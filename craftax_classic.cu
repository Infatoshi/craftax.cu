// craftax.cu -- all device code: game logic, worldgen, observations,
// the PufferLib-default policy (Linear encoder + MinGRU + heads), and
// every kernel. Host harness lives in main.cu, which #includes this file.
//
// Policy note: DefaultEncoder Linear(1345 -> 32, no activation) -> MinGRU
// (1 layer, hidden 32) -> logits Linear(32 -> 17) + value Linear(32 -> 1),
// matching PufferLib's default craftax config. The one-hot-dominated
// encoder collapses to a weight-column gather that is bit-exact vs the
// dense Linear (skipped terms are exact float zeros, same summation order).

#include <cstdint>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// ============================================================
// This is Craftax-CLASSIC, not Craftax-Full.
// Classic: 17 actions, 22 achievements, single 64x64 map, no
// dungeon floors / potions / enchantments / bosses.
//
// State is stored as structure-of-arrays (SoA): one thread per
// env means warp lanes touch adjacent envs, so per-field arrays
// indexed by env coalesce perfectly. The packed map stays as a
// per-env contiguous block (player positions diverge, so map
// reads never coalesce regardless of layout). Mob/plant slots
// are slot-major: field[slot * num_envs + env].
// ============================================================

// ============================================================
// Constants
// ============================================================
#define MAP_SIZE 64
#define MAX_ZOMBIES 3
#define MAX_COWS 3
#define MAX_SKELETONS 2
#define MAX_ARROWS 3
#define MAX_PLANTS 10
#define NUM_ACHIEVEMENTS 22
#define NUM_ACTIONS 17
#define NUM_BLOCK_TYPES 17
#define OBS_DIM 1345
#define OBS_MAP_ROWS 7
#define OBS_MAP_COLS 9
#define OBS_MAP_CELLS (OBS_MAP_ROWS * OBS_MAP_COLS)
#define OBS_MAP_CHANNELS 21  // 17 block types + 4 mob types
#define NUM_INVENTORY 12
#define MAX_TIMESTEPS 10000
#define DAY_LENGTH 300
#define MOB_DESPAWN_DIST 14

// Compact obs layout (uint8_t per env), exact-expandable to the
// 1345-float observation (light_level carried as raw float32):
//   [0..63)     block_ids   (63 tiles, values 0..16)
//   [63..126)   mob bitmask (bit0=zombie, bit1=cow, bit2=skel, bit3=arrow)
//   [126..138)  inventory   (12 slots, 0..9)
//   [138..142)  health,food,drink,energy (0..9)
//   [142]       player_dir  (1..4)
//   [143]       is_sleeping (0/1)
//   [144..148)  light_level (raw float32 bytes)
#define OBS_DIM_COMPACT 148

// Packed map: 2 blocks per byte (4-bit each), row = 32 bytes
#define MAP_PACKED_ROW 32
#define MAP_PACKED_SIZE (MAP_SIZE * MAP_PACKED_ROW)

// Block types
#define BLK_INVALID      0
#define BLK_OUT_OF_BOUNDS 1
#define BLK_GRASS        2
#define BLK_WATER        3
#define BLK_STONE        4
#define BLK_TREE         5
#define BLK_WOOD         6
#define BLK_PATH         7
#define BLK_COAL         8
#define BLK_IRON         9
#define BLK_DIAMOND      10
#define BLK_TABLE        11
#define BLK_FURNACE      12
#define BLK_SAND         13
#define BLK_LAVA         14
#define BLK_PLANT        15
#define BLK_RIPE_PLANT   16

// Actions
#define ACT_NOOP         0
#define ACT_LEFT         1
#define ACT_RIGHT        2
#define ACT_UP           3
#define ACT_DOWN         4
#define ACT_DO           5
#define ACT_SLEEP        6
#define ACT_PLACE_STONE  7
#define ACT_PLACE_TABLE  8
#define ACT_PLACE_FURNACE 9
#define ACT_PLACE_PLANT  10
#define ACT_MAKE_WOOD_PICK  11
#define ACT_MAKE_STONE_PICK 12
#define ACT_MAKE_IRON_PICK  13
#define ACT_MAKE_WOOD_SWORD 14
#define ACT_MAKE_STONE_SWORD 15
#define ACT_MAKE_IRON_SWORD  16

// Achievements
#define ACH_COLLECT_WOOD     0
#define ACH_PLACE_TABLE      1
#define ACH_EAT_COW          2
#define ACH_COLLECT_SAPLING  3
#define ACH_COLLECT_DRINK    4
#define ACH_MAKE_WOOD_PICK   5
#define ACH_MAKE_WOOD_SWORD  6
#define ACH_PLACE_PLANT      7
#define ACH_DEFEAT_ZOMBIE    8
#define ACH_COLLECT_STONE    9
#define ACH_PLACE_STONE      10
#define ACH_EAT_PLANT        11
#define ACH_DEFEAT_SKELETON  12
#define ACH_MAKE_STONE_PICK  13
#define ACH_MAKE_STONE_SWORD 14
#define ACH_WAKE_UP          15
#define ACH_PLACE_FURNACE    16
#define ACH_COLLECT_COAL     17
#define ACH_COLLECT_IRON     18
#define ACH_COLLECT_DIAMOND  19
#define ACH_MAKE_IRON_PICK   20
#define ACH_MAKE_IRON_SWORD  21

// Direction vectors: NOOP, LEFT, RIGHT, UP, DOWN
__constant__ int DIR_DR[5] = {0, 0, 0, -1, 1};
__constant__ int DIR_DC[5] = {0, -1, 1, 0, 0};

// ============================================================
// Game State: structure-of-arrays over envs
// ============================================================
struct EnvSoA {
    // Per-env contiguous packed map block: map_packed + e * MAP_PACKED_SIZE
    uint8_t* __restrict__ map_packed;

    // Player, indexed [e]
    int16_t* __restrict__ player_r;
    int16_t* __restrict__ player_c;
    int8_t*  __restrict__ player_dir; // 1=LEFT,2=RIGHT,3=UP,4=DOWN

    // Intrinsics, indexed [e]
    int8_t* __restrict__ health;
    int8_t* __restrict__ food;
    int8_t* __restrict__ drink;
    int8_t* __restrict__ energy;
    uint8_t* __restrict__ is_sleeping;
    float* __restrict__ recover;
    float* __restrict__ hunger;
    float* __restrict__ thirst;
    float* __restrict__ fatigue;

    // Inventory, slot-major: inv[slot * n + e]
    int8_t* __restrict__ inv;

    // Mobs, slot-major: field[slot * n + e]
    int16_t* __restrict__ zombie_r;
    int16_t* __restrict__ zombie_c;
    int8_t*  __restrict__ zombie_hp;
    int8_t*  __restrict__ zombie_cd;
    uint8_t* __restrict__ zombie_mask;

    int16_t* __restrict__ cow_r;
    int16_t* __restrict__ cow_c;
    int8_t*  __restrict__ cow_hp;
    uint8_t* __restrict__ cow_mask;

    int16_t* __restrict__ skel_r;
    int16_t* __restrict__ skel_c;
    int8_t*  __restrict__ skel_hp;
    int8_t*  __restrict__ skel_cd;
    uint8_t* __restrict__ skel_mask;

    int16_t* __restrict__ arrow_r;
    int16_t* __restrict__ arrow_c;
    int8_t*  __restrict__ arrow_dr;
    int8_t*  __restrict__ arrow_dc;
    uint8_t* __restrict__ arrow_mask;

    // Plants, slot-major
    int16_t* __restrict__ plant_r;
    int16_t* __restrict__ plant_c;
    int16_t* __restrict__ plant_age;
    uint8_t* __restrict__ plant_mask;

    // Misc, indexed [e]
    float*    __restrict__ light_level;
    uint32_t* __restrict__ ach;      // achievement bitmask, bit i = ACH_i
    int32_t*  __restrict__ timestep;

    // RNG, indexed [e]
    curandStatePhilox4_32_10_t* __restrict__ rng;

    int n; // num_envs (stride for slot-major fields)
};

// ============================================================
// Host-side SoA arena layout (shared by the C harness and the
// optional torch wrapper): one device allocation, 256B-aligned
// fields carved in a fixed order.
// ============================================================
struct SoAArena {
    uint8_t* base;
    size_t off = 0;
    template <typename T> T* take(size_t count) {
        off = (off + 255) & ~size_t(255);
        T* p = reinterpret_cast<T*>(base + off);
        off += count * sizeof(T);
        return p;
    }
};

inline size_t soa_bytes(int n) {
    size_t bytes = 0;
    bytes += (size_t)n * MAP_PACKED_SIZE;                 // map
    bytes += (size_t)n * 2 * 2 + n;                       // player r,c,dir
    bytes += (size_t)n * 4 + n;                           // health..energy, sleeping
    bytes += (size_t)n * 4 * 4;                           // recover..fatigue
    bytes += (size_t)n * NUM_INVENTORY;                   // inv
    bytes += (size_t)n * MAX_ZOMBIES * (2+2+1+1+1);
    bytes += (size_t)n * MAX_COWS * (2+2+1+1);
    bytes += (size_t)n * MAX_SKELETONS * (2+2+1+1+1);
    bytes += (size_t)n * MAX_ARROWS * (2+2+1+1+1);
    bytes += (size_t)n * MAX_PLANTS * (2+2+2+1);
    bytes += (size_t)n * (4 + 4 + 4);                     // light, ach, timestep
    bytes += (size_t)n * sizeof(curandStatePhilox4_32_10_t);
    bytes += 64 * 256;                                    // alignment slack
    return bytes;
}

inline EnvSoA carve_soa(uint8_t* base, int n) {
    SoAArena a{base};
    EnvSoA g;
    g.n = n;
    g.map_packed = a.take<uint8_t>((size_t)n * MAP_PACKED_SIZE);
    g.player_r = a.take<int16_t>(n);
    g.player_c = a.take<int16_t>(n);
    g.player_dir = a.take<int8_t>(n);
    g.health = a.take<int8_t>(n);
    g.food = a.take<int8_t>(n);
    g.drink = a.take<int8_t>(n);
    g.energy = a.take<int8_t>(n);
    g.is_sleeping = a.take<uint8_t>(n);
    g.recover = a.take<float>(n);
    g.hunger = a.take<float>(n);
    g.thirst = a.take<float>(n);
    g.fatigue = a.take<float>(n);
    g.inv = a.take<int8_t>((size_t)NUM_INVENTORY * n);
    g.zombie_r = a.take<int16_t>((size_t)MAX_ZOMBIES * n);
    g.zombie_c = a.take<int16_t>((size_t)MAX_ZOMBIES * n);
    g.zombie_hp = a.take<int8_t>((size_t)MAX_ZOMBIES * n);
    g.zombie_cd = a.take<int8_t>((size_t)MAX_ZOMBIES * n);
    g.zombie_mask = a.take<uint8_t>((size_t)MAX_ZOMBIES * n);
    g.cow_r = a.take<int16_t>((size_t)MAX_COWS * n);
    g.cow_c = a.take<int16_t>((size_t)MAX_COWS * n);
    g.cow_hp = a.take<int8_t>((size_t)MAX_COWS * n);
    g.cow_mask = a.take<uint8_t>((size_t)MAX_COWS * n);
    g.skel_r = a.take<int16_t>((size_t)MAX_SKELETONS * n);
    g.skel_c = a.take<int16_t>((size_t)MAX_SKELETONS * n);
    g.skel_hp = a.take<int8_t>((size_t)MAX_SKELETONS * n);
    g.skel_cd = a.take<int8_t>((size_t)MAX_SKELETONS * n);
    g.skel_mask = a.take<uint8_t>((size_t)MAX_SKELETONS * n);
    g.arrow_r = a.take<int16_t>((size_t)MAX_ARROWS * n);
    g.arrow_c = a.take<int16_t>((size_t)MAX_ARROWS * n);
    g.arrow_dr = a.take<int8_t>((size_t)MAX_ARROWS * n);
    g.arrow_dc = a.take<int8_t>((size_t)MAX_ARROWS * n);
    g.arrow_mask = a.take<uint8_t>((size_t)MAX_ARROWS * n);
    g.plant_r = a.take<int16_t>((size_t)MAX_PLANTS * n);
    g.plant_c = a.take<int16_t>((size_t)MAX_PLANTS * n);
    g.plant_age = a.take<int16_t>((size_t)MAX_PLANTS * n);
    g.plant_mask = a.take<uint8_t>((size_t)MAX_PLANTS * n);
    g.light_level = a.take<float>(n);
    g.ach = a.take<uint32_t>(n);
    g.timestep = a.take<int32_t>(n);
    g.rng = a.take<curandStatePhilox4_32_10_t>(n);
    return g;
}

// ============================================================
// Packed map accessors (per-env block)
// ============================================================
__device__ __forceinline__ uint8_t* env_map(const EnvSoA& g, int e) {
    return g.map_packed + (size_t)e * MAP_PACKED_SIZE;
}

__device__ __forceinline__ int8_t map_get(const uint8_t* __restrict__ map, int r, int c) {
    int idx = r * MAP_PACKED_ROW + (c >> 1);
    uint8_t byte = map[idx];
    return (c & 1) ? (byte >> 4) : (byte & 0x0F);
}

__device__ __forceinline__ void map_set(uint8_t* __restrict__ map, int r, int c, int8_t val) {
    int idx = r * MAP_PACKED_ROW + (c >> 1);
    uint8_t byte = map[idx];
    if (c & 1)
        map[idx] = (byte & 0x0F) | ((val & 0x0F) << 4);
    else
        map[idx] = (byte & 0xF0) | (val & 0x0F);
}

// ============================================================
// Inline helpers
// ============================================================
__device__ __forceinline__ bool in_bounds(int r, int c) {
    return (unsigned)r < MAP_SIZE && (unsigned)c < MAP_SIZE;
}

__device__ __forceinline__ bool is_solid(int8_t blk) {
    return blk == BLK_WATER || blk == BLK_STONE || blk == BLK_TREE ||
           blk == BLK_COAL || blk == BLK_IRON || blk == BLK_DIAMOND ||
           blk == BLK_TABLE || blk == BLK_FURNACE ||
           blk == BLK_PLANT || blk == BLK_RIPE_PLANT;
}

__device__ __forceinline__ int l1_dist(int r1, int c1, int r2, int c2) {
    int dr = r1 - r2; if (dr < 0) dr = -dr;
    int dc = c1 - c2; if (dc < 0) dc = -dc;
    return dr + dc;
}

__device__ __forceinline__ float rand_f(curandStatePhilox4_32_10_t* rng) {
    return curand_uniform(rng);
}

__device__ __forceinline__ int rand_int(curandStatePhilox4_32_10_t* rng, int n) {
    return curand(rng) % n;
}

__device__ __forceinline__ int clamp_i(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __forceinline__ int min_i(int a, int b) { return a < b ? a : b; }
__device__ __forceinline__ int max_i(int a, int b) { return a > b ? a : b; }
__device__ __forceinline__ float min_f(float a, float b) { return a < b ? a : b; }
__device__ __forceinline__ float max_f(float a, float b) { return a > b ? a : b; }

__device__ __forceinline__ int sign_i(int v) { return (v > 0) - (v < 0); }

// Check if position has a mob (scan compact arrays instead of 64x64 map)
__device__ __forceinline__ bool has_mob_at(const EnvSoA& g, int e, int r, int c) {
    const int n = g.n;
    for (int i = 0; i < MAX_ZOMBIES; i++)
        if (g.zombie_mask[i*n+e] && g.zombie_r[i*n+e] == r && g.zombie_c[i*n+e] == c) return true;
    for (int i = 0; i < MAX_COWS; i++)
        if (g.cow_mask[i*n+e] && g.cow_r[i*n+e] == r && g.cow_c[i*n+e] == c) return true;
    for (int i = 0; i < MAX_SKELETONS; i++)
        if (g.skel_mask[i*n+e] && g.skel_r[i*n+e] == r && g.skel_c[i*n+e] == c) return true;
    return false;
}

// Check if any of 8 neighbors contains a specific block type
__device__ bool is_near_block(const uint8_t* __restrict__ map, int pr, int pc, int8_t blk_type) {
    const int dr8[8] = {0, 0, -1, 1, -1, -1, 1, 1};
    const int dc8[8] = {-1, 1, 0, 0, -1, 1, -1, 1};
    for (int i = 0; i < 8; i++) {
        int nr = pr + dr8[i], nc = pc + dc8[i];
        if (in_bounds(nr, nc) && map_get(map, nr, nc) == blk_type) return true;
    }
    return false;
}

// Get sword damage
__device__ __forceinline__ int get_damage(const EnvSoA& g, int e) {
    const int n = g.n;
    if (g.inv[11*n+e] > 0) return 5; // iron sword
    if (g.inv[10*n+e] > 0) return 3; // stone sword
    if (g.inv[9*n+e] > 0)  return 2; // wood sword
    return 1;
}

#include <cmath>

// ============================================================
// Perlin Noise for World Generation
// ============================================================
__device__ float perlin_interp(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

__device__ float perlin_2d(float x, float y, const float* angles, int grid_size) {
    int x0 = (int)floorf(x), y0 = (int)floorf(y);
    float fx = x - x0, fy = y - y0;
    float u = perlin_interp(fx), v = perlin_interp(fy);

    auto grad_dot = [&](int ix, int iy, float dx, float dy) -> float {
        int idx = ((ix % grid_size) + grid_size) % grid_size * grid_size +
                  ((iy % grid_size) + grid_size) % grid_size;
        float a = angles[idx];
        return __cosf(a) * dx + __sinf(a) * dy;
    };

    float n00 = grad_dot(x0, y0, fx, fy);
    float n10 = grad_dot(x0+1, y0, fx-1, fy);
    float n01 = grad_dot(x0, y0+1, fx, fy-1);
    float n11 = grad_dot(x0+1, y0+1, fx-1, fy-1);

    float nx0 = n00 + u * (n10 - n00);
    float nx1 = n01 + u * (n11 - n01);
    return (nx0 + v * (nx1 - nx0) + 1.0f) * 0.5f;
}

// ============================================================
// World Generation (Reset)
//
// All RNG draws live at FIXED Philox counter offsets in the
// (seed, subsequence) stream, so tiles can be generated in any
// order -- serially by one thread or cooperatively by a warp --
// with bit-identical results. Draw budget per world:
//   [0, 400)                  Perlin angle table (4 layers x 100)
//   [WG_TILE_OFF + 2t, +2t+2) tile t: ore roll, tree roll
//   [WG_DIAMOND_OFF + 2a, ..) diamond fallback attempt a: r, c
//   [WG_GAMEPLAY_OFF, ...)    gameplay draws after reset
// ============================================================
#define WG_GRID 10
#define WG_ANGLE_COUNT (4 * WG_GRID * WG_GRID)
#define WG_TILE_OFF 512
#define WG_DIAMOND_OFF (WG_TILE_OFF + 2 * MAP_SIZE * MAP_SIZE)
#define WG_GAMEPLAY_OFF 32768

__device__ __forceinline__ float philox_f_at(uint64_t seed, uint64_t subseq, uint64_t offset) {
    curandStatePhilox4_32_10_t st;
    curand_init(seed, subseq, offset, &st);
    return curand_uniform(&st);
}

__device__ __forceinline__ uint32_t philox_u32_at(uint64_t seed, uint64_t subseq, uint64_t offset) {
    curandStatePhilox4_32_10_t st;
    curand_init(seed, subseq, offset, &st);
    return curand(&st);
}

// Classify one tile given the angle table. Pure function of
// (seed, subseq, r, c, angles) -- no draw-order dependence.
__device__ int8_t wg_classify_tile(
    uint64_t seed, uint64_t subseq, const float* angles, int r, int c
) {
    const float scale = (float)MAP_SIZE / (float)(WG_GRID - 1);
    const int center = MAP_SIZE / 2;
    float nr = (float)r / scale;
    float nc = (float)c / scale;

    float water_noise = perlin_2d(nr, nc, angles + 0 * WG_GRID * WG_GRID, WG_GRID);
    float mountain_noise = perlin_2d(nr, nc, angles + 1 * WG_GRID * WG_GRID, WG_GRID);
    float tree_noise = perlin_2d(nr, nc, angles + 2 * WG_GRID * WG_GRID, WG_GRID);
    float path_noise = perlin_2d(nr, nc, angles + 3 * WG_GRID * WG_GRID, WG_GRID);

    float dist = sqrtf((float)((r-center)*(r-center) + (c-center)*(c-center)));
    float prox = 1.0f - min_f(dist / 20.0f, 1.0f);

    float water_val = water_noise - prox * 0.3f;
    float mountain_val = mountain_noise - prox * 0.3f;

    int8_t blk = BLK_GRASS;

    if (water_val > 0.7f) {
        blk = BLK_WATER;
    } else if (water_val > 0.6f && water_val <= 0.75f) {
        blk = BLK_SAND;
    } else if (mountain_val > 0.7f) {
        blk = BLK_STONE;
        if (path_noise > 0.8f) blk = BLK_PATH;
        if (mountain_val > 0.85f && water_noise > 0.4f) blk = BLK_PATH;
        if (mountain_val > 0.85f && tree_noise > 0.7f) blk = BLK_LAVA;
    }

    int tile = r * MAP_SIZE + c;
    if (blk == BLK_STONE) {
        float ore_roll = philox_f_at(seed, subseq, WG_TILE_OFF + 2 * (uint64_t)tile);
        if (ore_roll < 0.005f && mountain_val > 0.8f)
            blk = BLK_DIAMOND;
        else if (ore_roll < 0.035f)
            blk = BLK_IRON;
        else if (ore_roll < 0.075f)
            blk = BLK_COAL;
    }

    if (blk == BLK_GRASS && tree_noise > 0.5f
        && philox_f_at(seed, subseq, WG_TILE_OFF + 2 * (uint64_t)tile + 1) > 0.8f)
        blk = BLK_TREE;

    return blk;
}

// Reset all non-map per-env fields (everything except the map).
__device__ void wg_reset_fields(EnvSoA& g, int e, uint64_t seed, int rng_subseq) {
    const int n = g.n;
    const int center = MAP_SIZE / 2;
    g.player_r[e] = center; g.player_c[e] = center;
    g.player_dir[e] = 4;
    g.health[e] = 9; g.food[e] = 9; g.drink[e] = 9; g.energy[e] = 9;
    g.is_sleeping[e] = 0;
    g.recover[e] = 0; g.hunger[e] = 0; g.thirst[e] = 0; g.fatigue[e] = 0;

    for (int i = 0; i < NUM_INVENTORY; i++) g.inv[i*n+e] = 0;

    for (int i = 0; i < MAX_ZOMBIES; i++) { g.zombie_mask[i*n+e] = 0; g.zombie_hp[i*n+e] = 0; g.zombie_cd[i*n+e] = 0; }
    for (int i = 0; i < MAX_COWS; i++) { g.cow_mask[i*n+e] = 0; g.cow_hp[i*n+e] = 0; }
    for (int i = 0; i < MAX_SKELETONS; i++) { g.skel_mask[i*n+e] = 0; g.skel_hp[i*n+e] = 0; g.skel_cd[i*n+e] = 0; }
    for (int i = 0; i < MAX_ARROWS; i++) { g.arrow_mask[i*n+e] = 0; }
    for (int i = 0; i < MAX_PLANTS; i++) { g.plant_mask[i*n+e] = 0; g.plant_age[i*n+e] = 0; }

    g.ach[e] = 0;
    g.timestep[e] = 0;
    g.light_level[e] = 1.0f;

    curandStatePhilox4_32_10_t rng;
    curand_init(seed, rng_subseq, WG_GAMEPLAY_OFF, &rng);
    g.rng[e] = rng;
}

// Serial worldgen (one thread per env). Bit-identical to the warp
// version below; used by the initial full reset and the torch ext.
__device__ void generate_world(EnvSoA& g, int e, uint64_t seed, int rng_subseq) {
    uint8_t* __restrict__ map = env_map(g, e);

    float angles[WG_ANGLE_COUNT];
    for (int i = 0; i < WG_ANGLE_COUNT; i++)
        angles[i] = philox_f_at(seed, rng_subseq, i) * 2.0f * 3.14159265f;

    const int center = MAP_SIZE / 2;
    bool has_diamond = false;
    for (int r = 0; r < MAP_SIZE; r++) {
        for (int c2 = 0; c2 < MAP_SIZE; c2 += 2) {
            int8_t b0 = (r == center && c2 == center) ? BLK_GRASS
                : wg_classify_tile(seed, rng_subseq, angles, r, c2);
            int8_t b1 = (r == center && c2 + 1 == center) ? BLK_GRASS
                : wg_classify_tile(seed, rng_subseq, angles, r, c2 + 1);
            has_diamond |= (b0 == BLK_DIAMOND) || (b1 == BLK_DIAMOND);
            map[r * MAP_PACKED_ROW + (c2 >> 1)] = (uint8_t)((b0 & 0x0F) | ((b1 & 0x0F) << 4));
        }
    }

    if (!has_diamond) {
        for (int attempts = 0; attempts < 1000; attempts++) {
            int r = (int)(philox_u32_at(seed, rng_subseq, WG_DIAMOND_OFF + 2 * (uint64_t)attempts) % MAP_SIZE);
            int c = (int)(philox_u32_at(seed, rng_subseq, WG_DIAMOND_OFF + 2 * (uint64_t)attempts + 1) % MAP_SIZE);
            if (map_get(map, r, c) == BLK_STONE) {
                map_set(map, r, c, BLK_DIAMOND);
                break;
            }
        }
    }

    wg_reset_fields(g, e, seed, rng_subseq);
}

// Warp-cooperative worldgen: 32 lanes generate one map together.
// Each lane owns byte pairs (2 adjacent tiles) so nibble packing
// never races. angles_smem must hold WG_ANGLE_COUNT floats of
// warp-private shared memory. Bit-identical to generate_world.
__device__ void generate_world_warp(
    EnvSoA& g, int e, uint64_t seed, int rng_subseq, float* angles_smem
) {
    const int lane = threadIdx.x & 31;
    uint8_t* __restrict__ map = env_map(g, e);

    for (int i = lane; i < WG_ANGLE_COUNT; i += 32)
        angles_smem[i] = philox_f_at(seed, rng_subseq, i) * 2.0f * 3.14159265f;
    __syncwarp();

    const int center = MAP_SIZE / 2;
    bool has_diamond = false;
    // 2048 packed bytes, lane-strided: lane handles bytes lane, lane+32, ...
    for (int byte = lane; byte < MAP_PACKED_SIZE; byte += 32) {
        int r = byte / MAP_PACKED_ROW;
        int c2 = (byte % MAP_PACKED_ROW) * 2;
        int8_t b0 = (r == center && c2 == center) ? BLK_GRASS
            : wg_classify_tile(seed, rng_subseq, angles_smem, r, c2);
        int8_t b1 = (r == center && c2 + 1 == center) ? BLK_GRASS
            : wg_classify_tile(seed, rng_subseq, angles_smem, r, c2 + 1);
        has_diamond |= (b0 == BLK_DIAMOND) || (b1 == BLK_DIAMOND);
        map[byte] = (uint8_t)((b0 & 0x0F) | ((b1 & 0x0F) << 4));
    }
    __threadfence_block();
    bool any_diamond = __any_sync(0xFFFFFFFFu, has_diamond);
    __syncwarp();

    if (!any_diamond && lane == 0) {
        for (int attempts = 0; attempts < 1000; attempts++) {
            int r = (int)(philox_u32_at(seed, rng_subseq, WG_DIAMOND_OFF + 2 * (uint64_t)attempts) % MAP_SIZE);
            int c = (int)(philox_u32_at(seed, rng_subseq, WG_DIAMOND_OFF + 2 * (uint64_t)attempts + 1) % MAP_SIZE);
            if (map_get(map, r, c) == BLK_STONE) {
                map_set(map, r, c, BLK_DIAMOND);
                break;
            }
        }
    }

    if (lane == 0)
        wg_reset_fields(g, e, seed, rng_subseq);
    __syncwarp();
}

// ============================================================
// Step Logic
// ============================================================
__device__ void do_crafting(EnvSoA& g, int e, int action) {
    const int n = g.n;
    const uint8_t* __restrict__ map = env_map(g, e);
    int pr = g.player_r[e], pc = g.player_c[e];
    bool near_table = is_near_block(map, pr, pc, BLK_TABLE);
    bool near_furnace = is_near_block(map, pr, pc, BLK_FURNACE);

    if (action == ACT_MAKE_WOOD_PICK && near_table && g.inv[0*n+e] >= 1) {
        g.inv[0*n+e]--; g.inv[6*n+e]++; g.ach[e] |= 1u << ACH_MAKE_WOOD_PICK;
    }
    if (action == ACT_MAKE_STONE_PICK && near_table && g.inv[0*n+e] >= 1 && g.inv[1*n+e] >= 1) {
        g.inv[0*n+e]--; g.inv[1*n+e]--; g.inv[7*n+e]++; g.ach[e] |= 1u << ACH_MAKE_STONE_PICK;
    }
    if (action == ACT_MAKE_IRON_PICK && near_table && near_furnace &&
        g.inv[0*n+e] >= 1 && g.inv[1*n+e] >= 1 && g.inv[3*n+e] >= 1 && g.inv[2*n+e] >= 1) {
        g.inv[0*n+e]--; g.inv[1*n+e]--; g.inv[3*n+e]--; g.inv[2*n+e]--;
        g.inv[8*n+e]++; g.ach[e] |= 1u << ACH_MAKE_IRON_PICK;
    }
    if (action == ACT_MAKE_WOOD_SWORD && near_table && g.inv[0*n+e] >= 1) {
        g.inv[0*n+e]--; g.inv[9*n+e]++; g.ach[e] |= 1u << ACH_MAKE_WOOD_SWORD;
    }
    if (action == ACT_MAKE_STONE_SWORD && near_table && g.inv[0*n+e] >= 1 && g.inv[1*n+e] >= 1) {
        g.inv[0*n+e]--; g.inv[1*n+e]--; g.inv[10*n+e]++; g.ach[e] |= 1u << ACH_MAKE_STONE_SWORD;
    }
    if (action == ACT_MAKE_IRON_SWORD && near_table && near_furnace &&
        g.inv[0*n+e] >= 1 && g.inv[1*n+e] >= 1 && g.inv[3*n+e] >= 1 && g.inv[2*n+e] >= 1) {
        g.inv[0*n+e]--; g.inv[1*n+e]--; g.inv[3*n+e]--; g.inv[2*n+e]--;
        g.inv[11*n+e]++; g.ach[e] |= 1u << ACH_MAKE_IRON_SWORD;
    }
}

__device__ void do_action(EnvSoA& g, int e, curandStatePhilox4_32_10_t* rng) {
    const int n = g.n;
    int tr = g.player_r[e] + DIR_DR[g.player_dir[e]];
    int tc = g.player_c[e] + DIR_DC[g.player_dir[e]];
    if (!in_bounds(tr, tc)) return;

    int dmg = get_damage(g, e);
    bool attacked = false;

    for (int i = 0; i < MAX_ZOMBIES && !attacked; i++) {
        if (g.zombie_mask[i*n+e] && g.zombie_r[i*n+e] == tr && g.zombie_c[i*n+e] == tc) {
            g.zombie_hp[i*n+e] -= dmg;
            if (g.zombie_hp[i*n+e] <= 0) {
                g.zombie_mask[i*n+e] = 0;
                g.ach[e] |= 1u << ACH_DEFEAT_ZOMBIE;
            }
            attacked = true;
        }
    }
    for (int i = 0; i < MAX_COWS && !attacked; i++) {
        if (g.cow_mask[i*n+e] && g.cow_r[i*n+e] == tr && g.cow_c[i*n+e] == tc) {
            g.cow_hp[i*n+e] -= dmg;
            if (g.cow_hp[i*n+e] <= 0) {
                g.cow_mask[i*n+e] = 0;
                g.ach[e] |= 1u << ACH_EAT_COW;
                g.food[e] = min_i(9, g.food[e] + 6);
                g.hunger[e] = 0;
            }
            attacked = true;
        }
    }
    for (int i = 0; i < MAX_SKELETONS && !attacked; i++) {
        if (g.skel_mask[i*n+e] && g.skel_r[i*n+e] == tr && g.skel_c[i*n+e] == tc) {
            g.skel_hp[i*n+e] -= dmg;
            if (g.skel_hp[i*n+e] <= 0) {
                g.skel_mask[i*n+e] = 0;
                g.ach[e] |= 1u << ACH_DEFEAT_SKELETON;
            }
            attacked = true;
        }
    }

    if (attacked) return;

    uint8_t* __restrict__ map = env_map(g, e);
    int8_t blk = map_get(map, tr, tc);
    switch (blk) {
        case BLK_TREE:
            map_set(map, tr, tc, BLK_GRASS);
            g.inv[0*n+e] = min_i(9, g.inv[0*n+e] + 1);
            g.ach[e] |= 1u << ACH_COLLECT_WOOD;
            break;
        case BLK_STONE:
            if (g.inv[6*n+e] > 0 || g.inv[7*n+e] > 0 || g.inv[8*n+e] > 0) {
                map_set(map, tr, tc, BLK_PATH);
                g.inv[1*n+e] = min_i(9, g.inv[1*n+e] + 1);
                g.ach[e] |= 1u << ACH_COLLECT_STONE;
            }
            break;
        case BLK_COAL:
            if (g.inv[6*n+e] > 0 || g.inv[7*n+e] > 0 || g.inv[8*n+e] > 0) {
                map_set(map, tr, tc, BLK_PATH);
                g.inv[2*n+e] = min_i(9, g.inv[2*n+e] + 1);
                g.ach[e] |= 1u << ACH_COLLECT_COAL;
            }
            break;
        case BLK_IRON:
            if (g.inv[7*n+e] > 0 || g.inv[8*n+e] > 0) {
                map_set(map, tr, tc, BLK_PATH);
                g.inv[3*n+e] = min_i(9, g.inv[3*n+e] + 1);
                g.ach[e] |= 1u << ACH_COLLECT_IRON;
            }
            break;
        case BLK_DIAMOND:
            if (g.inv[8*n+e] > 0) {
                map_set(map, tr, tc, BLK_PATH);
                g.inv[4*n+e] = min_i(9, g.inv[4*n+e] + 1);
                g.ach[e] |= 1u << ACH_COLLECT_DIAMOND;
            }
            break;
        case BLK_GRASS:
            if (rand_f(rng) < 0.1f) {
                g.inv[5*n+e] = min_i(9, g.inv[5*n+e] + 1);
                g.ach[e] |= 1u << ACH_COLLECT_SAPLING;
            }
            break;
        case BLK_WATER:
            g.drink[e] = min_i(9, g.drink[e] + 1);
            g.thirst[e] = 0;
            g.ach[e] |= 1u << ACH_COLLECT_DRINK;
            break;
        case BLK_RIPE_PLANT:
            map_set(map, tr, tc, BLK_PLANT);
            g.food[e] = min_i(9, g.food[e] + 4);
            g.hunger[e] = 0;
            g.ach[e] |= 1u << ACH_EAT_PLANT;
            for (int i = 0; i < MAX_PLANTS; i++) {
                if (g.plant_mask[i*n+e] && g.plant_r[i*n+e] == tr && g.plant_c[i*n+e] == tc) {
                    g.plant_age[i*n+e] = 0;
                    break;
                }
            }
            break;
    }
}

__device__ void place_block(EnvSoA& g, int e, int action) {
    const int n = g.n;
    int tr = g.player_r[e] + DIR_DR[g.player_dir[e]];
    int tc = g.player_c[e] + DIR_DC[g.player_dir[e]];
    if (!in_bounds(tr, tc)) return;
    if (has_mob_at(g, e, tr, tc)) return;

    uint8_t* __restrict__ map = env_map(g, e);
    int8_t blk = map_get(map, tr, tc);

    if (action == ACT_PLACE_TABLE && g.inv[0*n+e] >= 2 && !is_solid(blk)) {
        map_set(map, tr, tc, BLK_TABLE); g.inv[0*n+e] -= 2;
        g.ach[e] |= 1u << ACH_PLACE_TABLE;
    }
    else if (action == ACT_PLACE_FURNACE && g.inv[1*n+e] >= 1 && !is_solid(blk)) {
        map_set(map, tr, tc, BLK_FURNACE); g.inv[1*n+e] -= 1;
        g.ach[e] |= 1u << ACH_PLACE_FURNACE;
    }
    else if (action == ACT_PLACE_STONE && g.inv[1*n+e] >= 1 && (!is_solid(blk) || blk == BLK_WATER)) {
        map_set(map, tr, tc, BLK_STONE); g.inv[1*n+e] -= 1;
        g.ach[e] |= 1u << ACH_PLACE_STONE;
    }
    else if (action == ACT_PLACE_PLANT && g.inv[5*n+e] >= 1 && blk == BLK_GRASS) {
        map_set(map, tr, tc, BLK_PLANT); g.inv[5*n+e] -= 1;
        g.ach[e] |= 1u << ACH_PLACE_PLANT;
        for (int i = 0; i < MAX_PLANTS; i++) {
            if (!g.plant_mask[i*n+e]) {
                g.plant_r[i*n+e] = tr; g.plant_c[i*n+e] = tc;
                g.plant_age[i*n+e] = 0; g.plant_mask[i*n+e] = 1;
                break;
            }
        }
    }
}

__device__ void move_player(EnvSoA& g, int e, int action) {
    if (action < 1 || action > 4) return;
    int nr = g.player_r[e] + DIR_DR[action];
    int nc = g.player_c[e] + DIR_DC[action];
    g.player_dir[e] = action;
    if (!in_bounds(nr, nc)) return;
    if (is_solid(map_get(env_map(g, e), nr, nc))) return;
    if (has_mob_at(g, e, nr, nc)) return;
    g.player_r[e] = nr; g.player_c[e] = nc;
}

__device__ bool can_move_mob(const EnvSoA& g, int e, int r, int c) {
    if (!in_bounds(r, c)) return false;
    int8_t blk = map_get(env_map(g, e), r, c);
    if (is_solid(blk)) return false;
    if (blk == BLK_LAVA) return false;
    if (has_mob_at(g, e, r, c)) return false;
    if (r == g.player_r[e] && c == g.player_c[e]) return false;
    return true;
}

__device__ void update_mobs(EnvSoA& g, int e, curandStatePhilox4_32_10_t* rng) {
    const int n = g.n;
    int pr = g.player_r[e], pc = g.player_c[e];

    for (int i = 0; i < MAX_ZOMBIES; i++) {
        if (!g.zombie_mask[i*n+e]) continue;
        int zr = g.zombie_r[i*n+e], zc = g.zombie_c[i*n+e];
        int dist = l1_dist(zr, zc, pr, pc);

        if (dist >= MOB_DESPAWN_DIST) { g.zombie_mask[i*n+e] = 0; continue; }

        if (dist <= 1 && g.zombie_cd[i*n+e] <= 0) {
            int dmg = g.is_sleeping[e] ? 7 : 2;
            g.health[e] -= dmg;
            g.zombie_cd[i*n+e] = 5;
            g.is_sleeping[e] = 0;
        }
        g.zombie_cd[i*n+e] = max_i(0, g.zombie_cd[i*n+e] - 1);

        int dr = 0, dc = 0;
        if (dist < 10 && rand_f(rng) < 0.75f) {
            int adr = abs(pr - zr), adc = abs(pc - zc);
            if (adr > adc || (adr == adc && rand_f(rng) < 0.5f))
                dr = sign_i(pr - zr);
            else
                dc = sign_i(pc - zc);
        } else {
            int d = rand_int(rng, 4);
            dr = DIR_DR[d+1]; dc = DIR_DC[d+1];
        }
        int nr = zr + dr, nc = zc + dc;
        if (can_move_mob(g, e, nr, nc)) {
            g.zombie_r[i*n+e] = nr; g.zombie_c[i*n+e] = nc;
        }
    }

    for (int i = 0; i < MAX_COWS; i++) {
        if (!g.cow_mask[i*n+e]) continue;
        int cr = g.cow_r[i*n+e], cc = g.cow_c[i*n+e];
        int dist = l1_dist(cr, cc, pr, pc);

        if (dist >= MOB_DESPAWN_DIST) { g.cow_mask[i*n+e] = 0; continue; }

        int d = rand_int(rng, 8);
        if (d < 4) {
            int dr = DIR_DR[d+1], dc2 = DIR_DC[d+1];
            int nr = cr + dr, nc = cc + dc2;
            if (can_move_mob(g, e, nr, nc)) {
                g.cow_r[i*n+e] = nr; g.cow_c[i*n+e] = nc;
            }
        }
    }

    for (int i = 0; i < MAX_SKELETONS; i++) {
        if (!g.skel_mask[i*n+e]) continue;
        int sr = g.skel_r[i*n+e], sc = g.skel_c[i*n+e];
        int dist = l1_dist(sr, sc, pr, pc);

        if (dist >= MOB_DESPAWN_DIST) { g.skel_mask[i*n+e] = 0; continue; }

        if (dist >= 4 && dist <= 5 && g.skel_cd[i*n+e] <= 0) {
            for (int a = 0; a < MAX_ARROWS; a++) {
                if (!g.arrow_mask[a*n+e]) {
                    g.arrow_mask[a*n+e] = 1;
                    g.arrow_r[a*n+e] = sr; g.arrow_c[a*n+e] = sc;
                    int adr = abs(pr - sr), adc = abs(pc - sc);
                    g.arrow_dr[a*n+e] = (adr > 0) ? sign_i(pr - sr) : 0;
                    g.arrow_dc[a*n+e] = (adc > 0) ? sign_i(pc - sc) : 0;
                    break;
                }
            }
            g.skel_cd[i*n+e] = 4;
        }
        g.skel_cd[i*n+e] = max_i(0, g.skel_cd[i*n+e] - 1);

        int dr = 0, dc = 0;
        bool random_move = rand_f(rng) < 0.15f;
        if (!random_move) {
            if (dist >= 10) {
                int adr = abs(pr - sr), adc = abs(pc - sc);
                if (adr > adc || (adr == adc && rand_f(rng) < 0.5f))
                    dr = sign_i(pr - sr);
                else
                    dc = sign_i(pc - sc);
            } else if (dist <= 3) {
                int adr = abs(pr - sr), adc = abs(pc - sc);
                if (adr > adc || (adr == adc && rand_f(rng) < 0.5f))
                    dr = -sign_i(pr - sr);
                else
                    dc = -sign_i(pc - sc);
            } else {
                random_move = true;
            }
        }
        if (random_move) {
            int d = rand_int(rng, 4);
            dr = DIR_DR[d+1]; dc = DIR_DC[d+1];
        }
        int nr = sr + dr, nc = sc + dc;
        if (can_move_mob(g, e, nr, nc)) {
            g.skel_r[i*n+e] = nr; g.skel_c[i*n+e] = nc;
        }
    }

    for (int i = 0; i < MAX_ARROWS; i++) {
        if (!g.arrow_mask[i*n+e]) continue;
        int nr = g.arrow_r[i*n+e] + g.arrow_dr[i*n+e];
        int nc = g.arrow_c[i*n+e] + g.arrow_dc[i*n+e];

        if (!in_bounds(nr, nc)) { g.arrow_mask[i*n+e] = 0; continue; }
        uint8_t* __restrict__ map = env_map(g, e);
        int8_t blk = map_get(map, nr, nc);
        if (is_solid(blk) && blk != BLK_WATER) {
            if (blk == BLK_FURNACE || blk == BLK_TABLE) map_set(map, nr, nc, BLK_PATH);
            g.arrow_mask[i*n+e] = 0;
            continue;
        }
        if (nr == pr && nc == pc) {
            g.health[e] -= 2;
            g.is_sleeping[e] = 0;
            g.arrow_mask[i*n+e] = 0;
            continue;
        }
        g.arrow_r[i*n+e] = nr; g.arrow_c[i*n+e] = nc;
    }
}

__device__ void spawn_mobs(EnvSoA& g, int e, curandStatePhilox4_32_10_t* rng) {
    const int n = g.n;
    int pr = g.player_r[e], pc = g.player_c[e];
    const uint8_t* __restrict__ map = env_map(g, e);

    int n_cows = 0, n_zombies = 0, n_skels = 0;
    for (int i = 0; i < MAX_COWS; i++) n_cows += g.cow_mask[i*n+e];
    for (int i = 0; i < MAX_ZOMBIES; i++) n_zombies += g.zombie_mask[i*n+e];
    for (int i = 0; i < MAX_SKELETONS; i++) n_skels += g.skel_mask[i*n+e];

    auto try_spawn = [&](int min_dist, int max_dist, bool need_grass, bool need_path, int* out_r, int* out_c) -> bool {
        for (int attempts = 0; attempts < 20; attempts++) {
            int r = rand_int(rng, MAP_SIZE);
            int c = rand_int(rng, MAP_SIZE);
            int dist = l1_dist(r, c, pr, pc);
            if (dist < min_dist || dist >= max_dist) continue;
            if (has_mob_at(g, e, r, c)) continue;
            if (r == pr && c == pc) continue;
            int8_t blk = map_get(map, r, c);
            if (need_grass && blk != BLK_GRASS) continue;
            if (need_path && blk != BLK_PATH) continue;
            if (!need_grass && !need_path && blk != BLK_GRASS && blk != BLK_PATH) continue;
            *out_r = r; *out_c = c;
            return true;
        }
        return false;
    };

    if (n_cows < MAX_COWS && rand_f(rng) < 0.1f) {
        int r, c;
        if (try_spawn(3, MOB_DESPAWN_DIST, true, false, &r, &c)) {
            for (int i = 0; i < MAX_COWS; i++) {
                if (!g.cow_mask[i*n+e]) {
                    g.cow_mask[i*n+e] = 1; g.cow_r[i*n+e] = r; g.cow_c[i*n+e] = c; g.cow_hp[i*n+e] = 3;
                    break;
                }
            }
        }
    }

    float zombie_chance = 0.02f + 0.1f * (1.0f - g.light_level[e]) * (1.0f - g.light_level[e]);
    if (n_zombies < MAX_ZOMBIES && rand_f(rng) < zombie_chance) {
        int r, c;
        if (try_spawn(9, MOB_DESPAWN_DIST, false, false, &r, &c)) {
            for (int i = 0; i < MAX_ZOMBIES; i++) {
                if (!g.zombie_mask[i*n+e]) {
                    g.zombie_mask[i*n+e] = 1; g.zombie_r[i*n+e] = r; g.zombie_c[i*n+e] = c;
                    g.zombie_hp[i*n+e] = 5; g.zombie_cd[i*n+e] = 0;
                    break;
                }
            }
        }
    }

    if (n_skels < MAX_SKELETONS && rand_f(rng) < 0.05f) {
        int r, c;
        if (try_spawn(9, MOB_DESPAWN_DIST, false, true, &r, &c)) {
            for (int i = 0; i < MAX_SKELETONS; i++) {
                if (!g.skel_mask[i*n+e]) {
                    g.skel_mask[i*n+e] = 1; g.skel_r[i*n+e] = r; g.skel_c[i*n+e] = c;
                    g.skel_hp[i*n+e] = 3; g.skel_cd[i*n+e] = 0;
                    break;
                }
            }
        }
    }
}

__device__ void update_plants(EnvSoA& g, int e) {
    const int n = g.n;
    for (int i = 0; i < MAX_PLANTS; i++) {
        if (!g.plant_mask[i*n+e]) continue;
        g.plant_age[i*n+e]++;
        if (g.plant_age[i*n+e] >= 600) {
            int r = g.plant_r[i*n+e], c = g.plant_c[i*n+e];
            uint8_t* __restrict__ map = env_map(g, e);
            if (in_bounds(r, c) && map_get(map, r, c) == BLK_PLANT) {
                map_set(map, r, c, BLK_RIPE_PLANT);
            }
        }
    }
}

__device__ void update_intrinsics(EnvSoA& g, int e, int action) {
    if (action == ACT_SLEEP && g.energy[e] < 9) g.is_sleeping[e] = 1;
    if (g.energy[e] >= 9 && g.is_sleeping[e]) {
        g.is_sleeping[e] = 0;
        g.ach[e] |= 1u << ACH_WAKE_UP;
    }

    float sleep_mul = g.is_sleeping[e] ? 0.5f : 1.0f;

    g.hunger[e] += sleep_mul;
    if (g.hunger[e] > 25.0f) { g.food[e]--; g.hunger[e] = 0; }

    g.thirst[e] += sleep_mul;
    if (g.thirst[e] > 20.0f) { g.drink[e]--; g.thirst[e] = 0; }

    if (g.is_sleeping[e]) g.fatigue[e] -= 1.0f;
    else g.fatigue[e] += 1.0f;
    if (g.fatigue[e] > 30.0f) { g.energy[e]--; g.fatigue[e] = 0; }
    if (g.fatigue[e] < -10.0f) { g.energy[e] = min_i(g.energy[e] + 1, 9); g.fatigue[e] = 0; }

    bool all_needs = (g.food[e] > 0) && (g.drink[e] > 0) && (g.energy[e] > 0 || g.is_sleeping[e]);
    if (all_needs) g.recover[e] += g.is_sleeping[e] ? 2.0f : 1.0f;
    else g.recover[e] += g.is_sleeping[e] ? -0.5f : -1.0f;
    if (g.recover[e] > 25.0f) { g.health[e] = min_i(g.health[e] + 1, 9); g.recover[e] = 0; }
    if (g.recover[e] < -15.0f) { g.health[e]--; g.recover[e] = 0; }
}

// ============================================================
// Observation
// ============================================================
// Gather the 63-tile view: block ids plus a 4-bit mob mask per cell.
// Two arch-gated layouts; emitted values are identical. Consumers gate
// their view loops on VIEW_PACKED_LAYOUT. Mobs are scanned once each
// and stamped into their view cell (instead of scanning every mob for
// every cell).
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 1000
// Blackwell+: packed into registers. blk[r] holds row r's 9 block ids
// at 6 bits per cell, mob[r] its 9 mob masks at 4 bits per cell. Each
// view row spans <= 5 packed map bytes, loaded as two aligned words
// instead of 9 per-nibble byte loads (63 -> 14 loads per view, no
// local-memory view arrays); consumers fully unroll the row loop so
// the row words stay in registers. Measured +3% env / +1% rollout on
// sm_120; on sm_86 the extra packing ALU and unrolled consumers cost
// more than the saved loads (-8 to -15% env), hence the gate.
#define VIEW_PACKED_LAYOUT 1
struct ViewPacked {
    uint64_t blk[OBS_MAP_ROWS];
    uint64_t mob[OBS_MAP_ROWS];
};

// r must be a compile-time constant at every call site (unrolled loops)
// so the row words stay in registers.
__device__ __forceinline__ int vp_blk(const ViewPacked& v, int r, int c) {
    return (int)((v.blk[r] >> (c * 6)) & 63);
}
__device__ __forceinline__ int vp_mob(const ViewPacked& v, int r, int c) {
    return (int)((v.mob[r] >> (c * 4)) & 0xF);
}

__device__ __forceinline__ void gather_view(
    const EnvSoA& g, int e, ViewPacked& v
) {
    const int n = g.n;
    const uint8_t* __restrict__ map = env_map(g, e);
    int pr = g.player_r[e], pc = g.player_c[e];

    // Aligned base byte of the 8-byte window covering cols pc-4..pc+4;
    // s0 = bit offset of column pc-4 within that window.
    int a = ((pc - 4) >> 1) & ~3;
    a = max(0, min(a, MAP_PACKED_ROW - 8));
    int s0 = ((pc - 4) - 2 * a) * 4;

    uint64_t oob_row = 0;
    #pragma unroll
    for (int c = 0; c < OBS_MAP_COLS; c++)
        oob_row |= (uint64_t)BLK_OUT_OF_BOUNDS << (c * 6);

    #pragma unroll
    for (int r = 0; r < OBS_MAP_ROWS; r++) {
        int mr = pr - 3 + r;
        if (mr < 0 || mr >= MAP_SIZE) { v.blk[r] = oob_row; v.mob[r] = 0; continue; }
        const uint8_t* p = map + mr * MAP_PACKED_ROW + a;
        uint64_t bits = (uint64_t)*(const uint32_t*)p
                      | ((uint64_t)*(const uint32_t*)(p + 4) << 32);
        uint64_t row = 0;
        #pragma unroll
        for (int c = 0; c < OBS_MAP_COLS; c++) {
            int mc = pc - 4 + c;
            // In-bounds cells always have shift in [0,60]; the mask
            // just keeps the out-of-bounds lanes' shift defined.
            uint64_t nib = (bits >> ((s0 + c * 4) & 63)) & 0xF;
            row |= ((mc >= 0 && mc < MAP_SIZE) ? nib : (uint64_t)BLK_OUT_OF_BOUNDS)
                   << (c * 6);
        }
        v.blk[r] = row;
        v.mob[r] = 0;
    }

    auto stamp = [&](int r, int c, uint32_t bit) {
        int dr = r - pr, dc = c - pc;
        if (dr < -3 || dr > 3 || dc < -4 || dc > 4) return;
        uint64_t b = (uint64_t)bit << ((dc + 4) * 4);
        switch (dr + 3) {
            case 0: v.mob[0] |= b; break;
            case 1: v.mob[1] |= b; break;
            case 2: v.mob[2] |= b; break;
            case 3: v.mob[3] |= b; break;
            case 4: v.mob[4] |= b; break;
            case 5: v.mob[5] |= b; break;
            default: v.mob[6] |= b; break;
        }
    };

    for (int i = 0; i < MAX_ZOMBIES; i++)
        if (g.zombie_mask[i*n+e]) stamp(g.zombie_r[i*n+e], g.zombie_c[i*n+e], 1);
    for (int i = 0; i < MAX_COWS; i++)
        if (g.cow_mask[i*n+e]) stamp(g.cow_r[i*n+e], g.cow_c[i*n+e], 2);
    for (int i = 0; i < MAX_SKELETONS; i++)
        if (g.skel_mask[i*n+e]) stamp(g.skel_r[i*n+e], g.skel_c[i*n+e], 4);
    for (int i = 0; i < MAX_ARROWS; i++)
        if (g.arrow_mask[i*n+e]) stamp(g.arrow_r[i*n+e], g.arrow_c[i*n+e], 8);
}

#else
// Pre-Blackwell: per-nibble gather into two local byte arrays, flat
// cell loops in consumers. Kept token-identical to the pre-packed code
// (two bare arrays, not a struct): wrapping the arrays in a struct
// makes ptxas keep it in local memory instead of promoting to
// registers (obs_kernel 96 -> 48 regs, +72% LDL, -8% env SPS).
#define VIEW_PACKED_LAYOUT 0

__device__ __forceinline__ void gather_view(
    const EnvSoA& g, int e, int8_t* view_blk, uint8_t* view_mob
) {
    const int n = g.n;
    const uint8_t* __restrict__ map = env_map(g, e);
    int pr = g.player_r[e], pc = g.player_c[e];

    for (int dr = -3; dr <= 3; dr++) {
        for (int dc = -4; dc <= 4; dc++) {
            int r = pr + dr, c = pc + dc;
            int cell = (dr + 3) * OBS_MAP_COLS + (dc + 4);
            view_blk[cell] = in_bounds(r, c) ? map_get(map, r, c) : BLK_OUT_OF_BOUNDS;
            view_mob[cell] = 0;
        }
    }

    auto stamp = [&](int r, int c, uint8_t bit) {
        int dr = r - pr, dc = c - pc;
        if (dr < -3 || dr > 3 || dc < -4 || dc > 4) return;
        view_mob[(dr + 3) * OBS_MAP_COLS + (dc + 4)] |= bit;
    };

    for (int i = 0; i < MAX_ZOMBIES; i++)
        if (g.zombie_mask[i*n+e]) stamp(g.zombie_r[i*n+e], g.zombie_c[i*n+e], 1);
    for (int i = 0; i < MAX_COWS; i++)
        if (g.cow_mask[i*n+e]) stamp(g.cow_r[i*n+e], g.cow_c[i*n+e], 2);
    for (int i = 0; i < MAX_SKELETONS; i++)
        if (g.skel_mask[i*n+e]) stamp(g.skel_r[i*n+e], g.skel_c[i*n+e], 4);
    for (int i = 0; i < MAX_ARROWS; i++)
        if (g.arrow_mask[i*n+e]) stamp(g.arrow_r[i*n+e], g.arrow_c[i*n+e], 8);
}
#endif

// Per-layout view declaration / parameter / argument spelling so the
// consumers below stay single-source where the bodies don't differ.
#if VIEW_PACKED_LAYOUT
#define VIEW_DECL_GATHER(g_, e_) ViewPacked view; gather_view(g_, e_, view)
#define VIEW_PARAMS const ViewPacked& v
#define VIEW_ARGS view
#else
#define VIEW_DECL_GATHER(g_, e_) \
    int8_t view_blk[OBS_MAP_CELLS]; uint8_t view_mob[OBS_MAP_CELLS]; \
    gather_view(g_, e_, view_blk, view_mob)
#define VIEW_PARAMS const int8_t* view_blk, const uint8_t* view_mob
#define VIEW_ARGS view_blk, view_mob
#endif

__device__ void build_observation(const EnvSoA& g, int e, float* obs) {
    const int n = g.n;
    VIEW_DECL_GATHER(g, e);

    int obs_idx = 0;
#if VIEW_PACKED_LAYOUT
    #pragma unroll
    for (int r = 0; r < OBS_MAP_ROWS; r++) {
        for (int c = 0; c < OBS_MAP_COLS; c++) {
            int blk = vp_blk(view, r, c);
            for (int b = 0; b < NUM_BLOCK_TYPES; b++)
                obs[obs_idx++] = (blk == b) ? 1.0f : 0.0f;
            int m = vp_mob(view, r, c);
            obs[obs_idx++] = (m & 1) ? 1.0f : 0.0f;
            obs[obs_idx++] = (m & 2) ? 1.0f : 0.0f;
            obs[obs_idx++] = (m & 4) ? 1.0f : 0.0f;
            obs[obs_idx++] = (m & 8) ? 1.0f : 0.0f;
        }
    }
#else
    for (int cell = 0; cell < OBS_MAP_CELLS; cell++) {
        int8_t blk = view_blk[cell];
        for (int b = 0; b < NUM_BLOCK_TYPES; b++)
            obs[obs_idx++] = (blk == b) ? 1.0f : 0.0f;
        uint8_t m = view_mob[cell];
        obs[obs_idx++] = (m & 1) ? 1.0f : 0.0f;
        obs[obs_idx++] = (m & 2) ? 1.0f : 0.0f;
        obs[obs_idx++] = (m & 4) ? 1.0f : 0.0f;
        obs[obs_idx++] = (m & 8) ? 1.0f : 0.0f;
    }
#endif

    for (int i = 0; i < NUM_INVENTORY; i++)
        obs[obs_idx++] = (float)g.inv[i*n+e] / 10.0f;

    obs[obs_idx++] = (float)g.health[e] / 10.0f;
    obs[obs_idx++] = (float)g.food[e] / 10.0f;
    obs[obs_idx++] = (float)g.drink[e] / 10.0f;
    obs[obs_idx++] = (float)g.energy[e] / 10.0f;

    for (int d = 1; d <= 4; d++)
        obs[obs_idx++] = (g.player_dir[e] == d) ? 1.0f : 0.0f;

    obs[obs_idx++] = g.light_level[e];
    obs[obs_idx++] = g.is_sleeping[e] ? 1.0f : 0.0f;
}

// Compact uint8 observation (see OBS_DIM_COMPACT layout in craftax.cuh).
// 148 bytes vs 5380 -- 36x less obs traffic. expand_obs_kernel reproduces
// the float observation bit-for-bit.
__device__ void write_compact_obs(
    const EnvSoA& g, int e, VIEW_PARAMS, uint8_t* obs
);

__device__ void build_observation_compact(const EnvSoA& g, int e, uint8_t* obs) {
#if VIEW_PACKED_LAYOUT
    VIEW_DECL_GATHER(g, e);
    write_compact_obs(g, e, VIEW_ARGS, obs);
#else
    // Inline body, token-identical to the pre-packed code (nvcc's
    // inlining shape is sensitive here; see the layout comment above).
    const int n = g.n;
    VIEW_DECL_GATHER(g, e);

    for (int cell = 0; cell < OBS_MAP_CELLS; cell++) {
        obs[cell] = (uint8_t)view_blk[cell];
        obs[OBS_MAP_CELLS + cell] = view_mob[cell];
    }

    int idx = 2 * OBS_MAP_CELLS;
    for (int i = 0; i < NUM_INVENTORY; i++)
        obs[idx++] = (uint8_t)g.inv[i*n+e];
    obs[idx++] = (uint8_t)g.health[e];
    obs[idx++] = (uint8_t)g.food[e];
    obs[idx++] = (uint8_t)g.drink[e];
    obs[idx++] = (uint8_t)g.energy[e];
    obs[idx++] = (uint8_t)g.player_dir[e];
    obs[idx++] = g.is_sleeping[e] ? 1 : 0;
    float light = g.light_level[e];
    memcpy(obs + idx, &light, sizeof(float));
#endif
}

// Expand compact obs back to the exact 1345-float observation.
// One thread per (env, cell-ish chunk): simple env-per-thread version.
extern "C" __global__ void expand_obs_kernel(
    const uint8_t* __restrict__ compact, float* __restrict__ obs, int num_envs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_envs) return;

    const uint8_t* __restrict__ c = compact + (size_t)idx * OBS_DIM_COMPACT;
    float* __restrict__ o = obs + (size_t)idx * OBS_DIM;

    int obs_idx = 0;
    for (int cell = 0; cell < OBS_MAP_CELLS; cell++) {
        int8_t blk = (int8_t)c[cell];
        for (int b = 0; b < NUM_BLOCK_TYPES; b++)
            o[obs_idx++] = (blk == b) ? 1.0f : 0.0f;
        uint8_t m = c[OBS_MAP_CELLS + cell];
        o[obs_idx++] = (m & 1) ? 1.0f : 0.0f;
        o[obs_idx++] = (m & 2) ? 1.0f : 0.0f;
        o[obs_idx++] = (m & 4) ? 1.0f : 0.0f;
        o[obs_idx++] = (m & 8) ? 1.0f : 0.0f;
    }

    int idx2 = 2 * OBS_MAP_CELLS;
    for (int i = 0; i < NUM_INVENTORY; i++)
        o[obs_idx++] = (float)(int8_t)c[idx2++] / 10.0f;
    o[obs_idx++] = (float)(int8_t)c[idx2++] / 10.0f;
    o[obs_idx++] = (float)(int8_t)c[idx2++] / 10.0f;
    o[obs_idx++] = (float)(int8_t)c[idx2++] / 10.0f;
    o[obs_idx++] = (float)(int8_t)c[idx2++] / 10.0f;
    int8_t dir = (int8_t)c[idx2++];
    for (int d = 1; d <= 4; d++)
        o[obs_idx++] = (dir == d) ? 1.0f : 0.0f;
    uint8_t sleeping = c[idx2++];
    float light;
    memcpy(&light, c + idx2, sizeof(float));
    o[obs_idx++] = light;
    o[obs_idx++] = sleeping ? 1.0f : 0.0f;
}

// ============================================================
// Step core (shared by fused and split kernels)
// ============================================================
__device__ __forceinline__ bool step_env(
    EnvSoA& g, int e, int action_in, float* __restrict__ rewards
) {
    const int n = g.n;
    curandStatePhilox4_32_10_t rng = g.rng[e];

    int action = action_in;
    int old_health = g.health[e];
    uint32_t old_ach = g.ach[e];

    if (g.is_sleeping[e]) action = ACT_NOOP;

    do_crafting(g, e, action);
    if (action == ACT_DO) do_action(g, e, &rng);
    if (action >= ACT_PLACE_STONE && action <= ACT_PLACE_PLANT) place_block(g, e, action);
    move_player(g, e, action);
    update_mobs(g, e, &rng);
    spawn_mobs(g, e, &rng);
    update_plants(g, e);
    update_intrinsics(g, e, action_in);

    for (int i = 0; i < NUM_INVENTORY; i++)
        g.inv[i*n+e] = clamp_i(g.inv[i*n+e], 0, 9);

    int32_t t = ++g.timestep[e];
    float t_frac = fmodf((float)t / (float)DAY_LENGTH, 1.0f) + 0.3f;
    float cos_val = __cosf(3.14159265f * t_frac);
    g.light_level[e] = 1.0f - fabsf(cos_val * cos_val * cos_val);

    float ach_reward = (float)__popc(g.ach[e] & ~old_ach);
    float health_reward = (float)(g.health[e] - old_health) * 0.1f;
    rewards[e] = ach_reward + health_reward;

    bool done = (t >= MAX_TIMESTEPS) || (g.health[e] <= 0);
    int pr = g.player_r[e], pc = g.player_c[e];
    if (in_bounds(pr, pc) && map_get(env_map(g, e), pr, pc) == BLK_LAVA)
        done = true;

    g.rng[e] = rng;
    return done;
}

// ============================================================
// Main Kernels
// ============================================================
extern "C" __global__ void reset_kernel(
    EnvSoA g, float* obs, uint8_t* obs_compact, int obs_mode,
    int num_envs, uint64_t seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_envs) return;

    generate_world(g, idx, seed, idx);
    if (obs_mode != 1)
        build_observation(g, idx, obs + (size_t)idx * OBS_DIM);
    if (obs_mode >= 1)
        build_observation_compact(g, idx, obs_compact + (size_t)idx * OBS_DIM_COMPACT);
}

// Step kernel: game logic only, marks dones but does NOT auto-reset
extern "C" __global__ void step_only_kernel(
    EnvSoA g, const int32_t* __restrict__ actions,
    float* __restrict__ rewards, int8_t* __restrict__ dones,
    int num_envs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_envs) return;
    dones[idx] = step_env(g, idx, actions[idx], rewards) ? 1 : 0;
}

// Auto-reset kernel: only runs on done envs, then builds obs for ALL envs
extern "C" __global__ void autoreset_obs_kernel(
    EnvSoA g, const int8_t* __restrict__ dones,
    float* obs, uint8_t* obs_compact, int obs_mode,
    int num_envs, uint64_t reset_seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_envs) return;

    if (dones[idx]) {
        generate_world(g, idx, reset_seed, idx + num_envs);
    }
    if (obs_mode != 1)
        build_observation(g, idx, obs + (size_t)idx * OBS_DIM);
    if (obs_mode >= 1)
        build_observation_compact(g, idx, obs_compact + (size_t)idx * OBS_DIM_COMPACT);
}

// ============================================================
// Split-reset path: step marks dones into a compact list, a
// warp-cooperative kernel regenerates only the done envs, then a
// uniform kernel builds obs for everyone. reset_ctrl is 2 ints:
// [0] = done count (filled by step_mark), [1] = work cursor
// (used by reset_warp). Zero both before step_mark each step.
// ============================================================
extern "C" __global__ void step_mark_kernel(
    EnvSoA g, const int32_t* __restrict__ actions,
    float* __restrict__ rewards, int8_t* __restrict__ dones,
    int32_t* __restrict__ reset_list, int32_t* __restrict__ reset_ctrl,
    int num_envs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_envs) return;
    bool done = step_env(g, idx, actions[idx], rewards);
    dones[idx] = done ? 1 : 0;
    if (done) {
        int slot = atomicAdd(&reset_ctrl[0], 1);
        reset_list[slot] = idx;
    }
}

// One warp per resetting env, work-stealing over the compact list.
// Launch with a fixed grid (RESET_WARP_BLOCK threads/block); warps
// loop until the list is drained, so grid size is independent of
// the (device-side) done count.
#define RESET_WARP_BLOCK 128
extern "C" __global__ void reset_warp_kernel(
    EnvSoA g, const int32_t* __restrict__ reset_list,
    int32_t* __restrict__ reset_ctrl,
    int num_envs, uint64_t reset_seed
) {
    __shared__ float angles[RESET_WARP_BLOCK / 32][WG_ANGLE_COUNT];
    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    for (;;) {
        int i = 0;
        if (lane == 0) i = atomicAdd(&reset_ctrl[1], 1);
        i = __shfl_sync(0xFFFFFFFFu, i, 0);
        if (i >= reset_ctrl[0]) return;
        int e = reset_list[i];
        generate_world_warp(g, e, reset_seed, e + num_envs, angles[warp_in_block]);
    }
}

// Warp-cooperative full reset (initial reset of all envs).
// Launch num_envs warps; subseq = env index, matching reset_kernel.
extern "C" __global__ void reset_all_warp_kernel(
    EnvSoA g, int num_envs, uint64_t seed
) {
    __shared__ float angles[RESET_WARP_BLOCK / 32][WG_ANGLE_COUNT];
    const int warp_in_block = threadIdx.x >> 5;
    int e = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (e >= num_envs) return;
    generate_world_warp(g, e, seed, e, angles[warp_in_block]);
}

// Uniform obs pass, run after resets so every env sees fresh state.
extern "C" __global__ void obs_kernel(
    EnvSoA g, float* obs, uint8_t* obs_compact, int obs_mode, int num_envs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_envs) return;
    if (obs_mode != 1)
        build_observation(g, idx, obs + (size_t)idx * OBS_DIM);
    if (obs_mode >= 1)
        build_observation_compact(g, idx, obs_compact + (size_t)idx * OBS_DIM_COMPACT);
}

// Combined step kernel (for backward compat)
extern "C" __global__ void step_kernel(
    EnvSoA g, const int32_t* __restrict__ actions,
    float* obs, uint8_t* obs_compact, int obs_mode,
    float* __restrict__ rewards, int8_t* __restrict__ dones,
    int num_envs, uint64_t reset_seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_envs) return;

    bool done = step_env(g, idx, actions[idx], rewards);
    dones[idx] = done ? 1 : 0;

    if (done) {
        generate_world(g, idx, reset_seed, idx + num_envs);
    }
    if (obs_mode != 1)
        build_observation(g, idx, obs + (size_t)idx * OBS_DIM);
    if (obs_mode >= 1)
        build_observation_compact(g, idx, obs_compact + (size_t)idx * OBS_DIM_COMPACT);
}

#define CUDA_CHECK(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    exit(1); } } while (0)

// Deterministic action stream: same across trees and GPUs.
__device__ __forceinline__ uint64_t mix64(uint64_t x) {
    x ^= x >> 33; x *= 0xFF51AFD7ED558CCDULL;
    x ^= x >> 33; x *= 0xC4CEB9FE1A85EC53ULL;
    x ^= x >> 33; return x;
}

extern "C" __global__ void gen_actions_kernel(int32_t* actions, int n, int step, uint64_t seed) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    uint64_t x = mix64(seed ^ (uint64_t)step * 0x9E3779B97F4A7C15ULL ^ (uint64_t)e * 0xBF58476D1CE4E5B9ULL);
    actions[e] = (int32_t)(x % NUM_ACTIONS);
}


// ============================================================
// Policy: Linear(1345) -> MinGRU x3 (hidden 256) -> actor + value.
//
// Single fixed shape. Batched tensor-core execution: warp-cooperative
// encoder (lane owns HIDDEN/32 outputs) + cuBLAS GEMMs; the recurrent
// epilogue is a separate elementwise kernel.
//
// The old thread-per-env policy path (megakernel, scalar forward,
// smem-staged backward) was removed on measurement:
//  - hidden 256 spills the per-thread accumulator (1KB/layer), and
//  - every env re-reads ~2.4MB of weights scalar-fashion, ~6.6TB/s of
//    L2 at the resulting 2.7M SPS ceiling.
// Batched GEMM reuses each weight across the whole env batch.
// See README for the numbers. Parameter arena layout:
// ============================================================
#define HIDDEN 256
#define GRU_LAYERS 3
#define GRU_OUT (3 * HIDDEN)             // per-layer gate rows (h, g, p)
#define W_GRU_ELEMS (GRU_LAYERS * GRU_OUT * HIDDEN)

#define PARAM_W_ENC 0
#define PARAM_B_ENC (OBS_DIM * HIDDEN)
#define PARAM_W_GRU (PARAM_B_ENC + HIDDEN)                 // [L][3H][H] stacked
#define PARAM_W_A   (PARAM_W_GRU + W_GRU_ELEMS)
#define PARAM_B_A   (PARAM_W_A + NUM_ACTIONS * HIDDEN)
#define PARAM_W_V   (PARAM_B_A + NUM_ACTIONS)
#define PARAM_B_V   (PARAM_W_V + HIDDEN)
#define PARAM_COUNT (PARAM_B_V + 1)

// Policy weights (device pointers into the flat params arena).
struct Weights {
    const float* __restrict__ W_enc;  // [OBS_DIM][HIDDEN] row per input feature
    const float* __restrict__ b_enc;  // [HIDDEN]
    const float* __restrict__ W_gru;  // [GRU_LAYERS][GRU_OUT][HIDDEN] slices
    const float* __restrict__ W_a;    // [NUM_ACTIONS][HIDDEN]
    const float* __restrict__ b_a;    // [NUM_ACTIONS]
    const float* __restrict__ W_v;    // [HIDDEN]
    const float* __restrict__ b_v;    // [1]
};

extern "C" __global__ void init_weights_kernel(
    float* w, int count, float bound, uint64_t seed, int subseq
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    curandStatePhilox4_32_10_t st;
    curand_init(seed, subseq, i, &st);
    w[i] = (2.0f * curand_uniform(&st) - 1.0f) * bound;
}

// Categorical sample from logits with one uniform; returns action,
// fills logprob of the sampled action.
__device__ int sample_action(const float* logits, float u, float* logprob) {
    float m = logits[0];
    for (int a = 1; a < NUM_ACTIONS; a++) m = fmaxf(m, logits[a]);
    float p[NUM_ACTIONS], total = 0.0f;
    for (int a = 0; a < NUM_ACTIONS; a++) { p[a] = expf(logits[a] - m); total += p[a]; }
    float target = u * total, cum = 0.0f;
    int action = NUM_ACTIONS - 1;
    for (int a = 0; a < NUM_ACTIONS; a++) {
        cum += p[a];
        if (target <= cum) { action = a; break; }
    }
    *logprob = logits[action] - m - logf(total);
    return action;
}

// Write compact obs from an already-gathered view.
__device__ void write_compact_obs(
    const EnvSoA& g, int e, VIEW_PARAMS, uint8_t* obs
) {
    const int n = g.n;
#if VIEW_PACKED_LAYOUT
    #pragma unroll
    for (int r = 0; r < OBS_MAP_ROWS; r++) {
        for (int c = 0; c < OBS_MAP_COLS; c++) {
            int cell = r * OBS_MAP_COLS + c;
            obs[cell] = (uint8_t)vp_blk(v, r, c);
            obs[OBS_MAP_CELLS + cell] = (uint8_t)vp_mob(v, r, c);
        }
    }
#else
    for (int cell = 0; cell < OBS_MAP_CELLS; cell++) {
        obs[cell] = (uint8_t)view_blk[cell];
        obs[OBS_MAP_CELLS + cell] = view_mob[cell];
    }
#endif
    int idx = 2 * OBS_MAP_CELLS;
    for (int i = 0; i < NUM_INVENTORY; i++)
        obs[idx++] = (uint8_t)g.inv[i*n+e];
    obs[idx++] = (uint8_t)g.health[e];
    obs[idx++] = (uint8_t)g.food[e];
    obs[idx++] = (uint8_t)g.drink[e];
    obs[idx++] = (uint8_t)g.energy[e];
    obs[idx++] = (uint8_t)g.player_dir[e];
    obs[idx++] = g.is_sleeping[e] ? 1 : 0;
    float light = g.light_level[e];
    memcpy(obs + idx, &light, sizeof(float));
}
// ============================================================
// Device math helpers
// ============================================================
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float mingru_g(float x) { return x >= 0.0f ? x + 0.5f : sigmoidf_(x); }
__device__ __forceinline__ float dg_mingru(float x) {  // d/dx of mingru_g
    if (x >= 0.0f) return 1.0f;
    float s = sigmoidf_(x);
    return s * (1.0f - s);
}

// ============================================================
// Warp-cooperative encoder: one warp per env, lane owns ENC_W =
// HIDDEN/32 hidden units. All 32 lanes redundantly gather the same
// env's view (uniform loads broadcast through L1) and walk the
// features in fused-encoder order; each lane FMAs its W_enc row
// slice (warp covers each 1-KB column line-exactly). Output is
// column-major [HIDDEN][cols]: h_enc[k + col*HIDDEN], fed straight
// to cuBLAS. Same math as the old dense encoder, different width.
// ============================================================
#define ENC_W (HIDDEN / 32)
static_assert(HIDDEN == 256 && (HIDDEN % 32) == 0, "fixed policy is hidden 256");

#define ENC_ACC(hv, W_enc, f_, x_, lane_) do { \
    const float* col_ = (W_enc) + ((size_t)(f_) * HIDDEN) + (lane_) * ENC_W; \
    _Pragma("unroll") \
    for (int i_ = 0; i_ < ENC_W; i_++) hv[i_] = fmaf((x_), col_[i_], hv[i_]); \
} while (0)

#define ENC_INIT(hv, b_enc, lane_) do { \
    const float* be_ = (b_enc) + (lane_) * ENC_W; \
    _Pragma("unroll") \
    for (int i_ = 0; i_ < ENC_W; i_++) hv[i_] = be_[i_]; \
} while (0)

#define ENC_STORE(h_enc_, col_, hv, lane_) do { \
    float4* dst_ = reinterpret_cast<float4*>( \
        (h_enc_) + (size_t)(col_) * HIDDEN + (lane_) * ENC_W); \
    dst_[0] = make_float4(hv[0], hv[1], hv[2], hv[3]); \
    dst_[1] = make_float4(hv[4], hv[5], hv[6], hv[7]); \
} while (0)

// Live-env encoder (rollout forward, bootstrap). One warp per env.
extern "C" __global__ void encode_env_warp_kernel(
    EnvSoA g, const float* __restrict__ W_enc, const float* __restrict__ b_enc,
    float* __restrict__ h_enc, int num_envs
) {
    const int lane = threadIdx.x & 31;
    int e = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (e >= num_envs) return;
    VIEW_DECL_GATHER(g, e);
    const int n = g.n;

    float hv[ENC_W];
    ENC_INIT(hv, b_enc, lane);
#if VIEW_PACKED_LAYOUT
    for (int r = 0; r < OBS_MAP_ROWS; r++) {
        for (int c = 0; c < OBS_MAP_COLS; c++) {
            int cell = r * OBS_MAP_COLS + c;
            ENC_ACC(hv, W_enc, cell * (NUM_BLOCK_TYPES + 4) + vp_blk(view, r, c), 1.0f, lane);
            int m = vp_mob(view, r, c);
            int fm = cell * (NUM_BLOCK_TYPES + 4) + NUM_BLOCK_TYPES;
            for (int k = 0; k < 4; k++)
                if (m & (1 << k)) ENC_ACC(hv, W_enc, fm + k, 1.0f, lane);
        }
    }
#else
    for (int cell = 0; cell < OBS_MAP_CELLS; cell++) {
        ENC_ACC(hv, W_enc, cell * (NUM_BLOCK_TYPES + 4) + view_blk[cell], 1.0f, lane);
        uint8_t m = view_mob[cell];
        int fm = cell * (NUM_BLOCK_TYPES + 4) + NUM_BLOCK_TYPES;
        for (int k = 0; k < 4; k++)
            if (m & (1 << k)) ENC_ACC(hv, W_enc, fm + k, 1.0f, lane);
    }
#endif

    int f = OBS_MAP_CELLS * (NUM_BLOCK_TYPES + 4);
    for (int j = 0; j < NUM_INVENTORY; j++, f++)
        ENC_ACC(hv, W_enc, f, (float)g.inv[j*n+e] / 10.0f, lane);
    float intr[4] = {
        (float)g.health[e] / 10.0f, (float)g.food[e] / 10.0f,
        (float)g.drink[e] / 10.0f, (float)g.energy[e] / 10.0f
    };
    for (int j = 0; j < 4; j++, f++) ENC_ACC(hv, W_enc, f, intr[j], lane);
    for (int d = 1; d <= 4; d++, f++)
        ENC_ACC(hv, W_enc, f, (g.player_dir[e] == d) ? 1.0f : 0.0f, lane);
    ENC_ACC(hv, W_enc, f, g.light_level[e], lane); f++;
    ENC_ACC(hv, W_enc, f, g.is_sleeping[e] ? 1.0f : 0.0f, lane);

    ENC_STORE(h_enc, e, hv, lane);
}

// Encoder from stored compact obs records (backward recompute, loss).
// One thread per (hidden unit k, sample w); sample w maps to
// (t = t0 + w / mb, env offset env_start + w % mb). Output column w
// (tight [HIDDEN][count]).
// NOTE: this was a warp-cooperative kernel (warp per sample, ENC_ACC
// float4 walk, ENC_STORE). Under nvcc 13.2 for sm_120 (-O3
// --use_fast_math) that form miscompiled in a compilation-context-
// dependent way: tail record reads returned values not present in the
// record (garbage x for inventory/intrinsics/dir/light features) and
// the last warp's float4 store was intermittently dropped. A bytewise
// identical renamed clone in the same TU compiled fine. The live-env
// encoder above (encode_env_warp_kernel) never showed the defect and
// is hash-verified, so it is left alone. This thread-per-unit form
// matches a float64 spec to float32 rounding on every column
// (max|d| 1.63e-07 over all 512 gradcheck columns); do NOT "optimize"
// it back to the warp-cooperative gather without re-running the
// spec harness against /tmp/gcdump.
extern "C" __global__ void encode_obs_kernel(
    const uint8_t* __restrict__ r_obs,
    const float* __restrict__ W_enc, const float* __restrict__ b_enc,
    float* __restrict__ h_enc,
    int t0, int count, int mb, int n, int env_start
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= HIDDEN * count) return;
    int k = idx & (HIDDEN - 1);
    int w = idx / HIDDEN;
    int t = t0 + w / mb;
    int el = w - (w / mb) * mb;
    const uint8_t* obs = r_obs + ((size_t)t * n + env_start + el) * OBS_DIM_COMPACT;
    const float* Wk = W_enc + k;

    float hv = b_enc[k];
    for (int cell = 0; cell < OBS_MAP_CELLS; cell++) {
        hv = fmaf(1.0f, Wk[(size_t)(cell * (NUM_BLOCK_TYPES + 4) + (int8_t)obs[cell]) * HIDDEN], hv);
        uint8_t m = obs[OBS_MAP_CELLS + cell];
        int fm = cell * (NUM_BLOCK_TYPES + 4) + NUM_BLOCK_TYPES;
        for (int j = 0; j < 4; j++)
            if (m & (1 << j)) hv = fmaf(1.0f, Wk[(size_t)(fm + j) * HIDDEN], hv);
    }
    int oi = 2 * OBS_MAP_CELLS;
    int f = OBS_MAP_CELLS * (NUM_BLOCK_TYPES + 4);
    for (int j = 0; j < NUM_INVENTORY + 4; j++, f++)
        hv = fmaf((float)(int8_t)obs[oi++] / 10.0f, Wk[(size_t)f * HIDDEN], hv);
    int8_t dir = (int8_t)obs[oi++];
    for (int d = 1; d <= 4; d++, f++)
        hv = fmaf((dir == d) ? 1.0f : 0.0f, Wk[(size_t)f * HIDDEN], hv);
    uint8_t sleeping = obs[oi++];
    float light;
    memcpy(&light, obs + oi, sizeof(float));
    hv = fmaf(light, Wk[(size_t)f * HIDDEN], hv); f++;
    hv = fmaf(sleeping ? 1.0f : 0.0f, Wk[(size_t)f * HIDDEN], hv);

    h_enc[(size_t)w * HIDDEN + k] = hv;
}

// ============================================================
// MinGRU layer epilogue (elementwise, one thread per (unit, col)).
// pre/x/state/h_out are col-major; HIDDEN is a power of two so the
// (k, col) decode is a mask+shift. Semantics match the old scalar
// nn_head exactly: state zeroed on prev_done (post-zero input is
// what gets stored for BPTT), out = st + sg*(g(zh)-st),
// h_out = p*out + (1-p)*x.
// ============================================================
extern "C" __global__ void mingru_epi_fwd_kernel(
    const float* __restrict__ pre,     // [GRU_OUT][cols] col-major
    const float* __restrict__ x,       // [HIDDEN][cols]
    float* __restrict__ state,         // [HIDDEN][cols] live, in/out
    float* __restrict__ h_out,         // [HIDDEN][cols]
    float* __restrict__ r_state_store, // [HIDDEN][cols] post-zero inputs, or null
    const int8_t* __restrict__ prev_dones,  // [cols] or null
    int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= HIDDEN * cols) return;
    int k = idx & (HIDDEN - 1);
    int col = idx / HIDDEN;
    size_t o = (size_t)col * HIDDEN + k;

    float st = state[o];
    if (prev_dones && prev_dones[col]) st = 0.0f;
    if (r_state_store) r_state_store[o] = st;

    size_t o3 = (size_t)col * GRU_OUT + k;
    float zh = pre[o3];
    float zg = pre[o3 + HIDDEN];
    float zp = pre[o3 + 2 * HIDDEN];
    float out = st + sigmoidf_(zg) * (mingru_g(zh) - st);
    float p = sigmoidf_(zp);
    float xv = x[o];
    h_out[o] = p * out + (1.0f - p) * xv;
    state[o] = out;
}

// Live-recurrence replay for the backward's forward-recompute and
// loss(): launched once per (t, layer) over one step's columns. At a
// BPTT segment start the state input is reloaded from the stored
// slab (the truncation constant, matching the sweep's dcarry zeroing
// at segment boundaries); inside a segment it carries live from the
// previous step's out, zeroed on the done between the steps. The
// stored-slab replay used previously severed the recurrence, so FD
// probed a different function than the sweep differentiates. The
// post-zero input state actually used is optionally recorded in
// tight layout for the backward sweep.
extern "C" __global__ void mingru_epi_replay_kernel(
    const float* __restrict__ pre,       // [GRU_OUT][mb] this step
    const float* __restrict__ x,         // [HIDDEN][mb] this step
    const float* __restrict__ r_state_t, // stored slab at this (t, layer), k*n+el; segment start only, else null
    float* __restrict__ live,            // [HIDDEN][mb] carry in/out
    const int8_t* __restrict__ prev_dones, // dones between t-1 and t (+env offset); null at segment start
    float* __restrict__ st_store,        // [HIDDEN][mb] tight st record for the sweep, or null
    float* __restrict__ x_next,          // [HIDDEN][mb]
    int n, int mb
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= HIDDEN * mb) return;
    int k = idx & (HIDDEN - 1);
    int el = idx / HIDDEN;
    size_t o = (size_t)el * HIDDEN + k;

    float st;
    if (r_state_t) {
        st = r_state_t[(size_t)k * n + el];
    } else {
        st = live[o];
        if (prev_dones[el]) st = 0.0f;
    }
    if (st_store) st_store[o] = st;
    size_t o3 = (size_t)el * GRU_OUT + k;
    float zh = pre[o3];
    float zg = pre[o3 + HIDDEN];
    float zp = pre[o3 + 2 * HIDDEN];
    float out = st + sigmoidf_(zg) * (mingru_g(zh) - st);
    float p = sigmoidf_(zp);
    x_next[o] = p * out + (1.0f - p) * x[o];
    live[o] = out;
}

// ============================================================
// Heads: value dot + categorical sample. One thread per env.
// Sampler stream: Philox subsequence = env index, offset = *step_ctr
// (device counter so CUDA graphs can replay with advancing offsets);
// identical to the old split path's init(seed^A5A5.., e, step_count).
// ============================================================
extern "C" __global__ void value_sample_kernel(
    const float* __restrict__ h_out,   // [HIDDEN][cols] col-major
    const float* __restrict__ logits,  // [NUM_ACTIONS][cols] col-major
    const float* __restrict__ b_a,
    const float* __restrict__ W_v, const float* __restrict__ b_v,
    int32_t* __restrict__ actions, float* __restrict__ logprobs,
    float* __restrict__ values,
    int cols, uint64_t seed, const uint64_t* __restrict__ step_ctr
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= cols) return;

    const float* h = h_out + (size_t)e * HIDDEN;
    float v = b_v[0];
    for (int i = 0; i < HIDDEN; i++) v = fmaf(W_v[i], h[i], v);
    values[e] = v;

    float logits_e[NUM_ACTIONS];
    const float* lp = logits + (size_t)e * NUM_ACTIONS;
    for (int a = 0; a < NUM_ACTIONS; a++) logits_e[a] = lp[a] + b_a[a];

    curandStatePhilox4_32_10_t sampler;
    curand_init(seed ^ 0xA5A5A5A5A5A5A5A5ULL, e, *step_ctr, &sampler);
    float logp;
    int action = sample_action(logits_e, curand_uniform(&sampler), &logp);
    actions[e] = action;
    logprobs[e] = logp;
}

// Value only (GAE bootstrap on the post-rollout state).
extern "C" __global__ void value_dot_kernel(
    const float* __restrict__ h_out, const float* __restrict__ W_v,
    const float* __restrict__ b_v, float* __restrict__ v_out, int cols
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= cols) return;
    const float* h = h_out + (size_t)e * HIDDEN;
    float v = b_v[0];
    for (int i = 0; i < HIDDEN; i++) v = fmaf(W_v[i], h[i], v);
    v_out[e] = v;
}

// Record compact obs for the training record (rollout forward only).
extern "C" __global__ void record_obs_kernel(
    EnvSoA g, uint8_t* __restrict__ r_obs, int num_envs
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_envs) return;
    VIEW_DECL_GATHER(g, e);
    write_compact_obs(g, e, VIEW_ARGS, r_obs + (size_t)e * OBS_DIM_COMPACT);
}

// Episode stats at dones, read before resets wipe achievement bits.
extern "C" __global__ void ep_stats_kernel(
    EnvSoA g, const int8_t* __restrict__ dones,
    unsigned long long* __restrict__ ep_stats, int num_envs
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_envs || !dones[e]) return;
    atomicAdd(&ep_stats[0], 1ULL);
    uint32_t a = g.ach[e];
    while (a) { int b = __ffs(a) - 1; a &= a - 1; atomicAdd(&ep_stats[1 + b], 1ULL); }
}

// reset_warp with the seed schedule computed from a device step
// counter (graph-replayable): reset_seed = seed + (*ctr + 1) * 1e6,
// matching the old host-side schedule exactly.
extern "C" __global__ void reset_warp_ctr_kernel(
    EnvSoA g, const int32_t* __restrict__ reset_list,
    int32_t* __restrict__ reset_ctrl,
    int num_envs, uint64_t seed, const uint64_t* __restrict__ step_ctr
) {
    __shared__ float angles[RESET_WARP_BLOCK / 32][WG_ANGLE_COUNT];
    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    uint64_t reset_seed = seed + (*step_ctr + 1) * 1000000ULL;

    for (;;) {
        int i = 0;
        if (lane == 0) i = atomicAdd(&reset_ctrl[1], 1);
        i = __shfl_sync(0xFFFFFFFFu, i, 0);
        if (i >= reset_ctrl[0]) return;
        int e = reset_list[i];
        generate_world_warp(g, e, reset_seed, e + num_envs, angles[warp_in_block]);
    }
}

// Single-thread counter bump (graph node): step counter / adam step.
extern "C" __global__ void bump_ctr_kernel(uint64_t* __restrict__ ctr) {
    *ctr += 1;
}

// ============================================================
// Scalar L=3 reference policy (verify only): one thread per env,
// dense feature walk, strictly fp32 (no tensor cores). h[HIDDEN]
// lives in local memory; use small env counts.
// ============================================================
extern "C" __global__ void ref_policy_l3_kernel(
    EnvSoA g, Weights w, float* __restrict__ ref_state,  // [L][H][n]
    const int8_t* __restrict__ prev_dones,
    int32_t* __restrict__ r_act, float* __restrict__ r_logprob,
    float* __restrict__ r_value, float* __restrict__ ref_h3,  // [H][n]
    int num_envs, uint64_t seed, const uint64_t* __restrict__ step_ctr
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_envs) return;
    VIEW_DECL_GATHER(g, e);
    const int n = g.n;

    float h[HIDDEN], h2[HIDDEN];
    for (int i = 0; i < HIDDEN; i++) h[i] = w.b_enc[i];
#if VIEW_PACKED_LAYOUT
    for (int r = 0; r < OBS_MAP_ROWS; r++) {
        for (int c = 0; c < OBS_MAP_COLS; c++) {
            int cell = r * OBS_MAP_COLS + c;
            const float* col = w.W_enc +
                (size_t)(cell * (NUM_BLOCK_TYPES + 4) + vp_blk(view, r, c)) * HIDDEN;
            for (int i = 0; i < HIDDEN; i++) h[i] = fmaf(1.0f, col[i], h[i]);
            int m = vp_mob(view, r, c);
            int fm = cell * (NUM_BLOCK_TYPES + 4) + NUM_BLOCK_TYPES;
            for (int k = 0; k < 4; k++) {
                if (m & (1 << k)) {
                    const float* mc = w.W_enc + (size_t)(fm + k) * HIDDEN;
                    for (int i = 0; i < HIDDEN; i++) h[i] = fmaf(1.0f, mc[i], h[i]);
                }
            }
        }
    }
#else
    for (int cell = 0; cell < OBS_MAP_CELLS; cell++) {
        const float* col = w.W_enc +
            (size_t)(cell * (NUM_BLOCK_TYPES + 4) + view_blk[cell]) * HIDDEN;
        for (int i = 0; i < HIDDEN; i++) h[i] = fmaf(1.0f, col[i], h[i]);
        uint8_t m = view_mob[cell];
        int fm = cell * (NUM_BLOCK_TYPES + 4) + NUM_BLOCK_TYPES;
        for (int k = 0; k < 4; k++) {
            if (m & (1 << k)) {
                const float* mc = w.W_enc + (size_t)(fm + k) * HIDDEN;
                for (int i = 0; i < HIDDEN; i++) h[i] = fmaf(1.0f, mc[i], h[i]);
            }
        }
    }
#endif
    int f = OBS_MAP_CELLS * (NUM_BLOCK_TYPES + 4);
    for (int j = 0; j < NUM_INVENTORY; j++, f++) {
        float x = (float)g.inv[j*n+e] / 10.0f;
        const float* col = w.W_enc + (size_t)f * HIDDEN;
        for (int i = 0; i < HIDDEN; i++) h[i] = fmaf(x, col[i], h[i]);
    }
    float intr[4] = {
        (float)g.health[e] / 10.0f, (float)g.food[e] / 10.0f,
        (float)g.drink[e] / 10.0f, (float)g.energy[e] / 10.0f
    };
    for (int j = 0; j < 4; j++, f++) {
        const float* col = w.W_enc + (size_t)f * HIDDEN;
        for (int i = 0; i < HIDDEN; i++) h[i] = fmaf(intr[j], col[i], h[i]);
    }
    for (int d = 1; d <= 4; d++, f++) {
        float x = (g.player_dir[e] == d) ? 1.0f : 0.0f;
        const float* col = w.W_enc + (size_t)f * HIDDEN;
        for (int i = 0; i < HIDDEN; i++) h[i] = fmaf(x, col[i], h[i]);
    }
    {
        const float* col = w.W_enc + (size_t)f * HIDDEN; f++;
        for (int i = 0; i < HIDDEN; i++) h[i] = fmaf(g.light_level[e], col[i], h[i]);
    }
    {
        const float* col = w.W_enc + (size_t)f * HIDDEN;
        float x = g.is_sleeping[e] ? 1.0f : 0.0f;
        for (int i = 0; i < HIDDEN; i++) h[i] = fmaf(x, col[i], h[i]);
    }

    for (int l = 0; l < GRU_LAYERS; l++) {
        const float* Wl = w.W_gru + (size_t)l * GRU_OUT * HIDDEN;
        bool dz = prev_dones && prev_dones[e];
        for (int k = 0; k < HIDDEN; k++) {
            float st = dz ? 0.0f : ref_state[((size_t)l * HIDDEN + k) * n + e];
            float zh = 0.0f, zg = 0.0f, zp = 0.0f;
            for (int j = 0; j < HIDDEN; j++) {
                zh = fmaf(Wl[k * HIDDEN + j], h[j], zh);
                zg = fmaf(Wl[(HIDDEN + k) * HIDDEN + j], h[j], zg);
                zp = fmaf(Wl[(2 * HIDDEN + k) * HIDDEN + j], h[j], zp);
            }
            float out = st + sigmoidf_(zg) * (mingru_g(zh) - st);
            float p = sigmoidf_(zp);
            h2[k] = p * out + (1.0f - p) * h[k];
            ref_state[((size_t)l * HIDDEN + k) * n + e] = out;
        }
        for (int i = 0; i < HIDDEN; i++) h[i] = h2[i];
    }
    for (int i = 0; i < HIDDEN; i++) ref_h3[e * HIDDEN + i] = h[i];  // col-major [H][n]

    float logits[NUM_ACTIONS];
    for (int a = 0; a < NUM_ACTIONS; a++) {
        float z = w.b_a[a];
        for (int j = 0; j < HIDDEN; j++) z = fmaf(w.W_a[a * HIDDEN + j], h[j], z);
        logits[a] = z;
    }
    float value = w.b_v[0];
    for (int j = 0; j < HIDDEN; j++) value = fmaf(w.W_v[j], h[j], value);

    curandStatePhilox4_32_10_t sampler;
    curand_init(seed ^ 0xA5A5A5A5A5A5A5A5ULL, e, *step_ctr, &sampler);
    float logp;
    r_act[e] = sample_action(logits, curand_uniform(&sampler), &logp);
    r_logprob[e] = logp;
    r_value[e] = value;
}
// ============================================================
// Training: PPO with a batched-GEMM backward, entirely on device.
//
// The rollout stores only r_obs (148 B/sample), r_state (per-layer
// post-zero state inputs, 3 KB/sample) and the small scalars. Each
// minibatch recomputes the forward at the current theta from r_obs
// with live recurrence inside each BPTT segment (stored r_state used
// only as the constant at segment starts -- the truncation point
// where the sweep zeroes dcarry), then:
//   - head grads per sample (head_bwd_kernel)
//   - dh chain up the layers via cuBLAS (W^T @ dpre GEMMs)
//   - per-layer backward sweeps, thread per (unit, env), dcarry as
//     the only sequential part (mingru_sweep_bwd_kernel)
//   - weight grads as flat GEMMs over samples k = T*mb:
//       dW_l = dpre_l @ x_l^T (beta=1 accumulation)
//   - dW_enc as a thread-per-(unit, sample) sparse scatter (enc_bwd_kernel)
// dpre aliases pre (each sweep element is read before written).
// ============================================================

// Head gradients for one sample column. Computes softmax policy
// loss (clipped ratio), entropy bonus and value loss grads exactly
// as the old ppo_backward did, writing dlogits (tight
// [NUM_ACTIONS][T*mb]) and dvalue ([T*mb]) for the GEMM chain.
// Optional loss_acc adds raw (unnormalized) sums for logging.
extern "C" __global__ void head_bwd_kernel(
    const float* __restrict__ logits,   // tight [NUM_ACTIONS][T*mb]
    const float* __restrict__ h_out,    // tight [HIDDEN][T*mb]
    const float* __restrict__ b_a,
    const float* __restrict__ W_v, const float* __restrict__ b_v,
    const int32_t* __restrict__ r_act,      // slab + env_start
    const float* __restrict__ r_logprob,    // slab + env_start
    const float* __restrict__ adv, const float* __restrict__ ret,  // slab + env_start
    const double* __restrict__ adv_stats,
    float* __restrict__ dlogits,        // tight [NUM_ACTIONS][T*mb]
    float* __restrict__ dvalue,         // [T*mb]
    double* __restrict__ loss_acc,      // [3] pg/v/ent raw sums, or null
    int T, int mb, int n,
    float clip_eps, float vf_coef, float ent_coef
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int cols = T * mb;
    if (c >= cols) return;
    int t = c / mb;
    size_t o = (size_t)t * n + (c - t * mb);

    double inv_count = 1.0 / ((double)n * (double)T);
    float adv_mean = (float)(adv_stats[0] * inv_count);
    double m = adv_stats[0] * inv_count;
    double var = adv_stats[1] * inv_count - m * m;
    float adv_inv_std = 1.0f / ((float)sqrt(var > 0.0 ? var : 0.0) + 1e-8f);
    float inv_batch = 1.0f / (float)cols;

    const float* lp = logits + (size_t)c * NUM_ACTIONS;
    float lg[NUM_ACTIONS], pi[NUM_ACTIONS];
    float mx = -1e30f;
    #pragma unroll
    for (int a = 0; a < NUM_ACTIONS; a++) {
        lg[a] = lp[a] + b_a[a];
        mx = fmaxf(mx, lg[a]);
    }
    float total = 0.0f;
    #pragma unroll
    for (int a = 0; a < NUM_ACTIONS; a++) { pi[a] = expf(lg[a] - mx); total += pi[a]; }
    float inv_total = 1.0f / total;
    #pragma unroll
    for (int a = 0; a < NUM_ACTIONS; a++) pi[a] *= inv_total;
    float lse = mx + logf(total);

    int act = r_act[o];
    float logp_new = lg[act] - lse;
    float ratio = expf(logp_new - r_logprob[o]);
    float A = (adv[o] - adv_mean) * adv_inv_std;
    float u1 = ratio * A;
    float u2 = fminf(fmaxf(ratio, 1.0f - clip_eps), 1.0f + clip_eps) * A;
    float dlogp = (u1 <= u2) ? -A * ratio : 0.0f;
    float H = 0.0f;
    #pragma unroll
    for (int a = 0; a < NUM_ACTIONS; a++) H -= pi[a] * (lg[a] - lse);

    const float* h3 = h_out + (size_t)c * HIDDEN;
    float v = b_v[0];
    #pragma unroll 4
    for (int j = 0; j < HIDDEN; j++) v = fmaf(W_v[j], h3[j], v);
    float dv = inv_batch * vf_coef * (v - ret[o]);
    dvalue[c] = dv;

    float* dq = dlogits + (size_t)c * NUM_ACTIONS;
    #pragma unroll
    for (int a = 0; a < NUM_ACTIONS; a++) {
        float d = dlogp * ((a == act ? 1.0f : 0.0f) - pi[a]);
        d += ent_coef * pi[a] * ((lg[a] - lse) + H);
        dq[a] = d * inv_batch;
    }

    if (loss_acc) {
        __shared__ double s_pg, s_v, s_e;
        if (threadIdx.x == 0) { s_pg = 0.0; s_v = 0.0; s_e = 0.0; }
        __syncthreads();
        atomicAdd(&s_pg, -fminf(u1, u2));
        float ve = 0.5f * (v - ret[o]) * (v - ret[o]);
        atomicAdd(&s_v, ve);
        atomicAdd(&s_e, H);
        __syncthreads();
        if (threadIdx.x == 0) {
            atomicAdd(&loss_acc[0], s_pg);
            atomicAdd(&loss_acc[1], s_v);
            atomicAdd(&loss_acc[2], s_e);
        }
    }
}

// dh_out after the heads: dh3 = W_a^T @ dlogits (cuBLAS) plus the
// value-head contribution dv * W_v, added here with one add per elt.
extern "C" __global__ void add_dv_wv_kernel(
    float* __restrict__ dh,            // [HIDDEN][T*mb]
    const float* __restrict__ dv,      // [T*mb]
    const float* __restrict__ W_v,
    int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= HIDDEN * cols) return;
    int k = idx & (HIDDEN - 1);
    int col = idx / HIDDEN;
    dh[idx] += dv[col] * W_v[k];
}

// Backward sweep for one MinGRU layer over one minibatch. Thread per
// (unit k, env el), descending t with the scalar dcarry; everything
// else is column-parallel. bptt-split truncation = dcarry zeroed at
// each segment boundary. dhGEMM carries W_{l+1}^T @ dpre_{l+1} (or the
// head-side dh for the top layer), dhExtra the highway term from the
// layer above ({1-p} * dhout), st_used the post-zero input states the
// live replay actually consumed (tight layout, recorded by
// mingru_epi_replay_kernel). dpre aliases pre in place (read before
// write per element).
extern "C" __global__ void mingru_sweep_bwd_kernel(
    const float* __restrict__ pre,      // tight [GRU_OUT][T*mb] (aliased out)
    const float* __restrict__ x,        // tight [HIDDEN][T*mb] layer input
    const float* __restrict__ dhGEMM,   // tight [HIDDEN][T*mb]
    const float* __restrict__ dhExtra,  // tight [HIDDEN][T*mb] or null
    const float* __restrict__ st_used,  // tight [HIDDEN][T*mb] post-zero input states from replay
    const int8_t* __restrict__ r_done,  // slab base + env_start
    float* __restrict__ dpre,           // tight [GRU_OUT][T*mb] (may alias pre)
    float* __restrict__ dhExtraOut,     // tight [HIDDEN][T*mb] or null
    int T, int mb, int n, int seg_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= HIDDEN * mb) return;
    int k = idx & (HIDDEN - 1);
    int el = idx / HIDDEN;

    float dcarry = 0.0f;
    for (int t = T - 1; t >= 0; t--) {
        if (t == T - 1 || ((t + 1) % seg_len) == 0) dcarry = 0.0f;
        size_t col = (size_t)t * mb + el;
        size_t o3 = col * GRU_OUT + k;
        size_t o = col * HIDDEN + k;
        float zh = pre[o3];
        float zg = pre[o3 + HIDDEN];
        float zp = pre[o3 + 2 * HIDDEN];
        float dhout = dhGEMM[o];
        if (dhExtra) dhout += dhExtra[o];
        float s_in = st_used[o];
        float xv = x[o];
        bool done_t = r_done[(size_t)t * n + el] != 0;

        float sg = sigmoidf_(zg);
        float gh = mingru_g(zh);
        float p = sigmoidf_(zp);
        float out_k = s_in + sg * (gh - s_in);
        float dout = dhout * p + (done_t ? 0.0f : dcarry);
        float dp_ = dhout * (out_k - xv);
        dpre[o3 + 2 * HIDDEN] = dp_ * p * (1.0f - p);
        if (dhExtraOut) dhExtraOut[o] = dhout * (1.0f - p);
        dcarry = dout * (1.0f - sg);
        float dsg = dout * (gh - s_in);
        dpre[o3 + HIDDEN] = dsg * sg * (1.0f - sg);
        dpre[o3] = dout * sg * dg_mingru(zh);
    }
}

// Sparse encoder backward: one thread per (hidden unit k, sample
// column w), scalar atomicAdds into dW_enc row entries and db_enc.
// Was warp-cooperative with an ENC_ACC_BWD macro mirrored after the
// old warp encoder; that pattern hit the same sm_120 nvcc 13.2
// miscompile as encode_obs_warp_kernel (inventory-region byte reads
// returned garbage, fd-vs-analytic showed bogus dW_enc row 1325).
// Thread-per-unit matches the fixed encode_obs_kernel walk; do NOT
// re-warp-cooperatize without re-running the spec harness and
// gradcheck.
extern "C" __global__ void enc_bwd_kernel(
    const uint8_t* __restrict__ r_obs,
    const float* __restrict__ dhGEMM,   // W_1^T @ dpre_1, tight [HIDDEN][T*mb]
    const float* __restrict__ dhExtra,  // highway term from layer 1, tight, or null
    float* __restrict__ dW_enc,         // [OBS_DIM][HIDDEN] row-major
    float* __restrict__ db_enc,
    int T, int mb, int n, int env_start
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= HIDDEN * T * mb) return;
    int k = idx & (HIDDEN - 1);
    int w = idx / HIDDEN;
    int t = w / mb;
    int el = w - t * mb;
    const uint8_t* obs = r_obs + ((size_t)t * n + env_start + el) * OBS_DIM_COMPACT;

    size_t o = (size_t)w * HIDDEN + k;
    float dh = dhGEMM[o];
    if (dhExtra) dh += dhExtra[o];
    atomicAdd(&db_enc[k], dh);

#define ENC_ACC_BWD(f_, x_) atomicAdd(&dW_enc[(size_t)(f_) * HIDDEN + k], dh * (x_))

    for (int cell = 0; cell < OBS_MAP_CELLS; cell++) {
        ENC_ACC_BWD(cell * (NUM_BLOCK_TYPES + 4) + (int8_t)obs[cell], 1.0f);
        uint8_t mm = obs[OBS_MAP_CELLS + cell];
        int fm = cell * (NUM_BLOCK_TYPES + 4) + NUM_BLOCK_TYPES;
        for (int j = 0; j < 4; j++)
            if (mm & (1 << j)) ENC_ACC_BWD(fm + j, 1.0f);
    }
    int idx2 = 2 * OBS_MAP_CELLS;
    int f = OBS_MAP_CELLS * (NUM_BLOCK_TYPES + 4);
    for (int j = 0; j < NUM_INVENTORY + 4; j++, f++)
        ENC_ACC_BWD(f, (float)(int8_t)obs[idx2++] / 10.0f);
    int8_t dir = (int8_t)obs[idx2++];
    for (int d = 1; d <= 4; d++, f++) ENC_ACC_BWD(f, (dir == d) ? 1.0f : 0.0f);
    uint8_t sleeping = obs[idx2++];
    float light;
    memcpy(&light, obs + idx2, sizeof(float));
    ENC_ACC_BWD(f, light); f++;
    ENC_ACC_BWD(f, sleeping ? 1.0f : 0.0f);
#undef ENC_ACC_BWD
}

// Column sums of a [rows][cols] col-major matrix, accumulated into
// out (one block per row). Used for db_a (17 cols... rows) and db_v.
extern "C" __global__ void colsum_kernel(
    const float* __restrict__ M, float* __restrict__ out, int rows, int cols
) {
    int r = blockIdx.x;
    if (r >= rows) return;
    __shared__ float red[128];
    float acc = 0.0f;
    const float* col = M + r;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        acc += col[(size_t)c * rows];
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(&out[r], red[0]);
}

// Loss accumulation over one step of a full-batch loss evaluation.
// logits_t/h3_t are the tight per-step buffers ([NUM_ACTIONS][n] and
// [HIDDEN][n]); r_* are the full rollout slabs. Adds raw sums into
// loss_acc[3] (pg, v, ent), host divides by n*T_total. Matches the
// old ppo_loss math (same stats source, same per-term formulas).
extern "C" __global__ void loss_accum_kernel(
    const float* __restrict__ logits_t,  // tight [NUM_ACTIONS][n]
    const float* __restrict__ h3_t,      // tight [HIDDEN][n]
    const float* __restrict__ b_a,
    const float* __restrict__ W_v, const float* __restrict__ b_v,
    const int32_t* __restrict__ r_act,
    const float* __restrict__ r_logprob,
    const float* __restrict__ adv, const float* __restrict__ ret,
    const double* __restrict__ adv_stats,
    double* __restrict__ loss_acc,       // [3] pg/v/ent raw sums
    int t, int n, int T_total,
    float clip_eps
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    size_t o = (size_t)t * n + e;

    double inv_count = 1.0 / ((double)n * (double)T_total);
    float amean = (float)(adv_stats[0] * inv_count);
    double m = adv_stats[0] * inv_count;
    double var = adv_stats[1] * inv_count - m * m;
    float ainv_std = 1.0f / ((float)sqrt(var > 0.0 ? var : 0.0) + 1e-8f);

    const float* lp = logits_t + (size_t)e * NUM_ACTIONS;
    float lg[NUM_ACTIONS], pi[NUM_ACTIONS];
    float mx = -1e30f;
    #pragma unroll
    for (int a = 0; a < NUM_ACTIONS; a++) {
        lg[a] = lp[a] + b_a[a];
        mx = fmaxf(mx, lg[a]);
    }
    float total = 0.0f;
    #pragma unroll
    for (int a = 0; a < NUM_ACTIONS; a++) { pi[a] = expf(lg[a] - mx); total += pi[a]; }
    float inv_total = 1.0f / total;
    #pragma unroll
    for (int a = 0; a < NUM_ACTIONS; a++) pi[a] *= inv_total;
    float lse = mx + logf(total);

    float A = (adv[o] - amean) * ainv_std;
    float ratio = expf((lg[r_act[o]] - lse) - r_logprob[o]);
    float u1 = ratio * A;
    float u2 = fminf(fmaxf(ratio, 1.0f - clip_eps), 1.0f + clip_eps) * A;

    const float* h3 = h3_t + (size_t)e * HIDDEN;
    float v = b_v[0];
    #pragma unroll 4
    for (int j = 0; j < HIDDEN; j++) v = fmaf(W_v[j], h3[j], v);

    float H = 0.0f;
    #pragma unroll
    for (int a = 0; a < NUM_ACTIONS; a++) H -= pi[a] * (lg[a] - lse);

    __shared__ double s_pg, s_v, s_e;
    if (threadIdx.x == 0) { s_pg = 0.0; s_v = 0.0; s_e = 0.0; }
    __syncthreads();
    atomicAdd(&s_pg, -fminf(u1, u2));
    float ve = 0.5f * (v - ret[o]) * (v - ret[o]);
    atomicAdd(&s_v, ve);
    atomicAdd(&s_e, H);
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(&loss_acc[0], s_pg);
        atomicAdd(&loss_acc[1], s_v);
        atomicAdd(&loss_acc[2], s_e);
    }
}
// GAE scan, one thread per env, t = T-1 .. 0.
extern "C" __global__ void gae_kernel(
    const float* __restrict__ values, const float* __restrict__ rewards,
    const int8_t* __restrict__ dones, const float* __restrict__ v_boot,
    float* __restrict__ adv, float* __restrict__ ret,
    int num_envs, int T, float gamma, float lam
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_envs) return;
    float A = 0.0f;
    for (int t = T - 1; t >= 0; t--) {
        size_t o = (size_t)t * num_envs + e;
        float vnext = (t == T - 1) ? v_boot[e] : values[o + num_envs];
        float nonterm = dones[o] ? 0.0f : 1.0f;
        float delta = rewards[o] + gamma * vnext * nonterm - values[o];
        A = delta + gamma * lam * nonterm * A;
        adv[o] = A;
        ret[o] = A + values[o];
    }
}

// Sum and sum-of-squares of adv (for batch advantage normalization).
extern "C" __global__ void adv_stats_kernel(
    const float* __restrict__ adv, size_t count, double* __restrict__ stats  // [2]: sum, sumsq
) {
    __shared__ double s_sum[256], s_sq[256];
    double sum = 0.0, sq = 0.0;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += (size_t)gridDim.x * blockDim.x) {
        double a = adv[i];
        sum += a; sq += a * a;
    }
    s_sum[threadIdx.x] = sum; s_sq[threadIdx.x] = sq;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + w];
            s_sq[threadIdx.x] += s_sq[threadIdx.x + w];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(&stats[0], s_sum[0]);
        atomicAdd(&stats[1], s_sq[0]);
    }
}
// Reduce per-block gradient copies, Adam-update the flat params, and
// zero the copies for the next iteration. One thread per parameter.
// lr and the 1-based step come from device memory (a float and a
// uint64 counter) so the whole trainer iteration lives in one CUDA
// graph with host-side lr annealing as a single D2D float write.
extern "C" __global__ void adam_kernel(
    float* __restrict__ params, float* __restrict__ grads, int grad_copies,
    float* __restrict__ m, float* __restrict__ v,
    const uint64_t* __restrict__ step_ctr, const float* __restrict__ lr_ptr,
    float beta1, float beta2, float eps
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= PARAM_COUNT) return;
    float lr = *lr_ptr;
    int step = (int)(*step_ctr) + 1;
    float gsum = 0.0f;
    for (int c = 0; c < grad_copies; c++) {
        gsum += grads[(size_t)c * PARAM_COUNT + i];
        grads[(size_t)c * PARAM_COUNT + i] = 0.0f;
    }
    float mi = beta1 * m[i] + (1.0f - beta1) * gsum;
    float vi = beta2 * v[i] + (1.0f - beta2) * gsum * gsum;
    m[i] = mi; v[i] = vi;
    float mhat = mi / (1.0f - powf(beta1, (float)step));
    float vhat = vi / (1.0f - powf(beta2, (float)step));
    params[i] -= lr * mhat / (sqrtf(vhat) + eps);
}
