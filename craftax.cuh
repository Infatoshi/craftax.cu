#pragma once
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
