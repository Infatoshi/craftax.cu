#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <curand_kernel.h>

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
#define OBS_MAP_CHANNELS 21  // 17 block types + 4 mob types
#define NUM_INVENTORY 12
#define MAX_TIMESTEPS 10000
#define DAY_LENGTH 300
#define MOB_DESPAWN_DIST 14

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
// Game State (per environment) -- packed for minimal size
// ============================================================
struct __align__(16) EnvState {
    // Packed 4-bit map: map[r][c] stored as map_packed[r * 32 + c/2], nibble c%2
    uint8_t map_packed[MAP_SIZE * MAP_PACKED_ROW];  // 2048 bytes (was 4096)

    // Player
    int16_t player_r, player_c;
    int8_t player_dir; // 1=LEFT,2=RIGHT,3=UP,4=DOWN

    // Intrinsics
    int8_t health, food, drink, energy;
    bool is_sleeping;
    float recover, hunger, thirst, fatigue;

    // Inventory
    int8_t inv[NUM_INVENTORY]; // wood,stone,coal,iron,diamond,sapling,wpick,spick,ipick,wsword,ssword,isword

    // Mobs -- positions stored compactly, no mob_map needed
    int16_t zombie_r[MAX_ZOMBIES], zombie_c[MAX_ZOMBIES];
    int8_t zombie_hp[MAX_ZOMBIES], zombie_cd[MAX_ZOMBIES];
    bool zombie_mask[MAX_ZOMBIES];

    int16_t cow_r[MAX_COWS], cow_c[MAX_COWS];
    int8_t cow_hp[MAX_COWS];
    bool cow_mask[MAX_COWS];

    int16_t skel_r[MAX_SKELETONS], skel_c[MAX_SKELETONS];
    int8_t skel_hp[MAX_SKELETONS], skel_cd[MAX_SKELETONS];
    bool skel_mask[MAX_SKELETONS];

    int16_t arrow_r[MAX_ARROWS], arrow_c[MAX_ARROWS];
    int8_t arrow_dr[MAX_ARROWS], arrow_dc[MAX_ARROWS];
    bool arrow_mask[MAX_ARROWS];

    // Plants
    int16_t plant_r[MAX_PLANTS], plant_c[MAX_PLANTS];
    int16_t plant_age[MAX_PLANTS];
    bool plant_mask[MAX_PLANTS];

    // Misc
    float light_level;
    bool achievements[NUM_ACHIEVEMENTS];
    int32_t timestep;

    // RNG
    curandStatePhilox4_32_10_t rng;
};

// ============================================================
// Packed map accessors
// ============================================================
__device__ __forceinline__ int8_t map_get(const EnvState& s, int r, int c) {
    int idx = r * MAP_PACKED_ROW + (c >> 1);
    uint8_t byte = s.map_packed[idx];
    return (c & 1) ? (byte >> 4) : (byte & 0x0F);
}

__device__ __forceinline__ void map_set(EnvState& s, int r, int c, int8_t val) {
    int idx = r * MAP_PACKED_ROW + (c >> 1);
    uint8_t byte = s.map_packed[idx];
    if (c & 1)
        s.map_packed[idx] = (byte & 0x0F) | ((val & 0x0F) << 4);
    else
        s.map_packed[idx] = (byte & 0xF0) | (val & 0x0F);
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
__device__ __forceinline__ bool has_mob_at(const EnvState& s, int r, int c) {
    for (int i = 0; i < MAX_ZOMBIES; i++)
        if (s.zombie_mask[i] && s.zombie_r[i] == r && s.zombie_c[i] == c) return true;
    for (int i = 0; i < MAX_COWS; i++)
        if (s.cow_mask[i] && s.cow_r[i] == r && s.cow_c[i] == c) return true;
    for (int i = 0; i < MAX_SKELETONS; i++)
        if (s.skel_mask[i] && s.skel_r[i] == r && s.skel_c[i] == c) return true;
    return false;
}

// Check if any of 8 neighbors contains a specific block type
__device__ bool is_near_block(const EnvState& s, int8_t blk_type) {
    int pr = s.player_r, pc = s.player_c;
    const int dr8[8] = {0, 0, -1, 1, -1, -1, 1, 1};
    const int dc8[8] = {-1, 1, 0, 0, -1, 1, -1, 1};
    for (int i = 0; i < 8; i++) {
        int nr = pr + dr8[i], nc = pc + dc8[i];
        if (in_bounds(nr, nc) && map_get(s, nr, nc) == blk_type) return true;
    }
    return false;
}

// Get sword damage
__device__ __forceinline__ int get_damage(const EnvState& s) {
    if (s.inv[11] > 0) return 5; // iron sword
    if (s.inv[10] > 0) return 3; // stone sword
    if (s.inv[9] > 0)  return 2; // wood sword
    return 1;
}
