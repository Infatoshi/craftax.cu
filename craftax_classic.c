#define _GNU_SOURCE
// craftax.c -- pure C / AVX-512 CPU port of Craftax-Classic, single file.
// Same game logic as craftax.cu, optimized for many-core x86 (tuned on a
// Ryzen 9 9950X3D): OpenMP or a custom spin-barrier thread pool over envs,
// AVX-512 Perlin worldgen, and a pipelined world-generation pool that
// pre-generates reset maps on dedicated producer threads.
//
// Build:  gcc -O3 -march=native -mtune=native -ffast-math -fno-math-errno \
//             -funroll-loops -flto -fopenmp -o craftax_c craftax.c -lpthread -lm
// Run:    ./craftax_c [num_envs] [iters]     three configs: A libgomp+inline
//                                            reset, B thread pool, C thread
//                                            pool + world pool
// Requires AVX-512 (Zen 4/5, Ice Lake+).

// ============ from craftax.h ============
#include <stdint.h>
#include <stdbool.h>

// ============================================================
// This is Craftax-CLASSIC, not Craftax-Full.
// Classic: 17 actions, 22 achievements, single 64x64 map, no
// dungeon floors / potions / enchantments / bosses.
// ============================================================

// ============================================================
// Constants (mirror craftax.cuh)
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
#define OBS_MAP_CHANNELS 21

// Compact obs layout (uint8_t per env):
//   [0..63)    block_ids    (63 tiles, values 0..16)
//   [63..126)  mob bitmask  (bit0=zombie, bit1=cow, bit2=skel, bit3=arrow)
//   [126..138) inventory    (12 slots, 0..9)
//   [138..142) health,food,drink,energy (0..9)
//   [142]      player_dir   (0..4)
//   [143]      is_sleeping  (0/1)
//   [144]      light_level  (quantized 0..255)
#define OBS_DIM_COMPACT 145
#define NUM_INVENTORY 12
#define MAX_TIMESTEPS 10000
#define DAY_LENGTH 300
#define MOB_DESPAWN_DIST 14

#define MAP_PACKED_ROW 32
#define MAP_PACKED_SIZE (MAP_SIZE * MAP_PACKED_ROW)

#define BLK_INVALID       0
#define BLK_OUT_OF_BOUNDS 1
#define BLK_GRASS         2
#define BLK_WATER         3
#define BLK_STONE         4
#define BLK_TREE          5
#define BLK_WOOD          6
#define BLK_PATH          7
#define BLK_COAL          8
#define BLK_IRON          9
#define BLK_DIAMOND      10
#define BLK_TABLE        11
#define BLK_FURNACE      12
#define BLK_SAND         13
#define BLK_LAVA         14
#define BLK_PLANT        15
#define BLK_RIPE_PLANT   16

#define ACT_NOOP          0
#define ACT_LEFT          1
#define ACT_RIGHT         2
#define ACT_UP            3
#define ACT_DOWN          4
#define ACT_DO            5
#define ACT_SLEEP         6
#define ACT_PLACE_STONE   7
#define ACT_PLACE_TABLE   8
#define ACT_PLACE_FURNACE 9
#define ACT_PLACE_PLANT  10
#define ACT_MAKE_WOOD_PICK   11
#define ACT_MAKE_STONE_PICK  12
#define ACT_MAKE_IRON_PICK   13
#define ACT_MAKE_WOOD_SWORD  14
#define ACT_MAKE_STONE_SWORD 15
#define ACT_MAKE_IRON_SWORD  16

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
#define ACH_PLACE_STONE     10
#define ACH_EAT_PLANT       11
#define ACH_DEFEAT_SKELETON 12
#define ACH_MAKE_STONE_PICK 13
#define ACH_MAKE_STONE_SWORD 14
#define ACH_WAKE_UP         15
#define ACH_PLACE_FURNACE   16
#define ACH_COLLECT_COAL    17
#define ACH_COLLECT_IRON    18
#define ACH_COLLECT_DIAMOND 19
#define ACH_MAKE_IRON_PICK  20
#define ACH_MAKE_IRON_SWORD 21

// ============================================================
// PCG32 RNG -- tiny, fast, per-env
// ============================================================
typedef struct { uint64_t state; uint64_t inc; } pcg32_t;

static inline uint32_t pcg32_next(pcg32_t* r) {
    uint64_t old = r->state;
    r->state = old * 6364136223846793005ULL + r->inc;
    uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    uint32_t rot = (uint32_t)(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-(int32_t)rot) & 31));
}

static inline void pcg32_seed(pcg32_t* r, uint64_t seed, uint64_t seq) {
    r->state = 0;
    r->inc = (seq << 1u) | 1u;
    pcg32_next(r);
    r->state += seed;
    pcg32_next(r);
}

static inline float pcg32_uniform(pcg32_t* r) {
    // [0,1) with 24-bit mantissa
    return (pcg32_next(r) >> 8) * (1.0f / 16777216.0f);
}

static inline int pcg32_range(pcg32_t* r, int n) {
    // Bounded by n; unbiased enough for gameplay
    return (int)(pcg32_next(r) % (uint32_t)n);
}

// ============================================================
// Game State (per environment)
// ============================================================
typedef struct __attribute__((aligned(64))) {
    // Packed 4-bit map: nibble c%2 of byte [r*32 + c/2]
    uint8_t map_packed[MAP_SIZE * MAP_PACKED_ROW];

    // Per-row occupancy bitmaps (bit c = "mob/arrow at column c of row r")
    // mob_bits covers zombie|cow|skel (used by has_mob_at / can_move_mob).
    // The per-type bitmaps accelerate obs construction.
    uint64_t mob_bits[MAP_SIZE];
    uint64_t zombie_bits[MAP_SIZE];
    uint64_t cow_bits[MAP_SIZE];
    uint64_t skel_bits[MAP_SIZE];
    uint64_t arrow_bits[MAP_SIZE];

    int16_t player_r, player_c;
    int8_t player_dir;

    int8_t health, food, drink, energy;
    bool is_sleeping;
    float recover, hunger, thirst, fatigue;

    int8_t inv[NUM_INVENTORY];

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

    int16_t plant_r[MAX_PLANTS], plant_c[MAX_PLANTS];
    int16_t plant_age[MAX_PLANTS];
    bool plant_mask[MAX_PLANTS];

    float light_level;
    bool achievements[NUM_ACHIEVEMENTS];
    int32_t timestep;

    pcg32_t rng;
} EnvState;

// ============================================================
// Public API
// ============================================================
void craftax_reset(EnvState* s, uint64_t seed, uint64_t env_id);
void craftax_step(EnvState* s, int action, float* reward, int* done);
void craftax_build_obs(const EnvState* s, float* obs);
void craftax_build_obs_compact(const EnvState* s, uint8_t* obs);

// Batched helpers (OpenMP parallel over envs)
void craftax_reset_batch(EnvState* states, float* obs, int num_envs, uint64_t seed);
void craftax_step_batch(EnvState* states, const int32_t* actions,
                        float* obs, float* rewards, int8_t* dones,
                        int num_envs, uint64_t reset_seed);

void craftax_reset_batch_compact(EnvState* states, uint8_t* obs, int num_envs, uint64_t seed);
void craftax_step_batch_compact(EnvState* states, const int32_t* actions,
                                uint8_t* obs, float* rewards, int8_t* dones,
                                int num_envs, uint64_t reset_seed);

// ============================================================
// World-generation pool (pipelined resets)
// Producer threads pre-generate worlds; consumers pop on reset.
// ============================================================
typedef struct WorldPool WorldPool;

// capacity = ring size (slots). num_producers = background threads.
// Producer threads are pinned to SMT siblings of CCD1 cores (24..24+num_producers-1).
WorldPool* craftax_pool_create(int capacity, int num_producers, uint64_t master_seed);
void       craftax_pool_destroy(WorldPool* p);
void       craftax_pool_stats(WorldPool* p, uint64_t* produced, uint64_t* consumed,
                              uint64_t* fallbacks, int* ready_count);

// Same contract as craftax_step_batch_compact, but pops a pre-generated world
// on reset (falls back to inline generate_world if pool is empty).
void craftax_step_batch_compact_pool(WorldPool* pool,
                                     EnvState* states, const int32_t* actions,
                                     uint8_t* obs, float* rewards, int8_t* dones,
                                     int num_envs);

// ============================================================
// Worker-pool variants (use custom spin-barrier instead of OMP)
// ============================================================
struct ThreadPool;  // opaque; see worker_pool.h

void craftax_step_batch_compact_tp(struct ThreadPool* tp,
                                   EnvState* states, const int32_t* actions,
                                   uint8_t* obs, float* rewards, int8_t* dones,
                                   int num_envs, uint64_t reset_seed);

void craftax_step_batch_compact_pool_tp(struct ThreadPool* tp, WorldPool* pool,
                                        EnvState* states, const int32_t* actions,
                                        uint8_t* obs, float* rewards, int8_t* dones,
                                        int num_envs);

// ============ from worker_pool.h ============
// Minimal spin-barrier thread pool. Replaces #pragma omp parallel for to
// avoid libgomp team-barrier overhead (~30% of hot-path time at high thread
// counts when batches are tight).
//
// Usage:
//   int cpus[] = {0,1,2,...,15};
//   ThreadPool* tp = worker_pool_create(16, cpus);
//   worker_pool_run(tp, num_items, my_work_fn, my_arg);
//   worker_pool_destroy(tp);
//
// work_fn_t is called once per worker with its [begin, end) slice of [0, total).
#include <stdint.h>

typedef struct ThreadPool ThreadPool;

typedef void (*work_fn_t)(void* arg, int worker_id, int num_workers, int begin, int end);

// cpus: optional array of CPU ids (one per worker) for pthread_setaffinity.
// Pass NULL to leave threads unpinned.
ThreadPool* worker_pool_create(int num_workers, const int* cpus);
void        worker_pool_destroy(ThreadPool* p);

// Spin-based parallel for. Blocks until all workers finish.
void worker_pool_run(ThreadPool* p, int total, work_fn_t fn, void* arg);

int  worker_pool_num_workers(const ThreadPool* p);

// ============ from craftax.c ============
// Pure C port of craftax.cu -- CPU, OpenMP-parallel over envs.
// Same game logic and state layout as the CUDA version.
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <immintrin.h>

// ============================================================
// Map accessors / small helpers
// ============================================================
static inline int8_t map_get(const EnvState* s, int r, int c) {
    int idx = r * MAP_PACKED_ROW + (c >> 1);
    uint8_t byte = s->map_packed[idx];
    return (c & 1) ? (int8_t)(byte >> 4) : (int8_t)(byte & 0x0F);
}

static inline void map_set(EnvState* s, int r, int c, int8_t val) {
    int idx = r * MAP_PACKED_ROW + (c >> 1);
    uint8_t byte = s->map_packed[idx];
    if (c & 1) s->map_packed[idx] = (byte & 0x0F) | ((val & 0x0F) << 4);
    else       s->map_packed[idx] = (byte & 0xF0) | (val & 0x0F);
}

static inline bool in_bounds(int r, int c) {
    return (unsigned)r < MAP_SIZE && (unsigned)c < MAP_SIZE;
}

static inline bool is_solid(int8_t b) {
    return b == BLK_WATER || b == BLK_STONE || b == BLK_TREE ||
           b == BLK_COAL  || b == BLK_IRON  || b == BLK_DIAMOND ||
           b == BLK_TABLE || b == BLK_FURNACE ||
           b == BLK_PLANT || b == BLK_RIPE_PLANT;
}

static inline int l1_dist(int r1, int c1, int r2, int c2) {
    int dr = r1 - r2; if (dr < 0) dr = -dr;
    int dc = c1 - c2; if (dc < 0) dc = -dc;
    return dr + dc;
}

static inline int clamp_i(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
static inline int min_i(int a, int b) { return a < b ? a : b; }
static inline int max_i(int a, int b) { return a > b ? a : b; }
static inline float min_f(float a, float b) { return a < b ? a : b; }
static inline int sign_i(int v) { return (v > 0) - (v < 0); }

static const int DIR_DR[5] = {0, 0, 0, -1, 1};
static const int DIR_DC[5] = {0, -1, 1, 0, 0};

static inline float rand_f(EnvState* s) { return pcg32_uniform(&s->rng); }
static inline int   rand_int(EnvState* s, int n) { return pcg32_range(&s->rng, n); }

// O(1) mob bitmap query: bit c of mob_bits[r] = "mob at (r,c)"
static inline bool has_mob_at(const EnvState* s, int r, int c) {
    if ((unsigned)r >= MAP_SIZE || (unsigned)c >= MAP_SIZE) return false;
    return ((s->mob_bits[r] >> c) & 1ULL) != 0;
}

// Bitmap maintenance helpers (one bit per tile)
static inline void mb_set(uint64_t* bits, int r, int c)   { bits[r] |=  (1ULL << c); }
static inline void mb_clear(uint64_t* bits, int r, int c) { bits[r] &= ~(1ULL << c); }
static inline bool mb_get(const uint64_t* bits, int r, int c) { return (bits[r] >> c) & 1ULL; }

static bool is_near_block(const EnvState* s, int8_t blk_type) {
    int pr = s->player_r, pc = s->player_c;
    static const int dr8[8] = {0, 0, -1, 1, -1, -1, 1, 1};
    static const int dc8[8] = {-1, 1, 0, 0, -1, 1, -1, 1};
    for (int i = 0; i < 8; i++) {
        int nr = pr + dr8[i], nc = pc + dc8[i];
        if (in_bounds(nr, nc) && map_get(s, nr, nc) == blk_type) return true;
    }
    return false;
}

static inline int get_damage(const EnvState* s) {
    if (s->inv[11] > 0) return 5;
    if (s->inv[10] > 0) return 3;
    if (s->inv[9]  > 0) return 2;
    return 1;
}

// ============================================================
// Perlin noise worldgen
// ============================================================
static inline float perlin_interp(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

// Perlin kept for reference / API compat (no longer used in generate_world).
// The hot loop below evaluates all 4 layers inline, sharing the floor/frac/interp
// work across them.
__attribute__((unused))
static float perlin_2d(float x, float y, const float* cos_a, const float* sin_a, int grid) {
    int x0 = (int)floorf(x), y0 = (int)floorf(y);
    float fx = x - x0, fy = y - y0;
    float u = perlin_interp(fx), v = perlin_interp(fy);
    int i00 = (((x0  ) % grid) + grid) % grid * grid + (((y0  ) % grid) + grid) % grid;
    int i10 = (((x0+1) % grid) + grid) % grid * grid + (((y0  ) % grid) + grid) % grid;
    int i01 = (((x0  ) % grid) + grid) % grid * grid + (((y0+1) % grid) + grid) % grid;
    int i11 = (((x0+1) % grid) + grid) % grid * grid + (((y0+1) % grid) + grid) % grid;
    float n00 = cos_a[i00]*fx       + sin_a[i00]*fy;
    float n10 = cos_a[i10]*(fx-1.f) + sin_a[i10]*fy;
    float n01 = cos_a[i01]*fx       + sin_a[i01]*(fy-1.f);
    float n11 = cos_a[i11]*(fx-1.f) + sin_a[i11]*(fy-1.f);
    float nx0 = n00 + u * (n10 - n00);
    float nx1 = n01 + u * (n11 - n01);
    return (nx0 + v * (nx1 - nx0) + 1.0f) * 0.5f;
}

// Exposed for the pool (see craftax_pool.c). Prototype in pool impl.
void _craftax_generate_world(EnvState* s, uint64_t seed, uint64_t env_id);
static void generate_world(EnvState* s, uint64_t seed, uint64_t env_id) {
    _craftax_generate_world(s, seed, env_id);
}

void _craftax_generate_world(EnvState* s, uint64_t seed, uint64_t env_id) {
    pcg32_seed(&s->rng, seed, env_id);

    for (int i = 0; i < MAP_SIZE * MAP_PACKED_ROW; i++)
        s->map_packed[i] = (uint8_t)(BLK_GRASS | (BLK_GRASS << 4));

    enum { GRID = 10, GRID_PAD = GRID * GRID + 16 };
    // Precompute (cos,sin) of gradient angles. Padded by +16 floats so AVX-512
    // loads at max row (x0=8 => row1=90) don't read out of bounds.
    float cos_a[4][GRID_PAD];
    float sin_a[4][GRID_PAD];
    for (int layer = 0; layer < 4; layer++) {
        for (int i = 0; i < GRID * GRID; i++) {
            float a = rand_f(s) * 2.0f * 3.14159265f;
            cos_a[layer][i] = cosf(a);
            sin_a[layer][i] = sinf(a);
        }
        // Zero the pad region -- values never used but memory must be readable.
        for (int i = GRID * GRID; i < GRID_PAD; i++) {
            cos_a[layer][i] = 0.0f;
            sin_a[layer][i] = 0.0f;
        }
    }

    float scale = (float)MAP_SIZE / (float)(GRID - 1);
    float inv_scale = 1.0f / scale;
    int center = MAP_SIZE / 2;

    // AVX-512 Perlin: fill noise[4][MAP_SIZE][MAP_SIZE] in 16-column chunks.
    // Uses permutexvar instead of gather for the gradient lookup (tables are
    // only 10 entries per row, all lanes fit in one ZMM).
    // Stack-allocated so each caller thread has its own (generate_world is
    // called concurrently by producer threads).
    _Alignas(64) float noise[4][MAP_SIZE][MAP_SIZE];
    {
        const __m512 c_lane = _mm512_setr_ps(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
        const __m512 one = _mm512_set1_ps(1.0f);
        const __m512 half = _mm512_set1_ps(0.5f);
        const __m512 c6 = _mm512_set1_ps(6.0f);
        const __m512 c15 = _mm512_set1_ps(15.0f);
        const __m512 c10 = _mm512_set1_ps(10.0f);
        const __m512 invs = _mm512_set1_ps(inv_scale);
        const __m512i i_one = _mm512_set1_epi32(1);

        for (int r = 0; r < MAP_SIZE; r++) {
            float nr = (float)r * inv_scale;
            int x0 = (int)nr;
            float fx = nr - x0;
            float fx1 = fx - 1.0f;
            float u = perlin_interp(fx);
            int row0 = x0 * GRID, row1 = row0 + GRID;
            __m512 fx_v = _mm512_set1_ps(fx);
            __m512 fx1_v = _mm512_set1_ps(fx1);
            __m512 u_v  = _mm512_set1_ps(u);

            for (int c_base = 0; c_base < MAP_SIZE; c_base += 16) {
                __m512 c_v = _mm512_add_ps(_mm512_set1_ps((float)c_base), c_lane);
                __m512 nc_v = _mm512_mul_ps(c_v, invs);
                __m512i y0_v = _mm512_cvttps_epi32(nc_v);
                __m512 y0_f = _mm512_cvtepi32_ps(y0_v);
                __m512 fy_v  = _mm512_sub_ps(nc_v, y0_f);
                __m512 fy1_v = _mm512_sub_ps(fy_v, one);
                // v = fy^3 * (fy*(fy*6 - 15) + 10)  (smoothstep for Perlin)
                __m512 t = _mm512_fmsub_ps(fy_v, c6, c15);
                t = _mm512_fmadd_ps(fy_v, t, c10);
                __m512 fy2 = _mm512_mul_ps(fy_v, fy_v);
                __m512 fy3 = _mm512_mul_ps(fy2, fy_v);
                __m512 v_v = _mm512_mul_ps(fy3, t);
                __m512i y1_v = _mm512_add_epi32(y0_v, i_one);

                for (int k = 0; k < 4; k++) {
                    __m512 cos_r0 = _mm512_loadu_ps(&cos_a[k][row0]);
                    __m512 cos_r1 = _mm512_loadu_ps(&cos_a[k][row1]);
                    __m512 sin_r0 = _mm512_loadu_ps(&sin_a[k][row0]);
                    __m512 sin_r1 = _mm512_loadu_ps(&sin_a[k][row1]);

                    __m512 c00 = _mm512_permutexvar_ps(y0_v, cos_r0);
                    __m512 c10v= _mm512_permutexvar_ps(y0_v, cos_r1);
                    __m512 c01 = _mm512_permutexvar_ps(y1_v, cos_r0);
                    __m512 c11 = _mm512_permutexvar_ps(y1_v, cos_r1);
                    __m512 s00 = _mm512_permutexvar_ps(y0_v, sin_r0);
                    __m512 s10 = _mm512_permutexvar_ps(y0_v, sin_r1);
                    __m512 s01 = _mm512_permutexvar_ps(y1_v, sin_r0);
                    __m512 s11 = _mm512_permutexvar_ps(y1_v, sin_r1);

                    __m512 n00 = _mm512_fmadd_ps(c00,  fx_v,  _mm512_mul_ps(s00, fy_v));
                    __m512 n10 = _mm512_fmadd_ps(c10v, fx1_v, _mm512_mul_ps(s10, fy_v));
                    __m512 n01 = _mm512_fmadd_ps(c01,  fx_v,  _mm512_mul_ps(s01, fy1_v));
                    __m512 n11 = _mm512_fmadd_ps(c11,  fx1_v, _mm512_mul_ps(s11, fy1_v));

                    __m512 nx0 = _mm512_fmadd_ps(u_v, _mm512_sub_ps(n10, n00), n00);
                    __m512 nx1 = _mm512_fmadd_ps(u_v, _mm512_sub_ps(n11, n01), n01);
                    __m512 n = _mm512_fmadd_ps(v_v, _mm512_sub_ps(nx1, nx0), nx0);
                    n = _mm512_mul_ps(_mm512_add_ps(n, one), half);

                    _mm512_storeu_ps(&noise[k][r][c_base], n);
                }
            }
        }
    }

    // Tile-logic sweep reads from precomputed noise[][][].
    for (int r = 0; r < MAP_SIZE; r++) {
        for (int c = 0; c < MAP_SIZE; c++) {
            float water_noise    = noise[0][r][c];
            float mountain_noise = noise[1][r][c];
            float tree_noise     = noise[2][r][c];
            float path_noise     = noise[3][r][c];

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
                if (mountain_val > 0.85f && tree_noise > 0.7f)  blk = BLK_LAVA;
            }

            if (blk == BLK_STONE) {
                float ore = rand_f(s);
                if (ore < 0.005f && mountain_val > 0.8f) blk = BLK_DIAMOND;
                else if (ore < 0.035f) blk = BLK_IRON;
                else if (ore < 0.075f) blk = BLK_COAL;
            }
            if (blk == BLK_GRASS && tree_noise > 0.5f && rand_f(s) > 0.8f)
                blk = BLK_TREE;

            map_set(s, r, c, blk);
        }
    }

    map_set(s, center, center, BLK_GRASS);

    bool has_diamond = false;
    for (int r = 0; r < MAP_SIZE && !has_diamond; r++)
        for (int c = 0; c < MAP_SIZE && !has_diamond; c++)
            if (map_get(s, r, c) == BLK_DIAMOND) has_diamond = true;
    if (!has_diamond) {
        for (int att = 0; att < 1000; att++) {
            int r = rand_int(s, MAP_SIZE), c = rand_int(s, MAP_SIZE);
            if (map_get(s, r, c) == BLK_STONE) { map_set(s, r, c, BLK_DIAMOND); break; }
        }
    }

    s->player_r = center; s->player_c = center;
    s->player_dir = 4;
    s->health = 9; s->food = 9; s->drink = 9; s->energy = 9;
    s->is_sleeping = false;
    s->recover = s->hunger = s->thirst = s->fatigue = 0;

    memset(s->mob_bits, 0, sizeof(s->mob_bits));
    memset(s->zombie_bits, 0, sizeof(s->zombie_bits));
    memset(s->cow_bits, 0, sizeof(s->cow_bits));
    memset(s->skel_bits, 0, sizeof(s->skel_bits));
    memset(s->arrow_bits, 0, sizeof(s->arrow_bits));

    memset(s->inv, 0, sizeof(s->inv));
    memset(s->zombie_mask, 0, sizeof(s->zombie_mask));
    memset(s->zombie_hp,   0, sizeof(s->zombie_hp));
    memset(s->zombie_cd,   0, sizeof(s->zombie_cd));
    memset(s->cow_mask, 0, sizeof(s->cow_mask));
    memset(s->cow_hp,   0, sizeof(s->cow_hp));
    memset(s->skel_mask, 0, sizeof(s->skel_mask));
    memset(s->skel_hp,   0, sizeof(s->skel_hp));
    memset(s->skel_cd,   0, sizeof(s->skel_cd));
    memset(s->arrow_mask, 0, sizeof(s->arrow_mask));
    memset(s->plant_mask, 0, sizeof(s->plant_mask));
    memset(s->plant_age,  0, sizeof(s->plant_age));
    memset(s->achievements, 0, sizeof(s->achievements));
    s->timestep = 0;
    s->light_level = 1.0f;
}

// ============================================================
// Step sub-actions
// ============================================================
static void do_crafting(EnvState* s, int action) {
    bool t = is_near_block(s, BLK_TABLE);
    bool f = is_near_block(s, BLK_FURNACE);

    if (action == ACT_MAKE_WOOD_PICK  && t && s->inv[0] >= 1) { s->inv[0]--; s->inv[6]++; s->achievements[ACH_MAKE_WOOD_PICK] = true; }
    if (action == ACT_MAKE_STONE_PICK && t && s->inv[0] >= 1 && s->inv[1] >= 1) { s->inv[0]--; s->inv[1]--; s->inv[7]++; s->achievements[ACH_MAKE_STONE_PICK] = true; }
    if (action == ACT_MAKE_IRON_PICK  && t && f && s->inv[0] >= 1 && s->inv[1] >= 1 && s->inv[3] >= 1 && s->inv[2] >= 1) {
        s->inv[0]--; s->inv[1]--; s->inv[3]--; s->inv[2]--; s->inv[8]++; s->achievements[ACH_MAKE_IRON_PICK] = true;
    }
    if (action == ACT_MAKE_WOOD_SWORD  && t && s->inv[0] >= 1) { s->inv[0]--; s->inv[9]++;  s->achievements[ACH_MAKE_WOOD_SWORD] = true; }
    if (action == ACT_MAKE_STONE_SWORD && t && s->inv[0] >= 1 && s->inv[1] >= 1) { s->inv[0]--; s->inv[1]--; s->inv[10]++; s->achievements[ACH_MAKE_STONE_SWORD] = true; }
    if (action == ACT_MAKE_IRON_SWORD  && t && f && s->inv[0] >= 1 && s->inv[1] >= 1 && s->inv[3] >= 1 && s->inv[2] >= 1) {
        s->inv[0]--; s->inv[1]--; s->inv[3]--; s->inv[2]--; s->inv[11]++; s->achievements[ACH_MAKE_IRON_SWORD] = true;
    }
}

static void do_action(EnvState* s) {
    int tr = s->player_r + DIR_DR[s->player_dir];
    int tc = s->player_c + DIR_DC[s->player_dir];
    if (!in_bounds(tr, tc)) return;

    int dmg = get_damage(s);
    bool attacked = false;

    for (int i = 0; i < MAX_ZOMBIES && !attacked; i++) {
        if (s->zombie_mask[i] && s->zombie_r[i] == tr && s->zombie_c[i] == tc) {
            s->zombie_hp[i] -= dmg;
            if (s->zombie_hp[i] <= 0) {
                s->zombie_mask[i] = false;
                mb_clear(s->mob_bits, tr, tc);
                mb_clear(s->zombie_bits, tr, tc);
                s->achievements[ACH_DEFEAT_ZOMBIE] = true;
            }
            attacked = true;
        }
    }
    for (int i = 0; i < MAX_COWS && !attacked; i++) {
        if (s->cow_mask[i] && s->cow_r[i] == tr && s->cow_c[i] == tc) {
            s->cow_hp[i] -= dmg;
            if (s->cow_hp[i] <= 0) {
                s->cow_mask[i] = false;
                mb_clear(s->mob_bits, tr, tc);
                mb_clear(s->cow_bits, tr, tc);
                s->achievements[ACH_EAT_COW] = true;
                s->food = (int8_t)min_i(9, s->food + 6);
                s->hunger = 0;
            }
            attacked = true;
        }
    }
    for (int i = 0; i < MAX_SKELETONS && !attacked; i++) {
        if (s->skel_mask[i] && s->skel_r[i] == tr && s->skel_c[i] == tc) {
            s->skel_hp[i] -= dmg;
            if (s->skel_hp[i] <= 0) {
                s->skel_mask[i] = false;
                mb_clear(s->mob_bits, tr, tc);
                mb_clear(s->skel_bits, tr, tc);
                s->achievements[ACH_DEFEAT_SKELETON] = true;
            }
            attacked = true;
        }
    }
    if (attacked) return;

    int8_t blk = map_get(s, tr, tc);
    switch (blk) {
        case BLK_TREE:
            map_set(s, tr, tc, BLK_GRASS);
            s->inv[0] = (int8_t)min_i(9, s->inv[0] + 1);
            s->achievements[ACH_COLLECT_WOOD] = true;
            break;
        case BLK_STONE:
            if (s->inv[6] > 0 || s->inv[7] > 0 || s->inv[8] > 0) {
                map_set(s, tr, tc, BLK_PATH);
                s->inv[1] = (int8_t)min_i(9, s->inv[1] + 1);
                s->achievements[ACH_COLLECT_STONE] = true;
            } break;
        case BLK_COAL:
            if (s->inv[6] > 0 || s->inv[7] > 0 || s->inv[8] > 0) {
                map_set(s, tr, tc, BLK_PATH);
                s->inv[2] = (int8_t)min_i(9, s->inv[2] + 1);
                s->achievements[ACH_COLLECT_COAL] = true;
            } break;
        case BLK_IRON:
            if (s->inv[7] > 0 || s->inv[8] > 0) {
                map_set(s, tr, tc, BLK_PATH);
                s->inv[3] = (int8_t)min_i(9, s->inv[3] + 1);
                s->achievements[ACH_COLLECT_IRON] = true;
            } break;
        case BLK_DIAMOND:
            if (s->inv[8] > 0) {
                map_set(s, tr, tc, BLK_PATH);
                s->inv[4] = (int8_t)min_i(9, s->inv[4] + 1);
                s->achievements[ACH_COLLECT_DIAMOND] = true;
            } break;
        case BLK_GRASS:
            if (rand_f(s) < 0.1f) {
                s->inv[5] = (int8_t)min_i(9, s->inv[5] + 1);
                s->achievements[ACH_COLLECT_SAPLING] = true;
            } break;
        case BLK_WATER:
            s->drink = (int8_t)min_i(9, s->drink + 1);
            s->thirst = 0;
            s->achievements[ACH_COLLECT_DRINK] = true;
            break;
        case BLK_RIPE_PLANT:
            map_set(s, tr, tc, BLK_PLANT);
            s->food = (int8_t)min_i(9, s->food + 4);
            s->hunger = 0;
            s->achievements[ACH_EAT_PLANT] = true;
            for (int i = 0; i < MAX_PLANTS; i++) {
                if (s->plant_mask[i] && s->plant_r[i] == tr && s->plant_c[i] == tc) {
                    s->plant_age[i] = 0; break;
                }
            }
            break;
    }
}

static void place_block(EnvState* s, int action) {
    int tr = s->player_r + DIR_DR[s->player_dir];
    int tc = s->player_c + DIR_DC[s->player_dir];
    if (!in_bounds(tr, tc)) return;
    if (has_mob_at(s, tr, tc)) return;

    int8_t blk = map_get(s, tr, tc);
    if (action == ACT_PLACE_TABLE && s->inv[0] >= 2 && !is_solid(blk)) {
        map_set(s, tr, tc, BLK_TABLE); s->inv[0] -= 2;
        s->achievements[ACH_PLACE_TABLE] = true;
    } else if (action == ACT_PLACE_FURNACE && s->inv[1] >= 1 && !is_solid(blk)) {
        map_set(s, tr, tc, BLK_FURNACE); s->inv[1] -= 1;
        s->achievements[ACH_PLACE_FURNACE] = true;
    } else if (action == ACT_PLACE_STONE && s->inv[1] >= 1 && (!is_solid(blk) || blk == BLK_WATER)) {
        map_set(s, tr, tc, BLK_STONE); s->inv[1] -= 1;
        s->achievements[ACH_PLACE_STONE] = true;
    } else if (action == ACT_PLACE_PLANT && s->inv[5] >= 1 && blk == BLK_GRASS) {
        map_set(s, tr, tc, BLK_PLANT); s->inv[5] -= 1;
        s->achievements[ACH_PLACE_PLANT] = true;
        for (int i = 0; i < MAX_PLANTS; i++) {
            if (!s->plant_mask[i]) {
                s->plant_r[i] = tr; s->plant_c[i] = tc;
                s->plant_age[i] = 0; s->plant_mask[i] = true; break;
            }
        }
    }
}

static void move_player(EnvState* s, int action) {
    if (action < 1 || action > 4) return;
    int nr = s->player_r + DIR_DR[action];
    int nc = s->player_c + DIR_DC[action];
    s->player_dir = (int8_t)action;
    if (!in_bounds(nr, nc)) return;
    if (is_solid(map_get(s, nr, nc))) return;
    if (has_mob_at(s, nr, nc)) return;
    s->player_r = (int16_t)nr; s->player_c = (int16_t)nc;
}

static bool can_move_mob(const EnvState* s, int r, int c) {
    if (!in_bounds(r, c)) return false;
    int8_t blk = map_get(s, r, c);
    if (is_solid(blk)) return false;
    if (blk == BLK_LAVA) return false;
    if (has_mob_at(s, r, c)) return false;
    if (r == s->player_r && c == s->player_c) return false;
    return true;
}

static void update_mobs(EnvState* s) {
    int pr = s->player_r, pc = s->player_c;

    for (int i = 0; i < MAX_ZOMBIES; i++) {
        if (!s->zombie_mask[i]) continue;
        int zr = s->zombie_r[i], zc = s->zombie_c[i];
        int dist = l1_dist(zr, zc, pr, pc);
        if (dist >= MOB_DESPAWN_DIST) {
            s->zombie_mask[i] = false;
            mb_clear(s->mob_bits, zr, zc);
            mb_clear(s->zombie_bits, zr, zc);
            continue;
        }
        if (dist <= 1 && s->zombie_cd[i] <= 0) {
            int dmg = s->is_sleeping ? 7 : 2;
            s->health -= dmg;
            s->zombie_cd[i] = 5;
            s->is_sleeping = false;
        }
        s->zombie_cd[i] = (int8_t)max_i(0, s->zombie_cd[i] - 1);

        int dr = 0, dc = 0;
        if (dist < 10 && rand_f(s) < 0.75f) {
            int adr = abs(pr - zr), adc = abs(pc - zc);
            if (adr > adc || (adr == adc && rand_f(s) < 0.5f)) dr = sign_i(pr - zr);
            else                                                dc = sign_i(pc - zc);
        } else {
            int d = rand_int(s, 4);
            dr = DIR_DR[d+1]; dc = DIR_DC[d+1];
        }
        int nr = zr + dr, nc = zc + dc;
        if (can_move_mob(s, nr, nc)) {
            mb_clear(s->mob_bits, zr, zc); mb_clear(s->zombie_bits, zr, zc);
            s->zombie_r[i] = (int16_t)nr; s->zombie_c[i] = (int16_t)nc;
            mb_set(s->mob_bits, nr, nc);   mb_set(s->zombie_bits, nr, nc);
        }
    }

    for (int i = 0; i < MAX_COWS; i++) {
        if (!s->cow_mask[i]) continue;
        int cr = s->cow_r[i], cc = s->cow_c[i];
        int dist = l1_dist(cr, cc, pr, pc);
        if (dist >= MOB_DESPAWN_DIST) {
            s->cow_mask[i] = false;
            mb_clear(s->mob_bits, cr, cc);
            mb_clear(s->cow_bits, cr, cc);
            continue;
        }
        int d = rand_int(s, 8);
        if (d < 4) {
            int dr = DIR_DR[d+1], dc2 = DIR_DC[d+1];
            int nr = cr + dr, nc = cc + dc2;
            if (can_move_mob(s, nr, nc)) {
                mb_clear(s->mob_bits, cr, cc); mb_clear(s->cow_bits, cr, cc);
                s->cow_r[i] = (int16_t)nr; s->cow_c[i] = (int16_t)nc;
                mb_set(s->mob_bits, nr, nc);   mb_set(s->cow_bits, nr, nc);
            }
        }
    }

    for (int i = 0; i < MAX_SKELETONS; i++) {
        if (!s->skel_mask[i]) continue;
        int sr = s->skel_r[i], sc = s->skel_c[i];
        int dist = l1_dist(sr, sc, pr, pc);
        if (dist >= MOB_DESPAWN_DIST) {
            s->skel_mask[i] = false;
            mb_clear(s->mob_bits, sr, sc);
            mb_clear(s->skel_bits, sr, sc);
            continue;
        }

        if (dist >= 4 && dist <= 5 && s->skel_cd[i] <= 0) {
            for (int a = 0; a < MAX_ARROWS; a++) {
                if (!s->arrow_mask[a]) {
                    s->arrow_mask[a] = true;
                    s->arrow_r[a] = (int16_t)sr; s->arrow_c[a] = (int16_t)sc;
                    mb_set(s->arrow_bits, sr, sc);
                    int adr = abs(pr - sr), adc = abs(pc - sc);
                    s->arrow_dr[a] = (int8_t)((adr > 0) ? sign_i(pr - sr) : 0);
                    s->arrow_dc[a] = (int8_t)((adc > 0) ? sign_i(pc - sc) : 0);
                    break;
                }
            }
            s->skel_cd[i] = 4;
        }
        s->skel_cd[i] = (int8_t)max_i(0, s->skel_cd[i] - 1);

        int dr = 0, dc = 0;
        bool random_move = rand_f(s) < 0.15f;
        if (!random_move) {
            if (dist >= 10) {
                int adr = abs(pr - sr), adc = abs(pc - sc);
                if (adr > adc || (adr == adc && rand_f(s) < 0.5f)) dr = sign_i(pr - sr);
                else                                                dc = sign_i(pc - sc);
            } else if (dist <= 3) {
                int adr = abs(pr - sr), adc = abs(pc - sc);
                if (adr > adc || (adr == adc && rand_f(s) < 0.5f)) dr = -sign_i(pr - sr);
                else                                                dc = -sign_i(pc - sc);
            } else {
                random_move = true;
            }
        }
        if (random_move) {
            int d = rand_int(s, 4);
            dr = DIR_DR[d+1]; dc = DIR_DC[d+1];
        }
        int nr = sr + dr, nc = sc + dc;
        if (can_move_mob(s, nr, nc)) {
            mb_clear(s->mob_bits, sr, sc); mb_clear(s->skel_bits, sr, sc);
            s->skel_r[i] = (int16_t)nr; s->skel_c[i] = (int16_t)nc;
            mb_set(s->mob_bits, nr, nc);   mb_set(s->skel_bits, nr, nc);
        }
    }

    for (int i = 0; i < MAX_ARROWS; i++) {
        if (!s->arrow_mask[i]) continue;
        int ar = s->arrow_r[i], ac = s->arrow_c[i];
        int nr = ar + s->arrow_dr[i];
        int nc = ac + s->arrow_dc[i];
        if (!in_bounds(nr, nc)) { s->arrow_mask[i] = false; mb_clear(s->arrow_bits, ar, ac); continue; }
        int8_t blk = map_get(s, nr, nc);
        if (is_solid(blk) && blk != BLK_WATER) {
            if (blk == BLK_FURNACE || blk == BLK_TABLE) map_set(s, nr, nc, BLK_PATH);
            s->arrow_mask[i] = false; mb_clear(s->arrow_bits, ar, ac); continue;
        }
        if (nr == pr && nc == pc) {
            s->health -= 2; s->is_sleeping = false;
            s->arrow_mask[i] = false; mb_clear(s->arrow_bits, ar, ac); continue;
        }
        mb_clear(s->arrow_bits, ar, ac);
        s->arrow_r[i] = (int16_t)nr; s->arrow_c[i] = (int16_t)nc;
        mb_set(s->arrow_bits, nr, nc);
    }
}

static bool try_spawn(EnvState* s, int min_dist, int max_dist,
                     bool need_grass, bool need_path, int* out_r, int* out_c) {
    int pr = s->player_r, pc = s->player_c;
    for (int att = 0; att < 20; att++) {
        int r = rand_int(s, MAP_SIZE), c = rand_int(s, MAP_SIZE);
        int dist = l1_dist(r, c, pr, pc);
        if (dist < min_dist || dist >= max_dist) continue;
        if (has_mob_at(s, r, c)) continue;
        if (r == pr && c == pc) continue;
        int8_t blk = map_get(s, r, c);
        if (need_grass && blk != BLK_GRASS) continue;
        if (need_path && blk != BLK_PATH) continue;
        if (!need_grass && !need_path && blk != BLK_GRASS && blk != BLK_PATH) continue;
        *out_r = r; *out_c = c; return true;
    }
    return false;
}

static void spawn_mobs(EnvState* s) {
    int n_cows = 0, n_zombies = 0, n_skels = 0;
    for (int i = 0; i < MAX_COWS;      i++) n_cows    += s->cow_mask[i];
    for (int i = 0; i < MAX_ZOMBIES;   i++) n_zombies += s->zombie_mask[i];
    for (int i = 0; i < MAX_SKELETONS; i++) n_skels   += s->skel_mask[i];

    if (n_cows < MAX_COWS && rand_f(s) < 0.1f) {
        int r, c;
        if (try_spawn(s, 3, MOB_DESPAWN_DIST, true, false, &r, &c)) {
            for (int i = 0; i < MAX_COWS; i++) if (!s->cow_mask[i]) {
                s->cow_mask[i] = true; s->cow_r[i] = (int16_t)r; s->cow_c[i] = (int16_t)c; s->cow_hp[i] = 3;
                mb_set(s->mob_bits, r, c); mb_set(s->cow_bits, r, c);
                break;
            }
        }
    }
    float zombie_chance = 0.02f + 0.1f * (1.0f - s->light_level) * (1.0f - s->light_level);
    if (n_zombies < MAX_ZOMBIES && rand_f(s) < zombie_chance) {
        int r, c;
        if (try_spawn(s, 9, MOB_DESPAWN_DIST, false, false, &r, &c)) {
            for (int i = 0; i < MAX_ZOMBIES; i++) if (!s->zombie_mask[i]) {
                s->zombie_mask[i] = true; s->zombie_r[i] = (int16_t)r; s->zombie_c[i] = (int16_t)c;
                s->zombie_hp[i] = 5; s->zombie_cd[i] = 0;
                mb_set(s->mob_bits, r, c); mb_set(s->zombie_bits, r, c);
                break;
            }
        }
    }
    if (n_skels < MAX_SKELETONS && rand_f(s) < 0.05f) {
        int r, c;
        if (try_spawn(s, 9, MOB_DESPAWN_DIST, false, true, &r, &c)) {
            for (int i = 0; i < MAX_SKELETONS; i++) if (!s->skel_mask[i]) {
                s->skel_mask[i] = true; s->skel_r[i] = (int16_t)r; s->skel_c[i] = (int16_t)c;
                s->skel_hp[i] = 3; s->skel_cd[i] = 0;
                mb_set(s->mob_bits, r, c); mb_set(s->skel_bits, r, c);
                break;
            }
        }
    }
}

static void update_plants(EnvState* s) {
    for (int i = 0; i < MAX_PLANTS; i++) {
        if (!s->plant_mask[i]) continue;
        s->plant_age[i]++;
        if (s->plant_age[i] >= 600) {
            int r = s->plant_r[i], c = s->plant_c[i];
            if (in_bounds(r, c) && map_get(s, r, c) == BLK_PLANT)
                map_set(s, r, c, BLK_RIPE_PLANT);
        }
    }
}

static void update_intrinsics(EnvState* s, int action) {
    if (action == ACT_SLEEP && s->energy < 9) s->is_sleeping = true;
    if (s->energy >= 9 && s->is_sleeping) {
        s->is_sleeping = false;
        s->achievements[ACH_WAKE_UP] = true;
    }
    float sleep_mul = s->is_sleeping ? 0.5f : 1.0f;

    s->hunger += sleep_mul;
    if (s->hunger > 25.0f) { s->food--; s->hunger = 0; }

    s->thirst += sleep_mul;
    if (s->thirst > 20.0f) { s->drink--; s->thirst = 0; }

    if (s->is_sleeping) s->fatigue -= 1.0f; else s->fatigue += 1.0f;
    if (s->fatigue > 30.0f)   { s->energy--; s->fatigue = 0; }
    if (s->fatigue < -10.0f)  { s->energy = (int8_t)min_i(s->energy + 1, 9); s->fatigue = 0; }

    bool all_needs = (s->food > 0) && (s->drink > 0) && (s->energy > 0 || s->is_sleeping);
    if (all_needs) s->recover += s->is_sleeping ? 2.0f : 1.0f;
    else           s->recover += s->is_sleeping ? -0.5f : -1.0f;
    if (s->recover > 25.0f)  { s->health = (int8_t)min_i(s->health + 1, 9); s->recover = 0; }
    if (s->recover < -15.0f) { s->health--; s->recover = 0; }
}

// ============================================================
// Observation
// ============================================================
void craftax_build_obs(const EnvState* s, float* obs) {
    int pr = s->player_r, pc = s->player_c;
    int idx = 0;
    for (int dr = -3; dr <= 3; dr++) {
        for (int dc = -4; dc <= 4; dc++) {
            int r = pr + dr, c = pc + dc;
            int8_t blk = in_bounds(r, c) ? map_get(s, r, c) : BLK_OUT_OF_BOUNDS;
            // SIMD-friendly: wide zero then single scalar set.
            // 17 floats = 68B; compiler emits one AVX-512 ZMM store + tail.
            float* dst = obs + idx;
            for (int b = 0; b < NUM_BLOCK_TYPES; b++) dst[b] = 0.0f;
            dst[blk] = 1.0f;
            idx += NUM_BLOCK_TYPES;

            float mz = 0, mc = 0, ms = 0, ma = 0;
            if (in_bounds(r, c)) {
                mz = (float)mb_get(s->zombie_bits, r, c);
                mc = (float)mb_get(s->cow_bits,    r, c);
                ms = (float)mb_get(s->skel_bits,   r, c);
                ma = (float)mb_get(s->arrow_bits,  r, c);
            }
            obs[idx++] = mz; obs[idx++] = mc; obs[idx++] = ms; obs[idx++] = ma;
        }
    }
    for (int i = 0; i < NUM_INVENTORY; i++) obs[idx++] = (float)s->inv[i] / 10.0f;
    obs[idx++] = (float)s->health / 10.0f;
    obs[idx++] = (float)s->food   / 10.0f;
    obs[idx++] = (float)s->drink  / 10.0f;
    obs[idx++] = (float)s->energy / 10.0f;
    for (int d = 1; d <= 4; d++) obs[idx++] = (s->player_dir == d) ? 1.0f : 0.0f;
    obs[idx++] = s->light_level;
    obs[idx++] = s->is_sleeping ? 1.0f : 0.0f;
}

void craftax_build_obs_compact(const EnvState* s, uint8_t* obs) {
    int pr = s->player_r, pc = s->player_c;
    uint8_t mobs[OBS_MAP_ROWS * OBS_MAP_COLS];
    int tile = 0;
    for (int dr = -3; dr <= 3; dr++) {
        int r = pr + dr;
        bool row_ok = (unsigned)r < MAP_SIZE;
        // Extract 9 consecutive columns from each per-type row bitmap in one shot.
        uint64_t zb = row_ok ? s->zombie_bits[r] : 0;
        uint64_t cb = row_ok ? s->cow_bits[r]    : 0;
        uint64_t sb = row_ok ? s->skel_bits[r]   : 0;
        uint64_t ab = row_ok ? s->arrow_bits[r]  : 0;
        for (int dc = -4; dc <= 4; dc++) {
            int c = pc + dc;
            int8_t blk = (row_ok && (unsigned)c < MAP_SIZE) ? map_get(s, r, c) : BLK_OUT_OF_BOUNDS;
            obs[tile] = (uint8_t)blk;

            uint8_t m = 0;
            if (row_ok && (unsigned)c < MAP_SIZE) {
                uint64_t bit = 1ULL << c;
                m |= (zb & bit) ? 1 : 0;
                m |= (cb & bit) ? 2 : 0;
                m |= (sb & bit) ? 4 : 0;
                m |= (ab & bit) ? 8 : 0;
            }
            mobs[tile] = m;
            tile++;
        }
    }
    memcpy(obs + 63, mobs, 63);

    uint8_t* out = obs + 126;
    for (int i = 0; i < NUM_INVENTORY; i++) *out++ = (uint8_t)s->inv[i];
    *out++ = (uint8_t)s->health;
    *out++ = (uint8_t)s->food;
    *out++ = (uint8_t)s->drink;
    *out++ = (uint8_t)s->energy;
    *out++ = (uint8_t)s->player_dir;
    *out++ = (uint8_t)(s->is_sleeping ? 1 : 0);
    // light_level in [0,1] -> [0,255]
    float ll = s->light_level;
    if (ll < 0) ll = 0; else if (ll > 1.0f) ll = 1.0f;
    *out++ = (uint8_t)(ll * 255.0f + 0.5f);
}

// ============================================================
// Public single-env API
// ============================================================
void craftax_reset(EnvState* s, uint64_t seed, uint64_t env_id) {
    generate_world(s, seed, env_id);
}

void craftax_step(EnvState* s, int action, float* reward, int* done) {
    int old_health = s->health;
    bool old_ach[NUM_ACHIEVEMENTS];
    memcpy(old_ach, s->achievements, sizeof(old_ach));

    int eff_action = s->is_sleeping ? ACT_NOOP : action;

    do_crafting(s, eff_action);
    if (eff_action == ACT_DO) do_action(s);
    if (eff_action >= ACT_PLACE_STONE && eff_action <= ACT_PLACE_PLANT) place_block(s, eff_action);
    move_player(s, eff_action);
    update_mobs(s);
    spawn_mobs(s);
    update_plants(s);
    update_intrinsics(s, action);

    for (int i = 0; i < NUM_INVENTORY; i++) s->inv[i] = (int8_t)clamp_i(s->inv[i], 0, 9);

    s->timestep++;
    float t_frac = fmodf((float)s->timestep / (float)DAY_LENGTH, 1.0f) + 0.3f;
    float cv = cosf(3.14159265f * t_frac);
    s->light_level = 1.0f - fabsf(cv * cv * cv);

    float ach_r = 0;
    for (int i = 0; i < NUM_ACHIEVEMENTS; i++)
        ach_r += (float)(s->achievements[i] && !old_ach[i]);
    float hp_r = (float)(s->health - old_health) * 0.1f;
    *reward = ach_r + hp_r;

    bool d = (s->timestep >= MAX_TIMESTEPS) || (s->health <= 0);
    if (in_bounds(s->player_r, s->player_c) && map_get(s, s->player_r, s->player_c) == BLK_LAVA) d = true;
    *done = d ? 1 : 0;
}

// ============================================================
// Batched API (OpenMP)
// ============================================================
void craftax_reset_batch(EnvState* states, float* obs, int num_envs, uint64_t seed) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_envs; i++) {
        craftax_reset(&states[i], seed, (uint64_t)i);
        craftax_build_obs(&states[i], obs + (size_t)i * OBS_DIM);
    }
}

void craftax_step_batch(EnvState* states, const int32_t* actions,
                        float* obs, float* rewards, int8_t* dones,
                        int num_envs, uint64_t reset_seed) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_envs; i++) {
        float r; int d;
        craftax_step(&states[i], (int)actions[i], &r, &d);
        rewards[i] = r;
        dones[i] = (int8_t)d;
        if (d) craftax_reset(&states[i], reset_seed, (uint64_t)i + (uint64_t)num_envs);
        craftax_build_obs(&states[i], obs + (size_t)i * OBS_DIM);
    }
}

// ============================================================
// Thread-pool variant: no OpenMP, uses custom spin barrier.
// ============================================================
typedef struct {
    EnvState* states;
    const int32_t* actions;
    uint8_t* obs_compact;
    float* obs_float;
    float* rewards;
    int8_t* dones;
    int num_envs;
    uint64_t reset_seed;
} StepWork;

static void step_compact_worker(void* arg, int id, int nw, int begin, int end) {
    (void)id; (void)nw;
    StepWork* w = (StepWork*)arg;
    for (int i = begin; i < end; i++) {
        float r; int d;
        craftax_step(&w->states[i], (int)w->actions[i], &r, &d);
        w->rewards[i] = r;
        w->dones[i] = (int8_t)d;
        if (d) craftax_reset(&w->states[i], w->reset_seed,
                             (uint64_t)i + (uint64_t)w->num_envs);
        craftax_build_obs_compact(&w->states[i],
            w->obs_compact + (size_t)i * OBS_DIM_COMPACT);
    }
}

void craftax_step_batch_compact_tp(struct ThreadPool* tp,
                                   EnvState* states, const int32_t* actions,
                                   uint8_t* obs, float* rewards, int8_t* dones,
                                   int num_envs, uint64_t reset_seed) {
    StepWork w = {
        .states = states, .actions = actions,
        .obs_compact = obs, .obs_float = NULL,
        .rewards = rewards, .dones = dones,
        .num_envs = num_envs, .reset_seed = reset_seed,
    };
    worker_pool_run(tp, num_envs, step_compact_worker, &w);
}

void craftax_reset_batch_compact(EnvState* states, uint8_t* obs, int num_envs, uint64_t seed) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_envs; i++) {
        craftax_reset(&states[i], seed, (uint64_t)i);
        craftax_build_obs_compact(&states[i], obs + (size_t)i * OBS_DIM_COMPACT);
    }
}

void craftax_step_batch_compact(EnvState* states, const int32_t* actions,
                                uint8_t* obs, float* rewards, int8_t* dones,
                                int num_envs, uint64_t reset_seed) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_envs; i++) {
        float r; int d;
        craftax_step(&states[i], (int)actions[i], &r, &d);
        rewards[i] = r;
        dones[i] = (int8_t)d;
        if (d) craftax_reset(&states[i], reset_seed, (uint64_t)i + (uint64_t)num_envs);
        craftax_build_obs_compact(&states[i], obs + (size_t)i * OBS_DIM_COMPACT);
    }
}

// ============ from worker_pool.c ============
// Implementation: persistent worker threads spin on a generation counter.
// Main publishes work (fn/arg/total + bump generation); each worker grabs its
// slice, executes, and increments done_count. Main spins on done_count.
//
// Padding ensures done_count and generation don't share a cache line with
// other state (false-sharing kills spin-barrier latency).
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

#define CACHE_LINE 64
#define CL_ALIGN __attribute__((aligned(CACHE_LINE)))

struct ThreadPool {
    int num_workers;
    pthread_t* threads;
    int* cpus;

    // Hot control state on separate cache lines (avoid false sharing).
    CL_ALIGN _Atomic uint64_t generation;
    char _pad0[CACHE_LINE - sizeof(_Atomic uint64_t)];
    CL_ALIGN _Atomic int done_count;
    char _pad1[CACHE_LINE - sizeof(_Atomic int)];
    CL_ALIGN _Atomic int shutdown;
    char _pad2[CACHE_LINE - sizeof(_Atomic int)];

    // Payload (written by main before bumping generation, read by workers).
    work_fn_t fn;
    void* arg;
    int total;
} CL_ALIGN;

typedef struct { ThreadPool* pool; int id; } WorkerArg;

static void pin_cpu(int cpu) {
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
    pthread_setaffinity_np(pthread_self(), sizeof(s), &s);
}

static void* worker_main(void* arg_) {
    WorkerArg* a = (WorkerArg*)arg_;
    ThreadPool* p = a->pool;
    int id = a->id;
    if (p->cpus) pin_cpu(p->cpus[id]);

    uint64_t seen = 0;
    for (;;) {
        // Spin until a new generation is published.
        uint64_t g;
        for (;;) {
            g = atomic_load_explicit(&p->generation, memory_order_acquire);
            if (g != seen) break;
            if (atomic_load_explicit(&p->shutdown, memory_order_relaxed)) {
                free(a); return NULL;
            }
            _mm_pause();
        }
        seen = g;
        if (atomic_load_explicit(&p->shutdown, memory_order_relaxed)) { free(a); return NULL; }

        int total = p->total;
        int nw = p->num_workers;
        // Balanced chunking: workers < (total%nw) take chunk+1, rest take chunk.
        int base = total / nw;
        int rem  = total - base * nw;
        int begin = id * base + (id < rem ? id : rem);
        int extra = (id < rem) ? 1 : 0;
        int end = begin + base + extra;
        if (begin < end) p->fn(p->arg, id, nw, begin, end);

        atomic_fetch_add_explicit(&p->done_count, 1, memory_order_release);
    }
}

ThreadPool* worker_pool_create(int num_workers, const int* cpus) {
    ThreadPool* p = (ThreadPool*)aligned_alloc(CACHE_LINE, sizeof(*p));
    memset(p, 0, sizeof(*p));
    p->num_workers = num_workers;
    p->threads = (pthread_t*)calloc((size_t)num_workers, sizeof(pthread_t));
    if (cpus) {
        p->cpus = (int*)malloc((size_t)num_workers * sizeof(int));
        memcpy(p->cpus, cpus, (size_t)num_workers * sizeof(int));
    }
    atomic_store(&p->generation, 0);
    atomic_store(&p->done_count, 0);
    atomic_store(&p->shutdown, 0);
    for (int i = 0; i < num_workers; i++) {
        WorkerArg* a = (WorkerArg*)malloc(sizeof(*a));
        a->pool = p; a->id = i;
        pthread_create(&p->threads[i], NULL, worker_main, a);
    }
    return p;
}

void worker_pool_destroy(ThreadPool* p) {
    if (!p) return;
    atomic_store(&p->shutdown, 1);
    // Wake workers so they observe shutdown.
    atomic_fetch_add_explicit(&p->generation, 1, memory_order_release);
    for (int i = 0; i < p->num_workers; i++) pthread_join(p->threads[i], NULL);
    free(p->threads);
    free(p->cpus);
    free(p);
}

void worker_pool_run(ThreadPool* p, int total, work_fn_t fn, void* arg) {
    p->fn = fn; p->arg = arg; p->total = total;
    // Reset done_count, then publish new generation (release ordering so
    // workers see fn/arg/total after they see the bumped generation).
    atomic_store_explicit(&p->done_count, 0, memory_order_relaxed);
    atomic_fetch_add_explicit(&p->generation, 1, memory_order_release);
    int nw = p->num_workers;
    // Spin until all workers have incremented done_count.
    for (;;) {
        int d = atomic_load_explicit(&p->done_count, memory_order_acquire);
        if (d >= nw) break;
        _mm_pause();
    }
}

int worker_pool_num_workers(const ThreadPool* p) { return p->num_workers; }

// ============ from craftax_pool.c ============
// Pipelined world-generation pool.
// Producer pthreads (pinned to SMT siblings) continuously pre-generate worlds
// and publish them to a ring buffer. Consumers pop from the ring on reset.
//
// Slot state machine:  EMPTY (0) -> FILLING (1) -> READY (2) -> DRAINING (3) -> EMPTY
// Producers CAS EMPTY->FILLING, fill the slot, store READY.
// Consumers CAS READY->DRAINING, copy out, store EMPTY.
//
// produce_hint/consume_hint are non-authoritative atomics used as rolling
// starting points so producers and consumers don't collide on the same slots.
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

extern void _craftax_generate_world(EnvState* s, uint64_t seed, uint64_t env_id);

enum { SLOT_EMPTY=0, SLOT_FILLING=1, SLOT_READY=2, SLOT_DRAINING=3 };

struct WorldPool {
    int capacity;
    int num_producers;
    uint64_t master_seed;

    EnvState* worlds;               // [capacity]
    _Atomic uint32_t* state;        // [capacity]

    _Atomic uint64_t produce_hint;
    _Atomic uint64_t consume_hint;
    _Atomic uint64_t producer_env_id;   // monotonically-increasing env_id for seeding

    _Atomic uint64_t total_produced;
    _Atomic uint64_t total_consumed;
    _Atomic uint64_t total_fallbacks;

    _Atomic int running;
    pthread_t* threads;
};

static void pin_to_cpu(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

typedef struct { WorldPool* pool; int producer_idx; int cpu; } ProducerArg;

static void* producer_main(void* arg_) {
    ProducerArg* a = (ProducerArg*)arg_;
    WorldPool* p = a->pool;
    if (a->cpu >= 0) pin_to_cpu(a->cpu);

    while (atomic_load_explicit(&p->running, memory_order_relaxed)) {
        // Find an EMPTY slot starting from the rolling hint.
        uint64_t h = atomic_fetch_add_explicit(&p->produce_hint, 1, memory_order_relaxed);
        uint32_t idx = (uint32_t)(h % (uint64_t)p->capacity);

        uint32_t expect = SLOT_EMPTY;
        int claimed = atomic_compare_exchange_strong_explicit(
            &p->state[idx], &expect, SLOT_FILLING,
            memory_order_acquire, memory_order_relaxed);

        if (!claimed) {
            // Either filling/ready/draining -- try again with next hint.
            // If the whole ring is full (READY everywhere), yield briefly.
            if (expect == SLOT_READY) {
                // Pool full; back off a bit.
                for (int k = 0; k < 64; k++) __builtin_ia32_pause();
            }
            continue;
        }

        // Fill the slot.
        uint64_t eid = atomic_fetch_add_explicit(&p->producer_env_id, 1, memory_order_relaxed);
        _craftax_generate_world(&p->worlds[idx], p->master_seed, eid);

        atomic_store_explicit(&p->state[idx], SLOT_READY, memory_order_release);
        atomic_fetch_add_explicit(&p->total_produced, 1, memory_order_relaxed);
    }
    free(a);
    return NULL;
}

// Try to pop one ready world into dst. Returns true on success.
static inline bool pool_try_pop(WorldPool* p, EnvState* dst) {
    // Try up to capacity slots before giving up.
    for (int tries = 0; tries < p->capacity; tries++) {
        uint64_t h = atomic_fetch_add_explicit(&p->consume_hint, 1, memory_order_relaxed);
        uint32_t idx = (uint32_t)(h % (uint64_t)p->capacity);

        uint32_t expect = SLOT_READY;
        if (atomic_compare_exchange_strong_explicit(
                &p->state[idx], &expect, SLOT_DRAINING,
                memory_order_acquire, memory_order_relaxed)) {
            memcpy(dst, &p->worlds[idx], sizeof(EnvState));
            atomic_store_explicit(&p->state[idx], SLOT_EMPTY, memory_order_release);
            atomic_fetch_add_explicit(&p->total_consumed, 1, memory_order_relaxed);
            return true;
        }
    }
    return false;
}

WorldPool* craftax_pool_create(int capacity, int num_producers, uint64_t master_seed) {
    WorldPool* p = (WorldPool*)calloc(1, sizeof(*p));
    p->capacity = capacity;
    p->num_producers = num_producers;
    p->master_seed = master_seed;
    p->worlds = (EnvState*)aligned_alloc(64, sizeof(EnvState) * (size_t)capacity);
    p->state  = (_Atomic uint32_t*)aligned_alloc(64, sizeof(uint32_t) * (size_t)capacity);
    memset(p->worlds, 0, sizeof(EnvState) * (size_t)capacity);
    for (int i = 0; i < capacity; i++)
        atomic_store_explicit(&p->state[i], SLOT_EMPTY, memory_order_relaxed);
    atomic_store(&p->running, 1);

    // SMT siblings of physical cores live at +16 on this CPU topology:
    //   CCD0: cores 0..7  (SMT 16..23)
    //   CCD1: cores 8..15 (SMT 24..31)
    // Producers go on CCD1's SMT siblings (24..31), so consumers on the
    // physical cores can run steady-state step logic unimpeded.
    // Producer CPU list from env var CRAFTAX_POOL_CPUS (e.g. "12,13,14,15").
    // Default: CCD1 SMT siblings 24..31 (may collide with OMP, slow).
    const char* cpus_env = getenv("CRAFTAX_POOL_CPUS");
    int cpus[64]; int ncpus = 0;
    if (cpus_env && *cpus_env) {
        const char* s = cpus_env;
        while (*s && ncpus < 64) {
            while (*s == ',' || *s == ' ') s++;
            if (!*s) break;
            cpus[ncpus++] = atoi(s);
            while (*s && *s != ',') s++;
        }
    }
    p->threads = (pthread_t*)calloc((size_t)num_producers, sizeof(pthread_t));
    for (int i = 0; i < num_producers; i++) {
        ProducerArg* a = (ProducerArg*)malloc(sizeof(*a));
        a->pool = p;
        a->producer_idx = i;
        a->cpu = ncpus > 0 ? cpus[i % ncpus] : (24 + (i % 8));
        pthread_create(&p->threads[i], NULL, producer_main, a);
    }

    // Pre-fill: wait until pool has at least 3/4 ready worlds (bounded).
    for (int waits = 0; waits < 2000; waits++) {
        int ready = 0;
        for (int i = 0; i < capacity; i++)
            if (atomic_load_explicit(&p->state[i], memory_order_relaxed) == SLOT_READY) ready++;
        if (ready >= 3 * capacity / 4) break;
        usleep(1000);
    }
    return p;
}

void craftax_pool_destroy(WorldPool* p) {
    if (!p) return;
    atomic_store(&p->running, 0);
    for (int i = 0; i < p->num_producers; i++) pthread_join(p->threads[i], NULL);
    free(p->threads);
    free(p->worlds);
    free((void*)p->state);
    free(p);
}

void craftax_pool_stats(WorldPool* p, uint64_t* produced, uint64_t* consumed,
                        uint64_t* fallbacks, int* ready_count) {
    if (produced)  *produced  = atomic_load_explicit(&p->total_produced,  memory_order_relaxed);
    if (consumed)  *consumed  = atomic_load_explicit(&p->total_consumed,  memory_order_relaxed);
    if (fallbacks) *fallbacks = atomic_load_explicit(&p->total_fallbacks, memory_order_relaxed);
    if (ready_count) {
        int r = 0;
        for (int i = 0; i < p->capacity; i++)
            if (atomic_load_explicit(&p->state[i], memory_order_relaxed) == SLOT_READY) r++;
        *ready_count = r;
    }
}

void craftax_step_batch_compact_pool(
    WorldPool* pool,
    EnvState* states, const int32_t* actions,
    uint8_t* obs, float* rewards, int8_t* dones,
    int num_envs)
{
    uint64_t local_fallbacks = 0;
    #pragma omp parallel for schedule(static) reduction(+:local_fallbacks)
    for (int i = 0; i < num_envs; i++) {
        float r; int d;
        craftax_step(&states[i], (int)actions[i], &r, &d);
        rewards[i] = r;
        dones[i] = (int8_t)d;
        if (d) {
            if (!pool_try_pop(pool, &states[i])) {
                _craftax_generate_world(&states[i],
                    pool->master_seed,
                    atomic_fetch_add_explicit(&pool->producer_env_id, 1, memory_order_relaxed));
                local_fallbacks++;
            }
        }
        craftax_build_obs_compact(&states[i], obs + (size_t)i * OBS_DIM_COMPACT);
    }
    atomic_fetch_add_explicit(&pool->total_fallbacks, local_fallbacks, memory_order_relaxed);
}

// ============================================================
// Thread-pool + world-pool combined variant.
// ============================================================
typedef struct {
    WorldPool* wpool;
    EnvState* states;
    const int32_t* actions;
    uint8_t* obs;
    float* rewards;
    int8_t* dones;
    _Atomic uint64_t fallbacks;
} PoolStepWork;

static void pool_step_worker(void* arg, int id, int nw, int begin, int end) {
    (void)id; (void)nw;
    PoolStepWork* w = (PoolStepWork*)arg;
    uint64_t local_fb = 0;
    for (int i = begin; i < end; i++) {
        float r; int d;
        craftax_step(&w->states[i], (int)w->actions[i], &r, &d);
        w->rewards[i] = r;
        w->dones[i] = (int8_t)d;
        if (d) {
            if (!pool_try_pop(w->wpool, &w->states[i])) {
                _craftax_generate_world(&w->states[i],
                    w->wpool->master_seed,
                    atomic_fetch_add_explicit(&w->wpool->producer_env_id, 1,
                                              memory_order_relaxed));
                local_fb++;
            }
        }
        craftax_build_obs_compact(&w->states[i],
            w->obs + (size_t)i * OBS_DIM_COMPACT);
    }
    atomic_fetch_add_explicit(&w->fallbacks, local_fb, memory_order_relaxed);
}

void craftax_step_batch_compact_pool_tp(struct ThreadPool* tp, WorldPool* pool,
                                        EnvState* states, const int32_t* actions,
                                        uint8_t* obs, float* rewards, int8_t* dones,
                                        int num_envs) {
    PoolStepWork w = {
        .wpool = pool, .states = states, .actions = actions,
        .obs = obs, .rewards = rewards, .dones = dones,
    };
    atomic_store(&w.fallbacks, 0);
    worker_pool_run(tp, num_envs, pool_step_worker, &w);
    atomic_fetch_add_explicit(&pool->total_fallbacks,
        atomic_load(&w.fallbacks), memory_order_relaxed);
}

// ============ from bench_tp_pool.c ============
// Best-of-both: custom thread pool + world pool pipeline.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

int craftax_cpu_main(int NE, int ITERS) {

    EnvState* st = aligned_alloc(64, sizeof(EnvState)*(size_t)NE);
    uint8_t*  ob = aligned_alloc(64, (size_t)NE*OBS_DIM_COMPACT);
    float*    rw = aligned_alloc(64, sizeof(float)*(size_t)NE);
    int8_t*   dn = aligned_alloc(64, NE);
    int32_t*  ac = aligned_alloc(64, sizeof(int32_t)*(size_t)NE);
    memset(st, 0, sizeof(EnvState)*(size_t)NE);

    pcg32_t r; pcg32_seed(&r,42,1);
    for (int i=0;i<NE;i++) ac[i] = pcg32_next(&r)%NUM_ACTIONS;

    // --- A: libgomp + inline reset (old baseline) ---
    craftax_reset_batch_compact(st, ob, NE, 42);
    for (int w=0;w<20;w++) craftax_step_batch_compact(st,ac,ob,rw,dn,NE,42+w);
    double t0 = now_s();
    for (int k=0;k<ITERS;k++) craftax_step_batch_compact(st,ac,ob,rw,dn,NE,42+k+100);
    double dt_A = now_s()-t0;

    // --- B: custom threadpool + inline reset ---
    int cpus16[16]; for (int i=0;i<16;i++) cpus16[i] = i;
    ThreadPool* tp16 = worker_pool_create(16, cpus16);
    memset(st, 0, sizeof(EnvState)*(size_t)NE);
    craftax_reset_batch_compact(st, ob, NE, 42);
    for (int w=0;w<20;w++) craftax_step_batch_compact_tp(tp16,st,ac,ob,rw,dn,NE,42+w);
    t0 = now_s();
    for (int k=0;k<ITERS;k++) craftax_step_batch_compact_tp(tp16,st,ac,ob,rw,dn,NE,42+k+100);
    double dt_B = now_s()-t0;
    worker_pool_destroy(tp16);

    // --- C: custom threadpool (CCD0 phys+SMT) + world pool (CCD1 producers) ---
    // 16 consumer threads on cores {0..7, 16..23}, 8 producers on {8..15}.
    setenv("CRAFTAX_POOL_CPUS", "8,9,10,11,12,13,14,15", 1);
    int ccpus[16] = {0,1,2,3,4,5,6,7, 16,17,18,19,20,21,22,23};
    ThreadPool* tp_ccd0 = worker_pool_create(16, ccpus);
    WorldPool* wp = craftax_pool_create(4096, 8, 42);

    memset(st, 0, sizeof(EnvState)*(size_t)NE);
    craftax_reset_batch_compact(st, ob, NE, 42);
    for (int w=0;w<20;w++) craftax_step_batch_compact_pool_tp(tp_ccd0, wp, st,ac,ob,rw,dn,NE);
    uint64_t p0,c0,f0; int r0;
    craftax_pool_stats(wp, &p0, &c0, &f0, &r0);
    t0 = now_s();
    for (int k=0;k<ITERS;k++) craftax_step_batch_compact_pool_tp(tp_ccd0, wp, st,ac,ob,rw,dn,NE);
    double dt_C = now_s()-t0;
    uint64_t p1,c1,f1; int r1;
    craftax_pool_stats(wp, &p1, &c1, &f1, &r1);
    craftax_pool_destroy(wp);
    worker_pool_destroy(tp_ccd0);

    printf("NE=%d iters=%d\n", NE, ITERS);
    printf("  A  libgomp + inline reset (16 cores):        %.3fs  SPS=%10.0f  (1.00x)\n",
           dt_A, (double)NE*ITERS/dt_A);
    printf("  B  threadpool + inline reset (16 cores):     %.3fs  SPS=%10.0f  (%.2fx)\n",
           dt_B, (double)NE*ITERS/dt_B, dt_A/dt_B);
    printf("  C  threadpool + world pool (16C+8P):         %.3fs  SPS=%10.0f  (%.2fx)\n",
           dt_C, (double)NE*ITERS/dt_C, dt_A/dt_C);
    printf("     pool: consumed=%lu produced=%lu fallbacks=%lu ready=%d\n",
           c1-c0, p1-p0, f1-f0, r1);
    return 0;
}


#ifdef CRAFTAX_STANDALONE
int main(int argc, char** argv) {
    int envs = 32768, iters = 5000;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--envs") && i + 1 < argc) envs = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--backend") && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "cpu") != 0) {
                fprintf(stderr, "this build is CPU-only (no nvcc at compile time)\n");
                return 1;
            }
        }
        else if (!strcmp(argv[i], "bench")) {}  // only mode on CPU
        else {
            fprintf(stderr, "usage: %s [bench] [--envs N] [--iters N]\n", argv[0]);
            return 1;
        }
    }
    return craftax_cpu_main(envs, iters);
}
#endif
