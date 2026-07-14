// craftax_full.cu -- CUDA port of the full Craftax game (milestone 1:
// correctness only; one thread per env, naive AoS CraftaxState in global
// memory).
//
// GENERATED mechanically from craftax_full.c by scratchpad gen_cu.py:
//   - game logic kept token-identical except `static ` -> `static __device__ `
//     at line starts, `#pragma omp` lines dropped, and the targeted edits
//     marked "[cuda port]" below (host-alloc conveniences, light-level table,
//     pool-free reset-on-done).
//   - the pthread world pool and the OpenMP CLI harness are replaced by a
//     CUDA host harness at the bottom of this file (hash / cmp / stats /
//     bench modes; hash mode computes the same FNV-1a trajectory hash over
//     (obs, rewards, dones) as ./craftax_full hash).
//
// Build:  make craftax_full_cuda   (nvcc -O3 -arch=native --expt-relaxed-constexpr)
// Usage:  ./craftax_full_cuda hash  --envs N --steps M [--seed S]
//         ./craftax_full_cuda cmp   --dump FILE
//         ./craftax_full_cuda stats --envs N --steps M [--seed S]
//         ./craftax_full_cuda bench --envs N --iters M [--seed S]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>

// [cuda port] compatibility shims for the transformed C body below.
// gcc atomics: only touched by the single-threaded global-init kernel and
// (afterwards) read-only, so plain accesses are sufficient.
#define __atomic_load_n(ptr, order) (*(ptr))
#define __atomic_store_n(ptr, val, order) ((void)(*(ptr) = (val)))
// device ctz for the spawn bitboard scans (callers guarantee x != 0)
static __device__ inline int32_t craftax_cu_ctzll(unsigned long long x) {
    return __ffsll((long long)x) - 1;
}
#define __builtin_ctzll(x) craftax_cu_ctzll(x)
// never take the AVX-512 host paths
#define CRAFTAX_NO_SIMD_NOISE 1

// ============================================================
// ===== device game logic, transformed from craftax_full.c =====
// ============================================================
// craftax_full.c -- standalone single-file C port of the full Craftax game.
//
// Amalgamated mechanically (gen_amalgam.py) from PufferLib's parity-verified
// C implementation (ocean/craftax/*.h), which matches the JAX original with
// zero trajectory divergences. Game logic is byte-identical to the source
// headers; only the raylib renderer is #if 0-gated and a standalone
// bench/hash harness is appended at the bottom.
//
// Build:  make full   (gcc -O3 -march=native -ffast-math -fopenmp ...)
// Usage:  ./craftax_full bench --envs N --iters M [--threads T] [--seed S]
//         ./craftax_full hash  --envs N --steps M [--seed S]

#define CRAFTAX_ENABLE_ENV_IMPL

// ============================================================
// ===== threefry.h =====
// ============================================================
// Fast RNG helpers for Craftax.
// Replaces JAX Threefry with SplitMix64-based hashing for ~20-50x speedup.
// NOT cryptographically secure and NOT JAX-compatible.


#include <stdint.h>
#include <stddef.h>
#include <string.h>

typedef struct CraftaxThreefryKey {
    uint32_t word[2];
} CraftaxThreefryKey;

static __device__ inline uint64_t craftax_key_to_u64(CraftaxThreefryKey key) {
    return ((uint64_t)key.word[1] << 32) | key.word[0];
}

static __device__ inline CraftaxThreefryKey craftax_u64_to_key(uint64_t x) {
    CraftaxThreefryKey key = {{(uint32_t)x, (uint32_t)(x >> 32)}};
    return key;
}

static __device__ inline uint32_t craftax_rotl32(uint32_t x, uint32_t k) {
    return (uint32_t)((x << k) | (x >> (32u - k)));
}

static __device__ inline CraftaxThreefryKey craftax_prng_key(uint32_t seed) {
    CraftaxThreefryKey key = {{seed, seed ^ 0x9E3779B9u}};
    return key;
}

// MurmurHash3 64-bit finalizer — fast and good mixing
static __device__ inline uint64_t craftax_mix64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

// Core hash: mixes key state with counter, returns 64 bits of pseudo-randomness
static __device__ inline uint64_t craftax_fast_hash64(CraftaxThreefryKey key, uint64_t counter) {
    uint64_t x = craftax_key_to_u64(key);
    x ^= counter;
    return craftax_mix64(x);
}

static __device__ inline void craftax_threefry2x32(
    CraftaxThreefryKey key,
    uint32_t count0,
    uint32_t count1,
    uint32_t out[2]
) {
    uint64_t h = craftax_fast_hash64(key, ((uint64_t)count1 << 32) | count0);
    out[0] = (uint32_t)h;
    out[1] = (uint32_t)(h >> 32);
}

static __device__ inline CraftaxThreefryKey craftax_threefry_counter_key(
    CraftaxThreefryKey key,
    uint32_t count0,
    uint32_t count1
) {
    return craftax_u64_to_key(craftax_fast_hash64(key, ((uint64_t)count1 << 32) | count0));
}

// Fast split: sequential PCG-style advancement
static __device__ inline void craftax_threefry_split(
    CraftaxThreefryKey key,
    CraftaxThreefryKey* left,
    CraftaxThreefryKey* right
) {
    uint64_t state = craftax_key_to_u64(key);
    uint64_t s1 = state * 6364136223846793005ULL + 1;
    uint64_t s2 = s1 * 6364136223846793005ULL + 1;
    *left = craftax_u64_to_key(s1);
    *right = craftax_u64_to_key(s2);
}

static __device__ inline void craftax_threefry_split_n(
    CraftaxThreefryKey key,
    CraftaxThreefryKey* out,
    size_t count
) {
    uint64_t state = craftax_key_to_u64(key);
    for (size_t i = 0; i < count; i++) {
        state = state * 6364136223846793005ULL + 1;
        out[i] = craftax_u64_to_key(state);
    }
}

static __device__ inline CraftaxThreefryKey craftax_threefry_fold_in(
    CraftaxThreefryKey key,
    uint32_t data
) {
    return craftax_threefry_counter_key(key, 0u, data);
}

static __device__ inline uint32_t craftax_threefry_uniform_u32_at(
    CraftaxThreefryKey key,
    uint64_t index
) {
    uint64_t h = craftax_fast_hash64(key, index);
    return (uint32_t)h ^ (uint32_t)(h >> 32);
}

static __device__ inline uint32_t craftax_threefry_uniform_u32(CraftaxThreefryKey key) {
    return craftax_threefry_uniform_u32_at(key, 0u);
}

static __device__ inline float craftax_threefry_uniform_f32_at(
    CraftaxThreefryKey key,
    uint64_t index
) {
    uint32_t bits = craftax_threefry_uniform_u32_at(key, index);
    uint32_t float_bits = (bits >> 9u) | 0x3F800000u;
    float value;
    memcpy(&value, &float_bits, sizeof(value));
    return value - 1.0f;
}

static __device__ inline float craftax_threefry_uniform_f32(CraftaxThreefryKey key) {
    return craftax_threefry_uniform_f32_at(key, 0u);
}

// ============================================================
// ===== noise.h =====
// ============================================================
// Native C port of craftax/craftax/util/noise.py.


#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


#ifndef CRAFTAX_NOISE_PI2
#define CRAFTAX_NOISE_PI2 6.28318530717958647692f
#endif

#ifndef CRAFTAX_NOISE_SQRT2
#define CRAFTAX_NOISE_SQRT2 1.41421356237309504880f
#endif

static __device__ inline float craftax_noise_interpolant(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

static __device__ inline float craftax_noise_gradient_angle(
    CraftaxThreefryKey angle_key,
    int res_cols,
    int row,
    int col,
    const float* override_angles
) {
    int width = res_cols + 1;
    uint64_t index = (uint64_t)row * (uint64_t)width + (uint64_t)col;
    float unit = override_angles == NULL
        ? craftax_threefry_uniform_f32_at(angle_key, index)
        : override_angles[index];
    return CRAFTAX_NOISE_PI2 * unit;
}

static __device__ inline void craftax_noise_gradient(
    CraftaxThreefryKey angle_key,
    int res_cols,
    int row,
    int col,
    const float* override_angles,
    float* gx,
    float* gy
) {
    float angle = craftax_noise_gradient_angle(
        angle_key,
        res_cols,
        row,
        col,
        override_angles
    );
    *gx = cosf(angle);
    *gy = sinf(angle);
}

#ifndef CRAFTAX_NOISE_MAX_GRAD
// Largest gradient grid used by worldgen: res (6,24) at 48x48 -> 7x25 = 175.
// Sized with headroom; a grid larger than this falls back to per-cell lookups.
#define CRAFTAX_NOISE_MAX_GRAD 1024
#endif

static __device__ inline void craftax_generate_perlin_noise_2d_scalar(
    CraftaxThreefryKey rng,
    int rows,
    int cols,
    int res_rows,
    int res_cols,
    const float* override_angles,
    float* out
) {
    CraftaxThreefryKey unused;
    CraftaxThreefryKey angle_key;
    craftax_threefry_split(rng, &unused, &angle_key);

    int cell_rows = rows / res_rows;
    int cell_cols = cols / res_cols;

    // The gradient grid is tiny compared to the output map (e.g. 4x4 vs
    // 48x48), but the naive loop recomputes 4 sincos per output cell.
    // Precompute cos/sin for every grid corner once; values and their
    // consumers are unchanged, so output is bit-identical.
    int grad_w = res_cols + 1;
    int grad_h = res_rows + 1;
    float grad_x[CRAFTAX_NOISE_MAX_GRAD];
    float grad_y[CRAFTAX_NOISE_MAX_GRAD];
    bool table_ok = grad_w * grad_h <= CRAFTAX_NOISE_MAX_GRAD;
    if (table_ok) {
        for (int r = 0; r < grad_h; r++) {
            for (int c = 0; c < grad_w; c++) {
                float angle = craftax_noise_gradient_angle(
                    angle_key, res_cols, r, c, override_angles);
                grad_x[r * grad_w + c] = cosf(angle);
                grad_y[r * grad_w + c] = sinf(angle);
            }
        }
    }

    for (int row = 0; row < rows; row++) {
        int grad_row = row / cell_rows;
        float local_row = (float)(row - grad_row * cell_rows) / (float)cell_rows;
        float interp_row = craftax_noise_interpolant(local_row);

        for (int col = 0; col < cols; col++) {
            int grad_col = col / cell_cols;
            float local_col = (float)(col - grad_col * cell_cols) / (float)cell_cols;
            float interp_col = craftax_noise_interpolant(local_col);

            float g00x;
            float g00y;
            float g10x;
            float g10y;
            float g01x;
            float g01y;
            float g11x;
            float g11y;
            if (table_ok) {
                int i00 = grad_row * grad_w + grad_col;
                int i10 = i00 + grad_w;
                g00x = grad_x[i00];     g00y = grad_y[i00];
                g10x = grad_x[i10];     g10y = grad_y[i10];
                g01x = grad_x[i00 + 1]; g01y = grad_y[i00 + 1];
                g11x = grad_x[i10 + 1]; g11y = grad_y[i10 + 1];
            } else {
                craftax_noise_gradient(
                    angle_key, res_cols, grad_row, grad_col,
                    override_angles, &g00x, &g00y);
                craftax_noise_gradient(
                    angle_key, res_cols, grad_row + 1, grad_col,
                    override_angles, &g10x, &g10y);
                craftax_noise_gradient(
                    angle_key, res_cols, grad_row, grad_col + 1,
                    override_angles, &g01x, &g01y);
                craftax_noise_gradient(
                    angle_key, res_cols, grad_row + 1, grad_col + 1,
                    override_angles, &g11x, &g11y);
            }

            float n00 = local_row * g00x;
            n00 += local_col * g00y;
            float n10 = (local_row - 1.0f) * g10x;
            n10 += local_col * g10y;
            float n01 = local_row * g01x;
            n01 += (local_col - 1.0f) * g01y;
            float n11 = (local_row - 1.0f) * g11x;
            n11 += (local_col - 1.0f) * g11y;

            float n0 = n00 * (1.0f - interp_row) + interp_row * n10;
            float n1 = n01 * (1.0f - interp_row) + interp_row * n11;
            out[(size_t)row * (size_t)cols + (size_t)col] =
                CRAFTAX_NOISE_SQRT2 * ((1.0f - interp_col) * n0 + interp_col * n1);
        }
    }
}

#if defined(__AVX512F__) && !defined(CRAFTAX_NO_SIMD_NOISE)
#include <immintrin.h>

static __device__ inline __m512 craftax_noise_interpolant_v(__m512 t) {
    // t*t*t*(t*(t*6 - 15) + 10), plain mul/add to track the scalar formula.
    __m512 a = _mm512_add_ps(
        _mm512_mul_ps(t, _mm512_set1_ps(6.0f)), _mm512_set1_ps(-15.0f));
    a = _mm512_add_ps(_mm512_mul_ps(t, a), _mm512_set1_ps(10.0f));
    __m512 t3 = _mm512_mul_ps(_mm512_mul_ps(t, t), t);
    return _mm512_mul_ps(t3, a);
}

// AVX-512 inner loop, 16 output cells at a time. Requires cols % 16 == 0,
// power-of-two cell_cols, and a precomputed gradient table (all true for
// every worldgen call: 48-wide maps, cell_cols in {2,4,16}). May differ from
// the scalar path by ~1 ULP where the compiler contracted scalar mul+add
// into FMA; worlds are distributionally identical, not bit-identical.
static __device__ inline bool craftax_generate_perlin_noise_2d_avx512(
    CraftaxThreefryKey rng,
    int rows,
    int cols,
    int res_rows,
    int res_cols,
    const float* override_angles,
    float* out
) {
    int cell_rows = rows / res_rows;
    int cell_cols = cols / res_cols;
    int grad_w = res_cols + 1;
    int grad_h = res_rows + 1;
    if (cols % 16 != 0) return false;
    if (cell_cols <= 0 || (cell_cols & (cell_cols - 1)) != 0) return false;
    if (grad_w * grad_h > CRAFTAX_NOISE_MAX_GRAD) return false;

    CraftaxThreefryKey unused;
    CraftaxThreefryKey angle_key;
    craftax_threefry_split(rng, &unused, &angle_key);

    float grad_x[CRAFTAX_NOISE_MAX_GRAD];
    float grad_y[CRAFTAX_NOISE_MAX_GRAD];
    for (int r = 0; r < grad_h; r++) {
        for (int c = 0; c < grad_w; c++) {
            float angle = craftax_noise_gradient_angle(
                angle_key, res_cols, r, c, override_angles);
            grad_x[r * grad_w + c] = cosf(angle);
            grad_y[r * grad_w + c] = sinf(angle);
        }
    }

    uint32_t col_shift = (uint32_t)__builtin_ctz((unsigned)cell_cols);
    __m512i shift_v = _mm512_set1_epi32((int)col_shift);
    __m512i col_mask = _mm512_set1_epi32(cell_cols - 1);
    __m512 inv_cell_cols = _mm512_set1_ps(1.0f / (float)cell_cols);
    __m512i grad_w_v = _mm512_set1_epi32(grad_w);
    __m512i lane = _mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 sqrt2 = _mm512_set1_ps(CRAFTAX_NOISE_SQRT2);

    for (int row = 0; row < rows; row++) {
        int grad_row = row / cell_rows;
        float local_row_s =
            (float)(row - grad_row * cell_rows) / (float)cell_rows;
        float interp_row_s = craftax_noise_interpolant(local_row_s);
        __m512 local_row = _mm512_set1_ps(local_row_s);
        __m512 local_row_m1 = _mm512_set1_ps(local_row_s - 1.0f);
        __m512 interp_row = _mm512_set1_ps(interp_row_s);
        __m512 one_m_interp_row = _mm512_set1_ps(1.0f - interp_row_s);
        __m512i row_base = _mm512_set1_epi32(grad_row * grad_w);

        for (int col = 0; col < cols; col += 16) {
            __m512i col_v = _mm512_add_epi32(_mm512_set1_epi32(col), lane);
            __m512i grad_col = _mm512_srlv_epi32(col_v, shift_v);
            __m512 local_col = _mm512_mul_ps(
                _mm512_cvtepi32_ps(_mm512_and_si512(col_v, col_mask)),
                inv_cell_cols);
            __m512 interp_col = craftax_noise_interpolant_v(local_col);

            __m512i i00 = _mm512_add_epi32(row_base, grad_col);
            __m512i i10 = _mm512_add_epi32(i00, grad_w_v);
            __m512i i01 = _mm512_add_epi32(i00, _mm512_set1_epi32(1));
            __m512i i11 = _mm512_add_epi32(i10, _mm512_set1_epi32(1));

            __m512 g00x = _mm512_i32gather_ps(i00, grad_x, 4);
            __m512 g00y = _mm512_i32gather_ps(i00, grad_y, 4);
            __m512 g10x = _mm512_i32gather_ps(i10, grad_x, 4);
            __m512 g10y = _mm512_i32gather_ps(i10, grad_y, 4);
            __m512 g01x = _mm512_i32gather_ps(i01, grad_x, 4);
            __m512 g01y = _mm512_i32gather_ps(i01, grad_y, 4);
            __m512 g11x = _mm512_i32gather_ps(i11, grad_x, 4);
            __m512 g11y = _mm512_i32gather_ps(i11, grad_y, 4);

            __m512 local_col_m1 = _mm512_sub_ps(local_col, one);
            __m512 n00 = _mm512_add_ps(
                _mm512_mul_ps(local_row, g00x),
                _mm512_mul_ps(local_col, g00y));
            __m512 n10 = _mm512_add_ps(
                _mm512_mul_ps(local_row_m1, g10x),
                _mm512_mul_ps(local_col, g10y));
            __m512 n01 = _mm512_add_ps(
                _mm512_mul_ps(local_row, g01x),
                _mm512_mul_ps(local_col_m1, g01y));
            __m512 n11 = _mm512_add_ps(
                _mm512_mul_ps(local_row_m1, g11x),
                _mm512_mul_ps(local_col_m1, g11y));

            __m512 n0 = _mm512_add_ps(
                _mm512_mul_ps(n00, one_m_interp_row),
                _mm512_mul_ps(interp_row, n10));
            __m512 n1 = _mm512_add_ps(
                _mm512_mul_ps(n01, one_m_interp_row),
                _mm512_mul_ps(interp_row, n11));
            __m512 result = _mm512_mul_ps(sqrt2, _mm512_add_ps(
                _mm512_mul_ps(_mm512_sub_ps(one, interp_col), n0),
                _mm512_mul_ps(interp_col, n1)));
            _mm512_storeu_ps(&out[(size_t)row * (size_t)cols + (size_t)col],
                             result);
        }
    }
    return true;
}
#endif  // __AVX512F__ && !CRAFTAX_NO_SIMD_NOISE

static __device__ inline void craftax_generate_perlin_noise_2d(
    CraftaxThreefryKey rng,
    int rows,
    int cols,
    int res_rows,
    int res_cols,
    const float* override_angles,
    float* out
) {
#if defined(__AVX512F__) && !defined(CRAFTAX_NO_SIMD_NOISE)
    if (craftax_generate_perlin_noise_2d_avx512(
            rng, rows, cols, res_rows, res_cols, override_angles, out)) {
        return;
    }
#endif
    craftax_generate_perlin_noise_2d_scalar(
        rng, rows, cols, res_rows, res_cols, override_angles, out);
}

static __device__ inline void craftax_generate_fractal_noise_2d(
    CraftaxThreefryKey rng,
    int rows,
    int cols,
    int res_rows,
    int res_cols,
    int octaves,
    float persistence,
    int lacunarity,
    const float* override_angles,
    float* out
) {
    size_t size = (size_t)rows * (size_t)cols;
    for (size_t i = 0; i < size; i++) {
        out[i] = 0.0f;
    }

    int frequency = 1;
    float amplitude = 1.0f;
    // [cuda port] was a C VLA `float perlin[size]`; worldgen only ever
    // calls this with rows*cols <= 48*48.
    if (size > 2304) { __trap(); }
    float perlin[2304];

    for (int octave = 0; octave < octaves; octave++) {
        CraftaxThreefryKey next_rng;
        CraftaxThreefryKey noise_key;
        craftax_threefry_split(rng, &next_rng, &noise_key);
        rng = next_rng;

        craftax_generate_perlin_noise_2d(
            noise_key,
            rows,
            cols,
            frequency * res_rows,
            frequency * res_cols,
            override_angles,
            perlin
        );

        for (size_t i = 0; i < size; i++) {
            out[i] += amplitude * perlin[i];
        }

        frequency *= lacunarity;
        amplitude *= persistence;
    }

    float min_value = out[0];
    float max_value = out[0];
    for (size_t i = 1; i < size; i++) {
        if (out[i] < min_value) {
            min_value = out[i];
        }
        if (out[i] > max_value) {
            max_value = out[i];
        }
    }

    float scale = max_value - min_value;
    for (size_t i = 0; i < size; i++) {
        out[i] = (out[i] - min_value) / scale;
    }
}

// ============================================================
// ===== worldgen.h =====
// ============================================================
// Native Craftax reset world generation.
//
// This mirrors craftax/craftax/world_gen/world_gen.py for the default
// EnvParams and StaticEnvParams used by Craftax-Symbolic-v1 reset.


#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>


#define CRAFTAX_WG_MAP_SIZE 48
#define CRAFTAX_WG_MAP_CELLS (CRAFTAX_WG_MAP_SIZE * CRAFTAX_WG_MAP_SIZE)
#define CRAFTAX_WG_NUM_LEVELS 9
#define CRAFTAX_WG_OBS_ROWS 9
#define CRAFTAX_WG_OBS_COLS 11
#define CRAFTAX_WG_NUM_BLOCK_TYPES 37
#define CRAFTAX_WG_NUM_ITEM_TYPES 5
#define CRAFTAX_WG_NUM_MOB_CLASSES 5
#define CRAFTAX_WG_NUM_MOB_TYPES 8
#define CRAFTAX_WG_INVENTORY_OBS_SIZE 51

// Compact binary observation encoding.
// Each cell uses binary channels instead of one-hot:
//   6 bits: block type (0-63, covers 37 block types)
//   3 bits: item type+1 (0=no item, 1-5=item types)
//   4 bits per mob class: mob type+1 (0=no mob, 1-8=types) x 5 classes
//   1 bit : visibility
// Total: 30 binary channels per cell.
#define CRAFTAX_WG_BINARY_BLOCK_BITS 6
#define CRAFTAX_WG_BINARY_ITEM_BITS 3
#define CRAFTAX_WG_BINARY_MOB_BITS 4
#define CRAFTAX_WG_BINARY_VISIBILITY_BITS 1

#define CRAFTAX_WG_BINARY_CHANNELS_PER_CELL ( \
    CRAFTAX_WG_BINARY_BLOCK_BITS + \
    CRAFTAX_WG_BINARY_ITEM_BITS + \
    CRAFTAX_WG_NUM_MOB_CLASSES * CRAFTAX_WG_BINARY_MOB_BITS + \
    CRAFTAX_WG_BINARY_VISIBILITY_BITS \
)

#define CRAFTAX_WG_BINARY_MAP_OBS_SIZE ( \
    CRAFTAX_WG_OBS_ROWS * CRAFTAX_WG_OBS_COLS * CRAFTAX_WG_BINARY_CHANNELS_PER_CELL \
)
#define CRAFTAX_WG_OBS_WINDOW_CELLS (CRAFTAX_WG_OBS_ROWS * CRAFTAX_WG_OBS_COLS)
#define CRAFTAX_WG_CELL_TEMPLATE_BYTES ( \
    CRAFTAX_WG_BINARY_CHANNELS_PER_CELL * sizeof(float) \
)
#define CRAFTAX_WG_FULL_OBS_SIZE ( \
    CRAFTAX_WG_BINARY_MAP_OBS_SIZE + CRAFTAX_WG_INVENTORY_OBS_SIZE \
)

// Moonshot symbolic observation. Each visible cell stores compact float IDs:
// block, item+1, visible, and one mob type+1 slot for each mob class.
// The 51 scalar channels remain exact floats for oracle-expandability.
#define CRAFTAX_WG_PACKED_CHANNELS_PER_CELL (3 + CRAFTAX_WG_NUM_MOB_CLASSES)
#define CRAFTAX_WG_PACKED_MAP_OBS_SIZE ( \
    CRAFTAX_WG_OBS_ROWS * CRAFTAX_WG_OBS_COLS * CRAFTAX_WG_PACKED_CHANNELS_PER_CELL \
)
#define CRAFTAX_WG_PACKED_OBS_SIZE ( \
    CRAFTAX_WG_PACKED_MAP_OBS_SIZE + CRAFTAX_WG_INVENTORY_OBS_SIZE \
)

// Compact byte observation: every packed map channel is a small integer ID
// (block < 37, item+1 <= 5, visible bit, mob type+1 <= 8), so the 792 map
// channels fit in uint8 losslessly. The 51-float scalar tail is appended as
// raw float32 bytes; the GPU expands map bytes with .float() and reinterprets
// the tail, reproducing the float observation bit-for-bit at 3.4x less DMA.
#define CRAFTAX_WG_COMPACT_OBS_SIZE ( \
    CRAFTAX_WG_PACKED_MAP_OBS_SIZE \
    + CRAFTAX_WG_INVENTORY_OBS_SIZE * (int)sizeof(float) \
)

// Lookup tables for fast binary bit writing (eliminates loops/branches)
static __device__ const float CRAFTAX_WG_BLOCK_LUT[64][6] = {
    {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f},{1.0f,0.0f,0.0f,0.0f,0.0f,0.0f},{0.0f,1.0f,0.0f,0.0f,0.0f,0.0f},{1.0f,1.0f,0.0f,0.0f,0.0f,0.0f},
    {0.0f,0.0f,1.0f,0.0f,0.0f,0.0f},{1.0f,0.0f,1.0f,0.0f,0.0f,0.0f},{0.0f,1.0f,1.0f,0.0f,0.0f,0.0f},{1.0f,1.0f,1.0f,0.0f,0.0f,0.0f},
    {0.0f,0.0f,0.0f,1.0f,0.0f,0.0f},{1.0f,0.0f,0.0f,1.0f,0.0f,0.0f},{0.0f,1.0f,0.0f,1.0f,0.0f,0.0f},{1.0f,1.0f,0.0f,1.0f,0.0f,0.0f},
    {0.0f,0.0f,1.0f,1.0f,0.0f,0.0f},{1.0f,0.0f,1.0f,1.0f,0.0f,0.0f},{0.0f,1.0f,1.0f,1.0f,0.0f,0.0f},{1.0f,1.0f,1.0f,1.0f,0.0f,0.0f},
    {0.0f,0.0f,0.0f,0.0f,1.0f,0.0f},{1.0f,0.0f,0.0f,0.0f,1.0f,0.0f},{0.0f,1.0f,0.0f,0.0f,1.0f,0.0f},{1.0f,1.0f,0.0f,0.0f,1.0f,0.0f},
    {0.0f,0.0f,1.0f,0.0f,1.0f,0.0f},{1.0f,0.0f,1.0f,0.0f,1.0f,0.0f},{0.0f,1.0f,1.0f,0.0f,1.0f,0.0f},{1.0f,1.0f,1.0f,0.0f,1.0f,0.0f},
    {0.0f,0.0f,0.0f,1.0f,1.0f,0.0f},{1.0f,0.0f,0.0f,1.0f,1.0f,0.0f},{0.0f,1.0f,0.0f,1.0f,1.0f,0.0f},{1.0f,1.0f,0.0f,1.0f,1.0f,0.0f},
    {0.0f,0.0f,1.0f,1.0f,1.0f,0.0f},{1.0f,0.0f,1.0f,1.0f,1.0f,0.0f},{0.0f,1.0f,1.0f,1.0f,1.0f,0.0f},{1.0f,1.0f,1.0f,1.0f,1.0f,0.0f},
    {0.0f,0.0f,0.0f,0.0f,0.0f,1.0f},{1.0f,0.0f,0.0f,0.0f,0.0f,1.0f},{0.0f,1.0f,0.0f,0.0f,0.0f,1.0f},{1.0f,1.0f,0.0f,0.0f,0.0f,1.0f},
    {0.0f,0.0f,1.0f,0.0f,0.0f,1.0f},{1.0f,0.0f,1.0f,0.0f,0.0f,1.0f},{0.0f,1.0f,1.0f,0.0f,0.0f,1.0f},{1.0f,1.0f,1.0f,0.0f,0.0f,1.0f},
    {0.0f,0.0f,0.0f,1.0f,0.0f,1.0f},{1.0f,0.0f,0.0f,1.0f,0.0f,1.0f},{0.0f,1.0f,0.0f,1.0f,0.0f,1.0f},{1.0f,1.0f,0.0f,1.0f,0.0f,1.0f},
    {0.0f,0.0f,1.0f,1.0f,0.0f,1.0f},{1.0f,0.0f,1.0f,1.0f,0.0f,1.0f},{0.0f,1.0f,1.0f,1.0f,0.0f,1.0f},{1.0f,1.0f,1.0f,1.0f,0.0f,1.0f},
    {0.0f,0.0f,0.0f,0.0f,1.0f,1.0f},{1.0f,0.0f,0.0f,0.0f,1.0f,1.0f},{0.0f,1.0f,0.0f,0.0f,1.0f,1.0f},{1.0f,1.0f,0.0f,0.0f,1.0f,1.0f},
    {0.0f,0.0f,1.0f,0.0f,1.0f,1.0f},{1.0f,0.0f,1.0f,0.0f,1.0f,1.0f},{0.0f,1.0f,1.0f,0.0f,1.0f,1.0f},{1.0f,1.0f,1.0f,0.0f,1.0f,1.0f},
    {0.0f,0.0f,0.0f,1.0f,1.0f,1.0f},{1.0f,0.0f,0.0f,1.0f,1.0f,1.0f},{0.0f,1.0f,0.0f,1.0f,1.0f,1.0f},{1.0f,1.0f,0.0f,1.0f,1.0f,1.0f},
    {0.0f,0.0f,1.0f,1.0f,1.0f,1.0f},{1.0f,0.0f,1.0f,1.0f,1.0f,1.0f},{0.0f,1.0f,1.0f,1.0f,1.0f,1.0f},{1.0f,1.0f,1.0f,1.0f,1.0f,1.0f},
};
static __device__ const float CRAFTAX_WG_ITEM_LUT[8][3] = {
    {0.0f,0.0f,0.0f},{1.0f,0.0f,0.0f},{0.0f,1.0f,0.0f},{1.0f,1.0f,0.0f},
    {0.0f,0.0f,1.0f},{1.0f,0.0f,1.0f},{0.0f,1.0f,1.0f},{1.0f,1.0f,1.0f},
};
static __device__ const float CRAFTAX_WG_MOB_LUT[16][4] = {
    {0.0f,0.0f,0.0f,0.0f},{1.0f,0.0f,0.0f,0.0f},{0.0f,1.0f,0.0f,0.0f},{1.0f,1.0f,0.0f,0.0f},
    {0.0f,0.0f,1.0f,0.0f},{1.0f,0.0f,1.0f,0.0f},{0.0f,1.0f,1.0f,0.0f},{1.0f,1.0f,1.0f,0.0f},
    {0.0f,0.0f,0.0f,1.0f},{1.0f,0.0f,0.0f,1.0f},{0.0f,1.0f,0.0f,1.0f},{1.0f,1.0f,0.0f,1.0f},
    {0.0f,0.0f,1.0f,1.0f},{1.0f,0.0f,1.0f,1.0f},{0.0f,1.0f,1.0f,1.0f},{1.0f,1.0f,1.0f,1.0f},
};
static __device__ float CRAFTAX_WG_VISIBLE_CELL_TEMPLATE_LUT[64][8][CRAFTAX_WG_BINARY_CHANNELS_PER_CELL];
static __device__ float CRAFTAX_WG_EMPTY_CELL_TEMPLATE[CRAFTAX_WG_BINARY_CHANNELS_PER_CELL];
static __device__ bool CRAFTAX_WG_CELL_TEMPLATE_READY = false;

static __device__ inline void craftax_wg_init_cell_templates(void) {
    if (CRAFTAX_WG_CELL_TEMPLATE_READY) {
        return;
    }

    for (int block = 0; block < 64; block++) {
        for (int item = 0; item < 8; item++) {
            float* cell = CRAFTAX_WG_VISIBLE_CELL_TEMPLATE_LUT[block][item];
            memcpy(cell, CRAFTAX_WG_BLOCK_LUT[block], 6 * sizeof(float));
            memcpy(cell + CRAFTAX_WG_BINARY_BLOCK_BITS, CRAFTAX_WG_ITEM_LUT[item], 3 * sizeof(float));
            cell[CRAFTAX_WG_BINARY_CHANNELS_PER_CELL - 1] = 1.0f;
        }
    }

    CRAFTAX_WG_CELL_TEMPLATE_READY = true;
}

#define CRAFTAX_WG_OBS_SIZE CRAFTAX_WG_PACKED_OBS_SIZE
#define CRAFTAX_WG_NUM_ACHIEVEMENTS 67
#define CRAFTAX_WG_MAX_MELEE_MOBS 3
#define CRAFTAX_WG_MAX_PASSIVE_MOBS 3
#define CRAFTAX_WG_MAX_RANGED_MOBS 2
#define CRAFTAX_WG_MAX_MOB_PROJECTILES 3
#define CRAFTAX_WG_MAX_PLAYER_PROJECTILES 3
#define CRAFTAX_WG_MAX_GROWING_PLANTS 10
#define CRAFTAX_WG_MONSTERS_KILLED_TO_CLEAR_LEVEL 8

// Backwards-compatible names used by the phase-1 floor-0 test.
#define CRAFTAX_OVERWORLD_SIZE CRAFTAX_WG_MAP_SIZE
#define CRAFTAX_OVERWORLD_CELLS CRAFTAX_WG_MAP_CELLS

#define CRAFTAX_WG_BLOCK_INVALID 0
#define CRAFTAX_WG_BLOCK_OUT_OF_BOUNDS 1
#define CRAFTAX_WG_BLOCK_GRASS 2
#define CRAFTAX_WG_BLOCK_WATER 3
#define CRAFTAX_WG_BLOCK_STONE 4
#define CRAFTAX_WG_BLOCK_TREE 5
#define CRAFTAX_WG_BLOCK_WOOD 6
#define CRAFTAX_WG_BLOCK_PATH 7
#define CRAFTAX_WG_BLOCK_COAL 8
#define CRAFTAX_WG_BLOCK_IRON 9
#define CRAFTAX_WG_BLOCK_DIAMOND 10
#define CRAFTAX_WG_BLOCK_CRAFTING_TABLE 11
#define CRAFTAX_WG_BLOCK_FURNACE 12
#define CRAFTAX_WG_BLOCK_SAND 13
#define CRAFTAX_WG_BLOCK_LAVA 14
#define CRAFTAX_WG_BLOCK_PLANT 15
#define CRAFTAX_WG_BLOCK_RIPE_PLANT 16
#define CRAFTAX_WG_BLOCK_WALL 17
#define CRAFTAX_WG_BLOCK_DARKNESS 18
#define CRAFTAX_WG_BLOCK_WALL_MOSS 19
#define CRAFTAX_WG_BLOCK_STALAGMITE 20
#define CRAFTAX_WG_BLOCK_SAPPHIRE 21
#define CRAFTAX_WG_BLOCK_RUBY 22
#define CRAFTAX_WG_BLOCK_CHEST 23
#define CRAFTAX_WG_BLOCK_FOUNTAIN 24
#define CRAFTAX_WG_BLOCK_FIRE_GRASS 25
#define CRAFTAX_WG_BLOCK_ICE_GRASS 26
#define CRAFTAX_WG_BLOCK_GRAVEL 27
#define CRAFTAX_WG_BLOCK_FIRE_TREE 28
#define CRAFTAX_WG_BLOCK_ICE_SHRUB 29
#define CRAFTAX_WG_BLOCK_ENCHANTMENT_TABLE_FIRE 30
#define CRAFTAX_WG_BLOCK_ENCHANTMENT_TABLE_ICE 31
#define CRAFTAX_WG_BLOCK_NECROMANCER 32
#define CRAFTAX_WG_BLOCK_GRAVE 33
#define CRAFTAX_WG_BLOCK_GRAVE2 34
#define CRAFTAX_WG_BLOCK_GRAVE3 35
#define CRAFTAX_WG_BLOCK_NECROMANCER_VULNERABLE 36

#define CRAFTAX_WG_ITEM_NONE 0
#define CRAFTAX_WG_ITEM_TORCH 1
#define CRAFTAX_WG_ITEM_LADDER_DOWN 2
#define CRAFTAX_WG_ITEM_LADDER_UP 3
#define CRAFTAX_WG_ITEM_LADDER_DOWN_BLOCKED 4

#define CRAFTAX_WG_ACTION_UP 3
#define CRAFTAX_WG_BOSS_FIGHT_SPAWN_TURNS 7
#define CRAFTAX_WG_PI 3.14159265358979323846f
// [cuda port] table indexed by timestep 0..100000 inclusive
#define CRAFTAX_DEFAULT_MAX_TIMESTEPS_TABLE 100001

typedef struct CraftaxOverworldFloor {
    uint8_t map[CRAFTAX_OVERWORLD_SIZE][CRAFTAX_OVERWORLD_SIZE];
    uint8_t item_map[CRAFTAX_OVERWORLD_SIZE][CRAFTAX_OVERWORLD_SIZE];
    uint8_t light_map[CRAFTAX_OVERWORLD_SIZE][CRAFTAX_OVERWORLD_SIZE];
    int32_t ladder_down[2];
    int32_t ladder_up[2];
} CraftaxOverworldFloor;

typedef struct CraftaxWGInventory {
    int32_t wood;
    int32_t stone;
    int32_t coal;
    int32_t iron;
    int32_t diamond;
    int32_t sapling;
    int32_t pickaxe;
    int32_t sword;
    int32_t bow;
    int32_t arrows;
    int32_t armour[4];
    int32_t torches;
    int32_t ruby;
    int32_t sapphire;
    int32_t potions[6];
    int32_t books;
} CraftaxWGInventory;

typedef struct CraftaxWGMobs3 {
    int32_t position[CRAFTAX_WG_NUM_LEVELS][3][2];
    float health[CRAFTAX_WG_NUM_LEVELS][3];
    bool mask[CRAFTAX_WG_NUM_LEVELS][3];
    int32_t attack_cooldown[CRAFTAX_WG_NUM_LEVELS][3];
    int32_t type_id[CRAFTAX_WG_NUM_LEVELS][3];
} CraftaxWGMobs3;

typedef struct CraftaxWGMobs2 {
    int32_t position[CRAFTAX_WG_NUM_LEVELS][2][2];
    float health[CRAFTAX_WG_NUM_LEVELS][2];
    bool mask[CRAFTAX_WG_NUM_LEVELS][2];
    int32_t attack_cooldown[CRAFTAX_WG_NUM_LEVELS][2];
    int32_t type_id[CRAFTAX_WG_NUM_LEVELS][2];
} CraftaxWGMobs2;

typedef struct CraftaxWorldState {
    // === Hot data (accessed every step) ===
    int32_t player_position[2];
    int32_t player_level;
    int32_t player_direction;

    float player_health;
    int32_t player_food;
    int32_t player_drink;
    int32_t player_energy;
    int32_t player_mana;
    bool is_sleeping;
    bool is_resting;

    float player_recover;
    float player_hunger;
    float player_thirst;
    float player_fatigue;
    float player_recover_mana;

    int32_t player_xp;
    int32_t player_dexterity;
    int32_t player_strength;
    int32_t player_intelligence;

    CraftaxWGInventory inventory;

    CraftaxWGMobs3 melee_mobs;
    CraftaxWGMobs3 passive_mobs;
    CraftaxWGMobs2 ranged_mobs;

    CraftaxWGMobs3 mob_projectiles;
    int32_t mob_projectile_directions[CRAFTAX_WG_NUM_LEVELS][CRAFTAX_WG_MAX_MOB_PROJECTILES][2];
    CraftaxWGMobs3 player_projectiles;
    int32_t player_projectile_directions[CRAFTAX_WG_NUM_LEVELS][CRAFTAX_WG_MAX_PLAYER_PROJECTILES][2];

    int32_t growing_plants_positions[CRAFTAX_WG_MAX_GROWING_PLANTS][2];
    int32_t growing_plants_age[CRAFTAX_WG_MAX_GROWING_PLANTS];
    bool growing_plants_mask[CRAFTAX_WG_MAX_GROWING_PLANTS];

    int32_t potion_mapping[6];
    bool learned_spells[2];

    int32_t sword_enchantment;
    int32_t bow_enchantment;
    int32_t armour_enchantments[4];

    int32_t boss_progress;
    int32_t boss_timesteps_to_spawn_this_round;

    float light_level;
    bool achievements[CRAFTAX_WG_NUM_ACHIEVEMENTS];
    uint32_t state_rng[2];
    int32_t timestep;
    int32_t fractal_noise_angles[4];

    // === Medium-hot bitmaps ===
    uint64_t mob_bits[CRAFTAX_WG_NUM_LEVELS][CRAFTAX_WG_MAP_SIZE];
    uint64_t spawn_all_bits[CRAFTAX_WG_NUM_LEVELS][CRAFTAX_WG_MAP_SIZE];
    uint64_t spawn_grave_bits[CRAFTAX_WG_NUM_LEVELS][CRAFTAX_WG_MAP_SIZE];
    uint64_t spawn_water_bits[CRAFTAX_WG_NUM_LEVELS][CRAFTAX_WG_MAP_SIZE];

    // === Cold data (large maps) ===
    uint8_t map[CRAFTAX_WG_NUM_LEVELS][CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE];
    uint8_t item_map[CRAFTAX_WG_NUM_LEVELS][CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE];
    uint8_t light_map[CRAFTAX_WG_NUM_LEVELS][CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE];

    int32_t down_ladders[CRAFTAX_WG_NUM_LEVELS][2];
    int32_t up_ladders[CRAFTAX_WG_NUM_LEVELS][2];
    bool chests_opened[CRAFTAX_WG_NUM_LEVELS];
    int32_t monsters_killed[CRAFTAX_WG_NUM_LEVELS];

    // Lazy floor generation. Bit L of lazy_floors_pending set means floor L
    // has not been generated yet; its worldgen key is in lazy_floor_keys[L].
    // Zero-initialized states (test fixtures, JAX mirrors) read as
    // "all floors generated", so laziness is opt-in per state.
    uint32_t lazy_floor_keys[CRAFTAX_WG_NUM_LEVELS][2];
    uint32_t lazy_floors_pending;
} CraftaxWorldState;

typedef struct CraftaxSmoothGenConfig {
    int32_t default_block;
    int32_t sea_block;
    int32_t coast_block;
    int32_t mountain_block;
    int32_t path_block;
    int32_t inner_mountain_block;
    int32_t ore_requirement_blocks[5];
    int32_t ores[5];
    float ore_chances[5];
    int32_t tree_requirement_block;
    int32_t tree;
    int32_t lava;
    int32_t player_spawn;
    int32_t valid_ladder;
    bool ladder_up;
    bool ladder_down;
    float player_proximity_map_water_strength;
    float player_proximity_map_water_max;
    float player_proximity_map_mountain_strength;
    float player_proximity_map_mountain_max;
    float default_light;
    float water_threshold;
    float sand_threshold;
    float tree_threshold_uniform;
    float tree_threshold_perlin;
} CraftaxSmoothGenConfig;

typedef struct CraftaxDungeonConfig {
    int32_t special_block;
    int32_t fountain_block;
    int32_t rare_path_replacement_block;
} CraftaxDungeonConfig;

static __device__ const CraftaxSmoothGenConfig CRAFTAX_SMOOTHGEN_CONFIGS[6] = {
    {
        CRAFTAX_WG_BLOCK_GRASS,
        CRAFTAX_WG_BLOCK_WATER,
        CRAFTAX_WG_BLOCK_SAND,
        CRAFTAX_WG_BLOCK_STONE,
        CRAFTAX_WG_BLOCK_PATH,
        CRAFTAX_WG_BLOCK_PATH,
        {CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE},
        {CRAFTAX_WG_BLOCK_COAL, CRAFTAX_WG_BLOCK_IRON, CRAFTAX_WG_BLOCK_DIAMOND, CRAFTAX_WG_BLOCK_OUT_OF_BOUNDS, CRAFTAX_WG_BLOCK_OUT_OF_BOUNDS},
        {0.03f, 0.02f, 0.001f, 0.0f, 0.0f},
        CRAFTAX_WG_BLOCK_GRASS,
        CRAFTAX_WG_BLOCK_TREE,
        CRAFTAX_WG_BLOCK_LAVA,
        CRAFTAX_WG_BLOCK_GRASS,
        CRAFTAX_WG_BLOCK_PATH,
        false,
        true,
        5.0f,
        1.0f,
        5.0f,
        1.0f,
        1.0f,
        0.7f,
        0.6f,
        0.8f,
        0.5f,
    },
    {
        CRAFTAX_WG_BLOCK_PATH,
        CRAFTAX_WG_BLOCK_WATER,
        CRAFTAX_WG_BLOCK_PATH,
        CRAFTAX_WG_BLOCK_STONE,
        CRAFTAX_WG_BLOCK_STONE,
        CRAFTAX_WG_BLOCK_STONE,
        {CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE},
        {CRAFTAX_WG_BLOCK_COAL, CRAFTAX_WG_BLOCK_IRON, CRAFTAX_WG_BLOCK_DIAMOND, CRAFTAX_WG_BLOCK_SAPPHIRE, CRAFTAX_WG_BLOCK_RUBY},
        {0.04f, 0.02f, 0.005f, 0.0025f, 0.0025f},
        CRAFTAX_WG_BLOCK_PATH,
        CRAFTAX_WG_BLOCK_STALAGMITE,
        CRAFTAX_WG_BLOCK_LAVA,
        CRAFTAX_WG_BLOCK_PATH,
        CRAFTAX_WG_BLOCK_PATH,
        true,
        true,
        5.0f,
        1.0f,
        17.0f,
        1.5f,
        0.0f,
        0.7f,
        0.6f,
        0.8f,
        0.5f,
    },
    {
        CRAFTAX_WG_BLOCK_PATH,
        CRAFTAX_WG_BLOCK_WATER,
        CRAFTAX_WG_BLOCK_PATH,
        CRAFTAX_WG_BLOCK_STONE,
        CRAFTAX_WG_BLOCK_STONE,
        CRAFTAX_WG_BLOCK_STONE,
        {CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE},
        {CRAFTAX_WG_BLOCK_COAL, CRAFTAX_WG_BLOCK_IRON, CRAFTAX_WG_BLOCK_DIAMOND, CRAFTAX_WG_BLOCK_SAPPHIRE, CRAFTAX_WG_BLOCK_RUBY},
        {0.04f, 0.03f, 0.01f, 0.01f, 0.01f},
        CRAFTAX_WG_BLOCK_PATH,
        CRAFTAX_WG_BLOCK_STALAGMITE,
        CRAFTAX_WG_BLOCK_LAVA,
        CRAFTAX_WG_BLOCK_PATH,
        CRAFTAX_WG_BLOCK_PATH,
        true,
        true,
        5.0f,
        1.0f,
        17.0f,
        1.5f,
        0.0f,
        0.7f,
        0.6f,
        0.8f,
        0.5f,
    },
    {
        CRAFTAX_WG_BLOCK_FIRE_GRASS,
        CRAFTAX_WG_BLOCK_LAVA,
        CRAFTAX_WG_BLOCK_SAND,
        CRAFTAX_WG_BLOCK_STONE,
        CRAFTAX_WG_BLOCK_STONE,
        CRAFTAX_WG_BLOCK_STONE,
        {CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE},
        {CRAFTAX_WG_BLOCK_COAL, CRAFTAX_WG_BLOCK_IRON, CRAFTAX_WG_BLOCK_DIAMOND, CRAFTAX_WG_BLOCK_SAPPHIRE, CRAFTAX_WG_BLOCK_RUBY},
        {0.05f, 0.0f, 0.0f, 0.0f, 0.025f},
        CRAFTAX_WG_BLOCK_FIRE_GRASS,
        CRAFTAX_WG_BLOCK_FIRE_TREE,
        CRAFTAX_WG_BLOCK_LAVA,
        CRAFTAX_WG_BLOCK_FIRE_GRASS,
        CRAFTAX_WG_BLOCK_FIRE_GRASS,
        true,
        true,
        5.0f,
        1.0f,
        5.0f,
        1.0f,
        1.0f,
        0.5f,
        0.6f,
        0.8f,
        0.5f,
    },
    {
        CRAFTAX_WG_BLOCK_ICE_GRASS,
        CRAFTAX_WG_BLOCK_WATER,
        CRAFTAX_WG_BLOCK_ICE_GRASS,
        CRAFTAX_WG_BLOCK_STONE,
        CRAFTAX_WG_BLOCK_STONE,
        CRAFTAX_WG_BLOCK_STONE,
        {CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE},
        {CRAFTAX_WG_BLOCK_COAL, CRAFTAX_WG_BLOCK_IRON, CRAFTAX_WG_BLOCK_DIAMOND, CRAFTAX_WG_BLOCK_SAPPHIRE, CRAFTAX_WG_BLOCK_RUBY},
        {0.0f, 0.0f, 0.005f, 0.02f, 0.0f},
        CRAFTAX_WG_BLOCK_ICE_GRASS,
        CRAFTAX_WG_BLOCK_ICE_SHRUB,
        CRAFTAX_WG_BLOCK_WATER,
        CRAFTAX_WG_BLOCK_ICE_GRASS,
        CRAFTAX_WG_BLOCK_ICE_GRASS,
        true,
        true,
        5.0f,
        1.0f,
        17.0f,
        1.5f,
        0.0f,
        0.5f,
        0.6f,
        0.4f,
        0.5f,
    },
    {
        CRAFTAX_WG_BLOCK_PATH,
        CRAFTAX_WG_BLOCK_PATH,
        CRAFTAX_WG_BLOCK_PATH,
        CRAFTAX_WG_BLOCK_WALL,
        CRAFTAX_WG_BLOCK_WALL,
        CRAFTAX_WG_BLOCK_WALL,
        {CRAFTAX_WG_BLOCK_WALL, CRAFTAX_WG_BLOCK_GRAVE, CRAFTAX_WG_BLOCK_GRAVE, CRAFTAX_WG_BLOCK_WALL, CRAFTAX_WG_BLOCK_WALL},
        {CRAFTAX_WG_BLOCK_WALL_MOSS, CRAFTAX_WG_BLOCK_GRAVE2, CRAFTAX_WG_BLOCK_GRAVE3, CRAFTAX_WG_BLOCK_SAPPHIRE, CRAFTAX_WG_BLOCK_RUBY},
        {0.1f, 0.333f, 0.5f, 0.0f, 0.0f},
        CRAFTAX_WG_BLOCK_PATH,
        CRAFTAX_WG_BLOCK_GRAVE,
        CRAFTAX_WG_BLOCK_WALL,
        CRAFTAX_WG_BLOCK_NECROMANCER,
        CRAFTAX_WG_BLOCK_PATH,
        false,
        false,
        5.0f,
        1.0f,
        10.0f,
        10.0f,
        0.0f,
        0.7f,
        0.6f,
        0.95f,
        -1.0f,
    },
};

static __device__ const CraftaxDungeonConfig CRAFTAX_DUNGEON_CONFIGS[3] = {
    {CRAFTAX_WG_BLOCK_PATH, CRAFTAX_WG_BLOCK_FOUNTAIN, CRAFTAX_WG_BLOCK_PATH},
    {CRAFTAX_WG_BLOCK_ENCHANTMENT_TABLE_ICE, CRAFTAX_WG_BLOCK_WATER, CRAFTAX_WG_BLOCK_WATER},
    {CRAFTAX_WG_BLOCK_ENCHANTMENT_TABLE_FIRE, CRAFTAX_WG_BLOCK_FOUNTAIN, CRAFTAX_WG_BLOCK_PATH},
};

static __device__ inline float craftax_wg_clampf(float value, float low, float high) {
    if (value < low) {
        return low;
    }
    if (value > high) {
        return high;
    }
    return value;
}

static __device__ inline int craftax_wg_clampi(int value, int low, int high) {
    if (value < low) {
        return low;
    }
    if (value > high) {
        return high;
    }
    return value;
}

static __device__ inline size_t craftax_wg_index(int row, int col) {
    return (size_t)row * (size_t)CRAFTAX_WG_MAP_SIZE + (size_t)col;
}

static __device__ inline void craftax_threefry_split3(
    CraftaxThreefryKey key,
    CraftaxThreefryKey* first,
    CraftaxThreefryKey* second,
    CraftaxThreefryKey* third
) {
    CraftaxThreefryKey keys[3];
    craftax_threefry_split_n(key, keys, 3);
    *first = keys[0];
    *second = keys[1];
    *third = keys[2];
}

static __device__ inline CraftaxThreefryKey craftax_worldgen_key_from_seed(uint32_t seed) {
    CraftaxThreefryKey key = craftax_prng_key(seed);
    CraftaxThreefryKey carry;
    CraftaxThreefryKey reset_key;
    craftax_threefry_split(key, &carry, &reset_key);

    CraftaxThreefryKey reset_carry;
    CraftaxThreefryKey world_key;
    craftax_threefry_split(reset_key, &reset_carry, &world_key);
    return world_key;
}

static __device__ inline CraftaxThreefryKey craftax_overworld_rng_from_seed(uint32_t seed) {
    CraftaxThreefryKey world_key = craftax_worldgen_key_from_seed(seed);
    CraftaxThreefryKey world_keys[7];
    craftax_threefry_split_n(world_key, world_keys, 7);
    return world_keys[1];
}

static __device__ inline uint32_t craftax_randint_u32_at(
    CraftaxThreefryKey key,
    uint64_t index,
    uint32_t minval,
    uint32_t maxval
) {
    uint32_t span = maxval > minval ? maxval - minval : 1u;
    // Fast path for power-of-2 spans: just mask
    if ((span & (span - 1)) == 0) {
        uint32_t bits = craftax_threefry_uniform_u32_at(key, index);
        return minval + (bits & (span - 1));
    }
    // General path: use top-32 of hash, scale to span
    uint64_t h = craftax_fast_hash64(key, index);
    return minval + (uint32_t)(((h >> 32) * (uint64_t)span) >> 32);
}

static __device__ inline int32_t craftax_randint_i32_at(
    CraftaxThreefryKey key,
    uint64_t index,
    int32_t minval,
    int32_t maxval
) {
    return (int32_t)craftax_randint_u32_at(
        key,
        index,
        (uint32_t)minval,
        (uint32_t)maxval
    );
}

static __device__ inline int craftax_choice_bool_flat(
    CraftaxThreefryKey key,
    const bool* valid,
    int count
) {
    int valid_count = 0;
    int last_valid = 0;
    for (int i = 0; i < count; i++) {
        if (valid[i]) {
            valid_count++;
            last_valid = i;
        }
    }
    if (valid_count == 0) {
        return 0;
    }

    float draw = (float)valid_count * (1.0f - craftax_threefry_uniform_f32(key));
    float cumulative = 0.0f;
    for (int i = 0; i < count; i++) {
        if (valid[i]) {
            cumulative += 1.0f;
        }
        if (cumulative >= draw) {
            return i;
        }
    }
    return last_valid;
}

static __device__ inline float craftax_torch_light_value(int row, int col, float default_light) {
    float dr = (float)(row - 4);
    float dc = (float)(col - 4);
    float distance = sqrtf(dr * dr + dc * dc);
    float torch = craftax_wg_clampf(1.0f - distance * (1.0f / 5.0f), 0.0f, 1.0f);
    return torch * (1.0f - default_light) + default_light;
}

static __device__ inline void craftax_apply_ladder_light(
    uint8_t light_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    const int32_t ladder_up[2],
    float default_light
) {
    int start_row = ladder_up[0] - 4;
    int start_col = ladder_up[1] - 4;
    if (start_row < 0) {
        start_row += CRAFTAX_WG_MAP_SIZE;
    }
    if (start_col < 0) {
        start_col += CRAFTAX_WG_MAP_SIZE;
    }
    start_row = craftax_wg_clampi(start_row, 0, CRAFTAX_WG_MAP_SIZE - 9);
    start_col = craftax_wg_clampi(start_col, 0, CRAFTAX_WG_MAP_SIZE - 9);
    for (int row = 0; row < 9; row++) {
        for (int col = 0; col < 9; col++) {
            light_map[start_row + row][start_col + col] =
                (uint8_t)(craftax_torch_light_value(row, col, default_light) * 255.0f);
        }
    }
}

static __device__ inline void craftax_add_lava_light(
    uint8_t light_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    const bool lava_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    bool lava_emits_light
) {
    if (!lava_emits_light) {
        return;
    }

    static const float kernel[3][3] = {
        {0.2f, 0.7f, 0.2f},
        {0.7f, 1.0f, 0.7f},
        {0.2f, 0.7f, 0.2f},
    };

    for (int row = 0; row < CRAFTAX_WG_MAP_SIZE; row++) {
        for (int col = 0; col < CRAFTAX_WG_MAP_SIZE; col++) {
            float add = 0.0f;
            for (int kr = 0; kr < 3; kr++) {
                int src_row = row + kr - 1;
                if (src_row < 0 || src_row >= CRAFTAX_WG_MAP_SIZE) {
                    continue;
                }
                for (int kc = 0; kc < 3; kc++) {
                    int src_col = col + kc - 1;
                    if (src_col < 0 || src_col >= CRAFTAX_WG_MAP_SIZE) {
                        continue;
                    }
                    add += lava_map[src_row][src_col] ? kernel[kr][kc] : 0.0f;
                }
            }
            float new_light = craftax_wg_clampf(light_map[row][col] * (1.0f / 255.0f) + add, 0.0f, 1.0f);
            light_map[row][col] = (uint8_t)(new_light * 255.0f);
        }
    }
}

static __device__ inline int craftax_smooth_config_index_for_floor(int floor_idx) {
    switch (floor_idx) {
        case 0:
            return 0;
        case 2:
            return 1;
        case 5:
            return 2;
        case 6:
            return 3;
        case 7:
            return 4;
        case 8:
            return 5;
        default:
            return -1;
    }
}

static __device__ inline int craftax_dungeon_config_index_for_floor(int floor_idx) {
    switch (floor_idx) {
        case 1:
            return 0;
        case 3:
            return 1;
        case 4:
            return 2;
        default:
            return -1;
    }
}

// ============================================================
// Smoothworld tile passes: scalar reference implementations plus AVX-512
// twins. The SIMD versions use only IEEE-exact per-lane operations in the
// same order as the scalar code (no FMA, no reassociation), so their output
// is bit-identical; g_craftax_wg_force_scalar exists for testing that claim.
// ============================================================
static __device__ int g_craftax_wg_force_scalar = 0;

static __device__ inline void craftax_smoothworld_classify_scalar(
    const CraftaxSmoothGenConfig* config,
    CraftaxThreefryKey tree_uniform_key,
    float* water,
    float* mountain,
    const float* path_x,
    const float* tree_noise,
    uint8_t map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    uint8_t item_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    uint8_t light_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE]
) {
    const int size = CRAFTAX_WG_MAP_SIZE;
    const int player_row = CRAFTAX_WG_MAP_SIZE / 2;
    const int player_col = CRAFTAX_WG_MAP_SIZE / 2;
    for (int row = 0; row < size; row++) {
        int dr = row > player_row ? row - player_row : player_row - row;
        for (int col = 0; col < size; col++) {
            int dc = col > player_col ? col - player_col : player_col - col;
            float distance = sqrtf((float)(dr * dr + dc * dc));
            float proximity_water = craftax_wg_clampf(
                distance / config->player_proximity_map_water_strength,
                0.0f,
                config->player_proximity_map_water_max
            );
            float proximity_mountain = craftax_wg_clampf(
                distance / config->player_proximity_map_mountain_strength,
                0.0f,
                config->player_proximity_map_mountain_max
            );
            size_t idx = craftax_wg_index(row, col);

            water[idx] = water[idx] + proximity_water - 1.0f;
            int32_t block = water[idx] > config->water_threshold
                ? config->sea_block
                : config->default_block;
            bool sand = water[idx] > config->sand_threshold && block != config->sea_block;
            if (sand) {
                block = config->coast_block;
            }

            mountain[idx] = mountain[idx] + 0.05f + proximity_mountain - 1.0f;
            if (mountain[idx] > 0.7f) {
                block = config->mountain_block;
            }

            bool path = mountain[idx] > 0.7f && path_x[idx] > 0.8f;
            if (path) {
                block = config->path_block;
            }

            float path_y = path_x[craftax_wg_index(col, row)];
            path = mountain[idx] > 0.7f && path_y > 0.8f;
            if (path) {
                block = config->path_block;
            }

            bool cave = mountain[idx] > 0.85f && water[idx] > 0.4f;
            if (cave) {
                block = config->inner_mountain_block;
            }

            float tree_draw = craftax_threefry_uniform_f32_at(tree_uniform_key, idx);
            bool tree = tree_noise[idx] > config->tree_threshold_perlin
                && tree_draw > config->tree_threshold_uniform;
            if (tree && block == config->tree_requirement_block) {
                block = config->tree;
            }

            map[row][col] = (uint8_t)block;
            item_map[row][col] = CRAFTAX_WG_ITEM_NONE;
            light_map[row][col] = (uint8_t)(config->default_light * 255.0f);
        }
    }
}

static __device__ inline void craftax_smoothworld_ore_scalar(
    const CraftaxSmoothGenConfig* config,
    int ore_index,
    CraftaxThreefryKey ore_key,
    uint8_t map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE]
) {
    const int size = CRAFTAX_WG_MAP_SIZE;
    for (int row = 0; row < size; row++) {
        for (int col = 0; col < size; col++) {
            size_t idx = craftax_wg_index(row, col);
            bool is_ore = map[row][col] == config->ore_requirement_blocks[ore_index]
                && craftax_threefry_uniform_f32_at(ore_key, idx) < config->ore_chances[ore_index];
            if (is_ore) {
                map[row][col] = (uint8_t)config->ores[ore_index];
            }
        }
    }
}

static __device__ inline void craftax_smoothworld_lava_scalar(
    const CraftaxSmoothGenConfig* config,
    const float* mountain,
    const float* tree_noise,
    uint8_t map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    bool lava_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE]
) {
    const int size = CRAFTAX_WG_MAP_SIZE;
    for (int row = 0; row < size; row++) {
        for (int col = 0; col < size; col++) {
            size_t idx = craftax_wg_index(row, col);
            lava_map[row][col] = mountain[idx] > 0.85f && tree_noise[idx] > 0.7f;
            if (lava_map[row][col]) {
                map[row][col] = (uint8_t)config->lava;
            }
        }
    }
}

#if defined(__AVX512F__) && defined(__AVX512DQ__) && !defined(CRAFTAX_NO_SIMD_NOISE)
#define CRAFTAX_WG_SIMD 1
#include <immintrin.h>

static __device__ inline __m512i craftax_wg_mix64_v(__m512i x) {
    x = _mm512_xor_si512(x, _mm512_srli_epi64(x, 33));
    x = _mm512_mullo_epi64(x, _mm512_set1_epi64((long long)0xff51afd7ed558ccdULL));
    x = _mm512_xor_si512(x, _mm512_srli_epi64(x, 33));
    x = _mm512_mullo_epi64(x, _mm512_set1_epi64((long long)0xc4ceb9fe1a85ec53ULL));
    x = _mm512_xor_si512(x, _mm512_srli_epi64(x, 33));
    return x;
}

// craftax_threefry_uniform_f32_at for 16 consecutive counters [base, base+16).
static __device__ inline __m512 craftax_wg_uniform_f32x16(uint64_t key, uint64_t base) {
    __m512i lane8 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
    __m512i k = _mm512_set1_epi64((long long)key);
    __m512i idx_lo = _mm512_add_epi64(_mm512_set1_epi64((long long)base), lane8);
    __m512i idx_hi = _mm512_add_epi64(_mm512_set1_epi64((long long)(base + 8)), lane8);
    __m512i h_lo = craftax_wg_mix64_v(_mm512_xor_si512(k, idx_lo));
    __m512i h_hi = craftax_wg_mix64_v(_mm512_xor_si512(k, idx_hi));
    // u32 = (uint32_t)h ^ (uint32_t)(h >> 32), then keep low 32 of each lane
    h_lo = _mm512_xor_si512(h_lo, _mm512_srli_epi64(h_lo, 32));
    h_hi = _mm512_xor_si512(h_hi, _mm512_srli_epi64(h_hi, 32));
    __m256i lo32 = _mm512_cvtepi64_epi32(h_lo);
    __m256i hi32 = _mm512_cvtepi64_epi32(h_hi);
    __m512i bits = _mm512_inserti64x4(_mm512_castsi256_si512(lo32), hi32, 1);
    // float in [0,1): ((bits >> 9) | 0x3F800000) - 1.0f
    bits = _mm512_or_si512(_mm512_srli_epi32(bits, 9),
                           _mm512_set1_epi32(0x3F800000));
    return _mm512_sub_ps(_mm512_castsi512_ps(bits), _mm512_set1_ps(1.0f));
}

static __device__ inline void craftax_smoothworld_classify_avx512(
    const CraftaxSmoothGenConfig* config,
    CraftaxThreefryKey tree_uniform_key,
    float* water,
    float* mountain,
    const float* path_x,
    const float* tree_noise,
    uint8_t map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    uint8_t item_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    uint8_t light_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE]
) {
    const int size = CRAFTAX_WG_MAP_SIZE;
    const int player_col = CRAFTAX_WG_MAP_SIZE / 2;
    uint64_t tree_key = craftax_key_to_u64(tree_uniform_key);

    __m512i lane = _mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
    __m512 zero = _mm512_setzero_ps();
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 inv_w_strength =
        _mm512_set1_ps(config->player_proximity_map_water_strength);
    __m512 inv_m_strength =
        _mm512_set1_ps(config->player_proximity_map_mountain_strength);
    __m512 w_max = _mm512_set1_ps(config->player_proximity_map_water_max);
    __m512 m_max = _mm512_set1_ps(config->player_proximity_map_mountain_max);
    __m512 water_thresh = _mm512_set1_ps(config->water_threshold);
    __m512 sand_thresh = _mm512_set1_ps(config->sand_threshold);
    __m512 c07 = _mm512_set1_ps(0.7f);
    __m512 c08 = _mm512_set1_ps(0.8f);
    __m512 c085 = _mm512_set1_ps(0.85f);
    __m512 c04 = _mm512_set1_ps(0.4f);
    __m512 c005 = _mm512_set1_ps(0.05f);
    __m512 tree_perlin = _mm512_set1_ps(config->tree_threshold_perlin);
    __m512 tree_uniform = _mm512_set1_ps(config->tree_threshold_uniform);
    __m512i default_b = _mm512_set1_epi32(config->default_block);
    __m512i sea_b = _mm512_set1_epi32(config->sea_block);
    __m512i coast_b = _mm512_set1_epi32(config->coast_block);
    __m512i mountain_b = _mm512_set1_epi32(config->mountain_block);
    __m512i path_b = _mm512_set1_epi32(config->path_block);
    __m512i inner_b = _mm512_set1_epi32(config->inner_mountain_block);
    __m512i tree_req_b = _mm512_set1_epi32(config->tree_requirement_block);
    __m512i tree_b = _mm512_set1_epi32(config->tree);
    __m128i item_zero = _mm_setzero_si128();
    __m128i light_fill =
        _mm_set1_epi8((char)(uint8_t)(config->default_light * 255.0f));

    for (int row = 0; row < size; row++) {
        int player_row = CRAFTAX_WG_MAP_SIZE / 2;
        int dr = row > player_row ? row - player_row : player_row - row;
        __m512i dr2 = _mm512_set1_epi32(dr * dr);

        for (int col = 0; col < size; col += 16) {
            __m512i c = _mm512_add_epi32(_mm512_set1_epi32(col), lane);
            __m512i dc = _mm512_abs_epi32(
                _mm512_sub_epi32(c, _mm512_set1_epi32(player_col)));
            __m512 distance = _mm512_sqrt_ps(_mm512_cvtepi32_ps(
                _mm512_add_epi32(dr2, _mm512_mullo_epi32(dc, dc))));

            // clampf(distance / strength, 0, max) == min(max(v, 0), max)
            __m512 prox_w = _mm512_min_ps(
                _mm512_max_ps(_mm512_div_ps(distance, inv_w_strength), zero),
                w_max);
            __m512 prox_m = _mm512_min_ps(
                _mm512_max_ps(_mm512_div_ps(distance, inv_m_strength), zero),
                m_max);

            size_t idx = (size_t)row * (size_t)size + (size_t)col;
            __m512 w = _mm512_loadu_ps(&water[idx]);
            w = _mm512_sub_ps(_mm512_add_ps(w, prox_w), one);
            _mm512_storeu_ps(&water[idx], w);

            __mmask16 sea = _mm512_cmp_ps_mask(w, water_thresh, _CMP_GT_OQ);
            __m512i block = _mm512_mask_blend_epi32(sea, default_b, sea_b);
            // Scalar condition is `block != sea_block`, which differs from
            // `!sea` when default_block == sea_block (configs 1 and 4).
            __mmask16 sand = _mm512_kand(
                _mm512_cmpneq_epi32_mask(block, sea_b),
                _mm512_cmp_ps_mask(w, sand_thresh, _CMP_GT_OQ));
            block = _mm512_mask_blend_epi32(sand, block, coast_b);

            __m512 m = _mm512_loadu_ps(&mountain[idx]);
            m = _mm512_sub_ps(
                _mm512_add_ps(_mm512_add_ps(m, c005), prox_m), one);
            _mm512_storeu_ps(&mountain[idx], m);

            __mmask16 mnt = _mm512_cmp_ps_mask(m, c07, _CMP_GT_OQ);
            block = _mm512_mask_blend_epi32(mnt, block, mountain_b);

            __m512 px = _mm512_loadu_ps(&path_x[idx]);
            __mmask16 path = _mm512_kand(
                mnt, _mm512_cmp_ps_mask(px, c08, _CMP_GT_OQ));
            block = _mm512_mask_blend_epi32(path, block, path_b);

            // path_x transposed: path_x[col * size + row]
            __m512i t_idx = _mm512_add_epi32(
                _mm512_mullo_epi32(c, _mm512_set1_epi32(size)),
                _mm512_set1_epi32(row));
            __m512 py = _mm512_i32gather_ps(t_idx, path_x, 4);
            path = _mm512_kand(mnt, _mm512_cmp_ps_mask(py, c08, _CMP_GT_OQ));
            block = _mm512_mask_blend_epi32(path, block, path_b);

            __mmask16 cave = _mm512_kand(
                _mm512_cmp_ps_mask(m, c085, _CMP_GT_OQ),
                _mm512_cmp_ps_mask(w, c04, _CMP_GT_OQ));
            block = _mm512_mask_blend_epi32(cave, block, inner_b);

            __m512 draw = craftax_wg_uniform_f32x16(tree_key, (uint64_t)idx);
            __m512 tn = _mm512_loadu_ps(&tree_noise[idx]);
            __mmask16 tree = _mm512_kand(
                _mm512_cmp_ps_mask(tn, tree_perlin, _CMP_GT_OQ),
                _mm512_cmp_ps_mask(draw, tree_uniform, _CMP_GT_OQ));
            tree = _mm512_kand(
                tree, _mm512_cmpeq_epi32_mask(block, tree_req_b));
            block = _mm512_mask_blend_epi32(tree, block, tree_b);

            _mm_storeu_si128((__m128i*)&map[row][col],
                             _mm512_cvtepi32_epi8(block));
            _mm_storeu_si128((__m128i*)&item_map[row][col], item_zero);
            _mm_storeu_si128((__m128i*)&light_map[row][col], light_fill);
        }
    }
}

static __device__ inline void craftax_smoothworld_ore_avx512(
    const CraftaxSmoothGenConfig* config,
    int ore_index,
    CraftaxThreefryKey ore_key,
    uint8_t map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE]
) {
    const int size = CRAFTAX_WG_MAP_SIZE;
    uint64_t key = craftax_key_to_u64(ore_key);
    __m512i req = _mm512_set1_epi32(config->ore_requirement_blocks[ore_index]);
    __m512i ore = _mm512_set1_epi32(config->ores[ore_index]);
    __m512 chance = _mm512_set1_ps(config->ore_chances[ore_index]);
    for (int row = 0; row < size; row++) {
        for (int col = 0; col < size; col += 16) {
            size_t idx = (size_t)row * (size_t)size + (size_t)col;
            __m512i blocks = _mm512_cvtepu8_epi32(
                _mm_loadu_si128((const __m128i*)&map[row][col]));
            __mmask16 is_req = _mm512_cmpeq_epi32_mask(blocks, req);
            if (!is_req) continue;
            __m512 draw = craftax_wg_uniform_f32x16(key, (uint64_t)idx);
            __mmask16 is_ore = _mm512_kand(
                is_req, _mm512_cmp_ps_mask(draw, chance, _CMP_LT_OQ));
            blocks = _mm512_mask_blend_epi32(is_ore, blocks, ore);
            _mm_storeu_si128((__m128i*)&map[row][col],
                             _mm512_cvtepi32_epi8(blocks));
        }
    }
}

static __device__ inline void craftax_smoothworld_lava_avx512(
    const CraftaxSmoothGenConfig* config,
    const float* mountain,
    const float* tree_noise,
    uint8_t map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    bool lava_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE]
) {
    const int size = CRAFTAX_WG_MAP_SIZE;
    __m512 c085 = _mm512_set1_ps(0.85f);
    __m512 c07 = _mm512_set1_ps(0.7f);
    __m512i lava_b = _mm512_set1_epi32(config->lava);
    __m512i one32 = _mm512_set1_epi32(1);
    for (int row = 0; row < size; row++) {
        for (int col = 0; col < size; col += 16) {
            size_t idx = (size_t)row * (size_t)size + (size_t)col;
            __mmask16 lava = _mm512_kand(
                _mm512_cmp_ps_mask(
                    _mm512_loadu_ps(&mountain[idx]), c085, _CMP_GT_OQ),
                _mm512_cmp_ps_mask(
                    _mm512_loadu_ps(&tree_noise[idx]), c07, _CMP_GT_OQ));
            _mm_storeu_si128((__m128i*)&lava_map[row][col],
                             _mm512_cvtepi32_epi8(
                                 _mm512_maskz_mov_epi32(lava, one32)));
            __m512i blocks = _mm512_cvtepu8_epi32(
                _mm_loadu_si128((const __m128i*)&map[row][col]));
            blocks = _mm512_mask_blend_epi32(lava, blocks, lava_b);
            _mm_storeu_si128((__m128i*)&map[row][col],
                             _mm512_cvtepi32_epi8(blocks));
        }
    }
}
#endif  // CRAFTAX_WG_SIMD

static __device__ inline void craftax_generate_smoothworld_config(
    CraftaxThreefryKey rng,
    int config_idx,
    uint8_t map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    uint8_t item_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    uint8_t light_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    int32_t ladder_down[2],
    int32_t ladder_up[2]
) {
    const CraftaxSmoothGenConfig* config = &CRAFTAX_SMOOTHGEN_CONFIGS[config_idx];
    const int size = CRAFTAX_WG_MAP_SIZE;
    const int player_row = CRAFTAX_WG_MAP_SIZE / 2;
    const int player_col = CRAFTAX_WG_MAP_SIZE / 2;
    const size_t cells = CRAFTAX_WG_MAP_CELLS;

    CraftaxThreefryKey subkey;
    float water[CRAFTAX_WG_MAP_CELLS];
    float mountain[CRAFTAX_WG_MAP_CELLS];
    float path_x[CRAFTAX_WG_MAP_CELLS];
    float tree_noise[CRAFTAX_WG_MAP_CELLS];
    bool lava_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE];

    craftax_threefry_split(rng, &rng, &subkey);
    craftax_generate_fractal_noise_2d(subkey, size, size, 3, 3, 1, 0.5f, 2, NULL, water);

    craftax_threefry_split(rng, &rng, &subkey);
    (void)subkey;

    craftax_threefry_split(rng, &rng, &subkey);
    craftax_generate_fractal_noise_2d(subkey, size, size, 3, 3, 1, 0.5f, 2, NULL, mountain);

    craftax_threefry_split(rng, &rng, &subkey);
    craftax_generate_fractal_noise_2d(subkey, size, size, 6, 24, 1, 0.5f, 2, NULL, path_x);

    craftax_threefry_split(rng, &rng, &subkey);
    (void)subkey;

    craftax_threefry_split(rng, &rng, &subkey);
    CraftaxThreefryKey tree_uniform_key = rng;
    craftax_generate_fractal_noise_2d(subkey, size, size, 12, 12, 1, 0.5f, 2, NULL, tree_noise);

#ifdef CRAFTAX_WG_SIMD
    bool use_simd = !g_craftax_wg_force_scalar;
#else
    bool use_simd = false;
#endif
    (void)use_simd;

#ifdef CRAFTAX_WG_SIMD
    if (use_simd) {
        craftax_smoothworld_classify_avx512(
            config, tree_uniform_key, water, mountain, path_x, tree_noise,
            map, item_map, light_map);
    } else
#endif
    {
        craftax_smoothworld_classify_scalar(
            config, tree_uniform_key, water, mountain, path_x, tree_noise,
            map, item_map, light_map);
    }
    (void)player_col;

    CraftaxThreefryKey ore_rng;
    craftax_threefry_split(rng, &rng, &ore_rng);
    for (int ore_index = 0; ore_index < 5; ore_index++) {
        CraftaxThreefryKey ore_key;
        craftax_threefry_split(ore_rng, &ore_rng, &ore_key);
#ifdef CRAFTAX_WG_SIMD
        if (use_simd) {
            craftax_smoothworld_ore_avx512(config, ore_index, ore_key, map);
            continue;
        }
#endif
        craftax_smoothworld_ore_scalar(config, ore_index, ore_key, map);
    }

#ifdef CRAFTAX_WG_SIMD
    if (use_simd) {
        craftax_smoothworld_lava_avx512(config, mountain, tree_noise, map, lava_map);
    } else
#endif
    {
        craftax_smoothworld_lava_scalar(config, mountain, tree_noise, map, lava_map);
    }

    craftax_threefry_split(rng, &rng, &subkey);
    bool valid_diamond[CRAFTAX_WG_MAP_CELLS];
    for (int row = 0; row < size; row++) {
        for (int col = 0; col < size; col++) {
            valid_diamond[craftax_wg_index(row, col)] = map[row][col] == CRAFTAX_WG_BLOCK_STONE;
        }
    }
    int diamond_index = craftax_choice_bool_flat(subkey, valid_diamond, (int)cells);
    map[diamond_index / size][diamond_index % size] = (uint8_t)CRAFTAX_WG_BLOCK_STONE;

    map[player_row][player_col] = (uint8_t)config->player_spawn;

    bool valid_ladder[CRAFTAX_WG_MAP_CELLS];
    for (int row = 0; row < size; row++) {
        for (int col = 0; col < size; col++) {
            valid_ladder[craftax_wg_index(row, col)] = map[row][col] == config->valid_ladder;
        }
    }

    craftax_threefry_split(rng, &rng, &subkey);
    int ladder_down_index = craftax_choice_bool_flat(subkey, valid_ladder, (int)cells);
    ladder_down[0] = ladder_down_index / size;
    ladder_down[1] = ladder_down_index % size;
    if (config->ladder_down) {
        item_map[ladder_down[0]][ladder_down[1]] = CRAFTAX_WG_ITEM_LADDER_DOWN;
    }

    craftax_threefry_split(rng, &rng, &subkey);
    int ladder_up_index = craftax_choice_bool_flat(subkey, valid_ladder, (int)cells);
    ladder_up[0] = ladder_up_index / size;
    ladder_up[1] = ladder_up_index % size;

    craftax_apply_ladder_light(light_map, ladder_up, config->default_light);
    craftax_add_lava_light(light_map, lava_map, config->lava == CRAFTAX_WG_BLOCK_LAVA);

    if (config->ladder_up) {
        item_map[ladder_up[0]][ladder_up[1]] = CRAFTAX_WG_ITEM_LADDER_UP;
    }
}

static __device__ inline void craftax_generate_smoothworld_floor(
    CraftaxThreefryKey seed_key,
    int floor_idx,
    uint8_t map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    uint8_t item_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    uint8_t light_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    int32_t ladder_down[2],
    int32_t ladder_up[2]
) {
    int config_idx = craftax_smooth_config_index_for_floor(floor_idx);
    if (config_idx < 0) {
        memset(map, 0, CRAFTAX_WG_MAP_CELLS * sizeof(uint8_t));
        memset(item_map, 0, CRAFTAX_WG_MAP_CELLS * sizeof(uint8_t));
        memset(light_map, 0, CRAFTAX_WG_MAP_CELLS * sizeof(uint8_t));
        ladder_down[0] = 0;
        ladder_down[1] = 0;
        ladder_up[0] = 0;
        ladder_up[1] = 0;
        return;
    }
    craftax_generate_smoothworld_config(
        seed_key,
        config_idx,
        map,
        item_map,
        light_map,
        ladder_down,
        ladder_up
    );
}

static __device__ inline void craftax_generate_dungeon_config(
    CraftaxThreefryKey rng,
    int config_idx,
    uint8_t map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    uint8_t item_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    uint8_t light_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    int32_t ladder_down[2],
    int32_t ladder_up[2]
) {
    const CraftaxDungeonConfig* config = &CRAFTAX_DUNGEON_CONFIGS[config_idx];
    const int chunk_size = 16;
    const int world_chunk_height = CRAFTAX_WG_MAP_SIZE / chunk_size;
    const int num_rooms = 8;
    const int min_room_size = 5;
    const int max_room_size = 10;
    const int padded_size = CRAFTAX_WG_MAP_SIZE + 2 * max_room_size;

    uint8_t padded_map[68][68];
    uint8_t padded_item_map[68][68];
    bool room_occupancy_chunks[9];
    int32_t room_sizes[8][2];
    int32_t room_positions[8][2];

    for (int row = 0; row < padded_size; row++) {
        for (int col = 0; col < padded_size; col++) {
            bool inner = row >= max_room_size
                && row < max_room_size + CRAFTAX_WG_MAP_SIZE
                && col >= max_room_size
                && col < max_room_size + CRAFTAX_WG_MAP_SIZE;
            padded_map[row][col] = inner ? CRAFTAX_WG_BLOCK_WALL : 0;
            padded_item_map[row][col] = CRAFTAX_WG_ITEM_NONE;
        }
    }
    for (int i = 0; i < 9; i++) {
        room_occupancy_chunks[i] = true;
    }

    CraftaxThreefryKey room_scan_ignored_key;
    CraftaxThreefryKey room_size_key;
    craftax_threefry_split3(rng, &rng, &room_scan_ignored_key, &room_size_key);
    (void)room_scan_ignored_key;
    for (int room = 0; room < num_rooms; room++) {
        room_sizes[room][0] = craftax_randint_i32_at(room_size_key, (uint64_t)room * 2u, min_room_size, max_room_size);
        room_sizes[room][1] = craftax_randint_i32_at(room_size_key, (uint64_t)room * 2u + 1u, min_room_size, max_room_size);
    }

    CraftaxThreefryKey room_rng;
    craftax_threefry_split(rng, &rng, &room_rng);

    for (int room_index = 0; room_index < num_rooms; room_index++) {
        CraftaxThreefryKey choice_key;
        craftax_threefry_split(room_rng, &room_rng, &choice_key);
        int room_chunk = craftax_choice_bool_flat(choice_key, room_occupancy_chunks, 9);
        room_occupancy_chunks[room_chunk] = false;

        int room_row = (room_chunk % world_chunk_height) * chunk_size + max_room_size;
        int room_col = (room_chunk / world_chunk_height) * chunk_size + max_room_size;
        CraftaxThreefryKey position_key;
        craftax_threefry_split(room_rng, &room_rng, &position_key);
        room_row += craftax_randint_i32_at(position_key, 0, 0, chunk_size - min_room_size);
        room_col += craftax_randint_i32_at(position_key, 1, 0, chunk_size - min_room_size);
        room_positions[room_index][0] = room_row;
        room_positions[room_index][1] = room_col;

        for (int row = 0; row < max_room_size; row++) {
            for (int col = 0; col < max_room_size; col++) {
                if (row < room_sizes[room_index][0] && col < room_sizes[room_index][1]) {
                    padded_map[room_row + row][room_col + col] = CRAFTAX_WG_BLOCK_PATH;
                }
            }
        }

        padded_item_map[room_row][room_col] = CRAFTAX_WG_ITEM_TORCH;
        padded_item_map[room_row + room_sizes[room_index][0] - 1][room_col] = CRAFTAX_WG_ITEM_TORCH;
        padded_item_map[room_row][room_col + room_sizes[room_index][1] - 1] = CRAFTAX_WG_ITEM_TORCH;
        padded_item_map[room_row + room_sizes[room_index][0] - 1][room_col + room_sizes[room_index][1] - 1] = CRAFTAX_WG_ITEM_TORCH;

        CraftaxThreefryKey chest_key;
        craftax_threefry_split(room_rng, &room_rng, &chest_key);
        int chest_row = craftax_randint_i32_at(chest_key, 0, 1, room_sizes[room_index][0] - 1);
        int chest_col = craftax_randint_i32_at(chest_key, 1, 1, room_sizes[room_index][1] - 1);
        padded_map[room_row + chest_row][room_col + chest_col] = CRAFTAX_WG_BLOCK_CHEST;

        CraftaxThreefryKey fountain_key;
        CraftaxThreefryKey fountain_uniform_key;
        craftax_threefry_split3(room_rng, &room_rng, &fountain_key, &fountain_uniform_key);
        int fountain_row = craftax_randint_i32_at(fountain_key, 0, 1, room_sizes[room_index][0] - 1);
        int fountain_col = craftax_randint_i32_at(fountain_key, 1, 1, room_sizes[room_index][1] - 1);
        bool room_has_fountain = craftax_threefry_uniform_f32(fountain_uniform_key) > 0.5f;
        if (room_has_fountain) {
            padded_map[room_row + fountain_row][room_col + fountain_col] = config->fountain_block;
        }
    }

    CraftaxThreefryKey path_rng;
    craftax_threefry_split(rng, &rng, &path_rng);
    bool included_rooms_mask[8] = {false, false, false, false, false, false, false, true};

    for (int path_index = 0; path_index < num_rooms; path_index++) {
        int source_row = room_positions[path_index][0];
        int source_col = room_positions[path_index][1];

        CraftaxThreefryKey sink_key;
        craftax_threefry_split(path_rng, &path_rng, &sink_key);
        int sink_index = craftax_choice_bool_flat(sink_key, included_rooms_mask, num_rooms);
        int sink_row = room_positions[sink_index][0];
        int sink_col = room_positions[sink_index][1];

        int horizontal_distance = sink_col - source_col;
        int horizontal_sign = (horizontal_distance > 0) - (horizontal_distance < 0);
        if (horizontal_sign != 0) {
            int abs_distance = horizontal_distance > 0 ? horizontal_distance : -horizontal_distance;
            for (int col = 0; col < padded_size; col++) {
                int path_index_col = (col - source_col) * horizontal_sign;
                bool horizontal_mask = path_index_col >= 0
                    && path_index_col <= abs_distance
                    && padded_map[source_row][col] == CRAFTAX_WG_BLOCK_WALL;
                if (horizontal_mask) {
                    padded_map[source_row][col] = CRAFTAX_WG_BLOCK_PATH;
                }
            }
        }

        int vertical_distance = sink_row - source_row;
        int vertical_sign = (vertical_distance > 0) - (vertical_distance < 0);
        if (vertical_sign != 0) {
            int abs_distance = vertical_distance > 0 ? vertical_distance : -vertical_distance;
            for (int row = 0; row < padded_size; row++) {
                int path_index_row = (row - source_row) * vertical_sign;
                bool vertical_mask = path_index_row >= 0
                    && path_index_row <= abs_distance
                    && padded_map[row][sink_col] == CRAFTAX_WG_BLOCK_WALL;
                if (vertical_mask) {
                    padded_map[row][sink_col] = CRAFTAX_WG_BLOCK_PATH;
                }
            }
        }

        CraftaxThreefryKey unused_left;
        CraftaxThreefryKey next_path_rng;
        craftax_threefry_split(path_rng, &unused_left, &next_path_rng);
        path_rng = next_path_rng;
        included_rooms_mask[path_index] = true;
    }

    int special_row = room_positions[0][0] + 2;
    int special_col = room_positions[0][1] + 2;
    padded_map[special_row][special_col] = config->special_block;

    for (int row = 0; row < CRAFTAX_WG_MAP_SIZE; row++) {
        for (int col = 0; col < CRAFTAX_WG_MAP_SIZE; col++) {
            map[row][col] = padded_map[row + max_room_size][col + max_room_size];
            item_map[row][col] = padded_item_map[row + max_room_size][col + max_room_size];
        }
    }

    bool adjacent_path[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE];
    for (int row = 0; row < CRAFTAX_WG_MAP_SIZE; row++) {
        for (int col = 0; col < CRAFTAX_WG_MAP_SIZE; col++) {
            bool adjacent = map[row][col] != CRAFTAX_WG_BLOCK_WALL;
            adjacent = adjacent || (row > 0 && map[row - 1][col] != CRAFTAX_WG_BLOCK_WALL);
            adjacent = adjacent || (row + 1 < CRAFTAX_WG_MAP_SIZE && map[row + 1][col] != CRAFTAX_WG_BLOCK_WALL);
            adjacent = adjacent || (col > 0 && map[row][col - 1] != CRAFTAX_WG_BLOCK_WALL);
            adjacent = adjacent || (col + 1 < CRAFTAX_WG_MAP_SIZE && map[row][col + 1] != CRAFTAX_WG_BLOCK_WALL);
            adjacent_path[row][col] = adjacent;
        }
    }

    CraftaxThreefryKey rare_key;
    craftax_threefry_split(rng, &rng, &rare_key);
    for (int row = 0; row < CRAFTAX_WG_MAP_SIZE; row++) {
        for (int col = 0; col < CRAFTAX_WG_MAP_SIZE; col++) {
            size_t idx = craftax_wg_index(row, col);
            bool rare = (1.0f - craftax_threefry_uniform_f32_at(rare_key, idx)) > 0.9f;
            int32_t wall_map = rare ? CRAFTAX_WG_BLOCK_WALL_MOSS : CRAFTAX_WG_BLOCK_WALL;
            bool rare_path = rare && map[row][col] == CRAFTAX_WG_BLOCK_PATH && item_map[row][col] == CRAFTAX_WG_ITEM_NONE;
            int32_t path_map = rare_path ? config->rare_path_replacement_block : map[row][col];
            bool is_wall_map = map[row][col] == CRAFTAX_WG_BLOCK_WALL && adjacent_path[row][col];
            bool is_darkness_map = !adjacent_path[row][col];

            if (is_darkness_map) {
                map[row][col] = CRAFTAX_WG_BLOCK_DARKNESS;
            } else if (is_wall_map) {
                map[row][col] = wall_map;
            } else {
                map[row][col] = path_map;
            }
            light_map[row][col] = 255;
        }
    }

    bool valid_ladder[CRAFTAX_WG_MAP_CELLS];
    for (int row = 0; row < CRAFTAX_WG_MAP_SIZE; row++) {
        for (int col = 0; col < CRAFTAX_WG_MAP_SIZE; col++) {
            valid_ladder[craftax_wg_index(row, col)] = map[row][col] == CRAFTAX_WG_BLOCK_PATH;
        }
    }

    CraftaxThreefryKey ladder_down_key;
    craftax_threefry_split(rng, &rng, &ladder_down_key);
    int ladder_down_index = craftax_choice_bool_flat(ladder_down_key, valid_ladder, CRAFTAX_WG_MAP_CELLS);
    ladder_down[0] = ladder_down_index / CRAFTAX_WG_MAP_SIZE;
    ladder_down[1] = ladder_down_index % CRAFTAX_WG_MAP_SIZE;
    item_map[ladder_down[0]][ladder_down[1]] = CRAFTAX_WG_ITEM_LADDER_DOWN;

    CraftaxThreefryKey ladder_up_key;
    craftax_threefry_split(rng, &rng, &ladder_up_key);
    int ladder_up_index = craftax_choice_bool_flat(ladder_up_key, valid_ladder, CRAFTAX_WG_MAP_CELLS);
    ladder_up[0] = ladder_up_index / CRAFTAX_WG_MAP_SIZE;
    ladder_up[1] = ladder_up_index % CRAFTAX_WG_MAP_SIZE;
    item_map[ladder_up[0]][ladder_up[1]] = CRAFTAX_WG_ITEM_LADDER_UP;
}

static __device__ inline void craftax_generate_dungeon_floor(
    CraftaxThreefryKey seed_key,
    int floor_idx,
    uint8_t map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    uint8_t item_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    uint8_t light_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    int32_t ladder_down[2],
    int32_t ladder_up[2]
) {
    int config_idx = craftax_dungeon_config_index_for_floor(floor_idx);
    if (config_idx < 0) {
        memset(map, 0, CRAFTAX_WG_MAP_CELLS * sizeof(uint8_t));
        memset(item_map, 0, CRAFTAX_WG_MAP_CELLS * sizeof(uint8_t));
        memset(light_map, 0, CRAFTAX_WG_MAP_CELLS * sizeof(uint8_t));
        ladder_down[0] = 0;
        ladder_down[1] = 0;
        ladder_up[0] = 0;
        ladder_up[1] = 0;
        return;
    }
    craftax_generate_dungeon_config(
        seed_key,
        config_idx,
        map,
        item_map,
        light_map,
        ladder_down,
        ladder_up
    );
}

static __device__ inline void craftax_permutation_6(CraftaxThreefryKey key, int32_t out[6]) {
    CraftaxThreefryKey carry;
    CraftaxThreefryKey sort_key;
    craftax_threefry_split(key, &carry, &sort_key);
    (void)carry;

    uint32_t keys[6];
    for (int i = 0; i < 6; i++) {
        keys[i] = craftax_threefry_uniform_u32_at(sort_key, (uint64_t)i);
        out[i] = i;
    }

    for (int i = 1; i < 6; i++) {
        uint32_t key_value = keys[i];
        int32_t value = out[i];
        int j = i - 1;
        while (j >= 0 && keys[j] > key_value) {
            keys[j + 1] = keys[j];
            out[j + 1] = out[j];
            j--;
        }
        keys[j + 1] = key_value;
        out[j + 1] = value;
    }
}

// [cuda port] Light level is a pure function of timestep computed with
// host libm (cosf) + gcc -ffast-math powf(x,3)->x*x*x expansion in the C
// reference. Device libm rounds differently, so the host harness computes
// the whole table with host code and uploads it (see harness below).
__device__ float g_craftax_light_table[CRAFTAX_DEFAULT_MAX_TIMESTEPS_TABLE];

static __device__ inline float craftax_calculate_initial_light_level(void) {
    // reference: 1.0f - powf(fabsf(cosf(pi * 0.3f)), 3.0f) == table[0]
    return g_craftax_light_table[0];
}

static __device__ inline void craftax_init_empty_mobs3(CraftaxWGMobs3* mobs) {
    for (int level = 0; level < CRAFTAX_WG_NUM_LEVELS; level++) {
        for (int mob = 0; mob < 3; mob++) {
            mobs->health[level][mob] = 1.0f;
        }
    }
}

static __device__ inline void craftax_init_empty_mobs2(CraftaxWGMobs2* mobs) {
    for (int level = 0; level < CRAFTAX_WG_NUM_LEVELS; level++) {
        for (int mob = 0; mob < 2; mob++) {
            mobs->health[level][mob] = 1.0f;
        }
    }
}

// Per-level generator dispatch: floors {0,2,5,6,7,8} are smoothworld configs
// 0..5, floors {1,3,4} are dungeon configs 0..2 (must match the floor orders
// in craftax_generate_world_from_key_lazy below).
static __device__ inline void craftax_generate_floor_from_key(
    CraftaxThreefryKey key,
    int level,
    CraftaxWorldState* out
) {
    static const int8_t is_dungeon[CRAFTAX_WG_NUM_LEVELS] =
        {0, 1, 0, 1, 1, 0, 0, 0, 0};
    static const int8_t cfg_index[CRAFTAX_WG_NUM_LEVELS] =
        {0, 0, 1, 1, 2, 2, 3, 4, 5};
    if (is_dungeon[level]) {
        craftax_generate_dungeon_config(
            key, cfg_index[level],
            out->map[level], out->item_map[level], out->light_map[level],
            out->down_ladders[level], out->up_ladders[level]);
    } else {
        craftax_generate_smoothworld_config(
            key, cfg_index[level],
            out->map[level], out->item_map[level], out->light_map[level],
            out->down_ladders[level], out->up_ladders[level]);
    }
}

static __device__ inline void craftax_generate_world_from_key_lazy(
    CraftaxThreefryKey rng,
    CraftaxWorldState* out,
    bool lazy
) {
    memset(out, 0, sizeof(*out));

    CraftaxThreefryKey smooth_split[7];
    craftax_threefry_split_n(rng, smooth_split, 7);
    rng = smooth_split[0];

    static const int smooth_floor_order[6] = {0, 2, 5, 6, 7, 8};
    for (int i = 0; i < 6; i++) {
        int level = smooth_floor_order[i];
        out->lazy_floor_keys[level][0] = smooth_split[i + 1].word[0];
        out->lazy_floor_keys[level][1] = smooth_split[i + 1].word[1];
        if (lazy && level != 0) continue;
        craftax_generate_smoothworld_config(
            smooth_split[i + 1],
            i,
            out->map[level],
            out->item_map[level],
            out->light_map[level],
            out->down_ladders[level],
            out->up_ladders[level]
        );
    }

    CraftaxThreefryKey dungeon_split[4];
    craftax_threefry_split_n(rng, dungeon_split, 4);
    rng = dungeon_split[0];

    static const int dungeon_floor_order[3] = {1, 3, 4};
    for (int i = 0; i < 3; i++) {
        int level = dungeon_floor_order[i];
        out->lazy_floor_keys[level][0] = dungeon_split[i + 1].word[0];
        out->lazy_floor_keys[level][1] = dungeon_split[i + 1].word[1];
        if (lazy) continue;
        craftax_generate_dungeon_config(
            dungeon_split[i + 1],
            i,
            out->map[level],
            out->item_map[level],
            out->light_map[level],
            out->down_ladders[level],
            out->up_ladders[level]
        );
    }

    out->lazy_floors_pending = lazy ? 0x1FEu : 0u;  // floors 1..8 deferred

    craftax_init_empty_mobs3(&out->melee_mobs);
    craftax_init_empty_mobs3(&out->passive_mobs);
    craftax_init_empty_mobs2(&out->ranged_mobs);
    craftax_init_empty_mobs3(&out->mob_projectiles);
    craftax_init_empty_mobs3(&out->player_projectiles);
    for (int level = 0; level < CRAFTAX_WG_NUM_LEVELS; level++) {
        for (int projectile = 0; projectile < CRAFTAX_WG_MAX_MOB_PROJECTILES; projectile++) {
            out->mob_projectile_directions[level][projectile][0] = 1;
            out->mob_projectile_directions[level][projectile][1] = 1;
        }
        for (int projectile = 0; projectile < CRAFTAX_WG_MAX_PLAYER_PROJECTILES; projectile++) {
            out->player_projectile_directions[level][projectile][0] = 1;
            out->player_projectile_directions[level][projectile][1] = 1;
        }
    }

    CraftaxThreefryKey potion_key;
    craftax_threefry_split(rng, &rng, &potion_key);
    craftax_permutation_6(potion_key, out->potion_mapping);

    CraftaxThreefryKey state_key;
    craftax_threefry_split(rng, &rng, &state_key);
    (void)rng;
    out->state_rng[0] = state_key.word[0];
    out->state_rng[1] = state_key.word[1];

    out->monsters_killed[0] = 10;
    out->player_position[0] = CRAFTAX_WG_MAP_SIZE / 2;
    out->player_position[1] = CRAFTAX_WG_MAP_SIZE / 2;
    out->player_level = 0;
    out->player_direction = CRAFTAX_WG_ACTION_UP;
    out->player_health = 9.0f;
    out->player_food = 9;
    out->player_drink = 9;
    out->player_energy = 9;
    out->player_mana = 9;
    out->player_dexterity = 1;
    out->player_strength = 1;
    out->player_intelligence = 1;
    out->boss_timesteps_to_spawn_this_round = CRAFTAX_WG_BOSS_FIGHT_SPAWN_TURNS;
    out->light_level = craftax_calculate_initial_light_level();
}

static __device__ inline void craftax_generate_world_from_key(
    CraftaxThreefryKey rng,
    CraftaxWorldState* out
) {
    craftax_generate_world_from_key_lazy(rng, out, false);
}

static __device__ inline void craftax_generate_world_from_seed(
    uint32_t seed,
    CraftaxWorldState* out
) {
    craftax_generate_world_from_key(craftax_worldgen_key_from_seed(seed), out);
}

static __device__ inline void craftax_generate_overworld_from_rng(
    CraftaxThreefryKey rng,
    CraftaxOverworldFloor* out
) {
    craftax_generate_smoothworld_config(
        rng,
        0,
        out->map,
        out->item_map,
        out->light_map,
        out->ladder_down,
        out->ladder_up
    );
}

static __device__ inline void craftax_generate_overworld_from_seed(
    uint32_t seed,
    CraftaxOverworldFloor* out
) {
    craftax_generate_overworld_from_rng(craftax_overworld_rng_from_seed(seed), out);
}

static __device__ inline int craftax_wg_jax_index(int32_t index, int32_t size) {
    if (index < 0) {
        index += size;
    }
    if (index < 0) {
        return 0;
    }
    if (index >= size) {
        return size - 1;
    }
    return index;
}

static __device__ inline bool craftax_wg_scatter_index(
    int32_t index,
    int32_t size,
    int* mapped_index
) {
    if (index < -size || index >= size) {
        return false;
    }
    *mapped_index = index < 0 ? index + size : index;
    return true;
}

static __device__ inline bool craftax_wg_is_boss_vulnerable(
    const CraftaxWorldState* state
) {
    int level = craftax_wg_jax_index(state->player_level, CRAFTAX_WG_NUM_LEVELS);
    bool has_melee = false;
    bool has_ranged = false;
    for (int i = 0; i < CRAFTAX_WG_MAX_MELEE_MOBS; i++) {
        has_melee = has_melee || state->melee_mobs.mask[level][i];
    }
    for (int i = 0; i < CRAFTAX_WG_MAX_RANGED_MOBS; i++) {
        has_ranged = has_ranged || state->ranged_mobs.mask[level][i];
    }
    return !has_melee
        && !has_ranged
        && state->boss_timesteps_to_spawn_this_round <= 0;
}

static __device__ inline void craftax_encode_mobs3_observation(
    const CraftaxWorldState* state,
    const CraftaxWGMobs3* mobs,
    int mob_class_index,
    int channels,
    int mob_channels_offset,
    float* obs
) {
    int level = craftax_wg_jax_index(state->player_level, CRAFTAX_WG_NUM_LEVELS);
    for (int i = 0; i < 3; i++) {
        int local_row = mobs->position[level][i][0]
            - state->player_position[0]
            + CRAFTAX_WG_OBS_ROWS / 2;
        int local_col = mobs->position[level][i][1]
            - state->player_position[1]
            + CRAFTAX_WG_OBS_COLS / 2;
        int type_id = mobs->type_id[level][i];
        int scatter_row;
        int scatter_col;
        if (!craftax_wg_scatter_index(
                local_row,
                CRAFTAX_WG_OBS_ROWS,
                &scatter_row
            )
            || !craftax_wg_scatter_index(
                local_col,
                CRAFTAX_WG_OBS_COLS,
                &scatter_col
            )
            || type_id < 0
            || type_id >= CRAFTAX_WG_NUM_MOB_TYPES) {
            continue;
        }

        bool on_screen = local_row >= 0
            && local_row < CRAFTAX_WG_OBS_ROWS
            && local_col >= 0
            && local_col < CRAFTAX_WG_OBS_COLS;
        int world_row = mobs->position[level][i][0];
        int world_col = mobs->position[level][i][1];
        bool in_bounds = world_row >= 0
            && world_row < CRAFTAX_WG_MAP_SIZE
            && world_col >= 0
            && world_col < CRAFTAX_WG_MAP_SIZE;
        bool visible = in_bounds && state->light_map[level][world_row][world_col] > 12;
        int obs_base = (scatter_row * CRAFTAX_WG_OBS_COLS + scatter_col) * channels;
        int channel = mob_channels_offset
            + mob_class_index * CRAFTAX_WG_NUM_MOB_TYPES
            + type_id;
        obs[obs_base + channel] =
            mobs->mask[level][i] && on_screen && visible ? 1.0f : 0.0f;
    }
}

static __device__ inline void craftax_encode_mobs2_observation(
    const CraftaxWorldState* state,
    const CraftaxWGMobs2* mobs,
    int mob_class_index,
    int channels,
    int mob_channels_offset,
    float* obs
) {
    int level = craftax_wg_jax_index(state->player_level, CRAFTAX_WG_NUM_LEVELS);
    for (int i = 0; i < 2; i++) {
        int local_row = mobs->position[level][i][0]
            - state->player_position[0]
            + CRAFTAX_WG_OBS_ROWS / 2;
        int local_col = mobs->position[level][i][1]
            - state->player_position[1]
            + CRAFTAX_WG_OBS_COLS / 2;
        int type_id = mobs->type_id[level][i];
        int scatter_row;
        int scatter_col;
        if (!craftax_wg_scatter_index(
                local_row,
                CRAFTAX_WG_OBS_ROWS,
                &scatter_row
            )
            || !craftax_wg_scatter_index(
                local_col,
                CRAFTAX_WG_OBS_COLS,
                &scatter_col
            )
            || type_id < 0
            || type_id >= CRAFTAX_WG_NUM_MOB_TYPES) {
            continue;
        }

        bool on_screen = local_row >= 0
            && local_row < CRAFTAX_WG_OBS_ROWS
            && local_col >= 0
            && local_col < CRAFTAX_WG_OBS_COLS;
        int world_row = mobs->position[level][i][0];
        int world_col = mobs->position[level][i][1];
        bool in_bounds = world_row >= 0
            && world_row < CRAFTAX_WG_MAP_SIZE
            && world_col >= 0
            && world_col < CRAFTAX_WG_MAP_SIZE;
        bool visible = in_bounds && state->light_map[level][world_row][world_col] > 12;
        int obs_base = (scatter_row * CRAFTAX_WG_OBS_COLS + scatter_col) * channels;
        int channel = mob_channels_offset
            + mob_class_index * CRAFTAX_WG_NUM_MOB_TYPES
            + type_id;
        obs[obs_base + channel] =
            mobs->mask[level][i] && on_screen && visible ? 1.0f : 0.0f;
    }
}

static __device__ inline void craftax_write_binary_bits(
    float* obs,
    int base,
    int value,
    int num_bits
) {
    if (num_bits == 6) {
        memcpy(obs + base, CRAFTAX_WG_BLOCK_LUT[value], 6 * sizeof(float));
    } else if (num_bits == 3) {
        memcpy(obs + base, CRAFTAX_WG_ITEM_LUT[value], 3 * sizeof(float));
    } else if (num_bits == 4) {
        memcpy(obs + base, CRAFTAX_WG_MOB_LUT[value], 4 * sizeof(float));
    } else {
        for (int i = 0; i < num_bits; i++) {
            obs[base + i] = (value & (1 << i)) ? 1.0f : 0.0f;
        }
    }
}

static __device__ inline void craftax_encode_mobs3_binary(
    const CraftaxWorldState* state,
    const CraftaxWGMobs3* mobs,
    int mob_class_index,
    int channels_per_cell,
    int mob_bits_offset,
    float* obs
) {
    int level = craftax_wg_jax_index(state->player_level, CRAFTAX_WG_NUM_LEVELS);
    for (int i = 0; i < 3; i++) {
        int type_id = mobs->type_id[level][i];
        if (type_id < 0 || type_id >= CRAFTAX_WG_NUM_MOB_TYPES
            || !mobs->mask[level][i]) {
            continue;
        }

        int local_row = mobs->position[level][i][0]
            - state->player_position[0]
            + CRAFTAX_WG_OBS_ROWS / 2;
        int local_col = mobs->position[level][i][1]
            - state->player_position[1]
            + CRAFTAX_WG_OBS_COLS / 2;
        if (local_row < 0 || local_row >= CRAFTAX_WG_OBS_ROWS
            || local_col < 0 || local_col >= CRAFTAX_WG_OBS_COLS) {
            continue;
        }

        int world_row = mobs->position[level][i][0];
        int world_col = mobs->position[level][i][1];
        if (world_row < 0 || world_row >= CRAFTAX_WG_MAP_SIZE
            || world_col < 0 || world_col >= CRAFTAX_WG_MAP_SIZE
            || state->light_map[level][world_row][world_col] <= 12) {
            continue;
        }

        int obs_base = (local_row * CRAFTAX_WG_OBS_COLS + local_col)
            * channels_per_cell;
        int class_offset = mob_bits_offset
            + mob_class_index * CRAFTAX_WG_BINARY_MOB_BITS;
        memcpy(obs + obs_base + class_offset,
               CRAFTAX_WG_MOB_LUT[type_id + 1],
               CRAFTAX_WG_BINARY_MOB_BITS * sizeof(float));
    }
}

static __device__ inline void craftax_encode_mobs2_binary(
    const CraftaxWorldState* state,
    const CraftaxWGMobs2* mobs,
    int mob_class_index,
    int channels_per_cell,
    int mob_bits_offset,
    float* obs
) {
    int level = craftax_wg_jax_index(state->player_level, CRAFTAX_WG_NUM_LEVELS);
    for (int i = 0; i < 2; i++) {
        int type_id = mobs->type_id[level][i];
        if (type_id < 0 || type_id >= CRAFTAX_WG_NUM_MOB_TYPES
            || !mobs->mask[level][i]) {
            continue;
        }

        int local_row = mobs->position[level][i][0]
            - state->player_position[0]
            + CRAFTAX_WG_OBS_ROWS / 2;
        int local_col = mobs->position[level][i][1]
            - state->player_position[1]
            + CRAFTAX_WG_OBS_COLS / 2;
        if (local_row < 0 || local_row >= CRAFTAX_WG_OBS_ROWS
            || local_col < 0 || local_col >= CRAFTAX_WG_OBS_COLS) {
            continue;
        }

        int world_row = mobs->position[level][i][0];
        int world_col = mobs->position[level][i][1];
        if (world_row < 0 || world_row >= CRAFTAX_WG_MAP_SIZE
            || world_col < 0 || world_col >= CRAFTAX_WG_MAP_SIZE
            || state->light_map[level][world_row][world_col] <= 12) {
            continue;
        }

        int obs_base = (local_row * CRAFTAX_WG_OBS_COLS + local_col)
            * channels_per_cell;
        int class_offset = mob_bits_offset
            + mob_class_index * CRAFTAX_WG_BINARY_MOB_BITS;
        memcpy(obs + obs_base + class_offset,
               CRAFTAX_WG_MOB_LUT[type_id + 1],
               CRAFTAX_WG_BINARY_MOB_BITS * sizeof(float));
    }
}

static __device__ inline void craftax_encode_map_base_observation(
    const CraftaxWorldState* state,
    float* obs
) {
    const int channels = CRAFTAX_WG_BINARY_CHANNELS_PER_CELL;
    const int top = state->player_position[0] - CRAFTAX_WG_OBS_ROWS / 2;
    const int left = state->player_position[1] - CRAFTAX_WG_OBS_COLS / 2;
    const int level = state->player_level;
    const float* empty_cell = CRAFTAX_WG_EMPTY_CELL_TEMPLATE;

    for (int row = 0; row < CRAFTAX_WG_OBS_ROWS; row++) {
        int world_row = top + row;
        bool row_in_bounds = world_row >= 0 && world_row < CRAFTAX_WG_MAP_SIZE;
        for (int col = 0; col < CRAFTAX_WG_OBS_COLS; col++) {
            int world_col = left + col;
            int obs_base = (row * CRAFTAX_WG_OBS_COLS + col) * channels;
            const float* cell = empty_cell;

            if (row_in_bounds && world_col >= 0 && world_col < CRAFTAX_WG_MAP_SIZE
                && state->light_map[level][world_row][world_col] > 12) {
                uint8_t block = state->map[level][world_row][world_col];
                uint8_t item = state->item_map[level][world_row][world_col];
                cell = CRAFTAX_WG_VISIBLE_CELL_TEMPLATE_LUT[block][item + 1];
            }

            memcpy(obs + obs_base, cell, CRAFTAX_WG_CELL_TEMPLATE_BYTES);
        }
    }
}

static __device__ inline void craftax_encode_packed_map_base_observation(
    const CraftaxWorldState* state,
    float* obs
) {
    const int channels = CRAFTAX_WG_PACKED_CHANNELS_PER_CELL;
    const int top = state->player_position[0] - CRAFTAX_WG_OBS_ROWS / 2;
    const int left = state->player_position[1] - CRAFTAX_WG_OBS_COLS / 2;
    const int level = state->player_level;

    memset(obs, 0, CRAFTAX_WG_PACKED_MAP_OBS_SIZE * sizeof(float));
    for (int row = 0; row < CRAFTAX_WG_OBS_ROWS; row++) {
        int world_row = top + row;
        bool row_in_bounds = world_row >= 0 && world_row < CRAFTAX_WG_MAP_SIZE;
        for (int col = 0; col < CRAFTAX_WG_OBS_COLS; col++) {
            int world_col = left + col;
            int obs_base = (row * CRAFTAX_WG_OBS_COLS + col) * channels;
            if (row_in_bounds && world_col >= 0 && world_col < CRAFTAX_WG_MAP_SIZE
                && state->light_map[level][world_row][world_col] > 12) {
                obs[obs_base + 0] = (float)state->map[level][world_row][world_col];
                obs[obs_base + 1] = (float)state->item_map[level][world_row][world_col] + 1.0f;
                obs[obs_base + 2] = 1.0f;
            }
        }
    }
}

static __device__ inline void craftax_clear_mob_channels_observation(float* obs) {
    const int channels = CRAFTAX_WG_BINARY_CHANNELS_PER_CELL;
    const int mob_bits_offset = CRAFTAX_WG_BINARY_BLOCK_BITS + CRAFTAX_WG_BINARY_ITEM_BITS;
    const size_t mob_channel_bytes =
        CRAFTAX_WG_NUM_MOB_CLASSES * CRAFTAX_WG_BINARY_MOB_BITS * sizeof(float);

    for (int cell = 0; cell < CRAFTAX_WG_OBS_WINDOW_CELLS; cell++) {
        memset(obs + cell * channels + mob_bits_offset, 0, mob_channel_bytes);
    }
}

static __device__ inline void craftax_encode_mobs3_packed(
    const CraftaxWorldState* state,
    const CraftaxWGMobs3* mobs,
    int mob_class_index,
    float* obs
) {
    const int level = craftax_wg_jax_index(state->player_level, CRAFTAX_WG_NUM_LEVELS);
    const int mob_slot_offset = 3 + mob_class_index;
    for (int i = 0; i < 3; i++) {
        int type_id = mobs->type_id[level][i];
        if (type_id < 0 || type_id >= CRAFTAX_WG_NUM_MOB_TYPES
            || !mobs->mask[level][i]) {
            continue;
        }

        int local_row = mobs->position[level][i][0]
            - state->player_position[0]
            + CRAFTAX_WG_OBS_ROWS / 2;
        int local_col = mobs->position[level][i][1]
            - state->player_position[1]
            + CRAFTAX_WG_OBS_COLS / 2;
        if (local_row < 0 || local_row >= CRAFTAX_WG_OBS_ROWS
            || local_col < 0 || local_col >= CRAFTAX_WG_OBS_COLS) {
            continue;
        }

        int world_row = mobs->position[level][i][0];
        int world_col = mobs->position[level][i][1];
        if (world_row < 0 || world_row >= CRAFTAX_WG_MAP_SIZE
            || world_col < 0 || world_col >= CRAFTAX_WG_MAP_SIZE
            || state->light_map[level][world_row][world_col] <= 12) {
            continue;
        }

        int obs_base = (local_row * CRAFTAX_WG_OBS_COLS + local_col)
            * CRAFTAX_WG_PACKED_CHANNELS_PER_CELL;
        obs[obs_base + mob_slot_offset] = (float)(type_id + 1);
    }
}

static __device__ inline void craftax_encode_mobs2_packed(
    const CraftaxWorldState* state,
    const CraftaxWGMobs2* mobs,
    int mob_class_index,
    float* obs
) {
    const int level = craftax_wg_jax_index(state->player_level, CRAFTAX_WG_NUM_LEVELS);
    const int mob_slot_offset = 3 + mob_class_index;
    for (int i = 0; i < 2; i++) {
        int type_id = mobs->type_id[level][i];
        if (type_id < 0 || type_id >= CRAFTAX_WG_NUM_MOB_TYPES
            || !mobs->mask[level][i]) {
            continue;
        }

        int local_row = mobs->position[level][i][0]
            - state->player_position[0]
            + CRAFTAX_WG_OBS_ROWS / 2;
        int local_col = mobs->position[level][i][1]
            - state->player_position[1]
            + CRAFTAX_WG_OBS_COLS / 2;
        if (local_row < 0 || local_row >= CRAFTAX_WG_OBS_ROWS
            || local_col < 0 || local_col >= CRAFTAX_WG_OBS_COLS) {
            continue;
        }

        int world_row = mobs->position[level][i][0];
        int world_col = mobs->position[level][i][1];
        if (world_row < 0 || world_row >= CRAFTAX_WG_MAP_SIZE
            || world_col < 0 || world_col >= CRAFTAX_WG_MAP_SIZE
            || state->light_map[level][world_row][world_col] <= 12) {
            continue;
        }

        int obs_base = (local_row * CRAFTAX_WG_OBS_COLS + local_col)
            * CRAFTAX_WG_PACKED_CHANNELS_PER_CELL;
        obs[obs_base + mob_slot_offset] = (float)(type_id + 1);
    }
}

static __device__ inline void craftax_encode_packed_mobs_observation(
    const CraftaxWorldState* state,
    float* obs
) {
    craftax_encode_mobs3_packed(state, &state->melee_mobs, 0, obs);
    craftax_encode_mobs3_packed(state, &state->passive_mobs, 1, obs);
    craftax_encode_mobs2_packed(state, &state->ranged_mobs, 2, obs);
    craftax_encode_mobs3_packed(state, &state->mob_projectiles, 3, obs);
    craftax_encode_mobs3_packed(state, &state->player_projectiles, 4, obs);
}

static __device__ inline void craftax_encode_mobs_observation(
    const CraftaxWorldState* state,
    float* obs
) {
    const int channels = CRAFTAX_WG_BINARY_CHANNELS_PER_CELL;
    const int mob_bits_offset = CRAFTAX_WG_BINARY_BLOCK_BITS + CRAFTAX_WG_BINARY_ITEM_BITS;

    craftax_encode_mobs3_binary(
        state,
        &state->melee_mobs,
        0,
        channels,
        mob_bits_offset,
        obs
    );
    craftax_encode_mobs3_binary(
        state,
        &state->passive_mobs,
        1,
        channels,
        mob_bits_offset,
        obs
    );
    craftax_encode_mobs2_binary(
        state,
        &state->ranged_mobs,
        2,
        channels,
        mob_bits_offset,
        obs
    );
    craftax_encode_mobs3_binary(
        state,
        &state->mob_projectiles,
        3,
        channels,
        mob_bits_offset,
        obs
    );
    craftax_encode_mobs3_binary(
        state,
        &state->player_projectiles,
        4,
        channels,
        mob_bits_offset,
        obs
    );
}

static __device__ inline void craftax_encode_scalar_observation_tail_at(
    const CraftaxWorldState* state,
    float* obs,
    int index
) {
    const int level = state->player_level;
    obs[index++] = sqrtf((float)state->inventory.wood) * (1.0f / 10.0f);
    obs[index++] = sqrtf((float)state->inventory.stone) * (1.0f / 10.0f);
    obs[index++] = sqrtf((float)state->inventory.coal) * (1.0f / 10.0f);
    obs[index++] = sqrtf((float)state->inventory.iron) * (1.0f / 10.0f);
    obs[index++] = sqrtf((float)state->inventory.diamond) * (1.0f / 10.0f);
    obs[index++] = sqrtf((float)state->inventory.sapphire) * (1.0f / 10.0f);
    obs[index++] = sqrtf((float)state->inventory.ruby) * (1.0f / 10.0f);
    obs[index++] = sqrtf((float)state->inventory.sapling) * (1.0f / 10.0f);
    obs[index++] = sqrtf((float)state->inventory.torches) * (1.0f / 10.0f);
    obs[index++] = sqrtf((float)state->inventory.arrows) * (1.0f / 10.0f);
    obs[index++] = (float)state->inventory.books * (1.0f / 2.0f);
    obs[index++] = (float)state->inventory.pickaxe * (1.0f / 4.0f);
    obs[index++] = (float)state->inventory.sword * (1.0f / 4.0f);
    obs[index++] = (float)state->sword_enchantment;
    obs[index++] = (float)state->bow_enchantment;
    obs[index++] = (float)state->inventory.bow;

    for (int i = 0; i < 6; i++) {
        obs[index++] = sqrtf((float)state->inventory.potions[i]) * (1.0f / 10.0f);
    }

    obs[index++] = state->player_health * (1.0f / 10.0f);
    obs[index++] = (float)state->player_food * (1.0f / 10.0f);
    obs[index++] = (float)state->player_drink * (1.0f / 10.0f);
    obs[index++] = (float)state->player_energy * (1.0f / 10.0f);
    obs[index++] = (float)state->player_mana * (1.0f / 10.0f);
    obs[index++] = (float)state->player_xp * (1.0f / 10.0f);
    obs[index++] = (float)state->player_dexterity * (1.0f / 10.0f);
    obs[index++] = (float)state->player_strength * (1.0f / 10.0f);
    obs[index++] = (float)state->player_intelligence * (1.0f / 10.0f);

    int direction_index = state->player_direction - 1;
    for (int i = 0; i < 4; i++) {
        obs[index++] = i == direction_index ? 1.0f : 0.0f;
    }

    for (int i = 0; i < 4; i++) {
        obs[index++] = (float)state->inventory.armour[i] * (1.0f / 2.0f);
    }
    for (int i = 0; i < 4; i++) {
        obs[index++] = (float)state->armour_enchantments[i];
    }

    obs[index++] = state->light_level;
    obs[index++] = state->is_sleeping ? 1.0f : 0.0f;
    obs[index++] = state->is_resting ? 1.0f : 0.0f;
    obs[index++] = state->learned_spells[0] ? 1.0f : 0.0f;
    obs[index++] = state->learned_spells[1] ? 1.0f : 0.0f;
    obs[index++] = (float)state->player_level * (1.0f / 10.0f);
    obs[index++] = state->monsters_killed[level] >= CRAFTAX_WG_MONSTERS_KILLED_TO_CLEAR_LEVEL ? 1.0f : 0.0f;
    obs[index++] = craftax_wg_is_boss_vulnerable(state) ? 1.0f : 0.0f;
}

static __device__ inline void craftax_encode_scalar_observation_tail(
    const CraftaxWorldState* state,
    float* obs
) {
    craftax_encode_scalar_observation_tail_at(state, obs, CRAFTAX_WG_BINARY_MAP_OBS_SIZE);
}

static __device__ inline void craftax_encode_reset_observation(
    const CraftaxWorldState* state,
    float* obs
) {
    craftax_encode_packed_map_base_observation(state, obs);
    craftax_encode_packed_mobs_observation(state, obs);
    craftax_encode_scalar_observation_tail_at(state, obs, CRAFTAX_WG_PACKED_MAP_OBS_SIZE);
}

static __device__ inline void craftax_encode_compact_map_base_observation(
    const CraftaxWorldState* state,
    uint8_t* obs
) {
    const int channels = CRAFTAX_WG_PACKED_CHANNELS_PER_CELL;
    const int top = state->player_position[0] - CRAFTAX_WG_OBS_ROWS / 2;
    const int left = state->player_position[1] - CRAFTAX_WG_OBS_COLS / 2;
    const int level = state->player_level;

    memset(obs, 0, CRAFTAX_WG_PACKED_MAP_OBS_SIZE);
    for (int row = 0; row < CRAFTAX_WG_OBS_ROWS; row++) {
        int world_row = top + row;
        bool row_in_bounds = world_row >= 0 && world_row < CRAFTAX_WG_MAP_SIZE;
        for (int col = 0; col < CRAFTAX_WG_OBS_COLS; col++) {
            int world_col = left + col;
            int obs_base = (row * CRAFTAX_WG_OBS_COLS + col) * channels;
            if (row_in_bounds && world_col >= 0 && world_col < CRAFTAX_WG_MAP_SIZE
                && state->light_map[level][world_row][world_col] > 12) {
                obs[obs_base + 0] = (uint8_t)state->map[level][world_row][world_col];
                obs[obs_base + 1] = (uint8_t)(state->item_map[level][world_row][world_col] + 1);
                obs[obs_base + 2] = 1;
            }
        }
    }
}

static __device__ inline void craftax_encode_mobs3_compact(
    const CraftaxWorldState* state,
    const CraftaxWGMobs3* mobs,
    int mob_class_index,
    uint8_t* obs
) {
    const int level = craftax_wg_jax_index(state->player_level, CRAFTAX_WG_NUM_LEVELS);
    const int mob_slot_offset = 3 + mob_class_index;
    for (int i = 0; i < 3; i++) {
        int type_id = mobs->type_id[level][i];
        if (type_id < 0 || type_id >= CRAFTAX_WG_NUM_MOB_TYPES
            || !mobs->mask[level][i]) {
            continue;
        }

        int local_row = mobs->position[level][i][0]
            - state->player_position[0]
            + CRAFTAX_WG_OBS_ROWS / 2;
        int local_col = mobs->position[level][i][1]
            - state->player_position[1]
            + CRAFTAX_WG_OBS_COLS / 2;
        if (local_row < 0 || local_row >= CRAFTAX_WG_OBS_ROWS
            || local_col < 0 || local_col >= CRAFTAX_WG_OBS_COLS) {
            continue;
        }

        int world_row = mobs->position[level][i][0];
        int world_col = mobs->position[level][i][1];
        if (world_row < 0 || world_row >= CRAFTAX_WG_MAP_SIZE
            || world_col < 0 || world_col >= CRAFTAX_WG_MAP_SIZE
            || state->light_map[level][world_row][world_col] <= 12) {
            continue;
        }

        int obs_base = (local_row * CRAFTAX_WG_OBS_COLS + local_col)
            * CRAFTAX_WG_PACKED_CHANNELS_PER_CELL;
        obs[obs_base + mob_slot_offset] = (uint8_t)(type_id + 1);
    }
}

static __device__ inline void craftax_encode_mobs2_compact(
    const CraftaxWorldState* state,
    const CraftaxWGMobs2* mobs,
    int mob_class_index,
    uint8_t* obs
) {
    const int level = craftax_wg_jax_index(state->player_level, CRAFTAX_WG_NUM_LEVELS);
    const int mob_slot_offset = 3 + mob_class_index;
    for (int i = 0; i < 2; i++) {
        int type_id = mobs->type_id[level][i];
        if (type_id < 0 || type_id >= CRAFTAX_WG_NUM_MOB_TYPES
            || !mobs->mask[level][i]) {
            continue;
        }

        int local_row = mobs->position[level][i][0]
            - state->player_position[0]
            + CRAFTAX_WG_OBS_ROWS / 2;
        int local_col = mobs->position[level][i][1]
            - state->player_position[1]
            + CRAFTAX_WG_OBS_COLS / 2;
        if (local_row < 0 || local_row >= CRAFTAX_WG_OBS_ROWS
            || local_col < 0 || local_col >= CRAFTAX_WG_OBS_COLS) {
            continue;
        }

        int world_row = mobs->position[level][i][0];
        int world_col = mobs->position[level][i][1];
        if (world_row < 0 || world_row >= CRAFTAX_WG_MAP_SIZE
            || world_col < 0 || world_col >= CRAFTAX_WG_MAP_SIZE
            || state->light_map[level][world_row][world_col] <= 12) {
            continue;
        }

        int obs_base = (local_row * CRAFTAX_WG_OBS_COLS + local_col)
            * CRAFTAX_WG_PACKED_CHANNELS_PER_CELL;
        obs[obs_base + mob_slot_offset] = (uint8_t)(type_id + 1);
    }
}

// See CRAFTAX_WG_COMPACT_OBS_SIZE for the layout contract. The scalar tail
// goes through the float encoder and is memcpy'd, so the reinterpreted floats
// on the learner side are bit-identical to the float observation path.
static __device__ inline void craftax_encode_compact_observation(
    const CraftaxWorldState* state,
    uint8_t* obs
) {
    craftax_encode_compact_map_base_observation(state, obs);
    craftax_encode_mobs3_compact(state, &state->melee_mobs, 0, obs);
    craftax_encode_mobs3_compact(state, &state->passive_mobs, 1, obs);
    craftax_encode_mobs2_compact(state, &state->ranged_mobs, 2, obs);
    craftax_encode_mobs3_compact(state, &state->mob_projectiles, 3, obs);
    craftax_encode_mobs3_compact(state, &state->player_projectiles, 4, obs);

    float tail[CRAFTAX_WG_INVENTORY_OBS_SIZE];
    craftax_encode_scalar_observation_tail_at(state, tail, 0);
    memcpy(obs + CRAFTAX_WG_PACKED_MAP_OBS_SIZE, tail, sizeof(tail));
}

// ============================================================
// ===== craftax.h =====
// ============================================================
// Full native Craftax environment for PufferLib Ocean.


#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// ============================================================
// Optional step profiling (compile with -DCRAFTAX_PROFILE)
// ============================================================
#ifdef CRAFTAX_PROFILE

#define CRAFTAX_NUM_PROFILE_ZONES 18

typedef struct {
    const char* name;
    uint64_t total_ns;
    uint64_t count;
} CraftaxProfileZone;

static __device__ CraftaxProfileZone craftax_profile_zones[CRAFTAX_NUM_PROFILE_ZONES] = {
    {"change_floor", 0, 0},
    {"crafting", 0, 0},
    {"do_action", 0, 0},
    {"place+shoot+spell+potion", 0, 0},
    {"read_book", 0, 0},
    {"enchant", 0, 0},
    {"boss+attr+move", 0, 0},
    {"update_mobs", 0, 0},
    {"spawn_mobs", 0, 0},
    {"plants+intrinsics+achieve", 0, 0},
    {"reward+bookkeeping", 0, 0},
    {"encode_obs", 0, 0},
    {"rng_split", 0, 0},
    {"is_game_over", 0, 0},
    {"reset_on_done", 0, 0},
    {"copy_achievements", 0, 0},
    {"reward_bookkeeping", 0, 0},
    {"unprofiled", 0, 0},
};

static __device__ inline uint64_t craftax_profile_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static __device__ inline void craftax_profile_record(int zone, uint64_t start) {
    craftax_profile_zones[zone].total_ns += craftax_profile_now() - start;
    craftax_profile_zones[zone].count++;
}

static __device__ inline void craftax_profile_report(void) {
    fprintf(stderr, "\n=== Craftax Step Profile ===\n");
    uint64_t total = 0;
    for (int i = 0; i < CRAFTAX_NUM_PROFILE_ZONES; i++) {
        total += craftax_profile_zones[i].total_ns;
    }
    for (int i = 0; i < CRAFTAX_NUM_PROFILE_ZONES; i++) {
        CraftaxProfileZone* z = &craftax_profile_zones[i];
        if (z->count == 0) continue;
        double pct = total > 0 ? (100.0 * (double)z->total_ns / (double)total) : 0.0;
        double avg_us = (double)z->total_ns / (double)z->count / 1000.0;
        fprintf(stderr, "%-28s %8.3f%%  %10.2f us/step  (%lu calls)\n",
                z->name, pct, avg_us, (unsigned long)z->count);
    }
    fprintf(stderr, "%-28s %8.3f%%  %10.2f us/step\n",
            "TOTAL", 100.0, (double)total / (double)craftax_profile_zones[0].count / 1000.0);
}

#define CRAFTAX_PROFILE_START() uint64_t _prof_start = craftax_profile_now(); uint64_t _prof_zone_start;
#define CRAFTAX_PROFILE_ZONE(n) do { _prof_zone_start = craftax_profile_now(); } while(0)
#define CRAFTAX_PROFILE_END(n) craftax_profile_record((n), _prof_zone_start)
#define CRAFTAX_PROFILE_FINAL(n) craftax_profile_record((n), _prof_start)

#else

#define CRAFTAX_PROFILE_START() ((void)0)
#define CRAFTAX_PROFILE_ZONE(n) ((void)0)
#define CRAFTAX_PROFILE_END(n) ((void)0)
#define CRAFTAX_PROFILE_FINAL(n) ((void)0)
#define craftax_profile_report() ((void)0)

#endif // CRAFTAX_PROFILE

// ============================================================
// Constants
// ============================================================
#define CRAFTAX_OBS_ROWS 9
#define CRAFTAX_OBS_COLS 11
#define CRAFTAX_MAP_SIZE 48
#define CRAFTAX_NUM_LEVELS 9

#define CRAFTAX_NUM_BLOCK_TYPES 37
#define CRAFTAX_NUM_ITEM_TYPES 5
#define CRAFTAX_NUM_MOB_CLASSES 5
#define CRAFTAX_NUM_MOB_TYPES 8
#define CRAFTAX_INVENTORY_OBS_SIZE 51
// Compile with -DCRAFTAX_COMPACT_OBS to emit uint8 observations (996 bytes:
// 792 map ID bytes + 51 float32 scalars as raw bytes) instead of 843 floats.
// Pair with OBS_TENSOR_T ByteTensor and the CraftaxCompactEncoder policy
// encoder, which expands back to the identical 843-float observation on GPU.
#ifdef CRAFTAX_COMPACT_OBS
#define CRAFTAX_OBS_SIZE CRAFTAX_WG_COMPACT_OBS_SIZE
typedef uint8_t CraftaxObs;
#else
#define CRAFTAX_OBS_SIZE CRAFTAX_WG_OBS_SIZE
typedef float CraftaxObs;
#endif

#define CRAFTAX_NUM_ACTIONS 43
#define CRAFTAX_NUM_ACHIEVEMENTS 67

#define CRAFTAX_MAX_MELEE_MOBS 3
#define CRAFTAX_MAX_PASSIVE_MOBS 3
#define CRAFTAX_MAX_RANGED_MOBS 2
#define CRAFTAX_MAX_MOB_PROJECTILES 3
#define CRAFTAX_MAX_PLAYER_PROJECTILES 3
#define CRAFTAX_MAX_GROWING_PLANTS 10

#define CRAFTAX_DEFAULT_MAX_TIMESTEPS 100000
#define CRAFTAX_DAY_LENGTH 300
#define CRAFTAX_MAX_ATTRIBUTE 5
#define CRAFTAX_MOB_DESPAWN_DISTANCE 14
#define CRAFTAX_MONSTERS_KILLED_TO_CLEAR_LEVEL 8

// ============================================================
// Enums copied from craftax/craftax/constants.py
// ============================================================
typedef enum CraftaxBlockType {
    CRAFTAX_BLOCK_INVALID = 0,
    CRAFTAX_BLOCK_OUT_OF_BOUNDS = 1,
    CRAFTAX_BLOCK_GRASS = 2,
    CRAFTAX_BLOCK_WATER = 3,
    CRAFTAX_BLOCK_STONE = 4,
    CRAFTAX_BLOCK_TREE = 5,
    CRAFTAX_BLOCK_WOOD = 6,
    CRAFTAX_BLOCK_PATH = 7,
    CRAFTAX_BLOCK_COAL = 8,
    CRAFTAX_BLOCK_IRON = 9,
    CRAFTAX_BLOCK_DIAMOND = 10,
    CRAFTAX_BLOCK_CRAFTING_TABLE = 11,
    CRAFTAX_BLOCK_FURNACE = 12,
    CRAFTAX_BLOCK_SAND = 13,
    CRAFTAX_BLOCK_LAVA = 14,
    CRAFTAX_BLOCK_PLANT = 15,
    CRAFTAX_BLOCK_RIPE_PLANT = 16,
    CRAFTAX_BLOCK_WALL = 17,
    CRAFTAX_BLOCK_DARKNESS = 18,
    CRAFTAX_BLOCK_WALL_MOSS = 19,
    CRAFTAX_BLOCK_STALAGMITE = 20,
    CRAFTAX_BLOCK_SAPPHIRE = 21,
    CRAFTAX_BLOCK_RUBY = 22,
    CRAFTAX_BLOCK_CHEST = 23,
    CRAFTAX_BLOCK_FOUNTAIN = 24,
    CRAFTAX_BLOCK_FIRE_GRASS = 25,
    CRAFTAX_BLOCK_ICE_GRASS = 26,
    CRAFTAX_BLOCK_GRAVEL = 27,
    CRAFTAX_BLOCK_FIRE_TREE = 28,
    CRAFTAX_BLOCK_ICE_SHRUB = 29,
    CRAFTAX_BLOCK_ENCHANTMENT_TABLE_FIRE = 30,
    CRAFTAX_BLOCK_ENCHANTMENT_TABLE_ICE = 31,
    CRAFTAX_BLOCK_NECROMANCER = 32,
    CRAFTAX_BLOCK_GRAVE = 33,
    CRAFTAX_BLOCK_GRAVE2 = 34,
    CRAFTAX_BLOCK_GRAVE3 = 35,
    CRAFTAX_BLOCK_NECROMANCER_VULNERABLE = 36,
} CraftaxBlockType;

typedef enum CraftaxItemType {
    CRAFTAX_ITEM_NONE = 0,
    CRAFTAX_ITEM_TORCH = 1,
    CRAFTAX_ITEM_LADDER_DOWN = 2,
    CRAFTAX_ITEM_LADDER_UP = 3,
    CRAFTAX_ITEM_LADDER_DOWN_BLOCKED = 4,
} CraftaxItemType;

typedef enum CraftaxAction {
    CRAFTAX_ACTION_NOOP = 0,
    CRAFTAX_ACTION_LEFT = 1,
    CRAFTAX_ACTION_RIGHT = 2,
    CRAFTAX_ACTION_UP = 3,
    CRAFTAX_ACTION_DOWN = 4,
    CRAFTAX_ACTION_DO = 5,
    CRAFTAX_ACTION_SLEEP = 6,
    CRAFTAX_ACTION_PLACE_STONE = 7,
    CRAFTAX_ACTION_PLACE_TABLE = 8,
    CRAFTAX_ACTION_PLACE_FURNACE = 9,
    CRAFTAX_ACTION_PLACE_PLANT = 10,
    CRAFTAX_ACTION_MAKE_WOOD_PICKAXE = 11,
    CRAFTAX_ACTION_MAKE_STONE_PICKAXE = 12,
    CRAFTAX_ACTION_MAKE_IRON_PICKAXE = 13,
    CRAFTAX_ACTION_MAKE_WOOD_SWORD = 14,
    CRAFTAX_ACTION_MAKE_STONE_SWORD = 15,
    CRAFTAX_ACTION_MAKE_IRON_SWORD = 16,
    CRAFTAX_ACTION_REST = 17,
    CRAFTAX_ACTION_DESCEND = 18,
    CRAFTAX_ACTION_ASCEND = 19,
    CRAFTAX_ACTION_MAKE_DIAMOND_PICKAXE = 20,
    CRAFTAX_ACTION_MAKE_DIAMOND_SWORD = 21,
    CRAFTAX_ACTION_MAKE_IRON_ARMOUR = 22,
    CRAFTAX_ACTION_MAKE_DIAMOND_ARMOUR = 23,
    CRAFTAX_ACTION_SHOOT_ARROW = 24,
    CRAFTAX_ACTION_MAKE_ARROW = 25,
    CRAFTAX_ACTION_CAST_FIREBALL = 26,
    CRAFTAX_ACTION_CAST_ICEBALL = 27,
    CRAFTAX_ACTION_PLACE_TORCH = 28,
    CRAFTAX_ACTION_DRINK_POTION_RED = 29,
    CRAFTAX_ACTION_DRINK_POTION_GREEN = 30,
    CRAFTAX_ACTION_DRINK_POTION_BLUE = 31,
    CRAFTAX_ACTION_DRINK_POTION_PINK = 32,
    CRAFTAX_ACTION_DRINK_POTION_CYAN = 33,
    CRAFTAX_ACTION_DRINK_POTION_YELLOW = 34,
    CRAFTAX_ACTION_READ_BOOK = 35,
    CRAFTAX_ACTION_ENCHANT_SWORD = 36,
    CRAFTAX_ACTION_ENCHANT_ARMOUR = 37,
    CRAFTAX_ACTION_MAKE_TORCH = 38,
    CRAFTAX_ACTION_LEVEL_UP_DEXTERITY = 39,
    CRAFTAX_ACTION_LEVEL_UP_STRENGTH = 40,
    CRAFTAX_ACTION_LEVEL_UP_INTELLIGENCE = 41,
    CRAFTAX_ACTION_ENCHANT_BOW = 42,
} CraftaxAction;

typedef enum CraftaxMobType {
    CRAFTAX_MOB_PASSIVE = 0,
    CRAFTAX_MOB_MELEE = 1,
    CRAFTAX_MOB_RANGED = 2,
    CRAFTAX_MOB_PROJECTILE = 3,
} CraftaxMobType;

typedef enum CraftaxProjectileType {
    CRAFTAX_PROJECTILE_ARROW = 0,
    CRAFTAX_PROJECTILE_DAGGER = 1,
    CRAFTAX_PROJECTILE_FIREBALL = 2,
    CRAFTAX_PROJECTILE_ICEBALL = 3,
    CRAFTAX_PROJECTILE_ARROW2 = 4,
    CRAFTAX_PROJECTILE_SLIMEBALL = 5,
    CRAFTAX_PROJECTILE_FIREBALL2 = 6,
    CRAFTAX_PROJECTILE_ICEBALL2 = 7,
} CraftaxProjectileType;

typedef enum CraftaxAchievement {
    CRAFTAX_ACH_COLLECT_WOOD = 0,
    CRAFTAX_ACH_PLACE_TABLE = 1,
    CRAFTAX_ACH_EAT_COW = 2,
    CRAFTAX_ACH_COLLECT_SAPLING = 3,
    CRAFTAX_ACH_COLLECT_DRINK = 4,
    CRAFTAX_ACH_MAKE_WOOD_PICKAXE = 5,
    CRAFTAX_ACH_MAKE_WOOD_SWORD = 6,
    CRAFTAX_ACH_PLACE_PLANT = 7,
    CRAFTAX_ACH_DEFEAT_ZOMBIE = 8,
    CRAFTAX_ACH_COLLECT_STONE = 9,
    CRAFTAX_ACH_PLACE_STONE = 10,
    CRAFTAX_ACH_EAT_PLANT = 11,
    CRAFTAX_ACH_DEFEAT_SKELETON = 12,
    CRAFTAX_ACH_MAKE_STONE_PICKAXE = 13,
    CRAFTAX_ACH_MAKE_STONE_SWORD = 14,
    CRAFTAX_ACH_WAKE_UP = 15,
    CRAFTAX_ACH_PLACE_FURNACE = 16,
    CRAFTAX_ACH_COLLECT_COAL = 17,
    CRAFTAX_ACH_COLLECT_IRON = 18,
    CRAFTAX_ACH_COLLECT_DIAMOND = 19,
    CRAFTAX_ACH_MAKE_IRON_PICKAXE = 20,
    CRAFTAX_ACH_MAKE_IRON_SWORD = 21,
    CRAFTAX_ACH_MAKE_ARROW = 22,
    CRAFTAX_ACH_MAKE_TORCH = 23,
    CRAFTAX_ACH_PLACE_TORCH = 24,
    CRAFTAX_ACH_MAKE_DIAMOND_SWORD = 25,
    CRAFTAX_ACH_MAKE_IRON_ARMOUR = 26,
    CRAFTAX_ACH_MAKE_DIAMOND_ARMOUR = 27,
    CRAFTAX_ACH_ENTER_GNOMISH_MINES = 28,
    CRAFTAX_ACH_ENTER_DUNGEON = 29,
    CRAFTAX_ACH_ENTER_SEWERS = 30,
    CRAFTAX_ACH_ENTER_VAULT = 31,
    CRAFTAX_ACH_ENTER_TROLL_MINES = 32,
    CRAFTAX_ACH_ENTER_FIRE_REALM = 33,
    CRAFTAX_ACH_ENTER_ICE_REALM = 34,
    CRAFTAX_ACH_ENTER_GRAVEYARD = 35,
    CRAFTAX_ACH_DEFEAT_GNOME_WARRIOR = 36,
    CRAFTAX_ACH_DEFEAT_GNOME_ARCHER = 37,
    CRAFTAX_ACH_DEFEAT_ORC_SOLIDER = 38,
    CRAFTAX_ACH_DEFEAT_ORC_MAGE = 39,
    CRAFTAX_ACH_DEFEAT_LIZARD = 40,
    CRAFTAX_ACH_DEFEAT_KOBOLD = 41,
    CRAFTAX_ACH_DEFEAT_TROLL = 42,
    CRAFTAX_ACH_DEFEAT_DEEP_THING = 43,
    CRAFTAX_ACH_DEFEAT_PIGMAN = 44,
    CRAFTAX_ACH_DEFEAT_FIRE_ELEMENTAL = 45,
    CRAFTAX_ACH_DEFEAT_FROST_TROLL = 46,
    CRAFTAX_ACH_DEFEAT_ICE_ELEMENTAL = 47,
    CRAFTAX_ACH_DAMAGE_NECROMANCER = 48,
    CRAFTAX_ACH_DEFEAT_NECROMANCER = 49,
    CRAFTAX_ACH_EAT_BAT = 50,
    CRAFTAX_ACH_EAT_SNAIL = 51,
    CRAFTAX_ACH_FIND_BOW = 52,
    CRAFTAX_ACH_FIRE_BOW = 53,
    CRAFTAX_ACH_COLLECT_SAPPHIRE = 54,
    CRAFTAX_ACH_LEARN_FIREBALL = 55,
    CRAFTAX_ACH_CAST_FIREBALL = 56,
    CRAFTAX_ACH_LEARN_ICEBALL = 57,
    CRAFTAX_ACH_CAST_ICEBALL = 58,
    CRAFTAX_ACH_COLLECT_RUBY = 59,
    CRAFTAX_ACH_MAKE_DIAMOND_PICKAXE = 60,
    CRAFTAX_ACH_OPEN_CHEST = 61,
    CRAFTAX_ACH_DRINK_POTION = 62,
    CRAFTAX_ACH_ENCHANT_SWORD = 63,
    CRAFTAX_ACH_ENCHANT_ARMOUR = 64,
    CRAFTAX_ACH_DEFEAT_KNIGHT = 65,
    CRAFTAX_ACH_DEFEAT_ARCHER = 66,
} CraftaxAchievement;

// ============================================================
// State layout declarations matching craftax_state.py field order
// ============================================================
typedef struct CraftaxInventory {
    int32_t wood;
    int32_t stone;
    int32_t coal;
    int32_t iron;
    int32_t diamond;
    int32_t sapling;
    int32_t pickaxe;
    int32_t sword;
    int32_t bow;
    int32_t arrows;
    int32_t armour[4];
    int32_t torches;
    int32_t ruby;
    int32_t sapphire;
    int32_t potions[6];
    int32_t books;
} CraftaxInventory;

typedef struct CraftaxMobs3 {
    int32_t position[CRAFTAX_NUM_LEVELS][3][2];
    float health[CRAFTAX_NUM_LEVELS][3];
    bool mask[CRAFTAX_NUM_LEVELS][3];
    int32_t attack_cooldown[CRAFTAX_NUM_LEVELS][3];
    int32_t type_id[CRAFTAX_NUM_LEVELS][3];
} CraftaxMobs3;

typedef struct CraftaxMobs2 {
    int32_t position[CRAFTAX_NUM_LEVELS][2][2];
    float health[CRAFTAX_NUM_LEVELS][2];
    bool mask[CRAFTAX_NUM_LEVELS][2];
    int32_t attack_cooldown[CRAFTAX_NUM_LEVELS][2];
    int32_t type_id[CRAFTAX_NUM_LEVELS][2];
} CraftaxMobs2;

typedef struct CraftaxState {
    // === Hot data (accessed every step) ===
    int32_t player_position[2];
    int32_t player_level;
    int32_t player_direction;

    float player_health;
    int32_t player_food;
    int32_t player_drink;
    int32_t player_energy;
    int32_t player_mana;
    bool is_sleeping;
    bool is_resting;

    float player_recover;
    float player_hunger;
    float player_thirst;
    float player_fatigue;
    float player_recover_mana;

    int32_t player_xp;
    int32_t player_dexterity;
    int32_t player_strength;
    int32_t player_intelligence;

    CraftaxInventory inventory;

    CraftaxMobs3 melee_mobs;
    CraftaxMobs3 passive_mobs;
    CraftaxMobs2 ranged_mobs;

    CraftaxMobs3 mob_projectiles;
    int32_t mob_projectile_directions[CRAFTAX_NUM_LEVELS][CRAFTAX_MAX_MOB_PROJECTILES][2];
    CraftaxMobs3 player_projectiles;
    int32_t player_projectile_directions[CRAFTAX_NUM_LEVELS][CRAFTAX_MAX_PLAYER_PROJECTILES][2];

    int32_t growing_plants_positions[CRAFTAX_MAX_GROWING_PLANTS][2];
    int32_t growing_plants_age[CRAFTAX_MAX_GROWING_PLANTS];
    bool growing_plants_mask[CRAFTAX_MAX_GROWING_PLANTS];

    int32_t potion_mapping[6];
    bool learned_spells[2];

    int32_t sword_enchantment;
    int32_t bow_enchantment;
    int32_t armour_enchantments[4];

    int32_t boss_progress;
    int32_t boss_timesteps_to_spawn_this_round;

    float light_level;
    bool achievements[CRAFTAX_NUM_ACHIEVEMENTS];
    uint32_t state_rng[2];
    int32_t timestep;
    int32_t fractal_noise_angles[4];

    // === Medium-hot bitmaps, read during mob updates, spawn scans, encode_obs ===
    uint64_t mob_bits[CRAFTAX_NUM_LEVELS][CRAFTAX_MAP_SIZE];
    uint64_t spawn_all_bits[CRAFTAX_NUM_LEVELS][CRAFTAX_MAP_SIZE];
    uint64_t spawn_grave_bits[CRAFTAX_NUM_LEVELS][CRAFTAX_MAP_SIZE];
    uint64_t spawn_water_bits[CRAFTAX_NUM_LEVELS][CRAFTAX_MAP_SIZE];

    // === Cold data (large maps, scattered access) ===
    uint8_t map[CRAFTAX_NUM_LEVELS][CRAFTAX_MAP_SIZE][CRAFTAX_MAP_SIZE];
    uint8_t item_map[CRAFTAX_NUM_LEVELS][CRAFTAX_MAP_SIZE][CRAFTAX_MAP_SIZE];
    uint8_t light_map[CRAFTAX_NUM_LEVELS][CRAFTAX_MAP_SIZE][CRAFTAX_MAP_SIZE];

    int32_t down_ladders[CRAFTAX_NUM_LEVELS][2];
    int32_t up_ladders[CRAFTAX_NUM_LEVELS][2];
    bool chests_opened[CRAFTAX_NUM_LEVELS];
    int32_t monsters_killed[CRAFTAX_NUM_LEVELS];

    // Mirrors CraftaxWorldState: lazy floor generation bookkeeping.
    uint32_t lazy_floor_keys[CRAFTAX_NUM_LEVELS][2];
    uint32_t lazy_floors_pending;
} CraftaxState;

typedef char CraftaxStateMatchesWorldState[
    (sizeof(CraftaxState) == sizeof(CraftaxWorldState)) ? 1 : -1
];

static __device__ inline uint64_t craftax_spawn_all_bit(uint8_t block) {
    return (uint64_t)(
        block == CRAFTAX_BLOCK_GRASS
        || block == CRAFTAX_BLOCK_PATH
        || block == CRAFTAX_BLOCK_FIRE_GRASS
        || block == CRAFTAX_BLOCK_ICE_GRASS
    );
}

static __device__ inline uint64_t craftax_spawn_grave_bit(uint8_t block) {
    return (uint64_t)(
        block == CRAFTAX_BLOCK_GRAVE
        || block == CRAFTAX_BLOCK_GRAVE2
        || block == CRAFTAX_BLOCK_GRAVE3
    );
}

static __device__ inline uint64_t craftax_spawn_water_bit(uint8_t block) {
    return (uint64_t)(block == CRAFTAX_BLOCK_WATER);
}

static __device__ inline void craftax_refresh_spawn_bits_cell(
    CraftaxState* state,
    int32_t level,
    int32_t row,
    int32_t col
) {
    uint64_t bit = 1ULL << col;
    uint8_t block = state->map[level][row][col];

    state->spawn_all_bits[level][row] =
        (state->spawn_all_bits[level][row] & ~bit)
        | ((0ULL - craftax_spawn_all_bit(block)) & bit);
    state->spawn_grave_bits[level][row] =
        (state->spawn_grave_bits[level][row] & ~bit)
        | ((0ULL - craftax_spawn_grave_bit(block)) & bit);
    state->spawn_water_bits[level][row] =
        (state->spawn_water_bits[level][row] & ~bit)
        | ((0ULL - craftax_spawn_water_bit(block)) & bit);
}

static __device__ inline void craftax_set_map_block(
    CraftaxState* state,
    int32_t level,
    int32_t row,
    int32_t col,
    int32_t block
) {
    state->map[level][row][col] = (uint8_t)block;
    craftax_refresh_spawn_bits_cell(state, level, row, col);
}

static __device__ inline void craftax_refresh_spawn_bits_level(
    CraftaxState* state,
    int32_t level
) {
    for (int32_t row = 0; row < CRAFTAX_MAP_SIZE; row++) {
        uint64_t all_bits = 0;
        uint64_t grave_bits = 0;
        uint64_t water_bits = 0;
        for (int32_t col = 0; col < CRAFTAX_MAP_SIZE; col++) {
            uint8_t block = state->map[level][row][col];
            uint64_t bit = 1ULL << col;
            all_bits |= (0ULL - craftax_spawn_all_bit(block)) & bit;
            grave_bits |= (0ULL - craftax_spawn_grave_bit(block)) & bit;
            water_bits |= (0ULL - craftax_spawn_water_bit(block)) & bit;
        }
        state->spawn_all_bits[level][row] = all_bits;
        state->spawn_grave_bits[level][row] = grave_bits;
        state->spawn_water_bits[level][row] = water_bits;
    }
}

// Generate a deferred floor on first visit. No-op when the floor is already
// present (bit clear), so zero-initialized fixture states are unaffected.
static __device__ inline void craftax_ensure_floor_generated(
    CraftaxState* state,
    int32_t level
) {
    if (level < 0 || level >= CRAFTAX_NUM_LEVELS) return;
    uint32_t bit = 1u << (uint32_t)level;
    if (!(state->lazy_floors_pending & bit)) return;
    CraftaxThreefryKey key = {{
        state->lazy_floor_keys[level][0],
        state->lazy_floor_keys[level][1],
    }};
    craftax_generate_floor_from_key(key, level, (CraftaxWorldState*)(void*)state);
    state->lazy_floors_pending &= ~bit;
    craftax_refresh_spawn_bits_level(state, level);
}

static __device__ inline void craftax_refresh_spawn_bits_all(CraftaxState* state) {
    for (int32_t level = 0; level < CRAFTAX_NUM_LEVELS; level++) {
        craftax_refresh_spawn_bits_level(state, level);
    }
}

#define CRAFTAX_ARENA_PACKET_SIZE 64

typedef struct CraftaxArena {
    CraftaxState* states;
    int num_envs;
    int packet_size;
    int num_packets;
} CraftaxArena;

#ifdef CRAFTAX_ENABLE_ENV_IMPL
static __device__ inline void craftax_change_floor_native(CraftaxState* state, int32_t action);
static __device__ inline void craftax_do_crafting_native(CraftaxState* state, int32_t action);
static __device__ inline void craftax_do_action_native(
    CraftaxState* state,
    int32_t action,
    CraftaxThreefryKey rng
);
static __device__ inline void craftax_place_block_native(CraftaxState* state, int32_t action);
static __device__ inline void craftax_shoot_projectile_native(
    CraftaxState* state,
    int32_t action
);
static __device__ inline void craftax_cast_spell_native(CraftaxState* state, int32_t action);
static __device__ inline void craftax_drink_potion_native(CraftaxState* state, int32_t action);
static __device__ inline void craftax_read_book_native(
    CraftaxState* state,
    const uint32_t rng_words[2],
    int32_t action
);
static __device__ inline void craftax_enchant_native(
    CraftaxState* state,
    int32_t action,
    CraftaxThreefryKey rng
);
static __device__ inline void craftax_boss_logic_native(CraftaxState* state);
static __device__ inline void craftax_level_up_attributes_native(
    CraftaxState* state,
    int32_t action,
    int32_t max_attribute
);
static __device__ inline void craftax_move_player_native(
    CraftaxState* state,
    int32_t action,
    bool god_mode
);
static __device__ inline void craftax_update_mobs_native(
    CraftaxState* state,
    CraftaxThreefryKey rng
);
static __device__ inline void craftax_spawn_mobs_native(
    CraftaxState* state,
    CraftaxThreefryKey rng
);
static __device__ inline void craftax_update_plants_native(CraftaxState* state);
static __device__ inline void craftax_update_player_intrinsics_native(
    CraftaxState* state,
    int32_t action
);
static __device__ inline void craftax_clip_inventory_and_intrinsics_native(
    CraftaxState* state,
    bool god_mode
);
static __device__ inline void craftax_calculate_inventory_achievements_native(
    CraftaxState* state
);
#endif

typedef struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float achievements[CRAFTAX_NUM_ACHIEVEMENTS];
    float n;
} Log;

typedef struct Client {
    int unused;
} Client;

typedef struct Craftax {
    Client* client;
    Log log;

    CraftaxObs* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;

    unsigned int rng;
    uint64_t seed;
    CraftaxThreefryKey rng_key;
    CraftaxArena* arena;
    CraftaxState* state;
    int32_t packet_id;
    int32_t lane_id;
    bool owns_state_storage;

    float achievements[CRAFTAX_NUM_ACHIEVEMENTS];
    float episode_return_accum;
    int32_t episode_length_accum;
} Craftax;

#ifdef CRAFTAX_ENABLE_ENV_IMPL

// ============================================================
// Native reset, observation, reward, and step glue
// ============================================================
static __device__ const float CRAFTAX_ACHIEVEMENT_REWARD_MAP[CRAFTAX_NUM_ACHIEVEMENTS] = {
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 5.0f, 5.0f,
    5.0f, 8.0f, 8.0f, 8.0f, 3.0f, 3.0f, 3.0f, 3.0f,
    5.0f, 5.0f, 5.0f, 5.0f, 8.0f, 8.0f, 8.0f, 8.0f,
    8.0f, 8.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 5.0f,
    5.0f, 5.0f, 5.0f, 3.0f, 3.0f, 3.0f, 3.0f, 5.0f,
    5.0f, 5.0f, 5.0f,
};

static __device__ inline CraftaxThreefryKey craftax_step_native_next_key(
    CraftaxThreefryKey* rng
) {
    CraftaxThreefryKey subkey;
    craftax_threefry_split(*rng, rng, &subkey);
    return subkey;
}

static __device__ inline void craftax_copy_world_state_to_state(
    CraftaxState* dst,
    const CraftaxWorldState* src
) {
    memcpy(dst, src, sizeof(*dst));
}

// Lazy floor generation toggle (process-wide, like the reset pool). When on,
// resets generate only floor 0 and defer floors 1..8 to first visit. Maps are
// bit-identical to eager generation; only the time of generation changes.
static __device__ int g_craftax_lazy_floors = 0;

static __device__ inline void craftax_set_lazy_floors(int enabled) {
    g_craftax_lazy_floors = enabled;
}

static __device__ inline void craftax_generate_state_from_world_key(
    CraftaxThreefryKey world_key,
    CraftaxState* out
) {
    // CraftaxState and CraftaxWorldState are layout-identical (see the
    // sizeof static assert), so generate in place instead of filling an
    // 80KB temporary and copying it over.
    craftax_generate_world_from_key_lazy(
        world_key,
        (CraftaxWorldState*)(void*)out,
        g_craftax_lazy_floors != 0);
    // Deferred floors keep zeroed spawn bits; craftax_ensure_floor_generated
    // refreshes them when the floor materializes.
    for (int32_t level = 0; level < CRAFTAX_NUM_LEVELS; level++) {
        if (out->lazy_floors_pending & (1u << (uint32_t)level)) continue;
        craftax_refresh_spawn_bits_level(out, level);
    }
}

static __device__ inline void craftax_reset_state_from_reset_key(
    CraftaxState* out,
    CraftaxThreefryKey reset_key
) {
    CraftaxThreefryKey unused;
    CraftaxThreefryKey world_key;
    craftax_threefry_split(reset_key, &unused, &world_key);
    craftax_generate_state_from_world_key(world_key, out);
}

// ============================================================
// Reset pool: pre-generate N worlds once, then memcpy on reset.
// Trades world diversity (<= pool_size unique maps per process) for
// ~500x faster reset. Set pool_size=0 to disable (exact per-seed
// world; required for the parity harness).
// ============================================================
static __device__ int g_craftax_reset_pool_size = 0;
static __device__ CraftaxState* g_craftax_reset_pool = NULL;
static __device__ int g_craftax_reset_pool_ready = 0;

// [cuda port] craftax_set_reset_pool_size removed: host-side calloc pool
// machinery is never used by this harness (pools stay disabled, size 0).

static __device__ inline void craftax_ensure_state_storage(Craftax* env) {
    if (env->state != NULL) {
        return;
    }
    // [cuda port] state storage is always preassigned by the host harness;
    // the C file's calloc fallback is unreachable on device.
    __trap();
}

static __device__ inline void craftax_reset_state_from_seed(Craftax* env) {
    craftax_ensure_state_storage(env);
    CraftaxThreefryKey initial_key = craftax_prng_key((uint32_t)env->seed);
    if (g_craftax_reset_pool_size > 0) {
        CraftaxThreefryKey discard;
        craftax_threefry_split(initial_key, &env->rng_key, &discard);
        int idx = (int)(env->seed % (uint64_t)g_craftax_reset_pool_size);
        memcpy(env->state, &g_craftax_reset_pool[idx], sizeof(CraftaxState));
        return;
    }
    CraftaxThreefryKey reset_key;
    craftax_threefry_split(initial_key, &env->rng_key, &reset_key);
    craftax_reset_state_from_reset_key(env->state, reset_key);
}

// ============================================================
// [cuda port] The pipelined pthread world pool and static reset pool from
// craftax_full.c (lines 3616-3744) are host-side conveniences and are
// removed. With pools disabled the hot-path reset is exactly the direct
// per-key generation, matching the C hash-mode configuration.
// ============================================================
static __device__ inline void craftax_reset_state_on_done(
    CraftaxState* out,
    CraftaxThreefryKey reset_key
) {
    craftax_reset_state_from_reset_key(out, reset_key);
}

static __device__ inline void craftax_encode_native_observation(
    const CraftaxState* state,
    CraftaxObs* obs
) {
    if (obs == NULL) {
        return;
    }
#ifdef CRAFTAX_COMPACT_OBS
    craftax_encode_compact_observation((const CraftaxWorldState*)(const void*)state, obs);
#else
    craftax_encode_reset_observation((const CraftaxWorldState*)(const void*)state, obs);
#endif
}

static __device__ inline float craftax_calculate_light_level_native(int32_t timestep) {
    // [cuda port] host-computed table; reference formula:
    //   progress = fmodf(timestep * (1.0f / 300.0f), 1.0f) + 0.3f
    //   1.0f - powf(fabsf(cosf(pi * progress)), 3.0f)
    return g_craftax_light_table[timestep];
}

static __device__ inline bool craftax_is_game_over_native(const CraftaxState* state) {
    return state->timestep >= CRAFTAX_DEFAULT_MAX_TIMESTEPS
        || state->player_health <= 0.0f;
}

static __device__ inline void craftax_copy_achievements_to_env(
    Craftax* env,
    const CraftaxState* state
) {
    for (int i = 0; i < CRAFTAX_NUM_ACHIEVEMENTS; i++) {
        env->achievements[i] = state->achievements[i] ? 1.0f : 0.0f;
    }
}

static __device__ void add_log(Craftax* env) {
    int unlocked = 0;
    for (int i = 0; i < CRAFTAX_NUM_ACHIEVEMENTS; i++) {
        if (env->achievements[i] > 0.5f) {
            unlocked++;
            env->log.achievements[i] += 1.0f;
        }
    }
    env->log.perf += (float)unlocked / (float)CRAFTAX_NUM_ACHIEVEMENTS;
    env->log.score += env->episode_return_accum;
    env->log.episode_return += env->episode_return_accum;
    env->log.episode_length += (float)env->episode_length_accum;
    env->log.n += 1.0f;
}

static __device__ float craftax_gameplay_step_native(
    CraftaxState* state,
    int32_t action,
    CraftaxThreefryKey rng
) {
    CRAFTAX_PROFILE_START();
    bool init_achievements[CRAFTAX_NUM_ACHIEVEMENTS];
    memcpy(init_achievements, state->achievements, sizeof(init_achievements));
    float init_health = state->player_health;

    action = state->is_sleeping ? CRAFTAX_ACTION_NOOP : action;
    action = state->is_resting ? CRAFTAX_ACTION_NOOP : action;

    CRAFTAX_PROFILE_ZONE(0);
    craftax_change_floor_native(state, action);
    craftax_do_crafting_native(state, action);
    CRAFTAX_PROFILE_END(0);

    CraftaxThreefryKey subkey = craftax_step_native_next_key(&rng);
    CRAFTAX_PROFILE_ZONE(2);
    craftax_do_action_native(state, action, subkey);
    CRAFTAX_PROFILE_END(2);

    CRAFTAX_PROFILE_ZONE(3);
    craftax_place_block_native(state, action);
    craftax_shoot_projectile_native(state, action);
    craftax_cast_spell_native(state, action);
    craftax_drink_potion_native(state, action);
    CRAFTAX_PROFILE_END(3);

    subkey = craftax_step_native_next_key(&rng);
    CRAFTAX_PROFILE_ZONE(4);
    craftax_read_book_native(state, subkey.word, action);
    CRAFTAX_PROFILE_END(4);

    subkey = craftax_step_native_next_key(&rng);
    CRAFTAX_PROFILE_ZONE(5);
    craftax_enchant_native(state, action, subkey);
    CRAFTAX_PROFILE_END(5);

    CRAFTAX_PROFILE_ZONE(6);
    craftax_boss_logic_native(state);
    craftax_level_up_attributes_native(state, action, CRAFTAX_MAX_ATTRIBUTE);
    craftax_move_player_native(state, action, false);
    CRAFTAX_PROFILE_END(6);

    subkey = craftax_step_native_next_key(&rng);
    CRAFTAX_PROFILE_ZONE(7);
    craftax_update_mobs_native(state, subkey);
    CRAFTAX_PROFILE_END(7);

    subkey = craftax_step_native_next_key(&rng);
    CRAFTAX_PROFILE_ZONE(8);
    craftax_spawn_mobs_native(state, subkey);
    CRAFTAX_PROFILE_END(8);

    CRAFTAX_PROFILE_ZONE(9);
    craftax_update_plants_native(state);
    craftax_update_player_intrinsics_native(state, action);
    craftax_clip_inventory_and_intrinsics_native(state, false);
    craftax_calculate_inventory_achievements_native(state);
    CRAFTAX_PROFILE_END(9);

    CRAFTAX_PROFILE_ZONE(10);
    float reward = 0.0f;
    for (int i = 0; i < CRAFTAX_NUM_ACHIEVEMENTS; i++) {
        int32_t delta = (int32_t)state->achievements[i]
            - (int32_t)init_achievements[i];
        reward += (float)delta * CRAFTAX_ACHIEVEMENT_REWARD_MAP[i];
    }
    reward += (state->player_health - init_health) * 0.1f;

    subkey = craftax_step_native_next_key(&rng);
    state->timestep += 1;
    state->light_level = craftax_calculate_light_level_native(state->timestep);
    state->state_rng[0] = subkey.word[0];
    state->state_rng[1] = subkey.word[1];
    CRAFTAX_PROFILE_END(10);

    return reward;
}

// ============================================================
// Public API expected by vecenv.h
// ============================================================
static __device__ void c_init(Craftax* env) {
    env->client = NULL;
    env->num_agents = 1;
    craftax_ensure_state_storage(env);
    env->episode_return_accum = 0.0f;
    env->episode_length_accum = 0;
    memset(env->achievements, 0, sizeof(env->achievements));
    memset(&env->log, 0, sizeof(env->log));
    craftax_wg_init_cell_templates();
    craftax_reset_state_from_seed(env);
}

static __device__ void c_reset(Craftax* env) {
    if (env->rewards != NULL) {
        env->rewards[0] = 0.0f;
    }
    if (env->terminals != NULL) {
        env->terminals[0] = 0.0f;
    }
    env->episode_return_accum = 0.0f;
    env->episode_length_accum = 0;
    memset(env->achievements, 0, sizeof(env->achievements));

    craftax_reset_state_from_seed(env);
    craftax_encode_native_observation(env->state, env->observations);
}

#ifdef CRAFTAX_PROFILE
static __device__ void c_step_native(Craftax* env) {
    CRAFTAX_PROFILE_START();
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;

    int action = (int)env->actions[0];
    if (action < 0) {
        action = CRAFTAX_ACTION_NOOP;
    }
    if (action >= CRAFTAX_NUM_ACTIONS) {
        action = CRAFTAX_NUM_ACTIONS - 1;
    }

    CRAFTAX_PROFILE_ZONE(12);
    CraftaxThreefryKey step_key;
    craftax_threefry_split(env->rng_key, &env->rng_key, &step_key);

    CraftaxThreefryKey step_rng;
    CraftaxThreefryKey reset_key;
    craftax_threefry_split(step_key, &step_rng, &reset_key);
    CRAFTAX_PROFILE_END(12);

    float reward = craftax_gameplay_step_native(env->state, action, step_rng);

    CRAFTAX_PROFILE_ZONE(13);
    bool done = craftax_is_game_over_native(env->state);
    CRAFTAX_PROFILE_END(13);

    CRAFTAX_PROFILE_ZONE(15);
    craftax_copy_achievements_to_env(env, env->state);
    CRAFTAX_PROFILE_END(15);

    CRAFTAX_PROFILE_ZONE(16);
    env->rewards[0] = reward;
    env->terminals[0] = done ? 1.0f : 0.0f;
    env->episode_return_accum += reward;
    env->episode_length_accum += 1;
    CRAFTAX_PROFILE_END(16);

    if (done) {
        add_log(env);
        env->episode_return_accum = 0.0f;
        env->episode_length_accum = 0;
        memset(env->achievements, 0, sizeof(env->achievements));
        CRAFTAX_PROFILE_ZONE(14);
        craftax_reset_state_on_done(env->state, reset_key);
        CRAFTAX_PROFILE_END(14);
    }

    CRAFTAX_PROFILE_ZONE(11);
    craftax_encode_native_observation(env->state, env->observations);
    CRAFTAX_PROFILE_END(11);

    // Record unprofiled time
    CRAFTAX_PROFILE_ZONE(17);
    CRAFTAX_PROFILE_END(17);

#ifdef CRAFTAX_PROFILE
    static int profile_step_count = 0;
    profile_step_count++;
    if (profile_step_count >= 100000) {
        craftax_profile_report();
        profile_step_count = 0;
    }

#endif
}

#endif

static __device__ void c_step_gameplay(Craftax* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;

    int action = (int)env->actions[0];
    if (action < 0) action = CRAFTAX_ACTION_NOOP;
    if (action >= CRAFTAX_NUM_ACTIONS) action = CRAFTAX_NUM_ACTIONS - 1;

    CraftaxThreefryKey step_key;
    craftax_threefry_split(env->rng_key, &env->rng_key, &step_key);
    CraftaxThreefryKey step_rng;
    CraftaxThreefryKey reset_key;
    craftax_threefry_split(step_key, &step_rng, &reset_key);

    float reward = craftax_gameplay_step_native(env->state, action, step_rng);
    bool done = craftax_is_game_over_native(env->state);
    craftax_copy_achievements_to_env(env, env->state);

    env->rewards[0] = reward;
    env->terminals[0] = done ? 1.0f : 0.0f;
    env->episode_return_accum += reward;
    env->episode_length_accum += 1;

    if (done) {
        add_log(env);
        env->episode_return_accum = 0.0f;
        env->episode_length_accum = 0;
        memset(env->achievements, 0, sizeof(env->achievements));
        craftax_reset_state_on_done(env->state, reset_key);
    }
}

static __device__ void c_step_encode(Craftax* env) {
    craftax_encode_native_observation(env->state, env->observations);
}

static __device__ void c_step(Craftax* env) {
    c_step_gameplay(env);
    c_step_encode(env);
}

static __device__ void c_close(Craftax* env) {
    if (!env->owns_state_storage || env->arena == NULL) {
        return;
    }
    free(env->arena->states);
    free(env->arena);
    env->arena = NULL;
    env->state = NULL;
    env->owns_state_storage = false;
}

#if 0  // rendering stripped: raylib not used in this standalone port
// ------------------------------------------------------------
// Tile-based renderer using upstream Craftax 16x16 PNG assets
// ------------------------------------------------------------
// Packed layout (see ocean/craftax/pack_textures.py):
//   [0..36] block textures (indexed by CraftaxBlockType)
//   [37..41] player: down, up, left, right, sleep
//   [42..46] items: none, torch, ladder_down, ladder_up, ladder_down_blocked

#define CRAFTAX_TEX_TILE_PX 16
#define CRAFTAX_TEX_SCALE 4   // on-screen px = 64
#define CRAFTAX_TEX_DRAW_PX (CRAFTAX_TEX_TILE_PX * CRAFTAX_TEX_SCALE)
#define CRAFTAX_TEX_NUM (37 + 5 + 5 + 3 + 4)

// Render viewport (independent of agent obs window)
#define CRAFTAX_RENDER_ROWS 16
#define CRAFTAX_RENDER_COLS 16

#define CRAFTAX_TEX_PLAYER_DOWN 37
#define CRAFTAX_TEX_PLAYER_UP 38
#define CRAFTAX_TEX_PLAYER_LEFT 39
#define CRAFTAX_TEX_PLAYER_RIGHT 40
#define CRAFTAX_TEX_PLAYER_SLEEP 41
#define CRAFTAX_TEX_ITEM_BASE 42

static __device__ Texture2D craftax_textures[CRAFTAX_TEX_NUM];
static __device__ bool craftax_textures_loaded = false;

static __device__ void craftax_load_textures(void) {
    if (craftax_textures_loaded) return;
    const char* candidates[] = {
        "resources/craftax/textures.bin",
        "../resources/craftax/textures.bin",
        "../../resources/craftax/textures.bin",
    };
    FILE* f = NULL;
    for (size_t i = 0; i < sizeof(candidates)/sizeof(candidates[0]); i++) {
        f = fopen(candidates[i], "rb");
        if (f) break;
    }
    if (!f) {
        fprintf(stderr, "craftax: textures.bin not found in resources/craftax -- run ocean/craftax/pack_textures.py\n");
        exit(1);
    }
    const size_t tile_bytes = CRAFTAX_TEX_TILE_PX * CRAFTAX_TEX_TILE_PX * 4;
    uint8_t* buf = (uint8_t*)malloc(tile_bytes);
    for (int i = 0; i < CRAFTAX_TEX_NUM; i++) {
        if (fread(buf, 1, tile_bytes, f) != tile_bytes) {
            fprintf(stderr, "craftax: short read on textures.bin at tile %d\n", i);
            exit(1);
        }
        Image img = {
            .data = buf,
            .width = CRAFTAX_TEX_TILE_PX,
            .height = CRAFTAX_TEX_TILE_PX,
            .mipmaps = 1,
            .format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8,
        };
        craftax_textures[i] = LoadTextureFromImage(img);
        SetTextureFilter(craftax_textures[i], TEXTURE_FILTER_POINT);
    }
    free(buf);
    fclose(f);
    craftax_textures_loaded = true;
}

static __device__ int craftax_player_tex_id(int32_t direction, bool sleeping) {
    if (sleeping) return CRAFTAX_TEX_PLAYER_SLEEP;
    switch (direction) {
        case 1: return CRAFTAX_TEX_PLAYER_LEFT;
        case 2: return CRAFTAX_TEX_PLAYER_RIGHT;
        case 3: return CRAFTAX_TEX_PLAYER_UP;
        case 4: return CRAFTAX_TEX_PLAYER_DOWN;
        default: return CRAFTAX_TEX_PLAYER_DOWN;
    }
}

static __device__ void craftax_draw_tile(int tex_id, int dst_x, int dst_y, float tint_alpha) {
    if (tex_id < 0 || tex_id >= CRAFTAX_TEX_NUM) return;
    Rectangle src = {0, 0, CRAFTAX_TEX_TILE_PX, CRAFTAX_TEX_TILE_PX};
    Rectangle dst = {(float)dst_x, (float)dst_y, CRAFTAX_TEX_DRAW_PX, CRAFTAX_TEX_DRAW_PX};
    Color tint = {255, 255, 255, (unsigned char)(tint_alpha * 255.0f)};
    DrawTexturePro(craftax_textures[tex_id], src, dst, (Vector2){0, 0}, 0.0f, tint);
}

static __device__ void c_render(Craftax* env) {
    const int view_w = CRAFTAX_RENDER_COLS * CRAFTAX_TEX_DRAW_PX;
    const int view_h = CRAFTAX_RENDER_ROWS * CRAFTAX_TEX_DRAW_PX;
    const int hud_h = 80;

    if (!IsWindowReady()) {
        InitWindow(view_w, view_h + hud_h, "PufferLib Craftax");
        SetTargetFPS(30);
    }
    if (!craftax_textures_loaded) craftax_load_textures();
    if (IsKeyDown(KEY_ESCAPE)) exit(0);

    CraftaxState* s = env->state;
    int lvl = s->player_level;
    int pr = s->player_position[0];
    int pc = s->player_position[1];
    int half_r = CRAFTAX_RENDER_ROWS / 2;
    int half_c = CRAFTAX_RENDER_COLS / 2;

    BeginDrawing();
    ClearBackground(BLACK);

    for (int vr = 0; vr < CRAFTAX_RENDER_ROWS; vr++) {
        for (int vc = 0; vc < CRAFTAX_RENDER_COLS; vc++) {
            int wr = pr - half_r + vr;
            int wc = pc - half_c + vc;
            int dst_x = vc * CRAFTAX_TEX_DRAW_PX;
            int dst_y = vr * CRAFTAX_TEX_DRAW_PX;

            int blk = CRAFTAX_BLOCK_OUT_OF_BOUNDS;
            if (wr >= 0 && wr < CRAFTAX_MAP_SIZE && wc >= 0 && wc < CRAFTAX_MAP_SIZE) {
                blk = s->map[lvl][wr][wc];
                if (s->light_map[lvl][wr][wc] <= 12) blk = CRAFTAX_BLOCK_DARKNESS;
            }
            if (blk < 0 || blk >= CRAFTAX_NUM_BLOCK_TYPES) blk = 0;
            craftax_draw_tile(blk, dst_x, dst_y, 1.0f);

            // item overlay
            if (wr >= 0 && wr < CRAFTAX_MAP_SIZE && wc >= 0 && wc < CRAFTAX_MAP_SIZE) {
                int it = s->item_map[lvl][wr][wc];
                if (it > 0 && it < 5) {
                    craftax_draw_tile(CRAFTAX_TEX_ITEM_BASE + it, dst_x, dst_y, 1.0f);
                }
            }
        }
    }

    // player in center
    int pid = craftax_player_tex_id(s->player_direction, s->is_sleeping);
    craftax_draw_tile(pid, half_c * CRAFTAX_TEX_DRAW_PX, half_r * CRAFTAX_TEX_DRAW_PX, 1.0f);

    // night dim overlay
    if (s->light_level < 1.0f) {
        unsigned char a = (unsigned char)((1.0f - s->light_level) * 140.0f);
        DrawRectangle(0, 0, view_w, view_h, (Color){0, 0, 40, a});
    }

    // HUD
    int hud_y = view_h;
    DrawRectangle(0, hud_y, view_w, hud_h, (Color){20, 20, 20, 255});
    DrawText(TextFormat("HP:%.0f  F:%d  D:%d  E:%d  M:%d  L:%d  t:%d",
             s->player_health, s->player_food, s->player_drink,
             s->player_energy, s->player_mana, s->player_level, s->timestep),
             4, hud_y + 4, 14, WHITE);
    DrawText(TextFormat("XP:%d  DEX:%d  STR:%d  INT:%d  light:%.2f",
             s->player_xp, s->player_dexterity, s->player_strength,
             s->player_intelligence, s->light_level),
             4, hud_y + 22, 14, (Color){200, 200, 200, 255});
    int ach_count = 0;
    for (int i = 0; i < CRAFTAX_NUM_ACHIEVEMENTS; i++) ach_count += s->achievements[i] ? 1 : 0;
    DrawText(TextFormat("achievements: %d / %d", ach_count, CRAFTAX_NUM_ACHIEVEMENTS),
             4, hud_y + 40, 14, (Color){180, 220, 180, 255});
    DrawText(TextFormat("ret:%.2f len:%d", env->episode_return_accum, env->episode_length_accum),
             4, hud_y + 58, 14, (Color){200, 200, 140, 255});

    EndDrawing();
}

#endif  // rendering stripped
#endif

// ============================================================
// ===== step_simple.h =====
// ============================================================
// Standalone native ports of simple Craftax step subsystems.
//
// These helpers intentionally are not integrated into c_step yet. They mutate a
// full CraftaxState in place so tests can compare each subsystem directly
// against the installed JAX implementation.



static __device__ inline int32_t craftax_step_jax_index(int32_t index, int32_t size) {
    if (index < 0) {
        index += size;
    }
    if (index < 0) {
        return 0;
    }
    if (index >= size) {
        return size - 1;
    }
    return index;
}

static __device__ inline int32_t craftax_step_mini32(int32_t a, int32_t b) {
    return a < b ? a : b;
}

static __device__ inline int32_t craftax_step_maxi32(int32_t a, int32_t b) {
    return a > b ? a : b;
}

static __device__ inline float craftax_step_minf32(float a, float b) {
    if (isnan(a) || isnan(b)) {
        return NAN;
    }
    return a < b ? a : b;
}

static __device__ inline float craftax_step_maxf32(float a, float b) {
    if (isnan(a) || isnan(b)) {
        return NAN;
    }
    return a > b ? a : b;
}

static __device__ inline int32_t craftax_step_get_max_health(const CraftaxState* state) {
    return 8 + state->player_strength;
}

static __device__ inline int32_t craftax_step_get_max_food(const CraftaxState* state) {
    return 7 + 2 * state->player_dexterity;
}

static __device__ inline int32_t craftax_step_get_max_drink(const CraftaxState* state) {
    return 7 + 2 * state->player_dexterity;
}

static __device__ inline int32_t craftax_step_get_max_energy(const CraftaxState* state) {
    return 7 + 2 * state->player_dexterity;
}

static __device__ inline int32_t craftax_step_get_max_mana(const CraftaxState* state) {
    return 6 + 3 * state->player_intelligence;
}

static __device__ inline bool craftax_step_is_fighting_boss(const CraftaxState* state) {
    return state->player_level == CRAFTAX_NUM_LEVELS - 1;
}

static __device__ inline bool craftax_step_has_beaten_boss(const CraftaxState* state) {
    return state->boss_progress >= CRAFTAX_NUM_LEVELS - 1;
}

static __device__ inline void craftax_step_direction(int32_t action, int32_t direction[2]) {
    direction[0] = 0;
    direction[1] = 0;
    int32_t direction_index = craftax_step_jax_index(action, 16);
    if (direction_index == CRAFTAX_ACTION_LEFT) {
        direction[1] = -1;
    } else if (direction_index == CRAFTAX_ACTION_RIGHT) {
        direction[1] = 1;
    } else if (direction_index == CRAFTAX_ACTION_UP) {
        direction[0] = -1;
    } else if (direction_index == CRAFTAX_ACTION_DOWN) {
        direction[0] = 1;
    }
}

static __device__ inline bool craftax_step_is_solid_block(int32_t block) {
    switch (block) {
    case CRAFTAX_BLOCK_STONE:
    case CRAFTAX_BLOCK_TREE:
    case CRAFTAX_BLOCK_COAL:
    case CRAFTAX_BLOCK_IRON:
    case CRAFTAX_BLOCK_DIAMOND:
    case CRAFTAX_BLOCK_CRAFTING_TABLE:
    case CRAFTAX_BLOCK_FURNACE:
    case CRAFTAX_BLOCK_PLANT:
    case CRAFTAX_BLOCK_RIPE_PLANT:
    case CRAFTAX_BLOCK_WALL:
    case CRAFTAX_BLOCK_WALL_MOSS:
    case CRAFTAX_BLOCK_STALAGMITE:
    case CRAFTAX_BLOCK_RUBY:
    case CRAFTAX_BLOCK_SAPPHIRE:
    case CRAFTAX_BLOCK_CHEST:
    case CRAFTAX_BLOCK_FOUNTAIN:
    case CRAFTAX_BLOCK_FIRE_TREE:
    case CRAFTAX_BLOCK_ENCHANTMENT_TABLE_FIRE:
    case CRAFTAX_BLOCK_ENCHANTMENT_TABLE_ICE:
    case CRAFTAX_BLOCK_GRAVE:
    case CRAFTAX_BLOCK_GRAVE2:
    case CRAFTAX_BLOCK_GRAVE3:
    case CRAFTAX_BLOCK_NECROMANCER:
        return true;
    default:
        return false;
    }
}

static __device__ inline bool craftax_step_is_in_mob(
    const CraftaxState* state,
    int32_t row,
    int32_t col
) {
    int32_t level = craftax_step_jax_index(state->player_level, CRAFTAX_NUM_LEVELS);
    int32_t map_row = craftax_step_jax_index(row, CRAFTAX_MAP_SIZE);
    int32_t map_col = craftax_step_jax_index(col, CRAFTAX_MAP_SIZE);
    bool player_here = state->player_position[0] == row
        && state->player_position[1] == col;
    return ((state->mob_bits[level][map_row] >> map_col) & 1ULL) || player_here;
}

static __device__ inline bool craftax_step_valid_land_position(
    const CraftaxState* state,
    int32_t row,
    int32_t col
) {
    bool pos_in_bounds = row >= 0
        && row < CRAFTAX_MAP_SIZE
        && col >= 0
        && col < CRAFTAX_MAP_SIZE;
    int32_t level = craftax_step_jax_index(state->player_level, CRAFTAX_NUM_LEVELS);
    int32_t map_row = craftax_step_jax_index(row, CRAFTAX_MAP_SIZE);
    int32_t map_col = craftax_step_jax_index(col, CRAFTAX_MAP_SIZE);
    int32_t block = state->map[level][map_row][map_col];
    bool in_solid_block = craftax_step_is_solid_block(block);
    bool in_mob = craftax_step_is_in_mob(state, row, col);
    bool in_lava = block == CRAFTAX_BLOCK_LAVA;
    bool in_water = block == CRAFTAX_BLOCK_WATER;

    bool valid_move = pos_in_bounds && !in_mob && !in_solid_block;
    valid_move = valid_move && !in_water;
    valid_move = valid_move && !in_lava;
    return valid_move;
}

static __device__ inline void craftax_move_player_native(
    CraftaxState* state,
    int32_t action,
    bool god_mode
) {
    int32_t direction[2];
    craftax_step_direction(action, direction);

    int32_t proposed_row = state->player_position[0] + direction[0];
    int32_t proposed_col = state->player_position[1] + direction[1];
    bool valid_move = craftax_step_valid_land_position(
        state,
        proposed_row,
        proposed_col
    );
    valid_move = valid_move || god_mode;

    state->player_position[0] += (int32_t)valid_move * direction[0];
    state->player_position[1] += (int32_t)valid_move * direction[1];

    bool is_new_direction = direction[0] != 0 || direction[1] != 0;
    state->player_direction = state->player_direction * (1 - (int32_t)is_new_direction)
        + action * (int32_t)is_new_direction;
}

static __device__ inline void craftax_update_plants_native(CraftaxState* state) {
    bool finished_growing_plants[CRAFTAX_MAX_GROWING_PLANTS];

    for (int plant = 0; plant < CRAFTAX_MAX_GROWING_PLANTS; plant++) {
        state->growing_plants_age[plant] =
            (state->growing_plants_age[plant] + 1)
            * (int32_t)state->growing_plants_mask[plant];
        finished_growing_plants[plant] = state->growing_plants_age[plant] >= 600;
    }

    for (int plant = 0; plant < CRAFTAX_MAX_GROWING_PLANTS; plant++) {
        int32_t row = craftax_step_jax_index(
            state->growing_plants_positions[plant][0],
            CRAFTAX_MAP_SIZE
        );
        int32_t col = craftax_step_jax_index(
            state->growing_plants_positions[plant][1],
            CRAFTAX_MAP_SIZE
        );
        int32_t new_block = finished_growing_plants[plant]
            ? CRAFTAX_BLOCK_RIPE_PLANT
            : state->map[0][row][col];
        craftax_set_map_block(state, 0, row, col, new_block);
    }
}

static __device__ inline void craftax_boss_logic_native(CraftaxState* state) {
    state->achievements[CRAFTAX_ACH_DEFEAT_NECROMANCER] =
        state->achievements[CRAFTAX_ACH_DEFEAT_NECROMANCER]
        || craftax_step_has_beaten_boss(state);
    state->boss_timesteps_to_spawn_this_round -=
        (int32_t)craftax_step_is_fighting_boss(state);
}

static __device__ inline void craftax_level_up_attributes_native(
    CraftaxState* state,
    int32_t action,
    int32_t max_attribute
) {
    bool can_level_up = state->player_xp >= 1;
    bool is_levelling_up_dex = can_level_up
        && action == CRAFTAX_ACTION_LEVEL_UP_DEXTERITY
        && state->player_dexterity < max_attribute;
    bool is_levelling_up_str = can_level_up
        && action == CRAFTAX_ACTION_LEVEL_UP_STRENGTH
        && state->player_strength < max_attribute;
    bool is_levelling_up_int = can_level_up
        && action == CRAFTAX_ACTION_LEVEL_UP_INTELLIGENCE
        && state->player_intelligence < max_attribute;
    bool is_levelling_up = is_levelling_up_dex
        || is_levelling_up_str
        || is_levelling_up_int;

    state->player_dexterity += (int32_t)is_levelling_up_dex;
    state->player_strength += (int32_t)is_levelling_up_str;
    state->player_intelligence += (int32_t)is_levelling_up_int;
    state->player_xp -= (int32_t)is_levelling_up;
}

static __device__ inline void craftax_clip_inventory_and_intrinsics_native(
    CraftaxState* state,
    bool god_mode
) {
    state->inventory.wood = craftax_step_mini32(state->inventory.wood, 99);
    state->inventory.stone = craftax_step_mini32(state->inventory.stone, 99);
    state->inventory.coal = craftax_step_mini32(state->inventory.coal, 99);
    state->inventory.iron = craftax_step_mini32(state->inventory.iron, 99);
    state->inventory.diamond = craftax_step_mini32(state->inventory.diamond, 99);
    state->inventory.sapling = craftax_step_mini32(state->inventory.sapling, 99);
    state->inventory.pickaxe = craftax_step_mini32(state->inventory.pickaxe, 99);
    state->inventory.sword = craftax_step_mini32(state->inventory.sword, 99);
    state->inventory.bow = craftax_step_mini32(state->inventory.bow, 99);
    state->inventory.arrows = craftax_step_mini32(state->inventory.arrows, 99);
    for (int i = 0; i < 4; i++) {
        state->inventory.armour[i] = craftax_step_mini32(
            state->inventory.armour[i],
            99
        );
    }
    state->inventory.torches = craftax_step_mini32(state->inventory.torches, 99);
    state->inventory.ruby = craftax_step_mini32(state->inventory.ruby, 99);
    state->inventory.sapphire = craftax_step_mini32(state->inventory.sapphire, 99);
    for (int i = 0; i < 6; i++) {
        state->inventory.potions[i] = craftax_step_mini32(
            state->inventory.potions[i],
            99
        );
    }
    state->inventory.books = craftax_step_mini32(state->inventory.books, 99);

    float min_health = god_mode ? 9.0f : 0.0f;
    state->player_health = craftax_step_minf32(
        craftax_step_maxf32(state->player_health, min_health),
        (float)craftax_step_get_max_health(state)
    );
    state->player_food = craftax_step_mini32(
        craftax_step_maxi32(state->player_food, 0),
        craftax_step_get_max_food(state)
    );
    state->player_drink = craftax_step_mini32(
        craftax_step_maxi32(state->player_drink, 0),
        craftax_step_get_max_drink(state)
    );
    state->player_energy = craftax_step_mini32(
        craftax_step_maxi32(state->player_energy, 0),
        craftax_step_get_max_energy(state)
    );
    state->player_mana = craftax_step_mini32(
        craftax_step_maxi32(state->player_mana, 0),
        craftax_step_get_max_mana(state)
    );
}

static __device__ inline void craftax_calculate_inventory_achievements_native(
    CraftaxState* state
) {
    state->achievements[CRAFTAX_ACH_COLLECT_WOOD] =
        state->achievements[CRAFTAX_ACH_COLLECT_WOOD] || state->inventory.wood > 0;
    state->achievements[CRAFTAX_ACH_COLLECT_STONE] =
        state->achievements[CRAFTAX_ACH_COLLECT_STONE] || state->inventory.stone > 0;
    state->achievements[CRAFTAX_ACH_COLLECT_COAL] =
        state->achievements[CRAFTAX_ACH_COLLECT_COAL] || state->inventory.coal > 0;
    state->achievements[CRAFTAX_ACH_COLLECT_IRON] =
        state->achievements[CRAFTAX_ACH_COLLECT_IRON] || state->inventory.iron > 0;
    state->achievements[CRAFTAX_ACH_COLLECT_DIAMOND] =
        state->achievements[CRAFTAX_ACH_COLLECT_DIAMOND] || state->inventory.diamond > 0;
    state->achievements[CRAFTAX_ACH_COLLECT_RUBY] =
        state->achievements[CRAFTAX_ACH_COLLECT_RUBY] || state->inventory.ruby > 0;
    state->achievements[CRAFTAX_ACH_COLLECT_SAPPHIRE] =
        state->achievements[CRAFTAX_ACH_COLLECT_SAPPHIRE]
        || state->inventory.sapphire > 0;
    state->achievements[CRAFTAX_ACH_COLLECT_SAPLING] =
        state->achievements[CRAFTAX_ACH_COLLECT_SAPLING]
        || state->inventory.sapling > 0;
    state->achievements[CRAFTAX_ACH_FIND_BOW] =
        state->achievements[CRAFTAX_ACH_FIND_BOW] || state->inventory.bow > 0;
    state->achievements[CRAFTAX_ACH_MAKE_ARROW] =
        state->achievements[CRAFTAX_ACH_MAKE_ARROW] || state->inventory.arrows > 0;
    state->achievements[CRAFTAX_ACH_MAKE_TORCH] =
        state->achievements[CRAFTAX_ACH_MAKE_TORCH] || state->inventory.torches > 0;

    state->achievements[CRAFTAX_ACH_MAKE_WOOD_PICKAXE] =
        state->achievements[CRAFTAX_ACH_MAKE_WOOD_PICKAXE]
        || state->inventory.pickaxe >= 1;
    state->achievements[CRAFTAX_ACH_MAKE_STONE_PICKAXE] =
        state->achievements[CRAFTAX_ACH_MAKE_STONE_PICKAXE]
        || state->inventory.pickaxe >= 2;
    state->achievements[CRAFTAX_ACH_MAKE_IRON_PICKAXE] =
        state->achievements[CRAFTAX_ACH_MAKE_IRON_PICKAXE]
        || state->inventory.pickaxe >= 3;
    state->achievements[CRAFTAX_ACH_MAKE_DIAMOND_PICKAXE] =
        state->achievements[CRAFTAX_ACH_MAKE_DIAMOND_PICKAXE]
        || state->inventory.pickaxe >= 4;

    state->achievements[CRAFTAX_ACH_MAKE_WOOD_SWORD] =
        state->achievements[CRAFTAX_ACH_MAKE_WOOD_SWORD]
        || state->inventory.sword >= 1;
    state->achievements[CRAFTAX_ACH_MAKE_STONE_SWORD] =
        state->achievements[CRAFTAX_ACH_MAKE_STONE_SWORD]
        || state->inventory.sword >= 2;
    state->achievements[CRAFTAX_ACH_MAKE_IRON_SWORD] =
        state->achievements[CRAFTAX_ACH_MAKE_IRON_SWORD]
        || state->inventory.sword >= 3;
    state->achievements[CRAFTAX_ACH_MAKE_DIAMOND_SWORD] =
        state->achievements[CRAFTAX_ACH_MAKE_DIAMOND_SWORD]
        || state->inventory.sword >= 4;
}

static __device__ inline void craftax_update_player_intrinsics_native(
    CraftaxState* state,
    int32_t action
) {
    bool is_starting_sleep = action == CRAFTAX_ACTION_SLEEP
        && state->player_energy < craftax_step_get_max_energy(state);
    state->is_sleeping = state->is_sleeping || is_starting_sleep;

    bool is_waking_up = state->player_energy >= craftax_step_get_max_energy(state)
        && state->is_sleeping;
    state->is_sleeping = state->is_sleeping && !is_waking_up;
    state->achievements[CRAFTAX_ACH_WAKE_UP] =
        state->achievements[CRAFTAX_ACH_WAKE_UP] || is_waking_up;

    bool is_starting_rest = action == CRAFTAX_ACTION_REST
        && state->player_health < (float)craftax_step_get_max_health(state);
    state->is_resting = state->is_resting || is_starting_rest;

    is_waking_up = state->is_resting
        && (
            state->player_health >= (float)craftax_step_get_max_health(state)
            || state->player_food <= 0
            || state->player_drink <= 0
        );
    state->is_resting = state->is_resting && !is_waking_up;

    bool not_boss = !craftax_step_is_fighting_boss(state);
    float intrinsic_decay_coeff =
        1.0f - (0.125f * (float)(state->player_dexterity - 1));

    float hunger_add = (state->is_sleeping ? 0.5f : 1.0f) * intrinsic_decay_coeff;
    float new_hunger = state->player_hunger + hunger_add;
    int32_t hungered_food = craftax_step_maxi32(
        state->player_food - (int32_t)not_boss,
        0
    );
    int32_t new_food = new_hunger > 25.0f ? hungered_food : state->player_food;
    new_hunger = new_hunger > 25.0f ? 0.0f : new_hunger;
    state->player_hunger = new_hunger;
    state->player_food = new_food;

    float thirst_add = (state->is_sleeping ? 0.5f : 1.0f) * intrinsic_decay_coeff;
    float new_thirst = state->player_thirst + thirst_add;
    int32_t thirsted_drink = craftax_step_maxi32(
        state->player_drink - (int32_t)not_boss,
        0
    );
    int32_t new_drink = new_thirst > 20.0f ? thirsted_drink : state->player_drink;
    new_thirst = new_thirst > 20.0f ? 0.0f : new_thirst;
    state->player_thirst = new_thirst;
    state->player_drink = new_drink;

    float new_fatigue = state->is_sleeping
        ? craftax_step_minf32(state->player_fatigue - 1.0f, 0.0f)
        : state->player_fatigue + intrinsic_decay_coeff;
    int32_t new_energy = new_fatigue > 30.0f
        ? craftax_step_maxi32(state->player_energy - (int32_t)not_boss, 0)
        : state->player_energy;
    new_fatigue = new_fatigue > 30.0f ? 0.0f : new_fatigue;
    new_energy = new_fatigue < -10.0f
        ? craftax_step_mini32(
            state->player_energy + 1,
            craftax_step_get_max_energy(state)
        )
        : new_energy;
    new_fatigue = new_fatigue < -10.0f ? 0.0f : new_fatigue;
    state->player_fatigue = new_fatigue;
    state->player_energy = new_energy;

    bool all_necessities = state->player_food > 0
        && state->player_drink > 0
        && (state->player_energy > 0 || state->is_sleeping);
    float recover_all = state->is_sleeping ? 2.0f : 1.0f;
    float recover_not_all = (state->is_sleeping ? -0.5f : -1.0f)
        * (float)(int32_t)not_boss;
    float recover_add = all_necessities ? recover_all : recover_not_all;
    float new_recover = state->player_recover + recover_add;

    float recovered_health = craftax_step_minf32(
        state->player_health + 1.0f,
        (float)craftax_step_get_max_health(state)
    );
    float derecovered_health = state->player_health - 1.0f;
    float new_health = new_recover > 25.0f
        ? recovered_health
        : state->player_health;
    new_recover = new_recover > 25.0f ? 0.0f : new_recover;
    new_health = new_recover < -15.0f ? derecovered_health : new_health;
    new_recover = new_recover < -15.0f ? 0.0f : new_recover;
    state->player_recover = new_recover;
    state->player_health = new_health;

    float mana_recover_coeff =
        1.0f + 0.25f * (float)(state->player_intelligence - 1);
    float new_recover_mana = (
        state->is_sleeping
            ? state->player_recover_mana + 2.0f
            : state->player_recover_mana + 1.0f
    ) * mana_recover_coeff;
    int32_t new_mana = new_recover_mana > 30.0f
        ? state->player_mana + 1
        : state->player_mana;
    new_recover_mana = new_recover_mana > 30.0f ? 0.0f : new_recover_mana;
    state->player_recover_mana = new_recover_mana;
    state->player_mana = new_mana;
}

static __device__ inline void craftax_drink_potion_native(
    CraftaxState* state,
    int32_t action
) {
    int32_t drinking_potion_index = -1;
    bool is_drinking_potion = false;

    bool is_drinking_red_potion = action == CRAFTAX_ACTION_DRINK_POTION_RED
        && state->inventory.potions[0] > 0;
    drinking_potion_index = (int32_t)is_drinking_red_potion * 0
        + (1 - (int32_t)is_drinking_red_potion) * drinking_potion_index;
    is_drinking_potion = is_drinking_potion || is_drinking_red_potion;

    bool is_drinking_green_potion = action == CRAFTAX_ACTION_DRINK_POTION_GREEN
        && state->inventory.potions[1] > 0;
    drinking_potion_index = (int32_t)is_drinking_green_potion * 1
        + (1 - (int32_t)is_drinking_green_potion) * drinking_potion_index;
    is_drinking_potion = is_drinking_potion || is_drinking_green_potion;

    bool is_drinking_blue_potion = action == CRAFTAX_ACTION_DRINK_POTION_BLUE
        && state->inventory.potions[2] > 0;
    drinking_potion_index = (int32_t)is_drinking_blue_potion * 2
        + (1 - (int32_t)is_drinking_blue_potion) * drinking_potion_index;
    is_drinking_potion = is_drinking_potion || is_drinking_blue_potion;

    bool is_drinking_pink_potion = action == CRAFTAX_ACTION_DRINK_POTION_PINK
        && state->inventory.potions[3] > 0;
    drinking_potion_index = (int32_t)is_drinking_pink_potion * 3
        + (1 - (int32_t)is_drinking_pink_potion) * drinking_potion_index;
    is_drinking_potion = is_drinking_potion || is_drinking_pink_potion;

    bool is_drinking_cyan_potion = action == CRAFTAX_ACTION_DRINK_POTION_CYAN
        && state->inventory.potions[4] > 0;
    drinking_potion_index = (int32_t)is_drinking_cyan_potion * 4
        + (1 - (int32_t)is_drinking_cyan_potion) * drinking_potion_index;
    is_drinking_potion = is_drinking_potion || is_drinking_cyan_potion;

    bool is_drinking_yellow_potion = action == CRAFTAX_ACTION_DRINK_POTION_YELLOW
        && state->inventory.potions[5] > 0;
    drinking_potion_index = (int32_t)is_drinking_yellow_potion * 5
        + (1 - (int32_t)is_drinking_yellow_potion) * drinking_potion_index;
    is_drinking_potion = is_drinking_potion || is_drinking_yellow_potion;

    int32_t potion_index = craftax_step_jax_index(drinking_potion_index, 6);
    int32_t potion_effect_index = state->potion_mapping[potion_index];

    int32_t delta_health = 0;
    delta_health += (int32_t)is_drinking_potion * (int32_t)(potion_effect_index == 0) * 8;
    delta_health += (int32_t)is_drinking_potion * (int32_t)(potion_effect_index == 1) * -3;

    int32_t delta_mana = 0;
    delta_mana += (int32_t)is_drinking_potion * (int32_t)(potion_effect_index == 2) * 8;
    delta_mana += (int32_t)is_drinking_potion * (int32_t)(potion_effect_index == 3) * -3;

    int32_t delta_energy = 0;
    delta_energy += (int32_t)is_drinking_potion * (int32_t)(potion_effect_index == 4) * 8;
    delta_energy += (int32_t)is_drinking_potion * (int32_t)(potion_effect_index == 5) * -3;

    state->achievements[CRAFTAX_ACH_DRINK_POTION] =
        state->achievements[CRAFTAX_ACH_DRINK_POTION] || is_drinking_potion;
    state->inventory.potions[potion_index] =
        state->inventory.potions[potion_index] - (int32_t)is_drinking_potion;
    state->player_health += (float)delta_health;
    state->player_mana += delta_mana;
    state->player_energy += delta_energy;
}

static __device__ inline void craftax_read_book_native(
    CraftaxState* state,
    const uint32_t rng_words[2],
    int32_t action
) {
    bool is_reading_book = action == CRAFTAX_ACTION_READ_BOOK
        && state->inventory.books > 0;

    CraftaxThreefryKey rng = {{rng_words[0], rng_words[1]}};
    CraftaxThreefryKey unused;
    CraftaxThreefryKey choice_key;
    craftax_threefry_split(rng, &unused, &choice_key);

    float p0 = state->learned_spells[0] ? 0.0f : 1.0f;
    float p1 = state->learned_spells[1] ? 0.0f : 1.0f;
    float p_sum = p0 + p1;
    int32_t spell_to_learn_index = 0;
    if (p_sum != 0.0f) {
        p0 /= p_sum;
        float r = 1.0f - craftax_threefry_uniform_f32(choice_key);
        spell_to_learn_index = r <= p0 ? 0 : 1;
    }

    int32_t learn_spell_achievement = spell_to_learn_index
        ? CRAFTAX_ACH_LEARN_ICEBALL
        : CRAFTAX_ACH_LEARN_FIREBALL;

    state->achievements[learn_spell_achievement] =
        state->achievements[learn_spell_achievement] || is_reading_book;
    state->inventory.books -= (int32_t)is_reading_book;
    state->learned_spells[spell_to_learn_index] =
        state->learned_spells[spell_to_learn_index] || is_reading_book;
}

// ============================================================
// ===== step_crafting.h =====
// ============================================================
// Standalone native ports of Craftax crafting and placement subsystems.
//
// These helpers intentionally are not integrated into c_step yet. They mutate a
// full CraftaxState in place so tests can compare each subsystem directly
// against the installed JAX implementation.



static __device__ inline bool craftax_crafting_is_near_block(
    const CraftaxState* state,
    int32_t block_type
) {
    static const int32_t close_blocks[8][2] = {
        {0, -1},
        {0, 1},
        {-1, 0},
        {1, 0},
        {-1, -1},
        {-1, 1},
        {1, -1},
        {1, 1},
    };

    int32_t level = craftax_step_jax_index(
        state->player_level,
        CRAFTAX_NUM_LEVELS
    );
    for (int32_t i = 0; i < 8; i++) {
        int32_t row = state->player_position[0] + close_blocks[i][0];
        int32_t col = state->player_position[1] + close_blocks[i][1];
        bool in_bounds = row >= 0
            && row < CRAFTAX_MAP_SIZE
            && col >= 0
            && col < CRAFTAX_MAP_SIZE;
        if (in_bounds && state->map[level][row][col] == block_type) {
            return true;
        }
    }
    return false;
}

static __device__ inline int32_t craftax_crafting_first_armour_below(
    const CraftaxInventory* inventory,
    int32_t threshold,
    int32_t* count
) {
    int32_t first = 0;
    *count = 0;
    for (int32_t i = 0; i < 4; i++) {
        bool below = inventory->armour[i] < threshold;
        first = (*count == 0 && below) ? i : first;
        *count += (int32_t)below;
    }
    return first;
}

static __device__ inline void craftax_do_crafting_native(
    CraftaxState* state,
    int32_t action
) {
    bool is_at_crafting_table = craftax_crafting_is_near_block(
        state,
        CRAFTAX_BLOCK_CRAFTING_TABLE
    );
    bool is_at_furnace = craftax_crafting_is_near_block(
        state,
        CRAFTAX_BLOCK_FURNACE
    );

    CraftaxInventory* inventory = &state->inventory;

    bool can_craft_wood_pickaxe = inventory->wood >= 1;
    bool is_crafting_wood_pickaxe =
        action == CRAFTAX_ACTION_MAKE_WOOD_PICKAXE
        && can_craft_wood_pickaxe
        && is_at_crafting_table
        && inventory->pickaxe < 1;
    inventory->wood -= 1 * (int32_t)is_crafting_wood_pickaxe;
    inventory->pickaxe =
        inventory->pickaxe * (1 - (int32_t)is_crafting_wood_pickaxe)
        + 1 * (int32_t)is_crafting_wood_pickaxe;

    bool can_craft_stone_pickaxe =
        inventory->wood >= 1 && inventory->stone >= 1;
    bool is_crafting_stone_pickaxe =
        action == CRAFTAX_ACTION_MAKE_STONE_PICKAXE
        && can_craft_stone_pickaxe
        && is_at_crafting_table
        && inventory->pickaxe < 2;
    inventory->stone -= 1 * (int32_t)is_crafting_stone_pickaxe;
    inventory->wood -= 1 * (int32_t)is_crafting_stone_pickaxe;
    inventory->pickaxe =
        inventory->pickaxe * (1 - (int32_t)is_crafting_stone_pickaxe)
        + 2 * (int32_t)is_crafting_stone_pickaxe;

    bool can_craft_iron_pickaxe =
        inventory->wood >= 1
        && inventory->stone >= 1
        && inventory->iron >= 1
        && inventory->coal >= 1;
    bool is_crafting_iron_pickaxe =
        action == CRAFTAX_ACTION_MAKE_IRON_PICKAXE
        && can_craft_iron_pickaxe
        && is_at_furnace
        && is_at_crafting_table
        && inventory->pickaxe < 3;
    inventory->iron -= 1 * (int32_t)is_crafting_iron_pickaxe;
    inventory->wood -= 1 * (int32_t)is_crafting_iron_pickaxe;
    inventory->stone -= 1 * (int32_t)is_crafting_iron_pickaxe;
    inventory->coal -= 1 * (int32_t)is_crafting_iron_pickaxe;
    inventory->pickaxe =
        inventory->pickaxe * (1 - (int32_t)is_crafting_iron_pickaxe)
        + 3 * (int32_t)is_crafting_iron_pickaxe;

    bool can_craft_diamond_pickaxe =
        inventory->wood >= 1 && inventory->diamond >= 3;
    bool is_crafting_diamond_pickaxe =
        action == CRAFTAX_ACTION_MAKE_DIAMOND_PICKAXE
        && can_craft_diamond_pickaxe
        && is_at_crafting_table
        && inventory->pickaxe < 4;
    inventory->diamond -= 3 * (int32_t)is_crafting_diamond_pickaxe;
    inventory->wood -= 1 * (int32_t)is_crafting_diamond_pickaxe;
    inventory->pickaxe =
        inventory->pickaxe * (1 - (int32_t)is_crafting_diamond_pickaxe)
        + 4 * (int32_t)is_crafting_diamond_pickaxe;

    bool can_craft_wood_sword = inventory->wood >= 1;
    bool is_crafting_wood_sword =
        action == CRAFTAX_ACTION_MAKE_WOOD_SWORD
        && can_craft_wood_sword
        && is_at_crafting_table
        && inventory->sword < 1;
    inventory->wood -= 1 * (int32_t)is_crafting_wood_sword;
    inventory->sword =
        inventory->sword * (1 - (int32_t)is_crafting_wood_sword)
        + 1 * (int32_t)is_crafting_wood_sword;

    bool can_craft_stone_sword =
        inventory->stone >= 1 && inventory->wood >= 1;
    bool is_crafting_stone_sword =
        action == CRAFTAX_ACTION_MAKE_STONE_SWORD
        && can_craft_stone_sword
        && is_at_crafting_table
        && inventory->sword < 2;
    inventory->wood -= 1 * (int32_t)is_crafting_stone_sword;
    inventory->stone -= 1 * (int32_t)is_crafting_stone_sword;
    inventory->sword =
        inventory->sword * (1 - (int32_t)is_crafting_stone_sword)
        + 2 * (int32_t)is_crafting_stone_sword;

    bool can_craft_iron_sword =
        inventory->iron >= 1
        && inventory->wood >= 1
        && inventory->stone >= 1
        && inventory->coal >= 1;
    bool is_crafting_iron_sword =
        action == CRAFTAX_ACTION_MAKE_IRON_SWORD
        && can_craft_iron_sword
        && is_at_furnace
        && is_at_crafting_table
        && inventory->sword < 3;
    inventory->wood -= 1 * (int32_t)is_crafting_iron_sword;
    inventory->iron -= 1 * (int32_t)is_crafting_iron_sword;
    inventory->stone -= 1 * (int32_t)is_crafting_iron_sword;
    inventory->coal -= 1 * (int32_t)is_crafting_iron_sword;
    inventory->sword =
        inventory->sword * (1 - (int32_t)is_crafting_iron_sword)
        + 3 * (int32_t)is_crafting_iron_sword;

    bool can_craft_diamond_sword =
        inventory->diamond >= 2 && inventory->wood >= 1;
    bool is_crafting_diamond_sword =
        action == CRAFTAX_ACTION_MAKE_DIAMOND_SWORD
        && can_craft_diamond_sword
        && is_at_crafting_table
        && inventory->sword < 4;
    inventory->wood -= 1 * (int32_t)is_crafting_diamond_sword;
    inventory->diamond -= 2 * (int32_t)is_crafting_diamond_sword;
    inventory->sword =
        inventory->sword * (1 - (int32_t)is_crafting_diamond_sword)
        + 4 * (int32_t)is_crafting_diamond_sword;

    int32_t armour_count = 0;
    int32_t iron_armour_index_to_craft =
        craftax_crafting_first_armour_below(inventory, 1, &armour_count);
    bool can_craft_iron_armour =
        armour_count > 0 && inventory->iron >= 3 && inventory->coal >= 3;
    bool is_crafting_iron_armour =
        action == CRAFTAX_ACTION_MAKE_IRON_ARMOUR
        && can_craft_iron_armour
        && is_at_crafting_table
        && is_at_furnace;
    inventory->iron -= 3 * (int32_t)is_crafting_iron_armour;
    inventory->coal -= 3 * (int32_t)is_crafting_iron_armour;
    inventory->armour[iron_armour_index_to_craft] =
        (int32_t)is_crafting_iron_armour * 1
        + (1 - (int32_t)is_crafting_iron_armour)
        * inventory->armour[iron_armour_index_to_craft];
    state->achievements[CRAFTAX_ACH_MAKE_IRON_ARMOUR] =
        state->achievements[CRAFTAX_ACH_MAKE_IRON_ARMOUR]
        || is_crafting_iron_armour;

    int32_t diamond_armour_count = 0;
    int32_t diamond_armour_index_to_craft =
        craftax_crafting_first_armour_below(inventory, 2, &diamond_armour_count);
    bool can_craft_diamond_armour =
        diamond_armour_count > 0 && inventory->diamond >= 3;
    bool is_crafting_diamond_armour =
        action == CRAFTAX_ACTION_MAKE_DIAMOND_ARMOUR
        && can_craft_diamond_armour
        && is_at_crafting_table;
    inventory->diamond -= 3 * (int32_t)is_crafting_diamond_armour;
    inventory->armour[diamond_armour_index_to_craft] =
        (int32_t)is_crafting_diamond_armour * 2
        + (1 - (int32_t)is_crafting_diamond_armour)
        * inventory->armour[diamond_armour_index_to_craft];
    state->achievements[CRAFTAX_ACH_MAKE_DIAMOND_ARMOUR] =
        state->achievements[CRAFTAX_ACH_MAKE_DIAMOND_ARMOUR]
        || is_crafting_diamond_armour;

    bool can_craft_arrow = inventory->stone >= 1 && inventory->wood >= 1;
    bool is_crafting_arrow =
        action == CRAFTAX_ACTION_MAKE_ARROW
        && can_craft_arrow
        && is_at_crafting_table
        && inventory->arrows < 99;
    inventory->wood -= 1 * (int32_t)is_crafting_arrow;
    inventory->stone -= 1 * (int32_t)is_crafting_arrow;
    inventory->arrows += 2 * (int32_t)is_crafting_arrow;

    bool can_craft_torch = inventory->coal >= 1 && inventory->wood >= 1;
    bool is_crafting_torch =
        action == CRAFTAX_ACTION_MAKE_TORCH
        && can_craft_torch
        && is_at_crafting_table
        && inventory->torches < 99;
    inventory->wood -= 1 * (int32_t)is_crafting_torch;
    inventory->coal -= 1 * (int32_t)is_crafting_torch;
    inventory->torches += 4 * (int32_t)is_crafting_torch;
}

static __device__ inline bool craftax_crafting_can_place_item(int32_t block) {
    switch (block) {
    case CRAFTAX_BLOCK_GRASS:
    case CRAFTAX_BLOCK_SAND:
    case CRAFTAX_BLOCK_PATH:
    case CRAFTAX_BLOCK_FIRE_GRASS:
    case CRAFTAX_BLOCK_ICE_GRASS:
        return true;
    default:
        return false;
    }
}

static __device__ inline float craftax_crafting_torch_light(int32_t row, int32_t col) {
    static const float torch_light_map[9][9] = {
        {0.0f, 0.0f, 0.10557288f, 0.17537886f, 0.19999999f, 0.17537886f, 0.10557288f, 0.0f, 0.0f},
        {0.0f, 0.15147191f, 0.27888972f, 0.36754447f, 0.39999998f, 0.36754447f, 0.27888972f, 0.15147191f, 0.0f},
        {0.10557288f, 0.27888972f, 0.43431455f, 0.55278647f, 0.6f, 0.55278647f, 0.43431455f, 0.27888972f, 0.10557288f},
        {0.17537886f, 0.36754447f, 0.55278647f, 0.71715724f, 0.8f, 0.71715724f, 0.55278647f, 0.36754447f, 0.17537886f},
        {0.19999999f, 0.39999998f, 0.6f, 0.8f, 1.0f, 0.8f, 0.6f, 0.39999998f, 0.19999999f},
        {0.17537886f, 0.36754447f, 0.55278647f, 0.71715724f, 0.8f, 0.71715724f, 0.55278647f, 0.36754447f, 0.17537886f},
        {0.10557288f, 0.27888972f, 0.43431455f, 0.55278647f, 0.6f, 0.55278647f, 0.43431455f, 0.27888972f, 0.10557288f},
        {0.0f, 0.15147191f, 0.27888972f, 0.36754447f, 0.39999998f, 0.36754447f, 0.27888972f, 0.15147191f, 0.0f},
        {0.0f, 0.0f, 0.10557288f, 0.17537886f, 0.19999999f, 0.17537886f, 0.10557288f, 0.0f, 0.0f},
    };
    return torch_light_map[row][col];
}

static __device__ inline void craftax_crafting_add_torch_light(
    CraftaxState* state,
    int32_t level,
    int32_t row,
    int32_t col
) {
    for (int32_t dr = -4; dr <= 4; dr++) {
        int32_t map_row = row + dr;
        if (map_row < 0 || map_row >= CRAFTAX_MAP_SIZE) {
            continue;
        }
        for (int32_t dc = -4; dc <= 4; dc++) {
            int32_t map_col = col + dc;
            if (map_col < 0 || map_col >= CRAFTAX_MAP_SIZE) {
                continue;
            }
            float light = state->light_map[level][map_row][map_col] * (1.0f / 255.0f)
                + craftax_crafting_torch_light(dr + 4, dc + 4);
            state->light_map[level][map_row][map_col] =
                (uint8_t)(craftax_step_minf32(craftax_step_maxf32(light, 0.0f), 1.0f) * 255.0f);
        }
    }
}

static __device__ inline void craftax_add_new_growing_plant_native(
    CraftaxState* state,
    const int32_t position[2],
    bool is_placing_sapling
) {
    int32_t plant_index = 0;
    int32_t empty_count = 0;
    for (int32_t i = 0; i < CRAFTAX_MAX_GROWING_PLANTS; i++) {
        bool is_empty = !state->growing_plants_mask[i];
        plant_index = (empty_count == 0 && is_empty) ? i : plant_index;
        empty_count += (int32_t)is_empty;
    }

    bool is_adding_plant = empty_count > 0 && is_placing_sapling;
    if (!is_adding_plant) {
        return;
    }

    state->growing_plants_positions[plant_index][0] = position[0];
    state->growing_plants_positions[plant_index][1] = position[1];
    state->growing_plants_age[plant_index] = 0;
    state->growing_plants_mask[plant_index] = true;
}

static __device__ inline void craftax_place_block_native(
    CraftaxState* state,
    int32_t action
) {
    int32_t direction[2];
    craftax_step_direction(state->player_direction, direction);

    int32_t row = state->player_position[0] + direction[0];
    int32_t col = state->player_position[1] + direction[1];
    bool in_bounds = row >= 0
        && row < CRAFTAX_MAP_SIZE
        && col >= 0
        && col < CRAFTAX_MAP_SIZE;
    bool in_mob = in_bounds && craftax_step_is_in_mob(state, row, col);
    if (!in_bounds || in_mob) {
        return;
    }

    int32_t level = craftax_step_jax_index(
        state->player_level,
        CRAFTAX_NUM_LEVELS
    );
    int32_t original_block = state->map[level][row][col];
    int32_t original_item = state->item_map[level][row][col];
    bool is_placement_on_solid_block_or_item =
        craftax_step_is_solid_block(original_block)
        || original_item != CRAFTAX_ITEM_NONE;

    CraftaxInventory* inventory = &state->inventory;

    bool is_placing_crafting_table =
        action == CRAFTAX_ACTION_PLACE_TABLE
        && !is_placement_on_solid_block_or_item
        && inventory->wood >= 2;
    if (is_placing_crafting_table) {
        craftax_set_map_block(state, level, row, col, CRAFTAX_BLOCK_CRAFTING_TABLE);
    }
    inventory->wood -= 2 * (int32_t)is_placing_crafting_table;
    state->achievements[CRAFTAX_ACH_PLACE_TABLE] =
        state->achievements[CRAFTAX_ACH_PLACE_TABLE]
        || is_placing_crafting_table;

    bool is_placing_furnace =
        action == CRAFTAX_ACTION_PLACE_FURNACE
        && !is_placement_on_solid_block_or_item
        && inventory->stone > 0;
    if (is_placing_furnace) {
        craftax_set_map_block(state, level, row, col, CRAFTAX_BLOCK_FURNACE);
    }
    inventory->stone -= 1 * (int32_t)is_placing_furnace;
    state->achievements[CRAFTAX_ACH_PLACE_FURNACE] =
        state->achievements[CRAFTAX_ACH_PLACE_FURNACE]
        || is_placing_furnace;

    bool is_placing_on_valid_stone_block =
        original_block == CRAFTAX_BLOCK_WATER
        || !is_placement_on_solid_block_or_item;
    bool is_placing_stone =
        action == CRAFTAX_ACTION_PLACE_STONE
        && is_placing_on_valid_stone_block
        && inventory->stone > 0;
    if (is_placing_stone) {
        craftax_set_map_block(state, level, row, col, CRAFTAX_BLOCK_STONE);
    }
    inventory->stone -= 1 * (int32_t)is_placing_stone;
    state->achievements[CRAFTAX_ACH_PLACE_STONE] =
        state->achievements[CRAFTAX_ACH_PLACE_STONE]
        || is_placing_stone;

    bool is_placing_on_valid_torch_block =
        craftax_crafting_can_place_item(original_block)
        && state->item_map[level][row][col] == CRAFTAX_ITEM_NONE;
    bool is_placing_torch =
        action == CRAFTAX_ACTION_PLACE_TORCH
        && is_placing_on_valid_torch_block
        && inventory->torches > 0;
    if (is_placing_torch) {
        state->item_map[level][row][col] = CRAFTAX_ITEM_TORCH;
        craftax_crafting_add_torch_light(state, level, row, col);
    }
    inventory->torches -= 1 * (int32_t)is_placing_torch;
    state->achievements[CRAFTAX_ACH_PLACE_TORCH] =
        state->achievements[CRAFTAX_ACH_PLACE_TORCH]
        || is_placing_torch;

    bool is_placing_sapling =
        action == CRAFTAX_ACTION_PLACE_PLANT
        && state->map[level][row][col] == CRAFTAX_BLOCK_GRASS
        && inventory->sapling > 0
        && state->item_map[level][row][col] == CRAFTAX_ITEM_NONE;
    if (is_placing_sapling) {
        int32_t position[2] = {row, col};
        craftax_set_map_block(state, level, row, col, CRAFTAX_BLOCK_PLANT);
        craftax_add_new_growing_plant_native(
            state,
            position,
            is_placing_sapling
        );
    }
    inventory->sapling -= 1 * (int32_t)is_placing_sapling;
    state->achievements[CRAFTAX_ACH_PLACE_PLANT] =
        state->achievements[CRAFTAX_ACH_PLACE_PLANT]
        || is_placing_sapling;
}

// ============================================================
// ===== step_medium.h =====
// ============================================================
// Standalone native ports of medium Craftax step subsystems.
//
// These helpers intentionally are not integrated into c_step yet. They mutate a
// full CraftaxState, or an Inventory plus read-only state context, so tests can
// compare each subsystem directly against the installed JAX implementation.



static __device__ inline CraftaxThreefryKey craftax_medium_next_random_key(
    CraftaxThreefryKey* rng
) {
    CraftaxThreefryKey draw;
    craftax_threefry_split(*rng, rng, &draw);
    return draw;
}

static __device__ inline int32_t craftax_medium_randint(
    CraftaxThreefryKey key,
    int32_t minval,
    int32_t maxval
) {
    return craftax_randint_i32_at(key, 0u, minval, maxval);
}

static __device__ inline int32_t craftax_medium_choice_weighted(
    CraftaxThreefryKey key,
    const float* weights,
    int32_t count
) {
    float total = 0.0f;
    for (int32_t i = 0; i < count; i++) {
        total += weights[i];
    }

    float draw = total * (1.0f - craftax_threefry_uniform_f32(key));
    float cumulative = 0.0f;
    for (int32_t i = 0; i < count; i++) {
        cumulative += weights[i];
        if (cumulative >= draw) {
            return i;
        }
    }
    return count - 1;
}

static __device__ inline int32_t craftax_medium_projectile_count(const CraftaxState* state) {
    int32_t level = craftax_step_jax_index(
        state->player_level,
        CRAFTAX_NUM_LEVELS
    );
    int32_t count = 0;
    for (int32_t i = 0; i < CRAFTAX_MAX_PLAYER_PROJECTILES; i++) {
        count += (int32_t)state->player_projectiles.mask[level][i];
    }
    return count;
}

static __device__ inline int32_t craftax_medium_first_projectile_slot(
    const CraftaxState* state
) {
    int32_t level = craftax_step_jax_index(
        state->player_level,
        CRAFTAX_NUM_LEVELS
    );
    for (int32_t i = 0; i < CRAFTAX_MAX_PLAYER_PROJECTILES; i++) {
        if (!state->player_projectiles.mask[level][i]) {
            return i;
        }
    }
    return 0;
}

static __device__ inline void craftax_medium_spawn_player_projectile(
    CraftaxState* state,
    bool is_spawning_projectile,
    const int32_t new_projectile_position[2],
    const int32_t direction[2],
    int32_t projectile_type
) {
    if (!is_spawning_projectile) {
        return;
    }

    int32_t level = craftax_step_jax_index(
        state->player_level,
        CRAFTAX_NUM_LEVELS
    );
    int32_t index = craftax_medium_first_projectile_slot(state);
    state->player_projectiles.position[level][index][0] = new_projectile_position[0];
    state->player_projectiles.position[level][index][1] = new_projectile_position[1];
    state->player_projectiles.mask[level][index] = true;
    state->player_projectiles.type_id[level][index] = projectile_type;
    state->player_projectile_directions[level][index][0] = direction[0];
    state->player_projectile_directions[level][index][1] = direction[1];
}

static __device__ inline int32_t craftax_medium_level_achievement(int32_t level) {
    switch (craftax_step_jax_index(level, CRAFTAX_NUM_LEVELS)) {
    case 1:
        return CRAFTAX_ACH_ENTER_DUNGEON;
    case 2:
        return CRAFTAX_ACH_ENTER_GNOMISH_MINES;
    case 3:
        return CRAFTAX_ACH_ENTER_SEWERS;
    case 4:
        return CRAFTAX_ACH_ENTER_VAULT;
    case 5:
        return CRAFTAX_ACH_ENTER_TROLL_MINES;
    case 6:
        return CRAFTAX_ACH_ENTER_FIRE_REALM;
    case 7:
        return CRAFTAX_ACH_ENTER_ICE_REALM;
    case 8:
        return CRAFTAX_ACH_ENTER_GRAVEYARD;
    default:
        return CRAFTAX_ACH_COLLECT_WOOD;
    }
}

static __device__ inline void craftax_shoot_projectile_native(
    CraftaxState* state,
    int32_t action
) {
    bool is_shooting_arrow = action == CRAFTAX_ACTION_SHOOT_ARROW
        && state->inventory.bow >= 1
        && state->inventory.arrows >= 1
        && craftax_medium_projectile_count(state) < CRAFTAX_MAX_PLAYER_PROJECTILES;

    int32_t direction[2];
    craftax_step_direction(state->player_direction, direction);
    craftax_medium_spawn_player_projectile(
        state,
        is_shooting_arrow,
        state->player_position,
        direction,
        CRAFTAX_PROJECTILE_ARROW2
    );

    state->achievements[CRAFTAX_ACH_FIRE_BOW] =
        state->achievements[CRAFTAX_ACH_FIRE_BOW] || is_shooting_arrow;
    state->inventory.arrows -= (int32_t)is_shooting_arrow;
}

static __device__ inline void craftax_cast_spell_native(
    CraftaxState* state,
    int32_t action
) {
    bool has_projectile_slot =
        craftax_medium_projectile_count(state) < CRAFTAX_MAX_PLAYER_PROJECTILES;
    bool has_mana = state->player_mana >= 2;
    bool is_casting_fireball = action == CRAFTAX_ACTION_CAST_FIREBALL
        && has_mana
        && has_projectile_slot
        && state->learned_spells[0];
    bool is_casting_iceball = action == CRAFTAX_ACTION_CAST_ICEBALL
        && has_mana
        && has_projectile_slot
        && state->learned_spells[1];
    bool is_casting_spell = is_casting_fireball || is_casting_iceball;

    int32_t projectile_type =
        (int32_t)is_casting_fireball * CRAFTAX_PROJECTILE_FIREBALL
        + (int32_t)is_casting_iceball * CRAFTAX_PROJECTILE_ICEBALL;

    int32_t direction[2];
    craftax_step_direction(state->player_direction, direction);
    craftax_medium_spawn_player_projectile(
        state,
        is_casting_spell,
        state->player_position,
        direction,
        projectile_type
    );

    if (is_casting_fireball) {
        state->achievements[CRAFTAX_ACH_CAST_FIREBALL] = true;
    }
    if (is_casting_iceball) {
        state->achievements[CRAFTAX_ACH_CAST_ICEBALL] = true;
    }
    state->player_mana -= (int32_t)is_casting_spell * 2;
}

static __device__ inline void craftax_enchant_native(
    CraftaxState* state,
    int32_t action,
    CraftaxThreefryKey rng
) {
    int32_t direction[2];
    craftax_step_direction(state->player_direction, direction);

    int32_t level = craftax_step_jax_index(
        state->player_level,
        CRAFTAX_NUM_LEVELS
    );
    int32_t target_row = craftax_step_jax_index(
        state->player_position[0] + direction[0],
        CRAFTAX_MAP_SIZE
    );
    int32_t target_col = craftax_step_jax_index(
        state->player_position[1] + direction[1],
        CRAFTAX_MAP_SIZE
    );
    int32_t target_block = state->map[level][target_row][target_col];

    bool is_fire_table = target_block == CRAFTAX_BLOCK_ENCHANTMENT_TABLE_FIRE;
    bool is_ice_table = target_block == CRAFTAX_BLOCK_ENCHANTMENT_TABLE_ICE;
    bool target_block_is_enchantment_table = is_fire_table || is_ice_table;
    int32_t enchantment_type = is_fire_table ? 1 : 2;
    int32_t num_gems = is_fire_table
        ? state->inventory.ruby
        : state->inventory.sapphire;

    bool could_enchant = state->player_mana >= 9
        && target_block_is_enchantment_table
        && num_gems >= 1;
    bool is_enchanting_bow = could_enchant
        && action == CRAFTAX_ACTION_ENCHANT_BOW
        && state->inventory.bow > 0;
    bool is_enchanting_sword = could_enchant
        && action == CRAFTAX_ACTION_ENCHANT_SWORD
        && state->inventory.sword > 0;

    int32_t armour_count = 0;
    for (int32_t i = 0; i < 4; i++) {
        armour_count += state->inventory.armour[i];
    }
    bool is_enchanting_armour = could_enchant
        && action == CRAFTAX_ACTION_ENCHANT_ARMOUR
        && armour_count > 0;

    CraftaxThreefryKey armour_key = craftax_medium_next_random_key(&rng);
    int32_t unenchanted_count = 0;
    for (int32_t i = 0; i < 4; i++) {
        unenchanted_count += (int32_t)(state->armour_enchantments[i] == 0);
    }

    float armour_targets[4];
    for (int32_t i = 0; i < 4; i++) {
        bool unenchanted = state->armour_enchantments[i] == 0;
        bool opposite_enchanted = state->armour_enchantments[i] != 0
            && state->armour_enchantments[i] != enchantment_type;
        armour_targets[i] = (unenchanted || (
            unenchanted_count == 0 && opposite_enchanted
        )) ? 1.0f : 0.0f;
    }
    int32_t armour_target = craftax_medium_choice_weighted(
        armour_key,
        armour_targets,
        4
    );

    bool is_enchanting = is_enchanting_sword
        || is_enchanting_bow
        || is_enchanting_armour;
    if (is_enchanting_sword) {
        state->sword_enchantment = enchantment_type;
        state->achievements[CRAFTAX_ACH_ENCHANT_SWORD] = true;
    }
    if (is_enchanting_bow) {
        state->bow_enchantment = enchantment_type;
    }
    if (is_enchanting_armour) {
        state->armour_enchantments[armour_target] = enchantment_type;
        state->achievements[CRAFTAX_ACH_ENCHANT_ARMOUR] = true;
    }

    state->inventory.sapphire -=
        (int32_t)is_enchanting * (int32_t)(enchantment_type == 2);
    state->inventory.ruby -=
        (int32_t)is_enchanting * (int32_t)(enchantment_type == 1);
    state->player_mana -= (int32_t)is_enchanting * 9;
}

static __device__ inline void craftax_change_floor_native(
    CraftaxState* state,
    int32_t action
) {
    int32_t level = craftax_step_jax_index(
        state->player_level,
        CRAFTAX_NUM_LEVELS
    );
    int32_t player_row = craftax_step_jax_index(
        state->player_position[0],
        CRAFTAX_MAP_SIZE
    );
    int32_t player_col = craftax_step_jax_index(
        state->player_position[1],
        CRAFTAX_MAP_SIZE
    );

    bool on_down_ladder =
        state->item_map[level][player_row][player_col] == CRAFTAX_ITEM_LADDER_DOWN;
    bool is_moving_down = action == CRAFTAX_ACTION_DESCEND
        && on_down_ladder
        && state->monsters_killed[level] >= CRAFTAX_MONSTERS_KILLED_TO_CLEAR_LEVEL
        && state->player_level < CRAFTAX_NUM_LEVELS - 1;

    bool on_up_ladder =
        state->item_map[level][player_row][player_col] == CRAFTAX_ITEM_LADDER_UP;
    bool is_moving_up = action == CRAFTAX_ACTION_ASCEND
        && on_up_ladder
        && state->player_level > 0;

    int32_t delta_floor = (int32_t)is_moving_down - (int32_t)is_moving_up;
    int32_t new_level = state->player_level + delta_floor;
    int32_t achievement = craftax_medium_level_achievement(new_level);
    bool new_floor = new_level != 0 && !state->achievements[achievement];

    if (is_moving_down) {
        int32_t ladder_level = craftax_step_jax_index(
            state->player_level + 1,
            CRAFTAX_NUM_LEVELS
        );
        craftax_ensure_floor_generated(state, ladder_level);
        state->player_position[0] = state->up_ladders[ladder_level][0];
        state->player_position[1] = state->up_ladders[ladder_level][1];
    } else if (is_moving_up) {
        int32_t ladder_level = craftax_step_jax_index(
            state->player_level - 1,
            CRAFTAX_NUM_LEVELS
        );
        craftax_ensure_floor_generated(state, ladder_level);
        state->player_position[0] = state->down_ladders[ladder_level][0];
        state->player_position[1] = state->down_ladders[ladder_level][1];
    }

    state->player_level = new_level;
    state->achievements[achievement] =
        state->achievements[achievement] || new_level != 0;
    state->player_xp += (int32_t)new_floor;
}

static __device__ inline void craftax_add_items_from_chest_native(
    const CraftaxState* state,
    CraftaxInventory* inventory,
    bool is_opening_chest,
    CraftaxThreefryKey rng
) {
    CraftaxThreefryKey draw_key;

    draw_key = craftax_medium_next_random_key(&rng);
    bool is_looting_wood = craftax_threefry_uniform_f32(draw_key) < 0.6f;
    draw_key = craftax_medium_next_random_key(&rng);
    int32_t wood_loot_amount =
        craftax_medium_randint(draw_key, 1, 6) * (int32_t)is_looting_wood;
    (void)wood_loot_amount;

    draw_key = craftax_medium_next_random_key(&rng);
    bool is_looting_torch = craftax_threefry_uniform_f32(draw_key) < 0.6f;
    draw_key = craftax_medium_next_random_key(&rng);
    int32_t torch_loot_amount =
        craftax_medium_randint(draw_key, 4, 8) * (int32_t)is_looting_torch;

    draw_key = craftax_medium_next_random_key(&rng);
    bool is_looting_ore = craftax_threefry_uniform_f32(draw_key) < 0.6f;
    draw_key = craftax_medium_next_random_key(&rng);
    float ore_weights[5] = {0.3f, 0.3f, 0.15f, 0.125f, 0.125f};
    int32_t ore_loot_id = craftax_medium_choice_weighted(
        draw_key,
        ore_weights,
        5
    );
    draw_key = craftax_medium_next_random_key(&rng);

    int32_t coal_loot_amount =
        craftax_medium_randint(draw_key, 1, 4)
        * (int32_t)(ore_loot_id == 0)
        * (int32_t)is_looting_ore;
    int32_t iron_loot_amount =
        craftax_medium_randint(draw_key, 1, 3)
        * (int32_t)(ore_loot_id == 1)
        * (int32_t)is_looting_ore;
    int32_t diamond_loot_amount =
        craftax_medium_randint(draw_key, 1, 2)
        * (int32_t)(ore_loot_id == 2)
        * (int32_t)is_looting_ore;
    int32_t sapphire_loot_amount =
        craftax_medium_randint(draw_key, 1, 2)
        * (int32_t)(ore_loot_id == 3)
        * (int32_t)is_looting_ore;
    int32_t ruby_loot_amount =
        craftax_medium_randint(draw_key, 1, 2)
        * (int32_t)(ore_loot_id == 4)
        * (int32_t)is_looting_ore;

    draw_key = craftax_medium_next_random_key(&rng);
    bool is_looting_potion = craftax_threefry_uniform_f32(draw_key) < 0.5f;
    draw_key = craftax_medium_next_random_key(&rng);
    int32_t potion_loot_index = craftax_medium_randint(draw_key, 0, 6);
    draw_key = craftax_medium_next_random_key(&rng);
    int32_t potion_loot_amount = craftax_medium_randint(draw_key, 1, 3);

    draw_key = craftax_medium_next_random_key(&rng);
    bool is_looting_arrows = craftax_threefry_uniform_f32(draw_key) < 0.25f;
    draw_key = craftax_medium_next_random_key(&rng);
    int32_t arrows_loot_amount =
        craftax_medium_randint(draw_key, 1, 5) * (int32_t)is_looting_arrows;

    draw_key = craftax_medium_next_random_key(&rng);
    bool is_looting_tool = craftax_threefry_uniform_f32(draw_key) < 0.2f;
    draw_key = craftax_medium_next_random_key(&rng);
    int32_t tool_id = craftax_medium_randint(draw_key, 0, 2);

    bool is_looting_pickaxe = is_looting_tool
        && tool_id == 0
        && is_opening_chest;
    draw_key = craftax_medium_next_random_key(&rng);
    float tool_weights[4] = {0.4f, 0.3f, 0.2f, 0.1f};
    int32_t pickaxe_loot_level = (
        craftax_medium_choice_weighted(draw_key, tool_weights, 4) + 1
    ) * (int32_t)is_looting_pickaxe;
    pickaxe_loot_level = craftax_step_maxi32(
        pickaxe_loot_level,
        inventory->pickaxe
    );
    int32_t new_pickaxe_level = is_looting_pickaxe
        ? pickaxe_loot_level
        : inventory->pickaxe;

    bool is_looting_sword = is_looting_tool
        && tool_id == 1
        && is_opening_chest;
    draw_key = craftax_medium_next_random_key(&rng);
    int32_t sword_loot_level = (
        craftax_medium_choice_weighted(draw_key, tool_weights, 4) + 1
    ) * (int32_t)is_looting_sword;
    sword_loot_level = craftax_step_maxi32(sword_loot_level, inventory->sword);
    int32_t new_sword_level = is_looting_sword
        ? sword_loot_level
        : inventory->sword;

    int32_t level = craftax_step_jax_index(
        state->player_level,
        CRAFTAX_NUM_LEVELS
    );
    bool is_looting_bow = is_opening_chest
        && state->player_level == 1
        && !state->chests_opened[level];
    int32_t new_bow_level = is_looting_bow ? 1 : inventory->bow;

    bool is_looting_book = !state->chests_opened[level]
        && (state->player_level == 3 || state->player_level == 4);

    int32_t opening = (int32_t)is_opening_chest;
    inventory->torches += torch_loot_amount * opening;
    inventory->coal += coal_loot_amount * opening;
    inventory->iron += iron_loot_amount * opening;
    inventory->diamond += diamond_loot_amount * opening;
    inventory->sapphire += sapphire_loot_amount * opening;
    inventory->ruby += ruby_loot_amount * opening;
    inventory->arrows += arrows_loot_amount * opening;
    inventory->pickaxe = new_pickaxe_level;
    inventory->sword = new_sword_level;
    inventory->potions[potion_loot_index] +=
        potion_loot_amount * (int32_t)is_looting_potion * opening;
    inventory->bow = new_bow_level;
    inventory->books += (int32_t)is_looting_book * opening;
}

// ============================================================
// ===== step_do_action.h =====
// ============================================================
// Standalone native port of Craftax do_action.
//
// This helper intentionally is not integrated into c_step yet. It mutates a
// full CraftaxState in place so tests can compare the subsystem directly
// against the installed JAX implementation.



#define CRAFTAX_DO_ACTION_BOSS_FIGHT_SPAWN_TURNS 7

static __device__ inline float craftax_do_action_mob_defense(
    int32_t type_id,
    int32_t mob_class_index,
    int32_t damage_index
) {
    static const float defenses[8][4][3] = {
        {
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
        },
        {
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
        },
        {
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
        },
        {
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
        },
        {
            {0.0f, 0.0f, 0.0f},
            {0.5f, 0.0f, 0.0f},
            {0.5f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
        },
        {
            {0.0f, 0.0f, 0.0f},
            {0.2f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
        },
        {
            {0.0f, 0.0f, 0.0f},
            {0.9f, 1.0f, 0.0f},
            {0.9f, 1.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
        },
        {
            {0.0f, 0.0f, 0.0f},
            {0.9f, 0.0f, 1.0f},
            {0.9f, 0.0f, 1.0f},
            {0.0f, 0.0f, 0.0f},
        },
    };

    int32_t type_index = craftax_step_jax_index(type_id, 8);
    int32_t class_index = craftax_step_jax_index(mob_class_index, 4);
    int32_t component = craftax_step_jax_index(damage_index, 3);
    return defenses[type_index][class_index][component];
}

static __device__ inline int32_t craftax_do_action_mob_achievement(
    int32_t mob_class_index,
    int32_t type_id
) {
    static const int32_t achievements[3][8] = {
        {
            CRAFTAX_ACH_EAT_COW,
            CRAFTAX_ACH_EAT_BAT,
            CRAFTAX_ACH_EAT_SNAIL,
            0,
            0,
            0,
            0,
            0,
        },
        {
            CRAFTAX_ACH_DEFEAT_ZOMBIE,
            CRAFTAX_ACH_DEFEAT_GNOME_WARRIOR,
            CRAFTAX_ACH_DEFEAT_ORC_SOLIDER,
            CRAFTAX_ACH_DEFEAT_LIZARD,
            CRAFTAX_ACH_DEFEAT_KNIGHT,
            CRAFTAX_ACH_DEFEAT_TROLL,
            CRAFTAX_ACH_DEFEAT_PIGMAN,
            CRAFTAX_ACH_DEFEAT_FROST_TROLL,
        },
        {
            CRAFTAX_ACH_DEFEAT_SKELETON,
            CRAFTAX_ACH_DEFEAT_GNOME_ARCHER,
            CRAFTAX_ACH_DEFEAT_ORC_MAGE,
            CRAFTAX_ACH_DEFEAT_KOBOLD,
            CRAFTAX_ACH_DEFEAT_ARCHER,
            CRAFTAX_ACH_DEFEAT_DEEP_THING,
            CRAFTAX_ACH_DEFEAT_FIRE_ELEMENTAL,
            CRAFTAX_ACH_DEFEAT_ICE_ELEMENTAL,
        },
    };

    int32_t class_index = craftax_step_jax_index(mob_class_index, 3);
    int32_t type_index = craftax_step_jax_index(type_id, 8);
    return achievements[class_index][type_index];
}

static __device__ inline void craftax_do_action_player_damage_vector(
    const CraftaxState* state,
    float damage_vector[3]
) {
    static const float physical_damages[5] = {1.0f, 2.0f, 3.0f, 5.0f, 8.0f};

    int32_t sword_index = craftax_step_jax_index(state->inventory.sword, 5);
    float physical_damage = physical_damages[sword_index];
    float fire_damage =
        physical_damage * (float)(state->sword_enchantment == 1) * 0.5f;
    float ice_damage =
        physical_damage * (float)(state->sword_enchantment == 2) * 0.5f;

    physical_damage *= 1.0f + 0.25f * (float)(state->player_strength - 1);
    fire_damage *= 1.0f + 0.05f * (float)(state->player_intelligence - 1);
    ice_damage *= 1.0f + 0.05f * (float)(state->player_intelligence - 1);

    damage_vector[0] = physical_damage;
    damage_vector[1] = fire_damage;
    damage_vector[2] = ice_damage;
}

static __device__ inline float craftax_do_action_damage_done(
    const float damage_vector[3],
    int32_t type_id,
    int32_t mob_class_index
) {
    float damage = 0.0f;
    for (int32_t i = 0; i < 3; i++) {
        float defense = craftax_do_action_mob_defense(
            type_id,
            mob_class_index,
            i
        );
        damage += (1.0f - defense) * damage_vector[i];
    }
    return damage;
}

static __device__ inline void craftax_do_action_refresh_mobs3_masks(CraftaxMobs3* mobs) {
    for (int32_t level = 0; level < CRAFTAX_NUM_LEVELS; level++) {
        for (int32_t i = 0; i < 3; i++) {
            mobs->mask[level][i] =
                mobs->mask[level][i] && mobs->health[level][i] > 0.0f;
        }
    }
}

static __device__ inline void craftax_do_action_refresh_mobs2_masks(CraftaxMobs2* mobs) {
    for (int32_t level = 0; level < CRAFTAX_NUM_LEVELS; level++) {
        for (int32_t i = 0; i < 2; i++) {
            mobs->mask[level][i] =
                mobs->mask[level][i] && mobs->health[level][i] > 0.0f;
        }
    }
}

static __device__ inline void craftax_do_action_attack_mobs3(
    CraftaxState* state,
    CraftaxMobs3* mobs,
    int32_t row,
    int32_t col,
    const float damage_vector[3],
    bool can_get_achievement,
    int32_t mob_class_index,
    bool* did_kill_mob,
    bool* is_attacking_mob
) {
    int32_t level = craftax_step_jax_index(
        state->player_level,
        CRAFTAX_NUM_LEVELS
    );
    bool is_attacking_array[3];
    *is_attacking_mob = false;
    int32_t target_mob_index = 0;

    for (int32_t i = 0; i < 3; i++) {
        bool in_mob = mobs->position[level][i][0] == row
            && mobs->position[level][i][1] == col;
        is_attacking_array[i] = in_mob && mobs->mask[level][i];
        if (is_attacking_array[i] && !*is_attacking_mob) {
            target_mob_index = i;
        }
        *is_attacking_mob = *is_attacking_mob || is_attacking_array[i];
    }

    int32_t target_type_id = mobs->type_id[level][target_mob_index];
    float damage = craftax_do_action_damage_done(
        damage_vector,
        target_type_id,
        mob_class_index
    );
    mobs->health[level][target_mob_index] -=
        damage * (float)(int32_t)(*is_attacking_mob);

    bool old_mask = mobs->mask[level][target_mob_index];
    craftax_do_action_refresh_mobs3_masks(mobs);
    *did_kill_mob = old_mask && !mobs->mask[level][target_mob_index];

    int32_t achievement_for_kill = craftax_do_action_mob_achievement(
        mob_class_index,
        target_type_id
    );
    bool unlock = *did_kill_mob && can_get_achievement;
    state->achievements[achievement_for_kill] =
        state->achievements[achievement_for_kill] || unlock;
}

static __device__ inline void craftax_do_action_attack_mobs2(
    CraftaxState* state,
    CraftaxMobs2* mobs,
    int32_t row,
    int32_t col,
    const float damage_vector[3],
    bool can_get_achievement,
    int32_t mob_class_index,
    bool* did_kill_mob,
    bool* is_attacking_mob
) {
    int32_t level = craftax_step_jax_index(
        state->player_level,
        CRAFTAX_NUM_LEVELS
    );
    bool is_attacking_array[2];
    *is_attacking_mob = false;
    int32_t target_mob_index = 0;

    for (int32_t i = 0; i < 2; i++) {
        bool in_mob = mobs->position[level][i][0] == row
            && mobs->position[level][i][1] == col;
        is_attacking_array[i] = in_mob && mobs->mask[level][i];
        if (is_attacking_array[i] && !*is_attacking_mob) {
            target_mob_index = i;
        }
        *is_attacking_mob = *is_attacking_mob || is_attacking_array[i];
    }

    int32_t target_type_id = mobs->type_id[level][target_mob_index];
    float damage = craftax_do_action_damage_done(
        damage_vector,
        target_type_id,
        mob_class_index
    );
    mobs->health[level][target_mob_index] -=
        damage * (float)(int32_t)(*is_attacking_mob);

    bool old_mask = mobs->mask[level][target_mob_index];
    craftax_do_action_refresh_mobs2_masks(mobs);
    *did_kill_mob = old_mask && !mobs->mask[level][target_mob_index];

    int32_t achievement_for_kill = craftax_do_action_mob_achievement(
        mob_class_index,
        target_type_id
    );
    bool unlock = *did_kill_mob && can_get_achievement;
    state->achievements[achievement_for_kill] =
        state->achievements[achievement_for_kill] || unlock;
}

static __device__ inline bool craftax_do_action_update_index(
    int32_t index,
    int32_t size,
    int32_t* mapped_index
) {
    if (index < -size || index >= size) {
        return false;
    }
    *mapped_index = index < 0 ? index + size : index;
    return true;
}

static __device__ inline void craftax_do_action_update_mob_map(
    CraftaxState* state,
    int32_t row,
    int32_t col,
    bool did_kill_mob
) {
    int32_t update_row;
    int32_t update_col;
    if (!craftax_do_action_update_index(row, CRAFTAX_MAP_SIZE, &update_row)
        || !craftax_do_action_update_index(col, CRAFTAX_MAP_SIZE, &update_col)) {
        return;
    }

    int32_t level = craftax_step_jax_index(
        state->player_level,
        CRAFTAX_NUM_LEVELS
    );
    int32_t read_row = craftax_step_jax_index(row, CRAFTAX_MAP_SIZE);
    int32_t read_col = craftax_step_jax_index(col, CRAFTAX_MAP_SIZE);
    bool old_value = (state->mob_bits[level][read_row] >> read_col) & 1ULL;
    bool new_value = old_value && !did_kill_mob;
    if (new_value) {
        state->mob_bits[level][update_row] |= (1ULL << update_col);
    } else {
        state->mob_bits[level][update_row] &= ~(1ULL << update_col);
    }
}

static __device__ inline void craftax_do_action_attack_mob(
    CraftaxState* state,
    int32_t row,
    int32_t col,
    bool can_eat,
    bool* did_attack_mob,
    bool* did_kill_mob
) {
    float damage_vector[3];
    craftax_do_action_player_damage_vector(state, damage_vector);

    bool did_kill_melee_mob = false;
    bool is_attacking_melee_mob = false;
    craftax_do_action_attack_mobs3(
        state,
        &state->melee_mobs,
        row,
        col,
        damage_vector,
        true,
        1,
        &did_kill_melee_mob,
        &is_attacking_melee_mob
    );

    bool did_kill_passive_mob = false;
    bool is_attacking_passive_mob = false;
    craftax_do_action_attack_mobs3(
        state,
        &state->passive_mobs,
        row,
        col,
        damage_vector,
        can_eat,
        0,
        &did_kill_passive_mob,
        &is_attacking_passive_mob
    );

    if (did_kill_passive_mob && can_eat) {
        state->player_food = craftax_step_mini32(
            craftax_step_get_max_food(state),
            state->player_food + 6
        );
        state->player_hunger = 0.0f;
    }

    bool did_kill_ranged_mob = false;
    bool is_attacking_ranged_mob = false;
    craftax_do_action_attack_mobs2(
        state,
        &state->ranged_mobs,
        row,
        col,
        damage_vector,
        true,
        2,
        &did_kill_ranged_mob,
        &is_attacking_ranged_mob
    );

    *did_attack_mob = is_attacking_melee_mob
        || is_attacking_passive_mob
        || is_attacking_ranged_mob;
    bool did_kill_monster = did_kill_melee_mob || did_kill_ranged_mob;
    *did_kill_mob = did_kill_monster || did_kill_passive_mob;

    craftax_do_action_update_mob_map(state, row, col, *did_kill_mob);

    int32_t level = craftax_step_jax_index(
        state->player_level,
        CRAFTAX_NUM_LEVELS
    );
    state->monsters_killed[level] += (int32_t)did_kill_monster;
}

static __device__ inline bool craftax_do_action_in_bounds(int32_t row, int32_t col) {
    return row >= 0
        && row < CRAFTAX_MAP_SIZE
        && col >= 0
        && col < CRAFTAX_MAP_SIZE;
}

static __device__ inline bool craftax_do_action_boss_vulnerable(
    const CraftaxState* state
) {
    int32_t level = craftax_step_jax_index(
        state->player_level,
        CRAFTAX_NUM_LEVELS
    );
    int32_t melee_count = 0;
    int32_t ranged_count = 0;
    for (int32_t i = 0; i < CRAFTAX_MAX_MELEE_MOBS; i++) {
        melee_count += (int32_t)state->melee_mobs.mask[level][i];
    }
    for (int32_t i = 0; i < CRAFTAX_MAX_RANGED_MOBS; i++) {
        ranged_count += (int32_t)state->ranged_mobs.mask[level][i];
    }
    return melee_count == 0
        && ranged_count == 0
        && state->boss_timesteps_to_spawn_this_round <= 0;
}

static __device__ inline void craftax_do_action_update_plants_with_eat(
    CraftaxState* state,
    int32_t row,
    int32_t col
) {
    int32_t plant_index = 0;
    bool found = false;
    for (int32_t i = 0; i < CRAFTAX_MAX_GROWING_PLANTS; i++) {
        bool is_plant = state->growing_plants_positions[i][0] == row
            && state->growing_plants_positions[i][1] == col;
        if (is_plant && !found) {
            plant_index = i;
            found = true;
        }
    }
    state->growing_plants_age[plant_index] = 0;
}

static __device__ inline void craftax_do_action_native(
    CraftaxState* state,
    int32_t action,
    CraftaxThreefryKey rng
) {
    if (action != CRAFTAX_ACTION_DO) {
        return;
    }

    int32_t direction[2];
    craftax_step_direction(state->player_direction, direction);
    int32_t target_row = state->player_position[0] + direction[0];
    int32_t target_col = state->player_position[1] + direction[1];

    bool did_attack_mob = false;
    bool did_kill_mob = false;
    craftax_do_action_attack_mob(
        state,
        target_row,
        target_col,
        true,
        &did_attack_mob,
        &did_kill_mob
    );
    (void)did_kill_mob;

    int32_t level = craftax_step_jax_index(
        state->player_level,
        CRAFTAX_NUM_LEVELS
    );
    int32_t read_row = craftax_step_jax_index(target_row, CRAFTAX_MAP_SIZE);
    int32_t read_col = craftax_step_jax_index(target_col, CRAFTAX_MAP_SIZE);
    int32_t target_block = state->map[level][read_row][read_col];

    CraftaxThreefryKey sapling_key = craftax_medium_next_random_key(&rng);
    CraftaxThreefryKey chest_key = craftax_medium_next_random_key(&rng);

    bool is_opening_chest = target_block == CRAFTAX_BLOCK_CHEST;
    bool is_damaging_boss = target_block == CRAFTAX_BLOCK_NECROMANCER
        && craftax_do_action_boss_vulnerable(state)
        && craftax_step_is_fighting_boss(state);

    bool action_block_in_bounds =
        craftax_do_action_in_bounds(target_row, target_col) && !did_attack_mob;

    if (action_block_in_bounds) {
        bool is_block_tree = target_block == CRAFTAX_BLOCK_TREE;
        bool is_block_fire_tree = target_block == CRAFTAX_BLOCK_FIRE_TREE;
        bool is_block_ice_shrub = target_block == CRAFTAX_BLOCK_ICE_SHRUB;
        bool is_mining_tree =
            is_block_tree || is_block_fire_tree || is_block_ice_shrub;
        if (is_mining_tree) {
            int32_t replacement = is_block_tree
                ? CRAFTAX_BLOCK_GRASS
                : (is_block_fire_tree
                    ? CRAFTAX_BLOCK_FIRE_GRASS
                    : CRAFTAX_BLOCK_ICE_GRASS);
            craftax_set_map_block(state, level, target_row, target_col, replacement);
            state->inventory.wood += 1;
        }

        bool is_mining_stone = target_block == CRAFTAX_BLOCK_STONE
            && state->inventory.pickaxe >= 1;
        if (is_mining_stone) {
            craftax_set_map_block(state, level, target_row, target_col, CRAFTAX_BLOCK_PATH);
            state->inventory.stone += 1;
        }

        if (target_block == CRAFTAX_BLOCK_FURNACE) {
            craftax_set_map_block(state, level, target_row, target_col, CRAFTAX_BLOCK_PATH);
        }

        if (target_block == CRAFTAX_BLOCK_CRAFTING_TABLE) {
            craftax_set_map_block(state, level, target_row, target_col, CRAFTAX_BLOCK_PATH);
        }

        bool is_mining_coal = target_block == CRAFTAX_BLOCK_COAL
            && state->inventory.pickaxe >= 1;
        if (is_mining_coal) {
            craftax_set_map_block(state, level, target_row, target_col, CRAFTAX_BLOCK_PATH);
            state->inventory.coal += 1;
        }

        bool is_mining_iron = target_block == CRAFTAX_BLOCK_IRON
            && state->inventory.pickaxe >= 2;
        if (is_mining_iron) {
            craftax_set_map_block(state, level, target_row, target_col, CRAFTAX_BLOCK_PATH);
            state->inventory.iron += 1;
        }

        bool is_mining_diamond = target_block == CRAFTAX_BLOCK_DIAMOND
            && state->inventory.pickaxe >= 3;
        if (is_mining_diamond) {
            craftax_set_map_block(state, level, target_row, target_col, CRAFTAX_BLOCK_PATH);
            state->inventory.diamond += 1;
        }

        bool is_mining_sapphire = target_block == CRAFTAX_BLOCK_SAPPHIRE
            && state->inventory.pickaxe >= 4;
        if (is_mining_sapphire) {
            craftax_set_map_block(state, level, target_row, target_col, CRAFTAX_BLOCK_PATH);
            state->inventory.sapphire += 1;
        }

        bool is_mining_ruby = target_block == CRAFTAX_BLOCK_RUBY
            && state->inventory.pickaxe >= 4;
        if (is_mining_ruby) {
            craftax_set_map_block(state, level, target_row, target_col, CRAFTAX_BLOCK_PATH);
            state->inventory.ruby += 1;
        }

        bool is_mining_sapling = target_block == CRAFTAX_BLOCK_GRASS
            && craftax_threefry_uniform_f32(sapling_key) < 0.1f;
        state->inventory.sapling += (int32_t)is_mining_sapling;

        bool is_drinking_water = target_block == CRAFTAX_BLOCK_WATER
            || target_block == CRAFTAX_BLOCK_FOUNTAIN;
        if (is_drinking_water) {
            state->player_drink = craftax_step_mini32(
                craftax_step_get_max_drink(state),
                state->player_drink + 1
            );
            state->player_thirst = 0.0f;
            state->achievements[CRAFTAX_ACH_COLLECT_DRINK] = true;
        }

        bool is_eating_plant = target_block == CRAFTAX_BLOCK_RIPE_PLANT;
        if (is_eating_plant) {
            craftax_set_map_block(state, level, target_row, target_col, CRAFTAX_BLOCK_PLANT);
            state->player_food = craftax_step_mini32(
                craftax_step_get_max_food(state),
                state->player_food + 4
            );
            state->player_hunger = 0.0f;
            state->achievements[CRAFTAX_ACH_EAT_PLANT] = true;
            craftax_do_action_update_plants_with_eat(
                state,
                target_row,
                target_col
            );
        }

        bool is_mining_stalagmite = target_block == CRAFTAX_BLOCK_STALAGMITE
            && state->inventory.pickaxe >= 1;
        if (is_mining_stalagmite) {
            craftax_set_map_block(state, level, target_row, target_col, CRAFTAX_BLOCK_PATH);
            state->inventory.stone += 1;
        }

        if (is_opening_chest) {
            craftax_set_map_block(state, level, target_row, target_col, CRAFTAX_BLOCK_PATH);
            craftax_add_items_from_chest_native(
                state,
                &state->inventory,
                true,
                chest_key
            );
            state->achievements[CRAFTAX_ACH_OPEN_CHEST] = true;
        }

        if (is_damaging_boss) {
            state->achievements[CRAFTAX_ACH_DAMAGE_NECROMANCER] = true;
        }
    }

    state->chests_opened[level] =
        state->chests_opened[level] || is_opening_chest;

    state->boss_progress += (int32_t)is_damaging_boss;
    if (is_damaging_boss) {
        state->boss_timesteps_to_spawn_this_round =
            CRAFTAX_DO_ACTION_BOSS_FIGHT_SPAWN_TURNS;
    }
}

// ============================================================
// ===== step_update_mobs.h =====
// ============================================================
// Standalone native port of Craftax update_mobs.
//
// This helper intentionally is not integrated into c_step yet. It mutates a
// full CraftaxState in place so tests can compare the subsystem directly
// against the installed JAX implementation.



#define CRAFTAX_UPDATE_BOSS_FIGHT_EXTRA_DAMAGE 0.5f

static __device__ inline CraftaxThreefryKey craftax_update_mobs_next_random_key(
    CraftaxThreefryKey* rng
) {
    CraftaxThreefryKey draw;
    craftax_threefry_split(*rng, rng, &draw);
    return draw;
}

static __device__ inline bool craftax_update_mobs_scatter_index(
    int32_t index,
    int32_t size,
    int32_t* mapped_index
) {
    if (index < -size || index >= size) {
        return false;
    }
    *mapped_index = index < 0 ? index + size : index;
    return true;
}

static __device__ inline bool craftax_update_mobs_in_bounds(
    int32_t row,
    int32_t col
) {
    return row >= 0
        && row < CRAFTAX_MAP_SIZE
        && col >= 0
        && col < CRAFTAX_MAP_SIZE;
}

static __device__ inline int32_t craftax_update_mobs_read_block(
    const CraftaxState* state,
    int32_t level,
    int32_t row,
    int32_t col
) {
    int32_t map_level = craftax_step_jax_index(level, CRAFTAX_NUM_LEVELS);
    int32_t map_row = craftax_step_jax_index(row, CRAFTAX_MAP_SIZE);
    int32_t map_col = craftax_step_jax_index(col, CRAFTAX_MAP_SIZE);
    return state->map[map_level][map_row][map_col];
}

static __device__ inline void craftax_update_mobs_set_block(
    CraftaxState* state,
    int32_t level,
    int32_t row,
    int32_t col,
    int32_t block
) {
    int32_t map_level;
    int32_t map_row;
    int32_t map_col;
    if (!craftax_update_mobs_scatter_index(
            level,
            CRAFTAX_NUM_LEVELS,
            &map_level
        )
        || !craftax_update_mobs_scatter_index(
            row,
            CRAFTAX_MAP_SIZE,
            &map_row
        )
        || !craftax_update_mobs_scatter_index(
            col,
            CRAFTAX_MAP_SIZE,
            &map_col
        )) {
        return;
    }
    craftax_set_map_block(state, map_level, map_row, map_col, block);
}

static __device__ inline bool craftax_update_mobs_read_mob_map(
    const CraftaxState* state,
    int32_t level,
    int32_t row,
    int32_t col
) {
    int32_t map_level = craftax_step_jax_index(level, CRAFTAX_NUM_LEVELS);
    int32_t map_row = craftax_step_jax_index(row, CRAFTAX_MAP_SIZE);
    int32_t map_col = craftax_step_jax_index(col, CRAFTAX_MAP_SIZE);
    return (state->mob_bits[map_level][map_row] >> map_col) & 1ULL;
}

static __device__ inline void craftax_update_mobs_set_mob_map(
    CraftaxState* state,
    int32_t level,
    int32_t row,
    int32_t col,
    bool value
) {
    int32_t map_level;
    int32_t map_row;
    int32_t map_col;
    if (!craftax_update_mobs_scatter_index(
            level,
            CRAFTAX_NUM_LEVELS,
            &map_level
        )
        || !craftax_update_mobs_scatter_index(
            row,
            CRAFTAX_MAP_SIZE,
            &map_row
        )
        || !craftax_update_mobs_scatter_index(
            col,
            CRAFTAX_MAP_SIZE,
            &map_col
        )) {
        return;
    }
    if (value) {
        state->mob_bits[map_level][map_row] |= (1ULL << map_col);
    } else {
        state->mob_bits[map_level][map_row] &= ~(1ULL << map_col);
    }
}

static __device__ inline void craftax_update_mobs_clear_old_map_entry(
    CraftaxState* state,
    int32_t level,
    int32_t row,
    int32_t col,
    bool old_mask
) {
    bool old_value = craftax_update_mobs_read_mob_map(state, level, row, col);
    craftax_update_mobs_set_mob_map(
        state,
        level,
        row,
        col,
        old_value && !old_mask
    );
}

static __device__ inline void craftax_update_mobs_enter_new_map_entry(
    CraftaxState* state,
    int32_t level,
    int32_t row,
    int32_t col,
    bool new_mask
) {
    bool old_value = craftax_update_mobs_read_mob_map(state, level, row, col);
    craftax_update_mobs_set_mob_map(
        state,
        level,
        row,
        col,
        old_value || new_mask
    );
}

static __device__ inline void craftax_update_mobs_damage_vector(
    int32_t type_id,
    int32_t mob_class_index,
    float damage[3]
) {
    static const float damages[CRAFTAX_NUM_MOB_TYPES][4][3] = {
        {
            {0.0f, 0.0f, 0.0f},
            {2.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {2.0f, 0.0f, 0.0f},
        },
        {
            {0.0f, 0.0f, 0.0f},
            {4.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {4.0f, 0.0f, 0.0f},
        },
        {
            {0.0f, 0.0f, 0.0f},
            {3.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {0.0f, 3.0f, 0.0f},
        },
        {
            {0.0f, 0.0f, 0.0f},
            {5.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 3.0f},
        },
        {
            {0.0f, 0.0f, 0.0f},
            {6.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {5.0f, 0.0f, 0.0f},
        },
        {
            {0.0f, 0.0f, 0.0f},
            {6.0f, 1.0f, 1.0f},
            {0.0f, 0.0f, 0.0f},
            {4.0f, 3.0f, 3.0f},
        },
        {
            {0.0f, 0.0f, 0.0f},
            {3.0f, 5.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {3.0f, 5.0f, 0.0f},
        },
        {
            {0.0f, 0.0f, 0.0f},
            {4.0f, 0.0f, 5.0f},
            {0.0f, 0.0f, 0.0f},
            {4.0f, 0.0f, 5.0f},
        },
    };

    int32_t type_index = craftax_step_jax_index(
        type_id,
        CRAFTAX_NUM_MOB_TYPES
    );
    int32_t class_index = craftax_step_jax_index(mob_class_index, 4);
    for (int32_t i = 0; i < 3; i++) {
        damage[i] = damages[type_index][class_index][i];
    }
}

static __device__ inline void craftax_update_mobs_collision_map(
    int32_t type_id,
    int32_t mob_class_index,
    bool collision[3]
) {
    static const bool collisions[CRAFTAX_NUM_MOB_TYPES][4][3] = {
        {
            {false, true, true},
            {false, true, true},
            {false, true, true},
            {false, false, false},
        },
        {
            {false, false, false},
            {false, true, true},
            {false, true, true},
            {false, false, false},
        },
        {
            {false, true, true},
            {false, true, true},
            {false, true, true},
            {false, false, false},
        },
        {
            {false, true, true},
            {false, false, true},
            {false, true, true},
            {false, false, false},
        },
        {
            {false, true, true},
            {false, true, true},
            {false, true, true},
            {false, false, false},
        },
        {
            {false, true, true},
            {false, true, true},
            {true, false, true},
            {false, false, false},
        },
        {
            {false, true, true},
            {false, true, true},
            {false, false, false},
            {false, false, false},
        },
        {
            {false, true, true},
            {false, true, true},
            {false, false, false},
            {false, false, false},
        },
    };

    int32_t type_index = craftax_step_jax_index(
        type_id,
        CRAFTAX_NUM_MOB_TYPES
    );
    int32_t class_index = craftax_step_jax_index(mob_class_index, 4);
    for (int32_t i = 0; i < 3; i++) {
        collision[i] = collisions[type_index][class_index][i];
    }
}

static __device__ inline int32_t craftax_update_mobs_projectile_type_for_ranged(
    int32_t ranged_type
) {
    static const int32_t mapping[CRAFTAX_NUM_MOB_TYPES] = {
        CRAFTAX_PROJECTILE_ARROW,
        CRAFTAX_PROJECTILE_ARROW,
        CRAFTAX_PROJECTILE_FIREBALL,
        CRAFTAX_PROJECTILE_DAGGER,
        CRAFTAX_PROJECTILE_ARROW2,
        CRAFTAX_PROJECTILE_SLIMEBALL,
        CRAFTAX_PROJECTILE_FIREBALL2,
        CRAFTAX_PROJECTILE_ICEBALL2,
    };
    int32_t type_index = craftax_step_jax_index(
        ranged_type,
        CRAFTAX_NUM_MOB_TYPES
    );
    return mapping[type_index];
}

static __device__ inline void craftax_update_mobs_direction_choice(
    CraftaxThreefryKey key,
    int32_t count,
    int32_t direction[2]
) {
    int32_t choice = craftax_medium_randint(key, 0, count);
    direction[0] = 0;
    direction[1] = 0;
    if (choice == 0) {
        direction[1] = -1;
    } else if (choice == 1) {
        direction[1] = 1;
    } else if (choice == 2) {
        direction[0] = -1;
    } else if (choice == 3) {
        direction[0] = 1;
    }
}

static __device__ inline int32_t craftax_update_mobs_abs_i32(int32_t value) {
    return value < 0 ? -value : value;
}

static __device__ inline int32_t craftax_update_mobs_sign_i32(int32_t value) {
    if (value < 0) {
        return -1;
    }
    return value > 0 ? 1 : 0;
}

static __device__ inline int32_t craftax_update_mobs_player_axis_choice(
    CraftaxThreefryKey key,
    int32_t distance_row,
    int32_t distance_col
) {
    int32_t max_distance = distance_row > distance_col
        ? distance_row
        : distance_col;
    int32_t total_distance = distance_row + distance_col;
    if (total_distance == 0) {
        return 1;
    }

    float weights[2] = {
        (distance_row == max_distance) ? 1.0f / (float)total_distance : 0.0f,
        (distance_col == max_distance) ? 1.0f / (float)total_distance : 0.0f,
    };
    return craftax_medium_choice_weighted(key, weights, 2);
}

static __device__ inline bool craftax_update_mobs_valid_position(
    const CraftaxState* state,
    int32_t row,
    int32_t col,
    const bool collision[3]
) {
    int32_t level = craftax_step_jax_index(
        state->player_level,
        CRAFTAX_NUM_LEVELS
    );
    bool pos_in_bounds = craftax_update_mobs_in_bounds(row, col);
    int32_t block = craftax_update_mobs_read_block(state, level, row, col);
    bool in_solid_block = craftax_step_is_solid_block(block);
    bool in_mob = craftax_step_is_in_mob(state, row, col);
    bool in_lava = block == CRAFTAX_BLOCK_LAVA;
    bool in_water = block == CRAFTAX_BLOCK_WATER;
    bool on_ground_block = !in_solid_block && !in_water && !in_lava;

    bool valid_move = pos_in_bounds && !in_mob && !in_solid_block;
    valid_move = valid_move && (!collision[0] || !on_ground_block);
    valid_move = valid_move && (!collision[1] || !in_water);
    valid_move = valid_move && (!collision[2] || !in_lava);
    return valid_move;
}

static __device__ inline int32_t craftax_update_mobs_manhattan_to_player(
    const CraftaxState* state,
    int32_t row,
    int32_t col
) {
    return craftax_update_mobs_abs_i32(row - state->player_position[0])
        + craftax_update_mobs_abs_i32(col - state->player_position[1]);
}

static __device__ inline float craftax_update_mobs_damage_done_to_player(
    const CraftaxState* state,
    const float damage_vector[3]
) {
    float defense_vector[3] = {0.0f, 0.0f, 0.0f};
    for (int32_t i = 0; i < 4; i++) {
        defense_vector[0] += (float)state->inventory.armour[i] * 0.1f;
        defense_vector[1] +=
            (float)(int32_t)(state->armour_enchantments[i] == 1) * 0.2f;
        defense_vector[2] +=
            (float)(int32_t)(state->armour_enchantments[i] == 2) * 0.2f;
    }

    float boss_coeff = craftax_step_is_fighting_boss(state)
        ? 1.0f + CRAFTAX_UPDATE_BOSS_FIGHT_EXTRA_DAMAGE
        : 1.0f;
    float damage = 0.0f;
    for (int32_t i = 0; i < 3; i++) {
        damage += (1.0f - defense_vector[i]) * damage_vector[i] * boss_coeff;
    }
    return damage;
}

static __device__ inline int32_t craftax_update_mobs_count_mob_projectiles(
    const CraftaxState* state,
    int32_t level
) {
    const bool* mask = state->mob_projectiles.mask[level];
    return (int32_t)mask[0] + (int32_t)mask[1] + (int32_t)mask[2];
}

static __device__ inline int32_t craftax_update_mobs_first_empty_mob_projectile(
    const CraftaxState* state,
    int32_t level
) {
    const bool* mask = state->mob_projectiles.mask[level];
    if (!mask[0]) return 0;
    if (!mask[1]) return 1;
    if (!mask[2]) return 2;
    return 0;
}

static __device__ inline void craftax_update_mobs_spawn_mob_projectile(
    CraftaxState* state,
    int32_t level,
    bool is_spawning_projectile,
    const int32_t position[2],
    const int32_t direction[2],
    int32_t projectile_type
) {
    if (!is_spawning_projectile) {
        return;
    }

    int32_t index = craftax_update_mobs_first_empty_mob_projectile(
        state,
        level
    );
    state->mob_projectiles.position[level][index][0] = position[0];
    state->mob_projectiles.position[level][index][1] = position[1];
    state->mob_projectiles.mask[level][index] = true;
    state->mob_projectiles.type_id[level][index] = projectile_type;
    state->mob_projectile_directions[level][index][0] = direction[0];
    state->mob_projectile_directions[level][index][1] = direction[1];
}

static __device__ inline void craftax_update_mobs_attack_mob_with_damage(
    CraftaxState* state,
    int32_t row,
    int32_t col,
    const float damage_vector[3],
    bool can_eat,
    bool* did_attack_mob,
    bool* did_kill_mob
) {
    bool did_kill_melee_mob = false;
    bool is_attacking_melee_mob = false;
    craftax_do_action_attack_mobs3(
        state,
        &state->melee_mobs,
        row,
        col,
        damage_vector,
        true,
        CRAFTAX_MOB_MELEE,
        &did_kill_melee_mob,
        &is_attacking_melee_mob
    );

    bool did_kill_passive_mob = false;
    bool is_attacking_passive_mob = false;
    craftax_do_action_attack_mobs3(
        state,
        &state->passive_mobs,
        row,
        col,
        damage_vector,
        can_eat,
        CRAFTAX_MOB_PASSIVE,
        &did_kill_passive_mob,
        &is_attacking_passive_mob
    );

    if (did_kill_passive_mob && can_eat) {
        state->player_food = craftax_step_mini32(
            craftax_step_get_max_food(state),
            state->player_food + 6
        );
        state->player_hunger = 0.0f;
    }

    bool did_kill_ranged_mob = false;
    bool is_attacking_ranged_mob = false;
    craftax_do_action_attack_mobs2(
        state,
        &state->ranged_mobs,
        row,
        col,
        damage_vector,
        true,
        CRAFTAX_MOB_RANGED,
        &did_kill_ranged_mob,
        &is_attacking_ranged_mob
    );

    *did_attack_mob = is_attacking_melee_mob
        || is_attacking_passive_mob
        || is_attacking_ranged_mob;
    bool did_kill_monster = did_kill_melee_mob || did_kill_ranged_mob;
    *did_kill_mob = did_kill_monster || did_kill_passive_mob;

    craftax_do_action_update_mob_map(state, row, col, *did_kill_mob);

    int32_t level = craftax_step_jax_index(
        state->player_level,
        CRAFTAX_NUM_LEVELS
    );
    state->monsters_killed[level] += (int32_t)did_kill_monster;
}

static __device__ inline void craftax_update_mobs_player_projectile_damage_vector(
    const CraftaxState* state,
    int32_t level,
    int32_t projectile_index,
    float damage_vector[3]
) {
    int32_t projectile_type =
        state->player_projectiles.type_id[level][projectile_index];
    craftax_update_mobs_damage_vector(
        projectile_type,
        CRAFTAX_MOB_PROJECTILE,
        damage_vector
    );

    float mask = (float)(int32_t)
        state->player_projectiles.mask[level][projectile_index];
    for (int32_t i = 0; i < 3; i++) {
        damage_vector[i] *= mask;
    }

    bool is_arrow = projectile_type == CRAFTAX_PROJECTILE_ARROW
        || projectile_type == CRAFTAX_PROJECTILE_ARROW2;
    if (is_arrow) {
        float arrow_damage_add[3] = {0.0f, 0.0f, 0.0f};
        int32_t enchantment_index;
        if (craftax_update_mobs_scatter_index(
                state->bow_enchantment,
                3,
                &enchantment_index
            )) {
            arrow_damage_add[enchantment_index] = damage_vector[0] * (1.0f / 2.0f);
        }
        arrow_damage_add[0] = 0.0f;
        for (int32_t i = 0; i < 3; i++) {
            damage_vector[i] += arrow_damage_add[i];
        }
    }

    if (is_arrow) {
        float arrow_damage_coeff =
            1.0f + 0.2f * (float)(state->player_dexterity - 1);
        for (int32_t i = 0; i < 3; i++) {
            damage_vector[i] *= arrow_damage_coeff;
        }
    }

    bool is_magic_projectile = projectile_type == CRAFTAX_PROJECTILE_FIREBALL
        || projectile_type == CRAFTAX_PROJECTILE_ICEBALL;
    if (is_magic_projectile) {
        float magic_damage_coeff =
            1.0f + 0.5f * (float)(state->player_intelligence - 1);
        for (int32_t i = 0; i < 3; i++) {
            damage_vector[i] *= magic_damage_coeff;
        }
    }
}

static __device__ inline void craftax_update_mobs_move_melee(
    CraftaxState* state,
    CraftaxThreefryKey* rng,
    int32_t index
) {
    int32_t level = state->player_level;
    bool old_mask = state->melee_mobs.mask[level][index];
    // Dead slot early-out: no observable effect on obs/reward/terminal.
    // Skip body and RNG draws for speed. Breaks per-seed replay against
    // JAX; define CRAFTAX_JAX_PARITY at build time to restore the
    // branchless slow path (same pattern in every move_* below).
#ifndef CRAFTAX_JAX_PARITY
    if (!old_mask) return;
#endif
    int32_t old_row = state->melee_mobs.position[level][index][0];
    int32_t old_col = state->melee_mobs.position[level][index][1];
    int32_t old_cooldown = state->melee_mobs.attack_cooldown[level][index];
    int32_t mob_type = state->melee_mobs.type_id[level][index];

    CraftaxThreefryKey draw_key =
        craftax_update_mobs_next_random_key(rng);
    int32_t random_direction[2];
    craftax_update_mobs_direction_choice(draw_key, 4, random_direction);
    int32_t random_row = old_row + random_direction[0];
    int32_t random_col = old_col + random_direction[1];

    int32_t distance_row =
        craftax_update_mobs_abs_i32(state->player_position[0] - old_row);
    int32_t distance_col =
        craftax_update_mobs_abs_i32(state->player_position[1] - old_col);
    draw_key = craftax_update_mobs_next_random_key(rng);
    int32_t player_move_axis = craftax_update_mobs_player_axis_choice(
        draw_key,
        distance_row,
        distance_col
    );
    int32_t player_direction[2] = {0, 0};
    if (player_move_axis == 0) {
        player_direction[0] =
            craftax_update_mobs_sign_i32(state->player_position[0] - old_row);
    } else {
        player_direction[1] =
            craftax_update_mobs_sign_i32(state->player_position[1] - old_col);
    }
    int32_t player_row = old_row + player_direction[0];
    int32_t player_col = old_col + player_direction[1];

    int32_t distance_to_player = distance_row + distance_col;
    bool close_to_player = distance_to_player < 10
        || craftax_step_is_fighting_boss(state);
    draw_key = craftax_update_mobs_next_random_key(rng);
    close_to_player = close_to_player
        && craftax_threefry_uniform_f32(draw_key) < 0.75f;

    int32_t proposed_row = close_to_player ? player_row : random_row;
    int32_t proposed_col = close_to_player ? player_col : random_col;

    bool is_attacking_player = distance_to_player == 1
        && old_cooldown <= 0
        && old_mask;
    if (is_attacking_player) {
        proposed_row = old_row;
        proposed_col = old_col;
    }

    float base_damage[3];
    craftax_update_mobs_damage_vector(
        mob_type,
        CRAFTAX_MOB_MELEE,
        base_damage
    );
    float sleeping_coeff = 1.0f + 2.5f * (float)(int32_t)state->is_sleeping;
    for (int32_t i = 0; i < 3; i++) {
        base_damage[i] *= sleeping_coeff;
    }
    float damage = craftax_update_mobs_damage_done_to_player(
        state,
        base_damage
    );

    int32_t new_cooldown = is_attacking_player ? 5 : old_cooldown - 1;
    bool is_waking_player = state->is_sleeping && is_attacking_player;
    state->player_health -= damage * (float)(int32_t)is_attacking_player;
    state->is_sleeping = state->is_sleeping && !is_attacking_player;
    state->is_resting = state->is_resting && !is_attacking_player;
    state->achievements[CRAFTAX_ACH_WAKE_UP] =
        state->achievements[CRAFTAX_ACH_WAKE_UP] || is_waking_player;

    bool collision[3];
    craftax_update_mobs_collision_map(
        mob_type,
        CRAFTAX_MOB_MELEE,
        collision
    );
    bool valid_move = craftax_update_mobs_valid_position(
        state,
        proposed_row,
        proposed_col,
        collision
    );
    int32_t new_row = valid_move ? proposed_row : old_row;
    int32_t new_col = valid_move ? proposed_col : old_col;

    bool should_not_despawn = distance_to_player < CRAFTAX_MOB_DESPAWN_DISTANCE
        || craftax_step_is_fighting_boss(state);

    CraftaxThreefryKey unused_left;
    CraftaxThreefryKey returned_key;
    craftax_threefry_split(*rng, &unused_left, &returned_key);
    *rng = returned_key;

    craftax_update_mobs_clear_old_map_entry(
        state,
        level,
        old_row,
        old_col,
        old_mask
    );
    bool new_mask = old_mask && should_not_despawn;
    craftax_update_mobs_enter_new_map_entry(
        state,
        level,
        new_row,
        new_col,
        new_mask
    );

    state->melee_mobs.position[level][index][0] = new_row;
    state->melee_mobs.position[level][index][1] = new_col;
    state->melee_mobs.attack_cooldown[level][index] = new_cooldown;
    state->melee_mobs.mask[level][index] = new_mask;
}

static __device__ inline void craftax_update_mobs_move_passive(
    CraftaxState* state,
    CraftaxThreefryKey* rng,
    int32_t index
) {
    int32_t level = state->player_level;
    bool old_mask = state->passive_mobs.mask[level][index];
#ifndef CRAFTAX_JAX_PARITY
    if (!old_mask) return;
#endif
    int32_t old_row = state->passive_mobs.position[level][index][0];
    int32_t old_col = state->passive_mobs.position[level][index][1];
    int32_t mob_type = state->passive_mobs.type_id[level][index];

    CraftaxThreefryKey draw_key =
        craftax_update_mobs_next_random_key(rng);
    int32_t direction[2];
    craftax_update_mobs_direction_choice(draw_key, 8, direction);
    int32_t proposed_row = old_row + direction[0];
    int32_t proposed_col = old_col + direction[1];

    bool collision[3];
    craftax_update_mobs_collision_map(
        mob_type,
        CRAFTAX_MOB_PASSIVE,
        collision
    );
    bool valid_move = craftax_update_mobs_valid_position(
        state,
        proposed_row,
        proposed_col,
        collision
    );
    int32_t new_row = valid_move ? proposed_row : old_row;
    int32_t new_col = valid_move ? proposed_col : old_col;

    int32_t distance_to_player = craftax_update_mobs_manhattan_to_player(
        state,
        old_row,
        old_col
    );
    bool should_not_despawn =
        distance_to_player < CRAFTAX_MOB_DESPAWN_DISTANCE;

    craftax_update_mobs_clear_old_map_entry(
        state,
        level,
        old_row,
        old_col,
        old_mask
    );
    bool new_mask = old_mask && should_not_despawn;
    craftax_update_mobs_enter_new_map_entry(
        state,
        level,
        new_row,
        new_col,
        new_mask
    );

    state->passive_mobs.position[level][index][0] = new_row;
    state->passive_mobs.position[level][index][1] = new_col;
    state->passive_mobs.mask[level][index] = new_mask;
}

static __device__ inline void craftax_update_mobs_move_ranged(
    CraftaxState* state,
    CraftaxThreefryKey* rng,
    int32_t index
) {
    int32_t level = state->player_level;
    bool old_mask = state->ranged_mobs.mask[level][index];
#ifndef CRAFTAX_JAX_PARITY
    if (!old_mask) return;
#endif
    int32_t old_row = state->ranged_mobs.position[level][index][0];
    int32_t old_col = state->ranged_mobs.position[level][index][1];
    int32_t old_cooldown = state->ranged_mobs.attack_cooldown[level][index];
    int32_t mob_type = state->ranged_mobs.type_id[level][index];

    CraftaxThreefryKey draw_key =
        craftax_update_mobs_next_random_key(rng);
    int32_t random_direction[2];
    craftax_update_mobs_direction_choice(draw_key, 4, random_direction);
    int32_t random_row = old_row + random_direction[0];
    int32_t random_col = old_col + random_direction[1];

    int32_t distance_row =
        craftax_update_mobs_abs_i32(state->player_position[0] - old_row);
    int32_t distance_col =
        craftax_update_mobs_abs_i32(state->player_position[1] - old_col);
    draw_key = craftax_update_mobs_next_random_key(rng);
    int32_t player_move_axis = craftax_update_mobs_player_axis_choice(
        draw_key,
        distance_row,
        distance_col
    );
    int32_t player_direction[2] = {0, 0};
    if (player_move_axis == 0) {
        player_direction[0] =
            craftax_update_mobs_sign_i32(state->player_position[0] - old_row);
    } else {
        player_direction[1] =
            craftax_update_mobs_sign_i32(state->player_position[1] - old_col);
    }
    int32_t towards_row = old_row + player_direction[0];
    int32_t towards_col = old_col + player_direction[1];
    int32_t away_row = old_row - player_direction[0];
    int32_t away_col = old_col - player_direction[1];

    int32_t distance_to_player = distance_row + distance_col;
    bool far_from_player = distance_to_player >= 6;
    bool too_close_to_player = distance_to_player <= 3;
    int32_t proposed_row = far_from_player ? towards_row : random_row;
    int32_t proposed_col = far_from_player ? towards_col : random_col;
    if (too_close_to_player) {
        proposed_row = away_row;
        proposed_col = away_col;
    }

    draw_key = craftax_update_mobs_next_random_key(rng);
    if (!(craftax_threefry_uniform_f32(draw_key) > 0.85f)) {
        proposed_row = random_row;
        proposed_col = random_col;
    }

    bool collision[3];
    craftax_update_mobs_collision_map(
        mob_type,
        CRAFTAX_MOB_RANGED,
        collision
    );

    bool is_attacking_player =
        distance_to_player >= 4 && distance_to_player <= 5;
    bool proposed_valid = craftax_update_mobs_valid_position(
        state,
        proposed_row,
        proposed_col,
        collision
    );
    is_attacking_player = is_attacking_player
        || (too_close_to_player && !proposed_valid);
    is_attacking_player = is_attacking_player
        && old_cooldown <= 0
        && old_mask;

    bool can_spawn_projectile =
        craftax_update_mobs_count_mob_projectiles(state, level)
            < CRAFTAX_MAX_MOB_PROJECTILES;
    bool is_spawning_projectile =
        is_attacking_player && can_spawn_projectile;
    int32_t projectile_position[2] = {old_row, old_col};
    int32_t projectile_type =
        craftax_update_mobs_projectile_type_for_ranged(mob_type);
    craftax_update_mobs_spawn_mob_projectile(
        state,
        level,
        is_spawning_projectile,
        projectile_position,
        player_direction,
        projectile_type
    );

    if (is_attacking_player) {
        proposed_row = old_row;
        proposed_col = old_col;
    }
    int32_t new_cooldown = is_attacking_player ? 4 : old_cooldown - 1;

    bool valid_move = craftax_update_mobs_valid_position(
        state,
        proposed_row,
        proposed_col,
        collision
    );
    int32_t new_row = valid_move ? proposed_row : old_row;
    int32_t new_col = valid_move ? proposed_col : old_col;

    bool should_not_despawn = distance_to_player < CRAFTAX_MOB_DESPAWN_DISTANCE
        || craftax_step_is_fighting_boss(state);

    craftax_update_mobs_clear_old_map_entry(
        state,
        level,
        old_row,
        old_col,
        old_mask
    );
    bool new_mask = old_mask && should_not_despawn;
    craftax_update_mobs_enter_new_map_entry(
        state,
        level,
        new_row,
        new_col,
        new_mask
    );

    state->ranged_mobs.position[level][index][0] = new_row;
    state->ranged_mobs.position[level][index][1] = new_col;
    state->ranged_mobs.attack_cooldown[level][index] = new_cooldown;
    state->ranged_mobs.mask[level][index] = new_mask;
}

static __device__ inline void craftax_update_mobs_move_mob_projectile(
    CraftaxState* state,
    int32_t index
) {
    int32_t level = state->player_level;
    bool old_mask = state->mob_projectiles.mask[level][index];
#ifndef CRAFTAX_JAX_PARITY
    if (!old_mask) return;
#endif
    int32_t old_row = state->mob_projectiles.position[level][index][0];
    int32_t old_col = state->mob_projectiles.position[level][index][1];
    int32_t proposed_row =
        old_row + state->mob_projectile_directions[level][index][0];
    int32_t proposed_col =
        old_col + state->mob_projectile_directions[level][index][1];

    bool proposed_in_player =
        proposed_row == state->player_position[0]
        && proposed_col == state->player_position[1];
    bool proposed_in_bounds = craftax_update_mobs_in_bounds(
        proposed_row,
        proposed_col
    );
    int32_t proposed_block = craftax_update_mobs_read_block(
        state,
        level,
        proposed_row,
        proposed_col
    );
    bool in_wall = craftax_step_is_solid_block(proposed_block)
        && proposed_block != CRAFTAX_BLOCK_WATER;
    bool in_mob = craftax_step_is_in_mob(state, proposed_row, proposed_col);
    bool continue_move = proposed_in_bounds && !in_wall && !in_mob;

    bool hit_player0 =
        old_row == state->player_position[0]
        && old_col == state->player_position[1]
        && old_mask;
    bool hit_player1 = proposed_in_player && old_mask;
    bool hit_player = hit_player0 || hit_player1;
    continue_move = continue_move && !hit_player;

    bool new_mask = continue_move && old_mask;

    bool hit_bench_or_furnace = proposed_block == CRAFTAX_BLOCK_FURNACE
        || proposed_block == CRAFTAX_BLOCK_CRAFTING_TABLE;
    bool removing_block = hit_bench_or_furnace && old_mask;
    int32_t new_block = removing_block ? CRAFTAX_BLOCK_PATH : proposed_block;

    int32_t projectile_type =
        state->mob_projectiles.type_id[level][index];
    float damage_vector[3];
    craftax_update_mobs_damage_vector(
        projectile_type,
        CRAFTAX_MOB_PROJECTILE,
        damage_vector
    );
    float damage = craftax_update_mobs_damage_done_to_player(
        state,
        damage_vector
    );

    state->mob_projectiles.position[level][index][0] = proposed_row;
    state->mob_projectiles.position[level][index][1] = proposed_col;
    state->mob_projectiles.mask[level][index] = new_mask;
    state->player_health -= damage * (float)(int32_t)hit_player;
    state->is_sleeping = state->is_sleeping && !hit_player;
    state->is_resting = state->is_resting && !hit_player;
    craftax_update_mobs_set_block(
        state,
        level,
        proposed_row,
        proposed_col,
        new_block
    );
}

static __device__ inline void craftax_update_mobs_move_player_projectile(
    CraftaxState* state,
    int32_t index
) {
    int32_t level = state->player_level;
    bool old_mask = state->player_projectiles.mask[level][index];
#ifndef CRAFTAX_JAX_PARITY
    if (!old_mask) return;
#endif
    int32_t old_row = state->player_projectiles.position[level][index][0];
    int32_t old_col = state->player_projectiles.position[level][index][1];
    int32_t proposed_row =
        old_row + state->player_projectile_directions[level][index][0];
    int32_t proposed_col =
        old_col + state->player_projectile_directions[level][index][1];

    float damage_vector[3];
    craftax_update_mobs_player_projectile_damage_vector(
        state,
        level,
        index,
        damage_vector
    );

    bool proposed_in_bounds = craftax_update_mobs_in_bounds(
        proposed_row,
        proposed_col
    );
    int32_t proposed_block = craftax_update_mobs_read_block(
        state,
        level,
        proposed_row,
        proposed_col
    );
    bool in_wall = craftax_step_is_solid_block(proposed_block)
        && proposed_block != CRAFTAX_BLOCK_WATER;

    bool did_attack_mob0 = false;
    bool did_kill_mob0 = false;
    craftax_update_mobs_attack_mob_with_damage(
        state,
        old_row,
        old_col,
        damage_vector,
        false,
        &did_attack_mob0,
        &did_kill_mob0
    );
    (void)did_kill_mob0;

    float second_damage_vector[3];
    for (int32_t i = 0; i < 3; i++) {
        second_damage_vector[i] =
            damage_vector[i] * (float)(int32_t)(!did_attack_mob0);
    }

    bool did_attack_mob1 = false;
    bool did_kill_mob1 = false;
    craftax_update_mobs_attack_mob_with_damage(
        state,
        proposed_row,
        proposed_col,
        second_damage_vector,
        false,
        &did_attack_mob1,
        &did_kill_mob1
    );
    (void)did_kill_mob1;

    bool did_attack_mob = did_attack_mob0 || did_attack_mob1;
    bool continue_move = proposed_in_bounds && !in_wall && !did_attack_mob;
    bool new_mask = continue_move && old_mask;

    state->player_projectiles.position[level][index][0] = proposed_row;
    state->player_projectiles.position[level][index][1] = proposed_col;
    state->player_projectiles.mask[level][index] = new_mask;
}

static __device__ inline void craftax_update_mobs_native(
    CraftaxState* state,
    CraftaxThreefryKey rng
) {
    CraftaxThreefryKey unused;

    craftax_threefry_split(rng, &rng, &unused);
    craftax_update_mobs_move_melee(state, &rng, 0);
    craftax_update_mobs_move_melee(state, &rng, 1);
    craftax_update_mobs_move_melee(state, &rng, 2);

    craftax_threefry_split(rng, &rng, &unused);
    craftax_update_mobs_move_passive(state, &rng, 0);
    craftax_update_mobs_move_passive(state, &rng, 1);
    craftax_update_mobs_move_passive(state, &rng, 2);

    craftax_threefry_split(rng, &rng, &unused);
    craftax_update_mobs_move_ranged(state, &rng, 0);
    craftax_update_mobs_move_ranged(state, &rng, 1);

    craftax_threefry_split(rng, &rng, &unused);
    craftax_update_mobs_move_mob_projectile(state, 0);
    craftax_update_mobs_move_mob_projectile(state, 1);
    craftax_update_mobs_move_mob_projectile(state, 2);

    craftax_threefry_split(rng, &rng, &unused);
    craftax_update_mobs_move_player_projectile(state, 0);
    craftax_update_mobs_move_player_projectile(state, 1);
    craftax_update_mobs_move_player_projectile(state, 2);
}

// ============================================================
// ===== step_spawn_mobs.h =====
// ============================================================
// Craftax spawn_mobs, optimized for CPU.
//
// Bitwise-equivalent to the prior JAX-transliterated baseline (verified by
// ocean/craftax_exp/parity_vs_baseline.c over 1.28M paired steps), ~6-9x
// faster per step by stripping JAX-isms:
//   - full-grid validity masks -> compact coord list collected in one pass
//   - bounding-box scan (only cells within MOB_DESPAWN_DISTANCE)
//   - early return on mob-cap / probability-roll failure (no dead writes)
//   - merged count + first_empty loops
//
// The prior reference implementation is archived at
// ocean/craftax_exp/step_spawn_mobs_baseline.h.



#define CRAFTAX_SPAWN_MAP_CELLS (CRAFTAX_MAP_SIZE * CRAFTAX_MAP_SIZE)
#define CRAFTAX_SPAWN_BBOX_MAX_CELLS 729  // (2*DESPAWN-1)^2 at 14 = 27*27
#define CRAFTAX_SPAWN_ALL_VALID_BLOCK_MASK ( \
    (1ULL << CRAFTAX_BLOCK_GRASS) \
    | (1ULL << CRAFTAX_BLOCK_PATH) \
    | (1ULL << CRAFTAX_BLOCK_FIRE_GRASS) \
    | (1ULL << CRAFTAX_BLOCK_ICE_GRASS))
#define CRAFTAX_SPAWN_GRAVE_BLOCK_MASK ( \
    (1ULL << CRAFTAX_BLOCK_GRAVE) \
    | (1ULL << CRAFTAX_BLOCK_GRAVE2) \
    | (1ULL << CRAFTAX_BLOCK_GRAVE3))
#define CRAFTAX_SPAWN_WATER_BLOCK_MASK (1ULL << CRAFTAX_BLOCK_WATER)

typedef struct { int8_t dr, dc0, dc1; } CraftaxSpawnOffsetSpan;

static __device__ CraftaxSpawnOffsetSpan craftax_spawn_passive_spans[CRAFTAX_SPAWN_BBOX_MAX_CELLS];
static __device__ CraftaxSpawnOffsetSpan craftax_spawn_hostile_spans[CRAFTAX_SPAWN_BBOX_MAX_CELLS];
static __device__ CraftaxSpawnOffsetSpan craftax_spawn_boss_spans[CRAFTAX_SPAWN_BBOX_MAX_CELLS];
static __device__ int32_t craftax_spawn_passive_span_count = 0;
static __device__ int32_t craftax_spawn_hostile_span_count = 0;
static __device__ int32_t craftax_spawn_boss_span_count = 0;
static __device__ int32_t craftax_spawn_offsets_initialized = 0;

static __device__ inline void craftax_spawn_append_span(
    CraftaxSpawnOffsetSpan* spans,
    int32_t* count,
    int32_t dr,
    int32_t dc0,
    int32_t dc1
) {
    spans[*count] = (CraftaxSpawnOffsetSpan){
        (int8_t)dr, (int8_t)dc0, (int8_t)dc1
    };
    *count += 1;
}

static __device__ inline void craftax_spawn_build_spans_for_row(
    CraftaxSpawnOffsetSpan* spans,
    int32_t* count,
    int32_t dr,
    int32_t limit,
    int32_t min_exclusive,
    int32_t max_exclusive
) {
    bool active = false;
    int32_t start = 0;
    for (int32_t dc = -limit; dc <= limit; dc++) {
        int32_t distance2 = dr * dr + dc * dc;
        bool valid = distance2 > min_exclusive && distance2 < max_exclusive;
        if (valid && !active) {
            active = true;
            start = dc;
        } else if (!valid && active) {
            craftax_spawn_append_span(spans, count, dr, start, dc - 1);
            active = false;
        }
    }
    if (active) {
        craftax_spawn_append_span(spans, count, dr, start, limit);
    }
}

static __device__ inline void craftax_spawn_init_offsets_once(void) {
    if (__atomic_load_n(
            &craftax_spawn_offsets_initialized, __ATOMIC_ACQUIRE
    )) return;

    // [cuda port] #pragma omp removed
    {
        if (!__atomic_load_n(
                &craftax_spawn_offsets_initialized, __ATOMIC_RELAXED
        )) {
            int32_t passive_count = 0;
            int32_t hostile_count = 0;
            int32_t boss_count = 0;
            int32_t limit = CRAFTAX_MOB_DESPAWN_DISTANCE - 1;
            int32_t limit2 = CRAFTAX_MOB_DESPAWN_DISTANCE
                           * CRAFTAX_MOB_DESPAWN_DISTANCE;
            for (int32_t dr = -limit; dr <= limit; dr++) {
                craftax_spawn_build_spans_for_row(
                    craftax_spawn_passive_spans,
                    &passive_count,
                    dr,
                    limit,
                    9,
                    limit2
                );
                craftax_spawn_build_spans_for_row(
                    craftax_spawn_hostile_spans,
                    &hostile_count,
                    dr,
                    limit,
                    81,
                    limit2
                );
                craftax_spawn_build_spans_for_row(
                    craftax_spawn_boss_spans,
                    &boss_count,
                    dr,
                    limit,
                    -1,
                    37
                );
            }
            craftax_spawn_passive_span_count = passive_count;
            craftax_spawn_hostile_span_count = hostile_count;
            craftax_spawn_boss_span_count = boss_count;
            __atomic_store_n(
                &craftax_spawn_offsets_initialized, 1, __ATOMIC_RELEASE
            );
        }
    }
}

static __device__ inline bool craftax_spawn_block_matches(uint8_t block, uint64_t mask) {
    return ((mask >> block) & 1ULL) != 0;
}

static __device__ inline uint64_t craftax_spawn_row_bits_for_mask(
    const CraftaxState* state,
    int32_t level,
    int32_t row,
    uint64_t terrain_mask
) {
    if (terrain_mask == CRAFTAX_SPAWN_ALL_VALID_BLOCK_MASK) {
        return state->spawn_all_bits[level][row];
    }
    if (terrain_mask == CRAFTAX_SPAWN_GRAVE_BLOCK_MASK) {
        return state->spawn_grave_bits[level][row];
    }
    return state->spawn_water_bits[level][row];
}

static __device__ inline uint64_t craftax_spawn_col_mask(int32_t col0, int32_t col1) {
    uint64_t hi = (1ULL << (col1 + 1)) - 1ULL;
    uint64_t lo = col0 <= 0 ? 0ULL : ((1ULL << col0) - 1ULL);
    return hi & ~lo;
}

static __device__ inline CraftaxThreefryKey craftax_spawn_next_random_key(
    CraftaxThreefryKey* rng
) {
    CraftaxThreefryKey draw;
    craftax_threefry_split(*rng, rng, &draw);
    return draw;
}

static __device__ inline int32_t craftax_spawn_floor_mob_type(
    int32_t floor, int32_t mob_class
) {
    static const int32_t mapping[CRAFTAX_NUM_LEVELS][3] = {
        {0, 0, 0}, {2, 2, 2}, {1, 1, 1}, {2, 3, 3}, {2, 4, 4},
        {1, 5, 5}, {1, 6, 6}, {1, 7, 7}, {0, 0, 0},
    };
    int32_t level = craftax_step_jax_index(floor, CRAFTAX_NUM_LEVELS);
    int32_t class_index = craftax_step_jax_index(mob_class, 3);
    return mapping[level][class_index];
}

static __device__ inline float craftax_spawn_floor_spawn_chance(
    int32_t floor, int32_t chance_index
) {
    static const float chances[CRAFTAX_NUM_LEVELS][4] = {
        {0.1f, 0.02f, 0.05f, 0.1f},
        {0.1f, 0.06f, 0.05f, 0.0f},
        {0.1f, 0.06f, 0.05f, 0.0f},
        {0.1f, 0.06f, 0.05f, 0.0f},
        {0.1f, 0.06f, 0.05f, 0.0f},
        {0.1f, 0.06f, 0.05f, 0.0f},
        {0.1f, 0.06f, 0.05f, 0.0f},
        {0.0f, 0.06f, 0.05f, 0.0f},
        {0.1f, 0.06f, 0.05f, 0.0f},
    };
    int32_t level = craftax_step_jax_index(floor, CRAFTAX_NUM_LEVELS);
    int32_t index = craftax_step_jax_index(chance_index, 4);
    return chances[level][index];
}

static __device__ inline float craftax_spawn_mob_type_health(
    int32_t mob_type, int32_t mob_class
) {
    static const float health[CRAFTAX_NUM_MOB_TYPES][4] = {
        {3.0f, 5.0f, 3.0f, 0.0f}, {4.0f, 7.0f, 5.0f, 0.0f},
        {6.0f, 9.0f, 6.0f, 0.0f}, {8.0f, 11.0f, 8.0f, 0.0f},
        {0.0f, 12.0f, 12.0f, 0.0f}, {0.0f, 20.0f, 4.0f, 0.0f},
        {0.0f, 20.0f, 14.0f, 0.0f}, {0.0f, 24.0f, 16.0f, 0.0f},
    };
    int32_t type_index = craftax_step_jax_index(mob_type, CRAFTAX_NUM_MOB_TYPES);
    int32_t class_index = craftax_step_jax_index(mob_class, 4);
    return health[type_index][class_index];
}

static __device__ inline bool craftax_spawn_is_all_valid_block(int32_t block) {
    static const uint8_t flags[CRAFTAX_NUM_BLOCK_TYPES] = {
        [CRAFTAX_BLOCK_GRASS] = 1,
        [CRAFTAX_BLOCK_PATH] = 1,
        [CRAFTAX_BLOCK_FIRE_GRASS] = 1,
        [CRAFTAX_BLOCK_ICE_GRASS] = 1,
    };
    int32_t idx = craftax_step_jax_index(block, CRAFTAX_NUM_BLOCK_TYPES);
    return flags[idx] != 0;
}

static __device__ inline bool craftax_spawn_is_grave_block(int32_t block) {
    static const uint8_t flags[CRAFTAX_NUM_BLOCK_TYPES] = {
        [CRAFTAX_BLOCK_GRAVE] = 1,
        [CRAFTAX_BLOCK_GRAVE2] = 1,
        [CRAFTAX_BLOCK_GRAVE3] = 1,
    };
    int32_t idx = craftax_step_jax_index(block, CRAFTAX_NUM_BLOCK_TYPES);
    return flags[idx] != 0;
}

static __device__ inline bool craftax_spawn_is_water_block(int32_t block) {
    static const uint8_t flags[CRAFTAX_NUM_BLOCK_TYPES] = {
        [CRAFTAX_BLOCK_WATER] = 1,
    };
    int32_t idx = craftax_step_jax_index(block, CRAFTAX_NUM_BLOCK_TYPES);
    return flags[idx] != 0;
}

static __device__ inline int32_t craftax_spawn_player_distance_squared(
    const CraftaxState* state, int32_t row, int32_t col
) {
    int32_t dr = row - state->player_position[0];
    int32_t dc = col - state->player_position[1];
    if (dr < 0) dr = -dr;
    if (dc < 0) dc = -dc;
    return dr * dr + dc * dc;
}

static __device__ inline int32_t craftax_spawn_count_mobs3(
    const CraftaxMobs3* mobs, int32_t level
) {
    int32_t count = 0;
    for (int32_t i = 0; i < 3; i++) count += (int32_t)mobs->mask[level][i];
    return count;
}

static __device__ inline int32_t craftax_spawn_count_mobs2(
    const CraftaxMobs2* mobs, int32_t level
) {
    int32_t count = 0;
    for (int32_t i = 0; i < 2; i++) count += (int32_t)mobs->mask[level][i];
    return count;
}

static __device__ inline int32_t craftax_spawn_first_empty_mobs3(
    const CraftaxMobs3* mobs, int32_t level
) {
    for (int32_t i = 0; i < 3; i++) if (!mobs->mask[level][i]) return i;
    return 0;
}

static __device__ inline int32_t craftax_spawn_first_empty_mobs2(
    const CraftaxMobs2* mobs, int32_t level
) {
    for (int32_t i = 0; i < 2; i++) if (!mobs->mask[level][i]) return i;
    return 0;
}

static __device__ inline void craftax_spawn_mobs3_count_and_empty(
    const CraftaxMobs3* mobs, int32_t level,
    int32_t* count_out, int32_t* first_empty_out
) {
    int32_t count = 0, first_empty = 0;
    bool found = false;
    for (int32_t i = 0; i < 3; i++) {
        bool m = mobs->mask[level][i];
        count += (int32_t)m;
        if (!m && !found) { first_empty = i; found = true; }
    }
    *count_out = count;
    *first_empty_out = first_empty;
}

static __device__ inline void craftax_spawn_mobs2_count_and_empty(
    const CraftaxMobs2* mobs, int32_t level,
    int32_t* count_out, int32_t* first_empty_out
) {
    int32_t count = 0, first_empty = 0;
    bool found = false;
    for (int32_t i = 0; i < 2; i++) {
        bool m = mobs->mask[level][i];
        count += (int32_t)m;
        if (!m && !found) { first_empty = i; found = true; }
    }
    *count_out = count;
    *first_empty_out = first_empty;
}

// Baseline algorithm on a bool mask:
//   draw = valid_count * (1.0 - uniform_f32(key));
//   cum = 0;
//   for i: if valid[i] { cum += 1.0; if (cum >= draw) return i; }
// Over a compact list of length valid_count this collapses to a short loop
// using the same FP arithmetic, preserving bitwise-identical choice.
static __device__ inline int32_t craftax_spawn_pick_kth(
    int32_t valid_count, CraftaxThreefryKey key
) {
    float draw = (float)valid_count * (1.0f - craftax_threefry_uniform_f32(key));
    float cum = 0.0f;
    for (int32_t k = 0; k < valid_count; k++) {
        cum += 1.0f;
        if (cum >= draw) return k;
    }
    return valid_count - 1;
}

typedef struct { int16_t row, col; } CraftaxSpawnCoord;

typedef struct {
    CraftaxSpawnCoord passive[CRAFTAX_SPAWN_BBOX_MAX_CELLS];
    CraftaxSpawnCoord melee[CRAFTAX_SPAWN_BBOX_MAX_CELLS];
    CraftaxSpawnCoord ranged[CRAFTAX_SPAWN_BBOX_MAX_CELLS];
    int32_t passive_count;
    int32_t melee_count;
    int32_t ranged_count;
} CraftaxSpawnLists;

static __device__ inline int32_t craftax_spawn_collect_spans(
    const CraftaxState* state,
    int32_t level,
    const CraftaxSpawnOffsetSpan* spans,
    int32_t span_count,
    uint64_t terrain_mask,
    CraftaxSpawnCoord* coords
) {
    int32_t pr = state->player_position[0];
    int32_t pc = state->player_position[1];
    int32_t n = 0;
    for (int32_t i = 0; i < span_count; i++) {
        int32_t row = pr + spans[i].dr;
        if ((uint32_t)row >= CRAFTAX_MAP_SIZE) continue;
        int32_t col0 = pc + spans[i].dc0;
        int32_t col1 = pc + spans[i].dc1;
        if (col0 < 0) col0 = 0;
        if (col1 >= CRAFTAX_MAP_SIZE) col1 = CRAFTAX_MAP_SIZE - 1;
        if (col0 > col1) continue;
        uint64_t candidates =
            craftax_spawn_row_bits_for_mask(state, level, row, terrain_mask)
            & ~state->mob_bits[level][row]
            & craftax_spawn_col_mask(col0, col1);
        while (candidates != 0) {
            int32_t col = __builtin_ctzll(candidates);
            coords[n].row = (int16_t)row;
            coords[n].col = (int16_t)col;
            n++;
            candidates &= candidates - 1;
        }
    }
    return n;
}

static __device__ inline bool craftax_spawn_scan_spans(
    const CraftaxState* state,
    int32_t level,
    const CraftaxSpawnOffsetSpan* spans,
    int32_t span_count,
    uint64_t terrain_mask,
    CraftaxThreefryKey pos_key,
    int32_t* out_row,
    int32_t* out_col
) {
    CraftaxSpawnCoord coords[CRAFTAX_SPAWN_BBOX_MAX_CELLS];
    int32_t n = craftax_spawn_collect_spans(
        state, level, spans, span_count, terrain_mask, coords
    );
    if (n == 0) return false;
    int32_t k = craftax_spawn_pick_kth(n, pos_key);
    *out_row = coords[k].row;
    *out_col = coords[k].col;
    return true;
}

static __device__ inline bool craftax_spawn_coord_matches(
    CraftaxSpawnCoord coord, bool exclude, int32_t row, int32_t col
) {
    return exclude && coord.row == row && coord.col == col;
}

static __device__ inline bool craftax_spawn_pick_excluding(
    const CraftaxSpawnCoord* coords, int32_t count, CraftaxThreefryKey key,
    bool exclude_a, int32_t row_a, int32_t col_a,
    bool exclude_b, int32_t row_b, int32_t col_b,
    int32_t* out_row, int32_t* out_col
) {
    int32_t valid_count = 0;
    for (int32_t i = 0; i < count; i++) {
        bool excluded = craftax_spawn_coord_matches(
            coords[i], exclude_a, row_a, col_a
        ) || craftax_spawn_coord_matches(coords[i], exclude_b, row_b, col_b);
        valid_count += excluded ? 0 : 1;
    }
    if (valid_count == 0) return false;

    int32_t k = craftax_spawn_pick_kth(valid_count, key);
    for (int32_t i = 0; i < count; i++) {
        bool excluded = craftax_spawn_coord_matches(
            coords[i], exclude_a, row_a, col_a
        ) || craftax_spawn_coord_matches(coords[i], exclude_b, row_b, col_b);
        if (excluded) continue;
        if (k == 0) {
            *out_row = coords[i].row;
            *out_col = coords[i].col;
            return true;
        }
        k--;
    }
    return false;
}

static __device__ inline void craftax_spawn_scan_all(
    const CraftaxState* state,
    int32_t level,
    int32_t ranged_type,
    bool fighting_boss,
    bool need_passive,
    bool need_melee,
    bool need_ranged,
    CraftaxSpawnLists* out
) {
    out->passive_count = 0;
    out->melee_count = 0;
    out->ranged_count = 0;

    craftax_spawn_init_offsets_once();

    if (need_passive) {
        out->passive_count = craftax_spawn_collect_spans(
            state,
            level,
            craftax_spawn_passive_spans,
            craftax_spawn_passive_span_count,
            CRAFTAX_SPAWN_ALL_VALID_BLOCK_MASK,
            out->passive
        );
    }

    if (!need_melee && !need_ranged) return;

    int32_t pr = state->player_position[0];
    int32_t pc = state->player_position[1];
    const CraftaxSpawnOffsetSpan* spans = fighting_boss
        ? craftax_spawn_boss_spans
        : craftax_spawn_hostile_spans;
    int32_t span_count = fighting_boss
        ? craftax_spawn_boss_span_count
        : craftax_spawn_hostile_span_count;
    bool ranged_water_type = (ranged_type == 5);

    uint64_t melee_terrain_mask = fighting_boss
        ? CRAFTAX_SPAWN_GRAVE_BLOCK_MASK
        : CRAFTAX_SPAWN_ALL_VALID_BLOCK_MASK;
    uint64_t ranged_terrain_mask;
    if (fighting_boss) {
        ranged_terrain_mask = CRAFTAX_SPAWN_GRAVE_BLOCK_MASK;
    } else if (ranged_water_type) {
        ranged_terrain_mask = CRAFTAX_SPAWN_WATER_BLOCK_MASK;
    } else {
        ranged_terrain_mask = CRAFTAX_SPAWN_ALL_VALID_BLOCK_MASK;
    }

    for (int32_t i = 0; i < span_count; i++) {
        int32_t row = pr + spans[i].dr;
        if ((uint32_t)row >= CRAFTAX_MAP_SIZE) continue;
        int32_t col0 = pc + spans[i].dc0;
        int32_t col1 = pc + spans[i].dc1;
        if (col0 < 0) col0 = 0;
        if (col1 >= CRAFTAX_MAP_SIZE) col1 = CRAFTAX_MAP_SIZE - 1;
        if (col0 > col1) continue;
        uint64_t open_bits =
            ~state->mob_bits[level][row] & craftax_spawn_col_mask(col0, col1);

        if (need_melee) {
            uint64_t melee_candidates =
                craftax_spawn_row_bits_for_mask(
                    state, level, row, melee_terrain_mask
                ) & open_bits;
            while (melee_candidates != 0) {
                int32_t col = __builtin_ctzll(melee_candidates);
                int32_t n = out->melee_count++;
                out->melee[n].row = (int16_t)row;
                out->melee[n].col = (int16_t)col;
                melee_candidates &= melee_candidates - 1;
            }
        }

        if (need_ranged) {
            uint64_t ranged_candidates =
                craftax_spawn_row_bits_for_mask(
                    state, level, row, ranged_terrain_mask
                ) & open_bits;
            while (ranged_candidates != 0) {
                int32_t col = __builtin_ctzll(ranged_candidates);
                int32_t n = out->ranged_count++;
                out->ranged[n].row = (int16_t)row;
                out->ranged[n].col = (int16_t)col;
                ranged_candidates &= ranged_candidates - 1;
            }
        }
    }
}

static __device__ inline bool craftax_spawn_scan_passive(
    const CraftaxState* state, int32_t level, CraftaxThreefryKey pos_key,
    int32_t* out_row, int32_t* out_col
) {
    craftax_spawn_init_offsets_once();
    return craftax_spawn_scan_spans(
        state,
        level,
        craftax_spawn_passive_spans,
        craftax_spawn_passive_span_count,
        CRAFTAX_SPAWN_ALL_VALID_BLOCK_MASK,
        pos_key,
        out_row,
        out_col
    );
}

static __device__ inline bool craftax_spawn_scan_melee(
    const CraftaxState* state, int32_t level, bool fighting_boss,
    CraftaxThreefryKey pos_key, int32_t* out_row, int32_t* out_col
) {
    craftax_spawn_init_offsets_once();
    const CraftaxSpawnOffsetSpan* spans = fighting_boss
        ? craftax_spawn_boss_spans
        : craftax_spawn_hostile_spans;
    int32_t span_count = fighting_boss
        ? craftax_spawn_boss_span_count
        : craftax_spawn_hostile_span_count;
    uint64_t terrain_mask = fighting_boss
        ? CRAFTAX_SPAWN_GRAVE_BLOCK_MASK
        : CRAFTAX_SPAWN_ALL_VALID_BLOCK_MASK;
    return craftax_spawn_scan_spans(
        state, level, spans, span_count, terrain_mask, pos_key,
        out_row, out_col
    );
}

static __device__ inline bool craftax_spawn_scan_ranged(
    const CraftaxState* state, int32_t level, int32_t new_type,
    bool fighting_boss, CraftaxThreefryKey pos_key,
    int32_t* out_row, int32_t* out_col
) {
    craftax_spawn_init_offsets_once();
    const CraftaxSpawnOffsetSpan* spans = fighting_boss
        ? craftax_spawn_boss_spans
        : craftax_spawn_hostile_spans;
    int32_t span_count = fighting_boss
        ? craftax_spawn_boss_span_count
        : craftax_spawn_hostile_span_count;
    uint64_t terrain_mask;
    if (fighting_boss) {
        terrain_mask = CRAFTAX_SPAWN_GRAVE_BLOCK_MASK;
    } else if (new_type == 5) {
        terrain_mask = CRAFTAX_SPAWN_WATER_BLOCK_MASK;
    } else {
        terrain_mask = CRAFTAX_SPAWN_ALL_VALID_BLOCK_MASK;
    }
    return craftax_spawn_scan_spans(
        state, level, spans, span_count, terrain_mask, pos_key,
        out_row, out_col
    );
}

// Both RNG keys are always consumed (preserves baseline RNG sequence).
// Baseline quirk: type_id[level][slot] is written unconditionally, even
// when no mob spawns. We match that for bitwise parity.

static __device__ inline void craftax_spawn_passive_mob(
    CraftaxState* state, CraftaxThreefryKey* rng,
    int32_t level, bool fighting_boss
) {
    int32_t count, slot;
    craftax_spawn_mobs3_count_and_empty(&state->passive_mobs, level, &count, &slot);

    CraftaxThreefryKey prob_key = craftax_spawn_next_random_key(rng);
    CraftaxThreefryKey pos_key  = craftax_spawn_next_random_key(rng);

    int32_t type = craftax_spawn_floor_mob_type(level, CRAFTAX_MOB_PASSIVE);
    state->passive_mobs.type_id[level][slot] = type;

    if (fighting_boss) return;
    if (count >= CRAFTAX_MAX_PASSIVE_MOBS) return;
    if (craftax_threefry_uniform_f32(prob_key)
        >= craftax_spawn_floor_spawn_chance(level, 0)) return;

    int32_t row, col;
    if (!craftax_spawn_scan_passive(state, level, pos_key, &row, &col)) return;

    state->passive_mobs.position[level][slot][0] = row;
    state->passive_mobs.position[level][slot][1] = col;
    state->passive_mobs.health[level][slot]      =
        craftax_spawn_mob_type_health(type, CRAFTAX_MOB_PASSIVE);
    state->passive_mobs.mask[level][slot]        = true;
    state->mob_bits[level][row] |= (1ULL << col);
}

static __device__ inline void craftax_spawn_melee_mob(
    CraftaxState* state, CraftaxThreefryKey* rng,
    int32_t level, bool fighting_boss, int32_t monster_spawn_coeff
) {
    int32_t count, slot;
    craftax_spawn_mobs3_count_and_empty(&state->melee_mobs, level, &count, &slot);

    int32_t type = fighting_boss
        ? craftax_spawn_floor_mob_type(state->boss_progress, CRAFTAX_MOB_MELEE)
        : craftax_spawn_floor_mob_type(level, CRAFTAX_MOB_MELEE);

    CraftaxThreefryKey prob_key = craftax_spawn_next_random_key(rng);
    float night_coeff = 1.0f - state->light_level;
    float spawn_chance = craftax_spawn_floor_spawn_chance(level, 1)
        + craftax_spawn_floor_spawn_chance(level, 3) * night_coeff * night_coeff;
    CraftaxThreefryKey pos_key = craftax_spawn_next_random_key(rng);

    state->melee_mobs.type_id[level][slot] = type;

    if (count >= CRAFTAX_MAX_MELEE_MOBS) return;
    if (craftax_threefry_uniform_f32(prob_key)
        >= spawn_chance * (float)monster_spawn_coeff) return;

    int32_t row, col;
    if (!craftax_spawn_scan_melee(state, level, fighting_boss, pos_key, &row, &col))
        return;

    state->melee_mobs.position[level][slot][0] = row;
    state->melee_mobs.position[level][slot][1] = col;
    state->melee_mobs.health[level][slot]      =
        craftax_spawn_mob_type_health(type, CRAFTAX_MOB_MELEE);
    state->melee_mobs.mask[level][slot]        = true;
    state->mob_bits[level][row] |= (1ULL << col);
}

static __device__ inline void craftax_spawn_ranged_mob(
    CraftaxState* state, CraftaxThreefryKey* rng,
    int32_t level, bool fighting_boss, int32_t monster_spawn_coeff
) {
    int32_t count, slot;
    craftax_spawn_mobs2_count_and_empty(&state->ranged_mobs, level, &count, &slot);

    int32_t type = fighting_boss
        ? craftax_spawn_floor_mob_type(state->boss_progress, CRAFTAX_MOB_RANGED)
        : craftax_spawn_floor_mob_type(level, CRAFTAX_MOB_RANGED);

    CraftaxThreefryKey prob_key = craftax_spawn_next_random_key(rng);
    CraftaxThreefryKey pos_key  = craftax_spawn_next_random_key(rng);

    state->ranged_mobs.type_id[level][slot] = type;

    if (count >= CRAFTAX_MAX_RANGED_MOBS) return;
    if (craftax_threefry_uniform_f32(prob_key)
        >= craftax_spawn_floor_spawn_chance(level, 2) * (float)monster_spawn_coeff)
        return;

    int32_t row, col;
    if (!craftax_spawn_scan_ranged(state, level, type, fighting_boss, pos_key,
                                    &row, &col)) return;

    state->ranged_mobs.position[level][slot][0] = row;
    state->ranged_mobs.position[level][slot][1] = col;
    state->ranged_mobs.health[level][slot]      =
        craftax_spawn_mob_type_health(type, CRAFTAX_MOB_RANGED);
    state->ranged_mobs.mask[level][slot]        = true;
    state->mob_bits[level][row] |= (1ULL << col);
}

static __device__ inline void craftax_spawn_mobs_native(
    CraftaxState* state, CraftaxThreefryKey rng
) {
    int32_t level = craftax_step_jax_index(
        state->player_level, CRAFTAX_NUM_LEVELS
    );
    bool fighting_boss = craftax_step_is_fighting_boss(state);
    int32_t monster_spawn_coeff =
        1
        + (int32_t)(state->monsters_killed[level]
                    < CRAFTAX_MONSTERS_KILLED_TO_CLEAR_LEVEL) * 2;

    bool boss_spawn_wave =
        fighting_boss && state->boss_timesteps_to_spawn_this_round >= 1;
    if (fighting_boss) {
        monster_spawn_coeff *= (int32_t)boss_spawn_wave * 1000;
    }

    int32_t passive_count, passive_slot;
    craftax_spawn_mobs3_count_and_empty(
        &state->passive_mobs, level, &passive_count, &passive_slot
    );
    CraftaxThreefryKey passive_prob_key = craftax_spawn_next_random_key(&rng);
    CraftaxThreefryKey passive_pos_key = craftax_spawn_next_random_key(&rng);
    int32_t passive_type = craftax_spawn_floor_mob_type(
        level, CRAFTAX_MOB_PASSIVE
    );
    state->passive_mobs.type_id[level][passive_slot] = passive_type;

    int32_t melee_count, melee_slot;
    craftax_spawn_mobs3_count_and_empty(
        &state->melee_mobs, level, &melee_count, &melee_slot
    );
    int32_t melee_type = fighting_boss
        ? craftax_spawn_floor_mob_type(state->boss_progress, CRAFTAX_MOB_MELEE)
        : craftax_spawn_floor_mob_type(level, CRAFTAX_MOB_MELEE);
    CraftaxThreefryKey melee_prob_key = craftax_spawn_next_random_key(&rng);
    float night_coeff = 1.0f - state->light_level;
    float melee_spawn_chance = craftax_spawn_floor_spawn_chance(level, 1)
        + craftax_spawn_floor_spawn_chance(level, 3) * night_coeff * night_coeff;
    CraftaxThreefryKey melee_pos_key = craftax_spawn_next_random_key(&rng);
    state->melee_mobs.type_id[level][melee_slot] = melee_type;

    int32_t ranged_count, ranged_slot;
    craftax_spawn_mobs2_count_and_empty(
        &state->ranged_mobs, level, &ranged_count, &ranged_slot
    );
    int32_t ranged_type = fighting_boss
        ? craftax_spawn_floor_mob_type(state->boss_progress, CRAFTAX_MOB_RANGED)
        : craftax_spawn_floor_mob_type(level, CRAFTAX_MOB_RANGED);
    CraftaxThreefryKey ranged_prob_key = craftax_spawn_next_random_key(&rng);
    CraftaxThreefryKey ranged_pos_key = craftax_spawn_next_random_key(&rng);
    state->ranged_mobs.type_id[level][ranged_slot] = ranged_type;

    bool try_passive = !fighting_boss
        && passive_count < CRAFTAX_MAX_PASSIVE_MOBS
        && craftax_threefry_uniform_f32(passive_prob_key)
            < craftax_spawn_floor_spawn_chance(level, 0);
    bool try_melee = melee_count < CRAFTAX_MAX_MELEE_MOBS
        && craftax_threefry_uniform_f32(melee_prob_key)
            < melee_spawn_chance * (float)monster_spawn_coeff;
    bool try_ranged = ranged_count < CRAFTAX_MAX_RANGED_MOBS
        && craftax_threefry_uniform_f32(ranged_prob_key)
            < craftax_spawn_floor_spawn_chance(level, 2)
                * (float)monster_spawn_coeff;

    if (!try_passive && !try_melee && !try_ranged) return;

    int32_t try_count = (int32_t)try_passive
        + (int32_t)try_melee
        + (int32_t)try_ranged;
    if (try_count == 1) {
        int32_t row, col;
        if (try_passive && craftax_spawn_scan_passive(
                state, level, passive_pos_key, &row, &col
        )) {
            state->passive_mobs.position[level][passive_slot][0] = row;
            state->passive_mobs.position[level][passive_slot][1] = col;
            state->passive_mobs.health[level][passive_slot] =
                craftax_spawn_mob_type_health(
                    passive_type, CRAFTAX_MOB_PASSIVE
                );
            state->passive_mobs.mask[level][passive_slot] = true;
            state->mob_bits[level][row] |= (1ULL << col);
        } else if (try_melee && craftax_spawn_scan_melee(
                state, level, fighting_boss, melee_pos_key, &row, &col
        )) {
            state->melee_mobs.position[level][melee_slot][0] = row;
            state->melee_mobs.position[level][melee_slot][1] = col;
            state->melee_mobs.health[level][melee_slot] =
                craftax_spawn_mob_type_health(melee_type, CRAFTAX_MOB_MELEE);
            state->melee_mobs.mask[level][melee_slot] = true;
            state->mob_bits[level][row] |= (1ULL << col);
        } else if (try_ranged && craftax_spawn_scan_ranged(
                state, level, ranged_type, fighting_boss, ranged_pos_key,
                &row, &col
        )) {
            state->ranged_mobs.position[level][ranged_slot][0] = row;
            state->ranged_mobs.position[level][ranged_slot][1] = col;
            state->ranged_mobs.health[level][ranged_slot] =
                craftax_spawn_mob_type_health(ranged_type, CRAFTAX_MOB_RANGED);
            state->ranged_mobs.mask[level][ranged_slot] = true;
            state->mob_bits[level][row] |= (1ULL << col);
        }
        return;
    }

    CraftaxSpawnLists lists;
    craftax_spawn_scan_all(
        state, level, ranged_type, fighting_boss,
        try_passive, try_melee, try_ranged, &lists
    );

    bool passive_spawned = false;
    int32_t passive_row = 0;
    int32_t passive_col = 0;
    if (try_passive && craftax_spawn_pick_excluding(
            lists.passive, lists.passive_count, passive_pos_key,
            false, 0, 0, false, 0, 0, &passive_row, &passive_col
        )) {
        state->passive_mobs.position[level][passive_slot][0] = passive_row;
        state->passive_mobs.position[level][passive_slot][1] = passive_col;
        state->passive_mobs.health[level][passive_slot] =
            craftax_spawn_mob_type_health(passive_type, CRAFTAX_MOB_PASSIVE);
        state->passive_mobs.mask[level][passive_slot] = true;
        state->mob_bits[level][passive_row] |= (1ULL << passive_col);
        passive_spawned = true;
    }

    bool melee_spawned = false;
    int32_t melee_row = 0;
    int32_t melee_col = 0;
    if (try_melee && craftax_spawn_pick_excluding(
            lists.melee, lists.melee_count, melee_pos_key,
            passive_spawned, passive_row, passive_col,
            false, 0, 0, &melee_row, &melee_col
        )) {
        state->melee_mobs.position[level][melee_slot][0] = melee_row;
        state->melee_mobs.position[level][melee_slot][1] = melee_col;
        state->melee_mobs.health[level][melee_slot] =
            craftax_spawn_mob_type_health(melee_type, CRAFTAX_MOB_MELEE);
        state->melee_mobs.mask[level][melee_slot] = true;
        state->mob_bits[level][melee_row] |= (1ULL << melee_col);
        melee_spawned = true;
    }

    int32_t ranged_row = 0;
    int32_t ranged_col = 0;
    if (try_ranged && craftax_spawn_pick_excluding(
            lists.ranged, lists.ranged_count, ranged_pos_key,
            passive_spawned, passive_row, passive_col,
            melee_spawned, melee_row, melee_col, &ranged_row, &ranged_col
        )) {
        state->ranged_mobs.position[level][ranged_slot][0] = ranged_row;
        state->ranged_mobs.position[level][ranged_slot][1] = ranged_col;
        state->ranged_mobs.health[level][ranged_slot] =
            craftax_spawn_mob_type_health(ranged_type, CRAFTAX_MOB_RANGED);
        state->ranged_mobs.mask[level][ranged_slot] = true;
        state->mob_bits[level][ranged_row] |= (1ULL << ranged_col);
    }
}

// ============================================================
// ===== CUDA host harness (replaces the pthread/OpenMP harness) =====
// ============================================================
// hash mode reproduces ./craftax_full hash exactly: same shared-arena env
// layout (env->rng = index, env->seed = seed + index), same deterministic
// per-env xorshift32 action stream, same FNV-1a over (obs, rewards, dones)
// each step including the post-reset observations.

#define CU_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, \
                __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while (0)

// token-identical copy of the C harness action RNG, as device code
static __device__ inline uint32_t cf_xorshift32(uint32_t* s) {
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x ? x : 0xdeadbeefu;
    return *s;
}

static inline uint64_t cf_fnv1a(uint64_t h, const void* data, size_t n) {
    const uint8_t* p = (const uint8_t*)data;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

static double cf_now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

__global__ void k_global_init(void) {
    craftax_wg_init_cell_templates();
    craftax_spawn_init_offsets_once();
}

// Mirrors cf_vec_init in craftax_full.c: shared arena, env->rng = index,
// env->seed = seed + index, pools disabled. [env_lo, env_hi) per launch so
// worldgen-heavy init kernels stay short (display GPU watchdog).
__global__ void k_env_init(
    Craftax* envs, CraftaxState* states, CraftaxObs* observations,
    float* actions, float* rewards, float* terminals, uint32_t* action_rng,
    int env_lo, int env_hi, uint64_t seed
) {
    int i = env_lo + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= env_hi) return;
    Craftax* env = &envs[i];
    env->num_agents = 1;
    env->rng = (unsigned int)i;
    env->seed = seed + (uint64_t)i;
    env->arena = NULL;
    env->state = &states[i];
    env->packet_id = i / CRAFTAX_ARENA_PACKET_SIZE;
    env->lane_id = i % CRAFTAX_ARENA_PACKET_SIZE;
    env->owns_state_storage = false;
    env->observations = &observations[(size_t)i * CRAFTAX_OBS_SIZE];
    env->actions = &actions[i];
    env->rewards = &rewards[i];
    env->terminals = &terminals[i];
    c_init(env);
    craftax_encode_native_observation(env->state, env->observations);
    // Deterministic per-env action stream, identical to the C harness.
    action_rng[i] = (uint32_t)(seed ^ ((uint64_t)i * 2654435761ULL)) | 1u;
}

__global__ void k_step(Craftax* envs, uint32_t* action_rng, int num_envs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_envs) return;
    Craftax* env = &envs[i];
    env->actions[0] =
        (float)(cf_xorshift32(&action_rng[i]) % CRAFTAX_NUM_ACTIONS);
    c_step(env);
}

__device__ int g_nan_flag = 0;

__global__ void k_nan_scan(
    const CraftaxObs* observations, const float* rewards,
    const float* terminals, int num_envs
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_envs) return;
    if (isnan(rewards[i]) || isinf(rewards[i])
        || isnan(terminals[i])) {
        g_nan_flag = 1;
    }
    const CraftaxObs* obs = &observations[(size_t)i * CRAFTAX_OBS_SIZE];
    for (int k = 0; k < CRAFTAX_OBS_SIZE; k++) {
        if (isnan((float)obs[k]) || isinf((float)obs[k])) {
            g_nan_flag = 1;
        }
    }
}

typedef struct {
    int num_envs;
    Craftax* d_envs;
    CraftaxState* d_states;
    CraftaxObs* d_obs;
    float* d_actions;
    float* d_rewards;
    float* d_terminals;
    uint32_t* d_action_rng;
    CraftaxObs* h_obs;
    float* h_rewards;
    float* h_terminals;
} CuVec;

static void cu_fill_light_table(void) {
    // Same formula as craftax_calculate_light_level_native, computed with
    // HOST libm cosf (glibc, as in the reference binary) and the gcc
    // -ffast-math expansion powf(x, 3.0f) -> x*x*x.
    static float table[CRAFTAX_DEFAULT_MAX_TIMESTEPS_TABLE];
    for (int t = 0; t < CRAFTAX_DEFAULT_MAX_TIMESTEPS_TABLE; t++) {
        // gcc -ffast-math turns t / 300.0f into t * (1.0f / 300.0f)
        float progress =
            fmodf((float)t * (1.0f / (float)CRAFTAX_DAY_LENGTH), 1.0f) + 0.3f;
        float c = cosf(CRAFTAX_WG_PI * progress);
        float a = fabsf(c);
        // gcc contracts 1.0f - a*a*a into vfnmadd (verified bit-exact
        // against the reference binary for all dumped timesteps)
        table[t] = fmaf(-(a * a), a, 1.0f);
    }
    CU_CHECK(cudaMemcpyToSymbol(g_craftax_light_table, table, sizeof(table)));
}

static void cu_vec_init(CuVec* v, int num_envs, uint64_t seed) {
    v->num_envs = num_envs;
    size_t stack_bytes = 132 << 10;  // k_step worst case (reset worldgen) ~120KB per ptxas -v
    const char* stack_env = getenv("CRAFTAX_CU_STACK");
    if (stack_env != NULL) stack_bytes = (size_t)atol(stack_env);
    CU_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, stack_bytes));

    CU_CHECK(cudaMalloc(&v->d_envs, (size_t)num_envs * sizeof(Craftax)));
    CU_CHECK(cudaMalloc(&v->d_states, (size_t)num_envs * sizeof(CraftaxState)));
    CU_CHECK(cudaMalloc(&v->d_obs,
                        (size_t)num_envs * CRAFTAX_OBS_SIZE * sizeof(CraftaxObs)));
    CU_CHECK(cudaMalloc(&v->d_actions, (size_t)num_envs * sizeof(float)));
    CU_CHECK(cudaMalloc(&v->d_rewards, (size_t)num_envs * sizeof(float)));
    CU_CHECK(cudaMalloc(&v->d_terminals, (size_t)num_envs * sizeof(float)));
    CU_CHECK(cudaMalloc(&v->d_action_rng, (size_t)num_envs * sizeof(uint32_t)));
    // calloc semantics of the C harness
    CU_CHECK(cudaMemset(v->d_envs, 0, (size_t)num_envs * sizeof(Craftax)));
    CU_CHECK(cudaMemset(v->d_states, 0, (size_t)num_envs * sizeof(CraftaxState)));
    CU_CHECK(cudaMemset(v->d_obs, 0,
                        (size_t)num_envs * CRAFTAX_OBS_SIZE * sizeof(CraftaxObs)));
    CU_CHECK(cudaMemset(v->d_actions, 0, (size_t)num_envs * sizeof(float)));
    CU_CHECK(cudaMemset(v->d_rewards, 0, (size_t)num_envs * sizeof(float)));
    CU_CHECK(cudaMemset(v->d_terminals, 0, (size_t)num_envs * sizeof(float)));

    v->h_obs = (CraftaxObs*)malloc(
        (size_t)num_envs * CRAFTAX_OBS_SIZE * sizeof(CraftaxObs));
    v->h_rewards = (float*)malloc((size_t)num_envs * sizeof(float));
    v->h_terminals = (float*)malloc((size_t)num_envs * sizeof(float));

    cu_fill_light_table();
    k_global_init<<<1, 1>>>();
    CU_CHECK(cudaDeviceSynchronize());

    int batch = 8192;  // bounded init launches (full 9-level worldgen each)
    for (int lo = 0; lo < num_envs; lo += batch) {
        int hi = lo + batch < num_envs ? lo + batch : num_envs;
        int n = hi - lo;
        k_env_init<<<(n + 63) / 64, 64>>>(
            v->d_envs, v->d_states, v->d_obs, v->d_actions, v->d_rewards,
            v->d_terminals, v->d_action_rng, lo, hi, seed);
        CU_CHECK(cudaDeviceSynchronize());
    }
}

static void cu_vec_free(CuVec* v) {
    cudaFree(v->d_envs); cudaFree(v->d_states); cudaFree(v->d_obs);
    cudaFree(v->d_actions); cudaFree(v->d_rewards); cudaFree(v->d_terminals);
    cudaFree(v->d_action_rng);
    free(v->h_obs); free(v->h_rewards); free(v->h_terminals);
}

static void cu_copy_back(CuVec* v, bool with_obs) {
    if (with_obs) {
        CU_CHECK(cudaMemcpy(
            v->h_obs, v->d_obs,
            (size_t)v->num_envs * CRAFTAX_OBS_SIZE * sizeof(CraftaxObs),
            cudaMemcpyDeviceToHost));
    }
    CU_CHECK(cudaMemcpy(v->h_rewards, v->d_rewards,
                        (size_t)v->num_envs * sizeof(float),
                        cudaMemcpyDeviceToHost));
    CU_CHECK(cudaMemcpy(v->h_terminals, v->d_terminals,
                        (size_t)v->num_envs * sizeof(float),
                        cudaMemcpyDeviceToHost));
}

static void cu_print_logs(CuVec* v, bool histogram) {
    Craftax* h_envs = (Craftax*)malloc((size_t)v->num_envs * sizeof(Craftax));
    CU_CHECK(cudaMemcpy(h_envs, v->d_envs,
                        (size_t)v->num_envs * sizeof(Craftax),
                        cudaMemcpyDeviceToHost));
    double episodes = 0.0, ep_len = 0.0, ep_ret = 0.0, ach = 0.0;
    double hist[CRAFTAX_NUM_ACHIEVEMENTS] = {0};
    for (int i = 0; i < v->num_envs; i++) {
        Log* log = &h_envs[i].log;
        episodes += (double)log->n;
        ep_len += (double)log->episode_length;
        ep_ret += (double)log->score;
        for (int a = 0; a < CRAFTAX_NUM_ACHIEVEMENTS; a++) {
            ach += (double)log->achievements[a];
            hist[a] += (double)log->achievements[a];
        }
    }
    printf("episodes_completed=%.0f mean_episode_length=%.2f\n", episodes,
           episodes > 0.0 ? ep_len / episodes : 0.0);
    printf("completed_episode_reward=%.3f total_achievements=%.0f\n",
           ep_ret, ach);
    if (histogram) {
        printf("achievement_histogram");
        for (int a = 0; a < CRAFTAX_NUM_ACHIEVEMENTS; a++) {
            printf(" %.0f", hist[a]);
        }
        printf("\n");
    }
    free(h_envs);
}

static void cu_step_launch(CuVec* v) {
    k_step<<<(v->num_envs + 63) / 64, 64>>>(
        v->d_envs, v->d_action_rng, v->num_envs);
}

// ------------------------------------------------------------
// hash mode: identical FNV-1a stream to ./craftax_full hash
// ------------------------------------------------------------
static int cu_run_hash(int num_envs, int num_steps, uint64_t seed) {
    CuVec v;
    cu_vec_init(&v, num_envs, seed);
    cu_copy_back(&v, true);

    uint64_t h = 0xcbf29ce484222325ULL;
    h = cf_fnv1a(h, v.h_obs,
                 (size_t)num_envs * CRAFTAX_OBS_SIZE * sizeof(CraftaxObs));

    double total_reward = 0.0;
    int checkpoint = num_steps >= 10 ? num_steps / 10 : num_steps;

    for (int step = 0; step < num_steps; step++) {
        cu_step_launch(&v);
        CU_CHECK(cudaDeviceSynchronize());
        cu_copy_back(&v, true);
        h = cf_fnv1a(h, v.h_obs,
                     (size_t)num_envs * CRAFTAX_OBS_SIZE * sizeof(CraftaxObs));
        h = cf_fnv1a(h, v.h_rewards, (size_t)num_envs * sizeof(float));
        h = cf_fnv1a(h, v.h_terminals, (size_t)num_envs * sizeof(float));
        for (int i = 0; i < num_envs; i++) total_reward += (double)v.h_rewards[i];
        if ((step + 1) % checkpoint == 0 || step + 1 == num_steps) {
            printf("step %6d  hash 0x%016llx\n", step + 1, (unsigned long long)h);
        }
    }

    printf("trajectory_hash 0x%016llx\n", (unsigned long long)h);
    printf("envs=%d steps=%d seed=%llu\n", num_envs, num_steps,
           (unsigned long long)seed);
    printf("total_reward=%.3f\n", total_reward);
    cu_print_logs(&v, false);
    cu_vec_free(&v);
    return 0;
}

// ------------------------------------------------------------
// cmp mode: step in lockstep with a CPU dump file (from the scratchpad
// cpu_ref dump tool) and report the first diverging elements.
// Dump layout: int32 magic, num_envs, num_steps, obs_size;
//   initial obs block; then per step: obs, rewards, terminals.
// ------------------------------------------------------------
static void cu_decode_obs_index(int idx, char* buf, size_t bufsz) {
    if (idx < CRAFTAX_WG_PACKED_MAP_OBS_SIZE) {
        int cell = idx / CRAFTAX_WG_PACKED_CHANNELS_PER_CELL;
        int ch = idx % CRAFTAX_WG_PACKED_CHANNELS_PER_CELL;
        static const char* chname[8] = {
            "block_id", "item_id", "light", "mob_melee", "mob_passive",
            "mob_ranged", "mob_projectile", "player_projectile"};
        snprintf(buf, bufsz, "map cell (%d,%d) %s",
                 cell / CRAFTAX_WG_OBS_COLS, cell % CRAFTAX_WG_OBS_COLS,
                 chname[ch]);
    } else {
        snprintf(buf, bufsz, "scalar tail[%d]",
                 idx - CRAFTAX_WG_PACKED_MAP_OBS_SIZE);
    }
}

static int cu_compare_block(
    int step, int num_envs, const CraftaxObs* cpu_obs, const CraftaxObs* gpu_obs,
    const float* cpu_r, const float* gpu_r, const float* cpu_t,
    const float* gpu_t, int* reported, int max_report
) {
    int diverged = 0;
    for (int i = 0; i < num_envs; i++) {
        if (cpu_r != NULL) {
            uint32_t a, b;
            memcpy(&a, &cpu_r[i], 4); memcpy(&b, &gpu_r[i], 4);
            if (a != b && (*reported)++ < max_report) {
                printf("DIV step %d env %d reward cpu=%g(0x%08x) gpu=%g(0x%08x)\n",
                       step, i, cpu_r[i], a, gpu_r[i], b);
            }
            diverged |= (a != b);
            memcpy(&a, &cpu_t[i], 4); memcpy(&b, &gpu_t[i], 4);
            if (a != b && (*reported)++ < max_report) {
                printf("DIV step %d env %d terminal cpu=%g gpu=%g\n",
                       step, i, cpu_t[i], gpu_t[i]);
            }
            diverged |= (a != b);
        }
        for (int k = 0; k < CRAFTAX_OBS_SIZE; k++) {
            size_t off = (size_t)i * CRAFTAX_OBS_SIZE + k;
            uint32_t a, b;
            memcpy(&a, &cpu_obs[off], 4); memcpy(&b, &gpu_obs[off], 4);
            if (a != b) {
                diverged = 1;
                if ((*reported)++ < max_report) {
                    char what[96];
                    cu_decode_obs_index(k, what, sizeof(what));
                    printf("DIV step %d env %d obs[%d] (%s) cpu=%.9g(0x%08x) "
                           "gpu=%.9g(0x%08x)\n", step, i, k, what,
                           (double)cpu_obs[off], a, (double)gpu_obs[off], b);
                }
            }
        }
    }
    return diverged;
}

static int cu_run_cmp(const char* dump_path, uint64_t seed, int max_report) {
    FILE* f = fopen(dump_path, "rb");
    if (f == NULL) { fprintf(stderr, "cannot open %s\n", dump_path); return 1; }
    int32_t hdr[4];
    if (fread(hdr, sizeof(hdr), 1, f) != 1) { fprintf(stderr, "bad dump\n"); return 1; }
    if (hdr[0] != 0x43465231 || hdr[3] != CRAFTAX_OBS_SIZE) {
        fprintf(stderr, "dump header mismatch (magic 0x%x obs %d)\n", hdr[0], hdr[3]);
        return 1;
    }
    int num_envs = hdr[1], num_steps = hdr[2];
    printf("cmp: %d envs x %d steps from %s\n", num_envs, num_steps, dump_path);

    CuVec v;
    cu_vec_init(&v, num_envs, seed);
    cu_copy_back(&v, true);

    size_t obs_n = (size_t)num_envs * CRAFTAX_OBS_SIZE;
    CraftaxObs* cpu_obs = (CraftaxObs*)malloc(obs_n * sizeof(CraftaxObs));
    float* cpu_r = (float*)malloc((size_t)num_envs * sizeof(float));
    float* cpu_t = (float*)malloc((size_t)num_envs * sizeof(float));

    int reported = 0;
    int first_div_step = -1;
    if (fread(cpu_obs, sizeof(CraftaxObs), obs_n, f) != obs_n) return 1;
    if (cu_compare_block(-1, num_envs, cpu_obs, v.h_obs, NULL, NULL, NULL,
                         NULL, &reported, max_report)) {
        first_div_step = -1;
        printf("first divergence: initial (post-reset) observations\n");
    }

    int step = 0;
    for (; step < num_steps; step++) {
        cu_step_launch(&v);
        CU_CHECK(cudaDeviceSynchronize());
        cu_copy_back(&v, true);
        if (fread(cpu_obs, sizeof(CraftaxObs), obs_n, f) != obs_n) break;
        if (fread(cpu_r, sizeof(float), num_envs, f) != (size_t)num_envs) break;
        if (fread(cpu_t, sizeof(float), num_envs, f) != (size_t)num_envs) break;
        int div = cu_compare_block(step, num_envs, cpu_obs, v.h_obs, cpu_r,
                                   v.h_rewards, cpu_t, v.h_terminals,
                                   &reported, max_report);
        if (div && first_div_step < 0) first_div_step = step;
        if (reported >= max_report) {
            printf("... stopping after %d reported divergences (step %d)\n",
                   reported, step);
            break;
        }
    }
    if (reported == 0) {
        printf("NO divergence in %d envs x %d steps (bit-exact)\n",
               num_envs, step);
    } else {
        printf("first_divergence_step=%d reported=%d\n", first_div_step, reported);
    }
    fclose(f); free(cpu_obs); free(cpu_r); free(cpu_t);
    cu_vec_free(&v);
    return 0;
}

// ------------------------------------------------------------
// stats mode: distributional battery (episodes, lengths, rewards,
// per-achievement histogram) + NaN scan, no hashing.
// ------------------------------------------------------------
static int cu_run_stats(int num_envs, int num_steps, uint64_t seed) {
    CuVec v;
    cu_vec_init(&v, num_envs, seed);
    double total_reward = 0.0;
    for (int step = 0; step < num_steps; step++) {
        cu_step_launch(&v);
        k_nan_scan<<<(num_envs + 63) / 64, 64>>>(
            v.d_obs, v.d_rewards, v.d_terminals, num_envs);
        CU_CHECK(cudaMemcpy(v.h_rewards, v.d_rewards,
                            (size_t)num_envs * sizeof(float),
                            cudaMemcpyDeviceToHost));
        for (int i = 0; i < num_envs; i++)
            total_reward += (double)v.h_rewards[i];
    }
    CU_CHECK(cudaDeviceSynchronize());
    int nan_flag = 0;
    CU_CHECK(cudaMemcpyFromSymbol(&nan_flag, g_nan_flag, sizeof(int)));
    printf("envs=%d steps=%d seed=%llu\n", num_envs, num_steps,
           (unsigned long long)seed);
    printf("total_reward=%.3f nan_or_inf=%d\n", total_reward, nan_flag);
    cu_print_logs(&v, true);
    cu_vec_free(&v);
    return 0;
}

// ------------------------------------------------------------
// bench mode: pure device stepping, no per-step copies.
// ------------------------------------------------------------
static int cu_run_bench(int num_envs, int iters, uint64_t seed) {
    double t_init0 = cf_now_s();
    CuVec v;
    cu_vec_init(&v, num_envs, seed);
    double t_init = cf_now_s() - t_init0;

    int warmup = iters / 20 > 10 ? 10 : (iters / 20 > 0 ? iters / 20 : 1);
    for (int k = 0; k < warmup; k++) cu_step_launch(&v);
    CU_CHECK(cudaDeviceSynchronize());

    double t0 = cf_now_s();
    for (int k = 0; k < iters; k++) cu_step_launch(&v);
    CU_CHECK(cudaDeviceSynchronize());
    double dt = cf_now_s() - t0;
    double sps = (double)num_envs * (double)iters / dt;
    printf("envs=%d iters=%d\n", num_envs, iters);
    printf("init %.3fs  bench %.3fs  SPS=%12.0f  (%.2f us/step/env)\n",
           t_init, dt, sps, dt / (double)iters / (double)num_envs * 1e6);
    cu_print_logs(&v, false);
    cu_vec_free(&v);
    return 0;
}

static void cu_usage(const char* prog) {
    fprintf(stderr,
            "usage: %s hash  [--envs N] [--steps M] [--seed S]\n"
            "       %s cmp   --dump FILE [--seed S] [--max-report K]\n"
            "       %s stats [--envs N] [--steps M] [--seed S]\n"
            "       %s bench [--envs N] [--iters M] [--seed S]\n",
            prog, prog, prog, prog);
}

int main(int argc, char** argv) {
    if (argc < 2) { cu_usage(argv[0]); return 1; }
    const char* mode = argv[1];
    int envs = !strcmp(mode, "bench") ? 8192 : 64;
    int iters = 1000;
    int steps = 2000;
    uint64_t seed = 42;
    const char* dump = NULL;
    int max_report = 40;

    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--envs") && i + 1 < argc) envs = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--dump") && i + 1 < argc) dump = argv[++i];
        else if (!strcmp(argv[i], "--max-report") && i + 1 < argc) max_report = atoi(argv[++i]);
        else { cu_usage(argv[0]); return 1; }
    }
    if (envs <= 0 || iters <= 0 || steps <= 0) { cu_usage(argv[0]); return 1; }

    if (!strcmp(mode, "hash")) return cu_run_hash(envs, steps, seed);
    if (!strcmp(mode, "cmp")) {
        if (dump == NULL) { cu_usage(argv[0]); return 1; }
        return cu_run_cmp(dump, seed, max_report);
    }
    if (!strcmp(mode, "stats")) return cu_run_stats(envs, steps, seed);
    if (!strcmp(mode, "bench")) return cu_run_bench(envs, iters, seed);
    cu_usage(argv[0]);
    return 1;
}
