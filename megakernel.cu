// Fused env + policy rollout megakernel -- pure C/CUDA, no torch.
//
// Policy is the exact PufferLib default for craftax (config/craftax.ini):
//   DefaultEncoder: Linear(1345 -> 32), no activation
//   MinGRU (1 layer, hidden 32): z = W_gru h  (96 outputs, no bias)
//     hidden,gate,proj = chunk(z); out = lerp(state, g(hidden), sig(gate))
//     h' = sig(proj)*out + (1-sig(proj))*h;  state <- out
//   DefaultDecoder: logits = Linear(32 -> 17), value = Linear(32 -> 1)
//
// The encoder input is mostly one-hot (63 cells x (17 blk one-hot + 4 mob
// flags)), so Linear(obs) collapses to a column gather from W_enc. Skipped
// terms are exact float zeros, so the fused encoder is bit-identical to the
// dense reference (verified in `megaverify`).
//
// The megakernel runs a full T-step rollout per launch: one thread per env,
// blocks fully independent (no grid sync; the only cross-thread traffic is
// warp-cooperative worldgen for done lanes). Rollout buffers (compact obs,
// action, logprob, value, reward, done) are written per step for PPO.
//
// Build:  nvcc -O3 -arch=native --expt-relaxed-constexpr --use_fast_math \
//              megakernel.cu -o megakernel
// Usage:
//   ./megakernel mega      [envs] [T] [iters]   megakernel rollout SPS
//   ./megakernel split     [envs] [T] [iters]   same math, per-step kernels
//   ./megakernel verify    [envs] [steps]       fused-vs-dense forward exact,
//                                               mega-vs-split rollout bitwise
//   ./megakernel sweep     [T]                  env-count sweep, mega vs split

#include "craftax.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>

#include "craftax.cu"

#define CUDA_CHECK(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    exit(1); } } while (0)

#define HIDDEN 32
#define GRU_OUT (3 * HIDDEN)
#define MEGA_BLOCK 128

// ============================================================
// Policy weights (device pointers, passed to kernels by value)
// ============================================================
struct Weights {
    const float* __restrict__ W_enc;  // [OBS_DIM][HIDDEN] row per input feature
    const float* __restrict__ b_enc;  // [HIDDEN]
    const float* __restrict__ W_gru;  // [GRU_OUT][HIDDEN]
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

// Small weights staged in shared memory once per block.
struct NNShared {
    float W_gru[GRU_OUT * HIDDEN];
    float W_a[NUM_ACTIONS * HIDDEN];
    float b_a[NUM_ACTIONS];
    float W_v[HIDDEN];
    float b_enc[HIDDEN];
    float b_v;
};

__device__ void load_nn_shared(NNShared& s, const Weights& w) {
    for (int i = threadIdx.x; i < GRU_OUT * HIDDEN; i += blockDim.x) s.W_gru[i] = w.W_gru[i];
    for (int i = threadIdx.x; i < NUM_ACTIONS * HIDDEN; i += blockDim.x) s.W_a[i] = w.W_a[i];
    for (int i = threadIdx.x; i < NUM_ACTIONS; i += blockDim.x) s.b_a[i] = w.b_a[i];
    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) s.W_v[i] = w.W_v[i];
    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) s.b_enc[i] = w.b_enc[i];
    if (threadIdx.x == 0) s.b_v = w.b_v[0];
    __syncthreads();
}

// ============================================================
// Fused encoder: Linear(1345 float obs -> 32) computed directly
// from the gathered view + env state, in the SAME feature order
// as the dense loop, skipping only exact-zero terms.
// ============================================================
__device__ void fused_encoder(
    const EnvSoA& g, int e,
    const int8_t* view_blk, const uint8_t* view_mob,
    const Weights& w, const NNShared& s, float* h  // h[HIDDEN]
) {
    const int n = g.n;
    #pragma unroll
    for (int i = 0; i < HIDDEN; i++) h[i] = s.b_enc[i];

    for (int cell = 0; cell < OBS_MAP_CELLS; cell++) {
        int f = cell * (NUM_BLOCK_TYPES + 4) + view_blk[cell];
        const float* col = w.W_enc + (size_t)f * HIDDEN;
        #pragma unroll
        for (int i = 0; i < HIDDEN; i++) h[i] = fmaf(1.0f, col[i], h[i]);
        uint8_t m = view_mob[cell];
        int fm = cell * (NUM_BLOCK_TYPES + 4) + NUM_BLOCK_TYPES;
        for (int k = 0; k < 4; k++) {
            if (m & (1 << k)) {
                const float* mc = w.W_enc + (size_t)(fm + k) * HIDDEN;
                #pragma unroll
                for (int i = 0; i < HIDDEN; i++) h[i] = fmaf(1.0f, mc[i], h[i]);
            }
        }
    }

    int f = OBS_MAP_CELLS * (NUM_BLOCK_TYPES + 4);
    for (int j = 0; j < NUM_INVENTORY; j++, f++) {
        float x = (float)g.inv[j*n+e] / 10.0f;
        const float* col = w.W_enc + (size_t)f * HIDDEN;
        #pragma unroll
        for (int i = 0; i < HIDDEN; i++) h[i] = fmaf(x, col[i], h[i]);
    }
    float intr[4] = {
        (float)g.health[e] / 10.0f, (float)g.food[e] / 10.0f,
        (float)g.drink[e] / 10.0f, (float)g.energy[e] / 10.0f
    };
    for (int j = 0; j < 4; j++, f++) {
        const float* col = w.W_enc + (size_t)f * HIDDEN;
        #pragma unroll
        for (int i = 0; i < HIDDEN; i++) h[i] = fmaf(intr[j], col[i], h[i]);
    }
    for (int d = 1; d <= 4; d++, f++) {
        float x = (g.player_dir[e] == d) ? 1.0f : 0.0f;
        const float* col = w.W_enc + (size_t)f * HIDDEN;
        #pragma unroll
        for (int i = 0; i < HIDDEN; i++) h[i] = fmaf(x, col[i], h[i]);
    }
    {
        const float* col = w.W_enc + (size_t)f * HIDDEN; f++;
        float x = g.light_level[e];
        #pragma unroll
        for (int i = 0; i < HIDDEN; i++) h[i] = fmaf(x, col[i], h[i]);
    }
    {
        const float* col = w.W_enc + (size_t)f * HIDDEN;
        float x = g.is_sleeping[e] ? 1.0f : 0.0f;
        #pragma unroll
        for (int i = 0; i < HIDDEN; i++) h[i] = fmaf(x, col[i], h[i]);
    }
}

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float mingru_g(float x) { return x >= 0.0f ? x + 0.5f : sigmoidf_(x); }

// MinGRU cell + decoder heads. state in/out; logits[NUM_ACTIONS], value.
__device__ void nn_head(
    const float* h_enc, float* state, const NNShared& s,
    float* logits, float& value
) {
    float hidden[HIDDEN], gate[HIDDEN], proj[HIDDEN];
    #pragma unroll
    for (int k = 0; k < HIDDEN; k++) {
        float zh = 0.0f, zg = 0.0f, zp = 0.0f;
        #pragma unroll
        for (int j = 0; j < HIDDEN; j++) {
            zh = fmaf(s.W_gru[(k) * HIDDEN + j], h_enc[j], zh);
            zg = fmaf(s.W_gru[(HIDDEN + k) * HIDDEN + j], h_enc[j], zg);
            zp = fmaf(s.W_gru[(2*HIDDEN + k) * HIDDEN + j], h_enc[j], zp);
        }
        hidden[k] = zh; gate[k] = zg; proj[k] = zp;
    }
    float hout[HIDDEN];
    #pragma unroll
    for (int k = 0; k < HIDDEN; k++) {
        // torch.lerp(start, end, weight) = start + weight*(end-start)
        float out = state[k] + sigmoidf_(gate[k]) * (mingru_g(hidden[k]) - state[k]);
        float p = sigmoidf_(proj[k]);
        hout[k] = p * out + (1.0f - p) * h_enc[k];
        state[k] = out;
    }
    for (int a = 0; a < NUM_ACTIONS; a++) {
        float z = s.b_a[a];
        #pragma unroll
        for (int j = 0; j < HIDDEN; j++) z = fmaf(s.W_a[a * HIDDEN + j], hout[j], z);
        logits[a] = z;
    }
    float v = s.b_v;
    #pragma unroll
    for (int j = 0; j < HIDDEN; j++) v = fmaf(s.W_v[j], hout[j], v);
    value = v;
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

// Write compact obs from an already-gathered view (same bytes as
// build_observation_compact).
__device__ void write_compact_obs(
    const EnvSoA& g, int e,
    const int8_t* view_blk, const uint8_t* view_mob, uint8_t* obs
) {
    const int n = g.n;
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
}

// ============================================================
// Rollout megakernel: T steps, one thread per env, no grid sync.
// step0 is the global step count at entry (sampler stream offset
// and reset-seed schedule stay aligned with the split path).
// ============================================================
extern "C" __global__ void rollout_kernel(
    EnvSoA g, Weights w, float* __restrict__ h_state,
    uint8_t* __restrict__ r_obs, int32_t* __restrict__ r_act,
    float* __restrict__ r_logprob, float* __restrict__ r_value,
    float* __restrict__ r_reward, int8_t* __restrict__ r_done,
    int num_envs, int T, uint64_t seed, uint64_t step0
) {
    __shared__ NNShared snn;
    __shared__ float angles[MEGA_BLOCK / 32][WG_ANGLE_COUNT];
    load_nn_shared(snn, w);

    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_envs) return;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    curandStatePhilox4_32_10_t sampler;
    curand_init(seed ^ 0xA5A5A5A5A5A5A5A5ULL, e, step0, &sampler);

    float state[HIDDEN];
    #pragma unroll
    for (int i = 0; i < HIDDEN; i++) state[i] = h_state[(size_t)i * num_envs + e];
    bool was_done = false;

    for (int t = 0; t < T; t++) {
        int8_t view_blk[OBS_MAP_CELLS];
        uint8_t view_mob[OBS_MAP_CELLS];
        gather_view(g, e, view_blk, view_mob);
        write_compact_obs(g, e, view_blk, view_mob,
                          r_obs + ((size_t)t * num_envs + e) * OBS_DIM_COMPACT);

        if (was_done) {
            #pragma unroll
            for (int i = 0; i < HIDDEN; i++) state[i] = 0.0f;
        }
        float h_enc[HIDDEN], logits[NUM_ACTIONS], value, logprob;
        fused_encoder(g, e, view_blk, view_mob, w, snn, h_enc);
        nn_head(h_enc, state, snn, logits, value);
        int action = sample_action(logits, curand_uniform(&sampler), &logprob);

        size_t o = (size_t)t * num_envs + e;
        r_act[o] = action; r_logprob[o] = logprob; r_value[o] = value;

        bool done = step_env(g, e, action, r_reward + (size_t)t * num_envs);
        r_done[o] = done ? 1 : 0;
        was_done = done;

        // Warp-cooperative reset of done lanes; stalls stay in this warp.
        unsigned mask = __ballot_sync(0xFFFFFFFFu, done);
        uint64_t reset_seed = seed + (step0 + t + 1) * 1000000ULL;
        while (mask) {
            int l = __ffs(mask) - 1;
            mask &= mask - 1;
            int de = e - lane + l;
            generate_world_warp(g, de, reset_seed, de + num_envs, angles[warp_in_block]);
        }
        __syncwarp();
    }

    #pragma unroll
    for (int i = 0; i < HIDDEN; i++) h_state[(size_t)i * num_envs + e] = state[i];
}

// ============================================================
// Split-path forward kernel: identical math, one step per launch.
// Reads dones from the previous step to zero recurrent state.
// ============================================================
extern "C" __global__ void fused_forward_kernel(
    EnvSoA g, Weights w, float* __restrict__ h_state,
    const int8_t* __restrict__ prev_dones,
    uint8_t* __restrict__ r_obs, int32_t* __restrict__ r_act,
    float* __restrict__ r_logprob, float* __restrict__ r_value,
    int num_envs, uint64_t seed, uint64_t step_count
) {
    __shared__ NNShared snn;
    load_nn_shared(snn, w);
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_envs) return;

    int8_t view_blk[OBS_MAP_CELLS];
    uint8_t view_mob[OBS_MAP_CELLS];
    gather_view(g, e, view_blk, view_mob);
    write_compact_obs(g, e, view_blk, view_mob, r_obs + (size_t)e * OBS_DIM_COMPACT);

    float state[HIDDEN];
    if (prev_dones && prev_dones[e]) {
        #pragma unroll
        for (int i = 0; i < HIDDEN; i++) state[i] = 0.0f;
    } else {
        #pragma unroll
        for (int i = 0; i < HIDDEN; i++) state[i] = h_state[(size_t)i * num_envs + e];
    }

    float h_enc[HIDDEN], logits[NUM_ACTIONS], value, logprob;
    fused_encoder(g, e, view_blk, view_mob, w, snn, h_enc);
    nn_head(h_enc, state, snn, logits, value);

    curandStatePhilox4_32_10_t sampler;
    curand_init(seed ^ 0xA5A5A5A5A5A5A5A5ULL, e, step_count, &sampler);
    int action = sample_action(logits, curand_uniform(&sampler), &logprob);

    r_act[e] = action; r_logprob[e] = logprob; r_value[e] = value;
    #pragma unroll
    for (int i = 0; i < HIDDEN; i++) h_state[(size_t)i * num_envs + e] = state[i];
}

// Dense reference forward: same math from the materialized 1345-float
// obs, dense feature loop (no gather). Used only by `verify`.
extern "C" __global__ void ref_forward_kernel(
    Weights w, const float* __restrict__ obs, float* __restrict__ h_state,
    const int8_t* __restrict__ prev_dones,
    int32_t* __restrict__ r_act, float* __restrict__ r_logprob,
    float* __restrict__ r_value,
    int num_envs, uint64_t seed, uint64_t step_count
) {
    __shared__ NNShared snn;
    load_nn_shared(snn, w);
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_envs) return;

    const float* x = obs + (size_t)e * OBS_DIM;
    float h_enc[HIDDEN];
    #pragma unroll
    for (int i = 0; i < HIDDEN; i++) h_enc[i] = snn.b_enc[i];
    for (int f = 0; f < OBS_DIM; f++) {
        float xf = x[f];
        if (xf == 0.0f) continue;  // exact no-op terms
        const float* col = w.W_enc + (size_t)f * HIDDEN;
        #pragma unroll
        for (int i = 0; i < HIDDEN; i++) h_enc[i] = fmaf(xf, col[i], h_enc[i]);
    }

    float state[HIDDEN];
    if (prev_dones && prev_dones[e]) {
        #pragma unroll
        for (int i = 0; i < HIDDEN; i++) state[i] = 0.0f;
    } else {
        #pragma unroll
        for (int i = 0; i < HIDDEN; i++) state[i] = h_state[(size_t)i * num_envs + e];
    }
    float logits[NUM_ACTIONS], value, logprob;
    nn_head(h_enc, state, snn, logits, value);

    curandStatePhilox4_32_10_t sampler;
    curand_init(seed ^ 0xA5A5A5A5A5A5A5A5ULL, e, step_count, &sampler);
    int action = sample_action(logits, curand_uniform(&sampler), &logprob);

    r_act[e] = action; r_logprob[e] = logprob; r_value[e] = value;
    #pragma unroll
    for (int i = 0; i < HIDDEN; i++) h_state[(size_t)i * num_envs + e] = state[i];
}

// ============================================================
// Host side
// ============================================================
static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

static uint64_t fnv1a(uint64_t h, const void* data, size_t len) {
    const uint8_t* p = (const uint8_t*)data;
    for (size_t i = 0; i < len; i++) { h ^= p[i]; h *= 0x100000001B3ULL; }
    return h;
}

static float* alloc_weights(int count, float bound, uint64_t seed, int subseq) {
    float* w;
    CUDA_CHECK(cudaMalloc(&w, (size_t)count * sizeof(float)));
    init_weights_kernel<<<(count + 255) / 256, 256>>>(w, count, bound, seed, subseq);
    CUDA_CHECK(cudaGetLastError());
    return w;
}

struct Rollout {
    EnvSoA g;
    uint8_t* arena;
    Weights w;
    float* h_state;
    uint8_t* r_obs;
    int32_t* r_act;
    float* r_logprob;
    float* r_value;
    float* r_reward;
    int8_t* r_done;
    int32_t* reset_list;
    int32_t* reset_ctrl;
    int n, T;
    uint64_t seed;
    uint64_t step_count = 0;

    Rollout(int num_envs, int horizon, uint64_t seed_) : n(num_envs), T(horizon), seed(seed_) {
        if (n % 32 != 0) {
            fprintf(stderr, "num_envs must be a multiple of 32 (warp collectives)\n");
            exit(1);
        }
        CUDA_CHECK(cudaMalloc(&arena, soa_bytes(n)));
        CUDA_CHECK(cudaMemset(arena, 0, soa_bytes(n)));
        g = carve_soa(arena, n);

        float be = 1.0f / sqrtf((float)OBS_DIM);
        float bh = 1.0f / sqrtf((float)HIDDEN);
        w.W_enc = alloc_weights(OBS_DIM * HIDDEN, be, 1234, 0);
        w.b_enc = alloc_weights(HIDDEN, be, 1234, 1);
        w.W_gru = alloc_weights(GRU_OUT * HIDDEN, bh, 1234, 2);
        w.W_a   = alloc_weights(NUM_ACTIONS * HIDDEN, bh, 1234, 3);
        w.b_a   = alloc_weights(NUM_ACTIONS, bh, 1234, 4);
        w.W_v   = alloc_weights(HIDDEN, bh, 1234, 5);
        w.b_v   = alloc_weights(1, bh, 1234, 6);

        CUDA_CHECK(cudaMalloc(&h_state, (size_t)HIDDEN * n * sizeof(float)));
        CUDA_CHECK(cudaMemset(h_state, 0, (size_t)HIDDEN * n * sizeof(float)));

        size_t nt = (size_t)n * T;
        CUDA_CHECK(cudaMalloc(&r_obs, nt * OBS_DIM_COMPACT));
        CUDA_CHECK(cudaMalloc(&r_act, nt * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&r_logprob, nt * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&r_value, nt * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&r_reward, nt * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&r_done, nt));
        CUDA_CHECK(cudaMemset(r_done, 0, nt));
        CUDA_CHECK(cudaMalloc(&reset_list, n * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&reset_ctrl, 2 * sizeof(int32_t)));
    }
    ~Rollout() {
        cudaFree(arena); cudaFree(h_state);
        cudaFree((void*)w.W_enc); cudaFree((void*)w.b_enc); cudaFree((void*)w.W_gru);
        cudaFree((void*)w.W_a); cudaFree((void*)w.b_a); cudaFree((void*)w.W_v);
        cudaFree((void*)w.b_v);
        cudaFree(r_obs); cudaFree(r_act); cudaFree(r_logprob); cudaFree(r_value);
        cudaFree(r_reward); cudaFree(r_done);
        cudaFree(reset_list); cudaFree(reset_ctrl);
    }

    void reset() {
        int wgrid = ((size_t)n * 32 + RESET_WARP_BLOCK - 1) / RESET_WARP_BLOCK;
        reset_all_warp_kernel<<<wgrid, RESET_WARP_BLOCK>>>(g, n, seed);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemset(h_state, 0, (size_t)HIDDEN * n * sizeof(float)));
        step_count = 0;
    }

    void run_mega() {
        int grid = (n + MEGA_BLOCK - 1) / MEGA_BLOCK;
        rollout_kernel<<<grid, MEGA_BLOCK>>>(
            g, w, h_state, r_obs, r_act, r_logprob, r_value, r_reward, r_done,
            n, T, seed, step_count);
        CUDA_CHECK(cudaGetLastError());
        step_count += T;
    }

    // Same rollout via per-step kernels (forward, step+mark, warp reset).
    void run_split() {
        int block = 256, grid = (n + block - 1) / block;
        int warps_per_block = RESET_WARP_BLOCK / 32;
        int wgrid = (n + warps_per_block - 1) / warps_per_block;
        if (wgrid > 512) wgrid = 512;
        for (int t = 0; t < T; t++) {
            size_t o = (size_t)t * n;
            // r_done[t-1] row feeds recurrent-state zeroing; step 0 within
            // this rollout uses the last row of the previous rollout, which
            // we do not keep -- so split mode is only bitwise-comparable to
            // mega when starting from reset (prev_dones all zero), which is
            // how verify uses it.
            const int8_t* prev = (t == 0) ? nullptr : r_done + (size_t)(t-1) * n;
            fused_forward_kernel<<<grid, block>>>(
                g, w, h_state, prev, r_obs + o * OBS_DIM_COMPACT,
                r_act + o, r_logprob + o, r_value + o, n, seed, step_count);
            CUDA_CHECK(cudaMemsetAsync(reset_ctrl, 0, 2 * sizeof(int32_t)));
            step_mark_kernel<<<grid, block>>>(
                g, r_act + o, r_reward + o, r_done + o, reset_list, reset_ctrl, n);
            uint64_t reset_seed = seed + (step_count + 1) * 1000000ULL;
            reset_warp_kernel<<<wgrid, RESET_WARP_BLOCK>>>(g, reset_list, reset_ctrl, n, reset_seed);
            CUDA_CHECK(cudaGetLastError());
            step_count++;
        }
    }
};

static void run_bench(int num_envs, int T, int iters, bool mega) {
    // Rollout buffers are n*T*(148+13+4) bytes plus env state; skip
    // configs that do not fit.
    size_t need = (size_t)num_envs * T * (OBS_DIM_COMPACT + 17) + soa_bytes(num_envs);
    size_t free_b, total_b;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    if (need + (512ull << 20) > free_b) {
        printf("NE=%8d T=%d %s: skipped (needs %.1f GB, %.1f GB free)\n",
               num_envs, T, mega ? "mega " : "split", need / 1e9, free_b / 1e9);
        return;
    }
    Rollout r(num_envs, T, 42);
    r.reset();
    for (int i = 0; i < 3; i++) mega ? r.run_mega() : r.run_split();
    CUDA_CHECK(cudaDeviceSynchronize());
    double t0 = now_s();
    for (int i = 0; i < iters; i++) mega ? r.run_mega() : r.run_split();
    CUDA_CHECK(cudaDeviceSynchronize());
    double dt = now_s() - t0;
    double sps = (double)num_envs * T * iters / dt;
    printf("NE=%8d T=%d %s: %8.1f M SPS  (%.1f us/env-step-row)\n",
           num_envs, T, mega ? "mega " : "split", sps / 1e6, dt / (iters * (double)T) * 1e6);
}

static uint64_t rollout_hash(Rollout& r) {
    size_t nt = (size_t)r.n * r.T;
    std::vector<uint8_t> obs(nt * OBS_DIM_COMPACT);
    std::vector<int32_t> act(nt);
    std::vector<float> lp(nt), val(nt), rew(nt);
    std::vector<int8_t> done(nt);
    CUDA_CHECK(cudaMemcpy(obs.data(), r.r_obs, obs.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(act.data(), r.r_act, nt * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(lp.data(), r.r_logprob, nt * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(val.data(), r.r_value, nt * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(rew.data(), r.r_reward, nt * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(done.data(), r.r_done, nt, cudaMemcpyDeviceToHost));
    uint64_t h = 0xcbf29ce484222325ULL;
    h = fnv1a(h, obs.data(), obs.size());
    h = fnv1a(h, act.data(), nt * 4);
    h = fnv1a(h, lp.data(), nt * 4);
    h = fnv1a(h, val.data(), nt * 4);
    h = fnv1a(h, rew.data(), nt * 4);
    h = fnv1a(h, done.data(), nt);
    return h;
}

static int run_verify(int num_envs, int steps) {
    int fail = 0;

    // Check 1: fused (gather) forward == dense reference forward, bitwise,
    // along a real trajectory driven by the fused path's sampled actions.
    {
        Rollout r(num_envs, 1, 42);
        r.reset();
        float* obs_f;
        float* h_state_ref;
        int32_t* act_ref; float* lp_ref; float* val_ref;
        CUDA_CHECK(cudaMalloc(&obs_f, (size_t)num_envs * OBS_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_state_ref, (size_t)HIDDEN * num_envs * sizeof(float)));
        CUDA_CHECK(cudaMemset(h_state_ref, 0, (size_t)HIDDEN * num_envs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&act_ref, num_envs * 4));
        CUDA_CHECK(cudaMalloc(&lp_ref, num_envs * 4));
        CUDA_CHECK(cudaMalloc(&val_ref, num_envs * 4));

        int block = 256, grid = (num_envs + block - 1) / block;
        long bad = 0;
        std::vector<int32_t> a1(num_envs), a2(num_envs);
        std::vector<float> f1((size_t)HIDDEN * num_envs), f2((size_t)HIDDEN * num_envs);
        std::vector<float> v1(num_envs), v2(num_envs), l1(num_envs), l2(num_envs);
        for (int t = 0; t < steps; t++) {
            const int8_t* prev = (t == 0) ? nullptr : r.r_done;
            obs_kernel<<<grid, block>>>(r.g, obs_f, nullptr, 0, num_envs);
            fused_forward_kernel<<<grid, block>>>(
                r.g, r.w, r.h_state, prev, r.r_obs, r.r_act, r.r_logprob, r.r_value,
                num_envs, r.seed, r.step_count);
            ref_forward_kernel<<<grid, block>>>(
                r.w, obs_f, h_state_ref, prev, act_ref, lp_ref, val_ref,
                num_envs, r.seed, r.step_count);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(a1.data(), r.r_act, num_envs * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(a2.data(), act_ref, num_envs * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(f1.data(), r.h_state, f1.size() * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(f2.data(), h_state_ref, f2.size() * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(v1.data(), r.r_value, num_envs * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(v2.data(), val_ref, num_envs * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(l1.data(), r.r_logprob, num_envs * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(l2.data(), lp_ref, num_envs * 4, cudaMemcpyDeviceToHost));
            if (memcmp(a1.data(), a2.data(), num_envs * 4) != 0) bad++;
            else if (memcmp(f1.data(), f2.data(), f1.size() * 4) != 0) bad++;
            else if (memcmp(v1.data(), v2.data(), num_envs * 4) != 0) bad++;
            else if (memcmp(l1.data(), l2.data(), num_envs * 4) != 0) bad++;

            // advance env with the sampled actions
            CUDA_CHECK(cudaMemsetAsync(r.reset_ctrl, 0, 8));
            step_mark_kernel<<<grid, block>>>(
                r.g, r.r_act, r.r_reward, r.r_done, r.reset_list, r.reset_ctrl, num_envs);
            uint64_t reset_seed = r.seed + (r.step_count + 1) * 1000000ULL;
            int wgrid = (num_envs + 3) / 4; if (wgrid > 512) wgrid = 512;
            reset_warp_kernel<<<wgrid, RESET_WARP_BLOCK>>>(
                r.g, r.reset_list, r.reset_ctrl, num_envs, reset_seed);
            CUDA_CHECK(cudaGetLastError());
            r.step_count++;
        }
        printf("fused vs dense forward: %s (%ld/%d mismatching steps)\n",
               bad == 0 ? "EXACT" : "MISMATCH", bad, steps);
        if (bad != 0) fail = 1;
        cudaFree(obs_f); cudaFree(h_state_ref);
        cudaFree(act_ref); cudaFree(lp_ref); cudaFree(val_ref);
    }

    // Check 2: megakernel rollout == split-path rollout, bitwise.
    {
        Rollout a(num_envs, steps, 42), b(num_envs, steps, 42);
        a.reset(); b.reset();
        a.run_mega(); b.run_split();
        CUDA_CHECK(cudaDeviceSynchronize());
        uint64_t ha = rollout_hash(a), hb = rollout_hash(b);
        printf("mega rollout hash:  %016llx\n", (unsigned long long)ha);
        printf("split rollout hash: %016llx  (%s)\n", (unsigned long long)hb,
               ha == hb ? "MATCH" : "MISMATCH");
        if (ha != hb) fail = 1;
    }

    printf("envs=%d steps=%d  =>  %s\n", num_envs, steps, fail ? "FAIL" : "PASS");
    return fail;
}

int main(int argc, char** argv) {
    const char* mode = (argc > 1) ? argv[1] : "mega";
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s\n", prop.name);

    if (strcmp(mode, "verify") == 0) {
        int envs = (argc > 2) ? atoi(argv[2]) : 2048;
        int steps = (argc > 3) ? atoi(argv[3]) : 300;
        return run_verify(envs, steps);
    }
    if (strcmp(mode, "sweep") == 0) {
        int T = (argc > 2) ? atoi(argv[2]) : 128;
        int sizes[] = {4096, 16384, 65536, 262144, 1048576};
        for (int i = 0; i < 5; i++) run_bench(sizes[i], T, 10, false);
        for (int i = 0; i < 5; i++) run_bench(sizes[i], T, 10, true);
        return 0;
    }
    int envs = (argc > 2) ? atoi(argv[2]) : 262144;
    int T = (argc > 3) ? atoi(argv[3]) : 128;
    int iters = (argc > 4) ? atoi(argv[4]) : 10;
    run_bench(envs, T, iters, strcmp(mode, "split") != 0);
    return 0;
}
