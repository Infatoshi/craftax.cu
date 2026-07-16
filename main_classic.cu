// main_classic.cu -- launcher for craftax_classic.cu: env benchmarks,
// validation, and the fixed batched policy (Linear encoder -> MinGRU x3,
// hidden 256 -> actor/value heads) via per-step kernels + cuBLAS TF32
// GEMMs, optionally replayed as a CUDA graph.
//
// Build (Makefile does this):  nvcc -O3 -arch=native \
//     --expt-relaxed-constexpr --use_fast_math main_classic.cu \
//     craftax_classic_cpu.o -o craftax_classic -Xcompiler -fopenmp -lpthread -lcublas
//
// Usage:
//   ./craftax_classic bench  [envs] [iters] [obs_mode] [reset_mode]  env-only SPS
//   ./craftax_classic sweep                          env-only sweep
//   ./craftax_classic hash   [envs] [steps]          env validation: worldgen
//                                                    exactness, trajectory hash,
//                                                    obs expansion, map diversity
//   ./craftax_classic run    [envs] [T] [iters]      env+policy rollout SPS (graph)
//   ./craftax_classic split                          same rollout, eager launches
//   ./craftax_classic runsweep [T]                   rollout sweep, graph vs eager
//   ./craftax_classic verify [envs] [steps]          batched vs scalar reference,
//                                                    eager vs graph hash equality
//   ./craftax_classic train  [envs] [T] [iters]      on-device PPO (rollout +
//                                                    GAE + GEMM backward + Adam)
//   ./craftax_classic gradcheck                      analytic vs FD gradients
//
// obs_mode: 0 = float one-hot obs (1345), 1 = compact uint8 obs (148)
// reset_mode: 0 = fused serial autoreset, 1 = split warp-cooperative reset

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cublas_v2.h>

#include "craftax_classic.cu"

#define CUDA_CHECK(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    exit(1); } } while (0)

#define CUBLAS_CHECK(x) do { cublasStatus_t st_ = (x); if (st_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st_, __FILE__, __LINE__); \
    exit(1); } } while (0)

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}
static uint64_t fnv1a(uint64_t h, const void* data, size_t len) {
    const uint8_t* p = (const uint8_t*)data;
    for (size_t i = 0; i < len; i++) { h ^= p[i]; h *= 0x100000001B3ULL; }
    return h;
}

struct Env {
    EnvSoA g;
    uint8_t* arena;
    float* obs;           // obs_mode 0 or 2
    uint8_t* obs_compact; // obs_mode 1 or 2
    float* rewards;
    int8_t* dones;
    int32_t* actions;
    int32_t* reset_list;  // compacted done-env indices
    int32_t* reset_ctrl;  // [0]=done count, [1]=work cursor
    int n;
    int obs_mode;
    int reset_mode;
    uint64_t seed;
    uint64_t step_count = 0;

    Env(int num_envs, uint64_t seed_, int obs_mode_, int reset_mode_ = 1)
        : n(num_envs), obs_mode(obs_mode_), reset_mode(reset_mode_), seed(seed_) {
        CUDA_CHECK(cudaMalloc(&arena, soa_bytes(n)));
        CUDA_CHECK(cudaMemset(arena, 0, soa_bytes(n)));
        g = carve_soa(arena, n);
        obs = nullptr; obs_compact = nullptr;
        if (obs_mode != 1) CUDA_CHECK(cudaMalloc(&obs, (size_t)n * OBS_DIM * sizeof(float)));
        if (obs_mode >= 1) CUDA_CHECK(cudaMalloc(&obs_compact, (size_t)n * OBS_DIM_COMPACT));
        CUDA_CHECK(cudaMalloc(&rewards, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dones, n));
        CUDA_CHECK(cudaMalloc(&actions, n * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&reset_list, n * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&reset_ctrl, 2 * sizeof(int32_t)));
    }
    ~Env() {
        cudaFree(arena); cudaFree(rewards); cudaFree(dones); cudaFree(actions);
        cudaFree(reset_list); cudaFree(reset_ctrl);
        if (obs) cudaFree(obs);
        if (obs_compact) cudaFree(obs_compact);
    }

    void reset() {
        int block = 256, grid = (n + block - 1) / block;
        if (reset_mode == 1) {
            int wgrid = ((size_t)n * 32 + RESET_WARP_BLOCK - 1) / RESET_WARP_BLOCK;
            reset_all_warp_kernel<<<wgrid, RESET_WARP_BLOCK>>>(g, n, seed);
            obs_kernel<<<grid, block>>>(g, obs, obs_compact, obs_mode, n);
        } else {
            reset_kernel<<<grid, block>>>(g, obs, obs_compact, obs_mode, n, seed);
        }
        CUDA_CHECK(cudaGetLastError());
        step_count = 0;
    }

    void step() {
        int block = 256, grid = (n + block - 1) / block;
        step_count++;
        uint64_t reset_seed = seed + step_count * 1000000ULL;
        gen_actions_kernel<<<grid, block>>>(actions, n, (int)step_count, seed * 31 + 7);
        if (reset_mode == 1) {
            CUDA_CHECK(cudaMemsetAsync(reset_ctrl, 0, 2 * sizeof(int32_t)));
            step_mark_kernel<<<grid, block>>>(g, actions, rewards, dones, reset_list, reset_ctrl, n);
            int warps_per_block = RESET_WARP_BLOCK / 32;
            int wgrid = (n + warps_per_block - 1) / warps_per_block;
            if (wgrid > 512) wgrid = 512;  // work-stealing loop drains the rest
            reset_warp_kernel<<<wgrid, RESET_WARP_BLOCK>>>(g, reset_list, reset_ctrl, n, reset_seed);
            obs_kernel<<<grid, block>>>(g, obs, obs_compact, obs_mode, n);
        } else {
            step_only_kernel<<<grid, block>>>(g, actions, rewards, dones, n);
            autoreset_obs_kernel<<<grid, block>>>(g, dones, obs, obs_compact, obs_mode, n, reset_seed);
        }
        CUDA_CHECK(cudaGetLastError());
    }
};


static void run_bench(int num_envs, int iters, int obs_mode, int reset_mode) {
    Env env(num_envs, 42, obs_mode, reset_mode);
    env.reset();
    for (int i = 0; i < 20; i++) env.step();
    CUDA_CHECK(cudaDeviceSynchronize());
    double t0 = now_s();
    for (int i = 0; i < iters; i++) env.step();
    CUDA_CHECK(cudaDeviceSynchronize());
    double dt = now_s() - t0;
    double sps = (double)num_envs * iters / dt;
    printf("NE=%8d obs_mode=%d reset_mode=%d: %8.1f M SPS  (%.2f us/kernel-step)\n",
           num_envs, obs_mode, reset_mode, sps / 1e6, dt / iters * 1e6);
}

// FNV over one env's full trajectory-visible outputs for one step.
static uint64_t traj_hash_step(uint64_t hash, const std::vector<float>& h_obs,
                               const std::vector<float>& h_rew,
                               const std::vector<int8_t>& h_done, int num_envs) {
    hash = fnv1a(hash, h_obs.data(), h_obs.size() * sizeof(float));
    hash = fnv1a(hash, h_rew.data(), num_envs * sizeof(float));
    hash = fnv1a(hash, h_done.data(), num_envs);
    return hash;
}

// Run a full trajectory and return its hash (float obs, rewards, dones).
static uint64_t run_traj(Env& env, int steps) {
    std::vector<float> h_obs((size_t)env.n * OBS_DIM);
    std::vector<float> h_rew(env.n);
    std::vector<int8_t> h_done(env.n);
    env.reset();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_obs.data(), env.obs, h_obs.size() * sizeof(float), cudaMemcpyDeviceToHost));
    uint64_t hash = fnv1a(0xcbf29ce484222325ULL, h_obs.data(), h_obs.size() * sizeof(float));
    for (int t = 0; t < steps; t++) {
        env.step();
        CUDA_CHECK(cudaMemcpy(h_obs.data(), env.obs, h_obs.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_rew.data(), env.rewards, env.n * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_done.data(), env.dones, env.n, cudaMemcpyDeviceToHost));
        hash = traj_hash_step(hash, h_obs, h_rew, h_done, env.n);
    }
    return hash;
}

static const char* BLOCK_NAMES[17] = {
    "INVALID","OOB","GRASS","WATER","STONE","TREE","WOOD","PATH","COAL",
    "IRON","DIAMOND","CRAFT_TABLE","FURNACE","SAND","LAVA","PLANT","RIPE_PLANT"
};

// Map-level diversity + block histogram over all envs' packed maps.
static void map_stats(const Env& env) {
    std::vector<uint8_t> maps((size_t)env.n * MAP_PACKED_SIZE);
    CUDA_CHECK(cudaMemcpy(maps.data(), env.g.map_packed, maps.size(), cudaMemcpyDeviceToHost));
    std::set<uint64_t> map_hashes;
    long counts[17] = {0};
    for (int e = 0; e < env.n; e++) {
        const uint8_t* m = &maps[(size_t)e * MAP_PACKED_SIZE];
        map_hashes.insert(fnv1a(0xcbf29ce484222325ULL, m, MAP_PACKED_SIZE));
        for (int i = 0; i < MAP_PACKED_SIZE; i++) {
            counts[m[i] & 0x0F]++;
            counts[(m[i] >> 4) & 0x0F]++;
        }
    }
    printf("map diversity: %d/%d distinct maps\n", (int)map_hashes.size(), env.n);
    long total = (long)env.n * MAP_SIZE * MAP_SIZE;
    printf("block histogram (%% of %ld tiles):\n", total);
    for (int b = 0; b < 17; b++)
        if (counts[b] > 0)
            printf("  %-12s %8.4f%%\n", BLOCK_NAMES[b], 100.0 * counts[b] / total);
    if ((int)map_hashes.size() < env.n * 99 / 100)
        printf("WARNING: map diversity below 99%% -- worldgen may be degraded\n");
}

static int run_hash(int num_envs, int steps) {
    int fail = 0;

    // Guard 1: serial vs warp worldgen, bit-exact (full arena + obs).
    {
        Env a(num_envs, 42, 2, 0), b(num_envs, 42, 2, 1);
        a.reset(); b.reset();
        CUDA_CHECK(cudaDeviceSynchronize());
        size_t nb = soa_bytes(num_envs);
        std::vector<uint8_t> ha(nb), hb(nb);
        CUDA_CHECK(cudaMemcpy(ha.data(), a.arena, nb, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hb.data(), b.arena, nb, cudaMemcpyDeviceToHost));
        bool arena_eq = memcmp(ha.data(), hb.data(), nb) == 0;
        std::vector<float> oa((size_t)num_envs * OBS_DIM), ob((size_t)num_envs * OBS_DIM);
        CUDA_CHECK(cudaMemcpy(oa.data(), a.obs, oa.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ob.data(), b.obs, ob.size() * sizeof(float), cudaMemcpyDeviceToHost));
        bool obs_eq = memcmp(oa.data(), ob.data(), oa.size() * sizeof(float)) == 0;
        printf("serial vs warp worldgen: arena %s, obs %s\n",
               arena_eq ? "EXACT" : "MISMATCH", obs_eq ? "EXACT" : "MISMATCH");
        if (!arena_eq || !obs_eq) fail = 1;

        map_stats(b);
    }

    // Guard 2: reset_mode 0 and 1 produce the same trajectory hash.
    {
        Env a(num_envs, 42, 0, 0);
        Env b(num_envs, 42, 0, 1);
        uint64_t hs = run_traj(a, steps);
        uint64_t hw = run_traj(b, steps);
        printf("traj hash serial-reset: %016llx\n", (unsigned long long)hs);
        printf("traj hash warp-reset:   %016llx  (%s)\n", (unsigned long long)hw,
               hs == hw ? "MATCH" : "MISMATCH");
        if (hs != hw) fail = 1;
    }

    // Guard 3: compact obs expansion bit-exact along a warp-reset trajectory.
    {
        Env env(num_envs, 42, 2, 1);
        env.reset();
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> h_obs((size_t)num_envs * OBS_DIM), h_exp((size_t)num_envs * OBS_DIM);
        float* d_expanded;
        CUDA_CHECK(cudaMalloc(&d_expanded, (size_t)num_envs * OBS_DIM * sizeof(float)));
        long expand_mismatches = 0;
        int block = 256, grid = (num_envs + block - 1) / block;
        for (int t = 0; t < steps; t++) {
            env.step();
            expand_obs_kernel<<<grid, block>>>(env.obs_compact, d_expanded, num_envs);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(h_obs.data(), env.obs, h_obs.size() * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_exp.data(), d_expanded, h_exp.size() * sizeof(float), cudaMemcpyDeviceToHost));
            if (memcmp(h_obs.data(), h_exp.data(), h_obs.size() * sizeof(float)) != 0)
                expand_mismatches++;
        }
        cudaFree(d_expanded);
        printf("compact-obs expansion: %s (%ld mismatching steps)\n",
               expand_mismatches == 0 ? "EXACT" : "MISMATCH", expand_mismatches);
        if (expand_mismatches != 0) fail = 1;
    }

    printf("envs=%d steps=%d seed=42  =>  %s\n", num_envs, steps, fail ? "FAIL" : "PASS");
    return fail;
}

static void init_weight_seg(float* p, int count, float bound, uint64_t seed, int subseq) {
    init_weights_kernel<<<(count + 255) / 256, 256>>>(p, count, bound, seed, subseq);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// cuBLAS conveniences. All buffers are col-major. A row-major
// weight W [m][k] is consumed as col-major W^T [k][m], so the
// forward is OP_T and dh/dW use OP_N/OP_T on the SAME memory --
// no transposed weight copies anywhere.
// ============================================================
static cublasHandle_t g_blas = nullptr;
static float g_one = 1.0f, g_zero = 0.0f;

static void ensure_blas() {
    if (!g_blas) {
        CUBLAS_CHECK(cublasCreate(&g_blas));
        CUBLAS_CHECK(cublasSetMathMode(g_blas, CUBLAS_TF32_TENSOR_OP_MATH));
    }
}

// y[m][cols] = W[m][k] @ x[k][cols]   (forward, W row-major raw)
static void gemm_fwd(int m, int cols, int k, const float* W, const float* x,
                     float* y, cudaStream_t s) {
    CUBLAS_CHECK(cublasSetStream(g_blas, s));
    CUBLAS_CHECK(cublasSgemm(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, m, cols, k,
                             &g_one, W, k, x, k, &g_zero, y, m));
}

// dh[HIDDEN][cols] = W^T @ dpre[m3][cols]  (same raw W memory, OP_N)
static void gemm_dh(int m3, int cols, const float* W, const float* dpre,
                    float* dh, cudaStream_t s) {
    CUBLAS_CHECK(cublasSetStream(g_blas, s));
    CUBLAS_CHECK(cublasSgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_N, HIDDEN, cols, m3,
                             &g_one, W, HIDDEN, dpre, m3, &g_zero, dh, HIDDEN));
}

// dW[j-major][+] += x[HIDDEN][cols] @ dpre[rows][cols]^T, beta=1.
// C(j,i) lands at j + i*HIDDEN, i.e. raw [rows][HIDDEN] row-major,
// matching the forward weight layout exactly.
static void gemm_dw(int rows, int cols, const float* x, const float* dpre,
                    float* dW, cudaStream_t s) {
    CUBLAS_CHECK(cublasSetStream(g_blas, s));
    CUBLAS_CHECK(cublasSgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_T, HIDDEN, rows, cols,
                             &g_one, x, HIDDEN, dpre, rows, &g_one, dW, HIDDEN));
}

// dWv[j] += sum_s x[HIDDEN][cols](j,s) * dv[s]     (gemv, beta=1)
static void gemv_dwv(const float* x, const float* dv, float* dWv, int cols,
                     cudaStream_t s) {
    CUBLAS_CHECK(cublasSetStream(g_blas, s));
    CUBLAS_CHECK(cublasSgemv(g_blas, CUBLAS_OP_N, HIDDEN, cols,
                             &g_one, x, HIDDEN, dv, 1, &g_one, dWv, 1));
}

// ============================================================
// Rollout: env + fixed 3x256 policy driven per step. Each step is
// ~13 launches; `run` captures all T steps into one CUDA graph and
// replays it. Slabs r_obs/r_state/... hold T steps of data.
// ============================================================
struct Rollout {
    EnvSoA g;
    uint8_t* arena;
    Weights w;
    float* params;
    float* h_state;      // [GRU_LAYERS][HIDDEN][n] live recurrent state
    float* h_enc;        // [HIDDEN][n] col-major forward chain
    float* hout[2];      // [HIDDEN][n]
    float* h3;           // [HIDDEN][n]
    float* pre;          // [GRU_OUT][n]
    float* logits;       // [NUM_ACTIONS][n]
    uint8_t* r_obs;      // [T][n][OBS_DIM_COMPACT]
    int32_t* r_act;      // [T][n]
    float* r_logprob;    // [T][n]
    float* r_value;      // [T][n]
    float* r_reward;     // [T][n]
    int8_t* r_done;      // [T][n]
    float* r_state = nullptr;  // [T][GRU_LAYERS*HIDDEN][n] post-zero state inputs
    unsigned long long* ep_stats = nullptr;  // [1+22]
    int32_t* reset_list;
    int32_t* reset_ctrl;
    uint64_t* step_ctr;  // device counter: sampler offset + reset seed schedule
    cudaStream_t stream;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t gexec = nullptr;
    bool warmed = false;
    int n, T;
    uint64_t seed;

    Rollout(int num_envs, int horizon, uint64_t seed_) : n(num_envs), T(horizon), seed(seed_) {
        if (n % 32 != 0) {
            fprintf(stderr, "num_envs must be a multiple of 32 (warp collectives)\n");
            exit(1);
        }
        ensure_blas();
        CUDA_CHECK(cudaMalloc(&arena, soa_bytes(n)));
        CUDA_CHECK(cudaMemset(arena, 0, soa_bytes(n)));
        g = carve_soa(arena, n);

        CUDA_CHECK(cudaMalloc(&params, (size_t)PARAM_COUNT * sizeof(float)));
        float be = 1.0f / sqrtf((float)OBS_DIM);
        float bh = 1.0f / sqrtf((float)HIDDEN);
        init_weight_seg(params + PARAM_W_ENC, OBS_DIM * HIDDEN, be, 1234, 0);
        init_weight_seg(params + PARAM_B_ENC, HIDDEN, be, 1234, 1);
        init_weight_seg(params + PARAM_W_GRU, W_GRU_ELEMS, bh, 1234, 2);
        init_weight_seg(params + PARAM_W_A, NUM_ACTIONS * HIDDEN, bh, 1234, 3);
        init_weight_seg(params + PARAM_B_A, NUM_ACTIONS, bh, 1234, 4);
        init_weight_seg(params + PARAM_W_V, HIDDEN, bh, 1234, 5);
        init_weight_seg(params + PARAM_B_V, 1, bh, 1234, 6);
        w.W_enc = params + PARAM_W_ENC;
        w.b_enc = params + PARAM_B_ENC;
        w.W_gru = params + PARAM_W_GRU;
        w.W_a   = params + PARAM_W_A;
        w.b_a   = params + PARAM_B_A;
        w.W_v   = params + PARAM_W_V;
        w.b_v   = params + PARAM_B_V;

        size_t hn = (size_t)HIDDEN * n;
        CUDA_CHECK(cudaMalloc(&h_state, (size_t)GRU_LAYERS * hn * sizeof(float)));
        CUDA_CHECK(cudaMemset(h_state, 0, (size_t)GRU_LAYERS * hn * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_enc, hn * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&hout[0], hn * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&hout[1], hn * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h3, hn * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&pre, (size_t)GRU_OUT * n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&logits, (size_t)NUM_ACTIONS * n * sizeof(float)));

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
        CUDA_CHECK(cudaMalloc(&step_ctr, sizeof(uint64_t)));
        CUDA_CHECK(cudaMemset(step_ctr, 0, sizeof(uint64_t)));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }
    ~Rollout() {
        if (gexec) cudaGraphExecDestroy(gexec);
        if (graph) cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
        cudaFree(arena); cudaFree(h_state); cudaFree(params);
        cudaFree(h_enc); cudaFree(hout[0]); cudaFree(hout[1]); cudaFree(h3);
        cudaFree(pre); cudaFree(logits);
        cudaFree(r_obs); cudaFree(r_act); cudaFree(r_logprob); cudaFree(r_value);
        cudaFree(r_reward); cudaFree(r_done);
        if (r_state) cudaFree(r_state);
        cudaFree(reset_list); cudaFree(reset_ctrl); cudaFree(step_ctr);
        if (ep_stats) { cudaFree(ep_stats); ep_stats = nullptr; }
    }

    void alloc_states() {
        CUDA_CHECK(cudaMalloc(&r_state,
            (size_t)T * GRU_LAYERS * HIDDEN * n * sizeof(float)));
    }

    void reset() {
        int wgrid = ((size_t)n * 32 + RESET_WARP_BLOCK - 1) / RESET_WARP_BLOCK;
        reset_all_warp_kernel<<<wgrid, RESET_WARP_BLOCK, 0, stream>>>(g, n, seed);
        CUDA_CHECK(cudaMemsetAsync(h_state, 0,
            (size_t)GRU_LAYERS * HIDDEN * n * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(step_ctr, 0, sizeof(uint64_t), stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGetLastError());
    }

    // One env+policy step writing slab row t. t==0 never sees prev
    // dones (old mega/split semantic: state zeroing restarts each
    // rollout).
    void step(int t, bool record) {
        size_t o = (size_t)t * n;
        int block = 256, grid = (n + block - 1) / block;
        int wgrid = (int)(((size_t)n * 32 + block - 1) / block);
        const int8_t* prev = (t == 0) ? nullptr : r_done + (size_t)(t - 1) * n;
        CUDA_CHECK(cudaMemsetAsync(reset_ctrl, 0, 8, stream));
        encode_env_warp_kernel<<<wgrid, block, 0, stream>>>(g, w.W_enc, w.b_enc, h_enc, n);
        if (record)
            record_obs_kernel<<<grid, block, 0, stream>>>(g, r_obs + o * OBS_DIM_COMPACT, n);
        const float* x = h_enc;
        for (int l = 0; l < GRU_LAYERS; l++) {
            gemm_fwd(GRU_OUT, n, HIDDEN, w.W_gru + (size_t)l * (size_t)GRU_OUT * HIDDEN,
                     x, pre, stream);
            float* xn = (l < GRU_LAYERS - 1) ? hout[l] : h3;
            float* store = r_state
                ? r_state + ((size_t)t * GRU_LAYERS + l) * ((size_t)HIDDEN * n)
                : nullptr;
            mingru_epi_fwd_kernel<<<(int)(((size_t)HIDDEN * n + block - 1) / block), block, 0, stream>>>(
                pre, x, h_state + (size_t)l * HIDDEN * n, xn, store, prev, n);
            x = xn;
        }
        gemm_fwd(NUM_ACTIONS, n, HIDDEN, w.W_a, h3, logits, stream);
        value_sample_kernel<<<grid, block, 0, stream>>>(
            h3, logits, w.b_a, w.W_v, w.b_v, r_act + o, r_logprob + o, r_value + o,
            n, seed, step_ctr);
        step_mark_kernel<<<grid, block, 0, stream>>>(
            g, r_act + o, r_reward + o, r_done + o, reset_list, reset_ctrl, n);
        if (ep_stats) ep_stats_kernel<<<grid, block, 0, stream>>>(g, r_done + o, ep_stats, n);
        int warps_per_block = RESET_WARP_BLOCK / 32;
        int rwgrid = (n + warps_per_block - 1) / warps_per_block;
        if (rwgrid > 512) rwgrid = 512;
        reset_warp_ctr_kernel<<<rwgrid, RESET_WARP_BLOCK, 0, stream>>>(
            g, reset_list, reset_ctrl, n, seed, step_ctr);
        bump_ctr_kernel<<<1, 1, 0, stream>>>(step_ctr);
        CUDA_CHECK(cudaGetLastError());
    }

    // Full T-step rollout. Graph mode warms up eagerly once (cuBLAS
    // workspace allocation is illegal inside capture), then captures
    // and replays. Eager and graph runs are bitwise identical.
    void run(bool use_graph) {
        if (use_graph && !warmed) {
            for (int t = 0; t < T; t++) step(t, true);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            warmed = true;
        }
        if (!use_graph) {
            for (int t = 0; t < T; t++) step(t, true);
            return;
        }
        if (!gexec) {
            CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
            for (int t = 0; t < T; t++) step(t, true);
            CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
            cudaError_t ierr = cudaGraphInstantiate(&gexec, graph, nullptr, nullptr, 0);
            if (ierr != cudaSuccess) {
                fprintf(stderr, "graph instantiate failed: %s -- falling back to eager\n",
                        cudaGetErrorString(ierr));
                cudaGetLastError();
                gexec = nullptr;
                for (int t = 0; t < T; t++) step(t, true);
                return;
            }
        }
        CUDA_CHECK(cudaGraphLaunch(gexec, stream));
    }
};

static const char* ROLLOUT_NAMES[2] = {"split", "run  "};

static void run_bench_rollout(int num_envs, int T, int iters, int kind) {
    size_t need = (size_t)num_envs * T * (OBS_DIM_COMPACT + 17) + soa_bytes(num_envs)
                + (size_t)num_envs * (GRU_LAYERS * HIDDEN + 4 * HIDDEN + GRU_OUT + NUM_ACTIONS) * 4;
    size_t free_b, total_b;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    if (need + (512ull << 20) > free_b) {
        printf("NE=%8d T=%d %s: skipped (needs %.1f GB, %.1f GB free)\n",
               num_envs, T, ROLLOUT_NAMES[kind], need / 1e9, free_b / 1e9);
        return;
    }
    Rollout r(num_envs, T, 42);
    r.reset();
    for (int i = 0; i < 3; i++) r.run(kind == 1);
    CUDA_CHECK(cudaStreamSynchronize(r.stream));
    CUDA_CHECK(cudaDeviceSynchronize());
    double t0 = now_s();
    for (int i = 0; i < iters; i++) r.run(kind == 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    double dt = now_s() - t0;
    double sps = (double)num_envs * T * iters / dt;
    printf("NE=%8d T=%d %s: %8.2f M SPS  (%.1f us/env-step-row)\n",
           num_envs, T, ROLLOUT_NAMES[kind], sps / 1e6, dt / (iters * (double)T) * 1e6);
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

// Batched TF32 policy vs the scalar fp32 reference, along a real
// trajectory. The env always advances with the BATCHED path's action;
// the reference only mirrors the forward, so tf32 sampling flips can
// never poison the comparison. Gates on max |delta| along the whole
// trajectory; action flips are reported as information.
static int run_verify(int num_envs, int steps) {
    int fail = 0;
    ensure_blas();
    if (getenv("CRAFTAX_VERIFY_FP32"))
        CUBLAS_CHECK(cublasSetMathMode(g_blas, CUBLAS_DEFAULT_MATH));
    {
        Rollout r(num_envs, 1, 42);
        r.reset();
        float* ref_state;   // [L][H][n]
        float* ref_h3;      // [H][n]
        int32_t* act2; float* lp2; float* val2;
        size_t hbytes = (size_t)HIDDEN * num_envs * sizeof(float);
        CUDA_CHECK(cudaMalloc(&ref_state, (size_t)GRU_LAYERS * hbytes));
        CUDA_CHECK(cudaMemset(ref_state, 0, (size_t)GRU_LAYERS * hbytes));
        CUDA_CHECK(cudaMalloc(&ref_h3, hbytes));
        CUDA_CHECK(cudaMalloc(&act2, num_envs * 4));
        CUDA_CHECK(cudaMalloc(&lp2, num_envs * 4));
        CUDA_CHECK(cudaMalloc(&val2, num_envs * 4));

        int block = 256, grid = (num_envs + block - 1) / block;
        std::vector<float> b_h3((size_t)HIDDEN * num_envs), b_val(num_envs);
        std::vector<float> r_h((size_t)HIDDEN * num_envs);
        std::vector<int32_t> a1(num_envs), a2h(num_envs);
        std::vector<float> v2(num_envs);
        double max_dh = 0.0, max_dv = 0.0;
        long flips = 0;
        for (int t = 0; t < steps; t++) {
            const int8_t* prev = (t == 0) ? nullptr : r.r_done;
            CUDA_CHECK(cudaMemsetAsync(r.reset_ctrl, 0, 8, r.stream));
            encode_env_warp_kernel<<<(num_envs * 32 + 255) / 256, 256, 0, r.stream>>>(
                r.g, r.w.W_enc, r.w.b_enc, r.h_enc, num_envs);
            const float* x = r.h_enc;
            for (int l = 0; l < GRU_LAYERS; l++) {
                gemm_fwd(GRU_OUT, num_envs, HIDDEN,
                         r.w.W_gru + (size_t)l * (size_t)GRU_OUT * HIDDEN, x, r.pre, r.stream);
                float* xn = (l < GRU_LAYERS - 1) ? r.hout[l] : r.h3;
                mingru_epi_fwd_kernel<<<((size_t)HIDDEN * num_envs + 255) / 256, 256, 0, r.stream>>>(
                    r.pre, x, r.h_state + (size_t)l * HIDDEN * num_envs, xn, nullptr, prev, num_envs);
                x = xn;
            }
            gemm_fwd(NUM_ACTIONS, num_envs, HIDDEN, r.w.W_a, r.h3, r.logits, r.stream);
            value_sample_kernel<<<grid, block, 0, r.stream>>>(
                r.h3, r.logits, r.w.b_a, r.w.W_v, r.w.b_v, r.r_act, r.r_logprob, r.r_value,
                num_envs, r.seed, r.step_ctr);
            ref_policy_l3_kernel<<<grid, block, 0, r.stream>>>(
                r.g, r.w, ref_state, prev, act2, lp2, val2, ref_h3,
                num_envs, r.seed, r.step_ctr);
            step_mark_kernel<<<grid, block, 0, r.stream>>>(
                r.g, r.r_act, r.r_reward, r.r_done, r.reset_list, r.reset_ctrl, num_envs);
            int rwgrid = (num_envs + 3) / 4; if (rwgrid > 512) rwgrid = 512;
            reset_warp_ctr_kernel<<<rwgrid, RESET_WARP_BLOCK, 0, r.stream>>>(
                r.g, r.reset_list, r.reset_ctrl, num_envs, r.seed, r.step_ctr);
            bump_ctr_kernel<<<1, 1, 0, r.stream>>>(r.step_ctr);
            CUDA_CHECK(cudaGetLastError());
            // device-visible copies: the legacy default stream does
            // NOT serialize with our non-blocking stream
            CUDA_CHECK(cudaStreamSynchronize(r.stream));

            CUDA_CHECK(cudaMemcpy(b_h3.data(), r.h3, hbytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(b_val.data(), r.r_value, num_envs * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(r_h.data(), ref_h3, hbytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(v2.data(), val2, num_envs * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(a1.data(), r.r_act, num_envs * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(a2h.data(), act2, num_envs * 4, cudaMemcpyDeviceToHost));
            for (int i = 0; i < HIDDEN * num_envs; i++) {
                double d = fabs((double)b_h3[i] - (double)r_h[i]);
                if (d > max_dh) max_dh = d;
            }
            for (int e = 0; e < num_envs; e++) {
                double d = fabs((double)b_val[e] - (double)v2[e]);
                if (d > max_dv) max_dv = d;
                if (a1[e] != a2h[e]) flips++;
            }
        }
        printf("batched vs scalar ref: max |d_h3| %.4g  max |d_v| %.4g  action flips %ld/%lld  %s\n",
               max_dh, max_dv, flips, (long long)steps * num_envs,
               (max_dh < 5e-2 && max_dv < 5e-2) ? "PASS" : "FAIL");
        if (max_dh >= 5e-2 || max_dv >= 5e-2) fail = 1;
        cudaFree(ref_state); cudaFree(ref_h3);
        cudaFree(act2); cudaFree(lp2); cudaFree(val2);
    }

    // Eager and graph-replayed rollouts must be bitwise identical.
    // The first graph-mode run also executes its T warmup steps, so
    // build/warm the graph first, then reset both sides and compare
    // one clean rollout per mode.
    {
        Rollout a(num_envs, steps, 42), b(num_envs, steps, 42);
        b.reset();
        b.run(true);
        CUDA_CHECK(cudaStreamSynchronize(b.stream));
        a.reset(); b.reset();
        a.run(false);
        CUDA_CHECK(cudaStreamSynchronize(a.stream));
        b.run(true);
        CUDA_CHECK(cudaStreamSynchronize(b.stream));
        CUDA_CHECK(cudaDeviceSynchronize());
        uint64_t ha = rollout_hash(a), hb = rollout_hash(b);
        printf("rollout hash (eager): %016llx\n", (unsigned long long)ha);
        printf("rollout hash (graph): %016llx  (%s)\n", (unsigned long long)hb,
               ha == hb ? "MATCH" : "MISMATCH");
        if (ha != hb) fail = 1;
    }

    printf("envs=%d steps=%d  =>  %s\n", num_envs, steps, fail ? "FAIL" : "PASS");
    return fail;
}

// ============================================================
// PPO training: rollout -> bootstrap -> GAE -> GEMM backward ->
// Adam, everything resident on device as one capturable graph; the
// loop only syncs to log.
// ============================================================
struct PPOConfig {
    float lr = 3e-4f, gamma = 0.99f, lam = 0.95f;
    float clip = 0.2f, ent = 0.01f, vf = 0.5f;
    int epochs = 1;
    int minibatches = 1;  // contiguous env-range slices per epoch
    int lr_anneal = 0;
    int bptt_split = 1;   // BPTT segments per env (1 = exact)
};

struct Trainer {
    Rollout& r;
    PPOConfig cfg;
    float* grads;      // [PARAM_COUNT], accumulated per minibatch
    float* adam_m;
    float* adam_v;
    float* d_lr;       // device float (host writes annealed value)
    uint64_t* adam_ctr;
    float* v_boot;     // [n]
    float* boot_state; // [3H][n] scratch for the bootstrap value
    float* adv;        // [T][n]
    float* ret;        // [T][n]
    double* loss_acc;  // [3] raw sums (logging)
    double* stats;     // [2] adv sum, sumsq
    // tight minibatch buffers (cols = T * mb)
    float* x[4];       // [HIDDEN][cols]: enc out, layer outs
    float* preb[3];    // [GRU_OUT][cols] (sweeps write dpre in place)
    float* logitsb;    // [NUM_ACTIONS][cols]
    float* dlogits;    // [NUM_ACTIONS][cols]
    float* dvalue;     // [cols]
    float* dh3b;       // [HIDDEN][cols]
    float* dhG;        // [HIDDEN][cols]
    float* dhX;        // [HIDDEN][cols]
    float* stb[3];     // [HIDDEN][cols] post-zero input states from replay
    float* live_st[3]; // [HIDDEN][mb] replay carry across t
    int mb;              // envs per minibatch
    int chunk;           // envs per loss() chunk (= mb)
    long total_updates = 0;
    bool warmed = false;
    cudaGraph_t tgraph = nullptr;
    cudaGraphExec_t texec = nullptr;
    static const int MB_BYTES_PER_COL = 20100;  // measured from buffer set

    Trainer(Rollout& r_, PPOConfig cfg_) : r(r_), cfg(cfg_) {
        r.alloc_states();
        if (cfg.minibatches < 1 || r.n % cfg.minibatches != 0) {
            fprintf(stderr, "--minibatches must divide num_envs (%d %% %d != 0)\n",
                    r.n, cfg.minibatches);
            exit(1);
        }
        mb = r.n / cfg.minibatches;
        if (cfg.bptt_split < 1 || r.T % cfg.bptt_split != 0 || mb % 32 != 0) {
            fprintf(stderr, "--bptt-split must divide horizon (%d %% %d != 0), "
                    "and envs/minibatch must be a multiple of 32\n",
                    r.T, cfg.bptt_split);
            exit(1);
        }
        size_t cols = (size_t)r.T * mb;
        CUDA_CHECK(cudaMalloc(&grads, (size_t)PARAM_COUNT * sizeof(float)));
        CUDA_CHECK(cudaMemset(grads, 0, (size_t)PARAM_COUNT * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&adam_m, (size_t)PARAM_COUNT * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&adam_v, (size_t)PARAM_COUNT * sizeof(float)));
        CUDA_CHECK(cudaMemset(adam_m, 0, (size_t)PARAM_COUNT * sizeof(float)));
        CUDA_CHECK(cudaMemset(adam_v, 0, (size_t)PARAM_COUNT * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_lr, sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_lr, &cfg.lr, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&adam_ctr, sizeof(uint64_t)));
        CUDA_CHECK(cudaMemset(adam_ctr, 0, sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&v_boot, r.n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&boot_state, (size_t)GRU_LAYERS * HIDDEN * r.n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&adv, (size_t)r.n * r.T * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ret, (size_t)r.n * r.T * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&loss_acc, 3 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&stats, 2 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&r.ep_stats, 23 * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(r.ep_stats, 0, 23 * sizeof(unsigned long long)));
        for (int i = 0; i < 4; i++)
            CUDA_CHECK(cudaMalloc(&x[i], (size_t)HIDDEN * cols * sizeof(float)));
        for (int l = 0; l < 3; l++)
            CUDA_CHECK(cudaMalloc(&preb[l], (size_t)GRU_OUT * cols * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&logitsb, (size_t)NUM_ACTIONS * cols * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dlogits, (size_t)NUM_ACTIONS * cols * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dvalue, cols * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dh3b, (size_t)HIDDEN * cols * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dhG, (size_t)HIDDEN * cols * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dhX, (size_t)HIDDEN * cols * sizeof(float)));
        for (int l = 0; l < 3; l++) {
            CUDA_CHECK(cudaMalloc(&stb[l], (size_t)HIDDEN * cols * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&live_st[l], (size_t)HIDDEN * mb * sizeof(float)));
        }
        chunk = mb;
    }
    ~Trainer() {
        if (texec) cudaGraphExecDestroy(texec);
        if (tgraph) cudaGraphDestroy(tgraph);
        cudaFree(grads); cudaFree(adam_m); cudaFree(adam_v);
        cudaFree(d_lr); cudaFree(adam_ctr);
        cudaFree(v_boot); cudaFree(boot_state); cudaFree(adv); cudaFree(ret);
        cudaFree(loss_acc); cudaFree(stats);
        for (int i = 0; i < 4; i++) cudaFree(x[i]);
        for (int l = 0; l < 3; l++) {
            cudaFree(preb[l]); cudaFree(stb[l]); cudaFree(live_st[l]);
        }
        cudaFree(logitsb); cudaFree(dlogits); cudaFree(dvalue);
        cudaFree(dh3b); cudaFree(dhG); cudaFree(dhX);
        if (r.ep_stats) { cudaFree(r.ep_stats); r.ep_stats = nullptr; }
    }

    int seg_len() const { return r.T / cfg.bptt_split; }

    // Rollout + bootstrap value + GAE + advantage stats. The rollout
    // always launches eagerly here: the trainer's own graph capture
    // (update) subsumes it -- a nested cudaGraphLaunch inside another
    // capture is not allowed.
    void collect() {
        r.run(false);
        int block = 256, grid = (r.n + block - 1) / block;
        size_t hn = (size_t)HIDDEN * r.n;
        const int8_t* prev = r.r_done + (size_t)(r.T - 1) * r.n;
        CUDA_CHECK(cudaMemcpyAsync(boot_state, r.h_state,
                                   (size_t)GRU_LAYERS * hn * sizeof(float),
                                   cudaMemcpyDeviceToDevice, r.stream));
        encode_env_warp_kernel<<<((size_t)r.n * 32 + 255) / 256, block, 0, r.stream>>>(
            r.g, r.w.W_enc, r.w.b_enc, r.h_enc, r.n);
        const float* xx = r.h_enc;
        for (int l = 0; l < GRU_LAYERS; l++) {
            gemm_fwd(GRU_OUT, r.n, HIDDEN, r.w.W_gru + (size_t)l * (size_t)GRU_OUT * HIDDEN,
                     xx, r.pre, r.stream);
            float* xn = (l < GRU_LAYERS - 1) ? r.hout[l] : r.h3;
            mingru_epi_fwd_kernel<<<((size_t)hn + 255) / 256, block, 0, r.stream>>>(
                r.pre, xx, boot_state + (size_t)l * hn, xn, nullptr, prev, r.n);
            xx = xn;
        }
        value_dot_kernel<<<grid, block, 0, r.stream>>>(r.h3, r.w.W_v, r.w.b_v, v_boot, r.n);
        gae_kernel<<<grid, block, 0, r.stream>>>(
            r.r_value, r.r_reward, r.r_done, v_boot, adv, ret, r.n, r.T, cfg.gamma, cfg.lam);
        CUDA_CHECK(cudaMemsetAsync(stats, 0, 2 * sizeof(double), r.stream));
        adv_stats_kernel<<<256, 256, 0, r.stream>>>(adv, (size_t)r.n * r.T, stats);
        CUDA_CHECK(cudaGetLastError());
    }

    // Gradients over one contiguous env range, accumulated into the
    // (freshly zeroed) grads arena.
    void backward(int env_start, int env_count) {
        ensure_blas();
        const int T = r.T, n = r.n;
        const int cols = T * env_count;
        int bgrid = (cols + 255) / 256;
        int hgrid = (int)(((size_t)HIDDEN * cols + 255) / 256);
        int seg = seg_len();

        // forward recompute at current theta: live recurrence within
        // BPTT segments (stored state reloaded at segment starts),
        // the same function the sweep's dcarry chain differentiates
        encode_obs_kernel<<<(int)(((size_t)HIDDEN * cols + 255) / 256), 256, 0, r.stream>>>(
            r.r_obs, r.w.W_enc, r.w.b_enc, x[0], 0, cols, env_count, n, env_start);
        int egrid = (int)(((size_t)HIDDEN * env_count + 255) / 256);
        for (int t = 0; t < T; t++) {
            size_t co = (size_t)t * env_count;
            bool segstart = (t % seg) == 0;
            const int8_t* prev = segstart ? nullptr
                : r.r_done + (size_t)(t - 1) * n + env_start;
            for (int l = 0; l < GRU_LAYERS; l++) {
                gemm_fwd(GRU_OUT, env_count, HIDDEN,
                         r.w.W_gru + (size_t)l * (size_t)GRU_OUT * HIDDEN,
                         x[l] + co * HIDDEN, preb[l] + co * GRU_OUT, r.stream);
                mingru_epi_replay_kernel<<<egrid, 256, 0, r.stream>>>(
                    preb[l] + co * GRU_OUT, x[l] + co * HIDDEN,
                    segstart ? r.r_state + ((size_t)t * GRU_LAYERS + l) * HIDDEN * n + env_start
                             : nullptr,
                    live_st[l], prev, stb[l] + co * HIDDEN,
                    x[l + 1] + co * HIDDEN, n, env_count);
            }
        }
        gemm_fwd(NUM_ACTIONS, cols, HIDDEN, r.w.W_a, x[3], logitsb, r.stream);

        head_bwd_kernel<<<bgrid, 256, 0, r.stream>>>(
            logitsb, x[3], r.w.b_a, r.w.W_v, r.w.b_v,
            r.r_act + env_start, r.r_logprob + env_start, adv + env_start, ret + env_start,
            stats, dlogits, dvalue, loss_acc, T, env_count, n,
            cfg.clip, cfg.vf, cfg.ent);

        // dh chain: heads -> layer 3 -> 2 -> 1
        const float* Wl2 = r.w.W_gru + 2 * (size_t)GRU_OUT * HIDDEN;
        const float* Wl1 = r.w.W_gru + (size_t)GRU_OUT * HIDDEN;
        const float* Wl0 = r.w.W_gru;
        gemm_dh(NUM_ACTIONS, cols, r.w.W_a, dlogits, dh3b, r.stream);
        add_dv_wv_kernel<<<hgrid, 256, 0, r.stream>>>(dh3b, dvalue, r.w.W_v, cols);
        int sgrid = (int)(((size_t)HIDDEN * env_count + 255) / 256);
        const int8_t* rdone = r.r_done + env_start;
        mingru_sweep_bwd_kernel<<<sgrid, 256, 0, r.stream>>>(
            preb[2], x[2], dh3b, nullptr, stb[2],
            rdone, preb[2], dhX, T, env_count, n, seg);
        gemm_dh(GRU_OUT, cols, Wl2, preb[2], dhG, r.stream);
        mingru_sweep_bwd_kernel<<<sgrid, 256, 0, r.stream>>>(
            preb[1], x[1], dhG, dhX, stb[1],
            rdone, preb[1], dhX, T, env_count, n, seg);
        gemm_dh(GRU_OUT, cols, Wl1, preb[1], dhG, r.stream);
        mingru_sweep_bwd_kernel<<<sgrid, 256, 0, r.stream>>>(
            preb[0], x[0], dhG, dhX, stb[0],
            rdone, preb[0], dhX, T, env_count, n, seg);
        gemm_dh(GRU_OUT, cols, Wl0, preb[0], dhG, r.stream);
        enc_bwd_kernel<<<(int)(((size_t)HIDDEN * cols + 255) / 256), 256, 0, r.stream>>>(
            r.r_obs, dhG, dhX, grads + PARAM_W_ENC, grads + PARAM_B_ENC,
            T, env_count, n, env_start);

        // weight grads: flat GEMMs over samples (beta=1 accumulate)
        for (int l = 0; l < GRU_LAYERS; l++)
            gemm_dw(GRU_OUT, cols, x[l], preb[l],
                    grads + PARAM_W_GRU + (size_t)l * GRU_OUT * HIDDEN, r.stream);
        gemm_dw(NUM_ACTIONS, cols, x[3], dlogits, grads + PARAM_W_A, r.stream);
        gemv_dwv(x[3], dvalue, grads + PARAM_W_V, cols, r.stream);
        colsum_kernel<<<NUM_ACTIONS, 128, 0, r.stream>>>(dlogits, grads + PARAM_B_A,
                                                         NUM_ACTIONS, cols);
        colsum_kernel<<<1, 128, 0, r.stream>>>(dvalue, grads + PARAM_B_V, 1, cols);
        CUDA_CHECK(cudaGetLastError());
    }

    void adam() {
        adam_kernel<<<(PARAM_COUNT + 255) / 256, 256, 0, r.stream>>>(
            r.params, grads, 1, adam_m, adam_v, adam_ctr, d_lr, 0.9f, 0.999f, 1e-8f);
        bump_ctr_kernel<<<1, 1, 0, r.stream>>>(adam_ctr);
        CUDA_CHECK(cudaGetLastError());
    }

    void body() {
        collect();
        for (int e = 0; e < cfg.epochs; e++) {
            CUDA_CHECK(cudaMemsetAsync(loss_acc, 0, 3 * sizeof(double), r.stream));
            for (int m = 0; m < cfg.minibatches; m++) {
                backward(m * mb, mb);
                adam();
            }
        }
    }

    // One PPO iteration. Graph mode warms up eagerly first (a rollout
    // plus one bare backward, no adam -- params stay untouched), then
    // captures the whole body once and replays it.
    void update(bool use_graph) {
        if (!warmed) {
            r.run(false);
            backward(0, mb);
            CUDA_CHECK(cudaStreamSynchronize(r.stream));
            warmed = true;
        }
        if (use_graph && !texec) {
            CUDA_CHECK(cudaStreamBeginCapture(r.stream, cudaStreamCaptureModeGlobal));
            body();
            CUDA_CHECK(cudaStreamEndCapture(r.stream, &tgraph));
            cudaError_t ierr = cudaGraphInstantiate(&texec, tgraph, nullptr, nullptr, 0);
            if (ierr != cudaSuccess) {
                fprintf(stderr, "trainer graph instantiate failed: %s -- eager\n",
                        cudaGetErrorString(ierr));
                cudaGetLastError();
                texec = nullptr;
            }
        }
        if (texec) {
            CUDA_CHECK(cudaGraphLaunch(texec, r.stream));
        } else {
            body();
        }
        CUDA_CHECK(cudaStreamSynchronize(r.stream));
    }

    // Total PPO loss over the stored rollout at the current params.
    // Uses the same live-recurrence-within-segments replay as the
    // backward's forward-recompute, so FD measures exactly the
    // function the backward differentiates -- stored r_state enters
    // only as the constant at BPTT segment starts.
    // Envs are processed in chunks for buffer reuse.
    double loss() {
        ensure_blas();
        const int T = r.T, n = r.n;
        const int seg = seg_len();
        CUDA_CHECK(cudaMemsetAsync(loss_acc, 0, 3 * sizeof(double), r.stream));
        for (int e0 = 0; e0 < n; e0 += chunk) {
            int cn = chunk < n - e0 ? chunk : n - e0;
            size_t hn = (size_t)HIDDEN * n;
            for (int t = 0; t < T; t++) {
                bool segstart = (t % seg) == 0;
                const int8_t* prev = segstart ? nullptr
                    : r.r_done + (size_t)(t - 1) * n + e0;
                encode_obs_kernel<<<(int)(((size_t)HIDDEN * cn + 255) / 256), 256, 0, r.stream>>>(
                    r.r_obs, r.w.W_enc, r.w.b_enc, x[0], t, cn, cn, n, e0);
                for (int l = 0; l < GRU_LAYERS; l++) {
                    gemm_fwd(GRU_OUT, cn, HIDDEN,
                             r.w.W_gru + (size_t)l * (size_t)GRU_OUT * HIDDEN,
                             x[l], preb[l], r.stream);
                    mingru_epi_replay_kernel<<<(int)(((size_t)HIDDEN * cn + 255) / 256), 256, 0, r.stream>>>(
                        preb[l], x[l],
                        segstart ? r.r_state + ((size_t)t * GRU_LAYERS + l) * hn + e0
                                 : nullptr,
                        live_st[l], prev, nullptr, x[l + 1], n, cn);
                }
                gemm_fwd(NUM_ACTIONS, cn, HIDDEN, r.w.W_a, x[3], logitsb, r.stream);
                loss_accum_kernel<<<(cn + 255) / 256, 256, 0, r.stream>>>(
                    logitsb, x[3], r.w.b_a, r.w.W_v, r.w.b_v,
                    r.r_act + e0, r.r_logprob + e0, adv + e0, ret + e0,
                    stats, loss_acc, t, n, T, cfg.clip);
            }
        }
        double h[3];
        CUDA_CHECK(cudaStreamSynchronize(r.stream));
        CUDA_CHECK(cudaMemcpy(h, loss_acc, 24, cudaMemcpyDeviceToHost));
        double count = (double)n * T;
        return (h[0] + cfg.vf * h[1] - cfg.ent * h[2]) / count;
    }
};

static void run_train(int num_envs, int T, int iters, PPOConfig cfg) {
    // The GEMM backward's minibatch buffers scale with T*mb (~17
    // KB/sample). If the user did not size minibatches, pick the
    // smallest power of two that keeps the buffers under ~8 GB.
    {
        size_t max_cols = ((size_t)8 << 30) / Trainer::MB_BYTES_PER_COL;
        while (cfg.minibatches < num_envs / 32 &&
               (size_t)(num_envs / cfg.minibatches) * T > max_cols)
            cfg.minibatches *= 2;
    }
    Rollout r(num_envs, T, 42);
    Trainer tr(r, cfg);
    tr.total_updates = (long)iters * cfg.epochs * cfg.minibatches;
    r.reset();
    printf("train: hidden=%d layers=%d envs=%d horizon=%d iters=%d lr=%g gamma=%g lam=%g clip=%g ent=%g vf=%g epochs=%d minibatches=%d lr_anneal=%d bptt_split=%d\n",
           HIDDEN, GRU_LAYERS, num_envs, T, iters, cfg.lr, cfg.gamma, cfg.lam,
           cfg.clip, cfg.ent, cfg.vf, cfg.epochs, cfg.minibatches,
           cfg.lr_anneal, cfg.bptt_split);

    static const char* ach_names[22] = {
        "collect_wood", "place_table", "eat_cow", "collect_sapling",
        "collect_drink", "make_wood_pick", "make_wood_sword", "place_plant",
        "defeat_zombie", "collect_stone", "place_stone", "eat_plant",
        "defeat_skeleton", "make_stone_pick", "make_stone_sword", "wake_up",
        "place_furnace", "collect_coal", "collect_iron", "collect_diamond",
        "make_iron_pick", "make_iron_sword"};

    double* rew_stats;
    CUDA_CHECK(cudaMalloc(&rew_stats, 2 * sizeof(double)));
    double t0 = now_s();
    size_t steps_done = 0, steps_window = 0;
    unsigned long long total_eps = 0, total_ach[22] = {0};
    int64_t updates_done = 0;
    for (int it = 1; it <= iters; it++) {
        if (cfg.lr_anneal && tr.total_updates > 0) {
            float lr_it = cfg.lr * (1.0f - (float)updates_done / (float)tr.total_updates);
            CUDA_CHECK(cudaMemcpy(tr.d_lr, &lr_it, 4, cudaMemcpyHostToDevice));
        }
        tr.update(true);
        updates_done += (int64_t)cfg.epochs * cfg.minibatches;
        steps_done += (size_t)num_envs * T;
        steps_window += (size_t)num_envs * T;
        if (it == 1 || it % 10 == 0 || it == iters) {
            CUDA_CHECK(cudaMemsetAsync(rew_stats, 0, 16));
            adv_stats_kernel<<<256, 256>>>(r.r_reward, (size_t)num_envs * T, rew_stats);
            double h_loss[3];
            double h_rew[2];
            unsigned long long h_eps[23];
            CUDA_CHECK(cudaMemcpy(h_loss, tr.loss_acc, 24, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_rew, rew_stats, 16, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_eps, r.ep_stats, 23 * 8, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemsetAsync(r.ep_stats, 0, 23 * 8));
            total_eps += h_eps[0];
            for (int a = 0; a < 22; a++) total_ach[a] += h_eps[1 + a];
            double count = (double)num_envs * T;
            double sps = steps_done / (now_s() - t0);
            double eplen = h_eps[0] ? (double)steps_window / h_eps[0] : 0.0;
            double retep = (h_rew[0] / count) * eplen;
            printf("iter %5d  %7.1f M SPS  pg %+.4f  vf %8.4f  ent %6.3f  rew/step %+.5f  ret/ep %+.3f  eplen %5.0f\n",
                   it, sps / 1e6, h_loss[0] / count, h_loss[1] / count,
                   h_loss[2] / count, h_rew[0] / count, retep, eplen);
            steps_window = 0;
        }
    }
    if (total_eps) {
        printf("achievement rates over %llu episodes:\n", total_eps);
        for (int a = 0; a < 22; a++)
            printf("  %-18s %6.3f\n", ach_names[a], (double)total_ach[a] / total_eps);
    }
    cudaFree(rew_stats);
}

// Analytic gradients vs central finite differences of the PPO loss on
// a small fixed rollout. The loss treats actions, old logprobs, and
// advantages as constants, so it is a pure function of the parameters
// and FD is exact up to fp32 forward noise. cuBLAS runs in strict
// FP32 here (no TF32) so loss() and backward() agree op-for-op.
static int run_gradcheck() {
    ensure_blas();
    CUBLAS_CHECK(cublasSetMathMode(g_blas, CUBLAS_DEFAULT_MATH));
    const int n = 64, T = 8;
    Rollout r(n, T, 42);
    PPOConfig cfg;
    if (const char* s = getenv("CRAFTAX_GC_SPLIT")) cfg.bptt_split = atoi(s);
    Trainer tr(r, cfg);
    r.reset();
    tr.collect();  // eager: rollout + bootstrap + GAE + stats
    CUDA_CHECK(cudaStreamSynchronize(r.stream));
    tr.backward(0, n);
    CUDA_CHECK(cudaStreamSynchronize(r.stream));

    if (getenv("CRAFTAX_DUMP")) {
        auto dumpb = [](const char* name, const void* dev, size_t bytes) {
            std::vector<uint8_t> h(bytes);
            CUDA_CHECK(cudaMemcpy(h.data(), dev, bytes, cudaMemcpyDeviceToHost));
            char path[256]; snprintf(path, 256, "/tmp/gcdump/%s.bin", name);
            FILE* f = fopen(path, "wb"); fwrite(h.data(), 1, bytes, f); fclose(f);
            fprintf(stderr, "dumped %s (%zu B)\n", path, bytes);
        };
        size_t tn = (size_t)T * n;
        dumpb("params", r.params, (size_t)PARAM_COUNT * 4);
        dumpb("obs", r.r_obs, tn * OBS_DIM_COMPACT);
        dumpb("states", r.r_state, tn * GRU_LAYERS * HIDDEN * 4);
        dumpb("act", r.r_act, tn * 4);
        dumpb("lp", r.r_logprob, tn * 4);
        dumpb("done", r.r_done, tn);
        dumpb("adv", tr.adv, tn * 4);
        dumpb("ret", tr.ret, tn * 4);
        dumpb("stats", tr.stats, 2 * 8);
        for (int l = 0; l <= GRU_LAYERS; l++) {
            char nm[32]; snprintf(nm, 32, "x%d", l);
            dumpb(nm, tr.x[l], tn * HIDDEN * 4);
        }
        for (int l = 0; l < GRU_LAYERS; l++) {
            char nm[32]; snprintf(nm, 32, "pre%d", l);
            dumpb(nm, tr.preb[l], tn * GRU_OUT * 4);
        }
        dumpb("logits", tr.logitsb, tn * NUM_ACTIONS * 4);
        dumpb("dlogits", tr.dlogits, tn * NUM_ACTIONS * 4);
        dumpb("dvalue", tr.dvalue, tn * 4);
        dumpb("dh3b", tr.dh3b, tn * HIDDEN * 4);
        dumpb("dhG", tr.dhG, tn * HIDDEN * 4);
        dumpb("dhX", tr.dhX, tn * HIDDEN * 4);
        dumpb("grads", tr.grads, (size_t)PARAM_COUNT * 4);
    }

    std::vector<float> g_all(PARAM_COUNT);
    CUDA_CHECK(cudaMemcpy(g_all.data(), tr.grads, PARAM_COUNT * 4, cudaMemcpyDeviceToHost));
    std::vector<double> g(PARAM_COUNT, 0.0);
    for (int i = 0; i < PARAM_COUNT; i++) g[i] = g_all[i];

    struct Seg { const char* name; int off; int count; };
    Seg segs[] = {
        {"W_enc", PARAM_W_ENC, OBS_DIM * HIDDEN},
        {"b_enc", PARAM_B_ENC, HIDDEN},
        {"W_gru", PARAM_W_GRU, W_GRU_ELEMS},
        {"W_a",   PARAM_W_A,   NUM_ACTIONS * HIDDEN},
        {"b_a",   PARAM_B_A,   NUM_ACTIONS},
        {"W_v",   PARAM_W_V,   HIDDEN},
        {"b_v",   PARAM_B_V,   1},
    };
    uint64_t rng = 0x12345678ULL;
    int fail = 0;
    float fh = 1e-3f;
    if (getenv("CRAFTAX_FDH")) fh *= (float)atof(getenv("CRAFTAX_FDH"));
    for (const Seg& s : segs) {
        double gnorm = 0.0;
        for (int i = 0; i < s.count; i++) gnorm += fabs(g[s.off + i]);
        double max_rel = 0.0, max_diff = 0.0;
        std::vector<int> idx(s.count);
        for (int i = 0; i < s.count; i++) idx[i] = s.off + i;
        std::partial_sort(idx.begin(), idx.begin() + (s.count < 8 ? s.count : 8), idx.end(),
                          [&](int a, int b) { return fabs(g[a]) > fabs(g[b]); });
        int checks = s.count < 24 ? s.count : 24;
        for (int c = 0; c < checks; c++) {
            int i;
            if (c < 8 && c < s.count) i = idx[c];
            else {
                rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
                i = s.off + (int)((rng >> 33) % s.count);
            }
            float theta;
            CUDA_CHECK(cudaMemcpy(&theta, r.params + i, 4, cudaMemcpyDeviceToHost));
            float h = fh * fmaxf(fabsf(theta), 0.1f);
            float tp = theta + h, tm = theta - h;
            CUDA_CHECK(cudaMemcpy(r.params + i, &tp, 4, cudaMemcpyHostToDevice));
            double lp = tr.loss();
            CUDA_CHECK(cudaMemcpy(r.params + i, &tm, 4, cudaMemcpyHostToDevice));
            double lm = tr.loss();
            CUDA_CHECK(cudaMemcpy(r.params + i, &theta, 4, cudaMemcpyHostToDevice));
            double fd = (lp - lm) / (2.0 * (double)(tp - tm) * 0.5);
            double diff = fabs(fd - g[i]);
            double scale = fabs(fd) > fabs(g[i]) ? fabs(fd) : fabs(g[i]);
            double rel = diff / (scale > 1e-8 ? scale : 1e-8);
            if (diff > max_diff) max_diff = diff;
            if (diff > 2e-4 && rel > max_rel) max_rel = rel;
            if (diff > 2e-4 && rel > 3e-2) {
                printf("  MISMATCH %s[%d]: analytic %.6e  fd %.6e\n",
                       s.name, i - s.off, g[i], fd);
                fail = 1;
            }
        }
        printf("%-6s mean|g| %.3e  max |fd-g| %.3e  max rel err %.4f  (%d sampled)\n",
               s.name, gnorm / s.count, max_diff, max_rel, checks);
        if (gnorm == 0.0) { printf("  FAIL: all-zero gradient segment\n"); fail = 1; }
    }
    printf("gradcheck: envs=%d steps=%d  =>  %s\n", n, T, fail ? "FAIL" : "PASS");
    return fail;
}

extern "C" int craftax_cpu_main(int num_envs, int iters);

static void usage(const char* prog) {
    fprintf(stderr,
        "usage: %s <mode> [flags]\n"
        "modes:\n"
        "  bench      env-only SPS            (--envs --iters --obs-mode --reset-mode)\n"
        "  sweep      env-only sweep over env counts x obs x reset modes\n"
        "  hash       env validation suite    (--envs --steps)\n"
        "  run        env+policy rollout SPS, CUDA graph  (--envs --horizon --iters)\n"
        "  split      same rollout, eager launches\n"
        "  runsweep   rollout sweep, graph vs eager       (--horizon)\n"
        "  verify     batched-vs-reference + eager-vs-graph validation (--envs --steps)\n"
        "  train      on-device PPO training  (--envs --horizon --iters + PPO flags)\n"
        "  gradcheck  analytic vs finite-difference gradients (fp32, no TF32)\n"
        "flags:\n"
        "  --backend cuda|cpu   (default cuda; cpu supports bench only)\n"
        "  --envs N  --iters N  --steps N  --horizon N  --obs-mode 0|1  --reset-mode 0|1\n"
        "  --lr F  --gamma F  --gae-lambda F  --clip F  --ent F  --vf F  --epochs N\n"
        "  --minibatches M  (contiguous env-range slices; default auto: sized so the\n"
        "                    GEMM backward's ~17 KB/sample buffers stay under 8 GB)\n"
        "  --bptt-split S   (BPTT segments per env in backward, default 1 = exact)\n"
        "  --lr-anneal      (linear lr decay over the run)\n",
        prog);
}

int main(int argc, char** argv) {
    const char* mode = "bench";
    const char* backend = "cuda";
    int envs = -1, iters = -1, steps = -1, horizon = 128;
    int obs_mode = 0, reset_mode = 1;
    PPOConfig cfg;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--backend") && i + 1 < argc) backend = argv[++i];
        else if (!strcmp(argv[i], "--envs") && i + 1 < argc) envs = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--horizon") && i + 1 < argc) horizon = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--obs-mode") && i + 1 < argc) obs_mode = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reset-mode") && i + 1 < argc) reset_mode = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lr") && i + 1 < argc) cfg.lr = atof(argv[++i]);
        else if (!strcmp(argv[i], "--gamma") && i + 1 < argc) cfg.gamma = atof(argv[++i]);
        else if (!strcmp(argv[i], "--gae-lambda") && i + 1 < argc) cfg.lam = atof(argv[++i]);
        else if (!strcmp(argv[i], "--clip") && i + 1 < argc) cfg.clip = atof(argv[++i]);
        else if (!strcmp(argv[i], "--ent") && i + 1 < argc) cfg.ent = atof(argv[++i]);
        else if (!strcmp(argv[i], "--vf") && i + 1 < argc) cfg.vf = atof(argv[++i]);
        else if (!strcmp(argv[i], "--epochs") && i + 1 < argc) cfg.epochs = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--minibatches") && i + 1 < argc) cfg.minibatches = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bptt-split") && i + 1 < argc) cfg.bptt_split = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lr-anneal")) cfg.lr_anneal = 1;
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) { usage(argv[0]); return 0; }
        else if (argv[i][0] != '-') mode = argv[i];
        else { usage(argv[0]); return 1; }
    }

    if (strcmp(backend, "cpu") == 0) {
        if (strcmp(mode, "bench") != 0) {
            fprintf(stderr, "cpu backend supports only: bench\n");
            return 1;
        }
        return craftax_cpu_main(envs < 0 ? 32768 : envs, iters < 0 ? 5000 : iters);
    }
    if (strcmp(backend, "cuda") != 0) { usage(argv[0]); return 1; }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s\n", prop.name);

    if (strcmp(mode, "train") == 0) {
        run_train(envs < 0 ? 262144 : envs, horizon, iters < 0 ? 200 : iters, cfg);
        return 0;
    }
    if (strcmp(mode, "gradcheck") == 0)
        return run_gradcheck();
    if (strcmp(mode, "hash") == 0)
        return run_hash(envs < 0 ? 2048 : envs, steps < 0 ? 500 : steps);
    if (strcmp(mode, "verify") == 0)
        return run_verify(envs < 0 ? 2048 : envs, steps < 0 ? 300 : steps);
    if (strcmp(mode, "sweep") == 0) {
        int sizes[] = {4096, 16384, 65536, 262144, 1048576};
        for (int rm = 0; rm <= 1; rm++)
            for (int om = 0; om <= 1; om++)
                for (int i = 0; i < 5; i++)
                    run_bench(sizes[i], iters < 0 ? 1000 : iters, om, rm);
        return 0;
    }
    if (strcmp(mode, "runsweep") == 0) {
        int sizes[] = {4096, 16384, 65536, 262144, 1048576};
        for (int k = 0; k < 2; k++)
            for (int i = 0; i < 5; i++)
                run_bench_rollout(sizes[i], horizon, iters < 0 ? 10 : iters, k);
        return 0;
    }
    if (strcmp(mode, "run") == 0 || strcmp(mode, "split") == 0) {
        run_bench_rollout(envs < 0 ? 262144 : envs, horizon,
                          iters < 0 ? 10 : iters, strcmp(mode, "run") == 0 ? 1 : 0);
        return 0;
    }
    if (strcmp(mode, "bench") == 0) {
        run_bench(envs < 0 ? 65536 : envs, iters < 0 ? 1000 : iters, obs_mode, reset_mode);
        return 0;
    }
    usage(argv[0]);
    return 1;
}
