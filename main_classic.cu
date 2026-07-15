// main_classic.cu -- single launcher for craftax_classic.cu: benchmarks, validation,
// and NN-fused rollouts. Pure C/CUDA, no Python.
//
// Build:  nvcc -O3 -arch=native --expt-relaxed-constexpr --use_fast_math \
//              main.cu -o craftax
//
// Hidden-128 policy build (default is hidden 32; h32 hashes unaffected):
//   nvcc -O3 -arch=native --expt-relaxed-constexpr --use_fast_math \
//        -DCRAFTAX_HIDDEN=128 main_classic.cu craftax_classic_cpu.o \
//        -o craftax_classic_h128 -Xcompiler -fopenmp -lpthread
// Usage:
//   ./craftax bench  [envs] [iters] [obs_mode] [reset_mode]   env-only SPS
//   ./craftax sweep                          env-only sweep (obs x reset modes)
//   ./craftax hash   [envs] [steps]          env validation: worldgen exactness,
//                                            trajectory hash, obs expansion,
//                                            map diversity, block histogram
//   ./craftax run    [envs] [T] [iters]      fused env+policy rollout SPS
//   ./craftax split  [envs] [T] [iters]      same rollout via per-step kernels
//   ./craftax runsweep [T]                   rollout sweep, fused vs per-step
//   ./craftax verify [envs] [steps]          bitwise NN fusion + rollout checks
//   ./craftax train  [envs] [T] [iters]      on-device PPO training (rollout +
//                                            GAE + backward + Adam, no host sync)
//   ./craftax gradcheck                      analytic grads vs finite differences
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

#include "craftax_classic.cu"

#define CUDA_CHECK(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
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

struct Rollout {
    EnvSoA g;
    uint8_t* arena;
    Weights w;
    float* params;   // flat [PARAM_COUNT] arena the Weights pointers carve
    float* h_state;
    uint8_t* r_obs;
    int32_t* r_act;
    float* r_logprob;
    float* r_value;
    float* r_reward;
    int8_t* r_done;
    float* r_state = nullptr;  // optional [T][HIDDEN][n] for training
    unsigned long long* ep_stats = nullptr;  // optional [1+22] episode stats
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

        // One flat parameter arena (training indexes it by PARAM_*
        // offsets); per-segment init matches the old separate
        // allocations bit-for-bit (same seed/subsequence/bounds).
        CUDA_CHECK(cudaMalloc(&params, PARAM_COUNT * sizeof(float)));
        float be = 1.0f / sqrtf((float)OBS_DIM);
        float bh = 1.0f / sqrtf((float)HIDDEN);
        init_weight_seg(params + PARAM_W_ENC, OBS_DIM * HIDDEN, be, 1234, 0);
        init_weight_seg(params + PARAM_B_ENC, HIDDEN, be, 1234, 1);
        init_weight_seg(params + PARAM_W_GRU, GRU_OUT * HIDDEN, bh, 1234, 2);
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
        cudaFree(arena); cudaFree(h_state); cudaFree(params);
        cudaFree(r_obs); cudaFree(r_act); cudaFree(r_logprob); cudaFree(r_value);
        cudaFree(r_reward); cudaFree(r_done);
        if (r_state) cudaFree(r_state);
        cudaFree(reset_list); cudaFree(reset_ctrl);
    }

    void alloc_states() {
        CUDA_CHECK(cudaMalloc(&r_state, (size_t)T * HIDDEN * n * sizeof(float)));
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
            r_state, ep_stats, n, T, seed, step_count);
        CUDA_CHECK(cudaGetLastError());
        step_count += T;
    }

    void run_rollout(int kind) {  // 0 split, 1 mega
        if (kind == 1) run_mega();
        else run_split();
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

static const char* ROLLOUT_NAMES[2] = {"split", "run  "};

static void run_bench_rollout(int num_envs, int T, int iters, int kind) {
    // Rollout buffers are n*T*(148+13+4) bytes plus env state; skip
    // configs that do not fit.
    size_t need = (size_t)num_envs * T * (OBS_DIM_COMPACT + 17) + soa_bytes(num_envs);
    size_t free_b, total_b;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    if (need + (512ull << 20) > free_b) {
        printf("NE=%8d T=%d %s: skipped (needs %.1f GB, %.1f GB free)\n",
               num_envs, T, ROLLOUT_NAMES[kind], need / 1e9, free_b / 1e9);
        return;
    }
    Rollout r(num_envs, T, 42);
    r.reset();
    for (int i = 0; i < 3; i++) r.run_rollout(kind);
    CUDA_CHECK(cudaDeviceSynchronize());
    double t0 = now_s();
    for (int i = 0; i < iters; i++) r.run_rollout(kind);
    CUDA_CHECK(cudaDeviceSynchronize());
    double dt = now_s() - t0;
    double sps = (double)num_envs * T * iters / dt;
    printf("NE=%8d T=%d %s: %8.1f M SPS  (%.1f us/env-step-row)\n",
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

    // Check 2: megakernel, smem-map, and split-path rollouts, all bitwise equal.
    {
        Rollout a(num_envs, steps, 42), b(num_envs, steps, 42);
        a.reset(); b.reset();
        a.run_mega(); b.run_split();
        CUDA_CHECK(cudaDeviceSynchronize());
        uint64_t ha = rollout_hash(a), hb = rollout_hash(b);
        printf("run rollout hash:   %016llx\n", (unsigned long long)ha);
        printf("split rollout hash: %016llx  (%s)\n", (unsigned long long)hb,
               ha == hb ? "MATCH" : "MISMATCH");
        if (ha != hb) fail = 1;
    }

    printf("envs=%d steps=%d  =>  %s\n", num_envs, steps, fail ? "FAIL" : "PASS");
    return fail;
}


// ============================================================
// PPO training: rollout -> bootstrap -> GAE -> backward -> Adam,
// everything resident on device; the loop syncs only to log.
// ============================================================
struct PPOConfig {
    float lr = 3e-4f, gamma = 0.99f, lam = 0.95f;
    float clip = 0.2f, ent = 0.01f, vf = 0.5f;
    int epochs = 1;       // backward+adam passes per collected batch
    int minibatches = 1;  // contiguous env-range slices per epoch
    int lr_anneal = 0;    // 1: linear lr decay over the run
    int bptt_split = 1;   // BPTT segments per env in backward (1 = exact;
                          // >1 truncates state gradients at segment cuts)
};

struct Trainer {
    Rollout& r;
    PPOConfig cfg;
    float* grads;
    int grad_copies;
    float* adam_m;
    float* adam_v;
    float* v_boot;
    float* adv;
    float* ret;
    float* loss_acc;   // [3] pg, v, ent sums (fp32, logging)
    double* stats;     // [2] adv sum, sumsq
    int adam_step = 0;
    int mb_envs;           // envs per minibatch
    long total_updates = 0;  // adam steps over the whole run (lr anneal)

    Trainer(Rollout& r_, PPOConfig cfg_) : r(r_), cfg(cfg_) {
        r.alloc_states();
        if (cfg.minibatches < 1 || r.n % cfg.minibatches != 0) {
            fprintf(stderr, "--minibatches must divide num_envs (%d %% %d != 0)\n",
                    r.n, cfg.minibatches);
            exit(1);
        }
        mb_envs = r.n / cfg.minibatches;
        if (cfg.bptt_split < 1 || r.T % cfg.bptt_split != 0 ||
            (cfg.bptt_split > 1 && mb_envs % 32 != 0)) {
            fprintf(stderr, "--bptt-split must divide horizon (%d %% %d != 0), "
                    "and envs/minibatch must be a multiple of 32\n",
                    r.T, cfg.bptt_split);
            exit(1);
        }
        // Grad copies sized for the minibatch grid, not the full batch.
        int grid = ((size_t)mb_envs * cfg.bptt_split + 255) / 256;
        grad_copies = grid < 512 ? grid : 512;
        CUDA_CHECK(cudaMalloc(&grads, (size_t)grad_copies * PARAM_COUNT * sizeof(float)));
        CUDA_CHECK(cudaMemset(grads, 0, (size_t)grad_copies * PARAM_COUNT * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&adam_m, PARAM_COUNT * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&adam_v, PARAM_COUNT * sizeof(float)));
        CUDA_CHECK(cudaMemset(adam_m, 0, PARAM_COUNT * sizeof(float)));
        CUDA_CHECK(cudaMemset(adam_v, 0, PARAM_COUNT * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&v_boot, r.n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&adv, (size_t)r.n * r.T * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ret, (size_t)r.n * r.T * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&loss_acc, 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&stats, 2 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&r.ep_stats, 23 * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(r.ep_stats, 0, 23 * sizeof(unsigned long long)));
    }
    ~Trainer() {
        cudaFree(grads); cudaFree(adam_m); cudaFree(adam_v);
        cudaFree(v_boot); cudaFree(adv); cudaFree(ret);
        cudaFree(loss_acc); cudaFree(stats);
        cudaFree(r.ep_stats); r.ep_stats = nullptr;
    }

    // Rollout + advantage pipeline (no param update).
    void collect() {
        r.run_mega();
        int block = 256, grid = (r.n + block - 1) / block;
        bootstrap_value_kernel<<<grid, block>>>(
            r.g, r.w, r.h_state, r.r_done + (size_t)(r.T - 1) * r.n, v_boot, r.n);
        gae_kernel<<<grid, block>>>(
            r.r_value, r.r_reward, r.r_done, v_boot, adv, ret,
            r.n, r.T, cfg.gamma, cfg.lam);
        CUDA_CHECK(cudaMemsetAsync(stats, 0, 2 * sizeof(double)));
        adv_stats_kernel<<<256, 256>>>(adv, (size_t)r.n * r.T, stats);
        CUDA_CHECK(cudaGetLastError());
    }

    // Backward over a contiguous env range [env_start, env_start+env_count).
    void backward(int env_start, int env_count) {
        int block = 256;
        int grid = (int)(((size_t)env_count * cfg.bptt_split + block - 1) / block);
        ppo_backward_kernel<<<grid, block>>>(
            r.w, r.r_obs, r.r_act, r.r_logprob, r.r_done, r.r_state,
            adv, ret, grads, grad_copies, loss_acc, stats,
            r.n, r.T, env_start, env_count, cfg.bptt_split,
            cfg.clip, cfg.vf, cfg.ent);
        CUDA_CHECK(cudaGetLastError());
    }

    void adam() {
        adam_step++;
        float lr = cfg.lr;
        if (cfg.lr_anneal && total_updates > 0)
            lr *= 1.0f - (float)(adam_step - 1) / (float)total_updates;
        adam_kernel<<<(PARAM_COUNT + 255) / 256, 256>>>(
            r.params, grads, grad_copies, adam_m, adam_v,
            adam_step, lr, 0.9f, 0.999f, 1e-8f);
        CUDA_CHECK(cudaGetLastError());
    }

    void update() {
        collect();
        for (int e = 0; e < cfg.epochs; e++) {
            // loss_acc accumulates over the epoch's minibatches; the log
            // line after update() reports the last epoch's full-batch sums.
            CUDA_CHECK(cudaMemsetAsync(loss_acc, 0, 3 * sizeof(float)));
            for (int m = 0; m < cfg.minibatches; m++) {
                backward(m * mb_envs, mb_envs);
                adam();
            }
        }
    }

    // Total PPO loss over the stored rollout at the current params
    // (recomputes the recurrent forward; used by gradcheck FD).
    double loss() {
        double zero[3] = {0, 0, 0};
        double* d_losses;
        CUDA_CHECK(cudaMalloc(&d_losses, 3 * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_losses, zero, 24, cudaMemcpyHostToDevice));
        int block = 256, grid = (r.n + block - 1) / block;
        ppo_loss_kernel<<<grid, block>>>(
            r.w, r.r_obs, r.r_act, r.r_logprob, r.r_done, r.r_state,
            adv, ret, d_losses, stats, r.n, r.T, cfg.clip);
        CUDA_CHECK(cudaGetLastError());
        double h[3];
        CUDA_CHECK(cudaMemcpy(h, d_losses, 24, cudaMemcpyDeviceToHost));
        cudaFree(d_losses);
        double count = (double)r.n * r.T;
        return (h[0] + cfg.vf * h[1] - cfg.ent * h[2]) / count;
    }
};

static void run_train(int num_envs, int T, int iters, PPOConfig cfg) {
    Rollout r(num_envs, T, 42);
    Trainer tr(r, cfg);
    tr.total_updates = (long)iters * cfg.epochs * cfg.minibatches;
    r.reset();
    printf("train: hidden=%d envs=%d horizon=%d iters=%d lr=%g gamma=%g lam=%g clip=%g ent=%g vf=%g epochs=%d minibatches=%d lr_anneal=%d bptt_split=%d\n",
           HIDDEN, num_envs, T, iters, cfg.lr, cfg.gamma, cfg.lam, cfg.clip, cfg.ent, cfg.vf,
           cfg.epochs, cfg.minibatches, cfg.lr_anneal, cfg.bptt_split);

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
    for (int it = 1; it <= iters; it++) {
        tr.update();
        steps_done += (size_t)num_envs * T;
        steps_window += (size_t)num_envs * T;
        if (it == 1 || it % 10 == 0 || it == iters) {
            CUDA_CHECK(cudaMemsetAsync(rew_stats, 0, 16));
            adv_stats_kernel<<<256, 256>>>(r.r_reward, (size_t)num_envs * T, rew_stats);
            float h_loss[3];
            double h_rew[2];
            unsigned long long h_eps[23];
            CUDA_CHECK(cudaMemcpy(h_loss, tr.loss_acc, 12, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_rew, rew_stats, 16, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_eps, r.ep_stats, 23 * 8, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemsetAsync(r.ep_stats, 0, 23 * 8));
            total_eps += h_eps[0];
            for (int a = 0; a < 22; a++) total_ach[a] += h_eps[1 + a];
            double count = (double)num_envs * T;
            double sps = steps_done / (now_s() - t0);
            double eplen = h_eps[0] ? (double)steps_window / h_eps[0] : 0.0;
            double retep = (h_rew[0] / count) * eplen;  // rew/step * mean ep len
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
// and FD is exact up to fp32 forward noise.
static int run_gradcheck() {
    const int n = 64, T = 8;
    Rollout r(n, T, 42);
    PPOConfig cfg;
    Trainer tr(r, cfg);
    r.reset();
    tr.collect();
    CUDA_CHECK(cudaMemsetAsync(tr.loss_acc, 0, 3 * sizeof(float)));
    tr.backward(0, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> g_all((size_t)tr.grad_copies * PARAM_COUNT);
    CUDA_CHECK(cudaMemcpy(g_all.data(), tr.grads, g_all.size() * 4, cudaMemcpyDeviceToHost));
    std::vector<double> g(PARAM_COUNT, 0.0);
    for (int c = 0; c < tr.grad_copies; c++)
        for (int i = 0; i < PARAM_COUNT; i++) g[i] += g_all[(size_t)c * PARAM_COUNT + i];

    struct Seg { const char* name; int off, count; };
    Seg segs[] = {
        {"W_enc", PARAM_W_ENC, OBS_DIM * HIDDEN},
        {"b_enc", PARAM_B_ENC, HIDDEN},
        {"W_gru", PARAM_W_GRU, GRU_OUT * HIDDEN},
        {"W_a",   PARAM_W_A,   NUM_ACTIONS * HIDDEN},
        {"b_a",   PARAM_B_A,   NUM_ACTIONS},
        {"W_v",   PARAM_W_V,   HIDDEN},
        {"b_v",   PARAM_B_V,   1},
    };
    uint64_t rng = 0x12345678ULL;
    int fail = 0;
    for (const Seg& s : segs) {
        double gnorm = 0.0;
        for (int i = 0; i < s.count; i++) gnorm += fabs(g[s.off + i]);
        double max_rel = 0.0, max_diff = 0.0;
        // Check the 8 largest-|g| params (FD can resolve their relative
        // error) plus 16 random ones (absolute agreement).
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
            float h = 1e-3f * fmaxf(fabsf(theta), 0.1f);
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
        "  run        fused env+policy rollout SPS   (--envs --horizon --iters)\n"
        "  split      same rollout via per-step kernels\n"
        "  runsweep   rollout sweep, run vs split    (--horizon)\n"
        "  verify     NN fusion + rollout validation (--envs --steps)\n"
        "  train      on-device PPO training  (--envs --horizon --iters + PPO flags)\n"
        "  gradcheck  analytic vs finite-difference gradients\n"
        "flags:\n"
        "  --backend cuda|cpu   (default cuda; cpu supports bench only)\n"
        "  --envs N  --iters N  --steps N  --horizon N  --obs-mode 0|1  --reset-mode 0|1\n"
        "  --lr F  --gamma F  --gae-lambda F  --clip F  --ent F  --vf F  --epochs N\n"
        "  --minibatches M  (contiguous env-range slices per epoch, default 1)\n"
        "  --bptt-split S   (BPTT segments per env in backward, default 1 = exact;\n"
        "                    S>1 truncates state grads at segment cuts, more parallelism)\n"
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
