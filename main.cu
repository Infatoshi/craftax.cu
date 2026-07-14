// main.cu -- single launcher for craftax.cu: benchmarks, validation,
// and NN-fused rollouts. Pure C/CUDA, no Python.
//
// Build:  nvcc -O3 -arch=native --expt-relaxed-constexpr --use_fast_math \
//              main.cu -o craftax
// Usage:
//   ./craftax bench  [envs] [iters] [obs_mode] [reset_mode]   env-only SPS
//   ./craftax sweep                          env-only sweep (obs x reset modes)
//   ./craftax hash   [envs] [steps]          env validation: worldgen exactness,
//                                            trajectory hash, obs expansion,
//                                            map diversity, block histogram
//   ./craftax mega   [envs] [T] [iters]      fused env+policy rollout SPS
//   ./craftax split  [envs] [T] [iters]      same rollout via per-step kernels
//   ./craftax megasweep [T]                  rollout sweep, mega vs split
//   ./craftax verify [envs] [steps]          bitwise NN fusion + rollout checks
//
// obs_mode: 0 = float one-hot obs (1345), 1 = compact uint8 obs (148)
// reset_mode: 0 = fused serial autoreset, 1 = split warp-cooperative reset

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <vector>
#include <chrono>

#include "craftax.cu"

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

static void run_bench_rollout(int num_envs, int T, int iters, bool mega) {
    // Rollout buffers are n*T*(148+13+4) bytes plus env state; skip
    // configs that do not fit.
    size_t need = (size_t)num_envs * T * (OBS_DIM_COMPACT + 17) + soa_bytes(num_envs);
    size_t free_b, total_b;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    if (need + (512ull << 20) > free_b) {
        printf("NE=%8d T=%d %s: skipped (needs %.1f GB, %.1f GB free)\n",
               num_envs, T, mega ? "run  " : "split", need / 1e9, free_b / 1e9);
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
           num_envs, T, mega ? "run  " : "split", sps / 1e6, dt / (iters * (double)T) * 1e6);
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
        "flags:\n"
        "  --backend cuda|cpu   (default cuda; cpu supports bench only)\n"
        "  --envs N  --iters N  --steps N  --horizon N  --obs-mode 0|1  --reset-mode 0|1\n",
        prog);
}

int main(int argc, char** argv) {
    const char* mode = "bench";
    const char* backend = "cuda";
    int envs = -1, iters = -1, steps = -1, horizon = 128;
    int obs_mode = 0, reset_mode = 1;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--backend") && i + 1 < argc) backend = argv[++i];
        else if (!strcmp(argv[i], "--envs") && i + 1 < argc) envs = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--horizon") && i + 1 < argc) horizon = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--obs-mode") && i + 1 < argc) obs_mode = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reset-mode") && i + 1 < argc) reset_mode = atoi(argv[++i]);
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
        for (int i = 0; i < 5; i++) run_bench_rollout(sizes[i], horizon, iters < 0 ? 10 : iters, false);
        for (int i = 0; i < 5; i++) run_bench_rollout(sizes[i], horizon, iters < 0 ? 10 : iters, true);
        return 0;
    }
    if (strcmp(mode, "run") == 0 || strcmp(mode, "split") == 0) {
        run_bench_rollout(envs < 0 ? 262144 : envs, horizon,
                          iters < 0 ? 10 : iters, strcmp(mode, "run") == 0);
        return 0;
    }
    if (strcmp(mode, "bench") == 0) {
        run_bench(envs < 0 ? 65536 : envs, iters < 0 ? 1000 : iters, obs_mode, reset_mode);
        return 0;
    }
    usage(argv[0]);
    return 1;
}
