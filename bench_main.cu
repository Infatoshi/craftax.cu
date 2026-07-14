// Pure C/CUDA harness for craftax.cu -- no Python, no torch.
//
// Build:  nvcc -O3 -arch=native --expt-relaxed-constexpr --use_fast_math \
//              bench_main.cu -o bench_main
// Usage:
//   ./bench_main bench [num_envs] [iters] [obs_mode] [reset_mode]
//   ./bench_main sweep                                  SPS sweep over env counts
//   ./bench_main hash  [num_envs] [steps]               trajectory hash + guards
//
// obs_mode:   0 = float one-hot obs (1345), 1 = compact uint8 obs (148)
// reset_mode: 0 = fused serial autoreset, 1 = split warp-cooperative reset
//
// Hash mode guards (worldgen changed to offset-based Philox, so the
// trajectory hash vs the pre-offset baseline legitimately differs;
// these checks catch degradation instead):
//   1. serial generate_world == warp generate_world_warp, bit-exact
//      (full arena memcmp after reset, and reset_mode 0 vs 1 produce
//      the same trajectory hash over all steps)
//   2. compact obs expansion reproduces float obs bit-exactly
//   3. reset diversity: distinct full-map hashes across envs
//   4. block-type histogram over all reset maps (compare vs baseline)

#include "craftax.cuh"
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

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

static uint64_t fnv1a(uint64_t h, const void* data, size_t len) {
    const uint8_t* p = (const uint8_t*)data;
    for (size_t i = 0; i < len; i++) { h ^= p[i]; h *= 0x100000001B3ULL; }
    return h;
}

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

int main(int argc, char** argv) {
    const char* mode = (argc > 1) ? argv[1] : "bench";
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s\n", prop.name);

    if (strcmp(mode, "hash") == 0) {
        int envs = (argc > 2) ? atoi(argv[2]) : 2048;
        int steps = (argc > 3) ? atoi(argv[3]) : 500;
        return run_hash(envs, steps);
    }
    if (strcmp(mode, "sweep") == 0) {
        int sizes[] = {4096, 16384, 65536, 262144, 1048576};
        for (int reset_mode = 0; reset_mode <= 1; reset_mode++)
            for (int obs_mode = 0; obs_mode <= 1; obs_mode++)
                for (int i = 0; i < 5; i++)
                    run_bench(sizes[i], 1000, obs_mode, reset_mode);
        return 0;
    }
    int envs = (argc > 2) ? atoi(argv[2]) : 65536;
    int iters = (argc > 3) ? atoi(argv[3]) : 1000;
    int obs_mode = (argc > 4) ? atoi(argv[4]) : 0;
    int reset_mode = (argc > 5) ? atoi(argv[5]) : 1;
    run_bench(envs, iters, obs_mode, reset_mode);
    return 0;
}
