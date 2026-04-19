# The "major restructuring": block-per-env with shared-memory staging

## Current model (Oracle, opt, opt3)

```
thread idx  = env idx
warp        = 32 envs running different game logic lockstep
block (256) = 256 envs

kernel entry:
  EnvState s = global_states[idx];   // AoS: 32 threads hit 32 separate 2.3 KB regions
  int action = actions[idx];

  // game logic
  do_crafting(s, action);
  if (ACT_DO) do_action(s);
  ...
  // inside each: reads/writes state fields at non-coalesced strides

  build_observation(s, &obs[idx * OBS_DIM]);
  //               ^ single thread walks 63 cells sequentially

kernel exit:
  global_states[idx] = s;
```

One env = one thread. Warp threads diverge because their envs are in different
game states (different actions, different nearby mobs, different day/night).
Every state-field access is a scatter load.

## Proposed model: block-per-env + shared-memory state

```
blockIdx.x  = env idx
threadIdx.x = 0..BLOCK-1 (cooperating workers within this env)
block size  = 64 (two warps per env)

__shared__ EnvState s_state;
__shared__ uint8_t  s_map[2048];
__shared__ float    s_obs[OBS_DIM];

// --- PHASE 1: coalesced bulk load ---
// 64 threads collectively copy ~2.3 KB of state from global into shared.
// Each thread copies 4 bytes; 64 threads * 4B = 256B per iteration; 10 iters.
// This is ONE coalesced load — no per-field stride penalty.
for (int i = tid; i < sizeof(EnvState)/4; i += BLOCK)
    ((uint32_t*)&s_state)[i] = ((uint32_t*)&global_states[env])[i];
__syncthreads();

// --- PHASE 2: game step (mostly serial) ---
// The state transitions are inherently sequential per env. Only tid=0 runs them.
// The other 63 threads idle on __syncthreads(). This is fine — we gain back
// the time in phases 1, 3, and 4.
if (tid == 0) {
    int action = actions[env];
    do_crafting(s_state, s_map, action);
    if (action == ACT_DO) do_action(s_state, s_map);
    move_player(s_state, s_map, action);
    update_mobs(s_state, s_map);      // serial per-mob loop, ~11 iters
    spawn_mobs(s_state, s_map);
    update_plants(s_state, s_map);
    update_intrinsics(s_state, action);
    // reward, done, light_level
}
__syncthreads();

// --- PHASE 3: parallel observation build ---
// 63 cells of the obs map — perfectly parallel.
// Each thread handles ~1 cell (tid < 63).
if (tid < 63) {
    int dr = (tid / 9) - 3;
    int dc = (tid % 9) - 4;
    int r = s_state.player_r + dr;
    int c = s_state.player_c + dc;
    int8_t blk = in_bounds(r,c) ? map_get(s_map, r, c) : BLK_OUT_OF_BOUNDS;

    // Zero 21 channels for this cell, set the one-hot.
    float* cell_obs = &s_obs[tid * 21];
    #pragma unroll
    for (int k = 0; k < 21; k++) cell_obs[k] = 0.0f;
    cell_obs[blk] = 1.0f;
}
__syncthreads();

// 11 mob slots — each thread stamps one. Races on same-cell are fine
// because they write different channels (chan = slot-type).
if (tid < 11) { /* stamp mob if in obs window */ }
__syncthreads();

// Inventory + stats tail (22 floats). 22 threads each do one.
if (tid < 22) { s_obs[63*21 + tid] = ...; }
__syncthreads();

// --- PHASE 4: coalesced bulk write ---
// Write obs back to global, 4B per thread.
for (int i = tid; i < OBS_DIM; i += BLOCK)
    global_obs[env * OBS_DIM + i] = s_obs[i];

// Write EnvState back (ops already mutated shared copy).
for (int i = tid; i < sizeof(EnvState)/4; i += BLOCK)
    ((uint32_t*)&global_states[env])[i] = ((uint32_t*)&s_state)[i];
```

## Where the wins come from

**Phase 1 (load) and Phase 4 (store)**: perfectly coalesced bulk copies.
- 64 threads × 4 B = 256 B per iteration = exactly 2 cache lines per warp, both used.
- Vs current: 32 threads × 4 B at stride 2.3 KB = 32 separate lines, 1 byte used per line.
- Memory traffic drops to actual working set: ~4 KB in + ~6 KB out per env per step,
  down from NCU's reported `3.4M × 128B = 435 MB/kernel` excessive traffic.

**Phase 3 (obs build)**: parallelized across 63 threads.
- Current cost: one thread does 63 cells × (1 map lookup + 21 writes) = ~1,400 ops.
- New cost: 63 threads in parallel = ~22 ops of latency (1 lookup + 21 unrolled writes).
- Roughly 60× speedup on this phase. It's currently `~64 μs` of the `~686 μs` total, so
  the phase itself is not dominant, but the *memory traffic* from 63 scattered loads
  is what the current obs build pays for, and that disappears: the map is already in
  shared memory after phase 1.

**Phase 2 (game step)**: roughly unchanged wall-time since it's serial.
- tid=0 runs everything. 63 threads idle. That's acceptable because phases 1/3/4
  pay for those idle threads' existence.

## Where the costs come from

**Occupancy calculus changes**. Today, one SM runs many blocks (256 threads each,
handling 256 envs). New design: one block per env. Block size 64.
- `EnvState` = 272 B, `map` = 2048 B, `obs` = 5.4 KB, scratch ≈ 8 KB.
- Needed shared memory per block ≈ 10 KB. At ~228 KB/SM (sm_120), max ~22 blocks/SM.
- 22 blocks × 64 threads = 1408 threads/SM, vs theoretical max 2048. Fine.
- Grid size = num_envs. At NE=32768 that's 32768 blocks on 188 SMs — plenty of waves.

**tid=0 runs game logic, others idle**. You "waste" 63/64 threads during phase 2.
That's OK when phase 2 is a small slice of total kernel time (it is — most cycles
went to memory-stalled observation/map loads). If phase 2 *became* the bottleneck
we'd need to find intra-phase parallelism — only two places support it:

1. **Mob update loop** (11 mobs). Each mob is independent. 11 threads can do them
   in parallel, but writes to shared state (`can_move_mob` reads player pos and
   other mobs) need `__syncthreads()` barriers. Probably not worth the complexity.
2. **Worldgen Perlin sweep** (4096 cells). Embarrassingly parallel — this is the
   big win for the reset kernel. 64 threads × 64 cells each, all in parallel.

**Reset kernel benefits most**. It already does 4 Perlin layers × 4096 cells =
16,384 independent computations. Current code: one thread does all 16,384 serially.
New design: 64 threads × 256 cells each = 64× speedup on worldgen. Reset is
amortized across the whole training run (episodes are 10k steps), so the absolute
impact is small — but if you're resetting often during benchmarks, it matters.

## What actually changes in the code

The structural edits, in order of how invasive they are:

1. **Kernel launch config**: `<<<num_envs, 64>>>` instead of `<<<num_envs/256, 256>>>`.
2. **Helpers take `Ctx`** (shared_mem pointers + tid), similar to opt2 but where
   `Ctx::maps`/`Ctx::state` are now `__shared__` pointers, not global.
3. **Phase-structured kernel bodies**, with `__syncthreads()` between phases.
4. **Game step gated on `if (tid == 0)`** — all serial game logic runs in one thread.
5. **Parallel obs build rewritten** as 63-way parallel (by `tid`), with the
   inventory/stats tail done by `tid < 22`.
6. **Bulk copy helpers** for load/store of EnvState and obs.
7. **Worldgen** parallelized across threads: distribute (r,c) cells to threads.
   RNG state is per-env (shared), so the Perlin angle array init is done by tid=0,
   then cells iterate with local `curand` streams derived from the shared state.
   (Or: keep worldgen on tid=0 if the reset cost is not in the hot path.)

Everything that currently takes `EnvState&` needs to take `(EnvState& state, uint8_t* map)`
or the Ctx form. About 15 call sites in the game logic, plus the three kernels.

## Expected numbers

Rough estimate based on NCU stall breakdown:

| metric | current (opt) | expected |
|---|---:|---:|
| step_only duration (NE=32k) | 622 μs | ~350 μs |
| autoreset_obs duration | 64 μs | ~40 μs |
| uncoalesced sectors | 83% | <10% |
| warp cycles per inst | 17.97 | ~6-8 |
| achieved occupancy | 14% | 50-70% |

That would give step_only ~1.8× speedup and autoreset ~1.6×. Combined wall-clock
SPS at NE=32k: roughly `15.7M → ~25-28M` env-only. Training SPS: probably
1.2–1.4× because policy forward/backward is half the wall time at NE=4096.

## Why it's "major"

Compared to the `opt3` changes (edit 4 lines, rebuild), this is:
- ~400-500 lines of C++ to rewrite (3 kernels + helper signatures)
- Shared memory budget needs to be computed carefully — if you overshoot, blocks
  won't schedule and the kernel fails silently at occupancy
- Bank conflicts in shared memory on `s_map` need thought (2048 B / 4 B = 512
  uint32s; 32 banks means conflicts if multiple threads hit the same bank on
  the same cycle, which happens in the parallel obs cell loop)
- The RNG contract is different: currently each thread advances its own `curandState`.
  With tid=0-only game logic, only one RNG per env is ever active — fine, but the
  parallel worldgen would need multiple RNG streams (or serialize the RNG-consuming
  parts).
- Debugging becomes harder: warp divergence bugs get replaced with barrier-ordering
  bugs, which are worse.

## Recommendation before committing

Do it in two stages to de-risk:

1. **Stage A — obs-only restructuring**. Rewrite just `build_observation` as a
   block-per-env kernel with shared-memory-cached map. Leave `step_only` with the
   current 1-thread-per-env design. This is maybe 150 lines of new code, no RNG
   issues. Expected gain: modest, since obs is not the dominant cost, but it
   validates the shared-memory staging pattern and gives a real number.

2. **Stage B — full step kernel restructuring**. Only do this if stage A delivers
   and the NCU profile after stage A still blames step_only's AoS loads.
   This is the 400-line rewrite with all the hazards above.

If you want, I can implement Stage A now as a concrete test of the hypothesis
before committing to Stage B.
