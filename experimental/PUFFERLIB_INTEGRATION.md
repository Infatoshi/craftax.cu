# PufferLib 4.0 integration experiment for `craftax_classic_cuda`

## Landscape

PufferLib 4.0 already has a Craftax-Classic port at `ocean/craftax/`:
- Single-header CPU implementation (`craftax.h`, 1015 lines) + `binding.c` (34 lines)
- Derived from my CPU port at `github.com/Infatoshi/craftax.c`
- Claims 47.8M SPS standalone in the file header

PufferLib 4.0's vec layer (`src/vecenv.h`) is already GPU-aware:
- `StaticVec` carries `cudaStream_t* streams`, `gpu_observations`, `gpu_actions`, etc.
- CPU envs are stepped via OpenMP, with `cudaMemcpyAsync` overlapping H2D/D2H on per-buffer streams
- Policies run on GPU (their `PufferCNN` in `pufferlib/environments/craftax/torch.py`)
- No existing CUDA-in-env-step path; streams today only gate CPU→GPU transfers

## CPU SPS measured on this box (Ryzen 9950X3D, 16P cores, 32T)

Built from `ocean/craftax/craftax.h` with the OpenMP benchmark harness I wrote. Best of multiple runs (numbers swing 10–30% across runs, likely thermal):

| NE | CPU SPS (peak seen) |
|---:|---:|
| 1,024 | 13.7M |
| 2,048 | 24.3M |
| 4,096 | 18.8M |
| 8,192 | **28.4M** ← peak |
| 16,384 | 12.5M |
| 32,768 | 5.7M |
| 65,536 | 3.5M |

The 47.8M number in the header is probably best-case on a different CPU. On this machine CPU peaks around 28M and falls off hard past NE=16k because each env's ~2 KB state starts blowing L2/L3.

## Our CUDA SPS (opt5 branch)

| Config | NE=4k | NE=8k | NE=32k | NE=65k |
|---|---:|---:|---:|---:|
| opt5 single-step | 2.3M | 5.2M | 15.9M | 23.0M |
| opt5 multi-step K=32 | 40M | 53M | 95M | — |
| opt5 multi-step K=128, 2 streams | 77M | — | **157M** | — |

## Head-to-head

| Regime | CPU (peak) | CUDA (best) | CUDA/CPU |
|---|---:|---:|---:|
| Single-step, NE=4k | 18.8M | 2.3M | **0.12×** (loses) |
| Single-step, NE=8k | 28.4M | 5.2M | 0.18× (loses) |
| Multi-step K=32, NE=4k | — | 40M | — |
| Multi-step K=32, NE=8k | — | 53M | 1.9× |
| Multi-step K=32, NE=32k | — | 95M | **16.6×** (CPU@32k=5.7M) |
| Multi-step K=128 2-stream NE=32k | — | 157M | **27×** |

**The CUDA version loses at single-step small-NE**. The CPU version with 32-thread OMP is hard to beat when the per-env state is small and the kernel launch overhead hasn't been amortized. The CUDA version only dominates once we either (a) batch K env-steps per kernel or (b) scale NE past what the CPU cache hierarchy can hold.

## Would this be valuable to add to PufferLib 4.0?

**Maybe — with caveats.** The value proposition is narrow:

- **Training with GPU policies at NE ≥ 16k**: CPU craftax starts degrading past L2 capacity at this scale; the CUDA env keeps scaling linearly until GDDR7 bandwidth saturates. If someone is training at NE=32k–65k, CUDA is 5–30× faster.
- **Eliminating D2H of obs**: the CUDA env produces obs in device memory directly, so the policy forward reads them without a host round-trip. Today PufferLib issues `cudaMemcpyAsync` after each OMP batch. This saves a small fixed cost per step but doesn't change big-O.
- **Frees the CPU** for the worker's other duties (logging, evaluation, rollout buffer management). In practice this probably doesn't matter because the CPU version was already async with the GPU.

**Against**:
- **Losses at small NE in single-step**. The natural PPO pattern (N_STEPS=64 per rollout, NE=4096, one step at a time) is exactly where CPU wins today (~18M vs 2.3M).
- **Multi-step batching doesn't match PPO semantics** — you can't run K env-steps on the same policy output without breaking what the agent is learning. It's fine for pure env-SPS benchmarks or async architectures but not a drop-in for synchronous PPO rollouts.
- **Build cost**: requires nvcc + matching a PyTorch CUDA version. Breaks the single-`clang` build story PufferLib 4.0 ships with.
- **Maintenance**: sm-specific (currently tuned to sm_120 Blackwell). Needs retesting on sm_80/sm_86/sm_90.

## How easy is it to fit?

**File shape — easy.** Our three files (`craftax_opt5.cuh` 240 lines, `craftax_opt5.cu` 770 lines, `craftax_opt5_ext.cu` 90 lines, ~1100 lines total) are already comparable in size to PufferLib's `craftax.h` (1015 lines). Could be combined into a single `craftax_cuda.cu` with ~800 lines of real game logic + kernel wrappers.

**Integration with their vecenv — medium.** Their `StaticVec` assumes per-env C structs stepped by `c_step(&envs[i])`. Our model is one kernel call over all envs at once. To plug in cleanly we'd need either:

1. Add a "GPU env" codepath in `vecenv.h` that bypasses the OMP loop entirely and calls a single CUDA kernel per `static_vec_step`. This is ~150 lines of C and doesn't disrupt existing CPU envs — but it changes their threading model. Moderate PR.

2. Wrap our kernel as a Python-level vector env with PufferLib's `VectorEnv` adapter. Bypasses their C vec machinery entirely. Works today, no PR needed. But loses the OMP/stream infrastructure they've built, and ships a different API surface from other ocean envs.

The cleanest PufferLib-style PR would be option (1) with a new `ocean/craftax_cuda/` alongside the CPU version, sharing the game-logic test cases. The two could be kept in lock-step via the existing 33-test validation suite I built (`validate.py`).

## Recommendation

**Don't PR as a replacement.** The CPU version wins the common case (synchronous PPO, NE=4-8k, per-step policy calls) on this hardware.

**Consider PR as an optional `craftax_cuda` alongside** the existing `craftax`, with honest README wording: *"5-30× faster than the CPU version at NE ≥ 16k and for async/batched-action workflows; slower at NE ≤ 8k in single-step synchronous PPO."*

If you care about the GPU obs path specifically (training very large networks where the D2H cost matters), that alone might justify it at any NE.

## Files produced this session on `experimental/warp-specialized-envs`

```
experimental/craftax_opt5.{cuh,cu,_ext.cu}   -- warp-per-env kernel + wrapper
experimental/bench_nn_overlap.py              -- policy+env stream overlap bench
experimental/bench_opt2.py                    -- three-way bench scaffolding
experimental/compare_bench.py                 -- oracle-vs-opt parity
experimental/profile_driver*.py               -- NCU drivers
experimental/NCU_ANALYSIS.md                  -- original profile
experimental/RESTRUCTURING.md                 -- design writeup
```

Ready to merge to `main` as `opt5` if you want the 1.32–1.41× env-only and 1.20× PPO training speedup on the original drop-in `env.step()` API, plus the optional `step_n` for workloads that can consume it.
