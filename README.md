# craftax.cu

CUDA reimplementation of [Craftax-Classic](https://github.com/MichaelTMatthews/Craftax) (Matthews et al. 2024) for high-throughput RL training.

> **This is Craftax-Classic, not Craftax-Full.** 17 actions, 22 achievements, a single 64x64 map. No dungeon floors, potions, enchantments, or bosses. If you want the full game, see the native C port in [PufferLib's `ocean/craftax`](https://github.com/PufferAI/PufferLib/tree/4.0/ocean/craftax).

The idea: take the fastest PufferLib environment and see how far you can get by keeping everything on GPU and writing the game logic in pure CUDA. Turns out, pretty far.

<p align="center">
  <img src="gameplay.gif" alt="Trained agent gameplay" width="300">
</p>

## Results

Env steps per second, 1M parallel envs, deterministic device-generated actions:

| | RTX 3090 | RTX PRO 6000 Blackwell |
|---|---:|---:|
| Baseline (AoS state, float obs) | 14.4M | - |
| SoA state layout | 19.6M | 65.5M |
| + compact uint8 obs (148B vs 5380B) | 36.0M | 96.7M |
| + warp-cooperative resets | **124.5M** | **301.2M** |

With the full policy fused in (rollout megakernel: PufferLib's default craftax
policy -- Linear encoder + MinGRU + actor/critic heads -- plus categorical
sampling and PPO rollout storage, T=128):

| | RTX 3090 | RTX PRO 6000 Blackwell |
|---|---:|---:|
| Per-step kernels (split path) | 79.4M @ 262k envs | 277.3M @ 1M envs |
| Rollout megakernel | **114.3M @ 262k envs** | **306.2M @ 1M envs** |

The megakernel with policy inference outruns the env-only per-step path:
once per-step launch/sync overhead is fused away, a hidden-32 policy is free.
At small batch (4096 envs) the megakernel is 2.4x the split path.

Correctness is enforced at three levels:
- 33/33 structural validation tests against a JAX reference trajectory
  (`validate.py`).
- Layout changes (SoA, compact obs) were verified bit-exact against the
  original by FNV trajectory hash over (obs, rewards, dones); the compact
  observation expands back to the 1345-float obs bit-for-bit.
- The worldgen RNG restructure (fixed Philox counter offsets, enabling
  warp-parallel generation) is trajectory-changing by nature, so it is guarded
  distributionally: block-type histograms match the sequential-draw original
  (terrain noise fields are bit-identical -- Philox offset draws equal
  sequential draws -- and ore/tree placement differs only by sampling noise),
  full-map diversity is 100%, and the serial and warp generators are verified
  bit-identical to each other. The fused NN forward is bit-exact vs a dense
  reference, and megakernel rollouts are bitwise identical to the per-step
  path (`bench_main hash`, `megakernel verify`).

Earlier torch-based training comparison (RTX 3090, 4096 envs, matched PPO
hyperparameters): 7.4x faster than the JAX original end-to-end, converging to
the same mean return (5.48 vs 4.76 at 10M steps).

## Pure CUDA benchmark and validation (no Python)

```bash
nvcc -O3 -arch=native --expt-relaxed-constexpr --use_fast_math bench_main.cu -o bench_main
./bench_main sweep                 # SPS sweep: env counts x obs modes x reset modes
./bench_main hash 2048 500         # validation: worldgen exactness, trajectory
                                   # hash, obs expansion, map diversity, histogram
./bench_main bench 1048576 1000 1 1  # single config: envs iters obs_mode reset_mode

nvcc -O3 -arch=native --expt-relaxed-constexpr --use_fast_math megakernel.cu -o megakernel
./megakernel sweep 128             # fused env+NN rollouts, mega vs split
./megakernel verify 2048 300       # bitwise NN fusion + rollout verification
```

## PyTorch interface (optional)

Requires a CUDA toolkit version that matches your PyTorch install:

```bash
uv venv && uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --no-build-isolation -e .
```

```python
import torch
import craftax_cuda

env = craftax_cuda.CraftaxEnv(num_envs=4096, seed=42)
obs = env.reset()                    # (4096, 1345) float32 on GPU
actions = torch.randint(0, 17, (4096,), dtype=torch.int32, device='cuda')
obs, rewards, dones = env.step(actions)

# compact observations: obs_mode=1 -> (n, 148) uint8; env.expand() reproduces
# the float obs bit-exactly on GPU
```

```bash
uv run bench.py       # env-only + PPO training benchmark
uv run validate.py    # 33 tests against JAX reference
```

## Architecture

Everything lives on GPU. No CPU-GPU copies in the hot path.

- **`craftax.cuh`** -- SoA env state arena (per-field arrays over envs so warp
  lanes coalesce), 4-bit packed maps, compact obs layout, device helpers
- **`craftax.cu`** -- game logic (crafting, combat, mob AI, intrinsics),
  offset-Philox worldgen (serial + warp-cooperative, bit-identical), obs
  builders, all kernels
- **`bench_main.cu`** -- pure C/CUDA harness: benchmarks + validation suite
- **`megakernel.cu`** -- fused env + policy rollout kernel (PufferLib default
  policy: Linear 1345->32 encoder, MinGRU, actor/critic heads)
- **`craftax_ext.cu`** -- optional PyTorch extension (pybind11)

Key design choices:

- **SoA state layout**: one thread steps one env; struct-of-arrays makes
  neighboring lanes touch neighboring addresses (global store efficiency went
  from 11% to 47%)
- **Compact observations**: 148 uint8 bytes instead of 1345 floats (36x less
  obs traffic); a GPU expansion kernel reproduces the float obs bit-exactly
  when needed
- **Offset-based worldgen RNG**: every Perlin angle, ore roll, and tree roll
  lives at a fixed Philox counter offset, so tiles can be generated in any
  order -- one warp generates a map cooperatively, bit-identical to the serial
  generator. Resets cost ~4us instead of ~2.5ms per step at 65k envs
- **Done-list compaction**: the step kernel atomically appends finished envs
  to a list; a work-stealing warp kernel regenerates only those
- **Rollout megakernel**: one thread per env runs a full T-step rollout with
  the policy inlined. Blocks never sync with each other; done envs are reset
  warp-cooperatively so a reset stalls only its own warp. The one-hot-dominated
  encoder collapses to a weight-column gather (bit-exact vs the dense Linear)
- **4-bit packed map**: 64x64 map in 2048 bytes
- **Per-env RNG**: `curandStatePhilox4_32_10_t` per env, counter-based, so
  sampler streams stay aligned across kernel organizations

## What is Craftax-Classic?

A procedurally generated survival game used as an RL benchmark. The agent spawns on a 64x64 tile map with resources, mobs, and crafting. 17 actions (move, mine, craft, place, sleep), 22 achievements to unlock. Episodes run for 10k timesteps. The observation is a 7x9 local view + inventory + stats = 1345-dim vector.

Originally implemented in JAX by [Matthews et al.](https://github.com/MichaelTMatthews/Craftax) for fully-GPU training with `jax.lax.scan`.

## File structure

```
craftax.cuh          # Constants, SoA state arena, device helpers
craftax.cu           # Full game logic + CUDA kernels
bench_main.cu        # Pure C/CUDA benchmark + validation harness
megakernel.cu        # Fused env + policy rollout megakernel
craftax_ext.cu       # Optional PyTorch extension binding
setup.py             # Torch extension build (auto-detects GPU arch)
bench.py             # Torch benchmark script (env-only + PPO training)
validate.py          # 33-test validation suite
gen_reference.py     # Generates JAX reference data
reference_data.npz   # JAX reference trajectory (seed=42, 100 steps)
render_gif.py        # Train agent + render gameplay GIF
```

## Citation

If you use this in your work:

```bibtex
@software{craftax_cuda,
  title={craftax.cu: CUDA Craftax-Classic},
  url={https://github.com/Infatoshi/craftax.cu},
  year={2026}
}
```

The original Craftax environment:

```bibtex
@inproceedings{matthews2024craftax,
  title={Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning},
  author={Michael Matthews and Michael Beukman and Benjamin Ellis and Mikayel Samvelyan and Matthew Jackson and Samuel Coward and Jakob Foerster},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```
