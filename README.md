# craftax.cu

CUDA and C (AVX-512) implementations of [Craftax-Classic](https://github.com/MichaelTMatthews/Craftax) (Matthews et al. 2024) for high-throughput RL.

> Craftax-Classic, not Craftax-Full: 17 actions, 22 achievements, one 64x64 map. For the full game see [PufferLib's `ocean/craftax`](https://github.com/PufferAI/PufferLib/tree/4.0/ocean/craftax).

<p align="center">
  <img src="gameplay.gif" alt="Trained agent gameplay" width="300">
</p>

## Results

Env steps per second:

| Backend | Hardware | Env only | Env + policy rollout |
|---|---|---:|---:|
| CUDA, 1M envs | RTX PRO 6000 Blackwell | 312.4M | 306.9M |
| CUDA, 1M envs | RTX 3090 | 127.9M | 116.2M @ 262k |
| C, 32k envs | Ryzen 9 9950X3D | 47.8M | - |

"Env + policy" is a rollout megakernel: the PufferLib default craftax policy
(Linear 1345->32 encoder, MinGRU, actor/critic heads), categorical sampling,
and PPO rollout storage fused into one persistent kernel, one thread per env,
no grid sync. It runs within 2% of the env-only per-step path: fusing away
launch/sync overhead roughly pays for a hidden-32 policy.

## Build

```bash
make    # -> ./craftax with both backends; CPU-only if nvcc is not installed
```

CPU backend needs AVX-512 (Zen 4/5, Ice Lake+).

```bash
./craftax bench --envs 1048576 --iters 1000 --obs-mode 1 --reset-mode 1
./craftax bench --backend cpu --envs 32768 --iters 5000
./craftax sweep                    # env SPS sweep: env counts x obs x reset modes
./craftax run --envs 262144 --horizon 128    # fused env+policy rollouts
./craftax runsweep                 # rollout sweep, fused vs per-step kernels
./craftax hash --envs 2048 --steps 500       # env validation suite
./craftax verify --envs 2048 --steps 300     # NN fusion + rollout validation suite
```

## Verification

Layout changes (SoA state, compact 148-byte obs) are bit-exact vs the
original, checked by trajectory hash; the compact obs expands back to the
1345-float obs bit-for-bit. The worldgen RNG restructure (fixed Philox
counter offsets, enabling warp-parallel generation) changes trajectories by
design, so it is guarded distributionally: block histograms match the
original, full-map diversity is 100%, and the serial and warp generators are
bit-identical to each other. The fused policy forward is bit-exact vs a dense
reference, and megakernel rollouts are bitwise identical to the per-step
path. `./craftax hash` and `./craftax verify` run all of it; during
development 33/33 structural tests also passed against a JAX reference
trajectory.

## Design notes

CUDA (`craftax.cu` device code, `main.cu` launcher):

- SoA state: one thread steps one env; per-field arrays make warp lanes
  coalesce (store efficiency 11% -> 47%)
- Compact uint8 obs: 148 bytes vs 5380 (36x less traffic), exact GPU expansion
- Packed view gather (Blackwell+): each 9-tile view row loads as two aligned
  words instead of 9 nibble reads, view lives in registers, not local memory
  (arch-gated: the extra 64-bit ALU is a net loss on Ampere)
- Offset-Philox worldgen: every draw at a fixed counter offset, so one warp
  generates a map cooperatively, bit-identical to the serial generator;
  resets drop ~2.5ms -> ~4us per step at 65k envs
- Done-list compaction + work-stealing warp reset kernel
- Rollout megakernel: T-step rollouts per launch; done envs reset
  warp-cooperatively so a straggler stalls only its own warp; the
  one-hot-dominated encoder collapses to a weight-column gather

CPU (`craftax.c`, single file): OpenMP or custom spin-barrier thread pool
over envs, AVX-512 Perlin via `permutexvar`, pipelined world-gen pool
(producer threads pre-generate reset maps on one CCD, consumers step envs on
the V-Cache CCD). Progression at 32k envs: 5.6M naive -> 47.8M.

## Citation

```bibtex
@software{craftax_cuda,
  title={craftax.cu: CUDA and C Craftax-Classic},
  url={https://github.com/Infatoshi/craftax.cu},
  year={2026}
}
```

```bibtex
@inproceedings{matthews2024craftax,
  title={Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning},
  author={Michael Matthews and Michael Beukman and Benjamin Ellis and Mikayel Samvelyan and Matthew Jackson and Samuel Coward and Jakob Foerster},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```
