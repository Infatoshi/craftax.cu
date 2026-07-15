# craftax.cu

CUDA and C (AVX-512) implementations of [Craftax](https://github.com/MichaelTMatthews/Craftax) (Matthews et al. 2024) for high-throughput RL, as a grid of {classic, full} x {C, CUDA}:

| | C (CPU) | CUDA |
|---|---|---|
| Classic (17 actions, one 64x64 map) | `craftax_classic.c` | `craftax_classic.cu` |
| Full (43 actions, multi-floor dungeons) | `craftax_full.c` | `craftax_full.cu` |

The classic CUDA path is the most developed: fused env+policy rollout
megakernel plus fully on-device PPO training. The full game starts from
[PufferLib's `ocean/craftax`](https://github.com/PufferAI/PufferLib/tree/4.0/ocean/craftax)
C port (parity-verified against the JAX original) and follows the same
optimization ladder: C first, then CUDA, then a megakernel.

<p align="center">
  <img src="gameplay.gif" alt="Trained agent gameplay" width="300">
</p>

## Results (classic)

Env steps per second:

| Backend | Hardware | Env only | Env + policy rollout |
|---|---|---:|---:|
| CUDA, 1M envs | RTX PRO 6000 Blackwell | 312.4M | 306.9M |
| CUDA, 1M envs | RTX 3090 | 127.9M | 116.2M @ 262k |
| C, 32k envs | Ryzen 9 9950X3D | 47.8M | - |

Full game (env only, random actions, auto-reset included):

| Backend | Hardware | SPS |
|---|---|---:|
| C, 8192 envs, 32T | Ryzen 9 9950X3D | 5.6M |
| C, 1024 envs, 1T | Ryzen 9 9950X3D | 750K |
| CUDA, 262k envs, compact obs | RTX PRO 6000 Blackwell | 104.5M |
| CUDA, 262k envs | RTX PRO 6000 Blackwell | 94.3M |
| CUDA, 65k envs, compact obs | RTX PRO 6000 Blackwell | 89.5M |
| CUDA, 65k envs | RTX PRO 6000 Blackwell | 85.6M |
| CUDA, 65k envs, compact obs | RTX 3090 | 18.8M |
| CUDA, 65k envs | RTX 3090 | 18.3M |
| CUDA (naive port, milestone 1), 65k envs | RTX 3090 | 454K |

The full-game CUDA port is **bit-identical to the C build** -- the
trajectory hash matches the CPU reference exactly at 64x2000 and
4x20000 steps (gcc's -ffast-math float semantics are emulated:
reciprocal division at literal-divisor sites, a host-computed
glibc-cosf light table, and -fmad=false; fmad-on measured no faster,
so the exact build is the only one). Every optimization rung
re-verified the same anchors, so the 40x over the naive port changed
no trajectory bit:

- Split step/reset/encode kernels with done-list compaction, so the
  9-floor worldgen tail no longer stalls every stepping env
- Lazy floor generation: only floor 0 at reset, deeper floors on
  first descent (matching upstream Craftax semantics)
- Warp-cooperative floor-0 worldgen: per-floor RNG keys derive
  up-front (the "threefry" chain is a splittable 64-bit LCG) and
  per-cell draws are pure functions of (key, cell), so one warp
  generates a map in shared memory with order-free reductions and a
  chunked scan that reproduces the scalar FP cumsum bit-exactly
- Worldgen scratch moved to a global arena (per-thread stack 132KB ->
  16KB), warp-transposed coalesced obs encode, list-free two-pass
  mob-spawn scans, and optional compact byte obs (996B vs 3372B,
  its own hash universe, matched against the C compact build)
- SoA split of the hot scalar state (52 fields via one X-macro
  registry: player scalars, inventory, unified 5-class mob pool,
  spawn bitsets), so per-thread loads coalesce across envs; the env
  index is recovered from pointer arithmetic so game logic is
  untouched
- Obs encode writes each cell's 8 channels as one contiguous run
  (one lane per cell, light map loaded once instead of per channel);
  visible mobs scatter directly instead of 130 per-cell scans. The
  encode kernel now runs at ~71% of peak memory throughput, close to
  the write-bandwidth floor of the float obs tensor
- Lazy reset zeroing: on auto-reset a warp clears only the SoA slices
  of floors actually generated during the dying episode (tracked in
  the pending-floor bitmask); never-visited floors provably still
  hold post-reset values

"Env + policy" is a rollout megakernel: the PufferLib default craftax policy
(Linear 1345->32 encoder, MinGRU, actor/critic heads), categorical sampling,
and PPO rollout storage fused into one persistent kernel, one thread per env,
no grid sync. It runs within 2% of the env-only per-step path: fusing away
launch/sync overhead roughly pays for a hidden-32 policy.

## Build

```bash
make classic   # -> ./craftax_classic, CUDA+CPU backends; CPU-only without nvcc
make full      # -> ./craftax_full (CPU)
```

The classic CPU backend needs AVX-512 (Zen 4/5, Ice Lake+).

```bash
./craftax_classic bench --envs 1048576 --iters 1000 --obs-mode 1 --reset-mode 1
./craftax_classic bench --backend cpu --envs 32768 --iters 5000
./craftax_classic sweep                    # env SPS sweep: env counts x obs x reset modes
./craftax_classic run --envs 262144 --horizon 128    # fused env+policy rollouts
./craftax_classic runsweep                 # rollout sweep, fused vs per-step kernels
./craftax_classic hash --envs 2048 --steps 500       # env validation suite
./craftax_classic verify --envs 2048 --steps 300     # NN fusion + rollout validation suite
./craftax_classic train --envs 262144 --horizon 128 --iters 200   # on-device PPO training
./craftax_classic gradcheck                          # analytic grads vs finite differences
./craftax_full bench --envs 8192 --iters 2048 --threads 32   # full game, random actions
./craftax_full hash --envs 64 --steps 2000           # full-game determinism anchor
./craftax_full_cuda hash --envs 64 --steps 2000      # CUDA port, same hash bit-exactly
./craftax_full_cuda bench --envs 65536 --iters 512   # full-game CUDA throughput
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
path. `./craftax_classic hash` and `./craftax_classic verify` run all of it; during
development 33/33 structural tests also passed against a JAX reference
trajectory.

## Design notes

Classic CUDA (`craftax_classic.cu` device code, `main_classic.cu` launcher):

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

Classic CPU (`craftax_classic.c`, single file): OpenMP or custom spin-barrier thread pool
over envs, AVX-512 Perlin via `permutexvar`, pipelined world-gen pool
(producer threads pre-generate reset maps on one CCD, consumers step envs on
the V-Cache CCD). Progression at 32k envs: 5.6M naive -> 47.8M.

## Training

`./craftax_classic train` runs PPO entirely on device: fused
rollout, bootstrap value, GAE, advantage normalization, backward, and Adam,
with no tensor round-trips to the host inside an iteration. The 1345-float
observation is never materialized for training either: the backward pass
recomputes activations from the stored 148-byte compact obs plus a stored
per-step recurrent state (for BPTT through the MinGRU, truncated at episode
boundaries), and the encoder gradient is a sparse scatter-add into the ~70
active one-hot columns per sample. Gradients accumulate into per-block
copies in L2/HBM (block-level data parallelism, the single-GPU analogue of
an all-reduce) that the Adam kernel reduces. `./craftax_classic gradcheck` verifies
every parameter segment against central finite differences of the actual
PPO loss.

Training runs at ~81M SPS at 262k envs on the RTX PRO 6000 (vs 306.9M
rollout-only). Two backward-pass restructures got it there from an
initial 15M: dense grads (GRU + heads) are shuffle-reduced across the
warp so one lane issues one atomic per element instead of 32, and the
sparse encoder scatter is warp-cooperative -- the warp stages its 32
dh_enc vectors in shared memory and walks its samples together, so
every feature update is one coalesced line-atomic instead of 32
scattered ones. The remaining gap to rollout speed is the ~70
irreducible encoder column atomics per sample.

Learning was validated against PufferLib's PPO on craftax_classic at
matched hyperparameters (8192 envs, horizon 128, full-batch updates,
lr 3e-4, 200M steps): this trainer reaches episodic return 3.7 vs 1.1
for PuffeRL forced into the same single-update regime. With more
optimization pressure per batch (`--epochs 16 --lr 2e-3`) it reaches
13.1 (12.9 on the RTX 3090).

The trainer also supports minibatched updates (`--minibatches M`
slices each epoch into M contiguous env ranges with one backward+Adam
step per slice, keeping BPTT within an env intact; advantages stay
normalized with full-batch statistics) and flag-gated linear lr decay
(`--lr-anneal`). Minibatching is worth more per unit compute than
extra full-batch epochs: `--epochs 3 --minibatches 8 --lr 5e-3`
reaches episodic return 14.6 (eplen ~400, 17 of 22 achievements above
1%, wake_up 0.83 vs 0.08 for the full-batch recipe) in 344s on the
3090 -- better than the 16-epoch full-batch recipe at a quarter of
its update count. PufferLib's tuned-default recipe reaches 15.8 with
128 minibatch updates per batch, annealed lr 0.015, and a hidden-128
policy; the remaining gap at hidden 32 is dominated by model capacity
(deeper minibatching, lr annealing, and PufferLib's
gamma/lambda/vf/ent coefficients were all swept and plateau at
~14.2-14.6).

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
