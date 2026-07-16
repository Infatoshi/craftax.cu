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

Full game (random actions for env-only, auto-reset included; env+policy
is the `run` rollout path with the PufferLib default policy in the loop):

| Backend | Hardware | Env only | Env + policy rollout |
|---|---|---:|---:|
| CUDA, 655k envs, compact obs | RTX PRO 6000 Blackwell | 123.2M | 101.5M |
| CUDA, 524k envs, compact obs | RTX PRO 6000 Blackwell | 122.3M | 103.0M |
| CUDA, 524k envs | RTX PRO 6000 Blackwell | 102.0M | 103.0M |
| CUDA, 262k envs, compact obs | RTX PRO 6000 Blackwell | 104.5M | - |
| CUDA, 262k envs | RTX PRO 6000 Blackwell | 94.3M | 90.8M |
| CUDA, 65k envs, compact obs | RTX PRO 6000 Blackwell | 89.5M | - |
| CUDA, 65k envs | RTX PRO 6000 Blackwell | 85.6M | 69.2M |
| CUDA, 186k envs, compact obs | RTX 3090 | 46.4M | 34.9M |
| CUDA, 182k envs | RTX 3090 | 43.0M | 34.5M |
| CUDA, 65k envs, compact obs | RTX 3090 | 40.4M | 29.3M |
| CUDA, 65k envs | RTX 3090 | 38.2M | 28.9M |
| C, 8192 envs, 32T | Ryzen 9 9950X3D | 5.6M | - |
| C, 1024 envs, 1T | Ryzen 9 9950X3D | 750K | - |
| CUDA (naive port, milestone 1), 65k envs | RTX 3090 | 454K | - |

3090 env counts top out just under 192k (~184k default, ~187k compact):
the worldgen scratch arena (~41.5KB/env) plus per-env state exhausts 24GB
before the 262k the PRO 6000 fits.

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
- Mob-spawn request compaction: only ~2% of envs attempt a spawn per
  step, so k_step keeps just the RNG draws and try-flags inline and
  enqueues the rare candidate-scan tail into a worklist that a
  warp-cooperative kernel drains densely between step and reset
  (integer prefix-sum candidate selection, exact per-env draw order
  preserved); spawn_mobs fell from 25.9% of k_step to ~1.4%, +6.6% SPS

Full-game "env + policy" runs the same PufferLib default policy
(Linear 843->32 encoder, MinGRU, actor/critic heads, categorical
sampling) without ever materializing the observation tensor: the
encoder gathers weight columns straight from SoA game state, skipping
exact-zero features, which is bit-exact because only the set of
skipped fmaf terms changes, never the summation order. Two
hash-identical variants exist -- a split path (thread-per-env policy
kernel + gameplay kernel) and a fused megakernel (`--fused 1`, policy
forward + gameplay in one launch, the fastest path on the 3090). The
gathered forward is verified bitwise against a dense reference from
the materialized 843-float obs (`runverify`), and `runhash` seals the
whole rollout (actions, logprobs, values, rewards, terminals) across
fused/split, both obs builds, and repeated runs. Action sampling uses
a dedicated Philox stream per (env, step), so the env's game RNG
sequence is untouched -- random-action trajectory anchors still hold.

Classic "env + policy" is a batched per-step pipeline around one
fixed policy: Linear(1345->256) encoder -> MinGRU x3 (hidden 256) ->
actor/value heads. A warp-cooperative encoder gathers W_enc columns
straight from live env state (the 1345-float obs is never
materialized), each GRU layer is a cuBLAS GEMM plus an elementwise
epilogue, and a fused value/sampling kernel writes the rollout. The
whole step is replayed as a CUDA graph in `run` mode; eager (`split`)
and graph rollouts are bitwise identical (canonical rollout hash
9dc17c576f43f4a3), and `verify` checks the batched forward against a
scalar fp32 reference.

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
./craftax_classic run --envs 262144 --horizon 128    # env+policy rollouts (CUDA graph)
./craftax_classic runsweep                 # rollout sweep, graph vs eager
./craftax_classic hash --envs 2048 --steps 500       # env validation suite
./craftax_classic verify --envs 2048 --steps 300     # NN fusion + rollout validation suite
./craftax_classic train --envs 262144 --horizon 128 --iters 200   # on-device PPO training
./craftax_classic gradcheck                          # analytic grads vs finite differences
./craftax_full bench --envs 8192 --iters 2048 --threads 32   # full game, random actions
./craftax_full hash --envs 64 --steps 2000           # full-game determinism anchor
./craftax_full_cuda hash --envs 64 --steps 2000      # CUDA port, same hash bit-exactly
./craftax_full_cuda bench --envs 65536 --iters 512   # full-game CUDA throughput
./craftax_full_cuda run --envs 131072 --iters 500 --fused 1  # full-game env+policy rollouts
./craftax_full_cuda runhash --envs 1024 --steps 500  # rollout determinism anchor
./craftax_full_cuda runverify --envs 1024 --steps 128  # gathered vs dense forward, bitwise
./craftax_full_cuda train --envs 8192 --horizon 128 --iters 191   # full-game on-device PPO
./craftax_full_cuda gradcheck                        # analytic grads vs finite differences
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

`./craftax_classic train` runs PPO entirely on device: rollout,
bootstrap value, GAE, advantage normalization, backward, and Adam,
with no tensor round-trips to the host inside an iteration. The
1345-float observation is never materialized for training either:
each minibatch's backward recomputes the forward at the current
parameters from the stored 148-byte compact obs, with live recurrence
inside each BPTT segment -- the stored per-step recurrent state
enters only as the constant at segment starts, which are exactly the
points where the backward sweep truncates its carried state gradient,
so the analytic gradient is the true gradient of the replayed loss.
Dense math is cuBLAS GEMMs over sample columns (forward
pre-activations, the transposed dh chain between layers, and weight
gradients accumulated with beta=1); the recurrent sweep is one thread
per (unit, env) walking t backward with a scalar carry; the sparse
encoder forward and scatter-add backward are one thread per (unit,
sample).

`./craftax_classic gradcheck` verifies every parameter segment
against central finite differences of the actual PPO loss (fp32
cuBLAS with TF32 off; loss sums accumulate in double, without which
the FD quantization noise floor produces false mismatches at the
default step). It passes with max rel err 0.0000 on all segments,
including with mid-horizon segment boundaries (`CRAFTAX_GC_SPLIT=2`
or `4` builds the check at that `--bptt-split`). Getting it green
surfaced three real device bugs, documented in the source: nvcc 13.2
miscompiles the warp-cooperative byte-walk encoder kernels at sm_120
(reads of bytes that exist nowhere in the record, plus a dropped
final store that appears and disappears with unrelated compilation
context) -- the stored-obs encoder pair is deliberately
thread-per-unit for that reason -- and the replay originally fed
stored states at every step, severing the recurrence that the
backward sweep differentiates through.

On the RTX PRO 6000, rollout runs 22.9M SPS at 16k envs (roughly flat
out to 1M envs; CUDA graph and eager are within noise) and training
end-to-end saturates at 5.8M SPS from 16k envs up with default flags.
The auto-minibatcher caps backward slices at 4096 envs to bound
buffer memory (~20KB per sample column), so throughput above 16k envs
is pinned by per-slice GEMM efficiency; `--minibatches` can force
larger slices on big-memory GPUs.

`--minibatches M` slices each epoch into M contiguous env ranges with
one backward+Adam step per slice (advantages stay normalized with
full-batch statistics); `--bptt-split S` truncates BPTT at S segment
boundaries per trajectory (S=1, the default, is the exact
full-horizon backward); `--lr-anneal` decays lr linearly. These
recipes were swept extensively on the previous thread-per-env
hidden-32/128 architecture: minibatching beat extra full-batch epochs
per unit compute, hidden 32 plateaued at episodic return ~14.6 on
capacity, and hidden 128 with `--epochs 3 --minibatches 8 --lr 5e-3
--lr-anneal --gamma 0.995 --gae-lambda 0.90 --vf 2.0 --ent 0.001
--bptt-split 8` reached ~15.3 vs PufferLib's tuned 15.8 -- that
capacity result is what motivated consolidating on the hidden-256,
3-layer policy. On the new architecture the same (h128-tuned) recipe
reaches episodic return 12.3 at 200M steps (8192 envs, ~170s on the
PRO 6000) and is still climbing, with a survival-heavy profile
(eplen ~790); the deeper model wants its own lr/entropy sweep and
longer runs, which is open work.

### Full game

`./craftax_full_cuda train` ports the same on-device PPO stack to the
full game (43 actions, 67 achievements, 9 floors): fused rollout with
training storage, bootstrap value, GAE, full-batch advantage
normalization, a warp-cooperative backward, and Adam, all resident on
device. Per
step each env stores a 996-byte compact obs record (792 packed-map
bytes + the 51-float scalar tail -- the `CRAFTAX_WG_COMPACT_OBS_SIZE`
layout, written independently of the obs build flag), its MinGRU state
input, and action/logprob/value/reward/done. Backward recomputes the
forward from the record bit-identically to the rollout (verified by a
bitwise replay check in `gradcheck`): the encoder's active set is
exactly the nonzero record entries, so the W_enc gradient scatters
`x_f * dh_enc` into only the columns the forward touched. Unlike
classic's one-hot features, full-game features are scalar-coded
(block/item/mob ids) or continuous (inventory sqrt-scales, health,
light level), so every active column's gradient is weighted by its
actual feature value. `./craftax_full_cuda gradcheck` verifies all
seven parameter segments against central finite differences of the
PPO loss (max |fd-g| ~1.6e-4) plus the replay check; the training
kernels leave the inference `run` path untouched (all six
random-action anchors and both rollout-hash canonicals unchanged).

The backward (`k_ppo_backward_warp`) is warp-cooperative: one warp
per (env, BPTT segment), lane l owning hidden units [Ul, U(l+1)) with
U = hidden/32. Encoder recompute and GRU matvecs keep the scalar
kernel's per-unit fmaf order (encoder columns and the W_enc scatter
become coalesced line loads/atomics, no staging tile needed since the
warp IS one sample); dense grads (W_gru/W_a/heads) are staged per
timestep in shared memory and block-reduced across 4 warps into
per-thread registers that persist over the whole segment, so global
atomics for them drop to one flush per block per segment. The old
thread-per-env kernel (`k_ppo_backward`) is kept as a reference,
selectable with `CRAFTAX_CU_BWD=thread`; both pass gradcheck. The warp
kernel fixes the grid starvation that made backward the wall: at 8192
envs the thread kernel launches 8192 threads (7.6% SM busy on the RTX
PRO 6000), the warp kernel 262k (74.5% busy, 133.9ms -> 21.6ms per
full-batch backward; at 262k envs it runs at 82.6% SM / 97.8% L2 hit).

Full-game training SPS on the RTX PRO 6000 (h32, horizon 128, e1 mb1
exact backward; thread-kernel -> warp-kernel):
8192 envs 5.5 -> 12.0M, 32768 15.1 -> 19.2M, 65536 14.8 -> 19.1M,
131072 18.2 -> 21.1M, 262144 18.5 -> 21.7M SPS. Training saturates at
~21.7M SPS at 262k envs, now 60% rollout / 40% backward; the largest
h32 training fit is 327,680 envs on 96GB (75.8GB used at 262k) and
65,536 on the 24GB 3090. A 1.6B-step run (65536 envs, 191 iters, the
M5 recipe with minibatches 32) takes ~157s and reaches final-window
episodic return 10.3 (see below). On the 3090 the warp backward gives
2.5 -> 3.3M SPS at 8k envs (measured under light co-tenancy).

Learning at 200M steps (8192 envs, horizon 128, seed 42) with the
hidden-32 policy, final-window episodic return (random actions ~1.2):

| recipe (all `--epochs 3 --minibatches 8 --lr-anneal --gamma 0.995 --gae-lambda 0.90 --vf 2.0 --bptt-split 8`) | ret/ep | eplen | notes |
|---|---:|---:|---|
| lr 5e-3, ent 0.001 (classic-best) | 5.45 | 325 | entropy collapses to ~1.0, crafting dies |
| lr 1e-3, ent 0.001 | 5.47 | 352 | same collapse, slower |
| lr 2.5e-3, ent 0.01 | **5.79** | 354 | crafting tree stays alive |
| lr 5e-3, ent 0.01 | 4.73 | 294 | too hot |
| lr 1e-3, ent 0.01, full-batch e1 | 3.95 | 284 | conservative control |

Unlike classic, the full game punishes the low entropy bonus: with ent
0.001 the policy collapses into pure survival (eat_cow/wake_up/plant
farming, zero tool crafting); ent 0.01 keeps entropy at ~2.0 and holds
the crafting tree open. The best recipe's final window (56k episodes):
wood pickaxe 8.7%, wood sword 4.0%, collect stone 3.7%, place stone
2.9%, place furnace 3.0%, collect coal 0.5%, stone pickaxe/dungeon
descent just emerging (0.02% / 0.12%) -- clearly beyond random
(wood pickaxe 0.6%, stone 0.09%) but far from the full 67-achievement
tree at this scale and capacity. There is no PufferLib baseline at
these hypers for the full game; the classic-tuned coefficients
transfer except for the entropy coefficient.

Scaling the same recipe to 65536 envs (191 iters = 1.6B steps, lr
2.5e-3, ent 0.01, epochs 3, minibatches 32, bptt-split 8, ~157s wall)
validates large-batch training: final-window (272k episodes) episodic
return 10.3 at eplen 581, with wood pickaxe 86%, wood sword 67%,
collect stone 68%, place furnace 65%, collect coal 9.2%, stone
pickaxe/sword emerging (0.26%/0.23%) and first iron collected. More
parallel envs explore more of the tree per update; raising the
entropy coefficient at scale is the natural next sweep.

`make full-h128` builds the hidden-128 full-game binary
(`-DCRAFTAX_HIDDEN=128`, same flag as classic; the default hidden-32
build is bit-identical to all canonicals). Its three policy paths --
scalar split (`--fused 0`), scalar fused (`--fused 1`), and a
warp-cooperative path (`--fused 2`: one warp per env, each lane owns
4 hidden units, coalesced weight-column loads, GRU matvecs by warp
broadcast in scalar accumulation order) -- produce bitwise-identical
rollouts (h128 canonical runhash 1024x500 = 0x23288380b7245ff6), and
gradcheck/replay/gathered-vs-dense all pass. Measured on an idle 3090
at 65k envs: hidden-128 rollouts run 11.9M SPS (warp) / 11.8M (scalar
fused) vs 29.8M at hidden 32 -- a 2.5x capacity tax, with the warp
path only tying scalar (the step kernel, not the policy math,
dominates). Training at 8192 envs: 0.46M SPS vs 3.05M at hidden 32.
`--ent-anneal` (linear entropy-coefficient decay) is available for
long runs; 1B-step hidden-128 runs targeting the stone/iron tier are
the natural next experiment.

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
