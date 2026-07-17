# craftax.cu

CUDA and C (AVX-512) implementations of [Craftax](https://github.com/MichaelTMatthews/Craftax) (Matthews et al. 2024) for high-throughput RL, as a grid of {classic, full} x {C, CUDA}:

| | C (CPU) | CUDA |
|---|---|---|
| Classic (17 actions, one 64x64 map) | `craftax_classic.c` | `craftax_classic.cu` |
| Full (43 actions, multi-floor dungeons) | `craftax_full.c` | `craftax_full.cu` |

Both CUDA paths carry batched env+policy rollouts and fully on-device
PPO training around one fixed policy architecture (Linear encoder ->
MinGRU x3, hidden 256 -> actor/value heads). The full game starts from
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
is the `run` rollout path with the h256x3 policy in the loop -- it is
policy-GEMM-bound and flat at ~22.5M SPS from 65k envs on the PRO 6000;
the retired hidden-32 gathered-scalar policy reached ~103M at 524k,
see git history):

| Backend | Hardware | Env only | Env + policy rollout |
|---|---|---:|---:|
| CUDA, 655k envs, compact obs | RTX PRO 6000 Blackwell | 123.2M | - |
| CUDA, 524k envs, compact obs | RTX PRO 6000 Blackwell | 122.3M | - |
| CUDA, 524k envs | RTX PRO 6000 Blackwell | 102.0M | - |
| CUDA, 262k envs, compact obs | RTX PRO 6000 Blackwell | 104.5M | - |
| CUDA, 262k envs | RTX PRO 6000 Blackwell | 94.3M | 22.4M |
| CUDA, 65k envs, compact obs | RTX PRO 6000 Blackwell | 89.5M | - |
| CUDA, 65k envs | RTX PRO 6000 Blackwell | 85.6M | 22.8M |
| CUDA, 186k envs, compact obs | RTX 3090 | 46.4M | - |
| CUDA, 182k envs | RTX 3090 | 43.0M | - |
| CUDA, 65k envs, compact obs | RTX 3090 | 40.4M | - |
| CUDA, 65k envs | RTX 3090 | 38.2M | - |
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

Full-game "env + policy" runs the same fixed architecture as classic
-- Linear(843->256) encoder -> MinGRU x3 (hidden 256) -> actor(43) and
value heads -- as a batched per-step pipeline. Unlike classic's sparse
one-hots, full-game features are mostly nonzero (scalar-coded ids,
continuous scales), so gathering W_enc columns from live state is a
dense matmul in disguise; instead, a warp-cooperative kernel writes
the 996-byte compact obs record (one warp per env, lane per view
cell), the record expands into an fp32 feature matrix, and the encoder
forward is one cuBLAS GEMM -- the same record the trainer stores and
replays. Each GRU layer is a cuBLAS GEMM plus an elementwise epilogue,
and a fused value/sampling kernel writes the rollout and the env's
action slot. The whole step replays as a CUDA graph (`--graph 1`);
eager and graph rollouts are bitwise identical, and `runhash` seals
the whole rollout (actions, logprobs, values, rewards, terminals)
across graph/eager, both obs builds, and repeated runs (canonical
1024x500 rollout hash 0x667d0e43f5b14ead). `runverify` checks the
batched forward against a scalar fp32 reference (max |d_h3| 5.5e-4
under TF32 GEMMs, 23/131072 action flips) on top of the bitwise
eager-vs-graph gate. Action
sampling uses a dedicated Philox stream per (env, step), so the env's
game RNG sequence is untouched -- random-action trajectory anchors
still hold.

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
./craftax_full_cuda run --envs 131072 --iters 500 --graph 1  # full-game env+policy rollouts
./craftax_full_cuda runhash --envs 1024 --steps 500  # rollout determinism anchor
./craftax_full_cuda runverify --envs 1024 --steps 128  # batched vs scalar ref + eager==graph
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
bit-identical to each other. Batched policy forwards are checked against
scalar references (bit-exact for classic, tolerance-gated under TF32 for
the full game), and eager and CUDA-graph rollouts are bitwise identical. `./craftax_classic hash` and `./craftax_classic verify` run all of it; during
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

`--horizon T` is the rollout length per PPO iteration: each iteration
collects envs x T steps of experience (episodes pause mid-rollout and
the last state's value bootstraps GAE), runs the update on that batch,
and T is also the window BPTT unrolls over. `--iters N` is the number
of collect+update iterations, so total env steps = envs x horizon x
iters. `--minibatches M` slices each epoch into M contiguous env
ranges with one backward+Adam step per slice (advantages stay
normalized with full-batch statistics); `--bptt-split S` truncates
BPTT at S segment boundaries per trajectory (S=1, the default, is the
exact full-horizon backward); `--lr-anneal` decays lr linearly. These
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

`./craftax_full_cuda train` runs the same on-device PPO stack as
classic on the full game (43 actions, 67 achievements, 9 floors), on
the same h256x3 policy: rollout with training storage, bootstrap
value, GAE, full-batch advantage normalization, a batched cuBLAS-GEMM
backward, and Adam, all resident on device. One PPO iteration
(collect + epochs x minibatches of backward + Adam) is captured once
as a CUDA graph and replayed; lr/entropy annealing and the Adam/Philox
step counters live in device memory, so per-iteration host work is a
few 4-byte writes and one graph launch (`CRAFTAX_CU_TRAIN_PROF=1`
forces eager per-phase timing). Per step each env stores a 996-byte
compact obs record (792 packed-map bytes + the 51-float scalar tail --
the `CRAFTAX_WG_COMPACT_OBS_SIZE` layout, written independently of the
obs build flag), its three MinGRU state inputs, and
action/logprob/value/reward/done. The record write is
warp-cooperative (one warp per env, lane per view cell; the serial
scatter form ran 32 blocks on 188 SMs at 8192 envs), and the rollout
policy consumes the very record it just wrote (expand + GEMM, as in
`run` mode), so the encode work is not paid twice. Backward
recomputes the forward from the record: stored states enter as
constants at BPTT segment starts, the recurrence is live (done-gated)
within segments, per-layer sweeps run the dcarry chain per (unit, env)
descending t, and every dense gradient -- including the encoder's --
is a flat GEMM over all T x mb samples with beta=1 accumulation into a
single grads arena that Adam consumes and zeroes. Unlike classic's
sparse one-hots, full-game features are scalar-coded (block/item/mob
ids) or continuous (inventory sqrt-scales, health, light level) and
mostly nonzero, so the "sparse" per-feature encoder walk was really a
dense 843x256 matmul in scalar fmafs/atomics (k_enc_bwd alone was 25%
of all GPU time, L2-atomic-bound); the records now expand into an fp32
feature matrix per minibatch (k_expand_obs) and the encoder forward
and W_enc gradient are cuBLAS GEMMs. `./craftax_full_cuda gradcheck`
runs a two-pass replay gate -- the production GEMM-encode path,
replayed per step at the rollout's own GEMM shapes, must reproduce
stored logprobs/values BITWISE at every (env, t), sealing record
integrity and replay semantics, and an independent scalar fmaf
encoder reference must match within 1e-3 (measured max |d|
4.8e-7) -- then verifies all seven
parameter segments against central finite differences of the PPO loss
at bptt-split 1, 2, and 4 (max |fd-g| ~1.7e-4). The training kernels
leave the env code untouched: all random-action trajectory anchors and
the state hash are bit-exact.

Training SPS on the RTX PRO 6000 (h256x3, horizon 128, defaults
epochs 1 / minibatches 4): 8192 envs 6.4M, 32768 7.5M, 65536 7.5M,
131072 7.5M -- 2.4x over the scatter-encoder trainer (2.5M @8192,
saturating at 3.1M): GEMM-ifying the backward encoder gave 1.9x, and
replacing the rollout's live warp encoder with the record+GEMM path
(the standalone `run` rollout went 14.0M -> 22.8M SPS) gave the
rest. The split at 8192 is now 46% rollout / 54% backward. 131,072
envs train within the 96GB card. A 20-iteration smoke at 8192x128
(21M steps, ~3s) reaches final-window episodic return 3.4 at eplen
280 (random actions ~1.2) with the crafting tree already opening
(wood pickaxe 1.3%, wood sword 1.1%, collect stone 0.3%).
Hyperparameter sweeps and long runs for the h256x3 policy on the full
game are open work; the hidden-32/128 results this section previously
reported were measured on the retired architecture and live in git
history (f4af14d and earlier).

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
