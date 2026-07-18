# Overnight run: full-game train SPS at envs <= 8192

STATUS: PARTIAL (final) -- best 3.39 M SPS @8192 on RTX 3090 (1.51x, strong-win gate met there) and 7.56 M SPS @8192 on the RTX PRO 6000 (1.20x). PRO 6000 matrix filled 2026-07-18 on idle gpu0 (interleaved A/B, 2 reps, bit-stable SPS): the bf16 win does not transfer at the 3090's magnitude because sm_120 TF32 GEMMs were already fast; at 1024 the bf16 path REGRESSES (2.39 -> 2.23 M) and CF_GEMM_BF16=0 on HEAD matches baseline, so small-N default should stay/auto-gate to fp32-TF32.

Date: 2026-07-17/18. Agent: craftax-fable5-overnight. Branch: main, base 29fe464.

## What shipped (commits on main, not pushed)

1. `54bfc6d` Train: bf16 tensor-core GEMM path via cublasGemmEx (CF_GEMM_BF16, default 1)
   - All policy/trainer GEMMs (rollout forward, loss replay, windowed backward) take bf16 inputs via `cublasGemmEx` (CUDA_R_16BF in, CUBLAS_COMPUTE_32F), ~2x TF32 GEMM throughput on GA102 and sm_120.
   - No standalone conversion passes: producer kernels (obs expand, bias/epilogue, head backward) write bf16 shadows alongside fp32.
   - Gate GEMMs emit bf16 C directly (`pre_bf`); the MinGRU window backward reads/writes bf16 pre in place, eliminating the fp32 pre traffic.
   - Obs ld padded 843->848 so bf16 columns are 16B-aligned (GemmEx align8 kernels).
   - Coalesced column-sum kernels for bias grads (k_colsum256, k_colsum_small) replace strided reductions.
2. `831a023` Train backward: pipelined bf16 windows across three streams
   - Window w+1 forward recompute (stream st2) overlaps window w reverse chain (main stream); dW GEMMs drain on st3. Three cuBLAS handles (one per stream -- a shared handle races its workspace). Double-buffered window workspace by parity, disjoint grad segments, event-joined before Adam. Works under CUDA graph capture (events become graph edges).
3. `84f2171` Train: bf16 BPTT state record (r_state slab)
   - The rollout's recurrent-state record is stored bf16 (halves the slab and its bandwidth). Loss replay reads the same bf16 record, so gradcheck FD measures the differentiated function.

Verification gates keep strict fp32: `gradcheck` forces CUBLAS_DEFAULT_MATH + CF_GEMM_BF16=0 (same precedent as TF32 before this work), `runverify` forces bf16 off. `CF_GEMM_BF16=0` at runtime reproduces the 29fe464 build bit-exactly (verified: identical 1024x500 rollout hash 0xd849cb5574b70018 on the 3090).

## Acceptance gates

- `make craftax_full_cuda` builds: PASS
- `./craftax_full_cuda gradcheck --seed 42`: PASS (every step of the way; final: max rel err 0.0000, 24+1 samples W_v/b_v)
- `runverify`: PASS (max |d_h3| 5.5e-4, action flips 23/131072, eager==graph hash)
- Env trajectory anchors bit-exact vs C reference: PASS (64x2000 -> 0x38aca5eb88691859, 4x20000 -> 0x570443353b70a807, identical to `./craftax_full`)
- Convergence parity at 8192x128x12 (3090): ret/ep +2.846 vs baseline +2.850, vf 0.0920 vs 0.0923, ent 3.306 vs 3.304, run totals +1.845 vs +1.847 over ~42k episodes.

## SPS matrix -- RTX 3090 (GPU1), iters 12, seed 42, horizon 128, sole tenant on gpu1

| envs | 29fe464 baseline | after (bf16 default) | speedup |
|------|------------------|----------------------|---------|
| 1024 | 1.14 M SPS | 1.27 M SPS | 1.11x |
| 4096 | 1.86 M SPS | 2.71 M SPS | 1.46x |
| 8192 | 2.24 M SPS | 3.39 M SPS | 1.51x |

Baseline = the actual 29fe464 binary (worktree build), not a flag-off approximation.

Phase breakdown at 8192 (CRAFTAX_CU_TRAIN_PROF=1, same-session pair, ratio 1.49x):
- baseline: rollout 2.29s (43.9%), gae 0.3%, backward 2.91s (55.8%), adam 0.0%
- bf16:     rollout 1.77s (50.8%), gae 0.3%, backward 1.71s (48.9%), adam 0.0%
- at 1024 the bf16 build is rollout-bound: rollout 79.0%, backward 20.7% -- small-N is per-step latency floor (worldgen reset warp ~200us + divergent step kernel), which is env-side and anchor-risky, deliberately not touched.

## SPS matrix -- RTX PRO 6000 Blackwell (GPU0)

Filled 2026-07-18 ~14:20 MDT, idle gpu0, sole tenant, iters 12, seed 42, horizon 128. Interleaved baseline/bf16/bf16-off, 2 full repetitions -- every number reproduced to the displayed precision.

| envs | 29fe464 baseline | after (bf16 default) | HEAD w/ CF_GEMM_BF16=0 | speedup (bf16) |
|------|------------------|----------------------|------------------------|----------------|
| 1024 | 2.39 M SPS | 2.23 M SPS | 2.40 M SPS | 0.93x (regression) |
| 4096 | 5.36 M SPS | 5.74 M SPS | -- | 1.07x |
| 8192 | 6.29 M SPS | 7.56 M SPS | 6.43 M SPS | 1.20x |

Reading: sm_120's TF32 GEMMs are already fast enough that bf16 conversion overhead (producer shadow writes, bf16 round trips) eats most of the tensor-core gain; the non-bf16 HEAD changes (3-stream pipelined windows + slab) contribute only ~+2% here (6.29 -> 6.43). At 1024 bf16 is a straight loss -- consider auto-gating CF_GEMM_BF16 off below ~2-4k envs.

Phase breakdown on the PRO 6000 (CRAFTAX_CU_TRAIN_PROF=1, bf16 default):
- 8192: rollout 50.8% / backward 49.0% (same split as the 3090)
- 1024: rollout 82.0% / backward 17.7% (bf16 off: 82.5/17.2 -- rollout-latency floor, env-side)

## Tried, in order

1. nsys profile at 8192 (3090): ~52% of GPU time in cuBLAS TF32 GEMMs; per-iter ~435ms = rollout 192ms + backward 243ms. -> GEMM precision was the biggest lever.
2. bf16 cuBLAS GemmEx everywhere with producer-written shadows: 2.24 -> 2.80 M SPS @8192.
3. bf16 C-output for gate GEMMs + bf16 pre pipeline through the window backward: -> 3.13.
4. ld pad 843->848 (align8), bf16 value dot, coalesced colsum kernels: -> 3.25 (commit 54bfc6d).
5. dW GEMMs offloaded to a side stream: -> 3.31.
6. Full three-stream pipelined backward (fwd recompute w+1 under reverse w): -> 3.33 (commit 831a023). Small gain on the 3090 (kernels already fill 82 SMs); expected to matter more on the PRO 6000 (188 SMs) and at small N.
7. bf16 r_state BPTT slab: -> 3.39 (commit 84f2171).
8. W sweep under bf16: W=32 -> 3.38, W=64 -> 3.39 vs W=16 -> 3.33 at the time, but W>=32 trips the auto-minibatch memory cap at 8192 (splits into 2 minibatches, changes optimizer semantics). Kept W=16 default.
9. Ruled out: k_record_obs rework (already warp-cooperative, scatter-bound), worldgen reset parallelization (anchor-risky, serial warp-per-env ~217us), rollout overlap (strict policy->step->reset dependency).

Not re-done (per brief): full-horizon materialize, streaming reverse, W=32 default, custom WMMA/bf16-acc kernels, env megakernel, occupancy forcing.

## Runhash reseal

The 1024x500 rollout hash is GPU-arch-specific (cuBLAS picks different GEMM kernels per arch). The old README seal 0x667d0e43f5b14ead did not reproduce on the 3090 even with the 29fe464 binary (gives 0xd849cb5574b70018 there); it was presumably a PRO 6000 value. New seals documented in README once the PRO 6000 value is captured. bf16-default hash on the 3090: 0xf1bb646444f92b3f. Policy numerics changed by design (bf16 GEMM inputs); env trajectories did not.

## Follow-up (2026-07-18 afternoon session): all three next steps landed

Commits `000a14e` (bf16 auto-gate), `79c9e65` (block-cooperative reset,
CRAFTAX_CU_RESET_BLOCK default 128), `0edc71c` (reset/record overlap,
trainer-only, default on >=4096 envs). All env/statehash/runhash anchors
EXACT (both obs builds, mega path, NT=32/128/256), gradcheck PASS.

RTX 3090 train SPS (idle GPU1, iters 12 seed 42):
| envs | overnight best | after follow-ups |
|------|----------------|------------------|
| 1024 | 1.27 M | 1.44 M (+13%) |
| 4096 | 2.71 M | 2.93 M (+8%) |
| 8192 | 3.39 M | 3.56 M (+5%) |

Landmine: __restrict__ on smem pointer params of the extracted record
helper broke the lane0->warp smem handoff (illegal local write).

RTX PRO 6000 train SPS (idle gpu0 ~15:15 MDT, interleaved, 2 reps
bit-stable; "pre" = same binary with NT=32 + no overlap):
| envs | pre | after | vs 29fe464 baseline |
|------|-----|-------|---------------------|
| 1024 | 2.39 M | 2.80 M (+17%) | 2.39 M -> 1.17x |
| 4096 | 5.74 M | 6.21 M (+8%) | 5.36 M -> 1.16x |
| 8192 | 7.43 M | 7.96 M (+7%) | 6.29 M -> 1.27x |
Best number at <=8192 envs: 7.96 M train SPS @8192 (PRO 6000).
runhash 1024x500 = 0x667d0e43f5b14ead (fp32 canonical, autogate) and
64x2000 env anchor re-verified on gpu0 post-change.

## Follow-up 2 (2026-07-18 evening): value head folded into the actor GEMM

Commit `aa69062`: rollout head is one [44][256] GEMM (W_v packed as row
43 into W_av/W_av_bf after init/adam); k_value_sample is now only bias
add + categorical sample. Motivation: ncu showed the old kernel flat at
~27.5us from 1024 to 8192 envs (44 warp cycles/issued instruction --
pure serial latency from the 256-FMA value dot + uncoalesced h3 walk).
Same commit: under reset/record overlap, k_record_obs_list chases the
reset on st_reset (concurrent with the survivors' record) instead of
running on the main stream -- removes its 12-16us from the @8192
critical path.

Verification: env anchors EXACT both obs builds; gradcheck PASS with
replay EXACT 0/512 (the gradcheck replay uses the same fused-head GEMM
shape, and also passes with CRAFTAX_CU_RESET_OVERLAP=1, bit-exact slab
records from the side-stream record path); runverify PASS eager==graph.
Runhash resealed (value summation order changed by design): 1024x500
s42 = 0x1dc7e4e16d65e32d on the PRO 6000 fp32-autogate path (prior
0x667d0e43f5b14ead).

RTX 3090 train SPS (idle gpu1, interleaved base/after, 2 reps, iters 12
seed 42 horizon 128; base = commit 0edc71c binary):
| envs | base | after fold |
|------|------|------------|
| 1024 | 1.37 M | 1.47 M (+7%) |
| 4096 | 2.89 M | 2.97 M (+3%) |
| 8192 | 3.51 M | 3.56 M (+1.5%) |

RTX PRO 6000 train SPS (idle gpu0, interleaved base/after, 2 reps,
same protocol):
| envs | base | after fold | vs 29fe464 baseline |
|------|------|------------|---------------------|
| 1024 | 2.76 M | 2.97 M (+8%) | 2.39 -> 1.24x |
| 4096 | 6.05 M | 6.40 M (+6%) | 5.36 -> 1.19x |
| 8192 | 7.83 M | 8.09 M (+3%) | 6.29 -> 1.29x |
Best number at <=8192 envs: 8.09 M train SPS @8192 (PRO 6000).

Remaining ranked candidates: (1) k_mingru_window_bwd is ~700MB of
traffic in ~708us per call (~1TB/s, memory-bound) x24 calls @8192 --
an L2 env-chunking rework of the 3-stream pipelined backward is the
next big lever; (2) k_step_run (78-110us, 4% occupancy, divergent
gameplay latency) is the small-N floor -- only warp-cooperative
gameplay or more envs move it.

## Recommended next step

Rollout is now the bottleneck at every N (50.8% @8192, 79% @1024 on the 3090). The floor is env-side per-step latency: the worldgen reset warp (~200us serial latency whenever any env resets) and the divergent step kernel. Two candidate attacks, both need care with anchors:
1. Overlap the reset-list worldgen for envs that died at step t with the policy forward of step t+1 for the surviving envs (the dead envs' obs are not needed until t+1's record). Keeps bit-exactness if the graph ordering is preserved per env.
2. Split the reset warp's 9-floor worldgen across multiple warps per env (block-per-env with intra-block barriers) -- bit-exact if the per-floor RNG streams are already independent; verify against the 64x2000 / 4x20000 anchors.
