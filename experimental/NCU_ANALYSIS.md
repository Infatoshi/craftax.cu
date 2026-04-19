# NCU profile of `step_only_kernel` (Opt, NE=32768, sm_120)

Captured with `ncu --set full -k step_only_kernel` after 5 warmups.

## Headline numbers

| metric | value | what it means |
|---|---:|---|
| DRAM throughput | 3.7% | nowhere near memory-bound on DRAM |
| L1/TEX hit rate | 89.4% | L1 absorbing most reloads |
| L2 hit rate | 75.6% | L2 helping too |
| Mem pipes busy | 3.95% | load/store units idle most cycles |
| **Avg active threads per warp** | **4.81 / 32** | **85% warp divergence** |
| Warp cycles per issued inst | 17.97 | every issue waits ~18 cycles |
| Achieved occupancy | 14.0% | theoretical 100% — stalls kill it |
| Branch efficiency | 85.1%, 142 divergent branches/warp | per-env game logic diverges |
| **Uncoalesced global sectors** | **83% excessive** (3.40M of 4.12M) | AoS penalty |

NCU's own estimated speedups:

- **77.2%** — fix uncoalesced global accesses
- **62.7%** — cut L1TEX scoreboard stalls (11.3 of 18.0 cycles spent here)
- **35.2%** — fuse FP32 pairs into FMAs
- **81.0%** — raise occupancy (downstream of the above)

## Root cause: AoS `EnvState`

`EnvState` is 2320 B per env. With `threadIdx.x=i` handling env `i`, a warp reads 32 values 2320 B apart for every scalar field. Each `s.player_r` load requires 32 separate sectors — the 83% excessive sector number is exactly this.

It shows up everywhere in SASS as pairs of
```
LDG.E.U8 Rx, [Ry]        ; 1-byte per thread, 32 separate lines
SHF.R.U32.HI / LOP3.LUT  ; unpack nibble for map_get
```
and the long-scoreboard stall before the next dependent op.

## Where the cycles actually go (source-level hotspots)

Top stalls cluster in:

1. **`map_get` / `map_set`** (craftax_opt.cuh:86-96). Called in every cell of `build_observation` (63 cells) and `update_mobs`. Each call: byte load + nibble mask + branch. The byte load is uncoalesced across warp.
2. **`has_mob_at`** (craftax_opt.cuh:143-151). 8 byte compares per call, called from `move_player`, `can_move_mob`, `place_block`, `spawn_mobs`.
3. **`build_observation`** cell scan (craftax_opt.cu:470-481). 63 × `map_get` = 63 uncoalesced byte loads per env per step, plus the zero memset of 1323 floats.
4. **`is_solid(blk)`** (craftax_opt.cuh:70-74). A 10-way chain of `==` that compiles to a ladder of `ISETP + LOP3` — each warp thread takes a different branch path.
5. **Inventory clamp loop** (craftax_opt.cu step_only_kernel:604-605). Unrolled to 12 PRMT/LOP3/SEL chains for 1-byte clamps.
6. **Perlin grads** inside `generate_world`. Not hot in steady state (autoreset only runs on done envs), but contributes on resets.

## Optimization candidates (ranked)

### 1. Transpose `EnvState` to SoA (highest leverage)

Today:
```cpp
struct EnvState { int16_t player_r; int16_t player_c; ... };
states[idx].player_r;   // stride 2320 B per thread
```
SoA:
```cpp
struct EnvBuffers { int16_t* player_r; int16_t* player_c; uint8_t* map_packed; ... };
buf.player_r[idx];      // stride 2 B per thread -> fully coalesced
```
Estimated 2-4× throughput for `step_only_kernel` based on NCU's 77.2% / 62.7% numbers. The map (2048 B/env) dominates state size, so SoA'ing **just the map** (layout `map_packed[row][col_byte][env]`) gives most of the win without rewriting everything.

Inline PTX isn't the tool here — this is a data layout fix.

### 2. Pack the 4-bit map differently for vector loads

Current packed map: row-major 4-bit, 32 B/row. A thread reads one byte, masks 4 bits. With SoA the row for env `idx` is at `map[row][col_byte][idx]`. Coalescing works; but each map_get still does LDG.U8 + shift + mask.

Replace with a 32-bit `prmt.b32` unpack that peels 8 nibbles at once into 8 bytes, then indexed lookup. Inline PTX:

```cpp
__device__ __forceinline__ uint32_t unpack_nibbles_to_bytes(uint32_t packed4) {
    // Expand 8 nibbles (32 bits) into 8 bytes (64 bits) — do low half here
    uint32_t lo;
    asm("prmt.b32 %0, %1, 0, 0xF4F0;" : "=r"(lo) : "r"(packed4));  // bytes 0..3 high nibbles zeroed
    return lo & 0x0F0F0F0F;
}
```
Amortizes 8 `map_get` calls into one. Useful inside `build_observation`'s cell-grid scan where we fetch a contiguous 9-cell row.

### 3. `is_solid` as bitmask lookup

`is_solid` is 10 ORs today. Replace:
```cpp
__device__ __forceinline__ bool is_solid(int8_t blk) {
    constexpr uint32_t MASK = (1u<<BLK_WATER)|(1u<<BLK_STONE)|(1u<<BLK_TREE)|
                              (1u<<BLK_COAL)|(1u<<BLK_IRON)|(1u<<BLK_DIAMOND)|
                              (1u<<BLK_TABLE)|(1u<<BLK_FURNACE)|
                              (1u<<BLK_PLANT)|(1u<<BLK_RIPE_PLANT);
    return (MASK >> blk) & 1u;
}
```
Collapses 10 branchy compares to `SHR + AND` (2 SASS ops). No inline PTX needed. Called from `place_block`, `move_player`, `can_move_mob`, arrow collision — hits per step.

### 4. Inventory clamp with video SIMD intrinsics

Current end-of-step loop:
```cpp
for (int i = 0; i < NUM_INVENTORY; i++) s.inv[i] = clamp_i(s.inv[i], 0, 9);
```
`inv` is 12 bytes = three uint32s. Use CUDA's `__vminu4`/`__vmaxu4` video SIMD:
```cpp
uint32_t* p = reinterpret_cast<uint32_t*>(&s.inv[0]);
p[0] = __vminu4(p[0], 0x09090909u);
p[1] = __vminu4(p[1], 0x09090909u);
p[2] = __vminu4(p[2], 0x09090909u);
```
One SASS `VIMNMX` per uint32 (sm_90+ uses VIMNMX, older uses VABSDIFF4/VMAX4). 12 clamps → 3 SIMD ops. Only works with SoA if you also transpose `inv` across envs; with AoS you already have 4 contiguous bytes so this works today.

PTX equivalent if the intrinsic isn't lowered as you'd like:
```cpp
asm("vmin4.u32.u32.u32 %0, %1, %2, 0;" : "=r"(out) : "r"(in), "r"(0x09090909u));
```

### 5. `has_mob_at` bit-packed

Mob positions are `int16 r, int16 c` per slot. Pack into a single `uint32` as `(r<<16)|c`. Store all 8 mob slots as a fixed `uint32 mobs[8]` (+ mask bit in high bit of r since `r < 64`). Then one call compares against all 8 with a loop the compiler can unroll into `ISETP.EQ + VOTE`.

Better still, pack mask into the high bit so the "slot empty" check folds into the compare:
```cpp
uint32_t key = ((uint32_t)r << 16) | c;   // with mask bit in bit 31
bool any = (mob_keys[0]==key) | (mob_keys[1]==key) | ...; // branchless
```

Called many times per step — cuts 8 byte-loads + branches to a handful of reg-reg compares.

### 6. Cut block size from 256 to 64

At NE=4096, grid is `4096/256 = 16` on a 188-SM GPU — 91% of SMs idle. With block=64: grid=64. Combined with register pressure at 40 reg/thread (block limit 6), smaller blocks are also better for occupancy. This is a one-line change in `craftax_opt_ext.cu`:
```cpp
int block = 64;   // was 256
```
NCU flagged this as **91.5% est. speedup** for kernels at small NE.

### 7. FMA fusion

NCU reports 6482 fused FP32 and 15360 non-fused FP32 in Perlin + intrinsics path. `perlin_2d`'s `nx0 + v*(nx1-nx0)` and Perlin interp cube polynomial are the culprits. `--use_fast_math` is already set; most of these should fuse. Adding explicit `__fmaf_rn` at the bilerp step (`fmaf(u, n10-n00, n00)`) forces fusion and reclaims ~35% of the FP32 work in reset. Small absolute win since reset is amortized.

## Ranked by expected impact on steady-state SPS

| Change | Effort | Expected | Why |
|---|---|---:|---|
| SoA map layout | medium (touches map_get/set + kernel signatures) | 1.5-2.5× on step_only | Kills the 83% uncoalesced dominant stall |
| Block size 256 → 64 | 1 line | 1.1-1.3× at NE≤8192 | Fills more SMs |
| `is_solid` bitmask | trivial | 3-6% | Removes branch ladder hot in hot path |
| `__vminu4` inv clamp | trivial | 1-2% | 12 ops → 3 |
| `has_mob_at` packed | small | 3-5% | 24 byte-loads → 8 reg compares |
| FMA in Perlin | trivial | 1-2% (reset-dominated workloads only) | forces fusion |
| Full SoA struct transpose | large | 2-3× on step_only | superset of map SoA |

## What inline PTX actually buys you vs C++ intrinsics

Most of the wins above are achievable through `__vminu4`, `__byte_perm` (which lowers to `prmt.b32`), and bitmask tricks — the compiler already emits the right SASS for these. Cases where hand-written PTX is worth it:

- **`prmt.b32` with a custom selector** for unusual nibble extraction patterns the compiler won't synthesize from `__byte_perm` indices.
- **`ld.global.L2::128B` / `ld.global.ca` prefetch hints** once layout is SoA — force 128-byte sector loads for the map.
- **`vote.ballot.sync`** for the mob-presence broadcast if you ever put one env per warp instead of one env per thread (would need a whole different kernel organization).

Bottom line: the profile says the bottleneck is **memory layout and warp occupancy, not the arithmetic**. Fix AoS→SoA and block size first. Inline PTX for `prmt` and `vmin4` gives another few percent, but it's a rounding error next to the layout fix.
