---
layout: post
title: "Every Fix Moves the Bottleneck: Climbing the CUDA GEMM Ladder"
date: 2026-11-26
description: "A first-principles walkthrough of climbing an 8-stage CUDA matrix-multiplication ladder: exact FLOPs/byte math at every rung, why half the biggest speedups climb vertically under the HBM Roofline, and why custom kernels win through epilogue fusion."
categories: [Inference Engineering]
tags: [llm-inference, gpu-kernels, cuda, roofline, ncu]
math: true
---

> **Kernel Engineering Series — Part 3:** Builds on [Part 1: The Great AI Compromise](/ml/2026/08/26/the_great_ai_compromise.html) and [Part 2: Why Faster GPUs Leave Lazy Kernels Starving](/posts/why-faster-gpus-leave-lazy-kernels-starving/).
>
> **Prefer exploring interactively instead of reading a long post?** Skip straight to the [Interactive CUDA GEMM Ladder App](/gemm-ladder-cuda/) to step through all 9 stages, tweak the 7 autotuner knobs, and inspect the code side-by-side.
{: .prompt-info }

Most CUDA matrix-multiplication ("GEMM ladder") walkthroughs read like a recipe book: *here is Kernel 1, now paste Kernel 2, now paste Kernel 3*. That format hides the biggest surprise I hit while building the ladder from scratch in [`my_matmul_kernels.py`](https://github.com/karthikBalasubramanian/Kernel-Engg/blob/assignment-2/lecture8/my_matmul_kernels.py):

**In Part 2, the Roofline model made it look like moving *right* (higher HBM arithmetic intensity) is the whole game. It isn't.** Half of the biggest leaps on the GEMM ladder do not move right on the HBM Roofline at all. Instead, they climb *vertically*—clearing on-chip bottlenecks (memory queues, warp layout waste, and shared-memory bank conflicts) that trap a kernel far below the hardware roof.

To make that visible, we will track **two** numbers for every kernel in FP32 ($4\text{ bytes/float}$, $1\text{ FMA} = 2\text{ FLOPs}$):
- **$I_{\text{HBM}}$ (FLOPs per HBM byte fetched):** moves you *right* along the HBM Roofline.
- **$I_{\text{SMEM}}$ (FLOPs per on-chip SMEM byte read):** moves you *up* vertically toward the roof once HBM stops being the #1 limiter.

<div class="gemm-fig" data-scene="roofline-twist" data-step="3"></div>

*Methodology note: All Roofline ceilings come from official NVIDIA datasheets (H100 SXM5, A100 SXM4, and T4), and every X-axis intensity is exact closed-form FP32 math derived from the tile shapes in [`my_matmul_kernels.py`](https://github.com/karthikBalasubramanian/Kernel-Engg/blob/assignment-2/lecture8/my_matmul_kernels.py). The vertical gap below each roof illustrates conceptual stall headroom from CUDA architecture literature rather than measured GPU stopwatch runs; you can measure exact hardware counters on your own GPU with [`My_MatMul_Kernels_Colab.ipynb`](https://github.com/karthikBalasubramanian/Kernel-Engg/blob/assignment-2/lecture8/My_MatMul_Kernels_Colab.ipynb).*

---

## 1. Global Memory Coalescing: Bytes Requested ≠ Sectors Fetched

Start from first principles: how does a GPU actually read off-chip HBM?

Threads inside a Streaming Multiprocessor (SM) execute in lockstep groups of **32** called a **warp** (`threadIdx.x = 0..31`), and HBM does not fetch one 4-byte float at a time—it ships memory in **32-byte sectors ("crates" of 8 floats)**.

In Kernel 1 ([`my_naive_matmul`](https://github.com/karthikBalasubramanian/Kernel-Engg/blob/assignment-2/lecture8/my_matmul_kernels.py#L21-L36)), mapping `threadIdx.x` to `out_row` forces adjacent lanes in the same warp to jump down different rows of row-major `matA`. Lane 0 reads `matA[0, k]` while Lane 1 reads `matA[1, k]`—separated by $K \times 4\text{ bytes}$ ($4.8\text{ KB}$ at $K = 1200$). Every lane pulls its own 32-byte crate from HBM and throws away **7 of the 8 floats (87.5% waste)**.

Kernel 2 ([`my_mem_coalesced_matmul`](https://github.com/karthikBalasubramanian/Kernel-Engg/blob/assignment-2/lecture8/my_matmul_kernels.py#L38-L63)) fixes this with a one-line swap (`tx` → `out_col`, `ty` → `out_row`). Now all 32 lanes share the exact same `out_row` (1 hardware broadcast crate from `matA`) and walk 32 adjacent columns of `matB` (4 contiguous crates, 100% utilized; `matC` stores likewise drop from 32 sectors to 4).

```python
# Kernel 1 (Naive): tx strides across rows -> 32 crates from A + 1 from B = 33 sectors
out_row = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
out_col = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

# Kernel 2 (Coalesced): 1-line swap -> 1 broadcast from A + 4 contiguous from B = 5 sectors
out_row = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
out_col = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
```

**FLOPs / Byte moved (per 32-lane warp, per $k$-step = $32\text{ FMAs} = 64\text{ FLOPs}$):**

$$I_{\text{HBM}}^{\text{(K1 Naive)}} = \frac{64\text{ FLOPs}}{(32 + 1) \times 32\text{ B}} = \frac{64}{1{,}056} \approx \mathbf{0.061\text{ FLOPs/byte}}$$

$$I_{\text{HBM}}^{\text{(K2 Coalesced)}} = \frac{64\text{ FLOPs}}{(1 + 4) \times 32\text{ B}} = \frac{64}{160} = \mathbf{0.40\text{ FLOPs/byte}} \quad (6.6\times \text{ fewer HBM bytes})$$

Toggle between Kernel 1 and Kernel 2 below to watch Warp 0 rotate from vertical to horizontal and pack 33 crates down to 5:

<div class="gemm-fig" data-scene="coalescing" data-initial-naive="true"></div>

> **What Breaks & What Moves in `ncu`**
> - **Failure mode:** Swapping `out_row`/`out_col` inside the kernel while forgetting to swap `grid.x`/`grid.y` at launch, or using a leading dimension `N` that breaks 32-byte alignment.
> - **Profiler signal:** `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` drops ~6.6× and Sectors-per-Request on `matB`/`matC` snaps to **4.0**.
{: .prompt-warning }

---

## 2. Shared Memory & 1D Register Tiling: The MIO Queue Wall

Even at 100% sector efficiency, $0.40\text{ FLOPs/byte}$ is still **46× below** H100's FP32 ridge ($18.4\text{ FLOPs/byte}$). Every thread still walks all the way back to off-chip HBM (~400–800 cycles) on every $k$-step with zero sharing across threads.

### Kernel 3: Staging Tiles in On-Chip Shared Memory

Full matrices won't fit in per-SM Shared Memory (SMEM), but two `32 × 32` FP32 tiles take only $32 \times 32 \times 4\text{ B} \times 2 = \mathbf{8\text{ KiB}}$ per block. In [`make_shared_memory_matmul`](https://github.com/karthikBalasubramanian/Kernel-Engg/blob/assignment-2/lecture8/my_matmul_kernels.py#L66-L131), a `32 × 32` block (1,024 threads) cooperatively loads one tile of `matA` and one tile of `matB` into SMEM, calls `cuda.syncthreads()`, computes partial dot products from fast SRAM, and slides across $K$.

Why does a 1,024-thread block give **32×** HBM reuse rather than 1,024×? Because each element `sh_memA[r, k]` is only needed by the **32 threads** in row `r`, and `sh_memB[k, c]` by the **32 threads** in column `c`. Across one $K$-tile, total HBM reads drop from $1{,}024 \times 64 = 65{,}536$ down to $1{,}024 + 1{,}024 = 2{,}048$.

This gives a universal FP32 formula you can compute in your head for **any** tiled block of size $(BM \times BN)$ stepping by $BK$:

$$I_{\text{HBM}} = \frac{2 \cdot BM \cdot BN \cdot BK}{4\,(BM \cdot BK + BK \cdot BN)} = \frac{BM \cdot BN}{2\,(BM + BN)}$$

Notice something remarkable: **$BK$ cancels out completely.** Changing $BK$ does not change total HBM bytes per FLOP—it only trades off fewer `syncthreads()` barriers against a larger SMEM footprint.

**FLOPs / Byte moved (Kernel 3: $BM = BN = BK = 32$):**

$$I_{\text{HBM}}^{\text{(K3)}} = \frac{32 \times 32}{2(32 + 32)} = \mathbf{8.0\text{ FLOPs/byte}} \qquad I_{\text{SMEM}}^{\text{(K3)}} = \frac{2\text{ FLOPs}}{2 \times 4\text{ B}} = \mathbf{0.25\text{ FLOPs/byte}}$$

HBM just got 32× quieter—and immediately exposed the next wall. Sitting between your warp scheduler and SMEM is the **MIO (Memory I/O) queue**. Because every single FMA (`dot_product += sh_memA[ty, i] * sh_memB[i, tx]`) still issues **2 SMEM loads**, the MIO queue fills faster than it can drain, leaving the FP32 math units waiting on SRAM.

<div class="gemm-fig" data-scene="smem"></div>

> **What Breaks & What Moves in `ncu`**
> - **Failure mode:** Omitting the *second* `cuda.syncthreads()` after the compute loop lets fast warps advance to tile `t + 1` and overwrite `sh_memA`/`sh_memB` while slower warps are still reading tile `t`—causing silent numeric corruption.
> - **Profiler signal:** Global HBM traffic plunges ~32×, while `smsp__warp_issue_stalled_mio_throttle` becomes the #1 warp stall reason.
{: .prompt-warning }

### Kernel 4 (Roofline Twist #1): 1D Thread Tiling Climbs Vertically

Registers live directly inside each SM (65,536 32-bit registers/SM on H100)—faster than SMEM and completely off the MIO queue. In [`make_one_d_tiled_matmul`](https://github.com/karthikBalasubramanian/Kernel-Engg/blob/assignment-2/lecture8/my_matmul_kernels.py#L134-L217), we shrink the block from 1,024 threads to **128 threads (4 warps)** and give each thread a vertical strip of `TM = 8` output cells in register array `thread_results[0..7]`. At each `dot_k`, each thread reads `b_val` from `shmem_B` **once** and reuses it across 8 FMAs:

```python
# Kernel 4 (1D Tiling, BM=32, BN=32, BK=32, TM=8)
for dot_k in range(BK):
    b_val = shmem_B[dot_k, thread_col]            # 1 SMEM read into register...
    for tm in range(TM):                          # ...reused across 8 FMAs!
        thread_results[tm] += shmem_A[thread_row_base + tm, dot_k] * b_val
```

**FLOPs / Byte moved (Kernel 4: $BM = BN = BK = 32,\; TM = 8$):**

$$I_{\text{HBM}}^{\text{(K4)}} = \frac{32 \times 32}{2(32 + 32)} = \mathbf{8.0\text{ FLOPs/byte}}\;\text{(unchanged!)}$$

$$I_{\text{SMEM}}^{\text{(K4)}} = \frac{8 \times 2\text{ FLOPs}}{(8 + 1) \times 4\text{ B}} = \frac{16}{36} \approx \mathbf{0.44\text{ FLOPs/byte}} \quad \left(\frac{TM+1}{TM} = 1.125\text{ SMEM rd/FMA}\right)$$

HBM intensity did not move right at all, yet on-chip SMEM intensity jumped **1.78×**—lifting the kernel vertically toward the roof. The formula also reveals the hard ceiling of 1D tiling: because `shmem_A` still costs 1 SMEM load per FMA, $1 + \frac{1}{TM}$ can **never drop below `1.0` SMEM read per FMA** ($0.50\text{ FLOPs/SMEM byte}$).

<div class="gemm-fig" data-scene="1d"></div>

---

## 3. 2D Outer Products, Warp Tiling & `LDS.128`: Area Beats Perimeter at Every Level

### Kernel 5: 2D Thread Tiling Smashes the 1.0 SMEM Floor

To break past `1.0` SMEM read/FMA, each thread must reuse **both** operands in registers. In [`make_two_d_tiled_matmul`](https://github.com/karthikBalasubramanian/Kernel-Engg/blob/assignment-2/lecture8/my_matmul_kernels.py#L220-L314), every thread owns a 2D `TM × TN` (`4 × 4` = 16 accumulators) patch of `matC`. At each `dot_k`, it loads a tiny perimeter (`reg_M[4]` + `reg_N[4]` = 8 floats) and computes all 16 FMAs as a **register outer product**. At the same time, enlarging the block to `BM = 64, BN = 64, BK = 16` keeps SMEM at the exact same **8 KiB** while doubling HBM reuse.

**FLOPs / Byte moved (Kernel 5: $BM = BN = 64,\; BK = 16,\; TM = TN = 4$):**

$$I_{\text{HBM}}^{\text{(K5)}} = \frac{64 \times 64}{2(64 + 64)} = \mathbf{16.0\text{ FLOPs/byte}}$$

$$I_{\text{SMEM,thread}}^{\text{(K5)}} = \frac{16 \times 2\text{ FLOPs}}{(4 + 4) \times 4\text{ B}} = \mathbf{1.00\text{ FLOPs/byte}} \quad \left(\frac{TM+TN}{TM\cdot TN} = 0.50\text{ SMEM rd/FMA}\right)$$

At the single-thread level, everything looks solved. **Now click "Reveal Hidden Warp 0 Flaw" below.**

<div class="gemm-fig" data-scene="2d"></div>

### Kernel 6 (Roofline Twist #2): Warp Perimeter & Hardware Broadcasts

GPUs do not schedule threads individually—they schedule **32-lane warps**. In Kernel 5, `patches_per_row = 64 / 4 = 16`, so Warp 0 (`tid 0..31`) wraps across two full rows of 16 patches: a skinny **`8 × 64` ribbon**. Every `dot_k`, that warp fetches $8 + 64 = 72$ unique floats ($288\text{ B}$) from SMEM to run $32 \times 16 = 512\text{ FMAs}$ ($1{,}024\text{ FLOPs}$)—a warp-level SMEM intensity of only $\frac{1{,}024}{288} = \mathbf{3.56\text{ FLOPs/byte}}$.

In [`make_warp_tiled_matmul`](https://github.com/karthikBalasubramanian/Kernel-Engg/blob/assignment-2/lecture8/my_matmul_kernels.py#L319-L472), we add explicit 3-level nesting (`Block 64×64 → 8 Warps 16×32 → 4×8 Lanes of 4×4`):

```python
# Kernel 6: 3-level nested coordinates (Block -> Warp -> Thread patch)
warp_id, lane_id = tid // WARP_SIZE, tid % WARP_SIZE
warp_row = (warp_id // warps_per_BN) * WM
warp_col = (warp_id % warps_per_BN) * WN
patch_row = (lane_id // patches_per_warp_tile) * TM
patch_col = (lane_id % patches_per_warp_tile) * TN
```

Reshaping Warp 0 into a compact `16 × 32` rectangle shrinks its perimeter from **$72 \to 48$ unique floats (-33%)** for the exact same 512 FMAs, turning shared reads into free **8-way horizontal broadcasts on `tileA`** and **4-way vertical broadcasts on `tileB`**.

**FLOPs / Byte moved (Kernel 6: $BM = BN = 64,\; WM = 16,\; WN = 32$):**

$$I_{\text{HBM}}^{\text{(K6)}} = \mathbf{16.0\text{ FLOPs/byte}}\;\text{(unchanged!)} \qquad I_{\text{SMEM,warp}}^{\text{(K6)}} = \frac{1{,}024\text{ FLOPs}}{(16 + 32) \times 4\text{ B}} = \frac{1{,}024}{192} \approx \mathbf{5.33\text{ FLOPs/byte}}\;(+50\%)$$

<div class="gemm-fig" data-scene="warp"></div>

### Kernel 7: Transposing `sh_memA` for 32-Bank Clearance & `LDS.128`

Why did Kernel 5 still hit shared-memory stalls? Because SMEM isn't one monolithic bucket—it is striped across **32 parallel 4-byte memory banks** (like 32 checkout lanes). If 32 lanes in a warp read different banks (or broadcast the exact same address), hardware serves them all in 1 cycle; if two lanes request *different* addresses in the *same* bank, hardware serializes them into a **bank conflict**.

Storing `sh_memA` as `(BM, BK) = (64, 16)` meant reading `reg_M[0..3]` down a column stepped by stride 16—landing on `Bank 0, Bank 16, Bank 0, Bank 16` (bank conflicts!) and forcing 4 separate scalar `LDS.32` instructions. Transposing `sh_memA` to `(BK, BM) = (16, 64)` during the HBM→SMEM load ([Moves 1–3](https://github.com/karthikBalasubramanian/Kernel-Engg/blob/assignment-2/lecture8/my_matmul_kernels.py#L402-L458)) places both `reg_M[0..3]` and `reg_N[0..3]` across 4 contiguous adjacent banks:

- **FLOPs / Byte moved:** Still $I_{\text{HBM}} = \mathbf{16.0\text{ FLOPs/byte}}$ and $I_{\text{SMEM,warp}} = \mathbf{5.33\text{ FLOPs/byte}}$.
- **Instruction efficiency:** Bank conflicts drop to **0-way**, and 128-bit vector loads (`LOAD_SIZE = 4` / `LDS.128`) cut MIO load instructions **4×** (`0.50 → 0.125 LDS.128 / FMA`, i.e., **16 FLOPs per load instruction**).

Flip between **Before (`LDS.32`)** and **After (`LDS.128`)** below to watch the 32 SMEM banks clear out:

<div class="gemm-fig" data-scene="vectorized"></div>

> **What Breaks & What Moves in `ncu`**
> - **Failure mode:** Vectorized 4-float loads need explicit tail guards (`if global_a_col + 3 < K`, else scalar fallback) for non-multiple-of-4 shapes. Also, in Numba-CUDA, whether 4 consecutive scalar reads emit one `LDS.128`/`LDG.128` instruction depends on alignment and compiler codegen—always check the SASS in `ncu` before claiming vectorization.
> - **Profiler signal:** `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld` drops to **0**, and LSU/MIO load instruction count drops sharply.
{: .prompt-warning }

---

## 4. Autotuning 7 Hyperparameters: Crossing the Ridge Without Falling Off the Occupancy Cliff

Our `64 × 64 × 16` kernel is conflict-free, yet $16.0\text{ FLOPs/byte}$ still sits just under H100's `18.4` FP32 ridge. Scaling to `BM = BN = 128, WM = 32, WN = 64, TM = TN = 8` doubles every intensity metric at once:

**FLOPs / Byte moved (Kernel 8 Autotuned: $128 \times 128 \times 16,\; TM = TN = 8$):**

$$I_{\text{HBM}}^{\text{(K8)}} = \frac{128 \times 128}{2(128 + 128)} = \mathbf{32.0\text{ FLOPs/byte}}\quad (\text{crosses H100's } 18.4 \text{ FP32 ridge!})$$

$$I_{\text{SMEM,thread}}^{\text{(K8)}} = \frac{64 \times 2\text{ FLOPs}}{(8 + 8) \times 4\text{ B}} = \mathbf{2.00\text{ FLOPs/byte}} \qquad I_{\text{SMEM,warp}}^{\text{(K8)}} = \frac{4{,}096\text{ FLOPs}}{(32 + 64) \times 4\text{ B}} \approx \mathbf{10.67\text{ FLOPs/byte}}$$

So why not crank every tile to maximum? Because every SM has fixed ceilings—**65,536 registers, fixed SMEM, and max threads/SM**—and every knob fights another:

- Larger `TM, TN` halves SMEM traffic, but pushes registers/thread from ~48 to ~104—shrinking **active blocks/SM (occupancy)** and risking catastrophic **Local Memory spilling** (out to L2/HBM) past 255 regs/thread.
- Larger `BM, BN` raises HBM intensity, but reduces total grid blocks—triggering **wave quantization** on smaller shapes (for example, launching 140 blocks on 132 H100 SMs runs 1 full wave plus a tail wave where 124 SMs sit idle).

Because Numba freezes shared and local array shapes at compile time, [`make_warp_tiled_matmul`](https://github.com/karthikBalasubramanian/Kernel-Engg/blob/assignment-2/lecture8/my_matmul_kernels.py#L319-L472) captures all 7 knobs `(BM, BN, BK, WM, WN, TM, TN)` in a closure, and [`is_valid_config`](https://github.com/karthikBalasubramanian/Kernel-Engg/blob/assignment-2/lecture8/my_matmul_kernels.py#L489-L506) prunes 972 combinations down to **216 valid hardware kernels**.

Tweak all 7 knobs below (and switch between **H100**, **A100**, and **T4**) to watch the constraint validator and SM residency caps respond live:

<div class="gemm-fig" data-scene="autotune"></div>

---

## The Full GEMM Ladder at a Glance

All numbers below are exact analytical FP32 values ($4\text{ B/float}$, $32\text{ B}$ HBM sectors) for [`my_matmul_kernels.py`](https://github.com/karthikBalasubramanian/Kernel-Engg/blob/assignment-2/lecture8/my_matmul_kernels.py):

| Rung | Kernel / Mechanism | Tile Hierarchy | HBM FLOP/B | On-Chip SMEM FLOP/B | Bottleneck Crushed | Next Wall Exposed |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **01** | [`my_naive_matmul`](https://github.com/karthikBalasubramanian/Kernel-Engg/blob/assignment-2/lecture8/my_matmul_kernels.py#L21-L36) | `32×32` blk (1,024 thr) | `0.06` | — *(33 HBM sec/warp)* | Correctness baseline | Uncoalesced HBM (87.5% waste on A) |
| **02** | [`my_mem_coalesced_matmul`](https://github.com/karthikBalasubramanian/Kernel-Engg/blob/assignment-2/lecture8/my_matmul_kernels.py#L38-L63) | `32×32` blk (1,024 thr) | `0.40` | — *(5 HBM sec/warp)* | Sector waste (`33 → 5`) | Zero inter-thread data reuse |
| **03** | [`make_shared_memory_matmul`](https://github.com/karthikBalasubramanian/Kernel-Engg/blob/assignment-2/lecture8/my_matmul_kernels.py#L66-L131) | `32×32×32` (8 KiB SMEM) | `8.0` | `0.25` *(2.00 rd/FMA)* | 32× fewer HBM reads | SM MIO queue saturation |
| **04** | [`make_one_d_tiled_matmul`](https://github.com/karthikBalasubramanian/Kernel-Engg/blob/assignment-2/lecture8/my_matmul_kernels.py#L134-L217) | `32×32×32 · TM=8` (128 thr) | `8.0` | `0.44` *(1.125 rd/FMA)* | Half of MIO traffic (`b_val` in reg) | Hard 1D floor (`1 + 1/TM > 1.0`) |
| **05** | [`make_two_d_tiled_matmul`](https://github.com/karthikBalasubramanian/Kernel-Engg/blob/assignment-2/lecture8/my_matmul_kernels.py#L220-L314) | `64×64×16 · 4×4` (256 thr) | `16.0` | `1.00` thr / `3.56` warp | 1D floor + 2× HBM AI (8 KiB) | Skinny `8×64` warp ribbon + bank conflicts |
| **06** | [`make_warp_tiled_matmul`](https://github.com/karthikBalasubramanian/Kernel-Engg/blob/assignment-2/lecture8/my_matmul_kernels.py#L374-L389) *(Warp)* | `64×64 → 16×32 → 4×4` | `16.0` | `1.00` thr / `5.33` warp | Warp perimeter (`72 → 48`) + broadcasts | Scalar `LDS.32` + stride-`BK` layout |
| **07** | [`make_warp_tiled_matmul`](https://github.com/karthikBalasubramanian/Kernel-Engg/blob/assignment-2/lecture8/my_matmul_kernels.py#L402-L458) *(Vec)* | `sh_memA (BK×BM) · float4` | `16.0` | `5.33` warp · `0.125 LDS.128` | Stride-16 bank conflicts + 4× load inst | Fixed tile sits under H100 `18.4` ridge |
| **08** | [`run_grid_search`](https://github.com/karthikBalasubramanian/Kernel-Engg/blob/assignment-2/lecture8/my_matmul_kernels.py#L478-L607) *(7 Knobs)* | 216 valid `(BM..TN)` | `8–32` | up to `2.00` thr / `10.67` warp | Crosses FP32 ridge (`128×128`) | Regs vs. occupancy & wave tails |
| **09** | **Epilogue Fusion** *(LLM Serving)* | `GEMM + Bias + GELU` | `32.0` | `1×` output HBM store | 4 unfused `M×N` HBM round-trips | Tensor Cores (`wgmma`), TMA & Triton |
{: .table .table-bordered .table-striped .kb-ladder-table }

---

## 5. Why It Still Isn't cuBLAS — And Where Custom Kernels Actually Win

Even after Rung 8, our hand-written FP32 CUDA-core kernel will not beat `cuBLAS` across general shapes. Production libraries stack three extra hardware weapons on top of everything above:
1. **Tensor Cores (`mma` / Hopper `wgmma`):** H100's BF16 Tensor Cores deliver **989.4 TFLOP/s** vs. **61.8 TFLOP/s** on FP32 CUDA cores—a **16× compute ceiling gap** from [Part 2](/posts/why-faster-gpus-leave-lazy-kernels-starving/).
2. **Asynchronous multi-stage pipelining (`cp.async` / Hopper TMA) + XOR-swizzled SMEM:** Overlapping tile `t + 1`'s HBM load directly behind tile `t`'s math.
3. **Shape-specific heuristics:** Split-K / Stream-K dispatch for tall-skinny matrices in single-token LLM decode.

So where do custom kernels actually win? **When a GEMM doesn't live alone.**

In a Transformer MLP or projection layer, a matmul is almost always followed immediately by `+ Bias` and an activation (`GELU`/`SiLU`). Run those as three separate kernels (`cuBLAS GEMM → HBM → Bias → HBM → GELU → HBM`), and you write out and re-read the entire $M \times N$ output tensor twice—moving **$5 \times 4MN = 20MN$ bytes** of output HBM traffic.

Look at [Phase 3 of Kernel 6](https://github.com/karthikBalasubramanian/Kernel-Engg/blob/assignment-2/lecture8/my_matmul_kernels.py#L462-L471): `dot_products[i, j]` is **already sitting hot in SM registers**. Applying `+ bias[out_col]` and `gelu(val)` right before the single store to `matC` drops output HBM traffic **5×** (down to $4MN$ bytes) with zero extra kernel launches:

<div class="gemm-fig" data-scene="fusion"></div>

That single physical rule—**never round-trip an intermediate tensor to HBM when it is already on-chip**—is the bridge to what comes next in this series: **Triton & CuTe DSL** (automating all the tiling, swizzling, and pipelining boilerplate we just wrote by hand) and **FlashAttention** (fusing Softmax between two GEMMs so the $N \times N$ attention score matrix never touches HBM at all).

---

## Acknowledgments & Links

- **Interactive companion app:** Prefer hands-on exploration over prose? Launch the [Interactive CUDA GEMM Ladder App](/gemm-ladder-cuda/).
- **Source code & Colab `ncu` runner:** [`my_matmul_kernels.py`](https://github.com/karthikBalasubramanian/Kernel-Engg/blob/assignment-2/lecture8/my_matmul_kernels.py) and [`My_MatMul_Kernels_Colab.ipynb`](https://github.com/karthikBalasubramanian/Kernel-Engg/blob/assignment-2/lecture8/My_MatMul_Kernels_Colab.ipynb).
- **Interactive animations:** Built with assistance from **Google Neon**.
- **Learning acknowledgment:** Built while working through the [Vizuara AI Kernel Engineering Workshop](https://kernelworkshop.vizuara.ai/) by [Vizuara.ai](https://vizuara.ai/).
- **References:** Simon Boehm, [*How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance*](https://siboehm.com/articles/22/CUDA-MMM); Samuel Williams et al., *Roofline: An Insightful Visual Performance Model for Multicore Architectures* (*CACM*, 2009); and NVIDIA's [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) and [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/).

<script defer src="/assets/js/gemm-ladder-figs.js"></script>
