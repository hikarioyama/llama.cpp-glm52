# GLM-5.2 fast local inference — llama.cpp fork

Experimental changes to run **GLM-5.2** (754B MoE, `UD-Q2_K_XL`, ~236 GB) fast for
**single-stream decode** when the routed experts (224 GB) exceed VRAM (192 GB) and ~19+ MoE
layers must be computed on the CPU every token. Quantization is fixed (UD-Q2); we optimize
*around* it.

**Full write-up, measurements, and the open hack-hunt:**
https://github.com/hikarioyama/glm52-speedup

Branch: **`glm52-cpu-gpu-moe-split`**. Hardware target: 2× RTX PRO 6000 (192 GB, SM120) +
Threadripper 9965WX (24-core Zen5). **Ideas / PRs to go faster are very welcome.**

## What's changed vs upstream

- **`src/llama-graph.cpp` `build_moe_ffn`** — env-gated **CPU∥GPU expert split**: for host-offloaded
  MoE layers, split the top-8 experts by position (k on CPU, 8−k copied to GPU and computed
  concurrently). Disjoint id-slice = no double compute. Optional **dual-GPU** variant splits the
  GPU side across both PCIe links (host-staged inputs to keep the two GPUs independent).
- **`ggml/src/ggml-backend.cpp`** — per-site cross-backend **sync instrumentation**
  (`GGML_SCHED_SYNC_COUNT`, with `>=3` for per-token wall-clock buckets), and **events at
  `n_copies=1`** so the WAR-guard takes a non-blocking stream wait instead of a host-blocking
  `synchronize` (no buffer-explosion OOM, unlike forcing pipeline parallelism).
- **`common/arg.cpp`** — `-ot ...=CUDA_Host` (pinned offload) and `-ot ...=CPU_REPACK` exposure.
- Runtime-tunable split via **`LLAMA_MOE_CTL`** (a control file holding `"k dual"`, re-read on
  mtime change) so one persistent server can sweep configs without reloading the 237 GB model.

## Env flags / how to run

```
LLAMA_MOE_CTL=/tmp/moe_ctl        # file with "k dual" (e.g. "6 0"); runtime-tunable
LLAMA_MOE_CPU_SPLIT=k             # static alternative: k experts on CPU, 8-k on GPU (decode only)
LLAMA_MOE_DUAL_GPU=1              # split the GPU-side experts across both GPUs
GGML_SCHED_SYNC_COUNT=1|2|3       # sync-site counters (3 = per-token wall-clock buckets)
```
Build with AVX-512 on Zen5 (conda's `-march=nocona` otherwise silently disables it):
append `-march=native -mtune=native` to `CMAKE_C_FLAGS`/`CMAKE_CXX_FLAGS`.

## Status (honest)

Best verified: **+14% (24 → 27 t/s)**, output-correct. The decode bottleneck is the **CPU
`Q2_K`/`Q3_K` expert dequant (~74%, compute-bound)**; the stock kernels are AVX2-only and a
pure AVX-512 widening only buys ~1.25× because Zen5 double-pumps the 512-bit integer datapath.
See the write-up repo for the full diagnosis, the ranked roadmap, and where help is wanted.

## License

GLM-5.2-specific changes here are dedicated to the **public domain** (do whatever you want).
Upstream llama.cpp code remains under its original **MIT** license (see `LICENSE`).
