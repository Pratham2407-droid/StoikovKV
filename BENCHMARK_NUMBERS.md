# StoikovKV Benchmark Numbers Reference

This document consolidates the exact measured numbers across all StoikovKV benchmark suites (`benchmarks/bench_calibration_codec.py`, `benchmarks/bench_cache_ablation.py`, `benchmarks/comparative_audit/`, and `experiments/7b_gpu_validation/`).

> **Scope & Accounting Note:**
> - **Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM), CUDA 13.0, PyTorch `2.14.1+cu130` (`torch.float16`).
> - **Model Track:** Standalone GQA `Qwen/Qwen2.5-1.5B-Instruct` architecture (`28` layers, `12` Q heads, `2` KV heads, `q_per_kv = 6`, `head_dim = 128`) unless explicitly marked as Pretrained 7B (`Qwen/Qwen2.5-7B-Instruct`).
> - **Memory Metrics:** `KV Mem %` reports logical tier payload bytes (`compressed_bytes / fp16_bytes`); `+Dict %` (`memory_ratio_incl_dictionary`) adds the resident dictionary overhead (`D` + `W_ridge`) across active Lexico layers.

---

## 1. Offline Calibration & Write-Time Subspace Probe (`bench_calibration_codec.py`)

### 1.1 Write-Time Probe Correlation vs. Ground-Truth Attention
Evaluated on **336 held-out `(layer, prompt)` pairs** (`6` held-out prompts $\times$ `28` layers $\times$ `2` KV heads, strictly disjoint from the `16` calibration training prompts):

| Probe Variant | Spearman $\rho$ (Mean $\pm$ SD) | Recall@20% (Mean $\pm$ SD) | Sink Tokens | Sink Bonus |
|---|---:|---:|---:|---:|
| **Raw Subspace Probe** ($\|K_t U_Q\|_2 \cdot \|V_t\|_2$, $t \ge 4$) | **0.5705 $\pm$ 0.2117** | **59.20% $\pm$ 20.68%** | 0 | 0.0 |
| **Sink-Augmented Probe** ($+100.0$ on indices `0..3`) | 0.4863 $\pm$ 0.2704 | 53.10% $\pm$ 17.76% | 4 | +100.0 |
| **Value-Norm Baseline** ($\|V_t\|_2$ only) | 0.5615 | 58.67% | 0 | 0.0 |

### 1.2 Head Role Classification (`28` Layers $\times$ `2` KV Heads = `56` Total KV Heads)
- **Entropy Threshold (90th percentile):** `3.1672` nats (median entropy: `2.4473` nats)
- **Held-Out Needle Selectivity Score:** `1.0000` (`4` calibration needle prompts, `2` held-out needle prompts)

| Role ID | Role Name | Head Count | Share of Heads | Storage Policy |
|---:|---|---:|---:|---|
| `0` | `ROLE_RETRIEVAL_FP16` | **3** | **5.4%** | 100% uncompressed FP16 (`256 B/vec`) |
| `1` | `ROLE_PEAKED_H2O` | **47** | **83.9%** | Decayed heavy-hitter eviction (`k = int(seq_len * keep_ratio)`) |
| `2` | `ROLE_DIFFUSE_LEXICO` | **6** | **10.7%** | 1-step ElasticNet sparse dictionary coding (`3s + 2 B/vec`) |

### 1.3 Universal Dictionary & FP64 Ridge Preconditioner
- **Dictionary Shape per Layer:** $D \in \mathbb{R}^{128 \times 128}$, $W_{\text{ridge}} \in \mathbb{R}^{128 \times 128}$ ($\lambda_2 = 0.1$)
- **Mean Dictionary Reconstruction MSE:** `0.000220`
- **Max FP64 Linear Solve Relative Residual** ($\|(D^\top D + \lambda_2 I) W_{\text{ridge}} - D^\top\|_F / \|D^\top\|_F$): **`6.78e-16`**

---

## 2. Sparse Dictionary Codec: Fast 1-Step ElasticNet vs. OMP vs. FISTA

Evaluated on `1,024` held-out KV vectors (`head_dim = 128`, `torch.float16` on CUDA, `15` timed repetitions after `5` warmups):

| Sparsity $s$ | Bytes / Vec (`3s+2`) | Ratio vs. FP16 (`256 B`) | Fast 1-Step Cosine | OMP Cosine | FISTA Cosine | Fast Encode (ms) | Fast Decode (ms) | OMP Encode (ms) | FISTA Encode (ms) | Speedup vs. OMP | Speedup vs. FISTA |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **4** | 14 B | 18.29$\times$ | 0.6128 | 0.6142 | 0.6140 | 1.240 | 0.175 | 3.429 | 13.069 | **2.77$\times$** | **10.54$\times$** |
| **8** | 26 B | 9.85$\times$ | 0.7141 | 0.7166 | 0.7164 | 0.965 | 0.182 | 9.706 | 12.892 | **10.06$\times$** | **13.36$\times$** |
| **12** | 38 B | 6.74$\times$ | 0.7719 | 0.7758 | 0.7755 | 0.925 | 0.200 | 7.163 | 12.691 | **7.75$\times$** | **13.72$\times$** |
| **16** | 50 B | 5.12$\times$ | 0.8109 | 0.8161 | 0.8159 | 0.924 | 0.209 | 9.680 | 12.830 | **10.48$\times$** | **13.89$\times$** |
| **32** | 98 B | 2.61$\times$ | 0.8949 | 0.9071 | 0.9068 | 0.967 | 1.185 | 26.173 | 12.986 | **27.07$\times$** | **13.43$\times$** |

- **CPU Portability Check (`s = 16`, `256` vectors):** Encode median `2.874 ms` (IQR `0.108 ms`), Decode median `1.528 ms` (IQR `0.059 ms`).

---

## 3. 4-Way Equal-Budget (~20% KV Bytes) Cache Ablation (`bench_cache_ablation.py`)

Evaluated on a **2,037-token multi-turn prompt** (Turn 1: Python backend service logs containing early needle `"API_PORT = 8443"` across depths `{5%, 12%, 22%, 35%}`; Turn 2: abrupt topic shift to CSS glassmorphism stylesheet rules at token `1,170`). Compression parameters `(keep_ratio, s_budget)` are solved automatically by `solve_equal_budget_params` so every compressed configuration uses $\le 20\%$ of FP16 KV payload bytes:

| Configuration | Policy Mode | `keep_ratio` / `s` | KV Mem % | +Dict % | KV Payload Bytes | Dict Bytes | Needle `8443` Acc | Needle Logit Margin | Turn-2 CSS Logit Cosine | Turn-2 H2O Freshness | HMM Shifts | Composite Quality | Eval Time (s) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Full FP16** | `full_fp16` | `1.000` / `16` | 100.00% | 100.00% | 58,404,864 | 0 | **100.0%** (4/4) | +2457.81 | **1.0000** | 42.56% | 0 | 100.00% | 2.47 s |
| **Standalone H2O** | `standalone_h2o` | `0.168` / `16` | 19.93% | 19.93% | 11,640,832 | 0 | **100.0%** (4/4) | +2392.69 | **0.9807** | 40.85% | 0 | 99.03% | 3.16 s |
| **Standalone Lexico** | `standalone_lexico` | `1.000` / `14` | 19.79% | 22.93% | 11,557,952 | 1,835,008 | **50.0%** (2/4) | +1373.40 | 0.8354 | 0.00% | 0 | 66.77% | 10.65 s |
| **StoikovKV Hybrid** | `hybrid` | `0.1211` / `10` | **19.76%** | **20.44%** | **11,542,912** | **393,216** | **100.0%** (4/4) | **+2596.56** | **0.9071** | **100.00%** | **8** | **95.35%** | 3.98 s |

### Per-Tier Token-Head Allocation at `2,037` Tokens (`114,072` Total Token-Heads)
| Configuration | Local Window FP16 | Retrieval FP16 | Peaked H2O FP16 | Diffuse Lexico Sparse | Evicted H2O |
|---|---:|---:|---:|---:|---:|
| **Full FP16** | 3,584 | 110,488 | 0 | 0 | 0 |
| **Standalone H2O** | 3,584 | 0 | 19,152 | 0 | 91,336 |
| **Standalone Lexico** | 3,584 | 0 | 0 | 110,488 | 0 |
| **StoikovKV Hybrid** | 3,584 | 5,919 | 11,562 | 11,838 | 81,169 |

---

## 4. Matryoshka Self-Speculative Decoding Sweep (`bench_cache_ablation.py`)

Evaluated on a `1,068`-token prompt generating `64` new tokens with verify sparsity $s = 16$ (`50 B/vec`), speculative draft length $\gamma_{\text{spec}} = 4$, and draft atoms $m \in \{2, 4, 8, 12\}$:

| Draft Atoms $m$ | Draft Bytes / Vec (`3m+2`) | Empirical Acceptance $\alpha$ | Expected $\mathbb{E}[\alpha]$ | Avg Accepted / Verify Pass | Avg Tokens / Cycle | KV Bytes Read / Token | Bandwidth Savings vs. Standard Decode | Bandwidth Savings vs. Full FP16 | Passkey `8443` Retrieved |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **m = 2** | 8 B | **0.9219** | 0.9394 | 3.28 / 4 | 4.28 | 7,422.9 KB | **4.90%** | **75.88%** | Yes |
| **m = 4** | 14 B | **1.0000** | 0.9757 | 4.00 / 4 | 5.00 | 7,477.5 KB | **4.20%** | **75.70%** | Yes |
| **m = 8** | 26 B | **1.0000** | 0.9885 | 4.00 / 4 | 5.00 | 7,586.6 KB | **2.80%** | **75.35%** | Yes |
| **m = 12** | 38 B | **1.0000** | 0.9922 | 4.00 / 4 | 5.00 | 7,695.8 KB | **1.40%** | **74.99%** | Yes |

*(Standard decode on Hybrid reads `7,804.9 KB/token`; standard Full FP16 decode reads `30,772.0 KB/token`.)*

---

## 5. Multi-Length Needle Retrieval & Latency Audit (`benchmarks/comparative_audit`)

### 5.1 Needle-in-a-Haystack Retrieval (`"94812"` across Context Lengths & Positions)
`5` trials $\times$ `3` positions (`early`, `middle`, `late`) across `512`, `2,048`, and `4,096` tokens:

| Context Length | Needle Position | Full FP16 (B0) | Standalone H2O (B3) | Standalone Lexico (B3) | StoikovKV Hybrid (B4) | Unprotected Ablation |
|---:|---|---:|---:|---:|---:|---:|
| **512** | Early / Middle / Late | 100% / 100% / 100% | **0%** / 100% / 100% | 100% / 100% / 100% | **100% / 100% / 100%** | **0%** |
| **2,048** | Early / Middle / Late | 100% / 100% / 100% | **0%** / 100% / 100% | 100% / 100% / 100% | **100% / 100% / 100%** | **0%** |
| **4,096** | Early / Middle / Late | 100% / 100% / 100% | **0%** / 100% / 100% | 100% / 100% / 100% | **100% / 100% / 100%** | **0%** |

### 5.2 Prefill + 32-Token Generation Throughput (`512`-Token Context, `10` Repetitions)
| Method | Total Wall Time (s) | Throughput (tok/s) | Relative Throughput vs. FP16 |
|---|---:|---:|---:|
| **Full FP16 (B0)** | 1.09 s | **29.5 tok/s** | 100.0% |
| **Standalone H2O (B3)** | 1.58 s | **20.3 tok/s** | 68.8% |
| **StoikovKV Hybrid (B4)** | 1.95 s | **16.4 tok/s** | 55.6% (2.52$\times$ faster than Lexico) |
| **Standalone Lexico (B3)** | 4.89 s | **6.5 tok/s** | 22.0% |

---

## 6. WikiText-103 & 7B Validation Summary (`experiments/7b_gpu_validation`)

### 6.1 WikiText-103 Evaluation (`50` Samples, `3,835` Tokens, Compact 12L/4KV GQA Config)
| Policy | Throughput (tok/s) | Peak GPU VRAM | Logical KV Mem % | Logical Savings % | HMM Shifts Detected |
|---|---:|---:|---:|---:|---:|
| **Full FP16** | 1,457 tok/s | 0.41 GB | 100.0% | 0.0% | 0 |
| **Standalone H2O** | 1,578 tok/s | 0.37 GB | 13.3% | 86.7% | 0 |
| **StoikovKV Hybrid** | **1,007 tok/s** | **0.38 GB** | **19.8%** | **80.2%** | **14** |
| **Standalone Lexico** | 361 tok/s | 0.43 GB | 41.1% | 58.9% | 0 |

### 6.2 Pretrained `Qwen/Qwen2.5-7B-Instruct` Offloaded Hardware Check (8 GB RTX 4060 Laptop)
- **Placement:** Layers `0..9` on GPU (`5.75 GB`), Layers `10..27` offloaded to CPU (`8.25 GB`), peak VRAM `6.94 GB`.
- **Logical KV Memory Reduction on Real 7B:** `60.1%` (Early, 28 MB FP16 KV), `63.1%` (Middle, 32 MB FP16 KV), `67.3%` (Late, 40 MB FP16 KV) at `0.77 - 0.82 tok/s` (CPU offload bound).
