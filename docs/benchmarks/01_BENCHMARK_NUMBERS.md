# 01 — StoikovKV Benchmark Numbers Reference

> **Navigation:** [Folder Index (`docs/benchmarks/`)](./README.md) | **Next:** [02 — Benchmark Theory, Causal Mechanics & Interpretation Guide](./02_BENCHMARK_THEORY_AND_INTERPRETATION.md)

This document consolidates the exact measured numbers across all StoikovKV benchmark suites (`benchmarks/bench_calibration_codec.py`, `benchmarks/bench_cache_ablation.py`, `benchmarks/comparative_audit/`, `experiments/7b_gpu_validation/`, and `benchmarks/results/real_qwen15b_gpu_metrics.json`). For a complete explanation of **what each test means**, **how the algorithm produced each number**, and **how to interpret the results**, see [02 — Benchmark Theory, Causal Mechanics & Interpretation Guide](./02_BENCHMARK_THEORY_AND_INTERPRETATION.md).

> **Scope & Accounting Note:**
> - **Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM), CUDA 13.0, PyTorch `2.12.0+cu130` (`torch.float16`).
> - **Model Tracks:**
>   1. **Standalone GQA `Qwen/Qwen2.5-1.5B-Instruct` Architecture** (`28` layers, `12` Q heads, `2` KV heads, `q_per_kv = 6`, `head_dim = 128`) in Sections 1–5.
>   2. **WikiText-103 & Pretrained `Qwen/Qwen2.5-7B-Instruct` Track** in Section 6.
>   3. **Pretrained `Qwen/Qwen2.5-1.5B-Instruct` End-to-End GPU Track** (`Qwen2ForCausalLM` real HuggingFace weights on `cuda:0`) in Section 7.
> - **Memory Metrics:** `KV Mem %` reports logical and physical resident tier payload bytes (`compressed_bytes / fp16_bytes == physical_resident_bytes / fp16_bytes`); `+Dict %` (`memory_ratio_incl_dictionary`) adds the shared resident dictionary overhead (`D` + `W_ridge`) across active Lexico layers.

---

## 1. Offline Calibration & Write-Time Subspace Probe (`bench_calibration_codec.py`)

### 1.1 Write-Time Probe Correlation vs. Ground-Truth Attention
Evaluated on **336 held-out `(layer, prompt)` pairs** (`6` held-out prompts $\times$ `28` layers $\times$ `2` KV heads, strictly disjoint from the `16` calibration training prompts):

| Probe Variant | Spearman $\rho$ (Mean $\pm$ SD) | Recall@20% (Mean $\pm$ SD) | Sink Tokens | Sink Bonus |
|---|---:|---:|---:|---:|
| **Raw Directional Subspace Probe** ($\sigma((K_t U_Q)_0) \cdot \|K_t U_Q\|_2 \cdot \|V_t\|_2$, $t \ge 4$) | **0.5776 $\pm$ 0.2240** | **52.27% $\pm$ 17.50%** | 0 | 0.0 |
| **Sink-Augmented Probe** ($+100.0$ on indices `0..3`) | **0.4892 $\pm$ 0.2233** | **56.51% $\pm$ 18.02%** | 4 | +100.0 |
| **Value-Norm Baseline** ($\|V_t\|_2$ only) | 0.5612 | 58.70% | 0 | 0.0 |

### 1.2 Head Role Classification (`20` Layers $\times$ `2` KV Heads = `40` Total KV Heads in Benchmark Harness; `56` in 28-Layer Config)
- **Entropy Threshold (90th percentile):** `3.1755` nats (median entropy: `2.4518` nats)
- **Held-Out Needle Selectivity Score:** `1.0000` (`4` calibration needle prompts, `2` held-out needle prompts)

| Role ID | Role Name | Share of Heads | Storage Policy |
|---:|---|---:|---|
| `0` | `ROLE_RETRIEVAL_FP16` | **5.0% – 5.4%** (`2–3` heads) | 100% uncompressed FP16 (`256 B/vec`) |
| `1` | `ROLE_PEAKED_H2O` | **83.9% – 85.0%** (`34–47` heads) | Decayed heavy-hitter eviction (`k = int(seq_len * keep_ratio)`) |
| `2` | `ROLE_DIFFUSE_LEXICO` | **10.0% – 10.7%** (`4–6` heads) | 1-step ElasticNet sparse dictionary coding (`3s + 2 B/vec`) |

### 1.3 Universal Dictionary & FP64 Ridge Preconditioner
- **Dictionary Shape per Layer:** $D \in \mathbb{R}^{128 \times 128}$, $W_{\text{ridge}} \in \mathbb{R}^{128 \times 128}$ ($\lambda_2 = 0.1$)
- **Mean Dictionary Reconstruction MSE:** `0.000220`
- **Max FP64 Linear Solve Relative Residual** ($\|(D^\top D + \lambda_2 I) W_{\text{ridge}} - D^\top\|_F / \|D^\top\|_F$): **`7.20e-16`**

---

## 2. Sparse Dictionary Codec: Fast 1-Step ElasticNet vs. OMP vs. FISTA

Evaluated on `1,024` held-out KV vectors (`head_dim = 128`, `torch.float16` on CUDA, `15` timed repetitions after `5` warmups, zero GPU-to-CPU sync stalls on the hot path):

| Sparsity $s$ | Bytes / Vec (`3s+2`) | Ratio vs. FP16 (`256 B`) | Fast 1-Step Cosine | OMP Cosine | FISTA Cosine | Fast Encode (ms) | Fast Decode (ms) | OMP Encode (ms) | FISTA Encode (ms) | Speedup vs. OMP | Speedup vs. FISTA |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **4** | 14 B | 18.29$\times$ | 0.6129 | 0.6142 | 0.6140 | **0.529** | **0.108** | 1.676 | 8.917 | **3.17$\times$** | **16.87$\times$** |
| **8** | 26 B | 9.85$\times$ | 0.7133 | 0.7159 | 0.7156 | **0.547** | **0.137** | 2.702 | 8.766 | **4.94$\times$** | **16.03$\times$** |
| **12** | 38 B | 6.74$\times$ | 0.7713 | 0.7750 | 0.7748 | **0.531** | **0.139** | 3.816 | 8.365 | **7.19$\times$** | **15.76$\times$** |
| **16** | 50 B | 5.12$\times$ | 0.8103 | 0.8153 | 0.8151 | **0.527** | **0.167** | 4.957 | 8.336 | **9.40$\times$** | **15.81$\times$** |
| **32** | 98 B | 2.61$\times$ | 0.8945 | 0.9067 | 0.9064 | **0.594** | **0.243** | 18.776 | 8.979 | **31.60$\times$** | **15.11$\times$** |

- **Compressed-Cache Passkey Retrieval (`bench_calibration_codec.py`, `s = 32`, `m = 8`, `6` prompts):**
  - `full_fp16`: `256.00 B/vec` (`1.00x`), Top-1 Accuracy **`100.0%`**, Logit Cosine **`1.0000`**, KL **`0.0000`**
  - `stoikov_hybrid_s16`: `172.77 B/vec` (`1.48x`), Top-1 Accuracy **`100.0%`**, Logit Cosine **`0.9998`**, KL **`0.0000`**
  - `stoikov_hybrid_draft_m4`: `134.84 B/vec` (`1.90x`), Top-1 Accuracy **`100.0%`**, Logit Cosine **`0.9997`**, KL **`0.0000`**
  - `unprotected_ablation`: `5.00 B/vec` (`51.20x`), Top-1 Accuracy **`0.0%`**, Logit Cosine `0.0786`, KL `26.6241`

---

## 3. 4-Way Equal-Budget (~20% KV Bytes) Cache Ablation (`bench_cache_ablation.py`)

Evaluated on a **2,037-token multi-turn prompt** (Turn 1: Python backend service logs containing early needle `"API_PORT = 8443"` across depths `{5%, 12%, 22%, 35%}`; Turn 2: abrupt topic shift to CSS glassmorphism stylesheet rules at token `1,170`). Compression parameters `(keep_ratio, s_budget)` are solved automatically by `solve_equal_budget_params` so every compressed configuration uses $\le 20\%$ of FP16 KV payload bytes:

| Configuration | Policy Mode | `keep_ratio` / `s` | KV Mem % (Logical = Physical) | +Dict % | KV Payload Bytes | Dict Bytes | Needle `8443` Acc | Needle Logit Margin | Turn-2 CSS Logit Cosine | Turn-2 H2O Freshness | HMM Shifts | Composite Quality | Eval Time (s) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Full FP16** | `full_fp16` | `1.000` / `16` | 100.00% | 100.00% | 58,404,864 | 0 | **100.0%** (4/4) | +2457.81 | **1.0000** | 42.56% | 0 | 100.00% | **1.37 s** |
| **Standalone H2O** | `standalone_h2o` | `0.168` / `16` | 19.93% | 19.93% | 11,640,832 | 0 | **100.0%** (4/4) | +2346.73 | **0.9815** | 44.62% | 0 | 99.08% | **1.74 s** |
| **Standalone Lexico** | `standalone_lexico` | `1.000` / `14` | 19.79% | 22.93% | 11,557,952 | 1,835,008 | **50.0%** (2/4) | +1367.86 | 0.8569 | 0.00% | 0 | 67.84% | **5.83 s** |
| **StoikovKV Hybrid** | `hybrid` | `0.1211` / `10` | **19.76%** | **20.44%** | **11,542,912** | **393,216** | **100.0%** (4/4) | **+2540.95** | **0.9093** | **100.00%** | **8** | **95.47%** | **2.38 s** |

### Per-Tier Token-Head Allocation at `2,037` Tokens (`114,072` Total Token-Heads)
| Configuration | Local Window FP16 | Retrieval FP16 | Peaked H2O FP16 | Diffuse Lexico Sparse | Evicted H2O |
|---|---:|---:|---:|---:|---:|
| **Full FP16** | 3,584 | 110,488 | 0 | 0 | 0 |
| **Standalone H2O** | 3,584 | 0 | 19,152 | 0 | 91,336 |
| **Standalone Lexico** | 3,584 | 0 | 0 | 110,488 | 0 |
| **StoikovKV Hybrid** | 3,584 | 5,919 | 11,562 | 11,838 | 81,169 |

---

## 4. Matryoshka Self-Speculative Decoding Sweep (`bench_cache_ablation.py`)

Evaluated on a `1,068`-token prompt generating `64` new tokens with verify sparsity $s = 16$ (`50 B/vec`), speculative draft length $\gamma_{\text{spec}} = 4$, single-pass $\gamma_{\text{spec}} + 1$ verification with bonus token emission, and draft atoms $m \in \{2, 4, 8, 12\}$:

| Draft Atoms $m$ | Draft Bytes / Vec (`3m+2`) | Empirical Acceptance $\alpha$ | Expected $\mathbb{E}[\alpha]$ | Avg Accepted / Verify Pass | Avg Tokens / Cycle | KV Bytes Read / Token | Bandwidth Savings vs. Standard Decode | Bandwidth Savings vs. Full FP16 | Passkey `8443` Retrieved |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **m = 2** | 8 B | **0.9615** | 0.9445 | 3.33 / 4 | 4.33 | 7,422.9 KB | **4.90%** | **75.88%** | Yes |
| **m = 4** | 14 B | **0.9615** | 0.9527 | 3.33 / 4 | 4.33 | 7,477.5 KB | **4.20%** | **75.70%** | Yes |
| **m = 8** | 26 B | **0.9615** | 0.9584 | 3.33 / 4 | 4.33 | 7,586.6 KB | **2.80%** | **75.35%** | Yes |
| **m = 12** | 38 B | **0.9615** | 0.9601 | 3.33 / 4 | 4.33 | 7,695.8 KB | **1.40%** | **74.99%** | Yes |

*(Standard decode on Hybrid reads `7,804.9 KB/token`; standard Full FP16 decode reads `30,772.0 KB/token`.)*

---

## 5. Multi-Length Needle Retrieval & Latency Audit (`benchmarks/comparative_audit`)

### 5.1 Needle-in-a-Haystack Retrieval (`"94812"` across Context Lengths & Positions)
`15` runs (`5` trials $\times$ `3` seeds) per `(context_length, needle_position)` across `512`, `2,048`, and `4,096` tokens:

| Context Length | Needle Position | Full FP16 (B0) | Standalone H2O (B3) | StoikovKV Hybrid (B4) | Unprotected Ablation |
|---:|---|---:|---:|---:|---:|
| **512** | Early / Middle / Late | 100% / 100% / 100% | **0%** / 100% / 100% | **100% / 100% / 100%** | **0%** |
| **2,048** | Early / Middle / Late | 100% / 100% / 100% | **0%** / 100% / 100% | **100% / 100% / 100%** | **0%** |
| **4,096** | Early / Middle / Late | 100% / 100% / 100% | **0%** / 100% / 100% | **100% / 100% / 100%** | **0%** |

### 5.2 Prefill + 32-Token Generation Throughput (`512`-Token Context, `10` Repetitions $\times$ `3` Seeds)
| Method | Mean Wall Time (s) | Throughput (tok/s) | Relative Throughput vs. FP16 |
|---|---:|---:|---:|
| **Full FP16 (B0)** | **1.04 s** | **30.9 tok/s** | 100.0% |
| **Standalone H2O (B3)** | **1.39 s** | **23.1 tok/s** | 74.8% |
| **StoikovKV Hybrid (B4)** | **1.73 s** | **18.5 tok/s** | 59.9% |

---

## 6. WikiText-103 & 7B Validation Summary (`experiments/7b_gpu_validation`)

### 6.1 WikiText-103 Evaluation (`50` Samples, Compact 12L/4KV GQA Config)
After P0 physical KV cache compaction (removing duplicate `DynamicCache` buffers and cached FP16 Lexico reconstructions):

| Policy | Throughput (tok/s) | Peak GPU VRAM | Physical = Logical KV Mem % | Memory Savings % | HMM Shifts Detected |
|---|---:|---:|---:|---:|---:|
| **Full FP16** | **3,285.6 tok/s** | 0.32 GB | 100.0% | 0.0% | 0 |
| **Standalone H2O** | **3,109.3 tok/s** | **0.26 GB** | 13.5% | **86.5%** | 0 |
| **StoikovKV Hybrid** | **1,910.8 tok/s** | **0.29 GB** | **20.0%** | **80.0%** | **15** |
| **Standalone Lexico** | **687.6 tok/s** | **0.30 GB** | 41.2% | 58.8% | 0 |

### 6.2 Pretrained `Qwen/Qwen2.5-7B-Instruct` Offloaded Hardware Check (8 GB RTX 4060 Laptop)
- **Placement:** Layers `0..9` on GPU (`5.75 GB`), Layers `10..27` offloaded to CPU (`8.25 GB`), peak VRAM `6.94 GB`.
- **KV Memory Reduction on Real 7B:** `60.1%` (Early, 28 MB FP16 KV), `63.1%` (Middle, 32 MB FP16 KV), `67.3%` (Late, 40 MB FP16 KV) at `0.77 - 0.82 tok/s` (CPU offload bound).

---

## 7. Pretrained `Qwen/Qwen2.5-1.5B-Instruct` End-to-End GPU Validation (`real_qwen15b_gpu_metrics.json`)

Evaluated directly on real HuggingFace `Qwen2ForCausalLM` pretrained weights (`Qwen/Qwen2.5-1.5B-Instruct`, `28` layers, `12` Q heads, `2` KV heads, `head_dim = 128`, `torch.float16` on `cuda:0`) with streaming $O(B \cdot T)$ calibration (`output_attentions=False`) and SDPA compact mask slicing:

### 7.1 Real-Weights Write-Time Subspace Probe Correlation (`336` Held-Out `(layer, prompt)` Pairs)
- **Calibration Wall Time on RTX 4060 Laptop GPU:** **`8.92 s`** (`D` shape `(28, 128, 1024)`, mean reconstruction MSE `0.002341`, FP64 solve relative residual `1.45e-15`)
- **Held-Out Needle Selectivity Score:** **`0.9923`** (`ROLE_RETRIEVAL_FP16 = 3 (5.4%)`, `ROLE_PEAKED_H2O = 46 (82.1%)`, `ROLE_DIFFUSE_LEXICO = 7 (12.5%)`)

| Probe Variant (Real `Qwen/Qwen2.5-1.5B-Instruct`) | Spearman $\rho$ (Mean $\pm$ SD) | Recall@20% (Mean $\pm$ SD) | Gain over $\|V_t\|_2$ Baseline |
|---|---:|---:|---:|
| **Sink-Augmented Directional Subspace Probe** ($+100.0$ on `0..3`) | **0.5733 $\pm$ 0.2716** | **52.19% $\pm$ 18.09%** | **+0.6087 $\rho$** (+29.49% Recall@20%) |
| **Raw Directional Subspace Probe** ($t \ge 4$, no sink bonus) | **0.4020 $\pm$ 0.2887** | **36.73% $\pm$ 18.67%** | **+0.4374 $\rho$** (+14.03% Recall@20%) |
| **Value-Norm Baseline** ($\|V_t\|_2$ only) | -0.0354 | 22.70% | — |

### 7.2 Real-Weights Compressed-Cache Passkey Retrieval (`"94812"`, `s = 32`, `m = 8`)
| Policy (`Qwen/Qwen2.5-1.5B-Instruct`) | Effective Bytes / Vec | Compression vs. FP16 | Calibration Passkey Acc | Held-Out Passkey Acc | Overall Top-1 Acc | Passkey Prob Retention | Logit Cosine | Next-Token KL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Full FP16** (`full_fp16`) | 256.00 B | 1.00$\times$ | **100.0%** | **100.0%** | **100.0%** | 100.0% | 1.0000 | 0.0000 |
| **StoikovKV Hybrid Verify** (`stoikov_hybrid_s16`) | 171.36 B | **1.49$\times$** | **100.0%** | **100.0%** | **100.0%** | 82.80% | **0.9096** | 0.2394 |
| **StoikovKV Hybrid Draft** (`stoikov_hybrid_draft_m4`) | 132.79 B | **1.93$\times$** | **100.0%** | **100.0%** | **100.0%** | 77.19% | **0.9076** | 0.3432 |
| **Unprotected Ablation** (`unprotected_ablation`) | 5.00 B | 51.20$\times$ | **0.0%** | **0.0%** | **0.0%** | 2.68% | 0.5777 | 3.6446 |

### 7.3 Real-Weights Live `model.generate` & Matryoshka Self-Speculative Decoding (`412`-Token Vault Prompt)
- **Drop-In `model.generate(..., past_key_values=cache)` (`keep_ratio = 0.30, s_budget = 16, n_b = 32`):**
  - **Generated Text:** `'94812.'` (**exact 5-digit passkey retrieved**)
  - **Physical = Logical KV Memory:** `4,722,700 B` vs `11,956,224 B` FP16 (**`39.50%` of FP16, `60.50%` physical KV memory savings**)
- **Matryoshka Self-Speculative Decoding (`gamma_spec = 4, draft_atoms = 4, verify_atoms = 16`, single-pass $\gamma+1$ verification):**
  - **Generated Text:** `'94812. Remember this code. The secret passkey for the'`
  - **Empirical Acceptance Rate ($\alpha$):** **`1.0000` (`100.0%`)** (`3.25 / 4` accepted draft tokens per verify pass, `3,591.4 KB/tok` read)
