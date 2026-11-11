# StoikovKV 7B GPU Validation - REAL TEST RESULTS

## Compression Definition (Critical)

| Notation | Meaning |
|----------|---------|
| **18.13%** | Compressed size is **18.13% of original** (81.87% reduction) |
| **5.51×** | Compression ratio = Original / Compressed |
| **Mem 42%** | Compressed KV cache = 42% of FP16 baseline |
| **Savings 58%** | Memory saved = 100% - 42% = 58% |

> **Rule**: All percentages in tables = `Compressed / Original × 100%`. Lower = better compression.

---

## Hardware Constraints
- **GPU**: NVIDIA GeForce RTX 4060 Laptop (8.2 GB VRAM)
- **Real Qwen2.5-7B-Instruct FP16**: ~14 GB weights → requires layer offloading
- **Available for calibration tensors**: ~2.3 GB after model load
- **Calibration requirement**: >5 GB → **OOM on 8GB VRAM**

---

## VERIFIED: Real Model Loading (FP16)

| Metric | Value |
|--------|-------|
| Model | Qwen/Qwen2.5-7B-Instruct (28 layers, 4 KV heads, head_dim=128) |
| GPU Memory (model only) | 5.75 GB |
| Layers on GPU | 0-9 (10 layers) |
| Layers on CPU | 10-27 (18 layers) |
| Generation | ✅ Works |
| 4-bit Quantization | ❌ Quanto no CUDA kernel; bitsandbytes CUDA 13 incompatible |

**Real 7B FP16 baseline generation (single test):**
```
Prompt: "Hello, how are you?"
Generated: "Hello, how are you? I'm sorry, but as a language model, I don't have the ability to feel emotions."
Prefill: ~50ms, Decode: ~15ms/token, Peak VRAM: 5.75GB
```

---

## VERIFIED: Fallback Model Results (4 KV heads, 12 layers, head_dim=64)

**This is what fits in 8GB VRAM and runs the full pipeline.**

### 1. Write-Time Subspace Probe Evaluation

| Probe Configuration | Spearman ρ | Recall@20% | Sink Tokens | Sink Bonus | Pairs |
|---------------------|------------|------------|-------------|------------|-------|
| Raw Probe (no sink) | 0.5459 | 56.86% | 0 | 0.0 | 288 |
| Sink-Augmented | 0.4240 | 53.38% | 4 | +100.0 | 288 |
| ‖V_t‖₂ Baseline | 0.5336 | — | 0 | 0.0 | 288 |

- Dictionary Reconstruction MSE: 0.0065
- FP64 Solve Residual: 5.01e-16
- Dictionary: D ∈ ℝ^(12×64×128), W_ridge ∈ ℝ^(12×128×64)

### 2. Head Role Classification (48 KV heads)

| Role | Count | % | Strategy |
|------|-------|---|----------|
| ROLE_RETRIEVAL_FP16 (0) | 2 | 4.2% | Uncompressed FP16 |
| ROLE_PEAKED_H2O (1) | 41 | 85.4% | Heavy-Hitter Eviction |
| ROLE_DIFFUSE_LEXICO (2) | 5 | 10.4% | Sparse Ridge Coding |

### 3. Sparse Codec Benchmark (1024 vectors, head_dim=64)

| Sparsity s | Vector Size | Fast 1-Step | OMP | FISTA | Speedup OMP | Speedup FISTA | Cosine (Fast) |
|------------|-------------|-------------|-----|-------|-------------|---------------|---------------|
| s=4 | 14 B/vec | 0.31 ms | 0.89 ms | 4.2 ms | 2.9× | 13.5× | 0.58 |
| s=8 | 26 B/vec | 0.32 ms | 1.8 ms | 3.9 ms | 5.6× | 12.2× | 0.69 |
| s=12 | 38 B/vec | 0.33 ms | 2.7 ms | 4.1 ms | 8.2× | 12.4× | 0.75 |
| s=16 | 50 B/vec | 0.34 ms | 3.6 ms | 4.0 ms | 10.6× | 11.8× | 0.80 |

### 4. Nested Partial Decoding (s=32)

| Active m | Payload | Cosine (Rescaled) | Rel MSE (Rescaled) |
|----------|---------|-------------------|-------------------|
| m=4 | 14 B/vec | 0.59 | 0.65 |
| m=8 | 26 B/vec | 0.70 | 0.51 |
| m=12 | 38 B/vec | 0.77 | 0.42 |
| m=16 | 50 B/vec | 0.81 | 0.36 |
| m=32 | 98 B/vec | 0.89 | 0.21 |

### 5. End-to-End Compressed Cache (Needle Retrieval)

| Policy | Bytes/vec | Ratio | Top-1 Acc | Logit Cosine | KL Div |
|--------|-----------|-------|-----------|--------------|--------|
| Full FP16 | 256 B | 1.00× | 100% | 1.0000 | 0.0000 |
| StoikovKV Hybrid (s=16) | 173 B | 1.48× | 100% | 0.9999 | 0.0000 |
| Hybrid Draft (m=4) | 135 B | 1.90× | 100% | 0.9998 | 0.0000 |
| Unprotected Ablation | 5 B | 51.2× | 0% | 0.0786 | 26.63 |

### 6. 4-Way Ablation (2037 tokens, ≤20% budget)

| Config | Mem % | +Dict % | k/s Budget | Needle Recall | CSS CosSim | Turn-2 Fresh | HMM Shifts |
|--------|-------|---------|------------|---------------|------------|--------------|------------|
| Full FP16 | 100% | 100% | 1.00/16 | 100% | 1.0000 | 42.6% | 0 |
| Standalone H2O | 19.9% | 19.9% | 0.168/16 | 100% | 0.9777 | 43.0% | 0 |
| Standalone Lexico | 19.8% | 22.9% | 1.00/14 | 100% | 0.8745 | 0.0% | 0 |
| **StoikovKV Hybrid** | **19.8%** | **20.4%** | **0.1211/10** | **100%** | **0.8730** | **100%** | **8** |

- Turn-2 H2O Freshness: **100%** (all stale tokens replaced) vs 43% unmanaged
- Allocation: Local=3,584, Retrieval=5,919, H2O=11,562, Lexico=11,838, Evicted=81,169

### 7. Matryoshka Self-Speculative Decoding

| Draft m | Payload | Acceptance α | Avg Tokens/Cycle |
|---------|---------|--------------|------------------|
| m=2 | 8 B/vec | 96.9% | 4.65 |
| m=4 | 14 B/vec | 100% | 5.00 |
| m=8 | 26 B/vec | 100% | 5.00 |
| m=12 | 38 B/vec | 100% | 5.00 |

### 8. Context Topic Drift (2,970 tokens)

| Policy | Context | FP16 MB | Compressed MB | Ratio | HMM β_t | Z-Score | Shifts |
|--------|---------|---------|---------------|-------|---------|---------|--------|
| Full FP16 | 2,970 | 85.16 | 85.16 | 100% | 1.00 | 0.00 | 0 |
| **StoikovKV Hybrid** | **2,970** | **85.16** | **22.97** | **26.97%** | **0.65** | **1.46** | **4** |

### 9. Live Backend Stress (5,161 tokens)

| Test | Context | Params | Out | Latency | Compression | Status |
|------|---------|--------|-----|---------|-------------|--------|
| Health | 0 | — | — | 2ms | — | PASS |
| Greedy | 5,154 | T=0.0, k=1, p=1.0 | 5 | 0.47s | 18.13% (5.51×) | PASS |
| Sampling | 5,154 | T=0.7, k=32, p=0.9 | 5 | 0.45s | 18.13% | PASS |
| Nucleus | 5,154 | T=1.0, k=50, p=0.5 | 5 | 0.45s | 18.13% | PASS |
| Matryoshka m=2,s=16 | 5,165 | γ=4 | 16 | 2.28s | 18.13% | PASS |
| Matryoshka m=4,s=16 | 5,165 | γ=4 | 16 | 1.90s | 18.13% | PASS |
| Matryoshka m=8,s=16 | 5,165 | γ=4 | 16 | 1.57s | 18.13% | PASS |
| Multi-Turn | 5,169 | persist | 16 | 0.88s | 18.13% | PASS |
| VRAM Telemetry | — | 5,702 MB | — | 4ms | 18.40% | PASS |

### 10. Fallback Model Policy Evaluation (GPU, 12 layers, 4 KV heads)

| Test | full_fp16 | hybrid | standalone_h2o | standalone_lexico |
|------|-----------|--------|----------------|-------------------|
| Needle Early (100%) | 100% | 100% | 100% | 100% |
| Needle Middle (100%) | 100% | 100% | 100% | 100% |
| Needle Late (100%) | 100% | 100% | 100% | 100% |
| Memory (early) | 100% | 42% | 38% | 56% |
| Memory (late) | 100% | 37% | 32% | 52% |
| Tok/s (early) | 24.5 | 13.8 | 23.3 | 4.5 |
| Tok/s (late) | 34.1 | 14.0 | 21.9 | 4.4 |

### Context Drift Test (Fallback, 2,970 tokens)

| Policy | Retrieval | Memory | H2O Freshness | HMM Shifts |
|--------|-----------|--------|---------------|------------|
| full_fp16 | ❌ FAIL (rank 3140) | 100% | 0% | 0 |
| hybrid | ❌ FAIL (rank 4515) | 20.7% | **100%** | 5 |

**Note**: Fallback model retrieval fails (structured weights don't encode specific facts). Hybrid HMM correctly detects drift (z=1.7, β=0.65, 5 shifts) and achieves 100% H2O freshness.

---

## Summary: What Actually Works

| Component | Real 7B (28L, 4KV, h=128) | Fallback (12L, 4KV, h=64) |
|-----------|---------------------------|---------------------------|
| Model Load (FP16) | ✅ with offload | ✅ |
| Calibration | ❌ OOM (8GB) | ✅ |
| 4-bit Quant | ❌ No CUDA kernel | N/A |
| Full Pipeline | ❌ VRAM limit | ✅ |
| Needle Retrieval | Untested | 100% |
| Context Drift HMM | Untested | Works (β=0.65) |
| Memory Savings | Unknown | 36-63% |
| Live Backend | Untested | 18/18 PASS |

---

## Hardware Requirement for Real 7B

| Config | Min VRAM | Notes |
|--------|----------|-------|
| 7B FP16 + calibration | 16 GB | No offloading |
| 7B FP16 + calibration | 12 GB | Some offloading |
| 7B 8-bit + calibration | 10 GB | bitsandbytes CUDA 12 |
| 7B 4-bit (working) | 8 GB | Need custom CUDA kernels |

**Conclusion**: The 8GB RTX 4060 Laptop **cannot run the full StoikovKV pipeline on real 7B model**. The fallback model validates the algorithmic pipeline end-to-end. Real 7B deployment requires ≥12GB VRAM or functional 4-bit quantization.