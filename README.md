# StoikovKV: Adaptive KV-Cache Compression for Decoder-Only Transformers

Adaptive, per-head KV-cache compression unifying H2O eviction, Lexico sparse coding, and Matryoshka self-speculative decoding under live VRAM pressure control.

## Qwen2.5-7B-Instruct on 8GB VRAM (Real Model)

| Configuration | Needle Retrieval | Memory % | Compression | Tokens/sec | Peak VRAM |
|---------------|-----------------|----------|-------------|------------|-----------|
| Full FP16 (real 7B) | FAIL | 100% | 1.0x | 0.8 tok/s | 6.9 GB |
| StoikovKV Hybrid (real 7B) | FAIL | 33-41% | 2.4-3.0x | 0.7-0.8 tok/s | 6.9 GB |
| Standalone H2O (real 7B) | FAIL | 28-36% | 2.8-3.6x | 0.8 tok/s | 6.9 GB |

Root Cause: 8GB VRAM insufficient for real 7B calibration. Forced reduced prompts (3 prompts, seq_len=64) -> probe correlation ρ=-0.022 -> only 6 retrieval heads vs ~20 needed -> recall collapses. Memory savings 32-67% work but retrieval fails.

Real 7B Memory Savings (Hybrid Policy):
| Context | Full FP16 | Hybrid | Savings |
|---------|-----------|--------|---------|
| 28 MB (Early) | 100% | 39.9% | 60.1% |
| 32 MB (Middle) | 100% | 36.9% | 63.1% |
| 40 MB (Late) | 100% | 32.7% | 67.3% |

Hardware Requirements for Real 7B:
| Config | Min VRAM | Notes |
|--------|----------|-------|
| 7B FP16 + full calibration | 16 GB | No offloading |
| 7B FP16 + reduced calibration | 12 GB | Minimum viable |
| 7B 4-bit (working) | 8 GB | Need custom CUDA kernels |

## Qwen2.5-1.5B-Instruct (Fallback Model, 8GB RTX 4060)

| Configuration | Needle Retrieval | Physical Storage | Compression | Logit Cosine |
|---------------|-----------------|------------------|-------------|--------------|
| Full FP16 (baseline) | 100% | 256 B/vec | 1.0x | 1.000 |
| StoikovKV Hybrid (s=16) | 100% | 108.86 B/vec | 2.35x | 0.998 |
| StoikovKV Hybrid Draft (m=4) | 100% | 83.14 B/vec | 3.08x | 0.998 |
| Lexico only (s=16) | 100% | 256 B/vec | 1.0x | 0.998 |
| H2O only (r=0.3) | 0% | 44 B/vec | 5.8x | - |
| Unprotected Ablation | 16.7% | 5 B/vec | 51.2x | 0.11 |

Key Insight: Pure H2O/Lexico fail at high compression. Hybrid works by protecting retrieval heads in FP16 while compressing the rest.

## Architecture

StoikovKVCache:
- Per-Head Router (offline, entropy-based): ROLE_RETRIEVAL_FP16 (0) -> Exact FP16, never evicted; ROLE_PEAKED_H2O (1) -> Score-based eviction; ROLE_DIFFUSE_LEXICO (2) -> Sparse dictionary encoding
- Write-Time Importance Probe: ||K @ U_Q||_2 * ||V||_2
- H2O: Top-k by score + context window dilation
- Lexico: Fast 1-step Ridge + Soft-thresholding (s atoms)
- Matryoshka Self-Speculation: m=4 draft -> s=16 verify
- Avellaneda-Stoikov Controller: Live VRAM pressure control

## 7B GPU Validation (Qwen2.5-7B, 8GB RTX 4060)

| Aspect | Fallback (1.5B) | Real 7B (FP16 + CPU offload) |
|--------|-----------------|------------------------------|
| Model | 12L, 4 KV heads, h=64 | 28L, 4 KV heads, h=128 |
| Calibration | Full (ρ=0.546) | Reduced (ρ=-0.022) |
| Needle Retrieval | 100% | FAILS (ρ=-0.022) |
| Throughput | 1,007 tok/s | 0.8 tok/s |
| Peak VRAM | 0.38 GB | 6.9 GB |
| Hybrid Memory | 19.8% (80% savings) | 33-41% (59-67% savings) |
| Retrieval Heads | 2 (4.2%) | 6 (5.4%) |

Why 7B retrieval fails: 8GB VRAM forces reduced calibration (3 prompts, seq_len=64) -> probe correlation ρ=-0.022 -> only 6 retrieval heads vs ~20 needed -> recall collapses.

Real 7B Memory Savings (Hybrid Policy):
| Context | Full FP16 | Hybrid | Savings |
|---------|-----------|--------|---------|
| 28 MB (Early) | 100% | 39.9% | 60.1% |
| 32 MB (Middle) | 100% | 36.9% | 63.1% |
| 40 MB (Late) | 100% | 32.7% | 67.3% |

Hardware Requirements for Real 7B:
| Config | Min VRAM | Notes |
|--------|----------|-------|
| 7B FP16 + full calibration | 16 GB | No offloading |
| 7B FP16 + reduced calibration | 12 GB | Minimum viable |
| 7B 4-bit (working) | 8 GB | Need custom CUDA kernels |

## Benchmarks

| Benchmark | Command | Description |
|-----------|---------|-------------|
| Person 1 | python benchmarks/bench_person1.py | Full calibration + codec + passkey eval |
| Context Switch | python benchmarks/bench_context_switch.py | Topic-switching retrieval test |
| Unit Tests | python tests/test_person1_pipeline.py | 11 pipeline tests |

Key Metrics (from person1_metrics.json):

| Metric | Value |
|--------|-------|
| Probe Spearman (raw) | 0.4265 |
| Probe Recall@20% | 53.6% |
| Lexico Encode (s=16) | 0.98 ms |
| Lexico Cosine (s=16) | 0.73 |
| Speedup vs OMP | 12.8x |
| Speedup vs FISTA | 24x |
| Adaptation L* (Python) | 916 tokens |
| Adaptation L* (Kannada) | 442 tokens |

## 7B GPU Validation (Qwen2.5-7B, 8GB RTX 4060)

| Aspect | Fallback (1.5B) | Real 7B (FP16 + CPU offload) |
|--------|-----------------|------------------------------|
| Model | 12L, 4 KV heads, h=64 | 28L, 4 KV heads, h=128 |
| Calibration | Full (ρ=0.546) | Reduced (ρ=-0.022) |
| Needle Retrieval | 100% | FAILS (ρ=-0.022) |
| Throughput | 1,007 tok/s | 0.8 tok/s |
| Peak VRAM | 0.38 GB | 6.9 GB |
| Hybrid Memory | 19.8% (80% savings) | 33-41% (59-67% savings) |
| Retrieval Heads | 2 (4.2%) | 6 (5.4%) |

Why 7B retrieval fails: 8GB VRAM forces reduced calibration (3 prompts, seq_len=64) -> probe correlation ρ=-0.022 -> only 6 retrieval heads vs ~20 needed -> recall collapses.

Real 7B Memory Savings (Hybrid Policy):
| Context | Full FP16 | Hybrid | Savings |
|---------|-----------|--------|---------|
| 28 MB (Early) | 100% | 39.9% | 60.1% |
| 32 MB (Middle) | 100% | 36.9% | 63.1% |
| 40 MB (Late) | 100% | 32.7% | 67.3% |

Hardware Requirements for Real 7B:
| Config | Min VRAM | Notes |
|--------|----------|-------|
| 7B FP16 + full calibration | 16 GB | No offloading |
| 7B FP16 + reduced calibration | 12 GB | Minimum viable |
| 7B 4-bit (working) | 8 GB | Need custom CUDA kernels |

## WikiText-103 Production Evaluation (Fallback Model, 50 samples)

| Policy | Tok/s | Peak VRAM | Memory % | Savings | HMM β_t | HMM Shifts |
|--------|-------|-----------|----------|---------|---------|------------|
| Full FP16 | 1,457 | 0.41 GB | 100.0% | 0.0% | 1.00 | 0 |
| StoikovKV Hybrid | 1,007 | 0.38 GB | 19.8% | 80.2% | 0.99 | 14 |
| Standalone H2O | 1,578 | 0.37 GB | 13.3% | 86.7% | 1.00 | 0 |
| Standalone Lexico | 361 | 0.43 GB | 41.1% | 58.9% | 1.00 | 0 |

Per-Tier Token Allocation (Hybrid):
| Tier | Token Count | % of Total |
|------|-------------|------------|
| Local Window FP16 | 6,144 | 3.3% |
| Retrieval FP16 | 7,414 | 4.0% |
| H2O Peaked | 15,703 | 8.5% |
| Lexico Sparse | 18,535 | 10.1% |
| Evicted H2O | 136,284 | 74.1% |

HMM Topic Shift Detection: WikiText-103 triggers 14 HMM shift events with hybrid policy. HMM β_t drops from 1.0 to 0.99. Z-score peaks at 0.67.

Needle Retrieval (100% PASS all policies):
| Test | Full FP16 | Hybrid | Standalone H2O | Standalone Lexico |
|------|-----------|--------|----------------|-------------------|
| Needle Early | PASS | PASS | PASS | PASS |
| Needle Middle | PASS | PASS | PASS | PASS |
| Needle Late | PASS | PASS | PASS | PASS |

Needle Retrieval Detailed:
| Test | Policy | Memory % | Tokens/s |
|------|--------|----------|----------|
| Early | Full FP16 | 100% | 24.5 |
| | Hybrid | 42.1% | 13.8 |
| | Standalone H2O | 37.6% | 23.3 |
| | Standalone Lexico | 55.9% | 4.5 |

Context Drift (2,970 tokens):
| Policy | Retrieval | Memory % | H2O Freshness | HMM Shifts | β_t | Z-Score |
|--------|-----------|----------|---------------|------------|-----|---------|
| Full FP16 | FAIL | 100% | 0% | 0 | 1.00 | 0.00 |
| Hybrid | FAIL | 20.7% | 100% | 5 | 0.65 | 1.46 |

HMM detects shift (z=1.46, β_t=0.65) and achieves 100% H2O freshness post-drift.

Matryoshka Self-Speculative Decoding:
| Draft Atoms m | Draft Payload | Acceptance α | Avg Tokens/Cycle |
|---------------|---------------|--------------|------------------|
| m=2 | 8 B/vec | 96.9% | 4.65 |
| m=4 | 14 B/vec | 100% | 5.00 |
| m=8 | 26 B/vec | 100% | 5.00 |
| m=12 | 38 B/vec | 100% | 5.00 |

## Run

```bash
pip install torch transformers accelerate scipy matplotlib numpy
python benchmarks/bench_person1.py --model Qwen/Qwen2.5-1.5B-Instruct
```

## Repository Structure

```
StoikovKV/
├── stoikov_kv/
│   ├── __init__.py
│   ├── calibration.py      # Offline calibration, U_Q, head roles, dictionaries
│   ├── cache.py            # StoikovKVCache (DynamicCache subclass)
│   ├── elastic_lexico.py   # ElasticLexicoCodec (1-step encoder, OMP, FISTA)
│   └── matryoshka_spec.py  # Nested-atom speculative decoding
├── benchmarks/
│   ├── bench_person1.py    # Full calibration + codec + passkey eval
│   ├── bench_person2.py    # Legacy needle benchmark
│   └── bench_context_switch.py  # Multi-topic context switching
├── experiments/
│   └── 7b_gpu_validation/  # Real 7B validation (NEW)
│       ├── calibrate_7b_real.py      # Memory-optimized calibration
│       ├── bench_7b_stoikovkv.py     # Real 7B policy evaluation
│       ├── bench_7b_baseline.py      # FP16 baseline
│       ├── bench_wikitext.py         # WikiText-103 evaluation
│       ├── COMPLETE_EXPERIMENT_RESULTS.md  # All results in one file
│       └── calibrated_artifacts_7b_real.pt
├── tests/
│   └── test_person1_pipeline.py  # 11 unit tests (all pass)
├── server/
│   ├── app.py              # FastAPI backend
│   └── index.html          # FinOps dashboard (Tailwind + Chart.js)
├── calibrated_artifacts.pt  # Pre-calibrated artifacts (Qwen2.5-1.5B)
├── requirements.txt
```

## Key Implementation Details

### Calibration (calibration.py)
- Write-time probe: ||K @ U_Q||_2 * ||V||_2 with RoPE alignment
- Head routing: Entropy-based (peaked -> H2O, diffuse -> Lexico, top 5% -> Retrieval)
- Dictionary learning: SVD init + Adam (60 steps) + incoherence regularization
- Probe evaluation: Spearman ρ & Recall@20% vs attention-derived ground truth

### Elastic Lexico Codec (elastic_lexico.py)
- Fast 1-step encoder: indices, values = codec.encode(KV_vectors, layer_idx=0, s=16)
- Nested partial decoding: X_hat = codec.decode(indices, values, active_atoms=m)
- Online adaptation: adapt_info = codec.adapt_online_atoms(X_fit, X_hat, M_adapt=64, X_eval=X_test)

### Cache (cache.py)
- StoikovKVCache(model, keep_ratio=0.3, h2o_context_window=1, lexico_fp16_frac=0.5, dense_fallback=True)
- cache.set_matryoshka_mode(m=4) for draft mode
- cache.set_matryoshka_mode(16) for verify mode

## License

MIT License

## Acknowledgments

- H2O: Zhang et al., NeurIPS 2023
- Lexico: Kim et al., ICLR 2025
- Speculative Decoding: Leviathan et al., ICML 2023
- Avellaneda-Stoikov: Avellaneda & Stoikov, 2008
- Qwen2.5: Alibaba Cloud Qwen Team