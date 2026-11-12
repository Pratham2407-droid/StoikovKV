# StoikovKV Benchmark Numbers & Theory Documentation (`docs/benchmarks/`)

This directory contains the ordered benchmark documentation for **StoikovKV**, pairing the consolidated empirical numbers with the underlying mathematical derivations, causal explanations, and engineering interpretation guide.

---

## Ordered Reading Guide

1. **[`01_BENCHMARK_NUMBERS.md`](./01_BENCHMARK_NUMBERS.md) — StoikovKV Benchmark Numbers Reference**
   - Complete measured tables across all 7 benchmark tracks:
     1. Offline Calibration & Write-Time Query-Subspace Probe (`bench_calibration_codec.py`)
     2. Sparse Dictionary Codec: Fast 1-Step ElasticNet vs. OMP vs. FISTA (`bench_calibration_codec.py`)
     3. 4-Way Equal-Budget (~20% KV Bytes) Live Cache Ablation (`bench_cache_ablation.py`)
     4. Matryoshka Self-Speculative Decoding Sweep (`bench_cache_ablation.py`)
     5. Multi-Length (`512`, `2,048`, `4,096`) Needle Retrieval & Latency Audit (`benchmarks/comparative_audit/`)
     6. WikiText-103 Throughput & 7B Offloaded Hardware Validation (`experiments/7b_gpu_validation/`)
     7. Real Pretrained `Qwen/Qwen2.5-1.5B-Instruct` End-to-End GPU Validation (`real_qwen15b_gpu_metrics.json`)

2. **[`02_BENCHMARK_THEORY_AND_INTERPRETATION.md`](./02_BENCHMARK_THEORY_AND_INTERPRETATION.md) — Benchmark Theory, Causal Mechanics & Interpretation Guide**
   - Section-by-section and test-by-test breakdown explaining:
     - **What the Test Means:** The exact hypothesis, mathematical property, or hardware bottleneck evaluated by each benchmark and all 20 unit tests (`tests/`).
     - **How the Result Became the Result:** The step-by-step causal and mathematical derivation behind each measured metric (directional subspace probing, FP64 ridge preconditioning, 1-token virtual partition sinks, HMM topic-shift score decay, and Matryoshka nested prefix decoding).
     - **How to Infer & Interpret It:** How to read the tradeoffs between `Full FP16`, `Standalone H2O`, `Standalone Lexico`, and `StoikovKV Hybrid`, and how to diagnose regressions.
