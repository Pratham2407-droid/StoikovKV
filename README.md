# StoikovKV

**StoikovKV** is a hybrid KV-cache compression and routing library for Grouped-Query Attention (GQA) language models. It combines write-time query-subspace probing, per-head role routing, 1-step preconditioned ElasticNet dictionary coding, non-stationary HMM topic-shift decay, and Matryoshka self-speculative decoding.

> **📊 Benchmark Numbers & Theory Documentation:**
> Browse **[`docs/benchmarks/`](./docs/benchmarks/)** for the ordered benchmark documentation:
> 1. **[`01_BENCHMARK_NUMBERS.md`](./docs/benchmarks/01_BENCHMARK_NUMBERS.md)** — Complete measured numbers across calibration, 1-step ElasticNet vs. OMP/FISTA, 4-way equal-budget cache ablation, Matryoshka speculative decoding, multi-length Needle-in-a-Haystack, WikiText-103, and real pretrained `Qwen/Qwen2.5-1.5B-Instruct` GPU validation.
> 2. **[`02_BENCHMARK_THEORY_AND_INTERPRETATION.md`](./docs/benchmarks/02_BENCHMARK_THEORY_AND_INTERPRETATION.md)** — Detailed theory guide explaining **what each test means**, **how each result was produced**, and **how to interpret the metrics** across all benchmark suites and all 20 unit tests.

---

## Repository Layout

```text
.
├── pyproject.toml                  # Package metadata, dependencies & stoikov-kv CLI entrypoint
├── src/
│   └── stoikov_kv/                 # Core Python package
│       ├── __init__.py
│       ├── calibration.py          # Streaming O(B*T) subspace (U_Q), head-role & dictionary calibration
│       ├── elastic_lexico.py       # 1-step ElasticNet sparse dictionary codec + OMP/FISTA baselines
│       ├── cache.py                # StoikovKVCache(DynamicCache) + HMMTopicShiftController
│       ├── matryoshka_spec.py      # Single-pass (gamma+1) Matryoshka self-speculative decoding
│       └── cli.py                  # Command-line interface (calibrate, chat, bench, serve)
├── docs/
│   ├── benchmarks/                 # Ordered benchmark numbers & theory documentation
│   │   ├── README.md               # Ordered reading guide for benchmarks & theory
│   │   ├── 01_BENCHMARK_NUMBERS.md # Consolidated empirical benchmark tables (Sections 1-7)
│   │   └── 02_BENCHMARK_THEORY_AND_INTERPRETATION.md # Test meanings, derivations & interpretation
│   └── reports/historical/         # Archived hackathon reports & raw snapshots
├── server/                         # FastAPI OpenAI-compatible (/v1/chat/completions) & telemetry backend
├── frontend/                       # Next.js real-time cache telemetry dashboard
├── config/                         # YAML configuration files (model, cache, server)
├── scripts/                        # Launch & E2E verification scripts
├── benchmarks/                     # Reproducible benchmark suites
│   ├── bench_calibration_codec.py  # Probe correlation, ElasticNet vs OMP/FISTA, domain L* break-even
│   ├── bench_cache_ablation.py     # Equal-budget 4-way cache ablation & Matryoshka sweep
│   ├── bench_context_drift.py      # Multi-turn topic-shift & H2O freshness evaluation
│   ├── bench_context_switch.py     # Multi-turn context switching evaluation
│   ├── bench_longbench.py          # Long-context task evaluation suite
│   ├── comparative_audit/          # Configurable multi-seed comparative audit runner
│   └── results/                    # Generated plots & JSON metrics (gitignored)
├── experiments/
│   └── 7b_gpu_validation/          # WikiText-103 & 7B GPU calibration/offloading scripts
├── tests/                          # Unit test suites (20 tests)
│   ├── test_calibration_and_codec.py
│   └── test_cache_and_speculative.py
├── artifacts/                      # Local calibrated .pt artifacts (gitignored)
├── models/                         # Local model weight shards (gitignored)
└── data/                           # Local evaluation datasets (gitignored)
```

---

## Architecture Overview

1. **Offline Streaming Calibration (`stoikov_kv.calibration`):**
   - Extracts a rank-$r$ ($r = 16$) orthonormal query subspace basis $U_Q \in \mathbb{R}^{L \times H_{kv} \times d \times r}$ per KV head group with column `0` anchored to the normalized mean query direction $\hat{\mu}_Q$, running in $O(B \cdot T)$ memory with `output_attentions=False`.
   - Scores incoming tokens at write time without materializing the $O(T^2)$ attention matrix:
     $$I_t = \phi\!\left((K_t U_Q)_0\right) \cdot \|K_t U_Q\|_2 \cdot \psi\!\left(\|V_t\|_2\right)$$
   - Classifies each KV head into one of three tiers:
     - `ROLE_RETRIEVAL_FP16 (0)`: High needle-copy heads (`~5%`) retained in exact FP16.
     - `ROLE_PEAKED_H2O (1)`: Low-entropy heads (`~84%`) managed via decayed heavy-hitter eviction (`keep_ratio`).
     - `ROLE_DIFFUSE_LEXICO (2)`: High-entropy diffuse heads (`~11%`) compressed into $s$ sorted dictionary atoms (`3s + 2` bytes/vector).
   - Trains per-layer normalized dictionaries $D \in \mathbb{R}^{L \times d \times N}$ on pre-RoPE keys and values, and precomputes the ridge projection matrix $W_{\text{ridge}} = (D^\top D + \lambda_2 I)^{-1} D^\top$ in FP64 (`residual < 1.5e-15`).

2. **Sparse Dictionary Codec (`stoikov_kv.elastic_lexico`):**
   - Encodes vectors in a single matrix-multiply + soft-threshold + top-$s$ selection pass (`0.53 ms` on GPU, `3.17x–31.60x` faster than OMP and `15.11x–16.87x` faster than FISTA), sorting atoms by descending magnitude (`|y_(1)| >= ... >= |y_(s)|`) and quantizing values to `int8` with a per-vector FP16 scale.
   - Un-rotates keys into the pre-RoPE frame prior to dictionary encoding and re-applies RoPE at the stored token position upon decoding.

3. **Live Physically Compacted Hybrid Cache & HMM Controller (`stoikov_kv.cache`):**
   - Maintains a recent uncompressed sliding window (`n_b = 64` tokens) and routes overflowing tokens per KV head according to `head_roles` with zero duplicate storage (`physical_resident_bytes == compressed_bytes`).
   - Compacts pure-H2O layers to `[B, H_kv, k_kept + n_b, d]` with a 1-token virtual partition sink and automatic SDPA mask slicing.
   - Tracks query-subspace projection energy shifts via `HMMTopicShiftController`, accelerating score decay ($\beta_t = 0.65$) on topic transitions so stale heavy hitters are replaced (`100.0%` Turn-2 freshness).

4. **Matryoshka Self-Speculative Decoding (`stoikov_kv.matryoshka_spec`):**
   - Drafts $\gamma_{\text{spec}}$ candidate tokens using only the top $m < s$ dictionary atoms on `ROLE_DIFFUSE_LEXICO` heads (`3m + 2` bytes/vector) and verifies all $\gamma_{\text{spec}} + 1$ positions in a single batched forward pass using all $s$ atoms, rolling back rejected tokens losslessly via `StoikovKVCache.crop(max_length)`.

---

## Engineering Status & Validation Summary

- **Complete Unit & Integration Test Coverage:** All **20 unit tests** (`tests/test_calibration_and_codec.py` and `tests/test_cache_and_speculative.py`) pass.
- **Physical == Logical KV Memory:** `StoikovKVCache` stores only tier-specific physical buffers (`fp16`, `h2o`, or packed `uint8`/`int8` `lexico`) with zero duplicate `DynamicCache` storage.
- **Drop-In HuggingFace & Pretrained Weight Validation:** Validated end-to-end on real pretrained `Qwen/Qwen2.5-1.5B-Instruct` and `Qwen/Qwen2.5-7B-Instruct` weights (`model.generate(past_key_values=cache)` and `matryoshka_speculative_generate`). See **[`docs/benchmarks/`](./docs/benchmarks/)** for full tables and analysis.

---

## Installation & Quickstart

```bash
# Install in editable mode
pip install -e .

# Run all 20 unit tests
python3 -m unittest tests/test_calibration_and_codec.py tests/test_cache_and_speculative.py -v

# Run calibration + sparse codec benchmark (writes to benchmarks/results/)
python3 benchmarks/bench_calibration_codec.py

# Run 4-way equal-budget cache ablation & Matryoshka sweep (writes to benchmarks/results/)
python3 benchmarks/bench_cache_ablation.py

# Use the CLI or launch the OpenAI-compatible FastAPI server
stoikov-kv --help
python3 scripts/serve.py
```
