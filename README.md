# StoikovKV

**StoikovKV** is a hybrid KV-cache compression and routing library for Grouped-Query Attention (GQA) language models. It combines write-time query-subspace probing, per-head role routing, 1-step preconditioned ElasticNet dictionary coding, non-stationary HMM topic-shift decay, and Matryoshka self-speculative decoding.

---

## Repository Layout

```text
.
├── pyproject.toml                  # Package metadata & dependencies (src/ layout)
├── BENCHMARK_NUMBERS.md            # Complete consolidated benchmark tables & metrics
├── src/
│   └── stoikov_kv/                 # Core Python package
│       ├── __init__.py
│       ├── calibration.py          # Offline subspace (U_Q), head-role & dictionary calibration
│       ├── elastic_lexico.py       # 1-step ElasticNet sparse dictionary codec + OMP/FISTA baselines
│       ├── cache.py                # StoikovKVCache(DynamicCache) + HMMTopicShiftController
│       └── matryoshka_spec.py      # Matryoshka nested-atom self-speculative decoding
├── server/                         # FastAPI inference & telemetry backend
├── frontend/                       # Next.js real-time cache telemetry dashboard
├── config/                         # YAML configuration files (model, cache, server)
├── scripts/                        # Launch & E2E verification scripts
├── benchmarks/                     # Reproducible benchmark suites
│   ├── bench_calibration_codec.py  # Probe correlation, ElasticNet vs OMP/FISTA, domain L* break-even
│   ├── bench_cache_ablation.py     # Equal-budget 4-way cache ablation & Matryoshka sweep
│   ├── bench_context_drift.py      # Multi-turn topic-shift & H2O freshness evaluation
│   ├── bench_context_switch.py     # Multi-turn context switching evaluation
│   ├── bench_longbench.py          # Long-context task evaluation suite
│   ├── comparative_audit/          # Configurable comparative audit runner
│   └── results/                    # Generated plots & JSON metrics (gitignored)
├── experiments/
│   └── 7b_gpu_validation/          # 7B GPU calibration & offloading experiment scripts
├── tests/                          # Unit test suites
│   ├── test_calibration_and_codec.py
│   └── test_cache_and_speculative.py
├── docs/
│   └── reports/historical/         # Archived hackathon reports & raw snapshots
├── artifacts/                      # Local calibrated .pt artifacts (gitignored)
├── models/                         # Local model weight shards (gitignored)
└── data/                           # Local evaluation datasets (gitignored)
```

---

## Architecture Overview

1. **Offline Calibration (`stoikov_kv.calibration`):**
   - Extracts a rank-$r$ ($r = 16$) orthonormal query subspace basis $U_Q \in \mathbb{R}^{L \times H_{kv} \times d \times r}$ per KV head group.
   - Scores incoming tokens at write time without materializing the $O(T^2)$ attention matrix:
     $$I_t = \|K_t U_Q\|_2 \cdot \|V_t\|_2$$
   - Classifies each KV head into one of three tiers:
     - `ROLE_RETRIEVAL_FP16 (0)`: High needle-copy heads retained in exact FP16.
     - `ROLE_PEAKED_H2O (1)`: Low-entropy heads managed via decayed heavy-hitter eviction.
     - `ROLE_DIFFUSE_LEXICO (2)`: High-entropy diffuse heads compressed into $s$ sorted dictionary atoms (`3s + 2` bytes/vector).
   - Trains per-layer normalized dictionaries $D \in \mathbb{R}^{L \times d \times 1024}$ on pre-RoPE keys and values, and precomputes the ridge projection matrix $W_{\text{ridge}} = (D^\top D + \lambda_2 I)^{-1} D^\top$ in FP64.

2. **Sparse Dictionary Codec (`stoikov_kv.elastic_lexico`):**
   - Encodes vectors in a single matrix-multiply + soft-threshold + top-$s$ selection pass, sorting atoms by descending magnitude (`|y_(1)| >= ... >= |y_(s)|`) and quantizing values to `int8` with a per-vector FP16 scale.
   - Un-rotates keys into the pre-RoPE frame prior to dictionary encoding and re-applies RoPE at the stored token position upon decoding.

3. **Live Hybrid Cache & HMM Controller (`stoikov_kv.cache`):**
   - Maintains a recent uncompressed sliding window (`n_b = 64` tokens) and routes overflowing tokens per KV head according to `head_roles`.
   - Tracks query-subspace projection energy shifts via `HMMTopicShiftController`, accelerating score decay ($\beta_t = 0.65$) on topic transitions so stale heavy hitters are replaced.

4. **Matryoshka Self-Speculative Decoding (`stoikov_kv.matryoshka_spec`):**
   - Drafts $\gamma_{\text{spec}}$ candidate tokens using only the top $m < s$ dictionary atoms on `ROLE_DIFFUSE_LEXICO` heads (`3m + 2` bytes/vector) and verifies in parallel using all $s$ atoms, rolling back rejected tokens via `StoikovKVCache.crop(max_length)`.

---

## Current Engineering Status & Limitations

- **Verified Mechanics & Accounting:** All 15 unit tests (`tests/`) verify GQA subspace extraction, write-time probing, pre-RoPE dictionary coding, equal-budget tier routing, lossless `crop()` rollback, and Matryoshka draft/verify generation.
- **Logical vs. Physical Memory:** `StoikovKVCache` currently tracks exact logical tier bytes (`compressed_bytes` and `dictionary_bytes`), but materializes rectangular FP16 tensors per layer for standard SDPA and delegates sequence tracking to `DynamicCache`. Eliminating duplicate physical storage and adding a tier-aware attention kernel is tracked as Phase 1 of the productization roadmap.
- **Pretrained Model Validation:** By default, offline tests and benchmarks fall back to a deterministic synthetic GQA harness (`FallbackQwen2ForCausalLM`) when Hugging Face Hub weights are unavailable. Pretrained model benchmarks should be run with real checkpoints (`Qwen/Qwen2.5-1.5B-Instruct` or `Qwen/Qwen2.5-3B-Instruct`).

---

## Installation & Quickstart

```bash
# Activate virtualenv and install in editable mode
source .venv/bin/activate
pip install -e .

# Run unit tests
python -m unittest discover -s tests -v

# Run calibration + sparse codec benchmark (writes to benchmarks/results/)
python benchmarks/bench_calibration_codec.py

# Run 4-way equal-budget cache ablation & Matryoshka sweep (writes to benchmarks/results/)
python benchmarks/bench_cache_ablation.py

# Launch the FastAPI telemetry server
python scripts/serve.py
```
