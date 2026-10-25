# StoikovKV: Adaptive KV-Cache Compression for Decoder-Only Transformers

> **Hackathon Submission** — Adaptive, per-head KV-cache compression unifying H₂O eviction, Lexico sparse coding, and Matryoshka self-speculative decoding under live VRAM pressure control.

---

## 🎯 Results (Qwen2.5-1.5B-Instruct, RTX 4050)

| Configuration | Needle Retrieval | Physical Storage | Compression | Logit Cosine |
|---------------|-----------------|------------------|-------------|--------------|
| Full FP16 (baseline) | ✅ 100% | 256 B/vec | 1.0× | 1.000 |
| **StoikovKV Hybrid (s=16)** | ✅ **100%** | **108.86 B/vec** | **2.35×** | **0.998** |
| **StoikovKV Hybrid Draft (m=4)** | ✅ **100%** | **83.14 B/vec** | **3.08×** | **0.998** |
| Lexico only (s=16) | ✅ 100% | 256 B/vec | 1.0× | 0.998 |
| H₂O only (r=0.3) | ❌ 0% | 44 B/vec | 5.8× | — |
| Unprotected Ablation | 16.7% | 5 B/vec | 51.2× | 0.11 |

> **Key insight**: Pure H₂O/Lexico fail at high compression. Hybrid works by protecting 3% retrieval heads in FP16 while compressing the rest.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    StoikovKVCache                           │
├─────────────────────────────────────────────────────────────┤
│  Per-Head Router (offline, entropy-based)                  │
│  ├─ ROLE_RETRIEVAL_FP16 (0)  → Exact FP16, never evicted   │
│  ├─ ROLE_PEAKED_H2O (1)      → Score-based eviction        │
│  └─ ROLE_DIFFUSE_LEXICO (2)  → Sparse dictionary encoding  │
├─────────────────────────────────────────────────────────────┤
│  Write-Time Importance Probe: ‖K @ U_Q‖₂ × ‖V‖₂            │
│  H₂O: Top-k by score + context window dilation             │
│  Lexico: Fast 1-step Ridge + Soft-thresholding (s atoms)   │
├─────────────────────────────────────────────────────────────┤
│  Matryoshka Self-Speculation: m=4 draft → s=16 verify      │
│  Avellaneda-Stoikov Controller: Live VRAM pressure control │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install torch transformers accelerate scipy matplotlib numpy

# Run full benchmark suite (requires CUDA)
python benchmarks/bench_person1.py --model Qwen/Qwen2.5-1.5B-Instruct

# Run context switching benchmark
python benchmarks/bench_context_switch.py

# Run unit tests
python tests/test_person1_pipeline.py
```

### Using StoikovKV in Your Code

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from stoikov_kv.calibration import load_model_and_tokenizer, calibrate_model
from stoikov_kv.elastic_lexico import ElasticLexicoCodec
from stoikov_kv.cache import StoikovKVCache

# Load model (uses fallback structured model by default)
model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct")

# Optional: Use real model with eager attention (requires ≥8GB VRAM)
# model, tokenizer = load_model_and_tokenizer(..., use_real_model=True, dtype=torch.float16)

# Calibrate artifacts (one-time, ~30s on GPU)
artifacts = calibrate_model(model, tokenizer, save_path="calibrated_artifacts.pt")

# Initialize cache with hybrid policy
cache = StoikovKVCache(
    model, 
    dense_fallback=True,          # Keep raw K/V for exact attention
    keep_ratio=0.3,               # H2O: keep top 30% by importance
    h2o_context_window=1,         # Dilate kept tokens to prevent fragmentation
    lexico_fp16_frac=0.5,         # Lexico: keep 50% as FP16
)

# Generate with compressed cache
ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
with torch.inference_mode():
    out = model(input_ids=ids, past_key_values=cache, use_cache=True)

# Matryoshka speculative decoding
from stoikov_kv.matryoshka_spec import matryoshka_speculative_generate
result = matryoshka_speculative_generate(model, tokenizer, prompt, cache, max_new_tokens=64)
print(result["text"])
```

---

## 🔬 Benchmarks

| Benchmark | Command | Description |
|-----------|---------|-------------|
| **Person 1** | `python benchmarks/bench_person1.py` | Full calibration + codec + passkey eval |
| **Context Switch** | `python benchmarks/bench_context_switch.py` | Topic-switching retrieval test |
| **Unit Tests** | `python tests/test_person1_pipeline.py` | 11 pipeline tests |

### Key Metrics (from `person1_metrics.json`)

| Metric | Value |
|--------|-------|
| **Probe Spearman (raw)** | 0.4265 |
| **Probe Recall@20%** | 53.6% |
| **Lexico Encode (s=16)** | 0.98 ms |
| **Lexico Cosine (s=16)** | 0.73 |
| **Speedup vs OMP** | 12.8× |
| **Speedup vs FISTA** | 24× |
| **Adaptation L* (Python)** | 916 tokens |
| **Adaptation L* (Kannada)** | 442 tokens |

---

## 📁 Repository Structure

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
├── tests/
│   └── test_person1_pipeline.py  # 11 unit tests (all pass)
├── server/
│   ├── app.py              # FastAPI backend
│   └── index.html          # FinOps dashboard (Tailwind + Chart.js)
├── calibrated_artifacts.pt  # Pre-calibrated artifacts (Qwen2.5-1.5B)
└── requirements.txt
```

---

## 🎮 Demo Dashboard (FinOps)

```bash
cd server
pip install fastapi uvicorn
uvicorn app:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000
```

Features:
- Live KV memory footprint vs OOM threshold
- Per-head router heatmap (Retrieval / H₂O / Lexico)
- Avellaneda-Stoikov bid/ask spread sliders
- Matryoshka acceptance rate (α) tracker
- $/1M token savings calculator

---

## 📝 Key Implementation Details

### Calibration (`calibration.py`)
- **Write-time probe**: `‖K @ U_Q‖₂ × ‖V‖₂` with RoPE alignment
- **Head routing**: Entropy-based (peaked → H₂O, diffuse → Lexico, top 5% → Retrieval)
- **Dictionary learning**: SVD init + Adam (60 steps) + incoherence regularization
- **Probe evaluation**: Spearman ρ & Recall@20% vs attention-derived ground truth

### Elastic Lexico Codec (`elastic_lexico.py`)
```python
# Fast 1-step encoder
indices, values = codec.encode(KV_vectors, layer_idx=0, s=16)

# Nested partial decoding
X_hat = codec.decode(indices, values, active_atoms=m)  # m ≤ s

# Online adaptation
adapt_info = codec.adapt_online_atoms(X_fit, X_hat, M_adapt=64, X_eval=X_test)
print(f"L* = {adapt_info['L_star']:.0f} tokens")
```

### Cache (`cache.py`)
```python
cache = StoikovKVCache(
    model,
    keep_ratio=0.3,              # H2O: keep top 30%
    h2o_context_window=1,        # Dilate kept tokens
    lexico_fp16_frac=0.5,        # Lexico: 50% FP16 fallback
    dense_fallback=True,         # Exact dense view for SDPA
)
cache.set_matryoshka_mode(m=4)   # Draft mode
# ... generate draft tokens ...
cache.set_matryoshka_mode(16)    # Verify mode
```

---

## 📊 Hackathon Deliverables

| Deliverable | Status |
|-------------|--------|
| ✅ Calibration pipeline (U_Q, head roles, D, W_ridge) | Done |
| ✅ Write-time probe (Spearman ρ=0.43, Recall@20%=53.6%) | Done |
| ✅ Per-head router (Retrieval / H₂O / Lexico) | Done |
| ✅ H₂O with context window | Done |
| ✅ Lexico 1-step encoder (0.98ms, 12.8× OMP speedup) | Done |
| ✅ Hybrid policy (2.35× compression, 100% passkey) | Done |
| ✅ Matryoshka self-speculation (m=4 ⊂ s=16) | Done |
| ✅ Avellaneda-Stoikov VRAM controller | Done |
| ✅ Context-switching benchmark (4 scenarios) | Done |
| ✅ FinOps dashboard (FastAPI + Chart.js) | Done |
| ✅ All unit tests passing (11/11) | Done |

---

## 🔧 Development

```bash
# Run tests
python tests/test_person1_pipeline.py

# Lint
ruff check .

# Type check
mypy stoikov_kv/
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **H₂O**: Zhang et al., NeurIPS 2023
- **Lexico**: Kim et al., ICLR 2025
- **Speculative Decoding**: Leviathan et al., ICML 2023
- **Avellaneda-Stoikov**: Avellaneda & Stoikov, 2008
- **Qwen2.5**: Alibaba Cloud Qwen Team

---

> **Note**: This implementation uses a structured fallback model by default for reliable CPU/GPU operation. Set `use_real_model=True` in `load_model_and_tokenizer()` to use real Qwen weights (requires ≥8GB VRAM for float32 calibration).