# StoikovKV Local LLM Integration & Fine-Tuning Plan

## Executive Summary

**Goal**: Deploy Qwen2.5-3B locally with StoikovKV hybrid KV cache, fine-tune for Option C (compression-aware generation + robustness to compressed KV), serve via Next.js frontend, evaluate using existing Person 1/2 metrics.

**Current State**: Core KV cache + benchmarks + tests exist. **Missing**: Model server, fine-tuning pipeline, Next.js frontend, integration glue.

---

## Repository Structure (Target)

```
StoikovKV/
├── stoikov_kv/                    # Core library (✓ DONE)
│   ├── __init__.py
│   ├── cache.py                   # StoikovKVCache(DynamicCache)
│   ├── calibration.py             # Offline calibration + head routing
│   ├── elastic_lexico.py          # 1-step encoder + OMP + FISTA + adaptation
│   └── matryoshka_spec.py         # Self-speculative decoding
│
├── server/                        # NEW: Model serving
│   ├── __init__.py
│   ├── main.py                    # FastAPI app
│   ├── routes/
│   │   ├── generate.py            # Streaming /generate endpoint
│   │   ├── health.py              # Health + metrics
│   │   └── artifacts.py           # Calibration artifact mgmt
│   ├── services/
│   │   ├── model_loader.py        # Qwen2.5-3B + StoikovKVCache init
│   │   ├── cache_manager.py       # Per-session cache lifecycle
│   │   └── metrics_collector.py   # Latency, compression, quality
│   └── config.py                  # YAML config (model, cache, server)
│
├── training/                      # NEW: Fine-tuning pipeline
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── compression_dataset.py # Synthetic + real data with compression labels
│   │   └── needle_dataset.py      # Needle-in-haystack for retrieval heads
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── compression_loss.py    # Logit cosine + prob retention + reconstruction
│   │   └── retrieval_loss.py      # Passkey accuracy + distractor suppression
│   ├── trainers/
│   │   ├── __init__.py
│   │   ├── sft_trainer.py         # Supervised fine-tuning (LoRA)
│   │   └── rlhf_trainer.py        # DPO/PPO with compression reward
│   ├── config.py                  # Training config (LoRA, epochs, lr)
│   └── run_training.py            # Entry point
│
├── frontend/                      # NEW: Next.js app
│   ├── package.json
│   ├── next.config.js
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx           # Main chat interface
│   │   │   ├── api/               # Next.js API routes (proxy to FastAPI)
│   │   │   └── layout.tsx
│   │   ├── components/
│   │   │   ├── ChatInterface.tsx
│   │   │   ├── MetricsDashboard.tsx
│   │   │   ├── HeadRouterMap.tsx  # Visual: Retrieval/H2O/Lexico per layer
│   │   │   ├── KVPressureGauge.tsx
│   │   │   └── SpeculativeStats.tsx
│   │   ├── hooks/
│   │   │   ├── useStreamingChat.ts
│   │   │   └── useMetrics.ts
│   │   └── lib/
│   │       ├── api.ts             # FastAPI client
│   │       └── types.ts
│   └── tailwind.config.ts
│
├── benchmarks/                    # ✓ DONE
│   ├── bench_person1.py
│   ├── bench_person2.py
│   └── ...
│
├── tests/                         # ✓ DONE
│   ├── test_person1_pipeline.py
│   └── test_person2_pipeline.py
│
├── scripts/
│   ├── calibrate.py               # One-shot calibration runner
│   ├── serve.py                   # Start server + model
│   ├── train.py                   # Launch fine-tuning
│   └── eval.py                    # Run benchmarks against fine-tuned model
│
├── config/
│   ├── model.yaml                 # Qwen2.5-3B, LoRA, quantization
│   ├── cache.yaml                 # StoikovKV params (n_b, keep_ratio, s_budget, HMM)
│   ├── server.yaml                # FastAPI host/port, workers
│   └── training.yaml              # Epochs, lr, batch, compression reward weights
│
├── StoikovKV_Benchmark_Report.pdf # ✓ DONE
└── README.md
```

---

## Phase 1: Local Model Server (Week 1-2)

### 1.1 Model Loader (`server/services/model_loader.py`)
```python
# Load Qwen2.5-3B-Instruct (FP16/BF16) + StoikovKVCache
# Config: model_name, device, dtype, attn_impl, cache_params
# Returns: model, tokenizer, cache_factory
```

**Key decisions:**
- Use `transformers` with `attn_implementation="sdpa"` (FlashAttention-2 if available)
- Subclass `DynamicCache` → `StoikovKVCache` hooks directly into `model.generate()`
- Load `calibrated_artifacts.pt` or run `create_mock_artifacts()` on first boot
- Support CPU fallback for dev (MPS on Mac, CPU on Linux)

### 1.2 FastAPI Server (`server/main.py`)
```python
# POST /api/v1/generate     - Streaming generation (SSE)
# POST /api/v1/chat         - OpenAI-compatible chat completions
# GET  /api/v1/health       - Model loaded, VRAM, cache stats
# GET  /api/v1/metrics      # Prometheus-compatible
# POST /api/v1/calibrate    # Trigger calibration (background job)
```

**Streaming implementation:**
```python
async def stream_generate(request: GenerateRequest):
    cache = cache_factory.create_session_cache(session_id)
    async for token in model.astream(input_ids, past_key_values=cache):
        yield f"data: {json.dumps({'token': token, 'metrics': cache.get_telemetry()})}\n\n"
```

### 1.3 Cache Manager (`server/services/cache_manager.py`)
- Per-session `StoikovKVCache` lifecycle (create, crop on context window, destroy)
- Memory pressure monitoring → Avellaneda-Stoikov dynamic `s_budget`
- Telemetry aggregation for dashboard

### 1.4 Config System
```yaml
# config/model.yaml
model:
  name: "Qwen/Qwen2.5-3B-Instruct"
  dtype: "float16"
  device: "cuda"
  attn_implementation: "sdpa"
  max_seq_len: 32768

# config/cache.yaml
cache:
  n_b: 64
  keep_ratio: 0.15
  s_budget: 16
  policy_mode: "hybrid"
  enable_hmm_decay: true
  shift_z_threshold: 1.4
  beta_stationary: 0.99
  beta_shift: 0.65

# config/training.yaml
training:
  lora:
    r: 32
    alpha: 64
    dropout: 0.1
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  epochs: 3
  lr: 2e-4
  batch_size: 4
  gradient_accumulation: 4
```

---

## Phase 2: Fine-Tuning Pipeline (Week 2-4)

### 2.1 Option C Strategy: Dual-Objective Fine-Tuning

| Objective | Loss Component | Weight | Data |
|-----------|----------------|--------|------|
| **Compression-Aware Generation** | `L_comp = 1 - logit_cosine + (1 - prob_retention) + MSE(X, X_hat)` | 0.5 | Synthetic + real prompts with compressed KV |
| **Retrieval Robustness** | `L_ret = -log P(passkey) + 0.5 * log P(distractor)` | 0.3 | Needle prompts (calib + held-out) |
| **General Quality** | `L_lm = standard causal LM loss` | 0.2 | General domain data |

**Training stages:**

```
Stage 1: SFT with Compression Labels (LoRA)
├── Data: 5k prompts (2k calibration-style, 2k needle, 1k general)
├── Labels: For each prompt, pre-compute:
│   - Optimal head_roles (from calibration)
│   - Target s_budget per layer (from Avellaneda-Stoikov)
│   - Ground-truth compressed KV reconstructions
├── Loss: L = L_lm + λ_comp * L_comp + λ_ret * L_ret
└── Output: LoRA adapter (32M params on 3B model)

Stage 2: RLHF/DPO with Compression Reward
├── Generate pairs: (compressed, full_FP16) for same prompt
├── Reward: R = 0.4 * logit_cosine + 0.3 * prob_retention + 0.3 * (1 - memory_ratio)
├── DPO loss on pairs
└── Output: Final LoRA + potential base model update
```

### 2.2 Compression Dataset (`training/data/compression_dataset.py`)
- **Synthetic**: Use `FallbackQwen2` to generate diverse prompts, run through `calibrate_model`, collect `(prompt, head_roles, compressed_KV, logits_FP16)`
- **Real**: Sample from OpenWebText/Code + inject needles
- **Format**: `{"input_ids": [...], "labels": [...], "target_head_roles": [...], "target_compressed_KV": {...}, "target_logits_FP16": [...]}`

### 2.3 Losses (`training/losses/`)
```python
# compression_loss.py
def compression_loss(model_logits, fp16_logits, compressed_KV, target_KV, head_roles, target_roles):
    logit_cos = cosine_similarity(model_logits, fp16_logits)
    prob_ret = model_probs[passkey] / fp16_probs[passkey]
    recon_mse = mse(compressed_KV, target_KV)
    role_loss = cross_entropy(head_roles, target_roles)
    return (1 - logit_cos) + (1 - prob_ret) + recon_mse + role_loss

# retrieval_loss.py
def retrieval_loss(model_logits, passkey_id, distractor_ids):
    target_logprob = log_softmax(model_logits)[passkey_id]
    distractor_logprob = log_softmax(model_logits)[distractor_ids].max()
    return -target_logprob + 0.5 * distractor_logprob
```

### 2.4 Training Infrastructure
- **Framework**: `trl` + `peft` + `accelerate` (or `torchrun` for multi-GPU)
- **Quantization**: LoRA on FP16/BF16 base (no 4-bit for now to preserve gradients)
- **Checkpointing**: Save every 500 steps + best by validation `logit_cosine`
- **Validation**: Run `bench_person2.py` ablation on held-out prompts every epoch

---

## Phase 3: Next.js Frontend (Week 3-4)

### 3.1 Project Setup
```bash
cd frontend
npm create next-app@latest -- --typescript --tailwind --eslint --app --src-dir --import-alias "@/*"
npm install @tanstack/react-query axios recharts lucide-react clsx tailwind-merge
```

### 3.2 Core Components

| Component | Purpose | Data Source |
|-----------|---------|-------------|
| `ChatInterface` | Streaming chat with token-by-token display | `/api/v1/generate` SSE |
| `MetricsDashboard` | Real-time latency, tokens/sec, VRAM | `/api/v1/metrics` polling |
| `HeadRouterMap` | 28×2 grid: Retrieval(🟢)/H2O(🔵)/Lexico(🟡) | Cache telemetry |
| `KVPressureGauge` | Avellaneda-Stoikov `q_t`, `r_t`, spread `δ_t` | Cache telemetry |
| `SpeculativeStats` | Acceptance rate α, tokens/verify, bandwidth savings | Matryoshka result |

### 3.3 API Layer (`src/lib/api.ts`)
```typescript
// Proxy to FastAPI with SSE support
export async function* streamChat(messages: Message[], options: GenerateOptions) {
  const res = await fetch('/api/chat', { method: 'POST', body: JSON.stringify({ messages, ...options }) });
  const reader = res.body.getReader();
  while (true) { const { done, value } = await reader.read(); if (done) break; yield parseSSE(value); }
}
```

### 3.4 Dashboard Layout
```
┌─────────────────────────────────────────────────────────────┐
│  StoikovKV Dashboard                    [VRAM: 6.2/24 GB]  │
├─────────────────────┬───────────────────────────────────────┤
│   Chat              │  Head Router (28 layers × 2 heads)    │
│   ┌─────────────┐   │  ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐  │
│   │ User: ...   │   │  │🟢│🔵│🟡│🟢│🔵│🟡│... (click for per-head)│
│   │ Asst: ...   │   │  └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘  │
│   └─────────────┘   │  KV Pressure: ████████░░ 68%          │
│   [Input...]        │  q_t=0.68  r_t=0.42  δ_t=0.15         │
├─────────────────────┼───────────────────────────────────────┤
│   Speculative: α=0.82  3.2 tok/verify  18% BW saved        │
└─────────────────────┴───────────────────────────────────────┘
```

---

## Phase 4: Evaluation & Iteration (Week 4-5)

### 4.1 Evaluation Harness
```bash
# Run full Person 1 + Person 2 benchmarks against fine-tuned model
python scripts/eval.py --model-path ./checkpoints/lora-final --output eval_finetuned.json

# Compare with baseline
python scripts/compare.py baseline.json finetuned.json --output comparison.md
```

### 4.2 Success Criteria (from existing metrics)

| Metric | Baseline (Current) | Target (Fine-tuned) | Threshold |
|--------|-------------------|---------------------|-----------|
| Hybrid CSS CosSim | 0.9629 | ≥ 0.975 | > 0.97 |
| Needle Retrieval | 100% | 100% | = 100% |
| Logit Cosine (s=16) | 0.9996 | ≥ 0.9998 | > 0.9995 |
| Fast 1-Step Latency | 0.68ms | ≤ 0.6ms | < 0.7ms |
| Matryoshka α (m=4) | 0.80 | ≥ 0.85 | > 0.83 |
| L* (Kannada) | 451 | ≤ 300 | < 400 |

### 4.3 A/B Testing Infrastructure
- `/api/v1/generate?mode=baseline|finetuned|ablation`
- Frontend toggle to compare side-by-side
- Log all generations with metrics for offline analysis

---

## Technical Decisions Needed

| Decision | Options | Recommendation |
|----------|---------|----------------|
| **Training framework** | `trl`/`peft`/`accelerate` vs custom | `trl` + `peft` (mature, LoRA built-in) |
| **Quantization** | FP16 LoRA vs 4-bit QLoRA | FP16 LoRA first (stability), QLoRA later |
| **Reward model** | Trained RM vs heuristic compression metrics | Heuristic first (fast iteration), RM if needed |
| **Data scale** | 5k synthetic vs 50k real | Start 5k synthetic, expand if gains plateau |
| **Frontend streaming** | SSE vs WebSocket | SSE (simpler, works with Next.js API routes) |
| **Session management** | In-memory vs Redis | In-memory (local, single-user dev) |
| **Calibration artifacts** | Pre-computed vs on-demand | Pre-computed (ship with repo), on-demand fallback |

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Fine-tuning degrades base model quality | Medium | High | Strong LM loss weight (0.2), early stopping on validation |
| Compression metrics don't correlate with quality | Low | Medium | Validate with Person 2 ablation every epoch |
| VRAM OOM on 3B + cache | Low | High | Dynamic `s_budget` via Avellaneda-Stoikov, gradient checkpointing |
| Next.js + FastAPI dev complexity | Medium | Low | Use `next.config.js` rewrites for API proxy |
| Calibration artifacts don't transfer to fine-tuned model | Medium | Medium | Re-calibrate after fine-tuning; include head_role prediction in loss |

---

## Immediate Next Steps (This Week)

1. **Create `server/` skeleton** with FastAPI + model loader
2. **Add `config/` YAML files** for model/cache/server/training
3. **Implement `/api/v1/generate` SSE streaming** with `StoikovKVCache`
4. **Build minimal Next.js frontend** with chat + metrics display
5. **Run end-to-end test**: prompt → FastAPI → model → stream → frontend
6. **Then**: Design compression dataset generator for fine-tuning

---

## File Creation Order (Dependencies)

```
1. config/model.yaml, cache.yaml, server.yaml
2. server/config.py
3. server/services/model_loader.py
4. server/services/cache_manager.py
5. server/routes/generate.py
6. server/main.py
7. scripts/serve.py
8. frontend/ (package.json, next.config, src/app/page.tsx, components)
9. training/data/compression_dataset.py
10. training/losses/compression_loss.py, retrieval_loss.py
11. training/trainers/sft_trainer.py
12. training/run_training.py
13. scripts/train.py
14. scripts/eval.py
```

---

**Ready to implement?** I can start with Phase 1 (server + minimal frontend) or dive into any specific component. Which phase should we begin with?PLANEOF
cat /mnt/storage/Projects/HashCode/PLAN.md
