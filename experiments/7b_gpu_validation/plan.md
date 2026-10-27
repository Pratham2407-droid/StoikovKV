# StoikovKV 7B Model Integration Plan

## 1. Environment Audit

**GPU**: NVIDIA GeForce RTX 4060 Laptop (8.2 GB VRAM)  
**Driver**: 595.99.02  
**CUDA**: 13.2 (runtime), 13.0 (PyTorch)  
**PyTorch**: 2.12.0+cu130  
**Python**: 3.12.3  

**Available VRAM for model weights**: ~7.5 GB after OS/desktop (~0.7 GB)  
**Quantization requirement**: 4-bit NF4 mandatory for 7B model (FP16 = ~14 GB, 8-bit = ~7 GB, 4-bit = ~4 GB)

## 2. Target Model: Qwen/Qwen2.5-7B-Instruct

| Parameter | Value |
|-----------|-------|
| hidden_size | 3584 |
| num_hidden_layers | 28 |
| num_attention_heads | 28 |
| num_key_value_heads | 4 |
| head_dim | 128 |
| q_per_kv | 7 |
| vocab_size | 152064 |
| rope_theta | 1,000,000 |
| max_position_embeddings | 32768 |

**Architecture compatibility**: GQA with 7 query heads per KV head. Current StoikovKV supports arbitrary `q_per_kv` (tested with 6 for 1.5B). The `head_dim=128` matches 1.5B, so dictionary dimensions are compatible.

## 3. Current Implementation Status

### Working (validated by tests):
- All 11 Person 1 pipeline tests pass (CPU)
- All 5 Person 2 live inference tests pass (GPU with fallback model)
- Context drift benchmark executes on GPU (fallback model)
- Calibration pipeline produces artifacts for 1.5B (28 layers, 2 KV heads)

### Known gaps for 7B:
1. **No 4-bit quantization support** - `bitsandbytes` not installed; `load_model_and_tokenizer` doesn't support `load_in_4bit`
2. **Calibrated artifacts mismatch** - Existing `calibrated_artifacts.pt` is for 1.5B (2 KV heads), not 7B (4 KV heads)
3. **Context drift retrieval fails** - Both full_fp16 and hybrid fail on fallback model (predicts wrong token)
4. **Real model loading untested** - `use_real_model=True` path uses `device_map="auto"` with `max_memory={0: "5.5GB"}` which may not work for 7B

## 4. Experiment Sequence

### Phase 0: Infrastructure (PREREQUISITE)
- [ ] Install `bitsandbytes` with CUDA 12.x wheel
- [ ] Verify 4-bit loading works for a small model
- [ ] Extend `load_model_and_tokenizer` to support `load_in_4bit`, `bnb_4bit_compute_dtype`, `bnb_4bit_quant_type`

### Phase 1: Baseline Establishment (NO StoikovKV)
- [ ] Load Qwen2.5-7B-Instruct 4-bit NF4 on GPU
- [ ] Verify weights on GPU, not CPU
- [ ] Run greedy generation on fixed prompts (512, 1024, 2048 context)
- [ ] Measure: peak VRAM, prefill latency, decode latency, tokens/sec
- [ ] Save baseline outputs as reference
- [ ] Verify standard Transformers cache works (prefill + incremental decode)

### Phase 2: Context Drift Diagnosis
- [ ] Run context drift test with real 7B model (standard cache)
- [ ] Run context drift test with real 7B model (StoikovKV cache, compression disabled)
- [ ] Compare CPU vs GPU logits, cache shapes, positions
- [ ] Identify first divergence point

### Phase 3: StoikovKV Integration (Incremental)
- [ ] **Stage A**: Standard GPU cache, no StoikovKV modifications
- [ ] **Stage B**: StoikovKV cache with `policy_mode="full_fp16"` (compression disabled)
- [ ] **Stage C**: One-step sparse encoder + dictionary reconstruction
- [ ] **Stage D**: Per-head routing + retrieval-protected FP16 heads
- [ ] **Stage E**: H₂O eviction + context-drift handling
- [ ] **Stage F**: Full hybrid configuration
- [ ] At each stage: run correctness tests, record active components

### Phase 4: Controlled Experiments (Quantized Weights)
- [ ] **Experiment A**: 4-bit NF4 weights + Standard KV cache (baseline)
- [ ] **Experiment B**: 4-bit NF4 weights + StoikovKV Hybrid (s=16)
- [ ] Fixed prompts, decoding config, sequence lengths
- [ ] Measure memory, latency, retrieval, generation quality

### Phase 5: Evaluation & Reporting
- [ ] Context-drift test (CPU + GPU)
- [ ] Retrieval/needle tests (multiple positions)
- [ ] Generation quality (instruction following, long-context, multi-turn, summarization)
- [ ] Numerical correctness (logit diffs, top-1 agreement)
- [ ] Memory breakdown (weights, KV cache, dictionary, overhead)
- [ ] Latency breakdown (prefill, decode, encoder, decoder)
- [ ] Reconstruction metrics (cosine, MSE, sparsity)
- [ ] Produce `results.json`, `report.md`, `run_manifest.json`

## 5. Test Configurations

### Context Lengths
- 512 tokens (baseline)
- 1024 tokens (standard)
- 2048 tokens (target, subject to VRAM)

### Decoding
- Greedy (temperature=0.0, top_k=1)
- Fixed seed (42)
- Deterministic settings where supported

### Prompts
- Use existing `CALIBRATION_PROMPTS`, `NEEDLE_PROMPTS`, `HELDOUT_EVAL_PROMPTS`
- Add held-out retrieval prompts not used in calibration
- Add instruction-following, summarization, multi-turn prompts

## 6. Pass Criteria (Non-Negotiable)

1. ✅ Standard-cache real model generates on GPU
2. ✅ StoikovKV GPU path executes intended compression logic
3. ✅ Context-drift test passes under documented protocol
4. ✅ Retrieval/generation tests execute on real model outputs
5. ✅ Memory/latency metrics measured and reproducible
6. ✅ Standard-cache vs StoikovKV uses identical weights/quantization/inputs
7. ✅ Run manifest and raw measurements complete
8. ✅ Failures/limitations explicitly disclosed

## 7. Artifact Locations

```
/mnt/storage/Projects/HashCode/experiments/7b_gpu_validation/
├── plan.md                 # This file
├── baseline.json           # Standard-cache metrics
├── results.json            # All experiment metrics
├── report.md               # Final report
├── run_manifest.json       # Git hash, env, commands, hashes
├── logs/                   # Raw execution logs
└── outputs/                # Model outputs for inspection
```

## 8. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| 7B 4-bit doesn't fit in 8GB | Try 8-bit, reduce context to 512, or use CPU offload |
| `bitsandbytes` CUDA 13 incompatibility | Install from source or use `torch.compile` with `torch.ao.quantization` |
| Custom cache incompatible with quantized model | Diagnose at Stage A/B; fallback to Transformers cache API |
| Context drift root cause unclear | Bisect: CPU fallback → GPU standard → GPU StoikovKV disabled → GPU StoikovKV enabled |
| Retrieval fails on real model | Verify head role assignment produces ≥1 ROLE_RETRIEVAL_FP16 head per layer |

## 9. Integrity Safeguards

- Record `git commit`, `git status`, file hashes BEFORE final benchmark
- Compare repository state AFTER final benchmark
- Flag any changes to test definitions, metric formulas, reference outputs
- Preserve `data.md` and existing benchmarks unchanged
- New results in versioned experiment directory only

## 10. Next Immediate Action

**Install `bitsandbytes` and verify 4-bit loading works.**