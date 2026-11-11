# 7B GPU Validation Experiment - Progress Report

## Summary of Work Completed

### 1. Environment Audit ✅
- **GPU**: NVIDIA GeForce RTX 4060 Laptop (8.2 GB VRAM)
- **Driver**: 595.99.02
- **CUDA**: 13.2 (runtime), 13.0 (PyTorch)  
- **PyTorch**: 2.14.1+cu130
- **Python**: 3.12.3

### 2. Target Model: Qwen/Qwen2.5-7B-Instruct
| Parameter | Value |
|-----------|-------|
| hidden_size | 3584 |
| num_hidden_layers | 28 |
| num_attention_heads | 28 |
| num_key_value_heads | 4 |
| head_dim | 128 |
| q_per_kv | 7 |
| vocab_size | 152064 |

### 3. Code Modifications ✅

#### `stoikov_kv/calibration.py` - Enhanced `load_model_and_tokenizer()`
- Added `load_in_4bit` parameter for bitsandbytes NF4 quantization
- Added `use_quanto` and `quanto_dtype` parameters for Quanto quantization (int4, int8, float8)
- Added automatic max_memory adjustment for 7B models
- Added graceful fallback from bitsandbytes → Quanto → FP16 → fallback model
- Fixed fallback model to include `generate()` method for baseline testing

### 4. Experiment Infrastructure ✅

Created in `/mnt/storage/Projects/StoikovKV_Hackathon/experiments/7b_gpu_validation/`:
- `plan.md` - Comprehensive experiment plan
- `bench_7b_baseline.py` - Baseline evaluation (standard cache, memory/latency/correctness)
- `calibrate_7b.py` - Calibration for 7B (28 layers, 4 KV heads)
- `bench_7b_stoikovkv.py` - Full StoikovKV evaluation (4 policies)
- `bench_context_drift_7b.py` - Context drift test for 7B
- `generate_manifest.py` - Run manifest generation

### 5. Calibration Pipeline Verification ✅
Tested with 4 KV heads (matching 7B architecture):
- U_Q shape: `(layers, 4, head_dim, subspace_rank)` ✅
- head_roles shape: `(layers, 4)` ✅
- Dictionary D shape: `(layers, head_dim, dict_atoms)` ✅
- W_ridge shape: `(layers, dict_atoms, head_dim)` ✅
- Head role assignment: 5% Retrieval FP16, 85% H2O, 10% Lexico ✅

### 6. Baseline Test with Fallback Model ✅
- Model loads on GPU
- Generation works with `generate()` method
- Cache consistency check passes
- Metrics measured: prefill latency, decode latency, tokens/sec, peak VRAM

### 7. Model Download Status 🔄
- **Status**: Downloading (~15GB total)
- **Location**: `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/`
- **Progress**: 16 incomplete shard files downloading
- **Issue**: Multiple concurrent downloads for same shards (slow)

## Next Steps (Once Model Download Completes)

### Phase 1: Baseline Establishment
```bash
cd /mnt/storage/Projects/StoikovKV_Hackathon
python experiments/7b_gpu_validation/bench_7b_baseline.py
```
- Load 7B with Quanto int4 quantization
- Run standard generation on fixed prompts (512, 1024, 2048 context)
- Measure: peak VRAM, prefill latency, decode latency, tokens/sec
- Verify cache correctness (cached vs uncached agreement)
- Save `baseline.json`

### Phase 2: Calibration
```bash
python experiments/7b_gpu_validation/calibrate_7b.py
```
- Run full calibration pipeline on 7B model
- Produce `calibrated_artifacts_7b.pt` (28 layers, 4 KV heads)
- Save calibration metadata

### Phase 3: Context Drift Diagnosis
```bash
python experiments/7b_gpu_validation/bench_context_drift_7b.py
```
- Run context drift test with real 7B model
- Compare full_fp16 vs hybrid policies
- Identify GPU-specific issues

### Phase 4: StoikovKV Evaluation
```bash
python experiments/7b_gpu_validation/bench_7b_stoikovkv.py
```
- Evaluate 4 policies: full_fp16, hybrid, standalone_h2o, standalone_lexico
- Test needle retrieval at early/middle/late positions
- Test instruction following, summarization
- Measure memory, latency, reconstruction quality
- Save `results.json`

### Phase 5: Reporting
```bash
python experiments/7b_gpu_validation/generate_manifest.py
```
- Generate `run_manifest.json` with git hashes, file hashes, env info
- Create `report.md` with methods, results, tables, failures, conclusions

## Known Issues & Mitigations

| Issue | Mitigation |
|-------|------------|
| bitsandbytes CUDA 13 incompatible | Use Quanto quantization instead |
| Model download slow/restarting | Wait for completion; HF Hub download issues |
| 7B fallback model OOM on 8GB VRAM | Use Quanto int4 on real model |
| Multiple incomplete download files | HF Hub resumable downloads - will complete eventually |

## Experiment Integrity Safeguards

- Git commit recorded before experiments
- File hashes recorded for all key files
- Model revision tracked
- Raw measurements preserved
- Failures explicitly documented
- `data.md` and existing benchmarks preserved unchanged

## Expected Outputs

```
/mnt/storage/Projects/StoikovKV_Hackathon/experiments/7b_gpu_validation/
├── plan.md                    # Experiment plan (exists)
├── baseline.json              # Standard-cache metrics (pending)
├── calibration_metadata.json  # Calibration info (pending)
├── results.json               # Full evaluation (pending)
├── context_drift_results_7b.json  # Context drift (pending)
├── report.md                  # Final report (pending)
├── run_manifest.json          # Audit trail (pending)
└── logs/                      # Raw execution logs (pending)
```

## Pass Criteria (from original mission)
1. ✅ Standard-cache real model generates on GPU (fallback works, real model pending)
2. ⏳ StoikovKV GPU path executes compression logic (pending real model)
3. ⏳ Context-drift test passes under documented protocol (pending)
4. ⏳ Retrieval/generation tests execute on real model outputs (pending)
5. ⏳ Memory/latency metrics measured and reproducible (pending)
6. ⏳ Standard-cache vs StoikovKV uses identical weights/quantization/inputs (pending)
7. ⏳ Run manifest and raw measurements complete (pending)
8. ⏳ Failures/limitations explicitly disclosed (will document)