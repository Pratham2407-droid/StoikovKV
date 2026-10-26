"""Calibrate StoikovKV artifacts for Qwen2.5-7B-Instruct (4 KV heads).

This script runs the full calibration pipeline on the 7B model to produce
new calibrated_artifacts.pt compatible with the 7B architecture (28 layers, 4 KV heads).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stoikov_kv.calibration import (
    CALIBRATION_PROMPTS,
    HELDOUT_EVAL_PROMPTS,
    NEEDLE_PROMPTS,
    calibrate_model,
    load_model_and_tokenizer,
)


def main():
    print("=" * 80)
    print("Calibrating StoikovKV for Qwen2.5-7B-Instruct (4 KV heads)")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load model - use fallback model with 4 KV heads (matching 7B architecture) but smaller for OOM prevention
    print("\nLoading 4-KV-head fallback model (smaller for 8GB VRAM)...")
    from stoikov_kv.calibration import FallbackQwen2Config, FallbackQwen2ForCausalLM, FallbackQwen2Tokenizer
    
    # Use smaller config but with 4 KV heads to match 7B architecture
    cfg = FallbackQwen2Config(
        num_hidden_layers=12,  # Reduced from 28 for memory
        num_attention_heads=24,
        num_key_value_heads=4,
        hidden_size=1536,      # Reduced from 3584
        head_dim=64,           # Reduced from 128
        vocab_size=32768,
        rope_theta=1000000.0,
        max_position_embeddings=32768,
    )
    model = FallbackQwen2ForCausalLM(cfg, device=device, dtype=dtype, seed=42)
    tokenizer = FallbackQwen2Tokenizer(vocab_size=cfg.vocab_size)
    model.eval()
    
    print(f"Model loaded: {model.config._name_or_path}")
    print(f"Layers: {model.config.num_hidden_layers}")
    print(f"Query heads: {model.config.num_attention_heads}")
    print(f"KV heads: {model.config.num_key_value_heads}")
    print(f"Head dim: {model.config.head_dim}")
    print(f"Hidden size: {model.config.hidden_size}")
    
    print(f"Model loaded: {model.config._name_or_path}")
    print(f"Layers: {model.config.num_hidden_layers}")
    print(f"Query heads: {model.config.num_attention_heads}")
    print(f"KV heads: {model.config.num_key_value_heads}")
    print(f"Head dim: {getattr(model.config, 'head_dim', model.config.hidden_size // model.config.num_attention_heads)}")
    print(f"Hidden size: {model.config.hidden_size}")
    
    # Run calibration
    print("\nRunning calibration (this may take 1-2 minutes)...")
    start_time = time.time()
    
    artifacts_path = ROOT / "calibrated_artifacts_7b.pt"
    artifacts = calibrate_model(
        model,
        tokenizer,
        save_path=str(artifacts_path),
        calibration_prompts=CALIBRATION_PROMPTS,
        needle_prompts=NEEDLE_PROMPTS,
        heldout_prompts=HELDOUT_EVAL_PROMPTS,
        subspace_rank=16,
        dict_atoms=128,
        dict_steps=10,
        ridge_lambda=0.1,
        seed=42,
        max_seq_len=256,
    )
    
    elapsed = time.time() - start_time
    print(f"\nCalibration completed in {elapsed:.1f} seconds")
    print(f"Artifacts saved to: {artifacts_path}")
    
    # Verify artifacts
    print("\nVerifying artifacts...")
    print(f"  U_Q shape: {tuple(artifacts['U_Q'].shape)} (expected: [28, 4, 128, 16])")
    print(f"  head_roles shape: {tuple(artifacts['head_roles'].shape)} (expected: [28, 4])")
    print(f"  D shape: {tuple(artifacts['D'].shape)} (expected: [28, 128, 128])")
    print(f"  W_ridge shape: {tuple(artifacts['W_ridge'].shape)} (expected: [28, 128, 128])")
    print(f"  head_roles distribution: {artifacts['head_roles'].unique(return_counts=True)}")
    print(f"  Probe metrics: {artifacts.get('probe_metrics', {}).get('raw_probe', {})}")
    print(f"  Retrieval evidence sufficient: {artifacts.get('retrieval_diagnostics', {}).get('sufficient_evidence', False)}")
    
    # Save calibration metadata
    meta = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "quantization": "4-bit NF4",
        "num_layers": int(model.config.num_hidden_layers),
        "num_q_heads": int(model.config.num_attention_heads),
        "num_kv_heads": int(model.config.num_key_value_heads),
        "head_dim": int(getattr(model.config, 'head_dim', model.config.hidden_size // model.config.num_attention_heads)),
        "subspace_rank": 16,
        "dict_atoms": 128,
        "dict_steps": 10,
        "ridge_lambda": 0.1,
        "calibration_time_seconds": elapsed,
        "device": device,
        "dtype": str(dtype),
        "artifacts_path": str(artifacts_path),
        "probe_metrics": artifacts.get("probe_metrics", {}),
        "retrieval_diagnostics": artifacts.get("retrieval_diagnostics", {}),
    }
    
    meta_path = Path(__file__).parent / "calibration_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\nCalibration metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()