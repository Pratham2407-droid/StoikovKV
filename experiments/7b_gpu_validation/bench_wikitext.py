"""Production-level WikiText-103 evaluation for StoikovKV fallback model."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stoikov_kv.cache import StoikovKVCache, load_or_create_artifacts
from stoikov_kv.calibration import (
    FallbackQwen2Config, FallbackQwen2ForCausalLM, FallbackQwen2Tokenizer,
    load_model_and_tokenizer
)

def load_wikitext_test(num_samples: int = 100, max_tokens: int = 2048) -> List[str]:
    """Load WikiText-103 test samples from local parquet."""
    print(f"Loading WikiText-103 test split from local parquet...")
    parquet_path = Path("/mnt/storage/Projects/StoikovKV_Hackathon/datasets/wikitext-103/wikitext-103-raw-v1/test-00000-of-00001.parquet")
    df = pd.read_parquet(parquet_path)
    
    texts = []
    for text in df["text"].tolist():
        if text.strip():
            # Truncate to max_tokens * 4 chars (rough estimate)
            if len(text) > max_tokens * 4:
                text = text[:max_tokens * 4]
            texts.append(text)
        if len(texts) >= num_samples:
            break
    return texts[:num_samples]

@torch.inference_mode()
def evaluate_policy_wikitext(
    model: Any, tokenizer: Any, artifacts: Dict[str, Any], 
    policy_mode: str, texts: List[str], max_tokens: int = 2048
) -> Dict[str, Any]:
    """Evaluate policy on WikiText samples."""
    device = next(model.parameters()).device
    
    # Configure cache
    if policy_mode == "full_fp16":
        cache = StoikovKVCache(model=model, artifacts=artifacts, n_b=128, keep_ratio=0.20, s_budget=16, policy_mode="full_fp16", enable_hmm_decay=False)
    elif policy_mode == "hybrid":
        cache = StoikovKVCache(model=model, artifacts=artifacts, n_b=128, keep_ratio=0.10, s_budget=16, policy_mode="hybrid", enable_hmm_decay=True, h2o_context_window=1)
    elif policy_mode == "standalone_h2o":
        cache = StoikovKVCache(model=model, artifacts=artifacts, n_b=128, keep_ratio=0.10, s_budget=16, policy_mode="standalone_h2o", enable_hmm_decay=False)
    elif policy_mode == "standalone_lexico":
        cache = StoikovKVCache(model=model, artifacts=artifacts, n_b=128, keep_ratio=0.20, s_budget=16, policy_mode="standalone_lexico", enable_hmm_decay=False)
    else:
        raise ValueError(f"Unknown policy: {policy_mode}")
    
    total_tokens = 0
    total_time = 0.0
    peak_mem = 0.0
    samples_processed = 0
    
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    start = time.perf_counter()
    
    for text in tqdm(texts, desc=f"  {policy_mode}", leave=False):
        input_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens).input_ids.to(device)
        
        # Process in chunks
        chunk_size = 128
        for start_idx in range(0, input_ids.shape[1], chunk_size):
            end_idx = min(start_idx + chunk_size, input_ids.shape[1])
            chunk = input_ids[:, start_idx:end_idx]
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            step_start = time.perf_counter()
            
            _ = model(input_ids=chunk, past_key_values=cache, use_cache=True)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            step_time = time.perf_counter() - step_start
            
            total_time += step_time
            total_tokens += chunk.shape[1]
            
            if device.type == "cuda":
                peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / 1e9)
    
    total_elapsed = time.perf_counter() - start
    telemetry = cache.get_telemetry_summary()
    
    return {
        "policy": policy_mode,
        "total_tokens_processed": total_tokens,
        "total_time_s": total_elapsed,
        "tokens_per_second": total_tokens / total_elapsed if total_elapsed > 0 else 0,
        "peak_memory_gb": peak_mem,
        "fp16_bytes": telemetry.get("fp16_bytes", 0),
        "compressed_bytes": telemetry.get("compressed_bytes", 0),
        "memory_ratio_vs_fp16": telemetry.get("memory_ratio_vs_fp16", 1.0),
        "memory_savings_pct": telemetry.get("memory_savings_pct", 0.0),
        "bytes_read_last_step": telemetry.get("bytes_read_last_step", 0),
        "cumulative_bytes_read": telemetry.get("cumulative_bytes_read", 0),
        "per_tier_token_counts": telemetry.get("per_tier_token_counts", {}),
        "hmm_last_beta_t": telemetry.get("hmm_last_beta_t", 1.0),
        "hmm_last_z_score": telemetry.get("hmm_last_z_score", 0.0),
        "hmm_shift_events": telemetry.get("hmm_shift_events", 0),
    }

def evaluate_perplexity(model: Any, tokenizer: Any, artifacts: Dict[str, Any], 
                       policy_mode: str, texts: List[str], max_tokens: int = 2048) -> float:
    """Evaluate perplexity with given policy."""
    device = next(model.parameters()).device
    
    if policy_mode == "full_fp16":
        cache = StoikovKVCache(model=model, artifacts=artifacts, n_b=128, keep_ratio=0.20, s_budget=16, policy_mode="full_fp16", enable_hmm_decay=False)
    elif policy_mode == "hybrid":
        cache = StoikovKVCache(model=model, artifacts=artifacts, n_b=128, keep_ratio=0.10, s_budget=16, policy_mode="hybrid", enable_hmm_decay=True, h2o_context_window=1)
    elif policy_mode == "standalone_h2o":
        cache = StoikovKVCache(model=model, artifacts=artifacts, n_b=128, keep_ratio=0.10, s_budget=16, policy_mode="standalone_h2o", enable_hmm_decay=False)
    elif policy_mode == "standalone_lexico":
        cache = StoikovKVCache(model=model, artifacts=artifacts, n_b=128, keep_ratio=0.20, s_budget=16, policy_mode="standalone_lexico", enable_hmm_decay=False)
    else:
        raise ValueError(f"Unknown policy: {policy_mode}")
    
    total_nll = 0.0
    total_tokens = 0
    
    for text in tqdm(texts, desc=f"  PPL {policy_mode}", leave=False):
        input_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens).input_ids.to(device)
        
        if input_ids.shape[1] < 2:
            continue
            
        # Process all but last token for next-token prediction
        with torch.inference_mode():
            for i in range(input_ids.shape[1] - 1):
                chunk = input_ids[:, i:i+1]
                out = model(input_ids=chunk, past_key_values=cache, use_cache=True)
                
                logits = out.logits[0, -1].float()
                target = input_ids[0, i+1].item()
                
                log_probs = torch.log_softmax(logits, dim=-1)
                nll = -log_probs[target].item()
                
                total_nll += nll
                total_tokens += 1
    
    if total_tokens == 0:
        return float('inf')
    
    return torch.exp(torch.tensor(total_nll / total_tokens)).item()

def main():
    print("=" * 80)
    print("StoikovKV WikiText-103 Production Evaluation (Fallback Model)")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load fallback model with 4 KV heads
    print("\nLoading 4-KV-head fallback model...")
    cfg = FallbackQwen2Config(
        num_hidden_layers=12,
        num_attention_heads=24,
        num_key_value_heads=4,
        hidden_size=1536,
        head_dim=64,
        vocab_size=32768,
        rope_theta=1000000.0,
        max_position_embeddings=32768,
    )
    model = FallbackQwen2ForCausalLM(cfg, device=device, dtype=dtype, seed=42)
    tokenizer = FallbackQwen2Tokenizer(vocab_size=cfg.vocab_size)
    model.eval()
    
    print(f"Model: {model.config._name_or_path}")
    print(f"Layers: {model.config.num_hidden_layers}, Q heads: {model.config.num_attention_heads}, KV heads: {model.config.num_key_value_heads}, head_dim: {model.config.head_dim}")
    
    # Load calibration artifacts
    artifacts_path = ROOT / "calibrated_artifacts_7b.pt"
    if artifacts_path.exists():
        artifacts = load_or_create_artifacts(model, artifacts_path=str(artifacts_path), tokenizer=tokenizer)
        print(f"Artifacts loaded: U_Q={tuple(artifacts['U_Q'].shape)}")
    else:
        print("No calibration artifacts found!")
        return
    
    # Load WikiText test data
    print("\nLoading WikiText-103 test data...")
    wikitext_samples = load_wikitext_test(num_samples=50, max_tokens=2048)
    print(f"Loaded {len(wikitext_samples)} samples")
    
    # Run evaluation
    policies = ["full_fp16", "hybrid", "standalone_h2o", "standalone_lexico"]
    
    results = {
        "model": "Qwen2.5-1.5B-Instruct (Standalone GQA, Fallback 12L/4KV/h=64)",
        "dataset": "WikiText-103 test",
        "num_samples": len(wikitext_samples),
        "max_tokens_per_sample": 2048,
        "device": device,
        "policies": {},
    }
    
    for policy in policies:
        print(f"\n{'='*80}")
        print(f"Evaluating policy: {policy}")
        print(f"{'='*80}")
        
        # Throughput & memory
        perf = evaluate_policy_wikitext(model, tokenizer, artifacts, policy, wikitext_samples)
        
        # Perplexity (on subset for speed)
        ppl_subset = wikitext_samples[:10]  # 10 samples for PPL
        ppl = evaluate_perplexity(model, tokenizer, artifacts, policy, ppl_subset)
        
        results["policies"][policy] = {
            **perf,
            "perplexity": ppl,
        }
        
        print(f"  Tokens/s: {perf['tokens_per_second']:.2f}")
        print(f"  Perplexity: {ppl:.4f}")
        print(f"  Peak VRAM: {perf['peak_memory_gb']:.2f} GB")
        print(f"  Memory ratio: {perf['memory_ratio_vs_fp16']*100:.1f}%")
        print(f"  Savings: {perf['memory_savings_pct']:.1f}%")
    
    # Save results
    output_path = Path(__file__).parent / "wikitext_results.json"
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\n\nResults saved to: {output_path}")
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("WIKITEXT-103 COMPREHENSIVE COMPARISON")
    print("=" * 80)
    print(f"{'Policy':<20} {'Tok/s':>8} {'PPL':>8} {'VRAM (GB)':>10} {'Mem %':>8} {'Savings %':>10} {'HMM β_t':>8} {'Shifts':>8}")
    print("-" * 80)
    for policy, r in results["policies"].items():
        print(f"{policy:<20} {r['tokens_per_second']:>8.2f} {r['perplexity']:>8.4f} "
              f"{r['peak_memory_gb']:>10.2f} {r['memory_ratio_vs_fp16']*100:>7.1f}% "
              f"{r['memory_savings_pct']:>9.1f}% {r['hmm_last_beta_t']:>8.2f} {r['hmm_shift_events']:>8}")

if __name__ == "__main__":
    main()
