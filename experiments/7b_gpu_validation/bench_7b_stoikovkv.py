"""StoikovKV evaluation for Qwen2.5-7B-Instruct with 4-bit NF4 quantization.

This script evaluates the full StoikovKV hybrid cache on the 7B model and compares
against the standard cache baseline.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stoikov_kv.cache import StoikovKVCache, load_or_create_artifacts
from stoikov_kv.calibration import load_model_and_tokenizer
from stoikov_kv.elastic_lexico import ElasticLexicoCodec


# Test prompts for evaluation
EVAL_PROMPTS: List[Dict[str, Any]] = [
    {
        "name": "needle_retrieval_early",
        "prompt": (
            "System Audit Log: Cluster node alpha-07 initialized telemetry daemons across 8 PCIe lanes "
            "on port 71039. The secret passkey is 94812. Please remember this credential for final verification. "
            "Subsequent routine checks confirmed that NVMe queue depths remained below 32 IOPS, "
            "network tail latency stayed within 1.4ms at the 99th percentile, and thermal throttling "
            "was not observed on any GPU compute die during the 60-minute soak test. "
            + "Additional filler text to extend context. " * 50
            + "\nQuestion: What is the secret passkey? Answer: The secret passkey is"
        ),
        "target": "94812",
        "max_new_tokens": 16,
        "retrieval_position": "early",
    },
    {
        "name": "needle_retrieval_middle",
        "prompt": (
            "Filler text at the beginning. " * 30
            + "System Audit Log: Cluster node alpha-07 initialized telemetry daemons across 8 PCIe lanes "
            "on port 71039. The secret passkey is 94812. Please remember this credential for final verification. "
            "Subsequent routine checks confirmed that NVMe queue depths remained below 32 IOPS, "
            "network tail latency stayed within 1.4ms at the 99th percentile, and thermal throttling "
            "was not observed on any GPU compute die during the 60-minute soak test. "
            + "More filler text in the middle. " * 30
            + "\nQuestion: What is the secret passkey? Answer: The secret passkey is"
        ),
        "target": "94812",
        "max_new_tokens": 16,
        "retrieval_position": "middle",
    },
    {
        "name": "needle_retrieval_late",
        "prompt": (
            "Filler text at the beginning. " * 80
            + "System Audit Log: Cluster node alpha-07 initialized telemetry daemons across 8 PCIe lanes "
            "on port 71039. The secret passkey is 94812. Please remember this credential for final verification. "
            "Subsequent routine checks confirmed that NVMe queue depths remained below 32 IOPS, "
            "network tail latency stayed within 1.4ms at the 99th percentile, and thermal throttling "
            "was not observed on any GPU compute die during the 60-minute soak test. "
            + "\nQuestion: What is the secret passkey? Answer: The secret passkey is"
        ),
        "target": "94812",
        "max_new_tokens": 16,
        "retrieval_position": "late",
    },
    {
        "name": "instruction_following",
        "prompt": "Write a concise explanation of how grouped-query attention reduces KV cache memory in transformer models. Include the formula for memory reduction.",
        "target": None,
        "max_new_tokens": 128,
        "retrieval_position": "none",
    },
    {
        "name": "long_context_summary",
        "prompt": (
            "Summarize the following technical discussion in 3 sentences:\n\n"
            + "The transformer architecture uses self-attention to model relationships between tokens. "
            + "In autoregressive generation, the key-value cache stores historical activations. "
            + "Grouped-query attention shares KV heads across query heads to reduce memory. "
            + "StoikovKV further compresses the KV cache using hybrid eviction and sparse coding. "
            + "This enables longer context lengths on memory-constrained GPUs. " * 30
            + "\n\nSummary:"
        ),
        "target": None,
        "max_new_tokens": 64,
        "retrieval_position": "none",
    },
]


@torch.inference_mode()
def evaluate_policy(
    model: torch.nn.Module,
    tokenizer: Any,
    artifacts: Dict[str, Any],
    policy_mode: str,
    prompt: str,
    target: str = None,
    max_new_tokens: int = 128,
) -> Dict[str, Any]:
    """Evaluate a specific StoikovKV policy mode on a prompt."""
    device = next(model.parameters()).device
    
    # Configure cache based on policy
    if policy_mode == "full_fp16":
        cache = StoikovKVCache(
            model=model,
            artifacts=artifacts,
            n_b=128,
            keep_ratio=0.20,
            s_budget=16,
            policy_mode="full_fp16",
            enable_hmm_decay=False,
        )
    elif policy_mode == "hybrid":
        cache = StoikovKVCache(
            model=model,
            artifacts=artifacts,
            n_b=128,
            keep_ratio=0.10,
            s_budget=16,
            policy_mode="hybrid",
            enable_hmm_decay=True,
            h2o_context_window=1,
        )
    elif policy_mode == "standalone_h2o":
        cache = StoikovKVCache(
            model=model,
            artifacts=artifacts,
            n_b=128,
            keep_ratio=0.10,
            s_budget=16,
            policy_mode="standalone_h2o",
            enable_hmm_decay=False,
        )
    elif policy_mode == "standalone_lexico":
        cache = StoikovKVCache(
            model=model,
            artifacts=artifacts,
            n_b=128,
            keep_ratio=0.20,
            s_budget=16,
            policy_mode="standalone_lexico",
            enable_hmm_decay=False,
        )
    else:
        raise ValueError(f"Unknown policy_mode: {policy_mode}")
    
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    prompt_len = input_ids.shape[1]
    
    # Prefill in chunks to avoid OOM
    chunk_size = 128
    for start in range(0, prompt_len, chunk_size):
        end = min(start + chunk_size, prompt_len)
        chunk_ids = input_ids[:, start:end]
        _ = model(input_ids=chunk_ids, past_key_values=cache, use_cache=True)
    
    # Generate
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    gen_start = time.perf_counter()
    current_ids = input_ids[:, -1:]  # Last token for generation
    past_key_values = cache
    
    generated_ids = []
    for _ in range(max_new_tokens):
        out = model(input_ids=current_ids, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        generated_ids.append(next_token.item())
        current_ids = next_token
        
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    gen_end = time.perf_counter()
    
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0
    
    # Get telemetry
    telemetry = cache.get_telemetry_summary()
    
    result = {
        "policy": policy_mode,
        "prompt_tokens": prompt_len,
        "generated_tokens": len(generated_ids),
        "generated_text": generated_text,
        "generation_time_s": gen_end - gen_start,
        "tokens_per_second": len(generated_ids) / (gen_end - gen_start) if gen_end > gen_start else 0,
        "peak_memory_gb": peak_memory_gb,
        **telemetry,
    }
    
    # Check retrieval if target provided
    if target is not None:
        target_id = int(tokenizer(target, return_tensors="pt")["input_ids"][0, -1])
        # Get logits for the last generated token (or last token of prompt if no generation)
        out = model(input_ids=current_ids, past_key_values=past_key_values, use_cache=True)
        logits = out.logits[0, -1].float()
        probabilities = torch.softmax(logits, dim=-1)
        rank = int((logits > logits[target_id]).sum().item()) + 1
        top_id = int(torch.argmax(logits).item())
        
        result.update({
            "target": target,
            "target_token_id": target_id,
            "predicted_token_id": top_id,
            "retrieval_top1": top_id == target_id,
            "target_probability": float(probabilities[target_id].item()),
            "target_rank": rank,
        })
    
    return result


@torch.inference_mode()
def evaluate_codec_reconstruction(
    model: torch.nn.Module,
    artifacts: Dict[str, Any],
    layer_idx: int = 0,
    num_vectors: int = 256,
) -> Dict[str, Any]:
    """Evaluate the sparse codec reconstruction quality on real KV vectors."""
    device = next(model.parameters()).device
    codec = ElasticLexicoCodec.from_artifacts(artifacts)
    
    # Capture real K/V vectors from the model
    arch = model.config
    num_kv_heads = arch.num_key_value_heads
    head_dim = getattr(arch, 'head_dim', arch.hidden_size // arch.num_attention_heads)
    
    # Use a simple prompt to get KV states
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    prompt = "The quick brown fox jumps over the lazy dog. " * 20
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    captured_k = []
    captured_v = []
    
    def k_hook(_m, _inp, out):
        t = out[0] if isinstance(out, tuple) else out
        bsz, slen, _ = t.shape
        k_raw = t.detach().view(bsz, slen, num_kv_heads, head_dim).transpose(1, 2)
        # Apply RoPE
        from stoikov_kv.calibration import _compute_rope_cos_sin, _apply_rotary_pos_emb
        cos, sin = _compute_rope_cos_sin(slen, head_dim, device=k_raw.device, dtype=torch.float32, rope_theta=1000000.0)
        k_rope = _apply_rotary_pos_emb(k_raw.to(torch.float32), cos, sin).to(torch.float16)
        captured_k.append(k_rope.reshape(-1, head_dim).detach())
    
    def v_hook(_m, _inp, out):
        t = out[0] if isinstance(out, tuple) else out
        bsz, slen, _ = t.shape
        v_raw = t.detach().view(bsz, slen, num_kv_heads, head_dim).transpose(1, 2)
        captured_v.append(v_raw.reshape(-1, head_dim).detach())
    
    handles = []
    for l in range(min(3, model.config.num_hidden_layers)):
        attn = model.model.layers[l].self_attn
        handles.append(attn.k_proj.register_forward_hook(k_hook))
        handles.append(attn.v_proj.register_forward_hook(v_hook))
    
    try:
        with torch.no_grad():
            _ = model(input_ids=input_ids, use_cache=False, output_attentions=False)
    finally:
        for h in handles:
            h.remove()
    
    if not captured_k:
        return {"error": "No KV vectors captured"}
    
    all_k = torch.cat(captured_k, dim=0)[:num_vectors]
    all_v = torch.cat(captured_v, dim=0)[:num_vectors]
    
    results = {}
    for s in [4, 8, 12, 16]:
        # Encode
        k_idx, k_val = codec.encode(all_k, layer_idx=layer_idx, s=s, lambda1=0.04)
        v_idx, v_val = codec.encode(all_v, layer_idx=layer_idx, s=s, lambda1=0.04)
        
        # Decode
        k_rec = codec.decode(k_idx, k_val, layer_idx=layer_idx)
        v_rec = codec.decode(v_idx, v_val, layer_idx=layer_idx)
        
        # Evaluate
        k_metrics = codec.evaluate_reconstruction(all_k, k_rec)
        v_metrics = codec.evaluate_reconstruction(all_v, v_rec)
        
        results[f"s={s}"] = {
            "bytes_per_vector": k_val.bytes_per_vector * 2,  # K + V
            "k_cosine": k_metrics["cosine_similarity"],
            "k_rel_mse": k_metrics["relative_mse"],
            "v_cosine": v_metrics["cosine_similarity"],
            "v_rel_mse": v_metrics["relative_mse"],
        }
    
    return results


def main():
    print("=" * 80)
    print("StoikovKV Evaluation on Qwen2.5-7B-Instruct (4-bit NF4)")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load REAL 7B model in FP16 with CPU offload
    print("\nLoading REAL Qwen2.5-7B-Instruct in FP16 with CPU offload...")
    model, tokenizer = load_model_and_tokenizer(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        device=device,
        dtype=dtype,
        use_real_model=True,
        load_in_4bit=False,
        use_quanto=False,
    )
    
    print(f"Model loaded: {model.config._name_or_path}")
    print(f"Layers: {model.config.num_hidden_layers}")
    print(f"Query heads: {model.config.num_attention_heads}")
    print(f"KV heads: {model.config.num_key_value_heads}")
    head_dim = getattr(model.config, 'head_dim', model.config.hidden_size // model.config.num_attention_heads)
    print(f"Head dim: {head_dim}")
    print(f"Hidden size: {model.config.hidden_size}")
    print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # Load REAL 7B calibration artifacts
    artifacts_path = ROOT / "calibrated_artifacts_7b_real.pt"
    if artifacts_path.exists():
        print(f"\nLoading REAL 7B calibrated artifacts from: {artifacts_path}")
        artifacts = load_or_create_artifacts(model, artifacts_path=str(artifacts_path), tokenizer=tokenizer)
    else:
        print(f"\nReal artifacts not found at {artifacts_path}. Run calibrate_7b_real.py first.")
        print("Creating mock artifacts as fallback...")
        artifacts = load_or_create_artifacts(model, artifacts_path=str(artifacts_path), tokenizer=tokenizer)
    
    print(f"Artifacts loaded. U_Q shape: {tuple(artifacts['U_Q'].shape)}")
    print(f"Head roles distribution: {artifacts['head_roles'].unique(return_counts=True)}")
    
    # Evaluate policies
    print("\n" + "=" * 80)
    print("Policy Evaluation")
    print("=" * 80)
    
    policies = ["full_fp16", "hybrid", "standalone_h2o", "standalone_lexico"]
    all_results = {
        "model": "Qwen/Qwen2.5-7B-Instruct (REAL 28L, 4 KV heads, h=128)",
        "quantization": "FP16 (CPU offload)",
        "device": device,
        "policy_evaluation": {},
    }
    
    for test in EVAL_PROMPTS:
        print(f"\nTest: {test['name']} (retrieval: {test['retrieval_position']})")
        test_results = {}
        
        for policy in policies:
            print(f"  Evaluating {policy}...")
            try:
                result = evaluate_policy(
                    model, tokenizer, artifacts, policy,
                    test["prompt"], test["target"], test["max_new_tokens"]
                )
                test_results[policy] = result
                
                if test["target"]:
                    print(f"    Retrieval: {'PASS' if result['retrieval_top1'] else 'FAIL'} "
                          f"(rank={result['target_rank']}, prob={result['target_probability']:.2e})")
                print(f"    Memory: {result['memory_ratio_vs_fp16']*100:.1f}% of FP16 "
                      f"({result['compressed_bytes']/1e6:.1f}MB vs {result['fp16_bytes']/1e6:.1f}MB)")
                print(f"    Tokens/s: {result['tokens_per_second']:.2f}")
                print(f"    Peak VRAM: {result['peak_memory_gb']:.2f}GB")
                
            except Exception as e:
                print(f"    ERROR: {e}")
                test_results[policy] = {"error": str(e)}
        
        all_results["policy_evaluation"][test["name"]] = test_results
    
    # Save results
    output_path = Path(__file__).parent / "results.json"
    output_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n\nResults saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for test_name, test_results in all_results["policy_evaluation"].items():
        print(f"\n{test_name}:")
        for policy, result in test_results.items():
            if "error" in result:
                print(f"  {policy:20s} ERROR: {result['error']}")
            else:
                mem_pct = result['memory_ratio_vs_fp16'] * 100
                tok_s = result['tokens_per_second']
                if 'retrieval_top1' in result:
                    ret = "PASS" if result['retrieval_top1'] else "FAIL"
                    print(f"  {policy:20s} Mem: {mem_pct:5.1f}% | Tok/s: {tok_s:6.2f} | Retrieval: {ret}")
                else:
                    print(f"  {policy:20s} Mem: {mem_pct:5.1f}% | Tok/s: {tok_s:6.2f}")


if __name__ == "__main__":
    main()