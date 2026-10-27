"""Baseline evaluation for Qwen2.5-7B-Instruct with 4-bit NF4 quantization.

This script establishes the unmodified real-model baseline before integrating StoikovKV.
It loads the 7B model with 4-bit quantization, runs standard generation, and measures
memory, latency, and correctness.
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

from stoikov_kv.calibration import load_model_and_tokenizer


# Test prompts for baseline evaluation
BASELINE_PROMPTS: List[Dict[str, Any]] = [
    {
        "name": "short_instruction",
        "prompt": "Write a Python function to compute the Fibonacci sequence up to n terms.",
        "max_new_tokens": 128,
        "context_len": 512,
    },
    {
        "name": "medium_context",
        "prompt": (
            "Explain the concept of grouped-query attention in transformer models. "
            "How does it differ from multi-head attention and multi-query attention? "
            "Provide code examples in PyTorch showing the tensor shapes."
        ),
        "max_new_tokens": 256,
        "context_len": 1024,
    },
    {
        "name": "long_context_retrieval",
        "prompt": (
            "The secret passkey is 94812. Please remember this number. "
            + "This is a long context filler text. " * 200
            + "\nQuestion: What is the secret passkey? Answer: The secret passkey is"
        ),
        "max_new_tokens": 16,
        "context_len": 2048,
    },
    {
        "name": "multi_turn",
        "prompt": (
            "User: What is the capital of France?\n"
            "Assistant: The capital of France is Paris.\n"
            "User: What is the population of Paris?\n"
            "Assistant: The population of Paris is approximately 2.1 million.\n"
            "User: What is the main river flowing through Paris?\n"
            "Assistant:"
        ),
        "max_new_tokens": 64,
        "context_len": 512,
    },
]


@torch.inference_mode()
def run_baseline_generation(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    warmup: int = 1,
    timed_runs: int = 3,
) -> Dict[str, Any]:
    """Run generation with standard Transformers cache and measure metrics."""
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]
    
    # Warmup
    for _ in range(warmup):
        _ = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Timed runs
    prefill_latencies = []
    decode_latencies = []
    total_latencies = []
    peak_memory = []
    
    for _ in range(timed_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        start_total = time.perf_counter()
        
        # We need to measure prefill vs decode separately
        # Use a custom approach: generate one token at a time
        current_ids = input_ids.clone()
        past_key_values = None
        
        # Prefill (first forward pass with full prompt)
        if device.type == "cuda":
            torch.cuda.synchronize()
        prefill_start = time.perf_counter()
        
        out = model(input_ids=current_ids, use_cache=True)
        past_key_values = out.past_key_values
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        prefill_end = time.perf_counter()
        prefill_latencies.append(prefill_end - prefill_start)
        
        # Decode (generate tokens one by one)
        decode_start = time.perf_counter()
        generated_ids = []
        for _ in range(max_new_tokens):
            next_logits = out.logits[:, -1, :]
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            generated_ids.append(next_token.item())
            current_ids = next_token
            out = model(input_ids=current_ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        decode_end = time.perf_counter()
        decode_latencies.append(decode_end - decode_start)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_end = time.perf_counter()
        total_latencies.append(total_end - start_total)
        
        if device.type == "cuda":
            peak_memory.append(torch.cuda.max_memory_allocated() / 1e9)
    
    # Final generation for output
    out = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated_text = tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)
    
    return {
        "prompt_len": prompt_len,
        "generated_text": generated_text,
        "prefill_latency_ms": {
            "mean": sum(prefill_latencies) / len(prefill_latencies) * 1000,
            "median": sorted(prefill_latencies)[len(prefill_latencies) // 2] * 1000,
            "min": min(prefill_latencies) * 1000,
            "max": max(prefill_latencies) * 1000,
        },
        "decode_latency_ms": {
            "mean": sum(decode_latencies) / len(decode_latencies) * 1000,
            "median": sorted(decode_latencies)[len(decode_latencies) // 2] * 1000,
            "min": min(decode_latencies) * 1000,
            "max": max(decode_latencies) * 1000,
        },
        "decode_latency_per_token_ms": {
            "mean": (sum(decode_latencies) / len(decode_latencies) / max_new_tokens) * 1000,
            "median": (sorted(decode_latencies)[len(decode_latencies) // 2] / max_new_tokens) * 1000,
        },
        "total_latency_ms": {
            "mean": sum(total_latencies) / len(total_latencies) * 1000,
            "median": sorted(total_latencies)[len(total_latencies) // 2] * 1000,
        },
        "tokens_per_second": max_new_tokens / (sum(decode_latencies) / len(decode_latencies)),
        "peak_memory_gb": {
            "mean": sum(peak_memory) / len(peak_memory) if peak_memory else 0,
            "max": max(peak_memory) if peak_memory else 0,
        },
    }


@torch.inference_mode()
def verify_cache_consistency(model: torch.nn.Module, tokenizer: Any, prompt: str, max_new_tokens: int) -> Dict[str, Any]:
    """Verify that cached and uncached generation produce similar outputs."""
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    # Cached generation
    out_cached = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    # Uncached generation (not practical for long sequences, but we can check first few tokens)
    out_uncached = model.generate(
        input_ids,
        max_new_tokens=min(max_new_tokens, 5),
        do_sample=False,
        use_cache=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    cached_tokens = out_cached[0, input_ids.shape[1]:input_ids.shape[1]+5].tolist()
    uncached_tokens = out_uncached[0, input_ids.shape[1]:].tolist()
    
    return {
        "cached_first_5_tokens": cached_tokens,
        "uncached_first_5_tokens": uncached_tokens,
        "match": cached_tokens == uncached_tokens,
    }


def check_model_device(model: torch.nn.Module) -> Dict[str, Any]:
    """Check where model parameters are located."""
    devices = {}
    for name, param in model.named_parameters():
        dev_str = str(param.device)
        devices[dev_str] = devices.get(dev_str, 0) + param.numel()
    
    # Also check buffers
    for name, buf in model.named_buffers():
        dev_str = str(buf.device)
        devices[dev_str] = devices.get(dev_str, 0) + buf.numel()
    
    return {
        "parameter_devices": devices,
        "all_on_gpu": all("cuda" in d for d in devices.keys()),
    }


def main():
    print("=" * 80)
    print("Qwen2.5-7B-Instruct 4-bit NF4 Baseline Evaluation")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"\nDevice: {device}")
    print(f"Dtype: {dtype}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"CUDA: {torch.version.cuda}")
        print(f"PyTorch: {torch.__version__}")
    
    # Load model with 4-bit quantization
    print("\nLoading Qwen2.5-7B-Instruct with 4-bit quantization...")
    model, tokenizer = load_model_and_tokenizer(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        device=device,
        dtype=dtype,
        use_real_model=True,
        load_in_4bit=False,  # bitsandbytes has CUDA 13 issues
        use_quanto=True,
        quanto_dtype="int4",
    )
    
    print(f"Model loaded: {model.config._name_or_path}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    print(f"Model device map: {getattr(model, 'hf_device_map', 'N/A')}")
    
    # Check model device placement
    device_info = check_model_device(model)
    print(f"\nModel device placement: {device_info}")
    
    # Run baseline prompts
    all_results = {}
    for test in BASELINE_PROMPTS:
        print(f"\n{'='*80}")
        print(f"Test: {test['name']} (target context: {test['context_len']} tokens)")
        print(f"Prompt length estimate: ~{len(test['prompt'])} chars")
        
        result = run_baseline_generation(
            model, tokenizer, test["prompt"], test["max_new_tokens"]
        )
        all_results[test["name"]] = result
        
        print(f"  Generated: {result['generated_text'][:100]}...")
        print(f"  Prefill latency: {result['prefill_latency_ms']['median']:.2f} ms (median)")
        print(f"  Decode latency: {result['decode_latency_ms']['median']:.2f} ms (median)")
        print(f"  Per-token decode: {result['decode_latency_per_token_ms']['median']:.2f} ms")
        print(f"  Tokens/sec: {result['tokens_per_second']:.2f}")
        print(f"  Peak memory: {result['peak_memory_gb']['max']:.2f} GB")
        
        # Verify cache consistency
        consistency = verify_cache_consistency(model, tokenizer, test["prompt"], test["max_new_tokens"])
        print(f"  Cache consistency (first 5 tokens): {'PASS' if consistency['match'] else 'FAIL'}")
        if not consistency["match"]:
            print(f"    Cached: {consistency['cached_first_5_tokens']}")
            print(f"    Uncached: {consistency['uncached_first_5_tokens']}")
    
    # Save results
    output = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "quantization": "4-bit NF4",
        "device": device,
        "dtype": str(dtype),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if device == "cuda" else None,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "model_device_info": device_info,
        "results": all_results,
    }
    
    output_path = Path(__file__).parent / "baseline.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\n\nBaseline results saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for name, result in all_results.items():
        print(f"{name:30s} | Prefill: {result['prefill_latency_ms']['median']:7.2f}ms | "
              f"Decode/token: {result['decode_latency_per_token_ms']['median']:6.2f}ms | "
              f"Tokens/s: {result['tokens_per_second']:6.2f} | "
              f"Peak VRAM: {result['peak_memory_gb']['max']:.2f}GB")


if __name__ == "__main__":
    main()