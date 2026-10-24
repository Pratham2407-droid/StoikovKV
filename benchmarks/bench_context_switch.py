"""
Context Switching Benchmark for StoikovKV.

Evaluates KV cache compression performance under realistic context switching scenarios:
- Topic shifts within conversation (e.g., medical → cooking → code → medical)
- Multi-turn conversations with retrieval after topic switches
- Needle-in-haystack with topic boundaries
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stoikov_kv.calibration import (
    CALIBRATION_PROMPTS,
    NEEDLE_PROMPTS,
    HELDOUT_EVAL_PROMPTS,
    load_model_and_tokenizer,
    calibrate_model,
)
from stoikov_kv.elastic_lexico import ElasticLexicoCodec
from stoikov_kv.calibration import ROLE_RETRIEVAL_FP16, ROLE_PEAKED_H2O, ROLE_DIFFUSE_LEXICO
from stoikov_kv.cache import StoikovKVCache


# ============================================================================
# Context Switching Test Cases
# ============================================================================

CONTEXT_SWITCH_TESTS: List[Dict[str, Any]] = [
    {
        "name": "medical_to_cooking",
        "description": "Medical advice → Cooking recipe → Medical followup",
        "conversation": [
            {"role": "user", "content": "I have a fever of 102°F and chills. What should I do?"},
            {"role": "assistant", "content": "For a fever of 102°F, stay hydrated, rest, and take acetaminophen if needed. Contact a doctor if it persists over 3 days or exceeds 103°F."},
            {"role": "user", "content": "Thanks. By the way, can you give me a recipe for chicken tikka masala?"},
            {"role": "assistant", "content": "Sure! For chicken tikka masala: marinate chicken in yogurt and spices (turmeric, cumin, coriander, garam masala, chili), grill or bake, then simmer in tomato-cream sauce with butter and fenugreek leaves. Serve with naan or rice."},
            {"role": "user", "content": "What was my temperature again?"},  # Retrieval after topic switch
        ],
        "needle": "102°F",
        "needle_turn": 0,
        "retrieval_turn": 4,
    },
    {
        "name": "coding_to_travel",
        "description": "Python debugging → Travel planning → Code followup",
        "conversation": [
            {"role": "user", "content": "My Python code has a bug: def fib(n): return fib(n-1) + fib(n-2). It crashes with RecursionError."},
            {"role": "assistant", "content": "The issue is missing base cases. Add: if n <= 1: return n. Also consider memoization or iterative approach for efficiency."},
            {"role": "user", "content": "Thanks! By the way, I'm planning a trip to Japan in March. What should I see in Tokyo?"},
            {"role": "assistant", "content": "Tokyo in March is beautiful! Visit Senso-ji Temple, Shibuya Crossing, Meiji Shrine, Tsukiji Fish Market, and teamLab Planets. Cherry blossoms usually bloom late March."},
            {"role": "user", "content": "What was the base case for the fibonacci function again?"},  # Retrieval after topic switch
        ],
        "needle": "if n <= 1: return n",
        "needle_turn": 0,
        "retrieval_turn": 4,
    },
    {
        "name": "finance_to_fitness",
        "description": "Investment advice → Workout plan → Finance followup",
        "conversation": [
            {"role": "user", "content": "I want to invest $10,000 for retirement. I'm 30 years old. What's a good strategy?"},
            {"role": "assistant", "content": "At 30, consider a diversified portfolio: 70% stocks (VTI or S&P 500 ETF), 20% international, 10% bonds. Max out 401k match, then Roth IRA. Rebalance annually."},
            {"role": "user", "content": "Great advice! Also, I want to start running. Couch to 5K plan?"},
            {"role": "assistant", "content": "Couch to 5K: Week 1-2: 20 min (1 min run/1.5 min walk). Week 3-4: 25 min (3 min run/2 min walk). Week 5-6: 30 min (5 min run/3 min walk). Week 7-8: 30 min continuous run. 3x/week."},
            {"role": "user", "content": "What was the stock/bond split you recommended?"},  # Retrieval after topic switch
        ],
        "needle": "70% stocks, 20% international, 10% bonds",
        "needle_turn": 0,
        "retrieval_turn": 4,
    },
    {
        "name": "multi_topic_long",
        "description": "5-topic conversation with retrieval at end",
        "conversation": [
            {"role": "user", "content": "API documentation says the rate limit is 100 requests per minute. The endpoint is /api/v1/users."},
            {"role": "assistant", "content": "The rate limit is 100 requests per minute for /api/v1/users. Implement exponential backoff on 429 responses."},
            {"role": "user", "content": "Switching topics: best practices for React useEffect cleanup?"},
            {"role": "assistant", "content": "Return a cleanup function from useEffect. It runs before the next effect and on unmount. Use for subscriptions, timers, event listeners."},
            {"role": "user", "content": "What about PostgreSQL connection pooling with PgBouncer?"},
            {"role": "assistant", "content": "PgBouncer: transaction pooling for short transactions, session pooling for long. Set pool_mode=transaction. Monitor pool saturation with pg_stat_activity."},
            {"role": "user", "content": "Back to the API: what was the rate limit and endpoint?"},  # Retrieval across 3 topics
        ],
        "needle": "100 requests per minute for /api/v1/users",
        "needle_turn": 0,
        "retrieval_turn": 6,
    },
]


# ============================================================================
# Evaluation Functions
# ============================================================================

def _extract_needle_answer(text: str, needle: str) -> bool:
    """Check if generated text contains the needle answer."""
    text_lower = text.lower()
    needle_lower = needle.lower()
    # Check for exact needle or key parts
    needle_parts = [p for p in needle_lower.split() if len(p) > 3]
    if any(part in text_lower for part in needle_parts):
        return True
    # Fuzzy match for numbers
    import re
    needle_nums = re.findall(r'\d+', needle)
    if needle_nums:
        return any(n in text_lower for n in needle_nums)
    return False


def format_conversation_for_model(tokenizer, conversation: List[Dict[str, str]]) -> str:
    """Format conversation for model input."""
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
    # Fallback formatting
    text = ""
    for turn in conversation:
        role = turn["role"]
        content = turn["content"]
        if role == "user":
            text += f"User: {content}\n"
        elif role == "assistant":
            text += f"Assistant: {content}\n"
    text += "Assistant:"
    return text


def run_context_switch_test(
    model,
    tokenizer,
    test_case: Dict[str, Any],
    cache,
    max_new_tokens: int = 32,
) -> Tuple[bool, str, float]:
    """Run a single context switching test case."""
    conversation = test_case["conversation"]
    needle = test_case["needle"]
    retrieval_turn = test_case["retrieval_turn"]
    
    # Build conversation up to retrieval turn
    history = conversation[:retrieval_turn + 1]
    
    # Format for model
    text = format_conversation_for_model(tokenizer, history)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda")
    
    # Run with cache
    with torch.inference_mode():
        out = model(input_ids=inputs["input_ids"], past_key_values=cache, use_cache=True, return_dict=True)
    
    # Generate response
    logits = out.logits[:, -1]
    generated = []
    for _ in range(max_new_tokens):
        token = torch.argmax(logits, dim=-1)
        generated.append(token.item())
        step = model(input_ids=token.unsqueeze(0), past_key_values=cache, use_cache=True, return_dict=True)
        logits = step.logits[:, -1]
    
    generated_text = tokenizer.decode(generated, skip_special_tokens=True)
    ok = _extract_needle_answer(generated_text, needle)
    
    return ok, generated_text, cache.physical_storage_bytes


def format_conversation_for_model(tokenizer, conversation: List[Dict[str, str]]) -> str:
    """Format conversation for model input."""
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
    # Fallback formatting
    text = ""
    for turn in conversation:
        role = turn["role"]
        content = turn["content"]
        if role == "user":
            text += f"User: {content}\n"
        elif role == "assistant":
            text += f"Assistant: {content}\n"
    text += "Assistant:"
    return text


def _extract_needle_answer(text: str, needle: str) -> bool:
    """Check if generated text contains the needle answer."""
    text_lower = text.lower()
    needle_lower = needle.lower()
    # Check for exact needle or key parts
    needle_parts = [p for p in needle_lower.split() if len(p) > 3]
    if any(part in text_lower for part in needle_parts):
        return True
    # Fuzzy match for numbers
    import re
    needle_nums = re.findall(r'\d+', needle)
    if needle_nums:
        return any(n in text_lower for n in needle_nums)
    return False


def run_context_switch_benchmark(
    model,
    tokenizer,
    artifacts,
    codec,
    max_new_tokens: int = 32,
) -> List[Dict[str, Any]]:
    """Run all context switching test cases."""
    results = []
    
    for test_case in CONTEXT_SWITCH_TESTS:
        print(f"\nTesting: {test_case['name']} - {test_case['description']}")
        
        # Test different cache configurations
        configs = [
            ("Full FP16", {"dense_fallback": True}, ROLE_RETRIEVAL_FP16),
            ("H2O r=0.3", {"dense_fallback": True, "keep_ratio": 0.3, "h2o_context_window": 1}, ROLE_PEAKED_H2O),
            ("Lexico f=0.5", {"dense_fallback": True, "lexico_fp16_frac": 0.5}, ROLE_DIFFUSE_LEXICO),
            ("Hybrid", {"dense_fallback": True, "keep_ratio": 0.3, "lexico_fp16_frac": 0.5, "h2o_context_window": 1}, None),
        ]
        
        case_results = {"test_case": test_case["name"], "configs": []}
        
        for label, kwargs, roles in configs:
            c = StoikovKVCache(model, **kwargs)
            if roles is not None:
                c.head_roles.fill_(roles)
            
            # Run test 3 times for consistency
            passed = 0
            texts = []
            for trial in range(3):
                torch.manual_seed(100 + trial)
                ok, text, phys = run_context_switch_test(model, tokenizer, test_case, kwargs, max_new_tokens=32)
                texts.append(text)
                if ok:
                    passed += 1
            
            case_results["configs"].append({
                "label": label,
                "passed": f"{passed}/3",
                "physical_mb": c.physical_storage_bytes / 1e6,
                "example_text": texts[0][:80] if texts else "",
            })
            
            status = "PASS" if passed > 0 else "FAIL"
            print(f"  {label}: {passed}/3 {status} | {c.physical_storage_bytes/1e6:.1f}MB | {texts[0][:60] if texts else ''}")
        
        results.append(case_results)
    
    return results


def main():
    device = "cuda"
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct", seed=42)
    print(f"Model loaded: {model.__class__.__name__} on {next(model.parameters()).device}")
    
    print("Running calibration...")
    artifacts = calibrate_model(
        model, tokenizer, 
        save_path="/tmp/StoikovKV/calibrated_artifacts_cs.pt",
        seed=42, r_subspace=16, dict_atoms=64, dict_steps=10
    )
    
    codec = ElasticLexicoCodec.from_artifacts(artifacts)
    
    print("\n" + "="*60)
    print("CONTEXT SWITCHING BENCHMARK")
    print("="*60)
    
    results = run_context_switch_benchmark(model, tokenizer, artifacts, None, max_new_tokens=32)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for case in results:
        print(f"\n{case['test_case']}:")
        for cfg in case["configs"]:
            print(f"  {cfg['label']:20s} {cfg['passed']:>5s} | {cfg['physical_mb']:.1f}MB")
    
    # Save results
    output = {
        "results": results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open("context_switch_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("\nResults saved to context_switch_results.json")


if __name__ == "__main__":
    main()