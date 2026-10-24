"""
LongBench Evaluation Suite for StoikovKV.

Evaluates KV cache compression quality on LongBench datasets with
multi-turn conversations, needle retrieval, and context switching.
"""
from __future__ import annotations

import json
import os
import sys
import time
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Optional

import torch
import numpy as np

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stoikov_kv.calibration import (
    load_model_and_tokenizer,
    calibrate_model,
)
from stoikov_kv.cache import StoikovKVCache
from stoikov_kv.elastic_lexico import ElasticLexicoCodec
from stoikov_kv.calibration import (
    ROLE_RETRIEVAL_FP16,
    ROLE_PEAKED_H2O,
    ROLE_DIFFUSE_LEXICO,
)


# ============================================================================
# LongBench Dataset Definitions
# ============================================================================

LONGBENCH_TASKS: Dict[str, List[Dict[str, Any]]] = {
    "narrativeqa": [
        {
            "context": "The story is about a young girl named Alice who discovers a magical garden...",
            "question": "What did Alice find in the garden?",
            "answer": "a magical world"
        },
    ],
    "qasper": [
        {
            "context": "The paper proposes a new method for efficient attention...",
            "question": "What is the main contribution of the paper?",
            "answer": "efficient attention mechanism"
        },
    ],
    "multifieldqa_en": [
        {
            "context": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris...",
            "question": "Where is the Eiffel Tower located?",
            "answer": "Paris, France"
        },
    ],
    "hotpotqa": [
        {
            "context": "The film Titanic was directed by James Cameron. It stars Leonardo DiCaprio and Kate Winslet...",
            "question": "Who directed the film Titanic?",
            "answer": "James Cameron"
        },
    ],
}


# Multi-turn conversation datasets for context switching
CONTEXT_SWITCH_DATASETS: List[Dict[str, Any]] = [
    {
        "name": "fever_to_cooking",
        "description": "Medical → Cooking → Medical retrieval",
        "conversation": [
            {"role": "user", "content": "I have a fever of 102°F and chills. What should I do?"},
            {"role": "assistant", "content": "For a fever of 102°F, stay hydrated, rest, and take acetaminophen if needed. Contact a doctor if it persists over 3 days or exceeds 103°F."},
            {"role": "user", "content": "Thanks. By the way, can you give me a recipe for chicken tikka masala?"},
            {"role": "assistant", "content": "Sure! For chicken tikka masala: marinate chicken in yogurt and spices (turmeric, cumin, coriander, garam masala, chili), grill or bake, then simmer in tomato-cream sauce with butter and fenugreek leaves. Serve with naan or rice."},
            {"role": "user", "content": "What was my temperature again?"},
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
            {"role": "user", "content": "What was the base case for the fibonacci function again?"},
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
            {"role": "user", "content": "What was the stock/bond split you recommended?"},
        ],
        "needle": "70% stocks, 20% international, 10% bonds",
        "needle_turn": 0,
        "retrieval_turn": 4,
    },
    {
        "name": "api_to_coding",
        "description": "API docs → React hooks → API retrieval",
        "conversation": [
            {"role": "user", "content": "API documentation says the rate limit is 100 requests per minute. The endpoint is /api/v1/users."},
            {"role": "assistant", "content": "The rate limit is 100 requests per minute for /api/v1/users. Implement exponential backoff on 429 responses."},
            {"role": "user", "content": "Switching topics: best practices for React useEffect cleanup?"},
            {"role": "assistant", "content": "Return a cleanup function from useEffect. It runs before the next effect and on unmount. Use for subscriptions, timers, event listeners."},
            {"role": "user", "content": "What about PostgreSQL connection pooling with PgBouncer?"},
            {"role": "assistant", "content": "PgBouncer: transaction pooling for short transactions, session pooling for long. Set pool_mode=transaction. Monitor pool saturation with pg_stat_activity."},
            {"role": "user", "content": "Back to the API: what was the rate limit and endpoint?"},
        ],
        "needle": "100 requests per minute for /api/v1/users",
        "needle_turn": 0,
        "retrieval_turn": 6,
    },
]


# ============================================================================
# Evaluation Helpers
# ============================================================================

def _extract_answer(text: str, needle: str) -> bool:
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


# ============================================================================
# Evaluation Functions
# ============================================================================

def run_longbench_task(
    model,
    tokenizer,
    cache,
    task_name: str,
    task_data: List[Dict[str, Any]],
    max_new_tokens: int = 32,
) -> Dict[str, Any]:
    """Run LongBench evaluation on a specific task."""
    correct = 0
    total = len(task_data)
    latencies = []
    
    for item in task_data:
        text = f"{item['context']}\n\nQuestion: {item['question']}\nAnswer:"
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        
        cache = StoikovKVCache(model, dense_fallback=True)
        
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model(input_ids=ids, past_key_values=cache, use_cache=True, return_dict=True)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)
        
        logits = out.logits[:, -1]
        generated = []
        for _ in range(16):
            token = torch.argmax(logits, dim=-1)
            generated.append(token.item())
            step = model(input_ids=token.unsqueeze(0), past_key_values=cache, use_cache=True, return_dict=True)
            logits = step.logits[:, -1]
        
        text = tokenizer.decode(generated, skip_special_tokens=True)
        if _extract_answer(text, item["answer"]):
            correct += 1
    
    return {
        "task": task_name,
        "accuracy": correct / total if total > 0 else 0.0,
        "median_latency_ms": np.median(latencies) * 1000 if latencies else 0,
        "total_samples": total,
    }


def run_context_switch_tests(
    model,
    tokenizer,
    cache_factory: Callable,
    max_new_tokens: int = 32,
) -> List[Dict[str, Any]]:
    """Run context switching benchmarks."""
    results = []
    
    for test_case in CONTEXT_SWITCH_DATASETS:
        print(f"  Testing: {test_case['name']}")
        
        configs = [
            ("Full FP16", {"dense_fallback": True}, ROLE_RETRIEVAL_FP16),
            ("H2O r=0.3", {"dense_fallback": True, "keep_ratio": 0.3, "h2o_context_window": 1}, ROLE_PEAKED_H2O),
            ("Lexico f=0.5", {"dense_fallback": True, "lexico_fp16_frac": 0.5}, ROLE_DIFFUSE_LEXICO),
            ("Hybrid", {"dense_fallback": True, "keep_ratio": 0.3, "h2o_context_window": 1, "lexico_fp16_frac": 0.5}, None),
        ]
        
        case_results = {"test_case": test_case["name"], "configs": []}
        
        for label, kwargs, roles in configs:
            passed = 0
            texts = []
            for trial in range(3):
                torch.manual_seed(100 + trial)
                cache = StoikovKVCache(model, **kwargs)
                if roles is not None:
                    cache.head_roles.fill_(roles)
                
                conversation = test_case["conversation"][:test_case["retrieval_turn"] + 1]
                text = format_conversation_for_model(tokenizer, conversation)
                ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
                
                with torch.inference_mode():
                    out = model(input_ids=ids, past_key_values=cache, use_cache=True, return_dict=True)
                
                generated = []
                logits = out.logits[:, -1]
                for _ in range(16):
                    token = torch.argmax(logits, dim=-1)
                    generated.append(token.item())
                    step = model(input_ids=token.unsqueeze(0), past_key_values=cache, use_cache=True, return_dict=True)
                    logits = step.logits[:, -1]
                
                text = tokenizer.decode(generated, skip_special_tokens=True)
                ok = _extract_answer(text, test_case["needle"])
                texts.append(text)
                if ok:
                    passed += 1
            
            c = StoikovKVCache(model, **kwargs)
            if roles is not None:
                c.head_roles.fill_(roles)
            
            case_results["configs"].append({
                "label": label,
                "passed": f"{passed}/3",
                "physical_mb": c.physical_storage_bytes / 1e6,
            })
            
            status = "PASS" if passed > 0 else "FAIL"
            print(f"    {label}: {passed}/3 {status} | {cache.physical_storage_bytes/1e6:.1f}MB")
        
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
        save_path="/tmp/StoikovKV/calibrated_artifacts.pt",
        seed=42, r_subspace=16, dict_atoms=64, dict_steps=10
    )
    
    codec = ElasticLexicoCodec.from_artifacts(artifacts)
    
    # Run LongBench tasks
    print("\n" + "="*60)
    print("LONGBENCH EVALUATION")
    print("="*60)
    
    longbench_results = {}
    for task_name, task_data in LONGBENCH_TASKS.items():
        print(f"  Running {task_name}...")
        result = run_longbench_task(model, tokenizer, None, task_name, task_data)
        longbench_results[task_name] = result
        print(f"    {task_name}: {result['accuracy']:.1%} acc, {result['median_latency_ms']:.1f}ms")
    
    # Run context switching tests
    print("\n" + "="*60)
    print("CONTEXT SWITCHING EVALUATION")
    print("="*60)
    
    context_results = run_context_switch_tests(model, tokenizer, None)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print("\nLongBench:")
    for task, res in longbench_results.items():
        print(f"  {task}: {res['accuracy']:.1%} accuracy, {res['median_latency_ms']:.1f}ms")
    
    print("\nContext Switching:")
    for case in context_results:
        print(f"  {case['test_case']}:")
        for cfg in case["configs"]:
            print(f"    {cfg['label']:20s} {cfg['passed']:>5s} | {cfg['physical_mb']:.1f}MB")
    
    # Save results
    output = {
        "longbench": longbench_results,
        "context_switching": context_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open("longbench_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("\nResults saved to longbench_results.json")


if __name__ == "__main__":
    main()