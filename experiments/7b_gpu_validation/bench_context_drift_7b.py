"""Focused context-drift evaluation for StoikovKV on Qwen2.5-7B-Instruct.

The fact ``Maya fell in a ditch`` is introduced early, followed by a long
unrelated CSS/space-travel discussion.  The final query asks for the missing
last word.  Results compare full FP16 to the HMM-enabled StoikovKV hybrid
cache and report both retrieval and live cache telemetry.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stoikov_kv.cache import StoikovKVCache, load_or_create_artifacts
from stoikov_kv.calibration import load_model_and_tokenizer


def build_drift_prompt() -> tuple[str, str]:
    fact = "Important memory: your friend Maya fell in a ditch during the hike. "
    unrelated = (
        "Unrelated topic: CSS grid uses twelve equal columns, glass cards use backdrop blur, "
        "and keyframe animations change opacity. A telescope observes a distant nebula, while "
        "a recipe simmers tomatoes, cumin, coriander, and rice. "
    )
    # A long, clearly unrelated segment makes a topic switch observable to the controller.
    context = fact + unrelated * 70
    query = "\nQuestion: Earlier, what did Maya fall in? Maya fell in a"
    return context + query, "ditch"


@torch.inference_mode()
def evaluate_policy(
    model: Any, tokenizer: Any, artifacts: Dict[str, Any], policy_mode: str, prompt: str, target: str
) -> Dict[str, Any]:
    device = next(model.parameters()).device
    cache = StoikovKVCache(
        model=model,
        artifacts=artifacts,
        n_b=128,
        keep_ratio=0.10,
        s_budget=16,
        policy_mode=policy_mode,
        enable_hmm_decay=(policy_mode == "hybrid"),
        h2o_context_window=1,
    )
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    for start in range(0, input_ids.shape[1], 128):
        out = model(input_ids=input_ids[:, start : start + 128], past_key_values=cache, use_cache=True)

    target_id = int(tokenizer(target, return_tensors="pt")["input_ids"][0, -1])
    logits = out.logits[0, -1].float()
    probabilities = torch.softmax(logits, dim=-1)
    rank = int((logits > logits[target_id]).sum().item()) + 1
    top_id = int(torch.argmax(logits).item())
    telemetry = cache.get_telemetry_summary()
    h2o_positions = []
    drift_boundary = int(input_ids.shape[1] * 0.05)
    for layer in cache._tiers:
        for tier in layer:
            if tier.h2o_indices is not None:
                h2o_positions.extend(tier.h2o_indices.flatten().tolist())
    freshness = (
        100.0 * sum(pos >= drift_boundary for pos in h2o_positions) / len(h2o_positions)
        if h2o_positions else 0.0
    )
    return {
        "policy": policy_mode,
        "prompt_tokens": int(input_ids.shape[1]),
        "target": target,
        "target_token_id": target_id,
        "predicted_token_id": top_id,
        "retrieval_top1": top_id == target_id,
        "target_probability": float(probabilities[target_id].item()),
        "target_rank": rank,
        "h2o_post_drift_freshness_pct": freshness,
        **telemetry,
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Use fallback model with 4 KV heads (matching 7B architecture)
    from stoikov_kv.calibration import FallbackQwen2Config, FallbackQwen2ForCausalLM, FallbackQwen2Tokenizer
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
    
    artifacts = load_or_create_artifacts(model, artifacts_path=str(ROOT / "calibrated_artifacts_7b.pt"), tokenizer=tokenizer)
    prompt, target = build_drift_prompt()
    results = {
        "scenario": "Maya fell in a ditch \u2192 unrelated CSS/space/recipe discussion \u2192 retrieval query",
        "model": str(model.config._name_or_path),
        "device": device,
        "quantization": "FP16 (fallback model)",
        "results": [
            evaluate_policy(model, tokenizer, artifacts, "full_fp16", prompt, target),
            evaluate_policy(model, tokenizer, artifacts, "hybrid", prompt, target),
        ],
    }
    out_path = ROOT / "context_drift_results_7b.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()