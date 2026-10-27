"""
Person 2 Standalone Evaluation & Benchmark Suite (`benchmarks/bench_person2.py`).

Evaluates the live StoikovKV inference engine (`stoikov_kv/cache.py` & `stoikov_kv/matryoshka_spec.py`)
on CUDA/FP16 without ever requiring `output_attentions=True` during live generation:

1. **4-Way Ablation at ~20% KV Byte Budget on a 2,000-Token Multi-Turn Prompt**:
   - `Full FP16 (100% Memory)`
   - `Standalone H2O (20% Memory - All Heads Evict)`
   - `Standalone Lexico (20% Memory - All Heads Encode)`
   - `StoikovKV Hybrid (Retrieval Guard + Peaked H2O + Diffuse Lexico + HMM Decay)`
   Prompt structure (~2,000 tokens):
   - Turn 1 (Tokens 0..1150): System architecture & backend Python service logs containing an
     early Needle-in-a-Haystack fact (`"API_PORT = 8443"`).
   - Turn 2 (Tokens 1150..2000): Abrupt topic shift to modern CSS glassmorphism, flexbox/grid
     layout, and keyframe animation rules, followed by queries testing both the early Turn-1
     needle (`"API_PORT ="`) and Turn-2 CSS context fidelity.

2. **Matryoshka Self-Speculative Decoding Sweep (`draft_atoms m in {2, 4, 8, 12}` vs `verify_atoms s = 16`)**:
   - Measures empirical and expected acceptance rate `alpha`, average accepted tokens per
     parallel verification pass, and effective KV bytes read per generated token vs standard
     autoregressive decoding.

3. **Outputs**:
   - `pareto_quality_vs_bytes.png`
   - `matryoshka_acceptance.png`
   - `eval_results.json`
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_stoikov")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from stoikov_kv.cache import (
    HMMTopicShiftController,
    StoikovKVCache,
    create_mock_artifacts,
    load_or_create_artifacts,
)
from stoikov_kv.calibration import (
    ROLE_DIFFUSE_LEXICO,
    ROLE_PEAKED_H2O,
    ROLE_RETRIEVAL_FP16,
    calibrate_model,
    load_model_and_tokenizer,
)
from stoikov_kv.matryoshka_spec import matryoshka_speculative_generate


def build_2k_multiturn_prompt(
    tokenizer: Any,
    target_tokens: int = 2000,
    needle_depth_ratio: float = 0.08,
    query_mode: str = "needle",
) -> Tuple[str, torch.Tensor, Dict[str, int]]:
    """
    Constructs a ~2,000-token multi-turn prompt containing:
    - Sink prefix (Tokens 0..15)
    - Turn 1 Backend/Python haystack (first ~58% of tokens) with an early needle
      `"Configuration fact: API_PORT = 8443 ."` placed at `needle_depth_ratio`.
    - Turn 2 CSS Styling topic shift (remaining ~42% of tokens) with dense CSS rules
      (`.glass-modal`, `backdrop-filter`, `grid-template-columns`, `@keyframes`, `rgba`, `rem`).
    - Final query testing either:
      * `query_mode="needle"`: `"Recall the early server configuration: API_PORT ="` (target `"8443"`)
      * `query_mode="css"`: `"Based on the Turn 2 stylesheet, the glass modal uses backdrop"`
    """
    turn1_clauses = [
        "def dispatch_worker_pool(request_queue, max_retries=3, timeout_ms=250): "
        "for item in request_queue: validate_schema(item); commit_transaction(item)\n",
        "Backend telemetry log: worker_node_07 processed batch_id=4021 with p99 latency 4.2ms "
        "and zero packet drops across internal gRPC endpoints.\n",
        "Database connection pool initialized with read_replicas=4, pool_recycle=1800, "
        "isolation_level=READ_COMMITTED, statement_cache_size=512.\n",
        "class DistributedRateLimiter: def acquire_token(self, client_key, burst_limit): "
        "return self.redis_cluster.eval_lua_window(client_key, burst_limit)\n",
    ]
    turn2_css_clauses = [
        ".glass-card { display: flex; flex-direction: column; padding: 1.5rem; margin: 0.75rem; "
        "border-radius: 12px; backdrop-filter: blur(16px); background: rgba(255, 255, 255, 0.12); "
        "box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25); transition: transform 200ms ease-in-out; }\n",
        ".dashboard-grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 1.25rem; "
        "color: #f8fafc; font-weight: 600; opacity: 0.95; z-index: 10; overflow: hidden; }\n",
        "@keyframes pulse-glow { 0% { opacity: 0.65; transform: scale(0.98); } "
        "50% { opacity: 1.0; transform: scale(1.01); } 100% { opacity: 0.65; transform: scale(0.98); } }\n",
        "@media (max-width: 768px) { .sidebar-nav { position: fixed; width: 100vw; height: 4rem; "
        "padding: 0.5rem 1rem; border: 1px solid rgba(148, 163, 184, 0.2); } }\n",
    ]

    needle_text = " [CRITICAL CONFIG: API_PORT = 8443 ] "
    turn1_target = int(target_tokens * 0.58)
    needle_pos_target = max(16, int(target_tokens * needle_depth_ratio))

    # Pre-tokenize clause lengths once so prompt assembly is O(1) without repeated re-tokenization
    t1_lens = [
        int((tokenizer(c, return_tensors="pt", truncation=False)["input_ids"]).shape[1]) - 1
        for c in turn1_clauses
    ]
    t2_lens = [
        int((tokenizer(c, return_tensors="pt", truncation=False)["input_ids"]).shape[1]) - 1
        for c in turn2_css_clauses
    ]

    prefix_text = "System Session Turn 1 (Backend Infrastructure):\n"
    cur_est = 8
    idx_p = 0
    while cur_est < needle_pos_target and idx_p < 512:
        c = turn1_clauses[idx_p % len(turn1_clauses)]
        prefix_text += c
        cur_est += max(1, t1_lens[idx_p % len(t1_lens)])
        idx_p += 1

    needle_start_ids = (
        tokenizer(prefix_text, return_tensors="pt", truncation=False)["input_ids"]
    ).shape[1]
    body_text = prefix_text + needle_text
    needle_end_ids = (
        tokenizer(body_text, return_tensors="pt", truncation=False)["input_ids"]
    ).shape[1]

    cur_est = int(needle_end_ids)
    idx_c = 0
    while cur_est < turn1_target and idx_c < 512:
        c = turn1_clauses[idx_c % len(turn1_clauses)]
        body_text += c
        cur_est += max(1, t1_lens[idx_c % len(t1_lens)])
        idx_c += 1

    turn2_start_ids = (
        tokenizer(body_text, return_tensors="pt", truncation=False)["input_ids"]
    ).shape[1]
    body_text += "\nUser Session Turn 2 (Topic Shift -> Frontend CSS Design System):\n"
    cur_est = int(turn2_start_ids) + 12

    idx_css = 0
    while cur_est < target_tokens - 16 and idx_css < 512:
        c = turn2_css_clauses[idx_css % len(turn2_css_clauses)]
        body_text += c
        cur_est += max(1, t2_lens[idx_css % len(t2_lens)])
        idx_css += 1

    if query_mode == "needle":
        query_suffix = "\nVerification Query: What is the configured port? API_PORT ="
    else:
        query_suffix = "\nCSS Stylesheet Query: In .glass-card, display: flex; padding: 1.5rem; backdrop-filter:"

    full_prompt = body_text + query_suffix
    enc_full = tokenizer(full_prompt, return_tensors="pt", truncation=False)
    full_ids = enc_full["input_ids"] if isinstance(enc_full, dict) else enc_full.input_ids

    meta = {
        "total_tokens": int(full_ids.shape[1]),
        "needle_start_token": int(needle_start_ids),
        "needle_end_token": int(needle_end_ids),
        "turn2_start_token": int(turn2_start_ids),
    }
    return full_prompt, full_ids, meta


def predicted_memory_ratio(
    head_roles: torch.Tensor,
    seq_len: int,
    n_b: int,
    keep_ratio: float,
    s_budget: int,
    policy_mode: str,
    head_dim: int = 128,
) -> float:
    """
    Closed-form KV byte ratio vs FP16 using the SAME accounting as
    `StoikovKVCache._refresh_telemetry` (K+V per head per token):
      - local window (all heads):   n_b * 4*d bytes
      - ROLE_RETRIEVAL_FP16:         hist * 4*d
      - ROLE_PEAKED_H2O:             min(int(T*keep_ratio), hist) * 4*d
      - ROLE_DIFFUSE_LEXICO:         hist * 2*(3s+2)
    Dictionary/W_ridge overhead is reported separately (see `dictionary_bytes`).
    """
    n_layers, n_kv = int(head_roles.shape[0]), int(head_roles.shape[1])
    n_heads = n_layers * n_kv
    if policy_mode == "full_fp16":
        return 1.0
    if policy_mode == "standalone_h2o":
        n_ret, n_h2o, n_lex = 0, n_heads, 0
    elif policy_mode == "standalone_lexico":
        n_ret, n_h2o, n_lex = 0, 0, n_heads
    else:
        n_ret = int((head_roles == 0).sum().item())
        n_h2o = int((head_roles == 1).sum().item())
        n_lex = int((head_roles == 2).sum().item())
    vec = 4 * head_dim
    win = min(n_b, seq_len)
    hist = max(0, seq_len - win)
    k_kept = min(max(1, int(seq_len * keep_ratio)), hist)
    num = (
        n_heads * win * vec
        + n_ret * hist * vec
        + n_h2o * k_kept * vec
        + n_lex * hist * 2 * (3 * s_budget + 2)
    )
    return float(num) / float(n_heads * seq_len * vec)


def solve_equal_budget_params(
    head_roles: torch.Tensor,
    seq_len: int,
    n_b: int,
    target_ratio: float = 0.20,
    head_dim: int = 128,
    s_max: int = 16,
) -> Dict[str, Dict[str, float]]:
    """
    Picks compression knobs so that every compressed configuration stays at or below
    `target_ratio` of FP16 KV bytes. The rule is fixed in advance (no quality tuning):
      - Standalone H2O:    largest keep_ratio with ratio <= target.
      - Standalone Lexico: largest s <= s_max with ratio <= target.
      - Hybrid:            retrieval heads are paid for first (they are uncompressed);
                           the remaining budget is split with EQUAL BYTE FRACTION PER HEAD
                           between H2O and Lexico heads. With the new 75th percentile
                           entropy threshold, Lexico heads are reduced to ~25% (from ~47.5%),
                           so equal per-head allocation naturally gives more total budget
                           to high-fidelity H2O heads.
    """
    def ratio(mode: str, kr: float, s: int) -> float:
        return predicted_memory_ratio(head_roles, seq_len, n_b, kr, s, mode, head_dim)

    out: Dict[str, Dict[str, float]] = {"full_fp16": {"keep_ratio": 1.0, "s_budget": s_max}}

    kr_grid = [i / 1000.0 for i in range(1000, 0, -1)]
    kr_h2o = next((kr for kr in kr_grid if ratio("standalone_h2o", kr, s_max) <= target_ratio), 0.001)
    out["standalone_h2o"] = {"keep_ratio": kr_h2o, "s_budget": s_max}

    s_lex = next((s for s in range(s_max, 0, -1) if ratio("standalone_lexico", 1.0, s) <= target_ratio), 1)
    out["standalone_lexico"] = {"keep_ratio": 1.0, "s_budget": s_lex}

    hist_frac = max(0, seq_len - min(n_b, seq_len)) / float(seq_len)
    hy = None
    # Equal per-head byte fraction: H2O heads keep same byte fraction as Lexico heads spend.
    # With 75th percentile entropy threshold, Lexico heads are ~25% (vs old ~47.5%),
    # so equal per-head allocation naturally favors H2O majority.
    for s in range(s_max, 0, -1):
        kr = ((3 * s + 2) / (2.0 * head_dim)) * hist_frac
        if ratio("hybrid", kr, s) <= target_ratio:
            hy = {"keep_ratio": round(kr, 4), "s_budget": s}
            break
    if hy is None:
        hy = {"keep_ratio": 0.01, "s_budget": 1}
    out["hybrid"] = hy
    for mode, p in out.items():
        p["predicted_memory_ratio"] = ratio(mode, p["keep_ratio"], int(p["s_budget"]))
    return out


@torch.no_grad()
def evaluate_four_configs_2k(
    model: torch.nn.Module,
    tokenizer: Any,
    artifacts: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluates the 4 required KV cache configurations at ~20% KV byte budget on the
    2,000-token multi-turn Needle-in-a-Haystack (`API_PORT = 8443`) + CSS topic-shift benchmark:
    1. `Full FP16 (100% Memory)`
    2. `Standalone H2O (20% Memory - All Heads Evict)`
    3. `Standalone Lexico (20% Memory - All Heads Encode)`
    4. `StoikovKV Hybrid (Retrieval Guard + Peaked H2O + Diffuse Lexico + HMM Decay)`
    """
    device = next(model.parameters()).device
    port_tid = int(getattr(tokenizer, "port_token_id", 8443))

    # Needle depths across Turn 1 (5%, 12%, 22%, 35% of the 2,000-token prompt)
    needle_depths = [0.05, 0.12, 0.22, 0.35]

    configs = [
    {
        "name": "Full FP16 (100% Memory)",
        "short_name": "Full FP16",
        "policy_mode": "full_fp16",
        "keep_ratio": 1.0,
        "s_budget": 16,
        "n_b": 64,
        "enable_hmm_decay": False,
    },
    {
        "name": "Standalone H2O (20% Memory - All Heads Evict)",
        "short_name": "Standalone H2O",
        "policy_mode": "standalone_h2o",
        "keep_ratio": 0.17,
        "s_budget": 16,
        "n_b": 64,
        "enable_hmm_decay": False,
    },
    {
        "name": "Standalone Lexico (20% Memory - All Heads Encode)",
        "short_name": "Standalone Lexico",
        "policy_mode": "standalone_lexico",
        "keep_ratio": 0.20,
        "s_budget": 16,
        "n_b": 64,
        "enable_hmm_decay": False,
    },
    {
        "name": "StoikovKV Hybrid (Retrieval Guard + Peaked H2O + Diffuse Lexico + HMM Decay)",
        "short_name": "StoikovKV Hybrid",
        "policy_mode": "hybrid",
        "keep_ratio": 0.15,
        "s_budget": 16,
        "n_b": 64,
        "enable_hmm_decay": True,
    },
    ]
    _prompt_gen = (
        build_2k_multiturn_prompt(tokenizer, target_tokens=2000, needle_depth_ratio=d, query_mode="needle")
        for d in needle_depths
    )
    needle_prompts = list(_prompt_gen)
    css_prompt_text, css_prompt_ids, css_meta = build_2k_multiturn_prompt(
        tokenizer, target_tokens=2000, needle_depth_ratio=0.10, query_mode="css"
    )
    css_prompt_ids = css_prompt_ids.to(device)

    # Resolve the published "~20%" target against the *actual* calibrated
    # head-routing split and byte accounting.  Fixed hand-written values drift
    # over budget when the router changes (notably after re-calibration).
    budget_seq_len = int(needle_prompts[0][2]["total_tokens"])
    budget_params = solve_equal_budget_params(
        artifacts["head_roles"].detach().cpu(),
        seq_len=budget_seq_len,
        n_b=64,
        target_ratio=0.20,
        head_dim=int(artifacts["U_Q"].shape[2]),
    )
    for cfg in configs:
        resolved = budget_params[cfg["policy_mode"]]
        cfg["keep_ratio"] = float(resolved["keep_ratio"])
        cfg["s_budget"] = int(resolved["s_budget"])
        print(
            f"[Budget] {cfg['short_name']:<18s} keep_ratio={cfg['keep_ratio']:.4f} "
            f"s={cfg['s_budget']:2d} predicted KV mem="
            f"{100.0 * resolved['predicted_memory_ratio']:.2f}%"
        )

    # First run Full FP16 on the CSS prompt to record ground-truth Turn-2 contextual logits & multi-step generation
    ref_cache = StoikovKVCache(
        model=model,
        artifacts=artifacts,
        policy_mode="full_fp16",
        n_b=64,
    )
    # Feed in chunks of 256 tokens so local window & dynamic routing operate exactly as in streaming/chunked prefill
    chunk_size = 256
    for pos in range(0, css_prompt_ids.shape[1] - 1, chunk_size):
        end_pos = min(css_prompt_ids.shape[1] - 1, pos + chunk_size)
        _ = model(
            input_ids=css_prompt_ids[:, pos:end_pos],
            past_key_values=ref_cache,
            use_cache=True,
            output_attentions=False,
        )
    ref_out = model(
        input_ids=css_prompt_ids[:, -1:],
        past_key_values=ref_cache,
        use_cache=True,
        output_attentions=False,
    )
    base_css_logits = torch.matmul(
        model.model.embed_tokens(css_prompt_ids[:, -1:]),
        model.model.embed_tokens.weight.T,
    )[:, -1, :].detach().float()
    ref_css_logits = ref_out.logits[:, -1, :].detach().float() - base_css_logits

    results_by_config: Dict[str, Any] = {}

    for cfg in configs:
        name = cfg["name"]
        needle_hits = 0
        needle_probs: List[float] = []
        needle_margins: List[float] = []
        mem_ratios: List[float] = []
        mem_ratios_dict: List[float] = []
        dict_bytes_last: int = 0
        needle_per_depth: List[Dict[str, Any]] = []
        comp_bytes_list: List[int] = []
        fp16_bytes_list: List[int] = []
        tier_counts_last: Dict[str, int] = {}
        turn2_h2o_retention_pct: float = 0.0
        hmm_shifts_detected: int = 0

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        for d_idx, (_, p_ids, p_meta) in enumerate(needle_prompts):
            p_ids_dev = p_ids.to(device)
            cache = StoikovKVCache(
                model=model,
                artifacts=artifacts,
                n_b=cfg["n_b"],
                keep_ratio=cfg["keep_ratio"],
                s_budget=cfg["s_budget"],
                policy_mode=cfg["policy_mode"],
                enable_hmm_decay=cfg["enable_hmm_decay"],
            )

            # Stream prompt in 256-token chunks up to final query token
            seq_total = int(p_ids_dev.shape[1])
            for pos in range(0, seq_total - 1, chunk_size):
                end_pos = min(seq_total - 1, pos + chunk_size)
                _ = model(
                    input_ids=p_ids_dev[:, pos:end_pos],
                    past_key_values=cache,
                    use_cache=True,
                    output_attentions=False,
                )

            # Final query step predicting `"8443"`
            out = model(
                input_ids=p_ids_dev[:, -1:],
                past_key_values=cache,
                use_cache=True,
                output_attentions=False,
            )
            logits = out.logits[0, -1].float()
            pred_tid = int(torch.argmax(logits).item())
            hit = int(pred_tid == port_tid)
            needle_hits += hit

            probs = F.softmax(logits, dim=-1)
            target_prob = float(probs[port_tid].item())
            logits_other = logits.clone()
            logits_other[port_tid] = float("-inf")
            margin = float((logits[port_tid] - logits_other.max()).item())

            needle_probs.append(target_prob)
            needle_margins.append(margin)

            telem = cache.get_telemetry_summary()
            mem_ratios.append(float(telem["memory_ratio_vs_fp16"]))
            mem_ratios_dict.append(float(telem["memory_ratio_incl_dictionary"]))
            dict_bytes_last = int(telem["dictionary_bytes"])
            needle_per_depth.append({"depth": float(needle_depths[d_idx]), "correct": bool(hit), "pred_token": pred_tid})
            comp_bytes_list.append(int(telem["compressed_bytes"]))
            fp16_bytes_list.append(int(telem["fp16_bytes"]))
            tier_counts_last = telem["per_tier_token_counts"]
            hmm_shifts_detected = max(hmm_shifts_detected, int(telem["hmm_shift_events"]))

            # Measure what fraction of H2O heavy hitters belong to Turn 2 (>= turn2_start_token)
            t2_start = p_meta["turn2_start_token"]
            t2_kept = 0
            total_h2o_kept = 0
            for l_i in range(cache.num_layers):
                for h_i in range(cache.num_kv_heads):
                    st = cache._tiers[l_i][h_i]
                    if st.role == ROLE_PEAKED_H2O and st.h2o_indices is not None:
                        idx_t = st.h2o_indices[0]
                        t2_kept += int((idx_t >= t2_start).sum().item())
                        total_h2o_kept += int(idx_t.numel())
            if total_h2o_kept > 0:
                turn2_h2o_retention_pct = 100.0 * float(t2_kept) / float(total_h2o_kept)
            elif cfg["policy_mode"] == "full_fp16":
                turn2_h2o_retention_pct = 100.0 * float(seq_total - t2_start) / float(seq_total)

        # Evaluate Turn-2 CSS contextual logit cosine similarity vs Full FP16 reference
        css_cache = StoikovKVCache(
            model=model,
            artifacts=artifacts,
            n_b=cfg["n_b"],
            keep_ratio=cfg["keep_ratio"],
            s_budget=cfg["s_budget"],
            policy_mode=cfg["policy_mode"],
            enable_hmm_decay=cfg["enable_hmm_decay"],
        )
        for pos in range(0, css_prompt_ids.shape[1] - 1, chunk_size):
            end_pos = min(css_prompt_ids.shape[1] - 1, pos + chunk_size)
            _ = model(
                input_ids=css_prompt_ids[:, pos:end_pos],
                past_key_values=css_cache,
                use_cache=True,
                output_attentions=False,
            )
        css_out = model(
            input_ids=css_prompt_ids[:, -1:],
            past_key_values=css_cache,
            use_cache=True,
            output_attentions=False,
        )
        css_logits = css_out.logits[:, -1, :].detach().float() - base_css_logits
        css_logit_cos = float(
            F.cosine_similarity(ref_css_logits, css_logits, dim=-1).mean().item()
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0

        needle_acc = float(needle_hits) / float(len(needle_depths))
        mean_target_prob = float(np.mean(needle_probs))
        # Combined multi-turn quality score (50% Early Needle Retrieval + 50% Turn-2 CSS Context Fidelity)
        composite_quality = 0.50 * (needle_acc * 100.0) + 0.50 * (css_logit_cos * 100.0)

        results_by_config[name] = {
            "short_name": cfg["short_name"],
            "policy_mode": cfg["policy_mode"],
            "prompt_tokens": int(needle_prompts[0][2]["total_tokens"]),
            "memory_ratio_vs_fp16": float(np.mean(mem_ratios)),
            "memory_budget_pct": float(np.mean(mem_ratios)) * 100.0,
            "dictionary_bytes": int(dict_bytes_last),
            "memory_ratio_incl_dictionary": float(np.mean(mem_ratios_dict)),
            "keep_ratio": float(cfg["keep_ratio"]),
            "s_budget": int(cfg["s_budget"]),
            "needle_per_depth_correct": needle_per_depth,
            "fp16_bytes": int(np.mean(fp16_bytes_list)),
            "compressed_bytes": int(np.mean(comp_bytes_list)),
            "needle_passkey_accuracy": needle_acc,
            "needle_passkey_accuracy_pct": needle_acc * 100.0,
            "needle_target_prob": mean_target_prob,
            "needle_logit_margin": float(np.mean(needle_margins)),
            "turn2_css_logit_cosine": css_logit_cos,
            "turn2_h2o_freshness_pct": turn2_h2o_retention_pct,
            "composite_multiturn_quality_pct": composite_quality,
            "hmm_shift_events_detected": hmm_shifts_detected,
            "per_tier_token_counts": tier_counts_last,
            "eval_time_sec": float(elapsed),
        }

    return results_by_config


@torch.no_grad()
def evaluate_matryoshka_sweep(
    model: torch.nn.Module,
    tokenizer: Any,
    artifacts: Dict[str, Any],
    draft_atoms_list: Tuple[int, ...] = (2, 4, 8, 12),
    verify_atoms: int = 16,
    gamma_spec: int = 4,
    max_new_tokens: int = 64,
) -> Dict[str, Any]:
    """
    Sweeps `draft_atoms m in {2, 4, 8, 12}` vs `verify_atoms s = 16` using
    `matryoshka_speculative_generate` on a 2,000-token multi-turn prompt.
    Records Acceptance Rate `alpha`, Average Accepted Tokens per Verify Pass,
    and Effective KV Bytes Read per Token.
    Uses the published Person 2 setting: gamma_spec=4 and verify_atoms=16.
    """
    prompt_text, prompt_ids, prompt_meta = build_2k_multiturn_prompt(
        tokenizer, target_tokens=1024, needle_depth_ratio=0.10, query_mode="needle"
    )

    sweep_records: List[Dict[str, Any]] = []
    for m in draft_atoms_list:
        cache = StoikovKVCache(
            model=model,
            artifacts=artifacts,
            n_b=64,
            keep_ratio=0.15,
            s_budget=verify_atoms,
            policy_mode="hybrid",
            enable_hmm_decay=True,
        )
        res = matryoshka_speculative_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_ids,
            cache=cache,
            gamma_spec=gamma_spec,
            draft_atoms=m,
            verify_atoms=verify_atoms,
            max_new_tokens=max_new_tokens,
            temperature=0.5,
            top_k=32,
            seed=42,
        )
        sweep_records.append(
            {
                "draft_atoms_m": int(m),
                "verify_atoms_s": int(verify_atoms),
                "gamma_spec": int(gamma_spec),
                "max_new_tokens": int(max_new_tokens),
                "draft_lexico_bytes_per_vec": int(3 * m + 2),
                "verify_lexico_bytes_per_vec": int(3 * verify_atoms + 2),
                "acceptance_rate_alpha": float(res["acceptance_rate"]),
                "expected_acceptance_rate_alpha": float(res["expected_acceptance_rate"]),
                "avg_accepted_tokens_per_verify_pass": float(
                    res["avg_accepted_tokens_per_verify_pass"]
                ),
                "avg_tokens_per_cycle": float(res["avg_tokens_per_cycle"]),
                "kv_bytes_read_per_token": float(res["kv_bytes_read_per_token"]),
                "standard_decode_kv_bytes_per_token": float(
                    res["standard_decode_kv_bytes_per_token"]
                ),
                "standard_fp16_kv_bytes_per_token": float(
                    res["standard_fp16_kv_bytes_per_token"]
                ),
                "bandwidth_savings_vs_standard_decode_pct": float(
                    res["bandwidth_savings_vs_standard_decode_pct"]
                ),
                "bandwidth_savings_vs_fp16_pct": float(res["bandwidth_savings_vs_fp16_pct"]),
                "first_generated_token_id": int(res["generated_ids"][0]),
                "passkey_8443_retrieved_in_generation": bool(
                    int(getattr(tokenizer, "port_token_id", 8443)) in res["generated_ids"][:4]
                ),
                "generated_preview": str(res["generated_text"])[:120],
            }
        )

    return {
        "prompt_tokens": int(prompt_meta["total_tokens"]),
        "verify_atoms_s": int(verify_atoms),
        "gamma_spec": int(gamma_spec),
        "max_new_tokens": int(max_new_tokens),
        "sweep": sweep_records,
    }


def plot_pareto_quality_vs_bytes(
    ablation_results: Dict[str, Any],
    save_paths: List[Path],
) -> None:
    """
    Generates `pareto_quality_vs_bytes.png`:
    Left panel: Pareto frontier of Multi-Turn Quality & Needle Passkey Accuracy vs KV Memory Budget (%).
    Right panel: Breakdown of Early Needle Passkey Accuracy ("8443"), Turn-2 CSS Context Fidelity,
                 and Turn-2 H2O Freshness across the 4 configurations.
    """
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "figure.facecolor": "white",
        }
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.8), dpi=180)

    colors = {
        "Full FP16": "#334155",
        "Standalone H2O": "#e11d48",
        "Standalone Lexico": "#d97706",
        "StoikovKV Hybrid": "#059669",
    }
    markers = {
        "Full FP16": "s",
        "Standalone H2O": "v",
        "Standalone Lexico": "^",
        "StoikovKV Hybrid": "*",
    }

    # Panel 1: Pareto Frontier (KV Memory % of FP16 vs Composite Multi-Turn Quality & Needle Accuracy)
    for full_name, item in ablation_results.items():
        short = item["short_name"]
        x_mem = item["memory_budget_pct"]
        y_qual = item["composite_multiturn_quality_pct"]
        y_needle = item["needle_passkey_accuracy_pct"]
        c = colors.get(short, "#2563eb")
        m = markers.get(short, "o")
        sz = 260 if short == "StoikovKV Hybrid" else 140

        ax1.scatter(
            x_mem,
            y_qual,
            s=sz,
            color=c,
            marker=m,
            edgecolors="black",
            linewidths=1.2,
            zorder=5,
            label=f"{short} ({x_mem:.1f}% Mem, Needle={y_needle:.0f}%)",
        )
        offset_y = 3.2 if short != "Standalone Lexico" else -5.5
        offset_x = 1.8 if x_mem < 80 else -18.0
        ax1.annotate(
            f"{short}\n({x_mem:.1f}% KV, Qual={y_qual:.1f}%)",
            (x_mem, y_qual),
            xytext=(x_mem + offset_x, y_qual + offset_y),
            fontsize=8.8,
            fontweight="bold" if short == "StoikovKV Hybrid" else "normal",
            color=c,
        )

    ax1.axvline(20.0, color="#64748b", linestyle="--", linewidth=1.1, label="~20% Target Byte Budget")
    ax1.set_xlim(5, 110)
    ax1.set_ylim(35, 108)
    ax1.set_xlabel("KV Cache Memory Footprint (% of Uncompressed FP16)")
    ax1.set_ylabel("Multi-Turn Quality Score (% Needle + Turn-2 CSS Fidelity)")
    ax1.set_title("Pareto Frontier: Multi-Turn Quality vs. KV Byte Budget (2k Context)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=8.5, framealpha=0.92)

    # Panel 2: Grouped bar chart comparing the 4 configurations
    short_names = [v["short_name"] for v in ablation_results.values()]
    needle_accs = [v["needle_passkey_accuracy_pct"] for v in ablation_results.values()]
    css_fids = [v["turn2_css_logit_cosine"] * 100.0 for v in ablation_results.values()]
    mem_pcts = [v["memory_budget_pct"] for v in ablation_results.values()]

    x = np.arange(len(short_names))
    w = 0.26
    bars1 = ax2.bar(
        x - w,
        needle_accs,
        width=w,
        color="#059669",
        edgecolor="black",
        linewidth=0.8,
        label="Turn-1 Needle ('API_PORT=8443') Acc (%)",
    )
    bars2 = ax2.bar(
        x,
        css_fids,
        width=w,
        color="#2563eb",
        edgecolor="black",
        linewidth=0.8,
        label="Turn-2 CSS Logit Cosine Sim (%)",
    )
    bars3 = ax2.bar(
        x + w,
        mem_pcts,
        width=w,
        color="#94a3b8",
        edgecolor="black",
        linewidth=0.8,
        label="KV Memory Footprint (% vs FP16)",
    )

    for b in list(bars1) + list(bars2) + list(bars3):
        h = b.get_height()
        ax2.text(
            b.get_x() + b.get_width() / 2.0,
            h + 1.2,
            f"{h:.0f}%",
            ha="center",
            va="bottom",
            fontsize=7.8,
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, rotation=10)
    ax2.set_ylim(0, 115)
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title("4-Way Ablation at ~20% KV Budget (2,000-Token Multi-Turn Prompt)")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

    fig.tight_layout()
    for p in save_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_matryoshka_acceptance(
    matryoshka_results: Dict[str, Any],
    save_paths: List[Path],
) -> None:
    """
    Generates `matryoshka_acceptance.png`:
    Plots Acceptance Rate `alpha` vs. Effective KV Bytes Read per Token (and across
    `draft_atoms m in {2, 4, 8, 12}` vs `verify_atoms s = 16`).
    """
    sweep = matryoshka_results["sweep"]
    m_vals = [r["draft_atoms_m"] for r in sweep]
    emp_alphas = [r["acceptance_rate_alpha"] for r in sweep]
    exp_alphas = [r["expected_acceptance_rate_alpha"] for r in sweep]
    kb_per_tok = [r["kv_bytes_read_per_token"] / 1024.0 for r in sweep]
    std_lex_kb = sweep[0]["standard_decode_kv_bytes_per_token"] / 1024.0
    std_fp16_kb = sweep[0]["standard_fp16_kv_bytes_per_token"] / 1024.0
    vec_bytes = [r["draft_lexico_bytes_per_vec"] for r in sweep]

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "figure.facecolor": "white",
        }
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 5.6), dpi=180)

    # Panel 1: Acceptance Rate alpha vs KV Bytes Read per Token (KB/token)
    ax1.plot(
        kb_per_tok,
        emp_alphas,
        marker="o",
        markersize=9,
        linewidth=2.4,
        color="#059669",
        label="Empirical Acceptance Rate $\\alpha$",
    )
    ax1.plot(
        kb_per_tok,
        exp_alphas,
        marker="s",
        markersize=7,
        linewidth=1.8,
        linestyle="--",
        color="#0284c7",
        label="Expected Overlap $E[\\min(1, p/q)]$",
    )

    for m, kb, a_emp, vb in zip(m_vals, kb_per_tok, emp_alphas, vec_bytes):
        ax1.annotate(
            f"m={m} atoms\n({vb} B/vec, $\\alpha$={a_emp:.2f})",
            (kb, a_emp),
            xytext=(kb + 4.0, a_emp - 0.055),
            fontsize=8.8,
            fontweight="bold",
            color="#065f46",
        )

    ax1.axvline(
        std_lex_kb,
        color="#d97706",
        linestyle="-.",
        linewidth=1.5,
        label=(
            f"Standard s={matryoshka_results['verify_atoms_s']} Decode "
            f"({std_lex_kb:.0f} KB/tok)"
        ),
    )
    ax1.set_xlabel("Effective KV Cache Bytes Read per Generated Token (KB / token)")
    ax1.set_ylabel("Speculative Acceptance Rate $\\alpha$")
    ax1.set_ylim(0.35, 1.06)
    ax1.set_xlim(min(kb_per_tok) * 0.88, std_lex_kb * 1.12)
    ax1.set_title("Matryoshka Speculative Decoding: Acceptance $\\alpha$ vs. KV Bandwidth")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=8.8, framealpha=0.92)

    # Panel 2: Dual-axis plot of Draft Atoms m in {2, 4, 8, 12} -> Acceptance Rate & Bandwidth Savings vs FP16
    savings_vs_std = [r["bandwidth_savings_vs_standard_decode_pct"] for r in sweep]
    avg_acc_tokens = [r["avg_accepted_tokens_per_verify_pass"] for r in sweep]

    ax2_twin = ax2.twinx()
    l1 = ax2.plot(
        m_vals,
        emp_alphas,
        marker="o",
        markersize=8,
        linewidth=2.3,
        color="#059669",
        label="Acceptance Rate $\\alpha$ (left)",
    )
    l2 = ax2_twin.plot(
        m_vals,
        savings_vs_std,
        marker="^",
        markersize=8,
        linewidth=2.2,
        linestyle="--",
        color="#dc2626",
        label="HBM Read Savings vs s=16 Decode % (right)",
    )

    for m, a_emp, n_acc in zip(m_vals, emp_alphas, avg_acc_tokens):
        ax2.annotate(
            f"{n_acc:.2f}/4 accepted",
            (m, a_emp),
            xytext=(m - 0.6, a_emp + 0.03),
            fontsize=8.3,
            color="#065f46",
        )

    ax2.set_xticks(m_vals)
    ax2.set_xticklabels([f"m={m}\n({3*m+2} B/vec)" for m in m_vals])
    ax2.set_xlabel("Draft Dictionary Atoms $m$ (Verified against $s=16$, $\\gamma=4$)")
    ax2.set_ylabel("Speculative Acceptance Rate $\\alpha$", color="#059669")
    ax2_twin.set_ylabel("KV Bandwidth Savings vs Standard s=16 Decode (%)", color="#dc2626")
    ax2.set_ylim(0.35, 1.08)
    ax2_twin.set_ylim(0.0, max(savings_vs_std) * 1.28)
    ax2.set_title("Draft Atom Sweep $m \\in \\{2, 4, 8, 12\\}$ vs. Verify $s = 16$")
    ax2.grid(True, linestyle=":", alpha=0.6)

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="center right", fontsize=8.5, framealpha=0.92)

    fig.tight_layout()
    for p in save_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("=" * 84)
    print("StoikovKV Person 2 Live Inference & Matryoshka Speculative Decoding Benchmark")
    print("=" * 84)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    loaded_pair = load_model_and_tokenizer(device=device, dtype=dtype)
    if isinstance(loaded_pair, tuple) and len(loaded_pair) == 3:
        model, tokenizer, backend_info = loaded_pair
    else:
        model, tokenizer = loaded_pair
        backend_info = {
            "attn_implementation": getattr(model.config, "_attn_implementation", "sdpa"),
            "output_attentions_used_in_live_inference": False,
        }
    print(
        f"[Model] {getattr(model.config, '_name_or_path', 'Qwen2.5')} | "
        f"device={device} | dtype={dtype} | GQA: {model.config.num_attention_heads} Q / "
        f"{model.config.num_key_value_heads} KV heads"
    )

    artifacts_path = str(ROOT_DIR / "calibrated_artifacts.pt")
    if os.path.exists(artifacts_path):
        print(f"[Artifacts] Loading calibrated artifacts from {artifacts_path}")
        artifacts = load_or_create_artifacts(model, artifacts_path=artifacts_path, tokenizer=tokenizer)
    else:
        print("[Artifacts] calibrated_artifacts.pt not found; running create_mock_artifacts(model)...")
        artifacts = create_mock_artifacts(model, tokenizer=tokenizer, save_path=artifacts_path)

    # Also test standalone create_mock_artifacts(model) explicitly to verify the Independence Rule
    # Validate factory compatibility at the calibrated artifact dimensions.  This
    # must not hard-code 1024 atoms when Person 1 calibrated a 128-atom dictionary.
    mock_artifacts = create_mock_artifacts(
        model,
        tokenizer=tokenizer,
        dict_atoms=int(artifacts["D"].shape[-1]),
        subspace_rank=int(artifacts["U_Q"].shape[-1]),
    )
    assert mock_artifacts["U_Q"].shape == artifacts["U_Q"].shape
    assert mock_artifacts["D"].shape == artifacts["D"].shape
    assert mock_artifacts["W_ridge"].shape == artifacts["W_ridge"].shape
    print(
        f"[Independence Check] create_mock_artifacts(model) verified: "
        f"U_Q={tuple(mock_artifacts['U_Q'].shape)}, D={tuple(mock_artifacts['D'].shape)}, "
        f"W_ridge={tuple(mock_artifacts['W_ridge'].shape)}"
    )

    # 1. Run 4-Configuration Ablation at ~20% KV Byte Budget on 2,000-Token Multi-Turn Prompt
    print("\n" + "-" * 84)
    print("1. 4-Way KV Cache Ablation on 2,000-Token Multi-Turn Prompt ('API_PORT = 8443' + CSS Shift)")
    print("-" * 84)
    ablation_results = evaluate_four_configs_2k(model, tokenizer, artifacts)

    header = (
        f"{'Configuration':<20} | {'KV Mem %':>8} | {'+Dict %':>8} | {'k / s':>11} | {'Needle 8443':>11} | "
        f"{'CSS CosSim':>10} | {'Turn-2 H2O':>10}"
    )
    print(header)
    print("-" * len(header))
    for cfg_name, metrics in ablation_results.items():
        print(
            f"{metrics['short_name']:<20} | "
            f"{metrics['memory_budget_pct']:>7.1f}% | "
            f"{100.0 * metrics['memory_ratio_incl_dictionary']:>7.1f}% | "
            f"{metrics['keep_ratio']:>5.3f}/{metrics['s_budget']:<5d} | "
            f"{metrics['needle_passkey_accuracy_pct']:>10.1f}% | "
            f"{metrics['turn2_css_logit_cosine']:>10.4f} | "
            f"{metrics['turn2_h2o_freshness_pct']:>9.1f}%"
        )

    # Validity gate: the uncompressed baseline is the upper bound. If it cannot retrieve
    # the needle, the needle column measures the model/prompt, not the compression method.
    fp16_metrics = next(
        (m for m in ablation_results.values() if m["policy_mode"] == "full_fp16"), None
    )
    needle_test_valid = bool(fp16_metrics and fp16_metrics["needle_passkey_accuracy"] >= 1.0)
    over_budget = [
        m["short_name"]
        for m in ablation_results.values()
        if m["policy_mode"] != "full_fp16" and m["memory_ratio_vs_fp16"] > 0.2005
    ]
    print(f"\n[Validity] Full FP16 needle baseline passes: {needle_test_valid}")
    print(f"[Validity] Configs over the 20% KV budget: {over_budget or 'none'}")
    if not needle_test_valid:
        print("[Validity] WARNING: needle results are NOT interpretable for this run.")

    # 2. Run the report-compatible Matryoshka sweep.
    print("\n" + "-" * 84)
    print("2. Matryoshka Self-Speculative Decoding Sweep (m in {2, 4, 8, 12} vs s = 16, gamma = 4)")
    print("-" * 84)
    matryoshka_results = evaluate_matryoshka_sweep(
        model,
        tokenizer,
        artifacts,
        draft_atoms_list=(2, 4, 8, 12),
        verify_atoms=16,
        gamma_spec=4,
        max_new_tokens=64,
    )

    m_header = (
        f"{'Draft Atoms m':<15} | {'Bytes/Vec':>9} | {'Accept alpha':>12} | "
        f"{'Avg Accepted':>12} | {'KB Read/Tok':>12} | {'Read Savings':>12}"
    )
    print(m_header)
    print("-" * len(m_header))
    for row in matryoshka_results["sweep"]:
        print(
            f"m = {row['draft_atoms_m']:<11} | "
            f"{row['draft_lexico_bytes_per_vec']:>7} B | "
            f"{row['acceptance_rate_alpha']:>12.4f} | "
            f"{row['avg_accepted_tokens_per_verify_pass']:>10.2f}/"
            f"{row['gamma_spec']:<1d} | "
            f"{row['kv_bytes_read_per_token'] / 1024.0:>9.1f} KB | "
            f"{row['bandwidth_savings_vs_standard_decode_pct']:>10.1f}%"
        )

    # 3. Save Plots & JSON Results
    pareto_paths = [
        ROOT_DIR / "pareto_quality_vs_bytes.png",
        ROOT_DIR / "benchmarks" / "pareto_quality_vs_bytes.png",
    ]
    matryoshka_paths = [
        ROOT_DIR / "matryoshka_acceptance.png",
        ROOT_DIR / "benchmarks" / "matryoshka_acceptance.png",
    ]
    json_paths = [
        ROOT_DIR / "eval_results.json",
        ROOT_DIR / "benchmarks" / "eval_results.json",
    ]

    plot_pareto_quality_vs_bytes(ablation_results, pareto_paths)
    plot_matryoshka_acceptance(matryoshka_results, matryoshka_paths)

    payload = {
        "project": "StoikovKV",
        "module": "Person 2 Live Inference Engine & Matryoshka Self-Speculative Decoding",
        "device": device,
        "dtype": str(dtype),
        "model_name": getattr(model.config, "_name_or_path", "Qwen/Qwen2.5-1.5B-Instruct"),
        "backend_info": backend_info,
        "independence_mock_artifacts_verified": True,
        "validity": {
            "full_fp16_needle_baseline_passes": needle_test_valid,
            "configs_over_20pct_kv_budget": over_budget,
            "budget_rule": "equal-memory solver (solve_equal_budget_params), target <= 20% of FP16 KV bytes",
            "model_is_synthetic_fallback": "Standalone" in str(
                getattr(model.config, "_name_or_path", "")
            ),
        },
        "four_config_ablation_2k_context": ablation_results,
        "matryoshka_speculative_sweep": matryoshka_results,
    }
    for jp in json_paths:
        jp.parent.mkdir(parents=True, exist_ok=True)
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    print("\n" + "=" * 84)
    print("Saved artifacts:")
    print(f"  - {pareto_paths[0]}")
    print(f"  - {matryoshka_paths[0]}")
    print(f"  - {json_paths[0]}")
    print("=" * 84)


if __name__ == "__main__":
    main()
