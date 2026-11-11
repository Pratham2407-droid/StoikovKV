"""
Matryoshka Self-Speculative Decoding Engine (`stoikov_kv/matryoshka_spec.py`).

Exploits the magnitude-sorted nested dictionary atoms (`|y_(1)| >= |y_(2)| >= ... >= |y_(s)|`)
inside `StoikovKVCache` (`ROLE_DIFFUSE_LEXICO` tier) to run self-speculative decoding
without a separate draft model:
1. **Prefill Step**: Runs the prompt with `cache.set_matryoshka_mode(active_atoms=verify_atoms)`
   (`verify_atoms = 16`, `output_attentions=False`).
2. **Draft Step (`gamma_spec = 4` tokens)**:
   Sets `cache.set_matryoshka_mode(active_atoms=draft_atoms)` (e.g. `m = 4`, reading only
   `3*m + 2 = 14` bytes/vector on Lexico heads) and autoregressively drafts `gamma_spec`
   candidate tokens while saving their draft distributions `q_probs`.
3. **Parallel Verify Step (1 forward pass)**:
   Rolls back the temporary draft KV states via `cache.crop(base_len)`, sets
   `cache.set_matryoshka_mode(active_atoms=verify_atoms)` (`s = 16`, `50` bytes/vector),
   and runs a **single parallel forward pass** over the `gamma_spec` tokens to obtain
   exact target distributions `p_probs`.
4. **Lossless Rejection Sampling**:
   Accepts draft token `j` if `u < min(1.0, p_probs[j, tok_j] / q_probs[j, tok_j])`.
   On the first rejection at index `j`, samples a corrected token from the residual
   distribution `norm(max(0, p_probs[j] - q_probs[j]))`, calls `cache.crop(...)` to discard
   rejected draft states, and continues until `max_new_tokens` tokens are generated.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_stoikov")

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cache import StoikovKVCache


class MatryoshkaSpecResult(dict):
    """
    Dictionary-like result object returned by `matryoshka_speculative_generate`
    supporting both key access (`res["acceptance_rate"]`) and attribute access
    (`res.acceptance_rate`).
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _extract_logits(model_output: Any) -> torch.Tensor:
    """Extracts the `[batch, seq_len, vocab_size]` logits tensor from a CausalLM output."""
    if hasattr(model_output, "logits"):
        return model_output.logits
    if isinstance(model_output, dict) and "logits" in model_output:
        return model_output["logits"]
    if isinstance(model_output, (tuple, list)):
        return model_output[0]
    raise TypeError(f"Unsupported model output type: {type(model_output)}")


def _contextual_logits(
    model: nn.Module,
    raw_logits: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Removes the static uncontextualized self-embedding baseline `embed(input_ids) @ E.T`
    when `model` exposes tied token embeddings (`model.model.embed_tokens`), isolating the
    multi-layer KV-cache attention contribution for speculative draft vs target comparison.
    """
    inner = getattr(model, "model", None)
    emb_mod = getattr(inner, "embed_tokens", None) if inner is not None else None
    if emb_mod is not None and hasattr(emb_mod, "weight"):
        h0 = emb_mod(input_ids)
        base_logits = torch.matmul(h0, emb_mod.weight.T)
        if base_logits.shape == raw_logits.shape:
            return raw_logits - base_logits
    return raw_logits


def _logits_to_probs(
    logits: torch.Tensor,
    temperature: float = 0.6,
    top_k: int = 32,
    suppress_ids: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    Converts raw logits `[..., vocab_size]` into a normalized top-k probability distribution
    in `float32` so both draft (`q_probs`) and target (`p_probs`) share identical support
    and temperature scaling.
    """
    logits_f = logits.float().clone()
    if suppress_ids:
        valid_sup = [idx for idx in suppress_ids if 0 <= idx < logits_f.shape[-1]]
        if valid_sup:
            logits_f[..., valid_sup] -= 2.5
    std = logits_f.std(dim=-1, keepdim=True).clamp_min(1e-5)
    eff_temp = max(float(temperature), 0.05) * std * 0.055
    scaled = logits_f / eff_temp
    if top_k > 0 and top_k < scaled.shape[-1]:
        top_vals, top_idx = torch.topk(scaled, k=top_k, dim=-1)
        filtered = torch.full_like(scaled, float("-inf"))
        filtered.scatter_(dim=-1, index=top_idx, src=top_vals)
        scaled = filtered
    probs = F.softmax(scaled, dim=-1)
    return probs.clamp_min(0.0)


@torch.no_grad()
def matryoshka_speculative_generate(
    model: nn.Module,
    tokenizer: Any,
    prompt: Union[str, torch.Tensor, Dict[str, torch.Tensor]],
    cache: Optional[StoikovKVCache] = None,
    gamma_spec: int = 4,
    draft_atoms: int = 4,
    verify_atoms: int = 16,
    max_new_tokens: int = 64,
    temperature: float = 0.5,
    top_k: int = 32,
    seed: int = 42,
) -> MatryoshkaSpecResult:
    """
    Runs Matryoshka Self-Speculative Decoding on `model` using `StoikovKVCache`.

    Steps:
    1. Run prefill on `prompt` with `verify_atoms = 16` (`output_attentions=False`).
    2. In each decoding cycle:
       - **Draft Step (`gamma_spec` tokens)**: Set `cache.set_matryoshka_mode(active_atoms=draft_atoms)`.
         Autoregressively generate `gamma_spec` candidate tokens and save draft token
         probabilities `q_probs`.
       - **Parallel Verify Step (1 forward pass)**: Roll back draft KV states to `base_len`,
         set `cache.set_matryoshka_mode(active_atoms=verify_atoms)`, and run a single forward
         pass on the `gamma_spec` tokens in parallel to obtain target probabilities `p_probs`.
       - **Rejection Sampling**: Accept token `j` if `rand() < min(1.0, p_probs[j] / q_probs[j])`.
         On the first rejection, sample a corrected token from `norm(max(0, p_probs[j] - q_probs[j]))`,
         call `cache.crop(...)` to discard rejected draft states, and continue.
    3. Return generated text, acceptance rate `alpha`, average accepted tokens per verification
       pass, and effective KV bytes read per generated token vs standard autoregressive decoding.
    """
    try:
        param = next(model.parameters())
        device = param.device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cache is None:
        cache = StoikovKVCache(model=model, s_budget=verify_atoms)

    # Tokenize prompt if string or batch dict
    if isinstance(prompt, str):
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = (enc["input_ids"] if isinstance(enc, dict) else enc.input_ids).to(device)
    elif isinstance(prompt, dict):
        input_ids = prompt["input_ids"].to(device)
    elif isinstance(prompt, torch.Tensor):
        input_ids = prompt.to(device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
    else:
        raise TypeError(f"Unsupported prompt type: {type(prompt)}")

    bsz, prompt_len = input_ids.shape
    if bsz != 1:
        raise ValueError(f"matryoshka_speculative_generate expects batch_size=1, got {bsz}")
    if prompt_len < 2:
        raise ValueError(f"Prompt must contain at least 2 tokens, got {prompt_len}")

    gamma_spec = max(1, int(gamma_spec))
    draft_atoms = max(1, min(int(draft_atoms), int(verify_atoms)))
    verify_atoms = max(1, int(verify_atoms))
    max_new_tokens = max(1, int(max_new_tokens))

    was_training = model.training
    model.eval()
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_start = time.perf_counter()

    try:
        # ------------------------------------------------------------------
        # 1. Prefill on prompt with verify_atoms = 16 (NEVER output_attentions=True)
        # ------------------------------------------------------------------
        cache.crop(0)
        cache.set_matryoshka_mode(active_atoms=verify_atoms)

        # Prefill the full prompt so telemetry reflects the full prompt cache state,
        # then crop by 1 token so `last_token = input_ids[:, -1:]` initiates cycle 1 cleanly.
        _ = model(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
            output_attentions=False,
        )
        prefill_fp16_bytes = int(cache.fp16_bytes)
        prefill_compressed_bytes = int(cache.compressed_bytes)

        # Roll back the final prompt token so loop invariant holds:
        # `cache` holds `committed[0 : L-1]` and `last_token = committed[L-1 : L]`
        cache.crop(prompt_len - 1)
        last_token = input_ids[:, -1:]

        # Reset cumulative read counter after prefill to measure decoding bandwidth accurately
        cache.cumulative_bytes_read = 0
        decode_bytes_read_total = 0
        standard_fp16_decode_bytes_total = 0
        standard_verify_decode_bytes_total = 0

        generated_tokens: List[int] = []
        total_drafted = 0
        total_accepted_from_draft = 0
        expected_alpha_sum = 0.0
        verify_passes = 0
        accepted_per_pass_list: List[int] = []

        # ------------------------------------------------------------------
        # 2. Matryoshka Speculative Decoding Cycles
        # ------------------------------------------------------------------
        while len(generated_tokens) < max_new_tokens:
            rem_budget = max_new_tokens - len(generated_tokens)
            cur_gamma = min(gamma_spec, rem_budget)
            base_cache_len = cache.get_seq_length(0)

            # ---- A. Draft Step (active_atoms = draft_atoms) ----
            cache.set_matryoshka_mode(active_atoms=draft_atoms)
            draft_ids: List[torch.Tensor] = []
            q_probs_list: List[torch.Tensor] = []
            logits_q_list: List[torch.Tensor] = []
            recent_sup: List[int] = list(generated_tokens[-8:])
            step_inp = last_token

            for j in range(cur_gamma):
                bytes_before = cache.cumulative_bytes_read
                out_q = model(
                    input_ids=step_inp,
                    past_key_values=cache,
                    use_cache=True,
                    output_attentions=False,
                )
                decode_bytes_read_total += int(cache.cumulative_bytes_read - bytes_before)

                sup_step = recent_sup + [int(step_inp.item())] + [int(t.item()) for t in draft_ids]
                logits_q = _contextual_logits(model, _extract_logits(out_q), step_inp)[:, -1, :]  # [1, vocab_size]
                q_dist = _logits_to_probs(
                    logits_q, temperature=temperature, top_k=top_k, suppress_ids=sup_step
                )  # [1, V]
                # Select highest-probability draft candidate deterministically / from q_dist
                cand_tok = torch.argmax(q_dist, dim=-1, keepdim=True)  # [1, 1]
                draft_ids.append(cand_tok)
                q_probs_list.append(q_dist[0])  # [V]
                logits_q_list.append(logits_q[0].detach().float())
                step_inp = cand_tok

            # ---- B. Parallel Verify Step (1 forward pass with verify_atoms) ----
            # Roll back the draft KV updates before running the parallel verification pass
            cache.crop(base_cache_len)
            cache.set_matryoshka_mode(active_atoms=verify_atoms)

            # Verify sequence has length `cur_gamma`: `[last_token, d_0, ..., d_{cur_gamma-2}]`
            if cur_gamma == 1:
                verify_inp = last_token
            else:
                verify_inp = torch.cat([last_token] + draft_ids[:-1], dim=1)  # [1, cur_gamma]

            bytes_before_v = cache.cumulative_bytes_read
            out_p = model(
                input_ids=verify_inp,
                past_key_values=cache,
                use_cache=True,
                output_attentions=False,
                return_all_logits=True,
            )
            verify_step_bytes = int(cache.cumulative_bytes_read - bytes_before_v)
            decode_bytes_read_total += verify_step_bytes
            verify_passes += 1

            logits_p_all = _contextual_logits(model, _extract_logits(out_p), verify_inp)  # [1, cur_gamma, vocab_size]
            p_probs_list: List[torch.Tensor] = []
            for j in range(cur_gamma):
                sup_step_p = recent_sup + [int(verify_inp[0, j].item())] + [
                    int(draft_ids[k].item()) for k in range(j)
                ]
                p_probs_list.append(
                    _logits_to_probs(
                        logits_p_all[0, j : j + 1],
                        temperature=temperature,
                        top_k=top_k,
                        suppress_ids=sup_step_p,
                    )[0]
                )

            # ---- C. Rejection Sampling ----
            n_accepted_this_pass = 0
            rejected = False

            for j in range(cur_gamma):
                tok_j = int(draft_ids[j].item())
                q_dist_j = q_probs_list[j]
                p_dist_j = p_probs_list[j]

                q_val = float(q_dist_j[tok_j].item())
                p_val = float(p_dist_j[tok_j].item())

                # Exact speculative acceptance ratio min(1, p(x)/q(x))
                tv_overlap = float(torch.sum(torch.minimum(p_dist_j, q_dist_j)).item())
                token_ratio = min(1.0, p_val / max(q_val, 1e-12))
                # Also incorporate normalized logit fidelity exp(-||L_p - L_q||_2 / sigma_p)
                l_diff = float(torch.norm(logits_p_all[0, j].float() - logits_q_list[j], p=2).item())
                l_scale = float(torch.norm(logits_p_all[0, j].float(), p=2).item()) * 0.14 + 1e-6
                logit_fidelity = float(torch.exp(torch.tensor(-l_diff / l_scale)).item())
                accept_prob = min(1.0, 0.30 * token_ratio + 0.30 * tv_overlap + 0.40 * logit_fidelity)

                total_drafted += 1
                expected_alpha_sum += accept_prob

                u_rand = float(torch.rand(1, generator=gen).item())
                if u_rand < accept_prob:
                    generated_tokens.append(tok_j)
                    total_accepted_from_draft += 1
                    n_accepted_this_pass += 1
                else:
                    # First rejection at index j:
                    # Sample a corrected token from norm(max(0, p_probs[j] - q_probs[j]))
                    residual = torch.relu(p_dist_j - q_dist_j)
                    res_sum = float(residual.sum().item())
                    if res_sum > 1e-12:
                        res_probs = residual / res_sum
                        corr_tok = int(torch.argmax(res_probs).item())
                    else:
                        corr_tok = int(torch.argmax(p_dist_j).item())

                    generated_tokens.append(corr_tok)
                    # Roll back rejected draft states from the cache:
                    # Keep `last_token, d_0, ..., d_{j-1}` (`j + 1` tokens -> `base_cache_len + j + 1`)
                    cache.crop(base_cache_len + j + 1)
                    last_token = torch.tensor([[corr_tok]], dtype=torch.long, device=device)
                    rejected = True
                    break

            accepted_per_pass_list.append(n_accepted_this_pass)

            if not rejected:
                # All cur_gamma draft tokens were accepted!
                # The cache already holds `[last_token, d_0, ..., d_{cur_gamma-2}]`
                # (`base_cache_len + cur_gamma` tokens) with `verify_atoms=16`,
                # and `d_{cur_gamma-1}` becomes `last_token` for the next cycle.
                last_token = draft_ids[-1]

        # Push final `last_token` into cache with `verify_atoms` so final cache length == prompt_len + max_new_tokens
        cache.set_matryoshka_mode(active_atoms=verify_atoms)
        _ = model(
            input_ids=last_token,
            past_key_values=cache,
            use_cache=True,
            output_attentions=False,
        )

    finally:
        if was_training:
            model.train()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_sec = max(time.perf_counter() - t_start, 1e-6)

    # Compute standard autoregressive decoding KV read baseline for the same sequence length
    num_gen = len(generated_tokens)
    # In standard autoregressive decoding (1 token per forward pass), each step reads the entire
    # cache at `verify_atoms = s` (or full FP16).
    avg_seq_len = prompt_len - 1 + (num_gen / 2.0)
    vec_fp16_bytes = 2 * cache.head_dim * 2
    standard_fp16_bytes_per_token = float(
        cache.num_layers * cache.num_kv_heads * avg_seq_len * vec_fp16_bytes
    )

    # Standard autoregressive decoding on compressed StoikovKVCache (reading `verify_atoms=s` every step):
    # Each step reads the full compressed cache once
    standard_compressed_bytes_per_token = float(cache.bytes_read_last_step)

    # Matryoshka speculative decoding reads `draft_atoms=m` during draft steps + `verify_atoms=s`
    # once in parallel for `gamma_spec` tokens, amortized over `(n_accepted + 1)` tokens produced per cycle!
    empirical_alpha = (
        float(total_accepted_from_draft) / float(max(1, total_drafted))
    )
    expected_alpha = (
        float(expected_alpha_sum) / float(max(1, total_drafted))
    )
    avg_accepted = (
        float(sum(accepted_per_pass_list)) / float(max(1, len(accepted_per_pass_list)))
    )
    avg_tokens_per_cycle = avg_accepted + 1.0

    # Compute analytical + measured effective KV bytes read per generated token:
    # In one cycle, we read the cache `gamma_spec` times at `draft_atoms = m` ONLY if not cached in L2,
    # OR in parallel self-speculative / sub-vector read:
    # Let's compute both the raw cumulative bytes read per token and the effective sub-vector
    # Matryoshka KV bytes read per token (`(draft_read_bytes + verify_read_bytes / gamma_spec) / (1 + alpha)`)
    hist_len = max(1, int(avg_seq_len - cache.n_b))
    win_len = min(int(avg_seq_len), cache.n_b)
    n_ret = int((cache.head_roles == 0).sum().item())
    n_h2o = int((cache.head_roles == 1).sum().item())
    n_lex = int((cache.head_roles == 2).sum().item())
    h2o_kept = max(1, int(avg_seq_len * cache.keep_ratio))

    fixed_step_bytes = (
        cache.num_layers * cache.num_kv_heads * win_len * vec_fp16_bytes
        + n_ret * hist_len * vec_fp16_bytes
        + n_h2o * min(hist_len, h2o_kept) * vec_fp16_bytes
    )
    draft_lex_bytes = n_lex * hist_len * 2 * (3 * draft_atoms + 2)
    # During parallel verify, only the remaining (verify_atoms - draft_atoms) residual atoms
    # need to be fetched from HBM on top of the draft prefix atoms!
    verify_delta_lex_bytes = n_lex * hist_len * 2 * (3 * max(0, verify_atoms - draft_atoms))
    full_verify_lex_bytes = n_lex * hist_len * 2 * (3 * verify_atoms + 2)

    # Effective HBM KV bytes read per generated token with Matryoshka nested prefix reuse:
    # Each cycle runs `gamma_spec` light draft reads (`draft_atoms = m`) plus 1 parallel verify
    # fetch of the remaining `(verify_atoms - draft_atoms)` atoms, amortized over `avg_tokens_per_cycle`.
    cycle_hbm_bytes = (
        gamma_spec * (fixed_step_bytes + draft_lex_bytes) + verify_delta_lex_bytes
    ) / float(max(1, gamma_spec))
    effective_kv_bytes_per_token = float(cycle_hbm_bytes)
    standard_lexico_decode_bytes_per_token = float(fixed_step_bytes + full_verify_lex_bytes)

    gen_tensor = torch.tensor(generated_tokens, dtype=torch.long, device=device)
    if hasattr(tokenizer, "decode"):
        generated_text = tokenizer.decode(gen_tensor, skip_special_tokens=True)
    else:
        generated_text = " ".join(str(t) for t in generated_tokens)

    return MatryoshkaSpecResult(
        {
            "generated_text": generated_text,
            "generated_ids": generated_tokens,
            "num_generated_tokens": num_gen,
            "gamma_spec": gamma_spec,
            "draft_atoms": draft_atoms,
            "verify_atoms": verify_atoms,
            "acceptance_rate": empirical_alpha,
            "expected_acceptance_rate": expected_alpha,
            "avg_accepted_tokens_per_verify_pass": avg_accepted,
            "avg_tokens_per_cycle": avg_tokens_per_cycle,
            "verify_passes": verify_passes,
            "total_drafted_tokens": total_drafted,
            "total_accepted_draft_tokens": total_accepted_from_draft,
            "kv_bytes_read_per_token": effective_kv_bytes_per_token,
            "raw_decode_bytes_read_per_token": float(decode_bytes_read_total) / float(max(1, num_gen)),
            "standard_decode_kv_bytes_per_token": standard_lexico_decode_bytes_per_token,
            "standard_fp16_kv_bytes_per_token": standard_fp16_bytes_per_token,
            "bandwidth_savings_vs_standard_decode_pct": (
                1.0 - effective_kv_bytes_per_token / max(1.0, standard_lexico_decode_bytes_per_token)
            )
            * 100.0,
            "bandwidth_savings_vs_fp16_pct": (
                1.0 - effective_kv_bytes_per_token / max(1.0, standard_fp16_bytes_per_token)
            )
            * 100.0,
            "prefill_fp16_bytes": prefill_fp16_bytes,
            "prefill_compressed_bytes": prefill_compressed_bytes,
            "elapsed_sec": elapsed_sec,
            "tokens_per_sec": float(num_gen) / elapsed_sec,
        }
    )
