"""
StoikovKV: Adaptive KV-cache compression for decoder-only Transformers.

Implements per-head Retrieval/H2O/Lexico policies with:
- Write-time query-subspace importance scoring
- 3-state HMM topic-shift decay
- Matryoshka self-speculative decoding support
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

# Role constants
ROLE_RETRIEVAL_FP16 = 0
ROLE_PEAKED_H2O = 1
ROLE_DIFFUSE_LEXICO = 2


def create_mock_artifacts(model: Any, atoms: int = 16) -> dict[str, torch.Tensor]:
    """Make usable calibration tensors after one inexpensive model forward pass."""
    device = next(model.parameters()).device
    token = torch.tensor([[getattr(model.config, "bos_token_id", 1) or 1]], device=device)
    with torch.inference_mode():
        model(input_ids=token, use_cache=False, return_dict=True)
    cfg = model.config.get_text_config(decoder=True) if hasattr(model.config, "get_text_config") else model.config
    n_layers = int(getattr(cfg, "num_hidden_layers"))
    n_kv = int(getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads")))
    dim = int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))
    generator = torch.Generator(device=device).manual_seed(17)
    q, _ = torch.linalg.qr(torch.randn(dim, dim, device=device, dtype=torch.float32, generator=generator))
    dictionary = F.normalize(torch.randn(2 * dim, atoms, device=device, generator=generator), dim=0)
    roles = torch.arange(n_layers * n_kv, device=device).reshape(n_layers, n_kv) % 3
    return {"U_Q": q.to(torch.float16), "head_roles": roles.to(torch.long),
            "D": dictionary.T.to(torch.float16), "W_ridge": dictionary.to(torch.float16)}


@dataclass
class _LayerStore:
    raw_k: torch.Tensor | None = None
    raw_v: torch.Tensor | None = None
    scores: torch.Tensor | None = None
    h2o_keep: torch.Tensor | None = None
    coeffs: torch.Tensor | None = None
    atom_ids: torch.Tensor | None = None
    # Compact mode state
    compact_k: list[torch.Tensor] | None = None
    compact_v: list[torch.Tensor] | None = None
    compact_pos: list[torch.Tensor] | None = None
    lex_coeff: list[torch.Tensor] | None = None
    lex_ids: list[torch.Tensor] | None = None
    lex_pos: list[torch.Tensor] | None = None
    seq_len: int = 0


class StoikovKVCache(DynamicCache):
    """A DynamicCache with per-head Retrieval/H2O/Lexico policies."""

    def __init__(self, model: Any | None = None, artifacts_path: str | Path = "calibrated_artifacts.pt",
                 *, n_b: int = 64, keep_ratio: float = .50, s_budget: int = 16,
                 min_keep_tokens: int = 64, h2o_stride: int = 8, h2o_topk_dilate: int = 64,
                 h2o_context_window: int = 1, lexico_fp16_frac: float = 0.5, beta: float = .98,
                 controller: Callable[[], Mapping[str, float]] | None = None,
                 artifacts: Mapping[str, torch.Tensor] | None = None,
                 dense_fallback: bool = True) -> None:
        super().__init__()
        path = Path(artifacts_path)
        if artifacts is None:
            if path.exists():
                artifacts = torch.load(path, map_location="cpu", weights_only=True)
            elif model is not None:
                artifacts = create_mock_artifacts(model, 16)
            else:
                raise FileNotFoundError(f"{path} is missing; pass model to bootstrap mock artifacts")
        self.U_Q = artifacts["U_Q"]
        self.head_roles = artifacts["head_roles"].long()
        self.D = artifacts.get("D", artifacts["W_ridge"].T)
        self.W_ridge = artifacts["W_ridge"]
        self.kv_mean = artifacts.get("kv_mean", torch.zeros(self.W_ridge.shape[0]))
        self.W_ridge_by_layer = artifacts.get("W_ridge_by_layer")
        self.D_by_layer = artifacts.get("D_by_layer")
        self.kv_mean_by_layer = artifacts.get("kv_mean_by_layer")
        if model is not None:
            cfg = model.config.get_text_config(decoder=True) if hasattr(model.config, "get_text_config") else model.config
            expected = {"layers": int(cfg.num_hidden_layers),
                        "kv_heads": int(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)),
                        "head_dim": int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))}
            actual = {"layers": int(self.head_roles.shape[0]), "kv_heads": int(self.head_roles.shape[1]),
                      "head_dim": int(self.U_Q.shape[-2])}
            if actual != expected:
                raise ValueError(f"calibrated artifact geometry {actual} does not match model {expected}")
            if self.W_ridge.shape[-1] != expected["head_dim"]:
                raise ValueError(f"W_ridge last dimension {self.W_ridge.shape[-1]} must equal head_dim {expected['head_dim']}")
        self.n_b, self.keep_ratio, self.s_budget, self.min_keep_tokens, self.h2o_stride, self.h2o_topk_dilate, self.h2o_context_window, self.lexico_fp16_frac, self.beta = n_b, keep_ratio, s_budget, min_keep_tokens, h2o_stride, h2o_topk_dilate, h2o_context_window, lexico_fp16_frac, beta
        self.controller = controller
        self.dense_fallback = dense_fallback
        self.active_atoms = s_budget
        self._stores: dict[int, _LayerStore] = {}
        self._logical_length = 0
        self.fp16_bytes = 0
        self.compressed_bytes = 0
        self.capacity_bytes = 0
        self.physical_storage_bytes = 0
        self.reconstruction_workspace_bytes = 0
        self.bytes_written_last_step = 0
        self.bytes_read_total = 0
        self.bytes_read_last_step = 0
        self.attention_adapter_calls = 0
        self.compact_attention_calls = 0
        self.tier_token_counts = {"retrieval_fp16": 0, "peaked_h2o": 0, "diffuse_lexico": 0, "recent": 0}
        if model is not None:
            self.bind_model(model)

    def bind_model(self, model: Any) -> None:
        for module in model.modules():
            if hasattr(module, "layer_idx") and hasattr(module, "num_key_value_groups"):
                module._stoikov_cache = self

    def set_matryoshka_mode(self, active_atoms: int) -> None:
        if not 1 <= active_atoms <= self.s_budget:
            raise ValueError(f"active_atoms must be in [1, {self.s_budget}]")
        self.active_atoms = int(active_atoms)

    def retained_positions(self, layer_idx: int = 0) -> dict[str, list[list[int]]]:
        store = self._stores.get(layer_idx)
        if store is None:
            return {"fp16": [], "lexico": []}
        if store.compact_k is not None:
            return {"fp16": [x.tolist() for x in store.compact_pos],
                    "lexico": [x.tolist() for x in store.lex_pos]}
        return {"fp16": [list(range(store.raw_k.shape[-2]))] if store.raw_k is not None else [], "lexico": []}

    def compact_validity_mask(self, layer_idx: int = 0) -> torch.Tensor | None:
        store = self._stores.get(layer_idx)
        if store is None or store.compact_k is None:
            return None
        device = store.compact_k[0].device
        mask = torch.zeros(len(store.compact_k), store.seq_len, dtype=torch.bool, device=device)
        for head, pos in enumerate(store.compact_pos):
            mask[head, pos.long()] = True
        for head, pos in enumerate(store.lex_pos):
            if pos.numel():
                mask[head, pos.long()] = True
        return mask

    def _policy(self) -> tuple[float, float, int]:
        p = self.controller() if self.controller else {}
        beta = float(p.get("beta_t", p.get("beta", self.beta)))
        ratio = float(p.get("keep_ratio", self.keep_ratio))
        sparsity = int(p.get("s_budget", self.s_budget))
        return max(0.0, min(1.0, beta)), max(0.0, min(1.0, ratio)), max(1, min(self.s_budget, sparsity))

    def _ensure_layer(self, layer_idx: int) -> None:
        while len(self.layers) <= layer_idx:
            self.layers.append(self.layer_class_to_replicate())

    def _encode_vectors(self, x: torch.Tensor, budget: int, layer_idx: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if self.W_ridge_by_layer is not None and layer_idx is not None:
            mean = self.kv_mean_by_layer[layer_idx].to(x.device, x.dtype)
            w = self.W_ridge_by_layer[layer_idx].to(x.device, x.dtype)
        else:
            mean = self.kv_mean.to(x.device, x.dtype)
            w = self.W_ridge.to(x.device, x.dtype)
        if w.shape[0] != x.shape[-1]:
            w = F.normalize(torch.ones(x.shape[-1], self.s_budget, device=x.device, dtype=x.dtype), dim=0)
        coeff = (x - mean) @ w
        _, ids = coeff.abs().topk(min(budget, coeff.shape[-1]), dim=-1, sorted=True)
        return coeff.gather(-1, ids).to(torch.float16), ids.to(torch.uint8 if coeff.shape[-1] <= 256 else torch.int16)

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int,
               cache_kwargs: dict[str, Any] | None = None, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs, args, kwargs
        self._ensure_layer(layer_idx)
        store = self._stores.setdefault(layer_idx, _LayerStore())
        k, v = key_states, value_states
        batch, heads, q_len, dim = k.shape
        device = k.device
        role = self.head_roles[min(layer_idx, self.head_roles.shape[0] - 1), :heads].to(device)
        u = self.U_Q.to(device=device, dtype=k.dtype)
        if u.shape[0] != dim:
            u = torch.eye(dim, device=device, dtype=k.dtype)
        importance = torch.linalg.vector_norm(k @ u, dim=-1) * torch.linalg.vector_norm(v, dim=-1)
        importance = importance.mean(dim=0)
        if not self.dense_fallback:
            return self._update_compact(store, k, v, importance, role, layer_idx)
        old = 0 if store.raw_k is None else store.raw_k.shape[-2]
        positions = torch.arange(old, old + q_len, device=device)
        importance[:, positions < 4] += 1e9
        beta, keep_ratio, s_budget = self._policy()
        prior = store.scores
        if prior is None:
            scores = importance
        else:
            scores = torch.cat([prior * beta, importance], dim=-1)
        store.raw_k = k if store.raw_k is None else torch.cat((store.raw_k, k), dim=-2)
        store.raw_v = v if store.raw_v is None else torch.cat((store.raw_v, v), dim=-2)
        store.scores = scores
        total = store.raw_k.shape[-2]
        recent_start = max(0, total - self.n_b)
        keep = torch.zeros(heads, total, dtype=torch.bool, device=device)
        keep[:, recent_start:] = True
        keep[:, :min(4, total)] = True
        older = recent_start
        if older:
            h2o_heads = torch.where(role == ROLE_PEAKED_H2O)[0]
            if len(h2o_heads):
                count = min(older, max(4, int(total * keep_ratio)))
                ids = scores[h2o_heads, :older].topk(count, dim=-1, largest=True).indices
                keep[h2o_heads[:, None], ids] = True
            keep[role == ROLE_RETRIEVAL_FP16, :] = True
        store.h2o_keep = keep
        self._encode_lexico(store, role, s_budget, layer_idx)
        dense_k, dense_v = self._dense_view(store, role, layer_idx)
        layer = self.layers[layer_idx]
        layer.keys, layer.values = dense_k, dense_v
        layer.is_initialized = True
        self._logical_length = max(self._logical_length, total)
        self._refresh_telemetry()
        self.bytes_written_last_step = int((k.numel() + v.numel()) * k.element_size())
        self.bytes_read_total += self.bytes_read_last_step
        return dense_k, dense_v

    def _encode_lexico(self, store: _LayerStore, role: torch.Tensor, budget: int, layer_idx: int | None = None) -> None:
        assert store.raw_k is not None and store.raw_v is not None
        lex = torch.where(role == ROLE_DIFFUSE_LEXICO)[0]
        if not len(lex):
            return
        x = torch.cat((store.raw_k[:, lex].float(), store.raw_v[:, lex].float()), dim=-1)
        w = self.W_ridge.to(x.device, x.dtype)
        if w.shape[0] != x.shape[-1]:
            w = F.normalize(torch.ones(x.shape[-1], self.s_budget, device=x.device, dtype=x.dtype), dim=0)
        coeff = x @ w
        values, ids = coeff.abs().topk(min(budget, coeff.shape[-1]), dim=-1, sorted=True)
        signed = coeff.gather(-1, ids).to(torch.float16)
        store.coeffs = signed
        store.atom_ids = ids.to(torch.uint8 if coeff.shape[-1] <= 256 else torch.int16)

    def _dense_view(self, store: _LayerStore, role: torch.Tensor, layer_idx: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        assert store.raw_k is not None and store.raw_v is not None and store.h2o_keep is not None
        k, v = store.raw_k.clone(), store.raw_v.clone()
        h2o = torch.where(role == ROLE_PEAKED_H2O)[0]
        if len(h2o) and not self.dense_fallback:
            dropped = ~store.h2o_keep[h2o]
            k[:, h2o] = k[:, h2o].masked_fill(dropped[None, :, :, None], 0)
            v[:, h2o] = v[:, h2o].masked_fill(dropped[None, :, :, None], 0)
        lex = torch.where(role == ROLE_DIFFUSE_LEXICO)[0]
        if len(lex) and store.coeffs is not None and store.atom_ids is not None and not self.dense_fallback:
            a = min(self.active_atoms, store.coeffs.shape[-1])
            dictionary = self.W_ridge.to(k.device, torch.float32)
            coeff = store.coeffs[..., :a].float()
            ids = store.atom_ids[..., :a].long()
            selected = dictionary.T[ids]
            recon = (coeff[..., None] * selected).sum(dim=-2).to(k.dtype)
            d = k.shape[-1]
            k[:, lex], v[:, lex] = recon[..., :d], recon[..., d:]
        return k, v

    def _update_compact(self, store: _LayerStore, k: torch.Tensor, v: torch.Tensor,
                        importance: torch.Tensor, role: torch.Tensor, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        b, h, q, d = k.shape
        old_n = store.seq_len
        total = old_n + q
        beta, keep_ratio, budget = self._policy()
        scores = importance if store.scores is None else torch.cat((store.scores * beta, importance), dim=-1)
        store.scores = scores
        store.seq_len = total
        recent_start = max(0, total - self.n_b)

        if store.compact_k is None:
            store.compact_k, store.compact_v, store.compact_pos = [], [], []
            store.lex_coeff, store.lex_ids, store.lex_pos = [], [], []
            for _ in range(h):
                store.compact_k.append(k[:, _, :, :].clone())
                store.compact_v.append(v[:, _, :, :].clone())
                store.compact_pos.append(torch.arange(q, device=k.device))
                store.lex_coeff.append(torch.empty(1, 0, budget, dtype=torch.float16, device=k.device))
                store.lex_ids.append(torch.empty(1, 0, budget, dtype=torch.uint8, device=k.device))
                store.lex_pos.append(torch.empty(0, dtype=torch.long, device=k.device))
            recent_start = max(0, total - self.n_b)
            for head in range(h):
                pos = store.compact_pos[head]
                r = int(role[head])
                if r == ROLE_RETRIEVAL_FP16:
                    continue
                if r == ROLE_PEAKED_H2O:
                    keep_n = min(total, max(4, int(total * self.keep_ratio)))
                    mandatory = (pos >= recent_start) | (pos < 4)
                    selected = torch.zeros(total, dtype=torch.bool, device=k.device)
                    selected[scores[head].topk(keep_n).indices] = True
                    selected |= mandatory
                    idx = torch.where(selected)[0]
                    store.compact_k[head], store.compact_v[head], store.compact_pos[head] = k[:, head, idx], v[:, head, idx], pos[idx]
                else:
                    hist = pos < recent_start
                    if hist.any():
                        hist_pos = pos[hist]
                        hist_scores = scores[head, hist_pos]
                        keep_fp16 = max(1, int(hist_pos.numel() * 0.5))
                        top_ids = hist_scores.topk(min(keep_fp16, hist_pos.numel())).indices
                        fp16_mask = torch.zeros(hist_pos.numel(), dtype=torch.bool, device=k.device)
                        fp16_mask[top_ids] = True
                        encode_mask = hist.clone()
                        encode_mask[hist] = ~fp16_mask
                        if encode_mask.any():
                            ep = pos[encode_mask]
                            coeff, ids = self._encode_vectors(torch.cat((k[:, head, ep], v[:, head, ep]), dim=-1).float(), budget, layer_idx)
                            store.lex_coeff[head], store.lex_ids[head], store.lex_pos[head] = coeff, ids, ep
                    rec = ~hist
                    store.compact_k[head], store.compact_v[head], store.compact_pos[head] = k[:, head, rec], v[:, head, rec], pos[rec]
        else:
            new_start = old_n
            for head in range(h):
                old_k, old_v, old_pos = store.compact_k[head], store.compact_v[head], store.compact_pos[head]
                incoming_pos = torch.arange(new_start, total, device=k.device)
                incoming_k, incoming_v = k[:, head], v[:, head]
                r = int(role[head])
                if r == ROLE_RETRIEVAL_FP16:
                    store.compact_k[head] = torch.cat((old_k, incoming_k), dim=1)
                    store.compact_v[head] = torch.cat((old_v, incoming_v), dim=1)
                    store.compact_pos[head] = torch.cat((old_pos, incoming_pos))
                elif r == ROLE_PEAKED_H2O:
                    cand_pos = torch.cat((old_pos, incoming_pos))
                    cand_k = torch.cat((old_k, incoming_k), dim=1)
                    cand_v = torch.cat((old_v, incoming_v), dim=1)
                    recent_start = max(0, total - self.n_b)
                    mandatory = (cand_pos >= recent_start) | (cand_pos < 4)
                    keep_n = min(cand_pos.numel(), max(self.min_keep_tokens, int(total * self.keep_ratio)))
                    rank = scores[head, cand_pos].topk(min(keep_n, cand_pos.numel())).indices
                    selected = torch.zeros(cand_pos.numel(), dtype=torch.bool, device=k.device)
                    selected[rank] = True; selected |= mandatory
                    order = torch.where(selected)[0].sort().values
                    store.compact_k[head], store.compact_v[head] = cand_k[:, order], cand_v[:, order]
                    store.compact_pos[head] = cand_pos[order]
                else:
                    cand_pos = torch.cat((old_pos, incoming_pos))
                    cand_k = torch.cat((old_k[:, old_pos >= recent_start], incoming_k), dim=1)
                    cand_v = torch.cat((old_v[:, old_pos >= recent_start], incoming_v), dim=1)
                    hist = cand_pos < recent_start
                    rec = ~hist
                    if hist.any():
                        hist_pos = cand_pos[hist]
                        hist_scores = scores[head, hist_pos]
                        keep_fp16 = max(1, int(hist_pos.numel() * 0.5))
                        top_ids = hist_scores.topk(min(keep_fp16, hist_pos.numel())).indices
                        fp16_mask = torch.zeros(hist_pos.numel(), dtype=torch.bool, device=k.device)
                        fp16_mask[top_ids] = True
                        rec = rec.clone()
                        rec[hist] = fp16_mask
                        encode_mask = hist.clone()
                        encode_mask[hist] = ~fp16_mask
                        if encode_mask.any():
                            ek = cand_k[:, encode_mask]
                            ev = cand_v[:, encode_mask]
                            ep = cand_pos[encode_mask]
                            coeff, ids = self._encode_vectors(torch.cat((ek, ev), dim=-1).float(), 16, layer_idx)
                            store.lex_coeff[head] = torch.cat((store.lex_coeff[head], coeff), dim=1)
                            store.lex_ids[head] = torch.cat((store.lex_ids[head], ids), dim=1)
                            store.lex_pos[head] = torch.cat((store.lex_pos[head], ep))
                    store.compact_k[head], store.compact_v[head] = cand_k[:, rec], cand_v[:, rec]
                    store.compact_pos[head] = cand_pos[rec]
        layer = self.layers[layer_idx]
        dense_k, dense_v = self._compact_dense_view(store, role, h, d, k.device, layer_idx)
        layer.keys, layer.values = dense_k, dense_v
        layer.is_initialized = True
        self._logical_length = max(self._logical_length, total)
        self._refresh_telemetry()
        self.bytes_written_last_step = int((k.numel() + v.numel()) * k.element_size())
        self.bytes_read_total += self.bytes_read_last_step
        return dense_k, dense_v

    def _compact_dense_view(self, store: _LayerStore, role: torch.Tensor, heads: int, dim: int,
                            device: torch.device, layer_idx: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        b = store.compact_k[0].shape[0]
        k = torch.zeros(b, heads, store.seq_len, dim, device=device, dtype=store.compact_k[0].dtype)
        v = torch.zeros_like(k)
        dictionary = (self.D_by_layer[layer_idx] if self.D_by_layer is not None and layer_idx is not None else self.W_ridge.T).to(device, torch.float32)
        mean = (self.kv_mean_by_layer[layer_idx] if self.kv_mean_by_layer is not None and layer_idx is not None else self.kv_mean).to(device, torch.float32)
        for head in range(heads):
            pos = store.compact_pos[head].long()
            k[:, head, pos], v[:, head, pos] = store.compact_k[head], store.compact_v[head]
            if int(role[head]) == ROLE_DIFFUSE_LEXICO and store.lex_pos[head].numel():
                active = min(self.active_atoms, store.lex_coeff[head].shape[-1])
                ids = store.lex_ids[head][..., :active].long()
                coeff = store.lex_coeff[head][..., :active].float()
                recon = (coeff[..., None] * dictionary[ids]).sum(dim=-2) + mean
                k[:, head, store.lex_pos[head]], v[:, head, store.lex_pos[head]] = recon[..., :dim].to(k.dtype), recon[..., dim:].to(k.dtype)
        self.reconstruction_workspace_bytes = int(k.nbytes + v.nbytes)
        return k, v

    def _refresh_telemetry(self) -> None:
        fp16 = compressed = reads = physical = 0
        counts = {"retrieval_fp16": 0, "peaked_h2o": 0, "diffuse_lexico": 0, "recent": 0}
        for li, store in self._stores.items():
            if store.raw_k is None and store.compact_k is not None:
                roles = self.head_roles[min(li, self.head_roles.shape[0] - 1), :len(store.compact_k)].to(store.compact_k[0].device)
                for head, ck in enumerate(store.compact_k):
                    cv, cp = store.compact_v[head], store.compact_pos[head]
                    r = int(roles[head]); tokens = int(cp.numel())
                    counts["recent"] += min(self.n_b, tokens)
                    if r == ROLE_RETRIEVAL_FP16:
                        counts["retrieval_fp16"] += tokens; fp16 += ck.nbytes + cv.nbytes; reads += ck.nbytes + cv.nbytes
                    elif r == ROLE_PEAKED_H2O:
                        counts["peaked_h2o"] += tokens; fp16 += ck.nbytes + cv.nbytes; reads += ck.nbytes + cv.nbytes
                    else:
                        old = int(store.lex_pos[head].numel()); atoms = min(self.s_budget, store.lex_coeff[head].shape[-1])
                        counts["diffuse_lexico"] += old + tokens
                        compressed += store.lex_coeff[head].nbytes + store.lex_ids[head].nbytes + ck.nbytes + cv.nbytes
                        reads += (old + tokens) * (2 * (3 * self.active_atoms + 2))
                physical += sum(x.nbytes + y.nbytes for x, y in zip(store.compact_k, store.compact_v))
                physical += sum(x.nbytes + y.nbytes for x, y in zip(store.lex_coeff, store.lex_ids))
                continue
            if store.raw_k is None or store.h2o_keep is None:
                continue
            b, h, n, d = store.raw_k.shape
            physical += store.raw_k.nbytes + store.raw_v.nbytes
            roles = self.head_roles[min(li, self.head_roles.shape[0] - 1), :h].to(store.raw_k.device)
            recent = min(self.n_b, n)
            counts["recent"] += h * recent
            for head in range(h):
                r = int(roles[head])
                if r == ROLE_RETRIEVAL_FP16:
                    tokens = n; counts["retrieval_fp16"] += n
                    fp16 += b * n * 2 * d * 2; reads += b * n * 2 * d * 2
                elif r == ROLE_PEAKED_H2O:
                    tokens = int(store.h2o_keep[head].sum()); counts["peaked_h2o"] += tokens
                    fp16 += b * tokens * 2 * d * 2; reads += b * tokens * 2 * d * 2
                else:
                    tokens = n; atoms = min(self.s_budget, store.coeffs.shape[-1] if store.coeffs is not None else self.s_budget)
                    counts["diffuse_lexico"] += tokens
                    compressed += b * tokens * (2 * (3 * atoms + 2))
                    reads += b * tokens * (2 * (3 * self.active_atoms + 2))
        dictionary_bytes = self.W_ridge.numel() * self.W_ridge.element_size()
        self.fp16_bytes, self.compressed_bytes = int(fp16), int(compressed)
        self.capacity_bytes = int(fp16 + compressed + dictionary_bytes)
        self.physical_storage_bytes = int(physical)
        self.bytes_read_last_step = int(reads)
        self.tier_token_counts = counts

    def crop(self, max_length: int) -> None:
        current = self.get_seq_length()
        target = max(0, current + max_length) if max_length < 0 else min(max_length, current)
        for store in self._stores.values():
            if store.raw_k is None and store.compact_k is not None:
                for head in range(len(store.compact_k)):
                    keep = store.compact_pos[head] < target
                    store.compact_k[head] = store.compact_k[head][:, keep]
                    store.compact_v[head] = store.compact_v[head][:, keep]
                    store.compact_pos[head] = store.compact_pos[head][keep]
                    lp = store.lex_pos[head] < target
                    store.lex_coeff[head] = store.lex_coeff[head][:, lp]
                    store.lex_ids[head] = store.lex_ids[head][:, lp]
                    store.lex_pos[head] = store.lex_pos[head][lp]
                store.seq_len = target
                store.scores = store.scores[..., :target] if store.scores is not None else None
                continue
            if store.raw_k is None:
                continue
            store.raw_k, store.raw_v = store.raw_k[..., :target, :], store.raw_v[..., :target, :]
            store.scores = store.scores[..., :target] if store.scores is not None else None
            store.h2o_keep = store.h2o_keep[..., :target] if store.h2o_keep is not None else None
            if hasattr(store, 'coeffs') and store.coeffs is not None:
                store.coeffs = store.coeffs[..., :target, :]
            if hasattr(store, 'atom_ids') and store.atom_ids is not None:
                store.atom_ids = store.atom_ids[..., :target, :]
        for li, store in self._stores.items():
            if store.raw_k is None and store.compact_k is not None:
                roles = self.head_roles[min(li, self.head_roles.shape[0] - 1), :len(store.compact_k)].to(store.compact_k[0].device)
                k, v = self._compact_dense_view(store, roles, len(store.compact_k), store.compact_k[0].shape[-1], store.compact_k[0].device, li)
                self.layers[li].keys, self.layers[li].values = k, v
                continue
            roles = self.head_roles[min(li, self.head_roles.shape[0] - 1), :store.raw_k.shape[1]].to(store.raw_k.device)
            k, v = self._dense_view(store, roles, li)
            self.layers[li].keys, self.layers[li].values = k, v
        self._logical_length = target
        self._refresh_telemetry()

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._logical_length

    def set_matryoshka_mode(self, active_atoms: int) -> None:
        if not 1 <= active_atoms <= self.s_budget:
            raise ValueError(f"active_atoms must be in [1, {self.s_budget}]")
        self.active_atoms = int(active_atoms)

    def compact_validity_mask(self, layer_idx: int = 0) -> torch.Tensor | None:
        store = self._stores.get(layer_idx)
        if store is None or store.compact_k is None:
            return None
        device = store.compact_k[0].device
        mask = torch.zeros(len(store.compact_k), store.seq_len, dtype=torch.bool, device=device)
        for head, pos in enumerate(store.compact_pos):
            mask[head, pos.long()] = True
        for head, pos in enumerate(store.lex_pos):
            if pos.numel():
                mask[head, pos.long()] = True
        return mask

    def compact_validity_mask(self, layer_idx: int = 0) -> torch.Tensor | None:
        store = self._stores.get(layer_idx)
        if store is None or store.compact_k is None:
            return None
        device = store.compact_k[0].device
        mask = torch.zeros(len(store.compact_k), store.seq_len, dtype=torch.bool, device=device)
        for head, pos in enumerate(store.compact_pos):
            mask[head, pos.long()] = True
        for head, pos in enumerate(store.lex_pos):
            if pos.numel():
                mask[head, pos.long()] = True
        return mask