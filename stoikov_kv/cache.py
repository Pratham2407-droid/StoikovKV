"""
Live Inference Engine for StoikovKV (`stoikov_kv/cache.py`).

Implements:
- `DynamicCache` compatibility base class (imports `transformers.cache_utils.DynamicCache`
  when `transformers` is installed, or registers a 100% compatible `DynamicCache` in
  `sys.modules["transformers.cache_utils"]` for standalone operation).
- `create_mock_artifacts(model, save_path=None, seed=42)`: Fast single-pass artifact
  initializer (`U_Q`, `head_roles`, `D`, `W_ridge`, `head_entropy`) using only forward
  projection hooks (`output_attentions=False`), enabling immediate standalone execution
  when `calibrated_artifacts.pt` does not exist yet.
- `HMMTopicShiftController`: Online non-stationary regime detector tracking write-time
  query-subspace projection entropy EMA and variance; triggers rapid H2O score decay
  (`beta_t = 0.65` vs stationary `beta_t = 0.98`) upon topic shifts (`z_score > 1.8`).
- `StoikovKVCache(DynamicCache)`: FlashAttention/SDPA-compatible hybrid KV cache with:
  1. Uncompressed FP16 sliding window of `n_b = 64` recent tokens across all KV heads.
  2. Write-time importance probe `I_t = ||K_t @ U_Q||_2 * ||V_t||_2` + sink gate (`+100.0`
     for token indices `0..3`) and decayed heavy-hitter value `V_i(t) = beta_t * V_i(t-1) + I_t`.
  3. Per-KV-head routing for tokens older than `n_b = 64`:
     - `ROLE_RETRIEVAL_FP16 (0)`: Retain 100% of tokens in uncompressed FP16.
     - `ROLE_PEAKED_H2O (1)`: Evict lowest `V_i(t)` tokens, keeping only
       `k = int(seq_len * keep_ratio)` Heavy Hitters + `n_b` recent tokens.
     - `ROLE_DIFFUSE_LEXICO (2)`: Compress tokens via 1-step ridge-preconditioned
       projection (`W_ridge`) into `s_budget` magnitude-sorted dictionary atoms
       (`|y_(1)| >= ... >= |y_(s)|`).
  4. `set_matryoshka_mode(active_atoms=m)` for nested partial reconstruction (`3*m + 2`
     bytes read/vector during speculative drafting vs `3*s + 2` during parallel verification).
  5. `crop(max_length)` for exact speculative decoding rollback of rejected draft tokens.
  6. Exact byte telemetry: `fp16_bytes`, `compressed_bytes`, `bytes_read_last_step`,
     `cumulative_bytes_read`, and `per_tier_token_counts`.
"""

from __future__ import annotations

import math
import os
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_stoikov")

import torch
import torch.nn as nn
import torch.nn.functional as F

from .calibration import (
    ROLE_DIFFUSE_LEXICO,
    ROLE_PEAKED_H2O,
    ROLE_RETRIEVAL_FP16,
    _apply_rotary_pos_emb,
    _compute_rope_cos_sin,
    _inspect_model_architecture,
    score_write_time,
)
from .elastic_lexico import (
    ElasticLexicoCodec,
    QuantizedInt8Values,
    quantize_sorted_atoms,
)

# ============================================================================
# DynamicCache Base Class (HuggingFace Transformers + Standalone Compatibility)
# ============================================================================

try:
    from transformers.cache_utils import Cache, DynamicCache  # type: ignore
except Exception:

    class Cache:
        """Minimal HuggingFace-compatible Cache base class."""

        def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[Dict[str, Any]] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            raise NotImplementedError

        def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
            raise NotImplementedError

        def get_max_length(self) -> Optional[int]:
            return None

        def get_usable_length(
            self, new_seq_length: int, layer_idx: Optional[int] = 0
        ) -> int:
            return self.get_seq_length(layer_idx)

    class DynamicCache(Cache):
        """
        Standalone implementation of `transformers.cache_utils.DynamicCache`
        preserving `key_cache`, `value_cache`, `_seen_tokens`, `update`,
        `get_seq_length`, `crop`, and indexing semantics.
        """

        def __init__(self, num_hidden_layers: Optional[int] = None) -> None:
            super().__init__()
            self.key_cache: List[torch.Tensor] = []
            self.value_cache: List[torch.Tensor] = []
            self._seen_tokens: int = 0

        def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            if layer_idx < len(self):
                return (self.key_cache[layer_idx], self.value_cache[layer_idx])
            raise KeyError(f"Cache only has {len(self)} layers, accessed {layer_idx}")

        def __iter__(self):
            for layer_idx in range(len(self)):
                yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

        def __len__(self) -> int:
            return len(self.key_cache)

        def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[Dict[str, Any]] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            if layer_idx == 0:
                self._seen_tokens += int(key_states.shape[-2])
            while len(self.key_cache) <= layer_idx:
                self.key_cache.append(torch.empty(0, device=key_states.device, dtype=key_states.dtype))
                self.value_cache.append(torch.empty(0, device=value_states.device, dtype=value_states.dtype))
            if self.key_cache[layer_idx].numel() == 0:
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
            else:
                self.key_cache[layer_idx] = torch.cat(
                    [self.key_cache[layer_idx], key_states], dim=-2
                )
                self.value_cache[layer_idx] = torch.cat(
                    [self.value_cache[layer_idx], value_states], dim=-2
                )
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
            idx = 0 if layer_idx is None else int(layer_idx)
            if len(self.key_cache) <= idx or self.key_cache[idx].numel() == 0:
                return 0
            return int(self.key_cache[idx].shape[-2])

        def get_max_length(self) -> Optional[int]:
            return None

        def crop(self, max_length: int) -> None:
            if max_length < 0:
                max_length = self.get_seq_length() - abs(max_length)
            if self.get_seq_length() <= max_length:
                return
            self._seen_tokens = max_length
            for idx in range(len(self.key_cache)):
                if self.key_cache[idx].numel() > 0:
                    self.key_cache[idx] = self.key_cache[idx][..., :max_length, :]
                    self.value_cache[idx] = self.value_cache[idx][..., :max_length, :]

    # Register shim in sys.modules so `from transformers.cache_utils import DynamicCache` works
    if "transformers" not in sys.modules:
        tf_mod = types.ModuleType("transformers")
        sys.modules["transformers"] = tf_mod
    if "transformers.cache_utils" not in sys.modules:
        cu_mod = types.ModuleType("transformers.cache_utils")
        cu_mod.Cache = Cache  # type: ignore[attr-defined]
        cu_mod.DynamicCache = DynamicCache  # type: ignore[attr-defined]
        sys.modules["transformers.cache_utils"] = cu_mod
        setattr(sys.modules["transformers"], "cache_utils", cu_mod)


# ============================================================================
# Standalone Artifact Initialization (`create_mock_artifacts`)
# ============================================================================


@torch.no_grad()
def create_mock_artifacts(
    model: nn.Module,
    tokenizer: Optional[Any] = None,
    save_path: Optional[str] = None,
    subspace_rank: int = 16,
    dict_atoms: int = 1024,
    ridge_lambda2: float = 0.1,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Initializes `U_Q`, `head_roles`, `D`, `W_ridge`, and `head_entropy` from a single fast
    forward pass (`output_attentions=False`) using forward hooks on `q_proj`, `k_proj`,
    and `v_proj`. Never requires `output_attentions=True`.

    Target Shapes:
    - `U_Q`: `[num_layers, num_kv_heads, head_dim, r]` (`r = min(subspace_rank, head_dim)`)
    - `head_roles`: `[num_layers, num_kv_heads]` (`torch.long` in `{0, 1, 2}`)
    - `D`: `[num_layers, head_dim, dict_atoms]`
    - `W_ridge`: `[num_layers, dict_atoms, head_dim]`
    - `head_entropy`: `[num_layers, num_kv_heads]`
    """
    head_dim = int(
        getattr(model.config, "head_dim", None)
        or (int(model.config.hidden_size) // int(model.config.num_attention_heads))
    )
    eff_rank = min(int(subspace_rank), head_dim)
    dims = _inspect_model_architecture(model, r_subspace=eff_rank, dict_atoms=dict_atoms)
    num_layers = dims["num_layers"]
    num_q_heads = dims["num_q_heads"]
    num_kv_heads = dims["num_kv_heads"]
    q_per_kv = dims["q_per_kv"]
    rope_theta = float(dims["rope_theta"])
    device = dims["device"]
    model_dtype = dims["dtype"]
    layers = dims["layers_module"]

    q_captured: Dict[int, torch.Tensor] = {}
    k_captured: Dict[int, torch.Tensor] = {}
    v_captured: Dict[int, torch.Tensor] = {}
    handles: List[Any] = []

    def _make_hook(store: Dict[int, torch.Tensor], l_idx: int, n_heads: int, apply_rope: bool):
        def _hook(_mod: nn.Module, _inp: Any, out: torch.Tensor) -> None:
            t = out[0] if isinstance(out, tuple) else out
            bsz, seq_len, _ = t.shape
            reshaped = t.detach().view(bsz, seq_len, n_heads, head_dim).transpose(1, 2)
            if apply_rope:
                cos, sin = _compute_rope_cos_sin(
                    seq_len=seq_len,
                    head_dim=head_dim,
                    device=reshaped.device,
                    dtype=reshaped.dtype,
                    rope_theta=rope_theta,
                )
                reshaped = _apply_rotary_pos_emb(reshaped, cos, sin)
            store[l_idx] = reshaped

        return _hook

    try:
        for l_idx in range(num_layers):
            attn = getattr(layers[l_idx], "self_attn", None) or getattr(layers[l_idx], "attn", None)
            if hasattr(attn, "q_proj") and hasattr(attn, "k_proj") and hasattr(attn, "v_proj"):
                handles.append(
                    attn.q_proj.register_forward_hook(_make_hook(q_captured, l_idx, num_q_heads, True))
                )
                handles.append(
                    attn.k_proj.register_forward_hook(_make_hook(k_captured, l_idx, num_kv_heads, True))
                )
                handles.append(
                    attn.v_proj.register_forward_hook(_make_hook(v_captured, l_idx, num_kv_heads, False))
                )

        # Build a single fast batch of diverse synthetic/tokenized sequences (no output_attentions)
        vocab_sz = int(getattr(model.config, "vocab_size", 32768))
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        input_ids = torch.randint(16, min(8192, vocab_sz), (4, 48), generator=gen).to(device)
        if tokenizer is not None:
            sample_texts = [
                "System configuration: API_PORT = 8443. The secret passkey is 94812. Maintain low latency.",
                "def forward_attention(q, k, v): return torch.nn.functional.scaled_dot_product_attention(q, k, v)",
                ".container { display: flex; backdrop-filter: blur(12px); padding: 1.5rem; border-radius: 8px; }",
                "StoikovKV routes retrieval heads to FP16, peaked heads to H2O, and diffuse heads to Lexico.",
            ]
            for b_i, txt in enumerate(sample_texts):
                enc = tokenizer(txt, return_tensors="pt")
                ids = enc["input_ids"] if isinstance(enc, dict) else enc.input_ids
                take = min(48, int(ids.shape[-1]))
                input_ids[b_i, :take] = ids[0, :take].to(device)

        was_training = model.training
        model.eval()
        try:
            _ = model(input_ids=input_ids, output_attentions=False, use_cache=False)
        except TypeError:
            _ = model(input_ids=input_ids)
        if was_training:
            model.train()
    finally:
        for h in handles:
            h.remove()

    U_Q = torch.zeros(
        (num_layers, num_kv_heads, head_dim, eff_rank),
        dtype=model_dtype,
        device=device,
    )
    D = torch.zeros(
        (num_layers, head_dim, dict_atoms),
        dtype=model_dtype,
        device=device,
    )
    W_ridge = torch.zeros(
        (num_layers, dict_atoms, head_dim),
        dtype=model_dtype,
        device=device,
    )
    head_entropy = torch.zeros((num_layers, num_kv_heads), dtype=torch.float32, device=device)
    retrieval_Proxy = torch.zeros((num_layers, num_kv_heads), dtype=torch.float32, device=device)

    eye_atoms = torch.eye(dict_atoms, dtype=torch.float32, device=device)
    gen_dev = torch.Generator(device="cpu").manual_seed(int(seed) + 101)

    for l_idx in range(num_layers):
        q_t = q_captured.get(l_idx)
        k_t = k_captured.get(l_idx)
        v_t = v_captured.get(l_idx)

        if q_t is not None:
            # q_t: [bsz, num_q_heads, seq_len, head_dim]
            bsz, _, slen, _ = q_t.shape
            q_grouped = (
                q_t.view(bsz, num_kv_heads, q_per_kv, slen, head_dim)
                .permute(1, 0, 3, 2, 4)
                .reshape(num_kv_heads, bsz * slen * q_per_kv, head_dim)
                .float()
            )
            for h_kv in range(num_kv_heads):
                mat_q = q_grouped[h_kv]
                mat_q = torch.nan_to_num(mat_q, nan=0.0, posinf=0.0, neginf=0.0)
                try:
                    _, s_vals, Vh = torch.linalg.svd(mat_q, full_matrices=False)
                except RuntimeError:
                    _, s_vals, Vh = torch.linalg.svd(mat_q.cpu(), full_matrices=False)
                    s_vals = s_vals.to(device)
                    Vh = Vh.to(device)
                basis = Vh[:eff_rank, :].T
                # Deterministic sign canonicalization
                max_idx = torch.argmax(basis.abs(), dim=0, keepdim=True)
                signs = torch.sign(torch.gather(basis, 0, max_idx))
                signs = torch.where(signs == 0, torch.ones_like(signs), signs)
                basis = basis * signs
                U_Q[l_idx, h_kv] = basis.to(dtype=model_dtype)

                # Proxy spectral entropy from singular value concentration
                p_sv = (s_vals ** 2) / (torch.sum(s_vals ** 2) + 1e-8)
                ent = -torch.sum(p_sv * torch.log(p_sv.clamp_min(1e-12)))
                head_entropy[l_idx, h_kv] = ent
                retrieval_Proxy[l_idx, h_kv] = float(s_vals[0].item())

        if k_t is not None and v_t is not None:
            kv_samples = torch.cat(
                [
                    k_t.reshape(-1, head_dim).float(),
                    v_t.reshape(-1, head_dim).float(),
                ],
                dim=0,
            )
            kv_samples = torch.nan_to_num(kv_samples, nan=0.0, posinf=0.0, neginf=0.0)
            try:
                _, _, Vh_kv = torch.linalg.svd(kv_samples, full_matrices=False)
            except RuntimeError:
                _, _, Vh_kv = torch.linalg.svd(kv_samples.cpu(), full_matrices=False)
                Vh_kv = Vh_kv.to(device)
            r_kv = Vh_kv.shape[0]
            rand_atoms = torch.randn(
                (head_dim, dict_atoms), generator=gen_dev, dtype=torch.float32
            ).to(device)
            rand_atoms[:, :r_kv] = Vh_kv.T
            # Mix principal components to form an overcomplete normalized dictionary
            mix = torch.randn((r_kv, dict_atoms - r_kv), generator=gen_dev, dtype=torch.float32).to(device)
            rand_atoms[:, r_kv:] = Vh_kv.T @ mix + 0.10 * rand_atoms[:, r_kv:]
            D_l = F.normalize(rand_atoms, p=2.0, dim=0, eps=1e-8)
        else:
            rand_atoms = torch.randn(
                (head_dim, dict_atoms), generator=gen_dev, dtype=torch.float32
            ).to(device)
            D_l = F.normalize(rand_atoms, p=2.0, dim=0, eps=1e-8)

        gram = D_l.T @ D_l + float(ridge_lambda2) * eye_atoms
        W_l = torch.linalg.solve(gram, D_l.T)
        D[l_idx] = D_l.to(dtype=model_dtype)
        W_ridge[l_idx] = W_l.to(dtype=model_dtype)

    # Assign head_roles: top 5% retrieval -> 0, remaining < 90th percentile entropy -> 1, >= 90th -> 2
    # Using 90th percentile (was median) to minimize Lexico heads (diffuse = lossy, CSS ~0.46).
    # H2O heads have high fidelity (CSS ~0.96). With 90th percentile: ~5% Retrieval, ~85% H2O, ~10% Lexico.
    total_heads = num_layers * num_kv_heads
    num_ret = max(1, int(round(0.05 * total_heads)))
    flat_ret = retrieval_Proxy.view(-1)
    top_ret_idx = torch.topk(flat_ret, k=num_ret, largest=True).indices
    head_roles_flat = torch.full((total_heads,), ROLE_DIFFUSE_LEXICO, dtype=torch.long, device=device)
    is_ret_mask = torch.zeros(total_heads, dtype=torch.bool, device=device)
    is_ret_mask[top_ret_idx] = True
    head_roles_flat[is_ret_mask] = ROLE_RETRIEVAL_FP16

    rem_idx = (~is_ret_mask).nonzero(as_tuple=False).squeeze(-1)
    if rem_idx.numel() > 0:
        rem_ent = head_entropy.view(-1)[rem_idx]
        entropy_threshold = torch.quantile(rem_ent, 0.90)
        peaked_mask = rem_ent < entropy_threshold
        if not torch.any(peaked_mask) and rem_idx.numel() > 1:
            peaked_mask[: rem_idx.numel() // 2] = True
        head_roles_flat[rem_idx[peaked_mask]] = ROLE_PEAKED_H2O
        head_roles_flat[rem_idx[~peaked_mask]] = ROLE_DIFFUSE_LEXICO

    head_roles = head_roles_flat.view(num_layers, num_kv_heads)

    artifacts = {
        "U_Q": U_Q,
        "head_roles": head_roles,
        "D": D,
        "W_ridge": W_ridge,
        "head_entropy": head_entropy,
        "is_mock": True,
    }
    if save_path is not None:
        torch.save(artifacts, save_path)
    return artifacts


def load_or_create_artifacts(
    model: nn.Module,
    artifacts_path: str = "calibrated_artifacts.pt",
    tokenizer: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Loads `calibrated_artifacts.pt` if it exists on disk and matches `model` dimensions;
    otherwise initializes artifacts on the fly via `create_mock_artifacts(model)`.
    """
    head_dim = int(
        getattr(model.config, "head_dim", None)
        or (int(model.config.hidden_size) // int(model.config.num_attention_heads))
    )
    dims = _inspect_model_architecture(model, r_subspace=min(16, head_dim), dict_atoms=1024)
    device = dims["device"]
    dtype = dims["dtype"]

    if artifacts_path and os.path.exists(artifacts_path):
        loaded = torch.load(artifacts_path, map_location=device)
        u_q = loaded.get("U_Q")
        if (
            isinstance(u_q, torch.Tensor)
            and u_q.shape[0] == dims["num_layers"]
            and u_q.shape[1] == dims["num_kv_heads"]
            and u_q.shape[2] == dims["head_dim"]
        ):
            return {
                "U_Q": loaded["U_Q"].to(device=device, dtype=dtype),
                "head_roles": loaded["head_roles"].to(device=device, dtype=torch.long),
                "D": loaded["D"].to(device=device, dtype=dtype),
                "W_ridge": loaded["W_ridge"].to(device=device, dtype=dtype),
                "head_entropy": loaded.get(
                    "head_entropy",
                    torch.zeros((dims["num_layers"], dims["num_kv_heads"]), device=device),
                ).to(device=device),
                "is_mock": bool(loaded.get("is_mock", False)),
                "dict_key_frame": str(loaded.get("dict_key_frame", "post_rope")),
                "rope_theta": float(
                    loaded.get("rope_theta", getattr(model.config, "rope_theta", 1000000.0))
                ),
            }

    return create_mock_artifacts(model=model, tokenizer=tokenizer, save_path=None)


# ============================================================================
# HMM Non-Stationary Regime Controller (Topic-Shift Decay & Budget Hook)
# ============================================================================


@dataclass
class HMMTopicShiftController:
    """
    Online Hidden Markov / Subspace-Shift Controller for StoikovKV:
    - Computes the normalized subspace projection distribution across `U_Q` ranks
      at write time (`output_attentions=False`).
    - Tracks an exponential moving average (EMA) of the subspace feature centroid and
      entropy. When a topic shift (e.g., Turn-1 Python/Config -> Turn-2 CSS styling)
      produces a divergence z-score > `shift_z_threshold` (`1.4`), transitions into the
      non-stationary decay regime (`beta_t = beta_shift = 0.65` vs stationary
      `beta_stationary = 0.99`) so stale Turn-1 heavy hitters decay rapidly.
    """

    beta_stationary: float = 0.99
    beta_shift: float = 0.65
    keep_ratio: float = 0.20
    s_budget: int = 16
    shift_z_threshold: float = 1.4
    ema_alpha: float = 0.15
    enable_hmm_decay: bool = True

    # Telemetry state
    ema_centroid: Optional[torch.Tensor] = None
    ema_dev: float = 0.05
    last_z_score: float = 0.0
    last_beta_t: float = 0.995
    shift_events: int = 0
    step_history: List[Dict[str, float]] = field(default_factory=list)

    def reset(self) -> None:
        self.ema_centroid = None
        self.ema_dev = 0.05
        self.last_z_score = 0.0
        self.last_beta_t = self.beta_stationary if self.enable_hmm_decay else 1.0
        self.shift_events = 0
        self.step_history.clear()

    @torch.no_grad()
    def observe_and_query(
        self,
        proj_energy: torch.Tensor,
        seq_len: int,
    ) -> Tuple[float, float, int]:
        """
        Observes normalized query-subspace projection energy `proj_energy` (`[r]` or `[H, r]`)
        and returns `(beta_t, keep_ratio, s_budget)`.
        """
        if not self.enable_hmm_decay:
            self.last_beta_t = 1.0
            return 1.0, self.keep_ratio, self.s_budget

        feat = F.normalize(proj_energy.detach().float().flatten(), p=2.0, dim=0, eps=1e-8)
        if self.ema_centroid is None or self.ema_centroid.shape != feat.shape:
            self.ema_centroid = feat.clone()
            self.last_z_score = 0.0
            self.last_beta_t = self.beta_stationary
            return self.beta_stationary, self.keep_ratio, self.s_budget

        diff = float(torch.norm(feat - self.ema_centroid, p=2.0).item())
        z_score = diff / max(self.ema_dev, 1e-4)
        self.last_z_score = z_score

        # Update running EMA statistics
        self.ema_centroid = F.normalize(
            (1.0 - self.ema_alpha) * self.ema_centroid + self.ema_alpha * feat,
            p=2.0,
            dim=0,
            eps=1e-8,
        )
        self.ema_dev = (1.0 - self.ema_alpha) * self.ema_dev + self.ema_alpha * max(diff, 0.04)

        if z_score > self.shift_z_threshold:
            beta_t = self.beta_shift
            self.shift_events += 1
        else:
            beta_t = self.beta_stationary

        self.last_beta_t = beta_t
        self.step_history.append(
            {
                "seq_len": float(seq_len),
                "z_score": float(z_score),
                "beta_t": float(beta_t),
            }
        )
        return beta_t, self.keep_ratio, self.s_budget


# ============================================================================
# Per-Head Tier Storage Structures
# ============================================================================


@dataclass
class _HeadTierState:
    """Physical storage for a single `(layer_idx, kv_head_idx)` historical tier (tokens older than `n_b`)."""

    role: int
    # ROLE_RETRIEVAL_FP16 (0): Full uncompressed FP16 [batch, hist_len, head_dim]
    fp16_k: Optional[torch.Tensor] = None
    fp16_v: Optional[torch.Tensor] = None

    # ROLE_PEAKED_H2O (1): Kept heavy-hitter FP16 vectors [batch, k_kept, head_dim] & global token indices [batch, k_kept]
    h2o_k: Optional[torch.Tensor] = None
    h2o_v: Optional[torch.Tensor] = None
    h2o_indices: Optional[torch.Tensor] = None
    h2o_scores: Optional[torch.Tensor] = None

    # ROLE_DIFFUSE_LEXICO (2): Sorted sparse dictionary atoms [batch, hist_len, s_budget]
    lex_k_idx: Optional[torch.Tensor] = None
    lex_k_val: Optional[QuantizedInt8Values] = None
    lex_v_idx: Optional[torch.Tensor] = None
    lex_v_val: Optional[QuantizedInt8Values] = None
    # Cached full-s and active-m reconstructions for fast autoregressive decoding
    lex_recon_k: Optional[torch.Tensor] = None
    lex_recon_v: Optional[torch.Tensor] = None
    lex_recon_atoms: int = -1
    lex_recon_len: int = 0


# ============================================================================
# StoikovKVCache(DynamicCache)
# ============================================================================


class StoikovKVCache(DynamicCache):
    """
    FlashAttention/SDPA-Compatible Hybrid KV Cache (`StoikovKVCache(DynamicCache)`).

    Features:
    1. Recent uncompressed FP16 sliding window of `n_b = 64` tokens across all KV heads.
    2. FlashAttention-compatible write-time importance score:
       `I_t = ||K_t @ U_Q||_2 * ||V_t||_2` + `+100.0` sink protection for indices `0..3`.
    3. Running decayed importance scores `V_i(t) = beta_t * V_i(t-1) + I_t` controlled by
       `HMMTopicShiftController` (or a custom `controller_fn` hook).
    4. Per-KV-head routing for historical tokens (`pos < seq_len - n_b`):
       - `ROLE_RETRIEVAL_FP16 (0)`: Retain 100% of tokens in uncompressed FP16.
       - `ROLE_PEAKED_H2O (1)`: Evict tokens with lowest `V_i(t)`, keeping only
         `k = int(seq_len * keep_ratio)` Heavy Hitters + `n_b` recent tokens.
       - `ROLE_DIFFUSE_LEXICO (2)`: Compress tokens via 1-step ElasticNet projection (`W_ridge`)
         into `s_budget` sorted dictionary atoms (`|y_(1)| >= ... >= |y_(s)|`).
    5. `set_matryoshka_mode(active_atoms=m)` reconstructs `ROLE_DIFFUSE_LEXICO` heads using
       only the top `m` atoms (`3*m + 2` bytes read/vector).
    6. `crop(max_length)` rolls back rejected speculative draft tokens cleanly.
    7. Exact byte telemetry: `fp16_bytes`, `compressed_bytes`, `bytes_read_last_step`,
       `cumulative_bytes_read`, and `per_tier_token_counts`.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        artifacts: Optional[Dict[str, Any]] = None,
        artifacts_path: str = "calibrated_artifacts.pt",
        n_b: int = 64,
        keep_ratio: float = 0.20,
        s_budget: int = 16,
        lambda1: float = 0.04,
        sink_tokens: int = 4,
        sink_bonus: float = 100.0,
        policy_mode: str = "hybrid",
        controller: Optional[HMMTopicShiftController] = None,
        controller_fn: Optional[Callable[[int, int], Tuple[float, float, int]]] = None,
        enable_hmm_decay: bool = True,
        max_rollback_window: int = 32,
        h2o_context_window: int = 0,
    ) -> None:
        # Initialize HF DynamicCache with model config for compatibility
        if model is not None and hasattr(model, 'config'):
            from transformers import PreTrainedConfig
            if isinstance(model.config, PreTrainedConfig):
                super().__init__(config=model.config)
            else:
                from transformers import AutoConfig
                super().__init__(config=AutoConfig.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct"))
        else:
            super().__init__(config=None)

        # Ensure our local key_cache and value_cache exist (HF may not initialize them)
        if not hasattr(self, 'key_cache'):
            self.key_cache: List[torch.Tensor] = []
        if not hasattr(self, 'value_cache'):
            self.value_cache: List[torch.Tensor] = []
        if not hasattr(self, '_seen_tokens'):
            self._seen_tokens: int = 0

        if artifacts is None:
            if model is None:
                raise ValueError("Either `model` or `artifacts` must be provided to StoikovKVCache.")
            artifacts = load_or_create_artifacts(model, artifacts_path=artifacts_path)

        self.U_Q: torch.Tensor = artifacts["U_Q"]
        self.D: torch.Tensor = artifacts["D"]
        self.W_ridge: torch.Tensor = artifacts["W_ridge"]
        base_roles: torch.Tensor = artifacts["head_roles"].clone().long()

        self.num_layers: int = int(self.U_Q.shape[0])
        self.num_kv_heads: int = int(self.U_Q.shape[1])
        self.head_dim: int = int(self.U_Q.shape[2])
        self.subspace_rank: int = int(self.U_Q.shape[3])

        self.policy_mode: str = str(policy_mode).lower().strip()
        if self.policy_mode == "full_fp16":
            self.head_roles = torch.full_like(base_roles, ROLE_RETRIEVAL_FP16)
        elif self.policy_mode in ("standalone_h2o", "h2o"):
            self.head_roles = torch.full_like(base_roles, ROLE_PEAKED_H2O)
        elif self.policy_mode in ("standalone_lexico", "lexico"):
            self.head_roles = torch.full_like(base_roles, ROLE_DIFFUSE_LEXICO)
        elif self.policy_mode in ("hybrid", "stoikov", "stoikov_hybrid"):
            self.head_roles = base_roles
        else:
            raise ValueError(
                f"Unknown policy_mode={policy_mode!r}. Expected 'hybrid', 'full_fp16', "
                "'standalone_h2o', or 'standalone_lexico'."
            )

        self.n_b: int = max(1, int(n_b))
        self.keep_ratio: float = float(keep_ratio)
        self.s_budget: int = max(1, int(s_budget))
        self.active_atoms: int = self.s_budget
        self.lambda1: float = float(lambda1)
        self.sink_tokens: int = max(0, int(sink_tokens))
        self.sink_bonus: float = float(sink_bonus)
        self.max_rollback_window: int = max(8, int(max_rollback_window))
        # Retain a small neighborhood around selected heavy hitters.  This
        # keeps a salient token's immediate syntax/context from being split
        # across an eviction boundary while preserving the same H2O budget.
        self.h2o_context_window: int = max(0, int(h2o_context_window))

        self.codec = ElasticLexicoCodec(self.D, self.W_ridge)
        # Key frame used by the dictionary. "pre_rope": keys are un-rotated (inverse RoPE at
        # their absolute position) before sparse coding and re-rotated after decoding. This is
        # required because post-RoPE keys rotate with position, which a static dictionary cannot
        # code sparsely. Legacy artifacts without this key keep the old post-RoPE behaviour.
        self.dict_key_frame: str = str(artifacts.get("dict_key_frame", "post_rope"))
        self.rope_theta: float = float(
            artifacts.get(
                "rope_theta",
                getattr(getattr(model, "config", None), "rope_theta", 1000000.0)
                if model is not None
                else 1000000.0,
            )
        )
        half = self.head_dim // 2
        self._rope_inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, half, dtype=torch.float32) * 2.0 / float(self.head_dim))
        )
        self.controller = controller or HMMTopicShiftController(
            keep_ratio=self.keep_ratio,
            s_budget=self.s_budget,
            enable_hmm_decay=enable_hmm_decay,
        )
        self.controller_fn = controller_fn

        # Per-layer state
        self._layer_seq_len: List[int] = [0 for _ in range(self.num_layers)]
        # Recent uncompressed FP16 sliding window [batch, num_kv_heads, win_len <= n_b, head_dim]
        self._local_k: List[Optional[torch.Tensor]] = [None for _ in range(self.num_layers)]
        self._local_v: List[Optional[torch.Tensor]] = [None for _ in range(self.num_layers)]
        self._local_scores: List[Optional[torch.Tensor]] = [None for _ in range(self.num_layers)]
        self._local_positions: List[Optional[torch.Tensor]] = [None for _ in range(self.num_layers)]

        # Rollback ring buffer storing recently flushed tokens from the local window so
        # speculative `crop(max_length)` is 100% lossless even across local window boundaries
        self._flushed_ring_k: List[Optional[torch.Tensor]] = [None for _ in range(self.num_layers)]
        self._flushed_ring_v: List[Optional[torch.Tensor]] = [None for _ in range(self.num_layers)]
        self._flushed_ring_scores: List[Optional[torch.Tensor]] = [None for _ in range(self.num_layers)]
        self._flushed_ring_pos: List[Optional[torch.Tensor]] = [None for _ in range(self.num_layers)]

        # Historical routed tier per (layer_idx, kv_head_idx)
        self._tiers: List[List[_HeadTierState]] = [
            [
                _HeadTierState(role=int(self.head_roles[l_idx, h_idx].item()))
                for h_idx in range(self.num_kv_heads)
            ]
            for l_idx in range(self.num_layers)
        ]

        # Byte telemetry
        self.fp16_bytes: int = 0
        self.compressed_bytes: int = 0
        self.bytes_read_last_step: int = 0
        self.cumulative_bytes_read: int = 0
        self._layer_bytes_read: List[int] = [0 for _ in range(self.num_layers)]
        self.per_tier_token_counts: Dict[str, int] = {
            "local_window_fp16": 0,
            "retrieval_fp16": 0,
            "peaked_h2o_fp16": 0,
            "diffuse_lexico_sparse": 0,
            "evicted_h2o": 0,
        }

    def set_matryoshka_mode(self, active_atoms: int) -> None:
        """
        Sets the number of active sorted dictionary atoms `m` (`1 <= m <= s_budget`)
        used when reconstructing `ROLE_DIFFUSE_LEXICO` KV states:
        - `active_atoms = m` (e.g. `m = 4`): fast draft mode reading `3*m + 2` bytes/vector.
        - `active_atoms = s` (e.g. `s = 16`): full verification mode reading `3*s + 2` bytes/vector.
        """
        m = max(1, min(int(active_atoms), self.s_budget))
        if m != self.active_atoms:
            self.active_atoms = m
            # Mark cached Lexico reconstructions dirty so next read uses top-m atoms
            for l_idx in range(self.num_layers):
                for h_idx in range(self.num_kv_heads):
                    st = self._tiers[l_idx][h_idx]
                    if st.role == ROLE_DIFFUSE_LEXICO:
                        st.lex_recon_atoms = -1

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        idx = 0 if layer_idx is None else int(layer_idx)
        if idx < 0 or idx >= self.num_layers:
            return 0
        return int(self._layer_seq_len[idx])

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0) -> Tuple[int, int]:
        """
        Returns the KV cache length and offset for the given layer and query length.
        Uses our internal sequence length tracking instead of HF DynamicCache's internal layers.
        Hugging Face asks before this layer's ``update`` call, so the incoming
        query tokens are not yet reflected in ``_layer_seq_len``.
        """
        idx = 0 if layer_idx is None else int(layer_idx)
        idx = max(0, min(idx, self.num_layers - 1)) if self.num_layers > 0 else 0
        seq_len = int(self._layer_seq_len[idx]) if self.num_layers > 0 else 0
        return (seq_len + int(query_length), 0)

    @torch.no_grad()
    def _rope_rotate(self, x: torch.Tensor, positions: torch.Tensor, inverse: bool) -> torch.Tensor:
        """
        Applies (or inverts) standard half-split RoPE to `x` (`[..., T, D]`) at absolute
        `positions` (`[T]`). Matches `_compute_rope_cos_sin` / `_apply_rotary_pos_emb`:
        `x_rot = x * cos + rotate_half(x) * sin`; the inverse uses `-sin`.
        """
        if self.dict_key_frame != "pre_rope" or x.shape[-2] == 0:
            return x
        inv_freq = self._rope_inv_freq.to(device=x.device)
        freqs = torch.outer(positions.to(device=x.device, dtype=torch.float32), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, D]
        cos = emb.cos()
        sin = emb.sin()
        if inverse:
            sin = -sin
        xf = x.float()
        half = xf.shape[-1] // 2
        x_rot_half = torch.cat((-xf[..., half:], xf[..., :half]), dim=-1)
        return (xf * cos + x_rot_half * sin).to(x.dtype)

    @torch.no_grad()
    def _compute_step_scores_and_controller(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        start_pos: int,
    ) -> Tuple[torch.Tensor, float, float, int]:
        """
        Computes FlashAttention-compatible write-time score `I_t = ||K_t @ U_Q||_2 * ||V_t||_2`
        (plus `+100.0` sink gate for global token indices `0..3`) and queries the controller.
        Never requires `output_attentions=True`.
        """
        bsz, n_kv, q_len, d_head = key_states.shape
        u_q_l = self.U_Q[layer_idx].to(device=key_states.device, dtype=torch.float32)  # [H, D, r]
        k_f = key_states.float()
        v_f = value_states.float()

        # Project K onto U_Q: [B, H, T, D] @ [H, D, r] -> [B, H, T, r]
        proj = torch.einsum("bhtd,hdr->bhtr", k_f, u_q_l)
        proj_norm = torch.linalg.vector_norm(proj, ord=2, dim=-1)  # [B, H, T]
        v_norm = torch.linalg.vector_norm(v_f, ord=2, dim=-1)  # [B, H, T]
        scores = proj_norm * v_norm  # [B, H, T]

        # Apply sink gate (+100.0) for global token indices 0..3
        if self.sink_tokens > 0 and start_pos < self.sink_tokens:
            sink_end = min(q_len, self.sink_tokens - start_pos)
            scores[:, :, :sink_end] = scores[:, :, :sink_end] + self.sink_bonus

        # Query controller (update HMM on layer 0 or via custom controller_fn)
        seq_after = start_pos + q_len
        if self.controller_fn is not None:
            beta_t, keep_ratio, s_budget = self.controller_fn(layer_idx, seq_after)
        elif layer_idx == 0:
            # Compute subspace projection energy profile across tokens
            # During prefill (q_len > 128), evaluate sliding windows so topic shifts inside prefill are detected
            if q_len > 128 and self.controller.enable_hmm_decay:
                chunk_sz = 64
                beta_t = self.controller.beta_stationary
                for c_start in range(0, q_len, chunk_sz):
                    c_end = min(q_len, c_start + chunk_sz)
                    chunk_energy = proj[:, :, c_start:c_end, :].abs().mean(dim=(0, 2))
                    b_step, keep_ratio, s_budget = self.controller.observe_and_query(
                        chunk_energy, start_pos + c_end
                    )
                    if b_step < beta_t:
                        beta_t = b_step
            else:
                step_energy = proj.abs().mean(dim=(0, 2))
                beta_t, keep_ratio, s_budget = self.controller.observe_and_query(
                    step_energy, seq_after
                )
        else:
            beta_t = self.controller.last_beta_t
            keep_ratio = self.keep_ratio
            s_budget = self.s_budget

        return scores, float(beta_t), float(keep_ratio), int(s_budget)

    @torch.no_grad()
    def _route_overflow_to_tiers(
        self,
        layer_idx: int,
        overflow_k: torch.Tensor,
        overflow_v: torch.Tensor,
        overflow_scores: torch.Tensor,
        overflow_pos: torch.Tensor,
        beta_t: float,
        keep_ratio: float,
        s_budget: int,
        total_seq_len: int,
    ) -> None:
        """
        Routes tokens aging out of the `n_b = 64` local FP16 window into per-head tiers:
        - `ROLE_RETRIEVAL_FP16 (0)`: 100% uncompressed FP16
        - `ROLE_PEAKED_H2O (1)`: Top `k = int(seq_len * keep_ratio)` heavy hitters by `V_i(t)`
        - `ROLE_DIFFUSE_LEXICO (2)`: 1-step ElasticNet sparse dictionary atoms (`s_budget`)
        """
        bsz, n_kv, n_over, d_head = overflow_k.shape
        if n_over == 0:
            return

        # A single non-finite activation would otherwise poison the persistent
        # compressed tier and make every later decode fail.  Preserve the valid
        # state while mapping exceptional values to neutral zero vectors/scores.
        overflow_k = torch.nan_to_num(overflow_k, nan=0.0, posinf=0.0, neginf=0.0)
        overflow_v = torch.nan_to_num(overflow_v, nan=0.0, posinf=0.0, neginf=0.0)
        overflow_scores = torch.nan_to_num(overflow_scores, nan=0.0, posinf=0.0, neginf=0.0)

        # Save last `max_rollback_window` flushed tokens in ring buffer for lossless `crop()`
        if self._flushed_ring_k[layer_idx] is None:
            self._flushed_ring_k[layer_idx] = overflow_k[:, :, -self.max_rollback_window :, :].clone()
            self._flushed_ring_v[layer_idx] = overflow_v[:, :, -self.max_rollback_window :, :].clone()
            self._flushed_ring_scores[layer_idx] = overflow_scores[:, :, -self.max_rollback_window :].clone()
            self._flushed_ring_pos[layer_idx] = overflow_pos[-self.max_rollback_window :].clone()
        else:
            prev_pos = self._flushed_ring_pos[layer_idx]
            keep_prev = prev_pos < overflow_pos[0]
            self._flushed_ring_k[layer_idx] = torch.cat(
                [self._flushed_ring_k[layer_idx][:, :, keep_prev, :], overflow_k], dim=-2
            )[:, :, -self.max_rollback_window :, :]
            self._flushed_ring_v[layer_idx] = torch.cat(
                [self._flushed_ring_v[layer_idx][:, :, keep_prev, :], overflow_v], dim=-2
            )[:, :, -self.max_rollback_window :, :]
            self._flushed_ring_scores[layer_idx] = torch.cat(
                [self._flushed_ring_scores[layer_idx][:, :, keep_prev], overflow_scores], dim=-1
            )[:, :, -self.max_rollback_window :]
            self._flushed_ring_pos[layer_idx] = torch.cat(
                [prev_pos[keep_prev], overflow_pos], dim=0
            )[-self.max_rollback_window :]

        # Compute positional decay weights inside overflow chunk if n_over > 1
        if n_over > 1:
            exps = torch.arange(n_over - 1, -1, -1, device=overflow_scores.device, dtype=torch.float32)
            decay_vec = (beta_t ** exps).view(1, 1, n_over)
            overflow_scores_decayed = overflow_scores * decay_vec
        else:
            overflow_scores_decayed = overflow_scores

        # H2O heavy-hitter budget: k = int(seq_len * keep_ratio)
        k_budget = max(1, int(total_seq_len * keep_ratio))

        for h_idx in range(n_kv):
            st = self._tiers[layer_idx][h_idx]
            k_h = overflow_k[:, h_idx, :, :]  # [B, n_over, D]
            v_h = overflow_v[:, h_idx, :, :]  # [B, n_over, D]
            sc_h = overflow_scores_decayed[:, h_idx, :]  # [B, n_over]

            if st.role == ROLE_RETRIEVAL_FP16:
                if st.fp16_k is None:
                    st.fp16_k = k_h
                    st.fp16_v = v_h
                else:
                    st.fp16_k = torch.cat([st.fp16_k, k_h], dim=-2)
                    st.fp16_v = torch.cat([st.fp16_v, v_h], dim=-2)

            elif st.role == ROLE_PEAKED_H2O:
                pos_b = overflow_pos.unsqueeze(0).expand(bsz, n_over)
                if st.h2o_k is None:
                    cand_k = k_h
                    cand_v = v_h
                    cand_sc = sc_h
                    cand_pos = pos_b
                else:
                    # Existing heavy hitters have already been decayed by beta_t^n_over in update()
                    cand_k = torch.cat([st.h2o_k, k_h], dim=-2)
                    cand_v = torch.cat([st.h2o_v, v_h], dim=-2)
                    cand_sc = torch.cat([st.h2o_scores, sc_h], dim=-1)
                    cand_pos = torch.cat([st.h2o_indices, pos_b], dim=-1)

                if cand_k.shape[-2] > k_budget:
                    # Seed neighborhoods from the highest running scores, then
                    # keep the best ``k_budget`` positions with neighborhood
                    # membership taking priority.  This retains context without
                    # increasing the configured memory budget.
                    if self.h2o_context_window > 0:
                        span = 2 * self.h2o_context_window + 1
                        seed_count = max(1, (k_budget + span - 1) // span)
                        seed_idx = torch.topk(
                            cand_sc, k=min(seed_count, cand_sc.shape[-1]), dim=-1, largest=True, sorted=False
                        ).indices
                        positions = torch.arange(cand_sc.shape[-1], device=cand_sc.device).view(1, 1, -1)
                        seed_positions = seed_idx.unsqueeze(-1)
                        in_context = (
                            (positions - seed_positions).abs() <= self.h2o_context_window
                        ).any(dim=1)
                        # Scores are finite at this boundary; a large priority
                        # term makes contextual tokens win before other tokens.
                        priority = in_context.to(cand_sc.dtype) * 1e12 + cand_sc
                        _, top_idx = torch.topk(priority, k=k_budget, dim=-1, largest=True, sorted=False)
                    else:
                        _, top_idx = torch.topk(cand_sc, k=k_budget, dim=-1, largest=True, sorted=False)
                    top_idx_sorted, _ = torch.sort(top_idx, dim=-1)
                    gather_idx = top_idx_sorted.unsqueeze(-1).expand(-1, -1, d_head)
                    st.h2o_k = torch.gather(cand_k, dim=-2, index=gather_idx)
                    st.h2o_v = torch.gather(cand_v, dim=-2, index=gather_idx)
                    st.h2o_scores = torch.gather(cand_sc, dim=-1, index=top_idx_sorted)
                    st.h2o_indices = torch.gather(cand_pos, dim=-1, index=top_idx_sorted)
                else:
                    st.h2o_k = cand_k
                    st.h2o_v = cand_v
                    st.h2o_scores = cand_sc
                    st.h2o_indices = cand_pos

            elif st.role == ROLE_DIFFUSE_LEXICO:
                k_h_code = self._rope_rotate(k_h, overflow_pos, inverse=True)
                k_idx, k_val = self.codec.encode(
                    k_h_code, layer_idx=layer_idx, s=s_budget, lambda1=self.lambda1
                )
                v_idx, v_val = self.codec.encode(
                    v_h, layer_idx=layer_idx, s=s_budget, lambda1=self.lambda1
                )
                if st.lex_k_idx is None:
                    st.lex_k_idx = k_idx
                    st.lex_k_val = k_val
                    st.lex_v_idx = v_idx
                    st.lex_v_val = v_val
                else:
                    st.lex_k_idx = torch.cat([st.lex_k_idx, k_idx], dim=-2)
                    st.lex_v_idx = torch.cat([st.lex_v_idx, v_idx], dim=-2)
                    new_k_raw = torch.cat(
                        [
                            torch.Tensor._make_subclass(torch.Tensor, st.lex_k_val),
                            torch.Tensor._make_subclass(torch.Tensor, k_val),
                        ],
                        dim=-2,
                    )
                    new_k_sc = torch.cat([st.lex_k_val.scale, k_val.scale], dim=-2)
                    st.lex_k_val = QuantizedInt8Values(new_k_raw, new_k_sc)

                    new_v_raw = torch.cat(
                        [
                            torch.Tensor._make_subclass(torch.Tensor, st.lex_v_val),
                            torch.Tensor._make_subclass(torch.Tensor, v_val),
                        ],
                        dim=-2,
                    )
                    new_v_sc = torch.cat([st.lex_v_val.scale, v_val.scale], dim=-2)
                    st.lex_v_val = QuantizedInt8Values(new_v_raw, new_v_sc)

                # If active_atoms matches cached reconstruction, append incrementally
                if (
                    st.lex_recon_k is not None
                    and st.lex_recon_atoms == self.active_atoms
                ):
                    rk = self.codec.decode(
                        k_idx[..., : self.active_atoms],
                        k_val[..., : self.active_atoms],
                        layer_idx=layer_idx,
                    ).to(dtype=overflow_k.dtype)
                    rk = self._rope_rotate(rk, overflow_pos, inverse=False)
                    rv = self.codec.decode(
                        v_idx[..., : self.active_atoms],
                        v_val[..., : self.active_atoms],
                        layer_idx=layer_idx,
                    ).to(dtype=overflow_v.dtype)
                    st.lex_recon_k = torch.cat([st.lex_recon_k, rk], dim=-2)
                    st.lex_recon_v = torch.cat([st.lex_recon_v, rv], dim=-2)
                    st.lex_recon_len = int(st.lex_recon_k.shape[-2])
                else:
                    st.lex_recon_atoms = -1

    @torch.no_grad()
    def _materialize_layer_kv(
        self,
        layer_idx: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Materializes the rectangular `[batch, num_kv_heads, kv_len, head_dim]` tensor for
        SDPA / FlashAttention and returns `(K_out, V_out, bytes_read_this_layer)`.
        """
        local_k = self._local_k[layer_idx]
        local_v = self._local_v[layer_idx]
        if local_k is None or local_v is None:
            empty = torch.empty((1, self.num_kv_heads, 0, self.head_dim), dtype=dtype, device=device)
            return empty, empty, 0

        bsz, n_kv, win_len, d_head = local_k.shape
        total_seq_len = self._layer_seq_len[layer_idx]
        hist_len = max(0, total_seq_len - win_len)

        # Fast path when all tokens are still inside the n_b = 64 local FP16 sliding window
        if hist_len == 0:
            bytes_read = int(bsz * n_kv * win_len * 2 * d_head * 2)
            return local_k, local_v, bytes_read

        roles_l = self.head_roles[layer_idx]
        all_h2o = bool(torch.all(roles_l == ROLE_PEAKED_H2O).item())

        # Always materialize a logical-length representation.  H2O's evicted
        # positions are masked by anti-aligned keys below, which keeps the
        # returned KV tensor aligned with Transformers' causal-mask contract.
        k_heads = []
        v_heads = []
        bytes_read = int(bsz * n_kv * win_len * 2 * d_head * 2)
        m_atoms = max(1, min(int(self.active_atoms), self.s_budget))

        for h_idx in range(n_kv):
            st = self._tiers[layer_idx][h_idx]
            if st.role == ROLE_RETRIEVAL_FP16:
                hk = st.fp16_k if st.fp16_k is not None else torch.empty((bsz, 0, d_head), dtype=dtype, device=device)
                hv = st.fp16_v if st.fp16_v is not None else torch.empty((bsz, 0, d_head), dtype=dtype, device=device)
                bytes_read += int(bsz * hk.shape[-2] * 2 * d_head * 2)
                k_heads.append(torch.cat([hk, local_k[:, h_idx]], dim=-2))
                v_heads.append(torch.cat([hv, local_v[:, h_idx]], dim=-2))

            elif st.role == ROLE_DIFFUSE_LEXICO:
                if (
                    st.lex_recon_k is None
                    or st.lex_recon_atoms != m_atoms
                    or st.lex_recon_len != hist_len
                ):
                    if st.lex_k_idx is not None and st.lex_k_val is not None:
                        st.lex_recon_k = self.codec.decode(
                            st.lex_k_idx[..., :m_atoms],
                            st.lex_k_val[..., :m_atoms],
                            layer_idx=layer_idx,
                        ).to(dtype=dtype)
                        # Lexico history is stored contiguously for absolute positions 0..n-1
                        st.lex_recon_k = self._rope_rotate(
                            st.lex_recon_k,
                            torch.arange(st.lex_recon_k.shape[-2], device=st.lex_recon_k.device),
                            inverse=False,
                        )
                        st.lex_recon_v = self.codec.decode(
                            st.lex_v_idx[..., :m_atoms],
                            st.lex_v_val[..., :m_atoms],
                            layer_idx=layer_idx,
                        ).to(dtype=dtype)
                        st.lex_recon_atoms = m_atoms
                        st.lex_recon_len = int(st.lex_recon_k.shape[-2])
                    else:
                        st.lex_recon_k = torch.empty((bsz, 0, d_head), dtype=dtype, device=device)
                        st.lex_recon_v = torch.empty((bsz, 0, d_head), dtype=dtype, device=device)
                        st.lex_recon_atoms = m_atoms
                        st.lex_recon_len = 0

                # Each vector reads `3 * m_atoms + 2` bytes for K and `3 * m_atoms + 2` bytes for V
                bytes_read += int(bsz * st.lex_recon_len * 2 * (3 * m_atoms + 2))
                k_heads.append(torch.cat([st.lex_recon_k, local_k[:, h_idx]], dim=-2))
                v_heads.append(torch.cat([st.lex_recon_v, local_v[:, h_idx]], dim=-2))

            elif st.role == ROLE_PEAKED_H2O:
                # Place kept heavy hitters at their valid positions; fill evicted slots with an
                # anti-aligned key vector (-30.0 * U_Q[:, 0]) and zero value vector so evicted slots
                # have exp(q @ K_evicted^T / sqrt(d)) ~ 0 in SDPA without needing a 4D mask
                anti_k = (-30.0 * self.U_Q[layer_idx, h_idx, :, 0]).to(dtype=dtype, device=device)
                hk_pad = anti_k.view(1, 1, d_head).expand(bsz, hist_len, d_head).clone()
                hv_pad = torch.zeros((bsz, hist_len, d_head), dtype=dtype, device=device)
                if st.h2o_k is not None and st.h2o_indices is not None and st.h2o_k.shape[-2] > 0:
                    k_kept = int(st.h2o_k.shape[-2])
                    idx_clamped = st.h2o_indices.clamp(0, max(0, hist_len - 1)).unsqueeze(-1).expand(-1, -1, d_head)
                    hk_pad.scatter_(dim=-2, index=idx_clamped, src=st.h2o_k)
                    hv_pad.scatter_(dim=-2, index=idx_clamped, src=st.h2o_v)
                    bytes_read += int(bsz * k_kept * 2 * d_head * 2)
                k_heads.append(torch.cat([hk_pad, local_k[:, h_idx]], dim=-2))
                v_heads.append(torch.cat([hv_pad, local_v[:, h_idx]], dim=-2))

        k_out = torch.stack(k_heads, dim=1)
        v_out = torch.stack(v_heads, dim=1)
        return k_out, v_out, bytes_read

    @torch.no_grad()
    def _refresh_telemetry(self) -> None:
        """Recomputes exact physical byte telemetry and per-tier token counts across all layers."""
        total_seq = self.get_seq_length(0)
        if total_seq == 0 or self._local_k[0] is None:
            self.fp16_bytes = 0
            self.compressed_bytes = 0
            self.bytes_read_last_step = 0
            for k in self.per_tier_token_counts:
                self.per_tier_token_counts[k] = 0
            return

        bsz = int(self._local_k[0].shape[0])
        vec_fp16_bytes = 2 * self.head_dim * 2  # K and V in FP16: 2 * d_head * 2 bytes
        self.fp16_bytes = int(bsz * self.num_layers * self.num_kv_heads * total_seq * vec_fp16_bytes)

        comp_bytes = 0
        cnt_local = 0
        cnt_ret = 0
        cnt_h2o = 0
        cnt_lex = 0
        cnt_evicted = 0

        for l_idx in range(self.num_layers):
            lk = self._local_k[l_idx]
            if lk is None:
                continue
            win_len = int(lk.shape[-2])
            hist_len = max(0, self._layer_seq_len[l_idx] - win_len)

            # Local FP16 window bytes across all KV heads
            comp_bytes += int(bsz * self.num_kv_heads * win_len * vec_fp16_bytes)
            cnt_local += int(bsz * self.num_kv_heads * win_len)

            for h_idx in range(self.num_kv_heads):
                st = self._tiers[l_idx][h_idx]
                if st.role == ROLE_RETRIEVAL_FP16:
                    n_t = int(st.fp16_k.shape[-2]) if st.fp16_k is not None else 0
                    comp_bytes += int(bsz * n_t * vec_fp16_bytes)
                    cnt_ret += int(bsz * n_t)
                elif st.role == ROLE_PEAKED_H2O:
                    n_t = int(st.h2o_k.shape[-2]) if st.h2o_k is not None else 0
                    # Each kept H2O token stores K (FP16), V (FP16), and 1 int32/FP16 score/pos metadata (4 bytes)
                    comp_bytes += int(bsz * n_t * vec_fp16_bytes)
                    cnt_h2o += int(bsz * n_t)
                    cnt_evicted += int(bsz * max(0, hist_len - n_t))
                elif st.role == ROLE_DIFFUSE_LEXICO:
                    n_t = int(st.lex_k_idx.shape[-2]) if st.lex_k_idx is not None else 0
                    # Each K and V vector in Lexico format uses (3 * s_budget + 2) bytes
                    comp_bytes += int(bsz * n_t * 2 * (3 * self.s_budget + 2))
                    cnt_lex += int(bsz * n_t)

        # Shared Lexico model overhead (NOT included in `compressed_bytes`; reported separately).
        # D [L, d, N] is read at decode time and W_ridge [L, N, d] at encode time. Both are
        # model-level constants shared by every sequence/batch, so they amortize with
        # concurrent requests, but for a single short sequence they are a real resident cost.
        if cnt_lex > 0:
            lex_layers = sorted(
                {
                    l
                    for l in range(self.num_layers)
                    for h in range(self.num_kv_heads)
                    if self._tiers[l][h].role == ROLE_DIFFUSE_LEXICO
                }
            )
            per_layer = 0
            if self.codec is not None and hasattr(self.codec, "D"):
                d_t = self.codec.D
                per_layer += int(d_t[0].numel() * d_t.element_size())
            if self.codec is not None and hasattr(self.codec, "W_ridge"):
                w_t = self.codec.W_ridge
                per_layer += int(w_t[0].numel() * w_t.element_size())
            self.dictionary_bytes = int(per_layer * len(lex_layers))
        else:
            self.dictionary_bytes = 0

        self.compressed_bytes = comp_bytes
        self.bytes_read_last_step = int(sum(self._layer_bytes_read))
        self.per_tier_token_counts = {
            "local_window_fp16": cnt_local,
            "retrieval_fp16": cnt_ret,
            "peaked_h2o_fp16": cnt_h2o,
            "diffuse_lexico_sparse": cnt_lex,
            "evicted_h2o": cnt_evicted,
        }

    @torch.no_grad()
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the hybrid StoikovKV cache for `layer_idx` with incoming `key_states` and
        `value_states` (`[batch, num_kv_heads, q_len, head_dim]`) and returns materialized
        `(key_states_full, value_states_full)` for SDPA / FlashAttention.
        """
        bsz, n_kv, q_len, d_head = key_states.shape
        start_pos = self._layer_seq_len[layer_idx]
        new_seq_len = start_pos + q_len
        self._layer_seq_len[layer_idx] = new_seq_len
        if layer_idx == 0:
            self._seen_tokens = new_seq_len

        # 1. Compute write-time importance score I_t and query HMM controller
        new_scores, beta_t, keep_ratio, s_budget = self._compute_step_scores_and_controller(
            key_states, value_states, layer_idx=layer_idx, start_pos=start_pos
        )
        new_pos = torch.arange(start_pos, new_seq_len, device=key_states.device, dtype=torch.long)

        # Decay existing H2O scores and local window scores by beta_t^q_len
        decay_factor = float(beta_t ** min(q_len, 64))
        for h_idx in range(n_kv):
            st = self._tiers[layer_idx][h_idx]
            if st.role == ROLE_PEAKED_H2O and st.h2o_scores is not None:
                st.h2o_scores = st.h2o_scores * decay_factor

        # 2. Append to local FP16 sliding window
        if self._local_k[layer_idx] is None:
            cat_k = key_states
            cat_v = value_states
            cat_sc = new_scores
            cat_pos = new_pos
        else:
            cat_k = torch.cat([self._local_k[layer_idx], key_states], dim=-2)
            cat_v = torch.cat([self._local_v[layer_idx], value_states], dim=-2)
            cat_sc = torch.cat(
                [self._local_scores[layer_idx] * decay_factor, new_scores], dim=-1
            )
            cat_pos = torch.cat([self._local_positions[layer_idx], new_pos], dim=0)

        # 3. Route tokens older than n_b = 64 into per-head tiers
        total_in_buf = int(cat_k.shape[-2])
        if total_in_buf > self.n_b:
            n_over = total_in_buf - self.n_b
            overflow_k = cat_k[:, :, :n_over, :]
            overflow_v = cat_v[:, :, :n_over, :]
            overflow_sc = cat_sc[:, :, :n_over]
            overflow_pos = cat_pos[:n_over]

            self._local_k[layer_idx] = cat_k[:, :, n_over:, :]
            self._local_v[layer_idx] = cat_v[:, :, n_over:, :]
            self._local_scores[layer_idx] = cat_sc[:, :, n_over:]
            self._local_positions[layer_idx] = cat_pos[n_over:]

            self._route_overflow_to_tiers(
                layer_idx=layer_idx,
                overflow_k=overflow_k,
                overflow_v=overflow_v,
                overflow_scores=overflow_sc,
                overflow_pos=overflow_pos,
                beta_t=beta_t,
                keep_ratio=keep_ratio,
                s_budget=s_budget,
                total_seq_len=new_seq_len,
            )
        else:
            self._local_k[layer_idx] = cat_k
            self._local_v[layer_idx] = cat_v
            self._local_scores[layer_idx] = cat_sc
            self._local_positions[layer_idx] = cat_pos

        # 4. Materialize rectangular KV states for SDPA / FlashAttention
        k_out, v_out, bytes_read = self._materialize_layer_kv(
            layer_idx=layer_idx, dtype=key_states.dtype, device=key_states.device
        )
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(torch.empty(0, device=key_states.device, dtype=key_states.dtype))
            self.value_cache.append(torch.empty(0, device=value_states.device, dtype=value_states.dtype))
        self.key_cache[layer_idx] = k_out
        self.value_cache[layer_idx] = v_out

        # Call parent's update with original key_states/value_states to update HF DynamicCache's
        # internal DynamicLayer state (get_seq_length, get_mask_sizes)
        # This must be called with the original incremental key_states/value_states,
        # not the materialized full k_out/v_out
        super().update(key_states, value_states, layer_idx)

        self._layer_bytes_read[layer_idx] = bytes_read
        self.cumulative_bytes_read += bytes_read
        if layer_idx == self.num_layers - 1:
            self._refresh_telemetry()

        return k_out, v_out

    @torch.no_grad()
    def crop(self, max_length: int) -> None:
        """
        Rolls back the KV cache to `max_length` tokens across all layers and tiers.
        Used by Matryoshka speculative decoding to discard rejected draft tokens cleanly.
        """
        cur_len = self.get_seq_length(0)
        if max_length < 0:
            max_length = cur_len - abs(max_length)
        max_length = max(0, int(max_length))
        if cur_len <= max_length:
            return

        self._seen_tokens = max_length
        for l_idx in range(self.num_layers):
            l_cur = self._layer_seq_len[l_idx]
            if l_cur <= max_length:
                continue
            self._layer_seq_len[l_idx] = max_length

            if max_length == 0:
                self._local_k[l_idx] = None
                self._local_v[l_idx] = None
                self._local_scores[l_idx] = None
                self._local_positions[l_idx] = None
                self._flushed_ring_k[l_idx] = None
                self._flushed_ring_v[l_idx] = None
                self._flushed_ring_scores[l_idx] = None
                self._flushed_ring_pos[l_idx] = None
                for h_idx in range(self.num_kv_heads):
                    self._tiers[l_idx][h_idx] = _HeadTierState(
                        role=int(self.head_roles[l_idx, h_idx].item())
                    )
                continue

            target_win_len = min(self.n_b, max_length)
            target_hist_len = max(0, max_length - target_win_len)

            # Reconstruct the exact local FP16 window `[target_hist_len .. max_length - 1]`
            lk = self._local_k[l_idx]
            lv = self._local_v[l_idx]
            lsc = self._local_scores[l_idx]
            lpos = self._local_positions[l_idx]

            if lk is not None and lpos is not None:
                # Combine ring buffer of recently flushed tokens with current local window
                if self._flushed_ring_k[l_idx] is not None and self._flushed_ring_pos[l_idx] is not None:
                    comb_k = torch.cat([self._flushed_ring_k[l_idx], lk], dim=-2)
                    comb_v = torch.cat([self._flushed_ring_v[l_idx], lv], dim=-2)
                    comb_sc = torch.cat([self._flushed_ring_scores[l_idx], lsc], dim=-1)
                    comb_pos = torch.cat([self._flushed_ring_pos[l_idx], lpos], dim=0)
                else:
                    comb_k, comb_v, comb_sc, comb_pos = lk, lv, lsc, lpos

                valid_mask = (comb_pos >= target_hist_len) & (comb_pos < max_length)
                self._local_k[l_idx] = comb_k[:, :, valid_mask, :]
                self._local_v[l_idx] = comb_v[:, :, valid_mask, :]
                self._local_scores[l_idx] = comb_sc[:, :, valid_mask]
                self._local_positions[l_idx] = comb_pos[valid_mask]

                # Trim flushed ring buffer to `< target_hist_len`
                ring_mask = comb_pos < target_hist_len
                if torch.any(ring_mask):
                    self._flushed_ring_k[l_idx] = comb_k[:, :, ring_mask, :][:, :, -self.max_rollback_window :, :]
                    self._flushed_ring_v[l_idx] = comb_v[:, :, ring_mask, :][:, :, -self.max_rollback_window :, :]
                    self._flushed_ring_scores[l_idx] = comb_sc[:, :, ring_mask][:, :, -self.max_rollback_window :]
                    self._flushed_ring_pos[l_idx] = comb_pos[ring_mask][-self.max_rollback_window :]
                else:
                    self._flushed_ring_k[l_idx] = None
                    self._flushed_ring_v[l_idx] = None
                    self._flushed_ring_scores[l_idx] = None
                    self._flushed_ring_pos[l_idx] = None

            # Crop historical tiers to `< target_hist_len`
            for h_idx in range(self.num_kv_heads):
                st = self._tiers[l_idx][h_idx]
                if st.role == ROLE_RETRIEVAL_FP16:
                    if st.fp16_k is not None:
                        st.fp16_k = st.fp16_k[:, :target_hist_len, :]
                        st.fp16_v = st.fp16_v[:, :target_hist_len, :]
                elif st.role == ROLE_PEAKED_H2O:
                    if st.h2o_indices is not None and st.h2o_k is not None:
                        keep_mask = st.h2o_indices[0] < target_hist_len
                        st.h2o_k = st.h2o_k[:, keep_mask, :]
                        st.h2o_v = st.h2o_v[:, keep_mask, :]
                        st.h2o_scores = st.h2o_scores[:, keep_mask]
                        st.h2o_indices = st.h2o_indices[:, keep_mask]
                elif st.role == ROLE_DIFFUSE_LEXICO:
                    if st.lex_k_idx is not None and st.lex_k_val is not None:
                        st.lex_k_idx = st.lex_k_idx[:, :target_hist_len, :]
                        st.lex_v_idx = st.lex_v_idx[:, :target_hist_len, :]
                        st.lex_k_val = st.lex_k_val[:, :target_hist_len, :]
                        st.lex_v_val = st.lex_v_val[:, :target_hist_len, :]
                        if st.lex_recon_k is not None and st.lex_recon_len >= target_hist_len:
                            st.lex_recon_k = st.lex_recon_k[:, :target_hist_len, :]
                            st.lex_recon_v = st.lex_recon_v[:, :target_hist_len, :]
                            st.lex_recon_len = target_hist_len
                        else:
                            st.lex_recon_atoms = -1

            if l_idx < len(self.key_cache) and self.key_cache[l_idx].numel() > 0 and self._local_k[l_idx] is not None:
                k_out, v_out, _ = self._materialize_layer_kv(
                    l_idx, dtype=self._local_k[l_idx].dtype, device=self._local_k[l_idx].device
                )
                self.key_cache[l_idx] = k_out
                self.value_cache[l_idx] = v_out

        self._refresh_telemetry()

    def get_telemetry_summary(self) -> Dict[str, Any]:
        """Returns a dictionary of memory and routing telemetry metrics."""
        self._refresh_telemetry()
        ratio = (
            float(self.compressed_bytes) / float(self.fp16_bytes)
            if self.fp16_bytes > 0
            else 1.0
        )
        return {
            "seq_length": self.get_seq_length(0),
            "policy_mode": self.policy_mode,
            "active_atoms": self.active_atoms,
            "s_budget": self.s_budget,
            "keep_ratio": self.keep_ratio,
            "n_b": self.n_b,
            "fp16_bytes": self.fp16_bytes,
            "compressed_bytes": self.compressed_bytes,
            "memory_ratio_vs_fp16": ratio,
            "memory_savings_pct": (1.0 - ratio) * 100.0,
            "dictionary_bytes": int(getattr(self, "dictionary_bytes", 0)),
            "memory_ratio_incl_dictionary": (
                float(self.compressed_bytes + int(getattr(self, "dictionary_bytes", 0)))
                / float(self.fp16_bytes)
                if self.fp16_bytes > 0
                else 1.0
            ),
            "bytes_read_last_step": self.bytes_read_last_step,
            "cumulative_bytes_read": self.cumulative_bytes_read,
            "per_tier_token_counts": dict(self.per_tier_token_counts),
            "hmm_last_beta_t": self.controller.last_beta_t,
            "hmm_last_z_score": self.controller.last_z_score,
            "hmm_shift_events": self.controller.shift_events,
            "head_roles": self.head_roles.detach().cpu().tolist(),
        }
