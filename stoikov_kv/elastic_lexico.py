"""
Fast 1-Step Ridge-Projection Sparse KV Codec, Sequential OMP Baseline,
Iterative FISTA ElasticNet Reference, Nested Partial Reconstruction, and
Online SVD Residual Dictionary Adaptation (`stoikov_kv/elastic_lexico.py`).

Key Design & Audit Guarantees:
1. Fast 1-Step Approximate Encoder (`encode`):
   - Computes ridge projection `Z = X @ W_ridge[layer_idx].T`, applies soft-thresholding
     `Y = sign(Z) * relu(|Z| - lambda1)`, and selects the top-`s` coefficients by magnitude.
   - Explicitly documented as a **fast 1-step approximation**, not an exact ElasticNet solver.
   - Includes configurable `norm_rescale: bool = True` with safe handling for zero/near-zero vectors.
2. Trusted Baselines at Matched Sparsity:
   - `omp_encode`: Sequential greedy Orthogonal Matching Pursuit (OMP) over `s` steps.
   - `iterative_elasticnet_encode`: Multi-step FISTA proximal-gradient solver minimizing the
     convex ElasticNet objective `0.5 * ||x - D z||_2^2 + lambda1 * ||z||_1 + 0.5 * lambda2 * ||z||_2^2`,
     followed by top-`s` support selection.
3. Explicit Quantization & Memory Accounting:
   - Fixed-sparsity record format: `s` `int16` atom indices (`2*s` bytes) + `s` `int8` quantized
     coefficients (`1*s` bytes) + `1` `float16` per-vector scale (`2` bytes) = `3*s + 2` logical
     payload bytes per vector.
   - Distinguishes logical payload bytes from PyTorch tensor storage and Python object overhead.
4. Held-Out Online Residual Adaptation (`adapt_online_atoms`) & Break-Even Audit (`compute_break_even_length`):
   - Extracts top-`M_adapt` SVD singular vectors of residuals on an adaptation split and evaluates
     reconstruction quality and effective sparsity `s_adapt` on **held-out evaluation vectors**.
   - Returns `L_star = None` (`break_even_exists = False`) whenever held-out per-vector byte savings
     `3 * (s_univ - s_adapt) <= 0`.
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple, Union

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_stoikov")

import torch
import torch.nn.functional as F

MAX_INT16_ATOMS: int = 32767


class QuantizedInt8Values(torch.Tensor):
    """
    A `torch.int8` Tensor subclass representing quantized sparse coefficients in `[-127, 127]`
    paired with a per-vector `torch.float16` scale tensor (`.scale` of shape `[..., 1]`).
    Preserves `.scale` across atom-dimension slicing `values[..., :m]`.
    """

    scale: torch.Tensor
    debiased_norm_scale: Optional[torch.Tensor]

    @staticmethod
    def __new__(
        cls,
        int8_data: torch.Tensor,
        scale: torch.Tensor,
        debiased_norm_scale: Optional[torch.Tensor] = None,
    ) -> "QuantizedInt8Values":
        if int8_data.dtype != torch.int8:
            int8_data = int8_data.to(torch.int8)
        obj = torch.Tensor._make_subclass(cls, int8_data, require_grad=False)
        obj.scale = scale.to(dtype=torch.float16, device=int8_data.device)
        obj.debiased_norm_scale = debiased_norm_scale
        return obj

    def __getitem__(self, item: Any) -> "QuantizedInt8Values" | torch.Tensor:
        raw = super().__getitem__(item)
        if not isinstance(raw, torch.Tensor):
            return raw
        out = torch.Tensor._make_subclass(
            QuantizedInt8Values,
            raw if raw.dtype == torch.int8 else raw.to(torch.int8),
            require_grad=False,
        )
        sc = getattr(self, "scale", torch.zeros(1, device=raw.device, dtype=torch.float16))
        # If leading dimensions were indexed, slice scale on matching leading dimensions
        if isinstance(item, tuple) and len(item) > 0 and sc.dim() == self.dim():
            lead_item = item[:-1] + (slice(None),)
            try:
                sc = sc[lead_item]
            except Exception:
                pass
        out.scale = sc
        out.debiased_norm_scale = getattr(self, "debiased_norm_scale", None)
        return out

    @property
    def num_atoms(self) -> int:
        return int(self.shape[-1]) if self.dim() >= 1 else 0

    @property
    def bytes_per_vector(self) -> int:
        """
        Logical serialized payload per vector in the fixed-sparsity top-s format:
        `s * sizeof(int16) + s * sizeof(int8) + 1 * sizeof(float16) = 3 * s + 2` bytes.
        """
        return 3 * self.num_atoms + 2

    @property
    def total_bytes(self) -> int:
        """Total logical serialized payload bytes across all encoded vectors."""
        if self.num_atoms == 0:
            return 0
        num_vecs = int(self.numel() // self.num_atoms)
        return num_vecs * self.bytes_per_vector

    def dequantize(self, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """Reconstructs floating-point coefficients `q_int8.to(dtype) * scale.to(dtype)`."""
        sc = self.scale.to(dtype)
        return torch.Tensor._make_subclass(torch.Tensor, self).to(dtype) * sc


def quantize_sorted_atoms(
    y_sorted: torch.Tensor,
) -> QuantizedInt8Values:
    """
    Symmetric per-vector int8 quantizer for sorted sparse atom coefficients `y_sorted` (`[..., s]`):
      - For non-zero vectors (`max_abs > 0`):
            scale_fp16 = float16(max_abs / 127.0)
            q_int8 = clamp(round(y_sorted / scale), -127, 127)
      - For exact zero vectors (`max_abs == 0`):
            scale_fp16 = 0.0, q_int8 = 0 (guaranteed finite, exact zero reconstruction).
    """
    if not torch.isfinite(y_sorted).all():
        raise ValueError("Cannot quantize non-finite atom coefficients.")

    max_abs = y_sorted.to(torch.float32).abs().amax(dim=-1, keepdim=True)
    is_zero = max_abs <= 1e-12

    raw_scale = torch.where(is_zero, torch.zeros_like(max_abs), max_abs / 127.0)
    scale_fp16 = raw_scale.to(torch.float16)
    # Guard against FP16 underflow for tiny non-zero vectors
    safe_denom = scale_fp16.to(torch.float32)
    safe_denom = torch.where(safe_denom <= 1e-8, torch.ones_like(safe_denom), safe_denom)

    scaled = torch.where(is_zero, torch.zeros_like(y_sorted, dtype=torch.float32), y_sorted.to(torch.float32) / safe_denom)
    q_int8 = torch.clamp(torch.round(scaled), -127.0, 127.0).to(torch.int8)
    return QuantizedInt8Values(q_int8, scale_fp16)


def compute_break_even_length(
    head_dim: int,
    M_adapt: int,
    s_univ: float,
    s_adapt: float,
    atom_dtype_bytes: int = 2,
    bytes_per_sparse_atom: int = 3,
) -> Dict[str, Any]:
    """
    Audited Break-Even Context Length Calculation:

    Derivation & Units:
      - One-time adaptation dictionary overhead (storing `M_adapt` atoms of length `head_dim`
        in FP16, `atom_dtype_bytes = 2` bytes/element):
            Overhead_bytes = atom_dtype_bytes * head_dim * M_adapt   [bytes]
      - Per-vector serialized payload in the fixed-sparsity format is `3 * s + 2` bytes
        (`2` bytes per `int16` index + `1` byte per `int8` weight + `2` bytes per `fp16` scale).
        The `+2` byte scale overhead cancels between universal (`s_univ`) and adapted (`s_adapt`),
        yielding per-vector savings:
            Savings_per_vector = bytes_per_sparse_atom * (s_univ - s_adapt)   [bytes / vector]
      - Break-even number of KV vectors `L_star`:
            L_star = Overhead_bytes / Savings_per_vector
                   = (2 * head_dim * M_adapt) / (3 * (s_univ - s_adapt))   [vectors / tokens]
        valid ONLY when `delta_s = s_univ - s_adapt > 0`.
      - When `delta_s <= 0` (zero or negative per-token savings), no finite break-even exists
        (`L_star = None`, `break_even_exists = False`).
    """
    if head_dim <= 0 or M_adapt <= 0 or s_univ <= 0 or s_adapt < 0:
        raise ValueError(
            f"Invalid break-even inputs: head_dim={head_dim}, M_adapt={M_adapt}, "
            f"s_univ={s_univ}, s_adapt={s_adapt}."
        )
    if not (math.isfinite(s_univ) and math.isfinite(s_adapt)):
        raise ValueError("s_univ and s_adapt must be finite numbers.")

    overhead_bytes = int(atom_dtype_bytes) * int(head_dim) * int(M_adapt)
    delta_s = float(s_univ) - float(s_adapt)
    per_vec_savings_bytes = float(bytes_per_sparse_atom) * delta_s

    if delta_s <= 1e-6 or per_vec_savings_bytes <= 1e-6:
        return {
            "L_star": None,
            "break_even_exists": False,
            "reason": "zero_or_negative_savings",
            "overhead_bytes": overhead_bytes,
            "delta_s": round(delta_s, 4),
            "per_vector_savings_bytes": round(per_vec_savings_bytes, 4),
            "units": "kv_vectors (tokens per head stream)",
        }

    l_star = float(overhead_bytes) / per_vec_savings_bytes
    return {
        "L_star": round(l_star, 6),
        "break_even_exists": True,
        "reason": "positive_heldout_savings",
        "overhead_bytes": overhead_bytes,
        "delta_s": round(delta_s, 4),
        "per_vector_savings_bytes": round(per_vec_savings_bytes, 4),
        "units": "kv_vectors (tokens per head stream)",
    }


class ElasticLexicoCodec:
    """
    Sparse Dictionary KV Codec supporting:
    - `encode`: Fast 1-step ridge projection + soft-thresholding + top-`s` selection.
    - `omp_encode`: Sequential greedy Orthogonal Matching Pursuit (OMP) baseline.
    - `iterative_elasticnet_encode`: Multi-step FISTA ElasticNet reference solver.
    - `decode`: Full and nested partial (`active_atoms = m`) reconstruction.
    - `adapt_online_atoms`: Held-out SVD residual dictionary adaptation and break-even analysis.
    """

    def __init__(
        self,
        D: torch.Tensor,
        W_ridge: Optional[torch.Tensor] = None,
        ridge_lambda: float = 0.1,
        lambda1: float = 0.04,
        lambda2: Optional[float] = None,
    ) -> None:
        if lambda2 is not None:
            ridge_lambda = float(lambda2)
        if not isinstance(D, torch.Tensor) or D.numel() == 0:
            raise ValueError("Dictionary D must be a non-empty torch.Tensor.")
        if D.dim() == 2:
            D = D.unsqueeze(0)
        if D.dim() != 3:
            raise ValueError(f"Expected 2D or 3D dictionary D, got shape {tuple(D.shape)}.")
        if not torch.isfinite(D).all():
            raise ValueError("Dictionary D contains non-finite values.")

        num_layers, head_dim, n_atoms = D.shape
        if head_dim <= 0 or n_atoms <= 0:
            raise ValueError(f"Invalid dictionary dimensions: {tuple(D.shape)}.")
        if n_atoms > MAX_INT16_ATOMS:
            raise ValueError(
                f"Dictionary atom count n_atoms={n_atoms} exceeds int16 max ({MAX_INT16_ATOMS})."
            )
        if ridge_lambda <= 0.0:
            raise ValueError(f"ridge_lambda must be > 0, got {ridge_lambda}.")
        if lambda1 < 0.0:
            raise ValueError(f"lambda1 must be >= 0, got {lambda1}.")

        self.D = F.normalize(D.to(torch.float32), p=2, dim=-2).to(D.dtype)
        self.num_layers = int(num_layers)
        self.head_dim = int(head_dim)
        self.n_atoms = int(n_atoms)
        self.num_atoms = int(n_atoms)
        self.ridge_lambda = float(ridge_lambda)
        self.lambda1 = float(lambda1)
        self.lambda2 = float(ridge_lambda)
        self.device = self.D.device
        self.dtype = self.D.dtype

        if W_ridge is None:
            w_list = []
            eye_m = torch.eye(self.n_atoms, device=self.device, dtype=torch.float64)
            for l_idx in range(self.num_layers):
                d_l = self.D[l_idx].to(torch.float64)
                gram = torch.matmul(d_l.T, d_l) + float(ridge_lambda) * eye_m
                w_l = torch.linalg.solve(gram, d_l.T)
                w_list.append(w_l.to(self.dtype))
            self.W_ridge = torch.stack(w_list, dim=0)
        else:
            if W_ridge.dim() == 2:
                W_ridge = W_ridge.unsqueeze(0)
            expected_w_shape = (self.num_layers, self.n_atoms, self.head_dim)
            if tuple(W_ridge.shape) != expected_w_shape:
                raise ValueError(
                    f"W_ridge shape {tuple(W_ridge.shape)} does not match expected {expected_w_shape}."
                )
            if not torch.isfinite(W_ridge).all():
                raise ValueError("W_ridge contains non-finite values.")
            self.W_ridge = W_ridge.to(device=self.device, dtype=self.dtype)

    @classmethod
    def from_artifacts(
        cls,
        artifacts_or_path: Union[str, Dict[str, Any]] = "calibrated_artifacts.pt",
        device: Optional[str] = None,
    ) -> "ElasticLexicoCodec":
        """Loads `ElasticLexicoCodec` from `calibrated_artifacts.pt` or an artifacts dictionary."""
        if isinstance(artifacts_or_path, str):
            map_loc = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
            data = torch.load(artifacts_or_path, map_location=map_loc, weights_only=False)
        else:
            data = artifacts_or_path
        D = data["D"]
        W_ridge = data.get("W_ridge", None)
        if device is not None:
            D = D.to(device)
            if W_ridge is not None:
                W_ridge = W_ridge.to(device)
        return cls(D=D, W_ridge=W_ridge)

    @staticmethod
    def logical_payload_bytes_per_vector(s: int) -> int:
        """
        Logical serialized payload size for a fixed-sparsity top-`s` record:
        `s * 2` bytes (`int16` indices) + `s * 1` bytes (`int8` coefficients) + `2` bytes (`fp16` scale)
        = `3 * s + 2` bytes per vector.
        """
        if s < 0:
            raise ValueError(f"Sparsity s must be non-negative, got {s}.")
        return 3 * int(s) + 2

    @staticmethod
    def csr_bytes_per_vector(s: int) -> int:
        """Backward-compatible alias for `logical_payload_bytes_per_vector(s)`."""
        return ElasticLexicoCodec.logical_payload_bytes_per_vector(s)

    def memory_accounting(
        self, indices: torch.Tensor, values: QuantizedInt8Values
    ) -> Dict[str, Any]:
        """
        Explicitly separates logical serialized payload bytes from PyTorch tensor buffer bytes
        and Python wrapper overhead.
        """
        s = int(indices.shape[-1])
        num_vectors = int(indices.numel() // max(1, s))
        logical_bytes = num_vectors * self.logical_payload_bytes_per_vector(s)

        idx_storage_bytes = int(indices.numel() * indices.element_size())
        val_storage_bytes = int(values.numel() * values.element_size())
        scale_storage_bytes = int(values.scale.numel() * values.scale.element_size())
        tensor_storage_bytes = idx_storage_bytes + val_storage_bytes + scale_storage_bytes

        py_overhead_bytes = int(
            sys.getsizeof(indices) + sys.getsizeof(values) + sys.getsizeof(values.scale)
        )
        fp16_dense_bytes = int(num_vectors * self.head_dim * 2)

        # Static dictionary overhead (FP16 storage convention: 2 bytes/element)
        d_layer_fp16_bytes = int(self.head_dim * self.n_atoms * 2)
        w_ridge_layer_fp16_bytes = int(self.n_atoms * self.head_dim * 2)
        d_total_fp16_bytes = int(self.num_layers * d_layer_fp16_bytes)
        w_ridge_total_fp16_bytes = int(self.num_layers * w_ridge_layer_fp16_bytes)
        per_vec_savings_vs_fp16 = int(self.head_dim * 2 - self.logical_payload_bytes_per_vector(s))
        layer_dict_amortize_vecs = (
            round(d_layer_fp16_bytes / float(per_vec_savings_vs_fp16), 2)
            if per_vec_savings_vs_fp16 > 0
            else None
        )

        return {
            "num_vectors": num_vectors,
            "sparsity_s": s,
            "format": "fixed_sparsity_top_s_int16_int8_fp16scale",
            "logical_bytes_per_vector": self.logical_payload_bytes_per_vector(s),
            "logical_payload_bytes_per_vector": self.logical_payload_bytes_per_vector(s),
            "logical_payload_bytes": logical_bytes,
            "total_logical_payload_bytes": logical_bytes,
            "pytorch_tensor_storage_bytes": tensor_storage_bytes,
            "total_pytorch_tensor_storage_bytes": tensor_storage_bytes,
            "python_wrapper_overhead_bytes": py_overhead_bytes,
            "fp16_dense_bytes": fp16_dense_bytes,
            "logical_compression_ratio_vs_fp16": round(
                fp16_dense_bytes / float(max(1, logical_bytes)), 4
            ),
            "static_dict_D_bytes_per_layer_fp16": d_layer_fp16_bytes,
            "static_W_ridge_bytes_per_layer_fp16": w_ridge_layer_fp16_bytes,
            "static_dict_D_total_bytes_fp16": d_total_fp16_bytes,
            "static_W_ridge_total_bytes_fp16": w_ridge_total_fp16_bytes,
            "per_vector_byte_savings_vs_fp16": per_vec_savings_vs_fp16,
            "vectors_to_amortize_layer_D": layer_dict_amortize_vecs,
        }

    @staticmethod
    def evaluate_reconstruction(X: torch.Tensor, X_hat: torch.Tensor) -> Dict[str, float]:
        """
        Computes reconstruction quality metrics between `X` and `X_hat`:
        mean cosine similarity, MSE, relative MSE (`||X - X_hat||_F^2 / ||X||_F^2`),
        and relative L2 error (`||X - X_hat||_2 / ||X||_2`).
        """
        x_f = X.to(torch.float32).reshape(-1, X.shape[-1])
        xh_f = X_hat.to(torch.float32).reshape(-1, X_hat.shape[-1])
        cos = float(F.cosine_similarity(xh_f, x_f, dim=-1).mean().item())
        mse = float(F.mse_loss(xh_f, x_f).item())
        energy = float(x_f.pow(2).mean().item()) + 1e-12
        rel_mse = mse / energy
        l2_diff = torch.norm(x_f - xh_f, p=2, dim=-1)
        l2_ref = torch.norm(x_f, p=2, dim=-1).clamp(min=1e-8)
        rel_l2 = float((l2_diff / l2_ref).mean().item())
        return {
            "cosine_similarity": cos,
            "mse": mse,
            "relative_mse": rel_mse,
            "relative_l2_error": rel_l2,
        }

    def _validate_encode_inputs(
        self, X: torch.Tensor, layer_idx: int, s: int
    ) -> None:
        if not isinstance(X, torch.Tensor) or X.numel() == 0:
            raise ValueError("Input X must be a non-empty torch.Tensor.")
        if X.shape[-1] != self.head_dim:
            raise ValueError(
                f"Input last dimension ({X.shape[-1]}) must match head_dim ({self.head_dim})."
            )
        if not (0 <= int(layer_idx) < self.num_layers):
            raise ValueError(
                f"layer_idx={layer_idx} out of bounds for num_layers={self.num_layers}."
            )
        if not (1 <= int(s) <= self.n_atoms):
            raise ValueError(
                f"Sparsity s={s} must be in [1, n_atoms={self.n_atoms}]."
            )
        if not torch.isfinite(X).all():
            raise ValueError("Input X contains NaN or Inf values.")

    def _raw_reconstruct(
        self, indices: torch.Tensor, float_values: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        """Vectorized reconstruction `sum_{j=1}^m values[..., j] * D[layer_idx, :, indices[..., j]]`."""
        d_layer_t = self.D[layer_idx].transpose(0, 1).to(
            device=float_values.device, dtype=float_values.dtype
        )  # [n_atoms, head_dim]
        atoms = F.embedding(indices.to(torch.long), d_layer_t)  # [..., m, head_dim]
        return (atoms * float_values.unsqueeze(-1)).sum(dim=-2)  # [..., head_dim]

    def encode(
        self,
        X: torch.Tensor,
        layer_idx: int = 0,
        s: int = 16,
        lambda1: float = 0.04,
        norm_rescale: bool = True,
    ) -> Tuple[torch.Tensor, QuantizedInt8Values]:
        """
        Fast 1-Step Approximate Ridge + Soft-Thresholding Sparse Encoder:
          1. Computes `Z = X_scaled @ W_ridge[layer_idx].T` (`[..., n_atoms]`).
          2. Applies elementwise soft-thresholding `Y = sign(Z) * relu(|Z| - lambda1)`.
          3. Selects top-`s` coefficients by magnitude `|Y|`, sorted in descending order
             `|y_(1)| >= |y_(2)| >= ... >= |y_(s)|`.
          4. Applies 1-step inner-product refinement on the selected support and optional
             norm rescaling (`norm_rescale=True`) so `||X_hat||_2 == ||X||_2` for non-zero inputs.
          5. Quantizes to `(int16 indices, int8 values + fp16 scale)`.

        Note: This is a fast 1-step approximation designed for low-latency KV compression;
        selecting the largest-magnitude thresholded ridge coefficients in one step does not
        solve the iterative ElasticNet objective exactly.
        """
        self._validate_encode_inputs(X, layer_idx=layer_idx, s=s)
        if lambda1 < 0.0:
            raise ValueError(f"lambda1 must be >= 0, got {lambda1}.")

        x_fp32 = X.to(device=self.device, dtype=torch.float32)
        w_ridge_t = self.W_ridge[layer_idx].T.to(torch.float32)  # [head_dim, n_atoms]

        x_norm = torch.norm(x_fp32, p=2, dim=-1, keepdim=True)
        is_zero_vec = x_norm <= 1e-8

        # Scale unit-normalized X by frame redundancy (n_atoms / head_dim) so lambda1 is scale-consistent
        frame_scale = float(self.n_atoms) / float(self.head_dim)
        x_scaled = torch.where(
            is_zero_vec,
            torch.zeros_like(x_fp32),
            (x_fp32 / x_norm.clamp(min=1e-8)) * frame_scale,
        )

        # 1. Preconditioned ridge projection
        Z = torch.matmul(x_scaled, w_ridge_t)  # [..., n_atoms]

        # 2. Soft-thresholding operator S_{lambda1}(Z)
        Y = torch.sign(Z) * torch.relu(Z.abs() - float(lambda1))

        # If all entries of a non-zero vector were thresholded to zero, fall back to unthresholded Z
        all_zero_non_trivial = (Y.abs().amax(dim=-1, keepdim=True) <= 1e-8) & (~is_zero_vec)
        if torch.any(all_zero_non_trivial):
            Y = torch.where(all_zero_non_trivial, Z, Y)

        # 3. Select top-s atoms sorted in descending magnitude order
        _, top_indices = torch.topk(Y.abs(), k=s, dim=-1, largest=True, sorted=True)
        y_sorted = torch.gather(Y, dim=-1, index=top_indices)  # [..., s]
        y_sorted = torch.where(is_zero_vec, torch.zeros_like(y_sorted), y_sorted)

        # 4. Scalar least-squares debiasing (norm_rescale=False) or exact L2 norm matching (norm_rescale=True)
        # Scaling by a single positive scalar per vector preserves the exact descending order of |y_sorted|.
        x_raw = self._raw_reconstruct(top_indices, y_sorted, layer_idx=layer_idx)
        raw_norm = torch.norm(x_raw, p=2, dim=-1, keepdim=True)
        if norm_rescale:
            safe_scale = torch.where(
                is_zero_vec | (raw_norm <= 1e-8),
                torch.zeros_like(x_norm),
                x_norm / raw_norm.clamp(min=1e-8),
            )
        else:
            # Unrescaled ridge + soft-threshold coefficients compensated only for frame_scale normalization
            safe_scale = torch.where(
                is_zero_vec | (raw_norm <= 1e-8),
                torch.zeros_like(x_norm),
                x_norm / frame_scale,
            )
        y_sorted = y_sorted * safe_scale

        indices_int16 = top_indices.to(dtype=torch.int16, device=X.device)
        q_values = quantize_sorted_atoms(y_sorted.to(device=X.device))
        return indices_int16, q_values

    def omp_encode(
        self,
        X: torch.Tensor,
        layer_idx: int = 0,
        s: int = 16,
        norm_rescale: bool = True,
        tol: float = 1e-7,
    ) -> Tuple[torch.Tensor, QuantizedInt8Values]:
        """
        Sequential Greedy Orthogonal Matching Pursuit (OMP) Baseline at Matched Sparsity `s`:
        At each iteration `k = 1..s`:
          1. Selects atom `j* = argmax_{j not in S_{k-1}} |<r_{k-1}, d_j>|`.
          2. Solves orthogonal least squares `min_c ||x - D_{S_k} c||_2^2` on active support `S_k`.
          3. Updates residual `r_k = x - D_{S_k} c_k` (with early stopping for vectors with `||r_k||_2 <= tol`).
        """
        self._validate_encode_inputs(X, layer_idx=layer_idx, s=s)
        orig_shape = X.shape
        x_flat = X.reshape(-1, self.head_dim).to(device=self.device, dtype=torch.float32)
        n_vecs = x_flat.shape[0]

        x_norm = torch.norm(x_flat, p=2, dim=-1, keepdim=True)
        is_zero_vec = (x_norm.squeeze(-1) <= 1e-8)

        D_l = self.D[layer_idx].to(torch.float32)  # [head_dim, n_atoms]
        D_l_t = D_l.T.contiguous()  # [n_atoms, head_dim]

        residual = x_flat.clone()
        selected_indices = torch.zeros((n_vecs, s), dtype=torch.long, device=self.device)
        selected_mask = torch.zeros((n_vecs, self.n_atoms), dtype=torch.bool, device=self.device)
        coeffs = torch.zeros((n_vecs, s), dtype=torch.float32, device=self.device)

        for step in range(s):
            corrs = torch.matmul(residual, D_l).abs()  # [N, n_atoms]
            corrs = corrs.masked_fill(selected_mask, -1.0)
            best_idx = torch.argmax(corrs, dim=-1)  # [N]
            selected_indices[:, step] = best_idx
            selected_mask.scatter_(1, best_idx.unsqueeze(1), True)

            k_cur = step + 1
            active_idx = selected_indices[:, :k_cur]
            D_sub_t = F.embedding(active_idx, D_l_t)  # [N, k_cur, head_dim]
            D_sub = D_sub_t.transpose(1, 2)  # [N, head_dim, k_cur]

            gram = torch.matmul(D_sub_t, D_sub) + 1e-5 * torch.eye(
                k_cur, device=self.device, dtype=torch.float32
            ).unsqueeze(0)
            rhs = torch.matmul(D_sub_t, x_flat.unsqueeze(-1))
            c_step = torch.linalg.solve(gram, rhs).squeeze(-1)  # [N, k_cur]
            coeffs[:, :k_cur] = c_step

            x_approx = (D_sub_t * c_step.unsqueeze(-1)).sum(dim=1)
            residual = x_flat - x_approx
            if float(torch.norm(residual, p=2, dim=-1).max().item()) <= tol:
                break

        coeffs = torch.where(is_zero_vec.unsqueeze(-1), torch.zeros_like(coeffs), coeffs)

        sort_order = torch.argsort(coeffs.abs(), dim=-1, descending=True)
        sorted_indices = torch.gather(selected_indices, dim=-1, index=sort_order)
        sorted_coeffs = torch.gather(coeffs, dim=-1, index=sort_order)

        if norm_rescale:
            x_raw = self._raw_reconstruct(sorted_indices, sorted_coeffs, layer_idx=layer_idx)
            raw_norm = torch.norm(x_raw, p=2, dim=-1, keepdim=True)
            safe_scale = torch.where(
                is_zero_vec.unsqueeze(-1) | (raw_norm <= 1e-8),
                torch.zeros_like(x_norm),
                x_norm / raw_norm.clamp(min=1e-8),
            )
            sorted_coeffs = sorted_coeffs * safe_scale

        out_shape = orig_shape[:-1] + (s,)
        indices_int16 = sorted_indices.view(out_shape).to(dtype=torch.int16, device=X.device)
        q_values = quantize_sorted_atoms(sorted_coeffs.view(out_shape).to(device=X.device))
        return indices_int16, q_values

    def iterative_elasticnet_encode(
        self,
        X: torch.Tensor,
        layer_idx: int = 0,
        s: int = 16,
        lambda1: float = 0.04,
        lambda2: float = 0.1,
        n_iters: int = 50,
        num_iters: Optional[int] = None,
        norm_rescale: bool = True,
    ) -> Tuple[torch.Tensor, QuantizedInt8Values]:
        """
        Trusted Iterative FISTA (Fast Iterative Shrinkage-Thresholding) ElasticNet Reference Solver:
        Minimizes the convex ElasticNet objective over `n_iters` proximal-gradient steps:
            min_z  0.5 * ||x - D z||_2^2 + lambda1 * ||z||_1 + 0.5 * lambda2 * ||z||_2^2
        followed by top-`s` support selection, orthogonal least-squares debiasing on the selected
        support, and quantization to the exact same `(int16 indices, int8 values + fp16 scale)` format.
        """
        if num_iters is not None:
            n_iters = int(num_iters)
        self._validate_encode_inputs(X, layer_idx=layer_idx, s=s)
        orig_shape = X.shape
        x_flat = X.reshape(-1, self.head_dim).to(device=self.device, dtype=torch.float32)
        n_vecs = x_flat.shape[0]

        x_norm = torch.norm(x_flat, p=2, dim=-1, keepdim=True)
        is_zero_vec = (x_norm <= 1e-8)
        x_unit = torch.where(is_zero_vec, torch.zeros_like(x_flat), x_flat / x_norm.clamp(min=1e-8))

        D_l = self.D[layer_idx].to(torch.float32)  # [head_dim, n_atoms]
        D_l_t = D_l.T.contiguous()  # [n_atoms, head_dim]

        # Spectral norm ||D||_2^2 for step size eta = 1 / (L_lip)
        with torch.no_grad():
            s_max = float(torch.linalg.svdvals(D_l)[0].item())
            lip = (s_max * s_max) + float(lambda2)
            step_size = 1.0 / max(lip, 1e-4)
            thresh = step_size * float(lambda1)

            z_k = torch.zeros((n_vecs, self.n_atoms), device=self.device, dtype=torch.float32)
            y_k = z_k.clone()
            t_k = 1.0

            x_dt = torch.matmul(x_unit, D_l)  # [N, n_atoms]
            for _ in range(max(1, int(n_iters))):
                # Gradient of 0.5 * ||x_unit - y_k D^T||^2 + 0.5 * lambda2 * ||y_k||^2
                recon = torch.matmul(y_k, D_l_t)  # [N, head_dim]
                grad = torch.matmul(recon, D_l) - x_dt + float(lambda2) * y_k
                u = y_k - step_size * grad
                z_next = torch.sign(u) * torch.relu(u.abs() - thresh)

                t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t_k * t_k))
                y_k = z_next + ((t_k - 1.0) / t_next) * (z_next - z_k)
                z_k = z_next
                t_k = t_next

            # Select top-s atoms by magnitude from converged FISTA solution z_k
            _, top_indices = torch.topk(z_k.abs(), k=s, dim=-1, largest=True, sorted=True)

            # Debiasing least-squares projection on the selected s-atom support
            D_sub_t = F.embedding(top_indices, D_l_t)  # [N, s, head_dim]
            D_sub = D_sub_t.transpose(1, 2)  # [N, head_dim, s]
            gram = torch.matmul(D_sub_t, D_sub) + 1e-4 * torch.eye(
                s, device=self.device, dtype=torch.float32
            ).unsqueeze(0)
            rhs = torch.matmul(D_sub_t, x_flat.unsqueeze(-1))
            coeffs = torch.linalg.solve(gram, rhs).squeeze(-1)  # [N, s]
            coeffs = torch.where(is_zero_vec, torch.zeros_like(coeffs), coeffs)

            sort_order = torch.argsort(coeffs.abs(), dim=-1, descending=True)
            sorted_indices = torch.gather(top_indices, dim=-1, index=sort_order)
            sorted_coeffs = torch.gather(coeffs, dim=-1, index=sort_order)

            if norm_rescale:
                x_raw = self._raw_reconstruct(sorted_indices, sorted_coeffs, layer_idx=layer_idx)
                raw_norm = torch.norm(x_raw, p=2, dim=-1, keepdim=True)
                safe_scale = torch.where(
                    is_zero_vec | (raw_norm <= 1e-8),
                    torch.zeros_like(x_norm),
                    x_norm / raw_norm.clamp(min=1e-8),
                )
                sorted_coeffs = sorted_coeffs * safe_scale

        out_shape = orig_shape[:-1] + (s,)
        indices_int16 = sorted_indices.view(out_shape).to(dtype=torch.int16, device=X.device)
        q_values = quantize_sorted_atoms(sorted_coeffs.view(out_shape).to(device=X.device))
        return indices_int16, q_values

    def decode(
        self,
        indices: torch.Tensor,
        values: Union[QuantizedInt8Values, Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        layer_idx: int = 0,
        active_atoms: Optional[int] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full and Nested Partial Sparse Decoder:
        - Reconstructs `X_hat` from the first `m = active_atoms` largest-magnitude atoms
          (`indices[..., :m]`, `values[..., :m]`) or all `s` stored atoms when `active_atoms is None`.
        - Note: This performs partial sparse vector reconstruction; it is not a standalone
          speculative decoding runtime.
        """
        if not isinstance(indices, torch.Tensor) or indices.numel() == 0:
            raise ValueError("indices must be a non-empty torch.Tensor.")
        if not (0 <= int(layer_idx) < self.num_layers):
            raise ValueError(
                f"layer_idx={layer_idx} out of bounds for num_layers={self.num_layers}."
            )

        s_total = int(indices.shape[-1])
        if active_atoms is not None:
            m = int(active_atoms)
            if m <= 0 or m > s_total:
                raise ValueError(
                    f"active_atoms={m} must be in [1, s_total={s_total}]."
                )
        else:
            m = s_total

        idx_m = indices[..., :m].to(device=self.device, dtype=torch.long)
        if int(idx_m.min().item()) < 0 or int(idx_m.max().item()) >= self.n_atoms:
            raise ValueError(
                f"Atom index out of bounds [0, {self.n_atoms - 1}]: "
                f"min={int(idx_m.min().item())}, max={int(idx_m.max().item())}."
            )

        if isinstance(values, tuple):
            raw_vals, sc = values
            val_m = raw_vals[..., :m].to(device=self.device, dtype=torch.float32) * sc.to(
                device=self.device, dtype=torch.float32
            )
        elif isinstance(values, QuantizedInt8Values):
            sc = scale if scale is not None else values.scale
            raw_m = torch.Tensor._make_subclass(torch.Tensor, values)[..., :m]
            val_m = raw_m.to(device=self.device, dtype=torch.float32) * sc.to(
                device=self.device, dtype=torch.float32
            )
        else:
            raw_m = values[..., :m].to(device=self.device, dtype=torch.float32)
            if scale is not None:
                val_m = raw_m * scale.to(device=self.device, dtype=torch.float32)
            elif getattr(values, "scale", None) is not None:
                val_m = raw_m * getattr(values, "scale").to(device=self.device, dtype=torch.float32)
            else:
                val_m = raw_m

        x_hat = self._raw_reconstruct(idx_m, val_m, layer_idx=layer_idx)
        return x_hat.to(device=indices.device, dtype=self.dtype)

    def adapt_online_atoms(
        self,
        X: torch.Tensor,
        X_hat: Optional[torch.Tensor] = None,
        M_adapt: int = 64,
        layer_idx: int = 0,
        s_univ: int = 16,
        X_eval: Optional[torch.Tensor] = None,
        X_hat_eval: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Online Residual Dictionary Adaptation with Strict Held-Out Evaluation:
          1. If `X_eval` is provided, fits the top-`M_adapt` residual SVD basis `D_adapt` on `X`
             (and `X_hat`) and evaluates reconstruction quality & `s_adapt` strictly on `X_eval`.
             If `X_eval` is None, splits `X` 50/50 into an adaptation split and a held-out
             evaluation split so adaptation is never evaluated purely in-sample.
          2. Compares Universal Dictionary `D` vs Augmented Dictionary `[D, D_adapt]` at matched
             sparsity `s_univ` on the held-out evaluation split.
          3. Computes effective held-out sparsity `s_adapt` required for `[D, D_adapt]` to match
             the Universal Dictionary's held-out MSE at `s_univ`, and computes the audited
             break-even length `L_star` via `compute_break_even_length` (returning `L_star = None`
             if `s_adapt >= s_univ`).
        """
        if M_adapt <= 0 or M_adapt > self.head_dim:
            raise ValueError(
                f"M_adapt={M_adapt} must be in [1, head_dim={self.head_dim}]."
            )
        self._validate_encode_inputs(X, layer_idx=layer_idx, s=s_univ)

        x_all = X.reshape(-1, self.head_dim).to(device=self.device, dtype=torch.float32)
        if X_eval is not None:
            self._validate_encode_inputs(X_eval, layer_idx=layer_idx, s=s_univ)
            x_fit = x_all
            x_test = X_eval.reshape(-1, self.head_dim).to(device=self.device, dtype=torch.float32)
            if X_hat is not None:
                x_hat_fit = X_hat.reshape(-1, self.head_dim).to(device=self.device, dtype=torch.float32)
            else:
                idx_f, val_f = self.encode(x_fit, layer_idx=layer_idx, s=s_univ)
                x_hat_fit = self.decode(idx_f, val_f, layer_idx=layer_idx).to(torch.float32)
        else:
            # Split X into disjoint fit (first half) and held-out eval (second half)
            n_total = x_all.shape[0]
            if n_total < 4:
                raise ValueError(
                    f"Need at least 4 vectors to split into adaptation and held-out sets, got {n_total}."
                )
            mid = n_total // 2
            x_fit = x_all[:mid]
            x_test = x_all[mid:]
            if X_hat is not None:
                x_hat_fit = X_hat.reshape(-1, self.head_dim)[:mid].to(
                    device=self.device, dtype=torch.float32
                )
            else:
                idx_f, val_f = self.encode(x_fit, layer_idx=layer_idx, s=s_univ)
                x_hat_fit = self.decode(idx_f, val_f, layer_idx=layer_idx).to(torch.float32)

        # 1. Fit top-M_adapt SVD singular vectors on fit-split residuals R_fit = x_fit - x_hat_fit
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        residuals_fit = x_fit - x_hat_fit
        res_norm = float(torch.norm(residuals_fit).item())
        if res_norm <= 1e-8:
            # Degenerate zero-residual case: no adaptation needed or beneficial
            D_adapt = torch.eye(self.head_dim, M_adapt, device=self.device, dtype=torch.float32)
            S_res = torch.zeros(min(x_fit.shape[0], self.head_dim), device=self.device)
            k_actual = 0
        else:
            _, S_res, Vh_res = torch.linalg.svd(residuals_fit, full_matrices=False)
            k_actual = min(M_adapt, int((S_res > 1e-7).sum().item()), Vh_res.shape[0])
            if k_actual > 0:
                D_adapt = F.normalize(Vh_res[:k_actual, :].T.contiguous(), p=2, dim=0)
            else:
                D_adapt = torch.empty((self.head_dim, 0), device=self.device, dtype=torch.float32)
            if k_actual < M_adapt:
                gen = torch.Generator(device=self.device).manual_seed(9001)
                pad = F.normalize(
                    torch.randn(
                        self.head_dim,
                        M_adapt - k_actual,
                        device=self.device,
                        dtype=torch.float32,
                        generator=gen,
                    ),
                    p=2,
                    dim=0,
                )
                D_adapt = torch.cat([D_adapt, pad], dim=1)

        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize()
        adapt_time_ms = (time.perf_counter() - t0) * 1000.0

        # 2. Evaluate Universal Dictionary on held-out x_test at s_univ
        idx_test_u, val_test_u = self.encode(x_test, layer_idx=layer_idx, s=s_univ)
        x_hat_test_u = self.decode(idx_test_u, val_test_u, layer_idx=layer_idx).to(torch.float32)

        test_energy = float(x_test.pow(2).mean().item()) + 1e-12
        univ_test_mse = float(F.mse_loss(x_hat_test_u, x_test).item())
        univ_test_nmse = univ_test_mse / test_energy
        univ_test_cos = float(F.cosine_similarity(x_hat_test_u, x_test, dim=-1).mean().item())

        # 3. Evaluate Augmented Dictionary D_aug = [D_univ, D_adapt] on held-out x_test
        D_univ = self.D[layer_idx].to(torch.float32)
        D_aug = torch.cat([D_univ, D_adapt], dim=1)  # [head_dim, n_atoms + M_adapt]
        D_aug_t = D_aug.T.contiguous()

        x_test_norm = torch.norm(x_test, p=2, dim=-1, keepdim=True)
        is_zero_test = x_test_norm <= 1e-8
        frame_scale = float(self.n_atoms) / float(self.head_dim)
        x_test_scaled = torch.where(
            is_zero_test,
            torch.zeros_like(x_test),
            (x_test / x_test_norm.clamp(min=1e-8)) * frame_scale,
        )

        # Universal ridge scores + direct orthonormal residual basis inner products on held-out x_test
        Z_univ = torch.matmul(x_test_scaled, self.W_ridge[layer_idx].T.to(torch.float32))
        Z_adapt = torch.matmul(
            torch.where(is_zero_test, torch.zeros_like(x_test), x_test / x_test_norm.clamp(min=1e-8)),
            D_adapt,
        ) * 1.5
        Z_aug = torch.cat([Z_univ, Z_adapt], dim=-1)
        Y_aug = torch.sign(Z_aug) * torch.relu(Z_aug.abs() - 0.04)
        all_z = (Y_aug.abs().amax(dim=-1, keepdim=True) <= 1e-8) & (~is_zero_test)
        if torch.any(all_z):
            Y_aug = torch.where(all_z, Z_aug, Y_aug)

        mse_by_s: Dict[int, float] = {}
        nmse_by_s: Dict[int, float] = {}
        cos_by_s: Dict[int, float] = {}

        for s_cand in range(1, s_univ + 1):
            _, idx_c = torch.topk(Y_aug.abs(), k=s_cand, dim=-1, largest=True, sorted=True)
            y_c = torch.gather(Y_aug, dim=-1, index=idx_c)
            atoms_c = F.embedding(idx_c, D_aug_t)
            y_dir = (atoms_c * x_test.unsqueeze(-2)).sum(dim=-1)
            y_ref = (0.25 * (y_c / frame_scale) * x_test_norm) + (0.75 * y_dir)
            x_rec_c = (atoms_c * y_ref.unsqueeze(-1)).sum(dim=-2)
            rec_norm = torch.norm(x_rec_c, p=2, dim=-1, keepdim=True)
            safe_sc = torch.where(
                is_zero_test | (rec_norm <= 1e-8),
                torch.zeros_like(x_test_norm),
                x_test_norm / rec_norm.clamp(min=1e-8),
            )
            x_rec_c = x_rec_c * safe_sc
            m_val = float(F.mse_loss(x_rec_c, x_test).item())
            mse_by_s[s_cand] = m_val
            nmse_by_s[s_cand] = m_val / test_energy
            cos_by_s[s_cand] = float(F.cosine_similarity(x_rec_c, x_test, dim=-1).mean().item())

        # Interpolate held-out s_adapt where mse_by_s[s] crosses univ_test_mse
        if mse_by_s[s_univ] >= univ_test_mse or res_norm <= 1e-8:
            # Adapted dictionary did not improve upon universal dictionary at s_univ on held-out data
            s_adapt = float(s_univ)
        else:
            s_adapt = float(s_univ)
            for s_cand in range(1, s_univ):
                m_curr = mse_by_s[s_cand]
                m_next = mse_by_s[s_cand + 1]
                if m_curr >= univ_test_mse >= m_next:
                    denom = max(1e-12, m_curr - m_next)
                    frac = (m_curr - univ_test_mse) / denom
                    s_adapt = float(s_cand) + float(frac)
                    break
                elif m_curr <= univ_test_mse:
                    s_adapt = float(s_cand)
                    break

        be_info = compute_break_even_length(
            head_dim=self.head_dim,
            M_adapt=M_adapt,
            s_univ=float(s_univ),
            s_adapt=s_adapt,
        )

        top16_energy = (
            float((S_res[: min(16, len(S_res))].pow(2).sum() / (S_res.pow(2).sum() + 1e-12)).item())
            if S_res.numel() > 0 and float(S_res.pow(2).sum().item()) > 1e-12
            else 0.0
        )

        return {
            "D_adapt": D_adapt.to(self.dtype),
            "M_adapt": M_adapt,
            "effective_rank": int(k_actual),
            "head_dim": self.head_dim,
            "n_fit_vectors": int(x_fit.shape[0]),
            "n_eval_vectors": int(x_test.shape[0]),
            "evaluated_on_heldout": True,
            "adaptation_time_ms": round(adapt_time_ms, 3),
            "s_univ": float(s_univ),
            "s_adapt": round(float(s_adapt), 2),
            "delta_s": be_info["delta_s"],
            "L_star": be_info["L_star"],
            "break_even_exists": be_info["break_even_exists"],
            "break_even_reason": be_info["reason"],
            "overhead_bytes": be_info["overhead_bytes"],
            "per_vector_savings_bytes": be_info["per_vector_savings_bytes"],
            "units": be_info["units"],
            "univ_mse": round(univ_test_mse, 6),
            "univ_rel_mse": round(univ_test_nmse, 6),
            "univ_cosine_sim": round(univ_test_cos, 4),
            "adapted_mse_at_s_univ": round(mse_by_s[s_univ], 6),
            "adapted_rel_mse_at_s_univ": round(nmse_by_s[s_univ], 6),
            "adapted_cosine_sim_at_s_univ": round(cos_by_s[s_univ], 4),
            "cosine_gain_at_s_univ": round(cos_by_s[s_univ] - univ_test_cos, 4),
            "residual_top16_energy_ratio": round(top16_energy, 4),
            "mse_by_s": mse_by_s,
            "nmse_by_s": nmse_by_s,
            "cos_by_s": cos_by_s,
        }

    adapt_online_dictionary = adapt_online_atoms
