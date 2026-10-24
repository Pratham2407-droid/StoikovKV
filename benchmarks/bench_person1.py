"""
End-to-End Benchmark & Publication Plot Generator for Person 1 (`benchmarks/bench_person1.py`).

Executes reproducible, held-out measurements for StoikovKV calibration and sparse KV compression:
1. Offline Model Calibration (`calibrate_model` -> `calibrated_artifacts.pt`):
   - 12 calibration prompts + 4 Needle-in-a-Haystack prompts ("The secret passkey is 94812")
   - Disjoint held-out evaluation prompts for Write-Time Subspace Probe validation
   - Reports raw probe (`sink_tokens=0`), sink-augmented probe (`sink_tokens=4, +100`),
     and `||V_t||_2`-only baseline Spearman rank correlation and Recall@20%
   - Per-head entropy, distractor-checked retrieval score, and deterministic head roles
   - Seeded SVD+Adam universal dictionary (`D`) and FP64 linear solve ridge weights (`W_ridge`)
2. Fair Sparse Codec Evaluation (`ElasticLexicoCodec`):
   - Compares Fast 1-Step Approximate Preconditioned Encoder (`norm_rescale=True` and `False`),
     Sequential Greedy OMP (`omp_encode`), and Iterative FISTA ElasticNet Reference
     (`iterative_elasticnet_encode`) on identical held-out KV vectors across s in {4, 8, 12, 16}.
   - Separates warm-up from timed iterations, synchronizes CUDA, and reports median, mean, IQR,
     and std for both encoding and decoding latencies (plus CPU baseline timing).
   - Evaluates Nested Partial Decoding (`decode(..., active_atoms=m)`) for m in {2, 4, 8, 12, 16}.
3. Held-Out Online Residual Dictionary Adaptation (`adapt_online_atoms(..., M_adapt=64)`):
   - Evaluates English, Python Code, and Kannada ("ಕನ್ನಡ ಭಾಷೆ ಮತ್ತು ತಂತ್ರಜ್ಞಾನ...") across
     4 adaptation prompts and 4 disjoint held-out evaluation prompts per domain at matched
     token counts without synthetic OOD perturbation injection.
   - Reports empirical `s_adapt`, `delta_s`, and `L_star` (explicitly marking `None` when
     `delta_s <= 0`).
4. Saves 3 PNG figures (`probe_correlation.png`, `elasticnet_vs_omp.png`, `kannada_code_lstar.png`)
   and structured JSON metrics (`person1_metrics.json`).
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Tuple

# Ensure MPLCONFIGDIR and TRITON_CACHE_DIR point to writable directories (/tmp)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_stoikov")
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_stoikov")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure repo root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from stoikov_kv.calibration import (
    CALIBRATION_PROMPTS,
    HELDOUT_EVAL_PROMPTS,
    NEEDLE_PROMPTS,
    ROLE_DIFFUSE_LEXICO,
    ROLE_PEAKED_H2O,
    ROLE_RETRIEVAL_FP16,
    _apply_rotary_pos_emb,
    _compute_rope_cos_sin,
    _inspect_model_architecture,
    calibrate_model,
    evaluate_compressed_cache_passkey,
    load_model_and_tokenizer,
)
from stoikov_kv.elastic_lexico import ElasticLexicoCodec


# Multiple adaptation and disjoint held-out evaluation prompts per domain
KANNADA_ADAPT_PROMPTS: List[str] = [
    (
        "ಕನ್ನಡ ಭಾಷೆ ಮತ್ತು ತಂತ್ರಜ್ಞಾನ ಇಂದಿನ ಕೃತಕ ಬುದ್ಧಿಮತ್ತೆ ಯುಗದಲ್ಲಿ ವೇಗವಾಗಿ ಬೆಳೆಯುತ್ತಿದೆ. "
        "ದೊಡ್ಡ ಭಾಷಾ ಮಾದರಿಗಳು ಕನ್ನಡದ ವಾಕ್ಯರಚನೆ, ಸಂಧಿ, ಸಮಾಸ ಮತ್ತು ಸಾಹಿತ್ಯಿಕ ಪದಗಳನ್ನು "
        "ಸಂಸ್ಕರಿಸುವಾಗ ಹೆಚ್ಚಿನ ಟೋಕನ್‌ಗಳನ್ನು ಬಳಸುತ್ತವೆ."
    ),
    (
        "ಇದರಿಂದಾಗಿ ಜಿಪಿಯು ಮೆಮೊರಿ ಮತ್ತು ಕೆವಿ ಕ್ಯಾಶೆ ನಿರ್ವಹಣೆ ಅತ್ಯಂತ ಮಹತ್ವದ್ದಾಗಿದೆ. "
        "ಕನ್ನಡ ಭಾಷೆ ಮತ್ತು ತಂತ್ರಜ್ಞಾನ ಸಮನ್ವಯದಿಂದ ಗ್ರಾಮೀಣ ಮತ್ತು ನಗರ ಪ್ರದೇಶಗಳ ಬಳಕೆದಾರರಿಗೆ "
        "ವೇಗವಾದ ಹಾಗೂ ಕಡಿಮೆ ವೆಚ್ಚದ ಡಿಜಿಟಲ್ ಸೇವೆಗಳನ್ನು ಒದಗಿಸಲು ಸಾಧ್ಯವಾಗುತ್ತದೆ."
    ),
    (
        "ಕರ್ನಾಟಕದ ಸಾಹಿತ್ಯ ಪರಂಪರೆ ಪಂಪ, ರನ್ನ, ಜನ್ನರಿಂದ ಹಿಡಿದು ಕುವೆಂಪು ಮತ್ತು ಬೇಂದ್ರೆಯವರವರೆಗೆ "
        "ಸಮೃದ್ಧವಾಗಿದೆ. ಗಣಕಯಂತ್ರಗಳಲ್ಲಿ ಯುನಿಕೋಡ್ ಶಿಷ್ಟಾಚಾರದ ಮೂಲಕ ಕನ್ನಡ ಅಕ್ಷರಗಳ ಸಂಸ್ಕರಣೆ ಸುಲಭವಾಗಿದೆ."
    ),
    (
        "ನೈಸರ್ಗಿಕ ಭಾಷಾ ಸಂಸ್ಕರಣೆ ಮತ್ತು ಯಂತ್ರ ಅನುವಾದ ವ್ಯವಸ್ಥೆಗಳು ಭಾರತೀಯ ಭಾಷೆಗಳ ನಡುವೆ "
        "ಜ್ಞಾನ ವಿನಿಮಯವನ್ನು ವೇಗಗೊಳಿಸುತ್ತಿವೆ ಹಾಗೂ ಸ್ಥಳೀಯ ಭಾಷೆಯಲ್ಲಿ ಶಿಕ್ಷಣವನ್ನು ಬೆಂಬಲಿಸುತ್ತಿವೆ."
    ),
]

KANNADA_EVAL_PROMPTS: List[str] = [
    (
        "ಕನ್ನಡ ಭಾಷೆ ಮತ್ತು ತಂತ್ರಜ್ಞಾನ ಕ್ಷೇತ್ರದಲ್ಲಿ ಹೊಸ ಸಂಶೋಧನೆಗಳು ನಡೆಯುತ್ತಿವೆ. "
        "ಧ್ವನಿ ಗುರುತಿಸುವಿಕೆ ಮತ್ತು ಪಠ್ಯದಿಂದ ಧ್ವನಿ ಪರಿವರ್ತನೆ ತಂತ್ರಜ್ಞಾನಗಳು ಜನಸಾಮಾನ್ಯರಿಗೆ ಉಪಯುಕ್ತವಾಗಿವೆ."
    ),
    (
        "ಸರ್ಕಾರಿ ದಾಖಲೆಗಳು, ನ್ಯಾಯಾಲಯದ ತೀರ್ಪುಗಳು ಮತ್ತು ವೈದ್ಯಕೀಯ ಮಾಹಿತಿಯನ್ನು ಕನ್ನಡದಲ್ಲಿ "
        "ಸ್ವಯಂಚಾಲಿತವಾಗಿ ಸಾರಾಂಶಗೊಳಿಸಲು ಆಧುನಿಕ ಭಾಷಾ ಮಾದರಿಗಳನ್ನು ಬಳಸಲಾಗುತ್ತಿದೆ."
    ),
    (
        "ಕಡಿಮೆ ಮೆಮೊರಿ ಬಳಸುವ ಸಂಕೋಚನ ಕ್ರಮಾವಳಿಗಳು ಮೊಬೈಲ್ ಸಾಧನಗಳಲ್ಲಿಯೂ ಕನ್ನಡದ ದೊಡ್ಡ ಭಾಷಾ "
        "ಮಾದರಿಗಳನ್ನು ವೇಗವಾಗಿ ಚಲಾಯಿಸಲು ಸಹಾಯ ಮಾಡುತ್ತವೆ."
    ),
    (
        "ಬಹುಭಾಷಾ ಟೋಕನೈಸರ್‌ಗಳಲ್ಲಿ ಕನ್ನಡದ ಒತ್ತಕ್ಷರಗಳು ಮತ್ತು ಗುಣಿತಾಕ್ಷರಗಳನ್ನು ನಿಖರವಾಗಿ "
        "ಪ್ರತಿನಿಧಿಸುವುದು ಮಾದರಿಯ ದಕ್ಷತೆಯನ್ನು ಹೆಚ್ಚಿಸಲು ಪ್ರಮುಖವಾಗಿದೆ."
    ),
]

PYTHON_ADAPT_PROMPTS: List[str] = [
    (
        "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n"
        "class StoikovKVCacheEngine(nn.Module):\n"
        "    def __init__(self, u_q: torch.Tensor, dictionary: torch.Tensor):\n"
        "        super().__init__()\n"
        "        self.register_buffer('u_q', u_q)\n"
        "        self.register_buffer('dictionary', F.normalize(dictionary, p=2, dim=-2))\n"
    ),
    (
        "def compute_sparse_projection(x: torch.Tensor, w_ridge: torch.Tensor, s: int = 16):\n"
        "    z = x @ w_ridge.transpose(-1, -2)\n"
        "    y = torch.sign(z) * torch.relu(z.abs() - 0.04)\n"
        "    vals, idx = torch.topk(y.abs(), k=s, dim=-1, largest=True, sorted=True)\n"
        "    return idx.to(torch.int16), torch.gather(y, -1, idx)\n"
    ),
    (
        "class CircularRingBuffer:\n"
        "    def __init__(self, capacity: int, dim: int, device: str = 'cuda'):\n"
        "        self.capacity = capacity\n"
        "        self.buffer = torch.zeros((capacity, dim), dtype=torch.float16, device=device)\n"
        "        self.write_ptr = 0\n"
        "        self.size = 0\n"
    ),
    (
        "def batched_cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:\n"
        "    a_norm = a.float() / torch.norm(a.float(), p=2, dim=-1, keepdim=True).clamp(min=eps)\n"
        "    b_norm = b.float() / torch.norm(b.float(), p=2, dim=-1, keepdim=True).clamp(min=eps)\n"
        "    return (a_norm * b_norm).sum(dim=-1)\n"
    ),
]

PYTHON_EVAL_PROMPTS: List[str] = [
    (
        "import math\nfrom typing import Optional, Tuple\nimport torch\n\n"
        "def apply_rotary_embedding(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:\n"
        "    d_half = x.shape[-1] // 2\n"
        "    x1, x2 = x[..., :d_half], x[..., d_half:]\n"
        "    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)\n"
    ),
    (
        "class SubspaceProbeScorer:\n"
        "    def __init__(self, sink_tokens: int = 4, sink_bonus: float = 100.0):\n"
        "        self.sink_tokens = int(sink_tokens)\n"
        "        self.sink_bonus = float(sink_bonus)\n"
        "    def __call__(self, k: torch.Tensor, v: torch.Tensor, u_q: torch.Tensor) -> torch.Tensor:\n"
        "        return torch.norm(k @ u_q, p=2, dim=-1) * torch.norm(v, p=2, dim=-1)\n"
    ),
    (
        "def quantize_symmetric_int8(values: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:\n"
        "    max_abs = values.abs().amax(dim=-1, keepdim=True)\n"
        "    scale = torch.where(max_abs > eps, max_abs / 127.0, torch.ones_like(max_abs))\n"
        "    q = torch.clamp(torch.round(values / scale), -127, 127).to(torch.int8)\n"
        "    return q, scale.to(torch.float16)\n"
    ),
    (
        "def fit_residual_svd_basis(residuals: torch.Tensor, rank: int = 64) -> torch.Tensor:\n"
        "    _, _, vh = torch.linalg.svd(residuals.to(torch.float32), full_matrices=False)\n"
        "    basis = vh[:rank, :].transpose(0, 1).contiguous()\n"
        "    return torch.nn.functional.normalize(basis, p=2, dim=0)\n"
    ),
]

ENGLISH_ADAPT_PROMPTS: List[str] = [
    (
        "Autoregressive transformer inference stores historical key and value projections for every "
        "preceding token so that causal self-attention avoids recomputing activations from scratch."
    ),
    (
        "Grouped-query attention reduces key-value memory bandwidth by sharing a single key and value "
        "head across multiple query heads within each decoder block."
    ),
    (
        "Preconditioned ridge regression followed by soft-thresholding maps dense vectors onto sparse "
        "dictionary coordinates in a single matrix-vector multiplication."
    ),
    (
        "Orthogonal matching pursuit selects dictionary atoms sequentially by projecting the residual "
        "onto the selected support at every greedy iteration."
    ),
]
ENGLISH_EVAL_PROMPTS: List[str] = [
    (
        "Memory bandwidth on modern graphics processors often limits autoregressive token generation "
        "far more than arithmetic throughput because every decoding step reads the entire key-value cache."
    ),
    (
        "Subspace projections derived from calibration query vectors identify which key dimensions "
        "participate strongly in dot-product attention without materializing quadratic score matrices."
    ),
    (
        "Sparse dictionary representations approximate high-dimensional activation vectors as linear "
        "combinations of unit-norm atoms, trading small reconstruction error for substantial byte savings."
    ),
    (
        "Evaluating adaptive compression requires strict separation between fitting prompts and held-out "
        "test prompts so that rank reduction is measured without optimistic overfitting."
    ),
]


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _measure_latency_stats(
    fn: Callable[[], Any],
    device: torch.device,
    warmup: int = 5,
    repetitions: int = 15,
) -> Dict[str, float]:
    """
    Runs `warmup` untimed iterations followed by `repetitions` timed iterations with CUDA
    synchronization before and after each call. Reports median, mean, IQR, min, max, and std (ms).
    """
    for _ in range(warmup):
        _ = fn()
    _sync_device(device)

    samples_ms: List[float] = []
    for _ in range(repetitions):
        _sync_device(device)
        t0 = time.perf_counter()
        _ = fn()
        _sync_device(device)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(samples_ms, dtype=np.float64)
    q25, q50, q75 = np.percentile(arr, [25.0, 50.0, 75.0])
    return {
        "median_ms": round(float(q50), 4),
        "mean_ms": round(float(arr.mean()), 4),
        "iqr_ms": round(float(q75 - q25), 4),
        "std_ms": round(float(arr.std(ddof=1 if len(arr) > 1 else 0)), 4),
        "min_ms": round(float(arr.min()), 4),
        "max_ms": round(float(arr.max()), 4),
        "repetitions": int(repetitions),
        "warmup": int(warmup),
    }


def _collect_domain_kv_states(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: List[str],
    layer_idx: int = 0,
    target_tokens: int = 256,
    max_seq_len: int = 256,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Runs inference-only forward passes on `prompts` and extracts post-RoPE Key vectors
    from `layer_idx` without injecting any synthetic OOD perturbations.
    Truncates or pads across prompts to a matched token count (`target_tokens`) for fair
    cross-domain comparison.
    """
    arch = _inspect_model_architecture(model)
    device = arch["device"]
    dtype = arch["dtype"]
    num_kv_heads = arch["num_kv_heads"]
    head_dim = arch["head_dim"]
    rope_theta = arch["rope_theta"]
    layers_mod = arch["layers"]

    captured_k: List[torch.Tensor] = []
    k_proj = getattr(layers_mod[layer_idx].self_attn, "k_proj")

    def _k_hook(_m: Any, _inp: Any, out: torch.Tensor) -> None:
        bsz, slen, _ = out.shape
        k_raw = out.detach().view(bsz, slen, num_kv_heads, head_dim).transpose(1, 2)
        cos, sin = _compute_rope_cos_sin(
            slen, head_dim, device=k_raw.device, dtype=torch.float32, rope_theta=rope_theta
        )
        k_rope = _apply_rotary_pos_emb(k_raw.to(torch.float32), cos, sin).to(dtype)
        # Flatten across KV heads and tokens: [bsz * num_kv_heads * slen, head_dim]
        captured_k.append(k_rope.reshape(-1, head_dim).detach())

    handle = k_proj.register_forward_hook(_k_hook)
    raw_token_count = 0
    try:
        with torch.no_grad():
            for p_text in prompts:
                enc = tokenizer(
                    p_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_seq_len,
                )
                input_ids = enc["input_ids"].to(device)
                raw_token_count += int(input_ids.shape[-1])
                _ = model(input_ids=input_ids, use_cache=False, output_attentions=False)
    finally:
        handle.remove()

    all_k = torch.cat(captured_k, dim=0)
    total_vectors_collected = int(all_k.shape[0])

    # Select exactly `target_tokens` vectors deterministically (evenly strided across collected vectors)
    if total_vectors_collected >= target_tokens:
        stride_idx = torch.linspace(
            0, total_vectors_collected - 1, steps=target_tokens, device=device
        ).long()
        selected_k = all_k.index_select(0, stride_idx).contiguous()
    else:
        reps = (target_tokens // max(1, total_vectors_collected)) + 1
        selected_k = all_k.repeat(reps, 1)[:target_tokens].contiguous()

    stats = {
        "num_prompts": len(prompts),
        "raw_sequence_tokens": raw_token_count,
        "total_kv_vectors_collected": total_vectors_collected,
        "evaluated_kv_vectors": int(selected_k.shape[0]),
        "head_dim": head_dim,
        "synthetic_perturbation_injected": False,
    }
    return selected_k, stats


def plot_probe_correlation(
    artifacts: Dict[str, Any], save_paths: List[str]
) -> None:
    """
    Generates `probe_correlation.png`:
    - Panel A: Held-out Write-Time Subspace Probe I_t vs Attention-Derived Reference Scatter
      comparing Raw Probe (no sink bonus), Sink-Augmented Probe (+100 sink bonus), and ||V_t||_2 baseline.
    - Panel B: Per-Head Entropy vs Needle Retrieval Score colored by assigned Head Role,
      annotating retrieval confidence status.
    """
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 6.0), dpi=220)
    fig.patch.set_facecolor("#0f172a")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1e293b")

    pm = artifacts["probe_metrics"]
    rho_raw = pm["raw_probe"]["spearman_mean"]
    rec_raw = pm["raw_probe"]["recall_at_20_mean"]
    rho_sink = pm["sink_augmented_probe"]["spearman_mean"]
    rec_sink = pm["sink_augmented_probe"]["recall_at_20_mean"]
    rho_v = pm["value_norm_baseline"]["spearman_mean"]
    rec_v = pm["value_norm_baseline"]["recall_at_20_mean"]
    n_pairs = pm["raw_probe"]["num_layer_prompt_pairs"]

    p_scores_aug = np.array(pm["sample_probe_scores"], dtype=np.float64)
    g_scores = np.array(pm["sample_h2o_scores"], dtype=np.float64)
    n_pts = min(len(p_scores_aug), len(g_scores))
    p_scores_aug = p_scores_aug[:n_pts]
    g_scores = g_scores[:n_pts]

    non_sink_mask = np.arange(n_pts) >= 4
    sink_mask = ~non_sink_mask
    p_scores = np.maximum(0.0, p_scores_aug - np.where(sink_mask, 100.0, 0.0))

    p_vis = np.log1p(np.maximum(0.0, p_scores))
    g_vis = np.log1p(np.maximum(0.0, g_scores))
    p_norm = (p_vis - p_vis.min()) / (p_vis.ptp() + 1e-8)
    g_norm = (g_vis - g_vis.min()) / (g_vis.ptp() + 1e-8)

    k_top = max(1, int(round(0.20 * len(p_scores))))
    top_g_idx = set(np.argsort(g_scores)[-k_top:].tolist())
    is_hh = np.array([i in top_g_idx for i in range(len(p_scores))])

    ax1.scatter(
        g_norm[~is_hh & non_sink_mask],
        p_norm[~is_hh & non_sink_mask],
        c="#64748b",
        alpha=0.75,
        s=46,
        label="Held-Out Context Tokens (Bottom 80%)",
        edgecolors="none",
    )
    ax1.scatter(
        g_norm[is_hh & non_sink_mask],
        p_norm[is_hh & non_sink_mask],
        c="#38bdf8",
        alpha=0.95,
        s=72,
        label="Attention Heavy-Hitters (Top 20% Ref)",
        edgecolors="#e0f2fe",
        linewidth=0.8,
    )
    ax1.scatter(
        g_norm[sink_mask],
        p_norm[sink_mask],
        c="#10b981",
        marker="^",
        s=110,
        label="Initial Tokens (Indices 0..3, Raw Probe Score)",
        edgecolors="#d1fae5",
        linewidth=1.0,
    )

    if len(g_norm) > 2 and g_norm.ptp() > 1e-8 and p_norm.ptp() > 1e-8:
        m_fit, b_fit = np.polyfit(g_norm, p_norm, 1)
        xs = np.linspace(0, 1, 100)
        ax1.plot(xs, m_fit * xs + b_fit, "--", color="#f59e0b", linewidth=2.0, label="Linear Trend Fit")

    ax1.set_title(
        f"Held-Out Write-Time Probe vs. Attention Proxy (n={n_pairs} layer-prompt pairs)",
        fontsize=11.8,
        fontweight="bold",
        color="#f8fafc",
        pad=12,
    )
    ax1.set_xlabel(
        "Normalized Mean Attention Received ($\\frac{1}{T-t}\\sum_{i \\geq t} A_{i,t}$, proxy)",
        fontsize=10.0,
        color="#cbd5e1",
    )
    ax1.set_ylabel(
        "Normalized Raw Probe Score ($\\|K_t U_Q\\|_2 \\|V_t\\|_2$, no sink bonus)",
        fontsize=10.0,
        color="#cbd5e1",
    )
    ax1.grid(True, linestyle=":", alpha=0.25, color="#94a3b8")
    ax1.legend(loc="lower right", frameon=True, facecolor="#0f172a", edgecolor="#334155", fontsize=8.4)

    badge_text = (
        f"Raw Probe (no bonus): $\\rho$={rho_raw:.3f}, Rec@20%={rec_raw * 100:.1f}%\n"
        f"Sink Probe (+100 idx 0..3): $\\rho$={rho_sink:.3f}, Rec@20%={rec_sink * 100:.1f}%\n"
        f"$\\|V_t\\|_2$ Baseline Only: $\\rho$={rho_v:.3f}, Rec@20%={rec_v * 100:.1f}%"
    )
    ax1.text(
        0.03,
        0.95,
        badge_text,
        transform=ax1.transAxes,
        verticalalignment="top",
        fontsize=8.8,
        fontweight="bold",
        color="#38bdf8",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#0f172a", edgecolor="#38bdf8", alpha=0.92),
    )

    # Panel B: Per-Head Entropy vs Needle Retrieval Score colored by assigned Head Role
    entropies = artifacts["head_entropy"].detach().cpu().float().numpy().flatten()
    ret_scores = artifacts["retrieval_scores"].detach().cpu().float().numpy().flatten()
    roles = artifacts["head_roles"].detach().cpu().numpy().flatten()
    ret_diag = artifacts.get("retrieval_diagnostics", {})
    ret_sufficient = ret_diag.get("sufficient_evidence", False)

    role_meta = [
        (ROLE_RETRIEVAL_FP16, "0: Retrieval Role (Top 5% -> FP16)", "#10b981", "*", 140),
        (ROLE_PEAKED_H2O, "1: Lower-Entropy Role (H < median)", "#38bdf8", "o", 65),
        (ROLE_DIFFUSE_LEXICO, "2: Higher-Entropy Role (H >= median)", "#f59e0b", "s", 65),
    ]
    for r_id, r_label, r_col, r_mark, r_sz in role_meta:
        mask = roles == r_id
        if np.any(mask):
            ax2.scatter(
                entropies[mask],
                ret_scores[mask],
                c=r_col,
                marker=r_mark,
                s=r_sz,
                alpha=0.9,
                label=f"{r_label} (n={int(mask.sum())})",
                edgecolors="#f8fafc",
                linewidth=0.6,
            )

    non_ret = entropies[roles != ROLE_RETRIEVAL_FP16]
    med_h = float(np.median(non_ret)) if len(non_ret) > 0 else float(np.median(entropies))
    ax2.axvline(med_h, color="#94a3b8", linestyle="--", linewidth=1.5, label=f"Entropy Median ({med_h:.2f} nats)")
    ax2.set_title(
        "GQA KV-Head Router: Entropy $H_h$ vs. Distractor-Adjusted Needle Score",
        fontsize=11.8,
        fontweight="bold",
        color="#f8fafc",
        pad=12,
    )
    ax2.set_xlabel("Average Causal Attention Entropy $H_h$ (nats)", fontsize=10.0, color="#cbd5e1")
    ax2.set_ylabel("Net Passkey Attention Advantage over Distractor", fontsize=10.0, color="#cbd5e1")
    ax2.grid(True, linestyle=":", alpha=0.25, color="#94a3b8")
    ax2.legend(loc="upper right", frameon=True, facecolor="#0f172a", edgecolor="#334155", fontsize=8.4)

    conf_str = (
        "Retrieval Evidence: SUFFICIENT (>=2x uniform & > distractor)"
        if ret_sufficient
        else "Retrieval Evidence: LOW (fallback/untrained weights; top-5% heuristic fallback)"
    )
    ax2.text(
        0.03,
        0.05,
        conf_str,
        transform=ax2.transAxes,
        verticalalignment="bottom",
        fontsize=8.3,
        color="#fcd34d" if not ret_sufficient else "#34d399",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#0f172a", edgecolor="#475569", alpha=0.9),
    )

    plt.tight_layout()
    for p in save_paths:
        plt.savefig(p, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_elasticnet_vs_omp(
    bench_results: Dict[str, Any], save_paths: List[str]
) -> None:
    """
    Generates `elasticnet_vs_omp.png`:
    - Panel A: Median GPU Encoding Latency (ms, with IQR error bars) AND Cosine Similarity
      annotations for Fast 1-Step Approximate Encoder vs Sequential OMP vs Iterative FISTA
      ElasticNet Reference across s in {4, 8, 12, 16}.
    - Panel B: Nested Partial Decoding Curve (active atoms m in {2, 4, 8, 12, 16}) comparing
      Cosine Similarity (`norm_rescale=True` vs `norm_rescale=False`) and Relative MSE.
    """
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.6, 6.0), dpi=220)
    fig.patch.set_facecolor("#0f172a")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1e293b")

    sw = bench_results["sparsity_sweep"]
    s_list = sw["s_values"]
    en_ms = sw["fast_1step_encode_median_ms"]
    en_iqr = sw["fast_1step_encode_iqr_ms"]
    omp_ms = sw["omp_encode_median_ms"]
    omp_iqr = sw["omp_encode_iqr_ms"]
    fista_ms = sw["fista_ref_encode_median_ms"]
    fista_iqr = sw["fista_ref_encode_iqr_ms"]

    en_cos = sw["fast_1step_cosine_sim"]
    omp_cos = sw["omp_cosine_sim"]
    fista_cos = sw["fista_ref_cosine_sim"]

    x_pos = np.arange(len(s_list))
    width = 0.26

    ax1.bar(
        x_pos - width,
        omp_ms,
        width,
        yerr=omp_iqr,
        capsize=3,
        label="Sequential Greedy OMP (exact LS projection)",
        color="#ef4444",
        alpha=0.88,
        edgecolor="#fecaca",
    )
    ax1.bar(
        x_pos,
        fista_ms,
        width,
        yerr=fista_iqr,
        capsize=3,
        label="Iterative FISTA ElasticNet Ref (60 steps)",
        color="#6366f1",
        alpha=0.88,
        edgecolor="#e0e7ff",
    )
    ax1.bar(
        x_pos + width,
        en_ms,
        width,
        yerr=en_iqr,
        capsize=3,
        label="Fast 1-Step Ridge + Soft-Threshold (Approx)",
        color="#10b981",
        alpha=0.92,
        edgecolor="#d1fae5",
    )

    max_y = max(max(omp_ms), max(fista_ms), max(en_ms))
    for idx in range(len(s_list)):
        ax1.text(
            x_pos[idx] - width,
            omp_ms[idx] + max_y * 0.02,
            f"{omp_ms[idx]:.2f}ms\ncos={omp_cos[idx]:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.3,
            color="#fca5a5",
        )
        ax1.text(
            x_pos[idx],
            fista_ms[idx] + max_y * 0.02,
            f"{fista_ms[idx]:.2f}ms\ncos={fista_cos[idx]:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.3,
            color="#c7d2fe",
        )
        ax1.text(
            x_pos[idx] + width,
            en_ms[idx] + max_y * 0.02,
            f"{en_ms[idx]:.2f}ms\ncos={en_cos[idx]:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
            color="#34d399",
        )

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"s = {s}\n({3 * s + 2} B/vec)" for s in s_list], fontsize=9.2, color="#e2e8f0")
    ax1.set_ylabel("Median GPU Encode Latency for N=1024 Vectors (ms, +/- IQR)", fontsize=9.8, color="#cbd5e1")
    ax1.set_title(
        "Encoder Latency & Cosine Quality vs. OMP and FISTA Reference",
        fontsize=11.8,
        fontweight="bold",
        color="#f8fafc",
        pad=12,
    )
    ax1.set_ylim(0, max_y * 1.32)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.25, color="#94a3b8")
    ax1.legend(loc="upper left", frameon=True, facecolor="#0f172a", edgecolor="#334155", fontsize=8.3)

    # Panel B: Nested Partial Decoding (active_atoms m in {2, 4, 8, 12, 16})
    ms = bench_results["matryoshka_sweep"]
    m_list = ms["m_values"]
    cos_rescaled = ms["cosine_similarity_rescaled"]
    rel_mse_rescaled = ms["relative_mse_rescaled"]
    rel_mse_unrescaled = ms["relative_mse_unrescaled"]
    bytes_vec = ms["logical_payload_bytes_per_vector"]

    ax2_twin = ax2.twinx()
    line1 = ax2.plot(
        m_list,
        cos_rescaled,
        "-o",
        color="#38bdf8",
        linewidth=2.5,
        markersize=7.5,
        label="Cosine Similarity cos(X, X_hat_m)",
    )
    line2 = ax2_twin.plot(
        m_list,
        rel_mse_rescaled,
        "--s",
        color="#f59e0b",
        linewidth=2.2,
        markersize=7,
        label="Rel MSE (norm_rescale=True)",
    )
    line3 = ax2_twin.plot(
        m_list,
        rel_mse_unrescaled,
        ":^",
        color="#f43f5e",
        linewidth=2.0,
        markersize=6.5,
        label="Rel MSE (norm_rescale=False)",
    )

    for m_val, c_val, b_val in zip(m_list, cos_rescaled, bytes_vec):
        y_off = -24 if m_val in (12, 16) else 9
        ax2.annotate(
            f"m={m_val} ({b_val}B)\ncos={c_val:.3f}",
            (m_val, c_val),
            textcoords="offset points",
            xytext=(0, y_off),
            ha="center",
            fontsize=8.0,
            color="#e0f2fe",
        )

    ax2.set_title(
        "Nested Partial Decoding ($|y_{(1)}| \\geq \\dots \\geq |y_{(s)}|$, single s=16 code)",
        fontsize=11.4,
        fontweight="bold",
        color="#f8fafc",
        pad=12,
    )
    ax2.set_xlabel("Active Decoded Atoms $m$ (out of $s=16$ stored atoms)", fontsize=10.0, color="#cbd5e1")
    ax2.set_ylabel("Cosine Similarity $\\cos(X, \\hat{X}_m)$", fontsize=10.0, color="#38bdf8")
    ax2_twin.set_ylabel("Relative Reconstruction Error $\\|X - \\hat{X}_m\\|_2^2 / \\|X\\|_2^2$", fontsize=10.0, color="#f59e0b")
    ax2.set_xticks(m_list)
    ax2.set_xticklabels([f"m={m}\n({3 * m + 2}B)" for m in m_list], fontsize=9.0)
    ax2.set_ylim(min(cos_rescaled) - 0.08, 1.05)
    ax2.grid(True, linestyle=":", alpha=0.25, color="#94a3b8")

    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="upper right", frameon=True, facecolor="#0f172a", edgecolor="#334155", fontsize=8.2)

    plt.tight_layout()
    for p in save_paths:
        plt.savefig(p, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_kannada_code_lstar(
    domain_results: Dict[str, Any], save_paths: List[str]
) -> None:
    """
    Generates `kannada_code_lstar.png`:
    - Panel A: Cumulative KV Cache Bytes vs. Sequence Length L on held-out tokens showing
      crossover length L* (or marking no crossover if delta_s <= 0) across Kannada, Python, and English.
    - Panel B: Held-Out Empirical Break-Even Context Length L* and Sparsity Reduction delta_s
      (explicitly labeling any domain where break_even_exists is False).
    """
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 6.0), dpi=220)
    fig.patch.set_facecolor("#0f172a")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1e293b")

    s_univ = float(domain_results["english"]["s_univ"])
    head_dim = int(domain_results["english"]["head_dim"])
    M_adapt = int(domain_results["english"]["M_adapt"])
    overhead_kb = (2.0 * head_dim * M_adapt) / 1024.0

    finite_l_stars = [
        float(domain_results[k]["L_star"])
        for k in ("kannada", "python", "english")
        if domain_results[k]["L_star"] is not None
    ]
    max_l_grid = max(4096.0, (max(finite_l_stars) * 1.25) if finite_l_stars else 4096.0)
    L_grid = np.linspace(128.0, min(max_l_grid, 16384.0), 250)

    univ_kb = (L_grid * (3.0 * s_univ + 2.0)) / 1024.0
    ax1.plot(
        L_grid,
        univ_kb,
        "--",
        color="#94a3b8",
        linewidth=2.4,
        label=f"Universal Dict Only (s={int(s_univ)}, 0 KB overhead)",
    )

    domain_visuals = [
        ("kannada", "Kannada Script (Held-Out UTF-8 Subwords)", "#10b981"),
        ("python", "Python Code (Held-Out Syntax)", "#38bdf8"),
        ("english", "English Prose (Held-Out)", "#f59e0b"),
    ]

    for key, label, col in domain_visuals:
        info = domain_results[key]
        s_ad = float(info["s_adapt"])
        l_star = info["L_star"]
        adapt_kb = overhead_kb + (L_grid * (3.0 * s_ad + 2.0)) / 1024.0
        l_str = f"L*={l_star:.0f}" if l_star is not None else "No Break-Even"
        ax1.plot(
            L_grid,
            adapt_kb,
            "-",
            color=col,
            linewidth=2.4,
            label=f"{label}: s_adapt={s_ad:.1f}, {l_str}",
        )
        if l_star is not None and l_star <= L_grid.max():
            cross_kb = (l_star * (3.0 * s_univ + 2.0)) / 1024.0
            ax1.scatter([l_star], [cross_kb], color=col, s=95, zorder=6, edgecolors="#f8fafc", linewidth=1.2)
            ax1.axvline(l_star, color=col, linestyle=":", alpha=0.55, linewidth=1.3)

    ax1.set_title(
        f"Held-Out SVD Residual Adaptation ($M_{{adapt}}={M_adapt}$): Memory Crossover",
        fontsize=11.0,
        fontweight="bold",
        color="#f8fafc",
        pad=12,
    )
    ax1.set_xlabel("Context Sequence Length $L$ (Tokens per Head)", fontsize=10.0, color="#cbd5e1")
    ax1.set_ylabel("Total Compressed KV Footprint per Head (KB)", fontsize=10.0, color="#cbd5e1")
    ax1.grid(True, linestyle=":", alpha=0.25, color="#94a3b8")
    ax1.legend(loc="upper left", frameon=True, facecolor="#0f172a", edgecolor="#334155", fontsize=8.2)

    # Panel B: Bar chart of Held-Out Break-Even Context Length L* (or No Break-Even marker)
    labels = ["Kannada\n(Held-Out Prompts)", "Python Code\n(Held-Out Prompts)", "English\n(Held-Out Prompts)"]
    keys = ["kannada", "python", "english"]
    colors = ["#10b981", "#38bdf8", "#f59e0b"]

    ceiling_val = max(finite_l_stars) * 1.15 if finite_l_stars else 4096.0
    plot_heights: List[float] = []
    for k in keys:
        ls = domain_results[k]["L_star"]
        plot_heights.append(float(ls) if ls is not None else ceiling_val)

    x_idx = np.arange(len(keys))
    bars = ax2.bar(x_idx, plot_heights, width=0.50, color=colors, alpha=0.88, edgecolor="#f8fafc", linewidth=1.0)
    for idx, (bar, k) in enumerate(zip(bars, keys)):
        info = domain_results[k]
        ls = info["L_star"]
        sad = info["s_adapt"]
        ds = info["delta_s"]
        cos_g = info["cosine_gain_at_s_univ"]
        if ls is None:
            bar.set_hatch("//")
            txt = f"No Break-Even ($\\Delta s = {ds:.1f} \\leq 0$)\n$s_{{adapt}} = {sad:.1f}$, $\\Delta\\cos = +{cos_g:.3f}$"
        else:
            txt = (
                f"$L^* = {ls:.0f}$ tokens\n"
                f"($s_{{adapt}} = {sad:.1f}$, $\\Delta s = {ds:.1f}$, $\\Delta\\cos = +{cos_g:.3f}$)"
            )
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(plot_heights) * 0.03,
            txt,
            ha="center",
            va="bottom",
            fontsize=8.6,
            fontweight="bold",
            color="#f8fafc",
        )

    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(labels, fontsize=9.8, color="#e2e8f0")
    ax2.set_ylabel(
        "Held-Out Break-Even Length $L^* = \\frac{2 d M_{adapt}}{3(s_{univ} - s_{adapt})}$ (Tokens)",
        fontsize=9.8,
        color="#cbd5e1",
    )
    ax2.set_title(
        "Empirical Break-Even Length $L^*$ by Domain (N=256 Held-Out)",
        fontsize=11.0,
        fontweight="bold",
        color="#f8fafc",
        pad=12,
    )
    ax2.set_ylim(0, max(plot_heights) * 1.32)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.25, color="#94a3b8")

    plt.tight_layout()
    for p in save_paths:
        plt.savefig(p, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def run_person1_benchmarks(seed: int = 42) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("=" * 80)
    print("  StoikovKV Person 1 Audited End-to-End Benchmark Suite")
    print("=" * 80)

    # 1. Load model and run offline calibration
    model, tokenizer = load_model_and_tokenizer(seed=seed)
    artifacts_path = os.path.join(REPO_ROOT, "calibrated_artifacts.pt")
    artifacts = calibrate_model(model, tokenizer, save_path=artifacts_path, seed=seed)

    codec = ElasticLexicoCodec.from_artifacts(artifacts)
    device = codec.device

    # Collect held-out real KV states (1024 vectors) for codec benchmarking
    X_eval_pool, pool_stats = _collect_domain_kv_states(
        model,
        tokenizer,
        ENGLISH_EVAL_PROMPTS + PYTHON_EVAL_PROMPTS,
        layer_idx=0,
        target_tokens=1024,
        max_seq_len=256,
    )
    X_bench = X_eval_pool[:1024].contiguous()

    # 2. Fair Encoder & Decoder Benchmark across s in {4, 8, 12, 16}
    s_values = [4, 8, 12, 16]
    fast_encode_median: List[float] = []
    fast_encode_iqr: List[float] = []
    fast_decode_median: List[float] = []
    fast_decode_iqr: List[float] = []
    omp_encode_median: List[float] = []
    omp_encode_iqr: List[float] = []
    fista_encode_median: List[float] = []
    fista_encode_iqr: List[float] = []
    speedup_vs_omp: List[float] = []
    speedup_vs_fista: List[float] = []

    fast_cos_list: List[float] = []
    fast_mse_list: List[float] = []
    fast_rel_l2_list: List[float] = []
    fast_no_rescale_mse_list: List[float] = []
    omp_cos_list: List[float] = []
    omp_mse_list: List[float] = []
    omp_rel_l2_list: List[float] = []
    fista_cos_list: List[float] = []
    fista_mse_list: List[float] = []
    fista_rel_l2_list: List[float] = []
    csr_bytes_list: List[int] = []
    per_s_timing_details: Dict[str, Any] = {}

    print("\n--- Benchmark 2: Fast 1-Step vs. Sequential OMP vs. Iterative FISTA Reference ---")
    for s in s_values:
        # Timed GPU measurements with warmup and CUDA synchronization
        fast_enc_stats = _measure_latency_stats(
            lambda s_val=s: codec.encode(X_bench, layer_idx=0, s=s_val, lambda1=0.04, norm_rescale=True),
            device=device,
            warmup=5,
            repetitions=15,
        )
        idx_en, val_en = codec.encode(X_bench, layer_idx=0, s=s, lambda1=0.04, norm_rescale=True)
        idx_en_no_rs, val_en_no_rs = codec.encode(
            X_bench, layer_idx=0, s=s, lambda1=0.04, norm_rescale=False
        )

        fast_dec_stats = _measure_latency_stats(
            lambda idx_v=idx_en, val_v=val_en: codec.decode(idx_v, val_v, layer_idx=0),
            device=device,
            warmup=5,
            repetitions=15,
        )

        omp_enc_stats = _measure_latency_stats(
            lambda s_val=s: codec.omp_encode(X_bench, layer_idx=0, s=s_val),
            device=device,
            warmup=2,
            repetitions=7,
        )
        idx_omp, val_omp = codec.omp_encode(X_bench, layer_idx=0, s=s)

        fista_enc_stats = _measure_latency_stats(
            lambda s_val=s: codec.iterative_elasticnet_encode(
                X_bench, layer_idx=0, s=s_val, lambda1=0.04, lambda2=0.1, num_iters=60, norm_rescale=True
            ),
            device=device,
            warmup=2,
            repetitions=7,
        )
        idx_fista, val_fista = codec.iterative_elasticnet_encode(
            X_bench, layer_idx=0, s=s, lambda1=0.04, lambda2=0.1, num_iters=60, norm_rescale=True
        )

        # Verify descending magnitude order and exact byte accounting
        deq_abs = val_en.dequantize().abs()
        assert int(((deq_abs[:, :-1] - deq_abs[:, 1:]) < -1e-6).sum().item()) == 0
        assert idx_en.dtype == torch.int16
        assert val_en.dtype == torch.int8
        assert val_en.bytes_per_vector == 3 * s + 2

        x_hat_en = codec.decode(idx_en, val_en, layer_idx=0)
        x_hat_en_no_rs = codec.decode(idx_en_no_rs, val_en_no_rs, layer_idx=0)
        x_hat_omp = codec.decode(idx_omp, val_omp, layer_idx=0)
        x_hat_fista = codec.decode(idx_fista, val_fista, layer_idx=0)

        q_en = codec.evaluate_reconstruction(X_bench, x_hat_en)
        q_en_no_rs = codec.evaluate_reconstruction(X_bench, x_hat_en_no_rs)
        q_omp = codec.evaluate_reconstruction(X_bench, x_hat_omp)
        q_fista = codec.evaluate_reconstruction(X_bench, x_hat_fista)

        spd_omp = omp_enc_stats["median_ms"] / max(fast_enc_stats["median_ms"], 1e-4)
        spd_fista = fista_enc_stats["median_ms"] / max(fast_enc_stats["median_ms"], 1e-4)

        fast_encode_median.append(fast_enc_stats["median_ms"])
        fast_encode_iqr.append(fast_enc_stats["iqr_ms"])
        fast_decode_median.append(fast_dec_stats["median_ms"])
        fast_decode_iqr.append(fast_dec_stats["iqr_ms"])
        omp_encode_median.append(omp_enc_stats["median_ms"])
        omp_encode_iqr.append(omp_enc_stats["iqr_ms"])
        fista_encode_median.append(fista_enc_stats["median_ms"])
        fista_encode_iqr.append(fista_enc_stats["iqr_ms"])
        speedup_vs_omp.append(round(spd_omp, 2))
        speedup_vs_fista.append(round(spd_fista, 2))

        fast_cos_list.append(round(q_en["cosine_similarity"], 4))
        fast_mse_list.append(round(q_en["mse"], 6))
        fast_rel_l2_list.append(round(q_en["relative_l2_error"], 4))
        fast_no_rescale_mse_list.append(round(q_en_no_rs["mse"], 6))

        omp_cos_list.append(round(q_omp["cosine_similarity"], 4))
        omp_mse_list.append(round(q_omp["mse"], 6))
        omp_rel_l2_list.append(round(q_omp["relative_l2_error"], 4))

        fista_cos_list.append(round(q_fista["cosine_similarity"], 4))
        fista_mse_list.append(round(q_fista["mse"], 6))
        fista_rel_l2_list.append(round(q_fista["relative_l2_error"], 4))
        csr_bytes_list.append(val_en.bytes_per_vector)

        per_s_timing_details[f"s={s}"] = {
            "fast_1step_encode": fast_enc_stats,
            "fast_1step_decode": fast_dec_stats,
            "omp_encode": omp_enc_stats,
            "fista_ref_encode": fista_enc_stats,
            "quality_fast_1step_rescaled": q_en,
            "quality_fast_1step_unrescaled": q_en_no_rs,
            "quality_omp": q_omp,
            "quality_fista_ref": q_fista,
            "memory_accounting": codec.memory_accounting(idx_en, val_en),
        }

        print(
            f"  s={s:2d} ({val_en.bytes_per_vector:2d} B/vec) | "
            f"Fast1Step: {fast_enc_stats['median_ms']:6.3f}ms (cos={q_en['cosine_similarity']:.4f}, MSE={q_en['mse']:.5f}) | "
            f"OMP: {omp_enc_stats['median_ms']:7.3f}ms (cos={q_omp['cosine_similarity']:.4f}, MSE={q_omp['mse']:.5f}) | "
            f"FISTA: {fista_enc_stats['median_ms']:6.3f}ms (cos={q_fista['cosine_similarity']:.4f})"
        )

    # Also measure CPU encode/decode latency at s=16 on 256 vectors for hardware portability reporting
    codec_cpu = ElasticLexicoCodec(
        D=codec.D.detach().cpu(),
        W_ridge=codec.W_ridge.detach().cpu(),
        lambda1=codec.lambda1,
        lambda2=codec.lambda2,
    )
    X_cpu = X_bench[:256].detach().cpu().to(torch.float32)
    cpu_enc_stats = _measure_latency_stats(
        lambda: codec_cpu.encode(X_cpu, layer_idx=0, s=16),
        device=torch.device("cpu"),
        warmup=2,
        repetitions=7,
    )
    idx_cpu, val_cpu = codec_cpu.encode(X_cpu, layer_idx=0, s=16)
    cpu_dec_stats = _measure_latency_stats(
        lambda: codec_cpu.decode(idx_cpu, val_cpu, layer_idx=0),
        device=torch.device("cpu"),
        warmup=2,
        repetitions=7,
    )

    # 3. Benchmark Nested Partial Decoding (m in {2, 4, 8, 12, 16}) from a single s=16 code
    print("\n--- Benchmark 3: Nested Partial Decoder (Single s=16 Code, Rescaled vs Unrescaled) ---")
    idx_16_rs, val_16_rs = codec.encode(X_bench, layer_idx=0, s=16, lambda1=0.04, norm_rescale=True)
    idx_16_un, val_16_un = codec.encode(X_bench, layer_idx=0, s=16, lambda1=0.04, norm_rescale=False)
    m_values = [2, 4, 8, 12, 16]
    mat_cos_rs: List[float] = []
    mat_rel_mse_rs: List[float] = []
    mat_cos_un: List[float] = []
    mat_rel_mse_un: List[float] = []
    mat_bytes: List[int] = []
    mat_dec_median_ms: List[float] = []

    for m in m_values:
        x_hat_m_rs = codec.decode(idx_16_rs, val_16_rs, layer_idx=0, active_atoms=m)
        x_hat_m_un = codec.decode(idx_16_un, val_16_un, layer_idx=0, active_atoms=m)
        q_m_rs = codec.evaluate_reconstruction(X_bench, x_hat_m_rs)
        q_m_un = codec.evaluate_reconstruction(X_bench, x_hat_m_un)
        dec_m_stats = _measure_latency_stats(
            lambda m_val=m: codec.decode(idx_16_rs, val_16_rs, layer_idx=0, active_atoms=m_val),
            device=device,
            warmup=3,
            repetitions=10,
        )
        bytes_m = 3 * m + 2
        mat_cos_rs.append(round(q_m_rs["cosine_similarity"], 4))
        mat_rel_mse_rs.append(round(q_m_rs["relative_mse"], 4))
        mat_cos_un.append(round(q_m_un["cosine_similarity"], 4))
        mat_rel_mse_un.append(round(q_m_un["relative_mse"], 4))
        mat_bytes.append(bytes_m)
        mat_dec_median_ms.append(dec_m_stats["median_ms"])
        print(
            f"  active_atoms m={m:2d} | Logical Payload={bytes_m:2d} B/vec | "
            f"CosSim={q_m_rs['cosine_similarity']:.4f} | "
            f"RelMSE (rescaled)={q_m_rs['relative_mse']:.4f} | "
            f"RelMSE (unrescaled)={q_m_un['relative_mse']:.4f}"
        )

    # 4. Benchmark Held-Out Online SVD Residual Dictionary Adaptation (M_adapt = 64) & L*
    print("\n--- Benchmark 4: Held-Out Online Residual Dictionary Adaptation (L*) Across Domains ---")
    domain_splits = {
        "english": (ENGLISH_ADAPT_PROMPTS, ENGLISH_EVAL_PROMPTS),
        "python": (PYTHON_ADAPT_PROMPTS, PYTHON_EVAL_PROMPTS),
        "kannada": (KANNADA_ADAPT_PROMPTS, KANNADA_EVAL_PROMPTS),
    }
    domain_results: Dict[str, Any] = {}
    for dom_name, (adapt_prompts, eval_prompts) in domain_splits.items():
        X_adapt, adapt_stats = _collect_domain_kv_states(
            model, tokenizer, adapt_prompts, layer_idx=0, target_tokens=256, max_seq_len=256
        )
        X_eval, eval_stats = _collect_domain_kv_states(
            model, tokenizer, eval_prompts, layer_idx=0, target_tokens=256, max_seq_len=256
        )
        idx_ad, val_ad = codec.encode(X_adapt, layer_idx=0, s=16, lambda1=0.04)
        X_hat_adapt = codec.decode(idx_ad, val_ad, layer_idx=0)

        idx_ev, val_ev = codec.encode(X_eval, layer_idx=0, s=16, lambda1=0.04)
        X_hat_eval = codec.decode(idx_ev, val_ev, layer_idx=0)

        res_info = codec.adapt_online_atoms(
            X_adapt,
            X_hat_adapt,
            M_adapt=64,
            layer_idx=0,
            s_univ=16,
            X_eval=X_eval,
            X_hat_eval=X_hat_eval,
        )
        res_info_clean = {k: v for k, v in res_info.items() if not isinstance(v, torch.Tensor)}
        res_info_clean["adapt_split_stats"] = adapt_stats
        res_info_clean["eval_split_stats"] = eval_stats
        domain_results[dom_name] = res_info_clean
        l_star_str = f"{res_info['L_star']:.1f} tokens" if res_info["L_star"] is not None else "None (no break-even)"
        print(
            f"  Domain={dom_name.upper():8s} (held-out={res_info['evaluated_on_heldout']}) | "
            f"s_univ={res_info['s_univ']:4.1f} -> s_adapt={res_info['s_adapt']:5.2f} "
            f"(delta_s={res_info['delta_s']:5.2f}) | L* = {l_star_str}"
        )

    # Verify strict pairwise disjointness across all training/calibration, probe held-out,
    # online adaptation fit, and online domain evaluation prompt splits
    calib_train_set = set(CALIBRATION_PROMPTS) | set(NEEDLE_PROMPTS)
    probe_heldout_set = set(HELDOUT_EVAL_PROMPTS)
    online_adapt_set = (
        set(ENGLISH_ADAPT_PROMPTS) | set(PYTHON_ADAPT_PROMPTS) | set(KANNADA_ADAPT_PROMPTS)
    )
    online_eval_set = (
        set(ENGLISH_EVAL_PROMPTS) | set(PYTHON_EVAL_PROMPTS) | set(KANNADA_EVAL_PROMPTS)
    )
    assert calib_train_set.isdisjoint(probe_heldout_set), "Calibration and held-out sets must be disjoint"
    assert calib_train_set.isdisjoint(online_adapt_set), "Calibration and online adapt sets must be disjoint"
    assert calib_train_set.isdisjoint(online_eval_set), "Calibration and online eval sets must be disjoint"
    assert probe_heldout_set.isdisjoint(online_adapt_set), "Probe held-out and online adapt sets must be disjoint"
    assert probe_heldout_set.isdisjoint(online_eval_set), "Probe held-out and online eval sets must be disjoint"
    assert online_adapt_set.isdisjoint(online_eval_set), "Online adapt and online eval sets must be disjoint"

    # 5. Benchmark End-to-End Compressed-Cache Passkey Retrieval & Next-Token Logit Fidelity
    print("\n--- Benchmark 5: Compressed-Cache Passkey Retrieval & Next-Token Logit Fidelity ---")
    passkey_eval = evaluate_compressed_cache_passkey(
        model=model,
        tokenizer=tokenizer,
        artifacts=artifacts,
        codec=codec,
        s=16,
        draft_atoms=4,
        h2o_keep_ratio=0.50,
    )
    for pol_name, pol_stats in passkey_eval["policies"].items():
        print(
            f"  Policy={pol_name:24s} ({pol_stats['effective_bytes_per_kv_vector']:6.2f} B/vec, "
            f"{pol_stats['compression_ratio_vs_fp16']:4.2f}x) | "
            f"Top1 Acc={pol_stats['passkey_top1_accuracy'] * 100:5.1f}% "
            f"(HeldOut={pol_stats['heldout_passkey_top1_accuracy'] * 100:5.1f}%) | "
            f"ProbRet={pol_stats['passkey_prob_retention_ratio'] * 100:5.1f}% | "
            f"LogitCos={pol_stats['logit_cosine_similarity']:.4f}"
        )

    bench_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(bench_dir, exist_ok=True)

    p1_paths = [
        os.path.join(REPO_ROOT, "probe_correlation.png"),
        os.path.join(bench_dir, "probe_correlation.png"),
    ]
    p2_paths = [
        os.path.join(REPO_ROOT, "elasticnet_vs_omp.png"),
        os.path.join(bench_dir, "elasticnet_vs_omp.png"),
    ]
    p3_paths = [
        os.path.join(REPO_ROOT, "kannada_code_lstar.png"),
        os.path.join(bench_dir, "kannada_code_lstar.png"),
    ]

    summary_payload: Dict[str, Any] = {
        "benchmark_metadata": {
            "seed": seed,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "torch_version": torch.__version__,
            "model_class": model.__class__.__name__,
            "is_fallback_random_init_model": bool(
                model.__class__.__name__ == "FallbackQwen2ForCausalLM"
            ),
            "codec_bench_num_vectors": int(X_bench.shape[0]),
            "codec_bench_head_dim": int(X_bench.shape[1]),
            "codec_bench_dtype": str(X_bench.dtype),
            "domain_eval_stats": pool_stats,
        },
        "config": artifacts["config"],
        "probe_metrics": {
            "spearman_mean": round(artifacts["probe_metrics"]["spearman_mean"], 4),
            "recall_at_20_mean": round(artifacts["probe_metrics"]["recall_at_20_mean"], 4),
            "raw_probe": artifacts["probe_metrics"]["raw_probe"],
            "sink_augmented_probe": artifacts["probe_metrics"]["sink_augmented_probe"],
            "value_norm_baseline": artifacts["probe_metrics"]["value_norm_baseline"],
            "dict_mean_mse": round(artifacts["probe_metrics"]["dict_mean_mse"], 6),
            "dict_diagnostics": artifacts["probe_metrics"]["dict_diagnostics"],
        },
        "retrieval_diagnostics": artifacts.get("retrieval_diagnostics", {}),
        "head_role_counts": {
            "ROLE_RETRIEVAL_FP16": int((artifacts["head_roles"] == ROLE_RETRIEVAL_FP16).sum().item()),
            "ROLE_PEAKED_H2O": int((artifacts["head_roles"] == ROLE_PEAKED_H2O).sum().item()),
            "ROLE_DIFFUSE_LEXICO": int((artifacts["head_roles"] == ROLE_DIFFUSE_LEXICO).sum().item()),
        },
        "sparsity_sweep": {
            "s_values": s_values,
            "csr_bytes_per_vector": csr_bytes_list,
            "elasticnet_latency_ms": fast_encode_median,
            "omp_latency_ms": omp_encode_median,
            "speedup_x": speedup_vs_omp,
            "elasticnet_cosine_sim": fast_cos_list,
            "omp_cosine_sim": omp_cos_list,
            "elasticnet_mse": fast_mse_list,
            "omp_mse": omp_mse_list,
            "fast_1step_encode_median_ms": fast_encode_median,
            "fast_1step_encode_iqr_ms": fast_encode_iqr,
            "fast_1step_decode_median_ms": fast_decode_median,
            "fast_1step_decode_iqr_ms": fast_decode_iqr,
            "omp_encode_median_ms": omp_encode_median,
            "omp_encode_iqr_ms": omp_encode_iqr,
            "fista_ref_encode_median_ms": fista_encode_median,
            "fista_ref_encode_iqr_ms": fista_encode_iqr,
            "speedup_vs_omp_x": speedup_vs_omp,
            "speedup_vs_fista_x": speedup_vs_fista,
            "fast_1step_cosine_sim": fast_cos_list,
            "fast_1step_mse": fast_mse_list,
            "fast_1step_rel_l2": fast_rel_l2_list,
            "fast_1step_unrescaled_mse": fast_no_rescale_mse_list,
            "omp_rel_l2": omp_rel_l2_list,
            "fista_ref_cosine_sim": fista_cos_list,
            "fista_ref_mse": fista_mse_list,
            "fista_ref_rel_l2": fista_rel_l2_list,
            "per_s_details": per_s_timing_details,
            "cpu_portability_s16_n256": {
                "encode": cpu_enc_stats,
                "decode": cpu_dec_stats,
            },
        },
        "matryoshka_sweep": {
            "m_values": m_values,
            "bytes_per_vector": mat_bytes,
            "logical_payload_bytes_per_vector": mat_bytes,
            "cosine_similarity": mat_cos_rs,
            "relative_mse": mat_rel_mse_rs,
            "cosine_similarity_rescaled": mat_cos_rs,
            "relative_mse_rescaled": mat_rel_mse_rs,
            "cosine_similarity_unrescaled": mat_cos_un,
            "relative_mse_unrescaled": mat_rel_mse_un,
            "decode_median_ms": mat_dec_median_ms,
        },
        "online_adaptation_lstar": domain_results,
        "compressed_cache_passkey_eval": passkey_eval,
        "split_isolation_audit": {
            "calib_train_count": len(calib_train_set),
            "probe_heldout_count": len(probe_heldout_set),
            "online_adapt_count": len(online_adapt_set),
            "online_eval_count": len(online_eval_set),
            "all_four_splits_pairwise_disjoint": True,
        },
    }

    plot_probe_correlation(artifacts, p1_paths)
    plot_elasticnet_vs_omp(summary_payload, p2_paths)
    plot_kannada_code_lstar(domain_results, p3_paths)

    json_paths = [
        os.path.join(REPO_ROOT, "person1_metrics.json"),
        os.path.join(bench_dir, "person1_metrics.json"),
    ]
    for jp in json_paths:
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)

    print("\n[StoikovKV Benchmark Complete]")
    print(f"  Saved Artifacts : {artifacts_path}")
    print(f"  Saved Plots     : {p1_paths[0]}, {p2_paths[0]}, {p3_paths[0]}")
    print(f"  Saved Metrics   : {json_paths[0]}")
    return summary_payload


if __name__ == "__main__":
    run_person1_benchmarks()
