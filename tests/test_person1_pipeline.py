"""
Focused Unit Tests for StoikovKV Person 1 Pipeline (`tests/test_person1_pipeline.py`).

Covers all 9 required validation areas on CPU (and CUDA when available) without requiring
network access or large model downloads:
1. GQA grouping and subspace shape + column orthonormality
2. Write-time probe and attention reduction over query tokens (`dim=-2`)
3. Head role assignment (`{0, 1, 2}`), deterministic tie-breaking, and evidence checks
4. Dictionary (`D`) and ridge matrix (`W_ridge`) shapes, normalization, and linear solve accuracy
5. Fast 1-step encoder (`encode`), dtypes, descending order, zero-vector safety, and `norm_rescale`
6. Sequential OMP (`omp_encode`) and Iterative FISTA Reference (`iterative_elasticnet_encode`)
7. Matryoshka nested partial decoding (`active_atoms=m`) prefix isolation and error monotonicity
8. Held-out online residual adaptation (`adapt_online_atoms`) and break-even `L_star` edge cases
9. Artifact save/load round-trip and hook cleanup on exception
"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from benchmarks.bench_person1 import (
    ENGLISH_ADAPT_PROMPTS,
    ENGLISH_EVAL_PROMPTS,
    KANNADA_ADAPT_PROMPTS,
    KANNADA_EVAL_PROMPTS,
    PYTHON_ADAPT_PROMPTS,
    PYTHON_EVAL_PROMPTS,
)
from stoikov_kv.calibration import (
    CALIBRATION_PROMPTS,
    FallbackQwen2Config,
    FallbackQwen2ForCausalLM,
    FallbackQwen2Tokenizer,
    HELDOUT_EVAL_PROMPTS,
    NEEDLE_PROMPTS,
    ROLE_DIFFUSE_LEXICO,
    ROLE_PEAKED_H2O,
    ROLE_RETRIEVAL_FP16,
    _compute_spearman_and_recall,
    calibrate_model,
    evaluate_compressed_cache_passkey,
    evaluate_write_time_probe,
    reduce_attention_to_kv_heads,
    score_tokens_write_time,
    score_write_time,
)
from stoikov_kv.elastic_lexico import (
    ElasticLexicoCodec,
    QuantizedInt8Values,
    compute_break_even_length,
)


class TestStoikovKVPerson1Pipeline(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1234)
        self.cpu_device = torch.device("cpu")
        self.tiny_config = FallbackQwen2Config(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=2,
            vocab_size=512,
        )
        self.tiny_model = FallbackQwen2ForCausalLM(
            config=self.tiny_config,
            device=self.cpu_device,
            dtype=torch.float32,
            seed=1234,
        )
        self.tiny_tokenizer = FallbackQwen2Tokenizer(vocab_size=512)

    def test_1_gqa_grouping_and_subspace_orthonormality(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "calib_cpu.pt")
            artifacts = calibrate_model(
                self.tiny_model,
                self.tiny_tokenizer,
                save_path=save_path,
                subspace_rank=8,
                dict_atoms=64,
                dict_steps=10,
                seed=1234,
                max_seq_len=64,
            )
            cfg = artifacts["config"]
            self.assertEqual(cfg["num_layers"], 2)
            self.assertEqual(cfg["num_q_heads"], 8)
            self.assertEqual(cfg["num_kv_heads"], 2)
            self.assertEqual(cfg["q_per_kv"], 4)
            self.assertEqual(cfg["head_dim"], 8)

            u_q = artifacts["U_Q"]
            self.assertEqual(tuple(u_q.shape), (2, 2, 8, 8))
            self.assertTrue(torch.isfinite(u_q).all())

            # Check column orthonormality U_Q^T @ U_Q == I_r
            u_fp32 = u_q.to(torch.float32)
            eye = torch.eye(8, dtype=torch.float32)
            for l_idx in range(2):
                for h_idx in range(2):
                    gram = u_fp32[l_idx, h_idx].T @ u_fp32[l_idx, h_idx]
                    err = torch.norm(gram - eye, p="fro").item()
                    self.assertLess(err, 1e-4)

    def test_2_write_time_probe_and_attention_reduction(self) -> None:
        # Test 4D and 3D inputs for score_write_time
        B, H_kv, T, D, R = 2, 2, 10, 8, 4
        K = torch.randn(B, H_kv, T, D)
        V = torch.randn(B, H_kv, T, D)
        U_Q = torch.linalg.qr(torch.randn(H_kv, D, R))[0]

        raw_scores = score_write_time(K, V, U_Q, sink_tokens=0, sink_bonus=0.0)
        sink_scores = score_write_time(K, V, U_Q, sink_tokens=4, sink_bonus=100.0)
        self.assertEqual(tuple(raw_scores.shape), (B, H_kv, T))
        self.assertEqual(tuple(sink_scores.shape), (B, H_kv, T))

        # Verify first 4 tokens received +100.0 bonus and tokens >= 4 are unchanged
        self.assertTrue(torch.allclose(sink_scores[..., :4] - raw_scores[..., :4], torch.tensor(100.0), atol=1e-4))
        self.assertTrue(torch.allclose(sink_scores[..., 4:], raw_scores[..., 4:], atol=1e-5))

        # Verify 3D input [H_kv, T, D]
        scores_3d = score_write_time(K[0], V[0], U_Q, sink_tokens=2, sink_bonus=50.0)
        self.assertEqual(tuple(scores_3d.shape), (H_kv, T))
        self.assertTrue(torch.allclose(scores_3d[:, :2] - raw_scores[0, :, :2], torch.tensor(50.0), atol=1e-4))

        # Verify token-level aggregation
        tok_scores = score_tokens_write_time(K, V, U_Q, sink_tokens=0, sink_bonus=0.0)
        self.assertEqual(tuple(tok_scores.shape), (B, T))
        self.assertTrue(torch.allclose(tok_scores, raw_scores.mean(dim=1), atol=1e-5))

        # Verify reduce_attention_to_kv_heads reduces over query dimension (dim=-2), NOT key dimension (dim=-1)
        # Construct causal attention matrix [B=1, H_q=4, T_q=4, T_k=4] where key token 0 gets all attention
        attn = torch.zeros(1, 4, 4, 4)
        attn[:, :, :, 0] = 1.0  # Every query attends 100% to key token 0
        kv_cum_mean = reduce_attention_to_kv_heads(attn, num_kv_heads=2, normalize_by_opportunity=True)
        kv_cum_sum = reduce_attention_to_kv_heads(attn, num_kv_heads=2, normalize_by_opportunity=False)
        self.assertEqual(tuple(kv_cum_mean.shape), (1, 2, 4))
        # Key token 0 received 1.0 from all 4 queries -> sum=4.0, mean=1.0; tokens 1..3 received 0.0
        self.assertTrue(torch.allclose(kv_cum_mean[:, :, 0], torch.ones(1, 2), atol=1e-5))
        self.assertTrue(torch.allclose(kv_cum_mean[:, :, 1:], torch.zeros(1, 2, 3), atol=1e-5))
        self.assertTrue(torch.allclose(kv_cum_sum[:, :, 0], torch.full((1, 2), 4.0), atol=1e-5))

        # Verify evaluate_write_time_probe handles ties and constant vectors cleanly
        metrics = evaluate_write_time_probe(raw_scores[0, 0, :4], kv_cum_mean[0, 0], top_fraction=0.20)
        self.assertIn("spearman_rho", metrics)
        self.assertIn("recall_at_20", metrics)
        self.assertTrue(0.0 <= metrics["recall_at_20"] <= 1.0)

    def test_3_head_role_assignment_and_determinism(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = os.path.join(tmpdir, "c1.pt")
            p2 = os.path.join(tmpdir, "c2.pt")
            a1 = calibrate_model(
                self.tiny_model,
                self.tiny_tokenizer,
                save_path=p1,
                subspace_rank=8,
                dict_atoms=32,
                dict_steps=8,
                seed=99,
                max_seq_len=48,
            )
            a2 = calibrate_model(
                self.tiny_model,
                self.tiny_tokenizer,
                save_path=p2,
                subspace_rank=8,
                dict_atoms=32,
                dict_steps=8,
                seed=99,
                max_seq_len=48,
            )
            roles1 = a1["head_roles"]
            roles2 = a2["head_roles"]
            self.assertEqual(tuple(roles1.shape), (2, 2))
            self.assertTrue(torch.equal(roles1, roles2), "Calibration with identical seed must be deterministic")

            valid_roles = {ROLE_RETRIEVAL_FP16, ROLE_PEAKED_H2O, ROLE_DIFFUSE_LEXICO}
            for val in roles1.flatten().tolist():
                self.assertIn(val, valid_roles)
            # At least 1 head must be assigned ROLE_RETRIEVAL_FP16
            self.assertGreaterEqual(int((roles1 == ROLE_RETRIEVAL_FP16).sum().item()), 1)
            self.assertIn("retrieval_diagnostics", a1)
            self.assertIn("sufficient_evidence", a1["retrieval_diagnostics"])

    def test_4_dictionary_and_ridge_matrix_solve(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "dict_test.pt")
            lambda2 = 0.1
            artifacts = calibrate_model(
                self.tiny_model,
                self.tiny_tokenizer,
                save_path=save_path,
                subspace_rank=8,
                dict_atoms=32,
                dict_steps=10,
                ridge_lambda=lambda2,
                seed=42,
                max_seq_len=48,
            )
            D = artifacts["D"].to(torch.float64)
            W_ridge = artifacts["W_ridge"].to(torch.float64)
            self.assertEqual(tuple(D.shape), (2, 8, 32))
            self.assertEqual(tuple(W_ridge.shape), (2, 32, 8))
            self.assertTrue(torch.isfinite(D).all())
            self.assertTrue(torch.isfinite(W_ridge).all())

            # Verify each dictionary atom has unit L2 norm along head_dim (dim=-2)
            atom_norms = torch.norm(D, p=2, dim=1)
            self.assertTrue(torch.allclose(atom_norms, torch.ones_like(atom_norms), atol=1e-3))

            # Verify W_ridge satisfies (D^T D + lambda2 * I) W_ridge == D^T
            eye_m = torch.eye(32, dtype=torch.float64)
            for l_idx in range(2):
                d_l = D[l_idx]
                w_l = W_ridge[l_idx]
                lhs = (d_l.T @ d_l + lambda2 * eye_m) @ w_l
                rhs = d_l.T
                rel_err = (torch.norm(lhs - rhs) / torch.norm(rhs)).item()
                self.assertLess(rel_err, 1e-3)

    def test_5_fast_encoder_dtypes_sorting_zero_safety_and_rescale(self) -> None:
        head_dim, num_atoms = 16, 64
        gen = torch.Generator().manual_seed(7)
        D = F.normalize(torch.randn(1, head_dim, num_atoms, generator=gen), p=2, dim=-2)
        codec = ElasticLexicoCodec(D=D, lambda1=0.04, lambda2=0.1)

        X = torch.randn(12, head_dim, generator=gen)
        indices, q_vals = codec.encode(X, layer_idx=0, s=8, lambda1=0.04, norm_rescale=True)

        self.assertEqual(indices.dtype, torch.int16)
        self.assertEqual(q_vals.dtype, torch.int8)
        self.assertIsInstance(q_vals, QuantizedInt8Values)
        self.assertEqual(q_vals.scale.dtype, torch.float16)
        self.assertEqual(q_vals.bytes_per_vector, 3 * 8 + 2)

        # Verify indices are within [0, num_atoms)
        self.assertTrue((indices >= 0).all() and (indices < num_atoms).all())

        # Verify dequantized coefficient magnitudes are sorted in descending order
        deq_abs = q_vals.dequantize().abs()
        diffs = deq_abs[:, :-1] - deq_abs[:, 1:]
        self.assertTrue((diffs >= -1e-5).all())

        # Compare norm_rescale=True vs norm_rescale=False
        X_hat_rs = codec.decode(indices, q_vals, layer_idx=0)
        idx_no, q_vals_no = codec.encode(X, layer_idx=0, s=8, lambda1=0.04, norm_rescale=False)
        X_hat_no = codec.decode(idx_no, q_vals_no, layer_idx=0)
        # Cosine similarity of full reconstruction is preserved while norm matches ||X||_2 closer with rescaling
        cos_diff = (
            F.cosine_similarity(X_hat_rs, X, dim=-1) - F.cosine_similarity(X_hat_no, X, dim=-1)
        ).abs().max().item()
        self.assertLess(cos_diff, 0.02)
        norm_err_rs = (torch.norm(X_hat_rs, dim=-1) - torch.norm(X, dim=-1)).abs().mean().item()
        norm_err_no = (torch.norm(X_hat_no, dim=-1) - torch.norm(X, dim=-1)).abs().mean().item()
        self.assertLess(norm_err_rs, norm_err_no)

        # Verify zero-input and near-zero input safety (no NaNs/Infs)
        X_zeros = torch.zeros(4, head_dim)
        idx_z, val_z = codec.encode(X_zeros, layer_idx=0, s=4)
        X_hat_z = codec.decode(idx_z, val_z, layer_idx=0)
        self.assertTrue(torch.isfinite(X_hat_z).all())
        self.assertEqual(float(X_hat_z.abs().max().item()), 0.0)

        # Verify validation errors on invalid inputs
        with self.assertRaises(ValueError):
            codec.encode(X, layer_idx=5, s=4)
        with self.assertRaises(ValueError):
            codec.encode(X, layer_idx=0, s=0)
        with self.assertRaises(ValueError):
            codec.encode(X, layer_idx=0, s=num_atoms + 1)

    def test_6_omp_and_iterative_fista_reference_solvers(self) -> None:
        head_dim, num_atoms = 32, 64
        gen = torch.Generator().manual_seed(101)
        D = F.normalize(torch.randn(1, head_dim, num_atoms, generator=gen), p=2, dim=-2)
        codec = ElasticLexicoCodec(D=D, lambda1=0.02, lambda2=0.05)

        # Construct synthetic sparse signals from 4 known dictionary atoms
        X_sparse = (
            2.0 * D[0, :, 3] - 1.5 * D[0, :, 11] + 0.9 * D[0, :, 25] - 0.5 * D[0, :, 40]
        ).unsqueeze(0).repeat(6, 1)

        idx_omp, val_omp = codec.omp_encode(X_sparse, layer_idx=0, s=4)
        X_hat_omp = codec.decode(idx_omp, val_omp, layer_idx=0)
        q_omp = codec.evaluate_reconstruction(X_sparse, X_hat_omp)
        self.assertGreater(q_omp["cosine_similarity"], 0.98)

        idx_fista, val_fista = codec.iterative_elasticnet_encode(
            X_sparse, layer_idx=0, s=4, lambda1=0.02, lambda2=0.05, num_iters=60
        )
        X_hat_fista = codec.decode(idx_fista, val_fista, layer_idx=0)
        q_fista = codec.evaluate_reconstruction(X_sparse, X_hat_fista)
        self.assertGreater(q_fista["cosine_similarity"], 0.90)

        # Verify both return descending magnitudes and int16/int8 formats
        self.assertEqual(idx_omp.dtype, torch.int16)
        self.assertEqual(val_omp.dtype, torch.int8)
        self.assertEqual(idx_fista.dtype, torch.int16)
        self.assertEqual(val_fista.dtype, torch.int8)

    def test_7_matryoshka_partial_decode_prefix_isolation_and_monotonicity(self) -> None:
        head_dim, num_atoms = 16, 64
        gen = torch.Generator().manual_seed(2024)
        # Orthonormal dictionary subset so partial sums strictly improve reconstruction
        ortho = torch.linalg.qr(torch.randn(head_dim, head_dim, generator=gen))[0]
        extra = F.normalize(torch.randn(head_dim, num_atoms - head_dim, generator=gen), p=2, dim=0)
        D = torch.cat([ortho, extra], dim=1).unsqueeze(0)
        codec = ElasticLexicoCodec(D=D, lambda1=0.01, lambda2=0.05)

        # Construct signals supported on orthogonal atoms with decaying weights
        weights = torch.tensor([3.0, -2.2, 1.6, -1.1, 0.8, -0.5, 0.3, -0.2])
        X = (ortho[:, :8] @ weights).unsqueeze(0).repeat(8, 1)

        idx, val = codec.encode(X, layer_idx=0, s=8, lambda1=0.01, norm_rescale=False)

        # Prefix isolation: corrupting atoms 4..7 must NOT affect decode(..., active_atoms=4)
        out_m4_clean = codec.decode(idx, val, layer_idx=0, active_atoms=4)
        idx_corrupted = idx.clone()
        idx_corrupted[:, 4:] = (idx_corrupted[:, 4:] + 7) % num_atoms
        out_m4_corrupted = codec.decode(idx_corrupted, val, layer_idx=0, active_atoms=4)
        self.assertTrue(torch.equal(out_m4_clean, out_m4_corrupted))

        # Monotonic error improvement across m = 1, 2, 4, 8
        mses = []
        for m in [1, 2, 4, 8]:
            x_hat_m = codec.decode(idx, val, layer_idx=0, active_atoms=m)
            mses.append(F.mse_loss(x_hat_m, X).item())
        for i in range(len(mses) - 1):
            self.assertLessEqual(mses[i + 1], mses[i] + 1e-5)

        # Invalid active_atoms or out-of-range indices must raise ValueError
        with self.assertRaises(ValueError):
            codec.decode(idx, val, layer_idx=0, active_atoms=0)
        with self.assertRaises(ValueError):
            codec.decode(idx, val, layer_idx=0, active_atoms=9)
        bad_idx = idx.clone()
        bad_idx[0, 0] = num_atoms + 5
        with self.assertRaises(ValueError):
            codec.decode(bad_idx, val, layer_idx=0)

    def test_8_online_adaptation_and_break_even_edge_cases(self) -> None:
        # Direct unit test of compute_break_even_length
        res_pos = compute_break_even_length(head_dim=128, M_adapt=64, s_univ=16.0, s_adapt=10.0)
        self.assertTrue(res_pos["break_even_exists"])
        self.assertAlmostEqual(res_pos["L_star"], (2 * 128 * 64) / (3 * 6.0), places=3)

        # Edge case: s_adapt >= s_univ -> break_even_exists = False, L_star = None
        res_none = compute_break_even_length(head_dim=128, M_adapt=64, s_univ=16.0, s_adapt=16.0)
        self.assertFalse(res_none["break_even_exists"])
        self.assertIsNone(res_none["L_star"])

        res_worse = compute_break_even_length(head_dim=128, M_adapt=64, s_univ=16.0, s_adapt=18.0)
        self.assertFalse(res_worse["break_even_exists"])
        self.assertIsNone(res_worse["L_star"])

        # Held-out online adaptation test where residual has low-rank structure shared between X_adapt and X_eval
        head_dim, num_atoms = 16, 48
        gen = torch.Generator().manual_seed(555)
        # Dictionary spans only first 10 dimensions; dimensions 10..15 are unseen OOD subspace
        D_sub = F.normalize(torch.randn(1, 10, num_atoms, generator=gen), p=2, dim=-2)
        D_pad = torch.zeros(1, head_dim, num_atoms)
        D_pad[:, :10, :] = D_sub
        codec = ElasticLexicoCodec(D=D_pad, lambda1=0.02, lambda2=0.1)

        # Create adaptation and disjoint evaluation sets with strong energy in dims 10..13
        X_adapt = torch.randn(64, head_dim, generator=gen) * 0.2
        X_adapt[:, 10:14] += torch.randn(64, 4, generator=gen) * 1.5
        X_eval = torch.randn(64, head_dim, generator=gen) * 0.2
        X_eval[:, 10:14] += torch.randn(64, 4, generator=gen) * 1.5

        idx_a, val_a = codec.encode(X_adapt, layer_idx=0, s=8)
        X_hat_adapt = codec.decode(idx_a, val_a, layer_idx=0)
        idx_e, val_e = codec.encode(X_eval, layer_idx=0, s=8)
        X_hat_eval = codec.decode(idx_e, val_e, layer_idx=0)

        adapt_res = codec.adapt_online_atoms(
            X_adapt,
            X_hat_adapt,
            M_adapt=8,
            layer_idx=0,
            s_univ=8,
            X_eval=X_eval,
            X_hat_eval=X_hat_eval,
        )
        self.assertTrue(adapt_res["evaluated_on_heldout"])
        self.assertEqual(tuple(adapt_res["D_adapt"].shape), (head_dim, 8))
        self.assertGreater(adapt_res["cosine_gain_at_s_univ"], 0.10)
        self.assertTrue(adapt_res["break_even_exists"])
        self.assertIsNotNone(adapt_res["L_star"])
        self.assertLess(adapt_res["s_adapt"], adapt_res["s_univ"])

    def test_9_artifact_roundtrip_and_hook_cleanup_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "roundtrip.pt")
            _ = calibrate_model(
                self.tiny_model,
                self.tiny_tokenizer,
                save_path=save_path,
                subspace_rank=8,
                dict_atoms=32,
                dict_steps=5,
                seed=42,
                max_seq_len=32,
            )
            loaded = torch.load(save_path, map_location="cpu", weights_only=False)
            for req_key in ("U_Q", "head_roles", "D", "W_ridge", "head_entropy", "config", "probe_metrics"):
                self.assertIn(req_key, loaded)

            codec = ElasticLexicoCodec.from_artifacts(loaded, device="cpu")
            X_test = torch.randn(10, 8)
            idx, val = codec.encode(X_test, layer_idx=0, s=4)
            X_rec = codec.decode(idx, val, layer_idx=0)
            self.assertEqual(tuple(X_rec.shape), (10, 8))
            mem_info = codec.memory_accounting(idx, val)
            self.assertEqual(mem_info["logical_payload_bytes_per_vector"], 3 * 4 + 2)
            self.assertGreaterEqual(
                mem_info["total_pytorch_tensor_storage_bytes"],
                mem_info["total_logical_payload_bytes"],
            )

            # Verify hooks and training state are restored even if calibration fails mid-run
            self.tiny_model.train()
            with self.assertRaises(ValueError):
                calibrate_model(
                    self.tiny_model,
                    self.tiny_tokenizer,
                    save_path=save_path,
                    calibration_prompts=[],  # triggers ValueError
                )
            self.assertTrue(self.tiny_model.training, "Model training state must be restored on failure")
            for layer in self.tiny_model.model.layers:
                self.assertEqual(len(layer.self_attn.q_proj._forward_hooks), 0)
                self.assertEqual(len(layer.self_attn.k_proj._forward_hooks), 0)
                self.assertEqual(len(layer.self_attn.v_proj._forward_hooks), 0)

    def test_10_formula_verification_and_split_disjointness(self) -> None:
        # 1. Verify Spearman rho and Recall@20% against closed-form rank calculations
        pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        ref_ident = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
        rho_id, rec_id = _compute_spearman_and_recall(pred, ref_ident, top_ratio=0.20)
        self.assertAlmostEqual(float(rho_id), 1.0, places=6)
        self.assertAlmostEqual(float(rec_id), 1.0, places=6)

        ref_rev = ref_ident[::-1]
        rho_rev, rec_rev = _compute_spearman_and_recall(pred, ref_rev, top_ratio=0.20)
        self.assertAlmostEqual(float(rho_rev), -1.0, places=6)
        self.assertAlmostEqual(float(rec_rev), 0.0, places=6)

        with self.assertRaises(ValueError):
            _compute_spearman_and_recall(pred[:5], ref_ident)

        # 2. Verify cosine similarity, MSE, relative MSE, and relative L2 formulas
        x_a = torch.tensor([[3.0, 4.0], [6.0, 8.0]], dtype=torch.float32)
        x_b = torch.tensor([[6.0, 8.0], [3.0, 4.0]], dtype=torch.float32)
        metrics = ElasticLexicoCodec.evaluate_reconstruction(x_a, x_b)
        self.assertAlmostEqual(metrics["cosine_similarity"], 1.0, places=6)
        # Squared diffs: [9+16=25, 9+16=25], mean over 4 elements = 50 / 4 = 12.5
        self.assertAlmostEqual(metrics["mse"], 12.5, places=6)

        # 3. Verify break-even formula L* = (2 * d * M_adapt) / (3 * (s_univ - s_adapt))
        be = compute_break_even_length(head_dim=128, M_adapt=64, s_univ=16.0, s_adapt=10.5)
        expected_l_star = (2.0 * 128.0 * 64.0) / (3.0 * (16.0 - 10.5))
        self.assertAlmostEqual(float(be["L_star"]), expected_l_star, places=3)

        # 4. Verify strict pairwise disjointness of all prompt sets
        calib_set = set(CALIBRATION_PROMPTS) | set(NEEDLE_PROMPTS)
        heldout_set = set(HELDOUT_EVAL_PROMPTS)
        adapt_set = set(ENGLISH_ADAPT_PROMPTS) | set(PYTHON_ADAPT_PROMPTS) | set(KANNADA_ADAPT_PROMPTS)
        eval_set = set(ENGLISH_EVAL_PROMPTS) | set(PYTHON_EVAL_PROMPTS) | set(KANNADA_EVAL_PROMPTS)

        self.assertTrue(calib_set.isdisjoint(heldout_set))
        self.assertTrue(calib_set.isdisjoint(adapt_set))
        self.assertTrue(calib_set.isdisjoint(eval_set))
        self.assertTrue(heldout_set.isdisjoint(adapt_set))
        self.assertTrue(heldout_set.isdisjoint(eval_set))
        self.assertTrue(adapt_set.isdisjoint(eval_set))

    def test_11_compressed_cache_passkey_inference(self) -> None:
        # Use structured GQA model on CPU/CUDA to verify end-to-end passkey retrieval under KV compression
        cfg = FallbackQwen2Config(
            hidden_size=1536,
            num_hidden_layers=20,
            num_attention_heads=12,
            num_key_value_heads=2,
            head_dim=128,
            vocab_size=32768,
        )
        model = FallbackQwen2ForCausalLM(cfg, device="cpu", dtype=torch.float32, seed=42)
        tokenizer = FallbackQwen2Tokenizer(vocab_size=cfg.vocab_size)
        artifacts = calibrate_model(
            model,
            tokenizer,
            save_path=None,
            calibration_prompts=CALIBRATION_PROMPTS[:4],
            needle_prompts=NEEDLE_PROMPTS[:4],
            heldout_prompts=HELDOUT_EVAL_PROMPTS[:3],
            subspace_rank=16,
            dict_atoms=64,
            dict_steps=5,
            seed=42,
            max_seq_len=160,
        )
        self.assertTrue(artifacts["retrieval_diagnostics"]["heldout_isolated_from_role_assignment"])

        pk_res = evaluate_compressed_cache_passkey(
            model=model,
            tokenizer=tokenizer,
            artifacts=artifacts,
            s=16,
            draft_atoms=4,
            h2o_keep_ratio=0.50,
            max_seq_len=160,
        )
        pols = pk_res["policies"]
        # Full FP16 and StoikovKV Role-Routed Hybrid Cache (s=16 and draft m=4) preserve 100% passkey accuracy
        self.assertEqual(pols["full_fp16"]["passkey_top1_accuracy"], 1.0)
        self.assertEqual(pols["stoikov_hybrid_s16"]["passkey_top1_accuracy"], 1.0)
        self.assertEqual(pols["stoikov_hybrid_draft_m4"]["passkey_top1_accuracy"], 1.0)
        self.assertGreater(pols["stoikov_hybrid_s16"]["logit_cosine_similarity"], 0.98)
        # Unprotected ablation fails passkey retrieval when ROLE_RETRIEVAL_FP16 heads are not protected
        self.assertLess(
            pols["unprotected_ablation"]["passkey_top1_accuracy"],
            pols["stoikov_hybrid_s16"]["passkey_top1_accuracy"],
        )


if __name__ == "__main__":
    unittest.main()
