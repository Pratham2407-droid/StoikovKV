"""
Unit tests for StoikovKV Person 2 Live Inference Engine (`stoikov_kv/cache.py` & `stoikov_kv/matryoshka_spec.py`).
"""

from __future__ import annotations

import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from stoikov_kv.cache import (
    DynamicCache,
    HMMTopicShiftController,
    StoikovKVCache,
    create_mock_artifacts,
    load_or_create_artifacts,
)
from stoikov_kv.calibration import (
    ROLE_DIFFUSE_LEXICO,
    ROLE_PEAKED_H2O,
    ROLE_RETRIEVAL_FP16,
    FallbackQwen2Config,
    FallbackQwen2ForCausalLM,
    FallbackQwen2Tokenizer,
)
from stoikov_kv.matryoshka_spec import matryoshka_speculative_generate


class TestPerson2LiveInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.dtype = torch.float16 if cls.device == "cuda" else torch.float32
        cls.config = FallbackQwen2Config(
            hidden_size=96,
            num_hidden_layers=4,
            num_attention_heads=6,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=512,
        )
        cls.model = FallbackQwen2ForCausalLM(
            cls.config, device=cls.device, dtype=cls.dtype, seed=42
        )
        cls.tokenizer = FallbackQwen2Tokenizer(vocab_size=512)
        cls.artifacts = create_mock_artifacts(
            cls.model, tokenizer=cls.tokenizer, subspace_rank=8, dict_atoms=64
        )

    def test_01_create_mock_artifacts_standalone(self) -> None:
        """Verifies create_mock_artifacts(model) produces valid shapes without output_attentions."""
        art = self.artifacts
        self.assertEqual(tuple(art["U_Q"].shape), (4, 2, 16, 8))
        self.assertEqual(tuple(art["head_roles"].shape), (4, 2))
        self.assertEqual(tuple(art["D"].shape), (4, 16, 64))
        self.assertEqual(tuple(art["W_ridge"].shape), (4, 64, 16))
        self.assertTrue(art["is_mock"])

    def test_02_stoikov_cache_routing_and_telemetry(self) -> None:
        """Verifies StoikovKVCache routes tokens older than n_b across roles and tracks exact bytes."""
        cache = StoikovKVCache(
            model=self.model,
            artifacts=self.artifacts,
            n_b=16,
            keep_ratio=0.25,
            s_budget=8,
            policy_mode="hybrid",
        )
        self.assertIsInstance(cache, DynamicCache)

        input_ids = torch.randint(4, 250, (1, 48), device=self.device)
        out = self.model(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
            output_attentions=False,
        )
        self.assertIsNone(out.attentions)
        self.assertEqual(cache.get_seq_length(0), 48)
        self.assertGreater(cache.fp16_bytes, 0)
        self.assertGreater(cache.compressed_bytes, 0)
        self.assertLess(cache.compressed_bytes, cache.fp16_bytes)
        self.assertGreater(cache.bytes_read_last_step, 0)

        counts = cache.per_tier_token_counts
        # 4 layers * 2 kv_heads * 16 local window = 128 local window token-head slots
        self.assertEqual(counts["local_window_fp16"], 4 * 2 * 16)
        self.assertGreater(
            counts["retrieval_fp16"] + counts["peaked_h2o_fp16"] + counts["diffuse_lexico_sparse"],
            0,
        )

    def test_03_matryoshka_mode_and_crop_rollback(self) -> None:
        """Verifies set_matryoshka_mode(active_atoms=m) and crop(max_length) rollback."""
        cache = StoikovKVCache(
            model=self.model,
            artifacts=self.artifacts,
            n_b=16,
            keep_ratio=0.25,
            s_budget=8,
            policy_mode="standalone_lexico",
        )
        input_ids = torch.randint(4, 250, (1, 40), device=self.device)
        _ = self.model(input_ids=input_ids, past_key_values=cache, use_cache=True, output_attentions=False)

        cache.set_matryoshka_mode(active_atoms=2)
        self.assertEqual(cache.active_atoms, 2)
        step_tok = torch.randint(4, 250, (1, 1), device=self.device)
        _ = self.model(input_ids=step_tok, past_key_values=cache, use_cache=True, output_attentions=False)
        bytes_m2 = cache.bytes_read_last_step

        cache.crop(40)
        self.assertEqual(cache.get_seq_length(0), 40)

        cache.set_matryoshka_mode(active_atoms=8)
        _ = self.model(input_ids=step_tok, past_key_values=cache, use_cache=True, output_attentions=False)
        bytes_m8 = cache.bytes_read_last_step

        self.assertLess(bytes_m2, bytes_m8)

    def test_04_matryoshka_speculative_generate(self) -> None:
        """Verifies matryoshka_speculative_generate end-to-end."""
        cache = StoikovKVCache(
            model=self.model,
            artifacts=self.artifacts,
            n_b=16,
            keep_ratio=0.25,
            s_budget=8,
            policy_mode="hybrid",
        )
        prompt_ids = torch.randint(4, 250, (1, 36), device=self.device)
        res = matryoshka_speculative_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt_ids,
            cache=cache,
            gamma_spec=4,
            draft_atoms=2,
            verify_atoms=8,
            max_new_tokens=12,
        )
        self.assertEqual(len(res["generated_ids"]), 12)
        self.assertGreaterEqual(res["acceptance_rate"], 0.0)
        self.assertLessEqual(res["acceptance_rate"], 1.0)
        self.assertGreater(res["kv_bytes_read_per_token"], 0.0)
        self.assertEqual(cache.get_seq_length(0), 36 + 12)

    def test_05_h2o_context_window_and_legacy_rope_frame(self) -> None:
        """H2O accepts contextual retention and legacy post-RoPE artifacts stay unrotated."""
        artifacts = dict(self.artifacts)
        artifacts["dict_key_frame"] = "post_rope"
        cache = StoikovKVCache(
            model=self.model,
            artifacts=artifacts,
            n_b=2,
            keep_ratio=0.25,
            policy_mode="standalone_h2o",
            h2o_context_window=1,
        )
        states = torch.randn(1, 2, 3, 16, device=self.device, dtype=self.dtype)
        self.assertTrue(
            torch.equal(cache._rope_rotate(states, torch.arange(3, device=self.device), inverse=True), states)
        )

        input_ids = torch.randint(4, 250, (1, 24), device=self.device)
        _ = self.model(input_ids=input_ids, past_key_values=cache, use_cache=True, output_attentions=False)
        for tier in cache._tiers[0]:
            self.assertIsNotNone(tier.h2o_indices)
            self.assertLessEqual(int(tier.h2o_k.shape[-2]), max(1, int(24 * 0.25)))

    def test_06_p0_physical_memory_compaction_and_no_duplicates(self) -> None:
        """Verifies P0 physical memory compaction: no duplicate key_cache/layers/lex_recon and compact H2O decode."""
        cache = StoikovKVCache(
            model=self.model,
            artifacts=self.artifacts,
            n_b=8,
            keep_ratio=0.25,
            s_budget=4,
            policy_mode="hybrid",
        )
        input_ids = torch.randint(4, 250, (1, 64), device=self.device)
        _ = self.model(input_ids=input_ids, past_key_values=cache, use_cache=True, output_attentions=False)

        # Duplicate caches must be completely empty
        self.assertEqual(len(cache.key_cache), 0)
        self.assertEqual(len(cache.value_cache), 0)
        self.assertEqual(len(getattr(cache, "layers", [])), 0)

        # Lexico heads must not retain uncompressed FP16 reconstructions by default
        for l_idx in range(cache.num_layers):
            roles_l = cache.head_roles[l_idx]
            has_h2o = bool((roles_l == ROLE_PEAKED_H2O).any().item())
            all_h2o = bool((roles_l == ROLE_PEAKED_H2O).all().item())
            # Every layer containing an H2O head must be 100% pure H2O so anti_k is never needed
            self.assertEqual(has_h2o, all_h2o)
            for h_idx in range(cache.num_kv_heads):
                st = cache._tiers[l_idx][h_idx]
                if st.role == ROLE_DIFFUSE_LEXICO:
                    self.assertIsNone(st.lex_recon_k)
                    self.assertIsNone(st.lex_recon_v)

        # Physical resident bytes must equal logical compressed bytes and be strictly below FP16
        self.assertEqual(cache.physical_resident_bytes, cache.compressed_bytes)
        self.assertLess(cache.physical_resident_bytes, cache.fp16_bytes)
        self.assertLess(cache.raw_tensor_bytes, cache.raw_fp16_tensor_bytes)

        # Standalone H2O layer materializes compact [B, H_kv, k_kept + win_len, D] during decode
        h2o_cache = StoikovKVCache(
            model=self.model,
            artifacts=self.artifacts,
            n_b=8,
            keep_ratio=0.25,
            s_budget=4,
            policy_mode="standalone_h2o",
        )
        _ = self.model(input_ids=input_ids, past_key_values=h2o_cache, use_cache=True, output_attentions=False)
        k_compact, v_compact, _ = h2o_cache._materialize_layer_kv(
            0, dtype=self.dtype, device=torch.device(self.device), q_len=1
        )
        self.assertLess(int(k_compact.shape[-2]), h2o_cache.get_seq_length(0))
        self.assertEqual(int(k_compact.shape[-2]), int(v_compact.shape[-2]))

    def test_07_p1_native_generate_drop_in_compatibility(self) -> None:
        """Verifies P1 native model.generate(past_key_values=cache) and HF Cache API."""
        cache = StoikovKVCache(
            model=self.model,
            artifacts=self.artifacts,
            n_b=8,
            keep_ratio=0.25,
            s_budget=4,
            policy_mode="hybrid",
        )
        self.assertFalse(cache.is_initialized)
        self.assertEqual(len(cache), 0)

        input_ids = torch.randint(4, 250, (1, 24), device=self.device)
        out_ids = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=6,
            do_sample=False,
            past_key_values=cache,
            use_cache=True,
        )
        self.assertEqual(tuple(out_ids.shape), (1, 30))
        self.assertTrue(cache.is_initialized)
        self.assertEqual(len(cache), self.config.num_hidden_layers)
        self.assertEqual(cache.batch_size, 1)
        self.assertEqual(cache.get_seq_length(0), 29)
        self.assertEqual(cache.get_mask_sizes(1, 0), (30, 0))

        cache.reset()
        self.assertFalse(cache.is_initialized)
        self.assertEqual(cache.get_seq_length(0), 0)

    def test_08_p3_single_pass_verification_and_strict_sampling(self) -> None:
        """Verifies P3 single-pass cur_gamma+1 verification with bonus token and strict rejection sampling."""
        cache = StoikovKVCache(
            model=self.model,
            artifacts=self.artifacts,
            n_b=16,
            keep_ratio=0.25,
            s_budget=8,
            policy_mode="hybrid",
        )
        prompt_ids = torch.randint(4, 250, (1, 32), device=self.device)
        res = matryoshka_speculative_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt_ids,
            cache=cache,
            gamma_spec=4,
            draft_atoms=4,
            verify_atoms=8,
            max_new_tokens=10,
            strict_rejection_sampling=True,
            do_sample=False,
        )
        self.assertEqual(len(res["generated_ids"]), 10)
        self.assertEqual(cache.get_seq_length(0), 32 + 10)

        res_stoch = matryoshka_speculative_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt_ids,
            cache=cache,
            gamma_spec=4,
            draft_atoms=4,
            verify_atoms=8,
            max_new_tokens=10,
            strict_rejection_sampling=True,
            do_sample=True,
            temperature=0.7,
        )
        self.assertEqual(len(res_stoch["generated_ids"]), 10)
        self.assertEqual(cache.get_seq_length(0), 32 + 10)
        self.assertIn("analytical_subvector_kv_bytes_per_token", res_stoch)
        self.assertIn("measured_py_decode_bytes_per_token", res_stoch)

    def test_09_p4_openai_server_and_cli(self) -> None:
        """Verifies P4 StoikovInferenceEngine OpenAI routes and CLI entrypoint."""
        from stoikov_kv.cli import main as cli_main
        from stoikov_kv.server import StoikovInferenceEngine

        engine = StoikovInferenceEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            artifacts=self.artifacts,
            model_id="test-stoikov-model",
            n_b=8,
            keep_ratio=0.25,
            s_budget=4,
            draft_atoms=2,
            gamma_spec=3,
        )

        st_health, health = engine.handle_request("GET", "/health")
        self.assertEqual(st_health, 200)
        self.assertEqual(health["status"], "ok")

        st_models, models = engine.handle_request("GET", "/v1/models")
        self.assertEqual(st_models, 200)
        self.assertEqual(models["data"][0]["id"], "test-stoikov-model")

        st_chat, chat = engine.handle_request(
            "POST",
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "Hello world test prompt"}],
                "max_tokens": 6,
                "speculative": True,
            },
        )
        self.assertEqual(st_chat, 200)
        self.assertEqual(chat["object"], "chat.completion")
        self.assertIn("stoikov_telemetry", chat)
        self.assertLess(
            chat["stoikov_telemetry"]["physical_resident_bytes"],
            chat["stoikov_telemetry"]["fp16_bytes"] + 1,
        )

        st_tel, tel = engine.handle_request("GET", "/v1/telemetry")
        self.assertEqual(st_tel, 200)
        self.assertIn("physical_resident_bytes", tel["data"])


if __name__ == "__main__":
    unittest.main()
