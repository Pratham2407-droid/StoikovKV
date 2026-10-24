"""
Unit tests for StoikovKV Person 2 Live Inference Engine (`stoikov_kv/cache.py` & `stoikov_kv/matryoshka_spec.py`).
"""

from __future__ import annotations

import os
import unittest

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


if __name__ == "__main__":
    unittest.main()
