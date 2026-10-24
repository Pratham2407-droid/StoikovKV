"""StoikovKV: FlashAttention-Compatible Hybrid Evict-or-Encode KV Cache."""

from .calibration import (
    ROLE_RETRIEVAL_FP16,
    ROLE_PEAKED_H2O,
    ROLE_DIFFUSE_LEXICO,
    CALIBRATION_PROMPTS,
    NEEDLE_PROMPTS,
    HELDOUT_EVAL_PROMPTS,
    calibrate_model,
    score_write_time,
    score_tokens_write_time,
    reduce_attention_to_kv_heads,
    evaluate_write_time_probe,
    evaluate_compressed_cache_passkey,
    load_model_and_tokenizer,
)
from .elastic_lexico import (
    ElasticLexicoCodec,
    QuantizedInt8Values,
    compute_break_even_length,
)
from .cache import (
    DynamicCache,
    StoikovKVCache,
    HMMTopicShiftController,
    create_mock_artifacts,
    load_or_create_artifacts,
)
from .matryoshka_spec import (
    MatryoshkaSpecResult,
    matryoshka_speculative_generate,
)

__all__ = [
    "ROLE_RETRIEVAL_FP16",
    "ROLE_PEAKED_H2O",
    "ROLE_DIFFUSE_LEXICO",
    "CALIBRATION_PROMPTS",
    "NEEDLE_PROMPTS",
    "HELDOUT_EVAL_PROMPTS",
    "calibrate_model",
    "score_write_time",
    "score_tokens_write_time",
    "reduce_attention_to_kv_heads",
    "evaluate_write_time_probe",
    "evaluate_compressed_cache_passkey",
    "load_model_and_tokenizer",
    "ElasticLexicoCodec",
    "QuantizedInt8Values",
    "compute_break_even_length",
    "DynamicCache",
    "StoikovKVCache",
    "HMMTopicShiftController",
    "create_mock_artifacts",
    "load_or_create_artifacts",
    "MatryoshkaSpecResult",
    "matryoshka_speculative_generate",
]
