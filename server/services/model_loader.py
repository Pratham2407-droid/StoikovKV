import sys
from pathlib import Path

import torch

# Resolve ENGINE_ROOT relative to this file whether running inside repo root or as submodule
_repo_root = Path(__file__).resolve().parents[2]
if (_repo_root / "stoikov_kv").is_dir():
    ENGINE_ROOT = _repo_root
else:
    ENGINE_ROOT = Path(__file__).resolve().parents[3] / "HashCode"

if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from stoikov_kv.cache import StoikovKVCache, load_or_create_artifacts
from stoikov_kv.calibration import load_model_and_tokenizer as load_engine_model
from server.config import model_config, cache_config

_model = None
_tokenizer = None
_artifacts = None

def load_model_and_tokenizer():
    global _model, _tokenizer, _artifacts
    
    if _model is not None:
        return _model, _tokenizer, _artifacts
    
    device = model_config.device
    dtype = torch.float16 if model_config.dtype == "float16" else torch.bfloat16
    
    # The tested fallback is the safe default.  A real checkpoint is opt-in
    # because the long-context cache benchmark requires more VRAM than many
    # dashboard machines provide.
    model, tokenizer = load_engine_model(
        model_name=model_config.name,
        device=device,
        dtype=dtype,
        use_real_model=False,
    )
    
    artifacts_path = ENGINE_ROOT / "calibrated_artifacts.pt"
    _artifacts = load_or_create_artifacts(model, artifacts_path=str(artifacts_path), tokenizer=tokenizer)
    
    _model = model
    _tokenizer = tokenizer
    return model, tokenizer, _artifacts

def create_cache_factory():
    model, tokenizer, artifacts = load_model_and_tokenizer()
    
    def make_cache(session_id: str = "default"):
        return StoikovKVCache(
            model=model,
            artifacts=artifacts,
            n_b=cache_config.n_b,
            keep_ratio=cache_config.keep_ratio,
            s_budget=cache_config.s_budget,
            policy_mode=cache_config.policy_mode,
            enable_hmm_decay=cache_config.enable_hmm_decay,
        )
    
    return make_cache, model, tokenizer, artifacts
