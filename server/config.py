from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
import yaml
import os

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "config")

class ModelConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")
    name: str = "Qwen/Qwen2.5-3B-Instruct"
    dtype: str = "float16"
    device: str = "cuda"
    attn_implementation: str = "sdpa"
    max_seq_len: int = 32768
    trust_remote_code: bool = True
    fallback_to_mock: bool = True

class CacheConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")
    n_b: int = 64
    keep_ratio: float = 0.15
    s_budget: int = 16
    policy_mode: str = "hybrid"
    enable_hmm_decay: bool = True
    shift_z_threshold: float = 1.4
    beta_stationary: float = 0.99
    beta_shift: float = 0.65
    ema_alpha: float = 0.15
    max_rollback_window: int = 32
    sink_tokens: int = 4
    sink_bonus: float = 100.0
    lambda1: float = 0.04

class ServerConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    cors_origins: List[str] = ["http://localhost:3000"]
    request_timeout: int = 300

def load_yaml_config(path: str, cls):
    with open(path) as f:
        data = yaml.safe_load(f)
    key = cls.__name__.replace("Config", "").lower()
    return cls(**(data.get(key, {})))

model_config = load_yaml_config(os.path.join(CONFIG_DIR, "model.yaml"), ModelConfig)
cache_config = load_yaml_config(os.path.join(CONFIG_DIR, "cache.yaml"), CacheConfig)
server_config = load_yaml_config(os.path.join(CONFIG_DIR, "server.yaml"), ServerConfig)
