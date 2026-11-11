#!/usr/bin/env python3
"""
StoikovKV Comparative Audit Benchmark Runner
=============================================
Executes rigorous, reproducible benchmarks comparing StoikovKV against
industry-standard KV-cache management and compression approaches.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure repo root is on sys.path (auto-detect from script location)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Add src/ directory for stoikov_kv package
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Add benchmarks path for imports
BENCHMARKS_PATH = REPO_ROOT / "benchmarks"
if str(BENCHMARKS_PATH) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_PATH))

print(f"REPO_ROOT: {REPO_ROOT}")
print(f"sys.path: {sys.path[:5]}")

from stoikov_kv.calibration import (
    load_model_and_tokenizer,
    calibrate_model,
    evaluate_compressed_cache_passkey,
    FallbackQwen2Config,
    FallbackQwen2ForCausalLM,
    FallbackQwen2Tokenizer,
    ROLE_RETRIEVAL_FP16,
    ROLE_PEAKED_H2O,
    ROLE_DIFFUSE_LEXICO,
)
from stoikov_kv.elastic_lexico import ElasticLexicoCodec
from stoikov_kv.cache import StoikovKVCache, HMMTopicShiftController
from stoikov_kv.matryoshka_spec import matryoshka_speculative_generate


# ============================================================================
# Environment Detection
# ============================================================================

def detect_environment() -> Dict[str, Any]:
    """Detect and record hardware and software environment."""
    env = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        env["gpu_name"] = torch.cuda.get_device_name(0)
        env["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        env["gpu_compute_capability"] = f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}"
    
    # Try to get transformers version
    try:
        import transformers
        env["transformers_version"] = transformers.__version__
    except ImportError:
        env["transformers_version"] = "not_installed"
    
    # Git info
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=5
        )
        env["git_commit"] = result.stdout.strip() if result.returncode == 0 else "unknown"
        
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=5
        )
        env["git_dirty"] = len(result.stdout.strip()) > 0
    except Exception:
        env["git_commit"] = "unknown"
        env["git_dirty"] = False
    
    return env


# ============================================================================
# Benchmark Infrastructure
# ============================================================================

class BenchmarkResult:
    """Container for a single benchmark result."""
    
    def __init__(
        self,
        benchmark_name: str,
        method_name: str,
        config: Dict[str, Any],
        metrics: Dict[str, Any],
        success: bool = True,
        error: Optional[str] = None,
    ):
        self.benchmark_name = benchmark_name
        self.method_name = method_name
        self.config = config
        self.metrics = metrics
        self.success = success
        self.error = error
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "method_name": self.method_name,
            "config": self.config,
            "metrics": self.metrics,
            "success": self.success,
            "error": self.error,
            "timestamp": self.timestamp,
        }


class BenchmarkRunner:
    """Main benchmark runner orchestrating all experiments."""
    
    def __init__(
        self,
        config_path: str,
        output_dir: str,
        model_track: str = "fallback",
        verbose: bool = True,
    ):
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.model_track = model_track
        self.verbose = verbose
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "logs").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "results").mkdir(parents=True, exist_ok=True)
        
        self.results: List[BenchmarkResult] = []
        self.environment = detect_environment()
        
        # Load benchmark config
        import yaml
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.tokenizer = None
        self.artifacts = None
        self.codec = None
        
    def log(self, msg: str, level: str = "INFO"):
        if self.verbose:
            prefix = f"[{level}] " if level != "INFO" else ""
            print(f"{prefix}{msg}")
    
    def save_environment(self):
        """Save environment info to JSON."""
        env_path = self.output_dir / "environment.json"
        with open(env_path, "w") as f:
            json.dump(self.environment, f, indent=2, default=str)
        self.log(f"Environment saved to {env_path}")
    
    def save_results(self):
        """Save all results to JSONL and CSV."""
        # JSONL
        jsonl_path = self.output_dir / "results" / "raw_results.jsonl"
        with open(jsonl_path, "w") as f:
            for r in self.results:
                f.write(json.dumps(r.to_dict(), default=str) + "\n")
        
        # CSV (flattened)
        import csv
        csv_path = self.output_dir / "results" / "aggregate_results.csv"
        if self.results:
            # Collect all possible fieldnames first
            fieldnames = set()
            for r in self.results:
                d = r.to_dict()
                fieldnames.update(d.keys())
                fieldnames.update({f"config_{k}": v for k, v in r.config.items()})
                fieldnames.update({f"metric_{k}": v for k, v in r.metrics.items()})
            
            fieldnames = sorted(fieldnames)
            
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                for r in self.results:
                    row = r.to_dict()
                    row.update({f"config_{k}": v for k, v in r.config.items()})
                    row.update({f"metric_{k}": v for k, v in r.metrics.items()})
                    writer.writerow(row)
        
        self.log(f"Results saved to {jsonl_path} and {csv_path}")
    
    def add_result(self, result: BenchmarkResult):
        """Add a result and save incrementally."""
        self.results.append(result)
        self.save_results()
    
    # -------------------------------------------------------------------------
    # Model Setup
    # -------------------------------------------------------------------------
    
    def setup_model(self):
        """Load model, tokenizer, and calibration artifacts."""
        self.log(f"Setting up model for track: {self.model_track}")
        
        if self.model_track == "fallback":
            model_config = self.config["model"]["fallback"]
            device = model_config["device"]
            dtype = getattr(torch, model_config["dtype"])
            
            # Use fallback model
            cfg = FallbackQwen2Config(**model_config["config"])
            self.model = FallbackQwen2ForCausalLM(cfg, device=device, dtype=dtype, seed=42)
            self.tokenizer = FallbackQwen2Tokenizer(vocab_size=cfg.vocab_size)
            self.model.eval()
            
        elif self.model_track == "real_7b":
            model_config = self.config["model"]["real_7b"]
            device = model_config["device"]
            dtype = getattr(torch, model_config["dtype"])
            
            self.model, self.tokenizer = load_model_and_tokenizer(
                model_name=model_config["name"],
                device=device,
                dtype=dtype,
                use_real_model=True,
                load_in_4bit=model_config["quantization"]["enabled"],
                bnb_4bit_compute_dtype=getattr(torch, model_config["quantization"]["compute_dtype"]),
            )
        else:
            raise ValueError(f"Unknown model track: {self.model_track}")
        
        self.log(f"Model loaded: {self.model.__class__.__name__} on {next(self.model.parameters()).device}")
        
        # Load or create calibration artifacts
        artifacts_path = REPO_ROOT / "artifacts" / "calibrated_artifacts.pt"
        if artifacts_path.exists():
            self.log(f"Loading calibration artifacts from {artifacts_path}")
            from stoikov_kv.cache import load_or_create_artifacts
            self.artifacts = load_or_create_artifacts(
                self.model, artifacts_path=str(artifacts_path), tokenizer=self.tokenizer
            )
        else:
            self.log("Running calibration...")
            self.artifacts = calibrate_model(
                self.model, self.tokenizer,
                save_path=str(artifacts_path),
                seed=42, dict_atoms=128, dict_steps=10, max_seq_len=256, subspace_rank=16
            )
        
        self.codec = ElasticLexicoCodec.from_artifacts(self.artifacts)
        self.log("Calibration artifacts loaded/created")
    
    # -------------------------------------------------------------------------
    # Baseline Implementations
    # -------------------------------------------------------------------------
    
    def create_cache(self, baseline_config: Dict[str, Any]) -> StoikovKVCache:
        """Create a StoikovKVCache with the given configuration."""
        return StoikovKVCache(
            model=self.model,
            artifacts=self.artifacts,
            n_b=baseline_config.get("n_b", 64),
            keep_ratio=baseline_config.get("keep_ratio", 0.20),
            s_budget=baseline_config.get("s_budget", 16),
            lambda1=baseline_config.get("lambda1", 0.04),
            sink_tokens=baseline_config.get("sink_tokens", 4),
            sink_bonus=baseline_config.get("sink_bonus", 100.0),
            policy_mode=baseline_config.get("policy_mode", "hybrid"),
            enable_hmm_decay=baseline_config.get("enable_hmm_decay", True),
            max_rollback_window=baseline_config.get("max_rollback_window", 32),
            h2o_context_window=baseline_config.get("h2o_context_window", 1),
        )
    
    def run_inference(
        self,
        cache: StoikovKVCache,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Run inference and return generated tokens and telemetry."""
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        
        # Prefill
        chunk_size = 256
        seq_len = input_ids.shape[1]
        
        for pos in range(0, seq_len - 1, chunk_size):
            end_pos = min(seq_len - 1, pos + chunk_size)
            _ = self.model(
                input_ids=input_ids[:, pos:end_pos].to(device),
                past_key_values=cache,
                use_cache=True,
                output_attentions=False,
            )
        
        # Generate
        generated = []
        logits_list = []
        
        # First decode step
        out = self.model(
            input_ids=input_ids[:, -1:].to(device),
            past_key_values=cache,
            use_cache=True,
            output_attentions=False,
        )
        
        for _ in range(max_new_tokens):
            logits = out.logits[:, -1, :] / max(temperature, 1e-8)
            logits_list.append(logits.detach().float().cpu())
            
            if do_sample:
                # Top-k
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1:]
                    logits[indices_to_remove] = float("-inf")
                
                # Top-p
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float("-inf")
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            generated.append(next_token.item())
            next_token = next_token.to(device)
            
            if next_token.item() == getattr(self.tokenizer, "eos_token_id", 1):
                break
            
            out = self.model(
                input_ids=next_token,
                past_key_values=cache,
                use_cache=True,
                output_attentions=False,
            )
        
        telemetry = cache.get_telemetry_summary()
        
        return torch.tensor(generated, device=device), telemetry
    
    def measure_latency(
        self,
        cache: StoikovKVCache,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        warmup: int = 3,
        repetitions: int = 10,
    ) -> Dict[str, float]:
        """Measure inference latency with proper warmup and synchronization."""
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        
        def _run():
            # Prefill
            chunk_size = 256
            for pos in range(0, input_ids.shape[1] - 1, chunk_size):
                end_pos = min(input_ids.shape[1] - 1, pos + chunk_size)
                _ = self.model(
                    input_ids=input_ids[:, pos:end_pos].to(device),
                    past_key_values=cache,
                    use_cache=True,
                    output_attentions=False,
                )
            
            # Generate
            gen_ids = input_ids.clone()
            for _ in range(max_new_tokens):
                out = self.model(
                    input_ids=gen_ids[:, -1:].to(device),
                    past_key_values=cache,
                    use_cache=True,
                    output_attentions=False,
                )
                next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True).to(device)
                gen_ids = torch.cat([gen_ids, next_token], dim=-1)
        
        # Warmup
        for _ in range(warmup):
            cache.crop(0)
            _run()
        
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        
        # Timed runs
        latencies = []
        for _ in range(repetitions):
            cache.crop(0)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _run()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latencies.append(time.perf_counter() - t0)
        
        return {
            "mean_latency_sec": float(np.mean(latencies)),
            "median_latency_sec": float(np.median(latencies)),
            "std_latency_sec": float(np.std(latencies)),
            "min_latency_sec": float(np.min(latencies)),
            "max_latency_sec": float(np.max(latencies)),
            "tokens_per_sec": float(max_new_tokens / np.mean(latencies)),
            "prefill_tokens": int(input_ids.shape[1]),
            "gen_tokens": max_new_tokens,
        }
    
    def get_memory_usage(self, cache: StoikovKVCache) -> Dict[str, Any]:
        """Get detailed memory usage from cache telemetry."""
        telemetry = cache.get_telemetry_summary()
        device = next(self.model.parameters()).device
        
        if device.type == "cuda":
            gpu_allocated = torch.cuda.memory_allocated(device) / (1024**2)  # MB
            gpu_reserved = torch.cuda.memory_reserved(device) / (1024**2)
            gpu_max_allocated = torch.cuda.max_memory_allocated(device) / (1024**2)
        else:
            gpu_allocated = gpu_reserved = gpu_max_allocated = 0
        
        return {
            "gpu_allocated_mb": gpu_allocated,
            "gpu_reserved_mb": gpu_reserved,
            "gpu_max_allocated_mb": gpu_max_allocated,
            "cache_fp16_bytes": telemetry.get("fp16_bytes", 0),
            "cache_compressed_bytes": telemetry.get("compressed_bytes", 0),
            "cache_memory_ratio_vs_fp16": telemetry.get("memory_ratio_vs_fp16", 1.0),
            "cache_memory_ratio_incl_dict": telemetry.get("memory_ratio_incl_dictionary", 1.0),
            "dictionary_bytes": telemetry.get("dictionary_bytes", 0),
            "bytes_read_last_step": telemetry.get("bytes_read_last_step", 0),
            "cumulative_bytes_read": telemetry.get("cumulative_bytes_read", 0),
            "per_tier_token_counts": telemetry.get("per_tier_token_counts", {}),
        }
    
    # -------------------------------------------------------------------------
    # Benchmark 1: Needle-in-Haystack Retrieval
    # -------------------------------------------------------------------------
    
    def run_needle_retrieval(self, baseline_name: str, baseline_config: Dict[str, Any]) -> List[BenchmarkResult]:
        """Run needle-in-haystack retrieval benchmark."""
        results = []
        
        needle_config = self.config["datasets"]["needle_retrieval"]
        needle_text = needle_config["needle_text"]
        distractor_numbers = needle_config["distractor_numbers"]
        positions = needle_config["positions"]
        context_lengths = needle_config["context_lengths"]
        num_trials = needle_config["num_trials"]
        
        port_tid = getattr(self.tokenizer, "port_token_id", 8443)
        
        for ctx_len in context_lengths:
            for position in positions:
                for trial in range(num_trials):
                    # Build prompt with needle at specified position
                    prompt, prompt_ids, meta = self._build_needle_prompt(
                        needle_text, distractor_numbers, ctx_len, position
                    )
                    
                    # Run with each seed
                    for seed in self.config["benchmark"]["seeds"]:
                        torch.manual_seed(seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(seed)
                        
                        cache = self.create_cache(baseline_config)
                        
                        # Run inference
                        try:
                            generated_ids, telemetry = self.run_inference(
                                cache, prompt_ids, max_new_tokens=16, do_sample=False
                            )
                            
                            # Check if needle was retrieved
                            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                            retrieved = self._check_needle_retrieval(generated_text, needle_text, distractor_numbers)
                            
                            # Get target token probability
                            cache_for_prob = self.create_cache(baseline_config)
                            _ = self.run_inference(cache_for_prob, prompt_ids, max_new_tokens=1)
                            
                            metrics = {
                                "context_length": ctx_len,
                                "needle_position": position,
                                "trial": trial,
                                "seed": seed,
                                "retrieved": retrieved,
                                "generated_text": generated_text[:200],
                                "num_generated_tokens": len(generated_ids),
                                **telemetry,
                            }
                            
                            results.append(BenchmarkResult(
                                benchmark_name="needle_retrieval",
                                method_name=baseline_name,
                                config=baseline_config,
                                metrics=metrics,
                            ))
                            
                        except Exception as e:
                            self.log(f"Error in needle retrieval: {e}", "ERROR")
                            results.append(BenchmarkResult(
                                benchmark_name="needle_retrieval",
                                method_name=baseline_name,
                                config=baseline_config,
                                metrics={"context_length": ctx_len, "needle_position": position, "trial": trial},
                                success=False,
                                error=str(e),
                            ))
        
        return results
    
    def _build_needle_prompt(
        self, needle_text: str, distractors: List[str], ctx_len: int, position: str
    ) -> Tuple[str, torch.Tensor, Dict[str, Any]]:
        """Build a needle-in-haystack prompt with needle at specified position."""
        # Simple approach: repeat filler text and insert needle
        filler = "The system processes data through multiple layers. "
        
        if position == "early":
            needle_pos = 0.1
        elif position == "middle":
            needle_pos = 0.5
        else:  # late
            needle_pos = 0.9
        
        # Build prompt
        prompt = filler * 100  # Start with lots of filler
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=ctx_len)["input_ids"]
        
        # Insert needle
        needle_pos_tokens = int(prompt_ids.shape[1] * needle_pos)
        needle_ids = self.tokenizer(needle_text, return_tensors="pt")["input_ids"]
        
        # Reconstruct with needle
        before = prompt_ids[:, :needle_pos_tokens]
        after = prompt_ids[:, needle_pos_tokens:]
        prompt_ids = torch.cat([before, needle_ids, after], dim=1)
        prompt_ids = prompt_ids[:, :ctx_len]  # Truncate to target length
        
        query = "\nQuestion: What is the secret passkey? Answer: The secret passkey is"
        query_ids = self.tokenizer(query, return_tensors="pt")["input_ids"]
        full_ids = torch.cat([prompt_ids, query_ids], dim=1)
        
        meta = {
            "total_tokens": full_ids.shape[1],
            "needle_position": needle_pos_tokens,
            "context_length": ctx_len,
        }
        
        return self.tokenizer.decode(full_ids[0], skip_special_tokens=True), full_ids, meta
    
    def _check_needle_retrieval(self, generated_text: str, needle: str, distractors: List[str]) -> bool:
        """Check if needle was correctly retrieved."""
        text_lower = generated_text.lower()
        needle_lower = needle.lower()
        
        # Check for needle
        if "94812" in generated_text or "passkey" in text_lower:
            # Verify it's not a distractor
            for dist in distractors:
                if dist in generated_text and "94812" not in generated_text:
                    return False
            return True
        return False
    
    # -------------------------------------------------------------------------
    # Benchmark 2: Long-Context Quality
    # -------------------------------------------------------------------------
    
    def run_long_context_quality(self, baseline_name: str, baseline_config: Dict[str, Any]) -> List[BenchmarkResult]:
        """Run long-context instruction-following quality benchmarks."""
        results = []
        
        # Define test prompts
        test_cases = [
            {
                "name": "instruction_following",
                "prompt": """You are a helpful assistant. Follow the instruction precisely.

Context: The API configuration specifies port 8443 for secure connections. The database uses connection pooling with 4 read replicas. The cache layer uses Redis with TTL of 3600 seconds.

Instruction: List the three configuration parameters mentioned above in JSON format.

Response:""",
                "expected_keywords": ["8443", "4", "3600", "JSON"],
            },
            {
                "name": "summarization",
                "prompt": """Summarize the following technical document in 3 sentences.

Document: Modern inference engines use PagedAttention to manage KV caches efficiently. Instead of contiguous memory allocation, PagedAttention divides the KV cache into fixed-size blocks that can be dynamically allocated and freed. This reduces memory fragmentation and allows sharing of prefix blocks across requests. vLLM implements this with a block manager that tracks block reference counts and evicts unused blocks. The system also supports prefix caching for common prompt prefixes.

Summary:""",
                "expected_keywords": ["PagedAttention", "blocks", "vLLM", "prefix caching"],
            },
            {
                "name": "fact_retrieval",
                "prompt": """Based on the context below, answer the question.

Context: The microservice architecture consists of 12 services deployed across 3 availability zones. Service A handles authentication with JWT tokens expiring in 15 minutes. Service B manages user profiles with eventual consistency. Service C processes payments with idempotency keys. The circuit breaker threshold is 5 failures in 30 seconds.

Question: What is the JWT token expiration time?

Answer:""",
                "expected_keywords": ["15 minutes", "JWT"],
            },
            {
                "name": "multi_fact_qa",
                "prompt": """Answer the questions based on the context.

Context: The Kubernetes cluster runs version 1.28 with 50 worker nodes. Each node has 64 CPU cores and 256 GB RAM. The cluster uses Calico for CNI and metallb for load balancing. Persistent volumes use AWS EBS gp3 volumes. The monitoring stack includes Prometheus, Grafana, and Alertmanager.

Questions:
1. What Kubernetes version is running?
2. How many worker nodes?
3. What CNI is used?
4. What storage class for persistent volumes?

Answers:""",
                "expected_keywords": ["1.28", "50", "Calico", "gp3"],
            },
        ]
        
        for test_case in test_cases:
            for ctx_len in self.config["benchmark"]["context_lengths"]:
                for seed in self.config["benchmark"]["seeds"]:
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                    
                    # Extend prompt to target context length
                    prompt_text = test_case["prompt"]
                    prompt_ids = self.tokenizer(prompt_text, return_tensors="pt")["input_ids"]
                    
                    # Pad with filler if needed
                    if prompt_ids.shape[1] < ctx_len:
                        filler = "Additional context for length padding. " * 100
                        filler_ids = self.tokenizer(filler, return_tensors="pt", truncation=True, max_length=ctx_len - prompt_ids.shape[1])["input_ids"]
                        prompt_ids = torch.cat([prompt_ids, filler_ids], dim=1)
                    
                    prompt_ids = prompt_ids[:, :ctx_len]
                    
                    cache = self.create_cache(baseline_config)
                    
                    try:
                        generated_ids, telemetry = self.run_inference(
                            cache, prompt_ids, max_new_tokens=64, do_sample=False
                        )
                        
                        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                        
                        # Check for expected keywords
                        found_keywords = sum(1 for kw in test_case["expected_keywords"] if kw.lower() in generated_text.lower())
                        keyword_score = found_keywords / len(test_case["expected_keywords"])
                        
                        metrics = {
                            "task": test_case["name"],
                            "context_length": ctx_len,
                            "seed": seed,
                            "generated_text": generated_text[:500],
                            "keyword_score": keyword_score,
                            "keywords_found": found_keywords,
                            "keywords_total": len(test_case["expected_keywords"]),
                            **telemetry,
                        }
                        
                        results.append(BenchmarkResult(
                            benchmark_name="long_context_quality",
                            method_name=baseline_name,
                            config=baseline_config,
                            metrics=metrics,
                        ))
                        
                    except Exception as e:
                        self.log(f"Error in long-context quality: {e}", "ERROR")
                        results.append(BenchmarkResult(
                            benchmark_name="long_context_quality",
                            method_name=baseline_name,
                            config=baseline_config,
                            metrics={"task": test_case["name"], "context_length": ctx_len, "seed": seed},
                            success=False,
                            error=str(e),
                        ))
        
        return results
    
    # -------------------------------------------------------------------------
    # Benchmark 3: Context Drift and Adaptation
    # -------------------------------------------------------------------------
    
    def run_context_drift(self, baseline_name: str, baseline_config: Dict[str, Any]) -> List[BenchmarkResult]:
        """Run context drift and adaptation evaluation."""
        results = []
        
        # Build drift prompt (similar to existing bench_context_drift.py)
        fact = "Important memory: your friend Maya fell in a ditch during the hike. "
        unrelated = (
            "Unrelated topic: CSS grid uses twelve equal columns, glass cards use backdrop blur, "
            "and keyframe animations change opacity. A telescope observes a distant nebula, while "
            "a recipe simmers tomatoes, cumin, coriander, and rice. "
        )
        context = fact + unrelated * 70
        query = "\nQuestion: Earlier, what did Maya fall in? Maya fell in a"
        full_prompt = context + query
        
        target = "ditch"
        target_id = self.tokenizer(target, return_tensors="pt")["input_ids"][0, -1].item()
        
        for seed in self.config["benchmark"]["seeds"]:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            
            cache = self.create_cache(baseline_config)
            input_ids = self.tokenizer(full_prompt, return_tensors="pt")["input_ids"].to(next(self.model.parameters()).device)
            
            try:
                # Process in chunks
                chunk_size = 128
                for start in range(0, input_ids.shape[1], chunk_size):
                    out = self.model(
                        input_ids=input_ids[:, start:start+chunk_size],
                        past_key_values=cache,
                        use_cache=True,
                        output_attentions=False,
                    )
                
                logits = out.logits[0, -1].float()
                probabilities = torch.softmax(logits, dim=-1)
                rank = int((logits > logits[target_id]).sum().item()) + 1
                top_id = int(torch.argmax(logits).item())
                
                telemetry = cache.get_telemetry_summary()
                
                # Calculate H2O freshness
                drift_boundary = int(input_ids.shape[1] * 0.05)
                h2o_positions = []
                for layer in cache._tiers:
                    for tier in layer:
                        if tier.h2o_indices is not None:
                            h2o_positions.extend(tier.h2o_indices.flatten().tolist())
                
                freshness = (
                    100.0 * sum(pos >= drift_boundary for pos in h2o_positions) / len(h2o_positions)
                    if h2o_positions else 0.0
                )
                
                metrics = {
                    "target": target,
                    "target_token_id": target_id,
                    "predicted_token_id": top_id,
                    "retrieval_top1": top_id == target_id,
                    "target_probability": float(probabilities[target_id].item()),
                    "target_rank": rank,
                    "h2o_post_drift_freshness_pct": freshness,
                    "prompt_tokens": int(input_ids.shape[1]),
                    **telemetry,
                }
                
                results.append(BenchmarkResult(
                    benchmark_name="context_drift",
                    method_name=baseline_name,
                    config=baseline_config,
                    metrics=metrics,
                ))
                
            except Exception as e:
                self.log(f"Error in context drift: {e}", "ERROR")
                results.append(BenchmarkResult(
                    benchmark_name="context_drift",
                    method_name=baseline_name,
                    config=baseline_config,
                    metrics={"seed": seed},
                    success=False,
                    error=str(e),
                ))
        
        return results
    
    # -------------------------------------------------------------------------
    # Benchmark 4: Memory Accounting
    # -------------------------------------------------------------------------
    
    def run_memory_accounting(self, baseline_name: str, baseline_config: Dict[str, Any]) -> List[BenchmarkResult]:
        """Measure detailed memory accounting."""
        results = []
        
        for ctx_len in self.config["benchmark"]["context_lengths"]:
            for seed in self.config["benchmark"]["seeds"]:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                
                # Create a prompt of target length
                prompt = "The system configuration includes multiple parameters. " * 200
                prompt_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=ctx_len)["input_ids"]
                
                cache = self.create_cache(baseline_config)
                
                try:
                    # Prefill
                    _ = self.run_inference(cache, prompt_ids, max_new_tokens=1)
                    
                    # Get detailed memory accounting
                    memory_info = self.get_memory_usage(cache)
                    
                    # Calculate compression ratio
                    fp16_bytes = memory_info["cache_fp16_bytes"]
                    compressed_bytes = memory_info["cache_compressed_bytes"]
                    dict_bytes = memory_info["dictionary_bytes"]
                    
                    if fp16_bytes > 0:
                        compression_ratio = fp16_bytes / max(compressed_bytes, 1)
                        memory_savings_pct = 100 * (1 - (compressed_bytes + dict_bytes) / fp16_bytes)
                    else:
                        compression_ratio = 1.0
                        memory_savings_pct = 0.0
                    
                    metrics = {
                        "context_length": ctx_len,
                        "seed": seed,
                        "fp16_reference_bytes": fp16_bytes,
                        "compressed_payload_bytes": compressed_bytes,
                        "dictionary_bytes": dict_bytes,
                        "total_compressed_bytes": compressed_bytes + dict_bytes,
                        "compression_ratio": compression_ratio,
                        "memory_savings_pct": memory_savings_pct,
                        **memory_info,
                    }
                    
                    results.append(BenchmarkResult(
                        benchmark_name="memory_accounting",
                        method_name=baseline_name,
                        config=baseline_config,
                        metrics=metrics,
                    ))
                    
                except Exception as e:
                    self.log(f"Error in memory accounting: {e}", "ERROR")
                    results.append(BenchmarkResult(
                        benchmark_name="memory_accounting",
                        method_name=baseline_name,
                        config=baseline_config,
                        metrics={"context_length": ctx_len, "seed": seed},
                        success=False,
                        error=str(e),
                    ))
        
        return results
    
    # -------------------------------------------------------------------------
    # Benchmark 5: Latency and Throughput
    # -------------------------------------------------------------------------
    
    def run_latency_throughput(self, baseline_name: str, baseline_config: Dict[str, Any]) -> List[BenchmarkResult]:
        """Measure latency and throughput."""
        results = []
        
        for ctx_len in self.config["benchmark"]["context_lengths"]:
            for gen_len in self.config["benchmark"]["generation_lengths"]:
                for seed in self.config["benchmark"]["seeds"]:
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                    
                    prompt = "The system configuration includes multiple parameters. " * 200
                    prompt_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=ctx_len)["input_ids"]
                    
                    cache = self.create_cache(baseline_config)
                    
                    try:
                        latency_metrics = self.measure_latency(
                            cache, prompt_ids,
                            max_new_tokens=gen_len,
                            warmup=self.config["benchmark"]["warmup_iterations"],
                            repetitions=self.config["benchmark"]["timing_iterations"],
                        )
                        
                        memory_info = self.get_memory_usage(cache)
                        
                        metrics = {
                            "context_length": ctx_len,
                            "generation_length": gen_len,
                            "seed": seed,
                            **latency_metrics,
                            **memory_info,
                        }
                        
                        results.append(BenchmarkResult(
                            benchmark_name="latency_throughput",
                            method_name=baseline_name,
                            config=baseline_config,
                            metrics=metrics,
                        ))
                        
                    except Exception as e:
                        self.log(f"Error in latency throughput: {e}", "ERROR")
                        results.append(BenchmarkResult(
                            benchmark_name="latency_throughput",
                            method_name=baseline_name,
                            config=baseline_config,
                            metrics={"context_length": ctx_len, "generation_length": gen_len, "seed": seed},
                            success=False,
                            error=str(e),
                        ))
        
        return results
    
    # -------------------------------------------------------------------------
    # Benchmark 6: Encoder Quality and Cost
    # -------------------------------------------------------------------------
    
    def run_encoder_quality(self, baseline_name: str, baseline_config: Dict[str, Any]) -> List[BenchmarkResult]:
        """Compare encoder quality: Fast 1-step vs OMP vs FISTA."""
        results = []
        
        if baseline_config.get("policy_mode") not in ("hybrid", "standalone_lexico"):
            return results  # Only relevant for Lexico encoding
        
        # Collect KV states for encoding benchmark
        layer_idx = 0
        target_tokens = 1024
        
        # Use calibration prompts to get KV states
        from benchmarks.bench_person1 import (
            ENGLISH_EVAL_PROMPTS, PYTHON_EVAL_PROMPTS,
            _collect_domain_kv_states
        )
        
        try:
            X_bench, _ = _collect_domain_kv_states(
                self.model, self.tokenizer,
                ENGLISH_EVAL_PROMPTS + PYTHON_EVAL_PROMPTS,
                layer_idx=layer_idx,
                target_tokens=target_tokens,
                max_seq_len=256,
            )
            X_bench = X_bench[:target_tokens].contiguous()
            
            for s in [4, 8, 12, 16, 32]:
                # Fast 1-step
                idx_fast, val_fast = self.codec.encode(X_bench, layer_idx=layer_idx, s=s, lambda1=0.04, norm_rescale=True)
                x_hat_fast = self.codec.decode(idx_fast, val_fast, layer_idx=layer_idx)
                q_fast = self.codec.evaluate_reconstruction(X_bench, x_hat_fast)
                
                # OMP
                idx_omp, val_omp = self.codec.omp_encode(X_bench, layer_idx=layer_idx, s=s)
                x_hat_omp = self.codec.decode(idx_omp, val_omp, layer_idx=layer_idx)
                q_omp = self.codec.evaluate_reconstruction(X_bench, x_hat_omp)
                
                # FISTA
                idx_fista, val_fista = self.codec.iterative_elasticnet_encode(
                    X_bench, layer_idx=layer_idx, s=s, lambda1=0.04, lambda2=0.1, num_iters=60
                )
                x_hat_fista = self.codec.decode(idx_fista, val_fista, layer_idx=layer_idx)
                q_fista = self.codec.evaluate_reconstruction(X_bench, x_hat_fista)
                
                # Memory accounting
                mem_fast = self.codec.memory_accounting(idx_fast, val_fast)
                mem_omp = self.codec.memory_accounting(idx_omp, val_omp)
                mem_fista = self.codec.memory_accounting(idx_fista, val_fista)
                
                metrics = {
                    "sparsity_s": s,
                    "bytes_per_vector": 3 * s + 2,
                    "fast_cosine": q_fast["cosine_similarity"],
                    "fast_mse": q_fast["mse"],
                    "fast_rel_mse": q_fast["relative_mse"],
                    "fast_rel_l2": q_fast["relative_l2_error"],
                    "omp_cosine": q_omp["cosine_similarity"],
                    "omp_mse": q_omp["mse"],
                    "omp_rel_mse": q_omp["relative_mse"],
                    "omp_rel_l2": q_omp["relative_l2_error"],
                    "fista_cosine": q_fista["cosine_similarity"],
                    "fista_mse": q_fista["mse"],
                    "fista_rel_mse": q_fista["relative_mse"],
                    "fista_rel_l2": q_fista["relative_l2_error"],
                    "fast_logical_bytes": mem_fast["logical_payload_bytes"],
                    "omp_logical_bytes": mem_omp["logical_payload_bytes"],
                    "fista_logical_bytes": mem_fista["logical_payload_bytes"],
                }
                
                results.append(BenchmarkResult(
                    benchmark_name="encoder_quality",
                    method_name=baseline_name,
                    config={**baseline_config, "sparsity_s": s},
                    metrics=metrics,
                ))
                
        except Exception as e:
            self.log(f"Error in encoder quality: {e}", "ERROR")
            results.append(BenchmarkResult(
                benchmark_name="encoder_quality",
                method_name=baseline_name,
                config=baseline_config,
                metrics={},
                success=False,
                error=str(e),
            ))
        
        return results
    
    # -------------------------------------------------------------------------
    # Benchmark 7: Ablation Experiments
    # -------------------------------------------------------------------------
    
    def run_ablations(self) -> List[BenchmarkResult]:
        """Run all ablation experiments."""
        all_results = []
        
        ablation_configs = self.config["ablations"]
        
        for ablation in ablation_configs:
            name = ablation["name"]
            self.log(f"Running ablation: {name}")
            
            baseline_config = {k: v for k, v in ablation.items() if k not in ("name", "description")}
            
            # Run needle retrieval for this ablation
            needle_results = self.run_needle_retrieval(name, baseline_config)
            all_results.extend(needle_results)
            
            # Run context drift
            drift_results = self.run_context_drift(name, baseline_config)
            all_results.extend(drift_results)
            
            # Run memory accounting
            mem_results = self.run_memory_accounting(name, baseline_config)
            all_results.extend(mem_results)
            
            # Run latency
            lat_results = self.run_latency_throughput(name, baseline_config)
            all_results.extend(lat_results)
        
        return all_results
    
    # -------------------------------------------------------------------------
    # Benchmark 8: Equal Memory Budget Comparison
    # -------------------------------------------------------------------------
    
    def run_equal_memory_budget(self) -> List[BenchmarkResult]:
        """Compare methods at equal memory budgets."""
        all_results = []
        
        budgets = self.config["benchmark"]["memory_budgets"]
        
        # For each budget, configure each method to meet it
        for budget in budgets:
            self.log(f"Testing at memory budget: {budget*100:.0f}%")
            
            # B0: Full FP16 (only at 100%)
            if budget >= 1.0:
                baseline_config = self.config["baselines"]["B0_full_fp16"]
                results = self.run_needle_retrieval("B0_full_fp16", baseline_config)
                all_results.extend(results)
            
            # B3: Standalone H2O - adjust keep_ratio
            h2o_config = self.config["baselines"]["B3_token_eviction"]["methods"][2].copy()
            # We'd need to solve for keep_ratio to hit target budget
            # For now, use the default
            results = self.run_needle_retrieval("B3_h2o", h2o_config)
            all_results.extend(results)
            
            # B4: StoikovKV Hybrid - adjust keep_ratio and s_budget
            hybrid_config = self.config["baselines"]["B4_stoikovkv_hybrid"].copy()
            results = self.run_needle_retrieval("B4_stoikovkv", hybrid_config)
            all_results.extend(results)
        
        return all_results
    
    # -------------------------------------------------------------------------
    # Main Run Method
    # -------------------------------------------------------------------------
    
    def run_all(self):
        """Run all benchmarks."""
        self.log("=" * 60)
        self.log("Starting StoikovKV Comparative Audit Benchmark")
        self.log("=" * 60)
        
        self.save_environment()
        self.setup_model()
        
        # Define baselines to test
        baselines = {
            "B0_full_fp16": self.config["baselines"]["B0_full_fp16"],
            "B3_standalone_h2o": self.config["baselines"]["B3_token_eviction"]["methods"][2],
            "B4_stoikovkv_hybrid": self.config["baselines"]["B4_stoikovkv_hybrid"],
        }
        
        # Add standalone Lexico if configured
        baselines["B3_standalone_lexico"] = {
            "policy_mode": "standalone_lexico",
            "n_b": 64,
            "keep_ratio": 1.0,
            "s_budget": 14,
            "enable_hmm_decay": False,
        }
        
        # Run main benchmarks for each baseline
        for name, config in baselines.items():
            self.log(f"\n{'='*60}")
            self.log(f"Testing baseline: {name}")
            self.log(f"{'='*60}")
            
            # Needle retrieval
            self.log("Running needle retrieval benchmark...")
            results = self.run_needle_retrieval(name, config)
            for r in results:
                self.add_result(r)
            
            # Long-context quality
            self.log("Running long-context quality benchmark...")
            results = self.run_long_context_quality(name, config)
            for r in results:
                self.add_result(r)
            
            # Context drift
            self.log("Running context drift benchmark...")
            results = self.run_context_drift(name, config)
            for r in results:
                self.add_result(r)
            
            # Memory accounting
            self.log("Running memory accounting benchmark...")
            results = self.run_memory_accounting(name, config)
            for r in results:
                self.add_result(r)
            
            # Latency and throughput
            self.log("Running latency/throughput benchmark...")
            results = self.run_latency_throughput(name, config)
            for r in results:
                self.add_result(r)
            
            # Encoder quality (for Lexico methods)
            if config.get("policy_mode") in ("hybrid", "standalone_lexico"):
                self.log("Running encoder quality benchmark...")
                results = self.run_encoder_quality(name, config)
                for r in results:
                    self.add_result(r)
        
        # Run ablations
        self.log("\n" + "=" * 60)
        self.log("Running ablation experiments...")
        self.log("=" * 60)
        ablation_results = self.run_ablations()
        for r in ablation_results:
            self.add_result(r)
        
        # Run equal memory budget comparison
        self.log("\n" + "=" * 60)
        self.log("Running equal memory budget comparison...")
        self.log("=" * 60)
        eq_results = self.run_equal_memory_budget()
        for r in eq_results:
            self.add_result(r)
        
        self.log("\n" + "=" * 60)
        self.log("All benchmarks completed!")
        self.log("=" * 60)
        
        # Generate summary
        self.generate_summary()
    
    def generate_summary(self):
        """Generate a summary of results."""
        summary_path = self.output_dir / "SUMMARY.md"
        
        with open(summary_path, "w") as f:
            f.write("# StoikovKV Comparative Audit - Summary\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Model Track:** {self.model_track}\n")
            f.write(f"**Git Commit:** {self.environment.get('git_commit', 'unknown')}\n\n")
            
            f.write("## Environment\n")
            f.write(f"- GPU: {self.environment.get('gpu_name', 'N/A')}\n")
            f.write(f"- VRAM: {self.environment.get('gpu_memory_gb', 0):.1f} GB\n")
            f.write(f"- PyTorch: {self.environment.get('pytorch_version', 'N/A')}\n")
            f.write(f"- CUDA: {self.environment.get('cuda_version', 'N/A')}\n\n")
            
            f.write("## Results Summary\n\n")
            
            # Group by benchmark
            by_benchmark = {}
            for r in self.results:
                if r.benchmark_name not in by_benchmark:
                    by_benchmark[r.benchmark_name] = {}
                if r.method_name not in by_benchmark[r.benchmark_name]:
                    by_benchmark[r.benchmark_name][r.method_name] = {"success": 0, "total": 0}
                by_benchmark[r.benchmark_name][r.method_name]["total"] += 1
                if r.success:
                    by_benchmark[r.benchmark_name][r.method_name]["success"] += 1
            
            for bench_name, methods in by_benchmark.items():
                f.write(f"### {bench_name}\n\n")
                f.write("| Method | Success | Total |\n")
                f.write("|--------|---------|-------|\n")
                for method, stats in methods.items():
                    f.write(f"| {method} | {stats['success']} | {stats['total']} |\n")
                f.write("\n")
        
        self.log(f"Summary saved to {summary_path}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="StoikovKV Comparative Audit Benchmark")
    parser.add_argument("--config", default="benchmarks/comparative_audit/config/benchmark_config.yaml",
                        help="Path to benchmark configuration")
    parser.add_argument("--output", default="benchmarks/comparative_audit",
                        help="Output directory")
    parser.add_argument("--track", choices=["fallback", "real_7b"], default="fallback",
                        help="Model track to use")
    parser.add_argument("--benchmarks", nargs="+", 
                        choices=["needle", "long_context", "drift", "memory", "latency", "encoder", "ablation", "equal_budget", "all"],
                        default=["all"],
                        help="Specific benchmarks to run")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    runner = BenchmarkRunner(
        config_path=args.config,
        output_dir=args.output,
        model_track=args.track,
        verbose=not args.quiet,
    )
    
    try:
        runner.run_all()
    except KeyboardInterrupt:
        runner.log("Benchmark interrupted by user", "WARNING")
        runner.save_results()
        sys.exit(1)
    except Exception as e:
        runner.log(f"Benchmark failed: {e}", "ERROR")
        traceback.print_exc()
        runner.save_results()
        sys.exit(1)


if __name__ == "__main__":
    main()