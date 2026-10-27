#!/usr/bin/env python3
"""Generate run manifest for 7B GPU validation experiments."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def get_git_info():
    """Get git commit hash and status."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent.parent.parent, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=Path(__file__).parent.parent.parent, text=True
        ).strip()
        return commit, status
    except Exception:
        return "unknown", "unknown"


def get_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    if not filepath.exists():
        return "missing"
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_env_info():
    """Get environment information."""
    import torch
    import transformers
    
    info = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }
    
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_vram_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        info["cudnn_version"] = torch.backends.cudnn.version()
    
    return info


def main():
    root = Path(__file__).parent.parent.parent
    exp_dir = Path(__file__).parent
    
    # Git info
    git_commit, git_status = get_git_info()
    
    # Key files to hash
    key_files = {
        "calibration.py": root / "stoikov_kv" / "calibration.py",
        "cache.py": root / "stoikov_kv" / "cache.py",
        "elastic_lexico.py": root / "stoikov_kv" / "elastic_lexico.py",
        "bench_7b_baseline.py": exp_dir / "bench_7b_baseline.py",
        "calibrate_7b.py": exp_dir / "calibrate_7b.py",
        "bench_7b_stoikovkv.py": exp_dir / "bench_7b_stoikovkv.py",
        "plan.md": exp_dir / "plan.md",
    }
    
    file_hashes = {}
    for name, path in key_files.items():
        file_hashes[name] = get_file_hash(path)
    
    # Model checkpoint info
    model_path = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct"
    model_revision = "unknown"
    if model_path.exists():
        refs = list((model_path / "refs").glob("*"))
        if refs:
            model_revision = refs[0].read_text().strip()
        snapshots = list((model_path / "snapshots").glob("*"))
        if snapshots:
            model_revision = snapshots[0].name
    
    manifest = {
        "experiment": "7b_gpu_validation",
        "timestamp": datetime.now().isoformat(),
        "git": {
            "commit": git_commit,
            "status": git_status.split("\n") if git_status else [],
        },
        "environment": get_env_info(),
        "model": {
            "name": "Qwen/Qwen2.5-7B-Instruct",
            "revision": model_revision,
            "quantization": "4-bit NF4",
        },
        "file_hashes": file_hashes,
        "artifacts": {
            "calibrated_artifacts_1.5b": str(root / "calibrated_artifacts.pt"),
            "calibrated_artifacts_7b": str(root / "calibrated_artifacts_7b.pt"),
        },
    }
    
    # Save manifest
    manifest_path = exp_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Run manifest saved to: {manifest_path}")
    
    # Also save a summary for the final report
    summary = {
        "experiment_id": "7b_gpu_validation",
        "git_commit": git_commit[:8],
        "model": "Qwen/Qwen2.5-7B-Instruct (4-bit NF4)",
        "gpu": manifest["environment"].get("gpu_name", "unknown"),
        "vram_gb": manifest["environment"].get("gpu_vram_gb", 0),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()