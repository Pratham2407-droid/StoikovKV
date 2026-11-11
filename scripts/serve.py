#!/usr/bin/env python3
"""
Launch script for StoikovKV server.
Run: python scripts/serve.py
"""
import sys
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import uvicorn
from server.config import server_config

if __name__ == "__main__":
    print(f"[Serve] Starting StoikovKV server on {server_config.host}:{server_config.port}")
    uvicorn.run(
        "server.main:app",
        host=server_config.host,
        port=server_config.port,
        workers=server_config.workers,
        timeout_keep_alive=server_config.request_timeout,
        reload=False,
    )