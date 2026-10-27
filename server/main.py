from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch

from server.config import server_config, model_config, cache_config
from server.services.model_loader import load_model_and_tokenizer, create_cache_factory
from server.services.cache_manager import cache_manager
from server.routes import generate as generate_router

app = FastAPI(title="StoikovKV Server", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=server_config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(generate_router.router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    print("[Server] Loading model and tokenizer...")
    model, tokenizer, artifacts = load_model_and_tokenizer()
    
    print("[Server] Creating cache factory...")
    cache_factory, _, _, _ = create_cache_factory()
    
    cache_manager.cache_factory = cache_factory
    
    app.state.model = model
    app.state.tokenizer = tokenizer
    app.state.artifacts = artifacts
    
    print(f"[Server] Ready on {server_config.host}:{server_config.port}")

@app.on_event("shutdown")
async def shutdown_event():
    print("[Server] Shutting down...")

if __name__ == "__main__":
    uvicorn.run(
        "server.main:app",
        host=server_config.host,
        port=server_config.port,
        workers=server_config.workers,
        timeout_keep_alive=server_config.request_timeout,
    )