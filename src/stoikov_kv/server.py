"""
OpenAI-Compatible Local Inference Server for StoikovKV (`stoikov_kv/server.py`).

Exposes `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/v1/telemetry`, and `/health`
backed by `StoikovKVCache` (with optional `matryoshka_speculative_generate`) so local LLM harnesses
(OpenWebUI, Continue.dev, Cursor, `lm-eval-harness`) can connect over HTTP without extra dependencies.
"""

from __future__ import annotations

import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .cache import StoikovKVCache, load_or_create_artifacts
from .calibration import load_model_and_tokenizer
from .matryoshka_spec import matryoshka_speculative_generate


def _format_chat_messages(messages: List[Dict[str, Any]], tokenizer: Any) -> str:
    """Formats OpenAI-style `messages` into a single prompt string."""
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return str(
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
        except Exception:
            pass
    parts: List[str] = []
    for msg in messages:
        role = str(msg.get("role", "user")).strip()
        content = str(msg.get("content", "")).strip()
        parts.append(f"{role}: {content}")
    parts.append("assistant:")
    return "\n".join(parts)


class StoikovInferenceEngine:
    """
    Stateful local inference engine wrapping a CausalLM, tokenizer, and calibrated
    `StoikovKVCache` artifacts for OpenAI-compatible serving and CLI chat.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        artifacts: Dict[str, Any],
        model_id: str = "stoikov-qwen2.5",
        default_policy_mode: str = "hybrid",
        n_b: int = 64,
        keep_ratio: float = 0.25,
        s_budget: int = 16,
        draft_atoms: int = 4,
        gamma_spec: int = 4,
        default_speculative: bool = True,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.artifacts = artifacts
        self.model_id = str(model_id)
        self.default_policy_mode = str(default_policy_mode)
        self.n_b = int(n_b)
        self.keep_ratio = float(keep_ratio)
        self.s_budget = int(s_budget)
        self.draft_atoms = int(draft_atoms)
        self.gamma_spec = int(gamma_spec)
        self.default_speculative = bool(default_speculative)
        self.last_telemetry: Dict[str, Any] = {}

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        artifacts_path: str = "artifacts/calibrated_artifacts.pt",
        device: Optional[str] = None,
        use_real_model: bool = False,
        load_in_4bit: bool = False,
        policy_mode: str = "hybrid",
        n_b: int = 64,
        keep_ratio: float = 0.25,
        s_budget: int = 16,
        draft_atoms: int = 4,
        gamma_spec: int = 4,
        speculative: bool = True,
    ) -> "StoikovInferenceEngine":
        model, tokenizer = load_model_and_tokenizer(
            model_name=model_name,
            device=device,
            use_real_model=use_real_model,
            load_in_4bit=load_in_4bit,
        )
        artifacts = load_or_create_artifacts(
            model=model,
            artifacts_path=artifacts_path,
            tokenizer=tokenizer,
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            artifacts=artifacts,
            model_id=model_name,
            default_policy_mode=policy_mode,
            n_b=n_b,
            keep_ratio=keep_ratio,
            s_budget=s_budget,
            draft_atoms=draft_atoms,
            gamma_spec=gamma_spec,
            default_speculative=speculative,
        )

    @torch.no_grad()
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.6,
        top_k: int = 32,
        speculative: Optional[bool] = None,
        policy_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        use_spec = self.default_speculative if speculative is None else bool(speculative)
        mode = self.default_policy_mode if policy_mode is None else str(policy_mode)

        cache = StoikovKVCache(
            model=self.model,
            artifacts=self.artifacts,
            n_b=self.n_b,
            keep_ratio=self.keep_ratio,
            s_budget=self.s_budget,
            policy_mode=mode,
        )

        device = next(self.model.parameters()).device
        enc = self.tokenizer(prompt, return_tensors="pt")
        input_ids = (enc["input_ids"] if isinstance(enc, dict) else enc.input_ids).to(device)
        if input_ids.shape[-1] < 2:
            pad_id = int(getattr(self.tokenizer, "pad_token_id", 0) or 0)
            input_ids = torch.cat(
                [torch.tensor([[pad_id]], device=device, dtype=input_ids.dtype), input_ids],
                dim=-1,
            )
        prompt_tokens = int(input_ids.shape[-1])

        t0 = time.perf_counter()
        if use_spec:
            spec_res = matryoshka_speculative_generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=input_ids,
                cache=cache,
                gamma_spec=self.gamma_spec,
                draft_atoms=min(self.draft_atoms, self.s_budget),
                verify_atoms=self.s_budget,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 0.05),
                top_k=top_k,
            )
            text = str(spec_res["generated_text"])
            completion_tokens = int(spec_res["num_generated_tokens"])
            acceptance_rate = float(spec_res["acceptance_rate"])
            tokens_per_sec = float(spec_res["tokens_per_sec"])
        else:
            out_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=max(temperature, 1e-5),
                top_k=top_k,
                past_key_values=cache,
                use_cache=True,
            )
            elapsed = max(time.perf_counter() - t0, 1e-6)
            new_ids = out_ids[0, prompt_tokens:]
            completion_tokens = int(new_ids.shape[0])
            text = (
                self.tokenizer.decode(new_ids, skip_special_tokens=True)
                if hasattr(self.tokenizer, "decode")
                else " ".join(str(int(t)) for t in new_ids)
            )
            acceptance_rate = 1.0
            tokens_per_sec = float(completion_tokens) / elapsed

        telemetry = cache.get_telemetry_summary()
        telemetry["speculative"] = use_spec
        telemetry["acceptance_rate"] = acceptance_rate
        telemetry["tokens_per_sec"] = tokens_per_sec
        self.last_telemetry = telemetry

        return {
            "text": text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "telemetry": telemetry,
        }

    def handle_request(
        self, method: str, path: str, body: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Handles an HTTP request in-memory and returns `(status_code, json_payload)`.
        """
        m = method.upper().strip()
        p = path.split("?")[0].rstrip("/") or "/"

        if m == "GET" and p == "/health":
            return 200, {
                "status": "ok",
                "model": self.model_id,
                "default_policy_mode": self.default_policy_mode,
                "default_speculative": self.default_speculative,
            }

        if m == "GET" and p == "/v1/models":
            return 200, {
                "object": "list",
                "data": [
                    {
                        "id": self.model_id,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "stoikov-kv",
                    }
                ],
            }

        if m == "GET" and p == "/v1/telemetry":
            return 200, {"object": "stoikov.telemetry", "data": self.last_telemetry}

        if m == "POST" and p == "/v1/completions":
            req = body or {}
            prompt = str(req.get("prompt", "Hello"))
            max_tokens = int(req.get("max_tokens", 24))
            temperature = float(req.get("temperature", 0.5))
            speculative = req.get("speculative", None)
            policy_mode = req.get("policy_mode", None)

            gen = self.generate_text(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                speculative=speculative,
                policy_mode=policy_mode,
            )
            return 200, {
                "id": f"cmpl-{uuid.uuid4().hex[:12]}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": self.model_id,
                "choices": [
                    {
                        "text": gen["text"],
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "length",
                    }
                ],
                "usage": {
                    "prompt_tokens": gen["prompt_tokens"],
                    "completion_tokens": gen["completion_tokens"],
                    "total_tokens": gen["total_tokens"],
                },
                "stoikov_telemetry": gen["telemetry"],
            }

        if m == "POST" and p == "/v1/chat/completions":
            req = body or {}
            messages = req.get("messages", [{"role": "user", "content": "Hello"}])
            prompt = _format_chat_messages(messages, self.tokenizer)
            max_tokens = int(req.get("max_tokens", 24))
            temperature = float(req.get("temperature", 0.5))
            speculative = req.get("speculative", None)
            policy_mode = req.get("policy_mode", None)

            gen = self.generate_text(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                speculative=speculative,
                policy_mode=policy_mode,
            )
            return 200, {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": gen["text"]},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": gen["prompt_tokens"],
                    "completion_tokens": gen["completion_tokens"],
                    "total_tokens": gen["total_tokens"],
                },
                "stoikov_telemetry": gen["telemetry"],
            }

        return 404, {"error": {"message": f"Unknown route: {m} {p}", "type": "invalid_request_error"}}


def create_http_server(
    engine: StoikovInferenceEngine, host: str = "127.0.0.1", port: int = 8000
) -> ThreadingHTTPServer:
    """Creates a `ThreadingHTTPServer` bound to `(host, port)` serving `engine`."""

    class _Handler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            status, payload = engine.handle_request("GET", self.path)
            self._send_json(status, payload)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            try:
                body = json.loads(raw_body) if raw_body.strip() else {}
            except Exception:
                self._send_json(400, {"error": {"message": "Invalid JSON body"}})
                return

            if body.get("stream", False) and self.path.rstrip("/") == "/v1/chat/completions":
                status, payload = engine.handle_request("POST", self.path, body)
                if status != 200:
                    self._send_json(status, payload)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                chunk = {
                    "id": payload["id"],
                    "object": "chat.completion.chunk",
                    "created": payload["created"],
                    "model": payload["model"],
                    "choices": [
                        {
                            "index": 0,
                            "delta": payload["choices"][0]["message"],
                            "finish_reason": "stop",
                        }
                    ],
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode("utf-8"))
                return

            status, payload = engine.handle_request("POST", self.path, body)
            self._send_json(status, payload)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return ThreadingHTTPServer((host, int(port)), _Handler)
