"""
ollama_client.py - Minimal client for the local Ollama REST API (standard library only).

Provides model availability checks (with automatic download), chat completions constrained to a
JSON schema (structured outputs), text embeddings and explicit model unloading, which keeps the
peak memory low enough for a 16 GB laptop because the two LLMs are never resident together.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


class OllamaError(RuntimeError):
    """Raised when the Ollama server is unreachable or returns an unusable response."""


class OllamaClient:
    """Thin wrapper around the Ollama HTTP endpoints used by this project."""

    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout: int = 1800):
        """Store the server URL (scheme added if missing) and the per-request timeout in seconds."""
        if not base_url.startswith(("http://", "https://")):
            base_url = "http://" + base_url
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._capabilities: dict = {}

    def _request(self, path: str, payload: dict | None = None) -> dict:
        """Send a GET (no payload) or POST request and return the decoded JSON response."""
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(self.base_url + path, data=data,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore")
            raise OllamaError(f"Ollama {path} returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise OllamaError(f"Cannot reach Ollama at {self.base_url} - start it with `ollama serve`.") from exc

    @staticmethod
    def _full_name(model: str) -> str:
        """Normalise a model name so that 'name' and 'name:latest' compare equal."""
        return model if ":" in model else model + ":latest"

    def list_models(self) -> list:
        """Return the names of all locally installed models."""
        return [m["name"] for m in self._request("/api/tags").get("models", [])]

    def ensure_models(self, models: list, auto_pull: bool = True) -> None:
        """Verify that every required model is installed, downloading missing ones if allowed."""
        installed = {self._full_name(m) for m in self.list_models()}
        for model in models:
            if self._full_name(model) in installed:
                continue
            if not auto_pull:
                raise OllamaError(f"Model '{model}' is not installed. Run: ollama pull {model}")
            print(f"[setup] Model '{model}' is not installed - downloading it once (this can take a while)...")
            self._request("/api/pull", {"model": model, "stream": False})

    def supports_thinking(self, model: str) -> bool:
        """Return True when the model advertises the 'thinking' capability (cached per model)."""
        if model not in self._capabilities:
            self._capabilities[model] = self._request("/api/show", {"model": model}).get("capabilities", [])
        return "thinking" in self._capabilities[model]

    def chat_json(self, model: str, system: str, user: str, schema: dict, think: bool = False,
                  num_ctx: int = 8192, max_tokens: int = 2048, retries: int = 2):
        """
        Run a chat completion whose output is constrained to `schema` (Ollama structured outputs).
        Decoding is greedy (temperature 0, fixed seed) for reproducibility; a retry with slight
        sampling is used only if the reply cannot be parsed. Returns (parsed_json, statistics).
        """
        payload = {
            "model": model, "stream": False, "format": schema,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": 0, "seed": 42, "num_ctx": num_ctx, "num_predict": max_tokens},
        }
        if self.supports_thinking(model):
            payload["think"] = bool(think)
        last_error = None
        for attempt in range(retries + 1):
            if attempt:
                payload["options"].update({"temperature": 0.3, "seed": 42 + attempt})
            started = time.time()
            response = self._request("/api/chat", payload)
            try:
                parsed = json.loads(response.get("message", {}).get("content", ""))
            except json.JSONDecodeError as exc:
                last_error = exc
                continue
            stats = {"seconds": round(time.time() - started, 2),
                     "prompt_tokens": response.get("prompt_eval_count", 0),
                     "output_tokens": response.get("eval_count", 0)}
            return parsed, stats
        raise OllamaError(f"{model} did not return valid JSON after {retries + 1} attempts: {last_error}")

    def embed(self, model: str, texts: list) -> list:
        """Return one embedding vector per input text (over-long texts are truncated by the server)."""
        return self._request("/api/embed", {"model": model, "input": texts, "truncate": True})["embeddings"]

    def unload(self, model: str) -> None:
        """Ask the server to release a model's memory immediately (failures are harmless and ignored)."""
        try:
            self._request("/api/generate", {"model": model, "keep_alive": 0})
        except OllamaError:
            pass
