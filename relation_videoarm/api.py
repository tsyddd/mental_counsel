from __future__ import annotations

import json
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import requests


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)

    def add(self, payload: dict[str, Any], elapsed: float, purpose: str) -> None:
        usage = payload.get("usage") or {}
        p = int(usage.get("prompt_tokens") or 0)
        c = int(usage.get("completion_tokens") or 0)
        t = int(usage.get("total_tokens") or p + c)
        self.prompt_tokens += p
        self.completion_tokens += c
        self.total_tokens += t
        self.calls += 1
        self.events.append({
            "purpose": purpose,
            "prompt_tokens": p,
            "completion_tokens": c,
            "total_tokens": t,
            "elapsed_seconds": round(elapsed, 3),
        })

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "calls": self.calls,
            "events": self.events,
        }


class APIClient:
    def __init__(self, model: str | None = None, timeout: int = 300, attempts: int = 5):
        self.model = model or os.getenv("RELATION_VIDEOARM_MODEL", "gpt-6-luna")
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api2.mcben.top/v1").rstrip("/")
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.timeout = timeout
        self.attempts = attempts
        self.usage = Usage()
        self._lock = threading.Lock()

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        purpose: str,
        max_tokens: int = 2048,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for model calls")
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if tools:
            body["tools"] = tools
            if tool_choice is not None:
                body["tool_choice"] = tool_choice
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last: Exception | None = None
        for attempt in range(1, self.attempts + 1):
            started = time.time()
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=body,
                    timeout=self.timeout,
                )
                if response.status_code >= 400:
                    text = response.text[:1000]
                    raise RuntimeError(f"HTTP {response.status_code}: {text}")
                payload = response.json()
                message = payload["choices"][0]["message"]
                with self._lock:
                    self.usage.add(payload, time.time() - started, purpose)
                return message
            except Exception as exc:
                last = exc
                if attempt >= self.attempts:
                    break
                time.sleep(min(30.0, 2.0 ** attempt + random.random()))
        raise RuntimeError(f"API call failed after {self.attempts} attempts: {last}")



    def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        """Create OpenAI-compatible embeddings for the optional semantic index."""
        if not texts:
            return []
        started = time.time()
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": model, "input": texts},
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"embedding HTTP {response.status_code}: {response.text[:500]}")
        payload = response.json()
        rows = sorted(payload.get("data") or [], key=lambda item: int(item.get("index", 0)))
        vectors = [item.get("embedding") for item in rows]
        if len(vectors) != len(texts) or any(not isinstance(vector, list) for vector in vectors):
            raise RuntimeError("embedding response has an unexpected data shape")
        with self._lock:
            self.usage.add(payload, time.time() - started, "memory_embedding")
        return [[float(value) for value in vector] for vector in vectors]


def parse_json_object(text: str) -> dict[str, Any]:
    value = (text or "").strip()
    if value.startswith("```"):
        value = value.replace("```json", "", 1).replace("```", "").strip()
    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    start, end = value.find("{"), value.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(value[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("model output does not contain a JSON object")
