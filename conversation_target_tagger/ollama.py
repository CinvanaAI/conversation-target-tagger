"""Small Ollama transport with a loopback-only default."""

from __future__ import annotations

import ipaddress
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlparse


def validate_base_url(value: str, *, allow_remote: bool = False) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Ollama URL must be an http(s) origin without embedded credentials.")
    if parsed.path not in {"", "/"} or parsed.params or parsed.query or parsed.fragment:
        raise ValueError("Ollama URL must be an origin without a path, query, or fragment.")
    if not allow_remote:
        hostname = parsed.hostname.casefold()
        is_loopback = hostname == "localhost"
        if not is_loopback:
            try:
                is_loopback = ipaddress.ip_address(hostname).is_loopback
            except ValueError:
                is_loopback = False
        if not is_loopback:
            raise ValueError("Remote Ollama hosts require explicit allow_remote=True.")
    return value.rstrip("/")


@dataclass(frozen=True)
class OllamaTagger:
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen2.5:7b-instruct"
    timeout_seconds: int = 120
    retries: int = 2
    allow_remote: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", validate_base_url(self.base_url, allow_remote=self.allow_remote))
        if not self.model.strip() or len(self.model) > 200:
            raise ValueError("A bounded Ollama model name is required.")
        if not 1 <= self.timeout_seconds <= 600:
            raise ValueError("timeout_seconds must be between 1 and 600")
        if not 1 <= self.retries <= 5:
            raise ValueError("retries must be between 1 and 5")

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {"temperature": 0, "num_predict": 800},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + "/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    body = json.loads(response.read().decode("utf-8"))
                message = body.get("message") if isinstance(body, dict) else None
                content = message.get("content") if isinstance(message, dict) else None
                if not isinstance(content, str):
                    raise ValueError("Ollama response did not contain assistant text.")
                return content.strip()
            except (OSError, ValueError, urllib.error.URLError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(0.2 * attempt)
        raise RuntimeError(f"Ollama tagging failed after {self.retries} attempts: {type(last_error).__name__}")
