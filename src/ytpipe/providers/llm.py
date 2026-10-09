"""OpenAI-compatible chat client.

Any endpoint that speaks ``/chat/completions`` works (OpenAI, Groq, Together,
OpenRouter, a local llama.cpp or Ollama server) — only ``base_url`` and ``model``
change. Everything above this file is provider-agnostic on purpose: the writer
prompt should not care which model is answering.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from typing import Any

import httpx

from ..config import LLMSettings

log = logging.getLogger(__name__)

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class LLMError(RuntimeError):
    """Raised for transport or contract failures the caller may retry."""

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


def extract_json(raw: str) -> Any:
    """Best-effort JSON recovery: fences, prose preamble, trailing commas, comments.

    Models wrap payloads in ```json even when told not to, append a polite sentence
    after the object, and emit `True`/trailing commas. Each of those would otherwise
    fail the stage and re-spend a paid call, so this is deliberately lenient -- but
    only on the *fallback* path, never at the cost of masking a real error.
    """
    if not raw:
        raise LLMError("empty completion", retryable=True)
    text = raw.strip()
    if m := _JSON_FENCE.search(text):
        text = m.group(1).strip()
    if not (text.startswith("{") or text.startswith("[")):
        starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
        if not starts:
            raise LLMError(f"no JSON found in completion: {text[:160]!r}", retryable=True)
        text = text[min(starts) :]
    text = _balance(text)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        try:
            return json.loads(_repair(text))
        except Exception:
            raise LLMError(f"invalid JSON ({exc}): {text[:200]!r}", retryable=True) from exc


def _balance(text: str) -> str:
    """Cut at the closing brace that completes the first value, or close what is open.

    Truncating matters as much as closing: a model that writes `{"a":1} Hope that
    helps!` is otherwise parsed as a syntax error on perfectly good data.
    """
    openers = {"{": "}", "[": "]"}
    closers = {v: k for k, v in openers.items()}
    stack: list[str] = []
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in openers:
            stack.append(openers[ch])
        elif ch in closers and stack and ch == stack[-1]:
            stack.pop()
            if not stack:
                return text[: i + 1]
        elif ch in closers and stack:
            stack.pop()
    if in_str:  # an unterminated string cannot be closed usefully
        return text
    return text + "".join(reversed(stack)) if stack else text


def _repair(text: str) -> str:
    """Last resort: strip // comments, quote bare keys, translate Python literals."""
    text = re.sub(r"(?<!:)//[^\n]*", "", text)
    text = re.sub(r"([{,]\s*)([A-Za-z_][\w-]*)(\s*:)", r'\1"\2"\3', text)
    literals = {"True": "true", "False": "false", "None": "null"}
    text = re.sub(r"\b(True|False|None)\b", lambda m: literals[m.group(1)], text)
    return text


class ChatLLM:
    def __init__(self, settings: LLMSettings, *, client: httpx.Client | None = None) -> None:
        self.settings = settings
        self._client = client

    @property
    def available(self) -> bool:
        return bool(self.settings.api_key)

    @property
    def model(self) -> str:
        return self.settings.model

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._client is not None:
            resp = self._client.post("/chat/completions", json=payload)
        else:
            with httpx.Client(
                base_url=self.settings.base_url.rstrip("/"),
                timeout=self.settings.timeout_s,
                headers={"Authorization": f"Bearer {self.settings.api_key}"},
            ) as client:
                resp = client.post("/chat/completions", json=payload)
        if resp.status_code in (408, 409, 425, 429) or resp.status_code >= 500:
            raise LLMError(f"chat completions HTTP {resp.status_code}", retryable=True)
        if resp.status_code >= 400:
            raise LLMError(f"chat completions HTTP {resp.status_code}: {resp.text[:300]}", retryable=False)
        return resp.json()

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        attempts: int = 3,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "temperature": self.settings.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.settings.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        last: Exception | None = None
        for attempt in range(attempts):
            try:
                data = self._post(payload)
                choice = (data.get("choices") or [{}])[0]
                content = ((choice.get("message") or {}).get("content") or "").strip()
                if not content:
                    raise LLMError(f"empty message: {json.dumps(choice)[:200]}")
                return content
            except LLMError as exc:
                last = exc
                if not exc.retryable:
                    break
                delay = min(8.0, (2**attempt) + random.random())
                log.warning("llm retry %s/%s in %.1fs: %s", attempt + 1, attempts, delay, exc)
                time.sleep(delay)
        raise LLMError(f"llm call failed: {last}")

    def complete_json(self, system: str, user: str, **kwargs: Any) -> Any:
        kwargs.setdefault("json_mode", True)
        return extract_json(self.complete(system, user, **kwargs))
