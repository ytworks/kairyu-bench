"""Thread-safe HTTP calls: chat through the evaluated API, prompt logprobs
directly from vLLM replicas (the API under test may not expose them)."""

from __future__ import annotations

import itertools
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Connection failures and 5xx answers are infrastructure accidents and are
# retried; any 4xx is the model or request and is not.
RETRY_DELAYS_S = (2.0, 8.0, 30.0)


class ApiError(RuntimeError):
    """A request failed after its retries."""


def _post(url: str, payload: dict[str, Any], *, api_key: str | None, timeout: float) -> dict[str, Any]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = json.dumps(payload).encode("utf-8")
    last: Exception | None = None
    from kairyu_bench.wsp.observe.tracing import begin_request

    call_id = uuid.uuid4().hex
    for attempt_index, delay in enumerate((*RETRY_DELAYS_S, None)):
        observation = begin_request(url, payload, call_id=call_id, attempt_index=attempt_index)
        request_headers = {**headers, **(observation.headers if observation else {})}
        try:
            with urlopen(Request(url, data=body, headers=request_headers, method="POST"), timeout=timeout) as response:
                decoded = json.load(response)
            if not isinstance(decoded, dict):
                raise ApiError(f"{url}: non-object response")
            if observation:
                observation.finish(response=decoded)
            return decoded
        except ApiError as error:
            if observation:
                observation.finish(error=type(error).__name__)
            raise
        except HTTPError as error:
            if observation:
                observation.finish(error=f"HTTP {error.code}")
            detail = error.read().decode("utf-8", "replace")[:500]
            if error.code < 500:
                raise ApiError(f"{url}: HTTP {error.code}: {detail}") from error
            last = ApiError(f"{url}: HTTP {error.code}: {detail}")
        except (URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as error:
            if observation:
                observation.finish(error=type(error).__name__)
            last = error
        if delay is None:
            break
        time.sleep(delay)
    raise ApiError(f"{url}: {last}")


@dataclass
class ChatReply:
    content: str
    reasoning: str
    tool_calls: list[dict[str, Any]]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float


@dataclass
class ChatClient:
    base_url: str
    model: str
    api_key: str | None = None
    timeout: float = 7200.0
    # Requests in flight at once, across every thread (None: unbounded).
    concurrency: int | None = None
    calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _slots: threading.BoundedSemaphore | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.concurrency is not None:
            self._slots = threading.BoundedSemaphore(self.concurrency)

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        seed: int,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatReply:
        # Sampling fields are omitted on purpose: the server applies the
        # checkpoint's official generation_config.
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        if tools:
            payload["tools"] = tools
        if self._slots is not None:
            self._slots.acquire()
        try:
            started = time.monotonic()
            response = _post(
                f"{self.base_url}/chat/completions", payload, api_key=self.api_key, timeout=self.timeout
            )
        finally:
            if self._slots is not None:
                self._slots.release()
        with self._lock:
            self.calls += 1
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ApiError("chat response contains no choices")
        choice = choices[0]
        raw_message, raw_usage = choice.get("message"), response.get("usage")
        message: dict[str, Any] = raw_message if isinstance(raw_message, dict) else {}
        usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
        return ChatReply(
            content=message.get("content") or "",
            reasoning=message.get("reasoning_content") or message.get("reasoning") or "",
            tool_calls=list(message.get("tool_calls") or []),
            finish_reason=str(choice.get("finish_reason") or ""),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            seconds=time.monotonic() - started,
        )


@dataclass
class LogprobClient:
    """Prompt logprobs from one or more vLLM servers (round robin)."""

    base_urls: tuple[str, ...]
    model: str
    timeout: float = 600.0
    calls: int = 0
    _cycle: Any = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if not self.base_urls:
            raise ValueError("at least one logprob endpoint is required")
        self._cycle = itertools.cycle(self.base_urls)

    def _next(self) -> str:
        with self._lock:
            self.calls += 1
            return str(next(self._cycle))

    def tokenize(self, text: str, *, add_special_tokens: bool) -> list[int]:
        root = self._next().removesuffix("/v1")
        response = _post(
            f"{root}/tokenize",
            {"model": self.model, "prompt": text, "add_special_tokens": add_special_tokens},
            api_key=None,
            timeout=self.timeout,
        )
        tokens = response.get("tokens")
        if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
            raise ApiError("tokenize response contains no token ids")
        return tokens

    def prompt_logprobs(
        self, token_ids: list[int], *, top_k: int
    ) -> list[dict[int, tuple[float, int]] | None]:
        """Per position, ``token -> (logprob, rank)`` for the top-k and the
        actual token (None at position 0)."""

        response = _post(
            f"{self._next()}/completions",
            {
                "model": self.model,
                "prompt": token_ids,
                "max_tokens": 1,
                "prompt_logprobs": top_k,
            },
            api_key=None,
            timeout=self.timeout,
        )
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ApiError("completion response contains no choices")
        positions = choices[0].get("prompt_logprobs")
        if not isinstance(positions, list) or len(positions) != len(token_ids):
            raise ApiError("prompt_logprobs do not cover the prompt")
        return [
            None
            if entry is None
            else {int(token): (float(value["logprob"]), int(value["rank"])) for token, value in entry.items()}
            for entry in positions
        ]
