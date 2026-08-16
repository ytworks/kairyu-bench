from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


class PreflightError(RuntimeError):
    """The supplied endpoint cannot satisfy the benchmark API contract."""


@dataclass(frozen=True)
class Endpoint:
    base_url: str

    @classmethod
    def parse(cls, supplied: str) -> "Endpoint":
        parts = urlsplit(supplied.strip())
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            raise ValueError("endpoint must be an absolute http(s) URL")
        if parts.username or parts.password:
            raise ValueError("endpoint credentials must be supplied via KAIRYU_API_KEY")
        if parts.query or parts.fragment:
            raise ValueError("endpoint must not contain a query string or fragment")

        path = parts.path.rstrip("/")
        chat_suffix = "/chat/completions"
        if path.endswith(chat_suffix):
            path = path[: -len(chat_suffix)]
        if not path.endswith("/v1"):
            path = f"{path}/v1" if path else "/v1"
        normalized = urlunsplit((parts.scheme, parts.netloc, path, "", ""))
        return cls(normalized)

    @property
    def models_url(self) -> str:
        return f"{self.base_url}/models"

    @property
    def anthropic_base_url(self) -> str:
        return self.base_url.removesuffix("/v1")

    @property
    def chat_url(self) -> str:
        return f"{self.base_url}/chat/completions"


class TargetClient:
    def __init__(
        self,
        endpoint: Endpoint,
        *,
        api_key: str | None = None,
        timeout: float = 30,
    ) -> None:
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = timeout

    def discover_chat_model(self) -> str:
        payload = self._request_json("GET", self.endpoint.models_url)
        data = payload.get("data")
        if not isinstance(data, list):
            raise PreflightError("models response does not contain a data list")

        model_ids: list[str] = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            model_id = entry.get("id")
            if isinstance(model_id, str) and model_id:
                model_ids.append(model_id)
        if not model_ids:
            raise PreflightError("models response contains no model IDs")

        failures: list[str] = []
        for model_id in model_ids:
            try:
                response = self._request_json(
                    "POST",
                    self.endpoint.chat_url,
                    {
                        "model": model_id,
                        "messages": [
                            {
                                "role": "user",
                                "content": "Reply with exactly OK.",
                            }
                        ],
                        "max_tokens": 8,
                        "temperature": 0,
                    },
                )
            except PreflightError as error:
                failures.append(f"{model_id}: {error}")
                continue
            choices = response.get("choices")
            if isinstance(choices, list) and choices:
                return model_id
            failures.append(f"{model_id}: response contains no choices")

        detail = "; ".join(failures)
        raise PreflightError(f"no advertised model accepted chat requests ({detail})")

    def chat(
        self,
        model_id: str,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float = 0,
    ) -> str:
        response = self._request_json(
            "POST",
            self.endpoint.chat_url,
            {
                "model": model_id,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise PreflightError("chat response contains no choices")
        first = choices[0]
        if not isinstance(first, dict) or not isinstance(first.get("message"), dict):
            raise PreflightError("chat response contains no assistant message")
        content = first["message"].get("content")
        if not isinstance(content, str):
            raise PreflightError("chat response assistant content is not text")
        return content

    def _request_json(
        self,
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        body = None
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if payload is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(payload).encode("utf-8")
        request = Request(url, data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                decoded = json.load(response)
        except HTTPError as error:
            raise PreflightError(f"HTTP {error.code}") from error
        except URLError as error:
            raise PreflightError(f"connection failed: {error.reason}") from error
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise PreflightError("endpoint returned invalid JSON") from error
        if not isinstance(decoded, dict):
            raise PreflightError("endpoint returned a non-object JSON response")
        return decoded
