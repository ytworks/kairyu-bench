"""Atomic JSON artifacts, hashing and credential redaction shared by trial harnesses."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

PLACEHOLDER_SECRETS = {"", "not-required"}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read {label} artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} artifact must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def redact_artifacts(
    raw_dir: Path,
    keys: Iterable[str] = ("OPENAI_API_KEY", "HF_TOKEN"),
    env: Mapping[str, str] = os.environ,
) -> None:
    # Retain trajectories and official outputs, with credential literals removed.
    secrets = [
        value.encode()
        for value in (env.get(key, "") for key in keys)
        if value not in PLACEHOLDER_SECRETS
    ]
    if not secrets:
        return
    for path in raw_dir.rglob("*"):
        if path.is_file() and not path.is_symlink():
            content = path.read_bytes()
            redacted = content
            for secret in secrets:
                redacted = redacted.replace(secret, b"[REDACTED]")
            if redacted != content:
                path.write_bytes(redacted)


def env_integer(
    env: Mapping[str, str], key: str, default: int, low: int, high: int
) -> int:
    text = env.get(key, str(default))
    if not re.fullmatch(r"[0-9]+", text) or not low <= int(text) <= high:
        raise ValueError(f"{key} must be an integer from {low} to {high}")
    return int(text)
