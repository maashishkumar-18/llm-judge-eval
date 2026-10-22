"""Disk cache keyed by hash(model, params, prompt, nonce).

The nonce is part of the key: a different nonce is a guaranteed miss, which is how test-retest
runs get real (non-cached) judge calls."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def make_key(model: str, params: dict[str, Any], payload: str, nonce: str = "") -> str:
    blob = json.dumps(
        {"model": model, "params": params, "payload": payload, "nonce": nonce}, sort_keys=True
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


class DiskCache:
    def __init__(self, cache_dir: str | Path):
        self.root = Path(cache_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None  # torn write: treat as a miss

    def set(self, key: str, value: dict[str, Any]) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(value), encoding="utf-8")
        os.replace(tmp, p)  # atomic: a crash never leaves a half-written entry

    def __contains__(self, key: str) -> bool:
        return self._path(key).exists()
