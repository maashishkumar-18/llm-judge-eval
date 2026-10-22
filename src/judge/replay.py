"""Replay judge verdicts from a call log with zero API calls.

ReplayClient is a drop-in for LLMClient.complete(): it looks the request up in a JSONL call log by
(system, prompt, nonce) and returns the logged raw response. The pointwise code path (render prompt ->
parse ladder -> re-ask -> recompute overall) is the SAME code that ran live, so a replay reproduces the
verdicts exactly, including a re-ask (its prompt is a pure function of the first reply and the error)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ReplayMiss(LookupError):
    """The log has no row for this request (the live run never made it, or the prompt changed)."""


def call_key(system: str | None, prompt: str, nonce: str = "") -> str:
    blob = json.dumps({"system": system or "", "prompt": prompt, "nonce": nonce}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def read_log(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


class ReplayClient:
    """Serves logged raw responses. Has no HTTP client, no ledger and no keys: it cannot call an API."""

    def __init__(self, log_path: str | Path):
        self.rows: dict[str, dict[str, Any]] = {}
        for row in read_log(log_path):
            if row.get("role") == "judge":
                self.rows.setdefault(call_key(row.get("system"), row["prompt"], row.get("nonce", "")), row)
        self.served = 0

    async def complete(
        self, prompt: str, *, case_id: str = "", system: str | None = None, nonce: str = "", **_ignored: Any
    ) -> dict[str, Any]:
        row = self.rows.get(call_key(system, prompt, nonce))
        if row is None:
            raise ReplayMiss(f"no logged response for case {case_id!r} (prompt hash {hashlib.sha256(prompt.encode()).hexdigest()[:12]})")
        self.served += 1
        return {
            "text": row["raw_response"], "reasoning": row.get("reasoning", ""),
            "tokens_in": row["tokens_in"], "tokens_out": row["tokens_out"], "cached": True,
        }
