"""Checkpointed batch runner: finished items are appended to a JSONL file, so a run that stops at a
quota cap (or crashes) resumes by skipping everything already done."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from judge.ledger import QuotaStop


@dataclass
class RunStatus:
    results: dict[str, Any] = field(default_factory=dict)
    skipped: int = 0  # already in the checkpoint
    ran: int = 0
    remaining: list[str] = field(default_factory=list)
    stopped_reason: str | None = None

    @property
    def complete(self) -> bool:
        return not self.remaining


async def run_checkpointed(
    keys: list[str],
    fn: Callable[[str], Awaitable[Any]],
    path: str | Path,
    *,
    concurrency: int = 2,
) -> RunStatus:
    """Run fn(key) for every key not yet in the checkpoint. fn results must be JSON-serialisable.
    On a QuotaStop (daily/experiment cap) no new items start; the status says what is left."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    status = RunStatus()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                status.results[row["key"]] = row["result"]
    todo = [k for k in keys if k not in status.results]
    status.skipped = len(keys) - len(todo)

    sem = asyncio.Semaphore(concurrency)
    stop: dict[str, str] = {}
    pending: set[str] = set(todo)

    async def one(key: str) -> None:
        async with sem:
            if stop:
                return
            try:
                result = await fn(key)
            except QuotaStop as e:
                stop.setdefault("reason", str(e))
                return
            status.results[key] = result
            status.ran += 1
            pending.discard(key)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "result": result}, ensure_ascii=False) + "\n")

    await asyncio.gather(*(one(k) for k in todo))
    status.remaining = [k for k in todo if k in pending]
    status.stopped_reason = stop.get("reason")
    return status
