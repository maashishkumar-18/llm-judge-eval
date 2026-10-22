"""Judge-side quota guards: a sliding-window request limiter and a persistent daily ledger.

* RateLimiter  - at most `rpm` HTTP requests in any 60 s window (applies to every attempt,
  including retries, because the provider counts each request).
* JudgeLedger  - persistent count of NON-CACHED judge calls per Pacific day (Gemini's daily quota
  resets at midnight Pacific) with a hard cap, plus a lifetime count per budget experiment.
  `reserve()` is called before the network request, so the cap can never be exceeded; cached
  calls never touch it. A reserved call that later fails is not refunded (conservative).
  Single-process only: asyncio is single-threaded and reserve() has no await, so it is atomic
  there, but two processes sharing one ledger file could race.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

PACIFIC = ZoneInfo("America/Los_Angeles")


class QuotaStop(RuntimeError):
    """Base: a run should stop cleanly (checkpointed) and resume later."""


class DailyCapReached(QuotaStop):
    pass


class ExperimentCapReached(QuotaStop):
    pass


def pacific_today() -> str:
    return datetime.now(PACIFIC).strftime("%Y-%m-%d")


class RateLimiter:
    """Sliding window: the (rpm+1)-th request inside 60 s waits until the oldest one ages out."""

    def __init__(self, rpm: int, *, clock: Callable[[], float] = time.monotonic, sleep=asyncio.sleep, window: float = 60.0):
        if rpm <= 0:
            raise ValueError("rpm must be positive")
        self.rpm, self.window = rpm, window
        self._clock, self._sleep = clock, sleep
        self._stamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:  # serialises waiters so ordering and the window stay exact
            while True:
                now = self._clock()
                while self._stamps and now - self._stamps[0] >= self.window:
                    self._stamps.popleft()
                if len(self._stamps) < self.rpm:
                    self._stamps.append(now)
                    return
                await self._sleep(self._stamps[0] + self.window - now)


class JudgeLedger:
    def __init__(
        self,
        path: str | Path,
        daily_cap: int,
        experiment_caps: dict[str, int] | None = None,
        *,
        today: Callable[[], str] = pacific_today,
    ):
        self.path = Path(path)
        self.daily_cap = daily_cap
        self.experiment_caps = experiment_caps or {}
        self._today = today
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict:
        if not self.path.exists():
            return {"days": {}, "experiments": {}}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raise RuntimeError(f"corrupt ledger {self.path}; refusing to guess usage (delete it only if you are sure)")

    def _save(self, state: dict) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)

    def used_today(self) -> int:
        return self._load()["days"].get(self._today(), 0)

    def used_experiment(self, experiment: str) -> int:
        return self._load()["experiments"].get(experiment, 0)

    def remaining_today(self) -> int:
        return max(0, self.daily_cap - self.used_today())

    def reserve(self, experiment: str | None = None) -> None:
        state = self._load()
        day = self._today()
        if state["days"].get(day, 0) >= self.daily_cap:
            raise DailyCapReached(
                f"daily judge cap {self.daily_cap} reached for {day} (Pacific); resume from checkpoint after midnight Pacific"
            )
        if experiment is not None:
            cap = self.experiment_caps.get(experiment)
            if cap is not None and state["experiments"].get(experiment, 0) >= cap:
                raise ExperimentCapReached(f"experiment {experiment!r} reached its cap of {cap} judge calls")
            state["experiments"][experiment] = state["experiments"].get(experiment, 0) + 1
        state["days"][day] = state["days"].get(day, 0) + 1
        self._save(state)
