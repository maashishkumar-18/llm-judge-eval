"""Call budget: load budget.yaml, compute planned calls, refuse runs that would exceed a cap."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).with_name("budget.yaml")


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class Experiment:
    name: str
    desc: str
    planned: int
    cap: int


@dataclass(frozen=True)
class Budget:
    experiments: dict[str, Experiment]

    @property
    def total_planned(self) -> int:
        return sum(e.planned for e in self.experiments.values())

    @property
    def total_cap(self) -> int:
        return sum(e.cap for e in self.experiments.values())

    @property
    def caps(self) -> dict[str, int]:
        return {n: e.cap for n, e in self.experiments.items()}

    def get(self, name: str) -> Experiment:
        if name not in self.experiments:
            raise KeyError(f"unknown experiment {name!r}; known: {sorted(self.experiments)}")
        return self.experiments[name]

    def validate(self) -> list[str]:
        """Config errors: any experiment whose own plan already exceeds its cap."""
        return [f"{e.name}: planned {e.planned} > cap {e.cap}" for e in self.experiments.values() if e.planned > e.cap]

    def check_run(self, name: str, calls: int, spent: int = 0) -> None:
        """Refuse (raise) if `calls` more non-cached judge calls would push `name` past its cap."""
        e = self.get(name)
        if spent + calls > e.cap:
            raise BudgetExceeded(
                f"refused: {name} would use {spent}+{calls}={spent + calls} judge calls, cap is {e.cap}"
            )


def load_budget(path: str | Path = DEFAULT_PATH) -> Budget:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    exps: dict[str, Experiment] = {}
    for name, spec in raw["experiments"].items():
        planned = sum(math.prod(term) for term in spec.get("terms", []))
        exps[name] = Experiment(name, spec["desc"], planned, int(spec["cap"]))
    return Budget(exps)
