"""Load the candidate pool / curated suite and the generator configs (A / B)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from judge.checks import validate_spec

SUITE_PATH = Path("data/suite/candidates.yaml")
CONFIG_DIR = Path("data/configs")


class GenConfig(BaseModel):
    name: str
    prompt_version: str
    preamble: str

    def system_for(self, case_system_prompt: str) -> str:
        return f"{self.preamble}\n\n{case_system_prompt}".strip()


def load_config(name: str, config_dir: Path = CONFIG_DIR) -> GenConfig:
    return GenConfig(**yaml.safe_load((config_dir / f"{name}.yaml").read_text(encoding="utf-8")))


def load_candidates(path: Path = SUITE_PATH, *, include_dropped: bool = False) -> list[dict[str, Any]]:
    """Raw case dicts (no model_output yet). Cases with meta.drop: true are curated out unless include_dropped."""
    rows = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["cases"]
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case ids")
    for r in rows:
        checks = r.get("meta", {}).get("checks")
        if not checks:
            raise ValueError(f"{r['id']}: every case needs a checkable constraint in meta.checks")
        for spec in checks:
            validate_spec(spec)
    return rows if include_dropped else [r for r in rows if not r.get("meta", {}).get("drop")]


def summarize(rows: list[dict[str, Any]]) -> dict[str, int]:
    ref = sum(1 for r in rows if (r.get("expected_output") or "").strip())
    return {"cases": len(rows), "reference_based": ref, "reference_free": len(rows) - ref}
