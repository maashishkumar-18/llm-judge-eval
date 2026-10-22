"""Pydantic models: test cases, rubric, judge verdicts, anchors.

The overall score is never taken from the judge: it is recomputed here from the per-criterion scores and
the rubric weights. Unknown keys in a judge response (including a self-reported overall) are ignored.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

CRITERIA = ("faithfulness", "instruction_following", "completeness")
SCORE_MIN, SCORE_MAX = 1, 5
RUBRIC_PATH = Path(__file__).with_name("rubric.yaml")
ANCHORS_PATH = Path(__file__).with_name("anchors.yaml")

_EPS = 1e-9  # (4, 4, 2) is exactly 3.5 on paper but not in floating point


class Case(BaseModel):
    """One test case: what the generator was asked and what it produced."""

    id: str
    input: str
    system_prompt: str = ""
    model_output: str
    expected_output: str | None = None
    criteria: list[str] = Field(default_factory=list)  # extra case-specific requirements
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("expected_output")
    @classmethod
    def _blank_reference_is_none(cls, v: str | None) -> str | None:
        return v if v and v.strip() else None

    @property
    def reference_based(self) -> bool:
        return self.expected_output is not None


# ---------------------------------------------------------------- rubric

class CriterionSpec(BaseModel):
    name: str
    weight: float = Field(gt=0)
    description: str
    how_to_judge_ref: str
    how_to_judge_free: str
    levels: dict[int, str]

    @model_validator(mode="after")
    def _levels_cover_scale(self) -> CriterionSpec:
        if set(self.levels) != set(range(SCORE_MIN, SCORE_MAX + 1)):
            raise ValueError(f"{self.name}: levels must be exactly {SCORE_MIN}..{SCORE_MAX}")
        return self


class PassRule(BaseModel):
    faithfulness_min: int
    instruction_following_min: int
    overall_min: float


class Rubric(BaseModel):
    version: str
    rules: list[str]
    pass_rule: PassRule
    criteria: list[CriterionSpec]

    @model_validator(mode="after")
    def _matches_schema(self) -> Rubric:
        if tuple(c.name for c in self.criteria) != CRITERIA:
            raise ValueError(f"rubric criteria must be {CRITERIA} in that order")
        if abs(sum(c.weight for c in self.criteria) - 1.0) > 1e-6:
            raise ValueError("rubric weights must sum to 1.0")
        return self

    @property
    def weights(self) -> dict[str, float]:
        return {c.name: c.weight for c in self.criteria}

    def overall(self, scores: Mapping[str, int]) -> float:
        """sum(w_i * score_i) / sum(w_i) on the 1-5 scale."""
        w = self.weights
        return sum(w[c] * scores[c] for c in CRITERIA) / sum(w.values())

    def passes(self, scores: Mapping[str, int]) -> bool:
        r = self.pass_rule
        return (
            scores["faithfulness"] >= r.faithfulness_min
            and scores["instruction_following"] >= r.instruction_following_min
            and self.overall(scores) + _EPS >= r.overall_min
        )


@lru_cache(maxsize=None)
def load_rubric(path: str | Path = RUBRIC_PATH) -> Rubric:
    return Rubric.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------- evidence

_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})


def _norm(text: str) -> str:
    return " ".join(text.translate(_QUOTES).lower().split())


def is_none_evidence(evidence: str) -> bool:
    return evidence.strip().strip(".").upper() == "NONE"


def evidence_is_verbatim(evidence: str, text: str) -> bool:
    """True if every fragment of `evidence` (split at "..." / the ellipsis character) occurs in `text`.

    Case, whitespace and curly-vs-straight quotes are ignored; NONE is never verbatim.
    """
    if is_none_evidence(evidence):
        return False
    hay = _norm(text)
    fragments = [_norm(f).strip(" \"'") for f in re.split(r"\.\.\.|…", evidence)]
    fragments = [f for f in fragments if f]
    return bool(fragments) and all(f in hay for f in fragments)


# ---------------------------------------------------------------- verdicts

class CriterionVerdict(BaseModel):
    """Field order matters: evidence and rationale come before the score the judge commits to."""

    evidence: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    score: int = Field(ge=SCORE_MIN, le=SCORE_MAX)

    @field_validator("evidence", "rationale", mode="before")
    @classmethod
    def _strip(cls, v: Any) -> Any:
        return v.strip() if isinstance(v, str) else v


class PointwiseVerdict(BaseModel):
    faithfulness: CriterionVerdict
    instruction_following: CriterionVerdict
    completeness: CriterionVerdict

    @property
    def scores(self) -> dict[str, int]:
        return {c: getattr(self, c).score for c in CRITERIA}

    def overall(self, rubric: Rubric | None = None) -> float:
        return (rubric or load_rubric()).overall(self.scores)

    def passes(self, rubric: Rubric | None = None) -> bool:
        return (rubric or load_rubric()).passes(self.scores)

    def unsupported_high_scores(self, model_output: str) -> list[str]:
        """Criteria scored >= 4 whose evidence is not a verbatim span of the answer (sycophancy check)."""
        return [
            c
            for c in CRITERIA
            if getattr(self, c).score >= 4 and not evidence_is_verbatim(getattr(self, c).evidence, model_output)
        ]


def _upper_winner(v: Any) -> Any:
    return v.strip().upper() if isinstance(v, str) else v


Winner = Literal["A", "B", "TIE"]  # positions as shown to the judge, not configs


class PairCriterion(BaseModel):
    evidence_a: str = Field(min_length=1)
    evidence_b: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    winner: Winner

    _winner = field_validator("winner", mode="before")(_upper_winner)


class PairwiseVerdict(BaseModel):
    faithfulness: PairCriterion
    instruction_following: PairCriterion
    completeness: PairCriterion
    winner: Winner
    rationale: str = Field(min_length=1)

    _winner = field_validator("winner", mode="before")(_upper_winner)


class NaiveVerdict(BaseModel):
    """Baseline judge: one holistic 1-10 number and a reason."""

    score: int = Field(ge=1, le=10)
    reason: str = ""


class NaivePairVerdict(BaseModel):
    winner: Winner
    reason: str = ""

    _winner = field_validator("winner", mode="before")(_upper_winner)


# ---------------------------------------------------------------- anchors

class Anchor(BaseModel):
    """A graded few-shot example. Must come from outside the eval suite."""

    id: str
    case: Case
    verdict: PointwiseVerdict

    @model_validator(mode="after")
    def _evidence_obeys_own_rule(self) -> Anchor:
        # every quoted span must be verbatim, whatever its score (NONE is allowed)
        bad = [
            c
            for c in CRITERIA
            if not is_none_evidence(getattr(self.verdict, c).evidence)
            and not evidence_is_verbatim(getattr(self.verdict, c).evidence, self.case.model_output)
        ]
        if bad:
            raise ValueError(f"anchor {self.id}: evidence not verbatim in the answer for {bad}")
        return self


@lru_cache(maxsize=None)
def load_anchors(path: str | Path = ANCHORS_PATH) -> tuple[Anchor, ...]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    anchors = tuple(Anchor.model_validate(a) for a in raw["anchors"])
    ids = [a.id for a in anchors]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate anchor ids")
    return anchors
