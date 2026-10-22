"""Versioned prompt templates and their renderers.

Templates live in prompts/<name>.<version>.txt as "<system text>\\n---USER---\\n<user text>" with {{var}}
placeholders. Rendering is strict (a missing or unused variable raises) and single-pass, so a case that
itself contains "{{input}}" is not re-expanded. Bump VERSIONS when a template changes; the snapshot tests
in tests/test_prompts.py pin the rendered text.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

from .schemas import (
    CRITERIA,
    Anchor,
    Case,
    CriterionVerdict,
    PairCriterion,
    PairwiseVerdict,
    Rubric,
    load_anchors,
    load_rubric,
)

PROMPT_DIR = Path(__file__).with_name("prompts")
SEPARATOR = "\n---USER---\n"
VERSIONS = {
    "pointwise_ref": "v1",
    "pointwise_free": "v1",
    "pairwise": "v1",
    "naive_pointwise": "v1",
    "naive_pairwise": "v1",
}
_VAR = re.compile(r"\{\{(\w+)\}\}")


@dataclass(frozen=True)
class RenderedPrompt:
    template_id: str  # e.g. "pointwise_ref.v1"
    sha: str  # first 12 hex chars of the template file's SHA-256
    system: str
    user: str

    def as_snapshot(self) -> str:
        return f"[{self.template_id} sha={self.sha}]\n{self.system}\n=====USER=====\n{self.user}\n"


@lru_cache(maxsize=None)
def _load(name: str, version: str) -> tuple[str, str, str]:
    text = (PROMPT_DIR / f"{name}.{version}.txt").read_text(encoding="utf-8").replace("\r\n", "\n")
    system, user = text.split(SEPARATOR, 1)
    return system.strip(), user.rstrip("\n"), hashlib.sha256(text.encode()).hexdigest()[:12]


def _render(name: str, version: str | None, values: Mapping[str, str]) -> RenderedPrompt:
    version = version or VERSIONS[name]
    system, template, sha = _load(name, version)
    wanted = set(_VAR.findall(template))
    if wanted != set(values):
        raise ValueError(
            f"{name}.{version}: missing {sorted(wanted - set(values))}, unused {sorted(set(values) - wanted)}"
        )
    return RenderedPrompt(f"{name}.{version}", sha, system, _VAR.sub(lambda m: values[m.group(1)], template))


# ---------------------------------------------------------------- blocks

def render_rubric(rubric: Rubric, reference_based: bool) -> str:
    parts = []
    for c in rubric.criteria:
        how = c.how_to_judge_ref if reference_based else c.how_to_judge_free
        levels = "\n".join(f"{s} = {c.levels[s]}" for s in sorted(c.levels, reverse=True))
        parts.append(f"## {c.name}\nMeasures: {c.description}\nHow to judge: {how}\n{levels}")
    return "\n\n".join(parts)


def render_rules(rubric: Rubric) -> str:
    return "\n".join(f"- {r}" for r in rubric.rules)


def _case_criteria_block(case: Case) -> str:
    if not case.criteria:
        return ""
    bullets = "\n".join(f"- {c}" for c in case.criteria)
    return (
        "# Additional requirements for this case\n"
        "Treat these as part of the task when judging instruction_following and completeness:\n"
        f"{bullets}\n\n"
    )


def _anchors_block(anchors: Sequence[Anchor]) -> str:
    if not anchors:
        return ""
    out = [
        "# Calibration examples\n"
        "These are graded examples from OTHER cases. They show how to use the whole 1-5 scale consistently. "
        "Do not grade them and do not reuse their content.\n"
    ]
    for i, a in enumerate(anchors, 1):
        verdict = json.dumps(a.verdict.model_dump(), indent=2, ensure_ascii=False)
        reference = (
            f"<reference_answer>\n{a.case.expected_output}\n</reference_answer>\n\n" if a.case.reference_based else ""
        )
        out.append(
            f"## Example {i}\n<system_prompt>\n{a.case.system_prompt}\n</system_prompt>\n\n"
            f"<input>\n{a.case.input}\n</input>\n\n{reference}<answer>\n{a.case.model_output}\n</answer>\n\n"
            f"Correct grading:\n{verdict}\n"
        )
    return "\n".join(out) + "\n"


def _check_fields(model, expected: Sequence[str], what: str) -> None:
    if tuple(model.model_fields) != tuple(expected):
        raise ValueError(f"{what} placeholder keys drifted from {model.__name__}: {list(model.model_fields)}")


def pointwise_output_format() -> str:
    """The JSON shape the judge must return, generated from the schema so the two cannot drift."""
    crit = {
        "evidence": "<one verbatim quote from the answer, or NONE>",
        "rationale": "<one or two sentences tying the quote to a rubric level>",
        "score": "<integer 1-5>",
    }
    _check_fields(CriterionVerdict, crit, "pointwise")
    return json.dumps({c: crit for c in CRITERIA}, indent=2).replace('"<integer 1-5>"', "<integer 1-5>")


def pairwise_output_format() -> str:
    crit = {
        "evidence_a": "<verbatim quote from answer A, or NONE>",
        "evidence_b": "<verbatim quote from answer B, or NONE>",
        "rationale": "<one or two sentences comparing the two>",
        "winner": "<A or B or TIE>",
    }
    _check_fields(PairCriterion, crit, "pairwise criterion")
    shape = {c: crit for c in CRITERIA}
    shape.update(winner="<A or B or TIE>", rationale="<one sentence>")
    _check_fields(PairwiseVerdict, shape, "pairwise")
    return json.dumps(shape, indent=2)


# ---------------------------------------------------------------- renderers

def render_pointwise(
    case: Case,
    *,
    rubric: Rubric | None = None,
    anchors: Sequence[Anchor] | None = None,
    version: str | None = None,
) -> RenderedPrompt:
    """Mitigated pointwise prompt; reference-based when the case has an expected_output, else reference-free.

    anchors=None uses anchors.yaml; pass () to switch the anchors off. The reference-free prompt only shows
    reference-free anchors, so it never cites a reference answer the judge was not given.
    """
    rubric = rubric or load_rubric()
    anchors = load_anchors() if anchors is None else anchors
    if not case.reference_based:
        anchors = [a for a in anchors if not a.case.reference_based]
    values = {
        "rubric": render_rubric(rubric, case.reference_based),
        "rules": render_rules(rubric),
        "anchors": _anchors_block(anchors),
        "system_prompt": case.system_prompt,
        "input": case.input,
        "case_criteria": _case_criteria_block(case),
        "model_output": case.model_output,
        "output_format": pointwise_output_format(),
    }
    if case.reference_based:
        values["expected_output"] = case.expected_output
        return _render("pointwise_ref", version, values)
    return _render("pointwise_free", version, values)


def render_pairwise(
    case: Case,
    answer_a: str,
    answer_b: str,
    *,
    rubric: Rubric | None = None,
    version: str | None = None,
) -> RenderedPrompt:
    """Mitigated pairwise prompt. A/B are display positions; the caller swaps them for the second order."""
    rubric = rubric or load_rubric()
    reference = (
        f"<reference_answer>\n{case.expected_output}\n</reference_answer>\n\n" if case.reference_based else ""
    )
    return _render(
        "pairwise",
        version,
        {
            "rubric": render_rubric(rubric, case.reference_based),
            "rules": render_rules(rubric),
            "system_prompt": case.system_prompt,
            "input": case.input,
            "reference_block": reference,
            "case_criteria": _case_criteria_block(case),
            "answer_a": answer_a,
            "answer_b": answer_b,
            "output_format": pairwise_output_format(),
        },
    )


def _reference_line(case: Case) -> str:
    return f"Reference answer: {case.expected_output}\n\n" if case.reference_based else ""


def render_naive_pointwise(case: Case, *, version: str | None = None) -> RenderedPrompt:
    """Deliberately naive baseline: holistic 1-10, no rubric, anchors, length rule or evidence."""
    return _render(
        "naive_pointwise",
        version,
        {
            "system_prompt": case.system_prompt,
            "input": case.input,
            "reference_line": _reference_line(case),
            "model_output": case.model_output,
        },
    )


def render_naive_pairwise(case: Case, answer_a: str, answer_b: str, *, version: str | None = None) -> RenderedPrompt:
    return _render(
        "naive_pairwise",
        version,
        {
            "system_prompt": case.system_prompt,
            "input": case.input,
            "reference_line": _reference_line(case),
            "answer_a": answer_a,
            "answer_b": answer_b,
        },
    )
