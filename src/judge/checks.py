"""Deterministic checks for the constraint each suite case puts in its system_prompt.

These are NOT the judge: they say whether a hard, countable constraint (sentence cap, bullet count, JSON keys, ...)
was met. They give by-construction labels for instruction-following and a sanity check on the generator outputs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

_BULLET = re.compile(r"^\s*[-*•]\s+")


@dataclass(frozen=True)
class CheckResult:
    type: str
    passed: bool
    detail: str = ""


def _lines(text: str) -> list[str]:
    return [ln for ln in text.strip().splitlines() if ln.strip()]


def _words(text: str) -> int:
    return len(text.split())


def _sentences(text: str) -> int:
    """Count sentence-ending punctuation runs, ignoring decimals like 2.50 and list numbering like '1.'."""
    flat = re.sub(r"(?m)^\s*(?:\d+\.|[-*•])\s+", "", text.strip())
    flat = re.sub(r"(?<=\d)[.,](?=\d)", "", flat)
    return len(re.findall(r"[.!?]+(?=\s|$)", flat)) or (1 if flat else 0)


def _max_sentences(out: str, n: int) -> tuple[bool, str]:
    k = _sentences(out)
    return k <= n, f"{k} sentences (max {n})"


def _max_words(out: str, n: int) -> tuple[bool, str]:
    k = _words(out)
    return k <= n, f"{k} words (max {n})"


def _min_words(out: str, n: int) -> tuple[bool, str]:
    k = _words(out)
    return k >= n, f"{k} words (min {n})"


def _exact_bullets(out: str, n: int) -> tuple[bool, str]:
    k = sum(1 for ln in _lines(out) if _BULLET.match(ln))
    return k == n, f"{k} bullets (need {n})"


def _exact_lines(out: str, n: int) -> tuple[bool, str]:
    k = len(_lines(out))
    return k == n, f"{k} lines (need {n})"


def _regex(out: str, pattern: str) -> tuple[bool, str]:
    return re.search(pattern, out.strip()) is not None, f"pattern {pattern!r}"


def _regex_dotall(out: str, pattern: str) -> tuple[bool, str]:
    return re.search(pattern, out.strip(), re.DOTALL) is not None, f"pattern {pattern!r} (dotall)"


def _all_lines_match(out: str, pattern: str) -> tuple[bool, str]:
    lines = _lines(out)
    bad = [ln for ln in lines if not re.search(pattern, ln.strip())]
    return bool(lines) and not bad, f"{len(bad)} of {len(lines)} lines do not match {pattern!r}"


def _one_of(out: str, options: list[str]) -> tuple[bool, str]:
    return out.strip() in options, f"output must be one of {options}"


def _json_keys(out: str, keys: list[str]) -> tuple[bool, str]:
    try:
        obj = json.loads(out.strip())
    except json.JSONDecodeError as e:
        return False, f"not valid JSON: {e.msg}"
    if not isinstance(obj, dict):
        return False, "JSON is not an object"
    return set(obj) == set(keys), f"keys {sorted(obj)} (need {sorted(keys)})"


def _starts_with(out: str, text: str) -> tuple[bool, str]:
    return out.strip().startswith(text), f"must start with {text!r}"


def _all_lowercase(out: str) -> tuple[bool, str]:
    return out == out.lower(), "must be all lowercase"


def _must_include(out: str, words: list[str]) -> tuple[bool, str]:
    flat = re.sub(r"\s+", " ", out).lower()  # the generator emits U+202F between a number and its unit
    missing = [w for w in words if w.lower() not in flat]
    return not missing, f"missing {missing}" if missing else "all present"


def _must_not_include(out: str, words: list[str]) -> tuple[bool, str]:
    found = [w for w in words if re.search(rf"\b{re.escape(w.lower())}\b", out.lower())]
    return not found, f"forbidden {found}" if found else "none present"


CHECKS: dict[str, Callable[..., tuple[bool, str]]] = {
    "max_sentences": _max_sentences, "max_words": _max_words, "min_words": _min_words,
    "exact_bullets": _exact_bullets, "exact_lines": _exact_lines,
    "regex": _regex, "regex_dotall": _regex_dotall, "all_lines_match": _all_lines_match,
    "one_of": _one_of, "json_keys": _json_keys, "starts_with": _starts_with, "all_lowercase": _all_lowercase,
    "must_include": _must_include, "must_not_include": _must_not_include,
}


def validate_spec(spec: dict[str, Any]) -> None:
    """Raise ValueError if a check spec names an unknown type or the wrong parameters."""
    spec = dict(spec)
    kind = spec.pop("type", None)
    if kind not in CHECKS:
        raise ValueError(f"unknown check type {kind!r}")
    fn_params = list(CHECKS[kind].__code__.co_varnames[1:CHECKS[kind].__code__.co_argcount])
    if sorted(spec) != sorted(fn_params):
        raise ValueError(f"check {kind!r} needs params {fn_params}, got {sorted(spec)}")


def run_checks(output: str, specs: list[dict[str, Any]]) -> list[CheckResult]:
    results = []
    for spec in specs:
        validate_spec(spec)
        params = {k: v for k, v in spec.items() if k != "type"}
        ok, detail = CHECKS[spec["type"]](output, **params)
        results.append(CheckResult(spec["type"], ok, detail))
    return results


def complies(output: str, specs: list[dict[str, Any]]) -> bool:
    return all(r.passed for r in run_checks(output, specs))
