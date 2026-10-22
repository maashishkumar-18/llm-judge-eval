"""Parse ladder for judge responses: strict -> extract -> repair -> one re-ask with the error -> INVALID.

Each rung is a separate function so the order is visible in code and each rung's behaviour is unit-tested
against a corpus of malformed outputs (tests/corpus.py). Rules:

* strict   the whole reply is one JSON object that validates against the schema.
* extract  the JSON object is somewhere inside the reply: <think> blocks and harmony channels removed, fences
           and prose ignored, balanced {...} spans tried (string-aware). Reasoning text before the JSON, which
           gpt-oss emits, lands here. If several spans validate, the LAST one wins (the final answer).
* repair   a span that is almost JSON is fixed by small, named, cumulative text repairs (comments, trailing
           commas, smart quotes, unescaped inner quotes, raw newlines, python literals, unquoted keys,
           truncation). Repairs only fix syntax: a missing field is never invented, it fails validation.
* reask    one more judge call that quotes the error. It is a real judge call (counts against the cap).
* INVALID  everything failed. The case is kept and counted, never dropped.

Validation is the pydantic schema, so an out-of-range score or a missing criterion is a parse failure too.
Unknown keys (e.g. a self-reported "overall") are ignored by the schema; the overall is recomputed in code.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Awaitable, Callable, TypeVar

from pydantic import BaseModel, ValidationError

M = TypeVar("M", bound=BaseModel)

MAX_REASK_ECHO_CHARS = 6000
MAX_START_POINTS = 25  # '{' positions tried per reply, so a pathological reply cannot blow up the work


class ParseStatus(StrEnum):
    STRICT = "strict"
    EXTRACT = "extract"
    REPAIR = "repair"
    REASK = "reask"
    INVALID = "invalid"


@dataclass
class Attempt:
    rung: str
    ok: bool
    error: str = ""
    detail: str = ""


@dataclass
class ParseResult:
    status: ParseStatus
    value: BaseModel | None
    attempts: list[Attempt] = field(default_factory=list)
    repairs: list[str] = field(default_factory=list)
    error: str = ""  # last error; "" unless INVALID
    reask_raw: str | None = None  # the re-ask reply, when a re-ask happened

    @property
    def ok(self) -> bool:
        return self.value is not None


class NotAnObject(ValueError):
    pass


def describe_error(exc: Exception) -> str:
    """Short, model-readable description of why a reply failed (goes into the re-ask prompt)."""
    if isinstance(exc, json.JSONDecodeError):
        return f"not valid JSON ({exc.msg} at line {exc.lineno} column {exc.colno})"
    if isinstance(exc, ValidationError):
        parts = []
        for e in exc.errors()[:6]:
            where = ".".join(str(p) for p in e["loc"]) or "(root)"
            parts.append(f"{where}: {e['msg']}")
        more = "" if len(exc.errors()) <= 6 else f" (+{len(exc.errors()) - 6} more)"
        return "JSON does not match the required fields: " + "; ".join(parts) + more
    return str(exc)


def _validate(model: type[M], obj: Any, *, unwrap: bool) -> M:
    if unwrap and isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
        obj = obj[0]  # [{...}] -> {...}
    if not isinstance(obj, dict):
        raise NotAnObject(f"top-level JSON is a {type(obj).__name__}, expected an object")
    return model.model_validate(obj)


# ---------------------------------------------------------------- rung 1: strict

def strict(raw: str, model: type[M]) -> M:
    return _validate(model, json.loads(raw.strip().lstrip("﻿")), unwrap=False)


# ---------------------------------------------------------------- rung 2: extract

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_PREFIX = re.compile(r"^.*?</think>", re.DOTALL)  # opening tag was cut off, closing tag survives
_HARMONY_FINAL = "<|channel|>final<|message|>"
_HARMONY_TOKEN = re.compile(r"<\|[a-z_]+\|>")


def strip_reasoning(text: str) -> str:
    """Drop reasoning wrappers: <think>...</think> and gpt-oss harmony channels (keep the final channel)."""
    text = _THINK_BLOCK.sub("", text)
    text = _THINK_PREFIX.sub("", text)
    if _HARMONY_FINAL in text:
        text = text.rsplit(_HARMONY_FINAL, 1)[1]
    return _HARMONY_TOKEN.sub("", text)


def _balanced_end(text: str, start: int) -> int | None:
    """Index of the '}' closing the '{' at `start`, ignoring braces inside double-quoted strings."""
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def json_spans(text: str) -> list[Any]:
    """Every parseable JSON object embedded in `text`, in order of appearance (outermost first)."""
    found, i, tried = [], 0, 0
    while tried < MAX_START_POINTS:
        i = text.find("{", i)
        if i < 0:
            break
        tried += 1
        j = _balanced_end(text, i)
        if j is not None:
            try:
                found.append(json.loads(text[i : j + 1]))
                i = j + 1
                continue
            except json.JSONDecodeError:
                pass
        i += 1
    return found


def extract(raw: str, model: type[M]) -> M:
    """Last embedded JSON object that validates. Raises the most informative error when none does."""
    text = strip_reasoning(raw)
    objs = json_spans(text)
    if not objs:
        try:  # a bare list at top level, e.g. [{...}]
            objs = [json.loads(text[text.index("[") : text.rindex("]") + 1])]
        except ValueError:
            raise NotAnObject("no JSON object found in the reply") from None
    errors: list[Exception] = []
    good = None
    for obj in objs:
        try:
            good = _validate(model, obj, unwrap=True)
        except (ValidationError, NotAnObject) as e:
            errors.append(e)
    if good is not None:
        return good
    # a schema complaint is more useful than "not an object"; the last object is the judge's final answer
    raise next((e for e in reversed(errors) if isinstance(e, ValidationError)), errors[-1])


# ---------------------------------------------------------------- rung 3: repair

def _split_strings(s: str) -> list[tuple[bool, str]]:
    """[(is_inside_double_quoted_string, chunk)]; the quotes belong to the string chunk."""
    out, buf, in_str, esc = [], [], False, False
    for ch in s:
        if in_str:
            buf.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                out.append((True, "".join(buf)))
                buf, in_str = [], False
        elif ch == '"':
            if buf:
                out.append((False, "".join(buf)))
            buf, in_str = ['"'], True
        else:
            buf.append(ch)
    if buf:
        out.append((in_str, "".join(buf)))
    return out


def _map_outside(s: str, fn: Callable[[str], str]) -> str:
    return "".join(chunk if inside else fn(chunk) for inside, chunk in _split_strings(s))


_COMMENTS = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")
_UNQUOTED_KEY = re.compile(r'([{,]\s*)([A-Za-z_][\w\- ]*?)(\s*:)')


def _fix_smart_quotes(s: str) -> str:
    if '"' in s:  # mixed use: curly quotes may be legitimate string content, leave them
        return s
    return s.replace("“", '"').replace("”", '"')


def _fix_comments_and_commas(s: str) -> str:
    s = _map_outside(s, lambda c: _COMMENTS.sub("", c))
    # a comma and the bracket it precedes can sit in the same outside chunk only; strings never contain either
    return _map_outside(s, lambda c: _TRAILING_COMMA.sub(r"\1", c))


def _fix_inner_quotes(s: str) -> str:
    """Escape a '"' inside a string unless it plausibly closes it (followed by , " | , } | } | ] | :)."""
    out, in_str, esc, i, n = [], False, False, 0, len(s)
    while i < n:
        ch = s[i]
        if not in_str:
            out.append(ch)
            if ch == '"':
                in_str = True
        elif esc:
            out.append(ch)
            esc = False
        elif ch == "\\":
            out.append(ch)
            esc = True
        elif ch == '"':
            rest = s[i + 1 :].lstrip()
            closes = (
                not rest
                or rest[0] in "}]:"
                or (rest[0] == "," and (re.match(r',\s*["}\]{\[\-\d]', rest) is not None or rest == ","))
            )
            if closes:
                out.append(ch)
                in_str = False
            else:
                out.append('\\"')
        else:
            out.append(ch)
        i += 1
    return "".join(out)


_CONTROL = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}


def _fix_control_chars(s: str) -> str:
    return "".join(
        "".join(_CONTROL.get(ch, ch) for ch in chunk) if inside else chunk for inside, chunk in _split_strings(s)
    )


def _fix_python_literals(s: str) -> str:
    """Single-quoted strings, True/False/None: let python parse it as a literal and dump real JSON."""
    try:
        return json.dumps(ast.literal_eval(s.strip()), ensure_ascii=False)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return s


def _fix_unquoted_keys(s: str) -> str:
    return _map_outside(s, lambda c: _UNQUOTED_KEY.sub(lambda m: f'{m.group(1)}"{m.group(2).strip()}"{m.group(3)}', c))


def _fix_truncation(s: str) -> str:
    """Close an unterminated string and any open objects/arrays. Never adds a key or a value."""
    stack, in_str, esc = [], False, False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    if not stack and not in_str:
        return s
    if in_str:
        s = s[:-1] if esc else s
        s += '"'
    s = s.rstrip()
    s = re.sub(r',?\s*"[^"]*"\s*:$', "", s)  # a key with no value: drop the key
    s = s.rstrip().rstrip(",")
    return s + "".join(reversed(stack))


REPAIR_STAGES: tuple[tuple[str, Callable[[str], str]], ...] = (
    ("smart_quotes", _fix_smart_quotes),
    ("comments_trailing_commas", _fix_comments_and_commas),
    ("unquoted_keys", _fix_unquoted_keys),
    ("inner_quotes", _fix_inner_quotes),
    ("control_chars", _fix_control_chars),
    ("python_literals", _fix_python_literals),
    ("truncation", _fix_truncation),
)


def _repair_targets(text: str) -> list[str]:
    """Candidate substrings to repair: from each '{' to its matching '}', to the last '}', and to the end."""
    targets: list[str] = []
    last = text.rfind("}")
    i, tried = 0, 0
    while tried < MAX_START_POINTS:
        i = text.find("{", i)
        if i < 0:
            break
        tried += 1
        j = _balanced_end(text, i)
        for t in (text[i : j + 1] if j is not None else None, text[i : last + 1] if last > i else None, text[i:]):
            if t and t not in targets:
                targets.append(t)
        i += 1
    return targets


def repair(raw: str, model: type[M]) -> tuple[M, list[str]]:
    """Apply the repair stages cumulatively, parsing after each one that changed the text.

    Returns (value, names of the stages that changed the text). Raises the error of the best attempt."""
    text = strip_reasoning(raw)
    last_err: Exception = NotAnObject("no JSON object found in the reply")
    for target in _repair_targets(text):
        cur, applied = target, []
        for name, fn in REPAIR_STAGES:
            new = fn(cur)
            if new == cur:
                continue
            cur = new
            applied.append(name)
            try:
                obj = json.loads(cur)
            except json.JSONDecodeError as e:
                if not isinstance(last_err, ValidationError):
                    last_err = e
                continue
            try:
                return _validate(model, obj, unwrap=True), applied
            except (ValidationError, NotAnObject) as e:
                last_err = e  # valid JSON, wrong content: more text repairs cannot fix that
                break
    raise last_err


# ---------------------------------------------------------------- the ladder

def parse_text(raw: str, model: type[M]) -> ParseResult:
    """Rungs 1-3 (no API call). status is INVALID when all three fail; the caller may then re-ask."""
    attempts: list[Attempt] = []
    errors: Exception | None = None
    for rung, status in (("strict", ParseStatus.STRICT), ("extract", ParseStatus.EXTRACT), ("repair", ParseStatus.REPAIR)):
        try:
            if rung == "strict":
                value, repairs = strict(raw, model), []
            elif rung == "extract":
                value, repairs = extract(raw, model), []
            else:
                value, repairs = repair(raw, model)
        except (json.JSONDecodeError, ValidationError, NotAnObject) as e:
            attempts.append(Attempt(rung, False, describe_error(e)))
            # keep the most useful complaint: a schema complaint beats a syntax one, and later rungs beat earlier
            if errors is None or isinstance(e, ValidationError) or not isinstance(errors, ValidationError):
                errors = e
            continue
        attempts.append(Attempt(rung, True, detail=",".join(repairs)))
        return ParseResult(status, value, attempts, repairs)
    assert errors is not None
    if not raw.strip():
        return ParseResult(ParseStatus.INVALID, None, attempts, [], "the reply was empty")
    msg = describe_error(errors)
    if not isinstance(errors, ValidationError):
        msg = f"no usable JSON object found in the reply ({msg})"
    return ParseResult(ParseStatus.INVALID, None, attempts, [], msg)


Reask = Callable[[str], Awaitable[str]]


async def parse_with_ladder(raw: str, model: type[M], *, reask: Reask | None = None) -> ParseResult:
    """Full ladder. `reask(error)` must make ONE new judge call quoting the error and return its raw text.

    The re-ask reply goes through rungs 1-3 again; if that parses the status is REASK, else INVALID.
    There is exactly one re-ask, never a loop."""
    first = parse_text(raw, model)
    if first.ok or reask is None:
        return first
    new_raw = await reask(first.error)
    second = parse_text(new_raw, model)
    attempts = first.attempts + [Attempt("reask", second.ok, second.error, detail=",".join(second.repairs))]
    if second.ok:
        return ParseResult(ParseStatus.REASK, second.value, attempts, second.repairs, "", new_raw)
    return ParseResult(
        ParseStatus.INVALID, None, attempts, [], f"{first.error} | after re-ask: {second.error}", new_raw
    )


REASK_CLOSING = {
    "pointwise": (
        "Grade the same case again and reply with ONLY the JSON object in the required format: "
        "no markdown fences, no text before or after, every field present, scores as integers 1-5."
    ),
    "pairwise": (
        "Compare the same two answers again and reply with ONLY the JSON object in the required format: "
        "no markdown fences, no text before or after, every field present, winners as \"A\", \"B\" or \"TIE\"."
    ),
}


def build_reask_prompt(original_user_prompt: str, previous_reply: str, error: str, kind: str = "pointwise") -> str:
    """The re-ask: the original task, the unusable reply (truncated), and the exact problem."""
    echoed = previous_reply.strip() or "(empty)"
    if len(echoed) > MAX_REASK_ECHO_CHARS:
        echoed = echoed[:MAX_REASK_ECHO_CHARS] + "\n[...truncated]"
    return (
        f"{original_user_prompt}\n\n"
        "# Correction needed\n"
        "Your previous reply could not be used.\n"
        f"Problem: {error}\n\n"
        f"Previous reply:\n<previous_reply>\n{echoed}\n</previous_reply>\n\n"
        f"{REASK_CLOSING[kind]}"
    )
