"""Corpus of malformed judge replies. Each sample states the parse status the ladder must reach and, for
samples that parse, the scores it must return. `reask` samples are replies no text repair may rescue: they
have to go to the re-ask rung (and INVALID if that fails too)."""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Sample:
    name: str
    raw: str
    status: str  # strict | extract | repair | needs_reask
    scores: tuple[int, int, int] | None = None  # faithfulness, instruction_following, completeness
    repairs: tuple[str, ...] = ()  # stage names that must appear (repair samples)


def crit(score: int, evidence: str = "The fine is $2.50.", rationale: str = "Matches the reference.") -> dict:
    return {"evidence": evidence, "rationale": rationale, "score": score}


def good(f=5, i=4, c=3, **kw) -> dict:
    return {
        "faithfulness": crit(f, **kw), "instruction_following": crit(i, **kw), "completeness": crit(c, **kw),
    }


GOOD_JSON = json.dumps(good(), indent=2)
GOOD_SCORES = (5, 4, 3)
GOOD_COMPACT = json.dumps(good(), separators=(",", ":"))

# a draft in the reasoning with different scores, then the final answer
DRAFT = json.dumps(good(2, 2, 2))
FINAL = json.dumps(good(4, 3, 5))

SAMPLES: tuple[Sample, ...] = (
    # ---- rung 1: strict
    Sample("clean", GOOD_JSON, "strict", GOOD_SCORES),
    Sample("compact", GOOD_COMPACT, "strict", GOOD_SCORES),
    Sample("bom_and_whitespace", "﻿\n  " + GOOD_JSON + "\n\n", "strict", GOOD_SCORES),
    Sample("self_reported_overall_is_ignored", json.dumps({**good(), "overall": 1.0}), "strict", GOOD_SCORES),
    Sample("scores_as_digit_strings", json.dumps(good("5", "4", "3")), "strict", GOOD_SCORES),
    Sample("integral_float_scores", json.dumps(good(5.0, 4.0, 3.0)), "strict", GOOD_SCORES),
    # ---- rung 2: extract (valid JSON, wrapped in other text)
    Sample("json_fence", f"```json\n{GOOD_JSON}\n```", "extract", GOOD_SCORES),
    Sample("bare_fence", f"```\n{GOOD_JSON}\n```", "extract", GOOD_SCORES),
    Sample("prose_before_and_after", f"Here is my grading:\n{GOOD_JSON}\nHope this helps!", "extract", GOOD_SCORES),
    Sample(
        "gpt_oss_reasoning_before_json",
        "We need to evaluate the answer. The reference says $2.50. Let's check each criterion: "
        "faithfulness is {evidence} based, instruction-following fine. Now produce the JSON.\n" + GOOD_JSON,
        "extract", GOOD_SCORES,
    ),
    Sample(
        "reasoning_with_unbalanced_brace",
        "Plan: output a { object with three keys, then fill it in.\n" + GOOD_JSON,
        "extract", GOOD_SCORES,
    ),
    Sample("think_block", f"<think>Hmm {DRAFT} maybe not.</think>\n{FINAL}", "extract", (4, 3, 5)),
    Sample("think_opening_tag_lost", f"I should quote carefully {DRAFT}</think>{FINAL}", "extract", (4, 3, 5)),
    Sample(
        "harmony_channels_leaked",
        f"<|channel|>analysis<|message|>draft {DRAFT}<|end|><|start|>assistant<|channel|>final<|message|>{FINAL}<|return|>",
        "extract", (4, 3, 5),
    ),
    Sample("draft_then_final_last_wins", f"First attempt:\n{DRAFT}\nActually, final answer:\n{FINAL}", "extract", (4, 3, 5)),
    Sample("braces_inside_strings", json.dumps(good(evidence="the set {a, b} and a } stray")) + " done", "extract", GOOD_SCORES),
    Sample("wrapped_in_list", f"[{GOOD_JSON}]", "extract", GOOD_SCORES),
    Sample("prose_prefix_and_suffix_compact", "Result: " + GOOD_COMPACT + " (end)", "extract", GOOD_SCORES),
    # ---- rung 3: repair (almost JSON)
    Sample("trailing_comma", GOOD_JSON.replace('"score": 3\n  }', '"score": 3,\n  }'), "repair", GOOD_SCORES, ("comments_trailing_commas",)),
    Sample(
        "trailing_commas_everywhere",
        '{"faithfulness": {"evidence": "x", "rationale": "y", "score": 5,},'
        ' "instruction_following": {"evidence": "x", "rationale": "y", "score": 4,},'
        ' "completeness": {"evidence": "x", "rationale": "y", "score": 3,},}',
        "repair", GOOD_SCORES, ("comments_trailing_commas",),
    ),
    Sample(
        "comments",
        '{\n // faithfulness first\n "faithfulness": {"evidence": "see http://a.b/c", "rationale": "y", "score": 5}, /* ok */\n'
        ' "instruction_following": {"evidence": "x", "rationale": "y", "score": 4},\n'
        ' "completeness": {"evidence": "x", "rationale": "y", "score": 3}}',
        "repair", GOOD_SCORES, ("comments_trailing_commas",),
    ),
    Sample(
        "single_quotes_python_dict",
        str(good()),
        "repair", GOOD_SCORES, ("python_literals",),
    ),
    Sample(
        "unquoted_keys",
        '{faithfulness: {evidence: "x", rationale: "y", score: 5}, instruction_following: {evidence: "x", '
        'rationale: "y", score: 4}, completeness: {evidence: "x", rationale: "y", score: 3}}',
        "repair", GOOD_SCORES, ("unquoted_keys",),
    ),
    Sample("smart_quote_delimiters", GOOD_COMPACT.replace('"', "”"), "repair", GOOD_SCORES, ("smart_quotes",)),
    Sample(
        "raw_newline_in_string",
        GOOD_COMPACT.replace("The fine is $2.50.", "line one\nline two", 1),
        "repair", GOOD_SCORES, ("control_chars",),
    ),
    Sample(
        "unescaped_quotes_in_evidence",
        '{"faithfulness": {"evidence": "The notice says "no phone number" is listed", "rationale": "Quotes "NONE" wrongly.", "score": 5},'
        ' "instruction_following": {"evidence": "x", "rationale": "y", "score": 4},'
        ' "completeness": {"evidence": "x", "rationale": "y", "score": 3}}',
        "repair", GOOD_SCORES, ("inner_quotes",),
    ),
    Sample("truncated_closing_braces", GOOD_JSON.rstrip()[:-3], "repair", GOOD_SCORES, ("truncation",)),
    Sample("truncated_after_last_score_only_brace_missing", GOOD_COMPACT[:-1], "repair", GOOD_SCORES, ("truncation",)),
    Sample("fenced_and_trailing_comma", "```json\n" + GOOD_JSON.replace('"score": 3\n  }', '"score": 3,\n  }') + "\n```", "repair", GOOD_SCORES),
    Sample("reasoning_then_python_style", "Thinking...\n" + str(good()), "repair", GOOD_SCORES),
    # ---- rung 4: nothing local can fix these
    Sample("missing_criterion", json.dumps({k: v for k, v in good().items() if k != "completeness"}), "needs_reask"),
    Sample("score_out_of_range", json.dumps(good(7, 4, 3)), "needs_reask"),
    Sample("score_zero", json.dumps(good(0, 4, 3)), "needs_reask"),
    Sample("score_is_a_word", json.dumps(good("high", 4, 3)), "needs_reask"),
    Sample("fractional_score", json.dumps(good(4.5, 4, 3)), "needs_reask"),
    Sample("score_missing_in_a_criterion", json.dumps({**good(), "faithfulness": {"evidence": "x", "rationale": "y"}}), "needs_reask"),
    Sample("empty_evidence", json.dumps(good(evidence="")), "needs_reask"),
    Sample("truncated_mid_rationale", GOOD_COMPACT[: GOOD_COMPACT.rindex('"rationale":"') + 20], "needs_reask"),
    Sample("empty_reply", "", "needs_reask"),
    Sample("whitespace_only", " \n\t ", "needs_reask"),
    Sample("pure_prose_refusal", "I'm sorry, but I can't grade this answer because it is incomplete.", "needs_reask"),
    Sample("prose_with_scores_not_json", "Faithfulness: 5, instruction following: 4, completeness: 3.", "needs_reask"),
    Sample("json_null", "null", "needs_reask"),
    Sample("json_string", '"faithfulness 5"', "needs_reask"),
    Sample("only_inner_objects_no_wrapper", json.dumps(crit(5)) + json.dumps(crit(4)) + json.dumps(crit(3)), "needs_reask"),
    Sample("wrong_keys", json.dumps({"accuracy": crit(5), "format": crit(4), "coverage": crit(3)}), "needs_reask"),
)
