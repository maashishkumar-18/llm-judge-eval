"""Env-only configuration. Generator (GEN_*) and judge (JUDGE_*) are configured independently:
no shared model/temperature/concurrency fields, and model ids are required (no silent defaults),
so a missing id fails loudly instead of quietly falling back."""

from __future__ import annotations

import logging

from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger("judge.config")

# Ordered: first matching substring wins.
_FAMILY_PATTERNS: list[tuple[str, str]] = [
    ("gemini", "google"),
    ("gemma", "google"),
    ("gpt-oss", "openai"),
    ("gpt", "openai"),
    ("llama", "meta"),
    ("claude", "anthropic"),
    ("mistral", "mistral"),
    ("mixtral", "mistral"),
    ("qwen", "alibaba"),
    ("deepseek", "deepseek"),
]


class FamilyGuardError(RuntimeError):
    pass


def family_of(model_id: str) -> str | None:
    m = model_id.lower()
    for needle, fam in _FAMILY_PATTERNS:
        if needle in m:
            return fam
    return None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    # keys (one per provider)
    gemini_api_key: str = ""
    groq_api_key: str = ""

    # generator: Groq-hosted gpt-oss (OpenAI family). Roles were swapped in j0b; see CLAUDE.md A1/A8.
    gen_provider: str = "groq"
    gen_model: str = "openai/gpt-oss-20b"
    gen_temperature: float = 0.0
    gen_concurrency: int = 2
    gen_reasoning_effort: str = "low"  # low | medium | high (gpt-oss only)
    gen_strip_reasoning: bool = True  # drop the `reasoning` field; keep only final `content`

    # judge: Gemini (Google family). gemini-3.5-flash-lite is FINAL for this project (never change it
    # mid-project: validation numbers are only valid for the judge that produced them).
    judge_provider: str = "gemini"
    judge_model: str = "gemini-3.5-flash-lite"
    judge_temperature: float = 0.0
    judge_concurrency: int = 2
    # Free-tier limits (AI Studio): 15 RPM, 250K TPM, 500 RPD. We stay under them:
    judge_rpm_limit: int = 12  # requests/min, sliding window, applies to every HTTP attempt
    judge_daily_cap: int = 450  # hard cap on NON-cached judge calls per Pacific day (persistent ledger)
    judge_max_tokens: int = 8192  # generous: Gemini thinking tokens count against max output
    ledger_path: str = ".cache/judge_ledger.json"
    # Provider JSON/schema mode for judge calls. OFF = prompt-only JSON, so the parse ladder is exercised.
    judge_json_mode: bool = False

    allow_same_family: bool = False

    # client hygiene
    max_retries: int = 6
    backoff_base_seconds: float = 2.0
    backoff_max_seconds: float = 60.0
    request_timeout_seconds: float = 120.0

    cache_dir: str = ".cache/llm"
    log_dir: str = "logs"

    def api_key_for(self, provider: str) -> str:
        key = {"gemini": self.gemini_api_key, "groq": self.groq_api_key}.get(provider)
        if key is None:
            raise ValueError(f"unknown provider {provider!r}")
        if not key:
            raise ValueError(f"no API key configured for provider {provider!r}")
        return key

    def enforce_family_guard(self) -> tuple[str, str]:
        """Refuse judge family == generator family unless ALLOW_SAME_FAMILY=1.
        Unknown families are refused too: an unverifiable guard is no guard."""
        gf, jf = family_of(self.gen_model), family_of(self.judge_model)
        for role, model, fam in (("generator", self.gen_model, gf), ("judge", self.judge_model, jf)):
            if fam is None and not self.allow_same_family:
                raise FamilyGuardError(
                    f"cannot determine model family of {role} {model!r}; add it to _FAMILY_PATTERNS "
                    "or set ALLOW_SAME_FAMILY=1"
                )
        if gf == jf:
            if not self.allow_same_family:
                raise FamilyGuardError(
                    f"judge {self.judge_model!r} and generator {self.gen_model!r} are both family "
                    f"{gf!r}; self-enhancement bias risk. Set ALLOW_SAME_FAMILY=1 to override."
                )
            log.warning("ALLOW_SAME_FAMILY=1: judge and generator share family %r", gf)
        return gf or "unknown", jf or "unknown"


def load_settings() -> Settings:
    return Settings()
