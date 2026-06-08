"""Production routing: normalize → infer → validate → fallback."""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Protocol

from .prompt_normalizer import PromptNormalizer, ChatMessage
from .output_validator import OutputValidator, ValidationResult

logger = logging.getLogger(__name__)


DIRECTIVE_PREFIXES = {
    "json", "list", "classify", "extract", "summarize", "translate",
    "rewrite", "fix", "tulis", "kategorisasi", "ubah",
}

# Tasks known to degrade rapidly on the quantised 1.5B
_RISKY_PREFIXES = {
    "hitung", "hitunglah", "solve", "math", "calculate",
    "reason", "reasoning", "think", "pikir",
}


@dataclass
class InferenceResult:
    text: str
    tokens: int = 0
    tokens_per_sec: float = 0.0
    duration_sec: float = 0.0
    provider: str = "rkllm"
    model: str = "rkllm-qwen2.5-1.5b"
    error: str | None = None


class InferenceProvider(Protocol):
    def infer(self, prompt: str, *, max_tokens: int, temperature: float) -> InferenceResult:
        ...

    def is_available(self) -> bool:
        ...

    def name(self) -> str:
        ...


@dataclass
class RoutingDecision:
    task_type: str
    expected_json: bool = False
    fallback_allowed: bool = True


class ProductionRouter:
    """Coordinates the production pipeline: normalize → infer → validate → fallback."""

    def __init__(
        self,
        normalizer: PromptNormalizer | None = None,
        validator: OutputValidator | None = None,
        primary: InferenceProvider | None = None,
        fallback: InferenceProvider | None = None,
        min_quality_score: float = 0.65,
    ):
        self.normalizer = normalizer or PromptNormalizer()
        self.validator = validator or OutputValidator(min_score=min_quality_score)
        self.primary = primary
        self.fallback = fallback
        self.min_quality_score = min_quality_score

    def run(
        self,
        messages,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> InferenceResult:
        decision = self._classify(messages)

        prompt = self.normalizer.normalize(messages)
        logger.info("[ROUTER] prompt (%d chars, task=%s json=%s)",
                     len(prompt), decision.task_type, decision.expected_json)

        # Attempt primary (RKLLM NPU)
        attempt = 0
        result = None
        while attempt < 2:
            attempt += 1
            result = self._infer(self.primary, prompt, max_tokens, temperature)
            if result.error:
                logger.warning("[ROUTER] primary attempt %d error: %s", attempt, result.error)
                if attempt < 2:
                    time.sleep(0.3)
                continue

            # Validate output quality
            v = self.validator.validate(
                result.text,
                expected_json=decision.expected_json,
            )
            logger.info("[ROUTER] quality score: %.2f (ok=%s) reasons=%s",
                         v.score, v.ok, v.reasons)

            if v.ok:
                return result

            if attempt >= 2:
                break
            logger.info("[ROUTER] low quality (%.2f), retrying primary…", v.score)

        # All primary attempts failed or were low quality
        best_primary = result

        # Try fallback
        if decision.fallback_allowed and self.fallback and self.fallback.is_available():
            logger.info("[ROUTER] falling back to %s", self.fallback.name())
            fb = self._infer(self.fallback, prompt, max_tokens, temperature)
            if fb.text and not fb.error:
                v = self.validator.validate(fb.text, expected_json=decision.expected_json)
                logger.info("[ROUTER] fallback quality: %.2f (ok=%s)", v.score, v.ok)
                if v.ok:
                    return InferenceResult(
                        text=fb.text,
                        tokens=fb.tokens,
                        tokens_per_sec=fb.tokens_per_sec,
                        duration_sec=fb.duration_sec,
                        provider=self.fallback.name(),
                        model=f"fallback-{fb.model}",
                    )

        # Nothing passed → return the best attempt with an indicator
        if best_primary and not best_primary.error:
            return InferenceResult(
                text=best_primary.text,
                tokens=best_primary.tokens,
                tokens_per_sec=best_primary.tokens_per_sec,
                duration_sec=best_primary.duration_sec,
                provider=best_primary.provider,
                model=best_primary.model,
                error=f"low_quality score below {self.min_quality_score}",
            )
        return InferenceResult(
            text="",
            error="all_providers_failed",
        )

    def _classify(self, messages) -> RoutingDecision:
        user_msg = ""
        for m in reversed(list(messages)):
            if getattr(m, "role", None) == "user":
                user_msg = getattr(m, "content", "") or ""
                break
        first_word = user_msg.strip().lower().split(None, 1)[0] if user_msg.strip() else ""
        safe = first_word not in _RISKY_PREFIXES
        return RoutingDecision(
            task_type=first_word if first_word in DIRECTIVE_PREFIXES else "chat",
            expected_json=("json" in user_msg.lower() or first_word == "json"),
            fallback_allowed=safe,
        )

    def _infer(
        self,
        provider: InferenceProvider | None,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> InferenceResult:
        if not provider or not provider.is_available():
            return InferenceResult(text="", error="provider_unavailable")
        return provider.infer(prompt, max_tokens=max_tokens, temperature=temperature)
