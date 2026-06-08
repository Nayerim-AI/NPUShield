"""Output quality validation and confidence scoring."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    score: float
    reasons: list[str] = field(default_factory=list)


class OutputValidator:
    """Detect common RKLLM failure modes before returning output to users."""

    def __init__(self, min_score: float = 0.65):
        self.min_score = min_score

    def validate(self, text: str, *, expected_json: bool = False) -> ValidationResult:
        reasons: list[str] = []
        score = 1.0
        clean = (text or "").strip()

        if not clean:
            return ValidationResult(False, 0.0, ["empty_output"])

        if len(clean) < 3:
            score -= 0.35
            reasons.append("too_short")

        if len(clean) > 6000:
            score -= 0.2
            reasons.append("too_long")

        if self._has_runtime_noise(clean):
            score -= 0.25
            reasons.append("runtime_noise")

        if self._has_reasoning_trace(clean):
            score -= 0.2
            reasons.append("reasoning_trace")

        if self._has_repetition(clean):
            score -= 0.35
            reasons.append("repetition")

        if self._has_garbled_text(clean):
            score -= 0.3
            reasons.append("garbled_text")

        if expected_json and not self._is_valid_json(clean):
            score -= 0.45
            reasons.append("invalid_json")

        score = max(0.0, min(1.0, score))
        return ValidationResult(score >= self.min_score, score, reasons)

    @staticmethod
    def _has_runtime_noise(text: str) -> bool:
        patterns = [
            r"\[Token/s\]",
            r"\[Tokens\]",
            r"\[Seconds\]",
            r"Welcome to ezrkllm",
            r"<\|im_start\|>|<\|im_end\|>",
            r"^You:\s*",
        ]
        return any(re.search(p, text, re.IGNORECASE | re.MULTILINE) for p in patterns)

    @staticmethod
    def _has_reasoning_trace(text: str) -> bool:
        patterns = [
            r"Okay,\s*so I",
            r"Let me think",
            r"I need to figure",
            r"The user is asking",
            r"First,? I should",
            r"</think>|<think>",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    @staticmethod
    def _has_repetition(text: str) -> bool:
        words = re.findall(r"\b\w+\b", text.lower())
        if len(words) < 20:
            return False
        for n in (3, 4, 5):
            chunks = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
            if len(chunks) and len(chunks) - len(set(chunks)) > max(4, len(chunks) * 0.2):
                return True
        return bool(re.search(r"(.{12,80})\1{2,}", text, re.DOTALL))

    @staticmethod
    def _has_garbled_text(text: str) -> bool:
        if "�" in text:
            return True
        non_text = sum(1 for ch in text if ord(ch) < 32 and ch not in "\n\t\r")
        return non_text > 0

    @staticmethod
    def _is_valid_json(text: str) -> bool:
        try:
            json.loads(text)
            return True
        except Exception:
            return False
