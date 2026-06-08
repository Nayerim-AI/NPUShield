"""Stateless RKLLM provider.

This provider intentionally starts a fresh RKLLM process per request. It trades
latency for isolation: no cross-request context bleed, no unbounded KV/cache
state, and predictable cleanup on RK3588 memory-constrained systems.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from ..core.router import InferenceResult


DEFAULT_RKLLM_BINARY = os.getenv("NPUSHIELD_RKLLM_BINARY", "/usr/bin/rkllm")
DEFAULT_MODEL_PATH = os.getenv(
    "NPUSHIELD_RKLLM_MODEL",
    "models/rkllm/model.rkllm",
)


@dataclass
class StatelessRKLLMProvider:
    model_path: str = DEFAULT_MODEL_PATH
    binary_path: str = DEFAULT_RKLLM_BINARY
    context_len: int = int(os.getenv("NPUSHIELD_RKLLM_CONTEXT", "512"))
    timeout_sec: int = int(os.getenv("NPUSHIELD_RKLLM_TIMEOUT", "180"))

    def name(self) -> str:
        return "rkllm-stateless"

    def is_available(self) -> bool:
        return Path(self.binary_path).exists() and Path(self.model_path).exists()

    def infer(self, prompt: str, *, max_tokens: int = 1024, temperature: float = 0.7) -> InferenceResult:
        if not self.is_available():
            return InferenceResult(
                text="",
                provider=self.name(),
                model=Path(self.model_path).name,
                error=f"rkllm unavailable binary={self.binary_path} model={self.model_path}",
            )

        # RKLLM CLI accepts: rkllm <model> <max_tokens> <context_len>
        cmd = [self.binary_path, self.model_path, str(max_tokens), str(self.context_len)]
        start = time.time()
        proc = None
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            # Send one prompt then request process exit. This avoids persistent session bleed.
            stdin_payload = f"{prompt}\nexit\n"
            try:
                raw, _ = proc.communicate(stdin_payload, timeout=self.timeout_sec)
            except subprocess.TimeoutExpired:
                proc.kill()
                raw, _ = proc.communicate(timeout=5)
                return InferenceResult(
                    text=self._clean(raw or ""),
                    duration_sec=time.time() - start,
                    provider=self.name(),
                    model=Path(self.model_path).name,
                    error=f"timeout_after_{self.timeout_sec}s",
                )

            duration = time.time() - start
            tokens = self._parse_int(raw, r"\[Tokens\]:\s*(\d+)")
            tps = self._parse_float(raw, r"\[Token/s\]:\s*([\d.]+)")
            return InferenceResult(
                text=self._clean(raw or ""),
                tokens=tokens,
                tokens_per_sec=tps,
                duration_sec=duration,
                provider=self.name(),
                model=Path(self.model_path).name,
                error=None if proc.returncode == 0 else f"exit_code_{proc.returncode}",
            )
        except Exception as exc:
            if proc and proc.poll() is None:
                proc.kill()
            return InferenceResult(
                text="",
                duration_sec=time.time() - start,
                provider=self.name(),
                model=Path(self.model_path).name,
                error=str(exc),
            )

    @staticmethod
    def _parse_int(text: str, pattern: str) -> int:
        match = re.search(pattern, text or "")
        return int(match.group(1)) if match else 0

    @staticmethod
    def _parse_float(text: str, pattern: str) -> float:
        match = re.search(pattern, text or "")
        return float(match.group(1)) if match else 0.0

    @staticmethod
    def _clean(raw: str) -> str:
        text = raw or ""
        # Prefer content after assistant marker from ChatML prompt.
        markers = [
            r"<\|im_start\|>assistant\s*",
            r"Assistant:\s*",
            r"Answer:\s*",
            r"LLM:\s*",
        ]
        for marker in markers:
            matches = list(re.finditer(marker, text, flags=re.IGNORECASE))
            if matches:
                text = text[matches[-1].end():]
                break

        cleanup_patterns = [
            r"\[Token/s\]:\s*[\d.]+",
            r"\[Tokens\]:\s*\d+",
            r"\[Seconds\]:\s*[\d.]+",
            r"Welcome to ezrkllm.*?(?=\n|$)",
            r"To exit the model.*?(?=\n|$)",
            r"More information here.*?(?=\n|$)",
            r"Detailed information.*?(?=\n|$)",
            r"<\|im_end\|>",
            r"<\|im_start\|>assistant",
            r"^You:\s*.*$",
        ]
        for pattern in cleanup_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)

        # Drop trailing prompt after interactive CLI returns to `You:`.
        text = text.split("You:", 1)[0]
        return text.strip()
