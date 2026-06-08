"""Warm RKLLM worker with bounded lifecycle.

This provider keeps one RKLLM interactive process warm to avoid per-request cold
start, but it is NOT an infinite chat session:

- single-flight lock: one request at a time per worker
- max_requests: recycle before context/state becomes risky
- max_uptime_sec: recycle periodically
- hard timeout: kill stuck worker
- buffer drain: reduce cross-request contamination

This is the practical production compromise for RK3588/RKLLM where cold start is
expensive but persistent sessions can leak RAM/state.
"""

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import pexpect
except ModuleNotFoundError:  # allows importing module in dev/test env without runtime dependency installed
    pexpect = None

from ..core.router import InferenceResult


DEFAULT_RKLLM_BINARY = os.getenv("NPUSHIELD_RKLLM_BINARY", "/usr/bin/rkllm")
DEFAULT_MODEL_PATH = os.getenv(
    "NPUSHIELD_RKLLM_MODEL",
    "models/rkllm/model.rkllm",
)


@dataclass
class WarmRKLLMWorker:
    model_path: str = DEFAULT_MODEL_PATH
    binary_path: str = DEFAULT_RKLLM_BINARY
    context_len: int = int(os.getenv("NPUSHIELD_RKLLM_CONTEXT", "512"))
    max_tokens_default: int = int(os.getenv("NPUSHIELD_RKLLM_MAX_TOKENS", "1024"))
    load_timeout_sec: int = int(os.getenv("NPUSHIELD_RKLLM_LOAD_TIMEOUT", "90"))
    inference_timeout_sec: int = int(os.getenv("NPUSHIELD_RKLLM_TIMEOUT", "180"))
    max_requests: int = int(os.getenv("NPUSHIELD_RKLLM_WORKER_MAX_REQUESTS", "3"))
    max_uptime_sec: int = int(os.getenv("NPUSHIELD_RKLLM_WORKER_MAX_UPTIME", "900"))
    restart_on_low_confidence: bool = True

    _child: Optional[pexpect.spawn] = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _request_count: int = field(default=0, init=False)
    _started_at: float = field(default=0.0, init=False)
    _last_error: str | None = field(default=None, init=False)

    def name(self) -> str:
        return "rkllm-warm-worker"

    def is_available(self) -> bool:
        return Path(self.binary_path).exists() and Path(self.model_path).exists()

    def infer(self, prompt: str, *, max_tokens: int = 1024, temperature: float = 0.7) -> InferenceResult:
        if pexpect is None:
            return InferenceResult(
                text="",
                provider=self.name(),
                model=Path(self.model_path).name,
                error="missing_dependency_pexpect",
            )

        if not self.is_available():
            return InferenceResult(
                text="",
                provider=self.name(),
                model=Path(self.model_path).name,
                error=f"rkllm unavailable binary={self.binary_path} model={self.model_path}",
            )

        with self._lock:
            if self._should_recycle():
                self._shutdown_locked()

            ok = self._ensure_started_locked(max_tokens=max_tokens)
            if not ok:
                return InferenceResult(
                    text="",
                    provider=self.name(),
                    model=Path(self.model_path).name,
                    error=self._last_error or "worker_start_failed",
                )

            start = time.time()
            try:
                assert self._child is not None
                self._drain_buffer_locked()
                self._child.sendline(prompt)

                raw = self._read_until_done_locked(timeout=self.inference_timeout_sec)
                duration = time.time() - start
                self._request_count += 1

                tokens = self._parse_int(raw, r"\[Tokens\]:\s*(\d+)")
                tps = self._parse_float(raw, r"\[Token/s\]:\s*([\d.]+)")
                text = self._clean(raw)

                if not self._child.isalive():
                    self._last_error = "worker_died_after_request"
                    self._shutdown_locked()

                return InferenceResult(
                    text=text,
                    tokens=tokens,
                    tokens_per_sec=tps,
                    duration_sec=duration,
                    provider=self.name(),
                    model=Path(self.model_path).name,
                    error=None,
                )
            except Exception as exc:
                self._last_error = str(exc)
                self._shutdown_locked()
                return InferenceResult(
                    text="",
                    provider=self.name(),
                    model=Path(self.model_path).name,
                    duration_sec=time.time() - start,
                    error=str(exc),
                )

    def recycle(self) -> None:
        with self._lock:
            self._shutdown_locked()

    def status(self) -> dict:
        with self._lock:
            alive = bool(self._child and self._child.isalive())
            return {
                "provider": self.name(),
                "alive": alive,
                "pid": self._child.pid if alive and self._child else None,
                "request_count": self._request_count,
                "uptime_sec": round(time.time() - self._started_at, 1) if self._started_at else 0,
                "max_requests": self.max_requests,
                "max_uptime_sec": self.max_uptime_sec,
                "last_error": self._last_error,
            }

    def _ensure_started_locked(self, *, max_tokens: int) -> bool:
        if self._child and self._child.isalive():
            return True

        cmd = f"{self.binary_path} {self.model_path} {max_tokens or self.max_tokens_default} {self.context_len}"
        try:
            self._child = pexpect.spawn(cmd, encoding="utf-8", errors="replace", timeout=self.load_timeout_sec)
            self._started_at = time.time()
            self._request_count = 0
            self._last_error = None
            idx = self._child.expect(["You: ", "error", pexpect.EOF, pexpect.TIMEOUT], timeout=self.load_timeout_sec)
            if idx == 0:
                return True
            self._last_error = ["ready", "init_error", "eof", "load_timeout"][idx]
            self._shutdown_locked()
            return False
        except Exception as exc:
            self._last_error = str(exc)
            self._shutdown_locked()
            return False

    def _should_recycle(self) -> bool:
        if not self._child or not self._child.isalive():
            return True
        if self._request_count >= self.max_requests:
            return True
        if self._started_at and time.time() - self._started_at > self.max_uptime_sec:
            return True
        return False

    def _read_until_done_locked(self, *, timeout: int) -> str:
        assert self._child is not None
        deadline = time.time() + timeout
        output = ""
        while time.time() < deadline:
            try:
                chunk = self._child.readline()
                if chunk:
                    output += chunk
                if "[Token/s]:" in output or "You:" in output:
                    return output
            except pexpect.TIMEOUT:
                if "[Token/s]:" in output:
                    return output
                continue
        raise TimeoutError(f"inference_timeout_after_{timeout}s")

    def _drain_buffer_locked(self) -> None:
        if not self._child:
            return
        old_timeout = self._child.timeout
        self._child.timeout = 0.05
        try:
            for _ in range(20):
                try:
                    _ = self._child.read_nonblocking(size=4096, timeout=0.05)
                except Exception:
                    break
        finally:
            self._child.timeout = old_timeout

    def _shutdown_locked(self) -> None:
        child = self._child
        self._child = None
        self._started_at = 0.0
        self._request_count = 0
        if not child:
            return
        try:
            if child.isalive():
                child.sendline("exit")
                time.sleep(0.2)
            if child.isalive():
                child.close(force=True)
        except Exception:
            try:
                child.close(force=True)
            except Exception:
                pass

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

        for pattern in [
            r"\[Token/s\]:\s*[\d.]+",
            r"\[Tokens\]:\s*\d+",
            r"\[Seconds\]:\s*[\d.]+",
            r"Welcome to ezrkllm.*?(?=\n|$)",
            r"To exit the model.*?(?=\n|$)",
            r"More information here.*?(?=\n|$)",
            r"Detailed information.*?(?=\n|$)",
            r"<\|im_end\|>",
            r"<\|im_start\|>assistant",
        ]:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
        text = text.split("You:", 1)[0]
        return text.strip()
