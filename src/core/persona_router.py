"""Persona routing for NPUShield target-market modes."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

Persona = Literal["infra", "code"]


INFRA_KEYWORDS = {
    "server", "host", "gateway", "docker", "container", "disk", "ram",
    "cpu", "service", "systemd", "restart", "status", "log", "logs", "tailscale",
    "dns", "reverse-proxy", "proxy", "traefik", "nginx", "domain", "routing",
    "homelab", "ssh", "port", "tunnel", "database", "backup",
}

CODE_KEYWORDS = {
    "code", "python", "ctypes", "api", "function", "class",
    "provider", "streaming", "rkllm", "rknn", "rknpu", "rk3588", "npu", "model",
    "runtime", "callback", "librkllmrt", "snippet", "error", "debug", "driver",
    "convert", "quant", "quantization", "toolkit", "compile",
}

DESTRUCTIVE_PATTERNS = [
    r"\bdelete\b", r"\bremove\b", r"\brm\s+-rf\b", r"\bformat\b",
    r"\bprune\b", r"\breboot\b", r"\bpoweroff\b", r"\bshutdown\b",
    r"\bdrop\b", r"\btruncate\b", r"\breset\b",
]

SECRET_PATTERNS = [
    r"\btoken\b", r"\bpassword\b", r"\bpasswd\b", r"\bsecret\b",
    r"\bcredential\b", r"\bapi[_ -]?key\b", r"\bprivate[_ -]?key\b",
]


@dataclass(frozen=True)
class PersonaDecision:
    persona: Persona
    confidence: float
    signals: list[str] = field(default_factory=list)
    safety_flags: list[str] = field(default_factory=list)
    requires_confirmation: bool = False


class PersonaRouter:
    """Deterministic persona router for ARM/RKLLM-focused NPUShield."""

    def route(self, query: str) -> PersonaDecision:
        text = (query or "").lower()
        words = set(re.findall(r"[\w.-]+", text))

        infra_hits = sorted(k for k in INFRA_KEYWORDS if k in words or k in text)
        code_hits = sorted(k for k in CODE_KEYWORDS if k in words or k in text)

        # Target market default: ambiguous technical requests should go to code helper.
        if len(infra_hits) > len(code_hits):
            persona: Persona = "infra"
            signals = infra_hits
            confidence = min(1.0, 0.35 + 0.12 * len(infra_hits))
        else:
            persona = "code"
            signals = code_hits
            confidence = min(1.0, 0.25 + 0.12 * max(1, len(code_hits)))

        flags: list[str] = []
        if any(re.search(p, text, re.IGNORECASE) for p in DESTRUCTIVE_PATTERNS):
            flags.append("destructive")
            persona = "infra" if infra_hits or "container" in text or "server" in text else persona
        if any(re.search(p, text, re.IGNORECASE) for p in SECRET_PATTERNS):
            flags.append("secret")
            persona = "infra" if infra_hits or "cloudflare" in text or "vps" in text else persona

        requires_confirmation = bool(flags)
        return PersonaDecision(
            persona=persona,
            confidence=round(confidence, 2),
            signals=signals,
            safety_flags=flags,
            requires_confirmation=requires_confirmation,
        )
