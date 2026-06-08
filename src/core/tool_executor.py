"""Deterministic command executor for allowlisted NPUShield tools."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolTarget:
    name: str
    host: str | None = None
    user: str | None = None
    timeout_sec: int = 15

    @classmethod
    def local(cls) -> "ToolTarget":
        return cls(name="local")

    @classmethod
    def ssh(cls, name: str, user: str, host: str, timeout_sec: int = 15) -> "ToolTarget":
        return cls(name=name, user=user, host=host, timeout_sec=timeout_sec)


@dataclass(frozen=True)
class ToolResult:
    target_name: str
    command: str
    stdout: str
    stderr: str
    exit_code: int


class ToolExecutor:
    """Run fixed commands locally or via SSH with safety checks."""

    def __init__(self, dangerous_patterns: list[str] | None = None):
        self.dangerous_patterns = dangerous_patterns or [
            r"rm\s+-rf",
            r"docker\s+volume\s+rm",
            r"docker\s+system\s+prune",
            r"docker\s+restart",
            r"systemctl\s+restart",
            r"mkfs",
            r"dd\s+if=",
            r"shutdown",
            r"poweroff",
            r"reboot",
            r"drop\s+database",
            r"truncate",
        ]

    def run_commands(self, commands: list[str], target: ToolTarget) -> ToolResult:
        combined = " && ".join(commands)
        rejection = self._reject_if_dangerous(combined)
        if rejection:
            return ToolResult(
                target_name=target.name,
                command=combined,
                stdout="",
                stderr=rejection,
                exit_code=-1,
            )

        if target.name == "local" or not target.host:
            shell_command = combined
        else:
            if not target.user:
                return ToolResult(
                    target_name=target.name,
                    command=combined,
                    stdout="",
                    stderr="SSH target requires user and host",
                    exit_code=-1,
                )
            shell_command = (
                "ssh -o BatchMode=yes -o ConnectTimeout=8 "
                f"{target.user}@{target.host} {self._shell_quote(combined)}"
            )

        try:
            proc = subprocess.run(
                shell_command,
                shell=True,
                text=True,
                capture_output=True,
                timeout=target.timeout_sec,
            )
            return ToolResult(
                target_name=target.name,
                command=combined,
                stdout=self._truncate(proc.stdout),
                stderr=self._truncate(proc.stderr),
                exit_code=proc.returncode,
            )
        except subprocess.TimeoutExpired as exc:
            return ToolResult(
                target_name=target.name,
                command=combined,
                stdout=self._truncate(exc.stdout or ""),
                stderr=f"command timed out after {target.timeout_sec}s",
                exit_code=124,
            )

    def _reject_if_dangerous(self, command: str) -> str | None:
        for pattern in self.dangerous_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return f"command rejected by safety policy: {pattern}"
        return None

    @staticmethod
    def _truncate(text: str, max_chars: int = 12000) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n...[truncated]"

    @staticmethod
    def _shell_quote(value: str) -> str:
        return "'" + value.replace("'", "'\\''") + "'"
