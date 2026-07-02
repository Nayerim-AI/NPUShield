"""Allowlisted tool registry for deterministic infrastructure commands."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Tool:
    """A registered tool with fixed command templates."""
    name: str
    description: str
    commands: list[str] = field(default_factory=list)
    safe: bool = True
    requires_confirmation: bool = False
    keywords: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=lambda: ["local"])


class ToolRegistry:
    """Registry of allowlisted tools with intent matching."""

    # Services allowed to be restarted via safe_restart_service
    RESTARTABLE_SERVICES: list[str] = [
        "traefik",
        "cloudflared",
        "gitea",
        "rabbitmq",
        "dokploy",
        "npushield",
    ]

    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for t in (tools or []):
            self._tools[t.name] = t

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def build_restart_command(self, service: str) -> list[str] | None:
        """Build a safe restart command for an allowlisted service.

        Returns None if the service is not in the allowlist.
        """
        if service not in self.RESTARTABLE_SERVICES:
            return None
        return [f"docker restart {service}"]

    def match_intent(self, text: str) -> Tool | None:
        """Match a natural language query to the best tool."""
        text_lower = text.lower()
        best: tuple[int, Tool | None] = (0, None)

        for tool in self._tools.values():
            score = 0
            for kw in tool.keywords:
                if kw in text_lower:
                    score += 1
            if score > best[0]:
                best = (score, tool)

        return best[1]

    @classmethod
    def default(cls) -> ToolRegistry:
        tools = [
            Tool(
                name="server_status_top",
                description="Get live server health summary (uptime, cpu, memory, disk, top processes)",
                commands=[
                    "uptime",
                    "free -h",
                    "df -h /",
                    "top -b -n1 | head -20",
                ],
                safe=True,
                keywords=["top", "status", "uptime", "load", "health", "performance", "cpu", "memory", "ram", "disk"],
                targets=["local", "gateway", "backend"],
            ),
            Tool(
                name="service_status",
                description="Check whether a known Docker or system service is running",
                commands=[
                    "docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'",
                ],
                safe=True,
                keywords=["service", "running", "docker", "container", "healthy", "status"],
                targets=["local", "gateway", "backend"],
            ),
            Tool(
                name="docker_ps",
                description="List all running Docker containers with status and ports",
                commands=[
                    "docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}'",
                ],
                safe=True,
                keywords=["docker ps", "container list", "what's running", "running container"],
                targets=["local", "gateway", "backend"],
            ),
            Tool(
                name="safe_restart_service",
                description="Restart an allowlisted Docker service by name",
                commands=[],  # filled at runtime with dynamic service name
                safe=False,
                requires_confirmation=True,
                keywords=["restart", "reboot container", "reload service"],
                targets=["local", "gateway", "backend"],
            ),
        ]
        return cls(tools)
