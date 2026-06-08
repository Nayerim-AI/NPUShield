"""Prompt normalization for RKLLM production use."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal


Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


DEFAULT_SYSTEM_PROMPT = (
    "Kamu adalah asisten AI lokal yang berjalan di RK3588 NPU. "
    "Jawab langsung, singkat, akurat, dan dalam bahasa yang sama dengan user. "
    "Jika tidak yakin, katakan tidak yakin. Jangan mengarang."
)


class PromptNormalizer:
    """Build stable prompts for small RKLLM chat models."""

    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        self.system_prompt = system_prompt.strip()

    def normalize(self, messages: Iterable[ChatMessage | dict]) -> str:
        normalized = [self._coerce_message(m) for m in messages]
        if not normalized:
            normalized = [ChatMessage(role="user", content="Halo")]

        system_parts = [m.content.strip() for m in normalized if m.role == "system" and m.content.strip()]
        system_prompt = system_parts[-1] if system_parts else self.system_prompt

        conversation = [m for m in normalized if m.role in {"user", "assistant"} and m.content.strip()]
        if not any(m.role == "user" for m in conversation):
            conversation.append(ChatMessage(role="user", content="Halo"))

        # Qwen2.5-Instruct ChatML format. RKLLM sometimes receives this better than raw text.
        parts = [
            "<|im_start|>system",
            system_prompt,
            "<|im_end|>",
        ]
        for msg in conversation[-8:]:
            parts.extend([
                f"<|im_start|>{msg.role}",
                msg.content.strip(),
                "<|im_end|>",
            ])
        parts.extend(["<|im_start|>assistant", ""])
        return "\n".join(parts)

    @staticmethod
    def _coerce_message(message: ChatMessage | dict) -> ChatMessage:
        if isinstance(message, ChatMessage):
            return message
        role = str(message.get("role", "user")).lower()
        if role not in {"system", "user", "assistant"}:
            role = "user"
        return ChatMessage(role=role, content=str(message.get("content", "")))
