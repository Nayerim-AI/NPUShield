"""NPUShield — Production API Server

Singe-process FastAPI exposing the ProductionRouter pipeline over an
OpenAI-compatible /v1/chat/completions endpoint.

Environment variables:
  NPUSHIELD_RKLLM_BINARY    path to rkllm binary     [/usr/bin/rkllm]
  NPUSHIELD_RKLLM_MODEL     path to .rkllm model     [see code]
  NPUSHIELD_RKLLM_CONTEXT   context length           [512]
  NPUSHIELD_RKLLM_TIMEOUT   seconds per request      [180]
  NPUSHIELD_HOST            listen address            [0.0.0.0]
  NPUSHIELD_PORT            listen port               [18999]
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from src.core.output_validator import OutputValidator
from src.core.prompt_normalizer import PromptNormalizer
from src.core.router import ProductionRouter, InferenceResult
from src.core.persona_router import PersonaRouter
from src.core.kb_loader import load_kb
from src.core.kb_index import KBIndex
from src.core.tool_registry import ToolRegistry, Tool
from src.core.tool_executor import ToolExecutor, ToolTarget, ToolResult

logger = logging.getLogger("npushield")
logger.setLevel(logging.INFO)
_console = logging.StreamHandler()
_console.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
))
logger.addHandler(_console)


# ── Config ────────────────────────────────────────────────────────────────

HOST = os.getenv("NPUSHIELD_HOST", "0.0.0.0")
PORT = int(os.getenv("NPUSHIELD_PORT", "18999"))
MIN_QUALITY = float(os.getenv("NPUSHIELD_MIN_QUALITY", "0.65"))
KB_DIR = os.getenv("NPUSHIELD_KB_DIR", "kb")


def _augment_messages_with_rag(
    pydantic_messages: list,
    *,
    kb_max_chars: int = 3500,
) -> tuple[list[dict], dict]:
    """
    Route the conversation to a persona, retrieve relevant KB context,
    and return enriched messages with the RAG context injected.
    """
    messages = [{"role": m.role, "content": m.content} for m in pydantic_messages]

    if not messages:
        return messages, {"persona": "code", "rag_docs": 0}

    # Use the last user message for classification + KB search
    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = m.get("content", "")
            break

    if not last_user:
        return messages, {"persona": "code", "rag_docs": 0}

    # Route persona
    decision = persona_router.route(last_user)

    # Retrieve KB context
    docs: list = []
    if kb_index is not None:
        query = last_user
        path_prefix = None
        if decision.persona == "infra":
            path_prefix = "infra"
        elif decision.persona == "code":
            path_prefix = "code"
        results = kb_index.search(query, limit=4, path_prefix=path_prefix)
        if results:
            context = kb_index.format_context(results, max_chars=kb_max_chars)
            docs = [{"path": r.path, "score": r.score} for r in results]

            # Inject RAG context into the first system or user message
            injected = False
            for m in messages:
                if m["role"] == "system":
                    m["content"] = (
                        m["content"]
                        + f"\n\n---\nKnowledge Base Context:\n{context}\n---"
                    )
                    injected = True
                    break
            if not injected:
                messages.insert(
                    0,
                    {
                        "role": "system",
                        "content": (
                            "Berikut adalah konteks dari knowledge base:\n"
                            f"{context}\n---\n"
                            "Jawab hanya dari konteks di atas. "
                            "Jika tidak ada informasi yang relevan, katakan "
                            "'Saya belum punya info itu di knowledge base.'"
                        ),
                    },
                )
        else:
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": (
                        "Tidak ada informasi yang cocok di knowledge base. "
                        "Jangan mengarang jawaban. Katakan: "
                        "'Saya belum punya info itu di knowledge base.'"
                    ),
                },
            )

    meta = {
        "persona": decision.persona,
        "confidence": decision.confidence,
        "requires_confirmation": decision.requires_confirmation,
        "safety_flags": decision.safety_flags,
        "rag_docs": len(docs),
        "rag_paths": [d["path"] for d in docs],
    }
    return messages, meta

MODEL_ID = "rkllm-qwen2.5-1.5b"
MODEL_NAME = "NPUShield Qwen2.5-1.5B"
MODEL_OWNED_BY = "npushield"


# ── Schemas ───────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: List[Message]
    max_tokens: int = 1024
    stream: bool = False
    temperature: float = 0.7
    top_p: float = 0.95


class ToolRunRequest(BaseModel):
    tool: str
    target: str = "local"
    service: Optional[str] = None


class ToolRunResponse(BaseModel):
    tool: str
    target: str
    exit_code: int
    stdout: str
    stderr: str
    truncated: bool = False


# ── Global router ─────────────────────────────────────────────────────────

router: ProductionRouter | None = None
persona_router = PersonaRouter()
kb_index: KBIndex | None = None
tool_registry = ToolRegistry.default()
tool_executor = ToolExecutor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global router, kb_index
    normalizer = PromptNormalizer()
    validator = OutputValidator(min_score=MIN_QUALITY)
    try:
        kb_index = load_kb(KB_DIR)
        logger.info("knowledge base loaded from %s (%d docs)", KB_DIR, kb_index.document_count())
    except Exception as exc:
        kb_index = None
        logger.warning("knowledge base unavailable: %s", exc)

    mode = os.getenv("NPUSHIELD_WORKER_MODE", "capi")
    if mode == "stateless":
        from src.providers.rkllm_stateless import StatelessRKLLMProvider as RKLLMProvider
        logger.info("worker mode: stateless (spawn per request)")
    elif mode == "warm":
        from src.providers.rkllm_warm_worker import WarmRKLLMWorker as RKLLMProvider
        logger.info("worker mode: warm CLI (bounded persistent CLI worker)")
    else:
        from src.providers.rkllm_capi import RKLLMCAPIProvider as RKLLMProvider
        logger.info("worker mode: capi (direct librkllmrt.so in-process)")

    primary = RKLLMProvider()

    if primary.is_available():
        logger.info("rkllm backend available — loaded")
    else:
        logger.warning("rkllm binary or model not found — server starts but inference will fail")

    fallback = None
    llama_binary = os.getenv("NPUSHIELD_LLAMACPP_BINARY", "")
    llama_model = os.getenv("NPUSHIELD_LLAMACPP_MODEL", "")
    if llama_binary and llama_model:
        from src.providers.llamacpp_fallback import LlamaCppProvider
        fallback = LlamaCppProvider(
            binary_path=llama_binary,
            model_path=llama_model,
        )
        logger.info("llama.cpp fallback configured — enabled!")
    else:
        logger.info("no llama.cpp fallback configured (set NPUSHIELD_LLAMACPP_*)")

    router = ProductionRouter(
        normalizer=normalizer,
        validator=validator,
        primary=primary,
        fallback=fallback,
        min_quality_score=MIN_QUALITY,
    )
    yield
    if primary and hasattr(primary, "release"):
        primary.release()
    elif primary and hasattr(primary, "recycle"):
        primary.recycle()


def _resolve_tool_target(target_name: str) -> ToolTarget:
    """Resolve a safe tool target from environment configuration."""
    if target_name == "local":
        return ToolTarget.local()

    # Optional SSH targets, example:
    # NPUSHIELD_TOOL_TARGET_GATEWAY=user@10.0.0.2
    env_name = f"NPUSHIELD_TOOL_TARGET_{target_name.upper().replace('-', '_')}"
    raw = os.getenv(env_name, "")
    if raw and "@" in raw:
        user, host = raw.split("@", 1)
        return ToolTarget.ssh(name=target_name, user=user, host=host)

    raise HTTPException(404, f"Unknown or unconfigured tool target: {target_name}")


app = FastAPI(
    title="NPUShield",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    if router is None or not router.primary or not router.primary.is_available():
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "reason": "rkllm backend unavailable"},
        )
    return {
        "status": "healthy",
        "service": "npushield",
        "version": "0.1.0",
        "primary": router.primary.name(),
        "fallback": router.fallback.name() if router.fallback else None,
    }


@app.get("/v1/models")
async def list_models():
    fallback_models = []
    if router and router.fallback and router.fallback.is_available():
        fallback_models = [
            {
                "id": router.fallback.name(),
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "npushield-fallback",
                "permission": [],
                "root": router.fallback.name(),
                "parent": None,
            }
        ]
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": MODEL_OWNED_BY,
                "permission": [],
                "root": MODEL_ID,
                "parent": None,
            },
            *fallback_models,
        ],
    }


def _is_explain_query(text: str) -> bool:
    """Detect if user asks for explanation (should use LLM, not skip it)."""
    lowered = text.lower()
    explain_signals = [
        "jelaskan", "explain", "describe", "what is", "apa itu",
        "bagaimana", "how to", "how does", "how do",
        "apa saja", "what are", "what services",
    ]
    return any(s in lowered for s in explain_signals)


def _maybe_run_tool_for_chat(messages: list[Message]) -> dict | None:
    """Auto-run a safe allowlisted tool when chat intent clearly matches."""
    last_user = ""
    for m in reversed(messages):
        if m.role == "user":
            last_user = m.content
            break
    if not last_user:
        return None

    # Only infra persona should auto-run tools.
    decision = persona_router.route(last_user)
    if decision.persona != "infra":
        return None

    tool = tool_registry.match_intent(last_user)
    if tool is None or not tool.safe or tool.requires_confirmation:
        return None

    target = ToolTarget.local()
    result = tool_executor.run_commands(tool.commands, target=target)
    tool_output = (
        f"Tool executed: {tool.name}\n"
        f"Target: {result.target_name}\n"
        f"Exit code: {result.exit_code}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}\n"
    )
    summary = _summarize_tool_result(tool.name, result)
    meta = {
        "x-tool": tool.name,
        "x-tool-target": result.target_name,
        "x-tool-exit-code": result.exit_code,
    }

    # If user asks for explanation, run LLM with tool output as context
    skip_llm = not _is_explain_query(last_user)

    if skip_llm:
        return {
            "skip_llm": True,
            "response": _build_tool_response(summary, MODEL_ID, meta),
            "message": {
                "role": "system",
                "content": (
                    "A safe allowlisted infrastructure tool was executed. "
                    "Summarize the tool output in a concise operational status. "
                    "Do not invent facts not present in the output.\n\n"
                    f"{tool_output}"
                ),
            },
            "meta": meta,
        }

    # Explain query: pass tool output to LLM for explanation
    explain_msg = (
        "An allowlisted infrastructure tool was executed. "
        "The user asked for an explanation of what they see. "
        "Use the tool output below to explain each item clearly.\n\n"
        f"Tool: {tool.name}\nTarget: {result.target_name}\n\n"
        f"Raw output:\n{result.stdout}"
    )
    return {
        "skip_llm": False,
        "message": {"role": "system", "content": explain_msg},
        "meta": meta,
    }


@app.get("/v1/tools")
async def list_tools():
    return {
        "object": "list",
        "data": [
            {
                "name": tool.name,
                "description": tool.description,
                "safe": tool.safe,
                "requires_confirmation": tool.requires_confirmation,
                "targets": tool.targets,
            }
            for tool in tool_registry._tools.values()
        ],
    }


@app.post("/v1/tools/run")
async def run_tool(request: ToolRunRequest):
    tool = tool_registry.get(request.tool)
    if tool is None:
        raise HTTPException(404, f"Unknown tool: {request.tool}")
    if tool.requires_confirmation:
        raise HTTPException(409, f"Tool requires explicit confirmation: {request.tool}")

    target = _resolve_tool_target(request.target)
    result = tool_executor.run_commands(tool.commands, target=target)
    return ToolRunResponse(
        tool=tool.name,
        target=target.name,
        exit_code=result.exit_code,
        stdout=result.stdout,
        stderr=result.stderr,
        truncated="...[truncated]" in result.stdout or "...[truncated]" in result.stderr,
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if router is None:
        raise HTTPException(503, "Server not ready")

    if request.stream:
        return StreamingResponse(
            _stream_chat(request),
            media_type="text/event-stream",
        )

    tool_ctx = _maybe_run_tool_for_chat(request.messages)
    if tool_ctx and tool_ctx["skip_llm"]:
        return tool_ctx["response"]

    messages, persona_meta = _augment_messages_with_rag(request.messages)
    if tool_ctx:
        messages.insert(0, tool_ctx["message"])
        persona_meta.update(tool_ctx["meta"])

    result = router.run(
        messages=messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )

    if result.error and not result.text:
        raise HTTPException(503, detail=result.error)

    return _build_response(result, MODEL_ID, persona_meta=persona_meta)


async def _stream_chat(request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
    if router is None:
        yield f"data: {json.dumps({'error': 'server not ready'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    messages, persona_meta = _augment_messages_with_rag(request.messages)
    result = router.run(
        messages=messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )

    chat_id = f"chatcmpl-{int(time.time())}"
    created = int(datetime.now().timestamp())

    try:
        yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': created, 'model': MODEL_ID, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
        yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': created, 'model': MODEL_ID, 'choices': [{'index': 0, 'delta': {'content': result.text}, 'finish_reason': None}]})}\n\n"
        yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': created, 'model': MODEL_ID, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
    except Exception as e:
        logger.error("stream error: %s", e)
    finally:
        yield "data: [DONE]\n\n"


def _summarize_tool_result(tool_name: str, result: ToolResult) -> str:
    if result.exit_code != 0:
        return (
            f"Tool `{tool_name}` failed on `{result.target_name}` with exit code {result.exit_code}.\n\n"
            f"STDERR:\n{result.stderr.strip() or '(empty)'}"
        )
    if tool_name == "server_status_top":
        lines = result.stdout.splitlines()
        keep = []
        for line in lines:
            low = line.lower()
            if (
                "load average" in low
                or low.startswith("mem:")
                or low.startswith("swap:")
                or low.startswith("/dev/")
                or low.startswith("tasks:")
                or "cpu(s)" in low
            ):
                keep.append(line)
        details = "\n".join(keep[:12]) or result.stdout[:1200]
        return f"Server status tool ran successfully on `{result.target_name}`.\n\n```text\n{details}\n```"
    return f"Tool `{tool_name}` ran successfully on `{result.target_name}`.\n\n```text\n{result.stdout[:2000]}\n```"


def _build_tool_response(content: str, model_id: str, meta: dict) -> dict:
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(datetime.now().timestamp()),
        "model": model_id,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "tool"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "x-provider": "tool-executor",
        "x-duration-sec": 0,
        "x-tokens-per-sec": 0,
        "x-quality-error": None,
        "x-persona": "infra",
        "x-persona-confidence": 1.0,
        "x-rag-docs": 0,
        "x-rag-paths": [],
        "x-safety-flags": [],
        "x-tool": meta.get("x-tool"),
        "x-tool-target": meta.get("x-tool-target"),
        "x-tool-exit-code": meta.get("x-tool-exit-code"),
    }


def _build_response(result: InferenceResult, model_id: str, persona_meta: dict | None = None) -> dict:
    persona_meta = persona_meta or {}
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(datetime.now().timestamp()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.text,
                },
                "finish_reason": "stop" if not result.error else "error",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": result.tokens,
            "total_tokens": result.tokens,
        },
        "x-provider": result.provider,
        "x-duration-sec": round(result.duration_sec, 2),
        "x-tokens-per-sec": round(result.tokens_per_sec, 1),
        "x-quality-error": result.error or None,
        "x-persona": persona_meta.get("persona"),
        "x-persona-confidence": persona_meta.get("confidence"),
        "x-rag-docs": persona_meta.get("rag_docs", 0),
        "x-rag-paths": persona_meta.get("rag_paths", []),
        "x-safety-flags": persona_meta.get("safety_flags", []),
        "x-tool": persona_meta.get("x-tool"),
        "x-tool-target": persona_meta.get("x-tool-target"),
        "x-tool-exit-code": persona_meta.get("x-tool-exit-code"),
    }


if __name__ == "__main__":
    logger.info("NPUShield starting on %s:%s", HOST, PORT)
    uvicorn.run(app, host=HOST, port=PORT)
