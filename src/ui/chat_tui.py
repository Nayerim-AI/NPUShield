"""NPUShield interactive chat TUI — REPL client for local API."""

from __future__ import annotations

import json
import os
import shutil
import sys
import textwrap
from datetime import datetime
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

API_BASE = os.getenv("NPUSHIELD_HOST", "http://127.0.0.1:18999")
API_KEY = os.getenv("NPUSHIELD_API_KEY", "").strip() or None
DEFAULT_MODEL = "rkllm-qwen2.5-1.5b"

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.styles import Style
    from prompt_toolkit.formatted_text import FormattedText

    HAS_PROMPT = True
except ImportError:
    HAS_PROMPT = False


# ── ANSI helpers (fallback) ───────────────────────────────────────────


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m"


def green(t: str) -> str:
    return _c("92", t)


def yellow(t: str) -> str:
    return _c("93", t)


def red(t: str) -> str:
    return _c("91", t)


def dim(t: str) -> str:
    return _c("2", t)


def cyan(t: str) -> str:
    return _c("96", t)


def bold(t: str) -> str:
    return _c("1", t)


# ── API client ─────────────────────────────────────────────────────────


def _api_post(path: str, body: dict) -> dict | None:
    url = f"{API_BASE}{path}"
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    req = Request(url, data=data, headers=headers, method="POST")
    try:
        with urlopen(req, timeout=180) as resp:
            return json.loads(resp.read())
    except URLError as e:
        return {"_error": str(e)}
    except json.JSONDecodeError as e:
        return {"_error": f"json decode: {e}"}


def _api_get(path: str) -> dict | None:
    url = f"{API_BASE}{path}"
    headers = {}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    req = Request(url, headers=headers, method="GET")
    try:
        with urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"_error": str(e)}


# ── Format helpers ──────────────────────────────────────────────────────


def _format_meta(o: dict) -> list[str]:
    lines = []
    meta_keys = [
        ("x-provider", "provider"),
        ("x-tool", "tool"),
        ("x-persona", "persona"),
        ("x-persona-confidence", "confidence"),
        ("x-tool-exit-code", "exit"),
        ("x-rag-docs", "rag"),
        ("x-duration-sec", "dur"),
    ]
    for key, label in meta_keys:
        val = o.get(key)
        if val is not None and val != "" and val != 0:
            lines.append(f"{label}={val}")
    return lines


def _display_response(o: dict) -> str:
    if "_error" in o:
        return red(f"❌ {o['_error']}")

    lines: list[str] = []

    meta = _format_meta(o)
    if meta:
        lines.append(dim("  ".join(meta)))

    choices = o.get("choices", [])
    if choices:
        msg = choices[0].get("message", {})
        content = msg.get("content", "")
        reason = choices[0].get("finish_reason", "")
        if content:
            lines.append(content)
        if reason:
            lines.append(dim(f"\n  finish_reason: {reason}"))
    return "\n".join(lines)


def _display_tool_result(o: dict) -> str:
    if "_error" in o:
        return red(f"❌ {o['_error']}")
    lines = []
    lines.append(dim(f"tool={o['tool']}  target={o['target']}  exit={o['exit_code']}"))
    stdout = o.get("stdout", "").strip()
    stderr = o.get("stderr", "").strip()
    if stdout:
        lines.append(stdout)
    if stderr:
        lines.append(red(stderr))
    if not stdout and not stderr:
        lines.append("(empty output)")
    return "\n".join(lines)


# ── Built-in commands ────────────────────────────────────────────────


def _cmd_help() -> str:
    return textwrap.dedent(f"""
    {bold('Commands')}

      {cyan('Built-in:')}
        /help, /h        — this message
        /tools           — list available tools
        /run <tool>      — execute tool directly
        /model <name>    — switch model (default: rkllm-qwen2.5-1.5b)
        /health          — check API health
        /clear, /cls     — clear screen
        /history         — show command history
        /exit, /quit     — bye

      {cyan('Anything else')}
        — sent as chat query. If infra intent matches a tool,
          tool runs directly (no LLM). Otherwise routes to RKLLM.

    """).strip()


def _cmd_tools() -> str:
    resp = _api_get("/v1/tools")
    if not resp or "_error" in (resp or {}):
        return red("❌ gagal fetch tools")
    items = resp.get("data", [])
    if not items:
        return yellow("(no tools available)")
    lines = [bold("Available tools:")]
    for t in items:
        safe_flag = dim("safe") if t.get("safe") else yellow("with-confirmation")
        lines.append(f"  {cyan(t['name'])}  — {t['description']}  {safe_flag}")
    return "\n".join(lines)


def _cmd_run(args: list[str]) -> str:
    if not args:
        return yellow("Usage: /run <tool_name> [target=local]")
    tool_name = args[0]
    target = args[1] if len(args) > 1 else "local"
    resp = _api_post("/v1/tools/run", {"tool": tool_name, "target": target})
    if not resp:
        return red("❌ no response")
    return _display_tool_result(resp)


def _cmd_health() -> str:
    resp = _api_get("/health")
    if not resp or "_error" in (resp or {}):
        return red(f"❌ API unreachable at {API_BASE}")
    s = resp.get("status", "?")
    p = resp.get("primary", "?")
    return f"API {green(s)}  model: {p}"


# ── Main loop ────────────────────────────────────────────────────────


def main():
    # ── Sanity ──
    try:
        h = _api_get("/health")
        if not h or not h.get("status"):
            print(red(f"⚠  API not responding at {API_BASE}"))
            print(dim("  Start server: cd NPUShield && .venv/bin/python -m src.api.server"))
            sys.exit(1)
    except Exception:
        print(red(f"⚠  Cannot connect to API at {API_BASE}"))
        sys.exit(1)

    model = DEFAULT_MODEL
    history_path = os.path.expanduser("~/.local/share/npushield/history.txt")
    os.makedirs(os.path.dirname(history_path), exist_ok=True)

    if HAS_PROMPT:
        session = PromptSession(
            history=FileHistory(history_path),
            style=Style.from_dict({"prompt": "ansicyan bold"}),
        )
        prompt = session.prompt
    else:
        prompt = lambda msg: input(msg)  # noqa: E731

    # ── Show welcome ──
    cols, _ = shutil.get_terminal_size()
    print("=" * cols)
    print(bold("NPUShield Interactive"))
    print(dim(f"  API: {API_BASE}"))
    print(dim(f"  Model: {model}"))
    print(dim("  Type /help for commands"))
    print("=" * cols)
    print()

    # ── Loop ──
    while True:
        try:
            text = prompt(FormattedText([("class:prompt", "npushield> ")]) if HAS_PROMPT else "npushield> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        raw = text.strip()
        if not raw:
            continue

        # Built-ins
        if raw.startswith("/"):
            cmd_parts = raw[1:].split()
            cmd = cmd_parts[0].lower()
            args = cmd_parts[1:]

            if cmd in ("exit", "quit", "q"):
                print(green("👋 bye"))
                break
            elif cmd in ("help", "h"):
                print(_cmd_help())
            elif cmd == "tools":
                print(_cmd_tools())
            elif cmd == "run":
                print(_cmd_run(args))
            elif cmd == "health":
                print(_cmd_health())
            elif cmd == "model":
                if args:
                    model = args[0]
                    print(dim(f"model → {model}"))
                else:
                    print(dim(f"current: {model}"))
            elif cmd in ("clear", "cls"):
                os.system("clear" if os.name == "posix" else "cls")
            elif cmd == "history":
                if HAS_PROMPT:
                    for i, entry in enumerate(session.history.get_strings(), 1):
                        print(f"{i:>3}  {entry}")
                else:
                    try:
                        print(open(history_path).read())
                    except Exception:
                        print("(empty)")
            else:
                print(yellow(f"unknown command: {raw}"))

            print()
            continue

        # ── Chat ──
        resp = _api_post("/v1/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": raw}],
            "max_tokens": 480,
            "stream": False,
        })
        output = _display_response(resp)
        print(output)
        print()


if __name__ == "__main__":
    main()
