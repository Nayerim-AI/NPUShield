"""NPUShield interactive chat TUI — multi-panel dashboard client for local API."""

from __future__ import annotations

import json
import os
import shutil
import sys
import textwrap
import time
import threading
from datetime import datetime
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

API_BASE = os.getenv("NPUSHIELD_HOST", "http://127.0.0.1:18999")
API_KEY = os.getenv("NPUSHIELD_API_KEY", "").strip() or None
DEFAULT_MODEL = "rkllm-qwen2.5-1.5b"

try:
    from prompt_toolkit import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, VSplit, Layout
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.widgets import TextArea, Frame
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.styles import Style
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.filters import has_focus

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


def _api_get_text(path: str) -> str | None:
    url = f"{API_BASE}{path}"
    headers = {}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    req = Request(url, headers=headers, method="GET")
    try:
        with urlopen(req, timeout=10) as resp:
            return resp.read().decode()
    except Exception:
        return None


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


# ── Sparkline helper ─────────────────────────────────────────────────

_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def _sparkline(values: list[float]) -> str:
    if not values:
        return ""
    lo = min(values)
    hi = max(values)
    span = hi - lo or 1.0
    out = []
    for v in values:
        idx = int((v - lo) / span * (len(_SPARK_CHARS) - 1))
        idx = max(0, min(idx, len(_SPARK_CHARS) - 1))
        out.append(_SPARK_CHARS[idx])
    return "".join(out)


# ── Dashboard TUI ────────────────────────────────────────────────────

class NPUShieldDashboard:
    """Multi-panel prompt_toolkit dashboard."""

    def __init__(self) -> None:
        self.model = DEFAULT_MODEL
        self.start_time = time.time()
        self.chat_lines: list[str] = []
        self.log_lines: list[str] = []
        self.token_rate_history: list[float] = []
        self.health_data: dict[str, str] = {}
        self.metrics_data: dict[str, str] = {}
        self.running = True
        self.history_path = os.path.expanduser("~/.local/share/npushield/history.txt")
        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
        self._history = FileHistory(self.history_path)

    # ── formatting helpers for TUI panels ──

    def _uptime_str(self) -> str:
        elapsed = int(time.time() - self.start_time)
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}h{m:02d}m"
        return f"{m}m{s:02d}s"

    def _header_text(self) -> FormattedText:
        title = (
            "╔═╗╔╗╔╔═╗╦╔╗╔╔═╗╦  ╦╔═╗╦═╗\n"
            "╠═╣║║║╚═╗║║║║║ ║╚╗╔╝║╣ ╠╦╝\n"
            "╩ ╩╝╚╝╚═╝╩╝╚╝╚═╝ ╚╝ ╚═╝╩╚═"
        )
        return FormattedText([
            ("class:header", f"{title}\n"),
            ("class:header-info", f"  model: {self.model}  │  uptime: {self._uptime_str()}  │  API: {API_BASE}\n"),
        ])

    def _footer_text(self) -> FormattedText:
        now = datetime.now().strftime("%H:%M:%S")
        return FormattedText([
            ("class:footer", f" [Ctrl-C/Ctrl-D: exit] [Enter: send]  │  {now}  │  prompt_toolkit TUI"),
        ])

    def _chat_formatted(self) -> FormattedText:
        if not self.chat_lines:
            return FormattedText([("class:dim", "  (no messages yet — type a message or /help)")])
        result: list[tuple[str, str]] = []
        for line in self.chat_lines:
            if line.startswith("► "):
                result.append(("class:user-input", line + "\n"))
            elif line.startswith("◄ "):
                result.append(("class:assistant", line + "\n"))
            elif line.startswith("• "):
                result.append(("class:system-info", line + "\n"))
            else:
                result.append(("", line + "\n"))
        return FormattedText(result)

    def _logs_formatted(self) -> FormattedText:
        if not self.log_lines:
            return FormattedText([("class:dim", "  (no logs yet)")])
        result: list[tuple[str, str]] = []
        tail = self.log_lines[-20:]
        for line in tail:
            if "error" in line.lower() or "fail" in line.lower():
                result.append(("class:log-error", line + "\n"))
            elif "warn" in line.lower():
                result.append(("class:log-warn", line + "\n"))
            else:
                result.append(("class:log-info", line + "\n"))
        return FormattedText(result)

    def _stats_formatted(self) -> FormattedText:
        result: list[tuple[str, str]] = []
        result.append(("class:stat-header", "╔══ Health ══╗\n"))

        status = self.health_data.get("status", "—")
        if status == "healthy":
            result.append(("class:stat-ok", f"  status: ✓ {status}\n"))
        elif status == "—":
            result.append(("class:dim", "  status: polling…\n"))
        else:
            result.append(("class:stat-warn", f"  status: ✗ {status}\n"))

        primary = self.health_data.get("primary", "—")
        fallback = self.health_data.get("fallback", "—")
        result.append(("", f"  primary: {primary}\n"))
        if fallback and fallback != "None":
            result.append(("", f"  fallback: {fallback}\n"))

        result.append(("class:stat-header", "\n╔══ Metrics ══╗\n"))
        uptime_s = self.metrics_data.get("uptime", "—")
        reqs = self.metrics_data.get("requests", "—")
        inf_count = self.metrics_data.get("inference_count", "—")
        inf_sum = self.metrics_data.get("inference_sum", "—")
        result.append(("", f"  uptime: {uptime_s}s\n"))
        result.append(("", f"  requests: {reqs}\n"))
        result.append(("", f"  inferences: {inf_count}\n"))
        result.append(("", f"  inf time Σ: {inf_sum}s\n"))

        result.append(("class:stat-header", "\n╔══ Token Rate ══╗\n"))
        if self.token_rate_history:
            spark = _sparkline(self.token_rate_history)
            result.append(("class:sparkline", f"  {spark}\n"))
            result.append(("class:dim", f"  last: {self.token_rate_history[-1]:.1f} tok/s\n"))
        else:
            result.append(("class:dim", "  (waiting for data…)\n"))

        return FormattedText(result)

    # ── build layout ──

    def _build_layout(self) -> Layout:
        header = Frame(
            FormattedTextControl(self._header_text),
            style="class:header-frame",
        )

        self._chat_control = FormattedTextControl(self._chat_formatted)
        chat_area = Frame(
            self._chat_control,
            title="Chat",
            style="class:chat-frame",
        )

        self._log_control = FormattedTextControl(self._logs_formatted)
        log_area = Frame(
            self._log_control,
            title="Logs",
            style="class:log-frame",
        )

        self._stats_control = FormattedTextControl(self._stats_formatted)
        stats_area = Frame(
            self._stats_control,
            title="Stats",
            style="class:stats-frame",
        )

        self.input_area = TextArea(
            prompt=[("class:prompt", "npushield> ")],
            height=1,
            multiline=False,
            wrap_lines=False,
            style="class:input-area",
            history=self._history,
        )

        footer = Frame(
            FormattedTextControl(self._footer_text),
            style="class:footer-frame",
        )

        left_pane = HSplit([
            chat_area,
            log_area,
        ], padding=0)

        right_pane = HSplit([
            stats_area,
        ])

        body = VSplit([
            left_pane,
            right_pane,
        ], padding=1)

        root = HSplit([
            header,
            body,
            self.input_area,
            footer,
        ], padding=0)

        return Layout(root, focused_element=self.input_area)

    # ── key bindings ──

    def _build_bindings(self) -> KeyBindings:
        kb = KeyBindings()
        app_ref = self  # capture for closure

        @kb.add("enter", eager=True)
        def _(event):
            text = app_ref.input_area.text.strip()
            if not text:
                return
            app_ref.input_area.text = ""
            app_ref._handle_input(text)

        @kb.add("c-c")
        @kb.add("c-d")
        def _(event):
            app_ref.running = False
            event.app.exit()

        @kb.add("c-l")
        def _(event):
            app_ref.chat_lines.clear()

        @kb.add("c-x")
        def _(event):
            """Toggle focus between input and log panel."""
            pass

        return kb

    # ── input handling ──

    def _append_chat(self, line: str) -> None:
        self.chat_lines.append(line)

    def _append_log(self, line: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_lines.append(f"[{ts}] {line}")

    def _handle_input(self, raw: str) -> None:
        self._append_log(f">>> {raw}")

        if raw.startswith("/"):
            parts = raw[1:].split()
            cmd = parts[0].lower()
            args = parts[1:]

            if cmd in ("exit", "quit", "q"):
                self._append_chat("• bye!")
                self.running = False
                return
            elif cmd in ("help", "h"):
                self._append_chat(_cmd_help())
            elif cmd == "tools":
                self._append_chat(_cmd_tools())
            elif cmd == "run":
                self._append_chat(_cmd_run(args))
            elif cmd == "health":
                self._append_chat(_cmd_health())
            elif cmd == "model":
                if args:
                    self.model = args[0]
                    self._append_chat(f"• model → {self.model}")
                else:
                    self._append_chat(f"• current model: {self.model}")
            elif cmd in ("clear", "cls"):
                self.chat_lines.clear()
            elif cmd == "history":
                try:
                    content = open(self.history_path).read().strip()
                    self._append_chat(content if content else "(empty)")
                except Exception:
                    self._append_chat("(empty)")
            else:
                self._append_chat(yellow(f"unknown command: {raw}"))
            return

        # Chat request
        self._append_chat(f"► {raw}")
        self._append_log(f"sending chat request…")
        self._do_chat(raw)

    def _do_chat(self, text: str) -> None:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": 480,
            "stream": True,
        }
        url = f"{API_BASE}/v1/chat/completions"
        data = json.dumps(body).encode()
        headers = {"Content-Type": "application/json"}
        if API_KEY:
            headers["X-API-Key"] = API_KEY
        req = Request(url, data=data, headers=headers, method="POST")
        try:
            with urlopen(req, timeout=180) as resp:
                # Stream: read all and parse SSE
                raw = resp.read().decode()
                content_parts = []
                for line in raw.split("\n"):
                    line = line.strip()
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        c = delta.get("content")
                        if c:
                            content_parts.append(c)
                        # extract token rate if available
                        tps = chunk.get("x-tokens-per-sec")
                        if tps is not None:
                            self.token_rate_history.append(float(tps))
                            if len(self.token_rate_history) > 20:
                                self.token_rate_history = self.token_rate_history[-20:]
                    except (json.JSONDecodeError, IndexError, KeyError):
                        pass
                if content_parts:
                    self._append_chat("◄ " + "".join(content_parts))
                else:
                    # Try non-streaming parse fallback
                    try:
                        o = json.loads(raw)
                        self._append_chat(_display_response(o))
                    except Exception:
                        self._append_chat("◄ (empty response)")
                self._append_log("chat response received")
        except URLError as e:
            self._append_chat(red(f"❌ {e}"))
            self._append_log(f"error: {e}")
        except Exception as e:
            self._append_chat(red(f"❌ {e}"))
            self._append_log(f"error: {e}")

    # ── background polling ──

    def _poll_stats(self, app: Application) -> None:
        """Background task: poll /health and /metrics every 3s."""
        while self.running and not app.is_done:
            try:
                # Health
                h = _api_get("/health")
                if h and "_error" not in h:
                    self.health_data = {
                        "status": h.get("status", "?"),
                        "primary": h.get("primary", "—"),
                        "fallback": str(h.get("fallback", "—")),
                    }
                else:
                    self.health_data = {"status": "unreachable", "primary": "—", "fallback": "—"}

                # Metrics (Prometheus text format)
                raw = _api_get_text("/metrics")
                if raw:
                    for line in raw.split("\n"):
                        if line.startswith("npushield_uptime_seconds "):
                            self.metrics_data["uptime"] = line.split()[-1]
                        elif line.startswith("npushield_requests_total{"):
                            # Sum all request counters
                            pass
                        elif line.startswith("npushield_requests_total ") or (
                            "requests_total" in line and not line.startswith("#")
                        ):
                            pass
                        elif line.startswith("npushield_inference_duration_seconds_count "):
                            self.metrics_data["inference_count"] = line.split()[-1]
                        elif line.startswith("npushield_inference_duration_seconds_sum "):
                            self.metrics_data["inference_sum"] = line.split()[-1]

                    # Compute total requests
                    total_reqs = 0
                    for line in raw.split("\n"):
                        if line.startswith("npushield_requests_total{"):
                            try:
                                total_reqs += int(line.split()[-1])
                            except ValueError:
                                pass
                    self.metrics_data["requests"] = str(total_reqs)

                    # Compute avg inference time for sparkline
                    inf_sum_raw = self.metrics_data.get("inference_sum", "0")
                    inf_count_raw = self.metrics_data.get("inference_count", "0")
                    try:
                        s = float(inf_sum_raw)
                        c = float(inf_count_raw)
                        if c > 0:
                            avg = s / c
                            self.token_rate_history.append(avg)
                            if len(self.token_rate_history) > 20:
                                self.token_rate_history = self.token_rate_history[-20:]
                    except (ValueError, ZeroDivisionError):
                        pass

                # Trigger UI refresh
                try:
                    app.invalidate()
                except Exception:
                    pass

            except Exception:
                pass

            # Sleep 3s in small increments so we can exit promptly
            for _ in range(30):
                if not self.running or app.is_done:
                    return
                time.sleep(0.1)

    # ── main entry ──

    def run(self) -> None:
        # Sanity check
        try:
            h = _api_get("/health")
            if not h or not h.get("status"):
                print(red(f"⚠  API not responding at {API_BASE}"))
                print(dim("  Start server: cd NPUShield && .venv/bin/python -m src.api.server"))
                sys.exit(1)
        except Exception:
            print(red(f"⚠  Cannot connect to API at {API_BASE}"))
            sys.exit(1)

        style = Style.from_dict({
            "header": "bold ansicyan",
            "header-info": "ansicyan",
            "footer": "ansigray",
            "prompt": "ansicyan bold",
            "user-input": "ansigreen bold",
            "assistant": "ansiwhite",
            "system-info": "ansigray italic",
            "dim": "ansigray",
            "chat-frame": "ansicyan",
            "log-frame": "ansiyellow",
            "stats-frame": "ansimagenta",
            "header-frame": "ansicyan",
            "footer-frame": "ansigray",
            "input-area": "bg:ansigray ansiwhite",
            "stat-header": "bold ansimagenta",
            "stat-ok": "ansigreen bold",
            "stat-warn": "ansiyellow bold",
            "sparkline": "ansimagenta",
            "log-error": "ansired",
            "log-warn": "ansiyellow",
            "log-info": "ansigray",
        })

        self._append_chat("• NPUShield TUI started — type /help for commands")

        layout = self._build_layout()
        bindings = self._build_bindings()

        app = Application(
            layout=layout,
            key_bindings=bindings,
            style=style,
            full_screen=True,
            mouse_support=True,
        )

        # Start background polling thread
        poll_thread = threading.Thread(
            target=self._poll_stats, args=(app,), daemon=True
        )
        poll_thread.start()

        try:
            app.run()
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False


# ── Readline fallback ────────────────────────────────────────────────


def _fallback_main():
    """Simple readline loop if prompt_toolkit is not available."""
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
    cols, _ = shutil.get_terminal_size()
    print("=" * cols)
    print(bold("NPUShield Interactive (readline fallback)"))
    print(dim(f"  API: {API_BASE}"))
    print(dim(f"  Model: {model}"))
    print(dim("  Type /help for commands"))
    print("=" * cols)
    print()

    history_path = os.path.expanduser("~/.local/share/npushield/history.txt")
    os.makedirs(os.path.dirname(history_path), exist_ok=True)

    try:
        import readline
        if os.path.exists(history_path):
            readline.read_history_file(history_path)
    except Exception:
        readline = None  # type: ignore[assignment]

    try:
        while True:
            try:
                text = input("npushield> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break

            raw = text.strip()
            if not raw:
                continue

            # Save to file history
            try:
                with open(history_path, "a") as f:
                    f.write(raw + "\n")
            except Exception:
                pass

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
                    try:
                        print(open(history_path).read())
                    except Exception:
                        print("(empty)")
                else:
                    print(yellow(f"unknown command: {raw}"))
                print()
                continue

            # Chat
            resp = _api_post("/v1/chat/completions", {
                "model": model,
                "messages": [{"role": "user", "content": raw}],
                "max_tokens": 480,
                "stream": False,
            })
            output = _display_response(resp)
            print(output)
            print()
    finally:
        try:
            if readline:
                readline.write_history_file(history_path)
        except Exception:
            pass


# ── Main entry ───────────────────────────────────────────────────────


def main():
    if HAS_PROMPT:
        dashboard = NPUShieldDashboard()
        dashboard.run()
    else:
        _fallback_main()


if __name__ == "__main__":
    main()
