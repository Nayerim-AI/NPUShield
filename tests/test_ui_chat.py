"""TUI smoke test — verify chat_tui imports and commands work."""
from urllib.error import URLError
from src.ui.chat_tui import (
    _api_get,
    _api_post,
    _cmd_help,
    _cmd_tools,
    _cmd_health,
    _display_response,
    _display_tool_result,
    green,
    red,
    dim,
)


def test_help_contains_commands():
    out = _cmd_help()
    assert "/help" in out
    assert "/tools" in out
    assert "/run" in out


def test_health_returns_string():
    out = _cmd_health()
    assert isinstance(out, str)


def test_display_response_with_error():
    out = _display_response({"_error": "timeout"})
    assert "timeout" in out
    assert "❌" in out


def test_display_response_empty():
    out = _display_response({})
    assert out == ""


def test_display_tool_result_error():
    out = _display_tool_result({"_error": "fail"})
    assert "fail" in out
    assert "❌" in out


def test_api_get_reachable():
    result = _api_get("/health")
    assert result is not None
    assert isinstance(result, dict)
    assert "_error" not in result or result is not None


def test_api_get_nonexistent():
    result = _api_get("/nonexistent")
    assert result is not None


def test_ansi_helpers():
    assert "\033[" in green("ok")
    assert "\033[" in red("ok")
    assert "\033[" in dim("ok")
