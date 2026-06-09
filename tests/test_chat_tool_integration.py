"""Test chat-to-tool auto-execute integration."""

from src.core.tool_executor import ToolResult, ToolTarget, ToolExecutor
from src.core.tool_registry import ToolRegistry, Tool


def test_status_tool_matches_and_outputs_ok():
    registry = ToolRegistry.default()
    executor = ToolExecutor()

    tool = registry.match_intent("what is the top status of the server?")

    assert tool is not None
    assert tool.name == "server_status_top"
    assert tool.safe is True

    result = executor.run_commands(tool.commands, target=ToolTarget.local())

    assert result.exit_code == 0
    assert any(s in result.stdout.lower() for s in ["load", "mem", "cpu", "disk"])


def test_intent_no_match_falls_through_gracefully():
    registry = ToolRegistry.default()

    tool = registry.match_intent("what is the meaning of life?")

    assert tool is None


def test_safe_restart_rejected_without_confirmation():
    executor = ToolExecutor()
    result = executor.run_commands(
        ["ssh user@server 'docker restart gitea'"],
        target=ToolTarget.local()
    )

    assert result.exit_code == -1
    assert "rejected" in result.stderr.lower()


def test_restart_requires_confirmation_via_tool():
    registry = ToolRegistry.default()

    tool = registry.get("safe_restart_service")

    assert tool is not None
    assert tool.requires_confirmation is True
    assert tool.safe is False
def test_explain_query_should_not_skip_llm():
    """'jelaskan container' should keep skip_llm=False (pass output to LLM)."""
    from src.api.server import _is_explain_query

    assert _is_explain_query("jelaskan container apa aja itu") is True
    assert _is_explain_query("explain what containers are running") is True
    assert _is_explain_query("check top status server") is False
    assert _is_explain_query("apa itu rkllm_init") is True
    assert _is_explain_query("list docker containers") is False
    assert _is_explain_query("how to restart docker") is True
    assert _is_explain_query("show running containers") is False
