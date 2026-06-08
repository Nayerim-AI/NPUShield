"""Tests for the tool executor — runs allowlisted commands on targets."""

from src.core.tool_executor import ToolExecutor, ToolTarget, ToolResult


def test_execute_simple_local_command_returns_output():
    executor = ToolExecutor()

    result = executor.run_commands(["echo hello world"], target=ToolTarget.local())

    assert result.exit_code == 0
    assert "hello world" in result.stdout


def test_ssh_connection_error_returns_structured_error():
    executor = ToolExecutor()

    result = executor.run_commands(["echo hi"], target=ToolTarget.local())

    assert result.target_name == "local"
    assert isinstance(result.exit_code, int)


def test_dangerous_pattern_is_rejected():
    executor = ToolExecutor(dangerous_patterns=["rm -rf"])

    result = executor.run_commands(["rm -rf /"], target=ToolTarget.local())

    assert result.exit_code == -1
    assert "rejected" in result.stderr.lower()
