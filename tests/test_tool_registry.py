from src.core.tool_registry import ToolRegistry


def test_tool_registry_lists_server_status_top():
    registry = ToolRegistry.default()

    tool = registry.get("server_status_top")

    assert tool.name == "server_status_top"
    assert tool.safe is True
    assert tool.requires_confirmation is False
    assert "uptime" in "\n".join(tool.commands)


def test_rejects_unknown_tool():
    registry = ToolRegistry.default()

    assert registry.get("rm_everything") is None


def test_tool_intent_detects_server_status_request():
    registry = ToolRegistry.default()

    match = registry.match_intent("check top status server")

    assert match is not None
    assert match.name == "server_status_top"


def test_tool_intent_detects_service_status_request():
    registry = ToolRegistry.default()

    match = registry.match_intent("is docker service running?")

    assert match is not None
    assert match.name == "service_status"


def test_restart_service_requires_confirmation():
    registry = ToolRegistry.default()

    tool = registry.get("safe_restart_service")

    assert tool is not None
    assert tool.safe is False
    assert tool.requires_confirmation is True
