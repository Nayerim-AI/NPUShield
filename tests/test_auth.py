"""Tests for API key authentication middleware."""

from __future__ import annotations

import os
import pytest


@pytest.fixture(autouse=True)
def _clear_api_key(monkeypatch):
    """Ensure clean state for each test."""
    monkeypatch.delenv("NPUSHIELD_API_KEY", raising=False)


def test_auth_disabled_when_no_env(monkeypatch):
    monkeypatch.delenv("NPUSHIELD_API_KEY", raising=False)
    # Must reimport to pick up env change at module level
    import importlib
    import src.core.auth as auth_mod
    monkeypatch.setattr(auth_mod, "_CONFIGURED_KEY", None)
    # Should not raise
    auth_mod.require_api_key(api_key=None)


def test_auth_rejects_missing_key(monkeypatch):
    import importlib
    import src.core.auth as auth_mod
    from fastapi import HTTPException
    monkeypatch.setattr(auth_mod, "_CONFIGURED_KEY", "test-secret-123")
    with pytest.raises(HTTPException) as exc_info:
        auth_mod.require_api_key(api_key=None)
    assert exc_info.value.status_code == 401


def test_auth_rejects_wrong_key(monkeypatch):
    import src.core.auth as auth_mod
    from fastapi import HTTPException
    monkeypatch.setattr(auth_mod, "_CONFIGURED_KEY", "test-secret-123")
    with pytest.raises(HTTPException) as exc_info:
        auth_mod.require_api_key(api_key="wrong-key")
    assert exc_info.value.status_code == 403


def test_auth_accepts_correct_key(monkeypatch):
    import src.core.auth as auth_mod
    monkeypatch.setattr(auth_mod, "_CONFIGURED_KEY", "test-secret-123")
    # Should not raise
    auth_mod.require_api_key(api_key="test-secret-123")


def test_auth_uses_constant_time_compare(monkeypatch):
    """Ensure timing-safe comparison is used (no plain ==)."""
    import src.core.auth as auth_mod
    import inspect
    src = inspect.getsource(auth_mod.require_api_key)
    assert "compare_digest" in src, "Auth must use secrets.compare_digest"
    assert "==" not in src.split("compare_digest")[1].split("raise")[0], \
        "No plain == after compare_digest check"
