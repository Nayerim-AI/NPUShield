"""Tests for MetricsCollector — counters, inflight gauge, Prometheus format."""
from __future__ import annotations

import re
from src.metrics.collector import MetricsCollector


def _make() -> MetricsCollector:
    return MetricsCollector()


# ── counter tests ──────────────────────────────────────────────────────────

def test_record_request_increments():
    m = _make()
    m.record_request("/health", 200)
    m.record_request("/health", 200)
    m.record_request("/v1/chat/completions", 200)
    out = m.render_prometheus()
    assert 'endpoint="/health",status="200"} 2' in out
    assert 'endpoint="/v1/chat/completions",status="200"} 1' in out


def test_record_request_different_statuses():
    m = _make()
    m.record_request("/v1/chat/completions", 200)
    m.record_request("/v1/chat/completions", 401)
    m.record_request("/v1/chat/completions", 500)
    out = m.render_prometheus()
    assert 'status="200"} 1' in out
    assert 'status="401"} 1' in out
    assert 'status="500"} 1' in out


def test_record_inference():
    m = _make()
    m.record_inference(1.5)
    m.record_inference(2.5)
    out = m.render_prometheus()
    assert "npushield_inference_duration_seconds_sum 4.0" in out
    assert "npushield_inference_duration_seconds_count 2" in out


def test_record_tool_run():
    m = _make()
    m.record_tool_run("server_status_top", 0)
    m.record_tool_run("server_status_top", 0)
    m.record_tool_run("docker_ps", 1)
    out = m.render_prometheus()
    assert 'tool="server_status_top",exit_code="0"} 2' in out
    assert 'tool="docker_ps",exit_code="1"} 1' in out


def test_record_rag_docs():
    m = _make()
    m.record_rag_docs(3)
    m.record_rag_docs(5)
    out = m.render_prometheus()
    assert "npushield_rag_docs_retrieved_total 8" in out


# ── inflight gauge ─────────────────────────────────────────────────────────

def test_inflight_increments_and_decrements():
    m = _make()
    assert "npushield_inflight_requests 0" in m.render_prometheus()
    m.inc_inflight()
    m.inc_inflight()
    assert "npushield_inflight_requests 2" in m.render_prometheus()
    m.dec_inflight()
    assert "npushield_inflight_requests 1" in m.render_prometheus()


def test_inflight_no_negative():
    m = _make()
    m.dec_inflight()  # should not go below 0
    assert "npushield_inflight_requests 0" in m.render_prometheus()


def test_inflight_concurrent():
    """Basic thread-safety: many inc/dec pairs should net to zero."""
    import threading
    m = _make()
    errors = []

    def worker():
        try:
            for _ in range(100):
                m.inc_inflight()
                m.dec_inflight()
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert "npushield_inflight_requests 0" in m.render_prometheus()


# ── Prometheus format ──────────────────────────────────────────────────────

def test_prometheus_output_has_help_and_type_lines():
    m = _make()
    m.record_request("/health", 200)
    out = m.render_prometheus()
    assert "# HELP npushield_requests_total" in out
    assert "# TYPE npushield_requests_total counter" in out
    assert "# HELP npushield_uptime_seconds" in out
    assert "# TYPE npushield_uptime_seconds gauge" in out
    assert "# HELP npushield_inflight_requests" in out
    assert "# TYPE npushield_inflight_requests gauge" in out


def test_prometheus_uptime_is_positive():
    m = _make()
    out = m.render_prometheus()
    match = re.search(r"npushield_uptime_seconds (\d+\.\d+)", out)
    assert match
    assert float(match.group(1)) >= 0.0


def test_prometheus_ends_with_newline():
    m = _make()
    assert m.render_prometheus().endswith("\n")


def test_empty_collector_renders_without_error():
    m = _make()
    out = m.render_prometheus()
    assert "npushield_uptime_seconds" in out
    assert "npushield_inflight_requests 0" in out
