"""Lightweight in-process metrics collector for NPUShield.

No external dependencies — exposes a /metrics endpoint in Prometheus
text format using only stdlib counters.

Tracked:
  npushield_requests_total{endpoint, status}
  npushield_inference_duration_seconds_sum / _count
  npushield_tool_runs_total{tool, exit_code}
  npushield_rag_docs_retrieved_total
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict


class MetricsCollector:
    """Thread-safe in-process metrics store."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # counters: {label_tuple: float}
        self._request_counts: dict[tuple, int] = defaultdict(int)
        self._tool_counts: dict[tuple, int] = defaultdict(int)
        self._rag_docs_total: int = 0
        self._inference_duration_sum: float = 0.0
        self._inference_duration_count: int = 0
        self._start_time: float = time.time()
        self._inflight: int = 0

    def record_request(self, endpoint: str, status: int) -> None:
        with self._lock:
            self._request_counts[(endpoint, str(status))] += 1

    def record_inference(self, duration_sec: float) -> None:
        with self._lock:
            self._inference_duration_sum += duration_sec
            self._inference_duration_count += 1

    def record_tool_run(self, tool: str, exit_code: int) -> None:
        with self._lock:
            self._tool_counts[(tool, str(exit_code))] += 1

    def record_rag_docs(self, count: int) -> None:
        with self._lock:
            self._rag_docs_total += count

    def inc_inflight(self) -> None:
        with self._lock:
            self._inflight += 1

    def dec_inflight(self) -> None:
        with self._lock:
            self._inflight = max(0, self._inflight - 1)

    def render_prometheus(self) -> str:
        """Render all metrics in Prometheus text exposition format."""
        lines: list[str] = []
        uptime = time.time() - self._start_time

        with self._lock:
            # uptime
            lines.append("# HELP npushield_uptime_seconds Server uptime in seconds")
            lines.append("# TYPE npushield_uptime_seconds gauge")
            lines.append(f"npushield_uptime_seconds {uptime:.2f}")

            # request counter
            lines.append("# HELP npushield_requests_total Total HTTP requests")
            lines.append("# TYPE npushield_requests_total counter")
            for (endpoint, status), count in self._request_counts.items():
                lines.append(
                    f'npushield_requests_total{{endpoint="{endpoint}",status="{status}"}} {count}'
                )

            # inference duration
            lines.append("# HELP npushield_inference_duration_seconds_sum Sum of inference durations")
            lines.append("# TYPE npushield_inference_duration_seconds_sum counter")
            lines.append(f"npushield_inference_duration_seconds_sum {self._inference_duration_sum:.4f}")
            lines.append("# HELP npushield_inference_duration_seconds_count Number of inferences")
            lines.append("# TYPE npushield_inference_duration_seconds_count counter")
            lines.append(f"npushield_inference_duration_seconds_count {self._inference_duration_count}")

            # tool runs
            lines.append("# HELP npushield_tool_runs_total Total tool executions")
            lines.append("# TYPE npushield_tool_runs_total counter")
            for (tool, exit_code), count in self._tool_counts.items():
                lines.append(
                    f'npushield_tool_runs_total{{tool="{tool}",exit_code="{exit_code}"}} {count}'
                )

            # rag docs
            lines.append("# HELP npushield_rag_docs_retrieved_total Total RAG docs retrieved")
            lines.append("# TYPE npushield_rag_docs_retrieved_total counter")
            lines.append(f"npushield_rag_docs_retrieved_total {self._rag_docs_total}")

            # inflight
            lines.append("# HELP npushield_inflight_requests Current inflight requests")
            lines.append("# TYPE npushield_inflight_requests gauge")
            lines.append(f"npushield_inflight_requests {self._inflight}")

        return "\n".join(lines) + "\n"


# Global singleton
metrics = MetricsCollector()
