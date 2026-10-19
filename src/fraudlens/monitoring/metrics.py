"""In-process operational metric collection."""

from __future__ import annotations

import threading
import time
from collections import defaultdict


class MetricsCollector:
    """Thread-safe counters and latency timings for the /metrics endpoint."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = defaultdict(int)
        self._timings: dict[str, list[float]] = defaultdict(list)

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] += amount

    def observe(self, name: str, duration_ms: float) -> None:
        with self._lock:
            self._timings[name].append(duration_ms)

    def counter(self, name: str) -> int:
        with self._lock:
            return self._counters.get(name, 0)

    def timing(self, name: str) -> dict[str, float]:
        with self._lock:
            samples = self._timings.get(name, [])
        if not samples:
            return {"count": 0, "avg_ms": 0.0, "p95_ms": 0.0}
        ordered = sorted(samples)
        index = max(0, int(round(0.95 * len(ordered))) - 1)
        return {
            "count": len(ordered),
            "avg_ms": round(sum(ordered) / len(ordered), 2),
            "p95_ms": round(ordered[index], 2),
        }

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "counters": dict(self._counters),
                "timings": {name: self.timing(name) for name in list(self._timings)},
            }


class Timer:
    """Context manager that records elapsed milliseconds into the collector."""

    def __init__(self, collector: MetricsCollector, name: str) -> None:
        self.collector = collector
        self.name = name
        self._started = 0.0

    def __enter__(self) -> "Timer":
        self._started = time.perf_counter()
        return self

    def __exit__(self, *exc_info) -> None:
        self.collector.observe(self.name, (time.perf_counter() - self._started) * 1000)


metrics = MetricsCollector()
