"""
Real-time monitoring for Docker sandbox containers.

Tracks: memory usage, CPU percentage, PID count, OOM kills.
Provides anomaly detection for production deployments.
"""

import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ContainerMetrics:
    """Snapshot of container resource usage."""
    container_name: str
    timestamp: float
    memory_usage_mb: float = 0.0
    memory_limit_mb: float = 0.0
    cpu_percent: float = 0.0
    pids: int = 0
    exit_code: Optional[int] = None
    oom_killed: bool = False
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "container": self.container_name,
            "timestamp": self.timestamp,
            "memory_mb": round(self.memory_usage_mb, 2),
            "memory_limit_mb": round(self.memory_limit_mb, 2),
            "cpu_percent": round(self.cpu_percent, 2),
            "pids": self.pids,
            "exit_code": self.exit_code,
            "oom_killed": self.oom_killed,
        }


class SandboxMonitor:
    """Monitor Docker sandbox containers for anomalies."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.history: list[ContainerMetrics] = []
        self._stop_events: dict[str, threading.Event] = {}
        self._threads: dict[str, threading.Thread] = {}

    def start(self, container_name: str) -> None:
        """Start monitoring a container."""
        if container_name in self._threads:
            return

        stop_event = threading.Event()
        self._stop_events[container_name] = stop_event

        def monitor():
            while not stop_event.is_set():
                try:
                    proc = subprocess.run(
                        ["docker", "stats", container_name, "--no-stream",
                         "--format", "{{.MemUsage}},{{.CPUPerc}},{{.PIDs}}"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if proc.returncode == 0 and proc.stdout.strip():
                        parts = proc.stdout.strip().split(",")
                        # Parse memory: "50MiB / 256MiB"
                        mem_parts = parts[0].split(" / ")
                        mem_usage = self._parse_mem(mem_parts[0])
                        mem_limit = self._parse_mem(mem_parts[1]) if len(mem_parts) > 1 else 0

                        # Parse CPU: "25.00%"
                        cpu_str = parts[1].replace("%", "")
                        cpu = float(cpu_str) if cpu_str else 0.0

                        # Parse PIDs
                        pids = int(parts[2]) if len(parts) > 2 else 0

                        metrics = ContainerMetrics(
                            container_name=container_name,
                            timestamp=time.monotonic(),
                            memory_usage_mb=mem_usage,
                            memory_limit_mb=mem_limit,
                            cpu_percent=cpu,
                            pids=pids,
                        )
                        self.history.append(metrics)
                    else:
                        break
                except Exception:
                    break
                time.sleep(self.interval)

        thread = threading.Thread(target=monitor, daemon=True)
        self._threads[container_name] = thread
        thread.start()

    def stop(self, container_name: str) -> None:
        """Stop monitoring a container."""
        if container_name in self._stop_events:
            self._stop_events[container_name].set()
            del self._stop_events[container_name]
        if container_name in self._threads:
            del self._threads[container_name]

    def get_anomalies(self, container_name: Optional[str] = None) -> list[str]:
        """Detect anomalous behavior in monitored containers."""
        alerts = []
        metrics = self.history
        if container_name:
            metrics = [m for m in metrics if m.container_name == container_name]

        for m in metrics:
            if m.memory_usage_mb > m.memory_limit_mb * 0.9:
                alerts.append(
                    f"HIGH MEMORY: {m.memory_usage_mb:.1f}MB / {m.memory_limit_mb:.1f}MB "
                    f"({m.memory_usage_mb/m.memory_limit_mb*100:.0f}%) "
                    f"in {m.container_name}"
                )
            if m.cpu_percent > 90:
                alerts.append(
                    f"HIGH CPU: {m.cpu_percent:.1f}% in {m.container_name}"
                )
            if m.pids > 80:
                alerts.append(
                    f"HIGH PIDs: {m.pids} in {m.container_name}"
                )
            if m.oom_killed:
                alerts.append(
                    f"OOM KILLED: {m.container_name}"
                )

        return alerts

    def get_summary(self, container_name: Optional[str] = None) -> dict:
        """Get summary statistics."""
        metrics = self.history
        if container_name:
            metrics = [m for m in metrics if m.container_name == container_name]

        if not metrics:
            return {"count": 0}

        return {
            "count": len(metrics),
            "avg_memory_mb": round(sum(m.memory_usage_mb for m in metrics) / len(metrics), 2),
            "max_memory_mb": round(max(m.memory_usage_mb for m in metrics), 2),
            "avg_cpu_percent": round(sum(m.cpu_percent for m in metrics) / len(metrics), 2),
            "max_cpu_percent": round(max(m.cpu_percent for m in metrics), 2),
            "max_pids": max(m.pids for m in metrics),
            "oom_kills": sum(1 for m in metrics if m.oom_killed),
            "anomalies": len(self.get_anomalies(container_name)),
        }

    @staticmethod
    def _parse_mem(mem_str: str) -> float:
        """Parse memory string to MB."""
        mem_str = mem_str.strip()
        if not mem_str or mem_str == "0B":
            return 0.0

        units = {"KiB": 1/1024, "MiB": 1, "GiB": 1024, "TiB": 1024*1024,
                 "kB": 1/1000, "MB": 1, "GB": 1000, "TB": 1000*1000}

        for unit, factor in units.items():
            if unit in mem_str:
                val = float(mem_str.replace(unit, "").strip())
                return val * factor

        return float(mem_str)


if __name__ == "__main__":
    print("SandboxMonitor loaded.")
    print("Usage:")
    print("  monitor = SandboxMonitor(interval=0.5)")
    print("  monitor.start('container-name')")
    print("  # ... run sandbox ...")
    print("  alerts = monitor.get_anomalies('container-name')")
    print("  summary = monitor.get_summary('container-name')")
    print("  monitor.stop('container-name')")
