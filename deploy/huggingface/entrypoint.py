#!/usr/bin/env python3
"""Supervise the API, dashboard, and reverse proxy inside a single container.

Hugging Face Spaces publishes exactly one port to the internet, so the two
services that compose runs separately must share this container behind nginx:

    127.0.0.1:8000  uvicorn   <- internal only
    127.0.0.1:7861  streamlit <- internal only
    0.0.0.0:7860    nginx     <- the single port HF publishes

The proxy preserves the service boundary: the dashboard still calls the API over
HTTP rather than importing the models, so the deployed topology matches the
compose one instead of being a different architecture.

This script exists to solve the orphan problem. With a shell ``&`` and ``wait``,
a crashed API would leave nginx happily serving a broken UI. Here, any process
exiting tears the container down so HF reports the failure instead of showing a
half-working demo.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time

# Internal-only ports. Nothing external reaches these; nginx owns the traffic.
API_PORT = int(os.environ.get("WINE_API_PORT", "8000"))
DASHBOARD_PORT = int(os.environ.get("WINE_DASHBOARD_INTERNAL_PORT", "7861"))

# The single port HF publishes. Must match `app_port` in the Space README.
PUBLIC_PORT = int(os.environ.get("PORT", "7860"))

# How long to wait for each service before giving up.
STARTUP_TIMEOUT = float(os.environ.get("WINE_STARTUP_TIMEOUT", "90"))


def log(message: str) -> None:
    """Write a prefixed line to stdout, unbuffered for prompt log delivery."""
    print(f"[entrypoint] {message}", flush=True)


def wait_for_port(port: int, timeout: float, host: str = "127.0.0.1") -> bool:
    """Poll a TCP port until something accepts a connection.

    Args:
        port: Port to probe.
        timeout: Seconds to keep trying before giving up.
        host: Interface to connect to.

    Returns:
        ``True`` once the port accepts connections, ``False`` on timeout.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(1.0)
            if probe.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.5)
    return False


def main() -> int:
    """Start every service and return the exit code of whichever fails first.

    Returns:
        Exit code of the process that terminated, so the container reflects the
        real failure rather than a supervisor artefact.
    """
    log(f"project root: {os.environ.get('WINE_PROJECT_ROOT', '/app')}")
    log(f"api        -> 127.0.0.1:{API_PORT} (internal)")
    log(f"dashboard  -> 127.0.0.1:{DASHBOARD_PORT} (internal)")
    log(f"nginx      -> 0.0.0.0:{PUBLIC_PORT} (published)")

    env = {
        **os.environ,
        # The dashboard reaches the API over loopback. This is the single change
        # from the compose topology, where it used the `api` hostname. The /api
        # path prefix belongs to the browser's view of nginx, not to this call.
        "WINE_API_BASE_URL": f"http://127.0.0.1:{API_PORT}",
        "WINE_API_PORT": str(API_PORT),
    }

    api = subprocess.Popen(
        [
            "uvicorn",
            "wine_quality.serving.api:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(API_PORT),
            # Tells FastAPI it is mounted at /api, so the auto-generated docs
            # reference the publicly reachable URLs instead of bare paths.
            "--root-path",
            "/api",
            "--log-level",
            "info",
        ],
        env=env,
    )
    log(f"api started (pid {api.pid})")

    # Fail fast if the API never comes up. Otherwise nginx would serve a
    # dashboard whose every prediction 503s, which is much harder to diagnose.
    if not wait_for_port(API_PORT, STARTUP_TIMEOUT):
        log(f"ERROR: api did not accept connections within {STARTUP_TIMEOUT:.0f}s")
        api.terminate()
        return 1
    log("api is accepting connections")

    dashboard = subprocess.Popen(
        [
            "streamlit",
            "run",
            "app/Overview.py",
            "--server.port",
            str(DASHBOARD_PORT),
            "--server.address",
            "127.0.0.1",
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
            # nginx terminates TLS and rewrites headers, so XSRF protection would
            # reject the proxied WebSocket upgrade. Safe here because the Space
            # is a public read-only demo with no authenticated actions.
            "--server.enableCORS",
            "false",
            "--server.enableXsrfProtection",
            "false",
        ],
        env=env,
    )
    log(f"dashboard started (pid {dashboard.pid})")

    if not wait_for_port(DASHBOARD_PORT, STARTUP_TIMEOUT):
        log(f"ERROR: dashboard did not accept connections within {STARTUP_TIMEOUT:.0f}s")
        api.terminate()
        dashboard.terminate()
        return 1
    log("dashboard is accepting connections")

    nginx = subprocess.Popen(["nginx", "-g", "daemon off;"])
    log(f"nginx started (pid {nginx.pid})")

    processes = {"api": api, "dashboard": dashboard, "nginx": nginx}
    exit_code = 0

    def shutdown(signum: int, _frame: object) -> None:
        """Forward termination signals to every child."""
        log(f"received signal {signum}, shutting down")
        for name, process in processes.items():
            if process.poll() is None:
                log(f"terminating {name} (pid {process.pid})")
                process.terminate()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    # Wait for whichever process exits first, then take the rest down with it.
    while True:
        for name, process in processes.items():
            code = process.poll()
            if code is None:
                continue

            log(f"{name} exited with code {code}")
            exit_code = code

            for other_name, other in processes.items():
                if other_name != name and other.poll() is None:
                    log(f"terminating {other_name} (pid {other.pid})")
                    other.terminate()

            for process_to_join in processes.values():
                try:
                    process_to_join.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    log("forcing kill after timeout")
                    process_to_join.kill()

            return exit_code

        time.sleep(0.5)


if __name__ == "__main__":
    sys.exit(main())