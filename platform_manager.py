"""Platform integration catalog and startup-only executable detection."""

from __future__ import annotations

import os
import signal
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

from paths import INSTALL_ROOT


ExecutableResolver = Callable[[str], Optional[str]]
HealthChecker = Callable[[int], bool]
PortChecker = Callable[[int], bool]
PopenFactory = Callable[..., subprocess.Popen]


@dataclass(frozen=True)
class PlatformDefinition:
    backend_id: str
    provider: str
    display_name: str
    command_candidates: tuple[str, ...]


@dataclass(frozen=True)
class ExecutableDetection:
    detected: bool
    command: Optional[str] = None
    path: Optional[str] = None
    reason: Optional[str] = None


DEFAULT_PLATFORM_DEFINITIONS: tuple[PlatformDefinition, ...] = (
    PlatformDefinition(
        backend_id="platform:codex",
        provider="codex",
        display_name="Codex",
        command_candidates=("codex", "codex.cmd", "codex.exe"),
    ),
    PlatformDefinition(
        backend_id="platform:claude-code",
        provider="claude",
        display_name="Claude Code",
        command_candidates=("claude", "claude.cmd", "claude.exe"),
    ),
    PlatformDefinition(
        backend_id="platform:google-antigravity",
        provider="antigravity",
        display_name="Google Antigravity",
        command_candidates=("antigravity", "antigravity.cmd", "antigravity.exe"),
    ),
)

DEFAULT_CLIPROXY_CANDIDATES = (
    "CLIProxyAPI",
    "CLIProxyAPI.exe",
    "cli-proxy-api",
    "cli-proxy-api.exe",
    "cliproxyapi",
    "cliproxyapi.exe",
)

CLIPROXY_DEFAULT_PORT = 8317


class PlatformIntegrationError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class CLIProxySidecarError(Exception):
    pass


class CLIProxySidecarManager:
    """Supervises the shared local CLIProxyAPI HTTP sidecar."""

    def __init__(
        self,
        platform_manager: "PlatformIntegrationManager",
        *,
        runtime_dir: Optional[os.PathLike | str] = None,
        port_start: int = CLIPROXY_DEFAULT_PORT,
        popen_factory: PopenFactory = subprocess.Popen,
        health_checker: Optional[HealthChecker] = None,
        port_available: Optional[PortChecker] = None,
        health_timeout: float = 5.0,
    ) -> None:
        self._platform_manager = platform_manager
        self._runtime_dir = Path(runtime_dir or Path(INSTALL_ROOT) / "data" / "cliproxy")
        self._port_start = port_start
        self._popen = popen_factory
        self._health_checker = health_checker or self._default_health_check
        self._port_available = port_available or self._is_port_available
        self._health_timeout = health_timeout
        self._lock = threading.Lock()
        self._process: Optional[subprocess.Popen] = None
        self._port: Optional[int] = None
        self._config_path: Optional[Path] = None
        self._start_time: Optional[float] = None
        self._last_error: Optional[str] = None

    @property
    def port(self) -> Optional[int]:
        return self._port

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def ensure_running(self) -> dict:
        with self._lock:
            if self.is_running():
                return self.status()

            detection = self._platform_manager.cliproxy_detection
            if not detection.detected or not detection.path:
                self._last_error = detection.reason or "CLIProxyAPI executable not found"
                raise CLIProxySidecarError(self._last_error)

            port = self._allocate_port()
            config_path = self._write_config(port)
            cmd = [detection.path, "-config", str(config_path)]

            try:
                proc = self._popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
            except Exception as exc:
                self._last_error = f"Failed to start CLIProxyAPI: {exc}"
                raise CLIProxySidecarError(self._last_error) from exc

            self._process = proc
            self._port = port
            self._config_path = config_path
            self._start_time = time.time()
            if not self._wait_until_healthy(port):
                self._last_error = "CLIProxyAPI sidecar did not become healthy"
                self._terminate_locked()
                raise CLIProxySidecarError(self._last_error)

            self._last_error = None
            return self.status()

    def stop(self) -> dict:
        with self._lock:
            self._terminate_locked()
            return self.status()

    def status(self) -> dict:
        running = self.is_running()
        return {
            "status": "running" if running else "stopped",
            "port": self._port if running else None,
            "config_path": str(self._config_path) if self._config_path else None,
            "start_time": self._start_time if running else None,
            "last_error": self._last_error,
            "executable_path": self._platform_manager.cliproxy_detection.path,
        }

    def _write_config(self, port: int) -> Path:
        self._runtime_dir.mkdir(parents=True, exist_ok=True)
        auth_dir = self._runtime_dir / "auth"
        auth_dir.mkdir(parents=True, exist_ok=True)
        config_path = self._runtime_dir / "config.yaml"
        auth_dir_value = str(auth_dir).replace("\\", "/")
        config = "\n".join(
            [
                f"port: {port}",
                'host: "127.0.0.1"',
                f'auth-dir: "{auth_dir_value}"',
                "api-keys: []",
                "remote-management:",
                "  secret-key: \"\"",
                "  allow-remote: false",
                "",
            ]
        )
        config_path.write_text(config, encoding="utf-8")
        return config_path

    def _allocate_port(self) -> int:
        port = self._port_start
        while not self._port_available(port):
            port += 1
        return port

    def _wait_until_healthy(self, port: int) -> bool:
        deadline = time.time() + self._health_timeout
        while True:
            if self._process is None or self._process.poll() is not None:
                return False
            if self._health_checker(port):
                return True
            if time.time() >= deadline:
                return False
            time.sleep(0.1)

    def _terminate_locked(self) -> None:
        proc = self._process
        self._process = None
        if proc is not None and proc.poll() is None:
            try:
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception:
                pass
        self._port = None
        self._start_time = None

    @staticmethod
    def _is_port_available(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            return sock.connect_ex(("127.0.0.1", port)) != 0

    @staticmethod
    def _default_health_check(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.25)
            return sock.connect_ex(("127.0.0.1", port)) == 0


class PlatformIntegrationManager:
    """In-memory platform catalog built from startup-time executable detection."""

    def __init__(
        self,
        config_manager,
        *,
        executable_resolver: Optional[ExecutableResolver] = None,
        platform_definitions: Iterable[PlatformDefinition] = DEFAULT_PLATFORM_DEFINITIONS,
        cliproxy_candidates: Iterable[str] = DEFAULT_CLIPROXY_CANDIDATES,
    ) -> None:
        self._config = config_manager
        self._resolver = executable_resolver or shutil.which
        self._definitions = tuple(platform_definitions)
        self._cliproxy_candidates = tuple(cliproxy_candidates)
        self._cliproxy = self._detect_executable(
            self._cliproxy_candidates,
            "CLIProxyAPI executable not found",
        )
        self._detections = {
            definition.backend_id: self._detect_platform(definition)
            for definition in self._definitions
        }
        self._runtime: Dict[str, dict] = {}
        self._runtime_lock = threading.Lock()

    @property
    def cliproxy_detection(self) -> ExecutableDetection:
        return self._cliproxy

    def definitions(self) -> list[PlatformDefinition]:
        return list(self._definitions)

    def catalog(self) -> list[dict]:
        platform_configs = self._config.get_platform_configs()
        return [
            self._catalog_entry(definition, platform_configs.get(definition.backend_id, {}))
            for definition in self._definitions
        ]

    def get(self, backend_id: str) -> Optional[dict]:
        for item in self.catalog():
            if item["backend_id"] == backend_id:
                return item
        return None

    def runtime_states(self) -> list[dict]:
        return [self.runtime_state(item["backend_id"]) for item in self.catalog()]

    def runtime_state(self, backend_id: str) -> dict:
        item = self.get(backend_id)
        if item is None:
            return {
                "backend_id": backend_id,
                "backend_type": "platform",
                "active": False,
                "status": "missing",
                "last_error": "Platform integration not defined",
            }
        with self._runtime_lock:
            runtime = dict(self._runtime.get(backend_id, {}))
        return {
            **item,
            "active": bool(runtime.get("active")),
            "sidecar_port": runtime.get("sidecar_port"),
            "start_time": runtime.get("start_time"),
            "last_error": runtime.get("last_error"),
            "status": runtime.get("status") or item["status"],
        }

    def active_instances(self) -> list[dict]:
        instances = []
        for state in self.runtime_states():
            if not state.get("active") or state.get("status") != "running":
                continue
            instances.append({
                "port": state.get("sidecar_port"),
                "status": "running",
                "model": state.get("display_name") or state.get("name"),
                "model_path": None,
                "backend_id": state["backend_id"],
                "backend_type": "platform",
                "provider": state.get("provider"),
                "start_time": state.get("start_time"),
                "config": {
                    "backend_id": state["backend_id"],
                    "backend_type": "platform",
                    "provider": state.get("provider"),
                    "proxy_eligible": state.get("proxy_eligible"),
                    "max_parallel_requests": state.get("max_parallel_requests"),
                },
            })
        return instances

    def start_backend(
        self, backend_id: str, sidecar: CLIProxySidecarManager
    ) -> dict:
        item = self.get(backend_id)
        if item is None:
            raise PlatformIntegrationError(404, "Platform integration not defined")
        if not item.get("detected"):
            error = item.get("reason") or "Platform integration executable not found"
            self._set_runtime_error(backend_id, "missing", error)
            raise PlatformIntegrationError(400, error)
        if item.get("status") == "not_ready":
            error = item.get("reason") or "Platform integration is not ready"
            self._set_runtime_error(backend_id, "not_ready", error)
            raise PlatformIntegrationError(409, error)
        try:
            sidecar_status = sidecar.ensure_running()
        except CLIProxySidecarError as exc:
            error = str(exc)
            self._set_runtime_error(backend_id, "not_ready", error)
            raise PlatformIntegrationError(502, error) from exc
        runtime = {
            "active": True,
            "status": "running",
            "sidecar_port": sidecar_status.get("port"),
            "start_time": time.time(),
            "last_error": None,
        }
        with self._runtime_lock:
            self._runtime[backend_id] = runtime
        return self.runtime_state(backend_id)

    def stop_backend(
        self, backend_id: str, sidecar: CLIProxySidecarManager
    ) -> dict:
        if self.get(backend_id) is None:
            raise PlatformIntegrationError(404, "Platform integration not defined")
        with self._runtime_lock:
            self._runtime[backend_id] = {
                **self._runtime.get(backend_id, {}),
                "active": False,
                "status": "stopped",
                "sidecar_port": None,
            }
            any_active = any(
                state.get("active") for state in self._runtime.values()
            )
        if not any_active:
            sidecar.stop()
        return self.runtime_state(backend_id)

    def detection_for(self, backend_id: str) -> ExecutableDetection:
        return self._detections.get(
            backend_id,
            ExecutableDetection(False, reason="Platform integration not defined"),
        )

    def _set_runtime_error(self, backend_id: str, status: str, error: str) -> None:
        with self._runtime_lock:
            self._runtime[backend_id] = {
                **self._runtime.get(backend_id, {}),
                "active": False,
                "status": status,
                "sidecar_port": None,
                "last_error": error,
            }

    def _catalog_entry(self, definition: PlatformDefinition, config: Dict) -> dict:
        detection = self._detections[definition.backend_id]
        status, reason = self._status_and_reason(definition, detection)
        return {
            "backend_id": definition.backend_id,
            "backend_type": "platform",
            "provider": definition.provider,
            "name": definition.display_name,
            "display_name": definition.display_name,
            "detected": detection.detected,
            "status": status,
            "reason": reason,
            "executable_path": detection.path,
            "executable_command": detection.command,
            "cliproxy_detected": self._cliproxy.detected,
            "cliproxy_executable_path": self._cliproxy.path,
            "proxy_eligible": bool(config.get("proxy_eligible", False)),
            "max_parallel_requests": int(config.get("max_parallel_requests", 1) or 1),
        }

    def _status_and_reason(
        self, definition: PlatformDefinition, detection: ExecutableDetection
    ) -> tuple[str, Optional[str]]:
        if not detection.detected:
            return "missing", detection.reason or (
                f"{definition.display_name} executable not found"
            )
        if not self._cliproxy.detected:
            return "not_ready", self._cliproxy.reason
        return "detected", None

    def _detect_platform(self, definition: PlatformDefinition) -> ExecutableDetection:
        return self._detect_executable(
            definition.command_candidates,
            f"{definition.display_name} executable not found",
        )

    def _detect_executable(
        self, candidates: Iterable[str], missing_reason: str
    ) -> ExecutableDetection:
        for command in candidates:
            path = self._resolver(command)
            if path:
                return ExecutableDetection(True, command=command, path=path)
        return ExecutableDetection(False, reason=missing_reason)
