"""Platform integration catalog and startup-only executable detection."""

from __future__ import annotations

import glob
import hashlib
import os
import platform
import re
import signal
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from paths import INSTALL_ROOT

_IS_WINDOWS = platform.system() == "Windows"


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
        command_candidates=("agy", "antigravity", "antigravity.cmd", "antigravity.exe"),
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

# owned_by retornado pelo CLIProxyAPI em GET /v1/models por integração.
PLATFORM_MODEL_LISTING_MARKER = "-custom"
PLATFORM_MODEL_LISTING_SUFFIX = ".gguf"

# Substrings que o Cursor BYOK rejeita mesmo com sufixo -custom (validação server-side).
CURSOR_BLOCKED_MODEL_TOKENS = (
    "gemini",
    "claude",
    "gpt",
    "codex",
    "openai",
    "sonnet",
    "opus",
    "o1",
    "o3",
)

# listing_id exposto na API -> id real do sidecar
_PLATFORM_LISTING_REGISTRY: Dict[str, str] = {}


def clear_platform_listing_registry() -> None:
    _PLATFORM_LISTING_REGISTRY.clear()


def register_platform_listing(listing_id: str, bare_id: str) -> None:
    if listing_id and bare_id:
        _PLATFORM_LISTING_REGISTRY[listing_id] = bare_id


def lookup_platform_bare_id(listing_id: str) -> Optional[str]:
    return _PLATFORM_LISTING_REGISTRY.get(listing_id)


def lookup_platform_listing_id(bare_id: str) -> Optional[str]:
    matches = [
        listing_id
        for listing_id, root in _PLATFORM_LISTING_REGISTRY.items()
        if root == bare_id
    ]
    if not matches:
        return None
    virtual = [listing_id for listing_id in matches if listing_id != bare_id]
    return virtual[0] if virtual else matches[0]


def platform_listing_registry_populated() -> bool:
    return bool(_PLATFORM_LISTING_REGISTRY)


def _provider_slug(provider: str) -> str:
    slug = re.sub(r"[^a-z0-9]", "", (provider or "cloud").lower())
    return slug[:12] or "cloud"


def _needs_cursor_safe_name(bare: str) -> bool:
    lower = bare.lower()
    return any(token in lower for token in CURSOR_BLOCKED_MODEL_TOKENS)


def _cursor_safe_slug(bare: str, provider: str) -> str:
    """Nome estilo Qwen local — sem tokens bloqueados pelo Cursor."""
    prov = _provider_slug(provider)
    compact = re.sub(r"[^a-z0-9]", "", bare.lower())
    for token in CURSOR_BLOCKED_MODEL_TOKENS:
        compact = compact.replace(token, "")
    if len(compact) >= 4:
        return f"{prov}-{compact[:20]}"
    digest = hashlib.sha256(bare.encode()).hexdigest()[:10]
    return f"{prov}-{digest}"


def _bare_platform_model_id(model_id: str) -> str:
    """ID real do sidecar, sem sufixos virtuais de listagem."""
    bare = str(model_id or "").strip()
    if bare.endswith(PLATFORM_MODEL_LISTING_SUFFIX):
        bare = bare[: -len(PLATFORM_MODEL_LISTING_SUFFIX)]
    if bare.endswith(PLATFORM_MODEL_LISTING_MARKER):
        bare = bare[: -len(PLATFORM_MODEL_LISTING_MARKER)]
    return bare


def platform_model_listing_id(model_id: str, provider: str = "") -> str:
    """ID exposto em /v1/models — .gguf local; slug opaco se o Cursor bloquear o nome."""
    bare = _bare_platform_model_id(model_id)
    if not bare:
        return model_id
    if _needs_cursor_safe_name(bare):
        return f"{_cursor_safe_slug(bare, provider)}{PLATFORM_MODEL_LISTING_SUFFIX}"
    return f"{bare}{PLATFORM_MODEL_LISTING_MARKER}{PLATFORM_MODEL_LISTING_SUFFIX}"


def register_platform_model_listings(bare_id: str, provider: str = "") -> str:
    """Registra variantes de listagem (atual + legado) e retorna o ID principal."""
    bare = _bare_platform_model_id(bare_id)
    if not bare:
        return bare_id
    primary = platform_model_listing_id(bare, provider)
    register_platform_listing(primary, bare)
    legacy = f"{bare}{PLATFORM_MODEL_LISTING_MARKER}{PLATFORM_MODEL_LISTING_SUFFIX}"
    register_platform_listing(legacy, bare)
    prov_slug = _provider_slug(provider)
    if prov_slug == "codex":
        opaque = primary[: -len(PLATFORM_MODEL_LISTING_SUFFIX)]
        if opaque.startswith("codex-"):
            openai_listing = (
                f"openai-{opaque[len('codex-'):]}{PLATFORM_MODEL_LISTING_SUFFIX}"
            )
            register_platform_listing(openai_listing, bare)
    return primary


def is_platform_listing_id(
    model_name: str, local_model_ids: Optional[set[str]] = None
) -> bool:
    """True quando o ID é virtual de plataforma (não um .gguf local real)."""
    if not model_name:
        return False
    if local_model_ids and model_name in local_model_ids:
        return False
    if lookup_platform_bare_id(model_name):
        return True
    if model_name.endswith(f"{PLATFORM_MODEL_LISTING_MARKER}{PLATFORM_MODEL_LISTING_SUFFIX}"):
        return True
    if model_name.endswith(PLATFORM_MODEL_LISTING_SUFFIX):
        bare = model_name[: -len(PLATFORM_MODEL_LISTING_SUFFIX)]
        return bool(bare) and bare not in (local_model_ids or set())
    return False


def platform_model_listing_entry(model: Dict, provider: str = "") -> Dict:
    """Enriquece entrada de plataforma para espelhar /v1/models do llama-server."""
    root_id = str(model.get("id") or "")
    prov = provider or str(model.get("owned_by") or "")
    listing_id = register_platform_model_listings(root_id, prov)
    source_meta = model.get("meta") if isinstance(model.get("meta"), dict) else {}
    # Mesmo shape que llama-server — clientes como Cursor comparam estas chaves.
    meta = {
        "vocab_type": int(source_meta.get("vocab_type", 0)),
        "n_vocab": int(source_meta.get("n_vocab", 0)),
        "n_ctx": int(source_meta.get("n_ctx") or source_meta.get("context_length") or 1_048_576),
        "n_ctx_train": int(
            source_meta.get("n_ctx_train")
            or source_meta.get("n_ctx")
            or source_meta.get("context_length")
            or 1_048_576
        ),
        "n_embd": int(source_meta.get("n_embd", 0)),
        "n_params": int(source_meta.get("n_params", 0)),
        "size": int(source_meta.get("size", 0)),
        "root_model": root_id,
    }
    entry = {
        **model,
        "id": listing_id,
        "object": model.get("object") or "model",
        "owned_by": "llamacpp",
        "aliases": model.get("aliases") if isinstance(model.get("aliases"), list) else [],
        "tags": model.get("tags") if isinstance(model.get("tags"), list) else [],
        "created": int(model.get("created") or time.time()),
        "meta": meta,
    }
    entry.pop("root_model", None)
    return entry


def platform_client_facing_model(
    requested: str,
    local_model_ids: Optional[set[str]] = None,
    aliases: Optional[Dict[str, str]] = None,
    provider: str = "",
) -> str:
    """ID do campo `model` nas respostas /v1/* — espelha o que o cliente enviou."""
    original = str(requested or "").strip()
    if not original:
        return original
    if aliases and original in aliases:
        return original
    if is_platform_listing_id(original, local_model_ids):
        return original
    listing = lookup_platform_listing_id(original)
    if listing:
        return listing
    return platform_model_listing_id(original, provider)


def platform_listing_provider_prefix(model_name: str) -> Optional[str]:
    """Prefixo do slug opaco (codex-, antigravity-, openai- legado)."""
    if not model_name.endswith(PLATFORM_MODEL_LISTING_SUFFIX):
        return None
    slug = model_name[: -len(PLATFORM_MODEL_LISTING_SUFFIX)]
    if "-" not in slug:
        return None
    prefix = slug.split("-", 1)[0]
    if prefix == "openai":
        return "codex"
    return prefix


def platform_provider_for_listing(model_name: str) -> Optional[str]:
    """Provider da integração para um ID opaco de listagem."""
    return platform_listing_provider_prefix(model_name)


def resolve_platform_listing_model(
    model_name: str, local_model_ids: Optional[set[str]] = None
) -> str:
    """Converte ID virtual de plataforma de volta ao ID do sidecar."""
    if not model_name:
        return model_name
    if local_model_ids and model_name in local_model_ids:
        return model_name
    mapped = lookup_platform_bare_id(model_name)
    if mapped:
        return mapped
    custom_suffix = f"{PLATFORM_MODEL_LISTING_MARKER}{PLATFORM_MODEL_LISTING_SUFFIX}"
    if model_name.endswith(custom_suffix):
        return model_name[: -len(custom_suffix)]
    if model_name.endswith(PLATFORM_MODEL_LISTING_SUFFIX):
        slug = model_name[: -len(PLATFORM_MODEL_LISTING_SUFFIX)]
        if slug.startswith("openai-"):
            alt = f"codex-{slug[len('openai-'):]}{PLATFORM_MODEL_LISTING_SUFFIX}"
            mapped = lookup_platform_bare_id(alt)
            if mapped:
                return mapped
    if is_platform_listing_id(model_name, local_model_ids):
        return model_name[: -len(PLATFORM_MODEL_LISTING_SUFFIX)]
    return model_name


def should_skip_platform_model_listing(
    model_id: str, local_model_ids: Optional[set[str]] = None
) -> bool:
    """Evita listar no sidecar um modelo já exposto por instância local."""
    bare = _bare_platform_model_id(model_id)
    if not bare:
        return True
    local_ids = local_model_ids or set()
    return (
        model_id in local_ids
        or bare in local_ids
        or f"{bare}{PLATFORM_MODEL_LISTING_SUFFIX}" in local_ids
    )


PLATFORM_MODEL_OWNED_BY: Dict[str, tuple[str, ...]] = {
    "codex": ("openai",),
    "claude": ("claude",),
    "antigravity": ("antigravity",),
}


def filter_models_for_provider(
    models: List[Dict], provider: str
) -> List[Dict]:
    """Mantém apenas modelos do provedor da plataforma (sidecar agrega todos)."""
    owners = PLATFORM_MODEL_OWNED_BY.get((provider or "").strip().lower())
    if not owners:
        return list(models)
    allowed = {owner.lower() for owner in owners}
    return [
        model
        for model in models
        if str(model.get("owned_by") or "").lower() in allowed
    ]


def _is_executable(path: str) -> bool:
    if not path or not os.path.isfile(path):
        return False
    if _IS_WINDOWS:
        lower = path.lower()
        if lower.endswith((".exe", ".bat", ".cmd")):
            return True
    return os.access(path, os.X_OK)


def _user_home_dirs() -> List[str]:
    homes: List[str] = []
    home = os.path.expanduser("~")
    if home and home not in homes:
        homes.append(home)
    if not _IS_WINDOWS:
        for pattern in ("/home/*", "/root"):
            for path in glob.glob(pattern):
                if os.path.isdir(path) and path not in homes:
                    homes.append(path)
    return homes


def _search_bin_directories() -> List[str]:
    dirs: List[str] = []
    for home in _user_home_dirs():
        for rel in (".local/bin", "bin"):
            candidate = os.path.join(home, rel)
            if os.path.isdir(candidate) and candidate not in dirs:
                dirs.append(candidate)
    if not _IS_WINDOWS:
        for path in ("/usr/local/bin", "/usr/bin", "/opt/homebrew/bin"):
            if os.path.isdir(path) and path not in dirs:
                dirs.append(path)
    return dirs


def _tool_specific_paths(command: str) -> List[str]:
    paths: List[str] = []
    for home in _user_home_dirs():
        if command in ("codex", "codex.exe"):
            paths.append(
                os.path.join(
                    home, ".codex", "packages", "standalone", "current", "bin", "codex"
                )
            )
        if command in ("agy", "antigravity"):
            paths.append(os.path.join(home, ".local", "bin", command))
    return paths


def default_executable_resolver(command: str) -> Optional[str]:
    """Resolve a platform executable from PATH and known install locations."""
    found = shutil.which(command)
    if found and _is_executable(found):
        return os.path.normpath(found)

    seen: set[str] = set()

    def add(path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        norm = os.path.normpath(path)
        if norm in seen:
            return None
        seen.add(norm)
        if _is_executable(norm):
            return norm
        return None

    for directory in _search_bin_directories():
        resolved = add(os.path.join(directory, command))
        if resolved:
            return resolved

    for path in _tool_specific_paths(command):
        resolved = add(path)
        if resolved:
            return resolved

    return None


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
        log_manager=None,
        runtime_dir: Optional[os.PathLike | str] = None,
        port_start: int = CLIPROXY_DEFAULT_PORT,
        popen_factory: PopenFactory = subprocess.Popen,
        health_checker: Optional[HealthChecker] = None,
        port_available: Optional[PortChecker] = None,
        health_timeout: float = 5.0,
    ) -> None:
        self._platform_manager = platform_manager
        self._log_manager = log_manager
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

            if self._log_manager is not None:
                self._log_manager.start_streaming(port, proc, cmd=cmd)

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
        self._resolver = executable_resolver or default_executable_resolver
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
                    "auto_start": state.get("auto_start"),
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
            "auto_start": bool(config.get("auto_start", False)),
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
