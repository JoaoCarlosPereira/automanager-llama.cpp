"""CLIProxyAPI provider authentication helpers and login session management."""

from __future__ import annotations

import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from paths import INSTALL_ROOT

PopenFactory = Callable[..., subprocess.Popen]

_PROVIDER_PREFIXES = {
    "codex": ("codex-", "openai-"),
    "claude": ("claude-",),
    "antigravity": ("antigravity-", "agy-"),
}

_LOGIN_COMMANDS = {
    "codex": {
        "device": ("-codex-device-login",),
        "oauth": ("-codex-login", "-no-browser"),
    },
    "claude": {
        "oauth": ("-claude-login", "-no-browser"),
    },
    "antigravity": {
        "oauth": ("-antigravity-login", "-no-browser"),
    },
}

_DEFAULT_METHOD = {
    "codex": "oauth",
    "claude": "oauth",
    "antigravity": "oauth",
}


@dataclass(frozen=True)
class ProviderAuthStatus:
    provider: str
    authenticated: bool
    accounts: tuple[str, ...]
    default_method: str
    available_methods: tuple[str, ...]


def auth_dir_for(runtime_dir: Optional[Path] = None) -> Path:
    base = runtime_dir or Path(INSTALL_ROOT) / "data" / "cliproxy"
    return base / "auth"


def config_path_for(runtime_dir: Optional[Path] = None) -> Path:
    base = runtime_dir or Path(INSTALL_ROOT) / "data" / "cliproxy"
    return base / "config.yaml"


def ensure_runtime_config(runtime_dir: Optional[Path] = None, port: int = 8317) -> Path:
    base = runtime_dir or Path(INSTALL_ROOT) / "data" / "cliproxy"
    base.mkdir(parents=True, exist_ok=True)
    auth_dir = base / "auth"
    auth_dir.mkdir(parents=True, exist_ok=True)
    config_path = base / "config.yaml"
    if not config_path.is_file():
        auth_dir_value = str(auth_dir).replace("\\", "/")
        config = "\n".join(
            [
                f"port: {port}",
                'host: "127.0.0.1"',
                f'auth-dir: "{auth_dir_value}"',
                "api-keys: []",
                "remote-management:",
                '  secret-key: ""',
                "  allow-remote: false",
                "",
            ]
        )
        config_path.write_text(config, encoding="utf-8")
    return config_path


def list_provider_auth_status(
    runtime_dir: Optional[Path] = None,
) -> Dict[str, dict]:
    directory = auth_dir_for(runtime_dir)
    statuses: Dict[str, dict] = {}
    for provider, prefixes in _PROVIDER_PREFIXES.items():
        status = _provider_status(provider, directory, prefixes)
        statuses[provider] = {
            "provider": status.provider,
            "authenticated": status.authenticated,
            "accounts": list(status.accounts),
            "default_method": status.default_method,
            "available_methods": list(status.available_methods),
        }
    return statuses


def _provider_status(
    provider: str, directory: Path, prefixes: tuple[str, ...]
) -> ProviderAuthStatus:
    accounts: List[str] = []
    if directory.is_dir():
        for path in sorted(directory.glob("*.json")):
            name = path.name
            if any(name.startswith(prefix) for prefix in prefixes):
                accounts.append(name)
    methods = tuple(_LOGIN_COMMANDS.get(provider, {}).keys())
    return ProviderAuthStatus(
        provider=provider,
        authenticated=bool(accounts),
        accounts=tuple(accounts),
        default_method=_DEFAULT_METHOD.get(provider, "oauth"),
        available_methods=methods,
    )


def parse_login_output(text: str) -> dict:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    payload: dict = {
        "auth_url": None,
        "device_code": None,
        "instructions": [],
        "status_message": None,
        "needs_callback": False,
        "callback_hint": None,
    }
    lowered = text.lower()
    if "paste" in lowered and "callback" in lowered:
        payload["needs_callback"] = True
    for index, line in enumerate(lines):
        lower = line.lower()
        if lower.startswith("codex device url:"):
            payload["auth_url"] = line.split(":", 1)[1].strip()
        elif lower.startswith("codex device code:"):
            payload["device_code"] = line.split(":", 1)[1].strip()
        elif lower.startswith("visit the following url to continue authentication:"):
            if index + 1 < len(lines):
                payload["auth_url"] = lines[index + 1]
        elif lower.startswith("attempting to open url in browser:"):
            payload["auth_url"] = line.split(":", 1)[1].strip()
        elif "paste" in lower and "callback" in lower:
            payload["callback_hint"] = line
            payload["needs_callback"] = True
        elif "waiting for" in lower and "callback" in lower:
            payload["status_message"] = line
            payload["needs_callback"] = True
        elif "waiting for" in lower and "authentication" in lower:
            payload["status_message"] = line
        elif line.startswith("===="):
            block: List[str] = []
            cursor = index + 1
            while cursor < len(lines) and not lines[cursor].startswith("===="):
                block.append(lines[cursor])
                cursor += 1
            if block:
                payload["instructions"].extend(block)
    return payload


class CLIProxyAuthManager:
    """Runs CLIProxyAPI login flows and tracks in-progress sessions."""

    def __init__(
        self,
        platform_manager,
        *,
        runtime_dir: Optional[Path] = None,
        popen_factory: PopenFactory = subprocess.Popen,
        session_ttl: float = 900.0,
    ) -> None:
        self._platform_manager = platform_manager
        self._runtime_dir = Path(runtime_dir or Path(INSTALL_ROOT) / "data" / "cliproxy")
        self._popen = popen_factory
        self._session_ttl = session_ttl
        self._lock = threading.Lock()
        self._sessions: Dict[str, dict] = {}

    def list_status(self) -> Dict[str, dict]:
        return list_provider_auth_status(self._runtime_dir)

    def start_login(self, provider: str, method: Optional[str] = None) -> dict:
        provider = provider.strip().lower()
        if provider not in _PROVIDER_PREFIXES:
            raise ValueError(f"Unsupported provider: {provider}")

        detection = self._platform_manager.cliproxy_detection
        if not detection.detected or not detection.path:
            raise RuntimeError(detection.reason or "CLIProxyAPI executable not found")

        chosen = method or _DEFAULT_METHOD.get(provider, "oauth")
        commands = _LOGIN_COMMANDS.get(provider, {})
        if chosen not in commands:
            raise ValueError(f"Unsupported login method for {provider}: {chosen}")

        config_path = ensure_runtime_config(self._runtime_dir)
        before_accounts = set(
            list_provider_auth_status(self._runtime_dir).get(provider, {}).get("accounts", [])
        )
        session_id = uuid.uuid4().hex
        cmd = [detection.path, *commands[chosen], "-config", str(config_path)]

        proc = self._popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        session = {
            "id": session_id,
            "provider": provider,
            "method": chosen,
            "status": "pending",
            "created_at": time.time(),
            "updated_at": time.time(),
            "output": "",
            "parsed": {},
            "return_code": None,
            "error": None,
            "process": proc,
            "before_accounts": before_accounts,
        }
        with self._lock:
            self._cleanup_sessions_locked()
            self._sessions[session_id] = session

        threading.Thread(
            target=self._drain_session,
            args=(session_id,),
            daemon=True,
        ).start()
        return self.public_view(session_id)

    def get_session(self, session_id: str) -> Optional[dict]:
        with self._lock:
            self._cleanup_sessions_locked()
            session = self._sessions.get(session_id)
            if session is None:
                return None
            return self._public_view_locked(session)

    def cancel_session(self, session_id: str) -> Optional[dict]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            proc = session.get("process")
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            session["status"] = "cancelled"
            session["updated_at"] = time.time()
            return self._public_view_locked(session)

    def submit_callback(self, session_id: str, callback_url: str) -> dict:
        callback_url = (callback_url or "").strip()
        if not callback_url:
            raise ValueError("Callback URL is required")

        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(session_id)
            if session.get("status") in {"completed", "cancelled", "failed"}:
                raise RuntimeError("Authentication session is no longer active")
            proc = session.get("process")
            if proc is None or proc.stdin is None or proc.poll() is not None:
                raise RuntimeError("Authentication process is not waiting for callback")
            session["callback_submitted"] = True
            session["updated_at"] = time.time()

        try:
            proc.stdin.write(callback_url + "\n")
            proc.stdin.flush()
        except Exception as exc:
            raise RuntimeError(f"Failed to submit callback URL: {exc}") from exc

        return self.public_view(session_id)

    def public_view(self, session_id: str) -> dict:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(session_id)
            return self._public_view_locked(session)

    def _public_view_locked(self, session: dict) -> dict:
        parsed = dict(session.get("parsed") or {})
        return {
            "id": session["id"],
            "provider": session["provider"],
            "method": session["method"],
            "status": session["status"],
            "auth_url": parsed.get("auth_url"),
            "device_code": parsed.get("device_code"),
            "needs_callback": bool(parsed.get("needs_callback")),
            "callback_hint": parsed.get("callback_hint"),
            "callback_submitted": bool(session.get("callback_submitted")),
            "instructions": list(parsed.get("instructions") or []),
            "status_message": parsed.get("status_message"),
            "output_tail": (session.get("output") or "")[-4000:],
            "error": session.get("error"),
            "accounts": list_provider_auth_status(self._runtime_dir)
            .get(session["provider"], {})
            .get("accounts", []),
        }

    def _drain_session(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            proc = session.get("process")
        if proc is None or proc.stdout is None:
            return

        try:
            for chunk in iter(proc.stdout.readline, ""):
                if not chunk:
                    break
                with self._lock:
                    current = self._sessions.get(session_id)
                    if current is None:
                        break
                    current["output"] = (current.get("output") or "") + chunk
                    current["parsed"] = parse_login_output(current["output"])
                    current["updated_at"] = time.time()
                    if current["parsed"].get("needs_callback"):
                        current["status"] = "waiting_callback"
                    elif current["status"] == "pending" and (
                        current["parsed"].get("auth_url")
                        or current["parsed"].get("device_code")
                    ):
                        current["status"] = "waiting"
            return_code = proc.wait()
        except Exception as exc:
            with self._lock:
                current = self._sessions.get(session_id)
                if current is not None:
                    current["status"] = "failed"
                    current["error"] = str(exc)
                    current["updated_at"] = time.time()
            return

        with self._lock:
            current = self._sessions.get(session_id)
            if current is None:
                return
            current["return_code"] = return_code
            current["updated_at"] = time.time()
            after_accounts = set(
                list_provider_auth_status(self._runtime_dir)
                .get(current["provider"], {})
                .get("accounts", [])
            )
            if after_accounts - set(current.get("before_accounts") or set()):
                current["status"] = "completed"
            elif current["status"] == "cancelled":
                pass
            elif return_code == 0:
                current["status"] = "completed"
            else:
                current["status"] = "failed"
                current["error"] = current["error"] or "Authentication flow failed"

    def _cleanup_sessions_locked(self) -> None:
        cutoff = time.time() - self._session_ttl
        stale = [
            session_id
            for session_id, session in self._sessions.items()
            if session.get("updated_at", 0) < cutoff
            and session.get("status") in {"completed", "failed", "cancelled"}
        ]
        for session_id in stale:
            self._sessions.pop(session_id, None)
