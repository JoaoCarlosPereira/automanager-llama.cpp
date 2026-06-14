"""Config, token, and auth managers for automanager."""

import os
import json
import logging
import hashlib
import secrets
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from fastapi import Request
from fastapi.security import HTTPAuthorizationCredentials
import bcrypt

def _hash_password_bcrypt(password: str) -> str:
    """Hash password using bcrypt."""
    return bcrypt.hashpw(
        password.encode("utf-8"), bcrypt.gensalt()
    ).decode("utf-8")

def _verify_password_bcrypt(password: str, hashed: str) -> bool:
    """Verify password against bcrypt hash."""
    return bcrypt.checkpw(
        password.encode("utf-8"), hashed.encode("utf-8")
    )

from paths import CONFIG_PATH
from schemas import (
    DEFAULT_MTP_DRAFT_TOKENS,
    DEFAULT_MTP_ENABLED,
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_PARALLEL_SLOTS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CACHE_TYPE,
)

DEFAULT_THINKING_ENABLED = True

SESSION_IDLE_SECONDS = 86400  # 24h sem atividade

logger = logging.getLogger("automanager")


def normalize_model_path(model_path: str) -> str:
    """Canonical model path key for model_configs (absolute path + forward slashes)."""
    if not model_path:
        return ""
    path = model_path.replace("\\", "/")
    # Map Windows drive paths when running on POSIX (e.g. Z:/media/... -> /media/...)
    if os.name != "nt" and len(path) >= 3 and path[1] == ":" and path[0].isalpha():
        path = "/" + path[3:].lstrip("/")
    return os.path.abspath(path).replace("\\", "/")


def _migrate_config(config: dict) -> tuple[dict, bool]:
    """Normalize legacy keys and drop invalid auto-start entries."""
    if not config:
        return config, False
    changed = False

    model_configs = config.get("model_configs", {})
    if isinstance(model_configs, dict):
        migrated_configs: Dict[str, dict] = {}
        for key, value in model_configs.items():
            norm = normalize_model_path(key)
            if norm in migrated_configs:
                migrated_configs[norm] = {**migrated_configs[norm], **value}
            else:
                migrated_configs[norm] = value
            if norm != key:
                changed = True
        config["model_configs"] = migrated_configs

    defaults = config.get("default_models")
    if not isinstance(defaults, list):
        legacy = config.get("default_model")
        defaults = [legacy] if legacy else []
        changed = True

    normalized_defaults: list[str] = []
    for path in defaults:
        if not path:
            continue
        norm = normalize_model_path(path)
        if norm not in normalized_defaults:
            normalized_defaults.append(norm)

    valid_defaults: list[str] = []
    for path in defaults:
        if not path:
            continue
        norm = normalize_model_path(path)
        raw = path.replace("\\", "/")
        windows_legacy = (
            os.name != "nt"
            and len(raw) >= 3
            and raw[1] == ":"
            and raw[0].isalpha()
        )
        if windows_legacy and not os.path.exists(norm):
            changed = True
            continue
        if norm not in valid_defaults:
            valid_defaults.append(norm)
    if config.get("default_models") != valid_defaults:
        changed = True
    config["default_models"] = valid_defaults
    if "default_model" in config:
        config.pop("default_model", None)
        changed = True

    return config, changed


def lookup_model_config(model_configs: dict, model_path: str) -> dict:
    """Return saved settings for *model_path*, tolerating legacy key variants."""
    if not model_path:
        return {}
    norm = normalize_model_path(model_path)
    if norm in model_configs:
        return model_configs[norm]
    for key, cfg in model_configs.items():
        if normalize_model_path(key) == norm:
            return cfg
    return {}


class ConfigManager:
    """Thread-safe JSON config manager with atomic writes."""

    def __init__(self, config_path: str = CONFIG_PATH):
        self.config_path = config_path
        self._lock = threading.Lock()

    def load(self) -> dict:
        with self._lock:
            if os.path.exists(self.config_path):
                try:
                    with open(self.config_path, "r") as f:
                        config = json.load(f)
                except (json.JSONDecodeError, OSError):
                    return {}
            else:
                return {}
        config, changed = _migrate_config(config)
        if changed:
            self.save(config)
        return config

    def get_config(self) -> dict:
        return self.load()

    def save(self, data: dict) -> None:
        with self._lock:
            tmp_path = self.config_path + ".tmp"
            try:
                with open(tmp_path, "w") as f:
                    json.dump(data, f, indent=2)
                os.replace(tmp_path, self.config_path)
            except Exception as e:
                logger.error(f"Config save error: {e}")
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    def get_model_settings(self, model_path: str) -> dict:
        config = self.load()
        return lookup_model_config(config.get("model_configs", {}), model_path)

    def update_model_settings(self, model_path: str, settings: dict) -> None:
        config = self.load()
        if "model_configs" not in config:
            config["model_configs"] = {}
        model_configs = config["model_configs"]
        norm = normalize_model_path(model_path)
        prev = {}
        legacy_key = None
        if norm in model_configs:
            prev = model_configs[norm]
        else:
            for key in list(model_configs.keys()):
                if normalize_model_path(key) == norm:
                    prev = model_configs[key]
                    legacy_key = key
                    break
        merged = {**prev, **settings}
        entry = {
            "context_size": merged.get("context_size", DEFAULT_CONTEXT_SIZE),
            "parallel_slots": merged.get("parallel_slots", DEFAULT_PARALLEL_SLOTS),
            "batch_size": merged.get("batch_size", DEFAULT_BATCH_SIZE),
            "ubatch_size": merged.get("ubatch_size", 512),
            "cache_type_k": merged.get("cache_type_k", DEFAULT_CACHE_TYPE),
            "cache_type_v": merged.get("cache_type_v", DEFAULT_CACHE_TYPE),
            "numa_enabled": merged.get("numa_enabled", False),
            "threads": merged.get("threads", 0),
            "threads_batch": merged.get("threads_batch", 0),
            "mmproj_path": merged.get("mmproj_path"),
            "gpu_weights": merged.get("gpu_weights"),
            "split_mode": merged.get("split_mode", "layer"),
            "auto_balance": merged.get("auto_balance", False),
            "auto_balance_profile": merged.get("auto_balance_profile", False),
            "hardware_incapable": merged.get("hardware_incapable", False),
            "thinking_enabled": merged.get(
                "thinking_enabled", DEFAULT_THINKING_ENABLED
            ),
            "mtp_enabled": merged.get("mtp_enabled", DEFAULT_MTP_ENABLED),
            "mtp_draft_tokens": merged.get(
                "mtp_draft_tokens", DEFAULT_MTP_DRAFT_TOKENS
            ),
            "pinned_fields": merged.get("pinned_fields") or {},
            "last_started": datetime.now(timezone.utc).isoformat(),
        }
        if "hardware_incapable_message" in merged:
            entry["hardware_incapable_message"] = merged["hardware_incapable_message"]
        elif prev.get("hardware_incapable_message") and not entry["hardware_incapable"]:
            entry["hardware_incapable_message"] = None
        if legacy_key and legacy_key != norm:
            del model_configs[legacy_key]
        model_configs[norm] = entry
        self.save(config)

    def set_default_model(self, path: Optional[str], add: bool = True) -> None:
        config = self.load()
        defaults = config.get("default_models", [])
        if not isinstance(defaults, list):
            # Migrate legacy single default_model if it exists
            legacy = config.get("default_model")
            defaults = [legacy] if legacy else []
        
        if path:
            norm = normalize_model_path(path)
            if add:
                if norm not in defaults:
                    defaults.append(norm)
            else:
                if norm in defaults:
                    defaults.remove(norm)
        
        config["default_models"] = defaults
        # Clear legacy field for cleanliness
        config.pop("default_model", None)
        self.save(config)

    def get_default_models(self) -> list[str]:
        config = self.config.load() if hasattr(self, "config") else self.load()
        defaults = config.get("default_models")
        if isinstance(defaults, list):
            return [normalize_model_path(p) for p in defaults if p]
        legacy = config.get("default_model")
        return [normalize_model_path(legacy)] if legacy else []


class TokenManager:
    """Manages global API token in sk-... format."""

    PREFIX = "sk-"
    TOKEN_LENGTH = 32

    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager

    def generate(self) -> str:
        return f"{self.PREFIX}{secrets.token_hex(24)}"

    def validate(self, key: str) -> bool:
        if not isinstance(key, str):
            return False
        return key.startswith(self.PREFIX) and len(key) >= len(self.PREFIX) + 32

    def get_or_create(self) -> str:
        config = self.config.load()
        if "api_token" not in config or not self.validate(config["api_token"]):
            config["api_token"] = self.generate()
            self.config.save(config)
        return config["api_token"]

    def renew(self) -> str:
        config = self.config.load()
        config["api_token"] = self.generate()
        self.config.save(config)
        return config["api_token"]


class AuthManager:
    """Handles UI login (form-based sessions) and API key auth."""

    def __init__(self, config_manager: ConfigManager, token_manager: TokenManager):
        self.config = config_manager
        self.token_mgr = token_manager
        self._sessions: Dict[str, datetime] = {}
        self._lock = threading.Lock()
        self._init_admin_password()

    def _init_admin_password(self) -> None:
        """Initialize admin password hash if not present."""
        config = self.config.load()
        if "admin_password_hash" not in config:
            # Default password: "admin" — force user to change on first login
            config["admin_password_hash"] = _hash_password_bcrypt("admin")
            config["force_password_change"] = True
            self.config.save(config)

    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt (slow, salted)."""
        return _hash_password_bcrypt(password)

    def _verify_password(self, password: str, hashed: str) -> bool:
        """Verify password, migrating SHA-256 hashes to bcrypt."""
        if _verify_password_bcrypt(password, hashed):
            return True
        # Legacy SHA-256 hash — migrate to bcrypt
        legacy_hash = hashlib.sha256(password.encode()).hexdigest()
        if legacy_hash == hashed:
            config = self.config.load()
            config["admin_password_hash"] = self._hash_password(password)
            config["force_password_change"] = False
            self.config.save(config)
            return True
        return False

    def authenticate(self, username: str, password: str) -> Optional[dict]:
        """Returns session token dict on success, None on failure."""
        config = self.config.load()
        expected_hash = config.get("admin_password_hash", "")
        if not self._verify_password(password, expected_hash):
            logger.warning(f"Failed login attempt for user: {username}")
            return None
        session_token = secrets.token_urlsafe(32)
        with self._lock:
            self._sessions[session_token] = datetime.now(timezone.utc)
        force_change = config.get("force_password_change", False)
        return {"token": session_token, "force_password_change": force_change}

    def verify_session(self, session_token: str) -> bool:
        with self._lock:
            last_seen = self._sessions.get(session_token)
            if last_seen is None:
                return False
            idle = (datetime.now(timezone.utc) - last_seen).total_seconds()
            if idle > SESSION_IDLE_SECONDS:
                del self._sessions[session_token]
                logger.info("Session expired due to inactivity")
                return False
            self._sessions[session_token] = datetime.now(timezone.utc)
            return True

    def logout(self, session_token: str) -> None:
        with self._lock:
            self._sessions.pop(session_token, None)

    def verify_api_key(self, credentials: HTTPAuthorizationCredentials) -> bool:
        return self.token_mgr.validate(credentials.credentials)

    def change_password(self, old_password: str, new_password: str) -> bool:
        config = self.config.load()
        current_hash = config.get("admin_password_hash", "")
        if not self._verify_password(old_password, current_hash):
            return False
        if len(new_password) < 4:
            return False
        config["admin_password_hash"] = self._hash_password(new_password)
        config["force_password_change"] = False
        self.config.save(config)
        return True

    def check_auth(self, request: Request = None) -> bool:
        """FastAPI dependency: session cookie or Bearer API token."""
        if request is None:
            return False
        session_token = request.cookies.get("session_token")
        if session_token and self.verify_session(session_token):
            return True
        auth_header = request.headers.get("Authorization", "")
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()
            stored = self.config.load().get("api_token", "")
            if token and token == stored and self.token_mgr.validate(token):
                return True
        return False

    def check_auth_cookie(self, request: Request) -> bool:
        session_token = request.cookies.get("session_token")
        return bool(session_token and self.verify_session(session_token))
