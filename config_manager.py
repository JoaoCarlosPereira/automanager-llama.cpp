"""Config, token, and auth managers for automanager."""

import os
import json
import logging
import hashlib
import secrets
import threading
from datetime import datetime
from typing import Dict, Optional

from fastapi.security import HTTPAuthorizationCredentials

CONFIG_PATH = "/root/automanager_config.json"
MANAGER_LOG_PATH = "/root/manager.log"
DEFAULT_CONTEXT_SIZE = 65536
DEFAULT_PARALLEL_SLOTS = 1

logger = logging.getLogger("automanager")


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
                        return json.load(f)
                except (json.JSONDecodeError, OSError):
                    return {}
            return {}

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
        return config.get("model_configs", {}).get(model_path, {})

    def update_model_settings(self, model_path: str, settings: dict) -> None:
        config = self.load()
        if "model_configs" not in config:
            config["model_configs"] = {}
        config["model_configs"][model_path] = {
            "context_size": settings.get("context_size", DEFAULT_CONTEXT_SIZE),
            "parallel_slots": settings.get("parallel_slots", DEFAULT_PARALLEL_SLOTS),
            "mmproj_path": settings.get("mmproj_path"),
            "gpu_weights": settings.get("gpu_weights"),
            "split_mode": settings.get("split_mode", "layer"),
            "last_started": datetime.utcnow().isoformat(),
        }
        self.save(config)

    def set_default_model(self, path: Optional[str]) -> None:
        config = self.load()
        config["default_model"] = path
        self.save(config)

    def get_default_model(self) -> Optional[str]:
        return self.load().get("default_model")


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
            config["admin_password_hash"] = self._hash_password("admin")
            self.config.save(config)

    @staticmethod
    def _hash_password(password: str) -> str:
        """Simple SHA-256 hash — in production, use bcrypt via passlib."""
        return hashlib.sha256(password.encode()).hexdigest()

    def authenticate(self, username: str, password: str) -> Optional[str]:
        """Returns session token on success, None on failure."""
        config = self.config.load()
        expected_hash = config.get("admin_password_hash", "")
        actual_hash = hashlib.sha256(password.encode()).hexdigest()
        if actual_hash != expected_hash:
            logger.warning(f"Failed login attempt for user: {username}")
            return None
        session_token = secrets.token_urlsafe(32)
        with self._lock:
            self._sessions[session_token] = datetime.utcnow()
        return session_token

    def verify_session(self, session_token: str) -> bool:
        with self._lock:
            if session_token in self._sessions:
                # Extend session
                self._sessions[session_token] = datetime.utcnow()
                return True
            return False

    def logout(self, session_token: str) -> None:
        with self._lock:
            self._sessions.pop(session_token, None)

    def verify_api_key(self, credentials: HTTPAuthorizationCredentials) -> bool:
        return self.token_mgr.validate(credentials.credentials)

    def change_password(self, old_password: str, new_password: str) -> bool:
        config = self.config.load()
        current_hash = hashlib.sha256(old_password.encode()).hexdigest()
        if config.get("admin_password_hash") != current_hash:
            return False
        config["admin_password_hash"] = self._hash_password(new_password)
        self.config.save(config)
        return True
