"""Config, token, and auth managers for automanager."""

import os
import json
import logging
import hashlib
import secrets
import threading
import uuid as uuid_mod
from copy import deepcopy
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
    try:
        return bcrypt.checkpw(
            password.encode("utf-8"), hashed.encode("utf-8")
        )
    except (ValueError, TypeError):
        # Hash legado (SHA-256 hex) não é um hash bcrypt válido; o caller
        # tenta o caminho de migração SHA-256 em seguida.
        return False

from paths import CONFIG_PATH
from schemas import (
    DEFAULT_MTP_DRAFT_TOKENS,
    DEFAULT_MTP_ENABLED,
    DEFAULT_FLASH_ATTN_ENABLED,
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_PARALLEL_SLOTS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CACHE_TYPE,
    DEFAULT_PROXY_ELIGIBLE,
    DEFAULT_MAX_PARALLEL_REQUESTS,
    DEFAULT_PROXY_TTL_MINUTES,
    DEFAULT_PROXY_MAX_WAIT_SECONDS,
)
from utils import mask_api_key

DEFAULT_THINKING_ENABLED = True

DEFAULT_CONTEXT_OPTIMIZER = {
    "enabled": True,
    "audit_enabled": True,
    "tokenizers": {
        "models": {},
        "families": {},
    },
}

DEFAULT_SMART_PROXY = {
    "enabled": False,
    "primary_model_path": None,
    "primary_backend_id": None,
    "ttl_minutes": DEFAULT_PROXY_TTL_MINUTES,
    "max_wait_seconds": DEFAULT_PROXY_MAX_WAIT_SECONDS,
    "context_optimizer": DEFAULT_CONTEXT_OPTIMIZER,
}

DEFAULT_PLATFORM_BACKEND_IDS = (
    "platform:codex",
    "platform:claude-code",
    "platform:google-antigravity",
)

DEFAULT_PLATFORM_CONFIG = {
    "proxy_eligible": False,
    "max_parallel_requests": DEFAULT_MAX_PARALLEL_REQUESTS,
    "auto_start": False,
    "default_model": None,
}

# Nomes aceitos pelo Cursor em BYOK (validação server-side do editor).
CURSOR_COMPATIBLE_ALIAS_NAMES = (
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4",
    "gpt-3.5-turbo",
    "o3-mini",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4-turbo",
    "gpt-4-turbo-preview",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-4-0613",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-1106",
    "o1",
    "o1-mini",
    "o1-preview",
    "o1-pro",
    "o3",
    "o3-pro",
    "o4-mini",
    "gpt-5",
    "gpt-5.5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5-chat-latest",
    "codex-mini-latest",
)

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


def default_platform_configs() -> Dict[str, dict]:
    return {
        backend_id: dict(DEFAULT_PLATFORM_CONFIG)
        for backend_id in DEFAULT_PLATFORM_BACKEND_IDS
    }


def normalize_backend_id(backend_id: Optional[str]) -> str:
    if not backend_id:
        return ""
    return str(backend_id).strip()


def lookup_platform_config(platform_configs: dict, backend_id: str) -> dict:
    backend_id = normalize_backend_id(backend_id)
    if not backend_id:
        return {}
    defaults = default_platform_configs().get(
        backend_id, dict(DEFAULT_PLATFORM_CONFIG)
    )
    stored = platform_configs.get(backend_id)
    if not isinstance(stored, dict):
        stored = {}
    return {**defaults, **stored}


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
                # Save migration atomically inside the lock to avoid TOCTOU
                tmp_path = self.config_path + ".tmp"
                try:
                    with open(tmp_path, "w") as f:
                        json.dump(config, f, indent=2)
                    os.replace(tmp_path, self.config_path)
                except Exception as e:
                    logger.error(f"Config save error: {e}")
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
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

    def get_platform_configs(self) -> dict:
        config = self.load()
        stored = config.get("platform_configs")
        if not isinstance(stored, dict):
            stored = {}
        merged = default_platform_configs()
        for backend_id, settings in stored.items():
            norm = normalize_backend_id(backend_id)
            if not norm or not isinstance(settings, dict):
                continue
            merged[norm] = lookup_platform_config(stored, norm)
        return merged

    def get_platform_settings(self, backend_id: str) -> dict:
        config = self.load()
        stored = config.get("platform_configs")
        if not isinstance(stored, dict):
            stored = {}
        return lookup_platform_config(stored, backend_id)

    def update_platform_settings(self, backend_id: str, settings: dict) -> dict:
        backend_id = normalize_backend_id(backend_id)
        if not backend_id:
            raise ValueError("backend_id is required")
        config = self.load()
        platform_configs = config.get("platform_configs")
        if not isinstance(platform_configs, dict):
            platform_configs = {}
        prev = lookup_platform_config(platform_configs, backend_id)
        merged = {**prev, **(settings or {})}
        max_parallel = merged.get("max_parallel_requests")
        if not isinstance(max_parallel, int) or max_parallel < 1:
            max_parallel = DEFAULT_MAX_PARALLEL_REQUESTS
        default_model = merged.get("default_model")
        if isinstance(default_model, str):
            default_model = default_model.strip() or None
        elif default_model is not None:
            default_model = None
        entry = {
            **{
                k: v
                for k, v in merged.items()
                if k not in {
                    "proxy_eligible",
                    "max_parallel_requests",
                    "auto_start",
                    "default_model",
                }
            },
            "proxy_eligible": bool(
                merged.get("proxy_eligible", DEFAULT_PLATFORM_CONFIG["proxy_eligible"])
            ),
            "max_parallel_requests": max_parallel,
            "auto_start": bool(
                merged.get("auto_start", DEFAULT_PLATFORM_CONFIG["auto_start"])
            ),
            "default_model": default_model,
        }
        platform_configs[backend_id] = entry
        config["platform_configs"] = platform_configs
        self.save(config)
        return entry

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
        llama_server_bin = merged.get("llama_server_bin") or prev.get("llama_server_bin")
        turboquant_preset = merged.get("turboquant_preset")
        if turboquant_preset is None:
            turboquant_preset = prev.get("turboquant_preset")
        entry = {
            "context_size": merged.get("context_size", DEFAULT_CONTEXT_SIZE),
            "parallel_slots": merged.get("parallel_slots", DEFAULT_PARALLEL_SLOTS),
            "batch_size": merged.get("batch_size", DEFAULT_BATCH_SIZE),
            "ubatch_size": merged.get("ubatch_size", 512),
            "cache_type_k": merged.get("cache_type_k", DEFAULT_CACHE_TYPE),
            "cache_type_v": merged.get("cache_type_v", DEFAULT_CACHE_TYPE),
            "numa_enabled": merged.get("numa_enabled", False),
            "flash_attn_enabled": merged.get(
                "flash_attn_enabled", DEFAULT_FLASH_ATTN_ENABLED
            ),
            "threads": merged.get("threads", 0),
            "threads_batch": merged.get("threads_batch", 0),
            "mmproj_path": merged.get("mmproj_path"),
            "mmproj_disabled": bool(merged.get("mmproj_disabled", False)),
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
            "llama_server_bin": llama_server_bin,
            "turboquant_preset": turboquant_preset,
            "proxy_eligible": merged.get("proxy_eligible", DEFAULT_PROXY_ELIGIBLE),
            "max_parallel_requests": merged.get(
                "max_parallel_requests", DEFAULT_MAX_PARALLEL_REQUESTS
            ),
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

    def get_smart_proxy_settings(self) -> dict:
        """Chave global smart_proxy mesclada sobre defaults sem materializá-los."""
        config = self.load()
        stored = config.get("smart_proxy")
        if not isinstance(stored, dict):
            stored = {}
        return self._merge_smart_proxy_defaults(stored)

    def get_partial_config(self) -> dict:
        """Expe visão parcial administrativa sem expor senhas ou tokens."""
        config = self.load()
        smart_proxy = self.get_smart_proxy_settings()
        context_optimizer = smart_proxy.get("context_optimizer", {})
        tokenizers = context_optimizer.get("tokenizers", {})
        models_map = tokenizers.get("models", {}) if isinstance(tokenizers, dict) else {}
        families_map = tokenizers.get("families", {}) if isinstance(tokenizers, dict) else {}
        model_configs = config.get("model_configs", {}) if isinstance(config, dict) else {}
        return {
            "smart_proxy": smart_proxy,
            "context_optimizer": context_optimizer,
            "tokenizers_mapping_count": len(models_map) + len(families_map),
            "model_configs_count": len(model_configs),
        }

    @staticmethod
    def _merge_context_optimizer_defaults(stored: object) -> dict:
        stored = stored if isinstance(stored, dict) else {}
        defaults = deepcopy(DEFAULT_CONTEXT_OPTIMIZER)
        defaults.update({key: value for key, value in stored.items() if key != "tokenizers"})
        for key in ("enabled", "audit_enabled"):
            if key in stored:
                defaults[key] = bool(stored[key])
        tokenizers = stored.get("tokenizers")
        if isinstance(tokenizers, dict):
            defaults["tokenizers"] = {
                **defaults["tokenizers"],
                **tokenizers,
            }
            for key in ("models", "families"):
                if not isinstance(defaults["tokenizers"].get(key), dict):
                    defaults["tokenizers"][key] = {}
        return defaults

    @classmethod
    def _merge_smart_proxy_defaults(cls, stored: dict) -> dict:
        merged = {**DEFAULT_SMART_PROXY, **stored}
        merged["context_optimizer"] = cls._merge_context_optimizer_defaults(
            stored.get("context_optimizer")
        )
        return merged

    def update_smart_proxy_settings(self, partial: dict) -> dict:
        """Atualiza smart_proxy preservando campos desconhecidos e defaults legados."""
        config = self.load()
        stored = config.get("smart_proxy")
        if not isinstance(stored, dict):
            stored = {}
        partial = partial or {}
        primary_model_updated = "primary_model_path" in partial
        primary_backend_updated = "primary_backend_id" in partial
        merged = {**stored, **partial}
        context_partial = partial.get("context_optimizer")
        if isinstance(context_partial, dict):
            existing_context = stored.get("context_optimizer")
            merged["context_optimizer"] = {
                **(existing_context if isinstance(existing_context, dict) else {}),
                **context_partial,
            }
            if isinstance(context_partial.get("tokenizers"), dict):
                existing_tokenizers = (
                    existing_context.get("tokenizers", {})
                    if isinstance(existing_context, dict)
                    else {}
                )
                merged["context_optimizer"]["tokenizers"] = {
                    **(existing_tokenizers if isinstance(existing_tokenizers, dict) else {}),
                    **context_partial["tokenizers"],
                }
        merged = {**self._merge_smart_proxy_defaults(merged), **merged}
        merged["context_optimizer"] = self._merge_context_optimizer_defaults(
            merged.get("context_optimizer")
        )
        primary = merged.get("primary_model_path")
        merged["primary_model_path"] = (
            normalize_model_path(primary) if primary else None
        )
        primary_backend_id = normalize_backend_id(merged.get("primary_backend_id"))
        merged["primary_backend_id"] = primary_backend_id or None
        if primary_model_updated and merged["primary_model_path"]:
            merged["primary_backend_id"] = None
        if primary_backend_updated and not primary_backend_id:
            merged["primary_backend_id"] = None
        for key in ("ttl_minutes", "max_wait_seconds"):
            value = merged.get(key)
            if not isinstance(value, int) or value < 1:
                merged[key] = DEFAULT_SMART_PROXY[key]
        merged["enabled"] = bool(merged.get("enabled"))
        config["smart_proxy"] = merged
        self.save(config)
        return merged

    def get_model_aliases(self) -> Dict[str, str]:
        """Mapa alias_externo -> id_modelo_real (para clientes como Cursor)."""
        config = self.load()
        stored = config.get("model_aliases")
        if not isinstance(stored, dict):
            return {}
        cleaned: Dict[str, str] = {}
        for alias, target in stored.items():
            alias_key = str(alias or "").strip()
            target_val = str(target or "").strip()
            if alias_key and target_val:
                cleaned[alias_key] = target_val
        return cleaned

    def resolve_model_alias(self, model_name: Optional[str]) -> str:
        if not model_name:
            return model_name or ""
        aliases = self.get_model_aliases()
        return aliases.get(model_name, model_name)

    def is_removed_model_alias(self, model_name: Optional[str]) -> bool:
        """Return whether a previously configured alias was explicitly removed."""
        alias = str(model_name or "").strip()
        if not alias:
            return False
        config = self.load()
        removed = config.get("removed_model_aliases")
        return isinstance(removed, list) and alias in removed

    def set_model_alias(self, alias: str, target: Optional[str]) -> Dict[str, str]:
        alias_key = str(alias or "").strip()
        if not alias_key:
            raise ValueError("alias is required")
        config = self.load()
        aliases = config.get("model_aliases")
        if not isinstance(aliases, dict):
            aliases = {}
        removed = config.get("removed_model_aliases")
        if not isinstance(removed, list):
            removed = []
        target_val = str(target or "").strip()
        if target_val:
            aliases[alias_key] = target_val
            removed = [item for item in removed if item != alias_key]
        else:
            aliases.pop(alias_key, None)
            if alias_key not in removed:
                removed.append(alias_key)
        config["model_aliases"] = aliases
        config["removed_model_aliases"] = removed
        self.save(config)
        return self.get_model_aliases()

    def replace_model_alias_target(self, old_target: str, new_target: Optional[str]) -> None:
        """Update or remove aliases pointing at a local model target."""
        old_value = str(old_target or "").strip().replace("\\", "/")
        new_value = str(new_target or "").strip().replace("\\", "/")
        if not old_value:
            return
        config = self.load()
        aliases = config.get("model_aliases")
        if not isinstance(aliases, dict):
            return
        removed = config.get("removed_model_aliases")
        if not isinstance(removed, list):
            removed = []
        changed = False
        old_basename = os.path.basename(old_value)
        for alias, target in list(aliases.items()):
            target_value = str(target or "").strip().replace("\\", "/")
            if target_value != old_value and os.path.basename(target_value) != old_basename:
                continue
            if new_value:
                aliases[alias] = new_value
            else:
                aliases.pop(alias, None)
                if alias not in removed:
                    removed.append(alias)
            changed = True
        if changed:
            config["model_aliases"] = aliases
            config["removed_model_aliases"] = removed
            self.save(config)

    def get_default_models(self) -> list[str]:
        config = self.config.load() if hasattr(self, "config") else self.load()
        defaults = config.get("default_models")
        if isinstance(defaults, list):
            return [normalize_model_path(p) for p in defaults if p]
        legacy = config.get("default_model")
        return [normalize_model_path(legacy)] if legacy else []

    # ------------------------------------------------------------------ ollama_cloud_accounts
    @staticmethod
    def _mask_api_key(api_key: str) -> str:
        """Return a safe mask for *api_key* (``sk-****...****``).

        Delegado ao utilitário compartilhado ``utils.mask_api_key``.
        """
        return mask_api_key(api_key)

    def get_ollama_cloud_accounts(self) -> list[dict]:
        """Return a list of Ollama Cloud account dicts with masked api_key."""
        config = self.load()
        accounts = config.get("ollama_cloud_accounts")
        if not isinstance(accounts, list):
            return []
        return [
            {
                "id": acc["id"],
                "api_key": self._mask_api_key(acc.get("api_key", "")),
                "label": acc.get("label", ""),
                "created_at": acc.get("created_at"),
            }
            for acc in accounts
        ]

    def get_ollama_cloud_accounts_raw(self) -> list[dict]:
        """Return Ollama Cloud accounts for internal authenticated use only."""
        config = self.load()
        accounts = config.get("ollama_cloud_accounts")
        if not isinstance(accounts, list):
            return []
        return [dict(acc) for acc in accounts if isinstance(acc, dict)]

    def get_ollama_cloud_model_denials(self) -> dict[str, list[str]]:
        """Return model ids denied with subscription-required, by account id."""
        raw = self.load().get("ollama_cloud_model_denials")
        if not isinstance(raw, dict):
            return {}
        return {
            str(model_id): [str(account_id) for account_id in account_ids]
            for model_id, account_ids in raw.items()
            if isinstance(account_ids, list)
        }

    def record_ollama_cloud_model_denial(
        self, model_id: str, account_id: str
    ) -> None:
        model_id = str(model_id or "").strip()
        account_id = str(account_id or "").strip()
        if not model_id or not account_id:
            return
        config = self.load()
        denials = config.get("ollama_cloud_model_denials")
        if not isinstance(denials, dict):
            denials = {}
        account_ids = denials.get(model_id)
        if not isinstance(account_ids, list):
            account_ids = []
        if account_id not in account_ids:
            account_ids.append(account_id)
        denials[model_id] = account_ids
        config["ollama_cloud_model_denials"] = denials
        self.save(config)

    def clear_ollama_cloud_model_denials(self) -> None:
        config = self.load()
        if config.pop("ollama_cloud_model_denials", None) is not None:
            self.save(config)

    def add_ollama_cloud_account(self, api_key: str, label: str = "") -> dict:
        """Add a new Ollama Cloud account and return it (with masked api_key).

        The real api_key is stored; the returned dict always has a masked version.
        """
        if not api_key:
            raise ValueError("api_key is required")
        config = self.load()
        accounts: list = config.get("ollama_cloud_accounts")
        if not isinstance(accounts, list):
            accounts = []
        account = {
            "id": str(uuid_mod.uuid4()),
            "api_key": api_key,
            "label": label or "",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        accounts.append(account)
        config["ollama_cloud_accounts"] = accounts
        self.save(config)
        return {
            "id": account["id"],
            "api_key": self._mask_api_key(api_key),
            "label": account["label"],
            "created_at": account["created_at"],
        }

    def remove_ollama_cloud_account(self, account_id: str) -> bool:
        """Remove the Ollama Cloud account identified by *account_id*. Returns True if found."""
        config = self.load()
        accounts: list = config.get("ollama_cloud_accounts")
        if not isinstance(accounts, list):
            return False
        before = len(accounts)
        accounts = [acc for acc in accounts if acc.get("id") != account_id]
        if len(accounts) == before:
            return False  # not found
        config["ollama_cloud_accounts"] = accounts
        self.save(config)
        return True

    def update_ollama_cloud_account(self, account_id: str, updates: dict) -> Optional[dict]:
        """Partially update an Ollama Cloud account. Returns the updated dict (masked) or None."""
        config = self.load()
        accounts: list = config.get("ollama_cloud_accounts")
        if not isinstance(accounts, list):
            return None
        for acc in accounts:
            if acc.get("id") != account_id:
                continue
            if "api_key" in updates and updates["api_key"]:
                acc["api_key"] = updates["api_key"]
            if "label" in updates:
                acc["label"] = str(updates["label"] or "")
            if "created_at" in updates:
                acc["created_at"] = updates["created_at"]
            config["ollama_cloud_accounts"] = accounts
            self.save(config)
            return {
                "id": acc["id"],
                "api_key": self._mask_api_key(acc.get("api_key", "")),
                "label": acc.get("label", ""),
                "created_at": acc.get("created_at"),
            }
        return None


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

    def check_api_token(self, request: Request = None) -> bool:
        """Bearer API token only — for OpenAI-compatible /v1 routes."""
        if request is None:
            return False
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.lower().startswith("bearer "):
            return False
        token = auth_header[7:].strip()
        stored = self.config.load().get("api_token", "")
        return bool(token and token == stored and self.token_mgr.validate(token))

    def check_auth(self, request: Request = None) -> bool:
        """FastAPI dependency: session cookie or Bearer API token."""
        if request is None:
            return False
        session_token = request.cookies.get("session_token")
        if session_token and self.verify_session(session_token):
            return True
        return self.check_api_token(request)

    def check_auth_cookie(self, request: Request) -> bool:
        session_token = request.cookies.get("session_token")
        return bool(session_token and self.verify_session(session_token))
