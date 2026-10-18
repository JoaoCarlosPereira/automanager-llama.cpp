"""Fetch and normalize subscription quota for CLIProxyAPI platforms.

Providers expose dual windows that we map to card rings:

* **session** (inner ring) — short window (~5h), often called "daily/session"
* **weekly** (outer ring) — 7-day rolling window

Sources (via CLIProxy Management ``/api-call`` with stored OAuth):

| Provider     | Endpoint / shape                                              |
|--------------|---------------------------------------------------------------|
| Codex        | ``GET chatgpt.com/backend-api/wham/usage`` primary+secondary |
| Claude       | ``GET api.anthropic.com/api/oauth/usage`` five_hour+seven_day |
| Antigravity  | ``POST …/v1internal:retrieveUserQuotaSummary`` weekly+5h     |

Cursor / Ollama Cloud / generic OpenAI do not expose comparable dual windows
through the Management API, so their cards omit the rings until a source exists.
"""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from paths import INSTALL_ROOT

logger = logging.getLogger("automanager.platform_quota")

_CACHE_TTL_SECONDS = 90.0
_REQUEST_TIMEOUT = 20.0

_ANTIGRAVITY_HOSTS = (
    "https://daily-cloudcode-pa.googleapis.com",
    "https://daily-cloudcode-pa.sandbox.googleapis.com",
    "https://cloudcode-pa.googleapis.com",
)
_ANTIGRAVITY_UA = "antigravity/cli/1.0.13 (aidev_client; os_type=linux; arch=x86_64)"

# Providers we can query for dual-window quota.
_QUOTA_PROVIDERS = frozenset({"codex", "claude", "antigravity"})

# Map CLIProxy auth-file provider aliases → our provider key.
_PROVIDER_ALIASES = {
    "openai": "codex",
    "chatgpt": "codex",
    "anthropic": "claude",
    "gemini": "antigravity",
    "google": "antigravity",
    "agy": "antigravity",
}


def management_key_path(runtime_dir: Optional[Path] = None) -> Path:
    base = Path(runtime_dir or Path(INSTALL_ROOT) / "data" / "cliproxy")
    return base / "management.key"


def ensure_management_key(runtime_dir: Optional[Path] = None) -> str:
    """Return a stable plaintext management key, creating it on first use."""
    path = management_key_path(runtime_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        value = path.read_text(encoding="utf-8").strip()
        if value:
            return value
    value = secrets.token_urlsafe(32)
    path.write_text(value + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return value


def _to_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


def _clamp_pct(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return max(0.0, min(100.0, float(value)))


def _remaining_from_used(used: Optional[float]) -> Optional[float]:
    if used is None:
        return None
    return _clamp_pct(100.0 - float(used))


def _iso_or_unix(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(value)))
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    return text or None


def _normalize_provider(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    key = str(raw).strip().lower().replace("_", "-")
    key = _PROVIDER_ALIASES.get(key, key)
    if key in _QUOTA_PROVIDERS:
        return key
    return None


def _infer_provider(auth: Dict[str, Any]) -> Optional[str]:
    for field in ("provider", "account_type", "type", "label", "name", "id"):
        value = auth.get(field)
        if not value:
            continue
        direct = _normalize_provider(str(value))
        if direct:
            return direct
        lowered = str(value).lower()
        if "claude" in lowered or "anthropic" in lowered:
            return "claude"
        if "codex" in lowered or "openai" in lowered or "chatgpt" in lowered:
            return "codex"
        if "antigravity" in lowered or "gemini" in lowered or "agy" in lowered:
            return "antigravity"
    return None


def parse_codex_usage(payload: Any, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Parse Codex wham/usage JSON (and optional x-codex-* headers)."""
    data = payload if isinstance(payload, dict) else {}
    rate = data.get("rate_limit") if isinstance(data.get("rate_limit"), dict) else data
    primary = (
        rate.get("primary_window")
        if isinstance(rate.get("primary_window"), dict)
        else rate.get("primary")
        if isinstance(rate.get("primary"), dict)
        else {}
    )
    secondary = (
        rate.get("secondary_window")
        if isinstance(rate.get("secondary_window"), dict)
        else rate.get("secondary")
        if isinstance(rate.get("secondary"), dict)
        else {}
    )

    session_remaining = _remaining_from_used(
        _to_float(primary.get("used_percent") or primary.get("usedPercent"))
    )
    weekly_remaining = _remaining_from_used(
        _to_float(secondary.get("used_percent") or secondary.get("usedPercent"))
    )
    session_reset = _iso_or_unix(
        primary.get("reset_at") or primary.get("resets_at") or primary.get("resets_in_seconds")
    )
    weekly_reset = _iso_or_unix(
        secondary.get("reset_at") or secondary.get("resets_at") or secondary.get("resets_in_seconds")
    )

    headers = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
    if session_remaining is None:
        session_remaining = _remaining_from_used(
            _to_float(headers.get("x-codex-primary-used-percent"))
        )
        session_reset = session_reset or _iso_or_unix(headers.get("x-codex-primary-reset-at"))
    if weekly_remaining is None:
        weekly_remaining = _remaining_from_used(
            _to_float(headers.get("x-codex-secondary-used-percent"))
        )
        weekly_reset = weekly_reset or _iso_or_unix(headers.get("x-codex-secondary-reset-at"))

    return {
        "session_remaining_pct": session_remaining,
        "weekly_remaining_pct": weekly_remaining,
        "session_reset_at": session_reset,
        "weekly_reset_at": weekly_reset,
        "source": "codex",
    }


def parse_claude_usage(payload: Any) -> Dict[str, Any]:
    """Parse Anthropic OAuth usage JSON (five_hour + seven_day)."""
    data = payload if isinstance(payload, dict) else {}
    five = data.get("five_hour") if isinstance(data.get("five_hour"), dict) else {}
    seven = data.get("seven_day") if isinstance(data.get("seven_day"), dict) else {}

    session_used = _to_float(
        five.get("utilization")
        if five.get("utilization") is not None
        else five.get("used_percentage")
    )
    weekly_used = _to_float(
        seven.get("utilization")
        if seven.get("utilization") is not None
        else seven.get("used_percentage")
    )
    return {
        "session_remaining_pct": _remaining_from_used(session_used),
        "weekly_remaining_pct": _remaining_from_used(weekly_used),
        "session_reset_at": _iso_or_unix(five.get("resets_at")),
        "weekly_reset_at": _iso_or_unix(seven.get("resets_at")),
        "source": "claude",
    }


def parse_antigravity_quota(payload: Any) -> Dict[str, Any]:
    """Parse Antigravity retrieveUserQuotaSummary groups/buckets."""
    data = payload if isinstance(payload, dict) else {}
    session_vals: List[float] = []
    weekly_vals: List[float] = []
    session_reset: Optional[str] = None
    weekly_reset: Optional[str] = None

    for group in data.get("groups") or []:
        if not isinstance(group, dict):
            continue
        for bucket in group.get("buckets") or []:
            if not isinstance(bucket, dict):
                continue
            remaining = _to_float(bucket.get("remainingFraction"))
            if remaining is None:
                used = _to_float(bucket.get("usedFraction"))
                if used is not None:
                    remaining = 1.0 - used
            if remaining is None:
                continue
            pct = _clamp_pct(remaining * 100.0)
            if pct is None:
                continue
            kind = str(bucket.get("window") or bucket.get("kind") or "").lower()
            label = str(bucket.get("label") or "").lower()
            reset = _iso_or_unix(bucket.get("resetTime") or bucket.get("reset_at"))
            if "5h" in kind or "5h" in label or "five" in kind or "session" in kind:
                session_vals.append(pct)
                session_reset = session_reset or reset
            elif "week" in kind or "7d" in kind or "week" in label:
                weekly_vals.append(pct)
                weekly_reset = weekly_reset or reset

    return {
        "session_remaining_pct": min(session_vals) if session_vals else None,
        "weekly_remaining_pct": min(weekly_vals) if weekly_vals else None,
        "session_reset_at": session_reset,
        "weekly_reset_at": weekly_reset,
        "source": "antigravity",
    }


def parse_quota_headers(headers: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Best-effort parse of provider rate-limit headers from proxied responses."""
    lower = {str(k).lower(): str(v) for k, v in headers.items()}
    if any(k.startswith("x-codex-") for k in lower):
        parsed = parse_codex_usage({}, lower)
        if (
            parsed.get("session_remaining_pct") is not None
            or parsed.get("weekly_remaining_pct") is not None
        ):
            return parsed
    return None


def _empty_quota(provider: str, *, error: Optional[str] = None) -> Dict[str, Any]:
    return {
        "provider": provider,
        "available": False,
        "session_remaining_pct": None,
        "weekly_remaining_pct": None,
        "session_reset_at": None,
        "weekly_reset_at": None,
        "accounts": 0,
        "account_details": [],
        "focused_auth_index": None,
        "error": error,
        "updated_at": time.time(),
    }


def _account_sort_key(account: Dict[str, Any]) -> Tuple[int, float, str]:
    """Prefer higher priority, then most depleted weekly remaining, then label."""
    priority = int(account.get("priority") or 0)
    weekly = account.get("weekly_remaining_pct")
    weekly_key = float(weekly) if weekly is not None else 101.0
    label = str(account.get("label") or account.get("auth_index") or "")
    return (-priority, weekly_key, label.lower())


def _merge_account_quotas(provider: str, accounts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Keep real per-account windows; card summary follows the focused account.

    Focus defaults to the highest-priority account (CLIProxy routing order).
    When priorities tie, the most depleted weekly window is preferred so the
    card rings surface real pressure instead of a blended average.
    """
    details: List[Dict[str, Any]] = []
    for raw in accounts:
        session = raw.get("session_remaining_pct")
        weekly = raw.get("weekly_remaining_pct")
        has_values = session is not None or weekly is not None
        details.append(
            {
                "auth_index": raw.get("auth_index"),
                "label": raw.get("label") or raw.get("email") or raw.get("name") or "conta",
                "email": raw.get("email"),
                "priority": int(raw.get("priority") or 0),
                "session_remaining_pct": (
                    round(float(session), 1) if session is not None else None
                ),
                "weekly_remaining_pct": (
                    round(float(weekly), 1) if weekly is not None else None
                ),
                "session_reset_at": raw.get("session_reset_at"),
                "weekly_reset_at": raw.get("weekly_reset_at"),
                "available": has_values,
                "error": raw.get("error"),
            }
        )

    details.sort(key=_account_sort_key)
    usable = [a for a in details if a.get("available")]
    if not usable and not details:
        err = next((a.get("error") for a in accounts if a.get("error")), None)
        return _empty_quota(provider, error=err)

    focused = usable[0] if usable else details[0]
    return {
        "provider": provider,
        "available": bool(usable),
        "session_remaining_pct": focused.get("session_remaining_pct"),
        "weekly_remaining_pct": focused.get("weekly_remaining_pct"),
        "session_reset_at": focused.get("session_reset_at"),
        "weekly_reset_at": focused.get("weekly_reset_at"),
        "accounts": len(details),
        "account_details": details,
        "focused_auth_index": focused.get("auth_index"),
        "error": None if usable else next(
            (a.get("error") for a in details if a.get("error")), None
        ),
        "updated_at": time.time(),
    }


class PlatformQuotaService:
    """Cached quota snapshots keyed by provider, refreshed via Management API."""

    def __init__(
        self,
        *,
        runtime_dir: Optional[Path] = None,
        port_provider=None,
        http_client_factory=None,
        cache_ttl: float = _CACHE_TTL_SECONDS,
    ) -> None:
        self._runtime_dir = Path(runtime_dir or Path(INSTALL_ROOT) / "data" / "cliproxy")
        self._port_provider = port_provider
        self._http_client_factory = http_client_factory or (
            lambda: httpx.Client(timeout=_REQUEST_TIMEOUT)
        )
        self._cache_ttl = cache_ttl
        self._lock = threading.Lock()
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._fetched_at: float = 0.0

    def invalidate(self) -> None:
        with self._lock:
            self._cache.clear()
            self._fetched_at = 0.0

    def cached_snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Return the last known snapshot without contacting providers."""
        with self._lock:
            return {k: dict(v) for k, v in self._cache.items()}

    def update_from_headers(self, provider: str, headers: Dict[str, Any]) -> None:
        """Merge live response headers into the focused account cache."""
        key = _normalize_provider(provider)
        if not key:
            return
        parsed = parse_quota_headers(headers)
        if not parsed:
            return
        with self._lock:
            current = dict(self._cache.get(key) or _empty_quota(key))
            details = [dict(row) for row in (current.get("account_details") or [])]
            focus_id = current.get("focused_auth_index")
            target = None
            if details:
                for row in details:
                    if focus_id and row.get("auth_index") == focus_id:
                        target = row
                        break
                if target is None:
                    target = details[0]
            if target is None:
                target = {
                    "auth_index": focus_id or "live",
                    "label": key,
                    "priority": 0,
                }
                details = [target]
            for field in (
                "session_remaining_pct",
                "weekly_remaining_pct",
                "session_reset_at",
                "weekly_reset_at",
            ):
                if parsed.get(field) is not None:
                    target[field] = parsed[field]
                    current[field] = parsed[field]
            target["available"] = (
                target.get("session_remaining_pct") is not None
                or target.get("weekly_remaining_pct") is not None
            )
            current["account_details"] = details
            current["accounts"] = len(details)
            current["focused_auth_index"] = target.get("auth_index")
            current["available"] = any(bool(row.get("available")) for row in details)
            current["error"] = None
            current["updated_at"] = time.time()
            current["provider"] = key
            self._cache[key] = current

    def snapshot(self, *, force: bool = False) -> Dict[str, Dict[str, Any]]:
        """Return provider → quota map, refreshing when stale."""
        with self._lock:
            fresh = (time.time() - self._fetched_at) < self._cache_ttl and not force
            if fresh and self._cache:
                return {k: dict(v) for k, v in self._cache.items()}

        try:
            refreshed = self._refresh()
        except Exception as exc:
            logger.debug("platform quota refresh failed: %s", exc)
            with self._lock:
                return {k: dict(v) for k, v in self._cache.items()}

        with self._lock:
            self._cache = refreshed
            self._fetched_at = time.time()
            return {k: dict(v) for k, v in self._cache.items()}

    def for_provider(self, provider: str, *, force: bool = False) -> Dict[str, Any]:
        key = _normalize_provider(provider) or str(provider or "")
        snap = self.snapshot(force=force)
        return snap.get(key) or _empty_quota(key)

    def _base_url(self) -> Optional[str]:
        port = None
        if callable(self._port_provider):
            try:
                port = self._port_provider()
            except Exception:
                port = None
        if not port:
            return None
        return f"http://127.0.0.1:{int(port)}"

    def _management_headers(self) -> Dict[str, str]:
        key = ensure_management_key(self._runtime_dir)
        return {
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
            "User-Agent": "automanager-platform-quota",
        }

    def _refresh(self) -> Dict[str, Dict[str, Any]]:
        base = self._base_url()
        if not base:
            return {}

        with self._http_client_factory() as client:
            files = self._list_auth_files(client, base)
            by_provider: Dict[str, List[Dict[str, Any]]] = {
                p: [] for p in _QUOTA_PROVIDERS
            }
            for auth in files:
                if auth.get("disabled") or auth.get("unavailable"):
                    continue
                provider = _infer_provider(auth)
                if not provider or not auth.get("auth_index"):
                    continue
                try:
                    account = self._fetch_account(client, base, provider, auth)
                except Exception as exc:
                    account = {
                        "auth_index": auth.get("auth_index"),
                        "label": auth.get("email") or auth.get("label") or auth.get("name"),
                        "email": auth.get("email"),
                        "priority": int(auth.get("priority") or 0),
                        "session_remaining_pct": None,
                        "weekly_remaining_pct": None,
                        "session_reset_at": None,
                        "weekly_reset_at": None,
                        "error": str(exc),
                    }
                else:
                    account.setdefault("auth_index", auth.get("auth_index"))
                    account.setdefault("email", auth.get("email"))
                    account.setdefault("priority", int(auth.get("priority") or 0))
                by_provider[provider].append(account)

            return {
                provider: _merge_account_quotas(provider, accounts)
                for provider, accounts in by_provider.items()
                if accounts
            }

    def _list_auth_files(self, client: httpx.Client, base: str) -> List[Dict[str, Any]]:
        resp = client.get(
            f"{base}/v0/management/auth-files",
            headers=self._management_headers(),
        )
        if resp.status_code == 404:
            raise RuntimeError("CLIProxy Management API unavailable (secret-key not set)")
        resp.raise_for_status()
        payload = resp.json()
        files = payload.get("files") if isinstance(payload, dict) else None
        if not isinstance(files, list):
            raise RuntimeError("invalid auth-files response")
        return [f for f in files if isinstance(f, dict)]

    def _api_call(
        self,
        client: httpx.Client,
        base: str,
        *,
        auth_index: str,
        method: str,
        url: str,
        header: Optional[Dict[str, str]] = None,
        body: Optional[str] = None,
    ) -> Tuple[int, Any, Dict[str, str]]:
        payload: Dict[str, Any] = {
            "auth_index": auth_index,
            "method": method,
            "url": url,
            "header": header or {},
        }
        if body is not None:
            payload["body"] = body
        resp = client.post(
            f"{base}/v0/management/api-call",
            headers={**self._management_headers(), "Content-Type": "application/json"},
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        status = int(data.get("status_code") or 0)
        raw_body = data.get("body")
        if isinstance(raw_body, str):
            try:
                parsed_body: Any = json.loads(raw_body)
            except json.JSONDecodeError:
                parsed_body = raw_body
        else:
            parsed_body = raw_body
        headers_out: Dict[str, str] = {}
        raw_headers = data.get("header") or data.get("headers") or {}
        if isinstance(raw_headers, dict):
            for key, value in raw_headers.items():
                if isinstance(value, list) and value:
                    headers_out[str(key)] = str(value[0])
                elif value is not None:
                    headers_out[str(key)] = str(value)
        return status, parsed_body, headers_out

    def _fetch_account(
        self,
        client: httpx.Client,
        base: str,
        provider: str,
        auth: Dict[str, Any],
    ) -> Dict[str, Any]:
        auth_index = str(auth["auth_index"])
        label = auth.get("email") or auth.get("label") or auth.get("name") or auth_index
        if provider == "codex":
            headers = {
                "Authorization": "Bearer $TOKEN$",
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "codex_cli_rs/0.76.0 (Debian; x86_64) AutoManager",
            }
            account_id = auth.get("account_id") or auth.get("account")
            if account_id:
                headers["ChatGPT-Account-Id"] = str(account_id)
            status, body, resp_headers = self._api_call(
                client,
                base,
                auth_index=auth_index,
                method="GET",
                url="https://chatgpt.com/backend-api/wham/usage",
                header=headers,
            )
            if status < 200 or status >= 300:
                raise RuntimeError(f"codex usage HTTP {status}")
            parsed = parse_codex_usage(body, resp_headers)
        elif provider == "claude":
            status, body, _ = self._api_call(
                client,
                base,
                auth_index=auth_index,
                method="GET",
                url="https://api.anthropic.com/api/oauth/usage",
                header={
                    "Authorization": "Bearer $TOKEN$",
                    "Content-Type": "application/json",
                    "anthropic-beta": "oauth-2025-04-20",
                },
            )
            if status < 200 or status >= 300:
                raise RuntimeError(f"claude usage HTTP {status}")
            parsed = parse_claude_usage(body)
        elif provider == "antigravity":
            parsed = self._fetch_antigravity(client, base, auth_index, auth)
        else:
            raise RuntimeError(f"unsupported provider {provider}")

        return {
            "auth_index": auth_index,
            "label": label,
            "email": auth.get("email"),
            "priority": int(auth.get("priority") or 0),
            "session_remaining_pct": parsed.get("session_remaining_pct"),
            "weekly_remaining_pct": parsed.get("weekly_remaining_pct"),
            "session_reset_at": parsed.get("session_reset_at"),
            "weekly_reset_at": parsed.get("weekly_reset_at"),
            "error": None,
        }

    def _fetch_antigravity(
        self,
        client: httpx.Client,
        base: str,
        auth_index: str,
        auth: Dict[str, Any],
    ) -> Dict[str, Any]:
        header = {
            "Authorization": "Bearer $TOKEN$",
            "Content-Type": "application/json",
            "User-Agent": _ANTIGRAVITY_UA,
        }
        project = auth.get("project_id") or auth.get("project")
        if not project:
            try:
                status, body, _ = self._api_call(
                    client,
                    base,
                    auth_index=auth_index,
                    method="POST",
                    url=f"{_ANTIGRAVITY_HOSTS[0]}/v1internal:loadCodeAssist",
                    header=header,
                    body=json.dumps({"metadata": {"ideType": "ANTIGRAVITY"}}),
                )
                if 200 <= status < 300 and isinstance(body, dict):
                    cp = body.get("cloudaicompanionProject")
                    if isinstance(cp, str):
                        project = cp
                    elif isinstance(cp, dict):
                        project = cp.get("id")
            except Exception as exc:
                logger.debug("antigravity loadCodeAssist failed: %s", exc)

        last_error: Optional[Exception] = None
        if project:
            for host in _ANTIGRAVITY_HOSTS:
                try:
                    status, body, _ = self._api_call(
                        client,
                        base,
                        auth_index=auth_index,
                        method="POST",
                        url=f"{host}/v1internal:retrieveUserQuotaSummary",
                        header=header,
                        body=json.dumps({"project": project}),
                    )
                    if 200 <= status < 300:
                        return parse_antigravity_quota(body)
                    last_error = RuntimeError(f"antigravity quota HTTP {status}")
                except Exception as exc:
                    last_error = exc
        else:
            status, body, _ = self._api_call(
                client,
                base,
                auth_index=auth_index,
                method="POST",
                url="https://cloudcode-pa.googleapis.com/v1internal:retrieveUserQuotaSummary",
                header=header,
                body=json.dumps({"metadata": {"ideType": "ANTIGRAVITY"}}),
            )
            if 200 <= status < 300:
                return parse_antigravity_quota(body)
            last_error = RuntimeError(f"antigravity quota HTTP {status}")

        if last_error:
            raise last_error
        raise RuntimeError("antigravity quota unavailable")
