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

import base64
import json
import logging
import secrets
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from paths import INSTALL_ROOT

logger = logging.getLogger("automanager.platform_quota")

# Anthropic rate-limits /api/oauth/usage aggressively unless the request looks
# like Claude Code; community tools poll at ~180s with a claude-code UA.
_CACHE_TTL_SECONDS = 180.0
_REQUEST_TIMEOUT = 20.0
# Once /api/oauth/usage returns 429 for a token, Anthropic often keeps that
# access token in a sticky bucket for a long time. Keep polling away until the
# window cools down instead of re-hitting every cache TTL.
_CLAUDE_USAGE_429_BACKOFF_SECONDS = 900.0

_ANTIGRAVITY_HOSTS = (
    "https://cloudcode-pa.googleapis.com",
    "https://daily-cloudcode-pa.googleapis.com",
)
# Cloud Code rejects unrecognized UAs (403). Official clients identify as
# plain ``antigravity`` / ``antigravity/<ver> <os>/<arch>``.
_ANTIGRAVITY_UA = "antigravity/1.11.3"
# Required: without a plain ``claude-code/<ver>`` UA the usage endpoint parks
# the token in a persistent 429 bucket (anthropics/claude-code#31021 / #31637).
# Do not append suffixes like "(external, cli)" — working clients send only
# ``claude-code/<version>``.
_CLAUDE_USAGE_UA = "claude-code/2.1.80"

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


def quota_cache_path(runtime_dir: Optional[Path] = None) -> Path:
    """Last-known provider quotas on disk (survives manager restarts)."""
    base = Path(runtime_dir or Path(INSTALL_ROOT) / "data" / "cliproxy")
    return base / "quota_cache.json"


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
    if not text:
        return None
    try:
        num = float(text)
        if 1_000_000_000 <= num <= 2_500_000_000:
            return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(num))
    except (ValueError, TypeError, OverflowError, OSError):
        pass
    return text


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


def _codex_window(rate: Dict[str, Any], *keys: str) -> Dict[str, Any]:
    for key in keys:
        value = rate.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _codex_reset_at(window: Dict[str, Any]) -> Optional[str]:
    direct = _iso_or_unix(
        window.get("reset_at") or window.get("resets_at") or window.get("resets_in_seconds")
    )
    if direct:
        return direct
    after = _to_float(window.get("reset_after_seconds") or window.get("resetAfterSeconds"))
    if after is None:
        return None
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + float(after)))
    except (OverflowError, OSError, ValueError):
        return None


def parse_codex_usage(payload: Any, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Parse Codex wham/usage JSON (and optional x-codex-* headers)."""
    data = payload if isinstance(payload, dict) else {}
    rate = data.get("rate_limit") if isinstance(data.get("rate_limit"), dict) else data
    if not isinstance(rate, dict):
        rate = {}
    primary = _codex_window(rate, "primary_window", "primary")
    secondary = _codex_window(rate, "secondary_window", "secondary")

    # Alguns planos deixam secondary_window=null; ainda assim a sessão (5h) vem
    # em primary_window. Janelas extras às vezes aparecem em additional_rate_limits.
    if not secondary:
        extras = data.get("additional_rate_limits") or rate.get("additional_rate_limits") or []
        if isinstance(extras, list):
            for item in extras:
                if not isinstance(item, dict):
                    continue
                nested = item.get("rate_limit") if isinstance(item.get("rate_limit"), dict) else item
                if not isinstance(nested, dict):
                    continue
                candidate = _codex_window(nested, "secondary_window", "primary_window", "primary")
                seconds = _to_float(
                    candidate.get("limit_window_seconds") or candidate.get("limitWindowSeconds")
                )
                # Prefer weekly-sized windows (~7d) when primary already filled session.
                if candidate and seconds is not None and seconds >= 86400:
                    secondary = candidate
                    break

    session_remaining = _remaining_from_used(
        _to_float(
            primary.get("used_percent")
            if primary.get("used_percent") is not None
            else primary.get("usedPercent")
        )
    )
    weekly_remaining = _remaining_from_used(
        _to_float(
            secondary.get("used_percent")
            if secondary.get("used_percent") is not None
            else secondary.get("usedPercent")
        )
    )
    session_reset = _codex_reset_at(primary)
    weekly_reset = _codex_reset_at(secondary)

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


def _antigravity_groups(payload: Any) -> List[Dict[str, Any]]:
    """Locate ``groups`` at top-level or under response/summary wrappers."""
    data = payload if isinstance(payload, dict) else {}
    direct = data.get("groups")
    if isinstance(direct, list):
        return [g for g in direct if isinstance(g, dict)]
    cmd_data = data.get("command")
    if isinstance(cmd_data, dict):
        inner_data = cmd_data.get("data")
        if isinstance(inner_data, dict) and isinstance(inner_data.get("groups"), list):
            return [g for g in inner_data["groups"] if isinstance(g, dict)]
    for key in ("response", "summary", "quotaSummary", "userQuotaSummary", "data"):
        wrapper = data.get(key)
        if isinstance(wrapper, dict):
            inner = wrapper.get("groups")
            if isinstance(inner, list):
                return [g for g in inner if isinstance(g, dict)]
    return []


def _antigravity_remaining_fraction(bucket: Dict[str, Any]) -> Optional[float]:
    """Read remainingFraction across flat, nested, and case-tagged shapes."""
    for key in ("remainingFraction", "remaining_fraction"):
        value = _to_float(bucket.get(key))
        if value is not None:
            return value
    remaining = bucket.get("remaining")
    if isinstance(remaining, dict):
        for key in ("remainingFraction", "remaining_fraction"):
            value = _to_float(remaining.get(key))
            if value is not None:
                return value
        # Protobuf oneof: {"case":"remainingFraction","value":0.48}
        case = str(remaining.get("case") or "").strip()
        if case in ("remainingFraction", "remaining_fraction"):
            value = _to_float(remaining.get("value"))
            if value is not None:
                return value
    for key in ("usedFraction", "used_fraction"):
        used = _to_float(bucket.get(key))
        if used is not None:
            return 1.0 - used
    return None


def _antigravity_fraction_to_pct(fraction: Optional[float]) -> Optional[float]:
    """Convert 0..1 fraction (or accidental 0..100 percent) to remaining %."""
    if fraction is None:
        return None
    value = float(fraction)
    if value > 1.0:
        # Some payloads already emit percent (e.g. 48.94). Avoid 48.94*100→100.
        return _clamp_pct(value)
    return _clamp_pct(value * 100.0)


def _antigravity_bucket_period(bucket: Dict[str, Any]) -> Optional[str]:
    """Classify bucket as weekly or session (5h) from window/id/name."""
    label = " ".join(
        str(bucket.get(key) or "")
        for key in (
            "window",
            "kind",
            "label",
            "bucketId",
            "bucket_id",
            "id",
            "displayName",
            "display_name",
            "name",
        )
    ).lower()
    if not label.strip():
        return None
    if "week" in label or "7d" in label or "seven" in label or "168h" in label:
        return "weekly"
    if (
        "5h" in label
        or "five" in label
        or "session" in label
        or "300m" in label
        or ("hour" in label and "24" not in label)
    ):
        return "session"
    return None


def _antigravity_bucket_reset(bucket: Dict[str, Any]) -> Optional[str]:
    for key in ("resetTime", "reset_time", "resetAt", "reset_at", "resets_at"):
        reset = _iso_or_unix(bucket.get(key))
        if reset:
            return reset
    return None


def _antigravity_group_kind(group: Dict[str, Any]) -> str:
    name = str(
        group.get("displayName")
        or group.get("display_name")
        or group.get("name")
        or ""
    ).lower()
    if "gemini" in name:
        return "gemini"
    if "claude" in name or "gpt" in name or "3p" in name:
        return "claude_gpt"
    # Infer from first bucketId prefix (gemini-weekly / 3p-5h).
    for bucket in group.get("buckets") or []:
        if not isinstance(bucket, dict):
            continue
        bucket_id = str(bucket.get("bucketId") or bucket.get("bucket_id") or "").lower()
        if bucket_id.startswith("gemini"):
            return "gemini"
        if bucket_id.startswith("3p") or "claude" in bucket_id:
            return "claude_gpt"
    return "other"


def parse_antigravity_quota(payload: Any) -> Dict[str, Any]:
    """Parse Antigravity retrieveUserQuotaSummary groups/buckets.

    Prefer the Gemini group (what the agy ``Models & Quota`` TUI shows first).
    Fall back to the most depleted window across groups when Gemini is absent.
    """
    by_kind: Dict[str, Dict[str, Any]] = {}
    for group in _antigravity_groups(payload):
        kind = _antigravity_group_kind(group)
        slot = by_kind.setdefault(
            kind,
            {
                "session_remaining_pct": None,
                "weekly_remaining_pct": None,
                "session_reset_at": None,
                "weekly_reset_at": None,
            },
        )
        for bucket in group.get("buckets") or []:
            if not isinstance(bucket, dict):
                continue
            period = _antigravity_bucket_period(bucket)
            if period is None:
                continue
            pct = _antigravity_fraction_to_pct(_antigravity_remaining_fraction(bucket))
            if pct is None:
                continue
            reset = _antigravity_bucket_reset(bucket)
            key_pct = f"{period}_remaining_pct"
            key_reset = f"{period}_reset_at"
            previous = slot.get(key_pct)
            if previous is None or pct < float(previous):
                slot[key_pct] = pct
                if reset:
                    slot[key_reset] = reset
            elif slot.get(key_reset) is None and reset:
                slot[key_reset] = reset

    preferred = by_kind.get("gemini") or None
    if preferred is None:
        # Most depleted weekly, then session — matches routing risk.
        candidates = [row for row in by_kind.values() if row]
        if candidates:
            preferred = min(
                candidates,
                key=lambda row: (
                    float(row["weekly_remaining_pct"])
                    if row.get("weekly_remaining_pct") is not None
                    else 101.0,
                    float(row["session_remaining_pct"])
                    if row.get("session_remaining_pct") is not None
                    else 101.0,
                ),
            )

    if not preferred:
        return {
            "session_remaining_pct": None,
            "weekly_remaining_pct": None,
            "session_reset_at": None,
            "weekly_reset_at": None,
            "source": "antigravity",
        }

    # Round like other providers for stable card text.
    session = preferred.get("session_remaining_pct")
    weekly = preferred.get("weekly_remaining_pct")
    return {
        "session_remaining_pct": round(float(session), 1) if session is not None else None,
        "weekly_remaining_pct": round(float(weekly), 1) if weekly is not None else None,
        "session_reset_at": preferred.get("session_reset_at"),
        "weekly_reset_at": preferred.get("weekly_reset_at"),
        "source": "antigravity",
        "focus_group": "gemini" if "gemini" in by_kind else "depleted",
    }


def _find_antigravity_binary() -> Optional[Path]:
    for name in ("agy", "antigravity"):
        found = shutil.which(name)
        if found:
            p = Path(found)
            if p.is_file():
                return p
    for candidate in (
        Path.home() / ".local/bin/agy",
        Path.home() / ".local/bin/antigravity",
        Path("/root/.local/bin/agy"),
        Path("/root/.local/bin/antigravity"),
        Path("/usr/local/bin/agy"),
        Path("/usr/local/bin/antigravity"),
        Path("/home/orangepi/.local/bin/agy"),
    ):
        if candidate.is_file():
            return candidate
    return None


def _get_active_agy_email() -> Optional[str]:
    for candidate in (
        Path.home() / ".gemini/antigravity-cli/antigravity-oauth-token",
        Path("/root/.gemini/antigravity-cli/antigravity-oauth-token"),
        Path("/home/orangepi/.gemini/antigravity-cli/antigravity-oauth-token"),
    ):
        if not candidate.is_file():
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
            id_token = data.get("id_token")
            if id_token and "." in id_token:
                parts = id_token.split(".")
                if len(parts) >= 2:
                    payload_b64 = parts[1] + "=="
                    jwt_payload = json.loads(
                        base64.urlsafe_b64decode(payload_b64.encode("ascii")).decode("utf-8")
                    )
                    email = jwt_payload.get("email")
                    if email:
                        return str(email).strip().lower()
        except Exception:
            pass
    return None


def fetch_antigravity_cli(auth: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Query local agy CLI binary for authoritative live quota when available."""
    binary = _find_antigravity_binary()
    if not binary:
        return None

    if auth:
        auth_email = str(auth.get("email") or "").strip().lower()
        if auth_email:
            active_email = _get_active_agy_email()
            if active_email and active_email != auth_email:
                return None

    try:
        proc = subprocess.run(
            [str(binary), "-p", "/usage", "--output-format", "json"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout:
            data = json.loads(proc.stdout)
            parsed = parse_antigravity_quota(data)
            if (
                parsed.get("session_remaining_pct") is not None
                or parsed.get("weekly_remaining_pct") is not None
            ):
                return parsed
    except Exception as exc:
        logger.debug("antigravity CLI usage query failed: %s", exc)
    return None


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

    # Anthropic unified rate limit headers (5h + 7d utilization).
    uni_7d_used = _to_float(lower.get("anthropic-ratelimit-unified-7d-utilization"))
    uni_5h_used = _to_float(lower.get("anthropic-ratelimit-unified-5h-utilization"))
    if uni_7d_used is not None or uni_5h_used is not None:
        return {
            "session_remaining_pct": _remaining_from_used(uni_5h_used * 100.0) if uni_5h_used is not None else None,
            "weekly_remaining_pct": _remaining_from_used(uni_7d_used * 100.0) if uni_7d_used is not None else None,
            "session_reset_at": _iso_or_unix(lower.get("anthropic-ratelimit-unified-5h-reset")),
            "weekly_reset_at": _iso_or_unix(
                lower.get("anthropic-ratelimit-unified-7d-reset")
                or lower.get("anthropic-ratelimit-unified-reset")
            ),
            "source": "claude-unified-headers",
        }

    # Anthropic subscription / API responses sometimes expose remaining tokens.
    remaining = _to_float(
        lower.get("anthropic-ratelimit-tokens-remaining")
        or lower.get("anthropic-ratelimit-requests-remaining")
    )
    limit = _to_float(
        lower.get("anthropic-ratelimit-tokens-limit")
        or lower.get("anthropic-ratelimit-requests-limit")
    )
    if remaining is not None and limit is not None and limit > 0:
        pct = _clamp_pct((remaining / limit) * 100.0)
        return {
            "session_remaining_pct": pct,
            "weekly_remaining_pct": None,
            "session_reset_at": _iso_or_unix(
                lower.get("anthropic-ratelimit-tokens-reset")
                or lower.get("anthropic-ratelimit-requests-reset")
            ),
            "weekly_reset_at": None,
            "source": "claude-headers",
        }
    return None


def _extract_auth_quota(auth: Dict[str, Any]) -> Dict[str, Any]:
    """Extract quota metrics directly from stored auth-file signals or cooldowns."""
    parsed: Dict[str, Any] = {
        "session_remaining_pct": None,
        "weekly_remaining_pct": None,
        "session_reset_at": None,
        "weekly_reset_at": None,
        "source": "auth-signals",
    }
    signals = auth.get("quota", {}).get("signals") if isinstance(auth.get("quota"), dict) else None
    if not signals and isinstance(auth.get("model_quotas"), dict):
        for mq in auth["model_quotas"].values():
            if isinstance(mq, dict) and isinstance(mq.get("signals"), dict):
                signals = mq["signals"]
                break
    if isinstance(signals, dict) and signals:
        h = parse_quota_headers(signals)
        if h:
            for k in ("session_remaining_pct", "weekly_remaining_pct", "session_reset_at", "weekly_reset_at"):
                if h.get(k) is not None:
                    parsed[k] = h[k]

    for cd in auth.get("cooldowns") or []:
        if isinstance(cd, dict):
            retry = _iso_or_unix(cd.get("retry_at"))
            reason = str(cd.get("reason") or "").lower()
            if retry and any(w in reason for w in ("quota", "rate_limit", "429")):
                if parsed.get("weekly_remaining_pct") is None or parsed.get("weekly_remaining_pct") <= 0:
                    parsed["weekly_remaining_pct"] = 0.0
                    parsed["weekly_reset_at"] = retry

    return parsed


def _friendly_quota_error(err: Optional[str], *, has_values: bool) -> Optional[str]:
    """Humanize provider errors for the card (avoid raw HTTP 429 noise)."""
    if not err:
        return None
    text = str(err).strip()
    if not text:
        return None
    low = text.lower()
    waiting = "429" in low or "em espera" in low or "rate limit" in low
    if waiting and has_values:
        return "Monitoramento em espera · exibindo ultima cota"
    if waiting:
        return "Monitoramento em espera — tente mais tarde"
    if has_values:
        return f"{text} · exibindo ultima cota"
    return text


def _preserve_account_on_error(
    previous: Optional[Dict[str, Any]], failed: Dict[str, Any]
) -> Dict[str, Any]:
    """Keep last known percentages when a refresh fails (e.g. usage HTTP 429)."""
    has_values = (
        failed.get("session_remaining_pct") is not None
        or failed.get("weekly_remaining_pct") is not None
    )
    if not previous:
        out = dict(failed)
        out["error"] = _friendly_quota_error(failed.get("error"), has_values=has_values)
        return out
    merged = dict(failed)
    for field in (
        "session_remaining_pct",
        "weekly_remaining_pct",
        "session_reset_at",
        "weekly_reset_at",
    ):
        if merged.get(field) is None and previous.get(field) is not None:
            merged[field] = previous[field]
    has_values = (
        merged.get("session_remaining_pct") is not None
        or merged.get("weekly_remaining_pct") is not None
    )
    # Stale-but-real values beat an empty error row in the UI.
    merged["error"] = _friendly_quota_error(failed.get("error"), has_values=has_values)
    return merged


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
        # auth_index → unix timestamp until which Claude usage calls are skipped
        self._claude_backoff_until: Dict[str, float] = {}
        self._load_disk_cache()

    def invalidate(self) -> None:
        with self._lock:
            self._cache.clear()
            self._fetched_at = 0.0
            self._claude_backoff_until.clear()
        self._clear_disk_cache()

    def _load_disk_cache(self) -> None:
        path = quota_cache_path(self._runtime_dir)
        if not path.is_file():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            logger.debug("quota disk cache unreadable: %s", exc)
            return
        if not isinstance(raw, dict):
            return
        quotas = raw.get("quotas")
        if isinstance(quotas, dict):
            loaded: Dict[str, Dict[str, Any]] = {}
            for key, value in quotas.items():
                if isinstance(value, dict):
                    loaded[str(key)] = dict(value)
            self._cache = loaded
        fetched_at = _to_float(raw.get("fetched_at"))
        if fetched_at is not None:
            self._fetched_at = float(fetched_at)
        backoff = raw.get("claude_backoff_until")
        if isinstance(backoff, dict):
            now = time.time()
            self._claude_backoff_until = {
                str(auth): float(until)
                for auth, until in backoff.items()
                if _to_float(until) is not None and float(until) > now
            }

    def _save_disk_cache(self) -> None:
        path = quota_cache_path(self._runtime_dir)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "fetched_at": self._fetched_at,
                "quotas": self._cache,
                "claude_backoff_until": self._claude_backoff_until,
                "saved_at": time.time(),
            }
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            tmp.replace(path)
            try:
                path.chmod(0o600)
            except OSError:
                pass
        except OSError as exc:
            logger.debug("quota disk cache write failed: %s", exc)

    def _clear_disk_cache(self) -> None:
        path = quota_cache_path(self._runtime_dir)
        try:
            if path.is_file():
                path.unlink()
        except OSError:
            pass

    def _claude_in_backoff(self, auth_index: str) -> bool:
        with self._lock:
            until = self._claude_backoff_until.get(str(auth_index), 0.0)
        return time.time() < until

    def _claude_backoff_remaining(self, auth_index: str) -> float:
        with self._lock:
            until = self._claude_backoff_until.get(str(auth_index), 0.0)
        return max(0.0, until - time.time())

    def _mark_claude_backoff(
        self, auth_index: str, seconds: float = _CLAUDE_USAGE_429_BACKOFF_SECONDS
    ) -> None:
        with self._lock:
            self._claude_backoff_until[str(auth_index)] = time.time() + float(seconds)
            self._save_disk_cache()
        logger.info(
            "Claude usage HTTP 429 for %s — backing off %.0fs before next poll",
            auth_index,
            seconds,
        )

    def _clear_claude_backoff(self, auth_index: str) -> None:
        with self._lock:
            if str(auth_index) in self._claude_backoff_until:
                self._claude_backoff_until.pop(str(auth_index), None)
                self._save_disk_cache()

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
            self._save_disk_cache()

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
            self._save_disk_cache()
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

    def _previous_accounts(self, provider: str) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            snap = self._cache.get(provider) or {}
            rows = snap.get("account_details") or []
        return {
            str(row.get("auth_index")): dict(row)
            for row in rows
            if row.get("auth_index")
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
            previous_by_provider = {
                provider: self._previous_accounts(provider) for provider in _QUOTA_PROVIDERS
            }
            for auth in files:
                if auth.get("disabled"):
                    continue
                provider = _infer_provider(auth)
                if not provider or not auth.get("auth_index"):
                    continue
                auth_index = str(auth.get("auth_index"))
                previous = previous_by_provider.get(provider, {}).get(auth_index)
                auth_quota = _extract_auth_quota(auth)

                if auth.get("unavailable"):
                    base_account = {
                        "auth_index": auth.get("auth_index"),
                        "label": auth.get("email")
                        or auth.get("label")
                        or auth.get("name"),
                        "email": auth.get("email"),
                        "priority": int(auth.get("priority") or 0),
                        "session_remaining_pct": auth_quota.get("session_remaining_pct"),
                        "weekly_remaining_pct": auth_quota.get("weekly_remaining_pct"),
                        "session_reset_at": auth_quota.get("session_reset_at"),
                        "weekly_reset_at": auth_quota.get("weekly_reset_at"),
                        "error": None,
                    }
                    account = _preserve_account_on_error(previous, base_account)
                    by_provider[provider].append(account)
                    continue

                if provider == "claude" and self._claude_in_backoff(auth_index):
                    remaining = self._claude_backoff_remaining(auth_index)
                    mins = max(1, int((remaining + 59) // 60))
                    base_account = {
                        "auth_index": auth.get("auth_index"),
                        "label": auth.get("email")
                        or auth.get("label")
                        or auth.get("name"),
                        "email": auth.get("email"),
                        "priority": int(auth.get("priority") or 0),
                        "session_remaining_pct": auth_quota.get("session_remaining_pct"),
                        "weekly_remaining_pct": auth_quota.get("weekly_remaining_pct"),
                        "session_reset_at": auth_quota.get("session_reset_at"),
                        "weekly_reset_at": auth_quota.get("weekly_reset_at"),
                        "error": f"em espera ~{mins} min apos HTTP 429",
                    }
                    account = _preserve_account_on_error(previous, base_account)
                    by_provider[provider].append(account)
                    continue
                try:
                    account = self._fetch_account(client, base, provider, auth)
                except Exception as exc:
                    if provider == "claude" and "HTTP 429" in str(exc):
                        self._mark_claude_backoff(auth_index)
                    base_account = {
                        "auth_index": auth.get("auth_index"),
                        "label": auth.get("email")
                        or auth.get("label")
                        or auth.get("name"),
                        "email": auth.get("email"),
                        "priority": int(auth.get("priority") or 0),
                        "session_remaining_pct": auth_quota.get("session_remaining_pct"),
                        "weekly_remaining_pct": auth_quota.get("weekly_remaining_pct"),
                        "session_reset_at": auth_quota.get("session_reset_at"),
                        "weekly_reset_at": auth_quota.get("weekly_reset_at"),
                        "error": str(exc),
                    }
                    account = _preserve_account_on_error(previous, base_account)
                else:
                    if provider == "claude":
                        self._clear_claude_backoff(auth_index)
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
            account_id = (
                auth.get("account_id")
                or auth.get("accountId")
                or auth.get("chatgpt_account_id")
                or auth.get("chatgptAccountId")
                or auth.get("workspace_id")
                or auth.get("account")
            )
            if account_id:
                # Codex emite as duas grafias em caminhos diferentes.
                headers["ChatGPT-Account-Id"] = str(account_id)
                headers["ChatGPT-Account-ID"] = str(account_id)
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
            if (
                parsed.get("session_remaining_pct") is None
                and parsed.get("weekly_remaining_pct") is None
            ):
                raise RuntimeError("codex usage sem janelas de cota na resposta")
        elif provider == "claude":
            # User-Agent must identify as Claude Code or Anthropic puts the
            # token in a persistent 429 bucket for this undocumented endpoint.
            status, body, resp_headers = self._api_call(
                client,
                base,
                auth_index=auth_index,
                method="GET",
                url="https://api.anthropic.com/api/oauth/usage",
                header={
                    "Authorization": "Bearer $TOKEN$",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "anthropic-beta": "oauth-2025-04-20",
                    "User-Agent": _CLAUDE_USAGE_UA,
                },
            )
            if status < 200 or status >= 300:
                raise RuntimeError(f"claude usage HTTP {status}")
            parsed = parse_claude_usage(body)
            if (
                parsed.get("session_remaining_pct") is None
                and parsed.get("weekly_remaining_pct") is None
            ):
                header_parsed = parse_quota_headers(resp_headers)
                if header_parsed:
                    parsed = {**parsed, **{
                        k: v for k, v in header_parsed.items() if v is not None
                    }}
        elif provider == "antigravity":
            parsed = self._fetch_antigravity(client, base, auth_index, auth)
        else:
            raise RuntimeError(f"unsupported provider {provider}")

        auth_quota = _extract_auth_quota(auth)
        return {
            "auth_index": auth_index,
            "label": label,
            "email": auth.get("email"),
            "priority": int(auth.get("priority") or 0),
            "session_remaining_pct": (
                parsed.get("session_remaining_pct")
                if parsed.get("session_remaining_pct") is not None
                else auth_quota.get("session_remaining_pct")
            ),
            "weekly_remaining_pct": (
                parsed.get("weekly_remaining_pct")
                if parsed.get("weekly_remaining_pct") is not None
                else auth_quota.get("weekly_remaining_pct")
            ),
            "session_reset_at": (
                parsed.get("session_reset_at")
                if parsed.get("session_reset_at") is not None
                else auth_quota.get("session_reset_at")
            ),
            "weekly_reset_at": (
                parsed.get("weekly_reset_at")
                if parsed.get("weekly_reset_at") is not None
                else auth_quota.get("weekly_reset_at")
            ),
            "error": None,
        }

    def _fetch_antigravity(
        self,
        client: httpx.Client,
        base: str,
        auth_index: str,
        auth: Dict[str, Any],
    ) -> Dict[str, Any]:
        cli_result = fetch_antigravity_cli(auth)
        if cli_result:
            return cli_result

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
        # Official clients often POST `{}` (or project). Wrong project can yield
        # empty/stale buckets, so try project first then fall back to empty body.
        bodies: List[str] = []
        if project:
            bodies.append(json.dumps({"project": project}))
        bodies.append("{}")
        bodies.append(json.dumps({"metadata": {"ideType": "ANTIGRAVITY"}}))

        for host in _ANTIGRAVITY_HOSTS:
            for req_body in bodies:
                try:
                    status, body, _ = self._api_call(
                        client,
                        base,
                        auth_index=auth_index,
                        method="POST",
                        url=f"{host}/v1internal:retrieveUserQuotaSummary",
                        header=header,
                        body=req_body,
                    )
                    if 200 <= status < 300:
                        parsed = parse_antigravity_quota(body)
                        if (
                            parsed.get("session_remaining_pct") is not None
                            or parsed.get("weekly_remaining_pct") is not None
                        ):
                            return parsed
                        last_error = RuntimeError("antigravity quota empty summary")
                        continue
                    last_error = RuntimeError(f"antigravity quota HTTP {status}")
                except Exception as exc:
                    last_error = exc

        if last_error:
            raise last_error
        raise RuntimeError("antigravity quota unavailable")
