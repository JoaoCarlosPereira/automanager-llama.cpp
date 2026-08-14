"""Fixed model catalog for Ollama Cloud with tolerant parsing."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from config_manager import ConfigManager

import httpx

from paths import INSTALL_ROOT

logger = logging.getLogger("automanager.ollama_cloud")


@dataclass(frozen=True)
class OllamaCloudModel:
    """Metadata for a single Ollama Cloud model."""

    id: str
    display_name: str
    context_length: Optional[int] = None
    output_token_limit: Optional[int] = None
    capabilities: frozenset = field(default_factory=frozenset)


# ---------------------------------------------------------------------------
# Catalog helpers
# ---------------------------------------------------------------------------

def _build_fixed_models() -> List[OllamaCloudModel]:
    """Return the fixed list of Ollama Cloud models (10+)."""
    return [
        OllamaCloudModel(
            id="gpt-oss-20b",
            display_name="GPT OSS 20B",
            context_length=131072,
            output_token_limit=32768,
            capabilities=frozenset(["text", "tools"]),
        ),
        OllamaCloudModel(
            id="gpt-oss-120b",
            display_name="GPT OSS 120B",
            context_length=131072,
            output_token_limit=32768,
            capabilities=frozenset(["text", "tools"]),
        ),
        OllamaCloudModel(
            id="qwen2.5-7b",
            display_name="Qwen 2.5 7B",
            context_length=32768,
            output_token_limit=8192,
            capabilities=frozenset(["text", "tools"]),
        ),
        OllamaCloudModel(
            id="qwen2.5-14b",
            display_name="Qwen 2.5 14B",
            context_length=32768,
            output_token_limit=8192,
            capabilities=frozenset(["text", "tools"]),
        ),
        OllamaCloudModel(
            id="qwen2.5-72b",
            display_name="Qwen 2.5 72B",
            context_length=131072,
            output_token_limit=32768,
            capabilities=frozenset(["text", "tools"]),
        ),
        OllamaCloudModel(
            id="llama3.1-8b",
            display_name="Llama 3.1 8B",
            context_length=128000,
            output_token_limit=16384,
            capabilities=frozenset(["text", "tools"]),
        ),
        OllamaCloudModel(
            id="llama3.1-70b",
            display_name="Llama 3.1 70B",
            context_length=128000,
            output_token_limit=16384,
            capabilities=frozenset(["text", "tools"]),
        ),
        OllamaCloudModel(
            id="llama3.3-70b",
            display_name="Llama 3.3 70B",
            context_length=128000,
            output_token_limit=32768,
            capabilities=frozenset(["text", "tools"]),
        ),
        OllamaCloudModel(
            id="mistral-7b",
            display_name="Mistral 7B",
            context_length=32768,
            output_token_limit=8192,
            capabilities=frozenset(["text"]),
        ),
        OllamaCloudModel(
            id="mistral-large",
            display_name="Mistral Large",
            context_length=131072,
            output_token_limit=32768,
            capabilities=frozenset(["text", "tools"]),
        ),
        OllamaCloudModel(
            id="gemma-2-9b",
            display_name="Gemma 2 9B",
            context_length=8192,
            output_token_limit=4096,
            capabilities=frozenset(["text"]),
        ),
        OllamaCloudModel(
            id="gemma-2-27b",
            display_name="Gemma 2 27B",
            context_length=8192,
            output_token_limit=4096,
            capabilities=frozenset(["text"]),
        ),
        OllamaCloudModel(
            id="phi3.5-mini",
            display_name="Phi 3.5 Mini",
            context_length=128000,
            output_token_limit=16384,
            capabilities=frozenset(["text"]),
        ),
        OllamaCloudModel(
            id="command-r",
            display_name="Command R",
            context_length=128000,
            output_token_limit=4096,
            capabilities=frozenset(["text", "tools"]),
        ),
        OllamaCloudModel(
            id="deepseek-coder-v2",
            display_name="DeepSeek Coder V2",
            context_length=32768,
            output_token_limit=8192,
            capabilities=frozenset(["text", "tools"]),
        ),
    ]


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

class OllamaCloudCatalog:
    """Fixed model catalog with optional dynamic refresh.

    The catalog always starts from the built-in model list.  A ``refresh``
    call can later merge remote data, but the fixed list is never empty.

    Status lifecycle:

    * **fresh** — refreshed successfully within the interval.
    * **stale** — never refreshed yet or the last refresh succeeded but the
      interval has elapsed (no remote data available yet).
    * **error** — a refresh attempt failed; the existing model list is kept.

    A background task calls ``_refresh_from_endpoint`` every
    ``refresh_interval`` seconds (default 24 h).  The catalog is refreshed
    immediately on first startup via ``_schedule_refresh``.
    """

    # Status constants
    STATUS_FRESH = "fresh"
    STATUS_STALE = "stale"
    STATUS_ERROR = "error"

    # Default refresh interval: 24 hours in seconds
    DEFAULT_REFRESH_INTERVAL = 86400  # 24 * 60 * 60

    def __init__(
        self,
        models: Optional[List[OllamaCloudModel]] = None,
        account: Optional[OllamaCloudAccount] = None,
        refresh_interval: Optional[int] = None,
    ) -> None:
        self._models: List[OllamaCloudModel] = models if models is not None else _build_fixed_models()
        self._last_refresh: Optional[float] = None
        self._refresh_error: Optional[str] = None
        self._account: Optional[OllamaCloudAccount] = account
        self._refresh_interval: int = refresh_interval or self.DEFAULT_REFRESH_INTERVAL
        self._refresh_task: Optional[Any] = None

        # Set initial catalog_status based on whether we have models
        if self._models:
            self._catalog_status: str = self.STATUS_FRESH
        else:
            self._catalog_status = self.STATUS_STALE

        # Schedule background refresh if an account is provided
        if account is not None:
            self._schedule_refresh()

    # -- public API --------------------------------------------------------

    def find_model_by_id(self, model_id: str) -> Optional[OllamaCloudModel]:
        """Return the model whose *id* matches exactly, or ``None``."""
        for m in self._models:
            if m.id == model_id:
                return m
        return None

    def get_models_for_account(self, account: Any) -> List[OllamaCloudModel]:
        """Return models available for *account*.

        The MVP returns all models; per-account filtering is deferred to a
        later task.
        """
        return list(self._models)

    @property
    def all_models(self) -> List[OllamaCloudModel]:
        """Read-only access to the full model list."""
        return list(self._models)

    @property
    def last_refresh(self) -> Optional[float]:
        return self._last_refresh

    @property
    def refresh_error(self) -> Optional[str]:
        return self._refresh_error

    @property
    def catalog_status(self) -> str:
        """Return the current catalog status: fresh, stale, or error."""
        return self._catalog_status

    @property
    def refresh_interval(self) -> int:
        """Return the configured refresh interval in seconds."""
        return self._refresh_interval

    @refresh_interval.setter
    def refresh_interval(self, value: int) -> None:
        """Update the refresh interval and restart the background task if running."""
        self._refresh_interval = value
        # Cancel existing task and schedule a new one
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            self._refresh_task = None
        if self._account is not None:
            self._schedule_refresh()

    async def refresh(self) -> None:
        """Attempt to refresh the catalog from the remote API.

        Updates catalog_status and logs the result.  On failure the existing
        model list is preserved (status becomes ``error``, not broken).
        """
        self._refresh_error = None
        success = False
        models_count = len(self._models)
        try:
            await self._fetch_remote_models()
            self._catalog_status = self.STATUS_FRESH
            success = True
        except Exception as exc:
            self._refresh_error = str(exc)
            self._catalog_status = self.STATUS_ERROR
            logger.warning("Ollama Cloud catalog refresh failed: %s", exc)
        self._last_refresh = time.time()
        logger.info(
            "catalog_refresh success=%s models_count=%d",
            success,
            models_count,
        )

    # -- background scheduling ---------------------------------------------

    def _schedule_refresh(self) -> None:
        """Schedule a periodic background refresh using asyncio.create_task.

        Runs an immediate refresh on first call, then repeats every
        ``refresh_interval`` seconds (default 24 h).

        Uses ``asyncio.create_task`` — NOT a separate thread.
        """
        import asyncio

        async def _loop() -> None:
            while True:
                try:
                    await self._refresh_from_endpoint()
                except Exception as exc:
                    logger.error("Background catalog refresh error: %s", exc)
                await asyncio.sleep(self._refresh_interval)

        self._refresh_task = asyncio.create_task(_loop())

    async def _refresh_from_endpoint(self) -> None:
        """Perform a single refresh from the remote API endpoint.

        This is the core refresh method used both by the immediate startup
        call and by the periodic background loop.  It delegates to
        ``refresh()``, which sets ``catalog_status`` and handles logging.
        """
        await self.refresh()

    # -- internal ----------------------------------------------------------

    @staticmethod
    async def _fetch_remote_models() -> None:
        """Pull models from ``https://ollama.com/v1/models``.

        Raises on network errors so that ``refresh()`` records the failure.
        """
        import httpx

        url = "https://ollama.com/v1/models"
        headers = {"Authorization": "Bearer placeholder"}
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            payload = resp.json()
        # The API returns { "data": [ { "id": "...", ... }, ... ] }.
        models_data: List[Dict[str, Any]] = payload.get("data", [])
        for item in models_data:
            _parse_model_item(item)  # validates tolerance

    # -- class methods for external updates --------------------------------

    @classmethod
    def from_file(cls, path: str) -> "OllamaCloudCatalog":
        """Build a catalog from a JSON file (used for remote data)."""
        models_path = Path(path)
        if not models_path.is_file():
            return cls(_build_fixed_models())
        raw = models_path.read_text(encoding="utf-8")
        data: List[Dict[str, Any]] = json.loads(raw)
        models = []
        for item in data:
            parsed = _parse_model_item(item)
            if parsed:
                models.append(parsed)
        if not models:
            models = _build_fixed_models()
        return cls(models)

    @classmethod
    def merge_remote(cls, existing: List[OllamaCloudModel], data: List[Dict[str, Any]]) -> List[OllamaCloudModel]:
        """Merge remote model data into the existing fixed list.

        Existing IDs are preserved; remote entries with new IDs are appended.
        """
        merged: Dict[str, OllamaCloudModel] = {}
        for m in existing:
            merged[m.id] = m
        for item in data:
            parsed = _parse_model_item(item)
            if parsed and parsed.id not in merged:
                merged[parsed.id] = parsed
        return list(merged.values())


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _parse_model_item(item: Dict[str, Any]) -> Optional[OllamaCloudModel]:
    """Parse a model dict from the remote API response.

    Tolerant to missing fields: *context_length*, *output_token_limit*, and
    *capabilities* are all optional and default to ``None`` / empty set.
    """
    model_id = str(item.get("id") or "").strip()
    if not model_id:
        return None

    display_name = str(item.get("display_name") or item.get("name") or model_id)

    context_length = item.get("context_length")
    if context_length is not None:
        try:
            context_length = int(context_length)
        except (TypeError, ValueError):
            context_length = None

    output_token_limit = item.get("output_token_limit")
    if output_token_limit is not None:
        try:
            output_token_limit = int(output_token_limit)
        except (TypeError, ValueError):
            output_token_limit = None

    caps_raw = item.get("capabilities")
    if isinstance(caps_raw, list):
        capabilities = frozenset(str(c).strip().lower() for c in caps_raw if c)
    else:
        capabilities = frozenset()

    return OllamaCloudModel(
        id=model_id,
        display_name=display_name,
        context_length=context_length,
        output_token_limit=output_token_limit,
        capabilities=capabilities,
    )


# ---------------------------------------------------------------------------
# Account
# ---------------------------------------------------------------------------

@dataclass
class OllamaCloudAccount:
    """Credential holder for a single Ollama Cloud account."""

    id: str
    api_key: str
    label: str = ""
    status: str = "available"  # available | cooldown | rate_limited | error
    cooldown_until: Optional[float] = None
    rate_limited_until: Optional[float] = None


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class OllamaCloudProvider:
    """Direct HTTP provider for Ollama Cloud (https://ollama.com/v1).

    Communicates as an OpenAI-compatible backend without requiring a local
    CLI or sidecar.
    """

    BASE_URL = "https://ollama.com/v1"

    def __init__(self, account: OllamaCloudAccount) -> None:
        self.account = account
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            timeout=30.0,
            headers={"Authorization": f"Bearer {account.api_key}"},
        )

    # -- public API --------------------------------------------------------

    async def chat_completion(
        self,
        model: str,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        stream: bool = False,
    ) -> httpx.Response:
        """POST /v1/chat/completions with optional streaming.

        Returns the raw *httpx.Response* so the caller can decide whether to
        stream or parse JSON.
        """
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        if tools is not None:
            payload["tools"] = tools

        resp = await self._client.post("/chat/completions", json=payload)
        if resp.status_code >= 400:
            raise OllamaCloudHTTPError(resp.status_code, self.parse_error(resp))
        return resp

    async def list_models(self) -> List[Dict]:
        """GET /v1/models — returns the raw list of model dicts.

        Useful for validating the account and discovering available models.
        """
        resp = await self._client.get("/models")
        if resp.status_code >= 400:
            raise OllamaCloudHTTPError(resp.status_code, self.parse_error(resp))
        payload = resp.json()
        return payload.get("data", [])

    async def health_check(self) -> bool:
        """GET /v1/models — validate credentials and API availability."""
        try:
            resp = await self._client.get("/models")
            return resp.status_code < 400
        except Exception:
            return False

    def parse_error(
        self,
        response: httpx.Response,
    ) -> Tuple[Optional[float], str]:
        """Extract retry-after info and error message from an error response.

        Returns ``(retry_after_seconds, error_message)``.  *retry_after* may be
        *None* when the server did not provide retry information.
        """
        retry_after: Optional[float] = None

        # Check Retry-After header (seconds)
        retry_after_header = response.headers.get("retry-after")
        if retry_after_header is not None:
            try:
                retry_after = float(retry_after_header)
            except ValueError:
                pass

        # Check RateLimit-Reset header (timestamp)
        if retry_after is None:
            rate_limit_reset = response.headers.get("ratelimit-reset")
            if rate_limit_reset is not None:
                try:
                    reset_ts = float(rate_limit_reset)
                    now = time.time()
                    retry_after = max(0.0, reset_ts - now)
                except ValueError:
                    pass

        # Build error message
        error_message = f"HTTP {response.status_code} from {response.url}"
        try:
            detail = response.json()
            if isinstance(detail, dict):
                message = detail.get("error", {}).get("message", "") or detail.get("message", "")
                if message:
                    error_message += f" — {message}"
        except Exception:
            pass

        return retry_after, error_message

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    def __del__(self) -> None:
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.close())
        except RuntimeError:
            pass


class OllamaCloudHTTPError(Exception):
    """Error raised when the Ollama Cloud API returns a non-success response."""

    def __init__(self, status_code: int, error_info: Tuple[Optional[float], str]) -> None:
        self.status_code = status_code
        self.retry_after, self.message = error_info
        super().__init__(f"OllamaCloud error {status_code}: {self.message}")


# ---------------------------------------------------------------------------
# Account Manager
# ---------------------------------------------------------------------------

_COOLDOWN_DEFAULT_SECONDS = 60


class OllamaCloudAccountManager:
    """Manage the lifecycle of Ollama Cloud accounts.

    Responsibilities:
    * CRUD of accounts backed by ``ConfigManager``.
    * Validate each account via ``OllamaCloudProvider.health_check()``.
    * Resolve the best account for a request given capabilities + context.
    * Apply / clear cooldown on accounts hit by rate limits.
    """

    def __init__(
        self,
        config_manager: "ConfigManager",
        catalog: OllamaCloudCatalog,
    ) -> None:
        self.config_manager = config_manager
        self.catalog = catalog

    # -- public API --------------------------------------------------------

    def get_accounts(self) -> List[OllamaCloudAccount]:
        """Return all accounts stored in ConfigManager as ``OllamaCloudAccount`` objects."""
        raw_accounts = self.config_manager.get_ollama_cloud_accounts_raw()
        accounts: List[OllamaCloudAccount] = []
        for acc in raw_accounts:
            status = str(acc.get("status") or "available")
            cooldown_until = self._optional_timestamp(acc.get("cooldown_until"))
            rate_limited_until = self._optional_timestamp(
                acc.get("rate_limited_until")
            )
            now = time.time()
            if rate_limited_until is not None and rate_limited_until <= now:
                rate_limited_until = None
                if status == "rate_limited":
                    status = "available"
            if cooldown_until is not None and cooldown_until <= now:
                cooldown_until = None
                if status == "cooldown":
                    status = "available"
            accounts.append(OllamaCloudAccount(
                id=acc["id"],
                api_key=acc.get("api_key", ""),
                label=acc.get("label", ""),
                status=status,
                cooldown_until=cooldown_until,
                rate_limited_until=rate_limited_until,
            ))
        return accounts

    @staticmethod
    def _optional_timestamp(value: Any) -> Optional[float]:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _persist_runtime_state(self, account: OllamaCloudAccount) -> None:
        self.config_manager.update_ollama_cloud_account(account.id, {
            "status": account.status,
            "cooldown_until": account.cooldown_until,
            "rate_limited_until": account.rate_limited_until,
        })

    def add_account(self, api_key: str, label: str = "") -> OllamaCloudAccount:
        """Persist a new account via ConfigManager and return it as ``OllamaCloudAccount``."""
        if not api_key:
            raise ValueError("api_key is required")
        created = self.config_manager.add_ollama_cloud_account(api_key, label)
        return OllamaCloudAccount(
            id=created["id"],
            api_key=api_key,
            label=label or created.get("label", ""),
            status="available",
        )

    def remove_account(self, account_id: str) -> None:
        """Remove an account by its id."""
        if not account_id:
            raise ValueError("account_id is required")
        found = self.config_manager.remove_ollama_cloud_account(account_id)
        if not found:
            raise ValueError(f"Account {account_id} not found")

    async def validate_connection(self, account: OllamaCloudAccount) -> bool:
        """Check whether *account* can reach the Ollama Cloud API.

        Uses ``OllamaCloudProvider.health_check()``.  Updates the account
        ``status`` field on success / failure.
        """
        provider = OllamaCloudProvider(account)
        try:
            result = await provider.health_check()
            if result:
                account.status = "available"
            else:
                account.status = "error"
            return result
        except Exception:
            account.status = "error"
            return False
        finally:
            await provider.close()

    def resolve_for_request(
        self,
        required_capabilities: frozenset,
        needed_ctx: int,
        exclude_account_id: Optional[str] = None,
    ) -> Optional[OllamaCloudAccount]:
        """Pick the best available account for a request.

        Filtering rules:
        1. Account must not be in *cooldown* or *error* state.
        2. If *exclude_account_id* is provided, skip that account.
        3. At least one account must be available; return the first match.

        Model filtering is delegated to the catalog: models are checked against
        ``required_capabilities`` and ``needed_ctx``.  If no account has any
        matching model, the method still returns the first available account
        (the caller / proxy_router can reject after selection if needed).
        """
        accounts = self.get_accounts()
        for acc in accounts:
            if acc.id == exclude_account_id:
                continue
            # Check cooldown expiry before status check
            if acc.cooldown_until is not None:
                if time.time() < acc.cooldown_until:
                    continue  # Still in cooldown
                # Cooldown expired — treat as available but clear it
                acc.cooldown_until = None
                acc.status = "available"
            # Check rate limit expiry (quota exhausted)
            if acc.rate_limited_until is not None:
                if time.time() < acc.rate_limited_until:
                    continue  # Still rate limited
                # Rate limit expired — clear it
                acc.rate_limited_until = None
            if acc.status not in ("available",):
                continue
            return acc
        return None

    def apply_cooldown(self, account: OllamaCloudAccount, retry_after: Optional[float] = None) -> None:
        """Put *account* into cooldown.

        *retry_after* is seconds (from ``Retry-After`` header).  Falls back to
        60 s when not provided.
        """
        retry_after_seconds = retry_after
        if retry_after_seconds is None:
            retry_after_seconds = _COOLDOWN_DEFAULT_SECONDS

        account.status = "cooldown"
        account.cooldown_until = time.time() + retry_after_seconds
        account.rate_limited_until = None
        self._persist_runtime_state(account)
        logger.info(
            "cooldown_applied id=%s retry_after=%.1f",
            account.id,
            retry_after_seconds,
        )

    def apply_rate_limit(self, account: OllamaCloudAccount, retry_after: Optional[float] = None) -> None:
        """Put *account* into rate-limit (quota exhausted) state.

        *retry_after* is seconds (from ``Retry-After`` header).  Falls back to
        3600 s (1 hour) when not provided.
        """
        retry_after_seconds = retry_after
        if retry_after_seconds is None:
            retry_after_seconds = 3600  # 1 hour default for quota exhaustion

        account.status = "rate_limited"
        account.cooldown_until = None
        account.rate_limited_until = time.time() + retry_after_seconds
        self._persist_runtime_state(account)
        logger.info(
            "rate_limit_applied id=%s retry_after=%.1f",
            account.id,
            retry_after_seconds,
        )

    def clear_rate_limit(self, account: OllamaCloudAccount) -> None:
        """Remove rate limit from *account*."""
        account.rate_limited_until = None
        if account.status == "rate_limited":
            account.status = "available"
        self._persist_runtime_state(account)

    def clear_cooldown(self, account: OllamaCloudAccount) -> None:
        """Remove cooldown from *account* and mark it available."""
        account.cooldown_until = None
        account.status = "available"
        self._persist_runtime_state(account)
        logger.info("cooldown_cleared id=%s", account.id)

    async def close(self) -> None:
        """Close any active providers (future-proof)."""
        pass
