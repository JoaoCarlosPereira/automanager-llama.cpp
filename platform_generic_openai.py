import time
from contextlib import asynccontextmanager
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from config_manager import ConfigManager

import httpx

logger = logging.getLogger("automanager.generic_openai")


@dataclass
class GenericOpenAIAccount:
    id: str
    name: str
    base_url: str
    api_key: str
    status: str = "available"
    cooldown_until: Optional[float] = None
    rate_limited_until: Optional[float] = None

    @property
    def completions_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    @property
    def models_url(self) -> str:
        return f"{self.base_url}/models"


@dataclass(frozen=True)
class GenericOpenAIModel:
    id: str
    display_name: str
    context_length: Optional[int] = None
    output_token_limit: Optional[int] = None
    capabilities: frozenset = field(default_factory=lambda: frozenset(["text"]))
    backend_id: str = "platform:generic-openai"
    account_id: Optional[str] = None


class GenericOpenAIProvider:
    def __init__(self, account: GenericOpenAIAccount, timeout: float = 30.0):
        self.account = account
        self.timeout = timeout
        headers = {}
        if account.api_key:
            headers["Authorization"] = f"Bearer {account.api_key}"
        self._client = httpx.AsyncClient(timeout=timeout, headers=headers)

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get(self.account.models_url)
            resp.raise_for_status()
            return True
        except Exception as e:
            logger.warning(f"Health check failed for {self.account.name}: {e}")
            return False

    async def fetch_models(self) -> List[GenericOpenAIModel]:
        try:
            resp = await self._client.get(self.account.models_url)
            resp.raise_for_status()
            data = resp.json()
            models_list = data.get("data", [])
            return [
                GenericOpenAIModel(
                    id=m.get("id"),
                    display_name=m.get("id"),
                    context_length=m.get("context_length") or m.get("inputTokenLimit"),
                    output_token_limit=m.get("outputTokenLimit"),
                    account_id=self.account.id,
                )
                for m in models_list if m.get("id")
            ]
        except Exception as e:
            logger.warning(f"Failed to fetch models for {self.account.name}: {e}")
            return []

    async def chat_completion(self, payload: Dict[str, Any]) -> httpx.Response:
        """Forward a non-streaming OpenAI payload without changing its shape."""
        response = await self._client.post(
            self.account.completions_url,
            json=payload,
        )
        return response

    @asynccontextmanager
    async def stream_chat_completion(self, payload: Dict[str, Any]):
        """Open an upstream SSE response and keep it streaming end-to-end."""
        async with self._client.stream(
            "POST",
            self.account.completions_url,
            json=payload,
        ) as response:
            yield response

    async def close(self) -> None:
        await self._client.aclose()


class GenericOpenAICatalog:
    def __init__(self):
        self._models_by_account: Dict[str, List[GenericOpenAIModel]] = {}
        self._last_refresh: Optional[float] = None

    def update_account_models(self, account_id: str, models: List[GenericOpenAIModel]) -> None:
        self._models_by_account[account_id] = [
            GenericOpenAIModel(
                **{
                    **model.__dict__,
                    "backend_id": "platform:generic-openai",
                    "account_id": account_id,
                }
            )
            for model in models
        ]
        self._last_refresh = time.time()

    def clear_account_models(self, account_id: str) -> None:
        """Remove modelos que não podem mais ser atribuídos à conta."""
        self._models_by_account.pop(account_id, None)
        self._last_refresh = time.time()

    def clear_accounts_except(self, account_ids: set[str]) -> None:
        """Descarta origens que já não existem na configuração."""
        stale_ids = set(self._models_by_account) - set(account_ids)
        for account_id in stale_ids:
            self._models_by_account.pop(account_id, None)
        if stale_ids:
            self._last_refresh = time.time()

    def get_models_for_account(self, account_id: str) -> List[GenericOpenAIModel]:
        return self._models_by_account.get(account_id, [])

    @property
    def all_models(self) -> List[GenericOpenAIModel]:
        seen = set()
        unique = []
        for account_id in sorted(self._models_by_account):
            models = self._models_by_account[account_id]
            for m in models:
                if m.id not in seen:
                    seen.add(m.id)
                    unique.append(m)
        return unique

    def get_model_accounts(self, model_id: str) -> List[str]:
        """Return all account origins for a model in deterministic order."""
        return sorted(
            account_id
            for account_id, models in self._models_by_account.items()
            if any(model.id == model_id for model in models)
        )

    @property
    def catalog_status(self) -> str:
        return "fresh" if self._models_by_account else "stale"

    @property
    def last_refresh(self) -> Optional[float]:
        return self._last_refresh


class GenericOpenAIAccountManager:
    def __init__(self, config_manager: "ConfigManager", catalog: GenericOpenAICatalog):
        self.config_manager = config_manager
        self.catalog = catalog

    def get_accounts(self) -> List[GenericOpenAIAccount]:
        raw = self.config_manager.get_generic_openai_accounts_raw()
        accounts = []
        for acc in raw:
            status = acc.get("status", "available")
            cooldown_until = acc.get("cooldown_until")
            rate_limited_until = acc.get("rate_limited_until")
            now = time.time()
            if rate_limited_until and rate_limited_until <= now:
                rate_limited_until = None
                if status == "rate_limited":
                    status = "available"
            if cooldown_until and cooldown_until <= now:
                cooldown_until = None
                if status == "cooldown":
                    status = "available"
            accounts.append(GenericOpenAIAccount(
                id=acc["id"],
                name=acc["name"],
                base_url=acc["base_url"],
                api_key=acc.get("api_key", ""),
                status=status,
                cooldown_until=cooldown_until,
                rate_limited_until=rate_limited_until,
            ))
        return accounts

    def _persist_runtime_state(self, account: GenericOpenAIAccount) -> None:
        self.config_manager.update_generic_openai_account(account.id, {
            "status": account.status,
            "cooldown_until": account.cooldown_until,
            "rate_limited_until": account.rate_limited_until,
        })

    def add_account(self, name: str, base_url: str, api_key: str) -> GenericOpenAIAccount:
        created = self.config_manager.add_generic_openai_account(name, base_url, api_key)
        return GenericOpenAIAccount(
            id=created["id"],
            name=created["name"],
            base_url=created["base_url"],
            api_key=api_key,
        )

    def remove_account(self, account_id: str) -> None:
        if not self.config_manager.remove_generic_openai_account(account_id):
            raise ValueError(f"Account {account_id} not found")
        self.catalog.clear_account_models(account_id)

    async def validate_connection(self, account: GenericOpenAIAccount) -> bool:
        provider = GenericOpenAIProvider(account)
        try:
            is_valid = await provider.health_check()
            account.status = "available" if is_valid else "error"
            self._persist_runtime_state(account)
            return is_valid
        except Exception:
            account.status = "error"
            self._persist_runtime_state(account)
            return False
        finally:
            await provider.close()

    async def refresh_catalog(self) -> None:
        accounts = self.get_accounts()
        self.catalog.clear_accounts_except({account.id for account in accounts})
        for acc in accounts:
            # Cada refresh substitui o resultado anterior, inclusive quando o
            # upstream falha, para não expor modelos potencialmente obsoletos.
            self.catalog.clear_account_models(acc.id)
            if acc.status not in ("available", "cooldown", "rate_limited"):
                continue
            provider = GenericOpenAIProvider(acc)
            try:
                models = await provider.fetch_models()
                self.catalog.update_account_models(acc.id, models)
            finally:
                await provider.close()

    def resolve_for_request(self, exclude_account_id: Optional[str] = None) -> Optional[GenericOpenAIAccount]:
        return self.resolve_for_model(None, exclude_account_id=exclude_account_id)

    def resolve_for_model(
        self,
        model_id: Optional[str],
        exclude_account_id: Optional[str] = None,
    ) -> Optional[GenericOpenAIAccount]:
        accounts = self.get_accounts()
        by_id = {account.id: account for account in accounts}
        candidates = (
            self.catalog.get_model_accounts(model_id)
            if model_id
            else []
        )
        ordered_ids = candidates + sorted(
            account.id for account in accounts if account.id not in candidates
        )
        for account_id in ordered_ids:
            acc = by_id.get(account_id)
            if acc is None:
                continue
            if acc.id == exclude_account_id:
                continue
            if acc.status == "available":
                return acc
        return None
