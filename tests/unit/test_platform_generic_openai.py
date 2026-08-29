import pytest
import time
import httpx
from platform_generic_openai import (
    GenericOpenAIAccount,
    GenericOpenAIModel,
    GenericOpenAIProvider,
    GenericOpenAICatalog,
    GenericOpenAIAccountManager,
)
from config_manager import ConfigManager

@pytest.fixture
def tmp_config_manager(tmp_path):
    return ConfigManager(str(tmp_path / "config.json"))

def test_account_dataclass():
    acc = GenericOpenAIAccount(
        id="acc-1",
        name="Local VLLM",
        base_url="http://localhost:8000/v1",
        api_key="sk-test"
    )
    assert acc.completions_url == "http://localhost:8000/v1/chat/completions"
    assert acc.models_url == "http://localhost:8000/v1/models"
    assert acc.status == "available"

@pytest.mark.asyncio
async def test_account_manager_crud(tmp_config_manager):
    catalog = GenericOpenAICatalog()
    manager = GenericOpenAIAccountManager(tmp_config_manager, catalog)

    acc = manager.add_account("Test API", "https://api.test.com", "sk-12345")
    assert acc.name == "Test API"
    assert acc.base_url == "https://api.test.com"

    accounts = manager.get_accounts()
    assert len(accounts) == 1
    assert accounts[0].id == acc.id
    assert accounts[0].api_key == "sk-12345"

    resolved = manager.resolve_for_request()
    assert resolved.id == acc.id

    manager.remove_account(acc.id)
    assert len(manager.get_accounts()) == 0

@pytest.mark.asyncio
async def test_catalog_update():
    catalog = GenericOpenAICatalog()
    assert catalog.catalog_status == "stale"
    assert len(catalog.all_models) == 0

    models = [GenericOpenAIModel(id="gpt-3.5-turbo", display_name="GPT 3.5")]
    catalog.update_account_models("acc-1", models)

    assert catalog.catalog_status == "fresh"
    assert len(catalog.all_models) == 1
    assert catalog.all_models[0].id == "gpt-3.5-turbo"
    assert "text" in catalog.all_models[0].capabilities


def test_catalog_keeps_account_origins_and_selects_deterministically(tmp_config_manager):
    catalog = GenericOpenAICatalog()
    manager = GenericOpenAIAccountManager(tmp_config_manager, catalog)
    first = manager.add_account("First", "https://one.test/v1", "sk-one")
    second = manager.add_account("Second", "https://two.test/v1", "sk-two")
    model = GenericOpenAIModel(id="shared-model", display_name="Shared")
    catalog.update_account_models(second.id, [model])
    catalog.update_account_models(first.id, [model])

    account_ids = sorted([first.id, second.id])
    assert catalog.get_model_accounts("shared-model") == account_ids
    assert catalog.all_models[0].account_id == account_ids[0]
    assert manager.resolve_for_model("shared-model").id == account_ids[0]
    assert manager.resolve_for_model(
        "shared-model", exclude_account_id=account_ids[0]
    ).id == account_ids[1]


def test_remove_account_clears_catalog_models(tmp_config_manager):
    catalog = GenericOpenAICatalog()
    manager = GenericOpenAIAccountManager(tmp_config_manager, catalog)
    account = manager.add_account("To remove", "https://one.test/v1", "sk-one")
    catalog.update_account_models(
        account.id, [GenericOpenAIModel(id="stale-model", display_name="Stale")]
    )

    manager.remove_account(account.id)

    assert catalog.all_models == []
    assert manager.resolve_for_model("stale-model") is None


@pytest.mark.asyncio
async def test_refresh_clears_failed_and_removed_account_models(tmp_config_manager, monkeypatch):
    catalog = GenericOpenAICatalog()
    manager = GenericOpenAIAccountManager(tmp_config_manager, catalog)
    account = manager.add_account("Unavailable", "https://one.test/v1", "sk-one")
    catalog.update_account_models(
        account.id, [GenericOpenAIModel(id="stale-model", display_name="Stale")]
    )

    async def failed_fetch(self):
        return []

    monkeypatch.setattr("platform_generic_openai.GenericOpenAIProvider.fetch_models", failed_fetch)
    await manager.refresh_catalog()

    assert catalog.all_models == []


@pytest.mark.asyncio
async def test_provider_forwards_payload_and_streams():
    account = GenericOpenAIAccount(
        id="acc-1", name="Upstream", base_url="https://upstream.test/v1",
        api_key="sk-test",
    )
    seen = []

    def handler(request):
        seen.append(request)
        if request.content and b'"stream":true' not in request.content:
            return httpx.Response(
                200, headers={"content-type": "application/json"},
                json={"id": "chat-1", "model": "gpt-test"},
            )
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            content=b"data: ok\n\n",
        )

    provider = GenericOpenAIProvider(account)
    await provider._client.aclose()
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    payload = {
        "model": "gpt-test",
        "messages": [{"role": "user", "content": "oi"}],
        "temperature": 0.2,
    }
    response = await provider.chat_completion(payload)
    assert response.status_code == 200
    assert response.json()["model"] == "gpt-test"
    assert seen[0].url.path == "/v1/chat/completions"
    assert b"temperature" in seen[0].content

    async with provider.stream_chat_completion({**payload, "stream": True}) as response:
        assert b"data: ok" in await response.aread()
    await provider.close()
