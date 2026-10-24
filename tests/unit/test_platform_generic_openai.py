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
from platform_manager import PlatformIntegrationManager

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


def test_each_account_is_exposed_as_a_platform_backend(tmp_config_manager):
    catalog = GenericOpenAICatalog()
    account_manager = GenericOpenAIAccountManager(tmp_config_manager, catalog)
    account = account_manager.add_account(
        "Minha API", "https://example.test/v1", "sk-test"
    )
    platforms = PlatformIntegrationManager(
        tmp_config_manager,
        executable_resolver=lambda _command: None,
        generic_openai_account_manager=account_manager,
        generic_openai_catalog=catalog,
    )

    backend = platforms.get(f"platform:generic-openai:{account.id}")

    assert backend is not None
    assert backend["display_name"] == "Minha API"
    assert backend["account_id"] == account.id
    assert backend["base_url"] == "https://example.test/v1"
    assert backend["provider"] == "generic-openai"


def test_running_account_has_stable_virtual_port_for_proxy(tmp_config_manager):
    catalog = GenericOpenAICatalog()
    account_manager = GenericOpenAIAccountManager(tmp_config_manager, catalog)
    account = account_manager.add_account(
        "Minha API", "https://example.test/v1", "sk-test"
    )
    platforms = PlatformIntegrationManager(
        tmp_config_manager,
        executable_resolver=lambda _command: None,
        generic_openai_account_manager=account_manager,
        generic_openai_catalog=catalog,
    )
    backend_id = f"platform:generic-openai:{account.id}"
    platforms._runtime[backend_id] = {"active": True, "status": "running"}

    first = next(i for i in platforms.active_instances() if i["backend_id"] == backend_id)
    second = next(i for i in platforms.active_instances() if i["backend_id"] == backend_id)

    assert first["port"] < 0
    assert first["port"] == second["port"]
    assert first["backend_type"] == "platform"

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


def test_anthropic_account_urls_and_headers():
    acc_v1 = GenericOpenAIAccount(
        id="acc-ant-1",
        name="Claude API",
        base_url="https://api.anthropic.com/v1",
        api_key="sk-ant-test-key",
        api_type="anthropic",
    )
    assert acc_v1.completions_url == "https://api.anthropic.com/v1/messages"
    assert acc_v1.models_url == "https://api.anthropic.com/v1/models"

    acc_bare = GenericOpenAIAccount(
        id="acc-ant-2",
        name="Claude API Bare",
        base_url="https://api.anthropic.com",
        api_key="sk-ant-test-key",
        api_type="anthropic",
    )
    assert acc_bare.completions_url == "https://api.anthropic.com/v1/messages"
    assert acc_bare.models_url == "https://api.anthropic.com/v1/models"

    provider = GenericOpenAIProvider(acc_v1)
    assert provider._client.headers.get("x-api-key") == "sk-ant-test-key"
    assert provider._client.headers.get("anthropic-version") == "2023-06-01"
    assert "authorization" not in provider._client.headers


@pytest.mark.asyncio
async def test_anthropic_provider_health_and_fetch_models():
    account = GenericOpenAIAccount(
        id="acc-ant-3",
        name="Claude",
        base_url="https://api.anthropic.com/v1",
        api_key="sk-ant-key",
        api_type="anthropic",
    )

    # 1. Health check & model list when /models is supported
    def handler(request):
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                json={
                    "data": [
                        {
                            "id": "claude-3-5-sonnet-20241022",
                            "display_name": "Claude 3.5 Sonnet",
                            "type": "model",
                        }
                    ]
                },
            )
        return httpx.Response(404)

    provider = GenericOpenAIProvider(account)
    await provider._client.aclose()
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        headers={"x-api-key": account.api_key, "anthropic-version": "2023-06-01"},
    )

    assert await provider.health_check() is True
    models = await provider.fetch_models()
    assert len(models) == 1
    assert models[0].id == "claude-3-5-sonnet-20241022"
    assert models[0].display_name == "Claude 3.5 Sonnet"
    assert models[0].api_type == "anthropic"
    await provider.close()

    # 2. Fallback when /models returns 404 but endpoint is alive
    def handler_fallback(request):
        if request.url.path == "/v1/models":
            return httpx.Response(404)
        if request.url.path == "/v1/messages":
            return httpx.Response(400, json={"error": "messages required"})
        return httpx.Response(500)

    provider2 = GenericOpenAIProvider(account)
    await provider2._client.aclose()
    provider2._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler_fallback),
        headers={"x-api-key": account.api_key, "anthropic-version": "2023-06-01"},
    )

    assert await provider2.health_check() is True
    fallback_models = await provider2.fetch_models()
    assert len(fallback_models) >= 1
    assert any("claude" in m.id for m in fallback_models)
    assert fallback_models[0].api_type == "anthropic"
    await provider2.close()


def test_anthropic_account_exposed_in_platform_manager(tmp_config_manager):
    catalog = GenericOpenAICatalog()
    account_manager = GenericOpenAIAccountManager(tmp_config_manager, catalog)
    account = account_manager.add_account(
        "Claude API", "https://api.anthropic.com/v1", "sk-ant-test", api_type="anthropic"
    )
    platforms = PlatformIntegrationManager(
        tmp_config_manager,
        executable_resolver=lambda _command: None,
        generic_openai_account_manager=account_manager,
        generic_openai_catalog=catalog,
    )

    backend = platforms.get(f"platform:generic-openai:{account.id}")
    assert backend is not None
    assert backend["display_name"] == "Claude API"
    assert backend["api_type"] == "anthropic"
    assert backend["accounts"][0]["api_type"] == "anthropic"


def test_generic_account_api_type_routes(tmp_config_manager, monkeypatch):
    import llama_manager
    from fastapi.testclient import TestClient

    catalog = GenericOpenAICatalog()
    account_manager = GenericOpenAIAccountManager(tmp_config_manager, catalog)

    monkeypatch.setattr(llama_manager, "config_manager", tmp_config_manager)
    monkeypatch.setattr(llama_manager, "generic_openai_manager", account_manager)
    monkeypatch.setattr(llama_manager, "generic_openai_catalog", catalog)

    llama_manager.app.dependency_overrides[llama_manager.require_auth] = lambda: True
    try:
        client = TestClient(llama_manager.app)

        # 1. Create Anthropic account
        res = client.post(
            "/platforms/generic-openai/accounts",
            json={
                "name": "Anthropic Official",
                "base_url": "https://api.anthropic.com/v1",
                "api_key": "sk-ant-123456",
                "api_type": "anthropic",
            },
        )
        assert res.status_code == 201
        data = res.json()
        assert data["name"] == "Anthropic Official"
        assert data["api_type"] == "anthropic"
        account_id = data["id"]

        # 2. List accounts
        list_res = client.get("/platforms/generic-openai/accounts")
        assert list_res.status_code == 200
        accounts = list_res.json()["accounts"]
        target = next(a for a in accounts if a["id"] == account_id)
        assert target["api_type"] == "anthropic"

        # 3. Update account to openai
        patch_res = client.patch(
            f"/platforms/generic-openai/accounts/{account_id}",
            json={"api_type": "openai"},
        )
        assert patch_res.status_code == 200
        assert patch_res.json()["api_type"] == "openai"

        # 4. Invalid api_type
        bad_res = client.post(
            "/platforms/generic-openai/accounts",
            json={
                "name": "Invalid",
                "base_url": "https://api.test/v1",
                "api_key": "sk-123",
                "api_type": "other",
            },
        )
        assert bad_res.status_code == 422 or bad_res.status_code == 400
    finally:
        llama_manager.app.dependency_overrides.pop(llama_manager.require_auth, None)
