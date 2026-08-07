"""Unit tests for Ollama Cloud model catalog."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from platform_ollama_cloud import (
    OllamaCloudAccount,
    OllamaCloudCatalog,
    OllamaCloudModel,
    _build_fixed_models,
    _parse_model_item,
)


class TestBuildFixedModels:
    """Verify the built-in model list."""

    def test_returns_list(self):
        models = _build_fixed_models()
        assert isinstance(models, list)

    def test_contains_at_least_10_models(self):
        models = _build_fixed_models()
        assert len(models) >= 10

    def test_models_have_required_fields(self):
        models = _build_fixed_models()
        for m in models:
            assert isinstance(m.id, str) and len(m.id) > 0
            assert isinstance(m.display_name, str) and len(m.display_name) > 0
            assert isinstance(m.capabilities, frozenset)

    def test_capabilities_are_frozenset(self):
        models = _build_fixed_models()
        for m in models:
            assert type(m.capabilities) is frozenset

    def test_contains_expected_model_ids(self):
        models = _build_fixed_models()
        ids = {m.id for m in models}
        expected = {"llama3.1-8b", "qwen2.5-72b", "mistral-large", "gpt-oss-20b"}
        assert expected.issubset(ids)


class TestParseModelItem:
    """Tolerant parsing of a model dict."""

    def test_parses_full_item(self):
        item: Dict[str, Any] = {
            "id": "gpt-oss-20b",
            "display_name": "GPT OSS 20B",
            "context_length": 131072,
            "output_token_limit": 32768,
            "capabilities": ["text", "tools", "vision"],
        }
        model = _parse_model_item(item)
        assert model is not None
        assert model.id == "gpt-oss-20b"
        assert model.context_length == 131072
        assert model.output_token_limit == 32768
        assert model.capabilities == frozenset({"text", "tools", "vision"})

    def test_parses_minimal_item(self):
        item: Dict[str, Any] = {"id": "tiny-model"}
        model = _parse_model_item(item)
        assert model is not None
        assert model.id == "tiny-model"
        assert model.display_name == "tiny-model"
        assert model.context_length is None
        assert model.output_token_limit is None
        assert model.capabilities == frozenset()

    def test_returns_none_for_missing_id(self):
        model = _parse_model_item({})
        assert model is None

    def test_returns_none_for_empty_id(self):
        model = _parse_model_item({"id": ""})
        assert model is None

    def test_tolerates_string_context_length(self):
        item: Dict[str, Any] = {"id": "x", "context_length": "not_a_number"}
        model = _parse_model_item(item)
        assert model is not None
        assert model.context_length is None

    def test_tolerates_none_context_length(self):
        item: Dict[str, Any] = {"id": "x", "context_length": None}
        model = _parse_model_item(item)
        assert model is not None
        assert model.context_length is None

    def test_uses_name_when_display_name_missing(self):
        item: Dict[str, Any] = {"id": "x", "name": "Name Fallback"}
        model = _parse_model_item(item)
        assert model.display_name == "Name Fallback"

    def test_capabilities_as_non_list(self):
        item: Dict[str, Any] = {"id": "x", "capabilities": "text"}
        model = _parse_model_item(item)
        assert model.capabilities == frozenset()


class TestOllamaCloudCatalogInit:
    """Catalog construction."""

    def test_default_uses_fixed_models(self):
        catalog = OllamaCloudCatalog()
        assert len(catalog.all_models) >= 10

    def test_custom_models(self):
        custom = [OllamaCloudModel(id="custom-1", display_name="Custom 1")]
        catalog = OllamaCloudCatalog(models=custom)
        assert len(catalog.all_models) == 1

    def test_empty_list_uses_fixed_models(self):
        catalog = OllamaCloudCatalog(models=[])
        # An empty explicit list is kept as-is (not auto-replaced).
        assert len(catalog.all_models) == 0


class TestFindModelById:
    """find_model_by_id behaviour."""

    def test_finds_existing_model(self):
        catalog = OllamaCloudCatalog()
        model = catalog.find_model_by_id("llama3.1-8b")
        assert model is not None
        assert model.id == "llama3.1-8b"

    def test_returns_none_for_unknown_id(self):
        catalog = OllamaCloudCatalog()
        model = catalog.find_model_by_id("nonexistent-model-xyz")
        assert model is None

    def test_returns_none_for_empty_id(self):
        catalog = OllamaCloudCatalog()
        model = catalog.find_model_by_id("")
        assert model is None

    def test_case_sensitive_search(self):
        catalog = OllamaCloudCatalog()
        model_lower = catalog.find_model_by_id("llama3.1-8b")
        model_upper = catalog.find_model_by_id("LLAMA3.1-8B")
        assert model_lower is not None
        assert model_upper is None


class TestGetModelsForAccount:
    """get_models_for_account behaviour."""

    def test_returns_all_models(self):
        catalog = OllamaCloudCatalog()
        account = MagicMock()  # any object
        models = catalog.get_models_for_account(account)
        assert len(models) >= 10

    def test_returns_copy_not_reference(self):
        catalog = OllamaCloudCatalog()
        models1 = catalog.get_models_for_account(MagicMock())
        models2 = catalog.get_models_for_account(MagicMock())
        assert models1 is not models2


class TestRefresh:
    """refresh lifecycle."""

    @pytest.mark.asyncio
    async def test_refresh_sets_last_refresh_timestamp(self):
        catalog = OllamaCloudCatalog()
        assert catalog.last_refresh is None
        await catalog.refresh()
        assert catalog.last_refresh is not None

    @pytest.mark.asyncio
    async def test_refresh_records_error_on_failure(self):
        catalog = OllamaCloudCatalog()
        with patch.object(
            catalog,
            "_fetch_remote_models",
            new_callable=AsyncMock,
            side_effect=Exception("network error"),
        ):
            await catalog.refresh()
        assert catalog.refresh_error is not None
        assert "network error" in catalog.refresh_error

    @pytest.mark.asyncio
    async def test_refresh_is_async(self):
        catalog = OllamaCloudCatalog()
        with patch.object(catalog, "_fetch_remote_models", new_callable=AsyncMock) as mock:
            await catalog.refresh()
            mock.assert_called_once()


class TestAllModels:
    """all_models property."""

    def test_returns_copy(self):
        catalog = OllamaCloudCatalog()
        models = catalog.all_models
        assert isinstance(models, list)

    def test_immutable_reference(self):
        catalog = OllamaCloudCatalog()
        models1 = catalog.all_models
        models2 = catalog.all_models
        assert models1 is not models2


class TestFromFile:
    """from_file class method."""

    def test_from_file_with_valid_json(self):
        models_data = [
            {"id": "file-model-1", "display_name": "File Model 1", "context_length": 4096},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(models_data, f)
            f.flush()
            tmp_path = f.name
        try:
            catalog = OllamaCloudCatalog.from_file(tmp_path)
            assert len(catalog.all_models) == 1
            assert catalog.all_models[0].id == "file-model-1"
            assert catalog.all_models[0].context_length == 4096
        finally:
            os.unlink(tmp_path)

    def test_from_file_missing_file_uses_fixed(self):
        catalog = OllamaCloudCatalog.from_file("/nonexistent/path.json")
        assert len(catalog.all_models) >= 10

    def test_from_file_empty_array_uses_fixed(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([], f)
            f.flush()
            tmp_path = f.name
        try:
            catalog = OllamaCloudCatalog.from_file(tmp_path)
            assert len(catalog.all_models) >= 10
        finally:
            os.unlink(tmp_path)


class TestMergeRemote:
    """merge_remote class method."""

    def test_appends_new_models(self):
        existing = [OllamaCloudModel(id="fixed-1", display_name="Fixed")]
        remote_data = [{"id": "remote-1", "display_name": "Remote"}]
        merged = OllamaCloudCatalog.merge_remote(existing, remote_data)
        ids = {m.id for m in merged}
        assert "fixed-1" in ids
        assert "remote-1" in ids
        assert len(merged) == 2

    def test_does_not_duplicate_existing(self):
        existing = [OllamaCloudModel(id="shared", display_name="Shared")]
        remote_data = [{"id": "shared", "display_name": "Shared Updated"}]
        merged = OllamaCloudCatalog.merge_remote(existing, remote_data)
        ids = [m.id for m in merged]
        assert ids.count("shared") == 1

    def test_empty_remote_keeps_existing(self):
        existing = [OllamaCloudModel(id="x", display_name="X")]
        merged = OllamaCloudCatalog.merge_remote(existing, [])
        assert len(merged) == 1


class TestLastRefreshProperty:
    def test_none_before_refresh(self):
        catalog = OllamaCloudCatalog()
        assert catalog.last_refresh is None

    def test_set_after_refresh(self):
        catalog = OllamaCloudCatalog()
        assert catalog.last_refresh is None

    def test_refresh_error_is_none_initially(self):
        catalog = OllamaCloudCatalog()
        assert catalog.refresh_error is None


class TestTolerantParsingEdgeCases:
    """Additional edge cases for tolerant parsing."""

    def test_whitespace_in_id_is_trimmed(self):
        item: Dict[str, Any] = {"id": "  x  "}
        model = _parse_model_item(item)
        assert model is not None
        assert model.id == "x"

    def test_output_token_limit_from_non_int(self):
        item: Dict[str, Any] = {"id": "x", "output_token_limit": "abc"}
        model = _parse_model_item(item)
        assert model is not None
        assert model.output_token_limit is None

    def test_empty_capabilities_list(self):
        item: Dict[str, Any] = {"id": "x", "capabilities": []}
        model = _parse_model_item(item)
        assert model.capabilities == frozenset()

    def test_capabilities_with_empty_strings(self):
        item: Dict[str, Any] = {"id": "x", "capabilities": ["text", "", "tools"]}
        model = _parse_model_item(item)
        assert model.capabilities == frozenset({"text", "tools"})


# ---------------------------------------------------------------------------
# catalog_status
# ---------------------------------------------------------------------------


class TestCatalogStatus:
    """Tests for the catalog_status field and lifecycle."""

    def test_fresh_on_init_with_models(self):
        catalog = OllamaCloudCatalog()
        assert catalog.catalog_status == OllamaCloudCatalog.STATUS_FRESH

    def test_stale_on_init_without_models(self):
        catalog = OllamaCloudCatalog(models=[])
        assert catalog.catalog_status == OllamaCloudCatalog.STATUS_STALE

    @pytest.mark.asyncio
    async def test_error_on_refresh_failure(self):
        catalog = OllamaCloudCatalog()
        with patch.object(
            catalog,
            "_fetch_remote_models",
            new_callable=AsyncMock,
            side_effect=Exception("network error"),
        ):
            assert catalog.catalog_status == OllamaCloudCatalog.STATUS_FRESH
            await catalog.refresh()
        assert catalog.catalog_status == OllamaCloudCatalog.STATUS_ERROR

    @pytest.mark.asyncio
    async def test_fresh_on_refresh_success(self):
        catalog = OllamaCloudCatalog()
        with patch.object(catalog, "_fetch_remote_models", new_callable=AsyncMock):
            await catalog.refresh()
        assert catalog.catalog_status == OllamaCloudCatalog.STATUS_FRESH

    @pytest.mark.asyncio
    async def test_existing_models_preserved_on_error(self):
        """Catalog must NOT break on refresh failure — stale, not broken."""
        catalog = OllamaCloudCatalog()
        initial_models = len(catalog.all_models)
        with patch.object(
            catalog,
            "_fetch_remote_models",
            new_callable=AsyncMock,
            side_effect=Exception("network error"),
        ):
            await catalog.refresh()
        assert catalog.catalog_status == OllamaCloudCatalog.STATUS_ERROR
        assert len(catalog.all_models) == initial_models
        assert len(catalog.all_models) > 0


# ---------------------------------------------------------------------------
# _schedule_refresh / asyncio.create_task
# ---------------------------------------------------------------------------


class TestScheduleRefresh:
    """Background scheduling via asyncio.create_task."""

    def test_no_task_without_account(self):
        catalog = OllamaCloudCatalog()
        assert catalog._refresh_task is None

    @pytest.mark.asyncio
    async def test_task_created_when_account_provided(self):
        account = OllamaCloudAccount(id="acc-1", api_key="sk-test")
        catalog = OllamaCloudCatalog(account=account)
        assert catalog._refresh_task is not None
        assert not catalog._refresh_task.done()

    @pytest.mark.asyncio
    async def test_task_is_asyncio_task_not_thread(self):
        import asyncio

        account = OllamaCloudAccount(id="acc-1", api_key="sk-test")
        catalog = OllamaCloudCatalog(account=account)
        assert isinstance(catalog._refresh_task, asyncio.Task)

    @pytest.mark.asyncio
    async def test_refresh_from_endpoint_delegates_to_refresh(self):
        catalog = OllamaCloudCatalog()
        with patch.object(catalog, "refresh", new_callable=AsyncMock) as mock:
            await catalog._refresh_from_endpoint()
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_custom_refresh_interval(self):
        account = OllamaCloudAccount(id="acc-1", api_key="sk-test")
        catalog = OllamaCloudCatalog(account=account, refresh_interval=300)
        assert catalog.refresh_interval == 300

    @pytest.mark.asyncio
    async def test_default_refresh_interval_is_24h(self):
        account = OllamaCloudAccount(id="acc-1", api_key="sk-test")
        catalog = OllamaCloudCatalog(account=account)
        assert catalog.refresh_interval == OllamaCloudCatalog.DEFAULT_REFRESH_INTERVAL

    @pytest.mark.asyncio
    async def test_refresh_interval_setter_cancels_and_reschedules(self):
        account = OllamaCloudAccount(id="acc-1", api_key="sk-test")
        catalog = OllamaCloudCatalog(account=account, refresh_interval=300)
        old_task = catalog._refresh_task
        assert old_task is not None

        catalog.refresh_interval = 600
        assert catalog.refresh_interval == 600
        assert catalog._refresh_task is not old_task
        assert catalog._refresh_task is not None

    def test_catalog_status_constant_values(self):
        assert OllamaCloudCatalog.STATUS_FRESH == "fresh"
        assert OllamaCloudCatalog.STATUS_STALE == "stale"
        assert OllamaCloudCatalog.STATUS_ERROR == "error"


# ---------------------------------------------------------------------------
# Logging verification
# ---------------------------------------------------------------------------


class TestCatalogRefreshLogging:
    """Verify catalog_refresh log messages."""

    @pytest.mark.asyncio
    async def test_log_success_on_refresh(self, caplog):
        catalog = OllamaCloudCatalog()
        with patch.object(catalog, "_fetch_remote_models", new_callable=AsyncMock):
            await catalog.refresh()
        assert "catalog_refresh" in caplog.text
        assert "success=True" in caplog.text
        assert f"models_count={len(catalog.all_models)}" in caplog.text

    @pytest.mark.asyncio
    async def test_log_failure_on_refresh(self, caplog):
        catalog = OllamaCloudCatalog()
        with patch.object(
            catalog,
            "_fetch_remote_models",
            new_callable=AsyncMock,
            side_effect=Exception("network error"),
        ):
            await catalog.refresh()
        assert "catalog_refresh" in caplog.text
        assert "success=False" in caplog.text


# ---------------------------------------------------------------------------
# OllamaCloudAccount
# ---------------------------------------------------------------------------


class TestOllamaCloudAccount:
    """Account dataclass defaults and fields."""

    def test_default_label(self):
        account = OllamaCloudAccount(id="a1", api_key="sk-1")
        assert account.label == ""

    def test_default_status(self):
        account = OllamaCloudAccount(id="a1", api_key="sk-1")
        assert account.status == "available"

    def test_default_cooldown_until(self):
        account = OllamaCloudAccount(id="a1", api_key="sk-1")
        assert account.cooldown_until is None

    def test_custom_values(self):
        account = OllamaCloudAccount(
            id="a1",
            api_key="sk-1",
            label="Test Account",
            status="cooldown",
            cooldown_until=1000.0,
        )
        assert account.label == "Test Account"
        assert account.status == "cooldown"
        assert account.cooldown_until == 1000.0
