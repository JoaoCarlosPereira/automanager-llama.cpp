import os

import pytest

from platform_manager import (
    CLIProxySidecarError,
    CLIProxySidecarManager,
    PlatformIntegrationError,
    PlatformIntegrationManager,
    clear_platform_listing_registry,
    default_executable_resolver,
    filter_models_for_provider,
    lookup_platform_bare_id,
    merge_platform_model_metadata,
    platform_client_facing_model,
    platform_model_listing_entry,
    platform_model_listing_id,
    platform_provider_for_listing,
    register_platform_model_listings,
    register_platform_bare_model,
    resolve_platform_listing_model,
)


@pytest.fixture(autouse=True)
def reset_listing_registry():
    clear_platform_listing_registry()
    yield
    clear_platform_listing_registry()


def resolver_for(mapping):
    def _resolve(command):
        return mapping.get(command)

    return _resolve


def entry(catalog, backend_id):
    return next(item for item in catalog if item["backend_id"] == backend_id)


class TestFilterModelsForProvider:
    SAMPLE = [
        {"id": "gpt-5.4", "owned_by": "openai"},
        {"id": "gemini-3.1-pro-low", "owned_by": "antigravity"},
        {"id": "claude-sonnet-4-6", "owned_by": "claude"},
    ]

    def test_codex_keeps_openai_only(self):
        result = filter_models_for_provider(self.SAMPLE, "codex")
        assert [m["id"] for m in result] == ["gpt-5.4"]

    def test_antigravity_keeps_antigravity_only(self):
        result = filter_models_for_provider(self.SAMPLE, "antigravity")
        assert [m["id"] for m in result] == ["gemini-3.1-pro-low"]

    def test_claude_keeps_claude_only(self):
        result = filter_models_for_provider(self.SAMPLE, "claude")
        assert [m["id"] for m in result] == ["claude-sonnet-4-6"]

    def test_unknown_provider_returns_all(self):
        result = filter_models_for_provider(self.SAMPLE, "unknown")
        assert len(result) == 3

    def test_strict_filter_does_not_claim_another_providers_catalog(self):
        result = filter_models_for_provider(
            [{"id": "gemma4:31b", "owned_by": "ollama"}],
            "codex",
            strict=True,
        )
        assert result == []


class TestPlatformModelListing:
    def test_catalog_context_is_copied_to_sidecar_model(self):
        model = merge_platform_model_metadata(
            {"id": "gemini-3.1-pro-low", "owned_by": "antigravity"},
            "antigravity",
            {
                "antigravity": {
                    "gemini-3.1-pro-low": {
                        "id": "gemini-3.1-pro-low",
                        "inputTokenLimit": 1_048_576,
                        "outputTokenLimit": 65_536,
                    }
                }
            },
        )
        assert model["context_length"] == 1_048_576
        assert model["max_completion_tokens"] == 65_536

    def test_listing_id_uses_opaque_slug_for_blocked_names(self):
        assert (
            platform_model_listing_id("gemini-3.1-pro-low", "antigravity")
            == "antigravity-31prolow.gguf"
        )

    def test_listing_id_uses_custom_suffix_for_safe_names(self):
        assert platform_model_listing_id("some-model", "cloud") == "some-model-custom.gguf"

    def test_listing_entry_includes_metadata(self):
        entry = platform_model_listing_entry(
            {
                "id": "gemini-3.1-pro-low",
                "owned_by": "antigravity",
                "context_length": 1_048_576,
            },
            provider="antigravity",
        )
        assert entry["id"] == "antigravity-31prolow.gguf"
        assert entry["owned_by"] == "llamacpp"
        assert entry["meta"]["root_model"] == "gemini-3.1-pro-low"
        assert entry["meta"]["n_ctx"] == 1_048_576

    def test_resolve_uses_registry(self):
        register_platform_model_listings("gemini-3.1-pro-low", "antigravity")
        assert resolve_platform_listing_model("antigravity-31prolow.gguf") == "gemini-3.1-pro-low"

    def test_bare_model_id_can_be_bound_to_ollama_cloud(self):
        register_platform_bare_model("gemma4:31b", "ollama-cloud")
        assert platform_provider_for_listing("gemma4:31b") == "ollama-cloud"
        assert resolve_platform_listing_model("gemma4:31b") == "gemma4:31b"

    def test_ollama_bare_model_cannot_be_stolen_by_shared_cli_catalog(self):
        register_platform_bare_model("gemma4:31b", "ollama-cloud")
        register_platform_bare_model("gemma4:31b", "codex")
        register_platform_bare_model("gemma4:31b", "antigravity")
        assert platform_provider_for_listing("gemma4:31b") == "ollama-cloud"

    def test_resolve_proagent_listing(self):
        register_platform_model_listings("gemini-pro-agent", "antigravity")
        assert resolve_platform_listing_model("antigravity-proagent.gguf") == "gemini-pro-agent"

    def test_resolve_without_registry_keeps_listing_id(self):
        assert resolve_platform_listing_model("antigravity-proagent.gguf") == "antigravity-proagent.gguf"

    def test_resolve_openai_prefix_legacy_codex_listing(self):
        register_platform_model_listings("gpt-5.4-mini", "codex")
        assert lookup_platform_bare_id("openai-54mini.gguf") == "gpt-5.4-mini"
        assert resolve_platform_listing_model("openai-54mini.gguf") == "gpt-5.4-mini"

    def test_listing_provider_prefix_maps_openai_to_codex(self):
        from platform_manager import platform_provider_for_listing

        assert platform_provider_for_listing("openai-54mini.gguf") == "codex"
        assert platform_provider_for_listing("codex-54mini.gguf") == "codex"
        assert platform_provider_for_listing("antigravity-31prolow.gguf") == "antigravity"

    def test_resolve_skips_local_gguf_ids(self):
        local_ids = {"Qwen3.6-35B.gguf"}
        assert resolve_platform_listing_model("Qwen3.6-35B.gguf", local_ids) == "Qwen3.6-35B.gguf"

    def test_client_facing_model_uses_opaque_listing(self):
        register_platform_model_listings("gemini-3.1-pro-low", "antigravity")
        assert platform_client_facing_model("gemini-3.1-pro-low") == "antigravity-31prolow.gguf"

    def test_client_facing_model_preserves_alias(self):
        aliases = {"gpt-4o": "gemini-3.1-pro-low"}
        assert platform_client_facing_model("gpt-4o", aliases=aliases) == "gpt-4o"

    def test_client_facing_model_preserves_virtual_listing_id(self):
        register_platform_model_listings("gemini-3.1-pro-low", "antigravity")
        assert platform_client_facing_model("antigravity-31prolow.gguf") == "antigravity-31prolow.gguf"


class TestDefaultExecutableResolver:
    def test_finds_codex_in_user_local_bin(self, tmp_path, monkeypatch):
        home = tmp_path / "user"
        local_bin = home / ".local" / "bin"
        local_bin.mkdir(parents=True)
        codex = local_bin / "codex"
        codex.write_text("#!/bin/sh\necho codex\n", encoding="utf-8")
        codex.chmod(0o755)

        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr(
            "platform_manager._user_home_dirs",
            lambda: [str(home)],
        )
        monkeypatch.setattr("platform_manager.shutil.which", lambda _name: None)

        assert default_executable_resolver("codex") == str(codex)

    def test_finds_agy_in_user_local_bin(self, tmp_path, monkeypatch):
        home = tmp_path / "user"
        local_bin = home / ".local" / "bin"
        local_bin.mkdir(parents=True)
        agy = local_bin / "agy"
        agy.write_text("#!/bin/sh\necho agy\n", encoding="utf-8")
        agy.chmod(0o755)

        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr(
            "platform_manager._user_home_dirs",
            lambda: [str(home)],
        )
        monkeypatch.setattr("platform_manager.shutil.which", lambda _name: None)

        assert default_executable_resolver("agy") == str(agy)

    def test_finds_codex_standalone_package_path(self, tmp_path, monkeypatch):
        home = tmp_path / "user"
        codex = (
            home
            / ".codex"
            / "packages"
            / "standalone"
            / "current"
            / "bin"
            / "codex"
        )
        codex.parent.mkdir(parents=True)
        codex.write_text("#!/bin/sh\necho codex\n", encoding="utf-8")
        codex.chmod(0o755)

        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr(
            "platform_manager._user_home_dirs",
            lambda: [str(home)],
        )
        monkeypatch.setattr("platform_manager.shutil.which", lambda _name: None)

        assert default_executable_resolver("codex") == str(codex)

    def test_detects_antigravity_via_agy_candidate(self, tmp_config_manager, tmp_path):
        home = tmp_path / "user"
        local_bin = home / ".local" / "bin"
        local_bin.mkdir(parents=True)
        agy = local_bin / "agy"
        agy.write_text("#!/bin/sh\necho agy\n", encoding="utf-8")
        agy.chmod(0o755)

        manager = PlatformIntegrationManager(
            tmp_config_manager,
            executable_resolver=lambda command: (
                str(agy) if command == "agy" else None
            ),
        )

        antigravity = entry(manager.catalog(), "platform:google-antigravity")

        assert antigravity["detected"] is True
        assert antigravity["executable_command"] == "agy"
        assert antigravity["executable_path"] == str(agy)


class FakeProcess:
    def __init__(self):
        self.signals = []
        self.killed = False
        self._returncode = None

    def poll(self):
        return self._returncode

    def send_signal(self, sig):
        self.signals.append(sig)
        self._returncode = 0

    def wait(self, timeout=None):
        self._returncode = 0
        return 0

    def kill(self):
        self.killed = True
        self._returncode = -9


class FakePopenFactory:
    def __init__(self):
        self.calls = []
        self.processes = []

    def __call__(self, cmd, **kwargs):
        proc = FakeProcess()
        self.calls.append((cmd, kwargs))
        self.processes.append(proc)
        return proc


class TestPlatformIntegrationManager:
    def test_detects_codex_when_executable_resolves(self, tmp_config_manager):
        manager = PlatformIntegrationManager(
            tmp_config_manager,
            executable_resolver=resolver_for({
                "codex": "C:/Tools/Codex/codex.exe",
                "CLIProxyAPI": "D:/bin/CLIProxyAPI.exe",
            }),
        )

        codex = entry(manager.catalog(), "platform:codex")

        assert codex["detected"] is True
        assert codex["status"] == "detected"
        assert codex["executable_path"] == "C:/Tools/Codex/codex.exe"
        assert codex["cliproxy_detected"] is True

    def test_marks_claude_code_missing_with_reason(self, tmp_config_manager):
        manager = PlatformIntegrationManager(
            tmp_config_manager,
            executable_resolver=resolver_for({"CLIProxyAPI": "D:/bin/CLIProxyAPI.exe"}),
        )

        claude = entry(manager.catalog(), "platform:claude-code")

        assert claude["detected"] is False
        assert claude["status"] == "missing"
        assert claude["reason"] == "Claude Code executable not found"

    def test_catalog_includes_antigravity_when_missing(self, tmp_config_manager):
        manager = PlatformIntegrationManager(
            tmp_config_manager,
            executable_resolver=resolver_for({}),
        )

        antigravity = entry(manager.catalog(), "platform:google-antigravity")

        assert antigravity["provider"] == "antigravity"
        assert antigravity["status"] == "missing"
        assert antigravity["reason"] == "Google Antigravity executable not found"

    def test_detected_platform_is_not_ready_when_cliproxy_is_missing(
        self, tmp_config_manager
    ):
        manager = PlatformIntegrationManager(
            tmp_config_manager,
            executable_resolver=resolver_for({"codex": "C:/Tools/Codex/codex.exe"}),
        )

        codex = entry(manager.catalog(), "platform:codex")

        assert codex["detected"] is True
        assert codex["status"] == "not_ready"
        assert codex["reason"] == "CLIProxyAPI executable not found"

    def test_preferences_merge_without_changing_detection_status(
        self, tmp_config_manager
    ):
        tmp_config_manager.update_platform_settings(
            "platform:codex",
            {"proxy_eligible": True, "max_parallel_requests": 2, "auto_start": True},
        )
        manager = PlatformIntegrationManager(
            tmp_config_manager,
            executable_resolver=resolver_for({
                "codex": "C:/Tools/Codex/codex.exe",
                "CLIProxyAPI": "D:/bin/CLIProxyAPI.exe",
            }),
        )

        codex = entry(manager.catalog(), "platform:codex")

        assert codex["status"] == "detected"
        assert codex["proxy_eligible"] is True
        assert codex["max_parallel_requests"] == 2
        assert codex["auto_start"] is True

    def test_detection_is_startup_only_for_manager_instance(
        self, tmp_config_manager
    ):
        commands = {}
        manager = PlatformIntegrationManager(
            tmp_config_manager,
            executable_resolver=resolver_for(commands),
        )
        commands["codex"] = "C:/Tools/Codex/codex.exe"
        commands["CLIProxyAPI"] = "D:/bin/CLIProxyAPI.exe"

        codex = entry(manager.catalog(), "platform:codex")

        assert codex["detected"] is False
        assert codex["status"] == "missing"


class TestCLIProxySidecarManager:
    def test_ensure_running_starts_process_and_writes_config(
        self, tmp_config_manager, tmp_path
    ):
        platform_manager = PlatformIntegrationManager(
            tmp_config_manager,
            executable_resolver=resolver_for({"CLIProxyAPI": "D:/bin/CLIProxyAPI.exe"}),
        )
        popen = FakePopenFactory()
        sidecar = CLIProxySidecarManager(
            platform_manager,
            runtime_dir=tmp_path,
            port_start=9000,
            popen_factory=popen,
            health_checker=lambda port: True,
            port_available=lambda port: True,
        )

        status = sidecar.ensure_running()

        assert status["status"] == "running"
        assert status["port"] == 9000
        assert popen.calls[0][0] == [
            "D:/bin/CLIProxyAPI.exe",
            "-config",
            str(tmp_path / "config.yaml"),
        ]
        config_text = (tmp_path / "config.yaml").read_text(encoding="utf-8")
        assert "port: 9000" in config_text
        assert 'host: "127.0.0.1"' in config_text

    def test_ensure_running_reuses_existing_process(
        self, tmp_config_manager, tmp_path
    ):
        platform_manager = PlatformIntegrationManager(
            tmp_config_manager,
            executable_resolver=resolver_for({"CLIProxyAPI": "D:/bin/CLIProxyAPI.exe"}),
        )
        popen = FakePopenFactory()
        sidecar = CLIProxySidecarManager(
            platform_manager,
            runtime_dir=tmp_path,
            popen_factory=popen,
            health_checker=lambda port: True,
            port_available=lambda port: True,
        )

        first = sidecar.ensure_running()
        second = sidecar.ensure_running()

        assert first["port"] == second["port"]
        assert len(popen.calls) == 1

    def test_missing_cliproxy_executable_raises_concise_error(
        self, tmp_config_manager, tmp_path
    ):
        platform_manager = PlatformIntegrationManager(
            tmp_config_manager,
            executable_resolver=resolver_for({}),
        )
        sidecar = CLIProxySidecarManager(platform_manager, runtime_dir=tmp_path)

        with pytest.raises(CLIProxySidecarError) as exc:
            sidecar.ensure_running()

        assert str(exc.value) == "CLIProxyAPI executable not found"

    def test_health_failure_stops_process_and_records_error(
        self, tmp_config_manager, tmp_path
    ):
        platform_manager = PlatformIntegrationManager(
            tmp_config_manager,
            executable_resolver=resolver_for({"CLIProxyAPI": "D:/bin/CLIProxyAPI.exe"}),
        )
        popen = FakePopenFactory()
        sidecar = CLIProxySidecarManager(
            platform_manager,
            runtime_dir=tmp_path,
            popen_factory=popen,
            health_checker=lambda port: False,
            port_available=lambda port: True,
            health_timeout=0,
        )

        with pytest.raises(CLIProxySidecarError) as exc:
            sidecar.ensure_running()

        assert str(exc.value) == "CLIProxyAPI sidecar did not become healthy"
        assert popen.processes[0].signals
        assert sidecar.status()["last_error"] == (
            "CLIProxyAPI sidecar did not become healthy"
        )


class TestPlatformRuntime:
    def test_start_backend_starts_sidecar_and_marks_running(
        self, tmp_config_manager
    ):
        platform_manager = PlatformIntegrationManager(
            tmp_config_manager,
            executable_resolver=resolver_for({
                "codex": "C:/Tools/Codex/codex.exe",
                "CLIProxyAPI": "D:/bin/CLIProxyAPI.exe",
            }),
        )

        class Sidecar:
            def __init__(self):
                self.calls = 0

            def ensure_running(self):
                self.calls += 1
                return {"status": "running", "port": 9100}

            def stop(self):
                return {"status": "stopped", "port": None}

        sidecar = Sidecar()

        state = platform_manager.start_backend("platform:codex", sidecar)

        assert state["active"] is True
        assert state["status"] == "running"
        assert state["sidecar_port"] == 9100
        assert sidecar.calls == 1

    def test_start_missing_platform_raises_platform_error(self, tmp_config_manager):
        platform_manager = PlatformIntegrationManager(
            tmp_config_manager,
            executable_resolver=resolver_for({"CLIProxyAPI": "D:/bin/CLIProxyAPI.exe"}),
        )

        with pytest.raises(PlatformIntegrationError) as exc:
            platform_manager.start_backend("platform:codex", object())

        assert exc.value.status_code == 400
        assert exc.value.detail == "Codex executable not found"

    def test_start_second_platform_reuses_sidecar_contract(
        self, tmp_config_manager
    ):
        platform_manager = PlatformIntegrationManager(
            tmp_config_manager,
            executable_resolver=resolver_for({
                "codex": "C:/Tools/Codex/codex.exe",
                "claude": "C:/Tools/Claude/claude.exe",
                "CLIProxyAPI": "D:/bin/CLIProxyAPI.exe",
            }),
        )

        class Sidecar:
            def __init__(self):
                self.calls = 0

            def ensure_running(self):
                self.calls += 1
                return {"status": "running", "port": 9100}

            def stop(self):
                return {"status": "stopped", "port": None}

        sidecar = Sidecar()
        codex = platform_manager.start_backend("platform:codex", sidecar)
        claude = platform_manager.start_backend("platform:claude-code", sidecar)

        assert codex["sidecar_port"] == claude["sidecar_port"] == 9100
        assert sidecar.calls == 2

    def test_stop_last_active_platform_stops_sidecar(self, tmp_config_manager):
        platform_manager = PlatformIntegrationManager(
            tmp_config_manager,
            executable_resolver=resolver_for({
                "codex": "C:/Tools/Codex/codex.exe",
                "CLIProxyAPI": "D:/bin/CLIProxyAPI.exe",
            }),
        )

        class Sidecar:
            def __init__(self):
                self.stopped = False

            def ensure_running(self):
                return {"status": "running", "port": 9100}

            def stop(self):
                self.stopped = True
                return {"status": "stopped", "port": None}

        sidecar = Sidecar()
        platform_manager.start_backend("platform:codex", sidecar)

        state = platform_manager.stop_backend("platform:codex", sidecar)

        assert state["active"] is False
        assert state["status"] == "stopped"
        assert sidecar.stopped is True
