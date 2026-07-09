import pytest

from platform_manager import (
    CLIProxySidecarError,
    CLIProxySidecarManager,
    PlatformIntegrationError,
    PlatformIntegrationManager,
)


def resolver_for(mapping):
    def _resolve(command):
        return mapping.get(command)

    return _resolve


def entry(catalog, backend_id):
    return next(item for item in catalog if item["backend_id"] == backend_id)


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
            {"proxy_eligible": True, "max_parallel_requests": 2},
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
