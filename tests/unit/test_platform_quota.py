"""Unit tests for platform quota parsers and aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import platform_quota as pq


def test_parse_codex_usage_primary_secondary_windows():
    payload = {
        "plan_type": "plus",
        "rate_limit": {
            "primary_window": {
                "used_percent": 40,
                "reset_at": 1738300000,
                "limit_window_seconds": 18000,
            },
            "secondary_window": {
                "used_percent": 25,
                "reset_at": 1738900000,
                "limit_window_seconds": 604800,
            },
        },
    }
    parsed = pq.parse_codex_usage(payload)
    assert parsed["session_remaining_pct"] == 60.0
    assert parsed["weekly_remaining_pct"] == 75.0
    assert parsed["session_reset_at"]
    assert parsed["weekly_reset_at"]


def test_parse_codex_usage_falls_back_to_headers():
    parsed = pq.parse_codex_usage(
        {},
        {
            "x-codex-primary-used-percent": "10",
            "x-codex-secondary-used-percent": "90",
            "x-codex-primary-reset-at": "1738300000",
        },
    )
    assert parsed["session_remaining_pct"] == 90.0
    assert parsed["weekly_remaining_pct"] == 10.0


def test_parse_claude_usage_five_hour_and_seven_day():
    payload = {
        "five_hour": {"utilization": 23.5, "resets_at": "2026-03-20T15:00:00Z"},
        "seven_day": {"used_percentage": 41.2, "resets_at": "2026-03-24T00:00:00Z"},
    }
    parsed = pq.parse_claude_usage(payload)
    assert parsed["session_remaining_pct"] == 76.5
    assert parsed["weekly_remaining_pct"] == 58.8


def test_parse_antigravity_quota_prefers_gemini_group():
    # agy TUI "Models & Quota" shows Gemini first; Claude@30% must not win.
    payload = {
        "groups": [
            {
                "displayName": "GEMINI MODELS",
                "buckets": [
                    {"window": "weekly", "remainingFraction": 0.4894, "resetTime": "2026-10-23T00:00:00Z"},
                    {"window": "5h", "remainingFraction": 0.6755, "resetTime": "2026-10-21T05:00:00Z"},
                ],
            },
            {
                "displayName": "CLAUDE AND GPT",
                "buckets": [
                    {"window": "weekly", "remainingFraction": 1.0},
                    {"window": "5h", "remainingFraction": 1.0, "resetTime": "2026-10-21T08:00:00Z"},
                ],
            },
        ]
    }
    parsed = pq.parse_antigravity_quota(payload)
    assert parsed["weekly_remaining_pct"] == 48.9
    assert parsed["session_remaining_pct"] == 67.5
    assert parsed["focus_group"] == "gemini"
    assert parsed["weekly_reset_at"] == "2026-10-23T00:00:00Z"
    assert parsed["session_reset_at"] == "2026-10-21T05:00:00Z"


def test_parse_antigravity_quota_nested_remaining_and_wrapper():
    payload = {
        "response": {
            "groups": [
                {
                    "displayName": "Gemini Models",
                    "buckets": [
                        {
                            "bucketId": "gemini-weekly",
                            "displayName": "Weekly Limit Remaining",
                            "remaining": {"remainingFraction": 0.4894},
                            "resetTime": "2026-10-23T00:00:00Z",
                        },
                        {
                            "bucket_id": "gemini-5h",
                            "display_name": "Five Hour Limit Remaining",
                            "remaining": {
                                "case": "remainingFraction",
                                "value": 0.6755,
                            },
                            "reset_time": "2026-10-21T05:00:00Z",
                        },
                    ],
                }
            ]
        }
    }
    parsed = pq.parse_antigravity_quota(payload)
    assert parsed["weekly_remaining_pct"] == 48.9
    assert parsed["session_remaining_pct"] == 67.5


def test_parse_antigravity_quota_accepts_percent_scale():
    # Accidental percent payloads must not clamp to 100 via *100.
    payload = {
        "groups": [
            {
                "displayName": "GEMINI MODELS",
                "buckets": [
                    {"window": "weekly", "remainingFraction": 48.94},
                    {"window": "5h", "remaining_fraction": 67.55},
                ],
            }
        ]
    }
    parsed = pq.parse_antigravity_quota(payload)
    assert parsed["weekly_remaining_pct"] == 48.9
    assert parsed["session_remaining_pct"] == 67.5


def test_parse_antigravity_quota_falls_back_to_most_depleted():
    payload = {
        "groups": [
            {
                "displayName": "CLAUDE AND GPT",
                "buckets": [
                    {"window": "weekly", "remainingFraction": 0.3},
                    {"window": "5h", "remainingFraction": 0.9},
                ],
            },
            {
                "displayName": "Other pool",
                "buckets": [
                    {"window": "weekly", "remainingFraction": 0.8},
                    {"window": "5h", "remainingFraction": 0.5},
                ],
            },
        ]
    }
    parsed = pq.parse_antigravity_quota(payload)
    assert parsed["weekly_remaining_pct"] == 30.0
    assert parsed["session_remaining_pct"] == 90.0


def test_parse_antigravity_quota_from_agy_cli_output():
    payload = {
        "command": {
            "name": "usage",
            "data": {
                "description": "Within each group, models share a weekly limit...",
                "groups": [
                    {
                        "name": "Gemini Models",
                        "buckets": [
                            {
                                "id": "gemini-weekly",
                                "name": "Weekly Limit Remaining",
                                "window": "weekly",
                                "remaining_fraction": 0.443,
                                "reset_time": "2026-10-23T21:57:36Z",
                            },
                            {
                                "id": "gemini-5h",
                                "name": "Five Hour Limit Remaining",
                                "window": "5h",
                                "remaining_fraction": 0.952,
                                "reset_time": "2026-10-22T23:04:36Z",
                            },
                        ],
                    },
                    {
                        "name": "Claude and GPT models",
                        "buckets": [
                            {
                                "id": "3p-weekly",
                                "name": "Weekly Limit Remaining",
                                "window": "weekly",
                                "remaining_fraction": 1.0,
                                "reset_time": "2026-10-23T20:25:24Z",
                            },
                        ],
                    },
                ],
            },
        }
    }
    parsed = pq.parse_antigravity_quota(payload)
    assert parsed["weekly_remaining_pct"] == 44.3
    assert parsed["session_remaining_pct"] == 95.2
    assert parsed["weekly_reset_at"] == "2026-10-23T21:57:36Z"
    assert parsed["session_reset_at"] == "2026-10-22T23:04:36Z"
    assert parsed["focus_group"] == "gemini"


def test_fetch_antigravity_cli_matches_auth_email(monkeypatch):
    agy_payload = {
        "command": {
            "name": "usage",
            "data": {
                "groups": [
                    {
                        "name": "Gemini Models",
                        "buckets": [
                            {
                                "id": "gemini-weekly",
                                "window": "weekly",
                                "remaining_fraction": 0.443,
                                "reset_time": "2026-10-23T21:57:36Z",
                            }
                        ],
                    }
                ]
            },
        }
    }

    class DummyProc:
        returncode = 0
        stdout = json.dumps(agy_payload)

    monkeypatch.setattr(pq, "_find_antigravity_binary", lambda: Path("/fake/bin/agy"))
    monkeypatch.setattr(pq, "_get_active_agy_email", lambda: "user@example.com")
    monkeypatch.setattr(pq.subprocess, "run", lambda *args, **kwargs: DummyProc())

    # When email matches, returns parsed quota
    parsed = pq.fetch_antigravity_cli({"email": "user@example.com"})
    assert parsed is not None
    assert parsed["weekly_remaining_pct"] == 44.3

    # When email mismatches, returns None
    assert pq.fetch_antigravity_cli({"email": "other@example.com"}) is None


def test_merge_account_quotas_keeps_per_account_and_focuses_priority():
    merged = pq._merge_account_quotas(
        "codex",
        [
            {
                "auth_index": "low",
                "label": "low@example.com",
                "priority": 0,
                "session_remaining_pct": 20,
                "weekly_remaining_pct": 15,
            },
            {
                "auth_index": "high",
                "label": "high@example.com",
                "priority": 10,
                "session_remaining_pct": 80,
                "weekly_remaining_pct": 60,
                "session_reset_at": "soon",
            },
        ],
    )
    assert merged["available"] is True
    assert merged["accounts"] == 2
    assert merged["focused_auth_index"] == "high"
    assert merged["session_remaining_pct"] == 80.0
    assert merged["weekly_remaining_pct"] == 60.0
    assert [row["label"] for row in merged["account_details"]] == [
        "high@example.com",
        "low@example.com",
    ]
    assert merged["account_details"][1]["weekly_remaining_pct"] == 15.0


def test_merge_account_quotas_focuses_most_depleted_when_priority_ties():
    merged = pq._merge_account_quotas(
        "claude",
        [
            {
                "auth_index": "a",
                "label": "a@example.com",
                "priority": 1,
                "session_remaining_pct": 50,
                "weekly_remaining_pct": 90,
            },
            {
                "auth_index": "b",
                "label": "b@example.com",
                "priority": 1,
                "session_remaining_pct": 40,
                "weekly_remaining_pct": 12,
            },
        ],
    )
    assert merged["focused_auth_index"] == "b"
    assert merged["weekly_remaining_pct"] == 12.0


def test_ensure_management_key_persists(tmp_path: Path):
    key1 = pq.ensure_management_key(tmp_path)
    key2 = pq.ensure_management_key(tmp_path)
    assert key1 == key2
    assert (tmp_path / "management.key").read_text(encoding="utf-8").strip() == key1


def test_quota_service_update_from_headers(tmp_path: Path):
    service = pq.PlatformQuotaService(runtime_dir=tmp_path, port_provider=lambda: None)
    service.update_from_headers(
        "codex",
        {
            "x-codex-primary-used-percent": "30",
            "x-codex-secondary-used-percent": "10",
        },
    )
    snap = service.cached_snapshot()
    assert snap["codex"]["session_remaining_pct"] == 70.0
    assert snap["codex"]["weekly_remaining_pct"] == 90.0
    assert snap["codex"]["available"] is True


def test_quota_disk_cache_survives_service_restart(tmp_path: Path):
    """Apos restart, cotas boas em disco alimentam o preserve no 429."""

    class FakeResponse:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def json(self):
            return self._payload

    class OkThen429:
        calls = {"n": 0}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, headers=None):
            return FakeResponse(
                200,
                {
                    "files": [
                        {
                            "auth_index": "auth-claude",
                            "provider": "claude",
                            "email": "user@example.com",
                        }
                    ]
                },
            )

        def post(self, url, headers=None, json=None):
            OkThen429.calls["n"] += 1
            if OkThen429.calls["n"] == 1:
                body = {
                    "five_hour": {"utilization": 10, "resets_at": "2026-10-20T12:00:00Z"},
                    "seven_day": {"utilization": 80, "resets_at": "2026-10-24T00:00:00Z"},
                }
                return FakeResponse(
                    200,
                    {
                        "status_code": 200,
                        "body": json_module.dumps(body),
                        "header": {},
                    },
                )
            return FakeResponse(
                200,
                {"status_code": 429, "body": '{"error":"rate_limit"}', "header": {}},
            )

    import json as json_module

    first = pq.PlatformQuotaService(
        runtime_dir=tmp_path,
        port_provider=lambda: 8317,
        http_client_factory=OkThen429,
        cache_ttl=0,
    )
    snap = first.snapshot(force=True)
    assert snap["claude"]["weekly_remaining_pct"] == 20.0
    assert (tmp_path / "quota_cache.json").is_file()

    # "Restart": nova instancia, primeira chamada ja vem 429.
    OkThen429.calls["n"] = 1  # next post is 429
    second = pq.PlatformQuotaService(
        runtime_dir=tmp_path,
        port_provider=lambda: 8317,
        http_client_factory=OkThen429,
        cache_ttl=0,
    )
    # Disco ja carregou 20%/90% — preserve deve manter apos 429.
    assert second.cached_snapshot()["claude"]["weekly_remaining_pct"] == 20.0
    after = second.snapshot(force=True)
    assert after["claude"]["weekly_remaining_pct"] == 20.0
    assert after["claude"]["session_remaining_pct"] == 90.0
    err = after["claude"]["account_details"][0].get("error") or ""
    assert "Monitoramento em espera" in err


def test_preserve_account_on_error_keeps_last_known_percentages():
    previous = {
        "auth_index": "a1",
        "label": "user@example.com",
        "session_remaining_pct": 42.0,
        "weekly_remaining_pct": 77.0,
    }
    failed = {
        "auth_index": "a1",
        "label": "user@example.com",
        "session_remaining_pct": None,
        "weekly_remaining_pct": None,
        "error": "claude usage HTTP 429",
    }
    merged = pq._preserve_account_on_error(previous, failed)
    assert merged["session_remaining_pct"] == 42.0
    assert merged["weekly_remaining_pct"] == 77.0
    assert "Monitoramento em espera" in merged["error"]
    assert "HTTP 429" not in merged["error"]


def test_parse_codex_usage_primary_only_when_secondary_null():
    payload = {
        "plan_type": "plus",
        "rate_limit": {
            "primary_window": {
                "used_percent": 12,
                "reset_after_seconds": 3600,
                "limit_window_seconds": 18000,
            },
            "secondary_window": None,
        },
    }
    parsed = pq.parse_codex_usage(payload)
    assert parsed["session_remaining_pct"] == 88.0
    assert parsed["weekly_remaining_pct"] is None
    assert parsed["session_reset_at"]


def test_claude_fetch_sends_claude_code_user_agent(tmp_path: Path):
    captured = {}

    class FakeResponse:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def json(self):
            return self._payload

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, headers=None):
            return FakeResponse(
                200,
                {
                    "files": [
                        {
                            "auth_index": "auth-claude",
                            "provider": "claude",
                            "email": "user@example.com",
                        }
                    ]
                },
            )

        def post(self, url, headers=None, json=None):
            payload = json or {}
            captured["header"] = payload.get("header") or {}
            body = {
                "five_hour": {"utilization": 12, "resets_at": "2026-10-20T12:00:00Z"},
                "seven_day": {"utilization": 34, "resets_at": "2026-10-24T00:00:00Z"},
            }
            return FakeResponse(
                200,
                {
                    "status_code": 200,
                    "body": __import__("json").dumps(body),
                    "header": {},
                },
            )

    service = pq.PlatformQuotaService(
        runtime_dir=tmp_path,
        port_provider=lambda: 8317,
        http_client_factory=FakeClient,
        cache_ttl=0,
    )
    snap = service.snapshot(force=True)
    assert captured["header"].get("User-Agent") == "claude-code/2.1.80"
    assert "(" not in captured["header"].get("User-Agent", "")
    assert snap["claude"]["session_remaining_pct"] == 88.0
    assert snap["claude"]["weekly_remaining_pct"] == 66.0


def test_claude_usage_429_backs_off_and_skips_followup_polls(tmp_path: Path):
    """After Anthropic 429, do not keep hitting /api/oauth/usage every cache TTL."""
    calls = {"n": 0}

    class FakeResponse:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def json(self):
            return self._payload

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, headers=None):
            return FakeResponse(
                200,
                {
                    "files": [
                        {
                            "auth_index": "auth-claude",
                            "provider": "claude",
                            "email": "user@example.com",
                        }
                    ]
                },
            )

        def post(self, url, headers=None, json=None):
            calls["n"] += 1
            if calls["n"] == 1:
                # Seed cache with a successful read first.
                body = {
                    "five_hour": {"utilization": 7, "resets_at": "2026-10-20T12:00:00Z"},
                    "seven_day": {"utilization": 72, "resets_at": "2026-10-24T00:00:00Z"},
                }
                return FakeResponse(
                    200,
                    {
                        "status_code": 200,
                        "body": __import__("json").dumps(body),
                        "header": {},
                    },
                )
            return FakeResponse(
                200,
                {"status_code": 429, "body": '{"error":"rate_limit"}', "header": {}},
            )

    service = pq.PlatformQuotaService(
        runtime_dir=tmp_path,
        port_provider=lambda: 8317,
        http_client_factory=FakeClient,
        cache_ttl=0,
    )
    first = service.snapshot(force=True)
    assert first["claude"]["weekly_remaining_pct"] == 28.0
    assert calls["n"] == 1

    second = service.snapshot(force=True)
    assert calls["n"] == 2  # one 429 attempt
    assert second["claude"]["weekly_remaining_pct"] == 28.0
    second_err = second["claude"]["account_details"][0].get("error") or ""
    assert "Monitoramento em espera" in second_err

    third = service.snapshot(force=True)
    # Still in backoff — must not call Anthropic again.
    assert calls["n"] == 2
    err = third["claude"]["account_details"][0].get("error") or ""
    assert "espera" in err.lower() or "Monitoramento" in err
    assert third["claude"]["weekly_remaining_pct"] == 28.0
    assert third["claude"]["session_remaining_pct"] == 93.0


def test_quota_service_refresh_via_management_api(tmp_path: Path, monkeypatch):
    class FakeResponse:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def json(self):
            return self._payload

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, headers=None):
            assert url.endswith("/v0/management/auth-files")
            return FakeResponse(
                200,
                {
                    "files": [
                        {
                            "auth_index": "auth-1",
                            "provider": "claude",
                            "email": "user@example.com",
                        }
                    ]
                },
            )

        def post(self, url, headers=None, json=None):
            assert url.endswith("/v0/management/api-call")
            assert json["url"] == "https://api.anthropic.com/api/oauth/usage"
            body = {
                "five_hour": {"utilization": 10, "resets_at": "2026-10-18T12:00:00Z"},
                "seven_day": {"utilization": 40, "resets_at": "2026-10-24T00:00:00Z"},
            }
            return FakeResponse(
                200,
                {
                    "status_code": 200,
                    "body": json_module_dumps(body),
                    "header": {},
                },
            )

    def json_module_dumps(value):
        return json.dumps(value)

    service = pq.PlatformQuotaService(
        runtime_dir=tmp_path,
        port_provider=lambda: 8317,
        http_client_factory=FakeClient,
        cache_ttl=0,
    )
    snap = service.snapshot(force=True)
    assert snap["claude"]["available"] is True
    assert snap["claude"]["session_remaining_pct"] == 90.0
    assert snap["claude"]["weekly_remaining_pct"] == 60.0


def test_models_js_exposes_quota_ring_helpers():
    models_js = Path("static/js/models.js").read_text(encoding="utf-8")
    proxy_js = Path("static/js/proxy.js").read_text(encoding="utf-8")
    assert "wrapProxyBackendWithQuota" in models_js
    assert "applyAllQuotaShells" in models_js
    assert "quota-gauge" in models_js
    assert "buildQuotaGaugesHtml" in models_js
    assert "quota-account-chips" in models_js or "quota-chip" in models_js
    assert "buildQuotaAccountsHtml" in models_js
    assert "ensurePlatformQuotaRotation" in models_js
    assert "quota-gauges" in proxy_js
    assert "proxy-card-v2" in proxy_js
    assert "localGaugesHtml" in proxy_js or "Capacidade" in proxy_js
    assert "Prontidão" in proxy_js or "Prontidao" in proxy_js
    assert 'data-kind="capacity"' in proxy_js or 'data-kind="startup"' in proxy_js
    assert "rotateAllPlatformQuotaShells" in models_js
    assert "QUOTA_ACCOUNT_ROTATE_MS" in models_js
    assert "refreshPlatformQuotas" in models_js
    assert "/platforms/quota" in models_js
    assert "proxy-quota-shell" in proxy_js
    assert "data-quota-provider" in proxy_js
    assert "applyAllQuotaShells" in proxy_js
    assert "lastProxyQuotas" in proxy_js or "syncProxyQuotas" in proxy_js
    assert "patchBackendCard" in proxy_js or "patchPlatformGauges" in proxy_js
    assert "shortGpuLabel" in proxy_js
    assert "local-meta-host" in proxy_js
    assert "isQuotaExhausted" in models_js
    assert "isQuotaExhausted" in proxy_js
    assert "formatQuotaCountdown" in models_js
    assert "formatQuotaCountdown" in proxy_js
    assert "quota-gauge-pct-countdown" in models_js
    assert "quota-gauge-pct-countdown" in proxy_js
    assert "quota-gauge-pct-countdown" in Path("llama_manager.py").read_text(encoding="utf-8")


def test_parse_quota_headers_anthropic_unified():
    headers = {
        "Anthropic-Ratelimit-Unified-5h-Utilization": "0.06",
        "Anthropic-Ratelimit-Unified-5h-Reset": "1792709400",
        "Anthropic-Ratelimit-Unified-7d-Utilization": "1.0",
        "Anthropic-Ratelimit-Unified-7d-Reset": "1792818000",
    }
    parsed = pq.parse_quota_headers(headers)
    assert parsed is not None
    assert parsed["session_remaining_pct"] == 94.0
    assert parsed["weekly_remaining_pct"] == 0.0
    assert parsed["weekly_reset_at"] == "2026-10-24T05:00:00Z"


def test_extract_auth_quota_signals_and_cooldowns():
    auth = {
        "auth_index": "test_claude",
        "provider": "claude",
        "unavailable": True,
        "cooldowns": [
            {
                "scope": "credential",
                "reason": "credential_quota",
                "retry_at": "2026-10-24T05:00:01Z",
                "remaining_seconds": 120000,
            }
        ],
        "quota": {
            "signals": {
                "Anthropic-Ratelimit-Unified-5h-Utilization": "0.10",
                "Anthropic-Ratelimit-Unified-7d-Utilization": "1.0",
                "Anthropic-Ratelimit-Unified-7d-Reset": "1792818000",
            }
        },
    }
    extracted = pq._extract_auth_quota(auth)
    assert extracted["session_remaining_pct"] == 90.0
    assert extracted["weekly_remaining_pct"] == 0.0
    assert extracted["weekly_reset_at"] == "2026-10-24T05:00:01Z"


def test_refresh_handles_unavailable_accounts_without_skipping():
    class DummyClient:
        pass

    service = pq.PlatformQuotaService()
    files = [
        {
            "auth_index": "cl1",
            "provider": "claude",
            "email": "claude@sysmo.com.br",
            "unavailable": True,
            "disabled": False,
            "cooldowns": [
                {
                    "scope": "credential",
                    "reason": "quota",
                    "retry_at": "2026-10-24T05:00:01Z",
                }
            ],
            "quota": {
                "signals": {
                    "Anthropic-Ratelimit-Unified-5h-Utilization": "0.06",
                    "Anthropic-Ratelimit-Unified-7d-Utilization": "1.0",
                    "Anthropic-Ratelimit-Unified-7d-Reset": "1792818000",
                }
            },
        }
    ]
    service._base_url = lambda: "http://127.0.0.1:8317"
    service._list_auth_files = lambda client, base: files

    result = service._refresh()
    assert "claude" in result
    claude = result["claude"]
    assert claude["available"] is True
    assert claude["weekly_remaining_pct"] == 0.0
    assert claude["session_remaining_pct"] == 94.0
    assert claude["weekly_reset_at"] == "2026-10-24T05:00:01Z"
