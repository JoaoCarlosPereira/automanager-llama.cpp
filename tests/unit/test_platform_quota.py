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


def test_parse_antigravity_quota_min_across_groups():
    payload = {
        "groups": [
            {
                "displayName": "GEMINI MODELS",
                "buckets": [
                    {"window": "weekly", "remainingFraction": 0.8, "resetTime": "2026-10-20T00:00:00Z"},
                    {"window": "5h", "remainingFraction": 0.5, "resetTime": "2026-10-18T12:00:00Z"},
                ],
            },
            {
                "displayName": "CLAUDE AND GPT",
                "buckets": [
                    {"window": "weekly", "remainingFraction": 0.3},
                    {"window": "5h", "remainingFraction": 0.9},
                ],
            },
        ]
    }
    parsed = pq.parse_antigravity_quota(payload)
    assert parsed["weekly_remaining_pct"] == 30.0
    assert parsed["session_remaining_pct"] == 50.0


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
    assert "ultima cota conhecida" in merged["error"]


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
    assert captured["header"].get("User-Agent", "").startswith("claude-code/")
    assert snap["claude"]["session_remaining_pct"] == 88.0
    assert snap["claude"]["weekly_remaining_pct"] == 66.0


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
    assert "wrapPlatformCardWithQuota" in models_js
    assert "quota-ring-weekly" in models_js
    assert "quota-ring-session" in models_js
    assert "quota-account-list" in models_js
    assert "buildQuotaAccountsHtml" in models_js
    assert "ensurePlatformQuotaRotation" in models_js
    assert "rotateAllPlatformQuotaShells" in models_js
    assert "QUOTA_ACCOUNT_ROTATE_MS" in models_js
    assert "refreshPlatformQuotas" in models_js
    assert "/platforms/quota" in models_js
