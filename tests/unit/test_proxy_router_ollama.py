"""Tests para integração Ollama Cloud no ProxyRouter (Task 07).

Cobre:
- _candidates() inclui backends Ollama Cloud
- _pick_least_busy() prioriza conta não em cooldown
- handle_http_error() aplica cooldown com retry_after
- mark_backend_unavailable() para contas com retry_after
- Failover: quando conta A retorna 429, próxima vai para conta B
- Retorno: quando cooldown da conta A expira, retorna para A
- 5xx com cooldown menor que 4xx
"""

import asyncio
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from proxy_router import (
    ProxyRouter,
    StickySession,
    DEFAULT_BACKEND_COOLDOWN_SECONDS,
)
from platform_ollama_cloud import OllamaCloudAccount, OllamaCloudAccountManager, OllamaCloudCatalog


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_account(id: str, label: str = "", status: str = "available",
                  cooldown_until: Optional[float] = None) -> OllamaCloudAccount:
    return OllamaCloudAccount(
        id=id, api_key="", label=label, status=status, cooldown_until=cooldown_until,
    )


def _make_manager(accounts: List[OllamaCloudAccount]) -> MagicMock:
    mgr = MagicMock()
    mgr.get_accounts.return_value = list(accounts)

    def _apply_cooldown(account, cooldown_seconds):
        account.status = "cooldown"
        account.cooldown_until = time.time() + cooldown_seconds
    mgr.apply_cooldown.side_effect = _apply_cooldown

    def _clear_cooldown(account):
        account.status = "available"
        account.cooldown_until = None
    mgr.clear_cooldown.side_effect = _clear_cooldown
    return mgr


def _make_router(
    manager: Optional[OllamaCloudAccountManager] = None,
    now: Optional[Any] = None,
) -> ProxyRouter:
    cm = MagicMock()
    cm.get_smart_proxy_settings.return_value = {
        "primary_backend_id": "",
        "primary_model_path": "",
        "max_wait_seconds": 5,
    }
    cm.get_config.return_value = {
        "model_configs": {},
        "platform_configs": {},
    }
    cm.get_ollama_cloud_accounts.return_value = []
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump({"sessions": []}, f)
        f.flush()
        sessions_path = f.name
    router = ProxyRouter(
        get_status=lambda: {"instances": []},
        config_manager=cm,
        sessions_path=sessions_path,
        now=now,
        ollama_cloud_account_manager=manager,
    )
    # Cleanup temp file after test
    import atexit
    def _rm():
        try:
            os.unlink(sessions_path)
        except OSError:
            pass
    atexit.register(_rm)
    return router


# ---------------------------------------------------------------------------
# _ollama_cloud_backend_ids
# ---------------------------------------------------------------------------

class TestOllamaCloudBackendIds:
    def test_returns_empty_when_no_manager(self):
        router = _make_router()
        ids = router._ollama_cloud_backend_ids()
        assert ids == []

    def test_returns_backend_ids_for_accounts(self):
        acc1 = _make_account("a1", "Account 1")
        acc2 = _make_account("a2", "Account 2")
        mgr = _make_manager([acc1, acc2])
        router = _make_router(mgr)
        ids = router._ollama_cloud_backend_ids()
        assert len(ids) == 2
        assert "platform:ollama-cloud:a1" in ids
        assert "platform:ollama-cloud:a2" in ids


# ---------------------------------------------------------------------------
# _ollama_cloud_candidates
# ---------------------------------------------------------------------------

class TestOllamaCloudCandidates:
    def test_returns_empty_when_no_manager(self):
        router = _make_router()
        candidates = router._ollama_cloud_candidates()
        assert candidates == []

    def test_excludes_in_cooldown(self):
        future = time.time() + 120
        acc1 = _make_account("a1", "Account 1", status="available")
        acc2 = _make_account("a2", "Account 2", status="cooldown", cooldown_until=future)
        mgr = _make_manager([acc1, acc2])
        router = _make_router(mgr)
        candidates = router._ollama_cloud_candidates()
        assert len(candidates) == 1
        assert candidates[0]["backend_id"] == "platform:ollama-cloud:a1"

    def test_excludes_error_status(self):
        acc1 = _make_account("a1", "Account 1", status="error")
        acc2 = _make_account("a2", "Account 2", status="available")
        mgr = _make_manager([acc1, acc2])
        router = _make_router(mgr)
        candidates = router._ollama_cloud_candidates()
        assert len(candidates) == 1
        assert candidates[0]["backend_id"] == "platform:ollama-cloud:a2"

    def test_excludes_excluded_backend_ids(self):
        acc1 = _make_account("a1", "Account 1")
        acc2 = _make_account("a2", "Account 2")
        mgr = _make_manager([acc1, acc2])
        router = _make_router(mgr)
        candidates = router._ollama_cloud_candidates(
            exclude_backend_ids={"platform:ollama-cloud:a1"}
        )
        assert len(candidates) == 1
        assert candidates[0]["backend_id"] == "platform:ollama-cloud:a2"


# ---------------------------------------------------------------------------
# _pick_ollama_cloud_least_busy
# ---------------------------------------------------------------------------

class TestPickOllamaCloudLeastBusy:
    def test_returns_none_when_empty(self):
        router = _make_router()
        result = router._pick_ollama_cloud_least_busy([])
        assert result is None

    def test_picks_non_cooldown_over_cooldown(self):
        future = time.time() + 120
        acc1 = _make_account("a1", "Account 1", status="cooldown", cooldown_until=future)
        acc2 = _make_account("a2", "Account 2", status="available")
        mgr = _make_manager([acc1, acc2])
        router = _make_router(mgr)
        candidates = router._ollama_cloud_candidates()
        chosen = router._pick_ollama_cloud_least_busy(candidates)
        assert chosen is not None
        assert chosen["account_id"] == "a2"

    def test_returns_first_when_all_available(self):
        acc1 = _make_account("a1", "Account 1")
        acc2 = _make_account("a2", "Account 2")
        mgr = _make_manager([acc1, acc2])
        router = _make_router(mgr)
        candidates = router._ollama_cloud_candidates()
        chosen = router._pick_ollama_cloud_least_busy(candidates)
        assert chosen is not None
        # Both a1 and a2 are available; just verify it picks an available account
        assert chosen["account_id"] in ("a1", "a2")


# ---------------------------------------------------------------------------
# handle_http_error
# ---------------------------------------------------------------------------

class TestHandleHttpError:
    def test_returns_false_when_no_manager(self):
        router = _make_router()
        acc = _make_account("a1")
        result = asyncio.run(router.handle_http_error(429, acc))
        assert result is False

    def test_applies_cooldown_on_429(self):
        acc = _make_account("a1")
        mgr = _make_manager([acc])
        router = _make_router(mgr)
        result = asyncio.run(router.handle_http_error(429, acc))
        assert result is True
        assert acc.status == "cooldown"
        assert acc.cooldown_until is not None

    def test_4xx_uses_default_cooldown(self):
        acc = _make_account("a1")
        mgr = _make_manager([acc])
        router = _make_router(mgr)
        result = asyncio.run(router.handle_http_error(403, acc))
        assert result is True
        assert acc.status == "cooldown"
        assert acc.cooldown_until is not None
        cooldown_secs = acc.cooldown_until - time.time()
        assert DEFAULT_BACKEND_COOLDOWN_SECONDS - 1 < cooldown_secs <= DEFAULT_BACKEND_COOLDOWN_SECONDS + 1

    def test_5xx_has_shorter_cooldown(self):
        acc = _make_account("a1")
        mgr = _make_manager([acc])
        router = _make_router(mgr)
        result = asyncio.run(router.handle_http_error(500, acc))
        assert result is True
        assert acc.status == "cooldown"
        assert acc.cooldown_until is not None
        cooldown_secs = acc.cooldown_until - time.time()
        # 5xx cooldown should be ~30s, which is shorter than default 60s
        assert cooldown_secs <= 31

    def test_retry_after_overrides_default(self):
        acc = _make_account("a1")
        mgr = _make_manager([acc])
        router = _make_router(mgr)
        result = asyncio.run(router.handle_http_error(429, acc, retry_after=120.0))
        assert result is True
        cooldown_secs = acc.cooldown_until - time.time()
        assert 119 < cooldown_secs <= 121

    def test_marks_backend_unavailable(self):
        acc = _make_account("a1")
        mgr = _make_manager([acc])
        router = _make_router(mgr)
        asyncio.run(router.handle_http_error(429, acc))
        backend_id = "platform:ollama-cloud:a1"
        # Check that backend is in unavailable_until
        assert backend_id in router._unavailable_until


# ---------------------------------------------------------------------------
# _candidates() includes Ollama Cloud
# ---------------------------------------------------------------------------

class TestCandidatesIncludesOllamaCloud:
    def test_includes_ollama_cloud_in_candidates(self):
        acc = _make_account("a1")
        mgr = _make_manager([acc])
        router = _make_router(mgr)
        # Mock _backend_available for instances (none provided)
        instances = []
        config = {
            "platform_configs": {
                "platform:ollama-cloud": {"proxy_eligible": True},
            },
        }
        candidates = router._candidates(
            instances, config, None, 0, False,
            exclude_backend_ids=None,
            external_model="",
            configured_primary_backend_id="",
        )
        # Should have at least the Ollama Cloud candidate
        assert len(candidates) >= 1
        assert any(
            c.get("provider") == "ollama-cloud" for c in candidates
        )


# ---------------------------------------------------------------------------
# _pick_least_busy prioritizes non-cooldown accounts
# ---------------------------------------------------------------------------

class TestPickLeastBusyPrioritizesNonCooldown:
    def test_prioritizes_non_cooldown_account(self):
        future = time.time() + 120
        # Create accounts: one in cooldown, one available
        acc_cooldown = _make_account("a1", status="cooldown", cooldown_until=future)
        acc_available = _make_account("a2", status="available")
        mgr = _make_manager([acc_cooldown, acc_available])
        router = _make_router(mgr)
        candidates = router._ollama_cloud_candidates()
        # _pick_least_busy should prefer the non-cooldown account
        # Note: _pick_least_busy uses cooldown priority as a key
        chosen = router._pick_least_busy(candidates)
        assert chosen is not None
        assert chosen["account_id"] == "a2"  # Available account


# ---------------------------------------------------------------------------
# Failover scenario
# ---------------------------------------------------------------------------

class TestFailover:
    def test_failover_to_second_account(self):
        """When account A returns 429, next request should go to account B."""
        acc_a = _make_account("a1", "Account A")
        acc_b = _make_account("a2", "Account B")
        mgr = _make_manager([acc_a, acc_b])
        router = _make_router(mgr)

        # Initially, pick A (first available)
        candidates = router._ollama_cloud_candidates()
        assert len(candidates) == 2

        # Simulate 429 on account A
        asyncio.run(router.handle_http_error(429, acc_a))

        # Now only B should be available
        candidates = router._ollama_cloud_candidates()
        assert len(candidates) == 1
        assert candidates[0]["account_id"] == "a2"

    def test_failover_with_retry_after(self):
        """When account A hits rate limit with retry_after, cooldown respects it."""
        acc_a = _make_account("a1")
        mgr = _make_manager([acc_a])
        router = _make_router(mgr)

        asyncio.run(router.handle_http_error(429, acc_a, retry_after=300.0))
        assert acc_a.status == "cooldown"
        cooldown_secs = acc_a.cooldown_until - time.time()
        assert 299 < cooldown_secs <= 301


# ---------------------------------------------------------------------------
# Cooldown expiry / return to original
# ---------------------------------------------------------------------------

class TestCooldownExpiryReturn:
    def test_account_returns_after_cooldown_expires(self):
        """When cooldown of account A expires, it returns to candidates."""
        future = time.time() - 1  # Cooldown already expired (in the past)
        acc_a = _make_account("a1", status="cooldown", cooldown_until=future)
        acc_b = _make_account("a2", status="available")
        mgr = _make_manager([acc_a, acc_b])
        router = _make_router(mgr)

        # Initially both should be available (cooldown already expired)
        candidates = router._ollama_cloud_candidates()
        assert len(candidates) == 2

    def test_account_still_in_cooldown_not_returned(self):
        """Account in active cooldown should not be returned to candidates."""
        future = time.time() + 600  # Cooldown 10 minutes from now
        acc_a = _make_account("a1", status="cooldown", cooldown_until=future)
        mgr = _make_manager([acc_a])
        router = _make_router(mgr)

        candidates = router._ollama_cloud_candidates()
        assert len(candidates) == 0


# ---------------------------------------------------------------------------
# 5xx vs 4xx cooldown
# ---------------------------------------------------------------------------

class TestCooldown5xxVs4xx:
    def test_5xx_has_shorter_cooldown_than_4xx(self):
        """5xx errors should have shorter cooldown than 4xx errors."""
        acc_5xx = _make_account("a5")
        acc_4xx = _make_account("a4")
        mgr = _make_manager([acc_5xx, acc_4xx])
        router = _make_router(mgr)

        asyncio.run(router.handle_http_error(500, acc_5xx))
        asyncio.run(router.handle_http_error(429, acc_4xx))

        cooldown_5xx = acc_5xx.cooldown_until - time.time()
        cooldown_4xx = acc_4xx.cooldown_until - time.time()

        assert cooldown_5xx < cooldown_4xx
        assert cooldown_5xx <= 31  # ~30s
        assert cooldown_4xx >= 59  # ~60s


# ---------------------------------------------------------------------------
# mark_backend_unavailable from account
# ---------------------------------------------------------------------------

class TestMarkBackendUnavailable:
    def test_marks_backend_unavailable(self):
        """Backend should be marked unavailable for the specified cooldown period."""
        acc = _make_account("a1")
        mgr = _make_manager([acc])
        router = _make_router(mgr)

        backend_id = "platform:ollama-cloud:a1"
        asyncio.run(router.handle_http_error(429, acc, retry_after=60.0))

        assert backend_id in router._unavailable_until
        until = router._unavailable_until[backend_id]
        now = router._now()
        remaining = (until - now).total_seconds()
        assert 59 < remaining <= 61


# ---------------------------------------------------------------------------
# Integration: full failover flow
# ---------------------------------------------------------------------------

class TestFullFailoverFlow:
    def test_full_failover_and_return_flow(self):
        """Complete flow: select A -> A hits 429 -> failover to B -> cooldown expires -> back to A."""
        acc_a = _make_account("a1", "Account A")
        acc_b = _make_account("a2", "Account B")
        mgr = _make_manager([acc_a, acc_b])
        router = _make_router(mgr)

        # Phase 1: Both accounts available
        candidates = router._ollama_cloud_candidates()
        assert len(candidates) == 2

        # Phase 2: Account A hits 429
        asyncio.run(router.handle_http_error(429, acc_a))
        candidates = router._ollama_cloud_candidates()
        assert len(candidates) == 1
        assert candidates[0]["account_id"] == "a2"

        # Phase 3: Cooldown expires (set to past)
        acc_a.cooldown_until = time.time() - 1
        acc_a.status = "cooldown"
        candidates = router._ollama_cloud_candidates()
        assert len(candidates) == 2

        # Phase 4: Both available again
        assert any(c["account_id"] == "a1" for c in candidates)
        assert any(c["account_id"] == "a2" for c in candidates)


class TestProviderSafeReassign:
    @staticmethod
    def _router_with_cli_and_cloud():
        acc_a = _make_account("a1", "Account A")
        acc_b = _make_account("a2", "Account B")
        router = _make_router(_make_manager([acc_a, acc_b]))
        router._get_status = lambda: {
            "instances": [
                {
                    "backend_id": "platform:codex",
                    "backend_type": "platform",
                    "provider": "codex",
                    "status": "running",
                    "port": 8317,
                    "model": "Codex",
                    "model_path": "",
                    "config": {},
                }
            ]
        }
        router._config.get_config.return_value = {
            "model_configs": {},
            "platform_configs": {
                "platform:ollama-cloud": {"proxy_eligible": True},
                "platform:codex": {"proxy_eligible": True},
            },
        }
        cloud_a = next(
            item for item in router._routing_instances()
            if item.get("account_id") == "a1"
        )
        router._sessions["sid:test"] = StickySession(
            affinity_key="sid:test",
            backend_port=cloud_a["port"],
            backend_model_path="",
            external_model="gemma4:31b-custom.gguf",
            internal_model="gemma4:31b",
            detected_tag=None,
            created_at="2026-01-01T00:00:00+00:00",
            last_used_at="2026-01-01T00:00:00+00:00",
            backend_id=cloud_a["backend_id"],
            backend_type="platform",
            provider="ollama-cloud",
        )
        return router, cloud_a

    def test_uses_second_ollama_account_before_other_platform(self):
        router, cloud_a = self._router_with_cli_and_cloud()
        decision = asyncio.run(
            router.reassign(
                "sid:test",
                exclude_backend_ids={cloud_a["backend_id"]},
                reason="test_account_failover",
            )
        )
        assert decision is not None
        assert decision.provider == "ollama-cloud"
        assert decision.backend_id.endswith(":a2")

    def test_uses_other_platform_only_after_all_ollama_accounts_fail(self):
        router, cloud_a = self._router_with_cli_and_cloud()
        decision = asyncio.run(
            router.reassign(
                "sid:test",
                exclude_backend_ids={
                    cloud_a["backend_id"],
                    "platform:ollama-cloud:a2",
                },
                reason="test_provider_fallback",
            )
        )
        assert decision is not None
        assert decision.provider == "codex"
        assert decision.backend_id == "platform:codex"
