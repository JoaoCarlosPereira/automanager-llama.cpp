"""Integration tests for Ollama Cloud administrative endpoints.

Covers all 6 admin endpoints:
- GET /platforms/ollama-cloud/accounts
- POST /platforms/ollama-cloud/accounts
- DELETE /platforms/ollama-cloud/accounts/{account_id}
- PATCH /platforms/ollama-cloud/accounts/{account_id}
- POST /platforms/ollama-cloud/accounts/{account_id}/validate
- POST /platforms/ollama-cloud/catalog/refresh

All tests use TestClient (sync) against the FastAPI app.
"""

import json
import os
import sys
import uuid

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import llama_manager
from llama_manager import app
from paths import CONFIG_PATH

# ── Helpers ─────────────────────────────────────────────────────────────────

VALID_SESSION = "test-ollama-session-v1"


class _FakeAuth:
    """Minimal auth double that accepts a session cookie or bearer token."""

    def __init__(self):
        self.logged_out = []

    def check_auth(self, request=None):
        if request is None:
            return False
        session_token = request.cookies.get("session_token")
        if session_token and session_token not in self.logged_out:
            return True
        auth_header = request.headers.get("Authorization", "")
        if auth_header.lower().startswith("bearer "):
            return True
        return False


def _get_config_path():
    """Return the absolute config path."""
    return os.path.abspath(CONFIG_PATH)


def _make_unauth_client():
    """Return a TestClient where auth always fails (401)."""
    return TestClient(app)


def _add_account_via_config(api_key: str, label: str = "") -> dict:
    """Directly add an account to config.json and return its id."""
    path = _get_config_path()
    try:
        data = json.loads(open(path).read()) if os.path.isfile(path) else {}
    except Exception:
        data = {}
    accounts = data.get("ollama_cloud_accounts", [])
    if not isinstance(accounts, list):
        accounts = []
    acc_id = str(uuid.uuid4())
    accounts.append({
        "id": acc_id,
        "api_key": api_key,
        "label": label,
        "created_at": "2026-01-01T00:00:00Z",
    })
    data["ollama_cloud_accounts"] = accounts
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)
    return {"id": acc_id}


def _clear_accounts():
    """Remove ollama_cloud_accounts from config."""
    path = _get_config_path()
    try:
        data = json.loads(open(path).read()) if os.path.isfile(path) else {}
    except Exception:
        data = {}
    if "ollama_cloud_accounts" in data:
        del data["ollama_cloud_accounts"]
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clear_ollama_accounts():
    """Ensure config has no ollama_cloud_accounts before each test."""
    _clear_accounts()
    yield
    _clear_accounts()


@pytest.fixture()
def client(monkeypatch):
    """Client with valid session cookie set."""
    fake_auth = _FakeAuth()
    monkeypatch.setattr("llama_manager.auth_manager", fake_auth)
    tc = TestClient(app)
    tc.cookies.set("session_token", VALID_SESSION)
    return tc


# ── Auth enforcement (all 6 endpoints require auth) ─────────────────────────

def test_all_endpoints_require_auth():
    """All 6 admin endpoints return 401 without authentication."""
    client = _make_unauth_client()
    cases = [
        ("GET", "/platforms/ollama-cloud/accounts"),
        ("POST", "/platforms/ollama-cloud/accounts", {"api_key": "sk-x", "label": "x"}),
        ("PATCH", "/platforms/ollama-cloud/accounts/any-id", {"label": "y"}),
        ("DELETE", "/platforms/ollama-cloud/accounts/any-id"),
        ("POST", "/platforms/ollama-cloud/accounts/any-id/validate"),
        ("POST", "/platforms/ollama-cloud/catalog/refresh"),
    ]
    for i, item in enumerate(cases):
        method, path, *rest = item
        json_body = rest[0] if rest else None
        kwargs = {"json": json_body} if json_body is not None else {}
        resp = client.request(method, path, **kwargs)
        assert resp.status_code == 401, f"Endpoint {i} ({method} {path}) did not require auth"


# ── GET /platforms/ollama-cloud/accounts ────────────────────────────────────

def test_get_accounts_empty(client):
    """GET returns empty list when no accounts exist."""
    response = client.get("/platforms/ollama-cloud/accounts")
    assert response.status_code == 200
    assert response.json() == {"accounts": []}


def test_get_accounts_listed(client):
    """GET returns accounts with masked api_key."""
    # Add account directly to config (avoids endpoint masking)
    acc_info = _add_account_via_config("sk-get-list-111", "GetTest")
    acc_id = acc_info["id"]

    response = client.get("/platforms/ollama-cloud/accounts")
    assert response.status_code == 200
    accounts = response.json()["accounts"]
    found = next((a for a in accounts if a["id"] == acc_id), None)
    assert found is not None, f"Account {acc_id} not found in {accounts}"
    # api_key must be masked — never show full key in plaintext
    # get_accounts() returns masked api_key; it should NOT contain the raw key
    assert "sk-get-list-111" not in found["api_key"]
    assert found["api_key"] != "sk-get-list-111"


# ── POST /platforms/ollama-cloud/accounts ──────────────────────────────────

def test_add_account_returns_201(client):
    """POST adds account and returns 201 with masked api_key."""
    response = client.post(
        "/platforms/ollama-cloud/accounts",
        json={"api_key": "sk-add-test-222", "label": "AddAccount"},
    )
    assert response.status_code == 201
    body = response.json()
    assert "id" in body
    assert body["label"] == "AddAccount"
    assert "sk-" in body["api_key"]
    assert "sk-add-test-222" not in body["api_key"]


# ── PATCH /platforms/ollama-cloud/accounts/{id} ────────────────────────────

def test_patch_account_updates_label(client):
    """PATCH updates label without exposing api_key."""
    # Add
    add_resp = client.post(
        "/platforms/ollama-cloud/accounts",
        json={"api_key": "sk-patch-test-333", "label": "OldLabel"},
    )
    assert add_resp.status_code == 201
    account_id = add_resp.json()["id"]

    # Patch
    response = client.patch(
        f"/platforms/ollama-cloud/accounts/{account_id}",
        json={"label": "NewLabel"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "NewLabel"
    # api_key must be masked
    assert "sk-" in body["api_key"]
    assert "sk-patch-test-333" not in body["api_key"]


def test_patch_nonexistent_account_returns_404(client):
    """PATCH for nonexistent account returns 404."""
    response = client.patch(
        "/platforms/ollama-cloud/accounts/nonexistent-id",
        json={"label": "NewLabel"},
    )
    assert response.status_code == 404


# ── DELETE /platforms/ollama-cloud/accounts/{id} ───────────────────────────

def test_delete_account_removes_it(client):
    """DELETE removes account and verifies it's gone."""
    # Add
    add_resp = client.post(
        "/platforms/ollama-cloud/accounts",
        json={"api_key": "sk-del-test-444", "label": "ToDelete"},
    )
    assert add_resp.status_code == 201
    account_id = add_resp.json()["id"]

    # Delete
    response = client.delete(f"/platforms/ollama-cloud/accounts/{account_id}")
    assert response.status_code == 200
    assert response.json()["message"] == "Conta removida"

    # Verify it's gone
    get_resp = client.get("/platforms/ollama-cloud/accounts")
    accounts = get_resp.json()["accounts"]
    assert not any(a["id"] == account_id for a in accounts)


def test_delete_nonexistent_account_returns_404(client):
    """DELETE for nonexistent account returns 404."""
    response = client.delete("/platforms/ollama-cloud/accounts/nonexistent-id")
    assert response.status_code == 404


# ── POST /platforms/ollama-cloud/accounts/{id}/validate ────────────────────

def test_validate_connection(client):
    """POST validate returns valid status (may be False in test env)."""
    add_resp = client.post(
        "/platforms/ollama-cloud/accounts",
        json={"api_key": "sk-val-test-555", "label": "Validate"},
    )
    assert add_resp.status_code == 201
    account_id = add_resp.json()["id"]

    response = client.post(f"/platforms/ollama-cloud/accounts/{account_id}/validate")
    assert response.status_code == 200
    body = response.json()
    assert "valid" in body
    assert "status" in body


def test_validate_nonexistent_account_returns_404(client):
    """POST validate for nonexistent account returns 404."""
    response = client.post("/platforms/ollama-cloud/accounts/nonexistent-id/validate")
    assert response.status_code == 404


# ── POST catalog/refresh ───────────────────────────────────────────────────

def test_catalog_refresh(client):
    """POST catalog/refresh returns status and model count."""
    response = client.post("/platforms/ollama-cloud/catalog/refresh")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in ("fresh", "stale", "error")
    assert "models_count" in body
    assert isinstance(body["models_count"], int)
