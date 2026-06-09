"""Testes do servidor mock E2E."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(mock_client: TestClient) -> TestClient:
    return mock_client


def test_mock_login_sets_session_cookie(client: TestClient) -> None:
    response = client.post(
        "/api/auth/login",
        json={"username": "any", "password": "any"},
    )
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert "session_token=" in response.headers.get("set-cookie", "")


def test_status_reflects_start_and_stop(client: TestClient) -> None:
    assert client.get("/status").json()["running"] is False

    client.post(
        "/start",
        json={"path": "/models/llama/llama-3.1-8b.gguf", "gpu_weights": []},
    )
    status = client.get("/status").json()
    assert status["running"] is True
    assert status["model"] == "llama-3.1-8b.gguf"
    assert "config" in status

    client.post("/stop")
    assert client.get("/status").json()["running"] is False


def test_metrics_returns_fake_gpu_data(client: TestClient) -> None:
    data = client.get("/metrics").json()
    assert data["cpu"] == 42.5
    assert len(data["gpus"]) == 1
    assert data["gpus"][0]["index"] == 0


def test_models_returns_fake_list(client: TestClient) -> None:
    data = client.get("/models").json()
    assert len(data["models"]) >= 2
    assert len(data["projectors"]) >= 1
    llama = next(m for m in data["models"] if m["name"] == "llama-3.1-8b.gguf")
    assert llama["mmproj_candidates"] == ["/models/llama/llama-3.1-8b-mmproj.gguf"]
    mistral = next(m for m in data["models"] if m["name"] == "mistral-7b.gguf")
    assert mistral["mmproj_candidates"] == []


def test_models_mmproj_persists_selection(client: TestClient) -> None:
    path = "/models/llama/llama-3.1-8b.gguf"
    mmproj = "/models/llama/llama-3.1-8b-mmproj.gguf"
    response = client.post(
        "/models/mmproj",
        json={"model_path": path, "mmproj_path": mmproj},
    )
    assert response.status_code == 200
    assert response.json()["mmproj_path"] == mmproj
    config = client.get("/config").json()
    assert config["model_configs"][path]["mmproj_path"] == mmproj


def test_download_with_model_path_adds_projector(client: TestClient) -> None:
    response = client.post(
        "/downloads",
        json={
            "url": "https://example.com/mistral-vision.mmproj",
            "model_path": "/models/text/mistral-7b.gguf",
        },
    )
    assert response.status_code == 200
    data = client.get("/models").json()
    mistral = next(m for m in data["models"] if m["name"] == "mistral-7b.gguf")
    assert "/models/text/mistral-vision.mmproj" in mistral["mmproj_candidates"]


def test_logs_sse_stream(client: TestClient) -> None:
    response = client.get("/logs")
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    assert "[INFO] llama server started" in response.text


_FAKE_PATH = "/models/llama/llama-3.1-8b.gguf"


@pytest.mark.parametrize(
    "method,path,body",
    [
        ("POST", "/rename", {"path": _FAKE_PATH, "new_name": "renamed-model"}),
        ("POST", "/delete", {"path": "/models/text/mistral-7b.gguf"}),
        ("POST", "/set_default", {"path": _FAKE_PATH}),
    ],
)
def test_mutations_return_ok(
    client: TestClient, method: str, path: str, body: dict
) -> None:
    response = client.request(method, path, json=body)
    assert response.status_code == 200
    assert response.json().get("ok") is True


def test_protected_routes_work_without_credentials(client: TestClient) -> None:
    assert client.get("/status").status_code == 200
    assert client.get("/metrics").status_code == 200
