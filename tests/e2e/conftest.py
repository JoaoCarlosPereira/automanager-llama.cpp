"""Fixtures pytest para o servidor mock E2E."""

import pytest
from fastapi.testclient import TestClient

from llama_manager import app
from tests.e2e.mock_server import apply_mocks, remove_mocks, reset_mock_state


@pytest.fixture
def mock_client() -> TestClient:
    """TestClient com rotas mock e autenticação desabilitada via override."""
    apply_mocks()
    client = TestClient(app)
    yield client
    remove_mocks()
