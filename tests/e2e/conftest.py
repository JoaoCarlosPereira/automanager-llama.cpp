"""Fixtures pytest para o servidor mock E2E."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from llama_manager import app, limiter
from tests.e2e.mock_server import apply_mocks, remove_mocks, reset_mock_state


@pytest.fixture
def mock_client() -> TestClient:
    """TestClient com rotas mock, autenticação desabilitada e rate limit desligado."""
    apply_mocks()
    with patch.object(limiter, 'enabled', False):
        client = TestClient(app)
        yield client
    remove_mocks()
