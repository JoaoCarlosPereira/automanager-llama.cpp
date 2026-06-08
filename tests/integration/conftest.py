"""Isola testes de integração do estado global alterado por apply_mocks (E2E)."""

import pytest

from tests.e2e.mock_server import remove_mocks


@pytest.fixture(autouse=True)
def restore_llama_app_after_e2e_mocks():
    remove_mocks()
    yield
    remove_mocks()
