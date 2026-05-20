from pathlib import Path
from unittest.mock import MagicMock

import pytest

from config_manager import AuthManager, ConfigManager, TokenManager


@pytest.fixture
def tmp_config_path(tmp_path: Path) -> Path:
    return tmp_path / "automanager_config.json"


@pytest.fixture
def tmp_config_manager(tmp_config_path: Path) -> ConfigManager:
    return ConfigManager(str(tmp_config_path))


@pytest.fixture
def fake_models_dir(tmp_path: Path) -> Path:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "llama-test.gguf").write_bytes(b"fake gguf model")
    (models_dir / "llava-test.gguf").write_bytes(b"fake vision model")
    (models_dir / "llava-test.mmproj").write_bytes(b"fake projector")
    return models_dir


@pytest.fixture
def fake_model_path(fake_models_dir: Path) -> Path:
    return fake_models_dir / "llama-test.gguf"


@pytest.fixture
def llama_server_help_output() -> str:
    return "\n".join(
        [
            "Device 0: NVIDIA RTX 4090, compute capability 8.9, VRAM: 24564 MiB",
            "Device 1: NVIDIA RTX 3090, compute capability 8.6, VRAM: 24576 MiB",
        ]
    )


@pytest.fixture
def nvidia_smi_gpu_output() -> str:
    return "\n".join(
        [
            "0, NVIDIA RTX 4090, 24564",
            "1, NVIDIA RTX 3090, 24576",
        ]
    )


@pytest.fixture
def nvidia_smi_metrics_output() -> str:
    return "\n".join(
        [
            "0, 42, 8192, 24564, 61, 240.50",
            "1, 7, 1024, 24576, 45, 95.25",
        ]
    )


@pytest.fixture
def token_manager(tmp_config_manager: ConfigManager) -> TokenManager:
    return TokenManager(tmp_config_manager)


@pytest.fixture
def auth_manager(
    tmp_config_manager: ConfigManager,
    token_manager: TokenManager,
) -> AuthManager:
    return AuthManager(tmp_config_manager, token_manager)


@pytest.fixture
def valid_api_token(token_manager: TokenManager) -> str:
    return token_manager.get_or_create()


@pytest.fixture
def mock_http_credentials(valid_api_token: str) -> MagicMock:
    credentials = MagicMock()
    credentials.credentials = valid_api_token
    return credentials
