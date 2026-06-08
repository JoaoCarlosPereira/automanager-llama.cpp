"""Shared request/response schemas."""
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
DEFAULT_CONTEXT_SIZE = 65536
DEFAULT_PARALLEL_SLOTS = 1
DEFAULT_BATCH_SIZE = 2048
DEFAULT_MTP_ENABLED = False
DEFAULT_MTP_DRAFT_TOKENS = 3
MTP_DRAFT_TOKENS_MIN = 1
MTP_DRAFT_TOKENS_MAX = 6
BATCH_SIZE_PRESETS = [128, 256, 512, 1024, 2048, 4096, 8192]


class GPUWeight(BaseModel):
    index: int
    weight: float
    name: str
    active: bool = True
    is_main: bool = False
    pinned: bool = False
    device: Literal["gpu", "cpu"] = "gpu"


class StartRequest(BaseModel):
    path: str
    mmproj_path: Optional[str] = None
    gpu_weights: List[GPUWeight]
    context_size: int = DEFAULT_CONTEXT_SIZE
    parallel_slots: int = Field(default=DEFAULT_PARALLEL_SLOTS, ge=1, le=64)
    batch_size: int = Field(default=DEFAULT_BATCH_SIZE, ge=32, le=65536)
    split_mode: str = "layer"
    auto_balance: bool = False
    manual_gpu_override: bool = False
    thinking_enabled: bool = True
    mtp_enabled: bool = DEFAULT_MTP_ENABLED
    mtp_draft_tokens: int = Field(
        default=DEFAULT_MTP_DRAFT_TOKENS,
        ge=MTP_DRAFT_TOKENS_MIN,
        le=MTP_DRAFT_TOKENS_MAX,
    )
    total_layers: int = 0  # 0 = auto-detect from model file


class DeleteRequest(BaseModel):
    path: str


class DownloadRequest(BaseModel):
    url: str
    filename: Optional[str] = None


class SetDefaultRequest(BaseModel):
    path: Optional[str] = None


class RenameRequest(BaseModel):
    path: str
    new_name: str


class SetModelsDirRequest(BaseModel):
    models_dir: str = Field(..., min_length=1)


class LoginRequest(BaseModel):
    username: str
    password: str
