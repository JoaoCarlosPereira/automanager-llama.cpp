"""Shared request/response schemas."""
from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field
DEFAULT_CONTEXT_SIZE = 65536
DEFAULT_PARALLEL_SLOTS = 1
DEFAULT_BATCH_SIZE = 2048
DEFAULT_CACHE_TYPE = "f16"
DEFAULT_MTP_ENABLED = False
DEFAULT_MTP_DRAFT_TOKENS = 3
BATCH_SIZE_PRESETS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
CACHE_TYPE_PRESETS = ["f16", "q8_0", "q4_0"]


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
    ubatch_size: int = Field(default=512, ge=32, le=65536)
    cache_type_k: str = DEFAULT_CACHE_TYPE
    cache_type_v: str = DEFAULT_CACHE_TYPE
    numa_enabled: bool = False
    threads: int = 0  # 0 = auto
    threads_batch: int = 0  # 0 = auto
    split_mode: str = "layer"
    auto_balance: bool = False
    smart_calibration: bool = False
    pinned_fields: Optional[dict] = None
    manual_gpu_override: bool = False
    thinking_enabled: bool = True
    mtp_enabled: bool = DEFAULT_MTP_ENABLED
    mtp_draft_tokens: int = DEFAULT_MTP_DRAFT_TOKENS
    total_layers: int = 0  # 0 = auto-detect from model file
    cpu_enabled: Optional[bool] = None  # None = proporção da UI; True/False = válvula LoadDistributor
    port: Optional[int] = None  # Porta específica para o modelo
    auto_balance_profile: Optional[bool] = None  # None = False no /start; True após aplicar proposta


class DeleteRequest(BaseModel):
    path: str


class DownloadRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    url: str
    filename: Optional[str] = None
    model_path: Optional[str] = None


class DownloadCancelRequest(BaseModel):
    download_id: str


class SetMmprojRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_path: str
    mmproj_path: Optional[str] = None


class SetThinkingRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_path: str
    thinking_enabled: bool


class SetDefaultRequest(BaseModel):
    path: Optional[str] = None
    add: bool = True


class RenameRequest(BaseModel):
    path: str
    new_name: str


class SetModelsDirRequest(BaseModel):
    models_dir: str = Field(..., min_length=1)


class LoginRequest(BaseModel):
    username: str
    password: str


class VersionCommit(BaseModel):
    sha: str
    message: str
    author: str
    date: str


class VersionCheckResponse(BaseModel):
    status: Literal["ok", "unavailable", "error"]
    update_available: bool = False
    current_ref: Optional[str] = None
    remote_ref: Optional[str] = None
    branch: Optional[str] = None
    commits: List[VersionCommit] = Field(default_factory=list)
    error_message: Optional[str] = None
