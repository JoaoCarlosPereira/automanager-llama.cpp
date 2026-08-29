"""Shared request/response schemas."""
from typing import Dict, List, Literal, Optional, Union
from urllib.parse import urlparse
from pydantic import BaseModel, ConfigDict, Field, constr, field_validator
DEFAULT_CONTEXT_SIZE = 65536
DEFAULT_PARALLEL_SLOTS = 1
DEFAULT_BATCH_SIZE = 2048
DEFAULT_CACHE_TYPE = "f16"
DEFAULT_MTP_ENABLED = False
DEFAULT_MTP_DRAFT_TOKENS = 3
DEFAULT_FLASH_ATTN_ENABLED = True
DEFAULT_PROXY_ELIGIBLE = True
DEFAULT_MAX_PARALLEL_REQUESTS = 1
DEFAULT_PROXY_TTL_MINUTES = 180
DEFAULT_PROXY_MAX_WAIT_SECONDS = 30
BATCH_SIZE_PRESETS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
CACHE_TYPE_PRESETS = ["f16", "q8_0", "q4_0"]

TURBOQUANT_CACHE_K_PRESETS = ["f16", "q8_0"]
TURBOQUANT_CACHE_V_PRESETS = ["turbo4", "turbo3", "turbo2"]
TURBOQUANT_DEFAULT_CACHE_K = "q8_0"
TURBOQUANT_DEFAULT_CACHE_V = "turbo3"
TURBOQUANT_PRESETS = [
    {
        "id": "safest",
        "label": "Mais seguro (f16 / turbo4)",
        "cache_type_k": "f16",
        "cache_type_v": "turbo4",
    },
    {
        "id": "conservative",
        "label": "Conservador (q8_0 / turbo4)",
        "cache_type_k": "q8_0",
        "cache_type_v": "turbo4",
    },
    {
        "id": "recommended",
        "label": "Recomendado (q8_0 / turbo3)",
        "cache_type_k": "q8_0",
        "cache_type_v": "turbo3",
    },
    {
        "id": "aggressive",
        "label": "Agressivo (q8_0 / turbo2)",
        "cache_type_k": "q8_0",
        "cache_type_v": "turbo2",
    },
]


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
    flash_attn_enabled: bool = DEFAULT_FLASH_ATTN_ENABLED
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
    mtp_model_path: Optional[str] = None
    total_layers: int = 0  # 0 = auto-detect from model file
    cpu_enabled: Optional[bool] = None  # None = proporção da UI; True/False = válvula LoadDistributor
    port: Optional[int] = None  # Porta específica para o modelo
    auto_balance_profile: Optional[bool] = None  # None = False no /start; True após aplicar proposta
    llama_server_bin: Optional[str] = None  # Binário llama-server para este modelo
    turboquant_preset: Optional[str] = None  # Preset TurboQuant+ (UI / persistência por modelo)
    mmproj_disabled: bool = False  # True = usuário escolheu explicitamente "Sem visão"
    vision_enabled: Optional[bool] = None  # Preferência local de Vision; None = legado


class DeleteRequest(BaseModel):
    path: str


class DownloadRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    url: str
    filename: Optional[str] = None
    model_path: Optional[str] = None
    asset_type: Literal["model", "mmproj", "mtp"] = "model"


class DownloadCancelRequest(BaseModel):
    download_id: str


class SetMmprojRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_path: str
    mmproj_path: Optional[str] = None
    user_initiated: bool = False


class SetMtpModelRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_path: str
    mtp_model_path: Optional[str] = None


class SetThinkingRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_path: str
    thinking_enabled: bool


class SetLlamaBinRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_path: str
    llama_server_bin: Optional[str] = None
    cache_type_k: Optional[str] = None
    cache_type_v: Optional[str] = None
    turboquant_preset: Optional[str] = None


class SetDefaultRequest(BaseModel):
    path: Optional[str] = None
    add: bool = True


class TokenizerReference(BaseModel):
    """Referência explícita a um tokenizer Hugging Face."""
    identifier: str = Field(..., min_length=1)
    revision: Optional[str] = Field(default=None, min_length=1)


TokenizerMapping = Union[str, TokenizerReference]


class TokenizerMappings(BaseModel):
    """Mapeamentos explícitos por modelo exato e família."""
    models: Dict[constr(min_length=1), TokenizerMapping] = Field(default_factory=dict)
    families: Dict[constr(min_length=1), TokenizerMapping] = Field(default_factory=dict)


class ContextOptimizerConfig(BaseModel):
    """Atualização parcial da configuração administrativa do otimizador."""
    enabled: Optional[bool] = None
    audit_enabled: Optional[bool] = None
    tokenizers: Optional[TokenizerMappings] = None


class ProxyConfigRequest(BaseModel):
    """Atualização parcial da chave global smart_proxy."""
    enabled: Optional[bool] = None
    primary_model_path: Optional[str] = None
    primary_backend_id: Optional[str] = None
    ttl_minutes: Optional[int] = Field(default=None, ge=1)
    max_wait_seconds: Optional[int] = Field(default=None, ge=1)
    context_optimizer: Optional[ContextOptimizerConfig] = None
    custom_priority: Optional[List[str]] = None


class SetModelProxyRequest(BaseModel):
    """Flags de participação no proxy inteligente, por modelo."""
    model_config = ConfigDict(protected_namespaces=())

    model_path: Optional[str] = None
    backend_id: Optional[str] = None
    proxy_eligible: Optional[bool] = None
    vision_enabled: Optional[bool] = None
    max_parallel_requests: Optional[int] = Field(default=None, ge=1)
    auto_start: Optional[bool] = None
    default_model: Optional[str] = None


class RenameRequest(BaseModel):
    path: str
    new_name: str


class SetModelsDirRequest(BaseModel):
    models_dir: str = Field(..., min_length=1)


class LoginRequest(BaseModel):
    username: str
    password: str


class CLIProxyAuthStartRequest(BaseModel):
    method: Optional[str] = None


class CLIProxyAuthCallbackRequest(BaseModel):
    callback_url: str = Field(..., min_length=8)


class ModelAliasRequest(BaseModel):
    """Alias externo (ex.: gpt-4o no Cursor) -> modelo real no backend."""
    alias: str = Field(..., min_length=1)
    target: Optional[str] = None


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


class PartialConfigResponse(BaseModel):
    """Resposta metadata-only da configuração parcial administrativa."""
    smart_proxy: Optional[dict] = None
    context_optimizer: Optional[dict] = None
    tokenizers_mapping_count: int = 0
    model_configs_count: int = 0


class AuditLogItem(BaseModel):
    """Item de log de auditoria do Context Optimizer (metadata-only)."""
    strategy: str = ""
    original_cost: int = 0
    optimized_cost: int = 0
    savings_tokens: int = 0
    transformations_applied: List[str] = Field(default_factory=list)
    protected_units_preserved: int = 0
    blocks_removed: int = 0
    blocks_merged: int = 0
    blocks_deduplicated: int = 0
    validation_passed: bool = True
    validation_errors: List[str] = Field(default_factory=list)
    duration_ms: float = 0.0
    ts: float = 0.0
    model: Optional[str] = None


class AuditPaginatedResponse(BaseModel):
    """Resposta paginada do audit log do Context Optimizer."""
    page: int
    per_page: int
    total: int
    pages: int
    items: List[AuditLogItem] = Field(default_factory=list)


class StatusMetrics(BaseModel):
    """Métricas agregadas incluídas no status expandido."""
    cpu_percent: Optional[float] = None
    memory_percent: Optional[float] = None
    gpu_count: int = 0
    gpu_utilization: Optional[float] = None
    total_vram: int = 0
    used_vram: int = 0
    tokenizer_estimates: int = 0
    optimizer_audit_entries: int = 0


class GenericOpenAIAddAccountRequest(BaseModel):
    name: str = Field(..., min_length=1)
    base_url: str = Field(..., min_length=1)
    api_key: str = Field(..., min_length=1)

    @field_validator("name", "base_url", "api_key")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("campo obrigatório")
        return value

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("base_url deve ser uma URL HTTP(S) utilizável")
        return value.rstrip("/")

class GenericOpenAIUpdateAccountRequest(BaseModel):
    name: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    status: Optional[str] = None
    cooldown_until: Optional[float] = None
    rate_limited_until: Optional[float] = None

    @field_validator("name", "base_url", "api_key")
    @classmethod
    def validate_optional_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        value = value.strip()
        if not value:
            raise ValueError("não é permitido valor vazio")
        return value

    @field_validator("base_url")
    @classmethod
    def validate_optional_base_url(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        parsed = urlparse(value)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("base_url deve ser uma URL HTTP(S) utilizável")
        return value.rstrip("/")
