"""Classes de orçamento, limites, capacidades e resolvedores do Context Optimizer."""

import asyncio
import copy
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union


class ContextTooLargeError(Exception):
    """Estouro do limite de contexto no Context Optimizer."""

    def __init__(
        self,
        message: str = "Prompt excede o limite de contexto do modelo",
        code: str = "context_too_large",
    ):
        self.status_code = 413
        self.message = message
        self.code = code
        super().__init__(message)

    def payload(self) -> dict:
        return {"error": {"message": self.message, "type": "proxy_error", "code": self.code}}


DEFAULT_OUTPUT_RESERVE_FALLBACK = 2048
DEFAULT_PROTOCOL_OVERHEAD = 512
DEFAULT_SAFETY_MARGIN = 256
BACKOFF_BASE_SECONDS = 30.0
AUDIT_LOG_ROTATION_MAX_BYTES = 10 * 1024 * 1024
AUDIT_LOG_ROTATION_BACKUP_COUNT = 5

UNKNOWN_PLATFORM_CONTEXT_LIMIT = (1 << 63) - 1


_SOCIAL_NOISE_RE = re.compile(
    r"^(ob+rigad[oa]+|valeu|ok|certo|entendido|perfeito|pode continuar|tudo bem"
    r"|ob+rigad[oa]+,?\s*pode continuar|entendido,?\s*continue|estou bem,?\s*ob+rigad[oa]+!?"
    r"|muito ob+rigad[oa]+|ob+rigad[oa]+ pelo retorno|beleza|joia)[.!?]?$",
    re.IGNORECASE,
)

_TECHNICAL_DECISION_RE = re.compile(
    r"(usar\s+\w+|usando\s+\w+|postgresql|postgres|mysql|sqlite|redis|mongodb"
    r"|decisã[oó]|definid[oa]|requisito|arquitetura|mudança\s+(de\s+)?requisito"
    r"|confirmad[oa]|optamos\s+por|mudar\s+banco\s+para|alterar\s+para|mantendo\s+o"
    r"|usar\s+\w+\s+ao\s+invés\s+de)",
    re.IGNORECASE,
)

_CODE_SQL_URL_RE = re.compile(
    r"(```|select\s+.*\s+from|insert\s+into|update\s+.*\s+set|create\s+table|delete\s+from"
    r"|https?://|/\w+/\w+|\w+\.(py|js|ts|json|sh|sql|txt|md|yml|yaml|css|html)|[0-9a-f]{8}-[0-9a-f]{4})",
    re.IGNORECASE,
)

_LOG_IMPORTANT_RE = re.compile(
    r"(error|warn|warning|traceback|exception|fatal|critical|failed)",
    re.IGNORECASE,
)


def _extract_text_content(msg: Dict[str, Any]) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return " ".join(parts)
    return ""


def _classify_block_retention(
    msg: Dict[str, Any],
    role: str,
    kind: str,
    is_already_protected: bool,
) -> Tuple[str, bool]:
    """Classifica a retenção de um bloco e determina se deve ser protegido."""
    if role in ("system", "developer"):
        return ("system", True)

    if kind in ("multimodal_message", "file_content"):
        return ("media_file", True)

    text = _extract_text_content(msg).strip()

    if is_already_protected:
        if text and _TECHNICAL_DECISION_RE.search(text):
            return ("technical_decision", True)
        elif text and _CODE_SQL_URL_RE.search(text):
            return ("code_media_file", True)
        return ("current_turn", True)

    if not text:
        return ("normal", is_already_protected)

    if _SOCIAL_NOISE_RE.match(text) and not _CODE_SQL_URL_RE.search(text) and not _TECHNICAL_DECISION_RE.search(text):
        return ("social_noise", False)

    if _TECHNICAL_DECISION_RE.search(text):
        return ("technical_decision", True)

    if _CODE_SQL_URL_RE.search(text):
        return ("code_media_file", True)

    if "\n" in text or "log" in text.lower() or "[" in text:
        if _LOG_IMPORTANT_RE.search(text):
            return ("log_important", True)

    return ("normal", False)


AUDIT_ALLOWLIST_FIELDS: FrozenSet[str] = frozenset({
    "strategy",
    "original_cost",
    "optimized_cost",
    "savings_tokens",
    "transformations_applied",
    "protected_units_preserved",
    "blocks_removed",
    "blocks_merged",
    "blocks_deduplicated",
    "validation_passed",
    "validation_errors",
    "duration_ms",
    "ts",
    "model",
})


class AuditRecorder:
    """Registrador de auditoria metadata-only em JSONL rotativo.

    Escreve apenas metadados de ``OptimizationAudit`` — nunca o payload
    original.  Arquivos rotacionam por tamanho e mantêm ``backupCount``
    rotação em ``.1``, ``.2``, …
    """

    def __init__(self, log_dir: str, max_bytes: int = AUDIT_LOG_ROTATION_MAX_BYTES,
                 backup_count: int = AUDIT_LOG_ROTATION_BACKUP_COUNT) -> None:
        self._log_dir = log_dir
        self._max_bytes = max_bytes
        self._backup_count = backup_count
        self._lock = threading.Lock()
        self._current_fd = None
        self._current_size = 0
        try:
            os.makedirs(self._log_dir, exist_ok=True)
            self._current_path = os.path.join(self._log_dir, "audit.jsonl")
            self._current_fd = open(self._current_path, "a", encoding="utf-8")
            self._current_size = os.path.getsize(self._current_path)
        except OSError:
            # Fallback seguro caso o diretório de logs não seja gravável no ambiente
            fallback_dir = "/tmp/automanager_audit_logs"
            try:
                os.makedirs(fallback_dir, exist_ok=True)
                self._log_dir = fallback_dir
                self._current_path = os.path.join(self._log_dir, "audit.jsonl")
                self._current_fd = open(self._current_path, "a", encoding="utf-8")
                self._current_size = os.path.getsize(self._current_path)
            except OSError:
                self._current_fd = None

    def _should_rotate(self) -> None:
        if self._current_size >= self._max_bytes:
            self._current_fd.close()
            self._rotate()

    def _rotate(self) -> None:
        for i in range(self._backup_count - 1, 0, -1):
            src = f"{self._current_path}.{i}"
            dst = f"{self._current_path}.{i + 1}"
            try:
                if os.path.exists(src):
                    os.replace(src, dst)
            except OSError:
                pass
        try:
            if os.path.exists(self._current_path):
                os.replace(self._current_path, f"{self._current_path}.1")
        except OSError:
            pass
        self._current_fd = open(self._current_path, "w", encoding="utf-8")
        self._current_size = 0

    def record(self, audit: "OptimizationAudit", extra: Optional[Dict[str, Any]] = None) -> None:
        """Grava um registro metadata-only no arquivo JSONL, filtrando via allowlist."""
        raw_record: Dict[str, Any] = {
            "strategy": audit.strategy,
            "original_cost": audit.original_cost,
            "optimized_cost": audit.optimized_cost,
            "savings_tokens": audit.savings_tokens,
            "transformations_applied": audit.transformations_applied,
            "protected_units_preserved": audit.protected_units_preserved,
            "blocks_removed": audit.blocks_removed,
            "blocks_merged": audit.blocks_merged,
            "blocks_deduplicated": audit.blocks_deduplicated,
            "validation_passed": audit.validation_passed,
            "validation_errors": audit.validation_errors,
            "duration_ms": audit.duration_ms,
            "ts": time.time(),
        }
        if extra:
            for k, v in extra.items():
                if k in AUDIT_ALLOWLIST_FIELDS:
                    raw_record[k] = v

        # Strict allowlist filtering to prevent accidental payload leaking
        filtered_record = {
            k: v for k, v in raw_record.items() if k in AUDIT_ALLOWLIST_FIELDS
        }

        line = json.dumps(filtered_record, ensure_ascii=False) + "\n"
        with self._lock:
            if not self._current_fd or self._current_fd.closed:
                return
            self._should_rotate()
            self._current_fd.write(line)
            self._current_fd.flush()
            self._current_size += len(line.encode("utf-8"))

    def query(
        self,
        page: int = 1,
        per_page: int = 50,
        strategy_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Serviço de consulta paginada sem vazar payloads.

        Lê o arquivo de auditoria JSONL e aplica a allowlist strict antes de
        retornar os itens.
        """
        if page < 1:
            page = 1

        records: List[Dict[str, Any]] = []
        with self._lock:
            if os.path.exists(self._current_path):
                try:
                    with open(self._current_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                data = json.loads(line)
                                if not isinstance(data, dict):
                                    continue
                                # Filter strictly against allowlist
                                filtered = {
                                    k: v for k, v in data.items() if k in AUDIT_ALLOWLIST_FIELDS
                                }
                                if strategy_filter and filtered.get("strategy") != strategy_filter:
                                    continue
                                records.append(filtered)
                            except json.JSONDecodeError:
                                pass
                except OSError:
                    pass

        records.reverse()  # Mais recentes primeiro
        total = len(records)
        start = (page - 1) * per_page
        end = start + per_page

        return {
            "page": page,
            "per_page": per_page,
            "total": total,
            "pages": max(1, (total + per_page - 1) // per_page),
            "items": records[start:end],
        }

    def close(self) -> None:
        with self._lock:
            if self._current_fd and not self._current_fd.closed:
                self._current_fd.close()


class TokenCountSource(str, Enum):
    EXACT_MODEL = "exact_model"
    FAMILY = "family"
    ESTIMATED = "estimated"


@dataclass(frozen=True)
class TokenCount:
    tokens: int
    source: TokenCountSource
    tokenizer_ref: Optional[str] = None


class ConservativeEstimator:
    """Estimador conservador sem dependências externas."""

    @staticmethod
    def estimate_text(text: str) -> int:
        if not text:
            return 0
        b_len = len(text.encode("utf-8"))
        return max(1, int(b_len * 0.35) + 1)

    @classmethod
    def estimate_payload(cls, payload: Any) -> int:
        if payload is None:
            return 0
        if isinstance(payload, str):
            return cls.estimate_text(payload)
        if isinstance(payload, (int, float, bool)):
            return 1
        if isinstance(payload, list):
            return sum(cls.estimate_payload(item) for item in payload) + len(payload)
        if isinstance(payload, dict):
            total = 0
            for k, v in payload.items():
                total += cls.estimate_text(str(k)) + cls.estimate_payload(v) + 2
            return total
        return cls.estimate_text(str(payload))


class TokenizerRegistry:
    """Gerenciador de tokenizers Hugging Face com cache e download background."""

    def __init__(self, fetcher: Optional[Any] = None):
        self._cache: Dict[Tuple[str, Optional[str]], Any] = {}
        self._download_locks: Dict[Tuple[str, Optional[str]], asyncio.Lock] = {}
        self._failed_downloads: Dict[Tuple[str, Optional[str]], float] = {}
        self._lock = asyncio.Lock()
        self._fetcher = fetcher

    async def get_count(
        self,
        payload: Any,
        model_name: Optional[str] = None,
        family_name: Optional[str] = None,
        configured_mappings: Optional[Dict[str, Any]] = None,
    ) -> TokenCount:
        mappings = configured_mappings or {}
        models_map = mappings.get("models", {}) if isinstance(mappings, dict) else {}
        families_map = mappings.get("families", {}) if isinstance(mappings, dict) else {}

        target_ref: Optional[Tuple[str, Optional[str]]] = None
        found_source: Optional[TokenCountSource] = None

        if model_name and model_name in models_map:
            target_ref = self._normalize_ref(models_map[model_name])
            found_source = TokenCountSource.EXACT_MODEL
        elif family_name and family_name in families_map:
            target_ref = self._normalize_ref(families_map[family_name])
            found_source = TokenCountSource.FAMILY

        if not target_ref:
            est_tokens = ConservativeEstimator.estimate_payload(payload)
            return TokenCount(tokens=est_tokens, source=TokenCountSource.ESTIMATED)

        tokenizer = self._cache.get(target_ref)
        if tokenizer is not None:
            tokens = await asyncio.to_thread(self._tokenize_with_hf, tokenizer, payload)
            ref_str = f"{target_ref[0]}:{target_ref[1]}" if target_ref[1] else target_ref[0]
            return TokenCount(tokens=tokens, source=found_source, tokenizer_ref=ref_str)

        asyncio.create_task(self._trigger_background_download(target_ref))

        est_tokens = ConservativeEstimator.estimate_payload(payload)
        ref_str = f"{target_ref[0]}:{target_ref[1]}" if target_ref[1] else target_ref[0]
        return TokenCount(tokens=est_tokens, source=TokenCountSource.ESTIMATED, tokenizer_ref=ref_str)

    @staticmethod
    def _normalize_ref(val: Any) -> Tuple[str, Optional[str]]:
        if isinstance(val, str):
            return (val, None)
        if isinstance(val, dict):
            return (val.get("identifier", ""), val.get("revision"))
        if hasattr(val, "identifier"):
            return (getattr(val, "identifier"), getattr(val, "revision", None))
        return (str(val), None)

    async def _trigger_background_download(self, ref: Tuple[str, Optional[str]]) -> None:
        last_failed = self._failed_downloads.get(ref)
        if last_failed and (time.time() - last_failed < BACKOFF_BASE_SECONDS):
            return

        async with self._lock:
            if ref not in self._download_locks:
                self._download_locks[ref] = asyncio.Lock()
            dl_lock = self._download_locks[ref]

        if dl_lock.locked():
            return

        async with dl_lock:
            if ref in self._cache:
                return
            try:
                tokenizer = await asyncio.to_thread(self._load_tokenizer_sync, ref)
                if tokenizer:
                    self._cache[ref] = tokenizer
                    self._failed_downloads.pop(ref, None)
                else:
                    self._failed_downloads[ref] = time.time()
            except Exception:
                self._failed_downloads[ref] = time.time()

    def _load_tokenizer_sync(self, ref: Tuple[str, Optional[str]]) -> Any:
        if self._fetcher:
            return self._fetcher(ref[0], ref[1])
        try:
            from transformers import AutoTokenizer
            identifier, revision = ref
            return AutoTokenizer.from_pretrained(identifier, revision=revision)
        except Exception:
            return None

    @staticmethod
    def _tokenize_with_hf(tokenizer: Any, payload: Any) -> int:
        text_content = ""
        if isinstance(payload, str):
            text_content = payload
        elif isinstance(payload, dict):
            text_content = json.dumps(payload, ensure_ascii=False)
        else:
            text_content = str(payload)

        try:
            tokens = tokenizer.encode(text_content)
            return len(tokens)
        except Exception:
            return ConservativeEstimator.estimate_payload(payload)

DEFAULT_OUTPUT_RESERVE_FALLBACK = 2048
DEFAULT_PROTOCOL_OVERHEAD = 512
DEFAULT_SAFETY_MARGIN = 256

UNKNOWN_PLATFORM_CONTEXT_LIMIT = (1 << 63) - 1


class LimitConfidence(str, Enum):
    KNOWN_LOCAL = "known_local"
    KNOWN_PROVIDER = "known_provider"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ModelLimits:
    context_tokens: Optional[int]
    max_output_tokens: Optional[int]
    source: str
    confidence: LimitConfidence

    @property
    def is_known(self) -> bool:
        return self.confidence != LimitConfidence.UNKNOWN and self.context_tokens is not None and self.context_tokens > 0


@dataclass(frozen=True)
class RequiredCapabilities:
    text: bool = True
    vision: bool = False
    tools: bool = False
    structured_output: bool = False
    files: bool = False

    def as_set(self) -> FrozenSet[str]:
        caps = set()
        if self.text:
            caps.add("text")
        if self.vision:
            caps.add("vision")
        if self.tools:
            caps.add("tools")
        if self.structured_output:
            caps.add("structured_output")
        if self.files:
            caps.add("files")
        return frozenset(caps)

    def is_subset_of(self, available: FrozenSet[str]) -> bool:
        return self.as_set().issubset(available)


@dataclass(frozen=True)
class TargetBudget:
    context_limit: Optional[int]
    output_reserve: int
    protocol_overhead: int
    safety_margin: int
    input_budget: Optional[int]
    confidence: LimitConfidence
    source: str
    capabilities: FrozenSet[str]


def derive_required_capabilities(payload: Dict[str, Any]) -> RequiredCapabilities:
    """Deriva capacidades estritamente necessárias do payload da requisição."""
    if not isinstance(payload, dict):
        return RequiredCapabilities()

    has_tools = bool(payload.get("tools"))
    has_structured = bool(payload.get("response_format"))

    messages = payload.get("messages")
    has_vision = False
    has_files = False

    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        ptype = str(part.get("type") or "").lower()
                        if ptype in ("image_url", "image") or "image_url" in part:
                            has_vision = True
                        if ptype in ("file", "file_url") or "file_url" in part:
                            has_files = True

    return RequiredCapabilities(
        text=True,
        vision=has_vision,
        tools=has_tools,
        structured_output=has_structured,
        files=has_files,
    )


def resolve_model_limits(
    backend_info: Dict[str, Any],
    model_metadata: Optional[Dict[str, Any]] = None,
) -> ModelLimits:
    """Resolve os limites de um backend/modelo com origem e confiança explícitas."""
    backend_type = str(backend_info.get("backend_type") or "local").lower()
    model_metadata = model_metadata or {}

    if backend_type == "platform":
        raw_context = model_metadata.get("context_length")
        if raw_context is None:
            raw_context = model_metadata.get("inputTokenLimit")

        raw_output = model_metadata.get("max_completion_tokens")
        if raw_output is None:
            raw_output = model_metadata.get("outputTokenLimit")

        context_val = int(raw_context) if raw_context is not None and isinstance(raw_context, (int, float, str)) and str(raw_context).isdigit() else None
        output_val = int(raw_output) if raw_output is not None and isinstance(raw_output, (int, float, str)) and str(raw_output).isdigit() else None

        if context_val is None or context_val == UNKNOWN_PLATFORM_CONTEXT_LIMIT or context_val <= 0:
            return ModelLimits(
                context_tokens=None,
                max_output_tokens=output_val,
                source="platform_catalog",
                confidence=LimitConfidence.UNKNOWN,
            )

        return ModelLimits(
            context_tokens=context_val,
            max_output_tokens=output_val,
            source="platform_catalog",
            confidence=LimitConfidence.KNOWN_PROVIDER,
        )

    config = backend_info.get("config") or {}
    total_context = config.get("context_size")
    if total_context is None:
        total_context = backend_info.get("context_size", 65536)

    slots = config.get("parallel_slots")
    if slots is None:
        slots = backend_info.get("parallel_slots", 1)

    slots = max(1, int(slots))
    effective_context = max(1, int(total_context)) // slots

    return ModelLimits(
        context_tokens=effective_context,
        max_output_tokens=None,
        source="local_instance",
        confidence=LimitConfidence.KNOWN_LOCAL,
    )


def derive_target_capabilities(
    backend_info: Dict[str, Any],
    model_metadata: Optional[Dict[str, Any]] = None,
) -> FrozenSet[str]:
    """Deriva o conjunto de capacidades confirmadas do destino."""
    backend_type = str(backend_info.get("backend_type") or "local").lower()
    model_metadata = model_metadata or {}

    caps = {"text"}

    if backend_type == "platform":
        meta_caps = model_metadata.get("capabilities")
        if isinstance(meta_caps, (list, set, tuple, frozenset)):
            for c in meta_caps:
                cs = str(c).strip().lower()
                if cs in ("vision", "tools", "structured_output", "files"):
                    caps.add(cs)
        if model_metadata.get("supports_vision") or model_metadata.get("vision"):
            caps.add("vision")
        if model_metadata.get("supports_tools") or model_metadata.get("tools"):
            caps.add("tools")
        if model_metadata.get("supports_structured_output") or model_metadata.get("structured_output"):
            caps.add("structured_output")
    else:
        config = backend_info.get("config") or {}
        if config.get("mmproj_path") and not config.get("mmproj_disabled"):
            caps.add("vision")
        caps.add("tools")
        caps.add("structured_output")
        caps.add("files")

    return frozenset(caps)


def calculate_target_budget(
    payload: Dict[str, Any],
    limits: ModelLimits,
    capabilities: FrozenSet[str],
    protocol_overhead: int = DEFAULT_PROTOCOL_OVERHEAD,
    safety_margin: int = DEFAULT_SAFETY_MARGIN,
    default_output_reserve: int = DEFAULT_OUTPUT_RESERVE_FALLBACK,
) -> TargetBudget:
    """Calcula o orçamento de entrada do destino seguindo as regras da TechSpec."""
    req_max_completion = payload.get("max_completion_tokens")
    req_max_tokens = payload.get("max_tokens")

    if req_max_completion is not None and isinstance(req_max_completion, int) and req_max_completion > 0:
        reserve = req_max_completion
    elif req_max_tokens is not None and isinstance(req_max_tokens, int) and req_max_tokens > 0:
        reserve = req_max_tokens
    elif limits.max_output_tokens is not None and limits.max_output_tokens > 0:
        reserve = limits.max_output_tokens
    else:
        reserve = default_output_reserve

    if not limits.is_known:
        return TargetBudget(
            context_limit=None,
            output_reserve=reserve,
            protocol_overhead=protocol_overhead,
            safety_margin=safety_margin,
            input_budget=None,
            confidence=LimitConfidence.UNKNOWN,
            source=limits.source,
            capabilities=capabilities,
        )

    context_val = limits.context_tokens or 0
    budget = context_val - reserve - protocol_overhead - safety_margin

    return TargetBudget(
        context_limit=context_val,
        output_reserve=reserve,
        protocol_overhead=protocol_overhead,
        safety_margin=safety_margin,
        input_budget=max(0, budget),
        confidence=limits.confidence,
        source=limits.source,
        capabilities=capabilities,
    )


@dataclass
class ConversationBlock:
    block_id: str
    kind: str
    role: Optional[str] = None
    original_index: int = 0
    original_value: Any = None
    atomic_group_id: Optional[str] = None
    protected: bool = False
    token_cost: int = 0
    tool_call_ids: List[str] = field(default_factory=list)
    capabilities: FrozenSet[str] = field(default_factory=frozenset)
    retention_class: str = "normal"


@dataclass
class AtomicGroup:
    group_id: str
    block_ids: List[str] = field(default_factory=list)
    protected: bool = False
    tool_call_ids: List[str] = field(default_factory=list)
    kind: str = "tool_group"


@dataclass
class RequestEnvelope:
    original_payload: Dict[str, Any]
    recognized_conversation: bool = True
    model: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    response_format: Optional[Any] = None
    output_request: Optional[int] = None
    required_capabilities: RequiredCapabilities = field(default_factory=RequiredCapabilities)
    blocks: List[ConversationBlock] = field(default_factory=list)
    opaque_cost_sources: List[Dict[str, Any]] = field(default_factory=list)
    is_opaque: bool = False


@dataclass
class RequestIR:
    envelope: RequestEnvelope
    ordered_units: List[ConversationBlock] = field(default_factory=list)
    protected_unit_ids: Set[str] = field(default_factory=set)
    atomic_groups: Dict[str, AtomicGroup] = field(default_factory=dict)
    required_capabilities: RequiredCapabilities = field(default_factory=RequiredCapabilities)
    is_opaque: bool = False
    structural_validity: bool = True

    def calculate_total_tokens(self) -> int:
        if self.is_opaque:
            return ConservativeEstimator.estimate_payload(self.envelope.original_payload)
        tokens = sum(unit.token_cost for unit in self.ordered_units)
        if self.envelope.tools:
            tokens += ConservativeEstimator.estimate_payload(self.envelope.tools)
        if self.envelope.response_format:
            tokens += ConservativeEstimator.estimate_payload(self.envelope.response_format)
        return tokens

    def to_payload(self) -> Dict[str, Any]:
        return reconstruct_payload(self)


def parse_request_ir(payload: Dict[str, Any]) -> RequestIR:
    """Normaliza um payload em uma representação intermediária estrutural (RequestIR)."""
    if not isinstance(payload, dict):
        raw_payload = payload if isinstance(payload, dict) else {"raw": payload}
        env = RequestEnvelope(original_payload=raw_payload, recognized_conversation=False, is_opaque=True)
        return RequestIR(envelope=env, is_opaque=True, structural_validity=False)

    req_caps = derive_required_capabilities(payload)
    messages = payload.get("messages")
    tools = payload.get("tools")
    tool_choice = payload.get("tool_choice")
    response_format = payload.get("response_format")
    model = payload.get("model")
    req_max_completion = payload.get("max_completion_tokens") or payload.get("max_tokens")

    if not isinstance(messages, list):
        env = RequestEnvelope(
            original_payload=payload,
            recognized_conversation=False,
            model=model,
            tools=tools if isinstance(tools, list) else None,
            tool_choice=tool_choice,
            response_format=response_format,
            output_request=req_max_completion,
            required_capabilities=req_caps,
            is_opaque=True,
        )
        return RequestIR(envelope=env, is_opaque=True, structural_validity=False, required_capabilities=req_caps)

    defined_tool_call_ids: Dict[str, int] = {}
    called_tool_ids: List[str] = []
    tool_result_ids: Dict[str, List[int]] = {}
    has_invalid_structure = False

    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            has_invalid_structure = True
            break
        role = str(msg.get("role") or "").lower()

        if role == "assistant" and "tool_calls" in msg:
            tcs = msg.get("tool_calls")
            if not isinstance(tcs, list):
                has_invalid_structure = True
                break
            for tc in tcs:
                if not isinstance(tc, dict):
                    has_invalid_structure = True
                    break
                tc_id = tc.get("id")
                if not tc_id or not isinstance(tc_id, str):
                    has_invalid_structure = True
                    break
                if tc_id in defined_tool_call_ids:
                    has_invalid_structure = True
                    break
                defined_tool_call_ids[tc_id] = idx
                called_tool_ids.append(tc_id)

        if role == "tool":
            tc_id = msg.get("tool_call_id")
            if not tc_id or not isinstance(tc_id, str):
                has_invalid_structure = True
                break
            if tc_id not in tool_result_ids:
                tool_result_ids[tc_id] = []
            tool_result_ids[tc_id].append(idx)

    for tc_id in tool_result_ids:
        if tc_id not in defined_tool_call_ids:
            has_invalid_structure = True
            break

    if has_invalid_structure:
        env = RequestEnvelope(
            original_payload=payload,
            recognized_conversation=False,
            model=model,
            messages=messages,
            tools=tools if isinstance(tools, list) else None,
            tool_choice=tool_choice,
            response_format=response_format,
            output_request=req_max_completion,
            required_capabilities=req_caps,
            is_opaque=True,
        )
        return RequestIR(envelope=env, is_opaque=True, structural_validity=False, required_capabilities=req_caps)

    blocks: List[ConversationBlock] = []
    atomic_groups: Dict[str, AtomicGroup] = {}
    protected_unit_ids: Set[str] = set()

    tc_to_group: Dict[str, str] = {}
    for tc_id, asst_idx in defined_tool_call_ids.items():
        group_id = f"tool_group_asst_{asst_idx}"
        if group_id not in atomic_groups:
            atomic_groups[group_id] = AtomicGroup(group_id=group_id, kind="tool_group")
        atomic_groups[group_id].tool_call_ids.append(tc_id)
        tc_to_group[tc_id] = group_id

    last_user_idx = -1
    for idx, msg in enumerate(messages):
        if isinstance(msg, dict) and str(msg.get("role") or "").lower() == "user":
            last_user_idx = idx

    for idx, msg in enumerate(messages):
        role = str(msg.get("role") or "").lower()
        block_id = f"block_{idx}"
        kind = "opaque_message"

        is_protected = False
        if role in ("system", "developer"):
            kind = role
            is_protected = True
        elif role == "user":
            content = msg.get("content")
            if isinstance(content, list):
                has_img = any(isinstance(p, dict) and (p.get("type") in ("image_url", "image") or "image_url" in p) for p in content)
                has_file = any(isinstance(p, dict) and (p.get("type") in ("file", "file_url") or "file_url" in p) for p in content)
                if has_img:
                    kind = "multimodal_message"
                elif has_file:
                    kind = "file_content"
                else:
                    kind = "user_text"
            else:
                kind = "user_text"
            if idx == last_user_idx:
                is_protected = True
        elif role == "assistant":
            if "tool_calls" in msg and isinstance(msg.get("tool_calls"), list) and len(msg.get("tool_calls")) > 0:
                kind = "assistant_tool_calls"
            else:
                kind = "assistant_text"
        elif role == "tool":
            kind = "tool_results"

        block_tc_ids: List[str] = []
        block_group_id: Optional[str] = None

        if kind == "assistant_tool_calls":
            for tc in msg.get("tool_calls", []):
                tc_id = tc.get("id")
                if tc_id:
                    block_tc_ids.append(tc_id)
                    if not block_group_id and tc_id in tc_to_group:
                        block_group_id = tc_to_group[tc_id]
        elif kind == "tool_results":
            tc_id = msg.get("tool_call_id")
            if tc_id:
                block_tc_ids.append(tc_id)
                if tc_id in tc_to_group:
                    block_group_id = tc_to_group[tc_id]

        if block_group_id and block_group_id in atomic_groups:
            atomic_groups[block_group_id].block_ids.append(block_id)

        block_caps_set = {"text"}
        if kind == "multimodal_message":
            block_caps_set.add("vision")
        if kind == "file_content":
            block_caps_set.add("files")
        if kind in ("assistant_tool_calls", "tool_results"):
            block_caps_set.add("tools")

        retention_cls, final_protected = _classify_block_retention(msg, role, kind, is_protected)

        block = ConversationBlock(
            block_id=block_id,
            kind=kind,
            role=role,
            original_index=idx,
            original_value=msg,
            atomic_group_id=block_group_id,
            protected=final_protected,
            token_cost=ConservativeEstimator.estimate_payload(msg),
            tool_call_ids=block_tc_ids,
            capabilities=frozenset(block_caps_set),
            retention_class=retention_cls,
        )

        blocks.append(block)
        if final_protected:
            protected_unit_ids.add(block_id)

    # Propagar proteção para grupos atômicos (se um bloco for protegido, todos ficam protegidos)
    block_map = {b.block_id: b for b in blocks}
    for group_id, group in atomic_groups.items():
        group_blocks = [block_map[bid] for bid in group.block_ids if bid in block_map]
        if any(b.protected for b in group_blocks):
            group.protected = True
            for b in group_blocks:
                b.protected = True
                protected_unit_ids.add(b.block_id)

    envelope = RequestEnvelope(
        original_payload=payload,
        recognized_conversation=True,
        model=model,
        messages=messages,
        tools=tools if isinstance(tools, list) else None,
        tool_choice=tool_choice,
        response_format=response_format,
        output_request=req_max_completion,
        required_capabilities=req_caps,
        blocks=blocks,
        is_opaque=False,
    )

    return RequestIR(
        envelope=envelope,
        ordered_units=blocks,
        protected_unit_ids=protected_unit_ids,
        atomic_groups=atomic_groups,
        required_capabilities=req_caps,
        is_opaque=False,
        structural_validity=True,
    )


def build_request_ir(payload: Dict[str, Any]) -> RequestIR:
    """Alias para parse_request_ir."""
    return parse_request_ir(payload)


def reconstruct_payload(ir: RequestIR) -> Dict[str, Any]:
    """Reconstrói o payload JSON original preservando objetos originais e campos desconhecidos (lossless)."""
    if ir.is_opaque:
        return copy.deepcopy(ir.envelope.original_payload)

    reconstructed = copy.deepcopy(ir.envelope.original_payload)
    new_messages = [copy.deepcopy(block.original_value) for block in ir.ordered_units]
    reconstructed["messages"] = new_messages
    return reconstructed


class StructuralValidationError(ValueError):
    """Falha estrutural sem ecoar valores sensíveis do payload."""

    def __init__(self, code: str, *, stage: str = "validation") -> None:
        self.code = str(code)
        self.stage = str(stage)
        super().__init__(f"structural_validation_failed:{self.code}")


@dataclass(frozen=True)
class StructuralValidationReport:
    """Resultado metadata-only da validação de um payload transformado."""

    valid: bool
    original_cost: int
    candidate_cost: int
    retained_units: int


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise StructuralValidationError("not_json_serializable", stage="serialization") from exc


def _same_json(left: Any, right: Any) -> bool:
    return _canonical_json(left) == _canonical_json(right)


def _raise_if_unknown_fields_changed(original: Dict[str, Any], candidate: Dict[str, Any]) -> None:
    known_fields = {"messages", "tools", "response_format"}
    for key, value in original.items():
        if key in known_fields:
            continue
        if key not in candidate or not _same_json(value, candidate[key]):
            raise StructuralValidationError("envelope_field_changed", stage="envelope")
    if set(candidate) - set(original) - known_fields:
        raise StructuralValidationError("envelope_field_changed", stage="envelope")


def _validate_message_subsequence(
    ir: RequestIR,
    candidate_messages: List[Any],
    override_protected_ids: Optional[Set[str]] = None,
) -> List[ConversationBlock]:
    original_units = ir.ordered_units
    retained: List[ConversationBlock] = []
    cursor = 0

    for message in candidate_messages:
        match_index = None
        for index in range(cursor, len(original_units)):
            original_value = original_units[index].original_value
            if _same_json(original_value, message) or _same_json(_normalize_whitespace(original_value), message):
                match_index = index
                break
        if match_index is None:
            raise StructuralValidationError("message_changed_or_reordered", stage="messages")
        retained.append(original_units[match_index])
        cursor = match_index + 1

    retained_ids = {unit.block_id for unit in retained}
    protected_ids = ir.protected_unit_ids if override_protected_ids is None else override_protected_ids
    missing_protected = protected_ids - retained_ids
    if missing_protected:
        raise StructuralValidationError("protected_unit_removed", stage="messages")

    for group_id, group in ir.atomic_groups.items():
        group_present = bool(retained_ids.intersection(group.block_ids))
        if group_present and not set(group.block_ids).issubset(retained_ids):
            raise StructuralValidationError("atomic_group_split", stage="tools")

    return retained


def validate_transformed_payload(
    ir: RequestIR,
    candidate_payload: Dict[str, Any],
    *,
    original_cost: Optional[int] = None,
    override_protected_ids: Optional[Set[str]] = None,
) -> StructuralValidationReport:
    """Valida um resultado transformado antes do commit, sem aceitar mutação parcial."""
    if not isinstance(candidate_payload, dict):
        raise StructuralValidationError("payload_not_object", stage="serialization")

    _canonical_json(candidate_payload)
    original_cost = ir.calculate_total_tokens() if original_cost is None else int(original_cost)
    if original_cost < 0:
        raise StructuralValidationError("invalid_original_cost", stage="budget")

    if ir.is_opaque:
        if not _same_json(ir.envelope.original_payload, candidate_payload):
            raise StructuralValidationError("opaque_payload_changed", stage="opaque")
        return StructuralValidationReport(True, original_cost, original_cost, 0)

    _raise_if_unknown_fields_changed(ir.envelope.original_payload, candidate_payload)
    candidate_messages = candidate_payload.get("messages")
    if not isinstance(candidate_messages, list):
        raise StructuralValidationError("messages_not_array", stage="messages")

    retained = _validate_message_subsequence(ir, candidate_messages, override_protected_ids=override_protected_ids)
    candidate_ir = parse_request_ir(candidate_payload)
    if candidate_ir.is_opaque or not candidate_ir.structural_validity:
        raise StructuralValidationError("candidate_structure_invalid", stage="messages")

    if not _same_json(candidate_payload.get("tools"), ir.envelope.original_payload.get("tools")):
        raise StructuralValidationError("tools_changed", stage="tools")
    if not _same_json(candidate_payload.get("response_format"), ir.envelope.original_payload.get("response_format")):
        raise StructuralValidationError("response_format_changed", stage="envelope")

    candidate_cost = candidate_ir.calculate_total_tokens()
    if candidate_cost > original_cost:
        raise StructuralValidationError("cost_increased", stage="budget")

    return StructuralValidationReport(True, original_cost, candidate_cost, len(retained))


def validate_request_ir(
    ir: RequestIR,
    candidate_payload: Dict[str, Any],
    *,
    original_cost: Optional[int] = None,
) -> StructuralValidationReport:
    """Nome explícito para a barreira de validação usada pelo fluxo de commit."""
    return validate_transformed_payload(ir, candidate_payload, original_cost=original_cost)


# ---------------------------------------------------------------------------
# Safe-mode optimization pipeline — metadata-only audit & result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OptimizationAudit:
    """Relatório metadata-only da otimização Safe — não contém dados sensíveis."""

    strategy: str
    original_cost: int
    optimized_cost: int
    savings_tokens: int
    transformations_applied: List[str] = field(default_factory=list)
    protected_units_preserved: int = 0
    blocks_removed: int = 0
    blocks_merged: int = 0
    blocks_deduplicated: int = 0
    validation_passed: bool = True
    validation_errors: List[str] = field(default_factory=list)
    duration_ms: float = 0.0


@dataclass(frozen=True)
class OptimizationResult:
    """Resultado metadata-only do pipeline Safe — payload preservado, audit incluso."""

    audit: OptimizationAudit
    safe_payload: Dict[str, Any]


def _normalize_whitespace(payload: Any) -> Any:
    """Normaliza whitespace em strings mantendo semântica (lossless)."""
    if isinstance(payload, dict):
        return {
            key: _normalize_whitespace(value)
            for key, value in payload.items()
        }
    elif isinstance(payload, list):
        return [_normalize_whitespace(item) for item in payload]
    elif isinstance(payload, str):
        return _normalize_string("", payload)
    return payload


def _normalize_string(key: str, value: Any) -> Any:
    """Normaliza whitespace em strings, mantendo dados binários e numéricos intactos."""
    if not isinstance(value, str):
        return value

    # Preservar dados base64 e URLs (whitespace autorizado em URLs)
    if "base64" in value or "http" in value or "```" in value or _CODE_SQL_URL_RE.search(value):
        return value

    # Normalizar apenas strings puras de texto
    return " ".join(value.split())


def _remove_empty_blocks(
    ir: RequestIR,
    blocks: List[ConversationBlock],
) -> Tuple[List[ConversationBlock], int]:
    """Remove blocks vazios ou inofensivos mantendo integridade estrutural."""
    retained: List[ConversationBlock] = []
    removed_count = 0
    handled_atomic_groups: Set[str] = set()

    for block in blocks:
        # Não remover blocos protegidos nem grupos atômicos parcialmente.
        if block.protected:
            retained.append(block)
            continue
        if block.atomic_group_id:
            if block.atomic_group_id in handled_atomic_groups:
                continue
            handled_atomic_groups.add(block.atomic_group_id)
            group = ir.atomic_groups.get(block.atomic_group_id)
            if group:
                group_ids = set(group.block_ids)
                if all(any(candidate.block_id == group_id for candidate in blocks) for group_id in group_ids):
                    retained.extend(group_block for group_block in blocks if group_block.block_id in group_ids)
                    continue

        original = block.original_value

        # Ruído social antigo é descartável quando não pertence ao turno atual.
        if block.retention_class == "social_noise":
            removed_count += 1
            continue

        # Se for None ou vazio direto
        if original is None or (isinstance(original, str) and not original.strip()) or (isinstance(original, list) and len(original) == 0):
            removed_count += 1
            continue

        # Se for dict com campo 'content' vazio e sem tool_calls/outras chaves funcionais
        if isinstance(original, dict):
            content = original.get("content")
            has_tool_calls = bool(original.get("tool_calls"))
            is_content_empty = content is None or (isinstance(content, str) and not content.strip()) or (isinstance(content, list) and len(content) == 0)

            if is_content_empty and not has_tool_calls:
                removed_count += 1
                continue

        retained.append(block)

    return retained, removed_count


def _merge_duplicate_blocks(
    blocks: List[ConversationBlock],
) -> Tuple[List[ConversationBlock], int]:
    """Fusão segura de blocos duplicados consecutivos com mesmo conteúdo."""
    if len(blocks) < 2:
        return blocks, 0

    merged: List[ConversationBlock] = []
    dedup_count = 0
    i = 0

    while i < len(blocks):
        current = blocks[i]
        # Verificar duplicação com próximo bloco
        if i + 1 < len(blocks) and _same_json(current.original_value, blocks[i + 1].original_value):
            # Duplicação encontrada — manter o primeiro, descartar o segundo
            merged.append(current)
            dedup_count += 1
            i += 2
            continue

        merged.append(current)
        i += 1

    return merged, dedup_count


def _merge_consecutive_identical_blocks(
    blocks: List[ConversationBlock],
) -> Tuple[List[ConversationBlock], int]:
    """Fusão de blocos consecutivos idênticos (conservação agressiva mas segura)."""
    if len(blocks) < 2:
        return blocks, 0

    merged: List[ConversationBlock] = []
    merge_count = 0
    i = 0

    while i < len(blocks):
        current = blocks[i]
        next_blocks = []
        j = i + 1

        while j < len(blocks) and _same_json(current.original_value, blocks[j].original_value):
            next_blocks.append(blocks[j])
            j += 1

        if next_blocks:
            # Todos idênticos — manter um único bloco
            merged.append(current)
            merge_count += len(next_blocks)
            i = j
        else:
            merged.append(current)
            i += 1

    return merged, merge_count


async def _rebuild_with_blocks(
    ir: RequestIR,
    new_blocks: List[ConversationBlock],
) -> Dict[str, Any]:
    """Reconstrói o payload com os blocos atualizados e recalcula custo."""
    ir.ordered_units = new_blocks
    return ir.to_payload()


async def optimize_request_ir_safe(
    ir: RequestIR,
    budget: TargetBudget,
    max_iterations: int = 5,
    tokenizer_registry: Optional[TokenizerRegistry] = None,
    tokenizer_mappings: Optional[Dict[str, Any]] = None,
) -> OptimizationResult:
    """Pipeline principal de otimização no modo Safe.

    Executa transformações neutras, normalização de whitespace, remoção de
    elementos vazios, deduplicação e fusão seguras, com recálculo de custo
    via validador. Retorna OptimizationResult metadata-only.
    """
    import time as _time

    start_ms = _time.monotonic()
    original_cost = ir.calculate_total_tokens()

    transformations_applied: List[str] = []
    blocks_removed = 0
    blocks_merged = 0
    blocks_deduplicated = 0

    # Passos do pipeline Safe (iterativo, até max_iterations)
    for iteration in range(max_iterations):
        blocks = list(ir.ordered_units)

        # Step 1: Remoção de blocos vazios desprotegidos
        blocks, removed_count = _remove_empty_blocks(ir, blocks)
        if removed_count > 0:
            blocks_removed += removed_count
            transformations_applied.append(f"empty_blocks_removed_iter{iteration + 1}")

        # Step 2: Normalização de whitespace
        safe_payload = await _rebuild_with_blocks(ir, blocks)
        normalized_payload = _normalize_whitespace(safe_payload)
        normalized_ir = parse_request_ir(normalized_payload)

        # Step 3: Deduplicação de blocos consecutivos idênticos
        new_blocks, dedup_count = _merge_duplicate_blocks(normalized_ir.ordered_units)
        if dedup_count > 0:
            blocks_deduplicated += dedup_count
            transformations_applied.append(f"dedup_iter{iteration + 1}")

        # Step 4: Fusão de blocos consecutivos idênticos adicionais
        new_blocks, merge_count = _merge_consecutive_identical_blocks(new_blocks)
        if merge_count > 0:
            blocks_merged += merge_count
            transformations_applied.append(f"merge_iter{iteration + 1}")

        normalized_ir.ordered_units = new_blocks
        candidate_payload = reconstruct_payload(normalized_ir)
        candidate_ir = parse_request_ir(candidate_payload)

        # Validar estruturalmente e custo
        try:
            report = validate_transformed_payload(ir, candidate_payload, original_cost=original_cost)
            if not report.valid:
                break
        except StructuralValidationError:
            break

        ir = candidate_ir

    final_cost = ir.calculate_total_tokens()
    final_payload = ir.to_payload()
    duration_ms = (_time.monotonic() - start_ms) * 1000

    audit = OptimizationAudit(
        strategy="safe",
        original_cost=original_cost,
        optimized_cost=final_cost,
        savings_tokens=max(0, original_cost - final_cost),
        transformations_applied=transformations_applied,
        protected_units_preserved=len(ir.protected_unit_ids),
        blocks_removed=blocks_removed,
        blocks_merged=blocks_merged,
        blocks_deduplicated=blocks_deduplicated,
        validation_passed=True,
        duration_ms=duration_ms,
    )

    return OptimizationResult(audit=audit, safe_payload=final_payload)


async def optimize_request_ir_moderate(
    ir: RequestIR,
    budget: TargetBudget,
    max_iterations: int = 5,
    tokenizer_registry: Optional[TokenizerRegistry] = None,
    tokenizer_mappings: Optional[Dict[str, Any]] = None,
    original_cost: Optional[int] = None,
    safe_audit: Optional[OptimizationAudit] = None,
    cost_optimization: bool = False,
) -> OptimizationResult:
    """Pipeline de otimização modo Moderate (redução determinística de baixo risco).

    No modo normal, executa apenas quando o limite do destino é conhecido
    (confidence != UNKNOWN). Com ``cost_optimization=True``, também pode remover
    unidades históricas não protegidas sem depender de um limite conhecido.
    Remove apenas grupos atômicos completos antigos e blocos desprotegidos antigos.
    Preserva system/developer, turno atual, decisões técnicas, código, mídia, arquivos e logs críticos.
    Interrompe a remoção assim que o orçamento for satisfeito.
    """
    import time as _time

    start_ms = _time.monotonic()
    orig_cost = ir.calculate_total_tokens() if original_cost is None else original_cost
    current_tokens = ir.calculate_total_tokens()

    # Sem limite conhecido, Moderate só pode ser executado quando explicitamente
    # solicitado como otimização de custo. Nesse modo removemos apenas unidades
    # históricas não protegidas; não dependemos de um orçamento arbitrário.
    if (
        not cost_optimization
        and (
            not budget.confidence
            or budget.confidence == LimitConfidence.UNKNOWN
            or budget.input_budget is None
        )
    ):
        audit = safe_audit or OptimizationAudit(
            strategy="safe",
            original_cost=orig_cost,
            optimized_cost=current_tokens,
            savings_tokens=max(0, orig_cost - current_tokens),
            protected_units_preserved=len(ir.protected_unit_ids),
            validation_passed=True,
            duration_ms=(_time.monotonic() - start_ms) * 1000,
        )
        return OptimizationResult(audit=audit, safe_payload=ir.to_payload())

    # O modo normal só reduz quando necessário. O modo de economia de custo
    # continua até esgotar as unidades históricas não protegidas, mesmo que o
    # contexto já caiba no orçamento.
    if (
        not cost_optimization
        and budget.input_budget is not None
        and current_tokens <= budget.input_budget
    ):
        audit = OptimizationAudit(
            strategy="moderate",
            original_cost=orig_cost,
            optimized_cost=current_tokens,
            savings_tokens=max(0, orig_cost - current_tokens),
            transformations_applied=["moderate_already_fits"],
            protected_units_preserved=len(ir.protected_unit_ids),
            blocks_removed=safe_audit.blocks_removed if safe_audit else 0,
            blocks_merged=safe_audit.blocks_merged if safe_audit else 0,
            blocks_deduplicated=safe_audit.blocks_deduplicated if safe_audit else 0,
            validation_passed=True,
            duration_ms=(_time.monotonic() - start_ms) * 1000,
        )
        return OptimizationResult(audit=audit, safe_payload=ir.to_payload())

    # Montar unidades candidatas a remoção (apenas não protegidas)
    processed_groups = set()
    candidate_units = []

    for block in ir.ordered_units:
        if block.protected:
            continue
        if block.atomic_group_id:
            # Never discard tool calls/results in the moderate stage; preserving
            # the complete dependency is safer than reducing historical context.
            continue
        else:
            candidate_units.append({
                "kind": "single_block",
                "blocks": [block],
                "first_index": block.original_index,
                "retention_class": block.retention_class,
            })

    # Ordenar por tiers de retenção:
    # Tier 1: social_noise (do mais antigo ao mais recente)
    # Tier 2: tool_group (do mais antigo ao mais recente)
    # Tier 3: normal / outros desprotegidos (do mais antigo ao mais recente)
    tier1_social = sorted([u for u in candidate_units if u["retention_class"] == "social_noise"], key=lambda u: u["first_index"])
    tier2_tools = sorted([u for u in candidate_units if u["retention_class"] == "tool_group"], key=lambda u: u["first_index"])
    tier3_other = sorted([u for u in candidate_units if u["retention_class"] not in ("social_noise", "tool_group")], key=lambda u: u["first_index"])

    ordered_candidates = tier1_social + tier2_tools + tier3_other

    retained_blocks = list(ir.ordered_units)
    removed_blocks_count = 0
    transformations_applied = list(safe_audit.transformations_applied) if safe_audit else []

    for unit in ordered_candidates:
        if (
            not cost_optimization
            and budget.input_budget is not None
            and current_tokens <= budget.input_budget
        ):
            break

        unit_blocks = unit["blocks"]
        unit_block_ids = {b.block_id for b in unit_blocks}
        retained_blocks = [b for b in retained_blocks if b.block_id not in unit_block_ids]
        removed_blocks_count += len(unit_blocks)

        t_name = f"remove_{unit['retention_class']}"
        if t_name not in transformations_applied:
            transformations_applied.append(t_name)

        temp_ir = copy.deepcopy(ir)
        temp_ir.ordered_units = retained_blocks
        current_tokens = temp_ir.calculate_total_tokens()

    final_ir = copy.deepcopy(ir)
    final_ir.ordered_units = retained_blocks
    candidate_payload = reconstruct_payload(final_ir)

    try:
        report = validate_transformed_payload(ir, candidate_payload, original_cost=orig_cost)
        if report.valid:
            duration_ms = (_time.monotonic() - start_ms) * 1000
            final_cost = report.candidate_cost
            audit = OptimizationAudit(
                strategy="moderate",
                original_cost=orig_cost,
                optimized_cost=final_cost,
                savings_tokens=max(0, orig_cost - final_cost),
                transformations_applied=(
                    transformations_applied
                    or (["moderate_reduction"] if removed_blocks_count else ["cost_optimization_no_eligible_units"])
                ),
                protected_units_preserved=len(ir.protected_unit_ids),
                blocks_removed=(safe_audit.blocks_removed if safe_audit else 0) + removed_blocks_count,
                blocks_merged=safe_audit.blocks_merged if safe_audit else 0,
                blocks_deduplicated=safe_audit.blocks_deduplicated if safe_audit else 0,
                validation_passed=True,
                duration_ms=duration_ms,
            )
            return OptimizationResult(audit=audit, safe_payload=candidate_payload)
    except StructuralValidationError:
        pass

    safe_aud = safe_audit or OptimizationAudit(
        strategy="safe",
        original_cost=orig_cost,
        optimized_cost=ir.calculate_total_tokens(),
        savings_tokens=max(0, orig_cost - ir.calculate_total_tokens()),
        protected_units_preserved=len(ir.protected_unit_ids),
        validation_passed=True,
        duration_ms=(_time.monotonic() - start_ms) * 1000,
    )
    return OptimizationResult(audit=safe_aud, safe_payload=ir.to_payload())


async def optimize_request_ir_aggressive(
    ir: RequestIR,
    budget: TargetBudget,
    max_iterations: int = 5,
    tokenizer_registry: Optional[TokenizerRegistry] = None,
    tokenizer_mappings: Optional[Dict[str, Any]] = None,
    original_cost: Optional[int] = None,
    moderate_audit: Optional[OptimizationAudit] = None,
    cost_optimization: bool = False,
) -> OptimizationResult:
    """Pipeline de otimização modo Aggressive (redução drástica preservando apenas o conjunto mínimo).

    Executa apenas quando o limite do destino é conhecido (confidence != UNKNOWN)
    no fluxo de orçamento. O modo de economia de custo aceita a execução sem
    limite conhecido, embora o proxy só use Aggressive quando há orçamento para
    validar.
    Remove grupos/blocos antigos inteiros desprotegidos, descartando inclusive
    retenções estendidas (decisões técnicas, código antigo, mídias antigas e logs antigos).
    Preserva estritamente o conjunto mínimo obrigatório: system, developer, turno atual
    e grupos atômicos de ferramentas acoplados.
    Interrompe a remoção assim que o orçamento for satisfeito.
    Se após remover tudo o conjunto protegido ainda exceder o orçamento, gera resultado com
    strategy="context_too_large" e validation_passed=False.
    """
    import time as _time

    start_ms = _time.monotonic()
    orig_cost = ir.calculate_total_tokens() if original_cost is None else original_cost
    current_tokens = ir.calculate_total_tokens()

    # Invariante 1: Se limite é desconhecido ou input_budget é None, NÃO executar Aggressive
    if (
        not cost_optimization
        and (
            not budget.confidence
            or budget.confidence == LimitConfidence.UNKNOWN
            or budget.input_budget is None
        )
    ):
        audit = moderate_audit or OptimizationAudit(
            strategy="moderate",
            original_cost=orig_cost,
            optimized_cost=current_tokens,
            savings_tokens=max(0, orig_cost - current_tokens),
            protected_units_preserved=len(ir.protected_unit_ids),
            validation_passed=True,
            duration_ms=(_time.monotonic() - start_ms) * 1000,
        )
        return OptimizationResult(audit=audit, safe_payload=ir.to_payload())

    # Invariante 2: Se já cabe no orçamento, não remover nada
    if (
        not cost_optimization
        and budget.input_budget is not None
        and current_tokens <= budget.input_budget
    ):
        audit = OptimizationAudit(
            strategy="aggressive",
            original_cost=orig_cost,
            optimized_cost=current_tokens,
            savings_tokens=max(0, orig_cost - current_tokens),
            transformations_applied=["aggressive_already_fits"],
            protected_units_preserved=len(ir.protected_unit_ids),
            blocks_removed=moderate_audit.blocks_removed if moderate_audit else 0,
            blocks_merged=moderate_audit.blocks_merged if moderate_audit else 0,
            blocks_deduplicated=moderate_audit.blocks_deduplicated if moderate_audit else 0,
            validation_passed=True,
            duration_ms=(_time.monotonic() - start_ms) * 1000,
        )
        return OptimizationResult(audit=audit, safe_payload=ir.to_payload())

    # Identificar o índice do último user message (turno atual)
    messages = ir.envelope.messages or []
    last_user_idx = -1
    for idx, msg in enumerate(messages):
        if isinstance(msg, dict) and str(msg.get("role") or "").lower() == "user":
            last_user_idx = idx

    # Identificar quais blocos pertencem ao conjunto MÍNIMO protegido
    minimum_protected_block_ids: Set[str] = set()

    for block in ir.ordered_units:
        is_min_protected = False
        if block.protected:
            is_min_protected = True
        elif block.role in ("system", "developer"):
            is_min_protected = True
        elif last_user_idx >= 0 and block.original_index >= last_user_idx:
            is_min_protected = True

        if is_min_protected:
            minimum_protected_block_ids.add(block.block_id)

    # Tool calls/results são sempre protegidos como unidade atômica para evitar
    # órfãos, mesmo quando pertencem a um turno histórico.
    block_map = {b.block_id: b for b in ir.ordered_units}
    for group_id, group in ir.atomic_groups.items():
        group_blocks = [block_map[bid] for bid in group.block_ids if bid in block_map]
        for b in group_blocks:
            minimum_protected_block_ids.add(b.block_id)

    # Criar uma cópia do IR com a barreira de proteção restrita ao conjunto mínimo para validação
    aggressive_ir = copy.deepcopy(ir)
    aggressive_ir.protected_unit_ids = minimum_protected_block_ids
    for block in aggressive_ir.ordered_units:
        block.protected = block.block_id in minimum_protected_block_ids
    for group in aggressive_ir.atomic_groups.values():
        group.protected = any(bid in minimum_protected_block_ids for bid in group.block_ids)

    # Montar unidades candidatas a remoção no Aggressive
    processed_groups = set()
    candidate_units = []

    for block in aggressive_ir.ordered_units:
        if block.block_id in minimum_protected_block_ids:
            continue
        if block.atomic_group_id:
            gid = block.atomic_group_id
            if gid in processed_groups:
                continue
            processed_groups.add(gid)
            group_blocks = [b for b in aggressive_ir.ordered_units if b.atomic_group_id == gid]
            candidate_units.append({
                "kind": "atomic_group",
                "group_id": gid,
                "blocks": group_blocks,
                "first_index": min(b.original_index for b in group_blocks),
            })
        else:
            candidate_units.append({
                "kind": "single_block",
                "blocks": [block],
                "first_index": block.original_index,
            })

    # Ordenar candidatos do mais antigo ao mais recente
    ordered_candidates = sorted(candidate_units, key=lambda u: u["first_index"])

    retained_blocks = list(aggressive_ir.ordered_units)
    removed_blocks_count = 0
    transformations_applied = list(moderate_audit.transformations_applied) if moderate_audit else []

    for unit in ordered_candidates:
        if (
            not cost_optimization
            and budget.input_budget is not None
            and current_tokens <= budget.input_budget
        ):
            break

        unit_blocks = unit["blocks"]
        unit_block_ids = {b.block_id for b in unit_blocks}
        retained_blocks = [b for b in retained_blocks if b.block_id not in unit_block_ids]
        removed_blocks_count += len(unit_blocks)

        t_name = f"aggressive_remove_old_unit_{unit['first_index']}"
        if t_name not in transformations_applied:
            transformations_applied.append(t_name)

        temp_ir = copy.deepcopy(aggressive_ir)
        temp_ir.ordered_units = retained_blocks
        current_tokens = temp_ir.calculate_total_tokens()

    final_ir = copy.deepcopy(aggressive_ir)
    final_ir.ordered_units = retained_blocks
    candidate_payload = reconstruct_payload(final_ir)

    duration_ms = (_time.monotonic() - start_ms) * 1000

    try:
        report = validate_transformed_payload(
            aggressive_ir, candidate_payload, original_cost=orig_cost,
            override_protected_ids=minimum_protected_block_ids,
        )
        if report.valid:
            final_cost = report.candidate_cost
            fits_budget = (
                budget.input_budget is None
                or final_cost <= budget.input_budget
            )

            audit = OptimizationAudit(
                strategy="aggressive" if fits_budget else "context_too_large",
                original_cost=orig_cost,
                optimized_cost=final_cost,
                savings_tokens=max(0, orig_cost - final_cost),
                transformations_applied=transformations_applied or ["aggressive_reduction"],
                protected_units_preserved=len(minimum_protected_block_ids),
                blocks_removed=(moderate_audit.blocks_removed if moderate_audit else 0) + removed_blocks_count,
                blocks_merged=moderate_audit.blocks_merged if moderate_audit else 0,
                blocks_deduplicated=moderate_audit.blocks_deduplicated if moderate_audit else 0,
                validation_passed=fits_budget,
                validation_errors=[] if fits_budget else ["context_too_large"],
                duration_ms=duration_ms,
            )
            return OptimizationResult(audit=audit, safe_payload=candidate_payload)
    except StructuralValidationError:
        pass

    mod_aud = moderate_audit or OptimizationAudit(
        strategy="moderate",
        original_cost=orig_cost,
        optimized_cost=ir.calculate_total_tokens(),
        savings_tokens=max(0, orig_cost - ir.calculate_total_tokens()),
        protected_units_preserved=len(ir.protected_unit_ids),
        validation_passed=True,
        duration_ms=duration_ms,
    )
    return OptimizationResult(audit=mod_aud, safe_payload=ir.to_payload())


class ContextOptimizer:
    """Otimizador isolado do Context Optimizer (composition root)."""

    def __init__(
        self,
        config_manager: Optional[Any] = None,
        tokenizer_fetcher: Optional[Any] = None,
        audit_recorder: Optional[AuditRecorder] = None,
    ):
        self.config_manager = config_manager
        self.tokenizer_registry = TokenizerRegistry(fetcher=tokenizer_fetcher)
        self.audit_recorder = audit_recorder

    async def optimize(
        self,
        payload: Dict[str, Any],
        backend_info: Dict[str, Any],
        model_metadata: Optional[Dict[str, Any]] = None,
        stage_limit: Optional[str] = None,
        cost_optimization: bool = False,
    ) -> OptimizationResult:
        """Executa a otimização Safe/Moderate/Aggressive do payload para o backend de destino.

        Se o limite for excedido mesmo no estágio Aggressive, registra na auditoria e lança
        ContextTooLargeError(413, ..., code="context_too_large").
        Se o limite for desconhecido, o fluxo normal executa apenas Safe; quando
        ``cost_optimization`` está ativo, unidades históricas não protegidas
        ainda podem ser removidas. Erros internos continuam em fail-open.
        """
        try:
            ir = parse_request_ir(payload)
            limits = resolve_model_limits(backend_info, model_metadata)
            target_caps = derive_target_capabilities(backend_info, model_metadata)
            tokenizer_mappings = None
            if self.config_manager and hasattr(self.config_manager, "get_smart_proxy_settings"):
                sp = self.config_manager.get_smart_proxy_settings()
                co_cfg = sp.get("context_optimizer", {}) if isinstance(sp, dict) else {}
                tokenizer_mappings = co_cfg.get("tokenizers")

            budget = calculate_target_budget(
                payload=payload,
                limits=limits,
                capabilities=target_caps,
            )

            result = await optimize_request_ir_safe(
                ir=ir,
                budget=budget,
                tokenizer_registry=self.tokenizer_registry,
                tokenizer_mappings=tokenizer_mappings,
            )

            if stage_limit == "safe":
                if (
                    limits.is_known
                    and budget.input_budget is not None
                    and result.audit.optimized_cost > budget.input_budget
                ):
                    if self.audit_recorder:
                        self.audit_recorder.record(result.audit)
                    raise ContextTooLargeError(
                        message=f"Prompt excede o limite de contexto do modelo ({limits.context_tokens} tokens)",
                        code="context_too_large",
                    )
                if self.audit_recorder:
                    self.audit_recorder.record(result.audit)
                return result

            if (
                cost_optimization
                or (
                    limits.is_known
                    and budget.input_budget is not None
                    and result.audit.optimized_cost > budget.input_budget
                )
            ):
                moderate_ir = parse_request_ir(result.safe_payload)
                result = await optimize_request_ir_moderate(
                    ir=moderate_ir,
                    budget=budget,
                    tokenizer_registry=self.tokenizer_registry,
                    tokenizer_mappings=tokenizer_mappings,
                    original_cost=result.audit.original_cost,
                    safe_audit=result.audit,
                    cost_optimization=cost_optimization,
                )

            if stage_limit == "moderate":
                if self.audit_recorder:
                    self.audit_recorder.record(result.audit)
                return result

            if (
                limits.is_known
                and budget.input_budget is not None
                and result.audit.optimized_cost > budget.input_budget
            ):
                aggressive_ir = parse_request_ir(result.safe_payload)
                result = await optimize_request_ir_aggressive(
                    ir=aggressive_ir,
                    budget=budget,
                    tokenizer_registry=self.tokenizer_registry,
                    tokenizer_mappings=tokenizer_mappings,
                    original_cost=result.audit.original_cost,
                    moderate_audit=result.audit,
                    cost_optimization=cost_optimization,
                )

            if (
                limits.is_known
                and budget.input_budget is not None
                and result.audit.optimized_cost > budget.input_budget
            ):
                if self.audit_recorder:
                    self.audit_recorder.record(result.audit)
                raise ContextTooLargeError(
                    message=f"Prompt excede o limite de contexto do modelo ({limits.context_tokens} tokens)",
                    code="context_too_large",
                )

            if self.audit_recorder:
                self.audit_recorder.record(result.audit)
            return result
        except (ContextTooLargeError,):
            raise
        except Exception as exc:
            ir = parse_request_ir(payload)
            orig_cost = ir.calculate_total_tokens()
            audit = OptimizationAudit(
                strategy="fail_open",
                original_cost=orig_cost,
                optimized_cost=orig_cost,
                savings_tokens=0,
                transformations_applied=["fail_open_on_error"],
                validation_passed=True,
                validation_errors=[str(exc)],
            )
            if self.audit_recorder:
                self.audit_recorder.record(audit)
            return OptimizationResult(audit=audit, safe_payload=copy.deepcopy(payload))

    def query_audit_logs(
        self,
        page: int = 1,
        per_page: int = 50,
        strategy_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Consulta paginada de registros de auditoria (metadata-only)."""
        if self.audit_recorder:
            return self.audit_recorder.query(page=page, per_page=per_page, strategy_filter=strategy_filter)
        return {"page": page, "per_page": per_page, "total": 0, "pages": 1, "items": []}
