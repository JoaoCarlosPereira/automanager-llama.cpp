"""Roteador do Modo Proxy Inteligente.

Mantém a tabela de sessões sticky (afinidade conversa/subagente -> backend),
os contadores de requisições em andamento por porta e a seleção least-busy
de backends. Não importa llama_manager: dependências entram por injeção
(ADR-004). Persistência das sessões em JSON atômico (ADR-005).
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Set, Tuple, Union

from config_manager import (
    lookup_model_config,
    lookup_platform_config,
    normalize_backend_id,
    normalize_model_path,
)
from context_optimizer import (
    LimitConfidence,
    ModelLimits,
    RequiredCapabilities,
    derive_required_capabilities,
    derive_target_capabilities,
    resolve_model_limits,
)
from schemas import (
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_MAX_PARALLEL_REQUESTS,
    DEFAULT_PARALLEL_SLOTS,
    DEFAULT_PROXY_ELIGIBLE,
)

logger = logging.getLogger("automanager")

AGENT_TAG_RE = re.compile(r"\[AGENT:([A-Za-z0-9_-]+)\]")

# Margem de segurança sobre a estimativa de tokens (chars//4) — TechSpec.
TOKEN_ESTIMATE_MARGIN = 1.1
# Persistência oportunista dos contadores de sessão (a cada N usos).
PERSIST_EVERY_N_REQUESTS = 20
# Intervalo do polling de espera por slot livre.
BUSY_POLL_SECONDS = 0.25
# Limite de ramificações de uma sessão hash: (afinidade fraca) sob concorrência.
MAX_HASH_BRANCHES = 8
# Tempo padrao em que um backend que falhou fica fora do roteamento. O chamador
# pode fornecer um intervalo maior (por exemplo, para limite de assinatura).
DEFAULT_BACKEND_COOLDOWN_SECONDS = 60
# Limite sentinela usado apenas internamente quando o provedor nao publicou a
# capacidade do modelo. Nesse caso e mais seguro tentar o upstream do que
# rejeitar localmente um contexto que ele talvez suporte.
UNKNOWN_PLATFORM_CONTEXT_LIMIT = (1 << 63) - 1

MAIN_TAG = "main"


class ProxyError(Exception):
    """Erro de roteamento com resposta no formato de erro OpenAI."""

    def __init__(self, status_code: int, message: str, code: str = "proxy_error"):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.code = code

    def payload(self) -> dict:
        return {
            "error": {
                "message": self.message,
                "type": self.code,
                "code": self.code,
            }
        }


@dataclass
class StickySession:
    affinity_key: str
    backend_port: int
    backend_model_path: str
    external_model: str
    internal_model: str
    detected_tag: Optional[str]
    created_at: str
    last_used_at: str
    request_count: int = 0
    tokens_processed: int = 0
    backend_id: str = ""
    backend_type: str = "local"
    provider: Optional[str] = None


@dataclass
class RouteDecision:
    backend_port: int
    internal_model: str
    external_model: str
    affinity_key: str
    detected_tag: Optional[str]
    sticky_hit: bool
    reason: str
    rewrite: bool
    prompt_tokens_estimated: int = 0
    gpu: str = ""
    backend_id: str = ""
    backend_type: str = "local"
    provider: Optional[str] = None


@dataclass(frozen=True)
class RoutePlan:
    """Plano de roteamento imutável — sem efeitos colaterais.

    Contém um *commit_token* privado para impedir commit duplicado e
    validar revalidação sob lock em ``commit_route()``.
    """

    decision: RouteDecision
    commit_token: str = field(repr=False)
    created_at: str = field(repr=False)

    def __post_init__(self) -> None:
        if not self.commit_token:
            raise ValueError("commit_token obrigatorio")


class StaleRoutePlan(ProxyError):
    """Plano expirou ou backend destino nao esta mais disponivel.

    O chamador DEVE replanejar via ``plan_route()``.
    """

    def __init__(self, plan: RoutePlan) -> None:
        super().__init__(
            status_code=503,
            message=(
                f"Plano de roteamento expirou (token={plan.commit_token[:8]}). "
                "Backend destino pode ter mudado — replaneje."
            ),
            code="stale_route_plan",
        )


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _content_text(content: Any) -> str:
    """Extrai texto de um campo content (string ou lista de partes)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return ""


def gpu_label(instance: Dict[str, Any]) -> str:
    """Nome amigável da GPU da instância a partir de config.gpu_weights."""
    config = instance.get("config") or {}
    weights = config.get("gpu_weights") or []
    gpus = [
        w for w in weights
        if isinstance(w, dict) and w.get("device", "gpu") == "gpu" and w.get("active", True)
    ]
    if not gpus:
        return "CPU"
    main = next((w for w in gpus if w.get("is_main")), None)
    chosen = main or max(gpus, key=lambda w: w.get("weight", 0))
    name = chosen.get("name") or "GPU"
    index = chosen.get("index")
    return f"{name} #{index}" if index is not None else name


def _rewrite_sse_line(
    line: bytes, external_model: str, usage_holder: Optional[dict] = None
) -> bytes:
    """Reescreve o campo model de uma linha `data: {json}`; fail-open (ADR-006)."""
    has_cr = line.endswith(b"\r")
    stripped = line[:-1] if has_cr else line
    if not stripped.lstrip().startswith(b"data:"):
        return line
    prefix_len = stripped.index(b"data:") + len(b"data:")
    payload = stripped[prefix_len:].strip()
    if not payload or payload == b"[DONE]":
        return line
    try:
        obj = json.loads(payload)
    except ValueError:
        return line
    if not isinstance(obj, dict):
        return line
    if usage_holder is not None and isinstance(obj.get("usage"), dict):
        usage_holder["usage"] = obj["usage"]
    if "model" not in obj:
        return line
    obj["model"] = external_model
    rewritten = b"data: " + json.dumps(obj, ensure_ascii=False).encode("utf-8")
    return rewritten + b"\r" if has_cr else rewritten


async def rewrite_sse_stream(
    byte_iter, external_model: str, usage_holder: Optional[dict] = None
):
    """Buffer incremental por linha: reescreve eventos SSE sem reter o corpo.

    Emite somente linhas completas (delimitadas por \\n); o resíduo final é
    emitido no fechamento do stream para não truncar a resposta (ADR-006).
    """
    buffer = b""
    async for chunk in byte_iter:
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            yield _rewrite_sse_line(line, external_model, usage_holder) + b"\n"
    if buffer:
        yield _rewrite_sse_line(buffer, external_model, usage_holder)


def rewrite_json_model(
    content: bytes, external_model: str
) -> Tuple[bytes, Optional[dict]]:
    """Reescreve o model de uma resposta JSON não-stream; retorna (body, usage)."""
    try:
        obj = json.loads(content)
    except ValueError:
        return content, None
    if not isinstance(obj, dict):
        return content, None
    usage = obj.get("usage") if isinstance(obj.get("usage"), dict) else None
    if "model" in obj:
        obj["model"] = external_model
        return json.dumps(obj, ensure_ascii=False).encode("utf-8"), usage
    return content, usage


class ProxyRouter:
    """Decide o backend de cada requisição ao modelo principal."""

    def __init__(
        self,
        get_status: Callable[[], dict],
        config_manager,
        sessions_path,
        now: Optional[Callable[[], datetime]] = None,
        context_limit_resolver: Optional[
            Callable[[Dict[str, Any], str], Optional[int]]
        ] = None,
        requested_primary_resolver: Optional[
            Callable[[List[Dict[str, Any]], str], Optional[Dict[str, Any]]]
        ] = None,
        ollama_cloud_account_manager=None,
    ) -> None:
        self._get_status = get_status
        self._config = config_manager
        self._sessions_path = Path(sessions_path)
        self._now = now or _utcnow
        self._context_limit_resolver = context_limit_resolver
        self._requested_primary_resolver = requested_primary_resolver
        self._ollama_cloud_account_manager = ollama_cloud_account_manager
        self._lock = asyncio.Lock()
        self._sessions: Dict[str, StickySession] = {}
        self._in_flight: Dict[str, int] = {}
        self._disabled_ports: set = set()
        self._unavailable_until: Dict[str, datetime] = {}
        self._benchmark_latencies_ms: Dict[str, float] = {}
        self._benchmark_measured_at: Optional[str] = None
        self._unsaved_uses = 0
        self._committed_tokens: Set[str] = set()
        self._load_sessions()

    # ------------------------------------------------------------------
    # Extração de afinidade (PRD F5)
    # ------------------------------------------------------------------

    def extract_affinity(
        self,
        headers: Mapping[str, str],
        body: dict,
        client_ip: str,
        user_agent: str,
    ) -> Tuple[str, Optional[str]]:
        """Retorna (affinity_key, detected_tag) na ordem de precedência do PRD."""
        lower_headers = {str(k).lower(): v for k, v in headers.items()}
        tag = self.detect_tag(body)

        session_id = lower_headers.get("x-automanager-session-id")
        if session_id:
            return f"sid:{session_id}", tag
        agent_id = lower_headers.get("x-automanager-agent-id")
        if agent_id:
            return f"aid:{agent_id}", tag

        metadata = body.get("metadata")
        if isinstance(metadata, dict):
            if metadata.get("session_id"):
                return f"sid:{metadata['session_id']}", tag
            if metadata.get("agent_id"):
                return f"aid:{metadata['agent_id']}", tag

        digest = self._stable_hash(body, client_ip, user_agent)
        if tag:
            return f"agent:{tag}:{digest[:8]}", tag
        return f"hash:{digest[:16]}", tag

    @staticmethod
    def detect_tag(body: dict) -> Optional[str]:
        """Primeira tag [AGENT:...] no conteúdo (mensagens system primeiro)."""
        messages = body.get("messages")
        if not isinstance(messages, list):
            return None
        ordered = sorted(
            (m for m in messages if isinstance(m, dict)),
            key=lambda m: 0 if m.get("role") == "system" else 1,
        )
        for message in ordered:
            match = AGENT_TAG_RE.search(_content_text(message.get("content")))
            if match:
                return match.group(1)
        return None

    def _stable_hash(self, body: dict, client_ip: str, user_agent: str) -> str:
        """Hash estável: primeiro system + primeira user + modelo + IP + UA.

        Não usa timestamp/request-id — a mesma conversa gera sempre a
        mesma chave (PRD F5).
        """
        first_system = ""
        first_user = ""
        messages = body.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, dict):
                    continue
                text = _content_text(message.get("content"))
                if not first_system and message.get("role") == "system":
                    first_system = text
                elif not first_user and message.get("role") == "user":
                    first_user = text
                if first_system and first_user:
                    break
        raw = "\x1f".join(
            [first_system, first_user, str(body.get("model") or ""), client_ip or "",
             user_agent or ""]
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def estimate_prompt_tokens(body: dict) -> int:
        """Estimativa chars//4 sobre as mensagens serializadas (TechSpec)."""
        messages = body.get("messages") or []
        try:
            serialized = json.dumps(messages, ensure_ascii=False)
        except (TypeError, ValueError):
            serialized = str(messages)
        return max(0, len(serialized) // 4)

    # ------------------------------------------------------------------
    # Persistência (ADR-005)
    # ------------------------------------------------------------------

    def _load_sessions(self) -> None:
        if not self._sessions_path.exists():
            return
        try:
            with open(self._sessions_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("[proxy] sessions file unreadable, starting empty: %s", exc)
            return
        now = self._now()
        ttl = self._ttl()
        loaded = 0
        for item in raw.get("sessions", []):
            try:
                session = StickySession(**item)
            except TypeError:
                continue
            if self._is_expired(session, now, ttl):
                continue
            self._sessions[session.affinity_key] = session
            loaded += 1
        if loaded:
            logger.info("[proxy] restored %d sticky session(s) from disk", loaded)

    def _save_sessions(self) -> None:
        """Escrita atômica (.tmp + os.replace), mesmo padrão do ConfigManager."""
        tmp_path = str(self._sessions_path) + ".tmp"
        try:
            self._sessions_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"sessions": [asdict(s) for s in self._sessions.values()]},
                    f,
                    indent=2,
                )
            os.replace(tmp_path, str(self._sessions_path))
            self._unsaved_uses = 0
        except OSError as exc:
            logger.error("[proxy] failed to persist sessions: %s", exc)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    async def flush(self) -> None:
        """Persistência explícita (shutdown)."""
        async with self._lock:
            self._save_sessions()

    # ------------------------------------------------------------------
    # TTL / sessões
    # ------------------------------------------------------------------

    def _settings(self) -> dict:
        return self._config.get_smart_proxy_settings()

    def _ttl(self) -> timedelta:
        return timedelta(minutes=self._settings().get("ttl_minutes", 180))

    @staticmethod
    def _is_expired(session: StickySession, now: datetime, ttl: timedelta) -> bool:
        try:
            last = datetime.fromisoformat(session.last_used_at)
        except (TypeError, ValueError):
            return True
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        return (now - last) > ttl

    def _expire_locked(self) -> None:
        now = self._now()
        ttl = self._ttl()
        expired = [
            key for key, session in self._sessions.items()
            if self._is_expired(session, now, ttl)
        ]
        for key in expired:
            del self._sessions[key]
        if expired:
            logger.info("[proxy] expired %d sticky session(s) by TTL", len(expired))
            self._save_sessions()

    async def sessions(self) -> List[StickySession]:
        async with self._lock:
            self._expire_locked()
            return list(self._sessions.values())

    async def expire_idle(self) -> int:
        """Remove sessões ociosas além do TTL; retorna quantas foram apagadas."""
        async with self._lock:
            before = len(self._sessions)
            self._expire_locked()
            return before - len(self._sessions)

    async def clear_sessions(self, affinity_key: Optional[str] = None) -> int:
        async with self._lock:
            if affinity_key is None:
                removed = len(self._sessions)
                self._sessions.clear()
            else:
                removed = 1 if self._sessions.pop(affinity_key, None) else 0
            if removed:
                self._save_sessions()
            return removed

    # ------------------------------------------------------------------
    # Contadores in-flight
    # ------------------------------------------------------------------

    def _flight_key(self, instance: Dict[str, Any]) -> str:
        """Chave de contagem in-flight — backend_id (portas podem ser compartilhadas)."""
        return self._backend_id(instance)

    def _resolve_flight_key(self, key: str | int) -> str:
        if isinstance(key, str):
            return key
        for inst in self._routing_instances():
            if inst.get("port") == key:
                return self._backend_id(inst)
        return f"port:{key}"

    def in_flight_for(self, instance: Dict[str, Any]) -> int:
        return self._in_flight.get(self._flight_key(instance), 0)

    def in_flight(self, key: str | int) -> int:
        """Contagem por backend; int (porta) só para instâncias locais únicas."""
        return self._in_flight.get(self._resolve_flight_key(key), 0)

    async def acquire(self, backend_key: str | int) -> None:
        bid = self._resolve_flight_key(backend_key)
        async with self._lock:
            self._in_flight[bid] = self._in_flight.get(bid, 0) + 1

    async def release(
        self,
        backend_key: str | int,
        *,
        affinity_key: Optional[str] = None,
        usage: Optional[dict] = None,
    ) -> None:
        bid = self._resolve_flight_key(backend_key)
        async with self._lock:
            current = self._in_flight.get(bid, 0)
            self._in_flight[bid] = max(0, current - 1)
            if affinity_key and usage:
                session = self._sessions.get(affinity_key)
                if session:
                    total = usage.get("total_tokens")
                    if isinstance(total, (int, float)) and total > 0:
                        session.tokens_processed += int(total)

    # ------------------------------------------------------------------
    # Backends / elegibilidade
    # ------------------------------------------------------------------

    def set_backend_enabled(self, port: int, enabled: bool) -> None:
        if enabled:
            self._disabled_ports.discard(port)
            for instance in self._routing_instances():
                if instance.get("port") == port:
                    self._unavailable_until.pop(self._backend_id(instance), None)
        else:
            self._disabled_ports.add(port)
        logger.info(
            "[proxy] backend %s %s by admin", port,
            "enabled" if enabled else "disabled",
        )

    def is_backend_disabled(self, port: int) -> bool:
        return port in self._disabled_ports

    def _backend_cooldown_until(
        self, backend_id: str
    ) -> Optional[datetime]:
        until = self._unavailable_until.get(backend_id)
        if until is not None and until <= self._now():
            self._unavailable_until.pop(backend_id, None)
            return None
        return until

    def _backend_available(self, instance: Dict[str, Any]) -> bool:
        if instance.get("port") in self._disabled_ports:
            return False
        return self._backend_cooldown_until(self._backend_id(instance)) is None

    # ------------------------------------------------------------------
    # Ollama Cloud account integration (Task 07)
    # ------------------------------------------------------------------

    def _ollama_cloud_backend_ids(self) -> List[str]:
        """Return backend_ids for all Ollama Cloud accounts."""
        if self._ollama_cloud_account_manager is None:
            return []
        backend_ids = []
        for account in self._ollama_cloud_account_manager.get_accounts():
            backend_ids.append(f"platform:ollama-cloud:{account.id}")
        return backend_ids

    def _ollama_cloud_candidates(
        self,
        exclude_backend_ids: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        """Build virtual backend dicts for available Ollama Cloud accounts.

        Each account becomes a routing candidate with:
        * backend_id = ``platform:ollama-cloud:<account_id>``
        * ``backend_type = "platform"``
        * ``provider = "ollama-cloud"``
        * ``port`` is a synthetic negative key derived from account id.
        """
        if self._ollama_cloud_account_manager is None:
            return []

        candidates = []
        for account in self._ollama_cloud_account_manager.get_accounts():
            bid = f"platform:ollama-cloud:{account.id}"
            if exclude_backend_ids and bid in exclude_backend_ids:
                continue

            # Skip accounts in cooldown
            if account.cooldown_until is not None:
                if time.time() < account.cooldown_until:
                    continue
                # Cooldown expired — clear it
                account.cooldown_until = None
                account.status = "available"

            if account.status != "available":
                continue

            # Check cooldown expiry on status
            if account.cooldown_until is not None and time.time() < account.cooldown_until:
                continue

            synthetic_port = -(abs(hash(account.id)) % (10**9)) + 40000
            candidates.append({
                "backend_id": bid,
                "backend_type": "platform",
                "provider": "ollama-cloud",
                "status": "running",
                "port": synthetic_port,
                "model": account.label or "Ollama Cloud",
                "model_path": "",
                "config": {},
                "account_id": account.id,
                "account_label": account.label,
                "ollama_cloud_account": account,
            })
        return candidates

    def _pick_ollama_cloud_least_busy(
        self,
        candidates: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Pick the least-busy Ollama Cloud account.

        Prioritizes accounts NOT in cooldown. Among accounts at the same
        cooldown state, picks the one with the lowest in-flight count
        (synthetic accounts always have 0 in-flight).
        """
        if not candidates:
            return None

        def cooldown_priority(inst: Dict[str, Any]) -> int:
            """0 = not in cooldown (best), 1 = in cooldown (worst)."""
            acc = inst.get("ollama_cloud_account")
            if acc is not None and acc.cooldown_until is not None:
                return 1
            return 0

        return min(
            candidates,
            key=lambda i: (
                cooldown_priority(i),
                self.in_flight_for(i),
                i["port"],
                id(i),
            ),
        )

    async def handle_http_error(
        self,
        status_code: int,
        account: Any,
        *,
        retry_after: Optional[float] = None,
        reason: str = "upstream_error",
    ) -> bool:
        """Handle an HTTP error from the Ollama Cloud provider.

        Applies cooldown to the account and marks the backend unavailable.

        Returns ``True`` if cooldown was applied (failover may be needed).
        """
        if self._ollama_cloud_account_manager is None:
            return False

        # Determine cooldown duration based on status code
        if 400 <= status_code < 500:
            # 4xx — likely rate limit; use retry_after or default
            cooldown_seconds = retry_after or DEFAULT_BACKEND_COOLDOWN_SECONDS
        elif status_code >= 500:
            # 5xx — server error; shorter cooldown
            cooldown_seconds = (retry_after or 30) if retry_after else 30
        else:
            # Other errors — default cooldown
            cooldown_seconds = DEFAULT_BACKEND_COOLDOWN_SECONDS

        # Apply cooldown to the account
        self._ollama_cloud_account_manager.apply_cooldown(account, cooldown_seconds)

        # Mark the backend unavailable in proxy_router
        backend_id = f"platform:ollama-cloud:{account.id}"
        await self._mark_backend_unavailable_locked(
            backend_id,
            cooldown_seconds=cooldown_seconds,
            reason=reason,
        )
        return True

    async def _mark_backend_unavailable_locked(
        self,
        backend_id: str,
        cooldown_seconds: float,
        *,
        reason: str,
    ) -> datetime:
        """Internal: mark backend unavailable without re-checking normalization.

        Used by ``handle_http_error`` to avoid calling ``mark_backend_unavailable``
        directly (which would re-normalize and re-validate the backend_id).
        """
        seconds = max(1.0, float(cooldown_seconds))
        async with self._lock:
            until = self._now() + timedelta(seconds=seconds)
            previous = self._backend_cooldown_until(backend_id)
            if previous is not None and previous > until:
                until = previous
            self._unavailable_until[backend_id] = until
        logger.warning(
            "[proxy] backend_id=%s in cooldown until=%s reason=%s",
            backend_id, _iso(until), reason,
        )
        return until

    async def mark_backend_unavailable(
        self,
        backend_id: str,
        cooldown_seconds: float = DEFAULT_BACKEND_COOLDOWN_SECONDS,
        *,
        reason: str = "upstream_error",
    ) -> datetime:
        """Abre o circuito de um backend sem afetar integrações na mesma porta."""
        normalized = normalize_backend_id(backend_id)
        if not normalized:
            raise ValueError("backend_id obrigatorio")
        seconds = max(1.0, float(cooldown_seconds))
        async with self._lock:
            until = self._now() + timedelta(seconds=seconds)
            previous = self._backend_cooldown_until(normalized)
            if previous is not None and previous > until:
                until = previous
            self._unavailable_until[normalized] = until
        logger.warning(
            "[proxy] backend_id=%s in cooldown until=%s reason=%s",
            normalized, _iso(until), reason,
        )
        return until

    def _running_instances(self) -> List[Dict[str, Any]]:
        status = self._get_status() or {}
        return [
            inst for inst in status.get("instances", [])
            if inst.get("status") == "running" and inst.get("port") is not None
        ]

    def _routing_instances(self) -> List[Dict[str, Any]]:
        """Return local/sidecar instances plus direct Ollama Cloud accounts."""
        instances = self._running_instances()
        known_ids = {self._backend_id(inst) for inst in instances}
        instances.extend(
            inst for inst in self._ollama_cloud_candidates()
            if self._backend_id(inst) not in known_ids
        )
        return instances

    def _model_flags(self, model_configs: dict, model_path: str) -> Tuple[bool, int]:
        cfg = lookup_model_config(model_configs, model_path or "")
        eligible = cfg.get("proxy_eligible", DEFAULT_PROXY_ELIGIBLE)
        max_parallel = cfg.get("max_parallel_requests", DEFAULT_MAX_PARALLEL_REQUESTS)
        if not isinstance(max_parallel, int) or max_parallel < 1:
            max_parallel = DEFAULT_MAX_PARALLEL_REQUESTS
        return bool(eligible), max_parallel

    @staticmethod
    def _backend_type(instance: Dict[str, Any]) -> str:
        backend_type = instance.get("backend_type")
        if backend_type:
            return str(backend_type)
        backend_id = normalize_backend_id(instance.get("backend_id"))
        return "platform" if backend_id.startswith("platform:") else "local"

    def _backend_id(self, instance: Dict[str, Any]) -> str:
        backend_id = normalize_backend_id(instance.get("backend_id"))
        if backend_id:
            return backend_id
        if self._backend_type(instance) == "platform":
            return f"platform:{instance.get('port')}"
        model_path = instance.get("model_path") or ""
        if model_path:
            return f"local:{normalize_model_path(model_path)}"
        return f"local:{instance.get('port')}"

    def _config_backend_id(self, instance: Dict[str, Any]) -> str:
        """Map per-account cloud backends to their shared platform config."""
        backend_id = self._backend_id(instance)
        if instance.get("provider") == "ollama-cloud":
            return "platform:ollama-cloud"
        return backend_id

    def benchmark_key(self, instance: Dict[str, Any]) -> str:
        """Identidade estavel usada pelo ranking entre reinicializacoes."""
        if self._backend_type(instance) == "platform":
            return self._backend_id(instance)
        model_path = instance.get("model_path") or ""
        if model_path:
            return f"local:{normalize_model_path(model_path)}"
        return self._backend_id(instance)

    def set_benchmark_results(
        self,
        latencies_ms: Mapping[str, float],
        *,
        measured_at: Optional[str] = None,
    ) -> None:
        """Substitui atomicamente o ranking medido no startup."""
        clean: Dict[str, float] = {}
        for key, value in latencies_ms.items():
            try:
                latency = float(value)
            except (TypeError, ValueError):
                continue
            if key and latency > 0:
                clean[str(key)] = latency
        self._benchmark_latencies_ms = clean
        self._benchmark_measured_at = measured_at or _iso(self._now())
        logger.info(
            "[proxy] startup speed ranking loaded backends=%s",
            ", ".join(
                f"{key}={value:.0f}ms"
                for key, value in sorted(clean.items(), key=lambda item: item[1])
            ) or "none",
        )

    def _benchmark_latency(self, instance: Dict[str, Any]) -> Optional[float]:
        return self._benchmark_latencies_ms.get(self.benchmark_key(instance))

    def _backend_flags(
        self, config: dict, instance: Dict[str, Any]
    ) -> Tuple[bool, int]:
        if self._backend_type(instance) == "platform":
            cfg = lookup_platform_config(
                config.get("platform_configs", {}), self._config_backend_id(instance)
            )
            eligible = cfg.get("proxy_eligible", False)
            max_parallel = cfg.get(
                "max_parallel_requests", DEFAULT_MAX_PARALLEL_REQUESTS
            )
            if not isinstance(max_parallel, int) or max_parallel < 1:
                max_parallel = DEFAULT_MAX_PARALLEL_REQUESTS
            return bool(eligible), max_parallel
        return self._model_flags(
            config.get("model_configs", {}), instance.get("model_path") or ""
        )

    def _platform_default_model(self, instance: Dict[str, Any]) -> Optional[str]:
        """Modelo padrão configurado para a plataforma (usado quando secundária)."""
        if self._backend_type(instance) != "platform":
            return None
        config = self._config.get_config()
        cfg = lookup_platform_config(
            config.get("platform_configs", {}), self._config_backend_id(instance)
        )
        value = cfg.get("default_model")
        return value or None

    def _internal_model(
        self,
        instance: Dict[str, Any],
        external_model: str,
        is_primary: bool = True,
    ) -> str:
        if self._backend_type(instance) == "platform":
            # Secundária: o modelo pedido pertence ao principal e não vale para
            # esta plataforma; encaminha para o modelo padrão configurado.
            if not is_primary:
                default = self._platform_default_model(instance)
                if default:
                    return default
            return external_model or instance.get("model") or ""
        return instance.get("model") or ""

    def _ctx_per_slot(
        self, instance: Dict[str, Any], model_name: str = ""
    ) -> int:
        config = instance.get("config") or {}
        if (
            self._backend_type(instance) == "platform"
            and self._context_limit_resolver is not None
            and model_name
        ):
            try:
                platform_limit = self._context_limit_resolver(
                    instance, model_name
                )
            except Exception as exc:
                logger.warning(
                    "[proxy] failed to resolve context limit backend=%s model=%s: %s",
                    self._backend_id(instance), model_name, exc,
                )
            else:
                if isinstance(platform_limit, int) and platform_limit > 0:
                    return platform_limit
                if platform_limit == 0:
                    # O catalogo do provedor existe, mas este modelo concreto
                    # nao pertence a ele: backend incompativel, nao "desconhecido".
                    return 0
                return UNKNOWN_PLATFORM_CONTEXT_LIMIT
        ctx = config.get("context_size") or DEFAULT_CONTEXT_SIZE
        slots = config.get("parallel_slots") or DEFAULT_PARALLEL_SLOTS
        return int(ctx) // max(1, int(slots))

    def _find_primary(
        self, instances: List[Dict[str, Any]], settings: dict
    ) -> Optional[Dict[str, Any]]:
        primary_backend_id = normalize_backend_id(settings.get("primary_backend_id"))
        if primary_backend_id:
            matches = [
                inst for inst in instances
                if self._backend_id(inst) == primary_backend_id
                or self._config_backend_id(inst) == primary_backend_id
            ]
            if not matches:
                return None
            return min(matches, key=lambda i: i["port"])
        primary_model_path = settings.get("primary_model_path")
        if not primary_model_path:
            return None
        norm = normalize_model_path(primary_model_path)
        matches = [
            inst for inst in instances
            if normalize_model_path(inst.get("model_path") or "") == norm
        ]
        if not matches:
            return None
        # Duas instâncias do mesmo model_path: resolve para a de menor porta (ADR-005)
        return min(matches, key=lambda i: i["port"])

    def _find_requested_primary(
        self, instances: List[Dict[str, Any]], requested_model: str
    ) -> Optional[Dict[str, Any]]:
        """Resolve o principal dinamico indicado pelo campo ``model``."""
        if not requested_model:
            return None
        if self._requested_primary_resolver is not None:
            resolved = self._requested_primary_resolver(instances, requested_model)
            if resolved is not None:
                return resolved
        requested_norm = normalize_model_path(requested_model)
        matches = [
            instance for instance in instances
            if instance.get("model") == requested_model
            or instance.get("model_path") == requested_model
            or (
                self._backend_type(instance) != "platform"
                and normalize_model_path(instance.get("model_path") or "")
                == requested_norm
            )
        ]
        return min(matches, key=lambda item: item["port"]) if matches else None

    def backends_snapshot(self) -> List[Dict[str, Any]]:
        """Visão administrativa dos backends (estados PRD F3)."""
        settings = self._settings()
        primary_backend_id = normalize_backend_id(settings.get("primary_backend_id"))
        primary_norm = (
            normalize_model_path(settings["primary_model_path"])
            if not primary_backend_id and settings.get("primary_model_path") else None
        )
        config = self._config.get_config()
        snapshot = []
        for inst in self._routing_instances():
            port = inst["port"]
            model_path = inst.get("model_path")
            backend_id = self._backend_id(inst)
            backend_type = self._backend_type(inst)
            eligible, max_parallel = self._backend_flags(config, inst)
            if primary_backend_id:
                is_primary = (
                    backend_id == primary_backend_id
                    or self._config_backend_id(inst) == primary_backend_id
                )
            else:
                is_primary = (
                    primary_norm is not None
                    and normalize_model_path(model_path or "") == primary_norm
                )
            in_flight = self.in_flight_for(inst)
            cooldown_until = self._backend_cooldown_until(backend_id)
            if port in self._disabled_ports:
                state = "disabled"
            elif cooldown_until is not None:
                state = "cooldown"
            elif not eligible and (backend_type == "platform" or not is_primary):
                state = "not_eligible"
            elif in_flight > 0:
                # Qualquer requisição ativa = OCUPADO na UI (roteamento
                # continua usando max_parallel em _candidates).
                state = "busy"
            else:
                state = "online"
            latency_ms = self._benchmark_latency(inst)
            snapshot.append({
                "port": port,
                "model": inst.get("model"),
                "model_path": model_path,
                "backend_id": backend_id,
                "backend_type": backend_type,
                "provider": inst.get("provider"),
                "gpu": gpu_label(inst),
                "role": "primary" if is_primary else "secondary",
                "state": state,
                "in_flight": in_flight,
                "max_parallel": max_parallel,
                # O limite configurado é a capacidade inicial. Sob pressão o
                # roteador admite mais trabalho; exponha a capacidade efetiva
                # para a UI não mostrar um contador enganoso (ex.: 3/1).
                "effective_parallel": max(max_parallel, in_flight),
                "capacity_mode": "dynamic",
                "ctx_per_slot": self._ctx_per_slot(inst),
                "startup_latency_ms": (
                    round(latency_ms, 1) if latency_ms is not None else None
                ),
                "benchmark_measured_at": self._benchmark_measured_at,
                "cooldown_until": (
                    _iso(cooldown_until) if cooldown_until is not None else None
                ),
            })
        snapshot.sort(
            key=lambda backend: (
                0 if backend["role"] == "primary" else 1,
                backend["startup_latency_ms"] is None,
                backend["startup_latency_ms"] or float("inf"),
                backend["port"],
            )
        )
        rank = 1
        for backend in snapshot:
            if backend["role"] == "primary":
                backend["speed_rank"] = 0
            elif backend["startup_latency_ms"] is not None:
                backend["speed_rank"] = rank
                rank += 1
            else:
                backend["speed_rank"] = None
        return snapshot

    # ------------------------------------------------------------------
    # Seleção (ADR-001)
    # ------------------------------------------------------------------

    def _candidates(
        self,
        instances: List[Dict[str, Any]],
        config: dict,
        primary_port: Optional[int],
        needed_ctx: int,
        ignore_capacity: bool,
        exclude_ports: Optional[set] = None,
        exclude_backend_ids: Optional[set] = None,
        external_model: str = "",
        configured_primary_backend_id: str = "",
    ) -> List[Dict[str, Any]]:
        """Backends elegíveis para NOVA sessão (PRD F6)."""
        result = []
        all_instances = list(instances)
        known_ids = {self._backend_id(inst) for inst in all_instances}
        all_instances.extend(
            inst for inst in self._ollama_cloud_candidates(
                exclude_backend_ids=exclude_backend_ids
            )
            if self._backend_id(inst) not in known_ids
        )
        for inst in all_instances:
            port = inst["port"]
            backend_id = self._backend_id(inst)
            config_backend_id = self._config_backend_id(inst)
            if exclude_backend_ids and (
                backend_id in exclude_backend_ids
                or config_backend_id in exclude_backend_ids
            ):
                continue
            if exclude_ports and port in exclude_ports:
                continue
            if not self._backend_available(inst):
                continue
            eligible, max_parallel = self._backend_flags(config, inst)
            is_primary = (
                backend_id == configured_primary_backend_id
                or config_backend_id == configured_primary_backend_id
                if configured_primary_backend_id
                else port == primary_port
            )
            if not eligible and not is_primary:
                continue
            internal_model = self._internal_model(
                inst,
                external_model,
                is_primary,
            )
            if needed_ctx > self._ctx_per_slot(inst, internal_model):
                continue
            if not ignore_capacity and self.in_flight_for(inst) >= max_parallel:
                continue
            result.append(inst)
        return result

    def _pick_least_busy(
        self,
        candidates: List[Dict[str, Any]],
        primary_port: Optional[int] = None,
        primary_backend_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Menos ocupado por in-flight; empates: menos sessões sticky
        atribuídas, preferência por backend principal (por id, não porta),
        depois evita o mesmo sidecar do principal (outra integração na
        mesma porta tende a sofrer a mesma contenção), depois instâncias de
        plataforma (poupa GPU local dedicada), depois contas Ollama Cloud
        sem cooldown, e menor porta. (Task 07)"""
        if not candidates:
            return None
        session_counts: Dict[str, int] = {}
        for session in self._sessions.values():
            sid = session.backend_id or f"port:{session.backend_port}"
            session_counts[sid] = session_counts.get(sid, 0) + 1
        return min(
            candidates,
            key=lambda i: (
                0 if (
                    primary_backend_id
                    and self._backend_id(i) == primary_backend_id
                ) else 1,
                self._benchmark_latency(i) is None,
                self._benchmark_latency(i) or float("inf"),
                self.in_flight_for(i),
                session_counts.get(self._backend_id(i), 0),
                1 if (primary_port is not None and i["port"] == primary_port) else 0,
                0 if self._backend_type(i) == "platform" else 1,
                # (Task 07) Priorize Ollama Cloud accounts NOT in cooldown
                (
                    0 if (
                        i.get("backend_type") == "platform"
                        and i.get("provider") == "ollama-cloud"
                        and i.get("ollama_cloud_account") is not None
                        and i["ollama_cloud_account"].cooldown_until is None
                    ) else 1
                ),
                i["port"],
            ),
        )

    def _pick_for_dynamic_growth(
        self,
        candidates: List[Dict[str, Any]],
        config: dict,
        primary_backend_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Escolhe onde abrir capacidade adicional quando todos estão cheios.

        ``max_parallel_requests`` é a capacidade inicial (soft limit), não um
        teto. A carga relativa preserva a proporção configurada entre backends;
        em empate o principal, normalmente o modelo mais forte, recebe
        prioridade. Depois vêm os secundários menos carregados.
        """
        if not candidates:
            return None

        def load_key(instance: Dict[str, Any]) -> tuple:
            _, initial_capacity = self._backend_flags(config, instance)
            in_flight = self.in_flight_for(instance)
            return (
                in_flight / max(1, initial_capacity),
                0 if self._backend_id(instance) == primary_backend_id else 1,
                self._benchmark_latency(instance) is None,
                self._benchmark_latency(instance) or float("inf"),
                in_flight,
                0 if self._backend_type(instance) == "platform" else 1,
                instance["port"],
            )

        return min(candidates, key=load_key)

    def _register_session_locked(
        self,
        affinity_key: str,
        instance: Dict[str, Any],
        external_model: str,
        tag: Optional[str],
        is_primary: bool = True,
    ) -> StickySession:
        now_iso = _iso(self._now())
        session = StickySession(
            affinity_key=affinity_key,
            backend_port=instance["port"],
            backend_model_path=instance.get("model_path") or "",
            external_model=external_model,
            internal_model=self._internal_model(
                instance, external_model, is_primary
            ),
            detected_tag=tag,
            created_at=now_iso,
            last_used_at=now_iso,
            backend_id=self._backend_id(instance),
            backend_type=self._backend_type(instance),
            provider=instance.get("provider"),
        )
        self._sessions[affinity_key] = session
        self._save_sessions()
        return session

    def _touch_session_locked(self, session: StickySession) -> None:
        session.last_used_at = _iso(self._now())
        session.request_count += 1
        self._unsaved_uses += 1
        if self._unsaved_uses >= PERSIST_EVERY_N_REQUESTS:
            self._save_sessions()

    def _hash_branch_locked(
        self,
        affinity_key: str,
        by_port: Dict[int, Dict[str, Any]],
        instances: List[Dict[str, Any]],
        config: dict,
        primary_port: int,
        primary_backend_id: str,
        needed_ctx: int,
        external_model: str,
        configured_primary_backend_id: str,
        _decision,
        _commit,
    ) -> Optional[RouteDecision]:
        """Ramifica uma sessão hash: ocupada para um backend livre.

        Sessões de afinidade fraca não distinguem subagentes com prompts
        iniciais idênticos; sob concorrência, cada fluxo extra vira um ramo
        sticky próprio (hash:...#2, #3, ...) no backend menos ocupado.
        Retorna None quando não há backend livre (o chamador espera).
        """
        for n in range(2, MAX_HASH_BRANCHES + 1):
            branch_key = f"{affinity_key}#{n}"
            branch = self._sessions.get(branch_key)
            if branch is not None:
                b_inst = None
                if branch.backend_id:
                    b_inst = next(
                        (
                            i for i in instances
                            if self._backend_id(i) == branch.backend_id
                        ),
                        None,
                    )
                if b_inst is None:
                    b_inst = by_port.get(branch.backend_port)
                if b_inst is None or not self._backend_available(b_inst):
                    continue  # ramo órfão: deixa o TTL limpar
                b_internal_model = self._internal_model(
                    b_inst,
                    external_model,
                    self._backend_id(b_inst) == configured_primary_backend_id,
                )
                if needed_ctx > self._ctx_per_slot(b_inst, b_internal_model):
                    continue
                _, b_max = self._backend_flags(config, b_inst)
                if self.in_flight_for(b_inst) < b_max:
                    return _commit(
                        b_inst, True, "sticky_branch", branch, key=branch_key
                    )
                continue  # ramo também ocupado: tenta o próximo
            candidates = self._candidates(
                instances, config, primary_port, needed_ctx,
                ignore_capacity=False,
                external_model=external_model,
                configured_primary_backend_id=configured_primary_backend_id,
            )
            chosen = self._pick_least_busy(
                candidates, primary_port, primary_backend_id
            )
            if chosen is None:
                return None
            logger.info(
                "[proxy] concurrent hash session branched affinity_key=%s "
                "branch=%s selected_backend=%s reason=hash_branch",
                affinity_key, branch_key, chosen["port"],
            )
            return _commit(chosen, False, "hash_branch", None, key=branch_key)
        return None

    async def plan_route(
        self,
        *,
        headers: Mapping[str, str],
        body: dict,
        client_ip: str,
        user_agent: str,
    ) -> RoutePlan:
        """Planeja o roteamento de forma livre de efeitos colaterais.

        Não altera sessões sticky, não afeta contadores in_flight nem salva em disco.
        """
        settings = self._settings()
        max_wait = settings.get("max_wait_seconds", 30)
        deadline = asyncio.get_event_loop().time() + max_wait

        while True:
            async with self._lock:
                decision, wait_reason = self._resolve_locked(
                    settings=settings,
                    headers=headers,
                    body=body,
                    client_ip=client_ip,
                    user_agent=user_agent,
                    apply_side_effects=False,
                    ignore_capacity=False,
                )
                if decision is not None:
                    return RoutePlan(
                        decision=decision,
                        commit_token=str(uuid.uuid4()),
                        created_at=_iso(self._now()),
                    )
            if asyncio.get_event_loop().time() >= deadline:
                raise ProxyError(
                    503,
                    wait_reason or "Todos os backends ocupados; tente novamente",
                    code="backend_busy",
                )
            await asyncio.sleep(BUSY_POLL_SECONDS)

    async def plan_larger_window(
        self,
        *,
        headers: Mapping[str, str],
        body: dict,
        client_ip: str,
        user_agent: str,
        current_limit: int,
        required_capabilities: Any,
        candidate_evaluator: Optional[
            Callable[[Dict[str, Any]], Tuple[ModelLimits, FrozenSet[str]]]
        ] = None,
        fits_checker: Optional[
            Callable[[Dict[str, Any], int], bool]
        ] = None,
    ) -> Optional[RoutePlan]:
        """Planeja um desvio para backend de janela maior livre de efeitos colaterais.

        Procura um candidato elegível com janela estritamente maior que `current_limit`,
        com limite de contexto conhecido e que possua superset das `required_capabilities`.
        Não altera sticky sessions, in_flight nem salva em disco.
        """
        async with self._lock:
            self._expire_locked()
            affinity_key, tag = self.extract_affinity(headers, body, client_ip, user_agent)
            instances = self._routing_instances()
            config = self._config.get_config()
            settings = self._settings()

            requested_model = str(body.get("model") or "")
            requested_primary = self._find_requested_primary(instances, requested_model)
            configured_primary = self._find_primary(instances, settings)
            configured_backend_id = normalize_backend_id(
                settings.get("primary_backend_id")
            )
            if not configured_backend_id and settings.get("primary_model_path"):
                configured_backend_id = (
                    f"local:{normalize_model_path(settings['primary_model_path'])}"
                )

            primary = requested_primary or configured_primary
            primary_backend_id = (
                self._backend_id(primary) if primary else configured_backend_id
            )

            configured_primary_is_platform = configured_backend_id.startswith("platform:")
            if requested_primary is not None:
                external_model = requested_model or primary.get("model") or ""
            elif configured_primary_is_platform and requested_model:
                external_model = requested_model
            elif configured_primary is not None:
                external_model = configured_primary.get("model") or requested_model
            elif settings.get("primary_model_path"):
                external_model = Path(settings["primary_model_path"]).name
            else:
                external_model = requested_model

            if not external_model and primary:
                external_model = primary.get("model") or ""

            req_set: FrozenSet[str]
            if hasattr(required_capabilities, "as_set"):
                req_set = required_capabilities.as_set()
            elif isinstance(required_capabilities, (set, frozenset)):
                req_set = frozenset(required_capabilities)
            else:
                req_set = derive_required_capabilities(body).as_set()

            candidates_available = []
            candidates_all_capacity = []

            for inst in instances:
                if not self._backend_available(inst):
                    continue
                eligible, max_parallel = self._backend_flags(config, inst)
                is_primary = (
                    configured_primary is not None
                    and self._backend_id(inst) == configured_backend_id
                )
                if not eligible and not is_primary:
                    continue

                if candidate_evaluator is not None:
                    limits, caps = candidate_evaluator(inst)
                else:
                    limits = resolve_model_limits(inst)
                    caps = derive_target_capabilities(inst)

                # Requisito 2: Limite conhecido e estritamente maior que a janela atual
                if not limits.is_known or limits.context_tokens is None or limits.context_tokens <= current_limit:
                    continue

                # Requisito 3: Superset confirmado de capacidades
                if not req_set.issubset(caps):
                    continue

                # Requisito 4: Reavaliação no candidato
                if fits_checker is not None:
                    if not fits_checker(inst, limits.context_tokens):
                        continue
                else:
                    needed_ctx = int(self.estimate_prompt_tokens(body) * TOKEN_ESTIMATE_MARGIN)
                    if needed_ctx > limits.context_tokens:
                        continue

                candidates_all_capacity.append(inst)
                if self.in_flight_for(inst) < max_parallel:
                    candidates_available.append(inst)

            if not candidates_all_capacity:
                return None

            # Requisito 5: Escolha utilizando regras do roteador
            chosen = self._pick_least_busy(
                candidates_available or candidates_all_capacity,
                primary_backend_id=primary_backend_id or None,
            )
            if chosen is None:
                chosen = self._pick_for_dynamic_growth(
                    candidates_all_capacity, config, primary_backend_id
                )
            if chosen is None:
                return None

            chosen_is_primary = (
                configured_primary is not None
                and self._backend_id(chosen) == configured_backend_id
            )

            decision = RouteDecision(
                backend_port=chosen["port"],
                internal_model=self._internal_model(
                    chosen, external_model, chosen_is_primary
                ),
                external_model=external_model,
                affinity_key=affinity_key,
                detected_tag=tag,
                sticky_hit=False,
                reason="fallback_larger_window",
                rewrite=(
                    not chosen_is_primary
                    and self._backend_type(chosen) != "platform"
                ),
                prompt_tokens_estimated=self.estimate_prompt_tokens(body),
                gpu=gpu_label(chosen),
                backend_id=self._backend_id(chosen),
                backend_type=self._backend_type(chosen),
                provider=chosen.get("provider"),
            )

            return RoutePlan(
                decision=decision,
                commit_token=str(uuid.uuid4()),
                created_at=_iso(self._now()),
            )

    async def commit_route(self, plan: RoutePlan) -> RouteDecision:
        """Efetiva um RoutePlan sob lock após revalidação.

        Rejeita commits duplicados do mesmo plano e valida se o backend
        destino continua disponível. Em caso de plano obsoleto, levanta StaleRoutePlan.
        """
        async with self._lock:
            if plan.commit_token in self._committed_tokens:
                raise ProxyError(
                    400,
                    f"Plano com token {plan.commit_token[:8]} ja foi commitado.",
                    code="duplicate_commit",
                )

            decision = plan.decision
            instances = self._routing_instances()
            target_inst = None
            if decision.backend_id:
                target_inst = next(
                    (inst for inst in instances if self._backend_id(inst) == decision.backend_id),
                    None,
                )
            if target_inst is None:
                target_inst = next(
                    (inst for inst in instances if inst["port"] == decision.backend_port),
                    None,
                )

            if target_inst is None or not self._backend_available(target_inst):
                raise StaleRoutePlan(plan)

            # Revalidar capacidade de contexto do modelo
            needed_ctx = int(decision.prompt_tokens_estimated * TOKEN_ESTIMATE_MARGIN)
            settings = self._settings()
            configured_primary = self._find_primary(instances, settings)
            configured_backend_id = (
                self._backend_id(configured_primary) if configured_primary else ""
            )
            configured_primary_active = (
                configured_primary is not None and self._backend_available(configured_primary)
            )
            is_primary = bool(
                configured_primary_active
                and self._backend_id(target_inst) == configured_backend_id
            )
            internal_model = self._internal_model(
                target_inst, decision.external_model, is_primary
            )
            if needed_ctx > self._ctx_per_slot(target_inst, internal_model):
                raise StaleRoutePlan(plan)

            # Efetivar a sessão sticky
            existing = self._sessions.get(decision.affinity_key)
            if existing is None:
                existing = self._register_session_locked(
                    decision.affinity_key,
                    target_inst,
                    decision.external_model,
                    decision.detected_tag,
                    is_primary,
                )
                logger.info(
                    "[proxy] new sticky session affinity_key=%s "
                    "selected_backend=%s reason=%s",
                    decision.affinity_key, target_inst["port"], decision.reason,
                )
            else:
                old_port = existing.backend_port
                existing.backend_port = target_inst["port"]
                existing.backend_model_path = target_inst.get("model_path") or ""
                existing.internal_model = internal_model
                existing.backend_id = self._backend_id(target_inst)
                existing.backend_type = self._backend_type(target_inst)
                existing.provider = target_inst.get("provider")
                self._touch_session_locked(existing)
                if old_port != target_inst["port"] and decision.reason.startswith("reassign"):
                    reason_name = (
                        "backend_down"
                        if "backend_down" in decision.reason
                        else ("context_limit" if "context_limit" in decision.reason else decision.reason)
                    )
                    logger.warning(
                        "[proxy] reassigned affinity_key=%s old_backend=%s "
                        "new_backend=%s reason=%s",
                        decision.affinity_key, old_port, target_inst["port"], reason_name,
                    )

            # Incrementar in_flight
            fk = self._flight_key(target_inst)
            self._in_flight[fk] = self._in_flight.get(fk, 0) + 1

            self._committed_tokens.add(plan.commit_token)

            return decision

    async def resolve(
        self,
        *,
        headers: Mapping[str, str],
        body: dict,
        client_ip: str,
        user_agent: str,
        dry_run: bool = False,
    ) -> RouteDecision:
        """Decide o backend para a requisição ao modelo principal (wrapper retrocompatível).

        Quando dry_run=False a decisão é commitada e reserva o slot (in-flight +1);
        o chamador DEVE chamar release(port, ...) ao concluir.
        """
        if dry_run:
            async with self._lock:
                settings = self._settings()
                decision, wait_reason = self._resolve_locked(
                    settings=settings,
                    headers=headers,
                    body=body,
                    client_ip=client_ip,
                    user_agent=user_agent,
                    apply_side_effects=False,
                    ignore_capacity=True,
                )
                if decision is not None:
                    return decision
                raise ProxyError(503, wait_reason or "Nenhum backend disponivel")

        while True:
            plan = await self.plan_route(
                headers=headers,
                body=body,
                client_ip=client_ip,
                user_agent=user_agent,
            )
            try:
                return await self.commit_route(plan)
            except StaleRoutePlan:
                continue

    def _resolve_locked(
        self,
        *,
        settings: dict,
        headers: Mapping[str, str],
        body: dict,
        client_ip: str,
        user_agent: str,
        apply_side_effects: bool = True,
        ignore_capacity: bool = False,
    ) -> Tuple[Optional[RouteDecision], Optional[str]]:
        """Uma tentativa de decisão sob o lock.

        Retorna (decision, None) em sucesso, (None, motivo) quando deve
        aguardar slot livre. Levanta ProxyError para falhas definitivas.
        """
        self._expire_locked()
        affinity_key, tag = self.extract_affinity(headers, body, client_ip, user_agent)
        needed_ctx = int(self.estimate_prompt_tokens(body) * TOKEN_ESTIMATE_MARGIN)
        est_tokens = self.estimate_prompt_tokens(body)

        instances = self._routing_instances()
        config = self._config.get_config()
        requested_model = str(body.get("model") or "")
        requested_primary = self._find_requested_primary(instances, requested_model)
        configured_primary = self._find_primary(instances, settings)
        configured_backend_id = normalize_backend_id(
            settings.get("primary_backend_id")
        )
        if not configured_backend_id and settings.get("primary_model_path"):
            configured_backend_id = (
                f"local:{normalize_model_path(settings['primary_model_path'])}"
            )

        # O modelo explicitamente invocado passa a ser o principal apenas
        # desta requisicao. A configuracao fixa e o fallback quando ``model``
        # nao identifica uma instancia online.
        primary = requested_primary or configured_primary
        dynamic_primary = requested_primary is not None
        if dynamic_primary:
            configured_backend_id = self._backend_id(primary)

        if configured_primary is not None and not dynamic_primary:
            primary_eligible, _ = self._backend_flags(config, configured_primary)
            if (
                self._backend_type(configured_primary) == "platform"
                and not primary_eligible
            ):
                raise ProxyError(
                    503,
                    "Backend de plataforma principal nao esta habilitado para o proxy.",
                    code="backend_not_eligible",
                )

        configured_primary_is_platform = configured_backend_id.startswith("platform:")
        if dynamic_primary:
            external_model = requested_model or primary.get("model") or ""
        elif configured_primary_is_platform and requested_model:
            external_model = requested_model
        elif configured_primary is not None:
            external_model = configured_primary.get("model") or requested_model
        elif settings.get("primary_model_path"):
            external_model = Path(settings["primary_model_path"]).name
        else:
            external_model = requested_model

        configured_primary_active = (
            primary is not None and self._backend_available(primary)
        )
        if not configured_primary_active:
            fallback = self._candidates(
                instances,
                config,
                None,
                needed_ctx,
                ignore_capacity=ignore_capacity,
                exclude_backend_ids=(
                    {configured_backend_id} if configured_backend_id else None
                ),
                external_model=external_model,
                configured_primary_backend_id=configured_backend_id,
            )
            platform_fallback = [
                instance for instance in fallback
                if self._backend_type(instance) == "platform"
            ]
            primary = self._pick_least_busy(
                platform_fallback or fallback,
                primary_backend_id=configured_backend_id or None,
            )
            if primary is None:
                raise ProxyError(
                    503,
                    "Backend principal indisponivel e nenhum redundante esta online.",
                    code="no_backend",
                )

        primary_port = primary["port"]
        primary_backend_id = self._backend_id(primary)

        def _is_primary_backend(instance: Dict[str, Any]) -> bool:
            return bool(
                configured_primary_active
                and self._backend_id(instance) == configured_backend_id
            )

        if not external_model:
            external_model = primary.get("model") or ""

        def _context_limit(instance: Dict[str, Any]) -> int:
            return self._ctx_per_slot(
                instance,
                self._internal_model(
                    instance, external_model, _is_primary_backend(instance)
                ),
            )

        # Faz a validacao uma unica vez, usando o modelo concreto que cada
        # backend executaria. O limite do Antigravity pode habilitar failover
        # para ele, mas nunca altera o limite do Luna no catalogo publico.
        eligible_context_limits = []
        for instance in instances:
            if not self._backend_available(instance):
                continue
            eligible, _ = self._backend_flags(config, instance)
            if not eligible and self._backend_id(instance) != configured_backend_id:
                continue
            eligible_context_limits.append(_context_limit(instance))
        if (
            eligible_context_limits
            and needed_ctx > max(eligible_context_limits)
        ):
            max_context = max(eligible_context_limits)
            raise ProxyError(
                413,
                "O contexto desta conversa excede o limite dos modelos "
                f"disponiveis (estimado: {est_tokens} tokens; "
                f"limite seguro: {max_context}).",
                code="context_too_large",
            )

        def _decision(
            instance: Dict[str, Any], sticky_hit: bool, reason: str,
            key: Optional[str] = None,
        ) -> RouteDecision:
            return RouteDecision(
                backend_port=instance["port"],
                internal_model=self._internal_model(
                    instance, external_model, _is_primary_backend(instance)
                ),
                external_model=external_model,
                affinity_key=key or affinity_key,
                detected_tag=tag,
                sticky_hit=sticky_hit,
                reason=reason,
                rewrite=(
                    not _is_primary_backend(instance)
                    and self._backend_type(instance) != "platform"
                ),
                prompt_tokens_estimated=est_tokens,
                gpu=gpu_label(instance),
                backend_id=self._backend_id(instance),
                backend_type=self._backend_type(instance),
                provider=instance.get("provider"),
            )

        def _commit(
            instance: Dict[str, Any],
            sticky_hit: bool,
            reason: str,
            session: Optional[StickySession],
            key: Optional[str] = None,
        ) -> RouteDecision:
            key = key or affinity_key
            if apply_side_effects:
                if session is None:
                    session = self._register_session_locked(
                        key, instance, external_model, tag,
                        _is_primary_backend(instance),
                    )
                    logger.info(
                        "[proxy] new sticky session affinity_key=%s "
                        "selected_backend=%s reason=%s",
                        key, instance["port"], reason,
                    )
                self._touch_session_locked(session)
                fk = self._flight_key(instance)
                self._in_flight[fk] = self._in_flight.get(fk, 0) + 1
            return _decision(instance, sticky_hit, reason, key=key)

        by_port = {inst["port"]: inst for inst in instances}
        by_backend_id = {self._backend_id(inst): inst for inst in instances}
        existing = self._sessions.get(affinity_key)

        if existing is not None:
            # Uma sessão persistida não pode atravessar famílias/provedores.
            # Isso evita reutilizar estado criado pelo bug antigo que misturava
            # Codex e Antigravity no mesmo sidecar (ou uma sessão de plataforma
            # quando o principal atual é local).
            stored_inst = None
            if existing.backend_id:
                stored_inst = by_backend_id.get(existing.backend_id)
            if stored_inst is None:
                stored_inst = by_port.get(existing.backend_port)
            session_type = self._backend_type(stored_inst) if stored_inst else (
                existing.backend_type or "local"
            )
            session_provider = (
                stored_inst.get("provider")
                if stored_inst is not None
                else existing.provider
            )
            primary_type = self._backend_type(primary)
            incompatible_session = (
                (session_type == "platform") != (primary_type == "platform")
                or (
                    session_type == "platform"
                    and session_provider != primary.get("provider")
                )
            )
            # Durante failover, mantenha a sessão no secundário enquanto o
            # principal estiver em cooldown. A incompatibilidade só invalida
            # sessões antigas quando o principal configurado está utilizável.
            if (
                incompatible_session
                and self._backend_available(primary)
                and needed_ctx <= _context_limit(primary)
            ):
                logger.warning(
                    "[proxy] dropping incompatible sticky session "
                    "affinity_key=%s session=%s/%s primary=%s/%s",
                    affinity_key,
                    session_type,
                    session_provider,
                    primary_type,
                    primary.get("provider"),
                )
                if apply_side_effects:
                    self._sessions.pop(affinity_key, None)
                    self._save_sessions()
                existing = None

        if existing is not None:
            on_primary = (
                existing.backend_id == primary_backend_id
                if existing.backend_id else existing.backend_port == primary_port
            )
            if not on_primary and self._backend_available(primary):
                # Principal liberou: sessao sticky volta pra ele em vez de
                # continuar presa ao secundario onde foi parar (PRD F6 —
                # principal tem prioridade sempre que estiver livre).
                primary_ctx_ok = needed_ctx <= _context_limit(primary)
                if primary_ctx_ok:
                    _, primary_max = self._backend_flags(config, primary)
                    if ignore_capacity or self.in_flight_for(primary) < primary_max:
                        logger.info(
                            "[proxy] sticky session returning to primary "
                            "affinity_key=%s old_backend=%s",
                            affinity_key, existing.backend_port,
                        )
                        existing.backend_port = primary["port"]
                        existing.backend_model_path = primary.get("model_path") or ""
                        existing.internal_model = self._internal_model(
                            primary, existing.external_model, True
                        )
                        existing.backend_id = primary_backend_id
                        existing.backend_type = self._backend_type(primary)
                        existing.provider = primary.get("provider")
                        if apply_side_effects:
                            self._save_sessions()
                        return _commit(
                            primary, False, "sticky_return_primary", existing
                        ), None

            inst = None
            context_mismatch = False
            if existing.backend_id:
                inst = by_backend_id.get(existing.backend_id)
            if inst is None:
                inst = by_port.get(existing.backend_port)
            if inst is None:
                # Porta mudou (restart): re-vincula pelo model_path durável
                norm = normalize_model_path(existing.backend_model_path or "")
                inst = next(
                    (
                        i for i in instances
                        if normalize_model_path(i.get("model_path") or "") == norm
                    ),
                    None,
                )
                if inst is not None:
                    existing.backend_port = inst["port"]
            if inst is not None:
                existing.backend_port = inst["port"]
                existing.backend_id = self._backend_id(inst)
            if inst is not None and not self._backend_available(inst):
                inst = None
            if inst is not None and needed_ctx > _context_limit(inst):
                context_mismatch = True
                logger.info(
                    "[proxy] sticky backend context insufficient "
                    "affinity_key=%s backend=%s needed=%s limit=%s",
                    affinity_key, self._backend_id(inst), needed_ctx,
                    _context_limit(inst),
                )
                inst = None
            if inst is not None:
                _, max_parallel = self._backend_flags(config, inst)
                if ignore_capacity:
                    return _decision(inst, True, "sticky"), None
                if self.in_flight_for(inst) >= max_parallel:
                    # Afinidade fraca (hash:): requisições SIMULTÂNEAS com a
                    # mesma assinatura são fluxos paralelos (subagentes) — uma
                    # conversa não sobrepõe turnos. Ramifica para backend livre
                    # em vez de enfileirar (spec original: "muitas requisições
                    # simultâneas parecidas → pode distribuir").
                    if affinity_key.startswith("hash:"):
                        branched = self._hash_branch_locked(
                            affinity_key, by_port, instances, config,
                            primary_port, primary_backend_id, needed_ctx,
                            external_model, configured_backend_id,
                            _decision, _commit,
                        )
                        if branched is not None:
                            return branched, None
                    # Afinidade explícita permanece no mesmo backend, mas sua
                    # capacidade cresce em vez de enfileirar/retornar 503.
                    return _commit(
                        inst, True, "sticky_dynamic_capacity", existing
                    ), None
                return _commit(inst, True, "sticky", existing), None
            # Backend caiu/desabilitado: reatribui UMA vez (PRD F7)
            old_port = existing.backend_port
            candidates = self._candidates(
                instances, config, primary_port, needed_ctx,
                ignore_capacity=ignore_capacity,
                external_model=external_model,
                configured_primary_backend_id=configured_backend_id,
            )
            new_inst = self._pick_least_busy(
                candidates, primary_port, primary_backend_id
            )
            if new_inst is None:
                fallback = self._candidates(
                    instances, config, primary_port, needed_ctx,
                    ignore_capacity=True,
                    external_model=external_model,
                    configured_primary_backend_id=configured_backend_id,
                )
                if not fallback:
                    raise ProxyError(
                        503, "Nenhum backend com contexto suficiente disponivel",
                        code="no_backend",
                    )
                new_inst = self._pick_for_dynamic_growth(
                    fallback, config, primary_backend_id
                )
            reassign_reason = (
                "reassign_context_limit"
                if context_mismatch else "reassign_backend_down"
            )
            if not context_mismatch:
                logger.warning("[proxy] backend %s unavailable", old_port)
            if apply_side_effects:
                existing.backend_port = new_inst["port"]
                existing.backend_model_path = new_inst.get("model_path") or ""
                existing.internal_model = self._internal_model(
                    new_inst, existing.external_model,
                    _is_primary_backend(new_inst),
                )
                existing.backend_id = self._backend_id(new_inst)
                existing.backend_type = self._backend_type(new_inst)
                existing.provider = new_inst.get("provider")
                self._save_sessions()
                logger.warning(
                    "[proxy] reassigned affinity_key=%s old_backend=%s "
                    "new_backend=%s reason=%s",
                    affinity_key, old_port, new_inst["port"],
                    "context_limit" if context_mismatch else "backend_down",
                )
            return _commit(new_inst, False, reassign_reason, existing), None

        # ---------------- Nova sessão ----------------
        is_main = tag is None or tag == MAIN_TAG

        primary_backend_id = self._backend_id(primary)

        if is_main:
            # Conversa principal prefere o principal (PRD F6)
            if self._backend_available(primary):
                _, max_parallel = self._backend_flags(config, primary)
                primary_ctx_ok = needed_ctx <= _context_limit(primary)
                if primary_ctx_ok:
                    if ignore_capacity or self.in_flight_for(primary) < max_parallel:
                        return _commit(primary, False, "main_preference", None), None
                    # PRD F7: sessão NOVA não espera backend ocupado —
                    # transborda para um secundário elegível livre.
                    overflow = self._candidates(
                        instances, config, primary_port, needed_ctx,
                        ignore_capacity=False,
                        exclude_backend_ids={primary_backend_id},
                        external_model=external_model,
                        configured_primary_backend_id=configured_backend_id,
                    )
                    chosen = self._pick_least_busy(
                        overflow, primary_port, primary_backend_id
                    )
                    if chosen is not None:
                        return _commit(
                            chosen, False, "primary_busy_overflow", None
                        ), None
                    expandable = self._candidates(
                        instances, config, primary_port, needed_ctx,
                        ignore_capacity=True,
                        external_model=external_model,
                        configured_primary_backend_id=configured_backend_id,
                    )
                    chosen = self._pick_for_dynamic_growth(
                        expandable, config, primary_backend_id
                    )
                    if chosen is not None:
                        return _commit(
                            chosen, False, "dynamic_capacity", None
                        ), None
                    raise ProxyError(
                        503, "Nenhum backend com contexto suficiente disponivel",
                        code="no_backend",
                    )
            # Principal desabilitado ou sem contexto: least-busy nos demais
            candidates = self._candidates(
                instances, config, primary_port, needed_ctx,
                ignore_capacity=ignore_capacity,
                exclude_backend_ids={primary_backend_id},
                external_model=external_model,
                configured_primary_backend_id=configured_backend_id,
            )
            chosen = self._pick_least_busy(
                candidates, primary_port, primary_backend_id
            )
            if chosen is None:
                raise ProxyError(
                    503, "Nenhum backend disponivel para a conversa principal",
                    code="no_backend",
                )
            return _commit(chosen, False, "least_busy", None), None

        # Subagente: tenta o modelo principal primeiro (prioridade absoluta);
        # só se o primary estiver ocupado, escolhe o menos ocupado entre
        # secundários (PRD F6 — main-first para conversas principais e subagentes).
        if self._backend_available(primary):
            _, max_parallel = self._backend_flags(config, primary)
            primary_ctx_ok = needed_ctx <= _context_limit(primary)
            if primary_ctx_ok:
                if ignore_capacity or self.in_flight_for(primary) < max_parallel:
                    return _commit(primary, False, "subagent_main_preference", None), None

        # Primary ocupado ou desabilitado — least-busy entre secundários
        candidates = self._candidates(
            instances, config, primary_port, needed_ctx,
            ignore_capacity=ignore_capacity,
            exclude_backend_ids={primary_backend_id},
            external_model=external_model,
            configured_primary_backend_id=configured_backend_id,
        )
        chosen = self._pick_least_busy(
            candidates, primary_port, primary_backend_id
        )
        if chosen is None:
            any_eligible = self._candidates(
                instances, config, primary_port, needed_ctx,
                ignore_capacity=True,
                external_model=external_model,
                configured_primary_backend_id=configured_backend_id,
            )
            if not any_eligible:
                raise ProxyError(
                    503, "Nenhum backend com contexto suficiente disponivel",
                    code="no_backend",
                )
            chosen = self._pick_for_dynamic_growth(
                any_eligible, config, primary_backend_id
            )
            if chosen is None:
                raise ProxyError(
                    503, "Nenhum backend com contexto suficiente disponivel",
                    code="no_backend",
                )
            return _commit(chosen, False, "dynamic_capacity", None), None
        return _commit(chosen, False, "subagent_least_busy", None), None

    # ------------------------------------------------------------------
    # Reassign administrativo
    # ------------------------------------------------------------------

    async def reassign(
        self, affinity_key: str, *, exclude_current: bool = False,
        exclude_backend_ids: Optional[set] = None,
        reason: str = "reassign_admin",
    ) -> Optional[RouteDecision]:
        """Força a sessão a migrar para o melhor backend disponível.

        `exclude_current=True` / `exclude_backend_ids` evita reescolher backends
        que acabaram de falhar (conexão ou HTTP 429/502/503/504). O botão
        administrativo usa o padrao: o principal tem prioridade mesmo que a
        sessão ja esteja nele.
        """
        async with self._lock:
            session = self._sessions.get(affinity_key)
            if session is None:
                return None
            settings = self._settings()
            instances = self._routing_instances()
            primary = self._find_primary(instances, settings)
            config = self._config.get_config()
            primary_backend_id = (
                self._backend_id(primary)
                if primary is not None
                else normalize_backend_id(settings.get("primary_backend_id"))
            )
            if not primary_backend_id and settings.get("primary_model_path"):
                primary_backend_id = (
                    f"local:{normalize_model_path(settings['primary_model_path'])}"
                )
            excluded: set = set(exclude_backend_ids or ())
            if exclude_current and session.backend_id and len(instances) > 1:
                excluded.add(session.backend_id)

            chosen = None
            candidates = self._candidates(
                instances, config, primary["port"] if primary else None, 0,
                ignore_capacity=True,
                exclude_backend_ids=excluded or None,
            )
            # Em failover, esgote primeiro as contas do mesmo provedor da
            # sessao. So atravesse para outra plataforma quando nenhuma conta
            # daquele provedor continuar elegivel.
            preferred_provider = session.provider
            preferred_candidates = [
                inst for inst in candidates
                if preferred_provider
                and inst.get("provider") == preferred_provider
            ]
            if preferred_candidates:
                chosen = self._pick_for_dynamic_growth(
                    preferred_candidates, config, primary_backend_id
                )

            if chosen is None and (
                primary is not None
                and self._backend_available(primary)
                and self._backend_id(primary) not in excluded
            ):
                _, primary_max = self._backend_flags(config, primary)
                if self.in_flight_for(primary) < primary_max:
                    chosen = primary

            if chosen is None:
                # Plataformas primeiro: com o principal indisponível, a
                # próxima integração de plataforma assume; backends locais
                # só entram quando nenhuma plataforma está disponível.
                platform_candidates = [
                    inst for inst in candidates
                    if self._backend_type(inst) == "platform"
                ]
                chosen = self._pick_for_dynamic_growth(
                    platform_candidates or candidates,
                    config, primary_backend_id,
                )
            if chosen is None:
                raise ProxyError(503, "Nenhum backend disponivel para reassign",
                                 code="no_backend")
            old_port = session.backend_port
            chosen_is_primary = (
                primary is not None
                and self._backend_id(chosen) == self._backend_id(primary)
            )
            session.backend_port = chosen["port"]
            session.backend_model_path = chosen.get("model_path") or ""
            session.internal_model = self._internal_model(
                chosen, session.external_model, chosen_is_primary
            )
            session.backend_id = self._backend_id(chosen)
            session.backend_type = self._backend_type(chosen)
            session.provider = chosen.get("provider")
            session.last_used_at = _iso(self._now())
            self._save_sessions()
            logger.warning(
                "[proxy] reassigned affinity_key=%s old_backend=%s "
                "new_backend=%s reason=%s",
                affinity_key, old_port, chosen["port"], reason,
            )
            return RouteDecision(
                backend_port=chosen["port"],
                internal_model=self._internal_model(
                    chosen, session.external_model, chosen_is_primary
                ),
                external_model=session.external_model,
                affinity_key=affinity_key,
                detected_tag=session.detected_tag,
                sticky_hit=False,
                reason=reason,
                rewrite=(
                    not chosen_is_primary
                    and self._backend_type(chosen) != "platform"
                ),
                gpu=gpu_label(chosen),
                backend_id=self._backend_id(chosen),
                backend_type=self._backend_type(chosen),
                provider=chosen.get("provider"),
            )
