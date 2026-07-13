"""Model discovery, rename, delete, and downloads."""

import os
import hashlib
import re
import shutil
import time
import uuid
import urllib.parse
import logging
import threading
from ipaddress import ip_address, ip_network
from typing import Dict, List, Optional

import requests
from fastapi import HTTPException

from config_manager import ConfigManager, lookup_model_config, normalize_model_path
from paths import MODELS_DIR
from process_manager import ProcessManager
logger = logging.getLogger("automanager")
_GB = 1024**3

# SSRF prevention — block private/reserved IP ranges
_PRIVATE_NETWORKS = [
    ip_network("10.0.0.0/8"),
    ip_network("172.16.0.0/12"),
    ip_network("192.168.0.0/16"),
    ip_network("127.0.0.0/8"),
    ip_network("169.254.0.0/16"),
    ip_network("0.0.0.0/8"),
    ip_network("::1"),
    ip_network("fc00::/7"),
    ip_network("fe80::/10"),
]

# Max download size: 100 GB
MAX_DOWNLOAD_SIZE = 100 * 1024 * 1024 * 1024

_FAMILY_VARIANT_WORDS = frozenset({
    "instruct", "chat", "base", "mtp", "ud", "preview", "thinking",
    "it", "gguf", "ggml", "abliterated", "uncensored", "v0", "v1", "v2",
})
_SIZE_TOKEN_RE = re.compile(r"^\d+(\.\d+)?[bB]$")
_QUANT_TOKEN_RE = re.compile(
    r"^(Q\d|IQ\d|BF16|F16|F32|FP16|FP32|UD|quant|quantized)",
    re.IGNORECASE,
)


class DownloadCancelled(Exception):
    """Raised when a download is cancelled by the user."""


def infer_model_family(name_or_filename: str, url: str = "") -> str:
    """Infer model family from filename or HuggingFace URL (e.g. Qwen3.6, Llama-3.3)."""
    stem = os.path.basename(name_or_filename or "").replace(".gguf", "").replace(
        ".mmproj", ""
    )
    if not stem and url:
        stem = _stem_from_hf_repo(url) or ""

    family_parts = _family_parts_from_stem(stem)
    if family_parts:
        joined = "-".join(family_parts)
        if family_parts[0].lower() == "meta" and len(family_parts) > 1:
            joined = "-".join(family_parts[1:])
        if joined.lower() not in {"file", "model", "download", "misc"}:
            return joined

    repo_family = _family_from_hf_repo(url)
    if repo_family:
        return repo_family
    if family_parts:
        return "-".join(family_parts)
    return stem or "misc"


def _family_parts_from_stem(stem: str) -> List[str]:
    family: List[str] = []
    for part in stem.split("-"):
        if _SIZE_TOKEN_RE.match(part):
            break
        if _QUANT_TOKEN_RE.match(part):
            break
        if part.lower() in _FAMILY_VARIANT_WORDS and family:
            break
        family.append(part)
    return family


def _stem_from_hf_repo(url: str) -> str:
    match = re.search(
        r"huggingface\.co/[^/]+/([^/]+)/",
        url or "",
        re.IGNORECASE,
    )
    if not match:
        return ""
    repo = match.group(1)
    for suffix in ("-GGUF", "-gguf", "_GGUF", "_gguf"):
        if repo.endswith(suffix):
            repo = repo[: -len(suffix)]
    return repo


def _family_from_hf_repo(url: str) -> str:
    stem = _stem_from_hf_repo(url)
    if not stem:
        return ""
    parts = _family_parts_from_stem(stem)
    if parts and parts[0].lower() == "meta" and len(parts) > 1:
        return "-".join(parts[1:])
    return "-".join(parts) if parts else ""


def _validate_download_url(url: str) -> bool:
    """Prevent SSRF by blocking URLs to private/reserved IP ranges."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    host = parsed.hostname
    if not host:
        return False
    # Block localhost-like names
    if host.lower() in ("localhost", "localhost.localdomain", "metadata.google.internal", "instance-data"):
        return False
    # Block IPs in private/reserved ranges
    try:
        addr = ip_address(host)
        for net in _PRIVATE_NETWORKS:
            if addr in net:
                return False
    except ValueError:
        pass  # Not an IP — could be DNS; allow but log
    return True


def _directory_size_bytes(path: str) -> int:
    total = 0
    if not os.path.isdir(path):
        return 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _is_projector_filename(name_lower: str) -> bool:
    return any(
        marker in name_lower
        for marker in ("mmproj", "clip", "vision", "projector")
    )


def _projector_paths_for_model(model_path: str, projectors: List[dict]) -> List[str]:
    """Return projector paths in the same directory as the language model."""
    model_dir = os.path.dirname(model_path)
    return sorted(
        proj["path"]
        for proj in projectors
        if os.path.dirname(proj["path"]) == model_dir
    )


def get_repository_storage(models_dir: str = MODELS_DIR) -> dict:
    """Used GB in the models tree and total GB on the hosting filesystem."""
    used_bytes = _directory_size_bytes(models_dir)
    total_bytes = 0
    try:
        stat_path = models_dir if os.path.exists(models_dir) else os.path.dirname(
            models_dir.rstrip("/\\")
        )
        if stat_path:
            total_bytes = shutil.disk_usage(stat_path).total
    except OSError as exc:
        logger.warning("Storage stats unavailable for %s: %s", models_dir, exc)
    return {
        "path": models_dir,
        "used_gb": round(used_bytes / _GB, 1),
        "total_gb": round(total_bytes / _GB, 1),
    }


class ModelScanner:
    """Scans models directory for .gguf and .mmproj files."""

    def __init__(
        self,
        config_manager: ConfigManager,
        process_manager: ProcessManager,
        models_dir: str = MODELS_DIR,
    ):
        self.config = config_manager
        self.process_manager = process_manager
        self.models_dir = models_dir

    def scan(self) -> dict:
        models = []
        projectors = []
        try:
            for root, _dirs, files in os.walk(self.models_dir):
                for f in files:
                    full_path = os.path.join(root, f)
                    name_lower = f.lower()
                    item = {
                        # Stable, unique, DOM-safe id derived from the path so the
                        # UI can key tabs/elements per model. Without it every item
                        # rendered as id "undefined" and clicking any model just
                        # reopened the first model's tab.
                        "id": hashlib.md5(
                            full_path.replace("\\", "/").encode("utf-8")
                        ).hexdigest()[:12],
                        "path": full_path,
                        "name": f,
                        "dir": os.path.relpath(root, self.models_dir) or "/",
                    }
                    if _is_projector_filename(name_lower):
                        projectors.append(item)
                    elif name_lower.endswith(".gguf"):
                        models.append(item)
        except OSError as e:
            logger.error(f"Scan error: {e}")

        config = self.config.load()
        model_configs = config.get("model_configs", {})
        for m in models:
            m["last_config"] = lookup_model_config(model_configs, m["path"]) or None
        for p in projectors:
            p["last_config"] = lookup_model_config(model_configs, p["path"]) or None

        for m in models:
            candidates = _projector_paths_for_model(m["path"], projectors)
            m["mmproj_candidates"] = candidates
            m["auto_mmproj"] = candidates[0] if candidates else None

        return {
            "models": models,
            "projectors": projectors,
            "storage": get_repository_storage(self.models_dir),
        }

    def rename_model(self, old_path: str, new_name: str) -> str:
        real_models_dir = os.path.realpath(self.models_dir)
        real_old_path = os.path.realpath(old_path)
        if not real_old_path.startswith(real_models_dir + os.sep):
            raise HTTPException(status_code=403, detail="Acesso negado")
        if not os.path.exists(real_old_path):
            raise HTTPException(
                status_code=404, detail="Arquivo nao encontrado"
            )

        pm = self.process_manager
        normalized_old = normalize_model_path(real_old_path)
        status = pm.get_status()
        for inst in status.get("instances", []):
            if inst.get("status") != "running":
                continue
            if normalize_model_path(inst.get("model_path") or "") == normalized_old:
                raise HTTPException(
                    status_code=400,
                    detail="Impossivel renomear modelo em execucao",
                )

        dir_name = os.path.dirname(real_old_path)
        if not new_name.endswith(".gguf"):
            new_name += ".gguf"
        new_path = os.path.join(dir_name, new_name)
        new_path_norm = normalize_model_path(new_path)

        if os.path.exists(new_path):
            raise HTTPException(
                status_code=400, detail="Ja existe um arquivo com este nome"
            )

        os.rename(real_old_path, new_path)
        data = self.config.load()
        updated = False

        if data.get("default_model") and normalize_model_path(
            data["default_model"]
        ) == normalized_old:
            data["default_model"] = new_path_norm
            updated = True

        defaults = data.get("default_models", [])
        if isinstance(defaults, list):
            new_defaults = [
                new_path_norm if normalize_model_path(p) == normalized_old else p
                for p in defaults
            ]
            if new_defaults != defaults:
                data["default_models"] = new_defaults
                updated = True

        model_configs = data.get("model_configs", {})
        keys_to_move = [
            k for k in model_configs if normalize_model_path(k) == normalized_old
        ]
        for key in keys_to_move:
            model_configs[new_path_norm] = model_configs.pop(key)
            updated = True

        if updated:
            self.config.save(data)

        logger.info(f"Renamed: {real_old_path} -> {new_path_norm}")
        return new_path_norm

    def delete_model(self, file_path: str) -> None:
        real_models_dir = os.path.realpath(self.models_dir)
        real_file_path = os.path.realpath(file_path)
        if not real_file_path.startswith(real_models_dir + os.sep):
            raise HTTPException(status_code=403, detail="Acesso negado")
        if not os.path.exists(real_file_path):
            raise HTTPException(
                status_code=404, detail="Arquivo nao encontrado"
            )

        pm = self.process_manager
        normalized = normalize_model_path(real_file_path)
        status = pm.get_status()
        for inst in status.get("instances", []):
            if inst.get("status") != "running":
                continue
            inst_path = normalize_model_path(inst.get("model_path") or "")
            if inst_path == normalized:
                pm.stop(inst.get("port"))

        os.remove(real_file_path)

        data = self.config.load()
        updated = False
        if data.get("default_model") and normalize_model_path(
            data["default_model"]
        ) == normalized:
            data["default_model"] = None
            updated = True

        defaults = data.get("default_models", [])
        new_defaults = [
            p for p in defaults if normalize_model_path(p) != normalized
        ]
        if new_defaults != defaults:
            data["default_models"] = new_defaults
            updated = True

        model_configs = data.get("model_configs", {})
        keys_to_remove = [
            k for k in model_configs if normalize_model_path(k) == normalized
        ]
        for key in keys_to_remove:
            del model_configs[key]
            updated = True

        if updated:
            self.config.save(data)

        logger.info(f"Deleted: {real_file_path}")


class DownloadManager:
    """Manages model downloads with progress tracking."""

    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self._downloads: Dict[str, dict] = {}
        self._downloads_queue: List[tuple] = []
        self._lock = threading.Lock()

    def start_download(
        self,
        url: str,
        filename: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> str:
        # SSRF prevention
        if not _validate_download_url(url):
            raise HTTPException(
                status_code=400,
                detail="URL de download bloqueada por seguranca (SSRF prevention)"
            )
        download_id = str(uuid.uuid4())
        if model_path:
            normalized_root = os.path.normpath(self.models_dir)
            normalized_model = os.path.normpath(model_path)
            if not normalized_model.startswith(normalized_root):
                raise HTTPException(status_code=403, detail="Acesso negado")
            if not os.path.isfile(normalized_model):
                raise HTTPException(
                    status_code=404, detail="Modelo nao encontrado"
                )
            if not filename:
                filename = url.split("/")[-1].split("?")[0]
                if not _is_projector_filename(filename.lower()):
                    if filename.endswith(".gguf"):
                        filename = filename.replace(".gguf", ".mmproj")
                    else:
                        filename += ".mmproj"
            model_specific_dir = os.path.dirname(normalized_model)
            path = os.path.join(model_specific_dir, filename)
            if os.path.exists(path):
                base, ext = os.path.splitext(filename)
                filename = f"{base}_{int(time.time())}{ext}"
                path = os.path.join(model_specific_dir, filename)
        else:
            if not filename:
                filename = url.split("/")[-1].split("?")[0]
                if not filename.endswith(".gguf"):
                    filename += ".gguf"

            family = infer_model_family(filename, url)
            family_dir = os.path.join(self.models_dir, family)
            os.makedirs(family_dir, exist_ok=True)
            path = os.path.join(family_dir, filename)

            if os.path.exists(path):
                base, ext = os.path.splitext(filename)
                filename = f"{base}_{int(time.time())}{ext}"
                path = os.path.join(family_dir, filename)
            model_specific_dir = family_dir

        with self._lock:
            self._downloads[download_id] = {
                "filename": filename,
                "path": path,
                "url": url,
                "model_path": normalized_model.replace("\\", "/") if model_path else None,
                "family": infer_model_family(filename, url) if not model_path else None,
                "status": "downloading",
                "progress": 0,
                "downloaded_bytes": 0,
                "total_bytes": 0,
                "speed_bps": 0,
                "start_time": time.time(),
                "elapsed_seconds": 0,
                "eta_seconds": None,
                "cancel_requested": False,
            }
            self._downloads_queue.append(
                (download_id, url, filename, path)
            )
        return download_id

    def get_progress(self) -> dict:
        with self._lock:
            snapshot = {}
            now = time.time()
            for did, entry in self._downloads.items():
                item = dict(entry)
                if item.get("status") == "downloading":
                    started = item.get("start_time") or now
                    item["elapsed_seconds"] = max(0.0, now - started)
                snapshot[did] = item
            return snapshot

    def cancel_download(self, download_id: str) -> bool:
        with self._lock:
            entry = self._downloads.get(download_id)
            if not entry:
                return False
            if entry.get("status") != "downloading":
                return False
            if entry.get("cancel_requested"):
                return True
            was_queued = any(item[0] == download_id for item in self._downloads_queue)
            entry["cancel_requested"] = True
            entry["status"] = "cancelling"
            self._downloads_queue = [
                item for item in self._downloads_queue if item[0] != download_id
            ]
            path = entry.get("path")
            started = entry.get("start_time") or time.time()

        if was_queued:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError as exc:
                    logger.warning(
                        "Could not remove cancelled download %s: %s", path, exc
                    )
            with self._lock:
                entry = self._downloads.get(download_id)
                if entry:
                    entry["status"] = "cancelled"
                    entry["speed_bps"] = 0
                    entry["progress"] = 0
                    entry["eta_seconds"] = None
                    entry["elapsed_seconds"] = time.time() - started
            return True
        return True

    def clear_completed(self) -> int:
        """Remove finished downloads from memory. Returns count cleared."""
        with self._lock:
            to_remove = [
                did
                for did, d in self._downloads.items()
                if d.get("status") in ("completed", "cancelled", "failed")
            ]
            for did in to_remove:
                del self._downloads[did]
            return len(to_remove)

    def _do_download(
        self, download_id: str, url: str, filename: str, path: str
    ) -> None:
        try:
            with self._lock:
                if self._downloads.get(download_id, {}).get("cancel_requested"):
                    raise DownloadCancelled()

            # Segue redirects manualmente revalidando cada salto: requests segue
            # redirects por padrão, o que permitiria burlar _validate_download_url
            # apontando para um host interno (ex.: 169.254.169.254) via 302.
            current_url = url
            response = None
            for _ in range(6):
                if not _validate_download_url(current_url):
                    raise Exception("URL de download bloqueada (destino não permitido).")
                response = requests.get(
                    current_url, stream=True, timeout=300, allow_redirects=False
                )
                if response.is_redirect or response.status_code in (301, 302, 303, 307, 308):
                    next_url = response.headers.get("location")
                    response.close()
                    if not next_url:
                        raise Exception("Redirecionamento de download inválido.")
                    current_url = urllib.parse.urljoin(current_url, next_url)
                    continue
                break
            else:
                raise Exception("Excesso de redirecionamentos no download.")
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            if total_size > MAX_DOWNLOAD_SIZE:
                raise Exception(
                    f"Download muito grande ({total_size / _GB:.1f} GB). "
                    f"Maximo: {MAX_DOWNLOAD_SIZE / _GB:.0f} GB"
                )
            downloaded = 0
            start_time = time.time()
            with open(path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192 * 4):
                    with self._lock:
                        if self._downloads.get(download_id, {}).get("cancel_requested"):
                            raise DownloadCancelled()
                    if chunk:
                        downloaded += len(chunk)
                        if downloaded > MAX_DOWNLOAD_SIZE:
                            raise Exception(
                                "Download excede tamanho maximo permitido (100GB)"
                            )
                        f.write(chunk)
                        now = time.time()
                        elapsed = now - start_time
                        speed_bps = downloaded / elapsed if elapsed > 0 else 0
                        remaining = max(0, total_size - downloaded) if total_size else 0
                        eta_seconds = (
                            remaining / speed_bps
                            if speed_bps > 0 and total_size > 0
                            else None
                        )
                        with self._lock:
                            if download_id in self._downloads:
                                entry = self._downloads[download_id]
                                entry["downloaded_bytes"] = downloaded
                                entry["total_bytes"] = total_size
                                entry["speed_bps"] = speed_bps
                                entry["elapsed_seconds"] = elapsed
                                entry["eta_seconds"] = eta_seconds
                                if total_size > 0:
                                    entry["progress"] = round(
                                        (downloaded / total_size) * 100, 2
                                    )
            with self._lock:
                if download_id in self._downloads:
                    entry = self._downloads[download_id]
                    entry["status"] = "completed"
                    entry["progress"] = 100
                    entry["speed_bps"] = 0
                    entry["eta_seconds"] = 0
                    entry["elapsed_seconds"] = time.time() - start_time
            logger.info(f"Download completed: {filename}")
        except DownloadCancelled:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
            with self._lock:
                if download_id in self._downloads:
                    entry = self._downloads[download_id]
                    entry["status"] = "cancelled"
                    entry["speed_bps"] = 0
                    entry["progress"] = 0
                    entry["eta_seconds"] = None
                    if entry.get("start_time"):
                        entry["elapsed_seconds"] = time.time() - entry["start_time"]
            logger.info(f"Download cancelled: {filename}")
        except Exception as e:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
            logger.error(f"Download error {download_id}: {e}")
            with self._lock:
                if download_id in self._downloads:
                    entry = self._downloads[download_id]
                    entry["status"] = "failed"
                    entry["error"] = str(e)
                    entry["speed_bps"] = 0
                    entry["eta_seconds"] = None
                    if entry.get("start_time"):
                        entry["elapsed_seconds"] = time.time() - entry["start_time"]
