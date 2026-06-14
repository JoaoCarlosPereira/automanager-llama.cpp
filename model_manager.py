"""Model discovery, rename, delete, and downloads."""

import os
import hashlib
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

from config_manager import ConfigManager, lookup_model_config
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
    ip_network("127.0.0.0/8"),
]

# Max download size: 100 GB
MAX_DOWNLOAD_SIZE = 100 * 1024 * 1024 * 1024


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
        status = pm.get_status()
        if status["running"]:
            normalized_old = real_old_path.replace("\\", "/")
            normalized_run = status.get("model_path", "").replace("\\", "/")
            if normalized_old == normalized_run:
                raise HTTPException(
                    status_code=400,
                    detail="Impossivel renomear modelo em execucao",
                )

        dir_name = os.path.dirname(real_old_path)
        if not new_name.endswith(".gguf"):
            new_name += ".gguf"
        new_path = os.path.join(dir_name, new_name)

        if os.path.exists(new_path):
            raise HTTPException(
                status_code=400, detail="Ja existe um arquivo com este nome"
            )

        os.rename(real_old_path, new_path)
        data = self.config.load()
        updated = False

        if data.get("default_model") == real_old_path:
            data["default_model"] = new_path
            updated = True

        if "model_configs" in data and real_old_path in data["model_configs"]:
            data["model_configs"][new_path] = data["model_configs"].pop(
                real_old_path
            )
            updated = True

        if updated:
            self.config.save(data)

        logger.info(f"Renamed: {real_old_path} -> {new_path}")
        return new_path

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
        status = pm.get_status()
        if status["running"]:
            normalized = real_file_path.replace("\\", "/")
            normalized_run = status.get("model_path", "").replace("\\", "/")
            if normalized == normalized_run:
                pm.stop()

        os.remove(real_file_path)
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
        else:
            if not filename:
                filename = url.split("/")[-1].split("?")[0]
                if not filename.endswith(".gguf"):
                    filename += ".gguf"

            model_name_folder = filename.replace(".gguf", "")
            model_specific_dir = os.path.join(self.models_dir, model_name_folder)
            os.makedirs(model_specific_dir, exist_ok=True)
            path = os.path.join(model_specific_dir, filename)

        if os.path.exists(path):
            base, ext = os.path.splitext(filename)
            filename = f"{base}_{int(time.time())}{ext}"
            path = os.path.join(model_specific_dir, filename)

        with self._lock:
            self._downloads[download_id] = {
                "filename": filename,
                "path": path,
                "url": url,
                "status": "downloading",
                "progress": 0,
                "downloaded_bytes": 0,
                "total_bytes": 0,
                "speed_bps": 0,
                "start_time": time.time(),
            }
            self._downloads_queue.append(
                (download_id, url, filename, path)
            )
        return download_id

    def get_progress(self) -> dict:
        with self._lock:
            return dict(self._downloads)

    def clear_completed(self) -> int:
        """Remove completed downloads from memory. Returns count cleared."""
        with self._lock:
            to_remove = [
                did
                for did, d in self._downloads.items()
                if d.get("status") == "completed"
            ]
            for did in to_remove:
                del self._downloads[did]
            return len(to_remove)

    def _do_download(
        self, download_id: str, url: str, filename: str, path: str
    ) -> None:
        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            # Block downloads exceeding max size
            if total_size > MAX_DOWNLOAD_SIZE:
                raise Exception(f"Download muito grande ({total_size / _GB:.1f} GB). Maximo: {MAX_DOWNLOAD_SIZE / _GB:.0f} GB")
            downloaded = 0
            start_time = time.time()
            with open(path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192 * 4):
                    if chunk:
                        downloaded += len(chunk)
                        # Check max size during download
                        if downloaded > MAX_DOWNLOAD_SIZE:
                            raise Exception("Download excede tamanho maximo permitido (100GB)")
                        f.write(chunk)
                        now = time.time()
                        elapsed = now - start_time
                        speed_bps = downloaded / elapsed if elapsed > 0 else 0
                        with self._lock:
                            if download_id in self._downloads:
                                self._downloads[download_id]["downloaded_bytes"] = downloaded
                                self._downloads[download_id]["total_bytes"] = total_size
                                self._downloads[download_id]["speed_bps"] = speed_bps
                                if total_size > 0:
                                    self._downloads[download_id][
                                        "progress"
                                    ] = round(
                                        (downloaded / total_size) * 100, 2
                                    )
            with self._lock:
                if download_id in self._downloads:
                    self._downloads[download_id]["status"] = "completed"
                    self._downloads[download_id]["progress"] = 100
                    self._downloads[download_id]["speed_bps"] = 0
            logger.info(f"Download completed: {filename}")
        except Exception as e:
            logger.error(f"Download error {download_id}: {e}")
            with self._lock:
                if download_id in self._downloads:
                    self._downloads[download_id]["status"] = "failed"
                    self._downloads[download_id]["error"] = str(e)
