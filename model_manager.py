"""Model discovery, rename, delete, and downloads."""

import os
import time
import uuid
import logging
import threading
from typing import Dict, List, Optional

import requests
from fastapi import HTTPException

from config_manager import ConfigManager
from process_manager import ProcessManager

MODELS_DIR = "/media/docker/models"
logger = logging.getLogger("automanager")


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
                        "path": full_path,
                        "name": f,
                        "dir": os.path.relpath(root, self.models_dir) or "/",
                    }
                    if any(
                        x in name_lower
                        for x in ["mmproj", "clip", "vision", "projector"]
                    ):
                        projectors.append(item)
                    else:
                        models.append(item)
        except OSError as e:
            logger.error(f"Scan error: {e}")

        config = self.config.load()
        model_configs = config.get("model_configs", {})
        for m in models:
            m["last_config"] = model_configs.get(m["path"])
        for p in projectors:
            p["last_config"] = model_configs.get(p["path"])

        for m in models:
            base_name = os.path.splitext(m["name"])[0]
            candidates = []
            for proj in projectors:
                proj_base = os.path.splitext(proj["name"])[0]
                if proj_base == base_name or base_name in proj_base:
                    candidates.append(proj["path"])
            m["mmproj_candidates"] = candidates
            m["auto_mmproj"] = candidates[0] if candidates else None

        return {"models": models, "projectors": projectors}

    def rename_model(self, old_path: str, new_name: str) -> str:
        if not old_path.startswith(self.models_dir):
            raise HTTPException(status_code=403, detail="Acesso negado")
        if not os.path.exists(old_path):
            raise HTTPException(
                status_code=404, detail="Arquivo nao encontrado"
            )

        pm = self.process_manager
        status = pm.get_status()
        if status["running"]:
            normalized_old = old_path.replace("\\", "/")
            normalized_run = status.get("model_path", "").replace("\\", "/")
            if normalized_old == normalized_run:
                raise HTTPException(
                    status_code=400,
                    detail="Impossivel renomear modelo em execucao",
                )

        dir_name = os.path.dirname(old_path)
        if not new_name.endswith(".gguf"):
            new_name += ".gguf"
        new_path = os.path.join(dir_name, new_name)

        if os.path.exists(new_path):
            raise HTTPException(
                status_code=400, detail="Ja existe um arquivo com este nome"
            )

        os.rename(old_path, new_path)
        data = self.config.load()
        updated = False

        if data.get("default_model") == old_path:
            data["default_model"] = new_path
            updated = True

        if "model_configs" in data and old_path in data["model_configs"]:
            data["model_configs"][new_path] = data["model_configs"].pop(
                old_path
            )
            updated = True

        if updated:
            self.config.save(data)

        logger.info(f"Renamed: {old_path} -> {new_path}")
        return new_path

    def delete_model(self, file_path: str) -> None:
        if not file_path.startswith(self.models_dir):
            raise HTTPException(status_code=403, detail="Acesso negado")
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404, detail="Arquivo nao encontrado"
            )

        pm = self.process_manager
        status = pm.get_status()
        if status["running"]:
            normalized = file_path.replace("\\", "/")
            normalized_run = status.get("model_path", "").replace("\\", "/")
            if normalized == normalized_run:
                pm.stop()

        os.remove(file_path)
        logger.info(f"Deleted: {file_path}")


class DownloadManager:
    """Manages model downloads with progress tracking."""

    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self._downloads: Dict[str, dict] = {}
        self._downloads_queue: List[tuple] = []
        self._lock = threading.Lock()

    def start_download(self, url: str, filename: Optional[str] = None) -> str:
        download_id = str(uuid.uuid4())
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
            downloaded = 0
            start_time = time.time()
            with open(path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192 * 4):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
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
