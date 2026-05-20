#!/usr/bin/env python3
"""One-time script to extract modules from llama_manager.py."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "llama_manager.py"
lines = SRC.read_text(encoding="utf-8").splitlines(keepends=True)


def slice_lines(start: int, end: int) -> str:
    return "".join(lines[start - 1 : end])


def write(name: str, content: str) -> None:
    path = ROOT / name
    path.write_text(content, encoding="utf-8")
    print(f"Wrote {name} ({len(content.splitlines())} lines)")


# schemas.py - pydantic models
schemas = '''"""Shared request/response schemas."""
from typing import List, Optional
from pydantic import BaseModel
DEFAULT_CONTEXT_SIZE = 65536


class GPUWeight(BaseModel):
    index: int
    weight: float
    name: str
    active: bool = True
    is_main: bool = False


class StartRequest(BaseModel):
    path: str
    mmproj_path: Optional[str] = None
    gpu_weights: List[GPUWeight]
    context_size: int = DEFAULT_CONTEXT_SIZE
    split_mode: str = "layer"


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


class LoginRequest(BaseModel):
    username: str
    password: str
'''
write("schemas.py", schemas)

# gpu_manager.py
gpu_header = '''"""GPU detection and tensor split management."""
import os
import re
import logging
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import psutil

from schemas import GPUWeight

LLAMA_SERVER_BIN = "llama-server"
logger = logging.getLogger("automanager")


@dataclass
class GPUInfo:
    index: int
    name: str
    vram: int


'''
gpu_body = slice_lines(279, 366)
gpu_footer = '''

class GPUManager(GPUDetector):
    """GPU operations including strict tensor split enforcement."""

    def compute_tensor_split(self, gpu_weights: List[GPUWeight]) -> List[str]:
        active = [w for w in gpu_weights if w.active and w.weight > 0]
        if not active:
            return []
        total = sum(w.weight for w in active) or 1.0
        return [f"{w.weight / total:.4f}" for w in active]

    def get_visible_devices(self, gpu_weights: List[GPUWeight]) -> Optional[str]:
        active = [w for w in gpu_weights if w.active and w.weight > 0]
        if not active:
            return None
        return ",".join(str(w.index) for w in active)

    def validate_gpu_weights(self, gpu_weights: List[GPUWeight]) -> Tuple[bool, str]:
        active = [w for w in gpu_weights if w.active and w.weight > 0]
        if not active:
            return False, "No active GPUs selected. Enable at least one GPU with weight > 0."
        return True, ""
'''
write("gpu_manager.py", gpu_header + gpu_body + gpu_footer)

print("Done partial extract")
