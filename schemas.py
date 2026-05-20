"""Shared request/response schemas."""
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
