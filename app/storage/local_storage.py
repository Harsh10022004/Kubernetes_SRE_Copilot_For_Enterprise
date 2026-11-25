"""Local filesystem storage backend implementation."""

from __future__ import annotations

from pathlib import Path

from app.storage.storage_backend import StorageBackend


class LocalStorage(StorageBackend):
    """Stores bytes on the local filesystem under root_dir/key."""

    def __init__(self, root_dir: str | Path = ".cache/storage"):
        self.root = Path(root_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        clean_key = key.lstrip("/\\")
        target = (self.root / clean_key).resolve()
        if not target.is_relative_to(self.root):
            raise ValueError(f"Invalid storage key: {key}")
        return target

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def save_bytes(self, key: str, data: bytes) -> None:
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def read_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def url_for(self, key: str) -> str:
        return self._path(key).as_uri()
