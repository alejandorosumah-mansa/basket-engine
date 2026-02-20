"""Local file caching layer for API responses with incremental updates."""

import json
import os
import time
import logging
from pathlib import Path
from typing import Optional, Any

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"


class Cache:
    """File-based cache for API responses."""

    def __init__(self, platform: str):
        self.dir = BASE_DIR / platform
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta_file = self.dir / "_cache_meta.json"
        self.meta = self._load_meta()

    def _load_meta(self) -> dict:
        if self.meta_file.exists():
            return json.loads(self.meta_file.read_text())
        return {}

    def _save_meta(self):
        self.meta_file.write_text(json.dumps(self.meta, indent=2))

    def get(self, key: str) -> Optional[Any]:
        """Get cached data by key."""
        path = self.dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())
        return None

    def put(self, key: str, data: Any):
        """Store data in cache."""
        path = self.dir / f"{key}.json"
        path.write_text(json.dumps(data))

    def get_last_fetch(self, key: str) -> Optional[float]:
        """Get unix timestamp of last fetch for a key."""
        return self.meta.get(f"last_fetch:{key}")

    def set_last_fetch(self, key: str, ts: Optional[float] = None):
        """Record when data was last fetched."""
        self.meta[f"last_fetch:{key}"] = ts or time.time()
        self._save_meta()

    def needs_update(self, key: str, max_age_hours: float = 24) -> bool:
        """Check if cached data is stale."""
        last = self.get_last_fetch(key)
        if last is None:
            return True
        return (time.time() - last) > (max_age_hours * 3600)

    def invalidate(self, key: str):
        """Remove cached data."""
        path = self.dir / f"{key}.json"
        if path.exists():
            path.unlink()
        self.meta.pop(f"last_fetch:{key}", None)
        self._save_meta()
