"""Cache directory utilities for AlbumentationsX."""

from __future__ import annotations

import os
from pathlib import Path


def get_cache_dir() -> Path:
    """Get platform-appropriate cache directory.

    Returns:
        Path to the cache directory for AlbumentationsX.

    """
    # Check for environment variable override
    if cache_dir := os.environ.get("ALBUMENTATIONS_CACHE_DIR"):
        return Path(cache_dir)

    # Use platform-specific directories
    if os.name == "nt":  # Windows
        # Use %LOCALAPPDATA% on Windows (e.g., C:\Users\username\AppData\Local)
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "AlbumentationsX" / "Cache"
    # Unix-like
    # Follow XDG Base Directory spec
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "albumentationsx"
