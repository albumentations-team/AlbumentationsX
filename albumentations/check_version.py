"""Version checking functionality for AlbumentationsX.

This module provides functionality to check for updates to the AlbumentationsX package
by comparing the current version with the latest version available on PyPI.

Features:
    - Check for updates from PyPI with caching (24-hour cache)
    - DNS-based connectivity check for better reliability
    - Environment variable controls (NO_ALBUMENTATIONS_UPDATE, ALBUMENTATIONS_OFFLINE)
    - Graceful failure handling
    - Support for pre-release versions

Environment Variables:
    NO_ALBUMENTATIONS_UPDATE: Set to "1" or "true" to disable update checks
    ALBUMENTATIONS_OFFLINE: Set to "1" or "true" to force offline mode

Usage:
    >>> from albumentations.check_version import check_for_updates
    >>> check_for_updates()
"""

from __future__ import annotations

import functools
import json
import os
import re
import socket
import urllib.error
import urllib.request
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

from albumentations import __version__ as current_version

# Constants
HTTP_TIMEOUT = 3.0
PYPI_URL = "https://pypi.org/pypi/albumentationsx/json"
CACHE_HOURS = 24

# Environment variables
ENV_NO_UPDATE = "NO_ALBUMENTATIONS_UPDATE"
ENV_OFFLINE = "ALBUMENTATIONS_OFFLINE"

# DNS servers for connectivity check
DNS_SERVERS = [("1.1.1.1", 53), ("8.8.8.8", 53)]  # Cloudflare and Google


def _try_dns_connect(server: str, port: int) -> bool:
    """Try to connect to a DNS server."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            sock.connect((server, port))
            return True
    except (socket.timeout, OSError):
        return False


@functools.lru_cache(maxsize=1)
def check_connectivity() -> bool:
    """Check internet connectivity using DNS servers first, then HTTP."""
    # Check offline mode
    if os.environ.get(ENV_OFFLINE, "").lower() in {"1", "true"}:
        return False

    # Try DNS servers first (faster)
    if any(_try_dns_connect(server, port) for server, port in DNS_SERVERS):
        return True

    # Fallback to HTTP
    try:
        opener = urllib.request.build_opener()
        opener.open("https://pypi.org/simple/", timeout=1)
    except (urllib.error.URLError, OSError):
        return False
    else:
        return True


def fetch_pypi_version() -> str | None:
    """Fetch the latest version from PyPI."""
    if not check_connectivity():
        return None

    try:
        opener = urllib.request.build_opener()
        with opener.open(PYPI_URL, timeout=HTTP_TIMEOUT) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data.get("info", {}).get("version")
    except (urllib.error.URLError, json.JSONDecodeError, KeyError, OSError):
        return None


def get_cache_dir() -> Path:
    """Get platform-appropriate cache directory."""
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


CACHE_FILE = get_cache_dir() / "version_cache.json"


def read_cache() -> str | None:
    """Read version from cache file if not expired."""
    if not CACHE_FILE.exists():
        return None

    try:
        with CACHE_FILE.open("r") as f:
            data = json.load(f)

        # Check expiration
        timestamp_str = data.get("timestamp")
        if not timestamp_str:
            return None

        # Handle different timestamp formats
        if isinstance(timestamp_str, (int, float)):
            # Legacy format - unix timestamp
            timestamp = datetime.fromtimestamp(timestamp_str, tz=timezone.utc)
        else:
            # ISO format string
            timestamp = datetime.fromisoformat(timestamp_str)

        if datetime.now(timezone.utc) - timestamp > timedelta(hours=CACHE_HOURS):
            return None

        return data.get("version")
    except (json.JSONDecodeError, KeyError, ValueError, OSError, TypeError):
        return None


def write_cache(version: str) -> None:
    """Write version to cache file."""
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with CACHE_FILE.open("w") as f:
            json.dump(
                {
                    "version": version,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                f,
            )
    except OSError:
        pass  # Fail silently


def parse_version(version: str) -> tuple[int, ...] | None:
    """Parse version string into comparable tuple.

    Examples:
        "1.4.24" -> (1, 4, 24, 0, 0)
        "1.4.0-beta.2" -> (1, 4, 0, -2, 2)

    """
    # Match semantic versioning with optional pre-release
    match = re.match(
        r"^(\d+)\.(\d+)\.(\d+)"  # Major.Minor.Patch
        r"(?:-(alpha|beta|rc)(?:\.(\d+))?)?"  # Optional pre-release with optional number
        r"(?:\+.*)?$",  # Optional metadata
        version,
    )

    if not match:
        return None

    major, minor, patch, pre_type, pre_num = match.groups()
    result = [int(major), int(minor), int(patch)]

    # Handle pre-release (negative values for proper ordering)
    if pre_type:
        pre_order = {"alpha": -3, "beta": -2, "rc": -1}
        result.extend([pre_order[pre_type], int(pre_num or 0)])
    else:
        result.extend([0, 0])  # Stable version

    return tuple(result)


def get_latest_version() -> str | None:
    """Get the latest version with caching."""
    # Check if disabled
    if os.environ.get(ENV_NO_UPDATE, "").lower() in {"1", "true"}:
        return None

    # Try cache first
    if cached := read_cache():
        return cached

    # Fetch from PyPI
    version = fetch_pypi_version()
    if version:
        write_cache(version)

    return version


def check_for_updates(verbose: bool = True) -> tuple[bool, str | None]:
    """Check for available updates to AlbumentationsX.

    Args:
        verbose: Whether to print update messages.

    Returns:
        Tuple of (update_available, latest_version)

    """
    latest = get_latest_version()
    if not latest:
        return False, None

    latest_parsed = parse_version(latest)
    current_parsed = parse_version(current_version)

    if not latest_parsed or not current_parsed:
        return False, latest

    if latest_parsed > current_parsed:
        if verbose:
            warnings.warn(
                f"A new version of AlbumentationsX ({latest}) is available! "
                f"Your version is {current_version}. "
                f"Upgrade using: pip install -U albumentationsx",
                UserWarning,
                stacklevel=2,
            )
        return True, latest

    return False, latest
