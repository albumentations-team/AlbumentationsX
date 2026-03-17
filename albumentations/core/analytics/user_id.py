"""User ID management for telemetry.

This module provides functionality to manage persistent anonymous user IDs
for telemetry purposes, similar to how iterative-telemetry handles it.
"""

import contextlib
import json
import os
import tempfile
import uuid
from pathlib import Path

DO_NOT_TRACK_VALUE = "do-not-track"


def get_user_config_dir() -> Path:
    """Return the base config directory: XDG_CONFIG_HOME on Unix, APPDATA on Windows.
    Overridable with ALBUMENTATIONS_CONFIG_DIR.
    """
    # Check for environment variable override
    if config_dir := os.environ.get("ALBUMENTATIONS_CONFIG_DIR"):
        return Path(config_dir)

    # Use platform-specific directories
    if os.name == "nt":  # Windows
        # Use %APPDATA% on Windows (e.g., C:\Users\username\AppData\Roaming)
        return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    # Unix-like
    # Follow XDG Base Directory spec
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))


class UserIDManager:
    """Stores and retrieves a persistent anonymous ID in a JSON file under the user config dir.
    Atomic writes (temp file + rename) to avoid races.
    """

    def __init__(self, app_name: str = "albumentationsx"):
        """Create manager for a given app name; config stored in <config_dir>/<app_name>/user_id.json.
        Default app_name: albumentationsx.

        Args:
            app_name (str): Application name for config directory

        """
        self.app_name = app_name
        self.config_dir = Path(get_user_config_dir()) / app_name
        self.user_id_file = self.config_dir / "user_id.json"
        self._cached_user_id: str | None = None
        self._cache_loaded = False

    def _read_user_id(self) -> str | None:
        """Load user ID from the JSON file. Returns None if file missing, invalid, or set to do-not-track.
        Does not create a new ID.

        Returns:
            str | None: User ID string or None if not found/invalid.

        """
        if not self.user_id_file.exists():
            return None

        try:
            with self.user_id_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
                user_id = data.get("user_id", "")

                return None if user_id.lower() == DO_NOT_TRACK_VALUE.lower() else user_id
        except (json.JSONDecodeError, OSError):
            return None

    def _write_user_id_atomic(self, user_id: str) -> bool:
        """Write user ID to disk via temp file and atomic rename so concurrent processes do not corrupt file.
        Returns True on success.

        Args:
            user_id (str): User ID to write

        Returns:
            bool: True if write was successful, False otherwise.

        """
        # Create directory if it doesn't exist
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
        except OSError:
            return False

        # Write to temporary file first
        try:
            # Create temp file in the same directory for atomic rename
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=str(self.config_dir),
                prefix=".user_id_",
                suffix=".tmp",
                delete=False,
                encoding="utf-8",
            ) as tmp_file:
                json.dump({"user_id": user_id}, tmp_file, indent=2)
                tmp_path = Path(tmp_file.name)

            # Atomic rename (on Unix-like systems, this is atomic)
            # On Windows, it might fail if target exists, so we handle that
            try:
                tmp_path.replace(self.user_id_file)
            except OSError:
                # On Windows, try removing the target first
                try:
                    if self.user_id_file.exists():
                        self.user_id_file.unlink()
                    tmp_path.rename(self.user_id_file)
                except OSError:
                    # Clean up temp file
                    with contextlib.suppress(OSError):
                        tmp_path.unlink()
                    return False
                else:
                    return True
            else:
                return True
        except OSError:
            return False

    def get_or_create_user_id(self) -> str | None:
        """Return stored user ID, or create and persist a new UUID if none exists.
        Respects do-not-track; result cached for process.

        Returns:
            str | None: User ID string or None if user has opted out.

        """
        # Return cached value if already loaded
        if self._cache_loaded:
            return self._cached_user_id

        # Check if user has opted out by looking at the file directly
        if self.user_id_file.exists():
            try:
                with self.user_id_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("user_id", "").lower() == DO_NOT_TRACK_VALUE.lower():
                        # User has opted out
                        self._cached_user_id = None
                        self._cache_loaded = True
                        return None
            except (json.JSONDecodeError, OSError):
                pass

        # Try to read existing user ID
        user_id = self._read_user_id()

        if user_id is None and self.user_id_file.exists():
            # File exists but returned None - user has opted out
            self._cached_user_id = None
            self._cache_loaded = True
            return None

        if user_id is None:
            # File doesn't exist - generate new user ID
            new_user_id = str(uuid.uuid4())

            # Try to write it atomically
            if self._write_user_id_atomic(new_user_id):
                user_id = new_user_id
            else:
                # If write failed, try reading again
                # (another process might have created it)
                user_id = self._read_user_id()
                if user_id is None:
                    # If still None, use the generated ID without persisting
                    user_id = new_user_id

        # Cache the result
        self._cached_user_id = user_id
        self._cache_loaded = True

        return user_id

    def opt_out(self) -> None:
        """Persist 'do-not-track' and clear cache. Future get_or_create_user_id() returns None
        until user resets preference in config.
        """
        if self._write_user_id_atomic(DO_NOT_TRACK_VALUE):
            # Clear the cache
            self._cached_user_id = None
            self._cache_loaded = False

    def reset(self) -> None:
        """Delete the persisted user ID file and clear the in-memory cache. Next get_or_create_user_id()
        creates a new ID. Idempotent.
        """
        # Always clear the cache first
        self._cached_user_id = None
        self._cache_loaded = False

        # Try to delete the file
        with contextlib.suppress(OSError):
            if self.user_id_file.exists():
                self.user_id_file.unlink()


# Global instance for easy access
_user_id_manager: UserIDManager | None = None


def get_user_id_manager() -> UserIDManager:
    """Return the global UserIDManager singleton. First call creates it with default app name;
    later calls return the same instance.

    Returns:
        UserIDManager: The global UserIDManager instance.

    """
    global _user_id_manager  # noqa: PLW0603
    if _user_id_manager is None:
        _user_id_manager = UserIDManager()
    return _user_id_manager
