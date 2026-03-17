"""Settings management for AlbumentationsX."""

import json
import os
from pathlib import Path
from typing import Any

from albumentations.core.cache_utils import get_cache_dir


class SettingsManager:
    """Stores settings (e.g. telemetry on/off) in a JSON file under the cache dir.
    Env vars ALBUMENTATIONS_NO_TELEMETRY and OFFLINE override file.
    """

    def __init__(self, settings_file: Path | None = None):
        """Create manager; if no path given, use get_cache_dir()/settings.json.
        Loads from file and applies env overrides immediately.

        Args:
            settings_file (Path | None): Path to settings file. If None, uses default location.

        """
        self.settings_file = settings_file or (get_cache_dir() / "settings.json")
        self.defaults = {
            "telemetry": True,
        }
        self._settings = self._load_settings()

    def _load_settings(self) -> dict[str, Any]:
        """Merge defaults, file contents (if present), and env vars (ALBUMENTATIONS_NO_TELEMETRY, OFFLINE).
        Returns the merged dict.
        """
        settings = self.defaults.copy()

        # Load from file if exists
        if self.settings_file.exists():
            try:
                with self.settings_file.open() as f:
                    file_settings = json.load(f)
                    settings.update(file_settings)
            except (OSError, json.JSONDecodeError):
                pass

        # Override with environment variables
        if os.environ.get("ALBUMENTATIONS_NO_TELEMETRY", "").lower() in ("1", "true"):
            settings["telemetry"] = False

        if os.environ.get("ALBUMENTATIONS_OFFLINE", "").lower() in ("1", "true"):
            settings["telemetry"] = False

        return settings

    def get(self, key: str, default: Any = None) -> Any:
        """Return value for key, or default if missing. Typical key: 'telemetry' (bool).
        Does not mutate settings. File path is platform-specific.

        Args:
            key (str): Setting name
            default (Any): Default value if setting not found

        Returns:
            Any: Setting value.

        """
        return self._settings.get(key, default)

    def update(self, **kwargs: Any) -> None:
        """Merge kwargs into current settings and write the result to the JSON file in the cache directory.
        Overwrites file. Persists immediately.

        Args:
            **kwargs (Any): Settings to update

        """
        self._settings.update(kwargs)
        self._save_settings()

    def _save_settings(self) -> None:
        """Write self._settings to JSON file (indent=2). Creates parent dirs if needed.
        Ignores OSError on write failure. Called after update().
        """
        try:
            self.settings_file.parent.mkdir(parents=True, exist_ok=True)
            with self.settings_file.open("w") as f:
                json.dump(self._settings, f, indent=2)
        except OSError:
            pass

    @property
    def telemetry_enabled(self) -> bool:
        """True if telemetry is enabled (from settings file or env). Default True unless
        ALBUMENTATIONS_NO_TELEMETRY or OFFLINE is set.
        """
        return self.get("telemetry", True)


# Global settings instance
settings = SettingsManager()
