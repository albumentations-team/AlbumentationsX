"""Settings management for AlbumentationsX.

This module provides centralized configuration management for telemetry,
version checking, and other library-wide settings.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from albumentations.core.cache_utils import get_cache_dir


class SettingsManager:
    """Simple settings manager for AlbumentationsX configuration.

    This class manages user settings with environment variable overrides.
    Settings are stored in a JSON file in the user's cache directory.
    """

    def __init__(self, settings_file: Path | None = None):
        """Initialize the settings manager.

        Args:
            settings_file: Path to settings file. If None, uses default location.

        """
        self.settings_file = settings_file or (get_cache_dir() / "settings.json")
        self.defaults = {
            "telemetry": True,
        }
        self._settings = self._load_settings()

    def _load_settings(self) -> dict[str, Any]:
        """Load settings from file with defaults and env var overrides."""
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
        """Get a setting value.

        Args:
            key: Setting name
            default: Default value if setting not found

        Returns:
            Setting value

        """
        return self._settings.get(key, default)

    def update(self, **kwargs: Any) -> None:
        """Update settings and save to file.

        Args:
            **kwargs: Settings to update

        """
        self._settings.update(kwargs)
        self._save_settings()

    def _save_settings(self) -> None:
        """Save current settings to file."""
        try:
            self.settings_file.parent.mkdir(parents=True, exist_ok=True)
            with self.settings_file.open("w") as f:
                json.dump(self._settings, f, indent=2)
        except OSError:
            pass

    @property
    def telemetry_enabled(self) -> bool:
        """Check if telemetry is enabled."""
        return self.get("telemetry", True)


# Global settings instance
settings = SettingsManager()
