"""Mixpanel backend for telemetry."""

from __future__ import annotations

import base64
import json
import urllib.request
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from albumentations.core.analytics.events import ComposeInitEvent


class MixpanelBackend:
    """Mixpanel backend for sending telemetry data.

    Mixpanel is much simpler than GA4 for library telemetry:
    - No web stream complications
    - No parameter limits
    - Better suited for custom events
    """

    MIXPANEL_URL = "https://api.mixpanel.com/track"
    PROJECT_TOKEN = "9674977e5658e19ce4710845fdd68712"  # noqa: S105 - This is a public token, not a secret

    def _parse_timestamp(self, timestamp: str | datetime | None) -> int | None:
        """Parse timestamp to Unix epoch seconds.

        Args:
            timestamp: ISO format string or datetime object

        Returns:
            Unix timestamp in seconds or None

        """
        if timestamp is None:
            return None

        if isinstance(timestamp, str):
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                return int(dt.timestamp())
            except (ValueError, AttributeError):
                return None
        elif hasattr(timestamp, "timestamp"):
            return int(timestamp.timestamp())
        else:
            return None

    def send_event(self, event: ComposeInitEvent) -> None:
        """Send a compose initialization event to Mixpanel.

        Args:
            event: The ComposeInitEvent to send

        """
        try:
            # Convert event to Mixpanel format
            event_data: dict[str, Any] = {
                "event": "Compose Init",
                "properties": {
                    # User identification
                    "distinct_id": event.user_id or "anonymous",
                    "token": self.PROJECT_TOKEN,
                    # Event metadata
                    "$insert_id": event.session_id,  # For deduplication
                    "time": self._parse_timestamp(event.timestamp),
                    # Environment info
                    "pipeline_hash": event.pipeline_hash,
                    "version": event.albumentationsx_version,
                    "python_version": event.python_version,
                    "cpu": event.cpu,
                    "gpu": event.gpu,
                    "ram_gb": event.ram_gb,
                    "environment": event.environment,
                    # Pipeline info
                    "targets": event.targets,
                    "num_transforms": len(event.transforms),
                    "transforms": event.transforms,  # Full list - no limits!
                    # Mixpanel automatic properties
                    "$os": event.os,  # Mixpanel recognizes this
                    "$lib": "albumentationsx",
                    "$lib_version": event.albumentationsx_version,
                },
            }
            # Remove None values for cleaner data
            filtered_properties = {k: v for k, v in event_data["properties"].items() if v is not None}
            event_data["properties"] = filtered_properties

            # Encode the data (Mixpanel expects base64)
            encoded_data = base64.b64encode(
                json.dumps(event_data).encode("utf-8"),
            ).decode("utf-8")

            # Send request
            data = f"data={encoded_data}".encode()
            req = urllib.request.Request(
                self.MIXPANEL_URL,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

            # Use a short timeout to not block user code
            with urllib.request.urlopen(req, timeout=2) as response:
                # Mixpanel returns 1 for success, 0 for failure
                # We silently ignore failures
                response.read()

        except (OSError, urllib.error.URLError, UnicodeDecodeError, json.JSONDecodeError):
            # Never let telemetry errors affect user code
            pass
