"""Telemetry client for tracking anonymous usage statistics."""

import time
from typing import Any

from albumentations.core.analytics.backends.google import GoogleAnalyticsBackend
from albumentations.core.analytics.collectors import is_ci_environment, is_pytest_running
from albumentations.core.analytics.events import ComposeInitEvent
from albumentations.core.analytics.settings import settings


class TelemetryClient:
    """Singleton client for collecting and sending telemetry data with rate limiting and deduplication.

    The client ensures telemetry data fits within GA4's 25 parameter limit by:
    - Combining environment info into a single string
    - Using numbered transform parameters (transform_1, transform_2, etc.)
    - Excluding common transforms like Normalize and ToTensorV2
    - Limiting to 15 transforms to leave room for other parameters
    """

    _instance = None
    _initialized = False

    def __new__(cls) -> "TelemetryClient":
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not self._initialized:
            self.backend = GoogleAnalyticsBackend()
            # Disable telemetry in CI/test environments
            self.enabled = not (is_ci_environment() or is_pytest_running())
            self.sent_pipelines: set[str] = set()  # Track sent pipeline hashes
            self.last_send_time: float = 0
            self.rate_limit: float = 30.0  # 30 seconds between sends
            self._initialized = True

    def track_compose_init(self, compose_data: dict[str, Any], telemetry: bool = True) -> None:
        """Track Compose initialization event with rate limiting and deduplication.

        Args:
            compose_data: Data collected from the Compose instance
            telemetry: Whether telemetry is enabled for this specific instance

        """
        if not self.enabled or not telemetry:
            return

        # Check global settings
        if not settings.get("telemetry", True):
            return

        # Deduplication check
        pipeline_hash = compose_data.get("pipeline_hash")
        if pipeline_hash and pipeline_hash in self.sent_pipelines:
            return  # Skip if already sent

        # Rate limiting check
        current_time = time.time()
        if current_time - self.last_send_time < self.rate_limit:
            return  # Skip if too soon

        # Create event
        event = ComposeInitEvent(**compose_data)

        # Send event to backend
        telemetry_sent = True
        try:
            self.backend.send_event(event)
        except (OSError, ValueError):
            # Silently ignore telemetry errors
            # OSError: network issues
            # ValueError: data validation issues
            telemetry_sent = False

        # Update tracking only if telemetry was sent successfully
        if telemetry_sent and pipeline_hash:
            self.sent_pipelines.add(pipeline_hash)
        if telemetry_sent:
            self.last_send_time = current_time

    def disable(self) -> None:
        """Disable telemetry collection."""
        self.enabled = False

    def enable(self) -> None:
        """Enable telemetry collection."""
        self.enabled = True

    def reset(self) -> None:
        """Reset the telemetry client state (mainly for testing)."""
        self.sent_pipelines.clear()
        self.last_send_time = 0


# Global telemetry client instance
telemetry_client = None


def get_telemetry_client() -> TelemetryClient:
    """Get or create the global telemetry client.

    Returns:
        The global TelemetryClient instance

    """
    global telemetry_client  # noqa: PLW0603
    if telemetry_client is None:
        telemetry_client = TelemetryClient()
    return telemetry_client
