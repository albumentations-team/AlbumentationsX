"""Telemetry client for tracking anonymous usage statistics."""

import contextlib
import time
from threading import Thread
from typing import Any

from albumentations.core.analytics.backends.mixpanel import MixpanelBackend
from albumentations.core.analytics.collectors import is_ci_environment, is_pytest_running
from albumentations.core.analytics.events import ComposeInitEvent
from albumentations.core.analytics.settings import settings
from albumentations.core.analytics.user_id import get_user_id_manager


class TelemetryClient:
    """Singleton client for collecting and sending telemetry data with rate limiting and deduplication.

    Using Mixpanel backend for better library telemetry support:
    - No parameter limits
    - No web stream complications
    - Full transform list tracking
    - Better suited for custom events
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
            self.backend = MixpanelBackend()
            # Disable telemetry in CI/test environments
            self.enabled = not (is_ci_environment() or is_pytest_running())
            self.sent_pipelines: set[str] = set()  # Track sent pipeline hashes
            self.last_send_time: float = 0
            self.rate_limit: float = 30.0  # 30 seconds between sends
            self.user_id_manager = get_user_id_manager()
            self._initialized = True

    def track_compose_init(self, compose_data: dict[str, Any], telemetry: bool = True, use_thread: bool = True) -> None:
        """Track Compose initialization event with rate limiting and deduplication.

        Args:
            compose_data: Data collected from the Compose instance
            telemetry: Whether telemetry is enabled for this specific instance
            use_thread: If True, send telemetry in background thread (default)

        """
        if not self.enabled or not telemetry:
            return

        # Check global settings
        if not settings.telemetry_enabled:
            return

        # Get persistent user ID
        user_id = self.user_id_manager.get_or_create_user_id()
        if user_id is None:  # User opted out
            return

        # Deduplication check
        pipeline_hash = compose_data.get("pipeline_hash")
        if pipeline_hash and pipeline_hash in self.sent_pipelines:
            return  # Skip if already sent

        # Rate limiting check
        current_time = time.time()
        if current_time - self.last_send_time < self.rate_limit:
            return  # Skip if too soon

        # Add user ID to event data
        compose_data["user_id"] = user_id

        # Create event
        event = ComposeInitEvent(**compose_data)

        # Send event to backend
        if use_thread:
            # Send in background thread
            thread = Thread(target=self._send_event_thread, args=(event,), daemon=True)
            thread.start()
        else:
            # Send synchronously (mainly for testing)
            self._send_event(event)

        # Update tracking
        if pipeline_hash:
            self.sent_pipelines.add(pipeline_hash)
        self.last_send_time = current_time

    def _send_event_thread(self, event: ComposeInitEvent) -> None:
        """Send event in thread with proper error handling.

        Args:
            event: The event to send

        """
        with contextlib.suppress(Exception):
            # Silently ignore all errors in thread
            self._send_event(event)

    def _send_event(self, event: ComposeInitEvent) -> bool:
        """Send event to backend.

        Args:
            event: The event to send

        Returns:
            True if event was sent successfully, False otherwise

        """
        telemetry_sent = True
        try:
            self.backend.send_event(event)
        except (OSError, ValueError):
            # Silently ignore telemetry errors
            # OSError: network issues
            # ValueError: data validation issues
            telemetry_sent = False

        return telemetry_sent

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
