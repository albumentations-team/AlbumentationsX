"""Telemetry client for tracking anonymous usage statistics."""

import contextlib
import time
from threading import Thread
from typing import Any

from typing_extensions import Self

from albumentations.core.analytics.backends.mixpanel import MixpanelBackend
from albumentations.core.analytics.collectors import is_ci_environment, is_pytest_running
from albumentations.core.analytics.events import ComposeInitEvent
from albumentations.core.analytics.settings import settings
from albumentations.core.analytics.user_id import get_user_id_manager


class TelemetryClient:
    """Send Compose initialization events with pipeline deduplication and process-wide rate limiting.
    Disabled in CI and pytest.

    Mixpanel accepts the complete transform list without web-stream limits.
    """

    _instance = None
    _initialized = False

    def __new__(cls) -> Self:
        """Return the process-wide telemetry client."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not self._initialized:
            self.backend = MixpanelBackend()
            # Disable telemetry in CI/test environments
            self.enabled = not (is_ci_environment() or is_pytest_running())
            self.sent_pipelines: set[str] = set()
            self.last_send_time: float = 0
            self.rate_limit: float = 30.0
            self.user_id_manager = get_user_id_manager()
            self._initialized = True

    def track_compose_init(self, compose_data: dict[str, Any], telemetry: bool = True, use_thread: bool = True) -> None:
        """Record a Compose initialization event after opt-out, deduplication, and rate-limit checks.
        Send it in a daemon thread unless the caller asks for synchronous delivery.

        Args:
            compose_data (dict[str, Any]): Data collected from the Compose instance
            telemetry (bool): Whether telemetry is enabled for this specific instance
            use_thread (bool): If True, send telemetry in background thread (default)

        """
        if not self.enabled or not telemetry:
            return

        if not settings.telemetry_enabled:
            return

        user_id = self.user_id_manager.get_or_create_user_id()
        if user_id is None:  # The user opted out.
            return

        pipeline_hash = compose_data.get("pipeline_hash")
        if pipeline_hash and pipeline_hash in self.sent_pipelines:
            return

        current_time = time.time()
        if current_time - self.last_send_time < self.rate_limit:
            return

        compose_data["user_id"] = user_id
        event = ComposeInitEvent(**compose_data)

        if use_thread:
            thread = Thread(target=self._send_event_thread, args=(event,), daemon=True)
            thread.start()
        else:
            # Send synchronously (mainly for testing)
            self._send_event(event)

        if pipeline_hash:
            self.sent_pipelines.add(pipeline_hash)
        self.last_send_time = current_time

    def _send_event_thread(self, event: ComposeInitEvent) -> None:
        """Send an event without allowing telemetry failures to affect the caller."""
        with contextlib.suppress(Exception):
            self._send_event(event)

    def _send_event(self, event: ComposeInitEvent) -> bool:
        """Send an event synchronously without retries.

        Args:
            event (ComposeInitEvent): The event to send

        Returns:
            bool: True if event was sent successfully, False otherwise.

        """
        telemetry_sent = True
        try:
            self.backend.send_event(event)
        except (OSError, ValueError):
            telemetry_sent = False

        return telemetry_sent

    def disable(self) -> None:
        """Stop sending events until enable() is called."""
        self.enabled = False

    def enable(self) -> None:
        """Resume sending events subject to the global setting and rate limit."""
        self.enabled = True

    def reset(self) -> None:
        """Clear in-memory pipeline and rate-limit state."""
        self.sent_pipelines.clear()
        self.last_send_time = 0


telemetry_client = None


def get_telemetry_client() -> TelemetryClient:
    """Return the process-wide telemetry client, creating it on first use."""
    global telemetry_client  # noqa: PLW0603
    if telemetry_client is None:
        telemetry_client = TelemetryClient()
    return telemetry_client
