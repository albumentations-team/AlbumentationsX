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
    """Sends Compose init events to Mixpanel with rate limiting (e.g. 30s) and pipeline-hash
    deduplication. Disabled in CI and pytest.

    Using Mixpanel backend for better library telemetry support:
    - No parameter limits
    - No web stream complications
    - Full transform list tracking
    - Better suited for custom events
    """

    _instance = None
    _initialized = False

    def __new__(cls) -> Self:
        """Return the single TelemetryClient instance; create on first access.
        Overrides __new__ to enforce one instance per process.
        """
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
        """Record a Compose init: check rate limit and dedup by pipeline hash, then send to
        Mixpanel (default: in a daemon thread).

        Args:
            compose_data (dict[str, Any]): Data collected from the Compose instance
            telemetry (bool): Whether telemetry is enabled for this specific instance
            use_thread (bool): If True, send telemetry in background thread (default)

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
        """Run _send_event in a daemon thread; any exception suppressed so the main process is never affected.
        Non-blocking; fire-and-forget.

        Args:
            event (ComposeInitEvent): The event to send

        """
        with contextlib.suppress(Exception):
            # Silently ignore all errors in thread
            self._send_event(event)

    def _send_event(self, event: ComposeInitEvent) -> bool:
        """POST the event to Mixpanel track API. Returns True on success, False on network or
        validation error. Synchronous, no retries.

        Args:
            event (ComposeInitEvent): The event to send

        Returns:
            bool: True if event was sent successfully, False otherwise.

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
        """Stop sending events; track_compose_init no-ops until enable(). Idempotent.
        Call from CLI or config to turn off analytics.
        """
        self.enabled = False

    def enable(self) -> None:
        """Resume sending events; rate limit and global settings still apply. Idempotent.
        Call after disable() to turn analytics back on.
        """
        self.enabled = True

    def reset(self) -> None:
        """Clear sent pipeline hashes and last-send time so the same pipeline can be sent again.
        For tests; idempotent. In-memory only.
        """
        self.sent_pipelines.clear()
        self.last_send_time = 0


# Global telemetry client instance
telemetry_client = None


def get_telemetry_client() -> TelemetryClient:
    """Return the global TelemetryClient; create on first call so Compose can send init events
    without holding a reference. One instance per process.

    Returns:
        TelemetryClient: The global TelemetryClient instance.

    """
    global telemetry_client  # noqa: PLW0603
    if telemetry_client is None:
        telemetry_client = TelemetryClient()
    return telemetry_client
