"""Google Analytics 4 backend for telemetry.

This module implements the Google Analytics 4 Measurement Protocol
for sending telemetry data.

Note: Following Ultralytics' approach, the GA4 credentials are hardcoded in the library.
Users do NOT need to provide any API keys. The library maintainers should:

1. Create a GA4 property for AlbumentationsX
2. Generate an API secret (GA4 Admin > Data Streams > Measurement Protocol > Create)
3. Replace the placeholder values below with real credentials
4. All telemetry will be sent to the AlbumentationsX GA4 property

This approach ensures zero configuration for users while allowing the library
maintainers to collect anonymous usage statistics.
"""

from __future__ import annotations

import json
import urllib.request
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from albumentations.core.analytics.events import ComposeInitEvent


class GoogleAnalyticsBackend:
    """Google Analytics 4 backend for sending telemetry data.

    Following Ultralytics' approach, credentials are hardcoded here.
    Users do not need to configure anything - telemetry just works.
    """

    # GA4 Measurement Protocol endpoint with credentials
    # These are not sensitive - they only allow sending analytics data to our property
    GA4_URL = (
        "https://www.google-analytics.com/mp/collect?measurement_id=G-R131BKTHWD&api_secret=r8vC7YWMTeCidiR2OnB3dQ"
    )

    def __init__(self) -> None:
        """Initialize the Google Analytics backend."""
        # Generate a client ID for this session
        # In production, this should be persistent per installation
        # (like Ultralytics stores in SETTINGS["uuid"])
        self.client_id = str(uuid.uuid4())

    def send_event(self, event: ComposeInitEvent) -> None:
        """Send a compose initialization event to Google Analytics.

        Args:
            event: The ComposeInitEvent to send

        """
        try:
            # Get GA4-compliant parameters from the event
            params = event.to_ga4_params()

            # Prepare the GA4 event
            payload = {
                "client_id": self.client_id,
                "events": [
                    {
                        "name": "compose_init",
                        "params": params,
                    },
                ],
            }

            # Send the request
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.GA4_URL,
                data=data,
                headers={"Content-Type": "application/json"},
            )

            # Use a short timeout to not block user code
            with urllib.request.urlopen(req, timeout=2) as _response:
                # GA4 returns 204 No Content on success
                pass

        except (OSError, urllib.error.URLError):
            # Never let telemetry errors affect user code
            # OSError: network issues
            # URLError: urllib specific errors
            pass
