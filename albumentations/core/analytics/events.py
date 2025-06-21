"""Event definitions for telemetry data."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ComposeInitEvent:
    """Event data for Compose initialization tracking.

    Contains minimal information about pipeline configuration and environment.
    Structured to fit within GA4's 25 parameter limit.
    """

    # Core event data
    event_type: str = "compose_init"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""  # Persistent anonymous user ID
    pipeline_hash: str = ""

    # Environment info - kept as separate fields
    albumentationsx_version: str = ""
    python_version: str = ""
    os: str = ""
    cpu: str = ""
    gpu: str | None = None
    ram_gb: float | None = None
    environment: str = "unknown"  # colab/kaggle/jupyter/docker/local

    # Transform list (will be numbered transform_1, transform_2, etc.)
    transforms: list[str] = field(default_factory=list)

    # Target usage - combined field
    targets: str = "None"  # None/bboxes/keypoints/bboxes_keypoints

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for other uses (not GA4)."""
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "pipeline_hash": self.pipeline_hash,
            "environment": {
                "albumentationsx_version": self.albumentationsx_version,
                "python_version": self.python_version,
                "os": self.os,
                "cpu": self.cpu,
                "gpu": self.gpu,
                "ram_gb": self.ram_gb,
                "environment": self.environment,
            },
            "pipeline": {
                "transforms": self.transforms,
                "targets": self.targets,
            },
        }

    @staticmethod
    def generate_pipeline_hash(transforms: list[str]) -> str:
        """Generate a hash for pipeline deduplication.

        Args:
            transforms: List of transform names

        Returns:
            SHA-256 hash of the pipeline configuration

        """
        # Do NOT sort transforms - order matters in augmentation pipelines!
        pipeline_str = json.dumps(transforms, sort_keys=True)
        return hashlib.sha256(pipeline_str.encode()).hexdigest()
