"""Event definitions for telemetry data."""

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ComposeInitEvent:
    """Payload for one Compose init: pipeline hash, transform list, targets, environment (OS, CPU, GPU, RAM).
    to_dict() gives full nested dict.
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
        """Serialize event to dict with top-level keys plus nested 'environment' and 'pipeline'.
        For backends that need full payload.
        """
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
                "environment": self.environment,
                "gpu": self.gpu,
                "ram_gb": self.ram_gb,
            },
            "pipeline": {
                "transforms": self.transforms,
                "targets": self.targets,
            },
        }

    @staticmethod
    def generate_pipeline_hash(transforms: list[str]) -> str:
        """Compute SHA-256 hash of the transform list for deduplication. Order preserved (not sorted);
        different order gives different hash.

        Args:
            transforms (list[str]): List of transform names

        Returns:
            str: SHA-256 hash of the pipeline configuration.

        """
        # Do NOT sort transforms - order matters in augmentation pipelines!
        pipeline_str = json.dumps(transforms, sort_keys=True)
        return hashlib.sha256(pipeline_str.encode()).hexdigest()
