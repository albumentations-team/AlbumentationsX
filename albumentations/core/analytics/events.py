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

    def to_ga4_params(self) -> dict[str, Any]:
        """Convert event to GA4-compatible parameters (max 25 params).

        Returns compact dictionary suitable for GA4 event tracking.
        """
        params = {
            "pipeline_hash": self.pipeline_hash[:32],  # Truncate hash for space
            "version": self.albumentationsx_version,
            "python_version": self.python_version,
            "os": self.os,
            "cpu": self.cpu,
            "environment": self.environment,
            "targets": self.targets,
            "num_transforms": len(self.transforms),
        }

        # Add GPU if available
        if self.gpu:
            params["gpu"] = self.gpu

        # Add RAM if available
        if self.ram_gb is not None:
            params["ram_gb"] = round(self.ram_gb, 1)

        # Add individual transforms (up to 10 to leave room for other params)
        # Skip Normalize and ToTensorV2 as suggested
        excluded_transforms = {"Normalize", "ToTensorV2"}
        filtered_transforms = [t for t in self.transforms if t not in excluded_transforms]

        # Add comma-separated list of unique transforms for easier analysis
        unique_transforms = sorted(set(filtered_transforms))
        if unique_transforms:
            # Truncate if too long (GA4 parameter value limit is 100 chars)
            transform_list = ",".join(unique_transforms)
            if len(transform_list) > 100:
                # Take first 97 chars and add "..."
                transform_list = transform_list[:97] + "..."
            params["transform_types"] = transform_list

        for i, transform in enumerate(filtered_transforms[:10], 1):
            params[f"transform_{i}"] = transform

        return params

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for other uses (not GA4)."""
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
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
        # Sort transforms to ensure consistent hashing
        pipeline_str = json.dumps(sorted(transforms), sort_keys=True)
        return hashlib.sha256(pipeline_str.encode()).hexdigest()

    def anonymize(self) -> None:
        """Anonymize any potentially sensitive data."""
        # Session ID is already anonymized in telemetry client
        # Transform names don't contain sensitive data as they're just class names

    def _anonymize_params(self, params: dict[str, Any]) -> None:
        """Recursively anonymize parameters.

        Args:
            params: Parameters dictionary to anonymize

        """
        # Not used anymore since we don't collect transform parameters
