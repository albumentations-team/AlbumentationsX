"""Tests for telemetry functionality in AlbumentationsX."""

from __future__ import annotations

import time
from unittest.mock import Mock, patch
import pytest
import numpy as np

import albumentations as A
from albumentations.core.analytics.settings import settings
from albumentations.core.analytics.telemetry import TelemetryClient, get_telemetry_client
from albumentations.core.analytics.events import ComposeInitEvent
from albumentations.core.analytics.collectors import (
    get_environment_info,
    collect_pipeline_info,
)


@pytest.fixture(autouse=True)
def reset_telemetry_client():
    """Reset the telemetry client singleton between tests."""
    # Clear the global telemetry client
    import albumentations.core.analytics.telemetry
    albumentations.core.analytics.telemetry.telemetry_client = None

    # Clear the singleton instance
    TelemetryClient._instance = None
    TelemetryClient._initialized = False

    yield

    # Clean up after test
    albumentations.core.analytics.telemetry.telemetry_client = None
    TelemetryClient._instance = None
    TelemetryClient._initialized = False


class TestTelemetrySettings:
    """Test telemetry settings management."""

    def test_telemetry_enabled_by_default(self):
        """Test that telemetry is enabled by default."""
        # Create fresh settings instance
        from albumentations.core.analytics.settings import SettingsManager
        test_settings = SettingsManager()
        assert test_settings.telemetry_enabled is True

    def test_telemetry_disable_via_settings(self):
        """Test disabling telemetry via settings."""
        settings.update(telemetry=False)
        assert settings.telemetry_enabled is False
        # Reset
        settings.update(telemetry=True)

    def test_telemetry_disable_via_env_var(self, monkeypatch):
        """Test disabling telemetry via environment variable."""
        monkeypatch.setenv("ALBUMENTATIONS_NO_TELEMETRY", "1")
        # Create new settings instance to pick up env var
        from albumentations.core.analytics.settings import SettingsManager
        test_settings = SettingsManager()
        assert test_settings.telemetry_enabled is False

    def test_offline_mode_disables_telemetry(self, monkeypatch):
        """Test that offline mode disables telemetry."""
        monkeypatch.setenv("ALBUMENTATIONS_OFFLINE", "1")
        from albumentations.core.analytics.settings import SettingsManager
        test_settings = SettingsManager()
        assert test_settings.telemetry_enabled is False
        assert test_settings.check_updates_enabled is False


class TestTelemetryClient:
    """Test the main telemetry client."""

    def test_client_singleton(self):
        """Test that client is a singleton."""
        client1 = TelemetryClient()
        client2 = TelemetryClient()
        assert client1 is client2

    def test_client_enabled_by_default(self):
        """Test that client is disabled during pytest runs."""
        client = TelemetryClient()
        # Since we're running under pytest, telemetry should be disabled
        assert client.enabled is False

    def test_disable_enable_client(self):
        """Test disabling and enabling the client."""
        client = TelemetryClient()
        client.disable()
        assert client.enabled is False
        client.enable()
        assert client.enabled is True

    def test_rate_limiting(self):
        """Test that rate limiting prevents too frequent sends."""
        client = TelemetryClient()
        # Enable client for this test
        client.enable()
        client.reset()  # Clear state

        # First event should go through
        event1 = {
            "pipeline_hash": "hash1",
            "transforms": ["RandomCrop"],
        }
        client.track_compose_init(event1, telemetry=True)
        first_time = client.last_send_time
        assert first_time > 0

        # Second event within rate limit should be skipped
        time.sleep(0.1)  # Small delay
        event2 = {
            "pipeline_hash": "hash2",
            "transforms": ["HorizontalFlip"],
        }
        client.track_compose_init(event2, telemetry=True)
        assert client.last_send_time == first_time  # Time shouldn't update

        # Wait for rate limit to expire
        client.last_send_time = 0  # Force expire
        client.track_compose_init(event2, telemetry=True)
        assert client.last_send_time > first_time

    def test_deduplication(self):
        """Test that duplicate pipelines are not sent twice."""
        client = TelemetryClient()
        # Enable client for this test
        client.enable()
        client.reset()  # Clear state

        # First pipeline should be tracked
        event1 = {
            "pipeline_hash": "same_hash",
            "transforms": ["RandomCrop", "HorizontalFlip"],
        }
        client.track_compose_init(event1, telemetry=True)
        assert "same_hash" in client.sent_pipelines

        # Same pipeline hash should be skipped
        client.last_send_time = 0  # Bypass rate limit
        event2 = {
            "pipeline_hash": "same_hash",
            "transforms": ["RandomCrop", "HorizontalFlip"],
        }
        old_time = client.last_send_time
        client.track_compose_init(event2, telemetry=True)
        assert client.last_send_time == old_time  # Should not update

    def test_track_compose_init_when_disabled_globally(self):
        """Test that tracking is skipped when telemetry is disabled globally."""
        # Disable telemetry
        original = settings.get("telemetry", True)
        settings.update(telemetry=False)
        try:
            client = TelemetryClient()
            client.reset()

            event = {"pipeline_hash": "test", "transforms": ["RandomCrop"]}
            client.track_compose_init(event, telemetry=True)

            # No event should be tracked
            assert len(client.sent_pipelines) == 0
        finally:
            settings.update(telemetry=original)

    def test_track_compose_init_when_disabled_locally(self):
        """Test that tracking is skipped when telemetry is disabled for specific compose."""
        # Enable telemetry globally
        settings.update(telemetry=True)
        client = TelemetryClient()
        client.reset()

        event = {"pipeline_hash": "test", "transforms": ["RandomCrop"]}
        client.track_compose_init(event, telemetry=False)

        # No event should be tracked
        assert len(client.sent_pipelines) == 0

    def test_get_telemetry_client_singleton(self):
        """Test that get_telemetry_client returns singleton."""
        client1 = get_telemetry_client()
        client2 = get_telemetry_client()
        assert client1 is client2


class TestCIEnvironmentDetection:
    """Test CI/CD environment detection."""

    def test_ci_environment_detection(self, monkeypatch):
        """Test detection of various CI environments."""
        from albumentations.core.analytics.collectors import is_ci_environment

        # Test various CI environment variables
        ci_vars = [
            "CI",
            "CONTINUOUS_INTEGRATION",
            "GITHUB_ACTIONS",
            "GITLAB_CI",
            "JENKINS_HOME",
            "TRAVIS",
            "CIRCLECI",
            "BUILDKITE",
            "DRONE",
            "TEAMCITY_VERSION",
            "BITBUCKET_BUILD_NUMBER",
            "SEMAPHORE",
            "APPVEYOR",
            "CODEBUILD_BUILD_ID",
            "AZURE_PIPELINES_BUILD_ID",
            "TF_BUILD",
        ]

        # Test each CI variable
        for var in ci_vars:
            monkeypatch.setenv(var, "true")
            assert is_ci_environment() is True
            monkeypatch.delenv(var)

        # Test when no CI variables are set
        assert is_ci_environment() is False

    def test_pytest_detection(self, monkeypatch):
        """Test pytest environment detection."""
        from albumentations.core.analytics.collectors import is_pytest_running

        # Simulate pytest environment
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_file.py::test_function")
        assert is_pytest_running() is True

        # Remove pytest variable
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        assert is_pytest_running() is False

    def test_telemetry_disabled_in_ci(self, monkeypatch):
        """Test that telemetry is disabled in CI environments."""
        # Clear any existing telemetry client
        TelemetryClient._instance = None
        TelemetryClient._initialized = False

        # Set CI environment
        monkeypatch.setenv("GITHUB_ACTIONS", "true")

        # Create new client
        client = TelemetryClient()
        assert client.enabled is False

    def test_telemetry_disabled_in_pytest(self, monkeypatch):
        """Test that telemetry is disabled when pytest is running."""
        # The client should already be disabled since we're running in pytest
        # Just verify this behavior
        client = TelemetryClient()
        # Since we're running under pytest, telemetry should be disabled
        assert client.enabled is False

    def test_environment_returns_ci(self, monkeypatch):
        """Test that detect_environment returns 'ci' when in CI."""
        from albumentations.core.analytics.collectors import detect_environment

        # Set CI environment
        monkeypatch.setenv("CI", "true")
        assert detect_environment() == "ci"

        # CI should have highest priority
        monkeypatch.setenv("COLAB_GPU", "1")  # Also set Colab
        assert detect_environment() == "ci"  # Still returns CI

    def test_compose_no_telemetry_in_ci(self, monkeypatch):
        """Test that Compose doesn't send telemetry in CI environments."""
        # Since we're already in pytest, telemetry is disabled
        # Let's test that the compose still works
        transform = A.Compose([
            A.RandomCrop(256, 256),
            A.HorizontalFlip(p=0.5),
        ])

        # Transform should work normally
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        result = transform(image=image)
        assert result['image'].shape == (256, 256, 3)

        # Client should be disabled (because we're in pytest)
        client = get_telemetry_client()
        assert client.enabled is False


class TestComposeIntegration:
    """Test telemetry integration with Compose class."""

    @patch('albumentations.core.composition.get_telemetry_client')
    @patch('albumentations.core.composition.get_environment_info')
    @patch('albumentations.core.composition.collect_pipeline_info')
    def test_compose_with_telemetry_enabled(self, mock_collect_pipeline, mock_get_env, mock_get_client):
        """Test that Compose tracks telemetry when enabled."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_get_env.return_value = {"environment": "test"}
        mock_collect_pipeline.return_value = {"transforms": ["RandomCrop", "HorizontalFlip"], "pipeline_hash": "test_hash"}

        # Enable telemetry
        settings.update(telemetry=True)
        try:
            transform = A.Compose([
                A.RandomCrop(256, 256),
                A.HorizontalFlip(p=0.5),
            ], telemetry=True)

            # Verify telemetry was tracked
            mock_client.track_compose_init.assert_called_once()
            args = mock_client.track_compose_init.call_args
            # First argument should be the telemetry data dict
            telemetry_data = args[0][0]
            assert telemetry_data['environment'] == 'test'
            assert telemetry_data['transforms'] == ["RandomCrop", "HorizontalFlip"]
            assert args[1]['telemetry'] is True
        finally:
            settings.update(telemetry=True)

    @patch('albumentations.core.composition.get_telemetry_client')
    def test_compose_with_telemetry_disabled(self, mock_get_client):
        """Test that Compose tracks telemetry call with telemetry=False."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        # Enable telemetry globally
        settings.update(telemetry=True)
        try:
            transform = A.Compose([
                A.RandomCrop(256, 256),
                A.HorizontalFlip(p=0.5),
            ], telemetry=False)

            # Verify telemetry was called but with telemetry=False
            mock_client.track_compose_init.assert_called_once()
            args = mock_client.track_compose_init.call_args
            assert args[1]['telemetry'] is False
        finally:
            settings.update(telemetry=True)

    def test_compose_telemetry_never_raises(self):
        """Test that telemetry errors never affect user code."""
        # Make telemetry collection raise an exception
        with patch('albumentations.core.composition.get_telemetry_client', side_effect=Exception("Telemetry error")):
            # This should not raise
            transform = A.Compose([
                A.RandomCrop(256, 256),
                A.HorizontalFlip(p=0.5),
            ])

            # Transform should work normally
            image = np.zeros((512, 512, 3), dtype=np.uint8)
            result = transform(image=image)
            assert result['image'].shape == (256, 256, 3)


class TestDataCollectors:
    """Test data collection functions."""

    def test_get_environment_info(self):
        """Test environment info collection."""
        info = get_environment_info()

        # Check basic fields exist
        assert 'albumentationsx_version' in info
        assert 'python_version' in info
        assert 'os' in info
        assert 'cpu' in info
        assert 'environment' in info

        # Check optional fields (may or may not exist depending on system)
        # gpu and ram_gb can be None
        assert 'gpu' in info
        assert 'ram_gb' in info

        # Check environment is one of the expected values
        assert info['environment'] in ['ci', 'colab', 'kaggle', 'docker', 'jupyter', 'local']

        # OS should be more specific than just "Linux", "Darwin", "Windows"
        # e.g., "Ubuntu 22.04", "macOS 14.2", "Windows 11"
        assert info['os'] and len(info['os']) > 0

    def test_collect_pipeline_info(self):
        """Test pipeline info collection."""
        compose = A.Compose([
            A.RandomCrop(256, 256),
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.Blur(blur_limit=3),
                A.MedianBlur(blur_limit=3),
            ], p=0.5),
        ])

        info = collect_pipeline_info(compose)

        assert 'transforms' in info
        assert 'targets' in info
        assert 'pipeline_hash' in info

        # Check transform names (should include nested transforms)
        transform_names = info['transforms']
        assert 'RandomCrop' in transform_names
        assert 'HorizontalFlip' in transform_names
        assert 'OneOf' in transform_names
        assert 'Blur' in transform_names
        assert 'MedianBlur' in transform_names

        # Check target usage
        assert info['targets'] == 'None'

    def test_collect_pipeline_info_with_targets(self):
        """Test pipeline info collection with different target configurations."""
        # Test with bboxes only
        compose_bbox = A.Compose(
            [A.RandomCrop(256, 256)],
            bbox_params=A.BboxParams(format='pascal_voc')
        )
        info = collect_pipeline_info(compose_bbox)
        assert info['targets'] == 'bboxes'

        # Test with keypoints only
        compose_kp = A.Compose(
            [A.RandomCrop(256, 256)],
            keypoint_params=A.KeypointParams(format='xy')
        )
        info = collect_pipeline_info(compose_kp)
        assert info['targets'] == 'keypoints'

        # Test with both bboxes and keypoints
        compose_both = A.Compose(
            [A.RandomCrop(256, 256)],
            bbox_params=A.BboxParams(format='pascal_voc'),
            keypoint_params=A.KeypointParams(format='xy')
        )
        info = collect_pipeline_info(compose_both)
        assert info['targets'] == 'bboxes_keypoints'


class TestComposeInitEvent:
    """Test the ComposeInitEvent data structure."""

    def test_event_creation(self):
        """Test creating a ComposeInitEvent."""
        event = ComposeInitEvent(
            session_id="test_session",
            albumentationsx_version="2.0.0",
            python_version="3.10",
            os="Ubuntu 22.04",
            cpu="Intel Core i7-9700K",
            environment="local",
            transforms=["RandomCrop", "HorizontalFlip"],
            targets="bboxes",
        )

        assert event.event_type == "compose_init"
        assert event.session_id == "test_session"
        assert len(event.transforms) == 2
        assert event.targets == "bboxes"

    def test_event_to_dict(self):
        """Test converting event to dictionary."""
        event = ComposeInitEvent(
            session_id="test_session",
            albumentationsx_version="2.0.0",
            python_version="3.10",
            os="Ubuntu 22.04",
            cpu="Intel Core i7-9700K",
            environment="jupyter",
            gpu="NVIDIA RTX 3080",
            ram_gb=16.0,
            transforms=["RandomCrop", "HorizontalFlip"],
            targets="bboxes_keypoints",
        )

        data = event.to_dict()

        assert data['event_type'] == "compose_init"
        assert data['session_id'] == "test_session"
        assert data['environment']['os'] == "Ubuntu 22.04"
        assert data['environment']['environment'] == "jupyter"
        assert data['environment']['gpu'] == "NVIDIA RTX 3080"
        assert data['environment']['ram_gb'] == 16.0
        assert data['pipeline']['transforms'] == ["RandomCrop", "HorizontalFlip"]
        assert data['pipeline']['targets'] == "bboxes_keypoints"

    def test_generate_pipeline_hash(self):
        """Test pipeline hash generation."""
        transforms1 = ["RandomCrop", "HorizontalFlip", "Blur"]
        transforms2 = ["HorizontalFlip", "RandomCrop", "Blur"]  # Different order
        transforms3 = ["RandomCrop", "HorizontalFlip"]  # Different transforms

        hash1 = ComposeInitEvent.generate_pipeline_hash(transforms1)
        hash2 = ComposeInitEvent.generate_pipeline_hash(transforms2)
        hash3 = ComposeInitEvent.generate_pipeline_hash(transforms3)

        # Different transform orders should produce different hashes
        # because order matters in augmentation pipelines!
        assert hash1 != hash2
        # Different transforms should produce different hash
        assert hash1 != hash3
        assert hash2 != hash3

    def test_event_data_for_mixpanel(self):
        """Test event data structure is suitable for Mixpanel."""
        event = ComposeInitEvent(
            user_id="test-user-123",
            session_id="test_session",
            albumentationsx_version="2.0.0",
            python_version="3.10",
            os="Ubuntu 22.04",
            cpu="Intel Core i7-9700K",
            environment="jupyter",
            gpu="NVIDIA RTX 3080",
            ram_gb=16.0,
            transforms=["RandomCrop", "HorizontalFlip", "Normalize", "ToTensorV2", "Blur", "MedianBlur"],
            targets="bboxes_keypoints",
            pipeline_hash="abcdef1234567890" * 4,  # Long hash
        )

        # Get event dict
        data = event.to_dict()

        # Check that all transforms are included (no exclusions for Mixpanel!)
        assert len(data["pipeline"]["transforms"]) == 6
        assert "Normalize" in data["pipeline"]["transforms"]
        assert "ToTensorV2" in data["pipeline"]["transforms"]

        # Check environment data
        env = data["environment"]
        assert env["albumentationsx_version"] == "2.0.0"
        assert env["python_version"] == "3.10"
        assert env["os"] == "Ubuntu 22.04"
        assert env["cpu"] == "Intel Core i7-9700K"
        assert env["gpu"] == "NVIDIA RTX 3080"
        assert env["ram_gb"] == 16.0
        assert env["environment"] == "jupyter"

        # Check full pipeline hash is preserved
        assert data["pipeline_hash"] == "abcdef1234567890" * 4

    def test_event_minimal_data(self):
        """Test event with minimal data."""
        event = ComposeInitEvent(
            albumentationsx_version="2.0.0",
            python_version="3.10",
            os="macOS 14.2",
            cpu="Apple M1",
            environment="local",
            transforms=[],
        )

        data = event.to_dict()

        assert data["environment"]["os"] == "macOS 14.2"
        assert data["environment"]["cpu"] == "Apple M1"
        assert data["environment"]["environment"] == "local"
        assert data["pipeline"]["targets"] == "None"  # Default value
        assert len(data["pipeline"]["transforms"]) == 0
        assert data["environment"]["gpu"] is None  # Not provided
        assert data["environment"]["ram_gb"] is None  # Not provided

    def test_event_many_transforms(self):
        """Test event with many transforms - no limits with Mixpanel!"""
        # Create 20 transforms
        transforms = [f"Transform{i}" for i in range(20)]

        event = ComposeInitEvent(
            transforms=transforms,
            albumentationsx_version="2.0.0",
            python_version="3.10",
            os="Ubuntu 22.04",
            cpu="AMD Ryzen 9 5900X",
            environment="colab",
            targets="None",
        )

        data = event.to_dict()

        # All 20 transforms should be included with Mixpanel!
        assert len(data["pipeline"]["transforms"]) == 20
        assert data["pipeline"]["transforms"][0] == "Transform0"
        assert data["pipeline"]["transforms"][19] == "Transform19"


class TestComplexPipelines:
    """Test telemetry with complex pipeline configurations."""

    @patch('albumentations.core.composition.get_telemetry_client')
    @patch('albumentations.core.composition.get_environment_info')
    @patch('albumentations.core.composition.collect_pipeline_info')
    def test_compose_with_all_features(self, mock_collect_pipeline, mock_get_env, mock_get_client):
        """Test telemetry with fully-featured compose."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_get_env.return_value = {"environment": "test"}
        mock_collect_pipeline.return_value = {
            "transforms": ["RandomCrop", "HorizontalFlip", "OneOf", "Blur", "MedianBlur", "SomeOf", "RandomBrightnessContrast", "HueSaturationValue"],
            "pipeline_hash": "complex_hash",
            "targets": "bboxes_keypoints",
        }

        # Enable telemetry
        settings.update(telemetry=True)
        try:
            compose = A.Compose([
                A.RandomCrop(256, 256),
                A.HorizontalFlip(p=0.5),
                A.OneOf([
                    A.Blur(blur_limit=3),
                    A.MedianBlur(blur_limit=3),
                ], p=0.5),
                A.SomeOf([
                    A.RandomBrightnessContrast(p=0.5),
                    A.HueSaturationValue(p=0.5),
                ], n=1, p=0.5),
            ],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
            keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_labels']),
            additional_targets={'image2': 'image', 'mask2': 'mask'},
            )

            # Verify telemetry was called
            mock_client.track_compose_init.assert_called_once()

            # Verify the data was passed correctly
            args = mock_client.track_compose_init.call_args
            telemetry_data = args[0][0]
            assert telemetry_data['targets'] == 'bboxes_keypoints'
        finally:
            settings.update(telemetry=True)

    def test_nested_compose_telemetry(self):
        """Test telemetry behavior with nested compose structures.

        Note: Both main and nested compose track telemetry because disable_check_args_private
        is called after the nested compose's __init__ completes.
        """
        # Enable telemetry
        settings.update(telemetry=True)

        # Track telemetry calls
        telemetry_calls = []

        try:
            # Mock the telemetry client to track calls
            mock_client = Mock()

            def track_call(telemetry_data, telemetry):
                # Extract main_compose from the Compose object if passed
                # In the patched version, we get data dict, not Compose
                telemetry_calls.append({
                    'telemetry': telemetry,
                    'data': telemetry_data
                })

            mock_client.track_compose_init = track_call

            with patch('albumentations.core.composition.get_telemetry_client', return_value=mock_client):
                # Create compose with nested structure
                main_compose = A.Compose([
                    A.Compose([  # Nested compose
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                    ]),
                    A.RandomBrightnessContrast(p=0.3),
                ])

            # Both composes track telemetry
            assert len(telemetry_calls) == 2, f"Expected 2 telemetry calls, got {len(telemetry_calls)}"

            # After construction, nested compose should have main_compose=False
            nested = main_compose.transforms[0]
            assert isinstance(nested, A.Compose)
            assert nested.main_compose is False
            assert main_compose.main_compose is True

        finally:
            settings.update(telemetry=True)


@pytest.mark.parametrize("env_var,env_value,expected", [
            ("ALBUMENTATIONS_NO_TELEMETRY", "1", False),
        ("ALBUMENTATIONS_NO_TELEMETRY", "true", False),
        ("ALBUMENTATIONS_NO_TELEMETRY", "TRUE", False),
        ("ALBUMENTATIONS_NO_TELEMETRY", "0", True),
        ("ALBUMENTATIONS_NO_TELEMETRY", "false", True),
            ("ALBUMENTATIONS_OFFLINE", "1", False),
    ("NO_ALBUMENTATIONS_UPDATE", "1", True),  # Only affects updates, not telemetry
])
def test_environment_variables(monkeypatch, env_var, env_value, expected):
    """Test various environment variable configurations."""
    # Clear all relevant env vars first
    for var in ["ALBUMENTATIONS_NO_TELEMETRY", "ALBUMENTATIONS_OFFLINE", "NO_ALBUMENTATIONS_UPDATE"]:
        monkeypatch.delenv(var, raising=False)

    # Set the test env var
    monkeypatch.setenv(env_var, env_value)

    # Create new settings instance to pick up env vars
    from albumentations.core.analytics.settings import SettingsManager
    test_settings = SettingsManager()

    assert test_settings.telemetry_enabled == expected


def test_settings_manager():
    """Test the SettingsManager functionality."""
    from albumentations.core.analytics.settings import SettingsManager
    import tempfile
    from pathlib import Path

    # Create a temporary settings file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = Path(f.name)

    try:
        # Create settings with custom file
        settings = SettingsManager(settings_file=temp_path)

        # Test defaults
        assert settings.get("telemetry") is True
        assert settings.get("check_updates") is True

        # Test update
        settings.update(telemetry=False)
        assert settings.telemetry_enabled is False

        # Create new instance - should load from file
        settings2 = SettingsManager(settings_file=temp_path)
        assert settings2.telemetry_enabled is False

        # Test getting non-existent key with default
        assert settings.get("non_existent", "default") == "default"

    finally:
        # Clean up
        if temp_path.exists():
            temp_path.unlink()


class TestTelemetryIntegration:
    """Integration tests for telemetry with real pipelines."""

    def test_simple_pipeline_with_telemetry(self):
        """Test creating and using a simple pipeline with telemetry enabled."""
        # Enable telemetry
        settings.update(telemetry=True)
        # Create a simple pipeline
        transform = A.Compose([
            A.RandomCrop(256, 256),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ])

        # Test the pipeline works
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        result = transform(image=image)
        assert result['image'].shape == (256, 256, 3)

    def test_pipeline_with_telemetry_disabled(self):
        """Test creating a pipeline with telemetry explicitly disabled."""
        # Create a pipeline with telemetry disabled
        transform = A.Compose([
            A.RandomCrop(256, 256),
            A.HorizontalFlip(p=0.5),
        ], telemetry=False)

        # Test it works
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        result = transform(image=image)
        assert result['image'].shape == (256, 256, 3)

    def test_complex_pipeline_with_telemetry(self):
        """Test complex pipeline with nested structures and telemetry."""
        settings.update(telemetry=True)
        complex_transform = A.Compose([
            A.Compose([  # Nested compose
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ]),
            A.OneOf([
                A.Blur(blur_limit=3),
                A.MedianBlur(blur_limit=3),
            ], p=0.5),
            A.SomeOf([
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.5),
            ], n=1, p=0.5),
        ],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
        keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_labels']),
        )

        # Test with all targets
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        mask = np.zeros((512, 512), dtype=np.uint8)
        bboxes = [[10, 10, 50, 50]]
        labels = [1]
        keypoints = [[25, 25]]
        keypoint_labels = [0]

        result = complex_transform(
            image=image,
            mask=mask,
            bboxes=bboxes,
            labels=labels,
            keypoints=keypoints,
            keypoint_labels=keypoint_labels,
        )

        # Check all outputs are present
        assert 'image' in result
        assert 'mask' in result
        assert 'bboxes' in result
        assert 'labels' in result
        assert 'keypoints' in result
        assert 'keypoint_labels' in result

    def test_lambda_transform_excluded(self):
        """Test that Lambda transforms are excluded from telemetry."""
        from albumentations.augmentations.other.lambda_transform import Lambda

        # Create pipeline with Lambda
        transform = A.Compose([
            A.RandomCrop(256, 256),
            Lambda(name="custom_lambda", image=lambda x, **kwargs: x),
            A.HorizontalFlip(p=0.5),
        ])

        # Collect pipeline info
        info = collect_pipeline_info(transform)

        # Lambda should not be in transform names
        assert 'Lambda' not in info['transforms']
        assert 'RandomCrop' in info['transforms']
        assert 'HorizontalFlip' in info['transforms']

    def test_normalize_totensor_included(self):
        """Test that Normalize and ToTensorV2 are included with Mixpanel."""
        # Create event with Normalize and ToTensorV2
        event = ComposeInitEvent(
            transforms=["RandomCrop", "Normalize", "HorizontalFlip", "ToTensorV2", "Blur"],
            albumentationsx_version="2.0.0",
            python_version="3.10",
            os="Linux",
            cpu="Intel",
            environment="local",
        )

        # Check they're in the full transform list
        assert "Normalize" in event.transforms
        assert "ToTensorV2" in event.transforms

        # With Mixpanel, all transforms are included
        data = event.to_dict()
        pipeline_transforms = data["pipeline"]["transforms"]
        assert len(pipeline_transforms) == 5
        assert "RandomCrop" in pipeline_transforms
        assert "Normalize" in pipeline_transforms
        assert "HorizontalFlip" in pipeline_transforms
        assert "ToTensorV2" in pipeline_transforms
        assert "Blur" in pipeline_transforms

    def test_serialization_with_telemetry(self):
        """Test that telemetry doesn't interfere with serialization."""
        # Create a pipeline
        transform = A.Compose([
            A.RandomCrop(256, 256),
            A.HorizontalFlip(p=0.5),
        ], seed=137)

        # Serialize
        serialized = transform.to_dict()

        # Deserialize
        deserialized = A.from_dict(serialized)

        # Test both produce same results
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

        # Reset seed for both
        transform.set_random_seed(137)
        deserialized.set_random_seed(137)

        result1 = transform(image=image)
        result2 = deserialized(image=image)

        # Should produce identical results
        np.testing.assert_array_equal(result1['image'], result2['image'])
