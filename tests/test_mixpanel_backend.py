"""Tests for Mixpanel backend."""

import json
from unittest.mock import Mock, patch

import pytest

from albumentations.core.analytics.backends.mixpanel import MixpanelBackend
from albumentations.core.analytics.events import ComposeInitEvent


class TestMixpanelBackend:
    """Test the Mixpanel analytics backend."""

    def test_backend_initialization(self):
        """Test MixpanelBackend initialization."""
        backend = MixpanelBackend()
        assert backend.MIXPANEL_URL == "https://api.mixpanel.com/track"
        assert backend.PROJECT_TOKEN == "9674977e5658e19ce4710845fdd68712"

    def test_parse_timestamp(self):
        """Test timestamp parsing."""
        backend = MixpanelBackend()

        # Test ISO string
        timestamp = backend._parse_timestamp("2024-01-15T10:30:00.000Z")
        assert isinstance(timestamp, int)
        assert timestamp > 0

        # Test None
        assert backend._parse_timestamp(None) is None

        # Test invalid string
        assert backend._parse_timestamp("invalid") is None

    @patch('urllib.request.urlopen')
    def test_send_event_success(self, mock_urlopen):
        """Test successful event sending."""
        # Mock successful response
        mock_response = Mock()
        mock_response.read.return_value = b"1"
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=None)
        mock_urlopen.return_value = mock_response

        backend = MixpanelBackend()
        event = ComposeInitEvent(
            user_id="test-user-123",
            pipeline_hash="test_hash",
            albumentationsx_version="2.0.0",
            python_version="3.10",
            os="Ubuntu 22.04",
            cpu="Intel i7",
            gpu="NVIDIA RTX 3080",
            ram_gb=16.0,
            environment="local",
            transforms=["RandomCrop", "HorizontalFlip", "Normalize"],
            targets="bboxes",
        )

        # Should not raise
        backend.send_event(event)

        # Check request was made
        mock_urlopen.assert_called_once()

        # Check request details
        request = mock_urlopen.call_args[0][0]
        assert request.get_full_url() == backend.MIXPANEL_URL
        assert request.get_header("Content-type") == "application/x-www-form-urlencoded"

        # Check data format
        request_data = request.data.decode('utf-8')
        assert request_data.startswith("data=")

    def test_event_data_format(self):
        """Test that event data is formatted correctly for Mixpanel."""
        backend = MixpanelBackend()
        event = ComposeInitEvent(
            user_id="test-user-123",
            session_id="session-456",
            pipeline_hash="test_hash",
            albumentationsx_version="2.0.0",
            python_version="3.10",
            os="Ubuntu 22.04",
            cpu="Intel i7",
            gpu="NVIDIA RTX 3080",
            ram_gb=16.0,
            environment="jupyter",
            transforms=["RandomCrop", "HorizontalFlip", "Normalize", "ToTensorV2"],
            targets="bboxes_keypoints",
        )

        # Mock the send to capture the data
        sent_data = None

        def mock_urlopen(request, timeout=None):
            nonlocal sent_data
            # Decode the base64 data
            import base64
            data_param = request.data.decode('utf-8').replace('data=', '')
            sent_data = json.loads(base64.b64decode(data_param).decode('utf-8'))

            mock_response = Mock()
            mock_response.read.return_value = b"1"
            mock_response.__enter__ = Mock(return_value=mock_response)
            mock_response.__exit__ = Mock(return_value=None)
            return mock_response

        with patch('urllib.request.urlopen', side_effect=mock_urlopen):
            backend.send_event(event)

        # Check the data structure
        assert sent_data is not None
        assert sent_data["event"] == "Compose Init"

        props = sent_data["properties"]
        assert props["distinct_id"] == "test-user-123"
        assert props["token"] == backend.PROJECT_TOKEN
        assert props["$insert_id"] == "session-456"  # For deduplication

        # Check all fields are included
        assert props["pipeline_hash"] == "test_hash"
        assert props["version"] == "2.0.0"
        assert props["python_version"] == "3.10"
        assert props["cpu"] == "Intel i7"
        assert props["gpu"] == "NVIDIA RTX 3080"
        assert props["ram_gb"] == 16.0
        assert props["environment"] == "jupyter"
        assert props["targets"] == "bboxes_keypoints"
        assert props["num_transforms"] == 4

        # All transforms should be included (no exclusions!)
        assert props["transforms"] == ["RandomCrop", "HorizontalFlip", "Normalize", "ToTensorV2"]

        # Check Mixpanel automatic properties
        assert props["$os"] == "Ubuntu 22.04"
        assert props["$lib"] == "albumentationsx"
        assert props["$lib_version"] == "2.0.0"

    @patch('urllib.request.urlopen')
    def test_send_event_failure_handled(self, mock_urlopen):
        """Test that failures are handled gracefully."""
        # Mock failed response with OSError (which is caught)
        mock_urlopen.side_effect = OSError("Network error")

        backend = MixpanelBackend()
        event = ComposeInitEvent(
            transforms=["RandomCrop"],
            albumentationsx_version="2.0.0",
            python_version="3.10",
            os="Linux",
            cpu="Intel",
            environment="local",
        )

        # Should not raise - errors are silently handled
        backend.send_event(event)

    def test_none_values_removed(self):
        """Test that None values are removed from properties."""
        backend = MixpanelBackend()
        event = ComposeInitEvent(
            albumentationsx_version="2.0.0",
            python_version="3.10",
            os="Linux",
            cpu="Intel",
            environment="local",
            transforms=[],
            gpu=None,  # None value
            ram_gb=None,  # None value
        )

        sent_data = None

        def mock_urlopen(request, timeout=None):
            nonlocal sent_data
            import base64
            data_param = request.data.decode('utf-8').replace('data=', '')
            sent_data = json.loads(base64.b64decode(data_param).decode('utf-8'))

            mock_response = Mock()
            mock_response.read.return_value = b"1"
            mock_response.__enter__ = Mock(return_value=mock_response)
            mock_response.__exit__ = Mock(return_value=None)
            return mock_response

        with patch('urllib.request.urlopen', side_effect=mock_urlopen):
            backend.send_event(event)

        # Check None values are not in properties
        props = sent_data["properties"]
        assert "gpu" not in props
        assert "ram_gb" not in props
