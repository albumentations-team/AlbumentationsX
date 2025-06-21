"""Tests for the version checking functionality."""

import json
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
import socket
import urllib.error
import warnings
import sys

import pytest

from albumentations.check_version import (
    CACHE_HOURS,
    ENV_NO_UPDATE,
    ENV_OFFLINE,
    check_connectivity,
    check_for_updates,
    fetch_pypi_version,
    get_latest_version,
    parse_version,
    read_cache,
    write_cache,
)


@pytest.fixture
def clean_environment(monkeypatch):
    """Clean environment variables and clear caches."""
    monkeypatch.delenv(ENV_NO_UPDATE, raising=False)
    monkeypatch.delenv(ENV_OFFLINE, raising=False)

    # Clear LRU cache
    check_connectivity.cache_clear()

    yield

    # Clean up after test
    check_connectivity.cache_clear()


@pytest.fixture
def temp_cache_file(tmp_path, monkeypatch):
    """Use temporary cache file for testing."""
    cache_file = tmp_path / "version_cache.json"
    monkeypatch.setattr("albumentations.check_version.CACHE_FILE", cache_file)
    return cache_file


class TestParseVersion:
    """Test version parsing functionality."""

    def test_parse_standard_version(self):
        """Test parsing of standard semantic versions."""
        assert parse_version("1.4.24") == (1, 4, 24, 0, 0)
        assert parse_version("0.0.1") == (0, 0, 1, 0, 0)
        assert parse_version("10.20.30") == (10, 20, 30, 0, 0)

    def test_parse_prerelease_versions(self):
        """Test parsing of pre-release versions."""
        assert parse_version("1.4.0-alpha.1") == (1, 4, 0, -3, 1)
        assert parse_version("1.4.0-beta.2") == (1, 4, 0, -2, 2)
        assert parse_version("1.4.0-rc.3") == (1, 4, 0, -1, 3)
        assert parse_version("1.4.0-beta") == (1, 4, 0, -2, 0)

    def test_parse_with_metadata(self):
        """Test parsing versions with build metadata."""
        assert parse_version("1.4.0+build.123") == (1, 4, 0, 0, 0)
        assert parse_version("1.4.0-beta.1+build.456") == (1, 4, 0, -2, 1)

    def test_parse_invalid_versions(self):
        """Test parsing of invalid version strings."""
        assert parse_version("1.4") is None
        assert parse_version("not-a-version") is None
        assert parse_version("") is None
        assert parse_version("1.2.3.4") is None

    def test_version_comparison(self):
        """Test that parsed versions compare correctly."""
        # Standard versions
        assert parse_version("1.4.24") < parse_version("1.4.25")
        assert parse_version("1.4.0") < parse_version("1.5.0")
        assert parse_version("1.0.0") < parse_version("2.0.0")

        # Pre-release ordering
        assert parse_version("1.4.0-alpha.1") < parse_version("1.4.0-beta.1")
        assert parse_version("1.4.0-beta.1") < parse_version("1.4.0-rc.1")
        assert parse_version("1.4.0-rc.1") < parse_version("1.4.0")

        # Within same pre-release type
        assert parse_version("1.4.0-beta.1") < parse_version("1.4.0-beta.2")


class TestConnectivity:
    """Test connectivity checking functionality."""

    def test_check_connectivity_success(self, clean_environment):
        """Test successful connectivity check."""
        with patch("socket.socket") as mock_socket:
            mock_sock = MagicMock()
            # Set up context manager behavior
            mock_socket.return_value.__enter__.return_value = mock_sock
            mock_socket.return_value.__exit__.return_value = None

            assert check_connectivity() is True
            mock_sock.connect.assert_called_once()
            # Should not call close explicitly since context manager handles it
            mock_sock.close.assert_not_called()

    def test_check_connectivity_partial_dns_failure(self, clean_environment):
        """Test that connectivity check tries multiple DNS servers."""
        call_count = 0

        def socket_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            mock_sock = MagicMock()
            if call_count == 1:
                # First DNS server fails
                mock_sock.__enter__.return_value.connect.side_effect = socket.error("Connection failed")
            else:
                # Second DNS server succeeds
                mock_sock.__enter__.return_value = mock_sock

            mock_sock.__exit__.return_value = None
            return mock_sock

        with patch("socket.socket", side_effect=socket_side_effect):
            assert check_connectivity() is True
            # Should have tried two sockets (first failed, second succeeded)
            assert call_count == 2

    def test_check_connectivity_dns_failure_http_success(self, clean_environment):
        """Test fallback to HTTP when DNS fails."""
        with patch("socket.socket") as mock_socket:
            mock_socket.side_effect = socket.error("Connection failed")

            with patch("urllib.request.build_opener") as mock_opener_builder:
                mock_opener = MagicMock()
                mock_opener_builder.return_value = mock_opener

                assert check_connectivity() is True
                mock_opener.open.assert_called_once()

    def test_check_connectivity_all_failures(self, clean_environment):
        """Test when all connectivity checks fail."""
        with patch("socket.socket") as mock_socket:
            mock_socket.side_effect = socket.error("Connection failed")

            with patch("urllib.request.build_opener") as mock_opener_builder:
                mock_opener = MagicMock()
                mock_opener.open.side_effect = urllib.error.URLError("Failed")
                mock_opener_builder.return_value = mock_opener

                assert check_connectivity() is False

    def test_check_connectivity_offline_env(self, clean_environment, monkeypatch):
        """Test connectivity check with offline environment variable."""
        monkeypatch.setenv(ENV_OFFLINE, "1")
        assert check_connectivity() is False

        # Clear cache and test with "true"
        check_connectivity.cache_clear()
        monkeypatch.setenv(ENV_OFFLINE, "true")
        assert check_connectivity() is False

    def test_check_connectivity_cache(self, clean_environment):
        """Test that connectivity check is cached."""
        with patch("socket.socket") as mock_socket:
            mock_sock = MagicMock()
            # Set up context manager behavior
            mock_socket.return_value.__enter__.return_value = mock_sock
            mock_socket.return_value.__exit__.return_value = None

            # First call
            assert check_connectivity() is True
            assert mock_socket.call_count == 1

            # Second call should use cache
            assert check_connectivity() is True
            assert mock_socket.call_count == 1  # No additional calls


class TestCacheOperations:
    """Test cache read/write operations."""

    def test_write_and_read_cache(self, temp_cache_file):
        """Test writing and reading from cache."""
        write_cache("1.4.25")
        assert read_cache() == "1.4.25"

    def test_read_nonexistent_cache(self, temp_cache_file):
        """Test reading when cache doesn't exist."""
        assert read_cache() is None

    def test_read_expired_cache(self, temp_cache_file):
        """Test reading expired cache."""
        # Write old cache
        old_time = datetime.now(timezone.utc) - timedelta(hours=CACHE_HOURS + 1)
        with temp_cache_file.open("w") as f:
            json.dump({
                "version": "1.4.25",
                "timestamp": old_time.isoformat()
            }, f)

        assert read_cache() is None

    def test_read_malformed_cache(self, temp_cache_file):
        """Test reading malformed cache."""
        with temp_cache_file.open("w") as f:
            f.write("not json")

        assert read_cache() is None

    @pytest.mark.skipif(sys.platform == "win32", reason="File permission test not applicable on Windows")
    def test_cache_file_permissions_error(self, temp_cache_file, monkeypatch):
        """Test handling of permission errors."""
        # First, ensure the cache file doesn't exist
        assert not temp_cache_file.exists()

        # Make parent directory read-only
        temp_cache_file.parent.chmod(0o444)

        try:
            # write_cache should not raise, just fail silently
            write_cache("1.4.25")
            # No assertion needed - if no exception is raised, the test passes
        finally:
            # Restore permissions for cleanup
            temp_cache_file.parent.chmod(0o755)

        # After restoring permissions, verify file was not created
        assert not temp_cache_file.exists(), "Cache file should not have been created when parent dir was read-only"


class TestPyPIFetching:
    """Test PyPI version fetching."""

    def test_fetch_pypi_version_success(self, clean_environment):
        """Test successful version fetch from PyPI."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "info": {"version": "1.4.25"}
        }).encode()

        with patch("albumentations.check_version.check_connectivity", return_value=True):
            with patch("urllib.request.build_opener") as mock_opener_builder:
                mock_opener = MagicMock()
                mock_opener.open.return_value.__enter__.return_value = mock_response
                mock_opener_builder.return_value = mock_opener

                assert fetch_pypi_version() == "1.4.25"

    def test_fetch_pypi_version_offline(self, clean_environment):
        """Test PyPI fetch when offline."""
        with patch("albumentations.check_version.check_connectivity", return_value=False):
            assert fetch_pypi_version() is None

    def test_fetch_pypi_version_network_error(self, clean_environment):
        """Test PyPI fetch with network error."""
        with patch("albumentations.check_version.check_connectivity", return_value=True):
            with patch("urllib.request.build_opener") as mock_opener_builder:
                mock_opener = MagicMock()
                mock_opener.open.side_effect = urllib.error.URLError("Network error")
                mock_opener_builder.return_value = mock_opener

                assert fetch_pypi_version() is None

    def test_fetch_pypi_version_malformed_response(self, clean_environment):
        """Test PyPI fetch with malformed response."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"not json"

        with patch("albumentations.check_version.check_connectivity", return_value=True):
            with patch("urllib.request.build_opener") as mock_opener_builder:
                mock_opener = MagicMock()
                mock_opener.open.return_value.__enter__.return_value = mock_response
                mock_opener_builder.return_value = mock_opener

                assert fetch_pypi_version() is None

    def test_fetch_pypi_version_timeout(self, clean_environment):
        """Test PyPI fetch with network timeout."""
        with patch("albumentations.check_version.check_connectivity", return_value=True):
            with patch("urllib.request.build_opener") as mock_opener_builder:
                mock_opener = MagicMock()
                mock_opener.open.side_effect = socket.timeout("Network timeout")
                mock_opener_builder.return_value = mock_opener

                assert fetch_pypi_version() is None

    def test_fetch_pypi_version_missing_info(self, clean_environment):
        """Test PyPI fetch with valid JSON missing 'info' key."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"releases": {}}).encode()

        with patch("albumentations.check_version.check_connectivity", return_value=True):
            with patch("urllib.request.build_opener") as mock_opener_builder:
                mock_opener = MagicMock()
                mock_opener.open.return_value.__enter__.return_value = mock_response
                mock_opener_builder.return_value = mock_opener

                assert fetch_pypi_version() is None

    def test_fetch_pypi_version_missing_version(self, clean_environment):
        """Test PyPI fetch with valid JSON where 'info' exists but 'version' is missing."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"info": {"name": "albumentationsx"}}).encode()

        with patch("albumentations.check_version.check_connectivity", return_value=True):
            with patch("urllib.request.build_opener") as mock_opener_builder:
                mock_opener = MagicMock()
                mock_opener.open.return_value.__enter__.return_value = mock_response
                mock_opener_builder.return_value = mock_opener

                assert fetch_pypi_version() is None


class TestGetLatestVersion:
    """Test get_latest_version functionality."""

    def test_get_latest_version_disabled(self, clean_environment, temp_cache_file, monkeypatch):
        """Test when update check is disabled."""
        monkeypatch.setenv(ENV_NO_UPDATE, "1")
        assert get_latest_version() is None

        # Clear cache and test with "true"
        monkeypatch.setenv(ENV_NO_UPDATE, "true")
        assert get_latest_version() is None

        # Test that other values don't disable
        monkeypatch.setenv(ENV_NO_UPDATE, "0")
        with patch("albumentations.check_version.fetch_pypi_version", return_value="1.4.25"):
            assert get_latest_version() == "1.4.25"

    def test_get_latest_version_from_cache(self, clean_environment, temp_cache_file):
        """Test getting version from cache."""
        write_cache("1.4.25")
        assert get_latest_version() == "1.4.25"

    def test_get_latest_version_fetch_and_cache(self, clean_environment, temp_cache_file):
        """Test fetching and caching latest version."""
        with patch("albumentations.check_version.fetch_pypi_version", return_value="1.4.26"):
            assert get_latest_version() == "1.4.26"
            # Verify it was cached
            assert read_cache() == "1.4.26"

    def test_get_latest_version_fetch_failure(self, clean_environment, temp_cache_file):
        """Test when PyPI fetch fails."""
        with patch("albumentations.check_version.fetch_pypi_version", return_value=None):
            assert get_latest_version() is None

    def test_get_latest_version_corrupted_cache(self, clean_environment, temp_cache_file):
        """Test that corrupted cache files are handled gracefully."""
        # Write corrupted cache
        with temp_cache_file.open("w") as f:
            f.write("{ invalid json }")

        # Should fall back to fetching from PyPI
        with patch("albumentations.check_version.fetch_pypi_version", return_value="1.4.27"):
            assert get_latest_version() == "1.4.27"
            # Verify it was re-cached correctly
            assert read_cache() == "1.4.27"


class TestCheckForUpdates:
    """Test the main check_for_updates function."""

    def test_check_for_updates_newer_available(self, clean_environment, capsys):
        """Test when a newer version is available."""
        with patch("albumentations.check_version.get_latest_version", return_value="1.5.0"):
            with patch("albumentations.check_version.current_version", "1.4.0"):
                with pytest.warns(UserWarning) as warning_list:
                    update_available, latest = check_for_updates(verbose=True)

                assert update_available is True
                assert latest == "1.5.0"

                # Check warning content
                assert len(warning_list) == 1
                warning_msg = str(warning_list[0].message)
                assert "A new version of AlbumentationsX (1.5.0) is available!" in warning_msg
                assert "Your version is 1.4.0" in warning_msg
                assert "pip install -U albumentationsx" in warning_msg

    def test_check_for_updates_up_to_date(self, clean_environment, capsys):
        """Test when current version is up to date."""
        with patch("albumentations.check_version.get_latest_version", return_value="1.4.0"):
            with patch("albumentations.check_version.current_version", "1.4.0"):
                update_available, latest = check_for_updates(verbose=True)

                assert update_available is False
                assert latest == "1.4.0"

                captured = capsys.readouterr()
                assert captured.out == ""

    def test_check_for_updates_no_latest(self, clean_environment):
        """Test when latest version can't be determined."""
        with patch("albumentations.check_version.get_latest_version", return_value=None):
            update_available, latest = check_for_updates()

            assert update_available is False
            assert latest is None

    def test_check_for_updates_invalid_versions(self, clean_environment):
        """Test with unparsable version strings."""
        with patch("albumentations.check_version.get_latest_version", return_value="invalid-version"):
            with patch("albumentations.check_version.current_version", "also-invalid"):
                update_available, latest = check_for_updates()

                assert update_available is False
                assert latest == "invalid-version"

    def test_check_for_updates_silent(self, clean_environment, capsys):
        """Test silent mode (verbose=False)."""
        with patch("albumentations.check_version.get_latest_version", return_value="1.5.0"):
            with patch("albumentations.check_version.current_version", "1.4.0"):
                with warnings.catch_warnings(record=True) as warning_list:
                    update_available, latest = check_for_updates(verbose=False)

                assert update_available is True
                assert latest == "1.5.0"

                # Should not emit any warnings in silent mode
                assert len(warning_list) == 0

    def test_check_for_updates_disabled_by_env(self, clean_environment, monkeypatch):
        """Test that NO_ALBUMENTATIONS_UPDATE disables update check."""
        monkeypatch.setenv(ENV_NO_UPDATE, "1")

        # Even with a newer version "available", check should be disabled
        with patch("albumentations.check_version.fetch_pypi_version", return_value="1.5.0"):
            with patch("albumentations.check_version.current_version", "1.4.0"):
                # Should not emit any warnings
                with warnings.catch_warnings(record=True) as warning_list:
                    update_available, latest = check_for_updates(verbose=True)

                assert update_available is False
                assert latest is None
                assert len(warning_list) == 0  # No warnings should be emitted

        # Also test with "true" value
        monkeypatch.setenv(ENV_NO_UPDATE, "true")
        update_available, latest = check_for_updates(verbose=True)
        assert update_available is False
        assert latest is None


class TestIntegration:
    """Integration tests."""

    @pytest.mark.skipif(
        os.environ.get("CI") == "true" or os.environ.get(ENV_OFFLINE) == "1",
        reason="Skip integration tests in CI or offline environments"
    )
    def test_real_pypi_check(self, clean_environment):
        """Test actual PyPI connectivity (integration test)."""
        version = fetch_pypi_version()
        if version is None:
            # This is expected if the package doesn't exist on PyPI yet
            # or if there's a network issue - both are valid scenarios
            pytest.skip("Package not available on PyPI or network issue")
        else:
            # If we got a version, it should be valid semantic version
            parsed = parse_version(version)
            assert parsed is not None, f"Version '{version}' is not a valid semantic version"


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    def test_concurrent_cache_access(self, temp_cache_file):
        """Test concurrent cache access."""
        import threading

        results = []

        def write_version(version):
            write_cache(version)
            results.append(read_cache())

        threads = [
            threading.Thread(target=write_version, args=(f"1.4.{i}",))
            for i in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have handled concurrent access without crashing
        assert len(results) == 10
        # At least some reads should succeed (concurrent file I/O isn't guaranteed)
        assert any(r is not None for r in results)

        # Verify final cache state is valid
        final_version = read_cache()
        assert final_version is not None, "Cache should be readable after concurrent access"

        # Verify it's one of the versions we wrote (not corrupted)
        expected_versions = {f"1.4.{i}" for i in range(10)}
        assert final_version in expected_versions, f"Final version {final_version} should be one of the written versions"

        # Verify the cache file can be parsed (not corrupted)
        assert parse_version(final_version) is not None, "Final cache value should be a valid version"

    def test_concurrent_cache_corruption(self, temp_cache_file):
        """Test handling of cache corruption during concurrent access."""
        import threading
        import time

        results = []
        corruption_lock = threading.Lock()
        corruption_occurred = False
        successful_writes_before_corruption = 0

        def write_and_corrupt(version, should_corrupt=False):
            nonlocal corruption_occurred, successful_writes_before_corruption
            try:
                with corruption_lock:
                    if should_corrupt and not corruption_occurred:
                        # Ensure at least some writes complete before corruption
                        if successful_writes_before_corruption >= 2:
                            # Simulate corruption by writing invalid data
                            with temp_cache_file.open("w") as f:
                                f.write("{ corrupted during write")
                            corruption_occurred = True
                            results.append("corrupted")
                            return

                    if not corruption_occurred:
                        write_cache(version)
                        # Verify the write succeeded
                        if read_cache() == version:
                            successful_writes_before_corruption += 1
                            results.append(version)
                            return

                # Try to read after write/corruption
                result = read_cache()
                results.append(result)
            except Exception as e:
                results.append(f"error: {type(e).__name__}")

        # Mix normal writes with corruption attempts
        threads = []
        for i in range(10):
            # Every 3rd thread tries to corrupt (after some successes)
            should_corrupt = (i % 3 == 0)
            t = threading.Thread(
                target=write_and_corrupt,
                args=(f"1.4.{i}", should_corrupt)
            )
            threads.append(t)

        # Start threads with small delays to increase chance of some completing before corruption
        for i, t in enumerate(threads):
            t.start()
            if i < 3:  # Give first few threads a head start
                time.sleep(0.001)

        # Wait for completion
        for t in threads:
            t.join()

        # Verify behavior
        assert len(results) == 10, "All threads should complete"

        # Check that at least some operations succeeded before corruption
        valid_results = [r for r in results if r and isinstance(r, str) and r.startswith("1.4.")]
        assert len(valid_results) > 0, "Some cache operations should succeed before corruption"

        # Verify corruption was attempted
        assert "corrupted" in results or corruption_occurred, "Corruption should have been attempted"

        # Final cache state should be either valid or non-existent
        final_state = read_cache()
        if final_state is not None:
            # If there's a final state, it should be parseable (not corrupted)
            assert parse_version(final_state) is not None, "Final cache should be valid if it exists"
