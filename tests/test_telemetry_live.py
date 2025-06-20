"""Manual test to verify telemetry is working with real GA4 credentials.

This test is not run automatically by pytest. Run it manually to verify
telemetry is being sent to Google Analytics.

Usage:
    python tests/test_telemetry_live.py
"""

import os
import sys
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import albumentations as A
from albumentations.core.settings import settings
from albumentations.core.telemetry import get_telemetry_client


def test_telemetry_live():
    """Test telemetry with real GA4 credentials."""
    print("Testing AlbumentationsX telemetry...")
    print(f"Current settings: telemetry={settings.get('telemetry', True)}")

    # Temporarily enable telemetry for this test
    original_telemetry = settings.get("telemetry", True)
    settings.update(telemetry=True)

    # Get client and enable it (override pytest detection)
    client = get_telemetry_client()
    client.enable()
    print(f"Telemetry client enabled: {client.enabled}")

    try:
        # Create a test pipeline
        print("\nCreating test pipeline...")
        transform = A.Compose([
            A.RandomCrop(256, 256),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Blur(blur_limit=3, p=0.1),
            A.OneOf([
                A.MedianBlur(blur_limit=3),
                A.GaussianBlur(blur_limit=3),
            ], p=0.5),
        ], bbox_params=A.BboxParams(format='pascal_voc'))

        print("Pipeline created successfully!")
        print("\nTelemetry data should have been sent to Google Analytics.")
        print("\nTo verify:")
        print("1. Go to Google Analytics (https://analytics.google.com)")
        print("2. Select the AlbumentationsX property")
        print("3. Go to Reports > Realtime")
        print("4. Look for 'compose_init' events")
        print("\nNote: It may take a few minutes for events to appear in GA4.")

        # Create another different pipeline to test deduplication
        time.sleep(2)  # Small delay
        print("\nCreating second pipeline (different transforms)...")
        transform2 = A.Compose([
            A.Rotate(limit=30, p=0.5),
            A.RandomScale(scale_limit=0.1, p=0.5),
        ])
        print("Second pipeline created!")

        # Try to create duplicate pipeline (should not send telemetry)
        print("\nCreating duplicate of first pipeline (should not send telemetry due to deduplication)...")
        transform3 = A.Compose([
            A.RandomCrop(256, 256),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Blur(blur_limit=3, p=0.1),
            A.OneOf([
                A.MedianBlur(blur_limit=3),
                A.GaussianBlur(blur_limit=3),
            ], p=0.5),
        ], bbox_params=A.BboxParams(format='pascal_voc'))
        print("Duplicate pipeline created (telemetry should have been skipped)")

    finally:
        # Restore original settings
        settings.update(telemetry=original_telemetry)
        print(f"\nRestored telemetry setting to: {original_telemetry}")


if __name__ == "__main__":
    # Make sure we're not in CI
    if any(os.getenv(var) for var in ["CI", "GITHUB_ACTIONS", "GITLAB_CI"]):
        print("Skipping live telemetry test in CI environment")
        sys.exit(0)

    test_telemetry_live()
