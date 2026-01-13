from importlib.metadata import metadata

try:
    _metadata = metadata("albumentationsx")
    __version__ = _metadata["Version"]
    __author__ = _metadata["Author"]
    __maintainer__ = _metadata["Maintainer"]
except Exception:  # noqa: BLE001
    __version__ = "unknown"
    __author__ = "Vladimir Iglovikov"
    __maintainer__ = "Vladimir Iglovikov"

# Check for OpenCV at import time
try:
    import cv2  # noqa: F401
except ImportError as e:
    msg = (
        "AlbumentationsX requires OpenCV but it's not installed.\n\n"
        "Install one of the following:\n"
        "  pip install opencv-python                 # Full version with GUI (cv2.imshow)\n"
        "  pip install opencv-python-headless        # Headless for servers/docker\n"
        "  pip install opencv-contrib-python         # With extra algorithms\n"
        "  pip install opencv-contrib-python-headless # Contrib + headless\n\n"
        "Or use extras:\n"
        "  pip install albumentationsx[headless]     # Installs opencv-python-headless\n"
        "  pip install albumentationsx[gui]          # Installs opencv-python\n"
        "  pip install albumentationsx[contrib]      # Installs opencv-contrib-python"
    )
    raise ImportError(msg) from e

from contextlib import suppress

from .augmentations import *
from .core.composition import *
from .core.serialization import *
from .core.transforms_interface import *

with suppress(ImportError):
    from .pytorch import *
