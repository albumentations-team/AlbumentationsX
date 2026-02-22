from albumentations._version import __author__ as __author__
from albumentations._version import __maintainer__ as __maintainer__
from albumentations._version import __version__ as __version__

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
