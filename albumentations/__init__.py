from albumentations._version import __author__ as __author__
from albumentations._version import __maintainer__ as __maintainer__
from albumentations._version import __version__ as __version__

# OpenCV is an installation extra and Torch is intentionally absent from package
# metadata, so docs and dependency-only consumers do not select a CPU, CUDA, or MPS build.
# The public import graph requires both.
try:
    import cv2  # noqa: F401
except ModuleNotFoundError as e:
    if e.name != "cv2":
        raise
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

try:
    import torch  # noqa: F401
except ModuleNotFoundError as e:
    if e.name != "torch":
        raise
    msg = (
        "AlbumentationsX requires PyTorch when it is imported.\n\n"
        "Install the PyTorch build for your platform first. For Linux CPU-only:\n"
        '  pip install "torch>=2.13.0" --index-url https://download.pytorch.org/whl/cpu\n\n'
        "Then install AlbumentationsX with an OpenCV extra:\n"
        '  pip install "albumentationsx[headless]"\n\n'
        "Use PyTorch's platform-specific command for CUDA or MPS, and replace "
        "headless with gui, contrib, or contrib-headless if needed."
    )
    raise ImportError(msg) from e

from .augmentations import *
from .core.composition import *
from .core.serialization import *
from .core.tracing import *
from .core.transforms_interface import *
from .pytorch import *
