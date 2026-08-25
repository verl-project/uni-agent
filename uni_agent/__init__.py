"""Build, run, and train long-horizon agents at scale."""

import os

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

with open(os.path.join(version_folder, "version/version")) as f:
    __version__ = f.read().strip()

__all__ = ["__version__"]
