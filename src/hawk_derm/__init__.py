"""Reproducible 25-condition teledermatology experiments."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hawk-derm")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]
