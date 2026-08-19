"""Automated and modular EEG preprocessing."""

from importlib.metadata import PackageNotFoundError, version

from sEEGnal.pipeline import run_sEEGnal


try:
    __version__ = version("sEEGnal")
except PackageNotFoundError:  # pragma: no cover - source tree without install
    __version__ = "0+unknown"


__all__ = ["__version__", "run_sEEGnal"]
