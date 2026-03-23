"""spectrofast — Fast FFTW-based spectrogram computation."""

from spectrofast._spectrogram import (
    spectrogram,
    get_number_of_windows,
    ESTIMATE,
    MEASURE,
    PATIENT,
    EXHAUSTIVE,
)

__version__ = "0.3.0"

__all__ = [
    "spectrogram",
    "get_number_of_windows",
    "ESTIMATE",
    "MEASURE",
    "PATIENT",
    "EXHAUSTIVE",
]
