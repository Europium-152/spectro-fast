"""spectrofast — Fast FFTW-based spectrogram computation."""

from spectrofast._spectrogram import (
    get_number_of_windows,
    many_spectrograms,
    many_spectrograms_padded,
    spectrogram,
)

__version__ = "0.1.0"

__all__ = [
    "get_number_of_windows",
    "many_spectrograms",
    "many_spectrograms_padded",
    "spectrogram",
]
