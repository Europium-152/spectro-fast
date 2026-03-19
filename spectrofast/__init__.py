"""spectrofast — Fast FFTW-based spectrogram computation."""

from spectrofast._spectrogram import (
    get_number_of_windows,
    many_real_spectrograms,
    many_complex_spectrograms,
    complex_spectrogram,
    real_spectrogram,
)

__version__ = "0.2.0"

__all__ = [
    "get_number_of_windows",
    "many_real_spectrograms",
    "many_complex_spectrograms",
    "complex_spectrogram",
    "real_spectrogram",
]
