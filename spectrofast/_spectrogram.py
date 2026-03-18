import os
import sys
import numpy as np

# On Windows, add the vendor directory to the DLL search path
# so that libfftw3-3.dll can be found at runtime
if sys.platform == "win32":
    vendor_dir = os.path.join(os.path.dirname(__file__), "_vendor_win")
    if os.path.isdir(vendor_dir):
        os.add_dll_directory(vendor_dir)

from spectrofast._real_spectrogram_cffi import ffi, lib


def get_number_of_windows(signal_length, window_size, overlap):
    """Return the number of spectrogram windows for the given parameters."""
    return lib.get_number_of_windows(signal_length, window_size, overlap)


def many_spectrograms(signal, signal_length, number_of_signals, window_size, overlap):
    """
    Compute spectrograms of multiple signals using batched FFTW.

    Parameters
    ----------
    signal : np.ndarray
        Flattened 1D float64 array containing all signals concatenated.
    signal_length : int
        Length of each individual signal.
    number_of_signals : int
        Number of signals.
    window_size : int
        Window size for the FFT.
    overlap : int
        Overlap between consecutive windows.

    Returns
    -------
    np.ndarray
        Flattened 1D array of magnitude-squared spectrogram values.
    """
    signal = np.ascontiguousarray(signal, dtype=np.float64)
    num_windows = get_number_of_windows(signal_length, window_size, overlap)
    output_size = window_size // 2 + 1
    spectrogram_size = number_of_signals * num_windows * output_size

    spectrogram = np.empty(spectrogram_size, dtype=np.float64)

    signal_ptr = ffi.cast("double *", signal.ctypes.data)
    spec_ptr = ffi.cast("double *", spectrogram.ctypes.data)

    ret = lib.many_real_spectrograms(signal_ptr, signal_length,
                                     number_of_signals, window_size,
                                     overlap, spec_ptr)
    if ret != 0:
        raise RuntimeError(f"many_real_spectrograms failed with error code {ret}")

    return spectrogram


def many_spectrograms_padded(signal, signal_length, number_of_signals,
                              window_size, overlap, fft_size):
    """
    Compute zero-padded spectrograms of multiple signals.

    Parameters
    ----------
    signal : np.ndarray
        Flattened 1D float64 array containing all signals concatenated.
    signal_length : int
        Length of each individual signal.
    number_of_signals : int
        Number of signals.
    window_size : int
        Window size (number of signal samples per window).
    overlap : int
        Overlap between consecutive windows.
    fft_size : int
        FFT size (>= window_size). Zero-padding is applied if fft_size > window_size.

    Returns
    -------
    np.ndarray
        Flattened 1D array of magnitude-squared spectrogram values.
    """
    signal = np.ascontiguousarray(signal, dtype=np.float64)
    num_windows = get_number_of_windows(signal_length, window_size, overlap)
    output_size = fft_size // 2 + 1
    spectrogram_size = number_of_signals * num_windows * output_size

    spectrogram = np.empty(spectrogram_size, dtype=np.float64)

    signal_ptr = ffi.cast("double *", signal.ctypes.data)
    spec_ptr = ffi.cast("double *", spectrogram.ctypes.data)

    ret = lib.many_real_spectrograms_padded(signal_ptr, signal_length,
                                            number_of_signals, window_size,
                                            overlap, fft_size, spec_ptr)
    if ret != 0:
        raise RuntimeError(f"many_real_spectrograms_padded failed with error code {ret}")

    return spectrogram


def spectrogram(x, nperseg, noverlap, nfft=None):
    """
    Compute the spectrogram of one or more signals.

    A simple, fast alternative to scipy.signal.spectrogram, powered by FFTW.

    Parameters
    ----------
    x : np.ndarray
        Input signal(s). 1D array for a single signal, or 2D array
        (num_signals, signal_length) for batch processing.
    nperseg : int
        Window size (number of samples per segment).
    noverlap : int
        Number of overlapping samples between segments.
    nfft : int, optional
        FFT size. If larger than nperseg, zero-padding is applied.
        Defaults to nperseg (no padding).

    Returns
    -------
    np.ndarray
        Magnitude-squared spectrogram.
        Shape (num_windows, output_size) for 1D input, or
        (num_signals, num_windows, output_size) for 2D input.
        Where output_size = nfft // 2 + 1.
    """
    x = np.ascontiguousarray(x, dtype=np.float64)

    if nfft is None:
        nfft = nperseg

    if nfft < nperseg:
        raise ValueError(f"nfft ({nfft}) must be >= nperseg ({nperseg})")

    if x.ndim == 1:
        number_of_signals = 1
        signal_length = x.shape[0]
    elif x.ndim == 2:
        number_of_signals = x.shape[0]
        signal_length = x.shape[1]
        x = x.ravel()
    else:
        raise ValueError("Input array must be 1D or 2D.")

    num_windows = get_number_of_windows(signal_length, nperseg, noverlap)
    output_size = nfft // 2 + 1

    if nfft == nperseg:
        result = many_spectrograms(x, signal_length, number_of_signals,
                                   nperseg, noverlap)
    else:
        result = many_spectrograms_padded(x, signal_length, number_of_signals,
                                          nperseg, noverlap, nfft)

    if number_of_signals == 1:
        return result.reshape(num_windows, output_size)
    else:
        return result.reshape(number_of_signals, num_windows, output_size)
