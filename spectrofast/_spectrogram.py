import os
import sys
import numpy as np

try:
    import numba

    @numba.vectorize([numba.float64(numba.complex128)], nopython=True)
    def _abs2(x):
        return x.real ** 2 + x.imag ** 2

except ImportError:
    print("spectrofast: numba not found. Install it for ~3x faster magnitude computation: pip install numba")

    def _abs2(x):
        return x.real ** 2 + x.imag ** 2

# On Windows, add the vendor directory to the DLL search path
# so that libfftw3-3.dll can be found at runtime
if sys.platform == "win32":
    vendor_dir = os.path.join(os.path.dirname(__file__), "_vendor_win")
    if os.path.isdir(vendor_dir):
        os.add_dll_directory(vendor_dir)

from spectrofast._real_spectrogram_cffi import ffi, lib

# Planner flag constants matching the C get_fftw_flag() mapping
ESTIMATE = 0
MEASURE = 1
PATIENT = 2
EXHAUSTIVE = 3

_PLANNER_FLAGS = {
    "estimate": ESTIMATE,
    "measure": MEASURE,
    "patient": PATIENT,
    "exhaustive": EXHAUSTIVE,
}

_VALID_MODES = ('psd', 'complex', 'magnitude', 'angle', 'phase')
_VALID_SCALINGS = ('density', 'spectrum')


def _resolve_planner_flag(planner):
    """Convert a planner argument (str or int) to the integer flag."""
    if isinstance(planner, int):
        if planner not in (ESTIMATE, MEASURE, PATIENT, EXHAUSTIVE):
            raise ValueError(
                f"Invalid planner flag {planner}. "
                f"Use spectrofast.ESTIMATE (0), MEASURE (1), PATIENT (2), or EXHAUSTIVE (3)."
            )
        return planner
    if isinstance(planner, str):
        key = planner.lower()
        if key not in _PLANNER_FLAGS:
            raise ValueError(
                f"Invalid planner '{planner}'. "
                f"Choose from: 'estimate', 'measure', 'patient', 'exhaustive'."
            )
        return _PLANNER_FLAGS[key]
    raise TypeError(f"planner must be a str or int, got {type(planner).__name__}")


def _resolve_window(window, nperseg):
    """Resolve the window parameter into a numpy array or None.

    Parameters
    ----------
    window : None, str, tuple, float, or array_like
        - None: no windowing (returns None)
        - str or tuple: passed to scipy.signal.get_window
        - float: interpreted as Kaiser beta, passed to get_window
        - array_like: used directly, must have length nperseg

    Returns
    -------
    win : np.ndarray or None
        Window array of length nperseg (float64), or None for no windowing.
    """
    if window is None:
        return None

    if isinstance(window, str) and window.lower() == 'boxcar':
        return None

    if isinstance(window, (str, tuple, float)):
        try:
            from scipy.signal import get_window
        except ImportError:
            raise ImportError(
                "scipy is required for string/tuple window specifications. "
                "Install it with: pip install scipy\n"
                "Alternatively, pass a numpy array as the window parameter."
            )
        win = get_window(window, nperseg, fftbins=True)
        return np.ascontiguousarray(win, dtype=np.float64)

    # array_like
    win = np.ascontiguousarray(window, dtype=np.float64)
    if win.ndim != 1 or len(win) != nperseg:
        raise ValueError(
            f"Window array must be 1D with length nperseg ({nperseg}), "
            f"got shape {win.shape}."
        )
    return win


def _apply_detrend(segments, detrend):
    """Apply detrending to segments array.

    Parameters
    ----------
    segments : np.ndarray, shape (total_segments, nperseg)
        The extracted signal segments.
    detrend : str, callable, or False
        - 'constant': subtract the mean of each segment.
        - 'linear': subtract a linear least-squares fit (requires scipy).
        - callable: applied to each segment individually.
        - False: no detrending (should not reach here).

    Returns
    -------
    segments : np.ndarray, same shape as input
    """
    if detrend == 'constant':
        segments = segments - segments.mean(axis=-1, keepdims=True)
    elif detrend == 'linear':
        try:
            from scipy.signal import detrend as scipy_detrend
        except ImportError:
            raise ImportError(
                "scipy is required for linear detrending. "
                "Install it with: pip install scipy\n"
                "Alternatively, use detrend='constant' or detrend=False."
            )
        segments = scipy_detrend(segments, type='linear', axis=-1)
    elif callable(detrend):
        segments = np.array([detrend(seg) for seg in segments])
    else:
        raise ValueError(
            f"detrend must be False, 'constant', 'linear', or a callable, "
            f"got {detrend!r}."
        )
    return segments


def get_number_of_windows(signal_length, window_size, overlap):
    """Return the number of spectrogram windows for the given parameters."""
    return lib.get_number_of_windows(signal_length, window_size, overlap)


def _many_real(signal, signal_length, number_of_signals,
               window_size, overlap, fft_size, planner_flag, use_wisdom, win_arr):
    """Low-level wrapper for the real-to-complex C function."""
    output_size = fft_size // 2 + 1
    num_windows = get_number_of_windows(signal_length, window_size, overlap)
    spectrogram_size = number_of_signals * num_windows * output_size

    spectrogram = np.empty(spectrogram_size, dtype=np.complex128)

    signal_ptr = ffi.cast("double *", signal.ctypes.data)
    spec_ptr = ffi.cast("double *", spectrogram.ctypes.data)

    if win_arr is not None:
        win_ptr = ffi.cast("double *", win_arr.ctypes.data)
    else:
        win_ptr = ffi.cast("double *", 0)

    ret = lib.many_real_spectrograms(signal_ptr, signal_length,
                                     number_of_signals, window_size,
                                     overlap, fft_size,
                                     planner_flag, int(use_wisdom),
                                     win_ptr, spec_ptr)
    if ret != 0:
        raise RuntimeError(f"many_real_spectrograms failed with error code {ret}")

    return spectrogram


def _many_complex(signal, signal_length, number_of_signals,
                  window_size, overlap, fft_size, planner_flag, use_wisdom, win_arr):
    """Low-level wrapper for the complex-to-complex C function."""
    output_size = fft_size
    num_windows = get_number_of_windows(signal_length, window_size, overlap)
    spectrogram_size = number_of_signals * num_windows * output_size

    spectrogram = np.empty(spectrogram_size, dtype=np.complex128)

    signal_ptr = ffi.cast("double *", signal.ctypes.data)
    spec_ptr = ffi.cast("double *", spectrogram.ctypes.data)

    if win_arr is not None:
        win_ptr = ffi.cast("double *", win_arr.ctypes.data)
    else:
        win_ptr = ffi.cast("double *", 0)

    ret = lib.many_complex_spectrograms(signal_ptr, signal_length,
                                         number_of_signals, window_size,
                                         overlap, fft_size,
                                         planner_flag, int(use_wisdom),
                                         win_ptr, spec_ptr)
    if ret != 0:
        raise RuntimeError(f"many_complex_spectrograms failed with error code {ret}")

    return spectrogram


def spectrogram(x, fs=1.0, window=('tukey', 0.25), nperseg=256,
                noverlap=None, nfft=None, detrend='constant',
                return_onesided=True, scaling='density', axis=-1,
                mode='psd', planner="measure", use_wisdom=True):
    """
    Compute the spectrogram of one or more signals.

    A fast alternative to scipy.signal.spectrogram, powered by FFTW.
    Automatically handles both real and complex input signals.
    Call signature and default values match scipy.signal.spectrogram for drop-in migration.

    Parameters
    ----------
    x : np.ndarray
        Input signal(s). Can be any dimensionality (1D, 2D, N-D).
        The spectrogram is computed along the axis specified by `axis`.
        Supports both real (float64) and complex (complex128) data.
    fs : float, optional
        Sampling frequency of the input signal. Default: 1.0.
        Used for PSD scaling calculations.
    window : None, str, tuple, float, or array_like, optional
        Window function to apply to each segment before computing the FFT.
        Default: ('tukey', 0.25).
        - None or 'boxcar': no windowing (rectangular window, NULL passed to C).
        - str or tuple: passed to scipy.signal.get_window to generate the
          window values (periodic/DFT-even by default).
          Examples: 'hann', 'hamming', ('tukey', 0.25), ('kaiser', 8.0)
        - float: interpreted as Kaiser window shape parameter (beta).
        - array_like: used directly as window values. Must have length nperseg.
    nperseg : int, optional
        Window size (number of samples per segment). Default: 256.
    noverlap : int or None, optional
        Number of overlapping samples between segments. If None, defaults
        to nperseg // 8 (matching scipy's default). Default: None.
    nfft : int, optional
        FFT size. If larger than nperseg, zero-padding is applied.
        Defaults to nperseg (no padding).
    detrend : str, callable, or False, optional
        Specifies how to detrend each segment. Default: 'constant'.
        - 'constant': subtract the mean of each segment.
        - 'linear': subtract a linear least-squares fit (requires scipy).
        - callable: a function that takes a 1D segment and returns a
          detrended segment of the same length.
        - False: no detrending.
        Notes:
            Detrending increases computation time and memory usage.
            Avoid when possible or consider pre-detrending the signal
            before calling spectrogram for better performance.
    return_onesided : bool, optional
        If True (default), return a one-sided spectrum for real data
        (bins 0 to nfft//2, i.e. nfft//2+1 bins). If False, return the
        full two-sided spectrum (nfft bins) by mirroring the positive
        frequencies. For complex data, a two-sided spectrum is always
        returned regardless of this setting.
    scaling : str, optional
        Only used when mode='psd'. One of:
        - 'density' (default): Power spectral density with units V**2/Hz.
          Scale factor: 1 / (fs * sum(win**2)).
        - 'spectrum': Power spectrum with units V**2.
          Scale factor: 1 / sum(win)**2.
    axis : int, optional
        Axis along which the spectrogram is computed. Default: -1 (last axis).
    mode : str, optional
        Output type. One of:
        - 'psd' (default): Power spectral density (or power spectrum,
          depending on `scaling`). Real-valued, non-negative.
        - 'complex': Normalized complex FFT output.
        - 'magnitude': Absolute value of the FFT, |X|.
        - 'angle': Complex angle of the FFT, in radians [-pi, pi].
        - 'phase': Unwrapped phase of the FFT, in radians.
    planner : str or int, optional
        FFTW planner strategy. One of 'estimate', 'measure', 'patient',
        'exhaustive' or the corresponding integer constants (0-3).
        Default: 'measure'.
    use_wisdom : bool, optional
        If True (default), load/save FFTW wisdom from ~/.sfftw/wisdom.
        Set to False for thread-safe operation (skips all file I/O).

    Returns
    -------
    f : np.ndarray
        Array of sample frequencies. For one-sided real input, shape is
        (nfft//2+1,) with values from 0 to fs/2. For two-sided or complex
        input, shape is (nfft,) with values from 0 to fs (wrapping around
        negative frequencies, matching np.fft.fftfreq convention).
    t : np.ndarray
        Array of segment times, shape (num_windows,). Each value is the
        center time of the corresponding segment.
    Sxx : np.ndarray
        Spectrogram of x. The input axis is replaced by the frequency axis
        (output_size), and a new time axis (num_windows) is appended at the end.
        For example, input shape (A, B, C) with axis=1 and real signals produces
        output shape (A, output_size, C, num_windows).
        Where output_size = nfft // 2 + 1 for real one-sided input,
        nfft for complex input or real two-sided (return_onesided=False).
        dtype is float64 for all modes except 'complex' (complex128).
    """
    # Validate mode and scaling
    if mode not in _VALID_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Must be one of {_VALID_MODES}.")
    if scaling not in _VALID_SCALINGS:
        raise ValueError(f"Unknown scaling '{scaling}'. Must be one of {_VALID_SCALINGS}.")

    is_complex = np.issubdtype(x.dtype, np.complexfloating)

    if is_complex:
        x = np.ascontiguousarray(x, dtype=np.complex128)
    else:
        x = np.ascontiguousarray(x, dtype=np.float64)

    if nfft is None:
        nfft = nperseg

    if noverlap is None:
        noverlap = nperseg // 8

    if noverlap >= nperseg:
        raise ValueError(f"noverlap ({noverlap}) must be less than nperseg ({nperseg})")

    if nfft < nperseg:
        raise ValueError(f"nfft ({nfft}) must be >= nperseg ({nperseg})")

    if x.ndim == 0:
        raise ValueError("Input array must have at least 1 dimension.")

    # Normalize axis
    axis_norm = np.lib.array_utils.normalize_axis_index(axis, x.ndim)

    # Move target axis to last position
    x = np.moveaxis(x, axis_norm, -1)
    signal_length = x.shape[-1]

    # Save batch shape (all dims except signal dim)
    batch_shape = x.shape[:-1]  # e.g. (A, C, D) for input (A,B,C,D) axis=1
    number_of_signals = int(np.prod(batch_shape)) if batch_shape else 1

    # Flatten to 2D for C: (number_of_signals, signal_length)
    x = np.ascontiguousarray(x.reshape(max(number_of_signals, 1), signal_length))

    # Resolve window
    win_arr = _resolve_window(window, nperseg)

    planner_flag = _resolve_planner_flag(planner)
    num_windows = get_number_of_windows(signal_length, nperseg, noverlap)

    # ── Detrending ──
    # When detrend is active, we pre-segment in Python, apply detrend to each
    # segment, then pass the detrended segments as a flat batch to C
    # (each segment is a separate "signal" of length nperseg with overlap=0).
    if detrend is not False:
        step = nperseg - noverlap
        # Extract overlapping segments: shape (number_of_signals, num_windows, nperseg)
        segments = np.lib.stride_tricks.sliding_window_view(
            x, nperseg, axis=-1
        )[:, ::step, :]
        # Make a contiguous copy (sliding_window_view returns a view)
        segments = np.ascontiguousarray(segments.reshape(-1, nperseg))
        # Apply detrend to all segments at once
        segments = _apply_detrend(segments, detrend)
        # Ensure correct dtype and contiguity
        if is_complex:
            segments = np.ascontiguousarray(segments, dtype=np.complex128)
        else:
            segments = np.ascontiguousarray(segments, dtype=np.float64)
        # Pass to C: each segment is a "signal" of length nperseg, overlap=0
        c_signal_length = nperseg
        c_number_of_signals = number_of_signals * num_windows
        c_overlap = 0
    else:
        c_signal_length = signal_length
        c_number_of_signals = number_of_signals
        c_overlap = noverlap
        segments = x

    if is_complex:
        output_size = nfft
        result = _many_complex(segments, c_signal_length, c_number_of_signals,
                               nperseg, c_overlap, nfft, planner_flag, use_wisdom, win_arr)
    else:
        output_size = nfft // 2 + 1
        result = _many_real(segments, c_signal_length, c_number_of_signals,
                            nperseg, c_overlap, nfft, planner_flag, use_wisdom, win_arr)

    # ── Post-processing ──
    # FFTW outputs unnormalized transforms, so normalize by 1/nfft
    result = result / nfft

    # For real input with return_onesided=False, reconstruct the full two-sided
    # spectrum from the one-sided output using Hermitian symmetry: X[N-k] = conj(X[k])
    if not is_complex and not return_onesided:
        onesided = result.reshape(-1, output_size)  # (total_spectra, nfft//2+1)
        if nfft % 2 == 0:
            # Even nfft: DC and Nyquist are unpaired, mirror bins 1..nfft//2-1
            twosided = np.concatenate(
                [onesided, np.conj(onesided[:, -2:0:-1])], axis=-1
            )
        else:
            # Odd nfft: only DC is unpaired, mirror bins 1..(nfft-1)//2
            twosided = np.concatenate(
                [onesided, np.conj(onesided[:, -1:0:-1])], axis=-1
            )
        result = twosided.ravel()
        output_size = nfft

    if mode == 'psd':
        # Power spectral density: |X|² with scaling
        result = _abs2(result)

        # Compute window-aware scaling factors
        if win_arr is not None:
            # density: 1 / (fs * sum(win²))
            # spectrum: 1 / sum(win)²
            sum_win_sq = np.sum(win_arr ** 2)
            sum_win = np.sum(win_arr)
        else:
            # No window = rectangular (ones): sum(1²) = nperseg, sum(1) = nperseg
            sum_win_sq = float(nperseg)
            sum_win = float(nperseg)

        if scaling == 'density':
            result *= 1.0 / (fs * sum_win_sq)
        else:  # 'spectrum'
            result *= 1.0 / (sum_win ** 2)

    elif mode == 'complex':
        pass  # return normalized complex FFT output as-is

    elif mode == 'magnitude':
        result = np.abs(result)

    elif mode == 'angle':
        result = np.angle(result)

    elif mode == 'phase':
        result = np.angle(result)

    # ── Reshape (must happen before one-sided doubling and phase unwrap) ──
    # Intermediate layout: (*batch_shape, num_windows, output_size)
    result = result.reshape(*batch_shape, num_windows, output_size)

    # For one-sided (real input) PSD, double non-DC/Nyquist bins
    # to account for the energy in the negative frequencies.
    # Skip when return_onesided=False (two-sided already has both sides).
    if mode == 'psd' and not is_complex and return_onesided:
        if nfft % 2:  # odd nfft
            result[..., 1:] *= 2
        else:  # even nfft — last bin is unpaired Nyquist
            result[..., 1:-1] *= 2

    # Unwrap phase along the time axis (must happen after reshape)
    # In intermediate layout (..., num_windows, output_size), time is axis -2
    if mode == 'phase':
        result = np.unwrap(result, axis=-2)

    # ── Final axis rearrangement ──
    # Current layout: (*batch_shape, num_windows, output_size)
    # Target layout:  freq (output_size) at original axis position, time (num_windows) at the end
    result = np.moveaxis(result, -1, axis_norm)

    # ── Build frequency and time arrays ──
    if is_complex or not return_onesided:
        f = np.fft.fftfreq(nfft, d=1.0 / fs)
    else:
        f = np.fft.rfftfreq(nfft, d=1.0 / fs)

    t = np.arange(num_windows) * (nperseg - noverlap) / fs + (nperseg - 1) / 2.0 / fs

    return f, t, result
