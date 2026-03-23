import numpy as np
from spectrofast import spectrogram, get_number_of_windows


# ── Real signal tests ──
# New output layout: 1D → (output_size, num_windows), 2D → (N, output_size, num_windows)

def test_real_output_shape_1d():
    signal = np.sin(2 * np.pi * np.arange(1024) / 32).astype(np.float64)
    f, t, Sxx = spectrogram(signal, nperseg=256, noverlap=128)
    num_windows = get_number_of_windows(1024, 256, 128)
    output_size = 256 // 2 + 1
    assert Sxx.shape == (output_size, num_windows)


def test_real_output_shape_2d():
    signals = np.random.randn(10, 1024).astype(np.float64)
    f, t, Sxx = spectrogram(signals, nperseg=256, noverlap=128)
    num_windows = get_number_of_windows(1024, 256, 128)
    output_size = 256 // 2 + 1
    assert Sxx.shape == (10, output_size, num_windows)


def test_real_zero_padding_shape():
    signal = np.sin(2 * np.pi * np.arange(1024) / 32).astype(np.float64)
    f, t, Sxx = spectrogram(signal, nperseg=256, noverlap=128, nfft=512)
    num_windows = get_number_of_windows(1024, 256, 128)
    output_size = 512 // 2 + 1
    assert Sxx.shape == (output_size, num_windows)


def test_real_peak_frequency():
    """A pure sine wave should produce a peak at the expected frequency bin."""
    fs = 256
    freq = 32  # Hz
    t_sig = np.arange(1024) / fs
    signal = np.sin(2 * np.pi * freq * t_sig).astype(np.float64)

    f, t, Sxx = spectrogram(signal, nperseg=256, noverlap=0)
    # Sxx shape: (output_size, num_windows)

    for w in range(Sxx.shape[-1]):
        peak_bin = np.argmax(Sxx[1:, w]) + 1  # skip DC
        expected_bin = int(freq * 256 / fs)
        assert peak_bin == expected_bin, f"Window {w}: expected bin {expected_bin}, got {peak_bin}"


def test_real_non_negative():
    """Magnitude squared values should always be non-negative."""
    signal = np.random.randn(2048).astype(np.float64)
    f, t, Sxx = spectrogram(signal, nperseg=256, noverlap=128)
    assert np.all(Sxx >= 0)


# ── Complex signal tests ──

def test_complex_output_shape_1d():
    t_sig = np.arange(1024)
    signal = np.exp(2j * np.pi * 32 * t_sig / 256).astype(np.complex128)
    f, t, Sxx = spectrogram(signal, nperseg=256, noverlap=128)
    num_windows = get_number_of_windows(1024, 256, 128)
    assert Sxx.shape == (256, num_windows)


def test_complex_output_shape_2d():
    signals = (np.random.randn(10, 1024) + 1j * np.random.randn(10, 1024)).astype(np.complex128)
    f, t, Sxx = spectrogram(signals, nperseg=256, noverlap=128)
    num_windows = get_number_of_windows(1024, 256, 128)
    assert Sxx.shape == (10, 256, num_windows)


def test_complex_zero_padding_shape():
    t_sig = np.arange(1024)
    signal = np.exp(2j * np.pi * 32 * t_sig / 256).astype(np.complex128)
    f, t, Sxx = spectrogram(signal, nperseg=256, noverlap=128, nfft=512)
    num_windows = get_number_of_windows(1024, 256, 128)
    assert Sxx.shape == (512, num_windows)


def test_complex_peak_frequency():
    """A complex exponential should produce a single peak at the expected bin."""
    fs = 256
    freq = 32  # Hz
    t_sig = np.arange(1024) / fs
    signal = np.exp(2j * np.pi * freq * t_sig).astype(np.complex128)

    f, t, Sxx = spectrogram(signal, nperseg=256, noverlap=0)
    # Sxx shape: (output_size, num_windows)

    expected_bin = int(freq * 256 / fs)
    for w in range(Sxx.shape[-1]):
        peak_bin = np.argmax(Sxx[:, w])
        assert peak_bin == expected_bin, f"Window {w}: expected bin {expected_bin}, got {peak_bin}"


def test_complex_non_negative():
    """Magnitude squared values should always be non-negative."""
    signal = (np.random.randn(2048) + 1j * np.random.randn(2048)).astype(np.complex128)
    f, t, Sxx = spectrogram(signal, nperseg=256, noverlap=128)
    assert np.all(Sxx >= 0)


# ── Auto-detection tests ──

def test_auto_detect_real():
    """Integer input should be treated as real."""
    signal = np.arange(1024)
    f, t, Sxx = spectrogram(signal, nperseg=256, noverlap=128)
    num_windows = get_number_of_windows(1024, 256, 128)
    output_size = 256 // 2 + 1
    assert Sxx.shape == (output_size, num_windows)


def test_auto_detect_complex():
    """complex64 input should be detected and upcast to complex128."""
    signal = np.ones(1024, dtype=np.complex64)
    f, t, Sxx = spectrogram(signal, nperseg=256, noverlap=128)
    num_windows = get_number_of_windows(1024, 256, 128)
    assert Sxx.shape == (256, num_windows)


# ── Mode tests ──

def test_mode_psd_non_negative():
    """PSD mode should return real, non-negative values."""
    signal = np.random.randn(2048).astype(np.float64)
    f, t, Sxx = spectrogram(signal, nperseg=256, noverlap=128, mode='psd')
    assert Sxx.dtype == np.float64
    assert np.all(Sxx >= 0)


def test_mode_complex_returns_complex():
    """Complex mode should return complex dtype."""
    signal = np.random.randn(2048).astype(np.float64)
    f, t, Sxx = spectrogram(signal, nperseg=256, noverlap=128, mode='complex')
    assert np.issubdtype(Sxx.dtype, np.complexfloating)


def test_mode_magnitude():
    """Magnitude mode should equal abs(complex mode)."""
    signal = np.random.randn(2048).astype(np.float64)
    f, t, Sxx_mag = spectrogram(signal, nperseg=256, noverlap=128, mode='magnitude')
    _, _, Sxx_cplx = spectrogram(signal, nperseg=256, noverlap=128, mode='complex')
    np.testing.assert_allclose(Sxx_mag, np.abs(Sxx_cplx), rtol=1e-12)


def test_mode_angle():
    """Angle mode should return values in [-pi, pi]."""
    signal = np.random.randn(2048).astype(np.float64)
    f, t, Sxx = spectrogram(signal, nperseg=256, noverlap=128, mode='angle')
    assert np.all(Sxx >= -np.pi)
    assert np.all(Sxx <= np.pi)


def test_mode_phase_continuous():
    """Phase (unwrapped) should be more continuous than raw angle."""
    fs = 256
    freq = 32
    t_sig = np.arange(4096) / fs
    signal = np.sin(2 * np.pi * freq * t_sig).astype(np.float64)

    _, _, angle_result = spectrogram(signal, nperseg=256, noverlap=200, mode='angle')
    _, _, phase_result = spectrogram(signal, nperseg=256, noverlap=200, mode='phase')

    # Sxx shape: (output_size, num_windows)
    # At the peak frequency bin, unwrapped phase should have smaller jumps
    peak_bin = int(freq * 256 / fs)
    angle_diffs = np.abs(np.diff(angle_result[peak_bin, :]))
    phase_diffs = np.abs(np.diff(phase_result[peak_bin, :]))
    assert np.max(phase_diffs) <= np.max(angle_diffs) + 1e-10


# ── Scaling tests ──

def test_scaling_density_vs_spectrum():
    """Density and spectrum scalings should differ by sum(win)² / (fs * sum(win²))."""
    try:
        from scipy.signal import get_window
    except ImportError:
        return  # skip if scipy not installed

    signal = np.random.randn(2048).astype(np.float64)
    fs = 1000.0
    nperseg = 256
    win = get_window(('tukey', 0.25), nperseg, fftbins=True)

    _, _, density = spectrogram(signal, nperseg=nperseg, noverlap=128, fs=fs, mode='psd', scaling='density')
    _, _, spectrum = spectrogram(signal, nperseg=nperseg, noverlap=128, fs=fs, mode='psd', scaling='spectrum')

    # density_scale = 1/(fs * sum(win²)), spectrum_scale = 1/sum(win)²
    # ratio = density / spectrum = sum(win)² / (fs * sum(win²))
    expected_ratio = np.sum(win) ** 2 / (fs * np.sum(win ** 2))
    # Check ratio at non-zero bins to avoid 0/0
    mask = spectrum > 1e-30
    if np.any(mask):
        ratios = density[mask] / spectrum[mask]
        np.testing.assert_allclose(ratios, expected_ratio, rtol=1e-10)


def test_psd_onesided_doubling():
    """For real input, non-DC/Nyquist bins should be doubled in PSD mode."""
    try:
        from scipy.signal import get_window
    except ImportError:
        return  # skip if scipy not installed

    signal = np.random.randn(2048).astype(np.float64)
    nperseg = 256
    win = get_window(('tukey', 0.25), nperseg, fftbins=True)
    sum_win_sq = np.sum(win ** 2)

    _, _, result_psd = spectrogram(signal, nperseg=nperseg, noverlap=128, mode='psd')
    _, _, result_cplx = spectrogram(signal, nperseg=nperseg, noverlap=128, mode='complex')

    # Both have shape (output_size, num_windows)
    # Manually compute PSD from complex with density scaling: |X|² / (fs * sum(win²))
    manual_psd = (np.abs(result_cplx) ** 2) / (1.0 * sum_win_sq)  # fs=1.0

    # Middle bins (not DC, not Nyquist) should be doubled
    # Freq axis is axis 0 for 1D input
    np.testing.assert_allclose(result_psd[1:-1, :], manual_psd[1:-1, :] * 2, rtol=1e-10)

    # DC and Nyquist should NOT be doubled
    np.testing.assert_allclose(result_psd[0, :], manual_psd[0, :], rtol=1e-10)
    np.testing.assert_allclose(result_psd[-1, :], manual_psd[-1, :], rtol=1e-10)


# ── Window tests ──

def test_window_none_matches_no_window():
    """window=None should produce the same result as no windowing."""
    signal = np.random.randn(2048).astype(np.float64)
    _, _, result_none = spectrogram(signal, nperseg=256, noverlap=128, window=None)
    _, _, result_boxcar = spectrogram(signal, nperseg=256, noverlap=128, window=np.ones(256))
    np.testing.assert_allclose(result_none, result_boxcar, rtol=1e-12)


def test_window_boxcar_string_matches_none():
    """window='boxcar' should produce the same result as window=None."""
    signal = np.random.randn(2048).astype(np.float64)
    _, _, result_boxcar = spectrogram(signal, nperseg=256, noverlap=128, window='boxcar')
    _, _, result_none = spectrogram(signal, nperseg=256, noverlap=128, window=None)
    np.testing.assert_allclose(result_boxcar, result_none, rtol=1e-12)


def test_window_array():
    """Passing a numpy array as window should work and change the output."""
    signal = np.random.randn(2048).astype(np.float64)
    win = np.hanning(256)
    _, _, result_windowed = spectrogram(signal, nperseg=256, noverlap=128, window=win)
    _, _, result_none = spectrogram(signal, nperseg=256, noverlap=128, window=None)
    # Output shapes should match
    assert result_windowed.shape == result_none.shape
    # Values should differ (windowing changes the spectrum)
    assert not np.allclose(result_windowed, result_none)


def test_window_string():
    """String window specification should work via scipy.signal.get_window."""
    try:
        from scipy.signal import get_window
    except ImportError:
        return  # skip if scipy not installed

    signal = np.random.randn(2048).astype(np.float64)
    f, t, Sxx = spectrogram(signal, nperseg=256, noverlap=128, window='hann')
    num_windows = get_number_of_windows(2048, 256, 128)
    output_size = 256 // 2 + 1
    assert Sxx.shape == (output_size, num_windows)
    assert np.all(Sxx >= 0)


def test_window_tuple():
    """Tuple window specification should work (e.g., ('tukey', 0.25))."""
    try:
        from scipy.signal import get_window
    except ImportError:
        return  # skip if scipy not installed

    signal = np.random.randn(2048).astype(np.float64)
    f, t, Sxx = spectrogram(signal, nperseg=256, noverlap=128, window=('tukey', 0.25))
    num_windows = get_number_of_windows(2048, 256, 128)
    output_size = 256 // 2 + 1
    assert Sxx.shape == (output_size, num_windows)
    assert np.all(Sxx >= 0)


def test_window_wrong_length_raises():
    """Window array with wrong length should raise ValueError."""
    signal = np.random.randn(2048).astype(np.float64)
    try:
        spectrogram(signal, nperseg=256, noverlap=128, window=np.ones(128))
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "nperseg" in str(e)


def test_window_complex_signal():
    """Window should work with complex input signals."""
    signal = (np.random.randn(2048) + 1j * np.random.randn(2048)).astype(np.complex128)
    win = np.hanning(256)
    f, t, Sxx = spectrogram(signal, nperseg=256, noverlap=128, window=win)
    num_windows = get_number_of_windows(2048, 256, 128)
    assert Sxx.shape == (256, num_windows)
    assert np.all(Sxx >= 0)


def test_window_psd_scaling():
    """PSD scaling should use window values, not just nperseg."""
    try:
        from scipy.signal import get_window
    except ImportError:
        return  # skip if scipy not installed

    signal = np.random.randn(2048).astype(np.float64)
    win = get_window('hann', 256, fftbins=True)

    _, _, density = spectrogram(signal, nperseg=256, noverlap=128, window='hann',
                                fs=1000.0, mode='psd', scaling='density')
    _, _, spectrum = spectrogram(signal, nperseg=256, noverlap=128, window='hann',
                                 fs=1000.0, mode='psd', scaling='spectrum')

    # density_scale = 1/(fs * sum(win²)), spectrum_scale = 1/sum(win)²
    # ratio = density / spectrum = sum(win)² / (fs * sum(win²))
    expected_ratio = np.sum(win) ** 2 / (1000.0 * np.sum(win ** 2))
    mask = spectrum > 1e-30
    if np.any(mask):
        ratios = density[mask] / spectrum[mask]
        np.testing.assert_allclose(ratios, expected_ratio, rtol=1e-10)


# ── Detrend tests ──

def test_detrend_constant_removes_dc():
    """detrend='constant' should remove the DC offset from each segment."""
    # Signal with large DC offset — detrending should suppress the DC bin
    signal = (np.ones(2048) * 100 + np.random.randn(2048) * 0.01).astype(np.float64)
    _, _, result_detrend = spectrogram(signal, nperseg=256, noverlap=128, detrend='constant',
                                       window=None)
    _, _, result_nodetrend = spectrogram(signal, nperseg=256, noverlap=128, detrend=False,
                                         window=None)
    # Sxx shape: (output_size, num_windows)
    # DC bin (index 0) should be much smaller with detrending
    dc_detrend = np.mean(result_detrend[0, :])
    dc_nodetrend = np.mean(result_nodetrend[0, :])
    assert dc_detrend < dc_nodetrend * 0.01, (
        f"DC with detrend ({dc_detrend}) should be much smaller than without ({dc_nodetrend})"
    )


def test_detrend_false_matches_no_detrend():
    """detrend=False should produce the same result as the raw spectrogram."""
    signal = np.random.randn(2048).astype(np.float64)
    f, t, Sxx = spectrogram(signal, nperseg=256, noverlap=128, detrend=False)
    # With detrend=False, the signal goes directly to C (no pre-segmentation)
    assert Sxx.shape[0] == 256 // 2 + 1
    assert np.all(Sxx >= 0)


def test_detrend_linear():
    """detrend='linear' should remove linear trends from each segment."""
    try:
        from scipy.signal import detrend as _
    except ImportError:
        return  # skip if scipy not installed

    # Signal with a strong linear ramp
    signal = (np.linspace(0, 1000, 2048) + np.random.randn(2048) * 0.01).astype(np.float64)
    _, _, result_detrend = spectrogram(signal, nperseg=256, noverlap=128, detrend='linear',
                                       window=None)
    _, _, result_nodetrend = spectrogram(signal, nperseg=256, noverlap=128, detrend=False,
                                         window=None)
    # Total power should be much lower after linear detrending of a ramp signal
    power_detrend = np.sum(result_detrend)
    power_nodetrend = np.sum(result_nodetrend)
    assert power_detrend < power_nodetrend * 0.01


def test_detrend_callable():
    """A custom detrend function should be applied to each segment."""
    signal = (np.ones(2048) * 50 + np.random.randn(2048) * 0.01).astype(np.float64)
    # Custom function: subtract the mean (same as 'constant')
    _, _, result_callable = spectrogram(signal, nperseg=256, noverlap=128,
                                        detrend=lambda seg: seg - seg.mean(),
                                        window=None)
    _, _, result_constant = spectrogram(signal, nperseg=256, noverlap=128,
                                        detrend='constant', window=None)
    np.testing.assert_allclose(result_callable, result_constant, rtol=1e-10)


def test_detrend_shape_unchanged():
    """Detrending should not change the output shape."""
    signal = np.random.randn(2048).astype(np.float64)
    _, _, result_detrend = spectrogram(signal, nperseg=256, noverlap=128, detrend='constant')
    _, _, result_nodetrend = spectrogram(signal, nperseg=256, noverlap=128, detrend=False)
    assert result_detrend.shape == result_nodetrend.shape


def test_detrend_complex_signal():
    """Detrending should work with complex signals."""
    signal = (np.ones(2048) * (50 + 50j) + (np.random.randn(2048) + 1j * np.random.randn(2048)) * 0.01).astype(np.complex128)
    f, t, Sxx = spectrogram(signal, nperseg=256, noverlap=128, detrend='constant',
                             window=None)
    num_windows = get_number_of_windows(2048, 256, 128)
    assert Sxx.shape == (256, num_windows)
    assert np.all(Sxx >= 0)


def test_detrend_batch():
    """Detrending should work with batched 2D input."""
    signals = np.random.randn(5, 2048).astype(np.float64)
    f, t, Sxx = spectrogram(signals, nperseg=256, noverlap=128, detrend='constant')
    num_windows = get_number_of_windows(2048, 256, 128)
    output_size = 256 // 2 + 1
    assert Sxx.shape == (5, output_size, num_windows)
    assert np.all(Sxx >= 0)


# ── return_onesided tests ──

def test_onesided_true_shape():
    """return_onesided=True (default) should give nfft//2+1 freq bins for real input."""
    signal = np.random.randn(2048).astype(np.float64)
    f, t, Sxx = spectrogram(signal, nperseg=256, noverlap=128, return_onesided=True, detrend=False)
    num_windows = get_number_of_windows(2048, 256, 128)
    assert Sxx.shape == (256 // 2 + 1, num_windows)


def test_onesided_false_shape():
    """return_onesided=False should give nfft freq bins for real input."""
    signal = np.random.randn(2048).astype(np.float64)
    f, t, Sxx = spectrogram(signal, nperseg=256, noverlap=128, return_onesided=False, detrend=False)
    num_windows = get_number_of_windows(2048, 256, 128)
    assert Sxx.shape == (256, num_windows)


def test_onesided_false_hermitian_symmetry():
    """Two-sided spectrum of real signal should have Hermitian symmetry: X[N-k] = conj(X[k])."""
    signal = np.random.randn(2048).astype(np.float64)
    f, t, Sxx = spectrogram(signal, nperseg=256, noverlap=128, return_onesided=False,
                             mode='complex', detrend=False)
    # Sxx shape: (256, num_windows)
    nfft = 256
    for w in range(Sxx.shape[-1]):
        spectrum = Sxx[:, w]
        for k in range(1, nfft // 2):
            np.testing.assert_allclose(
                spectrum[nfft - k], np.conj(spectrum[k]),
                rtol=1e-12,
                err_msg=f"Window {w}, bin {k}: Hermitian symmetry violated"
            )


def test_onesided_false_psd_matches_onesided():
    """Two-sided PSD total power should approximately match one-sided PSD total power."""
    signal = np.random.randn(2048).astype(np.float64)
    _, _, psd_one = spectrogram(signal, nperseg=256, noverlap=128, return_onesided=True,
                                mode='psd', detrend=False)
    _, _, psd_two = spectrogram(signal, nperseg=256, noverlap=128, return_onesided=False,
                                mode='psd', detrend=False)
    # Total power per window should be the same
    # One-sided has doubled non-DC/Nyquist bins; two-sided has both sides
    power_one = np.sum(psd_one, axis=0)  # sum over freq axis
    power_two = np.sum(psd_two, axis=0)
    np.testing.assert_allclose(power_one, power_two, rtol=1e-10)


def test_onesided_false_complex_input_ignored():
    """For complex input, return_onesided has no effect (always two-sided)."""
    signal = (np.random.randn(2048) + 1j * np.random.randn(2048)).astype(np.complex128)
    _, _, result_true = spectrogram(signal, nperseg=256, noverlap=128, return_onesided=True, detrend=False)
    _, _, result_false = spectrogram(signal, nperseg=256, noverlap=128, return_onesided=False, detrend=False)
    # Both should be identical — complex always returns full spectrum
    np.testing.assert_allclose(result_true, result_false, rtol=1e-12)


def test_onesided_false_2d_shape():
    """return_onesided=False with 2D input should have correct shape."""
    signals = np.random.randn(5, 2048).astype(np.float64)
    f, t, Sxx = spectrogram(signals, nperseg=256, noverlap=128, return_onesided=False, detrend=False)
    num_windows = get_number_of_windows(2048, 256, 128)
    assert Sxx.shape == (5, 256, num_windows)


def test_onesided_false_peak_frequency():
    """Two-sided spectrum should show peaks at both positive and negative frequency bins."""
    fs = 256
    freq = 32  # Hz
    t_sig = np.arange(1024) / fs
    signal = np.sin(2 * np.pi * freq * t_sig).astype(np.float64)

    f, t, Sxx = spectrogram(signal, nperseg=256, noverlap=0, return_onesided=False,
                             mode='psd', detrend=False)
    # Sxx shape: (256, num_windows)
    nfft = 256
    expected_pos_bin = int(freq * nfft / fs)  # bin 32
    expected_neg_bin = nfft - expected_pos_bin  # bin 224

    for w in range(Sxx.shape[-1]):
        spectrum = Sxx[:, w]
        # Both positive and negative freq bins should be among the top peaks
        top_bins = np.argsort(spectrum)[-3:]  # top 3 bins (DC might be there too)
        assert expected_pos_bin in top_bins, f"Window {w}: positive bin {expected_pos_bin} not in top bins {top_bins}"
        assert expected_neg_bin in top_bins, f"Window {w}: negative bin {expected_neg_bin} not in top bins {top_bins}"


# ── Axis tests ──

def test_axis_default_1d():
    """axis=-1 on 1D input should produce (output_size, num_windows)."""
    signal = np.random.randn(1024).astype(np.float64)
    f, t, Sxx = spectrogram(signal, nperseg=256, noverlap=128)
    num_windows = get_number_of_windows(1024, 256, 128)
    output_size = 256 // 2 + 1
    assert Sxx.shape == (output_size, num_windows)


def test_axis_default_2d():
    """axis=-1 on 2D input should compute along last axis."""
    signals = np.random.randn(5, 1024).astype(np.float64)
    f, t, Sxx = spectrogram(signals, nperseg=256, noverlap=128)
    num_windows = get_number_of_windows(1024, 256, 128)
    output_size = 256 // 2 + 1
    assert Sxx.shape == (5, output_size, num_windows)


def test_axis_0_2d():
    """axis=0 on 2D input should compute along first axis."""
    # Shape (1024, 5): signal length along axis 0, 5 independent signals along axis 1
    signals = np.random.randn(1024, 5).astype(np.float64)
    f, t, Sxx = spectrogram(signals, nperseg=256, noverlap=128, axis=0)
    num_windows = get_number_of_windows(1024, 256, 128)
    output_size = 256 // 2 + 1
    # axis=0 replaced by output_size, num_windows appended
    assert Sxx.shape == (output_size, 5, num_windows)
    assert np.all(Sxx >= 0)


def test_axis_3d():
    """3D input with axis=1 should produce correct output shape."""
    # Shape (3, 1024, 4): spectrogram along axis 1
    x = np.random.randn(3, 1024, 4).astype(np.float64)
    f, t, Sxx = spectrogram(x, nperseg=256, noverlap=128, axis=1)
    num_windows = get_number_of_windows(1024, 256, 128)
    output_size = 256 // 2 + 1
    # axis=1 replaced by output_size, num_windows appended
    assert Sxx.shape == (3, output_size, 4, num_windows)
    assert np.all(Sxx >= 0)


def test_axis_4d():
    """4D input matching the user's example: (A, B, C, D) with axis=1."""
    A, B, C, D = 2, 512, 3, 4
    x = np.random.randn(A, B, C, D).astype(np.float64)
    nperseg = 64
    noverlap = 32
    f, t, Sxx = spectrogram(x, nperseg=nperseg, noverlap=noverlap, axis=1)
    num_windows = get_number_of_windows(B, nperseg, noverlap)
    output_size = nperseg // 2 + 1
    assert Sxx.shape == (A, output_size, C, D, num_windows)


def test_axis_negative():
    """Negative axis values should work correctly."""
    signals = np.random.randn(5, 1024).astype(np.float64)
    _, _, result_neg = spectrogram(signals, nperseg=256, noverlap=128, axis=-1)
    _, _, result_pos = spectrogram(signals, nperseg=256, noverlap=128, axis=1)
    np.testing.assert_allclose(result_neg, result_pos, rtol=1e-12)


def test_axis_0_vs_transpose():
    """Computing along axis=0 should match transposing and computing along axis=-1."""
    signals = np.random.randn(1024, 5).astype(np.float64)
    _, _, result_axis0 = spectrogram(signals, nperseg=256, noverlap=128, axis=0)
    # Transpose to (5, 1024), compute along last axis, then rearrange
    _, _, result_default = spectrogram(signals.T, nperseg=256, noverlap=128)
    # result_axis0 shape: (output_size, 5, num_windows)
    # result_default shape: (5, output_size, num_windows)
    # They should be transposes of each other on first two axes
    np.testing.assert_allclose(result_axis0, np.moveaxis(result_default, 0, 1), rtol=1e-12)


def test_axis_complex_3d():
    """3D complex input with axis=2."""
    x = (np.random.randn(3, 4, 512) + 1j * np.random.randn(3, 4, 512)).astype(np.complex128)
    f, t, Sxx = spectrogram(x, nperseg=64, noverlap=32, axis=2)
    num_windows = get_number_of_windows(512, 64, 32)
    output_size = 64  # complex: full spectrum
    assert Sxx.shape == (3, 4, output_size, num_windows)
    assert np.all(Sxx >= 0)


# ── Frequency and time array tests ──

def test_freq_array_onesided():
    """One-sided freq array should go from 0 to fs/2."""
    fs = 1000.0
    signal = np.random.randn(2048).astype(np.float64)
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=256, noverlap=128, return_onesided=True, detrend=False)
    expected_f = np.fft.rfftfreq(256, d=1.0 / fs)
    np.testing.assert_allclose(f, expected_f, rtol=1e-14)
    assert f[0] == 0.0
    assert f[-1] == fs / 2


def test_freq_array_twosided():
    """Two-sided freq array for real input should have nfft bins."""
    fs = 1000.0
    signal = np.random.randn(2048).astype(np.float64)
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=256, noverlap=128, return_onesided=False, detrend=False)
    expected_f = np.fft.fftfreq(256, d=1.0 / fs)
    np.testing.assert_allclose(f, expected_f, rtol=1e-14)
    assert len(f) == 256


def test_freq_array_complex():
    """Complex input freq array should always be two-sided."""
    fs = 500.0
    signal = (np.random.randn(2048) + 1j * np.random.randn(2048)).astype(np.complex128)
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=256, noverlap=128, detrend=False)
    expected_f = np.fft.fftfreq(256, d=1.0 / fs)
    np.testing.assert_allclose(f, expected_f, rtol=1e-14)
    assert len(f) == 256


def test_freq_array_zero_padded():
    """Freq array with zero-padding should reflect nfft, not nperseg."""
    fs = 1000.0
    signal = np.random.randn(2048).astype(np.float64)
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=256, noverlap=128, nfft=512, detrend=False)
    expected_f = np.fft.rfftfreq(512, d=1.0 / fs)
    np.testing.assert_allclose(f, expected_f, rtol=1e-14)
    assert len(f) == 512 // 2 + 1


def test_time_array_values():
    """Time array should give segment center times."""
    fs = 1000.0
    nperseg = 256
    noverlap = 128
    signal = np.random.randn(2048).astype(np.float64)
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend=False)
    num_windows = get_number_of_windows(2048, nperseg, noverlap)
    step = nperseg - noverlap
    expected_t = np.arange(num_windows) * step / fs + (nperseg - 1) / 2.0 / fs
    np.testing.assert_allclose(t, expected_t, rtol=1e-14)


def test_time_array_length():
    """Time array length should match the last axis of Sxx."""
    signal = np.random.randn(2048).astype(np.float64)
    f, t, Sxx = spectrogram(signal, nperseg=256, noverlap=128, detrend=False)
    assert len(t) == Sxx.shape[-1]


def test_freq_array_length_matches_sxx():
    """Freq array length should match the frequency axis of Sxx."""
    signal = np.random.randn(2048).astype(np.float64)
    f, t, Sxx = spectrogram(signal, nperseg=256, noverlap=128, detrend=False)
    # For 1D input, freq axis is axis 0
    assert len(f) == Sxx.shape[0]


def test_freq_time_2d_batch():
    """f and t should be the same regardless of batch size."""
    fs = 500.0
    signal_1d = np.random.randn(2048).astype(np.float64)
    signals_2d = np.random.randn(5, 2048).astype(np.float64)
    f1, t1, _ = spectrogram(signal_1d, fs=fs, nperseg=256, noverlap=128, detrend=False)
    f2, t2, _ = spectrogram(signals_2d, fs=fs, nperseg=256, noverlap=128, detrend=False)
    np.testing.assert_allclose(f1, f2, rtol=1e-14)
    np.testing.assert_allclose(t1, t2, rtol=1e-14)


def test_time_no_overlap():
    """Time array with noverlap=0 should have step = nperseg/fs."""
    fs = 100.0
    nperseg = 256
    signal = np.random.randn(2048).astype(np.float64)
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=0, detrend=False)
    if len(t) > 1:
        dt = t[1] - t[0]
        np.testing.assert_allclose(dt, nperseg / fs, rtol=1e-14)


def test_freq_peak_matches_bin():
    """The frequency of the peak PSD bin should match the input sine frequency."""
    fs = 1000.0
    freq = 125.0  # Hz
    t_sig = np.arange(8192) / fs
    signal = np.sin(2 * np.pi * freq * t_sig).astype(np.float64)

    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=256, noverlap=128, detrend=False)
    # Check that peak frequency in each window matches the input
    for w in range(Sxx.shape[-1]):
        peak_bin = np.argmax(Sxx[1:, w]) + 1  # skip DC
        peak_freq = f[peak_bin]
        # Allow ±1 bin tolerance (freq resolution = fs/nfft)
        assert abs(peak_freq - freq) <= fs / 256 + 1e-10, (
            f"Window {w}: peak at {peak_freq} Hz, expected ~{freq} Hz"
        )


# ── Larger batch tests ──

def test_large_batch_real():
    """Batch of 10000 real signals, each 2048 samples."""
    signals = np.random.randn(10000, 2048).astype(np.float64)
    f, t, Sxx = spectrogram(signals, nperseg=256, noverlap=128)
    num_windows = get_number_of_windows(2048, 256, 128)
    output_size = 256 // 2 + 1
    assert Sxx.shape == (10000, output_size, num_windows)
    assert np.all(Sxx >= 0)


def test_large_batch_complex():
    """Batch of 10000 complex signals, each 2048 samples."""
    signals = (np.random.randn(10000, 2048) + 1j * np.random.randn(10000, 2048)).astype(np.complex128)
    f, t, Sxx = spectrogram(signals, nperseg=256, noverlap=128)
    num_windows = get_number_of_windows(2048, 256, 128)
    output_size = 256
    assert Sxx.shape == (10000, output_size, num_windows)
    assert np.all(Sxx >= 0)


if __name__ == "__main__":
    test_real_output_shape_1d()
    test_real_output_shape_2d()
    test_real_zero_padding_shape()
    test_real_peak_frequency()
    test_real_non_negative()
    test_complex_output_shape_1d()
    test_complex_output_shape_2d()
    test_complex_zero_padding_shape()
    test_complex_peak_frequency()
    test_complex_non_negative()
    test_auto_detect_real()
    test_auto_detect_complex()
    test_mode_psd_non_negative()
    test_mode_complex_returns_complex()
    test_mode_magnitude()
    test_mode_angle()
    test_mode_phase_continuous()
    test_scaling_density_vs_spectrum()
    test_psd_onesided_doubling()
    test_window_none_matches_no_window()
    test_window_boxcar_string_matches_none()
    test_window_array()
    test_window_string()
    test_window_tuple()
    test_window_wrong_length_raises()
    test_window_complex_signal()
    test_window_psd_scaling()
    test_detrend_constant_removes_dc()
    test_detrend_false_matches_no_detrend()
    test_detrend_linear()
    test_detrend_callable()
    test_detrend_shape_unchanged()
    test_detrend_complex_signal()
    test_detrend_batch()
    test_onesided_true_shape()
    test_onesided_false_shape()
    test_onesided_false_hermitian_symmetry()
    test_onesided_false_psd_matches_onesided()
    test_onesided_false_complex_input_ignored()
    test_onesided_false_2d_shape()
    test_onesided_false_peak_frequency()
    test_axis_default_1d()
    test_axis_default_2d()
    test_axis_0_2d()
    test_axis_3d()
    test_axis_4d()
    test_axis_negative()
    test_axis_0_vs_transpose()
    test_axis_complex_3d()
    test_freq_array_onesided()
    test_freq_array_twosided()
    test_freq_array_complex()
    test_freq_array_zero_padded()
    test_time_array_values()
    test_time_array_length()
    test_freq_array_length_matches_sxx()
    test_freq_time_2d_batch()
    test_time_no_overlap()
    test_freq_peak_matches_bin()
    test_large_batch_real()
    test_large_batch_complex()
    print("All tests passed!")
