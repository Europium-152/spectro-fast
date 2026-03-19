import numpy as np
from spectrofast import real_spectrogram, complex_spectrogram, get_number_of_windows


def test_output_shape_1d():
    signal = np.sin(2 * np.pi * np.arange(1024) / 32).astype(np.float64)
    result = real_spectrogram(signal, nperseg=256, noverlap=128)
    num_windows = get_number_of_windows(1024, 256, 128)
    output_size = 256 // 2 + 1
    assert result.shape == (num_windows, output_size)


def test_output_shape_2d():
    signals = np.random.randn(10, 1024).astype(np.float64)
    result = real_spectrogram(signals, nperseg=256, noverlap=128)
    num_windows = get_number_of_windows(1024, 256, 128)
    output_size = 256 // 2 + 1
    assert result.shape == (10, num_windows, output_size)


def test_zero_padding_shape():
    signal = np.sin(2 * np.pi * np.arange(1024) / 32).astype(np.float64)
    result = real_spectrogram(signal, nperseg=256, noverlap=128, nfft=512)
    num_windows = get_number_of_windows(1024, 256, 128)
    output_size = 512 // 2 + 1
    assert result.shape == (num_windows, output_size)


def test_peak_frequency():
    """A pure sine wave should produce a peak at the expected frequency bin."""
    fs = 256
    freq = 32  # Hz
    t = np.arange(1024) / fs
    signal = np.sin(2 * np.pi * freq * t).astype(np.float64)

    result = real_spectrogram(signal, nperseg=256, noverlap=0)

    # For each window, find the peak frequency bin (excluding DC)
    for w in range(result.shape[0]):
        peak_bin = np.argmax(result[w, 1:]) + 1  # skip DC
        expected_bin = int(freq * 256 / fs)  # freq * nperseg / fs
        assert peak_bin == expected_bin, f"Window {w}: expected bin {expected_bin}, got {peak_bin}"


def test_non_negative():
    """Magnitude squared values should always be non-negative."""
    signal = np.random.randn(2048).astype(np.float64)
    result = real_spectrogram(signal, nperseg=256, noverlap=128)
    assert np.all(result >= 0)


# ── Complex spectrogram tests ──

def test_complex_output_shape_1d():
    """Complex spectrogram of 1D signal should have shape (num_windows, nfft)."""
    t = np.arange(1024)
    signal = np.exp(2j * np.pi * 32 * t / 256).astype(np.complex128)
    result = complex_spectrogram(signal, nperseg=256, noverlap=128)
    num_windows = get_number_of_windows(1024, 256, 128)
    assert result.shape == (num_windows, 256)


def test_complex_output_shape_2d():
    """Complex spectrogram of 2D batch should have shape (n_signals, num_windows, nfft)."""
    signals = (np.random.randn(10, 1024) + 1j * np.random.randn(10, 1024)).astype(np.complex128)
    result = complex_spectrogram(signals, nperseg=256, noverlap=128)
    num_windows = get_number_of_windows(1024, 256, 128)
    assert result.shape == (10, num_windows, 256)


def test_complex_zero_padding_shape():
    """Complex spectrogram with zero-padding should have output_size = nfft."""
    t = np.arange(1024)
    signal = np.exp(2j * np.pi * 32 * t / 256).astype(np.complex128)
    result = complex_spectrogram(signal, nperseg=256, noverlap=128, nfft=512)
    num_windows = get_number_of_windows(1024, 256, 128)
    assert result.shape == (num_windows, 512)


def test_complex_peak_frequency():
    """A complex exponential should produce a single peak at the expected bin."""
    fs = 256
    freq = 32  # Hz
    t = np.arange(1024) / fs
    signal = np.exp(2j * np.pi * freq * t).astype(np.complex128)

    result = complex_spectrogram(signal, nperseg=256, noverlap=0)

    # For a complex exponential at freq Hz, the peak should be at bin = freq * nperseg / fs
    expected_bin = int(freq * 256 / fs)
    for w in range(result.shape[0]):
        peak_bin = np.argmax(result[w, :])
        assert peak_bin == expected_bin, f"Window {w}: expected bin {expected_bin}, got {peak_bin}"


def test_complex_non_negative():
    """Magnitude squared values should always be non-negative."""
    signal = (np.random.randn(2048) + 1j * np.random.randn(2048)).astype(np.complex128)
    result = complex_spectrogram(signal, nperseg=256, noverlap=128)
    assert np.all(result >= 0)


if __name__ == "__main__":
    test_output_shape_1d()
    test_output_shape_2d()
    test_zero_padding_shape()
    test_peak_frequency()
    test_non_negative()
    test_complex_output_shape_1d()
    test_complex_output_shape_2d()
    test_complex_zero_padding_shape()
    test_complex_peak_frequency()
    test_complex_non_negative()
    print("All tests passed!")
