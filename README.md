# spectrofast

Fast FFTW-based spectrogram computation for Python. A simple, fast alternative to `scipy.signal.spectrogram`.

## Installation

### Prerequisites

spectrofast requires the FFTW3 library and a C compiler.

**Windows:**
- Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (select "Desktop development with C++")
- FFTW is bundled with the package, no extra steps needed

**Linux (Debian/Ubuntu):**
```bash
sudo apt install libfftw3-dev
```

**Linux (Fedora):**
```bash
sudo dnf install fftw-devel
```

**macOS:**
```bash
brew install fftw
```

### Install spectrofast

```bash
pip install spectrofast
```

Or install from source:

```bash
git clone https://github.com/yourusername/spectrofast.git
cd spectrofast
pip install .
```

## Usage

```python
import numpy as np
from spectrofast import spectrogram

# Single signal
signal = np.sin(2 * np.pi * np.arange(1024) / 32).astype(np.float64)
result = spectrogram(signal, nperseg=256, noverlap=128)
print(result.shape)  # (4, 129)

# Batch of signals
signals = np.random.randn(100, 1024).astype(np.float64)
result = spectrogram(signals, nperseg=256, noverlap=128)
print(result.shape)  # (100, 4, 129)

# With zero-padding
result = spectrogram(signal, nperseg=256, noverlap=128, nfft=512)
print(result.shape)  # (4, 257)
```

## Output

The spectrogram function returns **magnitude-squared** values (i.e., `real^2 + imag^2`). This avoids the expensive `sqrt` operation and is suitable for most spectral analysis tasks.

## License

MIT
