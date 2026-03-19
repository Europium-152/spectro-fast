import cffi
import platform
import os
import subprocess

ffibuilder = cffi.FFI()

here = os.path.dirname(os.path.abspath(__file__))

# Read function declarations from the header
with open(os.path.join(here, "csrc", "real_spectrogram.h")) as f:
    ffibuilder.cdef(f.read())

# Read the C source
with open(os.path.join(here, "csrc", "real_spectrogram.c")) as f:
    c_source = f.read()

# Platform-specific configuration
system = platform.system()

# CI builds set FFTW_STATIC_DIR to a directory containing libfftw3.a and fftw3.h
# This allows us to statically link FFTW into the wheel so users don't need it installed.
fftw_static_dir = os.environ.get("FFTW_STATIC_DIR")

if system == "Windows":
    vendor_dir = os.path.join(here, "_vendor_win")
    # Generate import lib from .def for MSVC (CI) or use directly for MinGW (local)
    ffibuilder.set_source(
        "spectrofast._real_spectrogram_cffi",
        c_source,
        libraries=["fftw3-3"],
        library_dirs=[vendor_dir],
        include_dirs=[vendor_dir],
    )

elif fftw_static_dir:
    # CI build: statically link FFTW (Linux & macOS)
    static_lib = os.path.join(fftw_static_dir, "lib", "libfftw3.a")
    static_inc = os.path.join(fftw_static_dir, "include")
    extra_compile = ["-fPIC"] if system == "Linux" else []
    extra_link = [static_lib, "-lm"]
    if system == "Linux":
        extra_link.insert(0, "-Wl,--no-as-needed")

    ffibuilder.set_source(
        "spectrofast._real_spectrogram_cffi",
        c_source,
        include_dirs=[static_inc],
        extra_compile_args=extra_compile,
        extra_link_args=extra_link,
    )

elif system == "Darwin":
    # macOS: try pkg-config, fall back to Homebrew paths
    include_dirs = []
    library_dirs = []
    try:
        cflags = subprocess.check_output(["pkg-config", "--cflags", "fftw3"], text=True).strip()
        libs = subprocess.check_output(["pkg-config", "--libs-only-L", "fftw3"], text=True).strip()
        include_dirs = [f.replace("-I", "") for f in cflags.split() if f.startswith("-I")]
        library_dirs = [f.replace("-L", "") for f in libs.split() if f.startswith("-L")]
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Homebrew defaults
        for prefix in ["/opt/homebrew", "/usr/local"]:
            inc = os.path.join(prefix, "include")
            lib = os.path.join(prefix, "lib")
            if os.path.exists(os.path.join(inc, "fftw3.h")):
                include_dirs.append(inc)
                library_dirs.append(lib)
                break

    ffibuilder.set_source(
        "spectrofast._real_spectrogram_cffi",
        c_source,
        libraries=["fftw3"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
    )

else:
    # Linux: try pkg-config, then $HOME/.local, then system default
    include_dirs = []
    library_dirs = []
    try:
        cflags = subprocess.check_output(["pkg-config", "--cflags", "fftw3"], text=True).strip()
        libs = subprocess.check_output(["pkg-config", "--libs-only-L", "fftw3"], text=True).strip()
        include_dirs = [f.replace("-I", "") for f in cflags.split() if f.startswith("-I")]
        library_dirs = [f.replace("-L", "") for f in libs.split() if f.startswith("-L")]
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fall back to ~/.local (user-installed FFTW without sudo)
        home = os.path.expanduser("~")
        local_inc = os.path.join(home, ".local", "include")
        local_lib = os.path.join(home, ".local", "lib")
        if os.path.exists(os.path.join(local_inc, "fftw3.h")):
            include_dirs.append(local_inc)
            library_dirs.append(local_lib)

    ffibuilder.set_source(
        "spectrofast._real_spectrogram_cffi",
        c_source,
        libraries=["fftw3"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        extra_compile_args=["-fPIC"],
        extra_link_args=["-Wl,--no-as-needed"],
    )

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
