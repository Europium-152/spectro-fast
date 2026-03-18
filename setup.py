import sys
import os

# Force MinGW compiler on Windows
if sys.platform == "win32":
    os.environ.setdefault("CC", "gcc")
    # Tell distutils to use mingw32 compiler
    if "build_ext" not in sys.argv:
        sys.argv.insert(1, "build_ext")
        sys.argv.insert(2, "--compiler=mingw32")
        # Remove the injected args after setup processes them
    from distutils.sysconfig import get_config_vars
    # Patch out MSVC-specific flags that break MinGW
    cfg = get_config_vars()
    for key in list(cfg.keys()):
        if isinstance(cfg[key], str) and '/MD' in cfg[key]:
            cfg[key] = cfg[key].replace('/MD', '')

from setuptools import setup

setup(
    cffi_modules=["spectrofast/_fft_build.py:ffibuilder"],
)
