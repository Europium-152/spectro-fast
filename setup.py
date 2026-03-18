import sys
import os
import sysconfig

# On non-Windows, prevent setuptools from generating .def export files
# which Linux gcc cannot handle
if sys.platform != "win32":
    # Remove EXT_SUFFIX patterns that trigger .def generation
    pass
else:
    # Force MinGW compiler on Windows
    os.environ.setdefault("CC", "gcc")
    if "build_ext" not in sys.argv:
        sys.argv.insert(1, "build_ext")
        sys.argv.insert(2, "--compiler=mingw32")
    from distutils.sysconfig import get_config_vars
    cfg = get_config_vars()
    for key in list(cfg.keys()):
        if isinstance(cfg[key], str) and '/MD' in cfg[key]:
            cfg[key] = cfg[key].replace('/MD', '')

from setuptools import setup

setup(
    cffi_modules=["spectrofast/_fft_build.py:ffibuilder"],
)
