import sys
import os

# On Linux/Mac, monkey-patch the linker to filter out .def files.
# cffi generates .abi3.def export files that only work on Windows (MinGW).
# On Linux, gcc treats them as linker scripts and fails.
if sys.platform != "win32":
    from distutils.unixccompiler import UnixCCompiler
    _original_link = UnixCCompiler.link

    def _patched_link(self, target_desc, objects, *args, **kwargs):
        objects = [o for o in objects if not o.endswith(".def")]
        return _original_link(self, target_desc, objects, *args, **kwargs)

    UnixCCompiler.link = _patched_link

elif not os.environ.get("CIBUILDWHEEL"):
    # Local Windows dev: force MinGW compiler (skip in CI where MSVC is used)
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
