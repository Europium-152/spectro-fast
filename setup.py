import sys
import os
from setuptools import setup
from setuptools.command.build_ext import build_ext


class BuildExtNoDef(build_ext):
    """Custom build_ext that removes .def files from the link step on Linux/Mac.

    cffi generates .abi3.def export files which are only valid on Windows (MinGW).
    On Linux, gcc treats them as linker scripts and fails.
    """
    def build_extension(self, ext):
        # Save original link method
        original_link = self.compiler.link_shared_object

        def patched_link(objects, *args, **kwargs):
            # Filter out .def files on non-Windows
            if sys.platform != "win32":
                objects = [o for o in objects if not o.endswith(".def")]
            return original_link(objects, *args, **kwargs)

        self.compiler.link_shared_object = patched_link
        super().build_extension(ext)
        self.compiler.link_shared_object = original_link


if sys.platform == "win32":
    os.environ.setdefault("CC", "gcc")
    if "build_ext" not in sys.argv:
        sys.argv.insert(1, "build_ext")
        sys.argv.insert(2, "--compiler=mingw32")
    from distutils.sysconfig import get_config_vars
    cfg = get_config_vars()
    for key in list(cfg.keys()):
        if isinstance(cfg[key], str) and '/MD' in cfg[key]:
            cfg[key] = cfg[key].replace('/MD', '')

setup(
    cffi_modules=["spectrofast/_fft_build.py:ffibuilder"],
    cmdclass={"build_ext": BuildExtNoDef},
)
