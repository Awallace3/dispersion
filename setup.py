import os
import re
import sys
import subprocess
from pathlib import Path
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools import Extension, setup, find_packages

# DFTD4 binary
from setuptools.command.install import install
from setuptools.command.develop import develop
import shutil
import site
import atexit

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        # Using this requires trailing slash for auto-detection & inclusion of
        # auxiliary "native" libs

        debug = int(os.environ.get("DEBUG", 0)
                    ) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
        ]
        build_args = []
        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [
                item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        # In this example, we pass in the version to C++. You might not need to.
        cmake_args += [
            f"-DEXAMPLE_VERSION_INFO={self.distribution.get_version()}"]

        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator or cmake_generator == "Ninja":
                try:
                    import ninja

                    ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"
                    cmake_args += [
                        "-GNinja",
                        f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
                    ]
                except ImportError:
                    pass

        else:
            # Single config generators are handled "normally"
            single_config = any(
                x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"
                ]
                build_args += ["--config", cfg]

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += [
                    "-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True
        )
        subprocess.run(
            ["cmake", "--build", ".", *build_args], cwd=build_temp, check=True
        )


class PostInstallCommand(install):
    """Post-installation for installation mode for DFTD4 install"""

    def run(self):
        # Run the standard install first
        install.run(self)

        # Get the conda bin path
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            # Path to your binary after build
            source_binary = os.path.join("build", "dftd4", "app", "dftd4")
            # Destination path in conda bin
            dest_path = os.path.join(conda_prefix, "bin")

            if os.path.exists(source_binary):
                print(f"Copying {source_binary} to {dest_path}")
                if not os.path.exists(dest_path):
                    os.makedirs(dest_path)
                shutil.copy2(source_binary, dest_path)
                # Make sure it's executable
                binary_path = os.path.join(dest_path, "dftd4")
                if os.path.exists(binary_path):
                    os.chmod(binary_path, 0o755)
                print(f"Successfully installed dftd4 binary to {dest_path}")
            else:
                print(
                    f"Warning: {source_binary} does not exist, skipping binary installation"
                )
        else:
            print(
                "Not in a conda environment, skipping binary installation to conda bin"
            )


class CustomInstallCommand(install):
    """Custom install command to copy binary to conda bin."""

    def run(self):
        install.run(self)
        self._copy_binary()

    def _copy_binary(self):
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            source_binary = os.path.join("build", "dftd4", "app", "dftd4")
            dest_path = os.path.join(conda_prefix, "bin")

            if os.path.exists(source_binary):
                if not os.path.exists(dest_path):
                    os.makedirs(dest_path)
                dest_file = os.path.join(dest_path, "dftd4")
                shutil.copy2(source_binary, dest_file)
                os.chmod(dest_file, 0o755)

                # Record the installed binary path
                self.distribution.data_files = self.distribution.data_files or []
                self.distribution.data_files.append(("bin", [dest_file]))

                print(f"Successfully installed dftd4 binary to {dest_path}")

                # Create .dist-info directory record
                record_file = os.path.join(
                    self.install_lib,
                    f"{self.distribution.get_name()}-{self.distribution.get_version()}.dist-info",
                    "installed-files.txt",
                )
                with open(record_file, "a") as f:
                    f.write(f"{dest_file}\n")
            else:
                print(
                    f"Warning: {source_binary} does not exist, skipping binary installation"
                )
        else:
            print(
                "Not in a conda environment, skipping binary installation to conda bin"
            )


# For development installs
class CustomDevelopCommand(develop):
    """Custom develop command to copy binary to conda bin."""

    def run(self):
        develop.run(self)
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            source_binary = os.path.join("build", "dftd4", "app", "dftd4")
            dest_path = os.path.join(conda_prefix, "bin")

            if os.path.exists(source_binary):
                if not os.path.exists(dest_path):
                    os.makedirs(dest_path)
                dest_file = os.path.join(dest_path, "dftd4")
                shutil.copy2(source_binary, dest_file)
                os.chmod(dest_file, 0o755)
                print(f"Successfully installed dftd4 binary to {dest_path}")
            else:
                print(
                    f"Warning: {source_binary} does not exist, skipping binary installation"
                )
        else:
            print(
                "Not in a conda environment, skipping binary installation to conda bin"
            )


# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
setup(
    name="dispersion_amw",
    version="0.1.0",
    author="Austin Wallace",
    author_email="austinwallace196@gmail.com",
    description="Dispersion module in C++ for Python",
    long_description="""
    Dispersion module for calculating dispersion energies using C++ for performance-critical operations.
    Includes Python utilities for molecular structure handling, analysis, and visualization.
    """,
    packages=find_packages(),
    ext_modules=[CMakeExtension("dispersion"), CMakeExtension("disp")],
    cmdclass={
        "build_ext": CMakeBuild,
        "install": PostInstallCommand,
    },
    zip_safe=False,
    extras_require={
        "test": ["pytest>=6.0"],
    },
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
