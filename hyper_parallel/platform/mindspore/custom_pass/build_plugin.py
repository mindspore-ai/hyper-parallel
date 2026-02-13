# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
Build plugin for mindspore custom pass.
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path


def find_cmake():
    """Locate CMake executable"""
    for candidate in ["cmake", "cmake3"]:
        if shutil.which(candidate):
            return candidate
    raise RuntimeError("CMake not found. Please install CMake.")


def build_plugin():
    """
    Build custom pass plugin with version safety and diagnostics.
    Exits with code 0 on intentional skip (version mismatch), non-zero on actual failure.
    """
    # Proceed with compilation
    build_dir = Path(os.environ.get("MS_PLUGIN_BUILD_DIR", "build")).resolve()
    so_output = Path(os.environ.get(
        "MS_SO_OUTPUT", "libhyper_parallel_mindspore.so")).resolve()
    src_dir = Path(__file__).parent.resolve()

    # Clean previous build
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    # Configure CMake
    cmake = find_cmake()
    cmake_args = [
        cmake,
        "-S", str(src_dir),
        "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
    ]

    if "MINDSPORE_ROOT" in os.environ:
        cmake_args.append(f"-DMINDSPORE_ROOT={os.environ['MINDSPORE_ROOT']}")
        print(
            f"Using user-specified MINDSPORE_ROOT: {os.environ['MINDSPORE_ROOT']}")

    print(f"\nRunning CMake configure: {' '.join(cmake_args)}")
    result = subprocess.run(
        cmake_args,
        cwd=src_dir,
        capture_output=True,
        text=True,
        check=False
    )

    if result.returncode != 0:
        print("CMake configure FAILED:")
        print(result.stdout)
        print(result.stderr)
        if "MINDSPORE_ROOT" not in os.environ:
            print("\n💡 SUGGESTION: Set MINDSPORE_ROOT to enable custom pass compilation:")
            print("   Linux/macOS: export MINDSPORE_ROOT=/path/to/mindspore-2.8.1")
            print("   Windows:     set MINDSPORE_ROOT=C:\\path\\to\\mindspore-2.8.1")
        raise RuntimeError("CMake configuration failed")

    # Build plugin
    build_args = [cmake, "--build",
                  str(build_dir), "--target", "hyper_parallel_mindspore", "-j4"]
    print(f"\nRunning CMake build: {' '.join(build_args)}")
    result = subprocess.run(
        build_args,
        cwd=src_dir,
        capture_output=True,
        text=True,
        check=False
    )

    if result.returncode != 0:
        print("Build FAILED:")
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("Build failed")

    # Locate and copy SO file
    candidates = [
        build_dir / "libhyper_parallel_mindspore.so",
        build_dir / "Release" / "libhyper_parallel_mindspore.so",
        build_dir / "Debug" / "libhyper_parallel_mindspore.so",
    ]

    built_so = next((p for p in candidates if p.exists()), None)
    if not built_so:
        raise FileNotFoundError(
            f"Built SO not found. Searched: {[str(p) for p in candidates]}"
        )

    so_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built_so, so_output)
    print(f"\n✓ Plugin successfully built: {so_output}")
    print(f"  Size: {so_output.stat().st_size / 1024:.2f} KB")


if __name__ == "__main__":
    try:
        build_plugin()
    except Exception as e:
        print(f"\n❌ Build failed: {e}", file=sys.stderr)
        print("\n💡 This is OPTIONAL - HyperParallel works without MindSpore jit(ast) mode.", file=sys.stderr)
        # Non-zero exit for actual failures (setup.py will show warning)
        sys.exit(1)
