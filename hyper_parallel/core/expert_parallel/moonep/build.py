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
"""Build shared MoonEP placement and optional MegaMoE SHMEM transport.

The default allocator build is independent of the MegaMoE SDK. Use --shared
only when the MegaMoE execution adapter is required.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sysconfig
from pathlib import Path

import torch
import torch_npu


def build_allocator(output_dir: Path) -> None:
    """Compile the common placement operator with the selected Torch/CANN ABI."""
    root = Path(__file__).resolve().parent
    cann = Path(os.environ["ASCEND_HOME_PATH"])
    torch_root, npu_root = Path(torch.__file__).parent, Path(torch_npu.__file__).parent
    includes = [torch_root / "include", torch_root / "include/torch/csrc/api/include",
                npu_root / "include", Path(sysconfig.get_path("include")), root / "csrc",
                cann / "asc", cann / "asc/include", cann / "asc/include/interface",
                cann / "asc/include/utils", cann / "asc/impl", cann / "include", cann / "include/runtime",
                cann / "pkg_inc", cann / "pkg_inc/runtime",
                cann / f"{platform.machine()}-linux/include", cann / f"{platform.machine()}-linux/pkg_inc"]
    opdev_headers = sorted(cann.rglob("bfloat16.h"))
    includes.extend(path.parent.parent for path in opdev_headers if path.parent.name == "opdev")
    command = ["bisheng", "-x", "asc", "--npu-arch=dav-2201", "--cce-auto-sync=off",
               "-shared", "-fPIC", "-O3", "-std=c++17",
               f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
               "-DTORCH_EXTENSION_NAME=_a3_allocator", str(root / "csrc/a3_allocator.asc"),
               "-o", str(output_dir / "_a3_allocator.so")]
    command.extend(f"-I{path}" for path in includes if path.is_dir())
    command.extend(f"-L{path}" for path in [torch_root / "lib", npu_root / "lib", cann / "lib64"])
    command.extend(["-ltorch_npu", "-ltorch", "-lc10", "-ltorch_python"])
    subprocess.run(command, check=True)


def stage_allocator(package_root: Path, output_dir: Path) -> None:
    """Stage the common allocator without any MegaMoE or SHMEM library dependency."""
    source = output_dir / "_a3_allocator.so"
    if not source.is_file():
        raise FileNotFoundError("Build the MoonEP allocator before staging it")
    target = package_root / "core/expert_parallel/moonep"
    target.mkdir(parents=True, exist_ok=True)
    if source.resolve() != (target / source.name).resolve():
        shutil.copy2(source, target / source.name)


def build_shared(output_dir: Path) -> None:
    """Compile transfers against the exact private SDK used by the HP native build."""
    root = Path(__file__).resolve().parent
    repo = next(parent for parent in root.parents
                if (parent / "hyper_parallel/core/multicore/_build/dependencies.lock.json").is_file())
    record = json.loads((repo / "build/native/work/multicore/shmem/sdk.json").read_text(encoding="utf-8"))
    sdk = Path(record["install_root"]) / "shmem"
    cann = Path(os.environ["ASCEND_HOME_PATH"])
    torch_root, npu_root = Path(torch.__file__).parent, Path(torch_npu.__file__).parent
    includes = [root / "csrc", sdk / "include", sdk / "include/device", sdk / "src",
                sdk / "src/device", torch_root / "include", torch_root / "include/torch/csrc/api/include",
                npu_root / "include", Path(sysconfig.get_path("include")), cann / "include",
                cann / "include/runtime", cann / "pkg_inc", cann / "pkg_inc/runtime",
                cann / "asc", cann / "asc/include", cann / "asc/include/interface", cann / "asc/include/utils",
                cann / "compiler/tikcpp", cann / "compiler/tikcpp/tikcfw", cann / "compiler/tikcpp/tikcfw/impl",
                cann / "compiler/tikcpp/tikcfw/interface"]
    includes.extend(path.parent.parent for path in cann.glob("include/**/opdev/bfloat16.h"))
    inc = [f"-I{path}" for path in includes if path.is_dir()]
    libs = [f"-L{path}" for path in [torch_root / "lib", npu_root / "lib", sdk / "lib", cann / "lib64"]]
    common = ["-shared", "-fPIC", "-O2", "-std=c++17",
              f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}"]
    runtime = ["-lhyper_parallel_shmem", "-lhyper_parallel_shmem_utils", "-lascendcl", "-lruntime", "-ldl"]
    subprocess.run(["bisheng", "-x", "asc", "--npu-arch=dav-2201", "--cce-auto-sync=off",
                    "--cce-fatobj-link", *common, str(root / "csrc/moonep_expert_prefetch.cpp"),
                    str(root / "csrc/moonep_expert_grad_reduce_bf16.cpp"), "-o",
                    str(output_dir / "libhp_moonep_shared_kernels.so"),
                    "-Wl,-soname,libhp_moonep_shared_kernels.so", *inc, *libs, *runtime], check=True)
    subprocess.run(["g++", *common, "-DTORCH_EXTENSION_NAME=_moonep_shared_bridge",
                    str(root / "csrc/moonep_shared_bridge.cpp"), "-o", str(output_dir / "_moonep_shared_bridge.so"),
                    *inc, f"-L{output_dir}", *libs, "-lhp_moonep_shared_kernels", "-ltorch_npu", "-ltorch",
                    "-lc10", "-ltorch_python", *runtime], check=True)


def stage_artifacts(package_root: Path, output_dir: Path) -> None:
    """Add only MoonEP libraries to an existing HP native payload package root."""
    source = output_dir
    names = ("_moonep_shared_bridge.so", "libhp_moonep_shared_kernels.so")
    missing = [name for name in names if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Build MoonEP with --shared before staging; missing {missing}")
    target = package_root / "core/expert_parallel/moonep"
    target.mkdir(parents=True, exist_ok=True)
    for name in names:
        if (source / name).resolve() != (target / name).resolve():
            shutil.copy2(source / name, target / name)
    stage_allocator(package_root, output_dir)


def main() -> None:
    """Compile only the selected backend and stage its native artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared", action="store_true", help="Also build MegaMoE SHMEM expert transport")
    parser.add_argument("--shared-only", action="store_true", help="Reuse the allocator and build SHMEM transport")
    parser.add_argument("--stage-to", type=Path, help="HP payload package root for wheel assembly")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[4] /
                        "build/native/payload/hyper_parallel/core/expert_parallel/moonep",
                        help="Directory for generated libraries; defaults to HP native payload")
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    if not arguments.shared_only:
        build_allocator(arguments.output_dir)
    if arguments.shared or arguments.shared_only:
        build_shared(arguments.output_dir)
    if arguments.stage_to is not None:
        if arguments.shared or arguments.shared_only:
            stage_artifacts(arguments.stage_to, arguments.output_dir)
        else:
            stage_allocator(arguments.stage_to, arguments.output_dir)


if __name__ == "__main__":
    main()
