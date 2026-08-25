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
"""Build script for hyper_parallel MoE-FFN PyTorch extension."""

import glob
import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension

import torch_npu
from torch_npu.utils.cpp_extension import NpuExtension

PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.abspath(torch_npu.__file__))
USE_NINJA = os.getenv('USE_NINJA') == '1'
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

source_files = glob.glob(os.path.join(BASE_DIR, "csrc", "*.cpp"), recursive=True)


def _find_vendor_libdir():
    """Return the explicit unified vendor library directory."""
    vendor_root = os.environ.get("HP_MULTICORE_VENDOR_ROOT", "")
    vendor_libdir = os.environ.get("CANN_VENDOR_LIBDIR", "")
    if vendor_root and not vendor_libdir:
        vendor_libdir = os.path.join(vendor_root, "op_api", "lib")
    if not os.path.isfile(os.path.join(vendor_libdir, "libcust_opapi.so")):
        raise RuntimeError(
            "Unified multicore vendor is not prepared. Set HP_MULTICORE_VENDOR_ROOT "
            "or CANN_VENDOR_LIBDIR to the verified hyper_parallel_multicore_nn payload."
        )
    return vendor_libdir


_VENDOR_LIBDIR = _find_vendor_libdir()
_rpath_args = ["-Wl,--enable-new-dtags", "-Wl,-rpath,$ORIGIN/../../vendors/"
               "hyper_parallel_multicore_nn/op_api/lib"]


class RelocatableBuildExtension(BuildExtension.with_options(use_ninja=USE_NINJA)):
    """Remove build-environment RPATH flags inherited from Python sysconfig."""

    def build_extensions(self) -> None:
        """Keep link search flags while rejecting host-absolute runtime paths."""
        self.compiler.linker_so = [
            argument
            for argument in self.compiler.linker_so
            if not argument.startswith(("-Wl,-rpath,/", "-Wl,-rpath-link,/"))
        ]
        super().build_extensions()

exts = []
ext = NpuExtension(
    name="hyper_parallel_mega_moe_pta",
    sources=source_files,
    extra_compile_args=[
        # torch link path: ABI must match PyTorch / torch_npu (=1, official default since 2.7).
        '-D_GLIBCXX_USE_CXX11_ABI=1',
        '-I' + os.path.join(PYTORCH_NPU_INSTALL_PATH, "include/third_party/acl/inc"),
        '-I' + os.path.join(PYTORCH_NPU_INSTALL_PATH, "include/third_party/op-plugin"),
        '-I' + os.path.join(PYTORCH_NPU_INSTALL_PATH, "include/third_party/op-plugin/op_plugin/include"),
    ],
    extra_link_args=_rpath_args,
)
exts.append(ext)

setup(
    name="hyper_parallel_mega_moe_pta",
    version='1.0',
    keywords='hyper_parallel_mega_moe_pta',
    ext_modules=exts,
    packages=find_packages(),
    cmdclass={"build_ext": RelocatableBuildExtension},
)
