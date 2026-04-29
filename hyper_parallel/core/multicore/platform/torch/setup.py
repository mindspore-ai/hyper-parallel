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

import os
import glob
import tarfile
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension

import torch_npu
from torch_npu.utils.cpp_extension import NpuExtension

PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.abspath(torch_npu.__file__))
USE_NINJA = os.getenv('USE_NINJA') == '1'
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

source_files = glob.glob(os.path.join(BASE_DIR, "csrc", "*.cpp"), recursive=True)


def _find_prebuild_vendor_libdirs():
    """Return (fwd_libdir, bwd_libdir) from env vars or the prebuild directory.

    Resolution order:
      1. CANN_VENDOR_FWD_LIBDIR / CANN_VENDOR_BWD_LIBDIR  (explicit per-op)
      2. CANN_VENDOR_LIBDIR  (legacy single-lib; used for both)
      3. prebuild/multicore_moe_ffn/vendors/... (auto-detect + extract)
    """
    fwd = os.environ.get("CANN_VENDOR_FWD_LIBDIR",
          os.environ.get("CANN_VENDOR_LIBDIR", ""))
    bwd = os.environ.get("CANN_VENDOR_BWD_LIBDIR",
          os.environ.get("CANN_VENDOR_LIBDIR", ""))
    if fwd and bwd:
        return fwd, bwd
    # Auto-detect: prebuild dir is 2 levels up from platform/torch/ (→ multicore/)
    prebuild_dir = os.path.normpath(
        os.path.join(BASE_DIR, "../../prebuild/multicore_moe_ffn"))
    tarball = prebuild_dir + ".tar.gz"
    if not os.path.isdir(prebuild_dir) and os.path.isfile(tarball):
        print(f"[setup] Extracting prebuild: {tarball}")
        with tarfile.open(tarball) as tf:
            tf.extractall(os.path.dirname(prebuild_dir))
    vendors = os.path.join(prebuild_dir, "vendors")
    fwd = fwd or os.path.join(vendors, "multicore_moe_ffn_nn", "op_api", "lib")
    bwd = bwd or os.path.join(vendors, "multicore_moe_ffn_grad_nn", "op_api", "lib")
    return fwd, bwd


_FWD_LIBDIR, _BWD_LIBDIR = _find_prebuild_vendor_libdirs()


def _write_vendor_env_to_bashrc(libdirs):
    """Append ASCEND_CUSTOM_OPP_PATH / LD_LIBRARY_PATH exports to ~/.bashrc.

    Idempotent: each line is only appended if not already present.
    Required because g_custom_lib_path in libopapi.so is a const file-scope
    global initialised once when libopapi.so loads (triggered by
    ``import torch_npu``).  Setting ASCEND_CUSTOM_OPP_PATH from Python at
    import time is too late if torch_npu was already imported.  Persisting the
    variable in ~/.bashrc ensures it is present before Python starts.
    """
    bashrc = os.path.expanduser("~/.bashrc")
    try:
        with open(bashrc, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        content = ""

    added = []
    for libdir in dict.fromkeys(libdirs):
        if not libdir:
            continue
        root = libdir
        for sfx in ("/op_api/lib/", "/op_api/lib"):
            if root.endswith(sfx):
                root = root[: -len(sfx)]
                break
        opp_line = f'export ASCEND_CUSTOM_OPP_PATH="{root}:${{ASCEND_CUSTOM_OPP_PATH}}"'
        ld_line  = f'export LD_LIBRARY_PATH="{libdir}:${{LD_LIBRARY_PATH}}"'
        for line in (opp_line, ld_line):
            if line not in content:
                content += f"\n{line}"
                added.append(line)

    if added:
        with open(bashrc, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[setup] Appended to {bashrc}:")
        for line in added:
            print(f"  {line}")
        print(f"[setup] Run 'source {bashrc}' or start a new shell to apply.")
    else:
        print(f"[setup] {bashrc} already up to date.")


def _print_env_banner(libdirs):
    """Print a copy-pasteable environment setup banner."""
    sep = "=" * 64
    print(f"\n{sep}")
    print("  hyper_parallel multicore: environment setup")
    print(sep)
    for libdir in dict.fromkeys(libdirs):
        if not libdir:
            continue
        root = libdir
        for sfx in ("/op_api/lib/", "/op_api/lib"):
            if root.endswith(sfx):
                root = root[: -len(sfx)]
                break
        print(f'  export ASCEND_CUSTOM_OPP_PATH="{root}:${{ASCEND_CUSTOM_OPP_PATH}}"')
        print(f'  export LD_LIBRARY_PATH="{libdir}:${{LD_LIBRARY_PATH}}"')
    print(sep + "\n")


# Add -Wl,-rpath for each vendor lib dir (deduplicated) so the dynamic linker
# can find libcust_opapi.so at runtime without LD_LIBRARY_PATH.
_rpath_args = []
for _ldir in dict.fromkeys([_FWD_LIBDIR, _BWD_LIBDIR]):
    if _ldir and os.path.isdir(_ldir):
        _rpath_args.append(f"-Wl,-rpath,{_ldir}")

exts = []
ext = NpuExtension(
    name="hyper_parallel_multicore_moe_ffn_pta",
    sources=source_files,
    extra_compile_args=[
        '-I' + os.path.join(PYTORCH_NPU_INSTALL_PATH, "include/third_party/acl/inc"),
        '-I' + os.path.join(PYTORCH_NPU_INSTALL_PATH, "include/third_party/op-plugin"),
        '-I' + os.path.join(PYTORCH_NPU_INSTALL_PATH, "include/third_party/op-plugin/op_plugin/include"),
    ],
    extra_link_args=_rpath_args,
)
exts.append(ext)

setup(
    name="hyper_parallel_multicore_moe_ffn_pta",
    version='1.0',
    keywords='hyper_parallel_multicore_moe_ffn_pta',
    ext_modules=exts,
    packages=find_packages(),
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=USE_NINJA)},
)

_write_vendor_env_to_bashrc([_FWD_LIBDIR, _BWD_LIBDIR])
_print_env_banner([_FWD_LIBDIR, _BWD_LIBDIR])
