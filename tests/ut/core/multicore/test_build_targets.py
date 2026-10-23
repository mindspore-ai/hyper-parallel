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
# WITHOUT WARRANTIES OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Hardware-independent contracts for Multicore build targets and private SHMEM SDKs."""

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import hyper_parallel


_ROOT = Path(hyper_parallel.__file__).resolve().parent.parent
_SDK = _ROOT / "hyper_parallel/core/multicore/shmem/_build/shmem_sdk.sh"
# Profile selection does not use the CANN version helper required by full SDK preparation.
_SOC_QUERY = 'set -e; hp_cann_version_at_least() { return 0; }; source "$1"; hp_shmem_soc_type "$2"'


class TestBuildTargets(unittest.TestCase):
    """Validate real shell profile selection without compiling or initializing a device."""

    def test_runtime_backend_accepts_only_its_device_generation(self) -> None:
        """Compile both host backend contracts without Torch, CANN or an NPU."""
        source = '''
#include <string_view>
#include "device/device_model.h"
using namespace hyper_parallel::multicore::shmem;
#ifdef HP_SHMEM_ASCEND950
constexpr bool is_a5 = true;
#else
constexpr bool is_a5 = false;
#endif
static_assert(BackendSupportsDevice(DeviceModel::kAscend950) == is_a5);
static_assert(BackendSupportsDevice(DeviceModel::kAscend910B) == !is_a5);
static_assert(BackendSupportsDevice(DeviceModel::kAscend910C) == !is_a5);
static_assert(!BackendSupportsDevice(static_cast<DeviceModel>(255)));
static_assert(std::string_view(kBackendSocType) == (is_a5 ? "Ascend950" : "Ascend910B"));
int main() { return 0; }
'''
        include = _ROOT / "hyper_parallel/core/multicore/shmem/ccsrc"
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "backend-contract"
            for definitions in ([], ["-DHP_SHMEM_ASCEND950"]):
                with self.subTest(definitions=definitions):
                    result = subprocess.run(
                        ["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror", f"-I{include}", *definitions,
                         "-x", "c++", "-", "-o", str(executable)], input=source, capture_output=True,
                        check=False, text=True)
                    self.assertEqual(result.returncode, 0, f"expected=0 actual={result.returncode}: {result.stderr}")
                    result = subprocess.run([str(executable)], capture_output=True, check=False, text=True)
                    self.assertEqual(result.returncode, 0, f"expected=0 actual={result.returncode}: {result.stderr}")

    def test_grouped_matmul_helper_architecture_selection(self) -> None:
        """Retain the A2/A3 helper and expose it on A5 without redefining device macros."""
        component = _ROOT / "hyper_parallel/core/multicore"
        adapter = component / "ops/grouped_matmul_fusion.patch"
        lock = json.loads((component / "_build/dependencies.lock.json").read_text(encoding="utf-8"))
        expected_hash = lock["components"]["multicore"]["ops_transformer"]["patches"][0]["sha256"]
        actual_hash = hashlib.sha256(adapter.read_bytes()).hexdigest()
        self.assertEqual(actual_hash, expected_hash, f"expected={expected_hash} actual={actual_hash}")
        helper_diff = adapter.read_text(encoding="utf-8").split(
            "diff --git a/gmm/grouped_matmul/op_kernel/grouped_matmul_utils.h", 1)[1]
        guard = "\n".join(line[1:] for line in helper_diff.splitlines()
                          if line.startswith("+") and not line.startswith("+++"))
        source = f"{guard}\nHP_GROUP_LIST_HELPER\n#endif\n"
        for core, architecture, expected in ((220, 2201, True), (310, 3510, True), (310, 3113, True),
                                             (310, 9999, False)):
            with self.subTest(core=core, architecture=architecture):
                result = subprocess.run(
                    ["c++", "-E", "-P", "-x", "c++", f"-D__CCE_AICORE__={core}",
                     f"-D__NPU_ARCH__={architecture}", "-"], input=source, capture_output=True,
                    check=False, text=True)
                self.assertEqual(result.returncode, 0, f"expected=0 actual={result.returncode}: {result.stderr}")
                actual = "HP_GROUP_LIST_HELPER" in result.stdout
                self.assertEqual(actual, expected, f"expected={expected} actual={actual}: {result.stdout}")

    def test_vendor_kernel_compute_units(self) -> None:
        """All six kernels use the selected CANN target, including the 950 target."""
        kernel_cmake = _ROOT / "hyper_parallel/core/multicore/cmake/vendor/op_kernel/CMakeLists.txt"
        script = '''
function(npu_op_kernel_library)
endfunction()
function(npu_op_kernel_options)
endfunction()
function(npu_op_kernel_sources)
    cmake_parse_arguments(KERNEL "" "OP_TYPE;COMPUTE_UNIT" "" ${ARGN})
    message(STATUS "HP_KERNEL=${KERNEL_OP_TYPE}:${KERNEL_COMPUTE_UNIT}")
endfunction()
function(npu_op_package_add)
endfunction()
include("${HP_KERNEL_CMAKE}")
'''
        targets = (("ascend910b", "ascend910b1"), ("ascend910_93", "ascend910_9391"),
                   ("ascend950", "ascend950pr_9599"))
        with tempfile.TemporaryDirectory() as temporary:
            script_path = Path(temporary) / "kernel-targets.cmake"
            script_path.write_text(script, encoding="utf-8")
            for selection, expected in targets:
                with self.subTest(selection=selection):
                    result = subprocess.run(
                        ["cmake", f"-DASCEND_COMPUTE_UNIT={selection}", f"-DHP_KERNEL_CMAKE={kernel_cmake}",
                         "-P", str(script_path)], capture_output=True, check=False, text=True)
                    self.assertEqual(result.returncode, 0, f"expected=0 actual={result.returncode}: {result.stderr}")
                    kernels = [line for line in result.stdout.splitlines() if "HP_KERNEL=" in line]
                    self.assertEqual(len(kernels), 6, f"expected=6 actual={kernels}")
                    for kernel in kernels:
                        self.assertTrue(kernel.endswith(f":{expected}"), f"expected={expected} actual={kernel}")

    def test_swiglu_bf16_downcasts_use_supported_rounding(self) -> None:
        """Preprocess the actual forward/backward adapter hunks for A2/A3 and A5."""
        component = _ROOT / "hyper_parallel/core/multicore"
        adapter = component / "ops/swi_glu_fusion.patch"
        lock = json.loads((component / "_build/dependencies.lock.json").read_text(encoding="utf-8"))
        expected_hash = lock["components"]["multicore"]["ops_nn"]["patches"][0]["sha256"]
        actual_hash = hashlib.sha256(adapter.read_bytes()).hexdigest()
        self.assertEqual(actual_hash, expected_hash, f"expected={expected_hash} actual={actual_hash}")
        for name in ("swi_glu", "swi_glu_grad"):
            path = f"activation/{name}/op_kernel/{name}_bf16.hpp"
            section = adapter.read_text(encoding="utf-8").split(f"diff --git a/{path} b/{path}\n", 1)[1]
            section = section.split("diff --git", 1)[0].split("@@", 2)[2]
            source = "\n".join(line[1:] for line in section.splitlines()
                               if line.startswith(("+", " ")))
            for core, architecture, expected in ((220, 2201, "CAST_RINT"), (220, 2203, "CAST_RINT"),
                                                 (310, 3510, "CAST_RINT"), (310, 3113, "CAST_NONE")):
                with self.subTest(kernel=name, core=core, architecture=architecture):
                    result = subprocess.run(
                        ["c++", "-E", "-P", "-x", "c++", f"-D__CCE_AICORE__={core}",
                         f"-D__NPU_ARCH__={architecture}", "-"], input=source, capture_output=True,
                        check=False, text=True)
                    self.assertEqual(result.returncode, 0, f"expected=0 actual={result.returncode}: {result.stderr}")
                    self.assertIn(f"RoundMode::{expected}", result.stdout,
                                  f"expected={expected} actual={result.stdout}")
                    other = "CAST_NONE" if expected == "CAST_RINT" else "CAST_RINT"
                    self.assertNotIn(other, result.stdout, f"expected={other} absent actual={result.stdout}")

    def test_vendor_host_registers_all_device_generations(self) -> None:
        """Every forward and backward definition advertises the same three SoCs."""
        definitions = list((_ROOT / "hyper_parallel/core/multicore/ops").glob("*/op_host/*_def.cpp"))
        self.assertEqual(len(definitions), 6, f"expected=6 actual={definitions}")
        for definition in definitions:
            source = definition.read_text(encoding="utf-8")
            for soc in ("ascend910b", "ascend910_93", "ascend950"):
                with self.subTest(operator=definition.parent.parent.name, soc=soc):
                    self.assertIn(f'AddConfig("{soc}",', source, f"expected={soc} actual={definition}")

    def test_worker_and_trace_select_the_actual_core_type(self) -> None:
        """Both device generations must compile Cube queues and profiler slots on Cube cores."""
        runtime = _ROOT / "hyper_parallel/core/multicore/ops/runtime"
        source = "\n".join(line for name in ("worker_kernel.h", "cycle_trace_recorder.h")
                           for line in (runtime / name).read_text(encoding="utf-8").splitlines()
                           if not line.startswith("#include"))
        for generation in ("C220", "C310"):
            for core in ("CUBE", "VEC"):
                with self.subTest(generation=generation, core=core):
                    result = subprocess.run(
                        ["c++", "-E", "-P", "-x", "c++", f"-D__DAV_{core}__=1",
                         f"-D__DAV_{generation}_{core}__=1", "-"], input=source, capture_output=True,
                        check=False, text=True)
                    self.assertEqual(result.returncode, 0, f"expected=0 actual={result.returncode}: {result.stderr}")
                    is_cube = core == "CUBE"
                    for fragment in ("return cube_task_indexs.GetValue(task_id);", "uint32_t core_type = 1;",
                                     "getGroupedMatmulGroupListOffsetById(runtimeConfigPtr, worker_id)"):
                        actual = fragment in result.stdout
                        self.assertEqual(actual, is_cube, f"expected={is_cube} actual={actual}: {fragment}")
                    actual = "return vector_task_indexs.GetValue(task_id);" in result.stdout
                    self.assertEqual(actual, not is_cube, f"expected={not is_cube} actual={actual}")

    def test_sdk_soc_selection(self) -> None:
        """Keep A2/A3 together and isolate the A5 SDK family."""
        for selection, expected in (("ascend910b", "Ascend910B"), ("ascend910_93", "Ascend910B"),
                                    ("ascend910b,ascend910_93", "Ascend910B"), (" ascend950 ", "Ascend950")):
            with self.subTest(selection=selection):
                result = subprocess.run(["bash", "-c", _SOC_QUERY,
                                         "sdk-profile-test", str(_SDK), selection], capture_output=True,
                                        check=False, text=True)
                self.assertEqual(result.returncode, 0, f"expected=0 actual={result.returncode}: {result.stderr}")
                self.assertEqual(result.stdout.strip(), expected, f"expected={expected} actual={result.stdout.strip()}")
                self.assertEqual(result.stderr, "", f"expected empty stderr actual={result.stderr}")

    def test_sdk_rejects_mixed_or_unknown_families(self) -> None:
        """A single SDK must never silently serve the wrong device generation."""
        for selection in ("ascend910b,ascend950", "ascend950,ascend910_93", "ascend970", ""):
            with self.subTest(selection=selection):
                result = subprocess.run(["bash", "-c", _SOC_QUERY,
                                         "sdk-profile-test", str(_SDK), selection], capture_output=True,
                                        check=False, text=True)
                self.assertEqual(result.returncode, 10, f"expected=10 actual={result.returncode}: {result.stderr}")
                self.assertEqual(result.stdout, "", f"expected=empty actual={result.stdout!r}")


if __name__ == "__main__":
    unittest.main()
