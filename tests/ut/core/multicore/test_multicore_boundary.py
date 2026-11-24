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

"""Verify component exports and absence of framework dispatch."""

import ast
import importlib
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import hyper_parallel
from hyper_parallel.core import multicore


class TestMulticoreBoundary(unittest.TestCase):
    """Check the component contract without initializing a native runtime."""

    @staticmethod
    def _run_cmake_project(
        cmake_lists: str,
        files: dict[str, str],
        *,
        build: bool = True,
        executable_files: tuple[str, ...] = (),
    ) -> subprocess.CompletedProcess[str]:
        """Configure and optionally build a temporary CMake project."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_root = Path(temporary_directory) / "source"
            build_root = Path(temporary_directory) / "build"
            source_root.mkdir()
            (source_root / "CMakeLists.txt").write_text(cmake_lists, encoding="utf-8")
            for relative_path, content in files.items():
                path = source_root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            for relative_path in executable_files:
                (source_root / relative_path).chmod(0o755)

            configure_result = subprocess.run(
                ["cmake", "-S", str(source_root), "-B", str(build_root)],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "TORCH_INSTALL_PREFIX": str(source_root / "polluted")},
            )
            if configure_result.returncode != 0 or not build:
                return configure_result

            build_result = subprocess.run(
                ["cmake", "--build", str(build_root), "--parallel", "2"],
                check=False,
                capture_output=True,
                text=True,
            )
            return subprocess.CompletedProcess(
                args=(configure_result.args, build_result.args),
                returncode=build_result.returncode,
                stdout=configure_result.stdout + build_result.stdout,
                stderr=configure_result.stderr + build_result.stderr,
            )

    @staticmethod
    def _fake_torch_stack(properties: str) -> dict[str, str]:
        """Provide only the package metadata consumed by the build helper."""
        return {
            "python": (
                f"#!{sys.executable}\n"
                "import os, sys\n"
                "os.execve(sys.executable, [sys.executable, '-S', *sys.argv[1:]], os.environ)\n"
            ),
            "packages/torch/__init__.py": "def compiled_with_cxx11_abi(): return True\n",
            "packages/torch_npu/__init__.py": "",
            "packages/torch/share/cmake/Caffe2/Caffe2Config.cmake": (
                "add_library(torch INTERFACE IMPORTED)\n"
            ),
            "packages/torch/share/cmake/Torch/TorchConfig.cmake": (
                "if(DEFINED ENV{TORCH_INSTALL_PREFIX})\n"
                '    message(FATAL_ERROR "TORCH_INSTALL_PREFIX was not cleared")\n'
                "endif()\n"
                "include(CMakeFindDependencyMacro)\n"
                "find_dependency(Caffe2)\n"
                f"{properties}\n"
            ),
        }

    def test_business_symbols_are_exported_only_by_multicore(self):
        """Verify the Multicore public export boundary.

        Feature: Multicore public exports.
        Description: Inspect the component and HyperParallel root package symbols.
        Expectation: Business modules and profiler are exported only by Multicore.
        """
        self.assertCountEqual(multicore.__all__, [
            "MegaMhc", "MegaGate", "MegaMoeExperts", "MegaKernelDiagnostics",
            "prepare_mega_kernel_diagnostics", "validate_mega_kernel_diagnostic_contract",
        ])
        self.assertTrue(callable(multicore.prepare_mega_kernel_diagnostics))
        for name in ("MhcState", "MegaMhcOutput", "GateOutput", "GateBackend"):
            self.assertFalse(hasattr(multicore, name))
        for name in (
            "HyperMegaMhc",
            "MegaMhc",
            "MhcState",
            "MegaMhcOutput",
            "GateOutput",
            "GateBackend",
            "MegaGate",
            "MegaMoeExperts",
            "MulticoreModule",
            "mega_moe",
            "mega_moe_grad",
        ):
            self.assertNotIn(name, hyper_parallel.__all__)
            self.assertFalse(hasattr(hyper_parallel, name))
        self.assertNotIn("hyper_parallel_shmem_torch", sys.modules)

    def test_mhc_exposes_only_canonical_module_name(self) -> None:
        """Feature: Single public mHC module name.
        Description: Import all business classes with the NPU backend blocked, then inspect mHC exports.
        Expectation: CPU imports need no HCCL; backend failures propagate and the canonical mHC identity is preserved.
        """
        code = """
import importlib.abc
import sys

class RejectNpuBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch_npu":
            raise ImportError("CPU import guard: libhccl.so unavailable")
        return None

sys.meta_path.insert(0, RejectNpuBackend())
if sys.argv[1] == "backward":
    backward = importlib.import_module("hyper_parallel.core.multicore.modules.mega_mhc_grad.function")
    assert callable(backward.hyper_mega_mhc_grad)
from hyper_parallel.core.multicore import MegaGate, MegaMhc, MegaMoeExperts
from hyper_parallel.core.multicore._loader import load_torch_npu
assert MegaGate.__name__ == "MegaGate"
assert MegaMhc.__name__ == "MegaMhc"
assert MegaMoeExperts.__name__ == "MegaMoeExperts"
import torch
from hyper_parallel.core.multicore.torch import ops, mega_gate_ops
router = MegaGate(hidden_size=4, num_experts=2, top_k=1, scoring_func="sigmoid", routed_scaling_factor=1.0)
assert router(torch.ones(1, 2, 4))[2].shape == (2, 1)
assert MegaMhc(128).norm_weight.shape == (128,)
experts = MegaMoeExperts(hidden_size=4, intermediate_size=4, num_experts=2, top_k=1)
try:
    experts(torch.ones(1, 2, 4, dtype=torch.bfloat16), torch.zeros(2, 1, dtype=torch.int32), torch.ones(2, 1))
except TypeError as error:
    assert "NPU hidden states" in str(error)
else:
    raise AssertionError("CPU expert execution must reject inputs before native initialization")
experts.close()
assert "torch_npu" not in sys.modules
for initialize in (ops._load_native, mega_gate_ops._load_native):
    try:
        initialize()
    except ImportError as error:
        assert "libhccl.so" in str(error)
    else:
        raise AssertionError("Backend registration must fail when HCCL is unavailable")
assert load_torch_npu.cache_info().currsize == 0
"""
        for order in ("forward", "backward"):
            result = subprocess.run(
                [sys.executable, "-c", code, order], check=False, capture_output=True, text=True,
                env={**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"},
            )
            self.assertEqual(result.returncode, 0, msg=f"{order}: {result.stdout}{result.stderr}")
        package = importlib.import_module("hyper_parallel.core.multicore.modules.mega_mhc")
        implementation = importlib.import_module("hyper_parallel.core.multicore.modules.mega_mhc.module")
        for namespace in (multicore, package, implementation):
            with self.subTest(namespace=namespace.__name__):
                self.assertIs(namespace.MegaMhc, multicore.MegaMhc)
                self.assertNotIn("HyperMegaMhc", getattr(namespace, "__all__", ()))
                self.assertFalse(hasattr(namespace, "HyperMegaMhc"))

    def test_runtime_sources_import_no_platform_layer(self):
        """Verify production sources stay Torch-only.

        Feature: Multicore runtime convergence.
        Description: Parse imports in the converged production Python tree.
        Expectation: Runtime sources import no HyperParallel Platform layer.
        """
        root = Path(multicore.__file__).parent
        for path in root.rglob("*.py"):
            # Benchmark examples import common MoE as an independent reference.
            if any(part in {"lib", "examples"} for part in path.relative_to(root).parts):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                elif isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                else:
                    continue
                for module in modules:
                    with self.subTest(path=path, module=module):
                        self.assertFalse(module.startswith("hyper_parallel.platform"))

    def test_native_mega_kernels_share_cached_op_api_bridge(self):
        """Feature: Shared ACLNN bridge.
        Description: Inspect the native delegation and existing operator call sites.
        Expectation: Operators use one wrapper backed by the native V1/V2 macros.
        """
        csrc = Path(multicore.__file__).parent / "torch" / "csrc"
        header = (csrc / "cached_op_api.h").read_text(encoding="utf-8")
        for macro in ("EXEC_NPU_CMD_V1_EXT", "EXEC_NPU_CMD_V2_EXT"):
            self.assertIn(macro, header, msg=f"Expected native delegation={macro}, present={macro in header}")
        callers = (
            "mega_gate.cpp", "mega_mhc.cpp", "mega_moe.cpp", "mega_moe_grad.cpp",
            "mega_moe_unpermute.cpp", "moe_token_permute_grad.cpp", "moe_token_permute_out.cpp",
        )
        for name in callers:
            with self.subTest(caller=name):
                source = (csrc / name).read_text(encoding="utf-8")
                self.assertIn(
                    "HYPER_EXEC_NPU_CMD_EXT", source,
                    msg=f"{name}: expected shared wrapper, present={'HYPER_EXEC_NPU_CMD_EXT' in source}",
                )
                direct = re.search(r"(?<!HYPER_)EXEC_NPU_CMD_(?:V1_|V2_)?EXT\s*\(", source)
                self.assertIsNone(direct, msg=f"{name}: expected no direct native calls, actual={direct!r}")

    def test_torch_build_contract_follows_active_stack(self):
        """Feature: Torch native build compatibility.
        Description: Compile legacy/interface contracts and reject unusable metadata.
        Expectation: Roots, ABI, minimum standard and environment restoration match the active stack.
        """
        cmake_module = (Path(multicore.__file__).parent / "cmake" / "torch_compat.cmake").as_posix()
        prefix = f'''cmake_minimum_required(VERSION 3.18)
project(torch_contract_test LANGUAGES CXX)
set(ENV{{PYTHONPATH}} "${{CMAKE_CURRENT_SOURCE_DIR}}/packages")
include("{cmake_module}")
hp_detect_torch_build_contract("${{CMAKE_CURRENT_SOURCE_DIR}}/python" STACK)
if(NOT "$ENV{{TORCH_INSTALL_PREFIX}}" STREQUAL "${{CMAKE_CURRENT_SOURCE_DIR}}/polluted")
    message(FATAL_ERROR "TORCH_INSTALL_PREFIX was not restored")
endif()
'''
        successes = (
            ("set_property(TARGET torch PROPERTY CXX_STANDARD 17)", 17, 201703),
            ("set_property(TARGET torch PROPERTY CXX_STANDARD 20)", 20, 201709),
            ("target_compile_features(torch INTERFACE cxx_std_17 cxx_std_20)", 20, 201709),
        )
        for properties, standard, minimum in successes:
            with self.subTest(properties=properties):
                cmake_lists = prefix
                for field, expected in (
                    ("CXX_STANDARD", standard), ("CXX11_ABI", 1),
                    ("PACKAGE_ROOT", "${CMAKE_CURRENT_SOURCE_DIR}/packages/torch"),
                    ("NPU_PACKAGE_ROOT", "${CMAKE_CURRENT_SOURCE_DIR}/packages/torch_npu"),
                ):
                    cmake_lists += f'''
if(NOT STACK_{field} STREQUAL "{expected}")
    message(FATAL_ERROR "{field}: expected={expected}, actual=${{STACK_{field}}}")
endif()
'''
                cmake_lists += '''
add_library(probe STATIC probe.cpp)
hp_apply_torch_cxx_standard(probe "${STACK_CXX_STANDARD}")
'''
                files = self._fake_torch_stack(properties)
                files["probe.cpp"] = f'static_assert(__cplusplus >= {minimum}L, "C++ standard too low");\n'
                result = self._run_cmake_project(cmake_lists, files, executable_files=("python",))
                self.assertEqual(
                    result.returncode, 0,
                    msg=f"Expected successful build, actual={result.returncode}: {result.stdout}{result.stderr}",
                )

        failures = (
            ("set_property(TARGET torch PROPERTY CXX_STANDARD 14)", "", "requires at least C++17"),
            ('target_compile_options(torch INTERFACE "-std=c++20")', "", "recognized C++ standard"),
            ("", "", "recognized C++ standard"),
            ("", "Torch", "missing a CMake config"),
            ("", "Caffe2", "missing a CMake config"),
        )
        for properties, missing_package, diagnostic in failures:
            with self.subTest(properties=properties, missing_package=missing_package):
                files = self._fake_torch_stack(properties)
                if missing_package:
                    del files[f"packages/torch/share/cmake/{missing_package}/{missing_package}Config.cmake"]
                result = self._run_cmake_project(prefix, files, build=False, executable_files=("python",))
                output = result.stdout + result.stderr
                self.assertNotEqual(
                    result.returncode, 0, msg=f"Expected configuration failure, actual={result.returncode}: {output}",
                )
                self.assertIn(diagnostic, output, msg=f"Expected diagnostic={diagnostic!r}, actual={output!r}")

    def test_torch_native_targets_use_shared_build_contract(self):
        """Feature: Torch native build compatibility.
        Description: Inspect adapter and SHMEM CMake integration boundaries.
        Expectation: Torch targets share the probe, host code stays C++17, no Torch version probing.
        """
        root = Path(multicore.__file__).parent
        sources = {name: (root / name).read_text(encoding="utf-8") for name in (
            "cmake/torch_compat.cmake", "torch/CMakeLists.txt", "shmem/ccsrc/CMakeLists.txt",
        )}
        for name, target_count in (("torch/CMakeLists.txt", 2), ("shmem/ccsrc/CMakeLists.txt", 1)):
            for call, expected in (("hp_detect_torch_build_contract(", 1),
                                   ("hp_apply_torch_cxx_standard(", target_count)):
                actual = sources[name].count(call)
                self.assertEqual(actual, expected, msg=f"{name}: {call} expected={expected}, actual={actual}")
        host_standard = "set(CMAKE_CXX_STANDARD 17)"
        shmem = sources["shmem/ccsrc/CMakeLists.txt"]
        self.assertIn(host_standard, shmem, msg=f"Expected {host_standard}, present={host_standard in shmem}")
        adapter = sources["torch/CMakeLists.txt"]
        definitions = re.findall(r"CURRENT_VERSION=([^\s)]+)", adapter)
        self.assertEqual(
            definitions, ["VERSION_NEWEST", "VERSION_NEWEST"],
            msg=f"Expected two VERSION_NEWEST definitions, actual={definitions}",
        )
        for forbidden in ("torch.__version__", "torch.__config__.show()"):
            present = [name for name, source in sources.items() if forbidden in source]
            self.assertEqual(present, [], msg=f"Expected no {forbidden} consumer, actual={present}")

    def test_diagnostics_dependency_boundaries(self):
        """Feature: MegaKernel diagnostics.
        Description: Keep common code tool-neutral and both tools independent of business adapters.
        Expectation: Diagnostics dependency boundaries.
        """
        root = Path(multicore.__file__).parent / "diagnostics"
        for path in root.rglob("*.py"):
            relative = path.relative_to(root)
            forbidden = ["hyper_parallel.core.multicore.modules"]
            if relative.parts[0] in ("common", "api.py", "__init__.py"):
                forbidden.extend(("diagnostics.profiler", "diagnostics.dump"))
            elif relative.parts[0] == "dump":
                forbidden.append("diagnostics.profiler")
            elif relative.parts[0] == "profiler":
                forbidden.append("diagnostics.dump")
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                modules = ([node.module or ""] if isinstance(node, ast.ImportFrom)
                           else [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
                for module in modules:
                    with self.subTest(path=path, module=module):
                        self.assertFalse(any(item in module for item in forbidden))
