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
"""Hardware-independent contracts for the optional Gram NS adapter."""

import importlib
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.components.functional import make_gram_newton_schulz

gram_module = importlib.import_module("hyper_parallel.components.functional.gram_newton_schulz")
_COEFFICIENTS = [(3.4445, -4.7750, 2.0315)] * 5


class TestGramNewtonSchulz(unittest.TestCase):
    """Validate dependency isolation and binding independently of CUDA kernels."""

    def test_public_import_does_not_load_optional_package(self) -> None:
        """Importing the public factory must work with the dependency blocked."""
        script = """
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('gram_newton_schulz', 'quack', 'cutlass'):
            raise RuntimeError('unexpected optional import: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
from hyper_parallel.components.functional import make_gram_newton_schulz
assert callable(make_gram_newton_schulz)
"""
        subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)

    def test_missing_or_broken_package_preserves_cause(self) -> None:
        """Missing dependencies and library failures must explain how to install."""
        for error in (ModuleNotFoundError("missing dependency"), OSError("missing shared library")):
            with self.subTest(error=error), patch.object(gram_module.importlib, "import_module", side_effect=error):
                with self.assertRaisesRegex(ImportError, "README") as caught:
                    make_gram_newton_schulz(_COEFFICIENTS)
                self.assertIs(caught.exception.__cause__, error)

    def test_missing_api_is_rejected(self) -> None:
        """An installed package with an incompatible public API must fail early."""
        with patch.object(gram_module.importlib, "import_module", return_value=SimpleNamespace()):
            with self.assertRaisesRegex(ImportError, "GramNewtonSchulz"):
                make_gram_newton_schulz(_COEFFICIENTS)

    def test_configuration_is_validated_before_import(self) -> None:
        """Invalid user settings must not require an optional installation."""
        configurations = (
            {"ns_coefficients": []},
            {"ns_coefficients": [(1, 2)]},
            {"ns_coefficients": [(1, 2, float("nan"))]},
            {"epsilon": 0}, {"epsilon": -1}, {"epsilon": float("inf")},
            {"reset_iterations": [0]}, {"reset_iterations": [True]},
            {"reset_iterations": [1.5]}, {"use_kernels": "auto"}, {"compile_kwargs": []},
        )
        with patch.object(gram_module, "_load_gram_newton_schulz") as loader:
            for overrides in configurations:
                with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                    make_gram_newton_schulz(**{"ns_coefficients": _COEFFICIENTS, **overrides})
            loader.assert_not_called()

    def test_step_schedule_cache_and_configuration_ownership(self) -> None:
        """Step prefixes/restarts must be bound once and insulated from mutation."""
        implementation = Mock(side_effect=lambda **kwargs: lambda inputs: inputs.clone())
        coefficients = [list(row) for row in _COEFFICIENTS]
        compile_kwargs = {"options": {"max_autotune": False}}
        with patch.object(gram_module, "_load_gram_newton_schulz", return_value=implementation):
            callback = make_gram_newton_schulz(
                coefficients, epsilon=1e-8, reset_iterations=[4, 2, 2],
                use_kernels=False, compile_kwargs=compile_kwargs,
            )
        coefficients[0][0] = 999
        compile_kwargs["options"]["max_autotune"] = True
        inputs = torch.randn(2, 4, 8, dtype=torch.bfloat16)
        for steps in (1, 5, 1, 5):
            result = callback(inputs, steps=steps)
            torch.testing.assert_close(result, inputs)
            self.assertEqual(result.dtype, inputs.dtype)
            self.assertEqual(result.device, inputs.device)
            result.mul_(0)
            self.assertGreater(inputs.abs().sum().item(), 0)
        self.assertEqual(implementation.call_count, 2)
        first, second = [call.kwargs for call in implementation.call_args_list]
        self.assertEqual(first["ns_coefficients"], [list(_COEFFICIENTS[0])])
        self.assertEqual(first["gram_newton_schulz_reset_iterations"], [])
        self.assertEqual(second["gram_newton_schulz_reset_iterations"], [2, 4])
        self.assertEqual(second["ns_epsilon"], 1e-8)
        self.assertEqual(second["compile_kwargs"], {"options": {"max_autotune": False}})

    def test_invalid_inputs_and_steps_do_not_construct_backend(self) -> None:
        """Reject unsupported callback inputs at the adapter boundary."""
        implementation = Mock()
        with patch.object(gram_module, "_load_gram_newton_schulz", return_value=implementation):
            callback = make_gram_newton_schulz(_COEFFICIENTS, use_kernels=False)
        for steps in (0, -1, 6, True, 1.5):
            with self.subTest(steps=steps), self.assertRaisesRegex(ValueError, "steps"):
                callback(torch.ones(2, 2), steps=steps)
        for inputs in (torch.ones(2), torch.ones(1, 1, 2, 2), torch.ones(2, 2, dtype=torch.int32)):
            with self.subTest(shape=inputs.shape), self.assertRaises(ValueError):
                callback(inputs, steps=1)
        implementation.assert_not_called()

    def test_custom_kernels_reject_cpu_without_fallback(self) -> None:
        """Explicit custom kernels must never silently run a different backend."""
        implementation = Mock()
        with patch.object(gram_module, "_load_gram_newton_schulz", return_value=implementation):
            callback = make_gram_newton_schulz(_COEFFICIENTS, use_kernels=True)
        with self.assertRaisesRegex(ValueError, "CUDA"):
            callback(torch.ones(2, 2), steps=5)
        implementation.assert_not_called()

    def test_default_selects_torch_for_cpu_without_probing_cuda(self) -> None:
        """Default selection must follow the input even on a host with available GPUs."""
        implementation = Mock(side_effect=lambda **kwargs: lambda inputs: inputs.clone())
        with patch.object(gram_module, "_load_gram_newton_schulz", return_value=implementation):
            callback = make_gram_newton_schulz(_COEFFICIENTS)
        with patch.object(torch.cuda, "get_device_capability") as capability:
            inputs = torch.randn(2, 4, 8)
            torch.testing.assert_close(callback(inputs, steps=5), inputs)
            capability.assert_not_called()
        self.assertIs(implementation.call_args.kwargs["ns_use_kernels"], False)

    def test_auto_selection_is_per_device_and_cached(self) -> None:
        """Metadata-only CUDA inputs exercise mixed hardware without a real accelerator."""
        implementation = Mock(side_effect=lambda **kwargs: lambda inputs: inputs.clone())
        with patch.object(gram_module, "_load_gram_newton_schulz", return_value=implementation):
            callback = make_gram_newton_schulz(_COEFFICIENTS)
        with patch.object(torch.cuda, "get_device_capability", side_effect=[(8, 0), (9, 0), (10, 0)]) as capability:
            for index in (0, 1, 2, 0, 1, 2):
                inputs = Mock(spec=torch.Tensor, ndim=2, dtype=torch.bfloat16, device=torch.device("cuda", index))
                inputs.numel.return_value = 4
                inputs.contiguous.return_value = torch.ones(2, 2)
                callback(inputs, steps=5)
        self.assertEqual(capability.call_count, 3)
        self.assertEqual(implementation.call_count, 3)
        self.assertEqual([call.kwargs["ns_use_kernels"] for call in implementation.call_args_list], [False, True, True])

    def test_explicit_cuda_overrides(self) -> None:
        """False must skip capability checks; True must reject unsupported CUDA hardware."""
        for use_kernels in (False, True):
            implementation = Mock(side_effect=lambda **kwargs: lambda inputs: inputs.clone())
            with patch.object(gram_module, "_load_gram_newton_schulz", return_value=implementation):
                callback = make_gram_newton_schulz(_COEFFICIENTS, use_kernels=use_kernels)
            inputs = Mock(spec=torch.Tensor, ndim=2, dtype=torch.bfloat16, device=torch.device("cuda", 0))
            inputs.numel.return_value = 4
            inputs.contiguous.return_value = torch.ones(2, 2)
            with self.subTest(use_kernels=use_kernels), patch.object(
                torch.cuda, "get_device_capability", return_value=(8, 0),
            ) as capability:
                if use_kernels:
                    with self.assertRaisesRegex(ValueError, "SM 8"):
                        callback(inputs, steps=5)
                    implementation.assert_not_called()
                else:
                    callback(inputs, steps=5)
                    capability.assert_not_called()
                    self.assertIs(implementation.call_args.kwargs["ns_use_kernels"], False)

    def test_unqualified_devices_are_rejected(self) -> None:
        """Automatic selection must not imply qualification for other device families."""
        implementation = Mock()
        with patch.object(gram_module, "_load_gram_newton_schulz", return_value=implementation):
            callback = make_gram_newton_schulz(_COEFFICIENTS)
        with self.assertRaisesRegex(ValueError, "CPU/CUDA"):
            callback(torch.empty(2, 2, device="meta"), steps=5)
        implementation.assert_not_called()

    def test_empty_and_noncontiguous_inputs(self) -> None:
        """Empty inputs avoid kernels; noncontiguous batches reach upstream contiguously."""
        backend = Mock(side_effect=lambda inputs: inputs.clone())
        implementation = Mock(return_value=backend)
        with patch.object(gram_module, "_load_gram_newton_schulz", return_value=implementation):
            callback = make_gram_newton_schulz(_COEFFICIENTS, use_kernels=False)
        empty = callback(torch.empty(0, 4, 8), steps=5)
        self.assertEqual(empty.shape, (0, 4, 8))
        implementation.assert_not_called()
        inputs = torch.randn(2, 8, 4).mT
        torch.testing.assert_close(callback(inputs, steps=5), inputs)
        self.assertTrue(backend.call_args.args[0].is_contiguous())

    def test_runtime_errors_propagate(self) -> None:
        """Automatic selection must propagate Quack kernel failures without a retry."""
        error = RuntimeError("kernel compilation failed")
        backend = Mock(side_effect=error)
        implementation = Mock(return_value=backend)
        with patch.object(gram_module, "_load_gram_newton_schulz", return_value=implementation):
            callback = make_gram_newton_schulz(_COEFFICIENTS, use_kernels=None)
        inputs = Mock(spec=torch.Tensor, ndim=2, dtype=torch.bfloat16, device=torch.device("cuda", 0))
        inputs.numel.return_value = 256 * 512
        with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
            with self.assertRaises(RuntimeError) as caught:
                callback(inputs, steps=5)
        self.assertIs(caught.exception, error)
        self.assertEqual(implementation.call_count, 1)
        self.assertIs(implementation.call_args.kwargs["ns_use_kernels"], True)
        backend.assert_called_once()
