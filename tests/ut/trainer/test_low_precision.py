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
"""CPU-only contract tests for low-precision setup and observation."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from hyper_models.components.training.low_precision import (
    LowPrecisionConfig,
    LowPrecisionConversionError,
    NpuQuantLinear,
    apply_low_precision,
)
from hyper_models.components.training.low_precision.functional import npu_quant_linear
from hyper_models.components.training.low_precision.functional.linear import (
    _dequantize_rowwise,
)
from hyper_models.components.training.low_precision.ops import NpuOps
from hyper_low_precision_observer import (
    DebugOutput,
    DebugOverlay,
    DebugSchedule,
    ObserveAction,
    PrecisionContext,
    PrecisionDebugProgram,
    PrecisionReport,
)
from hyper_models.components.training.low_precision.observer.bridge import (
    PrecisionDebugSession,
    find_precision_debug_session,
    install_precision_debug,
)
from hyper_models.components.training.low_precision.tensor import MXFP8Tensor


class _CustomLinear(nn.Linear):
    """A Linear subclass whose forward contract the generic converter cannot retain."""


class TestLowPrecisionSetup(unittest.TestCase):
    """Verify setup errors are explicit before an NPU runtime is required."""

    def test_converter_replaces_selected_exact_linear(self):
        model = nn.Sequential(nn.Linear(32, 32, bias=False))

        converted = apply_low_precision(
            model,
            LowPrecisionConfig(enabled=True, include_fqns=["0"]),
        )

        self.assertIs(converted, model)
        self.assertIsInstance(model[0], NpuQuantLinear)

    def test_converter_rejects_selected_linear_subclass(self):
        model = nn.Sequential(_CustomLinear(32, 32, bias=False))

        with self.assertRaisesRegex(
            LowPrecisionConversionError,
            "linear-subclass-is-not-supported",
        ):
            apply_low_precision(
                model,
                LowPrecisionConfig(enabled=True, include_fqns=["0"]),
            )

    def test_debug_requires_enabled_low_precision(self):
        with self.assertRaisesRegex(ValueError, "requires enabled=True"):
            LowPrecisionConfig(
                precision_debug={
                    "sections": [{"name": "fprop", "observe": True}],
                },
            )

    def test_public_debug_config_accepts_all_gemm_roles(self):
        program = PrecisionDebugProgram.from_mapping({
            "sections": [{
                "name": "all_roles",
                "select": {"gemm_roles": ["fprop", "dgrad", "wgrad"]},
                "observe": True,
            }],
        })

        self.assertEqual(
            program.sections[0].select.gemm_roles,
            ("fprop", "dgrad", "wgrad"),
        )

    def test_public_debug_config_rejects_yaml_output_directory(self):
        with self.assertRaisesRegex(ValueError, "Unknown precision_debug fields"):
            PrecisionDebugProgram.from_mapping({
                "output": {"root_dir": "/tmp/precision_debug"},
                "sections": [{"name": "fprop", "observe": True}],
            })

    def test_public_observer_has_no_hp_imports(self):
        import hyper_low_precision_observer

        package_root = Path(hyper_low_precision_observer.__file__).parent
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in package_root.glob("*.py")
        )

        self.assertNotIn("hyper_models", source)
        self.assertNotIn("hyper_parallel", source)

    def test_observer_imports_and_writes_rank_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            program = PrecisionDebugProgram.default()
            context = PrecisionContext()
            # The public program is parsed independently from installation;
            # use the minimal overlay that selects this direct unit observation.
            from hyper_low_precision_observer.config import (
                DebugOverlay,
                ObserveAction,
            )

            context.debug_overlay = DebugOverlay(
                observations={("module", "fprop"): (ObserveAction(),)},
            )
            context.record(
                "quantization",
                "module",
                "fprop",
                "lhs",
                torch.tensor([1.0]),
                torch.tensor([0.5]),
            )

            paths = PrecisionReport(DebugOutput(root_dir=directory)).flush(context)

            self.assertIsNotNone(paths)
            self.assertTrue(Path(paths["rank_artifact"]).is_file())

    def test_observer_rejects_existing_rank_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            overlay = DebugOverlay(
                observations={("module", "fprop"): (ObserveAction(),)},
            )
            first_context = PrecisionContext(debug_overlay=overlay)
            first_context.record(
                "quantization", "module", "fprop", "lhs",
                torch.tensor([1.0]), torch.tensor([0.5]),
            )
            first_paths = PrecisionReport(DebugOutput(root_dir=directory)).flush(
                first_context
            )

            resumed_context = PrecisionContext(debug_overlay=overlay)
            resumed_context.set_position(1)
            resumed_context.record(
                "quantization", "module", "fprop", "lhs",
                torch.tensor([1.0]), torch.tensor([0.25]),
            )
            self.assertIsNotNone(first_paths)
            with self.assertRaisesRegex(FileExistsError, "does not support resume"):
                PrecisionReport(DebugOutput(root_dir=directory)).flush(resumed_context)

    def test_debug_session_disables_after_record_failure(self):
        context = PrecisionContext()
        context.should_measure_quantization = MagicMock(return_value=True)
        context.record = MagicMock(side_effect=RuntimeError("diagnostic failed"))
        context.clear = MagicMock()
        report = MagicMock()
        session = PrecisionDebugSession(context, report)

        session.record_quantization(
            "module",
            "fprop",
            "lhs",
            torch.ones(1),
            torch.ones(1),
        )
        session.flush()

        self.assertFalse(session.enabled)
        self.assertIsInstance(session.failure, RuntimeError)
        context.clear.assert_called_once_with()
        report.flush.assert_not_called()

    def test_debug_session_disables_after_report_failure(self):
        context = PrecisionContext()
        report = MagicMock()
        report.flush.side_effect = OSError("output unavailable")
        session = PrecisionDebugSession(context, report)

        self.assertIsNone(session.flush())

        self.assertFalse(session.enabled)
        self.assertIsInstance(session.failure, OSError)

    def test_debug_session_swallows_cleanup_failure(self):
        context = PrecisionContext()
        context.should_measure_quantization = MagicMock(return_value=True)
        context.record = MagicMock(side_effect=RuntimeError("record failed"))
        context.clear = MagicMock(side_effect=RuntimeError("clear failed"))
        session = PrecisionDebugSession(context, MagicMock())

        self.assertIsNone(
            session.record_quantization(
                "module",
                "fprop",
                "lhs",
                torch.ones(1),
                torch.ones(1),
            )
        )
        self.assertFalse(session.enabled)
        self.assertIsInstance(session.failure, RuntimeError)

    def test_debug_session_applies_selector_and_schedule_before_recording(self):
        context = PrecisionContext(
            debug_overlay=DebugOverlay(
                observations={
                    ("selected", "fprop"): (
                        ObserveAction(
                            operands=("lhs",),
                            schedule=DebugSchedule(every_n_steps=2),
                        ),
                    ),
                },
            )
        )
        context.record = MagicMock()
        session = PrecisionDebugSession(context, MagicMock())

        session.set_step(1)
        session.record_quantization(
            "selected", "fprop", "lhs", torch.ones(1), torch.ones(1)
        )
        session.record_quantization(
            "other", "fprop", "lhs", torch.ones(1), torch.ones(1)
        )
        session.record_quantization(
            "selected", "fprop", "rhs", torch.ones(1), torch.ones(1)
        )
        context.record.assert_not_called()

        session.set_step(2)
        session.record_quantization(
            "selected", "fprop", "lhs", torch.ones(1), torch.ones(1)
        )
        context.record.assert_called_once()

    def test_debug_session_exposes_sampling_preflight(self):
        context = PrecisionContext(
            debug_overlay=DebugOverlay(
                observations={
                    ("selected", "fprop"): (
                        ObserveAction(
                            operands=("lhs",),
                            schedule=DebugSchedule(every_n_steps=2),
                        ),
                    ),
                },
            )
        )
        session = PrecisionDebugSession(context, MagicMock())

        session.set_step(1)
        self.assertFalse(
            session.should_observe_quantization("selected", "fprop", "lhs")
        )
        session.set_step(2)
        self.assertTrue(
            session.should_observe_quantization("selected", "fprop", "lhs")
        )
        self.assertFalse(
            session.should_observe_quantization("other", "fprop", "lhs")
        )
        self.assertFalse(
            session.should_observe_quantization("selected", "fprop", "rhs")
        )

    def test_debug_session_pauses_without_recording_or_disabling(self):
        context = PrecisionContext(
            debug_overlay=DebugOverlay(
                observations={
                    ("module", "fprop"): (ObserveAction(),),
                },
            )
        )
        context.record = MagicMock()
        session = PrecisionDebugSession(context, MagicMock())

        with session.paused():
            self.assertFalse(
                session.should_observe_quantization("module", "fprop", "lhs")
            )
            session.record_quantization(
                "module", "fprop", "lhs", torch.ones(1), torch.zeros(1)
            )

        context.record.assert_not_called()
        self.assertTrue(session.enabled)

    def test_dequantization_uses_npu_adapter_e8m0_dtype(self):
        adapter = MagicMock()
        adapter.is_e8m0_dtype.return_value = True
        quantizer = MagicMock()
        quantizer.npu_ops = adapter
        quantized = MXFP8Tensor(
            (1, 32),
            torch.float32,
            quantizer=quantizer,
            row_data=torch.ones(1, 32, dtype=torch.float8_e4m3fn),
            row_scale=torch.full((1, 1), 127, dtype=torch.uint8),
        )

        result = _dequantize_rowwise(quantized)

        adapter.is_e8m0_dtype.assert_called_once_with(torch.uint8)
        self.assertTrue(torch.equal(result, torch.ones(1, 32)))

    def test_install_precision_debug_samples_and_writes_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            model = nn.Sequential(nn.Linear(32, 32, bias=False))
            apply_low_precision(
                model,
                LowPrecisionConfig(enabled=True, include_fqns=["0"]),
            )
            session = install_precision_debug(
                model,
                {
                    "sections": [{
                        "name": "selected_lhs",
                        "select": {
                            "module_name_regex": "0",
                            "gemm_roles": ["fprop"],
                        },
                        "observe": {
                            "operands": ["lhs"],
                            "schedule": {"every_n_steps": 2},
                        },
                    }],
                },
                output_root=directory,
            )

            self.assertEqual(session.context.debug_overlay.output.root_dir, directory)

            session.set_step(1)
            session.record_quantization(
                "0", "fprop", "lhs", torch.ones(2), torch.zeros(2)
            )
            self.assertIsNone(session.flush())

            session.set_step(2)
            session.record_quantization(
                "0", "fprop", "lhs", torch.ones(2), torch.zeros(2)
            )
            paths = session.flush()

            self.assertTrue(session.enabled)
            self.assertIsNotNone(paths)
            artifact = Path(paths["rank_artifact"])
            self.assertTrue(artifact.is_file())
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            self.assertEqual(payload["iteration"], 2)
            self.assertIn(
                "quantization/0/fprop/lhs",
                payload["error_moments"],
            )

    def test_debug_installation_binds_converted_linear(self):
        with tempfile.TemporaryDirectory() as directory:
            model = nn.Sequential(nn.Linear(32, 32, bias=False))
            apply_low_precision(
                model,
                LowPrecisionConfig(enabled=True, include_fqns=["0"]),
            )
            session = install_precision_debug(
                model,
                PrecisionDebugProgram.default(),
                output_root=directory,
            )

            self.assertIs(model._precision_debug_session, session)
            self.assertIs(model[0]._precision_debug_session, session)

            with patch(
                "hyper_models.components.training.low_precision.modules.linear.npu_quant_linear",
                return_value=torch.zeros(1, 32),
            ) as quant_linear:
                model(torch.ones(1, 32))

            self.assertIs(quant_linear.call_args.kwargs["observer"], session)
            self.assertEqual(quant_linear.call_args.kwargs["module_fqn"], "0")

    def test_adapter_replaces_program_output_with_runtime_root(self):
        with tempfile.TemporaryDirectory() as directory:
            model = nn.Sequential(nn.Linear(32, 32, bias=False))
            apply_low_precision(
                model,
                LowPrecisionConfig(enabled=True, include_fqns=["0"]),
            )
            program = PrecisionDebugProgram(
                sections=PrecisionDebugProgram.default().sections,
                output=DebugOutput(root_dir="must-not-be-used"),
            )

            session = install_precision_debug(
                model,
                program,
                output_root=directory,
            )

            self.assertEqual(session.context.debug_overlay.output.root_dir, directory)

    def test_debug_installation_skips_unselected_converted_linear(self):
        with tempfile.TemporaryDirectory() as directory:
            model = nn.Sequential(
                nn.Linear(32, 32, bias=False),
                nn.Linear(32, 32, bias=False),
            )
            apply_low_precision(
                model,
                LowPrecisionConfig(enabled=True, include_fqns=["*"]),
            )
            session = install_precision_debug(
                model,
                {
                    "sections": [{
                        "name": "only_first_linear",
                        "select": {"module_name_regex": "0"},
                        "observe": True,
                    }],
                },
                output_root=directory,
            )

            self.assertIs(model[0]._precision_debug_session, session)
            self.assertIsNone(model[1]._precision_debug_session)

    def test_debug_installation_rejects_unmatched_selector(self):
        with tempfile.TemporaryDirectory() as directory:
            model = nn.Sequential(nn.Linear(32, 32, bias=False))
            apply_low_precision(
                model,
                LowPrecisionConfig(enabled=True, include_fqns=["0"]),
            )

            with self.assertRaisesRegex(
                ValueError,
                "selectors matched no converted NpuQuantLinear",
            ):
                install_precision_debug(
                    model,
                    {
                        "sections": [{
                            "name": "missing",
                            "select": {
                                "module_name_regex": "does-not-exist",
                            },
                            "observe": True,
                        }],
                    },
                    output_root=directory,
                )

    def test_debug_installation_rejects_unsupported_hp_role(self):
        with tempfile.TemporaryDirectory() as directory:
            model = nn.Sequential(nn.Linear(32, 32, bias=False))
            apply_low_precision(
                model,
                LowPrecisionConfig(enabled=True, include_fqns=["0"]),
            )

            with self.assertRaisesRegex(
                ValueError,
                r"supported GEMM roles \['fprop'\]",
            ):
                install_precision_debug(
                    model,
                    {
                        "sections": [{
                            "name": "unsupported_dgrad",
                            "select": {"gemm_roles": ["dgrad"]},
                            "observe": True,
                        }],
                    },
                    output_root=directory,
                )

    def test_converter_does_not_install_debug_session(self):
        with tempfile.TemporaryDirectory() as directory:
            model = nn.Sequential(nn.Linear(32, 32, bias=False))
            apply_low_precision(
                model,
                LowPrecisionConfig(
                    enabled=True,
                    include_fqns=["0"],
                    precision_debug={
                        "sections": [{
                            "name": "fprop_lhs",
                            "observe": {"operands": ["lhs"]},
                        }],
                    },
                ),
            )

            self.assertFalse(hasattr(model, "_precision_debug_session"))
            self.assertIsNone(model[0]._precision_debug_session)

    def test_native_quantization_emits_fprop_operands(self):
        class _Quantizer:
            def quantize(self, tensor, *, rowwise, colwise):
                del colwise
                assert rowwise
                return MXFP8Tensor(
                    tensor.shape,
                    tensor.dtype,
                    quantizer=self,
                    row_data=torch.ones_like(tensor, dtype=torch.float8_e4m3fn),
                    row_scale=torch.ones(
                        *tensor.shape[:-1],
                        tensor.shape[-1] // 32,
                        dtype=torch.float32,
                    ),
                )

        observer = MagicMock()
        quantizer = _Quantizer()
        inputs = torch.ones(1, 32)
        weight = torch.ones(32, 32)
        with patch(
            "hyper_models.components.training.low_precision.functional.linear.mxfp8_matmul",
            return_value=torch.zeros(1, 32),
        ):
            output = npu_quant_linear(
                inputs,
                weight,
                quantizer,
                observer=observer,
                module_fqn="module",
            )

        self.assertEqual(output.shape, (1, 32))
        self.assertEqual(observer.observe_quantization.call_count, 2)
        self.assertEqual(
            observer.observe_quantization.call_args_list[0].args[0:3],
            ("module", "fprop", "lhs"),
        )
        self.assertEqual(
            observer.observe_quantization.call_args_list[1].args[0:3],
            ("module", "fprop", "rhs"),
        )

    def test_native_quantization_keeps_autograd_contract_with_observer(self):
        class _Quantizer:
            def quantize(self, tensor, *, rowwise, colwise):
                del colwise
                assert rowwise
                return MXFP8Tensor(
                    tensor.shape,
                    tensor.dtype,
                    quantizer=self,
                    row_data=torch.ones_like(tensor, dtype=torch.float8_e4m3fn),
                    row_scale=torch.ones(
                        *tensor.shape[:-1],
                        tensor.shape[-1] // 32,
                        dtype=torch.float32,
                    ),
                )

        def _matmul(left, right, *, layout, output_dtype):
            del right
            if layout == "NT":
                shape = (left.shape[0], 32)
            elif layout == "NN":
                shape = (left.shape[0], 32)
            else:
                shape = (32, 32)
            return torch.zeros(shape, dtype=output_dtype)

        inputs = torch.ones(1, 32, requires_grad=True)
        weight = torch.ones(32, 32, requires_grad=True)
        with patch(
            "hyper_models.components.training.low_precision.functional.linear.mxfp8_matmul",
            side_effect=_matmul,
        ):
            npu_quant_linear(
                inputs,
                weight,
                _Quantizer(),
                observer=MagicMock(),
                module_fqn="module",
            ).sum().backward()

        self.assertEqual(inputs.grad.shape, inputs.shape)
        self.assertEqual(weight.grad.shape, weight.shape)

    def test_debug_session_defers_candidate_until_schedule_matches(self):
        context = PrecisionContext(
            debug_overlay=DebugOverlay(
                observations={
                    ("module", "fprop"): (
                        ObserveAction(
                            operands=("lhs",),
                            schedule=DebugSchedule(every_n_steps=2),
                        ),
                    ),
                },
            )
        )
        session = PrecisionDebugSession(context, MagicMock())
        candidate = MagicMock(return_value=torch.ones(1))

        session.set_step(1)
        session.observe_quantization(
            "module", "fprop", "lhs", torch.ones(1), candidate
        )
        candidate.assert_not_called()

        session.set_step(2)
        session.observe_quantization(
            "module", "fprop", "lhs", torch.ones(1), candidate
        )
        candidate.assert_called_once_with()

    def test_debug_session_is_found_after_model_wrapping(self):
        with tempfile.TemporaryDirectory() as directory:
            model = nn.Sequential(nn.Linear(32, 32, bias=False))
            apply_low_precision(
                model,
                LowPrecisionConfig(enabled=True, include_fqns=["0"]),
            )
            session = install_precision_debug(
                model,
                PrecisionDebugProgram.default(),
                output_root=directory,
            )
            wrapper = nn.Module()
            wrapper.add_module("inner", model)

            self.assertIs(find_precision_debug_session(wrapper), session)


if __name__ == "__main__":
    unittest.main()
