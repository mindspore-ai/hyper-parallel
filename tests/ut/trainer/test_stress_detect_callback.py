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
"""Hardware-independent tests for periodic NPU stress detection."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from hyper_parallel.trainer.callbacks import StressDetectCallback
from hyper_parallel.trainer.callbacks.stress_detect_callback import logger as stress_logger
from hyper_parallel.trainer.config import ModelStressDetectConfig, OpStressDetectConfig, StressDetectConfig
from hyper_parallel.trainer.config.resolver import resolve_config
from hyper_parallel.trainer.state import TrainerState
from tests.common.mark_utils import arg_mark


_MARK = {
    "plat_marks": ["cpu_linux", "cpu_macos"],
    "level_mark": "level0",
    "card_mark": "onecard",
    "essential_mark": "essential",
}


CALLBACK_MODULE = "hyper_parallel.trainer.callbacks.stress_detect_callback"


class TestStressDetectCallback(unittest.TestCase):
    """Exercise detection cadence and failure handling without accessing an NPU."""

    def setUp(self) -> None:
        """Provide a fake torch.npu namespace without requiring the optional backend."""
        self.npu = Mock()
        self.npu.stress_detect.return_value = 0
        npu_patch = patch(f"{CALLBACK_MODULE}.torch.npu", self.npu, create=True)
        npu_patch.start()
        self.addCleanup(npu_patch.stop)
        device = patch(f"{CALLBACK_MODULE}.get_device_type", return_value="npu")
        self.get_device_type = device.start()
        self.addCleanup(device.stop)

    @staticmethod
    def _trainer(
            interval: int = 10, count: int = 2, rank: int = 0, detect_type: str = "aic",
            exit_training_on_sdc: bool = False,
    ) -> SimpleNamespace:
        """Build the Trainer surface used by the callback."""
        return SimpleNamespace(
            mesh=None,
            global_rank=rank,
            config=SimpleNamespace(stress_detect=StressDetectConfig(
                op_stress_detect=OpStressDetectConfig(
                    detection_interval=interval,
                    num_detections=count,
                    detect_type=detect_type,
                    exit_training_on_sdc=exit_training_on_sdc,
                ),
            )),
        )

    @arg_mark(**_MARK)
    def test_disabled_by_default_does_not_access_npu(self) -> None:
        """Default configuration must not import or access the optional NPU backend.

        Feature: NPU stress detection.
        Description: Default configuration must not import or access the optional NPU backend.
        Expectation: No device detection, synchronization, or stress API call occurs.
        """
        trainer = self._trainer()
        trainer.config.stress_detect = StressDetectConfig()
        callback = StressDetectCallback(trainer)
        callback.on_step_end(TrainerState(global_step=10))

        self.get_device_type.assert_not_called()
        self.npu.synchronize.assert_not_called()
        self.npu.stress_detect.assert_not_called()

    @arg_mark(**_MARK)
    def test_cadence_counts_completed_steps_and_runs_on_nonzero_rank(self) -> None:
        """Run full rounds on nonzero ranks, including after a resumed step count.

        Feature: NPU stress detection.
        Description: Run full rounds on nonzero ranks, including after a resumed step count.
        Expectation: Each due step runs two checks after synchronization.
        """
        callback = StressDetectCallback(self._trainer(rank=3))
        for step in (0, 1, 9, 10, 11, 19, 20, 101, 110):
            callback.on_step_end(TrainerState(global_step=step), loss=1.0, loss_dict={}, grad_norm=0.5)

        self.assertEqual(self.npu.stress_detect.call_count, 6,
                         f"expected three rounds of two checks, got={self.npu.stress_detect.call_count}")
        expected_calls = [
            call.synchronize(), call.stress_detect(detect_type="aic"), call.stress_detect(detect_type="aic"),
        ]
        self.assertEqual(self.npu.mock_calls, expected_calls * 3,
                         f"training work must complete before each round, got={self.npu.mock_calls}")

    @arg_mark(**_MARK)
    def test_success_and_nonfatal_failure_logging(self) -> None:
        """Report failed checks and continue the remaining checks in the round.

        Feature: NPU stress detection.
        Description: Report failed checks and continue the remaining checks in the round.
        Expectation: Precision failure uses ERROR; execution and unknown failures use WARNING.
        """
        self.npu.stress_detect.side_effect = [0, 1, 2, 17]
        callback = StressDetectCallback(self._trainer(count=4))
        with self.assertLogs(stress_logger, level="INFO") as logs:
            callback.on_step_end(TrainerState(global_step=10))

        self.assertTrue(any("Stress detection passed" in line for line in logs.output),
                        f"missing success result, got={logs.output}")
        for code in (1, 2, 17):
            self.assertTrue(any(f"return_code={code}" in line for line in logs.output),
                            f"missing failure code={code}, logs={logs.output}")
        failures = [record for record in logs.records if "return_code=" in record.getMessage()]
        self.assertEqual([record.levelname for record in failures], ["WARNING", "ERROR", "WARNING"],
                         f"only hardware precision failure must use ERROR, got={logs.output}")
        self.assertIn("test case execution failed", failures[0].getMessage(),
                      f"execution failure must be identified, got={logs.output}")
        self.assertIn("hardware precision failure", failures[1].getMessage(),
                      f"precision failure must be identified, got={logs.output}")
        self.assertIn("unrecognized return status", failures[2].getMessage(),
                      f"unknown status must be identified, got={logs.output}")

    @arg_mark(**_MARK)
    def test_backend_exception_propagates(self) -> None:
        """Preserve errors raised directly by torch_npu instead of suppressing them.

        Feature: NPU stress detection.
        Description: Preserve errors raised directly by torch_npu instead of suppressing them.
        Expectation: The original voltage recovery exception reaches the caller unchanged.
        """
        error = RuntimeError("Stress detect error. Error code is 574007. Error message is Voltage recovery failed.")
        self.npu.stress_detect.side_effect = error
        callback = StressDetectCallback(self._trainer())
        with self.assertRaisesRegex(RuntimeError, "574007.*Voltage recovery failed") as context:
            callback.on_step_end(TrainerState(global_step=10))
        self.assertIs(context.exception, error, f"expected the original backend exception, got={context.exception}")
        self.npu.stress_detect.assert_called_once_with(detect_type="aic")

    @arg_mark(**_MARK)
    def test_precision_error_is_logged_before_later_backend_exception(self) -> None:
        """Keep completed fault results when a later detection raises.

        Feature: NPU stress detection failure reporting.
        Description: Return a precision fault before a same-type or HCCS call raises.
        Expectation: One HP ERROR is retained and the original backend exception propagates.
        """
        for detect_type, count in (("aic", 2), ("all", 1)):
            with self.subTest(detect_type=detect_type):
                self.npu.stress_detect.reset_mock()
                error = RuntimeError("574007 Voltage recovery failed")
                self.npu.stress_detect.side_effect = [2, error]
                callback = StressDetectCallback(self._trainer(count=count, detect_type=detect_type))
                with self.assertLogs(stress_logger, level="ERROR") as logs:
                    with self.assertRaises(RuntimeError) as context:
                        callback.on_step_end(TrainerState(global_step=10))
                self.assertIs(context.exception, error,
                              f"expected the original backend exception, got={context.exception}")
                self.assertEqual(len(logs.records), 1, f"expected one retained HP ERROR, got={logs.output}")
                self.assertIn("hardware precision failure", logs.output[0],
                              f"the completed precision fault must be reported, got={logs.output}")
                self.assertIn("detect_type=aic, return_code=2", logs.output[0],
                              f"the failed detection type and code must be retained, got={logs.output}")
                expected = [call(detect_type="aic"), call(detect_type="hccs" if detect_type == "all" else "aic")]
                self.assertEqual(self.npu.stress_detect.call_args_list, expected,
                                 f"unexpected backend invocation sequence, got={self.npu.stress_detect.call_args_list}")

    @arg_mark(**_MARK)
    def test_sdc_exit_stops_remaining_detections(self) -> None:
        """Exit on the first precision fault before another backend call can run.

        Feature: NPU stress detection early exit.
        Description: Enable SDC exit before a precision fault followed by a backend exception.
        Expectation: The precision fault is logged and raised after exactly one backend invocation.
        """
        for detect_type, count in (("aic", 2), ("all", 1)):
            with self.subTest(detect_type=detect_type):
                self.npu.stress_detect.reset_mock()
                self.npu.stress_detect.side_effect = [2, RuntimeError("574007 Voltage recovery failed")]
                callback = StressDetectCallback(self._trainer(
                    count=count, detect_type=detect_type, exit_training_on_sdc=True,
                ))
                with self.assertLogs(stress_logger, level="ERROR") as logs:
                    with self.assertRaisesRegex(RuntimeError, "hardware precision failure.*return_code=2"):
                        callback.on_step_end(TrainerState(global_step=10))
                self.npu.stress_detect.assert_called_once_with(detect_type="aic")
                self.assertEqual(len(logs.records), 1, f"expected one precision fault log, got={logs.output}")

    @arg_mark(**_MARK)
    def test_invalid_configuration_rejected_before_backend_access(self) -> None:
        """Reject invalid cadence/count values, including boolean and fractional values.

        Feature: NPU stress detection.
        Description: Reject invalid cadence/count values, including boolean and fractional values.
        Expectation: Invalid intervals and counts raise before NPU access.
        """
        for interval in (-1, True, 1.5, None):
            with self.subTest(interval=interval):
                with self.assertRaisesRegex(ValueError, "op_stress_detect.detection_interval"):
                    StressDetectCallback(self._trainer(interval=interval))
        for count in (0, -1, True, 1.5, None):
            with self.subTest(count=count):
                with self.assertRaisesRegex(ValueError, "op_stress_detect.num_detections"):
                    StressDetectCallback(self._trainer(count=count))
        self.npu.stress_detect.assert_not_called()

    @arg_mark(**_MARK)
    def test_enabled_requires_npu_device(self) -> None:
        """Explicitly enabling detection on CPU/CUDA must fail clearly.

        Feature: NPU stress detection.
        Description: Explicitly enabling detection on CPU/CUDA must fail clearly.
        Expectation: Explicit CPU and CUDA enablement raises without running detection.
        """
        for device in ("cpu", "cuda"):
            with self.subTest(device=device):
                self.get_device_type.return_value = device
                with self.assertRaisesRegex(RuntimeError, "requires an NPU"):
                    StressDetectCallback(self._trainer())
        self.npu.stress_detect.assert_not_called()

    @arg_mark(**_MARK)
    def test_missing_backend_api_rejected(self) -> None:
        """An older torch_npu without stress_detect must fail at initialization.

        Feature: NPU stress detection.
        Description: An older torch_npu without stress_detect must fail at initialization.
        Expectation: Missing backend support raises a descriptive initialization error.
        """
        with patch(f"{CALLBACK_MODULE}.torch.npu", SimpleNamespace()):
            with self.assertRaisesRegex(RuntimeError, "requires torch.npu.stress_detect"):
                StressDetectCallback(self._trainer())

    @arg_mark(**_MARK)
    def test_result_logging_can_be_called_independently(self) -> None:
        """Report collected results without accessing the NPU or a Trainer.

        Feature: NPU stress detection.
        Description: Report collected results without accessing the NPU or a Trainer.
        Expectation: The result helper reports each code without touching the NPU.
        """
        with self.assertLogs(stress_logger, level="INFO") as logs:
            StressDetectCallback.log_stress_detect_result([0, 2])
        self.assertEqual(len(logs.output), 2, f"expected one log per return code, got={logs.output}")
        self.npu.stress_detect.assert_not_called()

    @arg_mark(**_MARK)
    def test_yaml_settings_resolve_into_callback(self) -> None:
        """Resolve the dedicated stress detection section and schedule a detection round.

        Feature: NPU stress detection.
        Description: Resolve the dedicated stress detection section and schedule a detection round.
        Expectation: The callback uses the parsed HCCS mode and configured cadence.
        """
        config = resolve_config({
            "model": {"_target_": "torch.nn.Linear", "in_features": 2, "out_features": 2},
            "optimizer": {"_target_": "torch.optim.AdamW"},
            "stress_detect": {"op_stress_detect": {
                "detection_interval": 5, "num_detections": 3, "detect_type": "hccs", "exit_training_on_sdc": True,
            }},
        })
        trainer = SimpleNamespace(mesh=None, config=config)
        callback = StressDetectCallback(trainer)
        callback.on_step_end(TrainerState(global_step=4))
        self.npu.stress_detect.assert_not_called()
        callback.on_step_end(TrainerState(global_step=5))
        self.assertEqual(self.npu.stress_detect.call_count, 3,
                         f"expected the configured three checks, got={self.npu.stress_detect.call_count}")
        self.assertEqual(self.npu.stress_detect.call_args_list, [call(detect_type="hccs")] * 3,
                         f"expected the configured HCCS mode, got={self.npu.stress_detect.call_args_list}")

    @arg_mark(**_MARK)
    def test_omitted_section_defaults_to_disabled_and_serializes_separately(self) -> None:
        """Keep stress settings separate from training, including when YAML omits them.

        Feature: NPU stress detection.
        Description: Keep stress settings separate from training, including when YAML omits them.
        Expectation: Serialized stress settings remain disabled.
        """
        config = resolve_config({
            "model": {"_target_": "torch.nn.Linear", "in_features": 2, "out_features": 2},
            "optimizer": {"_target_": "torch.optim.AdamW"},
        })
        self.assertIsInstance(config.stress_detect, StressDetectConfig,
                              f"expected the dedicated config class, got={type(config.stress_detect)}")
        serialized = config.to_dict()
        expected = {
            "op_stress_detect": {"detection_interval": 0, "num_detections": 1, "detect_type": "aic",
                                 "exit_training_on_sdc": False},
            "model_stress_detect": {"detection_interval": 0},
        }
        self.assertEqual(serialized["stress_detect"], expected,
                         f"expected disabled defaults, got={serialized['stress_detect']}")
        self.assertFalse(any(key.startswith("stress_detect") for key in serialized["training"]),
                         f"stress settings must not be part of training, got={serialized['training']}")

    @arg_mark(**_MARK)
    def test_all_runs_each_type_after_precision_failure(self) -> None:
        """Report precision failure without suppressing remaining checks or the HCCS round.

        Feature: NPU stress detection.
        Description: Report precision failure without suppressing remaining checks or the HCCS round.
        Expectation: Both detection types complete and precision failure is logged.
        """
        self.npu.stress_detect.side_effect = [2, 0, 0, 0]
        callback = StressDetectCallback(self._trainer(detect_type="all"))
        with self.assertLogs(stress_logger, level="INFO") as logs:
            callback.on_step_end(TrainerState(global_step=10))
        self.assertEqual([record.levelname for record in logs.records].count("ERROR"), 1,
                         f"expected one precision failure error, got={logs.output}")
        expected = [call(detect_type="aic")] * 2 + [call(detect_type="hccs")] * 2
        self.assertEqual(self.npu.stress_detect.call_args_list, expected,
                         f"all must run two checks per type, got={self.npu.stress_detect.call_args_list}")

    @arg_mark(**_MARK)
    def test_all_logs_each_type(self) -> None:
        """Identify AIC and HCCS results separately in all mode.

        Feature: NPU stress detection.
        Description: Identify AIC and HCCS results separately in all mode.
        Expectation: Each result identifies its detection type and severity.
        """
        self.npu.stress_detect.side_effect = [0, 2]
        callback = StressDetectCallback(self._trainer(count=1, detect_type="all"))
        with self.assertLogs(stress_logger, level="INFO") as logs:
            callback.on_step_end(TrainerState(global_step=10))
        self.assertTrue(any("passed (aic)" in line for line in logs.output),
                        f"missing AIC success result, got={logs.output}")
        self.assertTrue(any(
            "hardware precision failure" in line and "detect_type=hccs" in line for line in logs.output),
                        f"missing HCCS failure result, got={logs.output}")

    @arg_mark(**_MARK)
    def test_precision_failure_raises_when_configured(self) -> None:
        """Stop the training step when an enabled policy receives a precision failure.

        Feature: NPU stress detection.
        Description: Stop the training step when an enabled policy receives a precision failure.
        Expectation: An enabled SDC exit policy logs ERROR and raises RuntimeError.
        """
        callback = StressDetectCallback(self._trainer(exit_training_on_sdc=True))
        self.npu.stress_detect.side_effect = [0, 2]
        with self.assertLogs(stress_logger, level="ERROR") as logs:
            with self.assertRaisesRegex(RuntimeError, "hardware precision failure.*detect_type=aic.*return_code=2"):
                callback.on_step_end(TrainerState(global_step=10))
        self.assertEqual(len(logs.records), 1, f"expected one precision error, got={logs.output}")

    @arg_mark(**_MARK)
    def test_exit_policy_does_not_raise_for_execution_failure(self) -> None:
        """The SDC exit policy must not treat execution failures as precision faults.

        Feature: NPU stress detection.
        Description: The SDC exit policy must not treat execution failures as precision faults.
        Expectation: Execution failures remain warnings with the exit policy on.
        """
        callback = StressDetectCallback(self._trainer(count=3, exit_training_on_sdc=True))
        self.npu.stress_detect.side_effect = [0, 1, 17]
        with self.assertLogs(stress_logger, level="WARNING") as logs:
            callback.on_step_end(TrainerState(global_step=10))
        self.assertEqual([record.levelname for record in logs.records], ["WARNING", "WARNING"],
                         f"non-precision failures must remain warnings, got={logs.output}")

    @arg_mark(**_MARK)
    def test_invalid_exit_policy_rejected(self) -> None:
        """Require a boolean exit policy for directly constructed configurations.

        Feature: NPU stress detection.
        Description: Require a boolean exit policy for directly constructed configurations.
        Expectation: Non-boolean exit policies fail configuration validation.
        """
        for value in (0, 1, "true", "false", None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "exit_training_on_sdc.*boolean"):
                    OpStressDetectConfig(exit_training_on_sdc=value)

    @arg_mark(**_MARK)
    def test_invalid_detection_type_rejected(self) -> None:
        """Reject unknown modes before accessing the NPU backend.

        Feature: NPU stress detection.
        Description: Reject unknown modes before accessing the NPU backend.
        Expectation: Unsupported detection types raise before backend access.
        """
        for detect_type in ("invalid", "AIC", "", None, ["aic", "hccs"]):
            with self.subTest(detect_type=detect_type):
                with self.assertRaisesRegex(ValueError, "op_stress_detect.detect_type"):
                    StressDetectCallback(self._trainer(detect_type=detect_type))
        self.get_device_type.assert_not_called()
        self.npu.stress_detect.assert_not_called()

    @arg_mark(**_MARK)
    def test_model_stress_entry_is_reserved_but_cannot_be_enabled(self) -> None:
        """Accept a disabled model-level entry and fail early when it is enabled.

        Feature: NPU stress detection.
        Description: Accept a disabled model-level entry and fail early when it is enabled.
        Expectation: Model detection can be disabled but not enabled yet.
        """
        raw = {
            "model": {"_target_": "torch.nn.Linear", "in_features": 2, "out_features": 2},
            "optimizer": {"_target_": "torch.optim.AdamW"},
            "stress_detect": {
                "op_stress_detect": {"detection_interval": 10, "detect_type": "all"},
                "model_stress_detect": {"detection_interval": 0},
            },
        }
        config = resolve_config(raw)
        operation = config.stress_detect.op_stress_detect
        model = config.stress_detect.model_stress_detect
        self.assertEqual(operation.detection_interval, 10,
                         f"expected enabled operation detection, got={operation.detection_interval}")
        self.assertEqual(model.detection_interval, 0,
                         f"expected disabled model detection, got={model.detection_interval}")
        raw["stress_detect"]["model_stress_detect"]["detection_interval"] = 20
        with self.assertRaisesRegex(NotImplementedError, "model_stress_detect.*not implemented"):
            resolve_config(raw)
        self.get_device_type.assert_not_called()
        self.npu.stress_detect.assert_not_called()

    @arg_mark(**_MARK)
    def test_invalid_model_interval_rejected(self) -> None:
        """Validate the reserved model interval independently of operation settings.

        Feature: NPU stress detection.
        Description: Validate the reserved model interval independently of operation settings.
        Expectation: Invalid model detection intervals fail validation.
        """
        for interval in (-1, True, 1.5, None):
            with self.subTest(interval=interval):
                with self.assertRaisesRegex(ValueError, "model_stress_detect.detection_interval"):
                    ModelStressDetectConfig(detection_interval=interval)
