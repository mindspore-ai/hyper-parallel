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
"""Periodic online NPU hardware precision stress detection."""

from typing import Any

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.trainer.runtime.device import get_device_type
from hyper_parallel.trainer.callbacks.base import Callback, TrainerState
from hyper_parallel.tools.logging import get_logger


logger = get_logger("HP")
HARDWARE_FAILURE_CODE = 2


class StressDetectCallback(Callback):
    """Run hardware precision stress detection between optimizer steps on every rank.

    ``stress_detect.op_stress_detect`` configures this callback.
    ``detection_interval`` sets the cadence in completed optimizer steps; zero
    disables detection. ``num_detections`` sets the number of consecutive checks
    in each round. Detection uses the current NPU and requires the memory headroom
    specified by torch_npu. ``detect_type`` selects ``aic``, ``hccs``, or ``all``;
    ``all`` runs the configured number of checks for each type, AIC first.
    ``exit_training_on_sdc`` raises on precision failure when enabled; otherwise
    the failure is logged at ERROR and training continues.
    """

    def __init__(self, trainer: Any) -> None:
        """Read detection settings and resolve the optional NPU API when enabled.

        Args:
            trainer: Trainer that owns the callback lifecycle.

        Raises:
            RuntimeError: If enabled on a non-NPU device or without the API.
        """
        super().__init__(trainer)
        config = trainer.config.stress_detect.op_stress_detect
        self.detection_interval = config.detection_interval
        self.num_detections = config.num_detections
        self.detect_type = config.detect_type
        self.exit_training_on_sdc = config.exit_training_on_sdc
        if self.detection_interval == 0:
            return
        if get_device_type() != "npu":
            raise RuntimeError("StressDetectCallback requires an NPU device when enabled")
        # Device discovery imports the optional torch_npu backend and registers torch.npu.
        if not callable(getattr(getattr(torch, "npu", None), "stress_detect", None)):
            raise RuntimeError("StressDetectCallback requires torch.npu.stress_detect")

    def on_step_end(self, state: TrainerState, **kwargs: Any) -> None:  # pylint: disable=arguments-differ
        """Synchronize training work and run a detection round at the configured step.

        Args:
            state: Training state containing the completed optimizer step count.
            **kwargs: Unused callback context.

        Raises:
            RuntimeError: If configured to stop on precision failure, or torch_npu
                raises an error such as voltage recovery failure.
        """
        del kwargs
        if self.detection_interval == 0 or state.global_step <= 0:
            return
        if state.global_step % self.detection_interval != 0:
            return
        torch.npu.synchronize()
        logger.info("Start hardware precision stress detection at step %s", state.global_step)
        detect_types = ("aic", "hccs") if self.detect_type == "all" else (self.detect_type,)
        for detect_type in detect_types:
            for _ in range(self.num_detections):
                result = torch.npu.stress_detect(detect_type=detect_type)
                self.log_stress_detect_result(
                    [result], detect_type=detect_type, exit_training_on_sdc=self.exit_training_on_sdc,
                )

    @staticmethod
    def log_stress_detect_result(
            detect_ret_list: list[int], detect_type: str = "aic", exit_training_on_sdc: bool = False,
    ) -> None:
        """Report completed detections without waiting for the remaining checks.

        Args:
            detect_ret_list: Detection return codes in execution order.
            detect_type: Detection type included in the result messages.
            exit_training_on_sdc: Whether hardware precision failure must raise.

        Raises:
            RuntimeError: If precision failure is detected and exit_training_on_sdc is true.
        """
        for result in detect_ret_list:
            if result == 0:
                logger.info("Stress detection passed (%s)", detect_type)
            elif result == HARDWARE_FAILURE_CODE:
                message = (
                    f"Stress detection completed but found a hardware precision failure (possible SDC): "
                    f"detect_type={detect_type}, return_code={result}"
                )
                logger.error(message)
                if exit_training_on_sdc:
                    raise RuntimeError(message)
            else:
                reason = "test case execution failed" if result == 1 else "unrecognized return status"
                logger.warning(
                    "Stress detection did not produce a precision verdict (%s): detect_type=%s, return_code=%s",
                    reason, detect_type, result,
                )
