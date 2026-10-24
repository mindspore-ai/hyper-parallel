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
"""Dedicated configuration for online stress detection."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class OpStressDetectConfig:
    """Configure operator-level NPU stress detection rounds.

    Args:
        detection_interval: Completed optimizer steps between rounds; zero disables
            callback creation and registration.
        num_detections: Consecutive checks per detection type in each round.
        detect_type: AIC, HCCS, or both types in AIC-then-HCCS order.
        exit_training_on_sdc: Raise on hardware precision failure when true;
            otherwise log an error and continue training.
    """

    detection_interval: int = 0
    num_detections: int = 1
    detect_type: Literal["aic", "hccs", "all"] = "aic"
    exit_training_on_sdc: bool = False

    def __post_init__(self) -> None:
        """Reject invalid stress detection settings at the configuration boundary."""
        if (
            isinstance(self.detection_interval, bool)
            or not isinstance(self.detection_interval, int)
            or self.detection_interval < 0
        ):
            raise ValueError("stress_detect.op_stress_detect.detection_interval must be a non-negative integer")
        if (
            isinstance(self.num_detections, bool)
            or not isinstance(self.num_detections, int)
            or self.num_detections < 1
        ):
            raise ValueError("stress_detect.op_stress_detect.num_detections must be a positive integer")
        if self.detect_type not in ("aic", "hccs", "all"):
            raise ValueError("stress_detect.op_stress_detect.detect_type must be one of: aic, hccs, all")
        if not isinstance(self.exit_training_on_sdc, bool):
            raise ValueError("stress_detect.op_stress_detect.exit_training_on_sdc must be a boolean")


@dataclass
class ModelStressDetectConfig:
    """Reserve independent model-level stress detection settings.

    Args:
        detection_interval: Completed optimizer steps between rounds. Only zero
            is currently supported because model-level detection is not implemented.
    """

    detection_interval: int = 0

    def __post_init__(self) -> None:
        """Reject invalid intervals and attempts to enable unimplemented detection."""
        if (
            isinstance(self.detection_interval, bool)
            or not isinstance(self.detection_interval, int)
            or self.detection_interval < 0
        ):
            raise ValueError("stress_detect.model_stress_detect.detection_interval must be a non-negative integer")
        if self.detection_interval > 0:
            raise NotImplementedError("stress_detect.model_stress_detect is reserved and not implemented yet")


@dataclass
class StressDetectConfig:
    """Group independent operator-level and model-level stress detection settings.

    Args:
        op_stress_detect: Operator-level detection cadence and AIC/HCCS settings.
        model_stress_detect: Reserved model-level detection configuration.
    """

    op_stress_detect: OpStressDetectConfig = field(default_factory=OpStressDetectConfig)
    model_stress_detect: ModelStressDetectConfig = field(default_factory=ModelStressDetectConfig)
