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
"""Dense Family configurations shared by CPU and real-device tests."""

from hyper_parallel.trainer.config import TrainerConfig
from hyper_parallel.trainer.config.resolver import resolve_config


def linear_config(*, quantized: bool = False) -> TrainerConfig:
    """Configure ordinary Linear or its existing MXFP8 implementation.

    Args:
        quantized: Reference the named MXFP8 scheme when true.
    """
    replacement = {"_target_": "hyper_parallel.components.modules.Linear"}
    if quantized:
        replacement["low_precision_dtype_scheme"] = "dense"
    return resolve_config({
        "model": {"_target_": "torch.nn.Identity"},
        "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.01},
        "training": {"low_precision": {"dtype_schemes": {
            "dense": {"weight_format": "mxfp8", "act_format": "mxfp8"},
        }}},
        "plan_overrides": [{"match": "0", "module_type": "torch.nn.Linear", "exact_type": True,
                            "replace_module": replacement}],
    })
