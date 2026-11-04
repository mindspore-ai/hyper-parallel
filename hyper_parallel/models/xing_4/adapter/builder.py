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
"""Build the Xing4 model from the recipe's configuration fields."""

from __future__ import annotations

from typing import Any

from transformers import PreTrainedModel

from hyper_parallel.models import HyperAutoModelForCausalLM
from hyper_parallel.models.xing_4.adapter.configuration import create_config
from hyper_parallel.models.xing_4.adapter.conversion.module_replacement import Xing4FusedHyperConnection
from hyper_parallel.models.xing_4.adapter.training import Xing4ForCausalLM
from hyper_parallel.models.xing_4.transformer.modeling_xing4_0 import Xing4_0HyperConnection

__all__ = ["build_model"]


def build_model(*, config: dict[str, Any], **options: Any) -> PreTrainedModel:
    """Build and parallelize the model through the shared builder.

    Args:
        config: Model configuration with HF fields and HP training options.
        **options: Construction options the Trainer passes to the shared builder.

    Returns:
        The constructed, parallelized model.
    """
    resolved = create_config(config)
    model = HyperAutoModelForCausalLM.from_config(resolved, **options)
    if not any(isinstance(module, Xing4ForCausalLM) for module in model.modules()):
        raise TypeError("Xing4 training requires the registered Xing4ForCausalLM adapter")
    for name, module in model.named_modules():
        if isinstance(module, Xing4_0HyperConnection) and not isinstance(module, Xing4FusedHyperConnection):
            raise ValueError(f"Xing4 training requires a Xing4FusedHyperConnection plan override for {name}")
    return model
