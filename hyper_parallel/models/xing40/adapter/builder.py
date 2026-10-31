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
"""Build the Xing4.0 model from the recipe's configuration fields."""

from typing import Any

from transformers import PreTrainedModel

from hyper_parallel.models import HyperAutoModelForCausalLM
from hyper_parallel.models.xing40.modeling_xing40 import Xing40Config

__all__ = ["build_model"]


def build_model(*, config: dict[str, Any], **options: Any) -> PreTrainedModel:
    """Build and parallelize the model through the shared builder.

    Args:
        config: Every model configuration field; the recipe is their only source.
        **options: Construction options the Trainer passes to the shared builder.

    Returns:
        The constructed, parallelized model.
    """
    return HyperAutoModelForCausalLM.from_config(Xing40Config(**config), **options)
