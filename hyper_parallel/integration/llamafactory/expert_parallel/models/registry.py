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

"""Registry primitives for model-specific expert-parallel patches."""

from collections.abc import Callable
from dataclasses import dataclass

# LlamaFactory is a PyTorch-only integration boundary.
# pylint: disable-next=forbidden-backend-import
from torch import nn


@dataclass(frozen=True)
class ExpertParallelModelPatch:
    """Describe the forward adaptation needed by one model family."""

    name: str
    supports: Callable[[nn.Module], bool]
    prepare: Callable[[nn.Module, object], None]
