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
"""Attention families with explicit mathematical and parallel members."""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hyper_parallel.components.modules.attention.deepseek_sparse_attention import (
        CPDeepseekSparseAttentionModule, DeepseekSparseAttention, DeepseekSparseAttentionModule,
    )

__all__ = [
    "DeepseekSparseAttention", "DeepseekSparseAttentionModule", "CPDeepseekSparseAttentionModule",
]


def __getattr__(name: str) -> Any:
    """Keep sequence-layout imports independent of attention family registration."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"{__name__}.deepseek_sparse_attention")
    value = getattr(module, name)
    globals()[name] = value
    return value
