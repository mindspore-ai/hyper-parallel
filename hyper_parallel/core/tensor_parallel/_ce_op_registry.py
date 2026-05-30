# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""CE operator name registry.

Set of operator names on the CE decomposition path; elements are platform.get_op_name results.
Allows extension via register_loss_parallel_op_names (for test injection).
"""

from typing import FrozenSet, Set

__all__ = [
    "get_loss_parallel_op_names",
    "get_decomposed_ce_op_names",
    "register_loss_parallel_op_names",
    "is_loss_parallel_op",
    "is_decomposed_ce_op",
]


_BASE_CE_OP_NAMES: FrozenSet[str] = frozenset({
    "cross_entropy",
    "torch_cross_entropy",
    "torch_nn_CrossEntropyLoss",
    "mint_nn_functional_cross_entropy",
    "CrossEntropy",
})

_DECOMPOSED_CE_OP_NAMES: FrozenSet[str] = frozenset({
    "nll_loss",
    "log_softmax",
    "softmax",
    "torch_nll_loss",
    "torch_log_softmax",
    "torch_softmax",
    "nll_loss_forward",
    "nll_loss_backward",
    "log_softmax_backward",
    "softmax_backward",
    "mint_nn_functional_nll_loss",
    "mint_nn_functional_log_softmax",
    "mint_nn_functional_softmax",
    "NLLLoss",
    "LogSoftmax",
    "Softmax",
})

_EXTENDED_CE_OP_NAMES: Set[str] = set()


def get_loss_parallel_op_names() -> FrozenSet[str]:
    """Return CE-related operator name set.

    Returns:
        FrozenSet[str]: Full set of operator names on CE decomposition path.
    """
    return _BASE_CE_OP_NAMES | frozenset(_EXTENDED_CE_OP_NAMES)


def get_decomposed_ce_op_names() -> FrozenSet[str]:
    """Return decomposed CE-related operator name set.

    Returns:
        FrozenSet[str]: Set of operator names that are part of CE decomposition
                       but should not be called directly in loss_parallel context.
    """
    return _DECOMPOSED_CE_OP_NAMES


# Backward compatibility aliases (will be deprecated in future)
LOSS_PARALLEL_OP_NAMES = get_loss_parallel_op_names
DECOMPOSED_CE_OP_NAMES = get_decomposed_ce_op_names


def register_loss_parallel_op_names(*names: str) -> None:
    """Extend CE operator name set (for test injection).

    Args:
        *names: Operator names to register (platform.get_op_name results).
    """
    _EXTENDED_CE_OP_NAMES.update(names)


def unregister_loss_parallel_op_names(*names: str) -> None:
    """Remove registered CE operator names (for test cleanup).

    Args:
        *names: Operator names to remove.
    """
    _EXTENDED_CE_OP_NAMES.difference_update(names)


def is_loss_parallel_op(op_name: str) -> bool:
    """Check if operator is a CE entry point (e.g., cross_entropy).

    Args:
        op_name: Operator name (platform.get_op_name result).

    Returns:
        bool: Whether operator is a CE entry point.
    """
    return op_name in get_loss_parallel_op_names()


def is_decomposed_ce_op(op_name: str) -> bool:
    """Check if operator is a decomposed CE op (e.g., log_softmax, nll_loss).

    Args:
        op_name: Operator name (platform.get_op_name result).

    Returns:
        bool: Whether operator is a decomposed CE op.
    """
    return op_name in get_decomposed_ce_op_names()


def clear_extended_op_names() -> None:
    """Clear extended operator name set (for test cleanup)."""
    _EXTENDED_CE_OP_NAMES.clear()
