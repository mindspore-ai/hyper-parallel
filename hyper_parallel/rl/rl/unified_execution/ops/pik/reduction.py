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
"""Opaque compiler boundary for the fixed FP32 tree; no device dependency at import."""
import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental.proxy_tensor import get_proxy_mode


def _run_tree(partials: torch.Tensor) -> torch.Tensor:
    """Launch exactly the same kernel in eager execution and compiled graphs."""
    if partials.device.type != "npu":
        raise ValueError("PIK fixed tree requires an NPU tensor")
    # Triton is optional for CPU-only model/configuration tools.
    from ._triton_reduction import reduce_tree  # pylint: disable=import-outside-toplevel

    return reduce_tree(partials)


@torch.library.custom_op("hyper_rl::pik_tree", mutates_args=())
def _tree_op(partials: torch.Tensor) -> torch.Tensor:
    """Keep Triton opaque to Dynamo/AOT while exposing a Fake/Meta implementation."""
    return _run_tree(partials)


@_tree_op.register_fake
def _tree_fake(partials: torch.Tensor) -> torch.Tensor:
    """Expose dtype and shape to the inference/training compiler."""
    return partials.new_empty(partials.shape[1:], dtype=torch.float32)


def fixed_tree(partials: torch.Tensor) -> torch.Tensor:
    """Combine packed BF16 leaves or FP32 rank roots without rank-dependent casts."""
    if partials.ndim != 3 or partials.shape[0] < 1 or partials.shape[0] & (partials.shape[0] - 1):
        raise ValueError("tree requires [power-of-two leaves, rows, columns]")
    if partials.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("tree accepts only BF16 leaves or FP32 internal nodes")
    if partials.shape[0] == 1:
        return partials[0].float()
    inputs = partials.contiguous()
    if torch.compiler.is_compiling() or isinstance(inputs, FakeTensor) or get_proxy_mode() is not None:
        return _tree_op(inputs)
    # Avoid Python custom-op dispatch overhead when no compiler is tracing.
    return _run_tree(inputs)
