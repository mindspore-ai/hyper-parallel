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
"""MindSpore adaptation for optional explicit Chunk KDA ACLNN stages."""
from typing import Any

import mindspore as ms

from hyper_parallel.core.dtensor.dtensor import DTensor

# Positional slots of cu_seqlens, chunk_indices and chunk_size in each bridge.
# A missing indices slot means cu_seqlens is a subinterval, not a chunk plan.
_METADATA = {
    "npu_chunk_kda_fwd_prepare": (10, 11, 6),
    "npu_chunk_kda_pre_process_fwd": (4, None, 5),
    "npu_chunk_kda_fwd_h": (7, 8, 6),
    "npu_chunk_kda_fwd_finalize": (6, 7, None),
    "npu_chunk_kda_bwd_prepare": (8, 9, 6),
    "npu_chunk_kda_pre_process_bwd": (8, None, 6),
    "npu_chunk_kda_bwd_dhu": (10, 11, 6),
    "npu_chunk_kda_bwd_finalize": (24, 25, 17),
}


def _host_ints(value):
    """Normalize host metadata; tensor data is copied to the host explicitly."""
    if value is None:
        return None
    if isinstance(value, ms.Tensor):
        value = value.asnumpy().reshape(-1).tolist()
    if hasattr(value, "reshape"):
        value = value.reshape(-1).tolist()
    return [int(item) for item in value]


def run_chunk_kda_stage(name: str, *args: Any) -> Any:
    """Call an optional bridge without constructing an autograd closure."""
    # Import lazily to reuse the existing extension and metadata builder.
    from hyper_parallel.platform.mindspore.custom_ops import custom_op_impl  # pylint: disable=C0415

    op = getattr(custom_op_impl._custom_ops, name, None)  # pylint: disable=protected-access
    if op is None:
        raise RuntimeError(
            f"{name} is unavailable in the loaded MindSpore extension. "
            "Rebuild the HP custom-op extension with Chunk KDA stage bridges; "
            "install and source a matching FLA-NPU OPP before calling this stage."
        )
    values = list(args)
    if any(isinstance(value, DTensor) for value in values):
        raise TypeError("Chunk KDA stages require local Tensors; the caller manages DTensor/CP distribution.")
    if name in _METADATA:
        cu_pos, index_pos, size_pos = _METADATA[name]
        values[cu_pos] = _host_ints(values[cu_pos])
        if index_pos is not None:
            values[index_pos] = _host_ints(values[index_pos])
            if values[index_pos] is None and size_pos is not None:
                values[index_pos] = custom_op_impl._canonical_chunk_indices(  # pylint: disable=protected-access
                    values[cu_pos], values[size_pos],
                )
    if name == "npu_chunk_kda_merge_fwd_bwd" and not values[0].is_contiguous():
        raise ValueError("state must be contiguous because merge updates it in place.")
    values = [value.contiguous() if isinstance(value, ms.Tensor) else value for value in values]
    result = op(*values)
    if name in ("npu_chunk_kda_fwd_finalize", "npu_chunk_kda_pre_process_bwd",
                "npu_chunk_kda_merge_fwd_bwd"):
        return result
    # PYBOOST_CALLER(1) unwraps its singleton result; FLA forward preprocess
    # exposes a one-element tuple, unlike the other single-output stages.
    if name == "npu_chunk_kda_pre_process_fwd":
        return (result,)
    return tuple(None if tuple(tensor.shape) == (0,) else tensor for tensor in result)
