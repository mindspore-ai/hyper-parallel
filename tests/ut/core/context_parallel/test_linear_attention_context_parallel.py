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
"""Unit tests for linear-attention context parallel helpers."""
import pytest
import torch

from hyper_parallel.core.context_parallel.linear_attention_context_parallel import (
    LinearAttentionContextParallel,
    _merge_gdn_prefix_state_summaries_torch,
    _pack_gdn_state_summary,
    _slice_qkv_local_cp,
    _unpack_gdn_state_summary,
)


@pytest.mark.parametrize("mode", ("ulysses", "p2p", "all_gather"))
def test_linear_attention_cp_accepts_supported_modes(mode):
    """All public linear-attention CP modes can be constructed."""
    assert LinearAttentionContextParallel(mode=mode).mode == mode


def test_linear_attention_cp_rejects_unknown_mode():
    """An unsupported execution mode fails before patching a module."""
    with pytest.raises(NotImplementedError, match="currently supports"):
        LinearAttentionContextParallel(mode="unknown")


def test_slice_qkv_local_cp_slices_each_projection_independently():
    """Fused Q/K/V projections preserve their channel boundaries when sharded."""
    tensor = torch.arange(20, dtype=torch.float32).reshape(1, 1, 20)

    actual = _slice_qkv_local_cp(
        tensor,
        key_dim=6,
        value_dim=8,
        dim=-1,
        cp_rank=1,
        cp_size=2,
    )

    expected = torch.tensor([[[3, 4, 5, 9, 10, 11, 16, 17, 18, 19]]], dtype=torch.float32)
    torch.testing.assert_close(actual, expected)


def test_gdn_summary_pack_merge_and_backward():
    """Packed summaries compose as affine state transitions and remain differentiable."""
    state_ext = torch.tensor(
        [[[[[1.0], [2.0]]]], [[[[3.0], [4.0]]]]],
        requires_grad=True,
    )
    transition = torch.tensor(
        [
            [[[[2.0, 0.0], [0.0, 3.0]]]],
            [[[[4.0, 0.0], [0.0, 5.0]]]],
        ],
        requires_grad=True,
    )

    packed = _pack_gdn_state_summary(state_ext, transition)
    unpacked_state, unpacked_transition = _unpack_gdn_state_summary(packed, v_head_dim=1)
    actual = _merge_gdn_prefix_state_summaries_torch(
        unpacked_state,
        unpacked_transition,
        rank=2,
    )

    expected = transition[1] @ state_ext[0] + state_ext[1]
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert state_ext.grad is not None
    assert transition.grad is not None
