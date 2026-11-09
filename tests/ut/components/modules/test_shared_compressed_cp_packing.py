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
"""Packed RoPE positions and physical CP shard geometry."""

import copy
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from hyper_parallel.components.modules import shared_compressed_dsa_attention as attention
from tests.common.mark_utils import arg_mark


def _attention_module() -> attention.SharedCompressedDSAAttention:
    """Build executable sink attention without a model-library dependency."""
    source = nn.Module()
    attributes = {
        "config": SimpleNamespace(o_groups=1, qk_rope_head_dim=2), "layer_idx": 0,
        "num_heads": 1, "num_key_value_groups": 1, "compress_ratio": 0,
        "is_kv_source": False, "is_index_source": False, "kv_source_layer_idx": None,
        "index_source_layer_idx": None, "candidate_source_layer_idx": None,
        "head_dim": 4, "sliding_window": 3, "scaling": 0.5,
    }
    for name, value in attributes.items():
        setattr(source, name, value)
    source.sinks = nn.Parameter(torch.zeros(1))
    for name in ("q_a_proj", "q_b_proj", "kv_proj", "o_a_proj", "o_b_proj"):
        source.add_module(name, nn.Linear(4, 4, bias=False))
    source.q_a_norm = nn.Identity()
    source.kv_norm = nn.Identity()
    return attention.SharedCompressedDSAAttention(source)


class TestPackedCPPositions(unittest.TestCase):
    """Exercise document position resets through real attention and autograd."""

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_packed_reset_positions_match_unsharded_outputs_and_gradients(self) -> None:
        """Feature: Packed-document positions under CP.
        Description: Compare split forward/backward with a full document-masked attention.
        Expectation: Outputs and all gradients match despite reset RoPE positions.
        """
        # The synchronous gather oracle uses remote projections without distributed setup.
        # pylint: disable=protected-access
        torch.manual_seed(41)
        reference = _attention_module()
        candidate = copy.deepcopy(reference)
        full_input = torch.randn(1, 8, 4, requires_grad=True)
        sharded_input = full_input.detach().clone().requires_grad_()
        positions = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4]])
        angles = positions.float().unsqueeze(-1).expand(-1, -1, 2) * 0.1
        cos, sin = angles.cos(), angles.sin()
        boundaries = torch.tensor([0, 3, 8])
        expected = reference(
            full_input, {"main": (cos, sin)}, positions, None,
            shared_attention_state=attention.SharedCompressedAttentionState(),
            packed_seq_params=attention.SharedCompressedPackedSequence(boundaries, 0, 8, 8),
        )[0]
        outputs = []
        for rank in range(2):
            local = slice(rank * 4, (rank + 1) * 4)
            remote = slice((1 - rank) * 4, (2 - rank) * 4)

            def _gather(value: torch.Tensor, dim: int, remote_slice: slice = remote,
                        local_rank: int = rank) -> torch.Tensor:
                """Project the other shard to emulate a differentiable gather."""
                remote_value, _ = candidate._project_raw_kv(
                    sharded_input[:, remote_slice], cos[:, remote_slice], sin[:, remote_slice], None,
                )
                parts = [value, remote_value] if local_rank == 0 else [remote_value, value]
                return torch.cat(parts, dim=dim)

            outputs.append(candidate(
                sharded_input[:, local], {"main": (cos[:, local], sin[:, local])}, positions[:, local], None,
                shared_attention_state=attention.SharedCompressedAttentionState(),
                shared_attention_cp_context=attention.SharedCompressedAttentionCPContext(2, rank, _gather),
                packed_seq_params=attention.SharedCompressedPackedSequence(boundaries, rank * 4, 4, 8),
            )[0])
        actual = torch.cat(outputs, dim=1)
        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(sharded_input.grad, full_input.grad)
        for parameter, target in zip(candidate.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter.grad, target.grad)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_packed_cp_rejects_inconsistent_physical_geometry(self) -> None:
        """Feature: Packed CP geometry validation.
        Description: Supply incorrect physical offsets, lengths and total sizes.
        Expectation: Each invalid geometry fails before projections or communication.
        """
        module = _attention_module()
        context = attention.SharedCompressedAttentionCPContext(2, 1, lambda value, dim: value)
        for start, length, total in ((0, 4, 8), (4, 3, 8), (4, 4, 12)):
            with self.subTest(start=start, length=length, total=total), self.assertRaisesRegex(
                    ValueError, "packed sequence geometry does not match"):
                module(
                    torch.zeros(1, 4, 2), {}, torch.tensor([[1, 2, 3, 4]]), None,
                    shared_attention_state=attention.SharedCompressedAttentionState(),
                    shared_attention_cp_context=context,
                    packed_seq_params=attention.SharedCompressedPackedSequence(
                        torch.tensor([0, 3, total]), start, length, total,
                    ),
                )
        with self.assertRaisesRegex(ValueError, "contiguous global position_ids"):
            module(torch.zeros(1, 4, 2), {}, torch.tensor([[1, 2, 3, 4]]), None,
                   shared_attention_state=attention.SharedCompressedAttentionState(),
                   shared_attention_cp_context=context)
