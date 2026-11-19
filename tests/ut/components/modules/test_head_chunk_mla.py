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
"""Independent literal MLA equations and hardware-independent API checks."""

from __future__ import annotations

import copy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from hyper_parallel.components.modules import HeadChunkMLA
from hyper_parallel.components.modules._head_chunk_mla_math import head_stage_schedule
from hyper_parallel.components.modules._head_chunk_mla_ops import _EagerAttention, _NpuAttention, _PackedNpuAttention


class _LiteralRMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.variance_epsilon = 1e-6

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        input_dtype = tensor.dtype
        normalized = tensor.float()
        variance = normalized.square().mean(-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * normalized.to(input_dtype)


class _LiteralExpandedMLA(nn.Module):
    """Full-head oracle; it does not call any candidate projection/FA helper."""

    def __init__(self) -> None:
        super().__init__()
        self.num_heads, self.q_lora_rank, self.kv_lora_rank = 8, 48, 32
        self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim = 16, 16, 16
        self.scaling, self.attention_dropout = 32**-0.5, 0.0
        self.is_causal, self.rotary_interleaved, self.sliding_window = True, False, None
        self.linear_qkv = nn.Linear(128, 96, bias=False)
        self.q_a_layernorm, self.kv_a_layernorm = _LiteralRMSNorm(48), _LiteralRMSNorm(32)
        self.q_b_proj = nn.Linear(48, 256, bias=False)
        self.kv_b_proj = nn.Linear(32, 256, bias=False)
        self.o_proj = nn.Linear(128, 128, bias=False)

    def forward(self, hidden: torch.Tensor, positions: tuple | None) -> torch.Tensor:
        batch, length = hidden.shape[:2]
        query_low, compressed_kv, key_rope = self.linear_qkv(hidden).split((48, 32, 16), dim=-1)
        query = self.q_b_proj(self.q_a_layernorm(query_low)).view(batch, length, 8, 32).transpose(1, 2)
        query_nope, query_rope = query.split((16, 16), dim=-1)
        expanded = self.kv_b_proj(self.kv_a_layernorm(compressed_kv)).view(batch, length, 8, 32).transpose(1, 2)
        key_nope, value = expanded.split((16, 16), dim=-1)
        key_rope = key_rope.unsqueeze(1)
        if positions is not None:
            cos, sin = positions
            if cos.ndim == 2:
                cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
            cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)

            def rotate(tensor: torch.Tensor) -> torch.Tensor:
                first, second = tensor.chunk(2, dim=-1)
                return tensor * cos + torch.cat((-second, first), dim=-1) * sin

            query_rope, key_rope = rotate(query_rope), rotate(key_rope)
        query = torch.cat((query_nope, query_rope), dim=-1)
        key = torch.cat((key_nope, key_rope.expand(-1, 8, -1, -1)), dim=-1)
        scores = query.matmul(key.transpose(-1, -2)) * self.scaling
        mask = torch.ones((length, length), dtype=torch.bool).triu(1)
        probability = scores.masked_fill(mask, -torch.inf).softmax(-1, dtype=torch.float32).to(hidden.dtype)
        attended = probability.matmul(value).transpose(1, 2).reshape(batch, length, -1)
        return self.o_proj(attended)



class TestHeadChunkMLA(unittest.TestCase):
    """CPU equations are independent; actual collectives are tested on NPU."""

    def test_packed_native_layout_and_backward_metadata(self) -> None:
        """TND transposes all operands and reuses invocation-owned document ends."""
        query = torch.randn(1, 2, 7, 4)
        tnd = query.squeeze(0).transpose(0, 1).contiguous()
        maximum, denominator = torch.randn(7, 2, 8), torch.randn(7, 2, 8)
        ends = [3, 7]
        backend = _PackedNpuAttention(ends)
        ends[0] = 2
        native = SimpleNamespace(
            npu_fusion_attention=Mock(return_value=(tnd, maximum, denominator)),
            npu_fusion_attention_grad=Mock(return_value=(tnd, tnd, tnd)),
        )
        with patch.dict(sys.modules, {"torch_npu": native}):
            output, saved = backend.forward_saved(query, query, query, 0.5)
            gradients = backend.backward_saved(query, query, query, output, 0.5, saved)
        torch.testing.assert_close(output, query, rtol=0, atol=0)
        for grad in gradients:
            torch.testing.assert_close(grad, query, rtol=0, atol=0)
        for operation in (native.npu_fusion_attention, native.npu_fusion_attention_grad):
            args, options = operation.call_args
            self.assertEqual(args[-1], "TND")
            self.assertEqual(options["sparse_mode"], 3)
            self.assertEqual(options["actual_seq_qlen"], [3, 7])
            self.assertEqual(options["actual_seq_kvlen"], [3, 7])
            torch.testing.assert_close(args[0], tnd, rtol=0, atol=0)
        self.assertIs(native.npu_fusion_attention_grad.call_args.kwargs["softmax_max"], maximum)
        self.assertEqual(native.npu_fusion_attention.call_count, 1)

    def test_native_fa_saved_state_and_mask_contract(self) -> None:
        """Use the old FA ABI, native mask modes and only three zero-dropout states."""
        for length, ends, mode in ((7, None, 0), (2049, None, 2)):
            with self.subTest(length=length, ends=ends):
                query = torch.zeros(1, length, 2, 4).transpose(1, 2)
                backend = _NpuAttention
                shaped = query.contiguous()
                native = SimpleNamespace(
                    npu_fusion_attention=Mock(return_value=(shaped, shaped, shaped, shaped, 0, 0, 0)),
                    npu_fusion_attention_grad=Mock(return_value=(shaped, shaped, shaped, None, None)),
                )
                with patch.dict(sys.modules, {"torch_npu": native}):
                    output, saved = backend.forward_saved(query, query, query, 0.5)
                    gradients = backend.backward_saved(query, query, query, output, 0.5, saved)
                self.assertEqual(len(saved), 3)
                self.assertEqual(tuple(output.shape), tuple(query.shape))
                self.assertEqual(tuple(gradients[0].shape), tuple(query.shape))
                self.assertEqual(native.npu_fusion_attention.call_count, 1)
                self.assertEqual(native.npu_fusion_attention_grad.call_count, 1)
                for operation in (native.npu_fusion_attention, native.npu_fusion_attention_grad):
                    args, options = operation.call_args
                    self.assertEqual(options["sparse_mode"], mode)
                    self.assertEqual(options.get("actual_seq_qlen"), ends)
                    self.assertEqual(options["keep_prob"], 1.0)
                    self.assertTrue(all(tensor.is_contiguous() for tensor in args[:3]))
        candidate = HeadChunkMLA(module=_LiteralExpandedMLA(), head_chunk_size=4)
        self.assertEqual(candidate.attention_backend, "npu")

    def test_literal_parity_tail_and_frozen_leaves(self) -> None:
        """Check all gradients through repeated calls, including a partial last stage."""
        for stage, rope, overlap, frozen in ((1, False, False, False), (3, True, False, False),
                                             (4, True, True, False), (8, True, True, True)):
            with self.subTest(stage=stage, rope=rope, overlap=overlap, frozen=frozen):
                torch.manual_seed(123)
                reference = _LiteralExpandedMLA()
                if frozen:
                    reference.q_b_proj.weight.requires_grad_(False)
                candidate = HeadChunkMLA(module=copy.deepcopy(reference), head_chunk_size=stage,
                                         attention_backend="eager", overlap_backward=overlap)
                for _ in range(2):
                    reference.zero_grad(set_to_none=True)
                    candidate.zero_grad(set_to_none=True)
                    hidden = torch.randn(2, 7, 128, requires_grad=not frozen)
                    other = hidden.detach().clone().requires_grad_(not frozen)
                    angle = torch.randn(1, 7, 8).repeat(1, 1, 2)
                    positions = (angle.cos(), angle.sin()) if rope else None
                    expected, actual = reference(hidden, positions), candidate(other, positions)[0]
                    probe = torch.randn_like(expected)
                    (expected * probe).sum().backward()
                    (actual * probe).sum().backward()
                    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=5e-6)
                    if not frozen:
                        torch.testing.assert_close(other.grad, hidden.grad, rtol=2e-5, atol=5e-6)
                    for (name, left), (other_name, right) in zip(
                            reference.named_parameters(), candidate.named_parameters()):
                        self.assertEqual(name, other_name)
                        if left.requires_grad:
                            torch.testing.assert_close(right.grad, left.grad, rtol=2e-5, atol=5e-6)
                        else:
                            self.assertIsNone(right.grad)

    def test_identity_no_replay_and_inference(self) -> None:
        """Wrapping preserves every parameter and backward never calls FA forward."""
        source = _LiteralExpandedMLA()
        original = dict(source.named_parameters())
        candidate = HeadChunkMLA(module=source, head_chunk_size=4, attention_backend="eager")
        self.assertEqual(set(original), set(candidate.state_dict()))
        for name, parameter in candidate.named_parameters():
            self.assertIs(parameter, original[name])
        hidden = torch.randn(1, 7, 128, requires_grad=True)
        with patch.object(_EagerAttention, "forward_saved", wraps=_EagerAttention.forward_saved) as forward:
            with patch.object(_EagerAttention, "backward_saved", wraps=_EagerAttention.backward_saved) as backward:
                candidate(hidden)[0].sum().backward()
                self.assertEqual(forward.call_count, 2)
                self.assertEqual(backward.call_count, 2)
        with torch.no_grad():
            self.assertFalse(candidate(hidden)[0].requires_grad)

    def test_schedule_and_unsupported_inputs(self) -> None:
        """Do not silently ignore packed documents, gates, masks or another CP wrapper."""
        self.assertEqual(head_stage_schedule(20, 8, 4), ((0, 8), (8, 8), (16, 4)))
        for values in ((8, 3, 2), (8, 10, 1), (True, 1, 1), (8, 0, 1)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                head_stage_schedule(*values)
        for name, value in (("use_output_gate", True), ("mla_cp_runtime", object()),
                            ("attention_dropout", 0.1), ("rotary_interleaved", True)):
            source = _LiteralExpandedMLA()
            setattr(source, name, value)
            with self.subTest(name=name), self.assertRaises(ValueError):
                HeadChunkMLA(module=source, head_chunk_size=4, attention_backend="eager")
        candidate = HeadChunkMLA(module=_LiteralExpandedMLA(), head_chunk_size=4, attention_backend="eager")
        for options in ({"attention_mask": torch.ones(7, 7)}, {"actual_seq_len": [2, 7]},
                        {"past_key_values": object()}, {"output_attentions": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                candidate(torch.randn(1, 7, 128), **options)

    def test_offload_capacity_and_backend_contract(self) -> None:
        """CPU is not a silent offload fallback and device slots are bounded."""
        for options in ({"prefetch_slots": 0}, {"prefetch_slots": True}, {"max_host_bytes": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                HeadChunkMLA(module=_LiteralExpandedMLA(), head_chunk_size=4, **options)
        candidate = HeadChunkMLA(module=_LiteralExpandedMLA(), head_chunk_size=4,
                                 attention_backend="eager", offload_saved_stats=True)
        with self.assertRaisesRegex(ValueError, "requires NPU"):
            candidate(torch.randn(1, 7, 128))

    def test_child_wrapping_cannot_bypass_hooks(self) -> None:
        """Reject child FSDP applied before or after head-chunk construction."""
        source = _LiteralExpandedMLA()
        source.q_b_proj.hsdp_scheduler = object()
        with self.assertRaisesRegex(ValueError, "child FSDP"):
            HeadChunkMLA(module=source, head_chunk_size=4, attention_backend="eager")
        candidate = HeadChunkMLA(module=_LiteralExpandedMLA(), head_chunk_size=4, attention_backend="eager")
        candidate.kv_b_proj.hsdp_scheduler = object()
        with self.assertRaisesRegex(ValueError, "child FSDP"):
            candidate(torch.randn(1, 7, 128))
