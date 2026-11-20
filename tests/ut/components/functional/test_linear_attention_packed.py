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
"""Packed linear-attention sample isolation and layout regression tests."""

from dataclasses import FrozenInstanceError
from typing import Any
from unittest.mock import patch
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from hyper_parallel.components.modules.attention.linear_attention.sequence_layout import (
    PackedDocumentInfo, PackedSequenceInfo, resolve_packed_sequence,
)
from hyper_parallel.components.functional.linear_attention_convolution import (
    packed_causal_conv1d,
)
from hyper_parallel.components.modules.attention.linear_attention.sequence_layout import (
    packed_eager,
)
from hyper_parallel.components.modules.attention.kimi_delta_attention import KimiDeltaAttention
from hyper_parallel.components.functional.gated_delta_net import chunk_gated_delta_rule
from hyper_parallel.data.batching.runtime_input import RuntimeInputContext
from hyper_parallel.models.qwen3_5.adapter.data.runtime import Qwen35PackedRuntime
from hyper_parallel.models.kimi_k3.adapter.data.runtime import KimiK3PackedRuntime
from hyper_parallel.models.qwen3_5.adapter.linear_attention_decoder import qwen3_5_linear_attention_decoder_wrapper
from hyper_parallel.distributed._builder.forward_rewriter import _commit_forward_rewrite
from tests.common.mark_utils import arg_mark


class TestLinearAttentionPacked(unittest.TestCase):
    """Check real reset semantics independently of device and transport."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_global_offsets_survive_local_cuts_and_mutation(self):
        """Feature: Packed document snapshots.

        Description: Slice CP intervals and mutate the original offset tensor.
        Expectation: Snapshots preserve true document boundaries independently of artificial CP cuts.
        """
        offsets = torch.tensor([0, 6000, 13000, 16384], dtype=torch.int32)
        info = resolve_packed_sequence({"cu_seq_lens": offsets}, 4096, cp_rank=1, cp_size=4)
        self.assertEqual(info.local_offsets, (0, 1904, 4096))
        self.assertTrue(info.continues_previous)
        self.assertFalse(info.transfer_active)
        self.assertEqual(info.ulysses(2, 1).local_offsets, (0, 6000, 8192))
        offsets[1] = 4000
        self.assertEqual(info.global_offsets, (0, 6000, 13000, 16384))
        plan = info.materialize(torch.device("cpu"), 64)
        self.assertEqual(plan.chunk_offsets.tolist(), [0, 30, 65])
        self.assertIs(info.materialize(torch.device("cpu"), 64), plan)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_interval_views_share_snapshot_and_preserve_reset_edges(self):
        """Feature: Packed interval views.

        Description: Construct local interval views from one immutable document snapshot.
        Expectation: Views share metadata and preserve reset edges without sharing recurrent states.
        """
        offsets = [0, 3, 8, 13, 20]
        documents = PackedDocumentInfo(offsets)
        root = PackedSequenceInfo(documents, 0, 20)
        offsets[1] = 2
        self.assertEqual(root.global_offsets, (0, 3, 8, 13, 20))
        with self.assertRaises(FrozenInstanceError):
            documents.offsets = (0, 20)
        for start in range(20):
            for end in range(start + 1, 21):
                view = root.interval(start, end)
                self.assertIs(view.documents, documents)
                self.assertEqual(view.local_offsets,
                                 (0, *(x - start for x in documents.offsets if start < x < end), end - start))
                self.assertEqual(view.continues_previous, start not in documents.offsets)
                self.assertEqual(view.continues_next, end not in documents.offsets)
                self.assertEqual(view.preceding_sample_tokens,
                                 start - max(x for x in documents.offsets if x <= start))
        first = root.interval(5, 10)
        expanded = first.ulysses(2, 1)
        self.assertIs(expanded.documents, documents)
        self.assertEqual(expanded.local_offsets, (0, 3, 8, 10))
        self.assertIs(root.interval(5, 10), first)
        self.assertIsNot(first.materialize(torch.device("cpu"), 64),
                         expanded.materialize(torch.device("cpu"), 64))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_aliases_fail_instead_of_silently_disagreeing(self):
        """Feature: Packed argument validation.

        Description: Supply conflicting or ambiguous document metadata aliases.
        Expectation: Invalid aliases are rejected before communication.
        """
        with self.assertRaisesRegex(ValueError, "disagree"):
            resolve_packed_sequence({"cu_seq_lens": [0, 3, 8], "cu_seqlens": [0, 4, 8]}, 8)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            resolve_packed_sequence({"cu_seq_lens": [0, 3, 3, 8]}, 8)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            resolve_packed_sequence({"seq_idx": torch.zeros(8)}, 8)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_segmented_conv_halo_and_all_gradients(self):
        """Feature: Segmented convolution halos.

        Description: Compare fragment convolution and halo gradients with independent full documents.
        Expectation: Outputs and all gradients match the per-document convolution reference.
        """
        torch.manual_seed(82)
        offsets = (0, 5, 17, 23)
        conv = nn.Conv1d(3, 3, 4, groups=3, padding=3, bias=True).double()
        inputs = torch.randn(1, 23, 3, dtype=torch.float64, requires_grad=True)
        reference = torch.cat(
            [
                F.silu(conv(inputs[:, start:end].transpose(1, 2))[:, :, : end - start]).transpose(1, 2)
                for start, end in zip(offsets, offsets[1:])
            ],
            dim=1,
        )
        pieces = []
        for start, end in ((0, 8), (8, 16), (16, 23)):
            info = PackedSequenceInfo(offsets, start, end)
            halo = None if start == 0 else inputs[:, start - 3 : start]
            pieces.append(packed_causal_conv1d(inputs[:, start:end], conv, info, halo))
        actual = torch.cat(pieces, dim=1)
        torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)
        grad = torch.randn_like(actual)
        tensors = (inputs, conv.weight, conv.bias)
        expected_grads = torch.autograd.grad(reference, tensors, grad, retain_graph=True)
        actual_grads = torch.autograd.grad(actual, tensors, grad)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, atol=1e-12, rtol=1e-12)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_kda_layer_tnd_and_cross_sample_gradient(self):
        """Feature: Packed KDA sample isolation.

        Description: Run the layer through packed aliases and backpropagate one sample.
        Expectation: Convolution and recurrent state do not leak gradients between samples.
        """
        torch.manual_seed(31)
        module = KimiDeltaAttention(
            hidden_size=16, num_heads=2, head_k_dim=4, head_v_dim=4, conv_kernel_size=3, chunk_size=4
        ).double()
        inputs = torch.randn(12, 16, dtype=torch.float64, requires_grad=True)
        actual = module(inputs, cu_seq_lens=torch.tensor([0, 5, 12]))
        expected = torch.cat([module(inputs[:5].unsqueeze(0)), module(inputs[5:].unsqueeze(0))], dim=1)[0]
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
        leak = torch.autograd.grad(actual[5:].square().sum(), inputs, retain_graph=True)[0][:5]
        self.assertEqual(torch.count_nonzero(leak).item(), 0)
        tensors = (inputs, *module.parameters())
        grad = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(actual, tensors, grad)
        expected_grads = torch.autograd.grad(expected, tensors, grad)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, atol=8e-6, rtol=1e-5)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_gdn_tnd_core_and_initial_state_slots(self):
        """Feature: Packed GDN document states.

        Description: Compare packed evaluation with independent local sample recurrences.
        Expectation: Outputs, gradients and initial-state slots follow document boundaries.
        """
        torch.manual_seed(73)
        query, key, value = [torch.randn(13, 2, 4, requires_grad=True) for _ in range(3)]
        gate = (-torch.rand(13, 2)).requires_grad_()
        beta = torch.rand(13, 2, requires_grad=True)
        initial = torch.randn(2, 2, 4, 4, requires_grad=True)
        tensors = (query, key, value, gate, beta)
        options = {"chunk_size": 4, "output_final_state": True, "use_qk_l2norm_in_kernel": True}
        actual, final = chunk_gated_delta_rule(*tensors, cu_seqlens=(0, 5, 13), initial_state=initial, **options)
        parts = [chunk_gated_delta_rule(
            *(tensor[start:end] for tensor in tensors), initial_state=initial[index:index + 1], **options
        ) for index, (start, end) in enumerate(((0, 5), (5, 13)))]
        expected, expected_final = torch.cat([part[0] for part in parts]), torch.cat([part[1] for part in parts])
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(final, expected_final, atol=0, rtol=0)
        output_grad, final_grad = torch.randn_like(actual), torch.randn_like(final)
        actual_grads = torch.autograd.grad((actual, final), (*tensors, initial), (output_grad, final_grad))
        expected_grads = torch.autograd.grad((expected, expected_final), (*tensors, initial), (output_grad, final_grad))
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, "token count"):
            packed_eager(lambda *args: args, (query.unsqueeze(0),), (0, 14))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_model_runtime_and_decoder_keep_invocation_metadata(self):
        """Feature: Packed invocation metadata ownership.

        Description: Exercise runtime packing and decoder forwarding for multiple calls.
        Expectation: Each call retains its own document offsets through the module-family entry.
        """

        class Mixer(nn.Module):
            """Record metadata at the decoder mixer boundary."""

            def __init__(self) -> None:
                """Build a minimal decoder fixture with stable parameter ownership."""
                super().__init__()
                self.weight = nn.Parameter(torch.ones(()))
                self.seen = None

            def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
                """Record the exact per-invocation metadata object."""
                self.seen = kwargs["packed_sequence_info"]
                return hidden_states * self.weight

        class Decoder(nn.Module):
            """Expose the decoder contract consumed by the packed adapter."""

            def __init__(self) -> None:
                """Build a minimal decoder fixture with stable parameter ownership."""
                super().__init__()
                self.layer_type = "linear_attention"
                self.linear_attn = Mixer()
                self.input_layernorm = nn.Identity()
                self.post_attention_layernorm = nn.Identity()
                self.mlp = nn.Identity()

            def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
                """Preserve the original dense path for the adapter control."""
                del kwargs
                return hidden_states

        context = RuntimeInputContext((1, 8), {"cp": 1}, {"cp": 2, "tp": 1}, {})
        batch = {"cu_seq_lens": torch.tensor([0, 11, 16])}
        runtime = Qwen35PackedRuntime()
        inputs = runtime.build_runtime_inputs(batch=batch, context=context)
        other = KimiK3PackedRuntime().build_runtime_inputs(batch=batch, context=context)
        self.assertEqual(inputs["packed_sequence_info"], other["packed_sequence_info"])
        self.assertEqual(inputs["packed_sequence_info"].local_offsets, (0, 3, 8))
        decoder = Decoder()
        parameters = {name: id(parameter) for name, parameter in decoder.named_parameters()}
        _commit_forward_rewrite(qwen3_5_linear_attention_decoder_wrapper(decoder, None, None, None, None))
        values = torch.randn(1, 8, 4, requires_grad=True)
        family = SimpleNamespace(resolved=SimpleNamespace(family=SimpleNamespace(name="gated_delta_net")))
        with patch("hyper_parallel.models.qwen3_5.adapter.linear_attention_decoder.get_family_build_info", return_value=family):
            decoder(values, **inputs).sum().backward()
        self.assertIs(decoder.linear_attn.seen, inputs["packed_sequence_info"])
        self.assertEqual(parameters, {name: id(parameter) for name, parameter in decoder.named_parameters()})
        self.assertIs(decoder(values), values)
