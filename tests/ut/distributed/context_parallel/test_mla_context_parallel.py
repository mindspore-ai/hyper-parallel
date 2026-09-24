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
"""CPU adapter contracts and real HF replacement/checkpoint coverage for MLA CP."""

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.components.checkpoint.weight_conversion import revert_weight_conversion
from hyper_parallel.components.functional.npu_fusion_attention import (
    _prepare_fusion_attention_context, resolve_packed_sequence_lengths,
)
from hyper_parallel.distributed.context_parallel.mla_context_parallel import (
    MLADimensions, MLACPPlan,
)
from hyper_parallel.models.deepseek_v3.adapter.distributed.context_parallel import mla_cp_wrapper
from tests.common.mla_cp_utils import (
    TestCPMesh, install_mla_cp, make_mla, make_deepseek_model, replace_and_load_mla,
)


class TestMLAContextParallel(unittest.TestCase):
    """Keep focused adapter contracts; distributed score math lives in the ST worker."""

    def test_packed_boundaries_and_long_causal_mask(self) -> None:
        """
        Feature: Shared packed-sequence metadata and bounded causal masks.
        Description: Resolve aliases and Tensor boundaries, reject conflicts, and prepare S4096 attention.
        Expectation: Valid document ends agree and default causal masks remain 2048 by 2048.
        """
        for carrier in ({"cu_seqlens_q": (0, 3, 12), "cu_seqlens_kv": (0, 3, 12)},
                        SimpleNamespace(cu_seqlens_q=(0, 3, 12), cu_seqlens_kv=(0, 3, 12))):
            options = {"actual_seq_len": (3, 12), "packed_seq_params": carrier}
            self.assertEqual(resolve_packed_sequence_lengths(options, 12, 12), ([3, 12], [3, 12]))
            options["actual_seq_len"] = (4, 12)
            with self.assertRaisesRegex(ValueError, "conflicting"):
                resolve_packed_sequence_lengths(options, 12, 12)
        for lengths in ((0, 12), (3.5, 12), (True, 12)):
            with self.subTest(lengths=lengths), self.assertRaises(ValueError):
                resolve_packed_sequence_lengths({"actual_seq_len": lengths}, 12, 12)
        self.assertEqual(resolve_packed_sequence_lengths({"actual_seq_len": torch.tensor([3, 12])}, 12, 12),
                         ([3, 12], [3, 12]))
        query = torch.empty(1, 2, 4096, 8)
        context = _prepare_fusion_attention_context(torch.nn.Module(), query, query, query, None, {})
        self.assertEqual(context.attention_mask.shape, (2048, 2048))
        self.assertEqual(context.sparse_mode, 3)
        self.assertIsNone(context.valid_rows)

    def test_projection_preserves_module_hooks_and_parameter_identity(self) -> None:
        """
        Feature: Head-selective KV projection with managed module hooks.
        Description: Project a contiguous head range through the original Linear and backpropagate.
        Expectation: Forward hooks run, parameter identities remain stable, and unused rows receive zero gradient.
        """
        module = make_mla()
        projection = module.kv_b_proj
        identities = [id(param) for param in projection.parameters()]
        calls = []
        pre = projection.register_forward_pre_hook(lambda _module, _args: calls.append("pre"))
        post = projection.register_forward_hook(lambda _module, _args, _out: calls.append("post"))
        install_mla_cp(module, TestCPMesh(singleton=True), "latent_kv_head", "sdpa")
        inputs = torch.randn(2, 3, 6, dtype=torch.float64)
        expected = torch.nn.functional.linear(  # pylint: disable=not-callable
            inputs, projection.weight[7:21], projection.bias[7:21],
        )
        actual = projection(inputs, mla_head_range=(1, 3))
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        self.assertEqual(calls, ["pre", "post"])
        self.assertEqual(identities, [id(param) for param in projection.parameters()])
        self.assertEqual(projection.weight.grad[:7].count_nonzero(), 0)
        pre.remove()
        post.remove()

    def test_invalid_metadata_and_options_fail_before_collectives(self) -> None:
        """
        Feature: MLA CP input validation before communication.
        Description: Supply invalid global boundaries, masks, cache options and local RoPE frequencies.
        Expectation: Each unsupported input raises a descriptive error before any collective executes.
        """
        module = make_mla()
        install_mla_cp(module, TestCPMesh(singleton=True), "latent_kv_head", "sdpa")
        inputs = torch.randn(1, 12, 12, dtype=torch.float64)
        cases = (
            ({"actual_seq_len": (3, 9)}, "query token count"),
            ({"attention_mask": torch.zeros(12, 12)}, "boolean"),
            ({"attention_mask": torch.ones(3, 12, dtype=torch.bool)}, "global Q/K"),
            ({"indices": torch.zeros(1)}, "Unsupported"),
            ({"past_key_values": object()}, "KV cache"),
            ({"actual_seq_len": (3, 12), "cu_seq_lens_q": (0, 4, 12)}, "conflicting"),
            ({"position_embeddings": (torch.ones(24, 2), torch.zeros(24, 2))}, "local tokens"),
        )
        with patch("hyper_parallel.distributed.context_parallel.mla_context_parallel.mla_all_gather") as collective:
            for kwargs, message in cases:
                with self.subTest(message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        module(inputs, **kwargs)
            collective.assert_not_called()
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            module(inputs, actual_seq_len=(0, 12))
        with self.assertRaisesRegex(ValueError, "batch_size=1"):
            module(inputs.expand(2, -1, -1), actual_seq_len=(24,))

    def test_plan_and_wrapper_fail_fast(self) -> None:
        """
        Feature: MLA CP plan and wrapper installation contracts.
        Description: Exercise invalid head division, TP layout, dropout and duplicate installation.
        Expectation: Invalid configurations fail and valid TP2 CP2 computes one head per rank.
        """
        dims = MLADimensions(4, 5, 6, 3, 2, 4)
        with self.assertRaisesRegex(ValueError, "divisible"):
            MLACPPlan(dims, 3)
        with self.assertRaisesRegex(ValueError, "divisible"):
            MLACPPlan(dims, 2, tp_degree=4)
        self.assertEqual(MLACPPlan(dims, 2, tp_degree=2).compute_heads, 1)
        module = make_mla()
        mesh = TestCPMesh(singleton=True)
        with self.assertRaisesRegex(ValueError, "TP-local"):
            mla_cp_wrapper(module, None, unittest.mock.Mock(ndim=1, size=lambda: 2), mesh, None)
        module.attention_dropout = 0.1
        with self.assertRaisesRegex(ValueError, "dropout=0"):
            mla_cp_wrapper(module, None, None, mesh, None)
        module.attention_dropout = 0.0
        install_mla_cp(module, mesh, "expanded_ulysses", "sdpa")
        with self.assertRaisesRegex(ValueError, "already configured"):
            mla_cp_wrapper(module, None, None, mesh, None)


class TestMLAModel(unittest.TestCase):
    """Exercise real decoder objects and the family-specific fused latent rules."""

    def test_replacement_decoder_and_checkpoint_roundtrip(self) -> None:
        """
        Feature: HF DeepSeek MLA replacement and checkpoint compatibility.
        Description: Replace a real decoder, restore original checkpoint keys, and compare forward and backward.
        Expectation: Checkpoint tensors are exact and logits and gradients match the HF model within tolerance.
        """
        torch.manual_seed(101)
        reference = make_deepseek_model()
        original_state = deepcopy(reference.state_dict())
        parallel, _ = replace_and_load_mla(deepcopy(reference))
        for layer in parallel.model.layers:
            install_mla_cp(layer.self_attn, None, "expanded_ulysses", "sdpa")
        restored = revert_weight_conversion(parallel, parallel.state_dict())
        self.assertEqual(set(restored), set(original_state))
        for name, value in original_state.items():
            torch.testing.assert_close(restored[name], value, atol=0, rtol=0)
        tokens = torch.randint(0, 128, (2, 16))
        allowed = torch.ones(1, 1, 16, 16, dtype=torch.bool).tril()
        expected = reference(input_ids=tokens, attention_mask=allowed, use_cache=False).logits
        actual = parallel(input_ids=tokens, attention_mask=allowed, use_cache=False).logits
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)
        target = torch.randn_like(expected)
        (expected * target).mean().backward()
        (actual * target).mean().backward()
        gradients = revert_weight_conversion(
            parallel, {name: parameter.grad for name, parameter in parallel.named_parameters()},
        )
        for name, parameter in reference.named_parameters():
            torch.testing.assert_close(gradients[name], parameter.grad, atol=2e-7, rtol=2e-5)



if __name__ == "__main__":
    unittest.main()
