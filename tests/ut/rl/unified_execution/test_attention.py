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
"""CPU contract checks for V4 dispatch, mask coordinates and training gradients."""

import copy
from functools import wraps
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch.utils.checkpoint import DefaultDeviceType
from transformers import Qwen3Config, Qwen3ForCausalLM

from rl.unified_execution.ops.attention.flash_attention_npu_v4 import (
    dense_attention, varlen_attention,
)
from rl.unified_execution.runtimes.hyper_parallel.training import (
    install_training_attention,
    prepare_training_attention_metadata,
    run_training_attention,
)

from hyper_parallel.core.activation_memory.wrapper import CheckpointWrapper
from hyper_parallel.distributed._builder import forward_rewriter


def _reference_dense(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs: object) -> torch.Tensor:
    """Use CPU SDPA solely as an independent mathematical operator stand-in."""
    # Torch exposes SDPA as a C-extension callable that pylint cannot infer.
    result = torch.nn.functional.scaled_dot_product_attention(  # pylint: disable=not-callable
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
        is_causal=True, enable_gqa=True, scale=kwargs.get("softmax_scale"),
    )
    return result.transpose(1, 2)


def _reference_varlen(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs: object) -> torch.Tensor:
    """Separate packed requests before computing the independent CPU reference."""
    boundaries = kwargs["cu_seqlens_q"].tolist()
    outputs = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        states = [tensor[start:end].unsqueeze(0) for tensor in (query, key, value)]
        outputs.append(_reference_dense(*states, **kwargs).squeeze(0))
    return torch.cat(outputs)


class TestUnifiedAttention(unittest.TestCase):
    """Verify adapter contracts without treating mocks as native kernel validation."""

    def setUp(self) -> None:
        """Use real CPU tensors and replace only the device-specific operator."""
        torch.manual_seed(17)
        self.provider = SimpleNamespace(
            flash_attn_func=Mock(side_effect=_reference_dense),
            flash_attn_varlen_func=Mock(side_effect=_reference_varlen),
            get_scheduler_metadata=Mock(return_value=torch.empty(16, dtype=torch.int32)),
        )
        provider_patch = patch("rl.unified_execution.ops.attention.flash_attention_npu_v4._provider",
                               return_value=self.provider)
        device_patch = patch("rl.unified_execution.ops.attention.flash_attention_npu_v4._validate_device")
        provider_patch.start()
        device_patch.start()
        self.addCleanup(provider_patch.stop)
        self.addCleanup(device_patch.stop)

    @staticmethod
    def _tiny_model() -> Qwen3ForCausalLM:
        config = Qwen3Config.from_dict({
            "vocab_size": 32, "hidden_size": 32, "intermediate_size": 48, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8, "attention_dropout": 0.0,
        })
        config._attn_implementation = "eager"  # pylint: disable=protected-access
        return Qwen3ForCausalLM(config).to(torch.bfloat16)

    def test_custom_forwards_cannot_claim_a_supported_wrapped_function(self) -> None:
        """A custom @wraps forwarding claim is not evidence that attention executes."""
        for target_name in ("root", "decoder", "layer", "attention"):
            with self.subTest(target=target_name):
                model = self._tiny_model()
                targets = {"root": model, "decoder": model.model, "layer": model.model.layers[0],
                           "attention": model.model.layers[0].self_attn}
                target = targets[target_name]

                def bypass(*args: object, **kwargs: object) -> None:
                    """Imitate a decorated function while intentionally skipping computation."""
                    del args, kwargs

                target.forward = wraps(target.forward)(bypass)
                with self.assertRaisesRegex(ValueError, "cannot verify the attention execution path"):
                    install_training_attention(model)
                self.assertFalse(getattr(model.model, "unified_attention_installed", False))

    def test_hp_boundary_and_checkpoint_wrappers_are_retained(self) -> None:
        """The audit must keep real HP boundary/AC wrappers and their execution order."""
        self.addCleanup(DefaultDeviceType.set_device_type, DefaultDeviceType.get_device_type())
        DefaultDeviceType.set_device_type("cpu")
        model = self._tiny_model()
        boundary = SimpleNamespace(
            redistribute_inputs=lambda args, kwargs, **unused: (args, kwargs),
            redistribute_outputs=lambda result, **unused: result,
        )
        attention = model.model.layers[0].self_attn
        forward_rewriter._wrap_production_forward(attention, boundary)  # pylint: disable=protected-access
        wrapped_attention = attention.forward
        model.model.layers[0] = CheckpointWrapper(model.model.layers[0], use_reentrant=False)
        install_training_attention(model)
        self.assertIs(attention.forward, wrapped_attention)
        output = model(torch.tensor([[1, 2, 3, 4]]), attention_mask=torch.tensor([[0, 1, 1, 1]]),
                       use_cache=False).logits
        output[:, 1:].float().square().sum().backward()
        self.assertGreaterEqual(self.provider.flash_attn_varlen_func.call_count, 2)

    def test_dense_preserves_backward_and_fixed_options(self) -> None:
        """Layout conversion must preserve Q/K/V gradients and numerical options."""
        states = [torch.randn(2, heads, 4, 8, dtype=torch.bfloat16, requires_grad=True) for heads in (4, 2, 2)]
        output, weights = run_training_attention(SimpleNamespace(is_causal=True), *states, None, scaling=0.25)
        output.float().square().sum().backward()
        self.assertEqual(output.shape, (2, 4, 4, 8))
        self.assertIsNone(weights)
        self.assertTrue(all(tensor.grad is not None and tensor.grad.abs().sum() > 0 for tensor in states))
        self.assertEqual(self.provider.flash_attn_func.call_args.kwargs,
                         {"softmax_scale": 0.25, "causal": True, "deterministic": True, "num_splits": 1})

    def test_left_right_padding_matches_independent_requests(self) -> None:
        """Packing must preserve order, request isolation and scatter gradients."""
        mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 0]])
        metadata = prepare_training_attention_metadata(mask, 2, 4, torch.device("cpu"))
        self.assertEqual(metadata.indices.tolist(), [2, 3, 4, 5, 6])
        self.assertEqual(metadata.cu_seqlens.tolist(), [0, 2, 5])
        states = [torch.randn(2, heads, 4, 8, dtype=torch.bfloat16, requires_grad=True) for heads in (4, 2, 2)]
        output, _ = run_training_attention(
            SimpleNamespace(is_causal=True), *states, None, unified_attention_metadata=metadata,
        )
        for row in range(2):
            indices = mask[row].nonzero().flatten()
            inputs = [tensor[row:row + 1].transpose(1, 2).index_select(1, indices) for tensor in states]
            expected = _reference_dense(*inputs)
            torch.testing.assert_close(output[row:row + 1].index_select(1, indices), expected)
        self.assertEqual(float(output[~mask.bool()].abs().sum()), 0)
        output.float().square().sum().backward()
        for tensor in states:
            self.assertEqual(float(tensor.grad.transpose(1, 2)[~mask.bool()].abs().sum()), 0)
        self.assertIsNone(self.provider.flash_attn_varlen_func.call_args.kwargs["seqused_k"])

    def test_all_padding_keeps_zero_gradients(self) -> None:
        """Entirely masked DP rows must remain differentiable without an empty kernel launch."""
        metadata = prepare_training_attention_metadata(torch.zeros(2, 4), 2, 4, torch.device("cpu"))
        states = [torch.randn(2, heads, 4, 8, dtype=torch.bfloat16, requires_grad=True) for heads in (4, 2, 2)]
        output, _ = run_training_attention(
            SimpleNamespace(is_causal=True), *states, None, unified_attention_metadata=metadata,
        )
        output.sum().backward()
        self.assertTrue(all(tensor.grad is not None and tensor.grad.count_nonzero() == 0 for tensor in states))
        self.provider.flash_attn_varlen_func.assert_not_called()

    def test_paged_metadata_preserved_and_backward_rejected(self) -> None:
        """Paged graph metadata must pass through without a host copy or inferred bounds."""
        query = torch.randn(3, 4, 8, dtype=torch.bfloat16)
        key = torch.randn(2, 64, 2, 8, dtype=torch.bfloat16)
        metadata = {
            "cu_seqlens_q": torch.tensor([0, 1, 3], dtype=torch.int32),
            "seqused_k": torch.tensor([13, 27], dtype=torch.int32),
            "page_table": torch.tensor([[1], [0]], dtype=torch.int32),
            "scheduler_metadata": torch.empty(16, dtype=torch.int32),
            "max_seqlen_q": 3, "max_seqlen_k": 128,
        }
        self.provider.flash_attn_varlen_func.side_effect = None
        self.provider.flash_attn_varlen_func.return_value = query
        with patch.object(torch.Tensor, "item", side_effect=AssertionError("host length query")), \
                patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host transfer")):
            self.assertIs(varlen_attention(query, key, key, **metadata), query)
        passed = self.provider.flash_attn_varlen_func.call_args.kwargs
        self.assertIs(passed["page_table"], metadata["page_table"])
        self.assertIs(passed["scheduler_metadata"], metadata["scheduler_metadata"])
        self.assertEqual(passed["num_splits"], 1)
        with self.assertRaisesRegex(ValueError, "Deterministic V4 backward"):
            varlen_attention(query.requires_grad_(True), key, key, **metadata)

    def test_real_qwen3_preserves_parameters_and_valid_token_gradients(self) -> None:
        """The installed model retains its parameters and matches independent HF eager attention."""
        config = Qwen3Config.from_dict({
            "vocab_size": 32, "hidden_size": 32, "intermediate_size": 48, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8, "attention_dropout": 0.0,
        })
        config._attn_implementation = "eager"  # pylint: disable=protected-access
        model = Qwen3ForCausalLM(config).to(torch.bfloat16)
        reference = copy.deepcopy(model)
        model.model.layers[0].self_attn.config = copy.deepcopy(model.config)
        identities = [id(parameter) for parameter in model.parameters()]
        install_training_attention(model)
        install_training_attention(model)
        self.assertEqual(identities, [id(parameter) for parameter in model.parameters()])
        mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 0]])
        ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        actual = model(ids, attention_mask=mask, use_cache=False).logits
        expected = reference(ids, attention_mask=mask, use_cache=False).logits
        valid = mask.bool()
        torch.testing.assert_close(actual[valid], expected[valid], atol=0.002, rtol=0.02)
        actual[valid].float().square().sum().backward()
        expected[valid].float().square().sum().backward()
        for parameter, reference_parameter in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter.grad, reference_parameter.grad, atol=0.02, rtol=0.05)
        self.assertEqual(self.provider.flash_attn_varlen_func.call_count, 2)

    def test_unsupported_inputs_rejected(self) -> None:
        """Unsupported dtype, dropout, cache shape and window must fail explicitly."""
        query = torch.randn(1, 2, 4, 8, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "BF16"):
            dense_attention(query.float(), query.float(), query.float())
        for options in ({"dropout": 0.1}, {"sliding_window": 4}, {"actual_seq_len": [4]}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                run_training_attention(SimpleNamespace(is_causal=True), query, query, query, None, **options)


if __name__ == "__main__":
    unittest.main()
