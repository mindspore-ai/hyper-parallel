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
"""Training attention, padding and sequence-sharded targets."""
from __future__ import annotations

import copy
import unittest
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import yaml
from transformers import Qwen3Config, Qwen3ForCausalLM

from rl.trainer.build import _TrainerBuilder
from rl.unified_execution.training.grouped_query_attention import (
    bind_gqa_v4_backend,
    prepare_v4_batch,
    gqa_v4_forward,
)
from rl.workers.adapters.context_parallel import compute_cp_token_log_probs


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
    q_ends, k_ends = (kwargs[name].tolist() for name in ("cu_seqlens_q", "cu_seqlens_k"))
    outputs = []
    for qa, qb, ka, kb in zip(q_ends[:-1], q_ends[1:], k_ends[:-1], k_ends[1:], strict=True):
        q, k, v = [tensor.unsqueeze(0).transpose(1, 2) for tensor in (query[qa:qb], key[ka:kb], value[ka:kb])]
        # V4 aligns a shorter local Q suffix at the right edge of its document's KV prefix.
        causal = torch.ones(qb - qa, kb - ka, dtype=torch.bool).tril((kb - ka) - (qb - qa))
        output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=causal, enable_gqa=True, scale=kwargs.get("softmax_scale"))
        outputs.append(output.transpose(1, 2).squeeze(0))
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
        provider_patch = patch("rl.unified_execution.modules.flash_attention_v4._provider",
                               return_value=self.provider)
        device_patch = patch("rl.unified_execution.modules.flash_attention_v4._validate_device")
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
                with self.assertRaisesRegex(ValueError, "Cannot verify the unified execution path"):
                    bind_gqa_v4_backend(model)
                self.assertFalse(getattr(model.model, "_rl_gqa_v4_bound", False))

    def test_left_right_padding_matches_independent_requests(self) -> None:
        """Packing must preserve order, request isolation and scatter gradients."""
        mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 0]])
        metadata = prepare_v4_batch(mask, 2, 4, torch.device("cpu"))
        self.assertEqual(metadata((), ()).query_rows.tolist(), [2, 3, 4, 5, 6])
        self.assertEqual(metadata((), ()).cu_q.tolist(), [0, 2, 5])
        states = [torch.randn(2, heads, 4, 8, dtype=torch.bfloat16, requires_grad=True) for heads in (4, 2, 2)]
        output, _ = gqa_v4_forward(
            SimpleNamespace(is_causal=True), *states, None, v4_batch=metadata,
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
        metadata = prepare_v4_batch(torch.zeros(2, 4), 2, 4, torch.device("cpu"))
        states = [torch.randn(2, heads, 4, 8, dtype=torch.bfloat16, requires_grad=True) for heads in (4, 2, 2)]
        output, _ = gqa_v4_forward(
            SimpleNamespace(is_causal=True), *states, None, v4_batch=metadata,
        )
        output.sum().backward()
        self.assertTrue(all(tensor.grad is not None and tensor.grad.count_nonzero() == 0 for tensor in states))
        self.provider.flash_attn_varlen_func.assert_not_called()

    def test_native_cp_operands_keep_padded_documents_isolated(self):
        """Use real HP document slicing/restoration; substitute CPU math only for the V4 kernel."""
        from hyper_parallel.components.modules.attention.grouped_query_attention.cp_grouped_query_attention import (
            CPGroupedQueryAttentionModule,
        )

        config = Qwen3Config(hidden_size=32, num_attention_heads=4, num_key_value_heads=2, head_dim=8)
        attention = CPGroupedQueryAttentionModule(config=config, ulysses_degree=1)
        attention.query_context = SimpleNamespace(rank=1)
        attention.all_gather_degree = 2
        attention.attention_interface = gqa_v4_forward
        mask = torch.tensor([[True, True, True, False, True, True, True, False, True, False, True, True]])
        states = [torch.randn(1, 12, heads, 8, dtype=torch.bfloat16) for heads in (4, 2, 2)]
        options = {"v4_batch": prepare_v4_batch(mask, 1, 6, torch.device("cpu"), mesh(1))}
        # Rank 1 owns queries 6..11. Supply the K/V operands after a transport-only gather.
        operands = attention._attention_operands(states[0][:, 6:], *states[1:], None, (3, 8, 12), options, None)
        output, _ = attention.run_attention(operands)
        output, _ = attention.restore_attention(output, operands.output_state)
        expected = torch.zeros_like(states[0])
        for begin, end in ((0, 3), (3, 8), (8, 12)):
            valid = mask[0, begin:end].nonzero().flatten() + begin
            expected[:, valid] = _reference_dense(*(state[:, valid] for state in states))
        torch.testing.assert_close(output, expected[:, 6:], atol=0.005, rtol=0.01)

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
        bind_gqa_v4_backend(model)
        bind_gqa_v4_backend(model)
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


def mesh(rank=0, size=2):
    """Supply topology without starting a distributed process group."""
    return SimpleNamespace(size=lambda: size, get_local_rank=lambda: rank, get_group=object)


class TestContextParallel(unittest.TestCase):
    """Test geometry independently of the native attention implementation."""

    def test_document_prefix_uses_global_offset(self):
        """A rank crossing packed boundaries gets a distinct KV prefix per document."""
        batch_layout = prepare_v4_batch(torch.ones(1, 12), 1, 6, torch.device("cpu"), mesh(1))
        layout = batch_layout((2, 6), (8, 12))
        self.assertIsNone(layout.query_rows)
        self.assertIsNone(layout.key_rows)
        self.assertEqual(layout.cu_q.tolist(), [0, 2, 6])
        self.assertEqual(layout.cu_k.tolist(), [0, 8, 12])
        self.assertIs(batch_layout((2, 6), (8, 12)), layout)

    def test_left_right_padding_and_empty_rank(self):
        """Padding rows index the actual operands returned by HP, including multiple documents."""
        mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 0], [1, 1, 0, 0, 0, 0, 0, 0]])
        layout = prepare_v4_batch(mask, 2, 4, torch.device("cpu"), mesh(1))((4, 8), (8, 16))
        self.assertEqual(layout.query_rows.tolist(), [0, 1, 2])
        self.assertEqual(layout.key_rows.tolist(), [2, 3, 4, 5, 6])
        empty = prepare_v4_batch(mask[:1] * 0, 1, 4, torch.device("cpu"), mesh(1))((4,), (8,))
        self.assertEqual(empty.query_rows.numel(), 0)
        self.assertEqual(empty.cu_q.tolist(), [0])

        # Rank 1 sees Q at 6..11 split across documents [0,3), [3,8), [8,12).
        # HP has already removed the first document from the physical KV operand.
        mask = torch.tensor([[1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1]])
        layout = prepare_v4_batch(mask, 1, 6, torch.device("cpu"), mesh(1))((2, 6), (5, 9))
        self.assertEqual(layout.query_rows.tolist(), [0, 2, 4, 5])
        self.assertEqual(layout.key_rows.tolist(), [1, 2, 3, 5, 7, 8])
        self.assertEqual(layout.cu_q.tolist(), [0, 1, 4])
        self.assertEqual(layout.cu_k.tolist(), [0, 3, 6])

    def test_next_token_target_crosses_cp_boundary(self):
        """The final local logit predicts the first token on the next CP rank."""
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        seen = []

        def forward(**kwargs):
            seen.append(kwargs)
            logits = torch.zeros(1, 4, 16)
            logits.scatter_(-1, (kwargs['input_ids'] + 1).unsqueeze(-1), 10.)
            return SimpleNamespace(logits=logits)

        output = compute_cp_token_log_probs(forward, ids, torch.ones_like(ids), mesh())
        self.assertTrue(bool((output > -.01).all()))
        self.assertEqual(seen[0]['position_ids'].tolist(), [[0, 1, 2, 3]])
        self.assertEqual(seen[0]['attention_mask'].shape, ids.shape)
        compute_cp_token_log_probs(forward, ids, torch.ones_like(ids), mesh(1))
        self.assertEqual(seen[1]['position_ids'].tolist(), [[4, 5, 6, 7]])

    @staticmethod
    def config():
        root = Path(__file__).resolve().parents[4]
        path = root / 'hyper_parallel/rl/examples/consistency/configs/qwen3_moe_4l_cp_rl.yaml'
        return yaml.safe_load(path.read_text(encoding='utf-8'))


class TestCPRequestOwnership(unittest.TestCase):
    """Logical DP remains the sampling domain when requests are shared across TP and CP."""

    def test_all_tp_cp_siblings_share_one_request_rank(self):
        """All TP/CP siblings share their logical DP request owner."""
        for cp_rank in range(2):
            for tp_rank in range(2):
                with self.subTest(cp=cp_rank, tp=tp_rank):
                    request_mesh = SimpleNamespace(get_group=lambda: 'request-group',
                                                   get_local_rank=lambda cp=cp_rank, tp=tp_rank: cp * 2 + tp,
                                                   size=lambda: 4)
                    device_mesh = {'tp': object(), ('cp', 'tp'): SimpleNamespace(
                        flatten=lambda name, mesh=request_mesh: mesh)}
                    builder = object.__new__(_TrainerBuilder)
                    builder.parallel_dims = SimpleNamespace(tp_size=2, cp_size=2, tp_rank=tp_rank, cp_rank=cp_rank,
                                                             dp_rank=1, dp_size=2, device_mesh=device_mesh)
                    configure = Mock()
                    builder.rollout_engine = SimpleNamespace(configure_trainer_tensor_parallel=configure)
                    builder._configure_rollout_tensor_parallel()
                    configure.assert_called_once_with(group='request-group', tp_rank=cp_rank * 2 + tp_rank,
                                                       tp_size=4, request_rank=1, request_size=2)
