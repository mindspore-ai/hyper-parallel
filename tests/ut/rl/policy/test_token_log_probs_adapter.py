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
"""Actual hybrid HF probabilities and gradients without vocabulary-sized retained outputs."""

from contextlib import ExitStack
from copy import deepcopy
from functools import partial
import inspect
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pytest
import torch
from torch.utils.checkpoint import DefaultDeviceType

from rl.roles.policy.token_log_probs_adapter import bind_token_log_probs

pytest.importorskip("transformers", minversion="5.5.4")

# Optional HF accelerator discovery must not initialize NPU in these actual CPU contracts.
with patch("transformers.utils.is_torch_npu_available", return_value=False), patch(
    "transformers.utils.import_utils.is_torch_npu_available", return_value=False
):
    from transformers import Qwen3ForCausalLM, Qwen3Config, Qwen3_5ForCausalLM, Qwen3_5TextConfig


def _model() -> Qwen3_5ForCausalLM:
    """Build a real two-layer text decoder with GDN and full attention."""
    config = Qwen3_5TextConfig.from_dict({
        "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128,
        "linear_num_key_heads": 2, "linear_num_value_heads": 6,
        "layer_types": ["linear_attention", "full_attention"], "pad_token_id": 0,
        "max_position_embeddings": 128, "use_cache": False,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0},
    })
    config._attn_implementation = "eager"  # pylint: disable=protected-access
    return Qwen3_5ForCausalLM(config).to(device="cpu", dtype=torch.float32)


def _dense(model: torch.nn.Module, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Use untouched HF forward as the full-vocabulary probability oracle."""
    logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, :-1]
    return logits.float().log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1)


class TestSelectedLogProbs(unittest.TestCase):
    """Verify the vector chain rule, real decoder recomputation and normal HF compatibility."""

    @staticmethod
    def make_model() -> torch.nn.Module:
        """Build the original hybrid text family for the shared acceptance suite."""
        return _model()

    def setUp(self) -> None:
        """Use deterministic CPU tensors and restore thread/RNG/checkpoint defaults afterwards."""
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, previous_threads)
        previous_device = DefaultDeviceType.get_device_type()
        DefaultDeviceType.set_device_type("cpu")
        self.addCleanup(DefaultDeviceType.set_device_type, previous_device)
        contexts = ExitStack()
        contexts.enter_context(torch.random.fork_rng(devices=[]))
        self.addCleanup(contexts.close)
        torch.manual_seed(7)
        self.ids = torch.tensor([[1, 4, 7, 3, 9, 5, 11, 6, 2], [2, 3, 8, 12, 5, 6, 4, 0, 0]])
        self.mask = self.ids.ne(0)

    def test_full_vector_gradients_match_dense_with_and_without_decoder_checkpointing(self) -> None:
        """Compare every real parameter under signed token gradients and a nonlinear clipped objective."""
        template = self.make_model()
        for full_ac in (False, True):
            for nonlinear in (False, True):
                with self.subTest(full_ac=full_ac, nonlinear=nonlinear):
                    dense, selected = deepcopy(template).train(), deepcopy(template).train()
                    if full_ac:
                        for model in (dense, selected):
                            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                    bind_token_log_probs(selected, chunk_size=3)
                    decoder_calls = [0, 0]

                    def count(module: torch.nn.Module, inputs: tuple, index: int, counts: list[int]) -> None:
                        """Count real decoder calls without changing their computation."""
                        del module, inputs
                        counts[index] += 1

                    for index, layer in enumerate(selected.model.layers):
                        layer.register_forward_pre_hook(partial(count, index=index, counts=decoder_calls))
                    expected = _dense(dense, self.ids, self.mask)
                    actual = selected(input_ids=self.ids, attention_mask=self.mask, use_cache=False,
                                      return_token_log_probs=True)
                    self.assertEqual(actual.dtype, torch.float32)
                    self.assertEqual(tuple(actual.shape), (2, 8))
                    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
                    weights = torch.linspace(-1.1, 1.3, expected.numel()).reshape_as(expected)
                    weights.view(-1)[::3] = 0
                    if nonlinear:
                        old = expected.detach() + torch.linspace(-0.04, 0.04, expected.numel()).reshape_as(expected)

                        def objective(values: torch.Tensor, old: torch.Tensor = old,
                                      weights: torch.Tensor = weights) -> torch.Tensor:
                            """Exercise signed clipped importance ratios on the full vector."""
                            ratio = (values - old).exp()
                            return -torch.minimum(ratio * weights, ratio.clamp(0.98, 1.02) * weights).sum()

                        expected_grads = torch.autograd.grad(objective(expected), tuple(dense.parameters()))
                        actual_grads = torch.autograd.grad(objective(actual), tuple(selected.parameters()))
                    else:
                        expected_grads = torch.autograd.grad(expected, tuple(dense.parameters()), grad_outputs=weights)
                        actual_grads = torch.autograd.grad(actual, tuple(selected.parameters()), grad_outputs=weights)
                    for (name, _), left, right in zip(dense.named_parameters(), expected_grads, actual_grads):
                        self.assertTrue(torch.isfinite(right).all(), msg=f"Nonfinite gradient for parameter={name}")
                        torch.testing.assert_close(right, left, rtol=2e-5, atol=2e-6, msg=name)
                    self.assertEqual(decoder_calls, [2, 2] if full_ac else [1, 1])

    def test_normal_forward_and_parameter_identity_are_unchanged(self) -> None:
        """Normal HF output stays exact and rebinding does not replace the model or its parameters."""
        model = self.make_model().eval()
        parameters = {name: id(value) for name, value in model.named_parameters()}
        modules = {name: id(value) for name, value in model.named_modules()}
        with torch.no_grad():
            expected = model(input_ids=self.ids, attention_mask=self.mask, use_cache=False).logits
            bind_token_log_probs(model, chunk_size=3)
            forward = model.forward
            bind_token_log_probs(model, chunk_size=3)
            self.assertIs(model.forward, forward)
            actual = model(input_ids=self.ids, attention_mask=self.mask, use_cache=False).logits
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            chosen = model(input_ids=self.ids, attention_mask=self.mask, use_cache=False, return_token_log_probs=True)
            torch.testing.assert_close(chosen, _dense(model, self.ids, self.mask), rtol=2e-5, atol=2e-6)
        self.assertEqual(parameters, {name: id(value) for name, value in model.named_parameters()})
        self.assertEqual(modules, {name: id(value) for name, value in model.named_modules()})
        project = model.forward
        self.assertFalse(any(torch.is_tensor(value) for value in inspect.getclosurevars(project).nonlocals.values()))

    def test_action_projection_preserves_all_gradients_including_zero_action_padding(self) -> None:
        """Masked nonlinear losses match dense HF, including every prompt-context gradient."""
        template = self.make_model()
        for full_ac in (False, True):
            for empty in (False, True):
                with self.subTest(full_ac=full_ac, empty=empty):
                    dense, selected = deepcopy(template).train(), deepcopy(template).train()
                    if full_ac:
                        for model in (dense, selected):
                            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                    bind_token_log_probs(selected, chunk_size=3)
                    action_mask = torch.zeros(2, 8, dtype=torch.bool)
                    if not empty:
                        action_mask[0, 4:] = True
                        action_mask[1, 3:6] = True
                    head_rows = []
                    selected.lm_head.register_forward_pre_hook(
                        lambda module, inputs, rows=head_rows: rows.append(inputs[0].shape[0]))
                    expected = _dense(dense, self.ids, self.mask)
                    actual = selected(input_ids=self.ids, attention_mask=self.mask, return_token_log_probs=True,
                                      token_log_probs_mask=action_mask)
                    torch.testing.assert_close(actual, expected.masked_fill(~action_mask, 0), rtol=2e-5, atol=2e-6)
                    self.assertEqual(sum(head_rows), max(1, int(action_mask.sum())))
                    weights = torch.linspace(-1.1, 1.3, expected.numel()).reshape_as(expected)
                    old = expected.detach() + torch.linspace(-0.04, 0.04, expected.numel()).reshape_as(expected)

                    def objective(values: torch.Tensor, old: torch.Tensor = old,
                                  weights: torch.Tensor = weights,
                                  action_mask: torch.Tensor = action_mask) -> torch.Tensor:
                        """Mask before the nonlinear objective, as policy loss does for unused positions."""
                        ratio = (values - old).masked_fill(~action_mask, 0).exp()
                        return -(torch.minimum(ratio * weights, ratio.clamp(0.98, 1.02) * weights) * action_mask).sum()

                    expected_grads = torch.autograd.grad(objective(expected), tuple(dense.parameters()))
                    actual_grads = torch.autograd.grad(objective(actual), tuple(selected.parameters()))
                    for (name, _), left, right in zip(dense.named_parameters(), expected_grads, actual_grads):
                        torch.testing.assert_close(right, left, rtol=2e-5, atol=2e-6, msg=name)

    def test_head_checkpoint_removes_saved_vocabulary_matrices(self) -> None:
        """Contrast real plain chunks with checkpointed chunks and count actual head recomputation."""
        model = self.make_model().train()

        def record(tensor: torch.Tensor) -> torch.Tensor:
            """Keep metadata only while autograd owns the saved tensor."""
            if tensor.ndim == 2 and 0 < tensor.shape[0] <= 3 and tensor.shape[1] == 64:
                retained.append(tuple(tensor.shape))
            return tensor

        retained = []
        with torch.autograd.graph.saved_tensors_hooks(record, lambda tensor: tensor):
            outputs = model.model(input_ids=self.ids, attention_mask=self.mask, use_cache=False)
            hidden = outputs.last_hidden_state[:, :-1]
            flat, targets = hidden.reshape(-1, 32), self.ids[:, 1:].reshape(-1)
            plain = torch.cat([model.lm_head(flat[start:start + 3]).float().log_softmax(-1).gather(
                -1, targets[start:start + 3, None]).squeeze(-1) for start in range(0, targets.numel(), 3)])
        self.assertTrue(retained)
        plain.square().sum().backward()
        model.zero_grad(set_to_none=True)
        bind_token_log_probs(model, chunk_size=3)
        retained = []
        head_calls = []

        def head_call(module: torch.nn.Module, inputs: tuple) -> None:
            """Observe the current head parameter identity and total token rows."""
            head_calls.append((inputs[0].shape[0], id(module.weight)))

        model.lm_head.register_forward_pre_hook(head_call)
        with torch.autograd.graph.saved_tensors_hooks(record, lambda tensor: tensor):
            chosen = model(input_ids=self.ids, attention_mask=self.mask, use_cache=False, return_token_log_probs=True)
        self.assertFalse(retained)
        chosen.square().sum().backward()
        self.assertEqual(len(head_calls), 12)
        self.assertTrue(all(rows <= 3 and weight == id(model.lm_head.weight) for rows, weight in head_calls))
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    def test_rejects_cache_partial_positions_and_unsafe_head_boundaries(self) -> None:
        """Fail before selected forward bypasses the complete-sequence or root-collective contract."""
        model = self.make_model()
        for bound in (0, 513, True):
            with self.subTest(chunk_size=bound), self.assertRaisesRegex(ValueError, "chunk_size"):
                bind_token_log_probs(model, chunk_size=bound)
        with patch("rl.roles.policy.token_log_probs_adapter.get_hsdp_state", return_value=object()):
            with self.assertRaisesRegex(ValueError, "independently sharded lm_head"):
                bind_token_log_probs(model)
        model.hsdp_scheduler = SimpleNamespace(reshard_after_forward=True)
        with patch("rl.roles.policy.token_log_probs_adapter.get_hsdp_state",
                   side_effect=lambda module: object() if module is model else None):
            if model.config.model_type == "qwen3_5_text":
                with self.assertRaisesRegex(ValueError, "root reshard_after_forward"):
                    bind_token_log_probs(model)
            else:
                bind_token_log_probs(model)
        bind_token_log_probs(model)
        for overrides in ({"use_cache": True}, {"past_key_values": object()}, {"logits_to_keep": 1}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                model(input_ids=self.ids, attention_mask=self.mask, return_token_log_probs=True, **overrides)
        for invalid in (torch.ones(2, 9, dtype=torch.bool), torch.ones(2, 8), [True]):
            with self.subTest(mask=type(invalid)), self.assertRaisesRegex(ValueError, "token_log_probs_mask"):
                model(input_ids=self.ids, return_token_log_probs=True, token_log_probs_mask=invalid)
        with self.assertRaisesRegex(ValueError, "requires return_token_log_probs"):
            model(input_ids=self.ids, token_log_probs_mask=torch.ones(2, 8, dtype=torch.bool))


class TestQwen3TokenLogProbs(TestSelectedLogProbs):
    """Run the same value, clipped-objective, gradient and identity checks on tied Qwen3."""

    @staticmethod
    def make_model() -> torch.nn.Module:
        """Use a real Qwen3 decoder with tied embedding and head parameters."""
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                             num_attention_heads=2, num_key_value_heads=1, head_dim=16, pad_token_id=0,
                             max_position_embeddings=128, use_cache=False, tie_word_embeddings=True)
        config._attn_implementation = "eager"  # pylint: disable=protected-access
        return Qwen3ForCausalLM(config).to(device="cpu", dtype=torch.float32)
