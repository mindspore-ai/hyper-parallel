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
"""CPU contracts for router replay gradients, checkpoint lifetime and Actor isolation."""
import copy
from dataclasses import replace
from functools import partial
from types import SimpleNamespace
import unittest

import torch
from torch.utils.checkpoint import DefaultDeviceType, checkpoint
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeTopKRouter

from rl.algorithm import build_algorithm
from rl.dataset.batch_builder import ExperiencePreparer, _append_padding_rows
from rl.dataset.contracts import ExperienceBatch, RouterTrace, Trajectory, Turn
from rl.roles.policy.actor import Actor
from rl.roles.policy.router_adapter import install_router_replay


class TinyMoe(torch.nn.Module):
    """Two MoE layers with token outputs sufficient to exercise Actor execution."""

    supports_token_log_probs = True
    supports_token_log_probs_mask = True

    def __init__(self, *, norm: bool = True, checkpointed: bool = False) -> None:
        """Construct deterministic native gates and independent expert parameters."""
        super().__init__()
        self.config = SimpleNamespace(
            model_type="qwen3_moe", num_hidden_layers=2, num_experts=3, hidden_size=2,
            num_experts_per_tok=2, mlp_only_layers=[], decoder_sparse_step=1, norm_topk_prob=norm,
        )
        self.layers = torch.nn.ModuleList()
        for _ in range(2):
            layer = torch.nn.Module()
            layer.mlp = torch.nn.Module()
            layer.mlp.gate = Qwen3MoeTopKRouter(self.config)
            with torch.no_grad():
                layer.mlp.gate.weight.copy_(torch.tensor([[0.8, 0.2], [0.1, 0.9], [-0.2, -0.7]]))
            layer.mlp.experts = torch.nn.Parameter(torch.tensor([[0.7, -0.1], [-0.4, 0.9], [0.5, 0.6]]))
            self.layers.append(layer)
        self.checkpointed = checkpointed
        self.fail = False
        self.input_hidden = None

    @staticmethod
    def _layer(layer: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor:
        _, scores, indices = layer.mlp.gate(hidden)
        return hidden + (layer.mlp.experts[indices] * scores[..., None]).sum(1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Return differentiable token-aligned probabilities, including checkpoint recomputations."""
        del attention_mask, kwargs
        hidden = torch.stack((input_ids.float() / 10 + 0.2, input_ids.float() / 7 + 0.1), -1)
        hidden = hidden.flatten(0, 1).requires_grad_()
        self.input_hidden = hidden
        for layer in self.layers:
            if self.checkpointed and self.training:
                hidden = checkpoint(self._layer, layer, hidden, use_reentrant=True)
            else:
                hidden = self._layer(layer, hidden)
        if self.fail:
            raise RuntimeError("test forward failure")
        return hidden.sum(-1).reshape(input_ids.shape)[:, :-1].float()


def _trace(length: int = 4, offset: int = 0) -> RouterTrace:
    return RouterTrace(
        expert_ids=(torch.tensor([[[2, 0], [1, 2]]], dtype=torch.int16)
                    + torch.arange(length, dtype=torch.int16)[:, None, None] + offset) % 3,
        valid_mask=torch.tensor([True] * (length - 1) + [False]),
        layer_ids=(0, 1),
    )


def _experience() -> ExperienceBatch:
    sequences = torch.tensor([[1, 2, 1, 4], [5, 1, 1, 0]])
    return ExperienceBatch(
        trajectories=(), sequences=sequences,
        attention_mask=torch.tensor([[True] * 4, [True, True, True, False]]),
        action_mask=torch.tensor([[False, False, True, True], [False, True, True, False]]),
        rewards=torch.ones(2), old_log_probs=torch.zeros(2, 3), responses=("a", "b"),
        generation_seconds=0.0, advantages=torch.ones(2, 3), reference_log_probs=torch.zeros(2, 3),
        router_traces=(_trace(), _trace(3, offset=1)),
    )


def _manual_forward(model: TinyMoe, experience: ExperienceBatch) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent selected-expert sum using per-expert exponentials instead of adapter gather."""
    tokens = experience.sequences
    hidden = torch.stack((tokens.float() / 10 + 0.2, tokens.float() / 7 + 0.1), -1)
    hidden = hidden.flatten(0, 1).requires_grad_()
    input_hidden = hidden
    for layer_id, layer in enumerate(model.layers):
        logits = hidden @ layer.mlp.gate.weight.T
        rows = []
        for token_index, token_logits in enumerate(logits):
            row, position = divmod(token_index, tokens.shape[1])
            trace = experience.router_traces[row]
            if position < trace.valid_mask.numel() and trace.valid_mask[position]:
                indices = trace.expert_ids[position, layer_id].long()
            else:
                indices = token_logits.topk(model.config.num_experts_per_tok).indices
            probabilities = token_logits.exp() / token_logits.exp().sum()
            selected = torch.stack([probabilities[index] for index in indices])
            if model.config.norm_topk_prob:
                selected = selected / selected.sum()
            rows.append(sum(selected[slot] * layer.mlp.experts[index] for slot, index in enumerate(indices)))
        hidden = hidden + torch.stack(rows)
    return hidden.sum(-1).reshape(tokens.shape)[:, :-1], input_hidden


class TestRouterReplay(unittest.TestCase):
    """Validate discrete replay independently from the current native top-k choices."""

    def setUp(self) -> None:
        """Use CPU checkpoint autocast even when an installed NPU backend registers itself."""
        self._checkpoint_device = DefaultDeviceType.get_device_type()
        DefaultDeviceType.set_device_type("cpu")

    def tearDown(self) -> None:
        """Restore checkpoint defaults for unrelated tests."""
        DefaultDeviceType.set_device_type(self._checkpoint_device)

    def test_selected_expert_values_and_gradients_match_manual_baseline(self) -> None:
        """Replayed choices use current weights and match independent input/router/expert gradients."""
        for norm in (True, False):
            with self.subTest(norm_topk_prob=norm):
                model = TinyMoe(norm=norm)
                baseline = copy.deepcopy(model)
                controller = install_router_replay(model)
                experience = _experience()
                targets = controller.prepare(
                    experience.router_traces, experience.attention_mask, experience.action_mask,
                )
                with controller.use(targets):
                    actual = model(experience.sequences, experience.attention_mask)
                    actual.square().sum().backward()
                expected, expected_input = _manual_forward(baseline, experience)
                expected.square().sum().backward()
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(model.input_hidden.grad, expected_input.grad)
                for actual_parameter, expected_parameter in zip(model.parameters(), baseline.parameters()):
                    torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad)
                self.assertGreater(float(model.layers[0].mlp.gate.weight.grad.abs().sum()), 0)
                self.assertIsNone(controller.current)

    def test_reentrant_checkpoint_keeps_targets_during_original_forward_and_backward(self) -> None:
        """No-grad original forward and grad-enabled reverse layer order read the same targets."""
        model = TinyMoe(checkpointed=True)
        controller = install_router_replay(model)
        experience = _experience()
        calls = []

        def observe(layer_id: int, module: torch.nn.Module, inputs: tuple, output: tuple) -> None:
            """Record checkpoint grad mode while checking bound selections."""
            del module, inputs
            self.assertIsNotNone(controller.current)
            indices, valid = controller.layer_targets(layer_id)
            torch.testing.assert_close(output[2][valid], indices[valid].long())
            calls.append((layer_id, torch.is_grad_enabled()))

        handles = [layer.mlp.gate.register_forward_hook(partial(observe, layer_id))
                   for layer_id, layer in enumerate(model.layers)]
        targets = controller.prepare(experience.router_traces, experience.attention_mask, experience.action_mask)
        with controller.use(targets):
            model(experience.sequences, experience.attention_mask).sum().backward()
        self.assertEqual(calls, [(0, False), (1, False), (1, True), (0, True)])
        self.assertIsNone(controller.current)
        for handle in handles:
            handle.remove()


    def test_actor_diagnostics_backward_cleanup_and_reference_isolation(self) -> None:
        """Only the trainable Actor replays; both normal and exceptional scopes clear state."""
        model = TinyMoe(checkpointed=True)
        reference_model = copy.deepcopy(model)
        controller = install_router_replay(model)
        algorithm = build_algorithm({"name": "grpo", "loss_aggregation": "token-mean"})
        actor = Actor(
            model, algorithm, 1, optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            device=torch.device("cpu"), router_replay=controller,
        )
        reference = Actor(reference_model, algorithm, 1)
        experience = _experience()
        replayed = actor.compute_log_probs(experience)
        self.assertTrue(actor.training)
        self.assertIsNone(controller.current)
        self.assertFalse(torch.allclose(replayed, reference.compute_log_probs(experience)))
        actor.forward_backward(replace(experience, old_log_probs=replayed), 0, 1, global_tokens=4)
        self.assertIsNone(controller.current)
        self.assertGreater(float(model.layers[0].mlp.gate.weight.grad.abs().sum()), 0)
        self.assertFalse(reference_model.layers[0].mlp.gate._forward_hooks)  # pylint: disable=protected-access
        model.fail = True
        with self.assertRaisesRegex(RuntimeError, "test forward failure"):
            actor.compute_log_probs(experience)
        self.assertIsNone(controller.current)
        self.assertTrue(actor.training)
        model.fail = False
        torch.testing.assert_close(actor.compute_log_probs(experience), replayed)
        self.assertIsNone(controller.current)


    def test_backward_exception_clears_targets(self) -> None:
        """A real checkpoint backward failure cannot leak routes into subsequent micro-batches."""
        model = TinyMoe(checkpointed=True)
        replay = install_router_replay(model)
        actor = Actor(
            model, build_algorithm({"name": "grpo", "loss_aggregation": "token-mean"}), 1,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1), device=torch.device("cpu"),
            router_replay=replay,
        )

        def fail_backward(*args: object, **kwargs: object) -> None:
            """Fail while replay targets are still bound."""
            del args, kwargs
            self.assertIsNotNone(replay.current)
            raise RuntimeError("test backward failure")

        handle = model.layers[1].mlp.gate.weight.register_hook(fail_backward)
        try:
            with self.assertRaisesRegex(RuntimeError, "test backward failure"):
                actor.forward_backward(_experience(), 0, 1, global_tokens=4)
        finally:
            handle.remove()
        self.assertIsNone(replay.current)

    def test_single_response_token_replays_its_prompt_and_keeps_final_input_native(self) -> None:
        """One action needs the prompt route; the unsent final token contributes no input gradient."""
        model = TinyMoe(checkpointed=True)
        baseline = copy.deepcopy(model)
        replay = install_router_replay(model)
        actor = Actor(
            model, build_algorithm({"name": "grpo", "loss_aggregation": "token-mean"}), 1,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1), device=torch.device("cpu"),
            router_replay=replay,
        )
        experience = replace(
            _experience(), sequences=torch.tensor([[1, 1]]), attention_mask=torch.ones(1, 2, dtype=torch.bool),
            action_mask=torch.tensor([[False, True]]), rewards=torch.ones(1), responses=("one",),
            old_log_probs=torch.zeros(1, 1), advantages=torch.ones(1, 1), reference_log_probs=torch.zeros(1, 1),
            router_traces=(_trace(2),),
        )
        expected, _ = _manual_forward(baseline, experience)
        actual = actor.compute_log_probs(experience)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.shape, (1, 1))
        actor.forward_backward(replace(experience, old_log_probs=actual), 0, 1, global_tokens=1)
        self.assertGreater(float(model.input_hidden.grad[0].abs().sum()), 0)
        torch.testing.assert_close(model.input_hidden.grad[1], torch.zeros(2))
        self.assertIsNone(replay.current)

    def test_dp_dummy_preserves_real_advantages_loss_and_router_expert_gradients(self) -> None:
        """Production padding contributes zero loss/gradients and leaves real GRPO targets unchanged."""
        source = _experience()
        rows = []
        for row, length in enumerate((4, 3)):
            action_start = int(source.action_mask[row].nonzero()[0])
            rows.append(Trajectory(
                trajectory_id=f"response-{row}", prompt_id="prompt", group_id="group", policy_version=0,
                turns=(Turn("user", "prompt", 0, action_start, False),
                       Turn("assistant", "answer", action_start, length, True)),
                token_ids=source.sequences[row, :length], attention_mask=source.attention_mask[row, :length],
                action_mask=source.action_mask[row, :length], rollout_log_probs=source.old_log_probs[row, :length - 1],
                reward=float(row), reward_components={"answer": float(row)}, done=True, truncated=False,
                terminal_reason="completed", worker_policy_version=0, router_trace=source.router_traces[row],
                metadata={"episode_id": f"episode-{row}", "call_index": 0, "call_count": 1},
            ))
        rollout = replace(source, trajectories=tuple(rows), rewards=torch.tensor([0.0, 1.0]),
                          advantages=None, reference_log_probs=None, worker_policy_version=0)
        algorithm = build_algorithm({"name": "grpo", "loss_aggregation": "token-mean", "kl_coef": 0.0})
        preparer = ExperiencePreparer(algorithm)
        real = preparer.prepare(rollout)
        padded = preparer.prepare(_append_padding_rows(rollout, 1))
        torch.testing.assert_close(padded.advantages[:2], real.advantages)
        torch.testing.assert_close(padded.advantages[2], torch.zeros(3))
        self.assertGreater(float(real.advantages.abs().sum()), 0)
        self.assertFalse(bool(padded.router_traces[2].valid_mask.any()))

        template = TinyMoe(checkpointed=True)
        gradients = []
        losses = []
        for experience in (real, padded):
            model = copy.deepcopy(template)
            replay = install_router_replay(model)
            actor = Actor(
                model, algorithm, 1, optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
                device=torch.device("cpu"), router_replay=replay,
            )
            metrics = actor.forward_backward(experience, 0, experience.sequences.shape[0], global_tokens=4)
            gradients.append([parameter.grad.clone() for parameter in model.parameters()])
            losses.append(metrics.total_loss_sum)
            self.assertIsNone(replay.current)
        torch.testing.assert_close(losses[0], losses[1])
        for actual, expected in zip(gradients[1], gradients[0]):
            torch.testing.assert_close(actual, expected)

        model.zero_grad(set_to_none=True)
        dummy_metrics = actor.forward_backward(padded, 2, 3, global_tokens=4)
        self.assertEqual(float(dummy_metrics.total_loss_sum), 0)
        for parameter in model.parameters():
            torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter))
        torch.testing.assert_close(model.input_hidden.grad, torch.zeros_like(model.input_hidden))
        self.assertIsNone(replay.current)
