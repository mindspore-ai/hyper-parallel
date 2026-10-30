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
"""Optimizer-step objectives must not depend on micro-batch partitioning."""

from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from typing import Union
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn
from torch.nn import functional as F
from torch.fx.experimental.proxy_tensor import make_fx

from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.runtime.loss_aggregation import count_loss_token
from hyper_parallel.trainer.runtime.metrics import mean_global_loss
from hyper_parallel.trainer.text_trainer import TextTrainer
from hyper_parallel.trainer.vlm_trainer import VLMTrainer
from tests.common.mark_utils import arg_mark


class TinyTokenModel(nn.Module):
    """Deterministic token classifier with no batch-dependent operations."""

    def __init__(self) -> None:
        """Initialize identical weights for every reference and training run."""
        super().__init__()
        self.weight = nn.Parameter(torch.arange(15, dtype=torch.float64).reshape(3, 5) / 20)

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        """Project per-token features to logits."""
        del use_cache
        return input_ids @ self.weight


def token_loss(model_output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return a local valid-token mean, including a differentiable empty loss."""
    return F.cross_entropy(model_output, labels, reduction="sum") / (labels != -100).sum().clamp_min(1)


def make_trainer(trainer_type: type, num_micro_batches: int) -> object:
    """Use real forward/backward and aggregation with CPU lifecycle stand-ins."""
    base = BaseTrainer.__new__(BaseTrainer)
    base.device = torch.device("cpu")
    base.model = TinyTokenModel()
    base.model_fwd_context = nullcontext()
    base.model_bwd_context = nullcontext()
    base.graph_compiler = None
    base.model_integration = Mock()
    base.config = SimpleNamespace(training=SimpleNamespace(empty_cache_before_backward=False))
    base.mesh = SimpleNamespace(dp_cp_mesh=None, sequence_parallel=False, dp_size=1, cp_size=1)
    base.loss_fn = token_loss
    base.num_micro_batches = num_micro_batches
    base.state = SimpleNamespace(global_step=0)
    base.get_batch = next
    for name in (
            "on_step_begin", "on_step_end", "on_micro_step_begin", "model_reshard",
            "configure_fsdp_gradient_sync", "begin_fsdp_runtime_diagnostics",
            "step_optimizers_and_schedulers", "_end_model_integration_step",
    ):
        setattr(base, name, Mock())
    base.prepare_optimizer_step = Mock(return_value=0.0)
    trainer = trainer_type.__new__(trainer_type)
    trainer.base = base
    return trainer


class TestStepTokenNormalization(unittest.TestCase):
    """Exercise both public trainer loops against a full-batch reference."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_partitioned_steps_match_full_batch_loss_and_gradient(self) -> None:
        """Feature: Whole-step token normalization.

        Description: Train identical tokens with several unequal micro-batch partitions.
        Expectation: Both trainer loops match the full-batch loss and parameter gradient.
        """
        features = torch.arange(36, dtype=torch.float64).reshape(12, 3) / 30
        labels = torch.tensor([0, -100, 2, 1, 4, -100, 0, 3, 1, 2, 4, 0])
        reference = TinyTokenModel()
        expected_loss = token_loss(reference(features), labels)
        expected_loss.backward()
        for trainer_type in (TextTrainer, VLMTrainer):
            for sizes in ((12,), (4, 4, 4), (1, 4, 7), (2, 10), (1, 1, 10)):
                with self.subTest(trainer=trainer_type.__name__, sizes=sizes), ExitStack() as stack:
                    for module in ("text_trainer", "vlm_trainer"):
                        stack.enter_context(patch(f"hyper_parallel.trainer.{module}.synchronize"))
                        stack.enter_context(patch(f"hyper_parallel.trainer.{module}.memory_profiler.step"))
                    stack.enter_context(patch(
                        "hyper_parallel.trainer.runtime.metrics.all_reduce", side_effect=lambda value, **_: value,
                    ))
                    batches = [
                        ({"input_ids": inputs}, {"labels": targets, "shift_labels": targets})
                        for inputs, targets in zip(features.split(sizes), labels.split(sizes))
                    ]
                    trainer = make_trainer(trainer_type, len(batches))
                    optimizer_step = Mock()
                    trainer.base.step_optimizers_and_schedulers = optimizer_step
                    actual = trainer.train_step(iter(batches))
                    self.assertAlmostEqual(actual["loss"], expected_loss.item(), places=12)
                    torch.testing.assert_close(trainer.base.model.weight.grad, reference.weight.grad)
                    self.assertEqual(int(trainer.base.step_token_counts["foundation_tokens"]), 10)
                    optimizer_step.assert_called_once()
                    self.assertEqual(trainer.base.state.global_step, 1)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_count_all_named_targets_across_micro_batches(self) -> None:
        """Feature: Per-objective token totals.

        Description: Count ignored labels, explicit masks and multiple modality targets.
        Expectation: Each objective sums its own valid targets across micro-batches.
        """
        batches = [
            {"shift_labels": torch.tensor([1, -100, 2]), "loss_mask": torch.tensor([1, 1, 0]),
             "audio_labels": torch.tensor([1, 2, -100]), "image_output_mask": torch.tensor([1, 0])},
            {"shift_labels": torch.tensor([3, 4]), "loss_mask": torch.tensor([1, 1]),
             "audio_labels": torch.tensor([3, 4, 5]), "image_output_mask": torch.tensor([1, 1])},
        ]
        counts = count_loss_token(batches)
        self.assertEqual({name: int(value) for name, value in counts.items()},
                         {"foundation_tokens": 3, "audio_tokens": 5, "image_decoder_tokens": 3})

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_single_micro_batch_preserves_unit_gradient(self) -> None:
        """Feature: Exact identity loss scaling.

        Description: Normalize a single micro-batch containing 164 valid targets.
        Expectation: The local-loss derivative is exactly one in eager and traced paths.
        """
        mesh = SimpleNamespace(dp_cp_mesh=None, sequence_parallel=False, dp_size=1, cp_size=1)
        for traced in (False, True):
            with self.subTest(traced=traced), ExitStack() as stack:
                stack.enter_context(patch("hyper_parallel.trainer.runtime.metrics._is_tracing_or_compiling",
                                          return_value=traced))
                stack.enter_context(patch("hyper_parallel.trainer.runtime.metrics.all_reduce",
                                          side_effect=lambda value, **_: value))
                stack.enter_context(patch("hyper_parallel.trainer.runtime.metrics.differentiable_all_reduce",
                                          side_effect=lambda value, *_: value))
                loss = torch.tensor(3.0, requires_grad=True)
                counts = {"foundation_tokens": torch.tensor(164)}
                actual = mean_global_loss(loss, counts, counts, mesh)["foundation_loss"]
                actual.backward()
                self.assertEqual(loss.grad.item(), 1.0)
                self.assertEqual(actual.item(), 3.0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_dp_cp_weighting_matches_global_objective(self) -> None:
        """Feature: DP/CP loss normalization.

        Description: Simulate uneven DP/CP shards and replicated TP token counts.
        Expectation: Reported loss and averaged gradients match the global token objective.
        """
        shards = (((1.0,), (2.0, 3.0)), ((4.0, 5.0), (6.0,)),
                  ((7.0, 8.0, 9.0), ()), ((10.0,), (11.0, 12.0)))
        for traced in (False, True):
            for sp_size in (1, 2):
                with self.subTest(traced=traced, sp_size=sp_size):
                    gradients = []
                    mesh = SimpleNamespace(
                        dp_cp_mesh=Mock(get_group=Mock(return_value="dp_cp")),
                        dp_size=2, cp_size=2, sequence_parallel=sp_size > 1,
                        device_mesh=_MockDeviceMesh({"tp": Mock(get_group=Mock(return_value="tp"))}),
                    )
                    for rank_shards in shards:
                        parameter = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
                        actual_loss = 0.0
                        for micro_step, values in enumerate(rank_shards):
                            weighted_sum = sum(sum(rank[micro_step]) for rank in shards) * 2 * sp_size
                            reductions = iter((12, weighted_sum))

                            def reduce_value(
                                    value: Union[float, torch.Tensor], *args: object,
                                    group: object = None, **_: object,
                            ) -> Union[float, torch.Tensor]:
                                """Replay mathematical SUM collectives without a process group."""
                                actual_group = args[1] if args else group
                                result = value * sp_size if actual_group == "tp" else next(reductions)
                                if torch.is_tensor(result):
                                    return result.clone()
                                return value.new_tensor(result) if torch.is_tensor(value) else result

                            with ExitStack() as stack:
                                stack.enter_context(patch(
                                    "hyper_parallel.trainer.runtime.metrics._is_tracing_or_compiling",
                                    return_value=traced,
                                ))
                                stack.enter_context(patch("hyper_parallel.trainer.runtime.metrics.all_reduce",
                                                          side_effect=reduce_value))
                                stack.enter_context(patch(
                                    "hyper_parallel.trainer.runtime.metrics.differentiable_all_reduce",
                                    side_effect=reduce_value,
                                ))
                                stack.enter_context(patch(
                                    "hyper_parallel.trainer.runtime.metrics.dist.get_world_size", return_value=sp_size,
                                ))
                                loss = parameter * (sum(values) / max(len(values), 1))
                                result = mean_global_loss(
                                    loss, {"foundation_tokens": torch.tensor(len(values))},
                                    {"foundation_tokens": torch.tensor(sum(map(len, rank_shards)))}, mesh,
                                )["foundation_loss"]
                                result.backward()
                                actual_loss += result.item()
                        self.assertAlmostEqual(actual_loss, 13.0)
                        gradients.append(parameter.grad)
                    torch.testing.assert_close(torch.stack(gradients).mean(), torch.tensor(6.5, dtype=torch.float64))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_empty_objective_has_finite_zero_loss_and_gradient(self) -> None:
        """Feature: Empty objective normalization.

        Description: Reduce a differentiable zero loss with zero valid targets.
        Expectation: Both aggregation paths return finite zero loss and gradients.
        """
        mesh = SimpleNamespace(dp_cp_mesh=None, sequence_parallel=False, dp_size=1, cp_size=1)
        for traced in (False, True):
            with self.subTest(traced=traced), ExitStack() as stack:
                stack.enter_context(patch("hyper_parallel.trainer.runtime.metrics._is_tracing_or_compiling",
                                          return_value=traced))
                stack.enter_context(patch("hyper_parallel.trainer.runtime.metrics.all_reduce",
                                          side_effect=lambda value, **_: value))
                stack.enter_context(patch("hyper_parallel.trainer.runtime.metrics.differentiable_all_reduce",
                                          side_effect=lambda value, *_: value))
                parameter = torch.tensor(3.0, requires_grad=True)
                counts = {"foundation_tokens": torch.tensor(0)}
                result = mean_global_loss(parameter * 0, counts, counts, mesh)["foundation_loss"]
                result.backward()
                self.assertEqual(result.item(), 0.0)
                self.assertEqual(parameter.grad.item(), 0.0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_traced_token_counts_remain_runtime_inputs(self) -> None:
        """Feature: Runtime token counts in captured graphs.

        Description: Capture aggregation once and execute it with different token counts.
        Expectation: The denominator remains dynamic and empty counts do not produce NaNs.
        """
        mesh = SimpleNamespace(dp_cp_mesh=None, sequence_parallel=False, dp_size=1, cp_size=1)

        def aggregate(loss: torch.Tensor, current: torch.Tensor, total: torch.Tensor) -> torch.Tensor:
            """Expose only loss and token counts as graph inputs."""
            return mean_global_loss(
                loss, {"foundation_tokens": current}, {"foundation_tokens": total}, mesh,
            )["foundation_loss"]

        with patch("hyper_parallel.trainer.runtime.metrics._is_tracing_or_compiling", return_value=True), patch(
                "hyper_parallel.trainer.runtime.metrics.differentiable_all_reduce", side_effect=lambda value, *_: value,
        ):
            graph = make_fx(aggregate)(torch.tensor(3.0), torch.tensor(164), torch.tensor(164))
        for current, total in ((164, 164), (10, 40), (0, 40), (0, 0)):
            with self.subTest(current=current, total=total):
                loss = torch.tensor(3.0, requires_grad=True)
                result = graph(loss, torch.tensor(current), torch.tensor(total))
                result.backward()
                expected_scale = current / max(total, 1)
                self.assertAlmostEqual(result.item(), 3 * expected_scale)
                self.assertAlmostEqual(loss.grad.item(), expected_scale)


class _MockDeviceMesh(dict):
    """Expose TP group lookup and dimension names for the metric contract."""

    mesh_dim_names = ("tp",)


if __name__ == "__main__":
    unittest.main()
