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
"""Numerical and streaming tests for optimizer-step loss objectives."""

from __future__ import annotations

from contextlib import nullcontext
from collections.abc import Iterator
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

import torch
from torch import nn

from hyper_parallel.components.functional.aux_loss import aux_loss_auto_scale
from hyper_parallel.components.optim.mixed_precision_optimizer import Float16OptimizerWithFloat16Params
from hyper_parallel.core.optimizer.adamw import AdamW
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer
from hyper_parallel.distributed.mesh import MeshContext
from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.config.resolver import ConfigResolutionError, normalize_value
from hyper_parallel.trainer.config.training import TrainingConfig
from hyper_parallel.trainer.runtime.loss_aggregation import count_loss_token
from hyper_parallel.trainer.runtime.loss_normalization import StepLossNormalization
from hyper_parallel.trainer.state import TrainerState
from hyper_parallel.trainer.text_trainer import TextTrainer
from hyper_parallel.trainer.vlm_trainer import VLMTrainer


class ScalarModel(nn.Module):
    """A mean task objective with an optional independently injected penalty."""

    def __init__(self, auxiliary: bool = False) -> None:
        """Create two independently differentiable scalar objectives."""
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.router = nn.Parameter(torch.tensor(3.0))
        self.auxiliary = auxiliary

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None, use_cache: bool = False,
    ) -> torch.Tensor:
        """Keep the target count independent of the differentiable task value."""
        del labels, use_cache
        loss = self.weight.square() * input_ids.float().mean()
        if self.auxiliary:
            loss = aux_loss_auto_scale(loss, self.router.square() * input_ids.float().mean())
        return loss


class TestStepLossNormalization(unittest.TestCase):
    """Compare actual gradients with explicitly constructed objectives."""

    def test_finish_selects_main_gradient_before_regular_gradient(self):
        """Match optimizer gradient selection, including aliases and missing buffers."""
        for storage in ("regular", "fallback", "main", "both", "alias", "empty"):
            with self.subTest(storage=storage):
                parameter = nn.Parameter(torch.tensor(2.0))
                normalizer = StepLossNormalization("token", 1, MeshContext())
                normalizer.reduce(parameter.square(), {"foundation_tokens": torch.tensor(600)})
                if storage in ("regular", "fallback", "both", "alias"):
                    parameter.grad = torch.tensor(1200.0)
                if storage != "regular":
                    parameter.main_grad = None
                if storage in ("main", "both"):
                    parameter.main_grad = torch.tensor(1200.0)
                if storage == "alias":
                    parameter.main_grad = parameter.grad
                loss, metrics = normalizer.finish(
                    [parameter, parameter], 2400.0, {"foundation_loss": 2400.0},
                )
                self.assertEqual(loss, 4.0)
                self.assertEqual(metrics["foundation_loss"], 4.0)
                if storage in ("main", "both", "alias"):
                    self.assertAlmostEqual(parameter.main_grad.item(), 2.0)
                if parameter.grad is not None:
                    self.assertAlmostEqual(parameter.grad.item(), 1200.0 if storage == "both" else 2.0)
                if storage in ("main", "empty"):
                    self.assertIsNone(parameter.grad)

    def test_fp32_main_gradient_clipping_and_optimizer_update(self):
        """FP32 master updates match a directly normalized objective after FSDP-style storage."""
        model = nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
        initial = torch.tensor([[1.0, 2.0]])
        with torch.no_grad():
            model.weight.copy_(initial)
        optimizer = Float16OptimizerWithFloat16Params(
            ChainedOptimizer(model, {"adamw": AdamW(model.parameters(), lr=0.01)}), model,
        )
        reference = nn.Parameter(initial.clone())
        reference_optimizer = AdamW([reference], lr=0.01)
        normalizer = StepLossNormalization("token", 2, MeshContext())
        total = 0.0
        for values in (torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]]).repeat(3, 1)):
            local_mean = (model.weight.float().square() * values).sum(dim=1).mean()
            loss = normalizer.reduce(local_mean, {"foundation_tokens": torch.tensor(len(values))})
            loss["foundation_loss"].backward()
            total += loss["foundation_loss"].item()
        expected = (reference.square() * torch.tensor([[1., 2.], [3., 4.], [3., 4.], [3., 4.]])).sum() / 4
        expected.backward()
        # Emulate FSDP's reduced FP32 buffer without requiring a process group.
        model.weight.main_grad = model.weight.grad.float()
        model.weight.grad = None
        actual, _ = normalizer.finish(model.parameters(), total, {})
        optimizer.prepare_grads()
        main_param = model.weight.main_param
        self.assertEqual(main_param.grad.dtype, torch.float32)
        self.assertAlmostEqual(actual, expected.item())
        torch.testing.assert_close(main_param.grad, reference.grad)
        actual_norm = nn.utils.clip_grad_norm_([main_param], max_norm=0.1)
        expected_norm = nn.utils.clip_grad_norm_([reference], max_norm=0.1)
        torch.testing.assert_close(actual_norm, expected_norm)
        optimizer.step_with_ready_grads()
        reference_optimizer.step()
        torch.testing.assert_close(main_param, reference)
        torch.testing.assert_close(model.weight, reference.to(torch.bfloat16))

    def test_formulas_and_partition_invariance(self):
        """Token normalization is invariant to repartitioning a fixed set of targets."""
        values = torch.tensor([1.0, 3.0, 5.0, 7.0])
        for mode in ("microbatch", "token"):
            for sizes in ((1, 3), (2, 2), (4,)):
                with self.subTest(mode=mode, sizes=sizes):
                    parameter = nn.Parameter(torch.tensor(2.0))
                    reference = nn.Parameter(torch.tensor(2.0))
                    chunks = values.split(sizes)
                    means = [reference.square() * chunk.mean() for chunk in chunks]
                    objective = (torch.stack(means).mean() if mode == "microbatch"
                                 else reference.square() * values.mean())
                    objective.backward()
                    normalizer = StepLossNormalization(mode, len(chunks), MeshContext())
                    total = 0.0
                    for chunk in chunks:
                        losses = normalizer.reduce(parameter.square() * chunk.mean(),
                                                   {"foundation_tokens": torch.tensor(chunk.numel())})
                        loss = losses["foundation_loss"]
                        loss.backward()
                        total += loss.item()
                    actual, metrics = normalizer.finish([parameter, parameter], total,
                                                        {"foundation_loss": total, "router_stat": 7.0})
                    torch.testing.assert_close(parameter.grad, reference.grad)
                    self.assertAlmostEqual(actual, objective.item(), places=5)
                    self.assertEqual(metrics["router_stat"], 7.0)
                    self.assertAlmostEqual(metrics["foundation_loss"], actual, places=5)

    def test_empty_microbatch_and_empty_step(self):
        """Empty micro-batches contribute zero but still occupy their micro-batch slot."""
        for mode in ("microbatch", "token"):
            for counts in ((0, 3), (0, 0)):
                with self.subTest(mode=mode, counts=counts):
                    parameter = nn.Parameter(torch.tensor(2.0))
                    normalizer = StepLossNormalization(mode, 2, MeshContext())
                    total = 0.0
                    for count in counts:
                        loss = normalizer.reduce(parameter.square(),
                                                 {"foundation_tokens": torch.tensor(count)})["foundation_loss"]
                        loss.backward()
                        total += loss.item()
                    total, _ = normalizer.finish([parameter], total, {})
                    expected = 0.0 if not sum(counts) else (2.0 if mode == "microbatch" else 4.0)
                    self.assertEqual(total, expected)
                    self.assertEqual(parameter.grad.item(), expected)

    def test_masked_ce_gradients_clipping_and_adam_update(self):
        """Masked CE, clipped gradients, and Adam updates match the chosen full objective."""
        generator = torch.Generator().manual_seed(42)
        inputs = torch.randn(2, 4, 3, generator=generator)
        labels = torch.tensor([[0, 1, -100, 2], [1, 0, 2, 1]])
        masks = torch.tensor([[1, 0, 1, 0], [1, 1, 0, 1]])
        initial = torch.randn(3, 3, generator=generator)
        for mode in ("microbatch", "token"):
            with self.subTest(mode=mode):
                parameter = nn.Parameter(initial.clone())
                reference = nn.Parameter(initial.clone())
                optimizer = torch.optim.Adam([parameter], lr=0.01)
                reference_optimizer = torch.optim.Adam([reference], lr=0.01)
                normalizer = StepLossNormalization(mode, 2, MeshContext())
                reference_sums, counts = [], []
                total = 0.0
                for data, targets, mask in zip(inputs, labels, masks):
                    token_counts = count_loss_token({"shift_labels": targets, "loss_mask": mask})
                    count = token_counts["foundation_tokens"]
                    valid = mask.bool() & (targets != -100)
                    per_token = nn.functional.cross_entropy(data @ parameter, targets, reduction="none")
                    local_mean = (per_token * valid).sum() / count
                    loss = normalizer.reduce(local_mean, token_counts)["foundation_loss"]
                    loss.backward()
                    total += loss.item()
                    reference_ce = nn.functional.cross_entropy(data @ reference, targets, reduction="none")
                    reference_sums.append((reference_ce * valid).sum())
                    counts.append(count)
                reference_sums, counts = torch.stack(reference_sums), torch.stack(counts)
                expected = ((reference_sums / counts).mean() if mode == "microbatch"
                            else reference_sums.sum() / counts.sum())
                expected.backward()
                total, _ = normalizer.finish([parameter], total, {})
                self.assertAlmostEqual(total, expected.item(), places=6)
                torch.testing.assert_close(parameter.grad, reference.grad)
                actual_norm = nn.utils.clip_grad_norm_([parameter], max_norm=0.1)
                expected_norm = nn.utils.clip_grad_norm_([reference], max_norm=0.1)
                torch.testing.assert_close(actual_norm, expected_norm)
                optimizer.step()
                reference_optimizer.step()
                torch.testing.assert_close(parameter, reference)

    def test_token_mode_rejects_independent_denominators(self):
        """One final scalar cannot normalize different objective families."""
        normalizer = StepLossNormalization("token", 2, MeshContext())
        with self.assertRaisesRegex(ValueError, "shared loss token family"):
            normalizer.reduce({"foundation_loss": torch.tensor(1.), "image_loss": torch.tensor(2.)}, {})

    def test_shared_family_is_counted_once(self):
        """Multiple terms sharing the LM divisor do not multiply its token count."""
        parameter = nn.Parameter(torch.tensor(2.0))
        normalizer = StepLossNormalization("token", 1, MeshContext())
        losses = normalizer.reduce(
            {"foundation_loss": parameter.square(), "foundation_loss_mtp": parameter.square() / 2},
            {"foundation_tokens": torch.tensor(3)},
        )
        total = sum(losses.values())
        total.backward()
        actual, metrics = normalizer.finish([parameter], total.item(),
                                            {key: value.item() for key, value in losses.items()})
        self.assertEqual(actual, 6.0)
        self.assertEqual(parameter.grad.item(), 6.0)
        self.assertEqual(metrics["foundation_loss_mtp"], 2.0)
        with self.assertRaisesRegex(ValueError, "shared loss token family"):
            normalizer.reduce({"image_loss": torch.tensor(1.)}, {"image_tokens": torch.tensor(3)})

    def test_mode_validation(self):
        """Validate parsed YAML and direct runtime construction at their existing boundaries."""
        self.assertEqual(TrainingConfig().loss_normalization, "legacy")
        self.assertEqual(normalize_value({}, TrainingConfig, path="training").loss_normalization, "legacy")
        for mode in ("legacy", "microbatch", "token"):
            config = normalize_value({"loss_normalization": mode}, TrainingConfig, path="training")
            self.assertEqual(config.loss_normalization, mode)
        with self.assertRaisesRegex(ConfigResolutionError, "training.loss_normalization"):
            normalize_value({"loss_normalization": "tokens"}, TrainingConfig, path="training")
        config = TrainingConfig(loss_normalization="tokens")
        with self.assertRaisesRegex(ValueError, "loss_normalization"):
            StepLossNormalization(config.loss_normalization, 1, MeshContext())

    def test_cp_denominator_and_dp_gradient_compensation(self):
        """CP shards form one micro-batch; DP replicas contribute independent micro-batches."""
        cp = SimpleNamespace(size=lambda: 2, get_group=lambda: "cp")
        dp_cp = SimpleNamespace(size=lambda: 4, get_group=lambda: "dp_cp")
        mesh = SimpleNamespace(cp_mesh=cp, dp_cp_mesh=dp_cp, cp_size=2, dp_size=2)
        for mode, expected_grad in (("microbatch", 0.5), ("token", 1.0)):
            with self.subTest(mode=mode):
                parameter = nn.Parameter(torch.tensor(2.0))
                normalizer = StepLossNormalization(mode, 2, mesh)
                def reduce(value: torch.Tensor, op: str, group: str) -> torch.Tensor:
                    """Supply unequal CP target counts without initializing process groups."""
                    del op
                    return value * (4 if group == "cp" else 8)
                with patch("hyper_parallel.trainer.runtime.loss_normalization.differentiable_all_reduce",
                           side_effect=reduce):
                    loss = normalizer.reduce(parameter.square(), {"foundation_tokens": torch.tensor(1)})
                    loss["foundation_loss"].backward()
                    # Emulate the FSDP average, with only this rank contributing a gradient.
                    parameter.grad.div_(4)
                    normalizer.finish([parameter], 0, {})
                self.assertEqual(parameter.grad.item(), expected_grad / 2)


class TestTrainerLossNormalization(unittest.TestCase):
    """Exercise real Trainer forward/backward, including input lifetime and aux hooks."""

    @staticmethod
    def make_trainer(
        trainer_type: type[TextTrainer] | type[VLMTrainer], mode: str, auxiliary: bool,
    ) -> TextTrainer | VLMTrainer:
        """Construct CPU dependencies without starting distributed training."""
        trainer = trainer_type.__new__(trainer_type)
        base = BaseTrainer.__new__(BaseTrainer)
        trainer.base = base
        base.model = ScalarModel(auxiliary)
        base.mesh = MeshContext()
        base.device = torch.device("cpu")
        base.config = SimpleNamespace(training=TrainingConfig(loss_normalization=mode))
        base.num_micro_batches = 2
        base.graph_compiler = None
        base.model_fwd_context = nullcontext()
        base.model_bwd_context = nullcontext()
        base.model_integration = Mock()
        # Mock call histories would themselves retain the inputs under test.
        base.model_integration.record_batch = lambda *args: None
        base.loss_fn = lambda model_output, labels: model_output
        base.state = TrainerState()
        base._callbacks = []
        base.fsdp_runtime_diagnostics = None
        base.model_reshard = Mock()
        base.configure_fsdp_gradient_sync = Mock()
        base.prepare_optimizer_step = Mock(return_value=0.0)
        base.step_optimizers_and_schedulers = Mock()
        base._end_model_integration_step = Mock()
        return trainer

    def test_streaming_backward_and_auxiliary_scaling(self):
        """Both entrypoints release inputs and normalize all gradients before optimizer preparation."""
        for trainer_type in (TextTrainer, VLMTrainer):
            for mode in ("legacy", "microbatch", "token"):
                for auxiliary in (False, True):
                    with self.subTest(trainer=trainer_type, mode=mode, auxiliary=auxiliary):
                        trainer = self.make_trainer(trainer_type, mode, auxiliary)
                        refs = []
                        def get_batch(
                            iterator: Iterator[tuple[int, float]],
                        ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
                            """Allocate inputs lazily to detect retention across micro-steps."""
                            self.assertTrue(all(ref() is None for ref in refs))
                            count, value = next(iterator)
                            data = torch.full((1, 3), value)
                            refs.append(weakref.ref(data))
                            labels = torch.tensor([[1, 2, 3]])
                            mask = torch.arange(3).reshape(1, 3) < count
                            return {"input_ids": data}, {"shift_labels": labels, "loss_mask": mask}
                        trainer.base.get_batch = get_batch
                        expected = 1.75 if mode == "token" else 1.5
                        def before_clip() -> float:
                            """Inspect final gradients at the optimizer preparation boundary."""
                            self.assertAlmostEqual(trainer.base.model.weight.grad.item(), 4 * expected)
                            if auxiliary:
                                self.assertAlmostEqual(trainer.base.model.router.grad.item(), 6 * expected)
                            else:
                                self.assertIsNone(trainer.base.model.router.grad)
                            return 0.0
                        trainer.base.prepare_optimizer_step.side_effect = before_clip
                        module = "text_trainer" if trainer_type is TextTrainer else "vlm_trainer"
                        with patch(f"hyper_parallel.trainer.{module}.synchronize"), \
                             patch(f"hyper_parallel.trainer.{module}.memory_profiler.step"), \
                             patch("hyper_parallel.trainer.runtime.metrics.all_reduce",
                                   side_effect=lambda value, **_: value):
                            result = trainer.train_step(iter(((1, 1.0), (3, 2.0))))
                        self.assertAlmostEqual(result["loss"], 4 * expected)
                        self.assertEqual(trainer.base.state.global_step, 1)
                        # A new step must reset its denominator, not accumulate previous steps.
                        with patch(f"hyper_parallel.trainer.{module}.synchronize"), \
                             patch(f"hyper_parallel.trainer.{module}.memory_profiler.step"), \
                             patch("hyper_parallel.trainer.runtime.metrics.all_reduce",
                                   side_effect=lambda value, **_: value):
                            trainer.base.model.zero_grad()
                            trainer.train_step(iter(((1, 1.0), (3, 2.0))))
                        self.assertEqual(trainer.base.state.global_step, 2)

    def test_token_mode_rejects_unmanaged_gradient_paths(self):
        """Fail before consuming data when gradients cannot be finalized safely."""
        trainer = self.make_trainer(TextTrainer, "token", False)
        trainer.base.graph_compiler = object()
        with self.assertRaisesRegex(ValueError, "eager"):
            trainer.base.begin_loss_step(2)
        trainer.base.graph_compiler = None
        trainer.base.model._hp_external_state = object()
        with self.assertRaisesRegex(ValueError, "externally managed"):
            trainer.base.begin_loss_step(2)
        del trainer.base.model._hp_external_state
        trainer.base.mesh.pp_size = 2
        with self.assertRaisesRegex(ValueError, "pipeline parallelism"):
            trainer.base.begin_loss_step(2)

    def test_legacy_preserves_existing_execution_paths(self):
        """Default mode bypasses new restrictions and clears a previous step's normalizer."""
        trainer = self.make_trainer(TextTrainer, "microbatch", False)
        base = trainer.base
        base.begin_loss_step(2)
        self.assertIsNotNone(base.loss_normalizer)
        base.config.training = TrainingConfig()
        base.graph_compiler = object()
        base.model._hp_external_state = object()
        base.mesh.pp_size = 2
        base.model.weight.grad = torch.tensor(7.0)
        base.begin_loss_step(2)
        self.assertIsNone(base.loss_normalizer)
        metrics = {"foundation_loss": 3.0}
        loss, actual_metrics = base.finish_loss_step(3.0, metrics)
        self.assertEqual(loss, 3.0)
        self.assertIs(actual_metrics, metrics)
        self.assertEqual(base.model.weight.grad.item(), 7.0)

    def test_microbatch_does_not_reject_pipeline_parallelism(self):
        """Only token mode introduces a PP restriction; microbatch has no final gradient scaling."""
        trainer = self.make_trainer(TextTrainer, "microbatch", False)
        trainer.base.mesh.pp_size = 2
        trainer.base.graph_compiler = object()
        trainer.base.begin_loss_step(2)
        self.assertEqual(trainer.base.loss_normalizer.mode, "microbatch")
        trainer.base.model.weight.grad = torch.tensor(7.0)
        trainer.base.finish_loss_step(3.0, {})
        self.assertEqual(trainer.base.model.weight.grad.item(), 7.0)

    def test_legacy_uses_original_dp_weighting(self):
        """Unequal DP counts retain the old per-micro-step token weighting and backward scale."""
        counts = ((1, 2), (4, 1))
        values = ((1.0, 4.0), (3.0, 8.0))
        gradients = []
        expected = sum(
            sum(counts[rank][step] * values[rank][step] for rank in range(2))
            / sum(counts[rank][step] for rank in range(2)) for step in range(2)
        ) / 2
        for rank in range(2):
            base = self.make_trainer(TextTrainer, "legacy", False).base
            base.mesh.dp_size = 2
            base.begin_loss_step(2)
            total = 0.0
            for step in range(2):
                count = torch.tensor(counts[rank][step])
                global_count = sum(counts[replica][step] for replica in range(2))
                global_sum = 4 * sum(counts[replica][step] * values[replica][step] for replica in range(2))
                with patch("hyper_parallel.trainer.runtime.metrics.all_reduce",
                           side_effect=(global_count * 2, global_sum)) as reduce_mock:
                    loss, _ = base.postforward(
                        base.model.weight.square() * values[rank][step], None,
                        current_token_counts={"foundation_tokens": count},
                        step_token_counts={"foundation_tokens": count * 2},
                    )
                    self.assertEqual(reduce_mock.call_count, 2)
                    self.assertEqual(reduce_mock.call_args_list[0].args[0], counts[rank][step] * 2)
                loss.backward()
                total += loss.item()
            total, _ = base.finish_loss_step(total, {})
            self.assertAlmostEqual(total, 4 * expected, places=5)
            gradients.append(base.model.weight.grad)
        torch.testing.assert_close(torch.stack(gradients).mean(), torch.tensor(4 * expected))
