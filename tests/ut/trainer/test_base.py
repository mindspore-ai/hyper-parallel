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
"""Unit tests for trainer lifecycle cleanup."""

import importlib
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import torch

from hyper_parallel.models.deepseek_v41.adapter.engram.host_state import DeepseekV41HostState
from hyper_parallel.models.deepseek_v41.adapter.engram.host_table import HostEngramTable

base_module = importlib.import_module("hyper_parallel.trainer.base")


class TestBaseTrainerCleanup(unittest.TestCase):
    """Tests for cleaning up the data iterator on exceptional exits."""

    def test_dataset_ownership_follows_loader_balance_block(self) -> None:
        """Feature: Dataset rank participation.
        Description: Build Dataset splits with absent, empty and populated loader balance settings.
        Expectation: The Trainer shares ownership context, enabling all ranks for an empty block too.
        """
        for options in (None, {}, {"metadata_mode": True}):
            with self.subTest(options=options):
                dataset = object()
                trainer = SimpleNamespace(
                    config=SimpleNamespace(
                        dataset=SimpleNamespace(build=MagicMock(return_value=(dataset, None, None))),
                        dataloader=SimpleNamespace(sampler_type="single", load_balance=options),
                        training=object(),
                    ),
                    data_transform=None, tokenizer=None, mesh=object(),
                )
                base_module.BaseTrainer._build_dataset(trainer)
                context = trainer.config.dataset.build.call_args.kwargs["dataloader_context"]
                self.assertIs(context, trainer.dataloader_context)
                self.assertEqual(context.tp_source_rank, 0 if options is None else None)
                self.assertIs(trainer.train_dataset, dataset)

    @staticmethod
    def _make_trainer() -> MagicMock:
        """Build the minimal trainer surface required by ``BaseTrainer.train``."""
        trainer = MagicMock()
        trainer.config = SimpleNamespace(
            dataloader=SimpleNamespace(
                use_background_prefetcher=True,
                drop_last=False,
            )
        )
        trainer.train_dataloader = object()
        trainer.local_rank = 0
        trainer.state = SimpleNamespace(global_step=0, epoch=0)
        trainer.train_iters = 1
        trainer.train_epochs = 1
        trainer.train_steps = 1
        return trainer

    def test_train_stops_data_iterator_when_training_or_callback_raises(self) -> None:
        """
        Feature: BaseTrainer lifecycle cleanup.
        Description: Raise from training and callback hooks while background prefetching is enabled.
        Expectation: The data iterator is stopped once and each original exception propagates unchanged.
        """
        for failure_hook in ("train_step", "on_epoch_end"):
            with self.subTest(failure_hook=failure_hook):
                trainer = self._make_trainer()
                data_iterator = MagicMock()
                expected_error = RuntimeError(f"{failure_hook} failed")
                getattr(trainer, failure_hook).side_effect = expected_error

                with patch.object(base_module, "HyperIter", return_value=data_iterator):
                    with self.assertRaises(RuntimeError) as error_context:
                        base_module.BaseTrainer.train(trainer)

                self.assertIs(error_context.exception, expected_error)
                data_iterator.stop.assert_called_once_with()


class TestHcclDeterminismSetup(unittest.TestCase):
    """Check that HCCL receives its reproducibility flag before group creation."""

    def test_hccl_flag_is_set_before_distributed_initialization(self) -> None:
        """The communicator must read the flag when it is initialized."""
        trainer = MagicMock()
        trainer.config.training.backend = "hccl"
        trainer.config.training.enable_full_determinism = True

        def check_initialization(*, backend: str) -> None:
            """Assert the HCCL flag is set before initialization, then stop."""
            self.assertEqual(backend, "hccl")
            self.assertEqual(base_module.os.environ["HCCL_DETERMINISTIC"], "true")
            raise RuntimeError("stop after preflight")

        with patch.dict(base_module.os.environ, {"HCCL_DETERMINISTIC": "false"}):
            with patch.object(base_module, "setup_logging"):
                with patch.object(base_module, "initialize_distributed", side_effect=check_initialization):
                    with self.assertRaisesRegex(RuntimeError, "stop after preflight"):
                        base_module.BaseTrainer._setup(trainer)


class TestBaseTrainerExternalState(unittest.TestCase):
    """Cover the host external-state branches of optimizer build and clip."""

    @staticmethod
    def _capture_build(captured: dict, built: Any) -> Any:
        """Return a build_optimizer stub that captures the wrap_dense hook."""
        def build_optimizer(optimizer_config: Any, *, wrap_dense: Any) -> Any:
            """Capture wrap_dense and return the prebuilt optimizer."""
            captured["wrap_dense"] = wrap_dense
            return built
        return build_optimizer

    @staticmethod
    def _make_trainer(fp32_main_params: bool, swap_enabled: bool) -> MagicMock:
        """Build the minimal trainer surface for the external optimizer path."""
        trainer = MagicMock()
        trainer.config = SimpleNamespace(
            optimizer=SimpleNamespace(
                fp32_main_params=fp32_main_params,
                swap=SimpleNamespace(enabled=swap_enabled),
            )
        )
        return trainer

    def test_build_optimizer_delegates_to_external_state(self) -> None:
        """
        Feature: Host external-state optimizer wiring.
        Description: Build the optimizer for every fp32-main/swap combination.
        Expectation: The external state owns construction and wrap_dense applies
            the fp32 main-parameter and swap wrappers in order.
        """
        for fp32_main_params in (False, True):
            for swap_enabled in (False, True):
                with self.subTest(fp32_main_params=fp32_main_params, swap_enabled=swap_enabled):
                    trainer = self._make_trainer(fp32_main_params, swap_enabled)
                    external = MagicMock()
                    built = object()
                    captured = {}
                    external.build_optimizer.side_effect = self._capture_build(captured, built)
                    with patch.object(base_module, "get_model_external_state", return_value=external), \
                            patch.object(base_module, "Float16OptimizerWithFloat16Params",
                                         side_effect=lambda raw, model: ("fp32", raw)) as fp32_wrap, \
                            patch.object(base_module, "_attach_optimizer_swap",
                                         side_effect=lambda dense, swap: ("swap", dense)) as swap_wrap:
                        base_module.BaseTrainer._build_optimizer(trainer)
                        raw = object()
                        wrapped = captured["wrap_dense"](raw)

                    self.assertIs(trainer.optimizer, built)
                    expected = raw
                    if fp32_main_params:
                        expected = ("fp32", raw)
                        fp32_wrap.assert_called_once_with(raw, trainer.model)
                    else:
                        fp32_wrap.assert_not_called()
                    if swap_enabled:
                        expected = ("swap", expected)
                        swap_wrap.assert_called_once()
                    else:
                        swap_wrap.assert_not_called()
                    self.assertEqual(wrapped, expected)

    def test_clip_grad_norm_uses_external_norm(self) -> None:
        """
        Feature: Host external-state gradient clipping.
        Description: Run clipping with external state present.
        Expectation: The external preparation drives after_clip and its global norm is returned.
        """
        trainer = MagicMock()
        trainer.config.training.max_grad_norm = 1.0
        trainer.fsdp_runtime_diagnostics = None
        external = MagicMock()
        preparation = SimpleNamespace(post_clip_norm=0.5, global_norm=2.0)
        external.prepare_optimizer_step.return_value = preparation
        with patch.object(base_module, "get_model_external_state", return_value=external):
            result = base_module.BaseTrainer.clip_grad_norm(trainer)
        self.assertEqual(result, 2.0)
        external.prepare_optimizer_step.assert_called_once_with(max_norm=1.0, gradient_divisor=1.0)
        trainer.model_integration.after_clip.assert_called_once_with(0.5)


    def test_main_grad_normalizes_before_clipping(self) -> None:
        """Feature: FP32 main-gradient normalization.

        Description: Exercise main_grad with absent or stale ordinary gradients.
        Expectation: Clipping sees the scaled FP32 buffer; diagnostic metrics stay unchanged.
        """
        for ordinary_grad in (None, torch.tensor([[99.]], dtype=torch.bfloat16)):
            trainer = base_module.BaseTrainer.__new__(base_module.BaseTrainer)
            trainer.model = torch.nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)
            trainer.model.weight.main_grad = torch.tensor([[1200.]])
            trainer.model.weight.grad = ordinary_grad
            trainer.mesh = SimpleNamespace(dp_cp_mesh=None)
            trainer.loss_names = ("foundation_loss",)
            trainer.step_token_counts = {"foundation_tokens": torch.tensor(600)}
            trainer.model.external_weight = torch.nn.Parameter(torch.tensor([0.]))
            trainer.model.external_weight.grad = torch.tensor([1200.])
            trainer.model._hp_external_state = SimpleNamespace(  # pylint: disable=protected-access
                parameters={"external_weight": trainer.model.external_weight},
            )
            trainer.clip_grad_norm = MagicMock(
                side_effect=lambda grad=trainer.model.weight.main_grad, **kwargs: grad.item(),
            )
            metrics = {"foundation_loss": 1200.0, "diagnostic": 7.0}
            self.assertEqual(trainer.prepare_optimizer_step(1200.0, metrics), (2.0, 2.0))
            self.assertEqual(metrics, {"foundation_loss": 2.0, "diagnostic": 7.0})
            trainer.clip_grad_norm.assert_called_once_with(gradient_divisor=600)
            torch.testing.assert_close(trainer.model.external_weight.grad, torch.tensor([1200.]))
            self.assertIs(trainer.model.weight.grad, ordinary_grad)
            if ordinary_grad is not None:
                torch.testing.assert_close(ordinary_grad, torch.tensor([[99.]], dtype=torch.bfloat16))


    def test_host_gradients_normalize_before_joint_clipping(self) -> None:
        """Feature: External gradient normalization.

        Description: Normalize dense and sparse gradients together, then apply joint clipping.
        Expectation: Normalization precedes the joint norm/clipping and each gradient is scaled once.
        """
        table = HostEngramTable(
            source_weight=torch.nn.Parameter(torch.empty((2, 1), device="meta")), logical_rows=2,
        )
        table.bind_planned_shard(ep_rank=0, ep_size=1)
        table.weight = torch.nn.Parameter(torch.zeros((2, 1)))
        model = torch.nn.Module()
        model.register_parameter("dense", torch.nn.Parameter(torch.zeros(1)))
        model.add_module("embed", table)
        model.dense.grad = torch.tensor([12.])
        for value in (4., 12.):
            table.append_rows(torch.tensor([0]), torch.tensor([[value]]))
        external = DeepseekV41HostState(model, {"embed": table}, None)
        external.optimizer = SimpleNamespace(state="ACCUMULATING")
        model._hp_external_state = external  # pylint: disable=protected-access
        trainer = base_module.BaseTrainer.__new__(base_module.BaseTrainer)
        trainer.model, trainer.device = model, torch.device("cpu")
        trainer.mesh = SimpleNamespace(dp_cp_mesh=None)
        trainer.config = SimpleNamespace(training=SimpleNamespace(max_grad_norm=2.5))
        trainer.model_integration = MagicMock()
        trainer.fsdp_runtime_diagnostics = None
        trainer.loss_names = ("foundation_loss",)
        trainer.begin_loss_step()
        trainer.step_token_counts = {"foundation_tokens": torch.tensor(4)}
        metrics = {"foundation_loss": 28., "diagnostic": 9.}
        loss, norm = trainer.prepare_optimizer_step(28., metrics)
        self.assertEqual(loss, 7.)
        self.assertEqual(metrics, {"foundation_loss": 7., "diagnostic": 9.})
        torch.testing.assert_close(norm, torch.tensor(5.))
        scale = (2.5 / (5. + 1.e-6)) / 4
        torch.testing.assert_close(model.dense.grad, torch.tensor([12. * scale]))
        torch.testing.assert_close(table.pending.values(), torch.tensor([[16. * scale]]))


if __name__ == "__main__":
    unittest.main()
