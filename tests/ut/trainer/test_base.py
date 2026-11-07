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

from hyper_parallel.models.build_options import CompileConfig
from hyper_parallel.trainer.config import OptimizerConfig, TrainerConfig
from tests.common.mark_utils import arg_mark

base_module = importlib.import_module("hyper_parallel.trainer.base")


class TestBaseTrainerFSDPRequirement(unittest.TestCase):
    """Require the FSDP parameter and gradient lifecycle before training."""

    def setUp(self) -> None:
        """Build the trainer boundary without loading a model or using a device."""
        config = TrainerConfig(model=MagicMock(), optimizer=OptimizerConfig(target=MagicMock()))
        self.trainer = MagicMock()
        self.trainer.config = config
        self.trainer.distributed_setup = SimpleNamespace(strategy_config=config.fsdp_config)
        self.trainer.global_rank = 1
        model = MagicMock(spec=base_module.torch.nn.Module)
        model.config = SimpleNamespace()
        self.trainer.config.model.build.return_value = model
        for target in (
                "_validate_optimize_dtype",
                "model_integration_runtime.build_model_integration_session",
        ):
            patcher = patch(f"hyper_parallel.trainer.base.{target}")
            patcher.start()
            self.addCleanup(patcher.stop)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_unwrapped_model_without_fsdp_strategy_fails(self) -> None:
        """Feature: Trainer FSDP/HSDP requirement.
        Description: Build an unwrapped model without an FSDP strategy.
        Expectation: Model construction raises ValueError regardless of the configured shard size.
        """
        self.trainer.config.fsdp_config.dp_shard_size = 8
        self.trainer.distributed_setup.strategy_config = None

        with self.assertRaisesRegex(ValueError, "requires FSDP/HSDP"):
            base_module.BaseTrainer._build_model(self.trainer)

        self.trainer.config.model.build.assert_called_once()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_strategy_without_actual_fsdp_wrapping_fails(self) -> None:
        """Feature: Trainer FSDP/HSDP requirement.
        Description: Build an unwrapped model with an FSDP strategy configured.
        Expectation: Model construction raises ValueError.
        """
        with self.assertRaisesRegex(ValueError, "Trainer requires FSDP/HSDP to be enabled"):
            base_module.BaseTrainer._build_model(self.trainer)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_empty_model_parts_fail(self) -> None:
        """Feature: Trainer FSDP/HSDP requirement.
        Description: Build a model exposing an empty parts list.
        Expectation: Model construction raises ValueError.
        """
        self.trainer.config.model.build.return_value.parts = []

        with self.assertRaisesRegex(ValueError, "Trainer requires FSDP/HSDP to be enabled"):
            base_module.BaseTrainer._build_model(self.trainer)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_unwrapped_pipeline_part_fails(self) -> None:
        """Feature: Trainer FSDP/HSDP requirement.
        Description: Build a model whose second part is unwrapped.
        Expectation: Model construction raises ValueError even when the first part is wrapped.
        """
        self.trainer.config.model.build.return_value.parts = [
            MagicMock(spec=base_module.HSDPModule),
            MagicMock(spec=base_module.torch.nn.Module),
        ]

        with self.assertRaisesRegex(ValueError, "Trainer requires FSDP/HSDP to be enabled"):
            base_module.BaseTrainer._build_model(self.trainer)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_wrapped_model_accepts_fsdp_shard_degree_one(self) -> None:
        """Feature: Trainer FSDP/HSDP requirement.
        Description: Build an FSDP-wrapped model with shard degree one.
        Expectation: Model construction succeeds and retains the wrapped model part.
        """
        model = MagicMock(spec=base_module.HSDPModule)
        model.config = SimpleNamespace()
        self.trainer.config.model.build.return_value = model

        base_module.BaseTrainer._build_model(self.trainer)

        self.assertEqual(
            self.trainer.hsdp_model_parts, [model],
            f"expected wrapped parts={[model]}, got={self.trainer.hsdp_model_parts}",
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_all_wrapped_pipeline_parts_are_accepted(self) -> None:
        """Feature: Trainer FSDP/HSDP requirement.
        Description: Build a model with two FSDP-wrapped parts.
        Expectation: Model construction succeeds and retains both wrapped parts.
        """
        parts = [MagicMock(spec=base_module.HSDPModule) for _ in range(2)]
        self.trainer.config.model.build.return_value.parts = parts

        base_module.BaseTrainer._build_model(self.trainer)

        self.assertEqual(
            self.trainer.hsdp_model_parts, parts,
            f"expected wrapped parts={parts}, got={self.trainer.hsdp_model_parts}",
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_joint_graph_uses_its_own_parallelism_contract(self) -> None:
        """Feature: Trainer joint-graph compatibility.
        Description: Build an unwrapped model with joint-graph compilation enabled.
        Expectation: Model construction succeeds without eager FSDP wrapping.
        """
        self.trainer.config.compile = CompileConfig(enabled=True, use_joint_graph=True)
        self.trainer.distributed_setup.strategy_config = None
        model = MagicMock(spec=base_module.torch.nn.Module)
        model.config = SimpleNamespace()
        self.trainer.config.model.build.return_value = model

        base_module.BaseTrainer._build_model(self.trainer)

        self.trainer.config.model.build.assert_called_once()
        self.assertEqual(
            self.trainer.hsdp_model_parts, [],
            f"expected no eager FSDP parts=[], got={self.trainer.hsdp_model_parts}",
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_decoder_compile_still_requires_fsdp(self) -> None:
        """Feature: Trainer FSDP/HSDP requirement.
        Description: Build an unwrapped model with ordinary decoder-layer compilation enabled.
        Expectation: Model construction raises ValueError.
        """
        self.trainer.config.compile = CompileConfig(enabled=True)
        self.trainer.distributed_setup.strategy_config = None

        with self.assertRaisesRegex(ValueError, "requires FSDP/HSDP"):
            base_module.BaseTrainer._build_model(self.trainer)

        self.trainer.config.model.build.assert_called_once()


class TestBaseTrainerCleanup(unittest.TestCase):
    """Tests for cleaning up the data iterator on exceptional exits."""

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

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="allcards",
        essential_mark="essential",
    )
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

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="allcards",
        essential_mark="essential",
    )
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
            del optimizer_config
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

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="allcards",
        essential_mark="essential",
    )
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

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="allcards",
        essential_mark="essential",
    )
    def test_prepare_optimizer_step_uses_external_norm(self) -> None:
        """
        Feature: Host external-state gradient clipping.
        Description: Run prepare_optimizer_step with external state present.
        Expectation: The external preparation drives after_clip and its global norm is returned.
        """
        trainer = MagicMock()
        trainer.config.training.max_grad_norm = 1.0
        trainer.fsdp_runtime_diagnostics = None
        external = MagicMock()
        preparation = SimpleNamespace(post_clip_norm=0.5, global_norm=2.0)
        external.prepare_optimizer_step.return_value = preparation
        with patch.object(base_module, "get_model_external_state", return_value=external):
            result = base_module.BaseTrainer.prepare_optimizer_step(trainer)
        self.assertEqual(result, 2.0)
        external.prepare_optimizer_step.assert_called_once_with(max_norm=1.0)
        trainer.model_integration.after_clip.assert_called_once_with(0.5)


if __name__ == "__main__":
    unittest.main()
