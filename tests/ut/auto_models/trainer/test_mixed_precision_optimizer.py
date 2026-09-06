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
"""Unit tests for Megatron-aligned fp32 main parameter optimization."""
# pylint: disable=wrong-import-position,abstract-method

import copy
import logging
import os
import tempfile
import unittest
from typing import Any
from unittest.mock import patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

# Snapshot logging.Logger attributes before importing optimizer modules, whose
# imports patch rank-aware helpers onto logging.Logger.
_LOGGING_HELPER_NAMES = (
    "info_rank0",
    "warning_rank0",
    "debug_rank0",
    "info_once",
    "warning_once",
)
_LOGGER_ATTR_SENTINEL = object()
_LOGGER_ATTR_SNAPSHOT = {
    helper_name: getattr(logging.Logger, helper_name, _LOGGER_ATTR_SENTINEL)
    for helper_name in _LOGGING_HELPER_NAMES
}

import torch  # pylint: disable=wrong-import-position
from torch import nn  # pylint: disable=wrong-import-position

from hyper_parallel import DeviceMesh, DTensor, Replicate
from hyper_parallel.components.optim.mixed_precision_optimizer import (
    Float16OptimizerWithFloat16Params,
)
from hyper_parallel.components.optim.builders import Muon
from hyper_parallel.components.checkpoint.dcp_checkpointer import (
    DistributedCheckpointer,
    initialize_optimizer_state,
)
from hyper_parallel.core.optimizer.adamw import AdamW as CoreAdamW
from hyper_parallel.core.optimizer.muon import Muon as CoreMuon
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer
from hyper_parallel.core.distributed_checkpoint import (
    save as dcp_save,
)
from tests.common.mark_utils import arg_mark


def tearDownModule() -> None:  # pylint: disable=invalid-name
    """Restore logging.Logger attributes patched by optimizer module imports."""
    for helper_name, original in _LOGGER_ATTR_SNAPSHOT.items():
        if original is _LOGGER_ATTR_SENTINEL:
            if hasattr(logging.Logger, helper_name):
                delattr(logging.Logger, helper_name)
        else:
            setattr(logging.Logger, helper_name, original)


class _MixedDtypeModel(nn.Module):
    """Model containing one low-precision and one native-fp32 parameter."""

    def __init__(self) -> None:
        """Register low-precision and native-fp32 parameters."""
        super().__init__()
        self.low = nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.bfloat16))
        self.fp32 = nn.Parameter(torch.tensor([3.0], dtype=torch.float32))


class _TwoLowPrecisionModel(nn.Module):
    """Model with two low-precision parameters for partial-state validation."""

    def __init__(self) -> None:
        """Register two independently checkpointed parameters."""
        super().__init__()
        self.first = nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
        self.second = nn.Parameter(torch.full((2,), 2.0, dtype=torch.bfloat16))


def _build_optimizer() -> tuple[
        _MixedDtypeModel,
        Float16OptimizerWithFloat16Params,
]:
    """Build a production AdamW chain behind the mixed-precision wrapper."""
    model = _MixedDtypeModel()
    for name, parameter in model.named_parameters():
        parameter.model_name = name
        parameter.main_grad = None
    leaf_optimizer = CoreAdamW(
        [{"params": list(model.parameters()), "lr": 0.25}],
        lr=0.25,
    )
    chained_optimizer = ChainedOptimizer(
        model,
        {"adamw": leaf_optimizer},
    )
    wrapper = Float16OptimizerWithFloat16Params(chained_optimizer, model)
    return model, wrapper


def _optimizer_parameters(optimizer: Any) -> list[nn.Parameter]:
    """Return parameters routed through an optimizer's groups."""
    return [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]


def _build_two_parameter_optimizer() -> Float16OptimizerWithFloat16Params:
    """Build a wrapper with two distinct fp32 main-parameter state keys."""
    model = _TwoLowPrecisionModel()
    optimizer = ChainedOptimizer(
        model,
        {"adamw": CoreAdamW(model.parameters(), lr=0.01)},
    )
    return Float16OptimizerWithFloat16Params(optimizer, model)


def _muon_grouping_parameters(muon_optimizer: Any) -> list[nn.Parameter]:
    """Return parameters referenced by Muon's HSDP grouping cache."""
    # pylint: disable=protected-access
    return [
        parameter
        for no_comm_params, hsdp_groups in muon_optimizer._hsdp_grouping.values()
        for parameter in (
            list(no_comm_params)
            + [record.param for group in hsdp_groups for record in group.records]
        )
    ]


def _muon_assignment_parameters(muon_optimizer: Any) -> list[nn.Parameter]:
    """Return parameters referenced by Muon's HSDP assignment cache."""
    # pylint: disable=protected-access
    return [
        parameter
        for assignment in muon_optimizer._hsdp_assignment_batches.values()
        for parameter in assignment["no_comm"]
    ]


def _muon_step_parameters(muon_optimizer: Any) -> list[nn.Parameter]:
    """Return parameters referenced by Muon's unsharded step cache."""
    return [
        parameter
        for parameters in muon_optimizer.unshard_params_by_group.values()
        for parameter in parameters
    ]


class TestFloat16OptimizerWithFloat16Params(unittest.TestCase):
    """Cover group routing, gradient movement, copy-back, reset, and DCP state."""

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_initialize_hyper_optimizer_state_without_step(self):
        """Materialize AdamW and Muon checkpoint state without optimizer updates.

        Feature: Lazy optimizer-state checkpoint restore.
        Description: Initialize a chained Hyper AdamW/Muon optimizer before DCP load.
        Expectation: Required zero state exists without calling either optimizer step.
        """
        model = nn.Module()
        model.adamw_param = nn.Parameter(torch.tensor([1.0, 2.0]))
        model.muon_param = nn.Parameter(torch.arange(4.0).reshape(2, 2))
        adamw = CoreAdamW([model.adamw_param], lr=0.1)
        muon = CoreMuon([model.muon_param], lr=0.1)
        optimizer = ChainedOptimizer(model, {"adamw": adamw, "muon": muon})
        expected_adamw = model.adamw_param.detach().clone()
        expected_muon = model.muon_param.detach().clone()

        with patch.object(adamw, "step", side_effect=AssertionError("AdamW step called")), \
                patch.object(muon, "step", side_effect=AssertionError("Muon step called")):
            self.assertTrue(initialize_optimizer_state(optimizer))

        self.assertEqual(adamw.param_groups[0]["step"], 0)
        self.assertEqual(muon.param_groups[0]["step"], 0)
        self.assertEqual(set(adamw.state[model.adamw_param]), {"exp_avg", "exp_avg_sq"})
        self.assertEqual(set(muon.state[model.muon_param]), {"momentum_buffer"})
        self.assertTrue(torch.equal(adamw.state[model.adamw_param]["exp_avg"], torch.zeros(2)))
        self.assertTrue(torch.equal(muon.state[model.muon_param]["momentum_buffer"], torch.zeros(2, 2)))
        self.assertTrue(torch.equal(model.adamw_param, expected_adamw))
        self.assertTrue(torch.equal(model.muon_param, expected_muon))

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_groups_separate_low_precision_and_native_fp32_params(self):
        """Separate low-precision and native fp32 parameters.

        Feature: Mixed-precision main-parameter grouping.
        Description: Route bfloat16 and fp32 parameters through the wrapper.
        Expectation: Only the bfloat16 parameter receives a distinct main copy.
        """
        model, optimizer = _build_optimizer()

        self.assertEqual(optimizer.float16_groups, [[model.low]])
        self.assertEqual(
            optimizer.fp32_from_float16_groups,
            [[model.low.main_param]],
        )
        self.assertEqual(optimizer.fp32_from_fp32_groups, [[model.fp32]])
        self.assertIsNot(model.low.main_param, model.low)
        self.assertEqual(model.low.main_param.dtype, torch.float32)
        self.assertIs(model.fp32.main_param, model.fp32)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    @patch("hyper_parallel.core.dtensor.device_mesh.platform.get_rank", return_value=0)
    def test_main_param_preserves_dtensor_layout_and_optimizer_metadata(
            self,
            mock_get_rank,
    ):
        """Preserve distributed layout and optimizer metadata.

        Feature: Distributed fp32 main-parameter construction.
        Description: Clone a bfloat16 distributed parameter for an optimizer.
        Expectation: Layout and model FQN metadata are retained.
        """
        del mock_get_rank
        mesh = DeviceMesh(
            "cpu",
            [0],
            mesh_dim_names=("dp",),
            _init_backend=False,
        )
        model = nn.Module()
        parameter = nn.Parameter(
            DTensor.from_local(
                torch.ones(2, dtype=torch.bfloat16),
                mesh,
                (Replicate(),),
            )
        )
        parameter.model_name = "weight"
        model.register_parameter("weight", parameter)

        leaf_optimizer = CoreAdamW([parameter], lr=0.1)
        optimizer = Float16OptimizerWithFloat16Params(
            ChainedOptimizer(model, {"adamw": leaf_optimizer}),
            model,
        )
        main_param = optimizer.fp32_from_float16_groups[0][0]

        self.assertIsInstance(main_param, DTensor)
        self.assertIs(main_param.device_mesh, mesh)
        self.assertEqual(tuple(main_param.placements), (Replicate(),))
        self.assertEqual(main_param.model_name, "weight")
        self.assertFalse(hasattr(main_param, "is_muon"))
        self.assertIs(optimizer.param_groups[0]["params"][0], main_param)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_prepare_step_copy_back_and_zero_grad(self):
        """Exercise gradient preparation, update, copy-back, and reset.

        Feature: Mixed-precision optimizer step lifecycle.
        Description: Apply main gradients through a leaf optimizer.
        Expectation: Main params update, model params copy back, and grads clear.
        """
        model, optimizer = _build_optimizer()
        original_model_low = model.low.detach().clone()  # pylint: disable=not-callable
        original_main_low = model.low.main_param.detach().clone()
        model.low.main_grad = torch.tensor([0.5, -0.25], dtype=torch.float32)
        model.fp32.main_grad = torch.tensor([1.0], dtype=torch.float32)

        optimizer.prepare_grads()
        optimizer.optimizer.step()

        torch.testing.assert_close(model.low, original_model_low)
        self.assertFalse(torch.equal(model.low.main_param, original_main_low))
        optimizer._copy_main_params_to_model_params()  # pylint: disable=protected-access
        torch.testing.assert_close(
            model.low.float(),
            model.low.main_param.to(dtype=torch.bfloat16).float(),
        )

        optimizer.zero_grad()
        self.assertIsNone(model.low.grad)
        self.assertIsNone(model.low.main_grad)
        self.assertIsNone(model.low.main_param.grad)
        self.assertIsNone(model.fp32.grad)
        self.assertIsNone(model.fp32.main_grad)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_reload_model_params_uses_current_model_value(self):
        """Reload main parameters from model storage.

        Feature: Model-only mixed-precision restoration.
        Description: Change a model parameter while its main copy is stale.
        Expectation: Reload refreshes the fp32 main parameter from the model.
        """
        model, optimizer = _build_optimizer()
        with torch.no_grad():
            model.low.fill_(2.5)
            model.low.main_param.zero_()

        optimizer.reload_model_params()

        torch.testing.assert_close(
            model.low.main_param,
            model.low.float(),
        )

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_explicit_fp32_main_param_state_restores_unrepresentable_value(self):
        """Restore main-parameter values beyond bfloat16 precision.

        Feature: Fp32-main optimizer checkpoint state.
        Description: Save fp32 main values that cannot round-trip through bfloat16.
        Expectation: Loading restores the exact checkpointed fp32 values.
        """
        model, optimizer = _build_optimizer()
        with torch.no_grad():
            model.low.main_param.copy_(
                torch.tensor([1.0001, 2.0003], dtype=torch.float32)
            )
        state_dict = copy.deepcopy(optimizer.state_dict())
        expected = state_dict["_mixed_precision_optimizer"][
            "fp32_from_fp16_params"
        ]["low"].clone()
        with torch.no_grad():
            model.low.fill_(0)
            model.low.main_param.zero_()

        optimizer.load_state_dict(state_dict)

        self.assertTrue(torch.equal(model.low.main_param, expected))
        self.assertFalse(torch.equal(model.low.main_param, model.low.float()))

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    @patch("hyper_parallel.core.distributed_checkpoint.api.platform.barrier")
    @patch(
        "hyper_parallel.core.distributed_checkpoint.api.platform.get_world_size",
        return_value=1,
    )
    @patch(
        "hyper_parallel.core.distributed_checkpoint.api.platform.get_rank",
        return_value=0,
    )
    def test_dcp_round_trip_restores_explicit_fp32_main_param(
            self,
            mock_get_rank,
            mock_get_world_size,
            mock_barrier,
    ):
        """Round-trip fp32 main weights through the DCP tensor format.

        Feature: Mixed-precision optimizer DCP state.
        Description: Persist a value that cannot round-trip through bfloat16.
        Expectation: Metadata exposes the main-param branch and DCP restores it exactly.
        """
        del mock_get_rank, mock_get_world_size, mock_barrier
        _, source_optimizer = _build_optimizer()
        with torch.no_grad():
            source_optimizer.fp32_from_float16_groups[0][0].copy_(
                torch.tensor([1.0001, 2.0003], dtype=torch.float32)
            )
        expected = source_optimizer.fp32_from_float16_groups[0][0].clone()

        with tempfile.TemporaryDirectory() as checkpoint_path:
            dcp_save(
                {"optimizer": source_optimizer.state_dict()},
                checkpoint_id=checkpoint_path,
                no_dist=True,
            )
            _, restored_optimizer = _build_optimizer()
            load_state = {"optimizer": restored_optimizer.state_dict()}
            DistributedCheckpointer().load(
                checkpoint_path,
                load_state,
            )

        restored_optimizer.load_state_dict(load_state["optimizer"])
        restored_main_param = restored_optimizer.fp32_from_float16_groups[0][0]
        self.assertTrue(torch.equal(restored_main_param, expected))

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    @patch("hyper_parallel.core.distributed_checkpoint.api.platform.barrier")
    @patch(
        "hyper_parallel.core.distributed_checkpoint.api.platform.get_world_size",
        return_value=1,
    )
    @patch(
        "hyper_parallel.core.distributed_checkpoint.api.platform.get_rank",
        return_value=0,
    )
    def test_dcp_ordinary_optimizer_state_reloads_main_params(
            self,
            mock_get_rank,
            mock_get_world_size,
            mock_barrier,
    ):
        """Load an ordinary optimizer checkpoint into the fp32 wrapper.

        Feature: Coexisting optimizer checkpoint formats.
        Description: Restore a checkpoint without the mixed-precision subtree.
        Expectation: Inner state loads and main params reload from model values.
        """
        del mock_get_rank, mock_get_world_size, mock_barrier
        source_model = _MixedDtypeModel()
        with torch.no_grad():
            source_model.low.fill_(4.5)
        source_optimizer = ChainedOptimizer(
            source_model,
            {"adamw": CoreAdamW(source_model.parameters(), lr=0.01)},
        )

        with tempfile.TemporaryDirectory() as checkpoint_path:
            dcp_save(
                {
                    "model": source_model.state_dict(),
                    "optimizer": source_optimizer.state_dict(),
                },
                checkpoint_id=checkpoint_path,
                no_dist=True,
            )
            restored_model, restored_optimizer = _build_optimizer()
            load_state = {
                "model": restored_model.state_dict(),
                "optimizer": restored_optimizer.state_dict(),
            }
            DistributedCheckpointer().load(checkpoint_path, load_state)

        self.assertNotIn("_mixed_precision_optimizer", load_state["optimizer"])
        restored_model.load_state_dict(load_state["model"])
        with self.assertLogs(
                "hyper_parallel.components.optim.mixed_precision_optimizer",
                level="WARNING",
        ):
            restored_optimizer.load_state_dict(load_state["optimizer"])
        torch.testing.assert_close(
            restored_model.low.main_param,
            restored_model.low.float(),
        )

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    @patch("hyper_parallel.core.distributed_checkpoint.api.platform.barrier")
    @patch(
        "hyper_parallel.core.distributed_checkpoint.api.platform.get_world_size",
        return_value=1,
    )
    @patch(
        "hyper_parallel.core.distributed_checkpoint.api.platform.get_rank",
        return_value=0,
    )
    def test_dcp_partial_main_param_state_fails(
            self,
            mock_get_rank,
            mock_get_world_size,
            mock_barrier,
    ):
        """Reject a checkpoint containing only part of the main-param subtree.

        Feature: Mixed-precision checkpoint integrity.
        Description: Remove one main-param FQN while retaining another.
        Expectation: Load planning reports the checkpoint as incomplete.
        """
        del mock_get_rank, mock_get_world_size, mock_barrier
        source_optimizer = _build_two_parameter_optimizer()
        source_state = source_optimizer.state_dict()
        source_main_params = source_state["_mixed_precision_optimizer"][
            "fp32_from_fp16_params"
        ]
        source_main_params.pop("second")

        with tempfile.TemporaryDirectory() as checkpoint_path:
            dcp_save(
                {"optimizer": source_state},
                checkpoint_id=checkpoint_path,
                no_dist=True,
            )
            restored_optimizer = _build_two_parameter_optimizer()
            load_state = {"optimizer": restored_optimizer.state_dict()}
            with self.assertRaisesRegex(RuntimeError, "checkpoint is incomplete"):
                DistributedCheckpointer().load(checkpoint_path, load_state)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_ordinary_optimizer_state_has_no_main_param_branch(self):
        """Keep the ordinary optimizer checkpoint path structurally independent.

        Feature: Coexisting optimizer checkpoint branches.
        Description: Build the same model without the fp32 main-param wrapper.
        Expectation: Its state contains no mixed-precision optimizer subtree.
        """
        model = _MixedDtypeModel()
        optimizer = ChainedOptimizer(
            model,
            {"adamw": CoreAdamW(model.parameters(), lr=0.01)},
        )

        self.assertNotIn("_mixed_precision_optimizer", optimizer.state_dict())

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_ordinary_state_reloads_main_params_and_corrupt_state_fails(self):
        """Reload ordinary state and reject corrupted main-parameter state.

        Feature: Mixed-precision checkpoint validation.
        Description: Load state without the mixed subtree and with an incomplete subtree.
        Expectation: Ordinary state reloads from the model and incomplete state fails.
        """
        model, optimizer = _build_optimizer()
        with torch.no_grad():
            model.low.fill_(2.5)
            model.low.main_param.zero_()
        missing_state = optimizer.state_dict()
        missing_state.pop("_mixed_precision_optimizer")

        with self.assertLogs(
                "hyper_parallel.components.optim.mixed_precision_optimizer",
                level="WARNING",
        ) as logs:
            optimizer.load_state_dict(missing_state)
        torch.testing.assert_close(model.low.main_param, model.low.float())
        self.assertIn("cannot be recovered", "\n".join(logs.output))

        corrupt_state = optimizer.state_dict()
        corrupt_state["_mixed_precision_optimizer"][
            "fp32_from_fp16_params"
        ] = {}
        with self.assertRaisesRegex(RuntimeError, "missing fp32 main"):
            optimizer.load_state_dict(corrupt_state)

        invalid_state = optimizer.state_dict()
        invalid_state["_mixed_precision_optimizer"] = None
        with self.assertRaisesRegex(RuntimeError, "must be a mapping"):
            optimizer.load_state_dict(invalid_state)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_native_fp32_only_wrapper_has_empty_main_param_state(self):
        """Allow an empty main-param state for a native-fp32 model.

        Feature: Mixed-precision checkpoint validation.
        Description: Wrap an optimizer whose model parameters are already fp32.
        Expectation: An empty main-param subtree needs no DCP tensor metadata.
        """
        model = nn.Linear(2, 2, bias=False)
        optimizer = Float16OptimizerWithFloat16Params(
            ChainedOptimizer(
                model,
                {"adamw": CoreAdamW(model.parameters(), lr=0.01)},
            ),
            model,
        )

        state_dict = optimizer.state_dict()
        optimizer.load_state_dict(state_dict)

        self.assertEqual(
            state_dict["_mixed_precision_optimizer"]["fp32_from_fp16_params"],
            {},
        )

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_chained_optimizer_uses_model_fqns_for_main_param_state(self):
        """Use model FQNs for main-parameter optimizer state.

        Feature: Chained optimizer checkpoint naming.
        Description: Create Adam state for fp32 main parameters.
        Expectation: Moment state remains keyed by original model FQNs.
        """
        model = _MixedDtypeModel()
        adam = CoreAdamW(model.parameters(), lr=0.01)
        optimizer = Float16OptimizerWithFloat16Params(
            ChainedOptimizer(model, {"adamw": adam}),
            model,
        )
        model.low.main_grad = torch.ones_like(model.low, dtype=torch.float32)
        model.fp32.main_grad = torch.ones_like(model.fp32)
        optimizer.step()
        optimizer.zero_grad()

        state_dict = optimizer.state_dict()

        self.assertEqual(set(state_dict["state"]), {"low", "fp32"})
        self.assertEqual(set(state_dict["state"]["low"]), {"exp_avg", "exp_avg_sq"})
        self.assertEqual(state_dict["param_groups"][0]["step"], 1)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    @patch("hyper_parallel.core.distributed_checkpoint.api.platform.barrier")
    @patch(
        "hyper_parallel.core.distributed_checkpoint.api.platform.get_world_size",
        return_value=1,
    )
    @patch(
        "hyper_parallel.core.distributed_checkpoint.api.platform.get_rank",
        return_value=0,
    )
    def test_adam_moments_muon_momentum_and_main_params_round_trip(
            self,
            mock_get_rank,
            mock_get_world_size,
            mock_barrier,
    ):
        """Round-trip both optimizer families and fp32 main weights.

        Feature: Mixed AdamW and Muon checkpoint state.
        Description: Save moments, momentum buffers, and main parameters together.
        Expectation: Every optimizer and main-parameter value is restored.
        """

        del mock_get_rank, mock_get_world_size, mock_barrier

        class _SplitModel(nn.Module):
            def __init__(self) -> None:
                """Register parameters assigned to separate optimizer families."""
                super().__init__()
                self.adam_weight = nn.Parameter(
                    torch.ones(2, dtype=torch.bfloat16)
                )
                self.muon_weight = nn.Parameter(
                    torch.ones((2, 2), dtype=torch.bfloat16)
                )

        model = _SplitModel()
        for name, parameter in model.named_parameters():
            parameter.model_name = name
            parameter.main_grad = None
        expected_adam_config = {
            "lr": 0.0123,
            "betas": (0.71, 0.82),
            "eps": 3.0e-7,
            "weight_decay": 0.045,
        }
        expected_muon_config = {
            "lr": 0.234,
            "momentum": (0.67, 0.67),
            "weight_decay": 0.056,
            "ns_steps": 3,
            "nesterov": False,
        }
        adam = CoreAdamW(
            [model.adam_weight],
            **expected_adam_config,
        )
        muon_state_owner = CoreMuon(
            [model.muon_weight],
            **expected_muon_config,
        )
        chained = ChainedOptimizer(
            model,
            {"adamw": adam, "muon": muon_state_owner},
            flatten=True,
        )
        optimizer = Float16OptimizerWithFloat16Params(chained, model)
        for parameter in model.parameters():
            parameter.main_grad = torch.ones_like(
                parameter,
                dtype=torch.float32,
            )
        optimizer.step()
        optimizer.zero_grad()
        state_dict = copy.deepcopy(optimizer.state_dict())

        self.assertIn("state.adam_weight.exp_avg", state_dict)
        self.assertIn("state.adam_weight.exp_avg_sq", state_dict)
        self.assertIn("state.muon_weight.momentum_buffer", state_dict)
        for config_name, expected_value in expected_adam_config.items():
            with self.subTest(optimizer="adamw", config_name=config_name):
                self.assertEqual(
                    state_dict[f"param_groups.adam_weight.{config_name}"],
                    expected_value,
                )
        for config_name, expected_value in expected_muon_config.items():
            with self.subTest(optimizer="muon", config_name=config_name):
                self.assertEqual(
                    state_dict[f"param_groups.muon_weight.{config_name}"],
                    expected_value,
                )
        self.assertNotIn("param_groups.adam_weight.momentum", state_dict)
        self.assertNotIn("param_groups.muon_weight.betas", state_dict)
        expected_main_params = copy.deepcopy(
            state_dict["_mixed_precision_optimizer"][
                "fp32_from_fp16_params"
            ]
        )
        for leaf_optimizer in (adam, muon_state_owner):
            for parameter_state in leaf_optimizer.state.values():
                for state_value in parameter_state.values():
                    if isinstance(state_value, torch.Tensor):
                        state_value.zero_()
        adam.param_groups[0].update({
            "lr": 9.1,
            "betas": (0.11, 0.22),
            "eps": 8.0e-4,
            "weight_decay": 0.91,
        })
        muon_state_owner.param_groups[0].update({
            "lr": 8.2,
            "momentum": (0.19, 0.19),
            "weight_decay": 0.82,
            "ns_steps": 5,
            "nesterov": True,
        })
        with torch.no_grad():
            for main_param in optimizer.model_param_by_optimizer_param:
                main_param.zero_()

        with tempfile.TemporaryDirectory() as checkpoint_path:
            dcp_save(
                {"optimizer": state_dict},
                checkpoint_id=checkpoint_path,
                no_dist=True,
            )
            load_state = {"optimizer": optimizer.state_dict()}
            DistributedCheckpointer().load(checkpoint_path, load_state)

        optimizer.load_state_dict(load_state["optimizer"])

        self.assertTrue(
            torch.equal(
                adam.state[model.adam_weight.main_param]["exp_avg"],
                state_dict["state.adam_weight.exp_avg"],
            )
        )
        self.assertTrue(
            torch.equal(
                muon_state_owner.state[model.muon_weight.main_param][
                    "momentum_buffer"
                ],
                state_dict["state.muon_weight.momentum_buffer"],
            )
        )
        for config_name, expected_value in expected_adam_config.items():
            with self.subTest(loaded_optimizer="adamw", config_name=config_name):
                self.assertEqual(
                    adam.param_groups[0][config_name],
                    expected_value,
                )
        for config_name, expected_value in expected_muon_config.items():
            with self.subTest(loaded_optimizer="muon", config_name=config_name):
                self.assertEqual(
                    muon_state_owner.param_groups[0][config_name],
                    expected_value,
                )
        for parameter_fqn, expected_main_param in expected_main_params.items():
            model_parameter = dict(model.named_parameters())[parameter_fqn]
            self.assertTrue(
                torch.equal(model_parameter.main_param, expected_main_param)
            )


class TestMuonMainParamConstruction(unittest.TestCase):
    """Muon must rebuild identity-based caches after wrapper replacement."""

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_muon_wrapper_rebuilds_all_parameter_caches(self):
        """Rebuild Muon caches after replacing model parameters.

        Feature: Muon mixed-precision optimizer construction.
        Description: Build ordinary Muon first, then apply the outer wrapper.
        Expectation: Leaf groups and all HSDP grouping caches reference main params.
        """
        model = nn.Sequential(nn.Linear(4, 4, bias=False)).to(torch.bfloat16)
        old_parameter_ids = {id(parameter) for parameter in model.parameters()}
        inner_optimizer = Muon(
            muon_config={},
            adamw_config={},
            model=model,
        ).get_optimizer()
        original_leaf_parameters = _optimizer_parameters(inner_optimizer)
        self.assertEqual(
            {id(parameter) for parameter in original_leaf_parameters},
            old_parameter_ids,
        )

        built = Float16OptimizerWithFloat16Params(inner_optimizer, model)
        routed_parameters = _optimizer_parameters(built)
        muon_optimizer = built.optimizers_dict["muon"]
        cached_parameters = _muon_grouping_parameters(muon_optimizer)
        assignment_parameters = _muon_assignment_parameters(muon_optimizer)
        step_parameters = _muon_step_parameters(muon_optimizer)
        self.assertTrue(routed_parameters)
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in routed_parameters))
        self.assertTrue(all(id(parameter) not in old_parameter_ids for parameter in routed_parameters))
        self.assertEqual(
            {id(parameter) for parameter in cached_parameters},
            {id(parameter) for parameter in routed_parameters},
        )
        self.assertEqual(
            {id(parameter) for parameter in assignment_parameters},
            {id(parameter) for parameter in routed_parameters},
        )
        self.assertEqual(
            {id(parameter) for parameter in step_parameters},
            {id(parameter) for parameter in routed_parameters},
        )


if __name__ == "__main__":
    unittest.main()
