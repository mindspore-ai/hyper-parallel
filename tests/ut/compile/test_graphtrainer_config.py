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
"""Unit tests for graph-mode integration helpers."""

import math
import os
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

import torch

from hyper_parallel.compile.compiler import GraphCompiler
from hyper_parallel.compile.pass_config import PassConfig, build_pass_config_from_trainer_config
from hyper_parallel.compile.trainer import GraphTrainer
from hyper_parallel.core.dtensor.device_mesh import clear_device_mesh_cache, init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.layout import infer_slice_area_by_layout
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard, StridedShard
from hyper_parallel.core.fully_shard.hsdp_param import HSDPParamV2
from hyper_parallel.core.fully_shard.utils import FSDPMeshInfo
from hyper_parallel.distributed._builder.fsdp_adapter import FSDP2Manager
from hyper_parallel.distributed.compile import _resolve_compile_config
from hyper_parallel.distributed.mesh import MeshContext
from hyper_parallel.models._transformers.model_builder import instantiate_infrastructure
from hyper_parallel.models.build_options import CompileConfig
from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.config import (
    AcceleratorConfig,
    OptimizerConfig,
    Target,
    TrainerConfig,
)
from hyper_parallel.trainer.runtime.model_integration import DisabledModelIntegrationSession


def _build_model(*, distributed_setup, width):
    """Return model construction inputs for config assertions."""
    return distributed_setup, width


def _build_optimizer(*, params):
    """Minimal optimizer target unused by these tests."""
    return params


def _make_kw_model() -> torch.nn.Module:
    """Tiny module following the ``input_ids``/``use_cache`` call protocol."""

    class _KWModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)

        def forward(self, input_ids, use_cache=False):
            return self.linear(input_ids)

    return _KWModel()


class TestGraphModeIntegration(unittest.TestCase):
    """Test graph-mode selection without the old target-wrapping bridge."""

    def test_basetrainer_builds_graph_compiler_only_for_joint_graph(self):
        """BaseTrainer should own compiler creation without affecting eager mode."""
        base = BaseTrainer.__new__(BaseTrainer)
        base.config = SimpleNamespace(
            compile=CompileConfig(enabled=True, use_joint_graph=True),
            accelerator=SimpleNamespace(
                tp_size=1,
                sequence_parallel=False,
                loss_parallel=False,
            ),
            fsdp_config=SimpleNamespace(dp_shard_size=1, edp_shard_size=1),
        )
        base.model = torch.nn.Linear(2, 2)
        source_info = {"weight": ((), None)}
        base.model._source_shard_info = source_info
        base.device = torch.device("cpu")
        base.mesh = SimpleNamespace()
        base.model_fwd_context = nullcontext()
        base.loss_fn = lambda model_output, labels: (
            (model_output - labels) ** 2
        ).mean()
        base.graph_compiler = None

        base._build_graph_compiler()

        self.assertIsInstance(base.graph_compiler, GraphCompiler)
        self.assertEqual(base.graph_compiler.train_fn.__self__, base)
        self.assertIs(base.graph_compiler.source_shard_info, source_info)

        base.config.compile = CompileConfig(enabled=True, use_joint_graph=False)
        base.graph_compiler = None
        base._build_graph_compiler()
        self.assertIsNone(base.graph_compiler)

    def test_basetrainer_graph_train_fn_uses_postforward(self):
        """The graph wrapper should return postforward's (loss, loss_dict)."""
        base = BaseTrainer.__new__(BaseTrainer)
        base.current_token_counts = {"foundation_tokens": torch.tensor(4)}
        base.step_token_counts = {"foundation_tokens": torch.tensor(8)}
        base.model_fwd_context = nullcontext()
        joint_loss = torch.tensor(2.0)
        joint_loss_dict = {"main": joint_loss.clone()}
        base.postforward = Mock(
            return_value=(joint_loss, joint_loss_dict)
        )
        model = Mock(return_value=torch.tensor([1.0]))
        labels = torch.tensor([0.0])

        loss, loss_dict = base._graph_train_fn(
            model,
            model_inputs={"input_ids": torch.tensor([1])},
            labels=labels,
            current_token_counts=base.current_token_counts,
            step_token_counts=base.step_token_counts,
        )

        model.assert_called_once_with(input_ids=torch.tensor([1]), use_cache=False)
        base.postforward.assert_called_once_with(
            model.return_value, labels,
            current_token_counts=base.current_token_counts,
            step_token_counts=base.step_token_counts,
        )
        # Both the backward loss and the named losses travel to the tracer:
        # the loss_dict values are emitted as extra graph outputs.
        self.assertIs(loss, joint_loss)
        self.assertIs(loss_dict, joint_loss_dict)

    def test_forward_backward_step_runs_joint_graph_when_compiler_present(self):
        """A configured compiler replaces the eager forward/backward path."""
        base = BaseTrainer.__new__(BaseTrainer)
        base.current_token_counts = {"foundation_tokens": torch.tensor(4)}
        base.step_token_counts = {"foundation_tokens": torch.tensor(8)}
        base.config = SimpleNamespace(
            training=SimpleNamespace(empty_cache_before_backward=False),
        )
        base.device = torch.device("cpu")
        base.state = None
        base.mesh = None
        base.model_integration = DisabledModelIntegrationSession()
        base.model_fwd_context = nullcontext()
        base.graph_compiler = Mock()
        joint_loss = torch.tensor(3.5)
        joint_loss_dict = {"foundation_loss": joint_loss.detach()}
        base.graph_compiler.forward_backward.return_value = (
            joint_loss,
            joint_loss_dict,
        )
        micro_batch = {"input_ids": torch.randn(2, 4)}
        labels = torch.randn(2, 4)

        loss, loss_dict = base.forward_backward_step(
            micro_batch,
            loss_inputs={"labels": labels},
        )

        base.graph_compiler.forward_backward.assert_called_once_with(
            model_inputs=micro_batch,
            labels=labels,
            current_token_counts=base.current_token_counts,
            step_token_counts=base.step_token_counts,
        )
        self.assertIs(loss, joint_loss)
        # The named losses flow back from the graph as extra outputs.
        self.assertIs(loss_dict, joint_loss_dict)

    def test_forward_backward_step_runs_joint_graph_end_to_end(self):
        """The BaseTrainer-captured joint graph should execute a real step."""
        base = BaseTrainer.__new__(BaseTrainer)
        base.current_token_counts = {"foundation_tokens": torch.tensor(4)}
        base.step_token_counts = {"foundation_tokens": torch.tensor(8)}
        base.config = SimpleNamespace(
            training=SimpleNamespace(empty_cache_before_backward=False),
        )
        base.device = torch.device("cpu")
        base.state = None
        base.mesh = None
        base.model_integration = DisabledModelIntegrationSession()
        base.model_fwd_context = nullcontext()
        base.model = _make_kw_model()
        base.loss_fn = lambda model_output, labels: (
            (model_output - labels) ** 2
        ).mean()
        # ``_graph_train_fn`` routes through ``postforward``; the plain
        # local loss keeps this UT dist-free (no token counting / mesh).
        base.postforward = lambda outputs, labels, *, current_token_counts, step_token_counts: (
            base.loss_fn(model_output=outputs, labels=labels),
            {"foundation_loss": base.loss_fn(model_output=outputs, labels=labels).detach()},
        )
        base.graph_compiler = GraphCompiler(
            model=base.model,
            train_fn=base._graph_train_fn,
            device=torch.device("cpu"),
        )

        micro_batch = {"input_ids": torch.randn(2, 4)}
        labels = torch.randn(2, 4)

        loss, loss_dict = base.forward_backward_step(
            micro_batch,
            loss_inputs={"labels": labels},
        )

        self.assertIsInstance(loss, torch.Tensor)
        self.assertIn("foundation_loss", loss_dict)
        self.assertIsNotNone(base.model.linear.weight.grad)
        self.assertGreater(
            float(base.model.linear.weight.grad.abs().sum()),
            0.0,
            "a real joint-graph backward must fill parameter grads",
        )

    def test_forward_backward_step_warns_on_empty_cache_before_backward(self):
        """Graph mode must warn when empty_cache_before_backward is requested."""
        base = BaseTrainer.__new__(BaseTrainer)
        base.current_token_counts = {"foundation_tokens": torch.tensor(4)}
        base.step_token_counts = {"foundation_tokens": torch.tensor(8)}
        base.config = SimpleNamespace(
            training=SimpleNamespace(empty_cache_before_backward=True),
        )
        base.device = torch.device("cpu")
        base.state = None
        base.mesh = None
        base.model_integration = DisabledModelIntegrationSession()
        base.model_fwd_context = nullcontext()
        base.graph_compiler = Mock()
        base.graph_compiler.forward_backward.return_value = (torch.tensor(1.0), {})
        micro_batch = {"input_ids": torch.randn(2, 4)}

        with self.assertLogs("hyper_parallel.trainer.base", level="WARNING") as captured:
            base.forward_backward_step(micro_batch, loss_inputs={"labels": None})

        self.assertTrue(
            any("empty_cache_before_backward" in line for line in captured.output),
            "graph mode should warn that empty_cache_before_backward is ignored",
        )
        base.graph_compiler.forward_backward.assert_called_once()

    def test_graphcompiler_infers_pass_config_from_trainer_config(self):
        """GraphCompiler should infer pass config from trainer topology when omitted."""
        trainer_config = SimpleNamespace(
            accelerator=SimpleNamespace(
                tp_size=2,
                sequence_parallel=True,
                loss_parallel=False,
            ),
            fsdp_config=SimpleNamespace(dp_shard_size=4, edp_shard_size=1),
        )

        compiler = GraphCompiler(
            model=torch.nn.Linear(2, 2),
            train_fn=lambda model, x, y: ((model(x) - y) ** 2).mean(),
            trainer_config=trainer_config,
            device=torch.device("cpu"),
        )

        self.assertTrue(compiler.pass_config.fsdp_enabled)
        self.assertEqual(compiler.pass_config.fsdp_degree, 4)

    def test_graphcompiler_prefers_explicit_pass_config(self):
        """An explicit pass config should override trainer topology inference."""
        trainer_config = SimpleNamespace(
            accelerator=SimpleNamespace(
                tp_size=2,
                sequence_parallel=True,
                loss_parallel=True,
            ),
            fsdp_config=SimpleNamespace(dp_shard_size=4, edp_shard_size=1),
        )
        pass_config = PassConfig(fsdp_enabled=False)

        compiler = GraphCompiler(
            model=torch.nn.Linear(2, 2),
            train_fn=lambda model, x, y: ((model(x) - y) ** 2).mean(),
            pass_config=pass_config,
            trainer_config=trainer_config,
            device=torch.device("cpu"),
        )

        self.assertIs(compiler.pass_config, pass_config)

    def test_graphtrainer_infers_pass_config_from_trainer_config(self):
        """GraphTrainer should infer pass config from trainer topology when omitted."""
        trainer_config = SimpleNamespace(
            accelerator=SimpleNamespace(
                tp_size=2,
                sequence_parallel=True,
                loss_parallel=False,
            ),
            fsdp_config=SimpleNamespace(dp_shard_size=4, edp_shard_size=1),
        )

        trainer = GraphTrainer(
            model=torch.nn.Linear(2, 2),
            train_fn=lambda model, x, y: ((model(x) - y) ** 2).mean(),
            trainer_config=trainer_config,
            device=torch.device("cpu"),
        )

        self.assertTrue(trainer.pass_config.fsdp_enabled)
        self.assertEqual(trainer.pass_config.fsdp_degree, 4)

    def test_use_joint_graph_skips_eager_fsdp2_instantiation(self):
        """Graph-mode selection should keep mesh planning but skip eager FSDP2."""
        distributed_setup = SimpleNamespace(
            mesh_context="mesh",
            strategy_config=object(),
            fp32_main_params=False,
        )

        with patch(
            "hyper_parallel.models._transformers.model_builder._instantiate_fsdp2"
        ) as mock_instantiate:
            sharding_planner, fsdp_manager = instantiate_infrastructure(
                distributed_setup=distributed_setup,
                compile_config=CompileConfig(enabled=True, use_joint_graph=True),
            )

        self.assertIsNotNone(sharding_planner)
        self.assertIsNone(fsdp_manager)
        mock_instantiate.assert_not_called()

    def test_non_graph_mode_preserves_eager_fsdp2_instantiation(self):
        """Non-graph builds should still instantiate eager FSDP2 as before."""
        distributed_setup = SimpleNamespace(
            mesh_context="mesh",
            strategy_config=object(),
            fp32_main_params=False,
        )
        fsdp_manager = SimpleNamespace(config=SimpleNamespace())

        with patch(
            "hyper_parallel.models._transformers.model_builder._instantiate_fsdp2",
            return_value=fsdp_manager,
        ) as mock_instantiate:
            _, built_manager = instantiate_infrastructure(
                distributed_setup=distributed_setup,
                compile_config=CompileConfig(enabled=True, use_joint_graph=False),
            )

        self.assertIs(built_manager, fsdp_manager)
        mock_instantiate.assert_called_once_with(
            config=distributed_setup.strategy_config,
            mesh_context=distributed_setup.mesh_context,
            fp32_main_params=False,
        )

    def test_use_joint_graph_skips_eager_decoder_compile(self):
        """Graph-mode compile selection should not trigger eager layer compile."""
        compile_config, compile_for_execution = _resolve_compile_config(
            CompileConfig(
                enabled=True,
                use_joint_graph=True,
                mode="reduce-overhead",
            ),
            validate_placement=False,
            fsdp2_manager=None,
        )

        self.assertIsNotNone(compile_config)
        self.assertTrue(compile_config.selects_graph_compiler())
        self.assertFalse(
            compile_for_execution,
            f"graph mode should skip eager decoder compile: {compile_config!r}",
        )

    def test_graph_mode_allows_pipeline_parallel_topology(self):
        """Graph trainer selection should bypass the eager compile PP guard."""
        model_target = Target(
            _target_=_build_model,
            target_path="tests.ut.compile.test_graphtrainer_config._build_model",
            width=16,
        )
        optimizer_target = Target(
            _target_=_build_optimizer,
            target_path="tests.ut.compile.test_graphtrainer_config._build_optimizer",
        )

        config = TrainerConfig(
            model=model_target,
            optimizer=OptimizerConfig(target=optimizer_target),
            accelerator=AcceleratorConfig(pp_size=2),
            compile=CompileConfig(enabled=True, use_joint_graph=True),
        )

        self.assertTrue(config.compile.selects_graph_compiler())


    def test_build_pass_config_projects_parallel_topology(self):
        """Trainer topology and inferred FSDP intent are copied to PassConfig."""
        config = SimpleNamespace(
            accelerator=SimpleNamespace(
                tp_size=2,
                sequence_parallel=True,
                loss_parallel=True,
            ),
            fsdp_config=SimpleNamespace(dp_shard_size=4, edp_shard_size=1),
        )

        pass_config = build_pass_config_from_trainer_config(
            config,
            enable_overlap=True,
        )

        self.assertTrue(
            pass_config.fsdp_enabled,
            f"FSDP should be inferred as enabled: got={pass_config!r}",
        )
        self.assertEqual(
            pass_config.fsdp_degree,
            4,
            f"FSDP degree mismatch: expected=4, got={pass_config.fsdp_degree}",
        )
        self.assertTrue(
            pass_config.enable_overlap,
            f"overlap flag mismatch: expected=True, got={pass_config.enable_overlap}",
        )


class TestGraphParameterLayout(unittest.TestCase):
    """Graph-local storage metadata preserves source order and padding."""

    def setUp(self) -> None:
        """Keep synthetic CPU meshes independent across cases."""
        clear_device_mesh_cache()
        self.addCleanup(clear_device_mesh_cache)

    @staticmethod
    def _mesh(shape, names):
        """Construct a metadata-only CPU mesh without a process group."""
        return init_device_mesh("cpu", shape, mesh_dim_names=names,
                                rank_list=tuple(range(math.prod(shape))), init_backend=False)

    def test_same_dim_shards_preserve_source_first_order(self):
        """Graph metadata preserves source coordinates without mutating them."""
        mesh = self._mesh((2, 4), ("fsdp_shard", "tp"))
        source = [Shard(0)]
        parameter = torch.nn.Parameter(torch.zeros(2, 3))
        layout = GraphCompiler._build_parameter_layout(
            parameter, mesh["fsdp_shard"], (source, mesh["tp"]),
            sharded=True, unsharded_shape=(4, 3),
        )
        self.assertEqual(layout.tensor_shape, (16, 3))
        self.assertEqual(layout.tensor_stride, (3, 1))
        self.assertEqual(layout.tensor_dtype, torch.float32)
        self.assertEqual(layout.placements, (StridedShard(0, split_factor=4), Shard(0)))
        self.assertEqual(source, [Shard(0)])

    def test_eager_placement_input_and_noncontiguous_stride_contract(self):
        """Restored eager methods retain list mutation and logical strides."""
        mesh = self._mesh((2, 4), ("dp", "tp"))
        state = object.__new__(HSDPParamV2)
        state.mesh_info = object.__new__(FSDPMeshInfo)
        state._spmd_mesh = mesh
        state._spmd_shard_mesh_dim = 0
        placements = [Replicate(), Shard(1)]
        actual = state._apply_data_parallel_placements(placements, Shard(1))
        expected = (StridedShard(1, split_factor=4), Shard(1))
        self.assertEqual(actual, expected)
        self.assertEqual(tuple(placements), expected)
        state._spmd_placements = actual
        source = DTensor.from_local(torch.zeros(4, 3).t(), mesh, (Replicate(), Shard(1)),
                                    shape=(3, 16), stride=(1, 3))
        layout = state._build_sharding_spec(source, source.to_local())
        self.assertEqual(layout.tensor_shape, (3, 16))
        self.assertEqual(layout.tensor_stride, (1, 3))
        self.assertEqual(layout.placements, expected)

    def test_distinct_dims_and_replication_do_not_add_strided_shards(self):
        """Retain ordinary sharding when the source splits another dimension."""
        mesh = self._mesh((2, 4), ("fsdp_shard", "tp"))
        parameter = torch.nn.Parameter(torch.zeros(2, 3))
        for source in (Replicate(), Shard(1)):
            with self.subTest(source=source):
                layout = GraphCompiler._build_parameter_layout(
                    parameter, mesh["fsdp_shard"], ((source,), mesh["tp"]),
                    sharded=True, unsharded_shape=(4, 3),
                )
                self.assertEqual(layout.placements, (Shard(0), source))

    def test_graph_padding_is_applied_before_restoring_source_global_shape(self):
        """Describe equal physical chunks after padding and source sharding."""
        mesh = self._mesh((2, 2), ("fsdp_shard", "tp"))
        parameter = torch.nn.Parameter(torch.zeros(2, 4))
        layout = GraphCompiler._build_parameter_layout(
            parameter, mesh["fsdp_shard"], ((Shard(0),), mesh["tp"]),
            sharded=True, unsharded_shape=(3, 4),
        )
        self.assertEqual(layout.tensor_shape, (8, 4))
        for rank in range(4):
            area = infer_slice_area_by_layout(layout, rank, layout.tensor_shape)
            self.assertEqual(tuple(end - start for start, end in area), (2, 4))

    def test_replicated_scalar_and_invalid_shard_axis(self):
        """Keep scalars replicated and reject invalid physical declarations."""
        mesh = self._mesh((2,), ("fsdp_replicate",))
        scalar = torch.nn.Parameter(torch.tensor(1.))
        layout = GraphCompiler._build_parameter_layout(scalar, mesh, None, sharded=False, unsharded_shape=())
        self.assertEqual(layout.tensor_shape, ())
        self.assertEqual(layout.placements, (Replicate(),))
        for parameter in (scalar, torch.nn.Parameter(torch.ones(2))):
            with self.subTest(shape=parameter.shape), self.assertRaisesRegex(ValueError, "shard mesh axis"):
                GraphCompiler._build_parameter_layout(parameter, mesh, None,
                                                      sharded=True, unsharded_shape=parameter.shape)

    def test_unit_mesh_selection_dense_expert_and_world_fallback(self):
        """Original manager methods preserve dense, expert and world rules."""
        dense = self._mesh((2, 2, 2), ("fsdp_replicate", "fsdp_shard", "tp"))
        expert = self._mesh((1, 2, 4), ("edp_replicate", "edp_shard", "ep"))
        manager = object.__new__(FSDP2Manager)
        manager.mesh_context = MeshContext(fsdp_non_moe_mesh=dense, fsdp_moe_mesh=expert, dp_replicate_size=2)
        self.assertEqual(manager._build_fsdp_actual_mesh().mesh_dim_names, ("fsdp_replicate", "fsdp_shard"))
        self.assertEqual(manager._build_fsdp_actual_mesh(True).mesh_dim_names, ("edp_replicate", "edp_shard"))
        world = self._mesh((2, 2, 2), ("dp_replicate", "dp_shard", "cp"))
        manager.mesh_context = MeshContext(device_mesh=world)
        fallback = manager._build_fsdp_actual_mesh()
        self.assertEqual(fallback.mesh_dim_names, ("dp_replicate", "fsdp_shard"))
        self.assertEqual(fallback.mesh_shape, (2, 4))
        manager.mesh_context = MeshContext(fsdp_non_moe_mesh=dense)
        with self.assertRaisesRegex(ValueError, "fsdp_moe_mesh"):
            manager._build_fsdp_actual_mesh(True)
        manager.mesh_context = MeshContext()
        with self.assertRaisesRegex(ValueError, "fsdp_non_moe_mesh"):
            manager._build_fsdp_actual_mesh()


if __name__ == "__main__":
    unittest.main()
