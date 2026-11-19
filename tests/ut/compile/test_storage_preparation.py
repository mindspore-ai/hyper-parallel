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
"""Storage preparation is independent of lazy graph capture and optimizer state."""

from contextlib import contextmanager
from unittest import TestCase
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from hyper_parallel.compile.compiler import GraphCompiler
from hyper_parallel.compile.pass_config import PassConfig
from hyper_parallel.compile.passes.parallel.fsdp_pass import FSDPPass
from hyper_parallel.compile.tracer.graph_tracer import trace_model_graph
from hyper_parallel.components.optim.mixed_precision_optimizer import Float16OptimizerWithFloat16Params
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer


class _Model(nn.Module):
    """Include aliases, padding, a scalar, frozen state and a buffer."""

    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        """Build the state categories that preparation must distinguish."""
        super().__init__()
        self.weight = nn.Parameter(torch.arange(15, dtype=dtype).reshape(5, 3) / 16)
        self.alias = self.weight
        self.scale = nn.Parameter(torch.tensor(0.5, dtype=dtype))
        self.frozen = nn.Parameter(torch.ones(5, dtype=dtype), requires_grad=False)
        self.register_buffer("offset", torch.ones(5, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Read full logical tensors during forward."""
        return (x @ self.weight.T + self.offset) * self.scale + self.frozen


def _loss(model, *, x):
    """Use a scalar loss without labels."""
    return model(x).square().mean()


@contextmanager
def _distributed(compiler):
    """Mock transport setup; slicing, tracing and graph lowering remain real."""
    transport = MagicMock()
    transport.is_initialized.return_value = True
    transport.get_world_size.return_value = 2
    transport.get_rank.return_value = 0
    with (
        patch("hyper_parallel.compile.compiler.dist", transport),
        patch("hyper_parallel.compile.passes.parallel.fsdp_pass.dist", transport),
        patch("hyper_parallel.compile.passes.parallel.fsdp_pass._resolve_process_group"),
        patch.object(compiler, "_init_device_mesh"),
    ):
        yield


class TestStoragePreparation(TestCase):
    """Preparation owns physical slicing; the graph only lowers logical state."""

    def test_prepare_once_preserves_parameter_identity_and_nonsharded_state(self):
        """A second preparation neither re-shards nor changes loaded values."""
        model = _Model()
        compiler = GraphCompiler(model, _loss, PassConfig(fsdp_degree=2), device=torch.device("cpu"))
        original = torch.detach(model.weight).clone()
        identity = id(model.weight)
        with _distributed(compiler):
            compiler.prepare_model()
            self.assertEqual(compiler.parameter_shapes["weight"], (5, 3))
            self.assertEqual(compiler.sharded_parameter_names, {"weight"})
            self.assertEqual(model.weight.shape, (3, 3))
            torch.testing.assert_close(model.weight, original[:3], rtol=0, atol=0)
            with torch.no_grad():
                model.weight.add_(1)
            compiler.prepare_model()
        self.assertEqual(id(model.weight), identity)
        self.assertIs(model.alias, model.weight)
        self.assertEqual(model.scale.shape, ())
        self.assertEqual(model.frozen.shape, (5,))
        self.assertEqual(model.offset.shape, (5,))
        self.assertFalse(compiler.is_compiled)
        torch.testing.assert_close(model.weight, original[:3] + 1, rtol=0, atol=0)

    def test_logical_trace_and_pass_do_not_cut_prepared_storage_again(self):
        """Trace sees the original five rows while FSDP accepts the three-row shard."""
        model = _Model()
        compiler = GraphCompiler(model, _loss, PassConfig(fsdp_degree=2), device=torch.device("cpu"))
        with _distributed(compiler):
            compiler.prepare_model()
            shard = torch.detach(model.weight).clone()
            for dynamic in (False, True):
                with self.subTest(dynamic=dynamic):
                    joint = trace_model_graph(model, _loss, {"x": torch.ones(2, 3)},
                                              state_shapes=compiler.parameter_shapes, dynamic=dynamic)
                    self.assertEqual(joint.param_shapes["weight"], (5, 3))
                    self.assertEqual(joint.example_inputs[0].shape, (5, 3))
                    FSDPPass().run(joint.graph_module, compiler.pass_config, **compiler._build_pass_kwargs())
                    self.assertEqual(joint.graph_module.fsdp_sharded_param_names, {"weight"})
                    torch.testing.assert_close(model.weight, shard, rtol=0, atol=0)

    def test_fp32_main_parameters_are_built_from_prepared_shards(self):
        """BF16 storage does not leave full-size FP32 optimizer copies behind."""
        model = _Model(torch.bfloat16)
        compiler = GraphCompiler(model, _loss, PassConfig(fsdp_degree=2), device=torch.device("cpu"))
        with _distributed(compiler):
            compiler.prepare_model()
        raw = torch.optim.AdamW(model.parameters(), lr=0.01)
        optimizer = Float16OptimizerWithFloat16Params(ChainedOptimizer(model, {"adamw": raw}), model)
        for parameter, main in optimizer.optimizer_param_by_model_param.items():
            self.assertEqual(parameter.shape, main.shape)
            self.assertEqual(main.dtype, torch.float32)
        self.assertEqual(model.weight.main_param.shape, (3, 3))

    def test_failed_preparation_cannot_repeat_partial_slicing(self):
        """A partially prepared model must be rebuilt rather than silently reused."""
        compiler = GraphCompiler(_Model(), _loss, PassConfig(fsdp_degree=2), device=torch.device("cpu"))
        with _distributed(compiler), patch.object(FSDPPass, "prepare_model", side_effect=ValueError("slice failed")):
            with self.assertRaisesRegex(ValueError, "slice failed"):
                compiler.prepare_model()
            with self.assertRaisesRegex(RuntimeError, "rebuild the model"):
                compiler.prepare_model()

    def test_trace_failure_keeps_prepared_storage_and_logical_shapes(self):
        """A retry after capture failure cannot re-shard the live model."""
        model = _Model()
        compiler = GraphCompiler(model, _loss, PassConfig(fsdp_degree=2), device=torch.device("cpu"))
        with _distributed(compiler):
            compiler.prepare_model()
            shard = torch.detach(model.weight).clone()
            with patch("hyper_parallel.compile.compiler.trace_model_graph", side_effect=ValueError("trace failed")):
                for _ in range(2):
                    with self.assertRaisesRegex(ValueError, "trace failed"):
                        compiler.compile(x=torch.ones(2, 3))
            torch.testing.assert_close(model.weight, shard, rtol=0, atol=0)
            self.assertEqual(compiler.parameter_shapes["weight"], (5, 3))

    def test_replacing_prepared_parameter_requires_a_new_model(self):
        """Parameter identity is part of the prepared-storage contract."""
        model = _Model()
        compiler = GraphCompiler(model, _loss, PassConfig(fsdp_enabled=False))
        compiler.prepare_model()
        model.weight = nn.Parameter(torch.detach(model.weight).clone())
        with self.assertRaisesRegex(ValueError, "storage changed"):
            compiler.prepare_model()
