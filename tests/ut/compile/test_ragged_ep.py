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
"""CPU regressions for graph EP mesh and parameter contracts."""

import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from hyper_parallel.compile import GraphCompiler, PassConfig
from hyper_parallel.distributed.expert_parallel.experts import bind_local_expert_forward
from hyper_parallel.distributed.mesh import MeshContext


class _LocalExperts(nn.Module):
    """Keep actual variable-size expert computation without a process group."""

    def __init__(self) -> None:
        """Bind two ordinary stacked experts."""
        super().__init__()
        self.experts = nn.Module()
        self.experts.num_experts = 2
        self.experts.gate_up_proj = nn.Parameter(torch.randn(2, 6, 4))
        self.experts.down_proj = nn.Parameter(torch.randn(2, 4, 3))
        bind_local_expert_forward(self, 1)

    def forward(self, hidden: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """Run each input row through its selected local expert."""
        return self.experts(hidden, ids)


def _loss(model, hidden, ids):
    """Expose outputs alongside the scalar backward objective."""
    output = model(hidden, ids)
    return output.square().sum(), {"output": output.detach()}


class TestRaggedExperts(unittest.TestCase):
    """Reject unsupported meshes and retain exact expert ownership."""

    def test_compile_rejects_unsupported_ep_topologies(self):
        """Reject extra parallel axes and overlap before tracing or communication."""
        for axis in ("tp_size", "cp_size", "pp_size"):
            context = MeshContext(ep_size=2, **{axis: 2})
            compiler = GraphCompiler(_LocalExperts(), _loss, PassConfig(enable_overlap=False), mesh_context=context)
            with self.subTest(axis=axis), self.assertRaisesRegex(ValueError, "TP=CP=PP=1"):
                compiler.compile()
        compiler = GraphCompiler(_LocalExperts(), _loss, PassConfig(enable_overlap=True),
                                 mesh_context=MeshContext(ep_size=2))
        with self.assertRaisesRegex(ValueError, "enable_overlap=False"):
            compiler.compile()

    def test_compile_resolves_expert_parameter_aliases(self):
        """Pass exact expert FQNs, including tied aliases, to the FSDP pass."""
        model = _LocalExperts()
        model.alias = model.experts
        dense, expert = MagicMock(), MagicMock()
        dense.mesh_dim_names = ("fsdp_shard",)
        expert.mesh_dim_names = ("edp_shard", "ep")
        context = MeshContext(ep_size=2, fsdp_non_moe_mesh=dense, fsdp_moe_mesh=expert)
        compiler = GraphCompiler(model, _loss, PassConfig(enable_overlap=False), mesh_context=context)
        joint = MagicMock(input_guards=None)
        with (
            patch("hyper_parallel.compile.compiler.dist.is_initialized", return_value=True),
            patch("hyper_parallel.compile.compiler._register_process_group") as register,
            patch("hyper_parallel.compile.compiler.trace_model_graph", return_value=joint),
            patch("hyper_parallel.compile.compiler.PassPipeline.from_config") as pipeline,
        ):
            compiler.compile()
        self.assertEqual(register.call_args.args, ("fsdp_expert", expert["edp_shard"].get_group()))
        names = pipeline.return_value.run.call_args.kwargs["expert_param_names"]
        self.assertEqual(names, {"experts.gate_up_proj", "experts.down_proj", "alias.gate_up_proj", "alias.down_proj"})

    def test_compile_rejects_missing_or_replicated_expert_mesh(self):
        """Fail before tracing when the expert DP contract cannot be satisfied."""
        for case in ("missing_mesh", "replication", "unbound_model"):
            model = nn.Linear(4, 4) if case == "unbound_model" else _LocalExperts()
            dense, expert = MagicMock(), MagicMock()
            dense.mesh_dim_names = ("fsdp_shard",)
            expert.mesh_dim_names = ("edp_shard", "edp_replicate", "ep") if case == "replication" else ("edp_shard",)
            expert["edp_replicate"].size.return_value = 2
            context = MeshContext(ep_size=2, fsdp_non_moe_mesh=dense,
                                  fsdp_moe_mesh=None if case == "missing_mesh" else expert)
            compiler = GraphCompiler(model, _loss, PassConfig(enable_overlap=False), mesh_context=context)
            with (
                self.subTest(case=case),
                patch("hyper_parallel.compile.compiler.dist.is_initialized", return_value=True),
                patch("hyper_parallel.compile.compiler._register_process_group"),
                self.assertRaisesRegex(ValueError, "Graph EP"),
            ):
                compiler.compile()
