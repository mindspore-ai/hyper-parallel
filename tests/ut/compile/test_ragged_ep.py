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

    def test_compile_rejects_ep_with_pipeline_parallelism(self):
        """Pipeline stages bring cross-stage P2P that graph EP cannot schedule yet."""
        context = MeshContext(ep_size=2, pp_size=2)
        compiler = GraphCompiler(
            _LocalExperts(),
            _loss,
            PassConfig(enable_overlap=False),
            mesh_context=context,
        )
        with self.assertRaisesRegex(ValueError, "PP=1"):
            compiler.compile()

    def test_compile_allows_ep_with_context_parallelism(self):
        """CP folds into the expert DP domain and the gradient normalization."""
        dense, expert = MagicMock(), MagicMock()
        dense.mesh_dim_names = ("fsdp_shard",)
        dense["fsdp_shard"].size.return_value = 4
        expert.mesh_dim_names = ("edp_shard", "ep")
        expert["edp_shard"].size.return_value = 2
        context = MeshContext(
            ep_size=2,
            cp_size=2,
            dp_size=4,
            fsdp_non_moe_mesh=dense,
            fsdp_moe_mesh=expert,
        )
        compiler = GraphCompiler(
            _LocalExperts(),
            _loss,
            PassConfig(enable_overlap=False),
            mesh_context=context,
        )
        joint = MagicMock(input_guards=None)
        with (
            patch(
                "hyper_parallel.compile.compiler.dist.is_initialized", return_value=True
            ),
            patch("hyper_parallel.compile.compiler._register_process_group"),
            patch(
                "hyper_parallel.compile.compiler.trace_model_graph", return_value=joint
            ),
            patch("hyper_parallel.compile.compiler.PassPipeline.from_config"),
        ):
            compiler.compile()
        self.assertEqual(compiler.pass_config.fsdp_degree, 4)
        self.assertEqual(compiler._expert_shard_degree, 2)
        reductions = compiler._build_pass_kwargs()["gradient_reductions"]
        self.assertTrue(reductions)
        self.assertTrue(all(policy.divisor == 8 for policy in reductions.values()))

    def test_compile_allows_enable_overlap_with_ep(self):
        """The overlap flag does not block graph EP while AutoOverlap is a no-op."""
        dense, expert = MagicMock(), MagicMock()
        dense.mesh_dim_names = ("fsdp_shard",)
        expert.mesh_dim_names = ("edp_shard", "ep")
        expert["edp_shard"].size.return_value = 2
        context = MeshContext(ep_size=2, fsdp_non_moe_mesh=dense, fsdp_moe_mesh=expert)
        compiler = GraphCompiler(_LocalExperts(), _loss, PassConfig(enable_overlap=True), mesh_context=context)
        joint = MagicMock(input_guards=None)
        with (
            patch("hyper_parallel.compile.compiler.dist.is_initialized", return_value=True),
            patch("hyper_parallel.compile.compiler._register_process_group"),
            patch("hyper_parallel.compile.compiler.trace_model_graph", return_value=joint),
            patch("hyper_parallel.compile.compiler.PassPipeline.from_config") as pipeline,
        ):
            compiler.compile()
        pipeline.return_value.run.assert_called_once()

    def test_compile_resolves_expert_parameter_aliases(self):
        """Pass exact expert FQNs, including tied aliases, to the FSDP pass."""
        model = _LocalExperts()
        model.alias = model.experts
        dense, expert = MagicMock(), MagicMock()
        dense.mesh_dim_names = ("fsdp_shard",)
        expert.mesh_dim_names = ("edp_shard", "ep")
        expert["edp_shard"].size.return_value = 2
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
        plan = pipeline.return_value.run.call_args.kwargs["fsdp_plan"]
        self.assertEqual(
            set(plan.params),
            {"experts.gate_up_proj", "experts.down_proj", "alias.gate_up_proj", "alias.down_proj"},
        )
        self.assertEqual({spec.group_name for spec in plan.params.values()}, {"fsdp_expert"})
        self.assertEqual({spec.degree for spec in plan.params.values()}, {2})

    def test_compile_rejects_missing_expert_mesh_and_unbound_model(self):
        """Fail before tracing when the expert DP contract cannot be satisfied."""
        for case in ("missing_mesh", "unbound_model"):
            model = nn.Linear(4, 4) if case == "unbound_model" else _LocalExperts()
            dense, expert = MagicMock(), MagicMock()
            dense.mesh_dim_names = ("fsdp_shard",)
            expert.mesh_dim_names = ("edp_shard",)
            context = MeshContext(
                ep_size=2,
                fsdp_non_moe_mesh=dense,
                fsdp_moe_mesh=None if case == "missing_mesh" else expert,
            )
            compiler = GraphCompiler(model, _loss, PassConfig(enable_overlap=False), mesh_context=context)
            with (
                self.subTest(case=case),
                patch("hyper_parallel.compile.compiler.dist.is_initialized", return_value=True),
                patch("hyper_parallel.compile.compiler._register_process_group"),
                self.assertRaisesRegex(ValueError, "Graph EP"),
            ):
                compiler.compile()

    def test_compile_composes_dense_hsdp_with_expert_edp_replication(self):
        """Dense HSDP and expert EDP replication use independent replicate groups."""
        model = _LocalExperts()
        dense, expert = MagicMock(), MagicMock()
        dense.mesh_dim_names = ("fsdp_replicate", "fsdp_shard")
        rep_mesh, dense_shard = MagicMock(), MagicMock()
        rep_mesh.size.return_value = 2
        dense_shard.size.return_value = 2
        dense.__getitem__.side_effect = lambda key: {
            "fsdp_replicate": rep_mesh,
            "fsdp_shard": dense_shard,
        }.get(key, MagicMock())
        expert.mesh_dim_names = ("edp_replicate", "edp_shard", "ep")
        expert_rep, expert_shard = MagicMock(), MagicMock()
        expert_rep.size.return_value = 2
        expert_shard.size.return_value = 2
        expert.__getitem__.side_effect = lambda key: {
            "edp_replicate": expert_rep,
            "edp_shard": expert_shard,
        }.get(key, MagicMock())
        context = MeshContext(
            ep_size=2,
            fsdp_non_moe_mesh=dense,
            fsdp_moe_mesh=expert,
            dp_size=4,
            dp_replicate_size=2,
            dp_shard_size=2,
        )
        compiler = GraphCompiler(
            model, _loss, PassConfig(enable_overlap=False), mesh_context=context
        )
        joint = MagicMock(input_guards=None)
        with (
            patch(
                "hyper_parallel.compile.compiler.dist.is_initialized", return_value=True
            ),
            patch(
                "hyper_parallel.compile.compiler._register_process_group"
            ) as register,
            patch(
                "hyper_parallel.compile.compiler.trace_model_graph", return_value=joint
            ),
            patch(
                "hyper_parallel.compile.compiler.PassPipeline.from_config"
            ) as pipeline,
        ):
            compiler.compile()
        self.assertEqual(compiler.pass_config.dp_mode, "hsdp")
        self.assertEqual(compiler.pass_config.dp_replicate_degree, 2)
        registered = {call.args[0] for call in register.call_args_list}
        self.assertTrue(
            {"dp_replicate", "fsdp", "fsdp_expert", "fsdp_expert_replicate"}
            <= registered
        )
        plan = pipeline.return_value.run.call_args.kwargs["fsdp_plan"]
        specs = [
            spec for name, spec in plan.params.items() if name.startswith("experts.")
        ]
        self.assertTrue(specs)
        for spec in specs:
            self.assertEqual(spec.group_name, "fsdp_expert")
            self.assertEqual(spec.degree, 2)
            self.assertEqual(spec.replicate_group_name, "fsdp_expert_replicate")
            self.assertEqual(spec.replicate_degree, 2)

    def test_compile_supports_expert_data_parallel_replication(self):
        """Expert EDP replication reduces on its own group, not the dense one."""
        for shard_degree in (1, 2):
            with self.subTest(edp_shard=shard_degree):
                model = _LocalExperts()
                dense, expert = MagicMock(), MagicMock()
                dense.mesh_dim_names = ("fsdp_shard",)
                expert.mesh_dim_names = ("edp_replicate", "edp_shard", "ep")
                shard_mesh, replicate_mesh = MagicMock(), MagicMock()
                shard_mesh.size.return_value = shard_degree
                replicate_mesh.size.return_value = 2
                expert.__getitem__.side_effect = lambda key, shard=shard_mesh, replicate=replicate_mesh: {
                    "edp_shard": shard,
                    "edp_replicate": replicate,
                }.get(key, MagicMock())
                context = MeshContext(
                    ep_size=2, fsdp_non_moe_mesh=dense, fsdp_moe_mesh=expert
                )
                compiler = GraphCompiler(
                    model, _loss, PassConfig(enable_overlap=False), mesh_context=context
                )
                joint = MagicMock(input_guards=None)
                with (
                    patch(
                        "hyper_parallel.compile.compiler.dist.is_initialized",
                        return_value=True,
                    ),
                    patch(
                        "hyper_parallel.compile.compiler._register_process_group"
                    ) as register,
                    patch(
                        "hyper_parallel.compile.compiler.trace_model_graph",
                        return_value=joint,
                    ),
                    patch(
                        "hyper_parallel.compile.compiler.PassPipeline.from_config"
                    ) as pipeline,
                ):
                    compiler.compile()
                registered = {call.args[0] for call in register.call_args_list}
                self.assertIn("fsdp_expert", registered)
                self.assertIn("fsdp_expert_replicate", registered)
                plan = pipeline.return_value.run.call_args.kwargs["fsdp_plan"]
                specs = [
                    spec
                    for name, spec in plan.params.items()
                    if name.startswith("experts.")
                ]
                self.assertTrue(specs)
                for spec in specs:
                    self.assertEqual(spec.degree, shard_degree)
                    self.assertEqual(
                        spec.mode, "replicate" if shard_degree == 1 else "shard"
                    )
                    self.assertEqual(spec.replicate_group_name, "fsdp_expert_replicate")
                    self.assertEqual(spec.replicate_degree, 2)
                    self.assertEqual(spec.pad_policy, "pad")
