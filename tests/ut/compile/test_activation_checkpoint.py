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
"""Unit tests for static-graph activation checkpointing.

Covers tagging (full / selective / RNG / collectives), remat numerics and
duplicate shape, layer selection, FSDP composition, ``PassConfig``
validation, trainer-config projection, and the GraphCompiler end-to-end path.
"""

import copy
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Dict, Iterator, Tuple
from unittest.mock import MagicMock, patch

import torch
from torch import nn
from torch.utils.checkpoint import CheckpointPolicy, DefaultDeviceType

from hyper_parallel.compile import GraphCompiler
from hyper_parallel.compile.graph_parallel_plan import GraphParallelPlan
from hyper_parallel.compile.pass_config import (
    PassConfig,
    build_pass_config_from_trainer_config,
)
from hyper_parallel.compile.passes.activation_checkpoint import (
    ActivationCheckpointPass,
    _CheckpointScope,
    _GRAPH_AC_COMM_OPS,
    _GRAPH_FORWARD_ALL_GATHER_OPS,
    _remat,
    _tag_sac_policy,
    prepare_activation_checkpoint,
)
from hyper_parallel.compile.passes.parallel.fsdp_pass import (
    FSDPPass,
    _free_tensor_storage,
)
from hyper_parallel.compile.passes.pipeline import PassPipeline
from hyper_parallel.compile.tracer.graph_tracer import (
    extract_module_state,
    trace_model_graph,
    run_traced_graph,
)
from hyper_parallel.distributed.activation_checkpoint import (
    _apply_activation_checkpointing,
)
from hyper_parallel.trainer.config.parallelism import (
    FullCheckpointConfig,
    FullExcludeRegionConfig,
    SelectiveCheckpointConfig,
)

_DIST_PATH = "hyper_parallel.compile.passes.parallel.fsdp_pass.dist"
_RESOLVE_PG_PATH = (
    "hyper_parallel.compile.passes.parallel.fsdp_pass._resolve_process_group"
)
_DEGREE = 2


class _Block(nn.Module):
    """One checkpointable block: linear-relu-linear."""

    def __init__(self, dim: int) -> None:
        """Create the two linears of the block."""
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run linear-relu-linear."""
        return self.fc2(torch.relu(self.fc1(x)))


class _Model(nn.Module):
    """Two-block model with a head and the HF checkpointing marker."""

    def __init__(self, dim: int = 8, num_blocks: int = 2) -> None:
        """Register the repeated block container and the head."""
        super().__init__()
        self.gradient_checkpointing = False
        self.layers = nn.ModuleList(_Block(dim) for _ in range(num_blocks))
        self.head = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the blocks, then the head."""
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


class _DropModel(_Model):
    """Model whose blocks end in active dropout (RNG op)."""

    def __init__(self, dim: int = 8) -> None:
        """Replace the second block's submodules with a dropout block."""
        super().__init__(dim=dim, num_blocks=1)
        self.layers[0] = _DropBlock(dim)


class _AliasBlock(nn.Module):
    """Block that reads a view and then mutates the shared storage in place."""

    def __init__(self, dim: int) -> None:
        """Create the linear whose output is mutated after a view read."""
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Take a view, read it, then mutate the base through the same storage."""
        hidden = self.linear(x)
        view = hidden.view_as(hidden)
        before = view + 0.25
        hidden.add_(1.0)
        return before.sin() + hidden


class _AliasModel(nn.Module):
    """Alias-mutation block wrapped in the HF checkpointing discovery contract."""

    def __init__(self, dim: int = 8) -> None:
        """Register the block container and the head."""
        super().__init__()
        self.gradient_checkpointing = False
        self.layers = nn.ModuleList([_AliasBlock(dim)])
        self.head = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the alias block, then the head."""
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


class _DropBlock(nn.Module):
    """Block with an active dropout between the linears."""

    def __init__(self, dim: int) -> None:
        """Create the linears and the dropout."""
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run linear-dropout-linear."""
        return self.fc2(self.drop(torch.relu(self.fc1(x))))


def _train_fn(model: nn.Module, *, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """MSE loss of ``model(x)`` against ``y``."""
    return ((model(x) - y) ** 2).mean()


def _trace(model: nn.Module, x: torch.Tensor, y: torch.Tensor):
    """Trace ``model`` into a joint graph and return the JointGraph."""
    return trace_model_graph(model, _train_fn, {"x": x, "y": y})


def _run_ac(model: nn.Module, pass_config: PassConfig):
    """Trace + run the AC pass; return (joint, model)."""
    joint = _trace(model, torch.randn(4, 8), torch.randn(4, 8))
    ActivationCheckpointPass().run(joint.graph_module, pass_config, model=model)
    return joint, model


def _functional_op(name: str):
    """Return the default overload of a ``_c10d_functional`` op."""
    return getattr(getattr(torch.ops, "_c10d_functional"), name).default


def _duplicates(graph_module) -> list:
    """Return the recompute duplicate nodes of a transformed graph."""
    return [node for node in graph_module.graph.nodes if node.meta.get("ac_recomputed")]


def _forward_policy(graph_module, name: str):
    """Return the recompute policy of the forward node called ``name``."""
    for node in graph_module.graph.nodes:
        if node.name == name and not node.meta.get("autograd_backward", False):
            return node.meta.get("recompute")
    return None


def _forward_policies_under(graph_module, fqn: str) -> list:
    """Return the recompute policies of forward nodes under module ``fqn``."""
    policies = []
    for node in graph_module.graph.nodes:
        if node.op != "call_function" or node.meta.get("autograd_backward", False):
            continue
        stack = [
            value for value, _ in (node.meta.get("nn_module_stack") or {}).values()
        ]
        if any(value == fqn or value.startswith(f"{fqn}.") for value in stack):
            policies.append(node.meta.get("recompute"))
    return policies


def _selective_config(module_paths) -> PassConfig:
    """Build an eager-aligned selective config targeting whole block modules."""
    return PassConfig(
        ac_mode="selective",
        ac_selective_recompute_modules=list(module_paths),
        enable_overlap=False,
        fsdp_enabled=False,
    )


def _eager_reference(model: nn.Module, x: torch.Tensor, y: torch.Tensor):
    """Compute the eager loss and gradients from a fresh copy of ``model``."""
    reference = copy.deepcopy(model)
    loss = _train_fn(reference, x=x, y=y)
    grads = torch.autograd.grad(loss, list(reference.parameters()))
    return loss, grads


@contextmanager
def _patch_dist(world_size: int = _DEGREE, rank: int = 0) -> Iterator[MagicMock]:
    """Patch ``dist`` / ``_resolve_process_group`` inside ``fsdp_pass``."""
    mock_dist = MagicMock()
    mock_dist.is_initialized.return_value = True
    mock_dist.get_world_size.return_value = world_size
    mock_dist.get_rank.return_value = rank
    with (
        patch(_DIST_PATH, mock_dist),
        patch(_RESOLVE_PG_PATH, return_value=MagicMock()),
    ):
        yield mock_dist


class _SimulatedRank(torch.fx.Interpreter):
    """Execute a transformed graph as one rank with simulated collectives."""

    def __init__(
        self,
        graph_module: torch.fx.GraphModule,
        id_to_full: Dict[int, torch.Tensor],
        degree: int,
    ) -> None:
        """Store the full-parameter lookup and the simulated world size."""
        super().__init__(graph_module)
        self.id_to_full = id_to_full
        self.degree = degree

    def call_function(
        self, target: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        """Simulate the c10d collectives and execute the storage free."""
        name = str(target)
        if "all_gather_into_tensor" in name:
            return self.id_to_full[id(args[0])].clone()
        if "reduce_scatter_tensor" in name:
            return args[0].chunk(self.degree, dim=0)[0].clone()
        if "wait_tensor" in name:
            return args[0]
        if "_free_tensor_storage" in name:
            return _free_tensor_storage(*args, **kwargs)
        return super().call_function(target, args, kwargs)


class TestTagging(unittest.TestCase):
    """Policy tags respect the scope and the structural exceptions."""

    @classmethod
    def setUpClass(cls) -> None:
        """Fix the inputs used by every tagging test."""
        torch.manual_seed(0)
        cls.model = _Model()
        cls.config = PassConfig(
            ac_mode="full", enable_overlap=False, fsdp_enabled=False
        )

    def test_full_mode_tags_block_interior_and_saves_outside(self):
        """Block nodes are recompute candidates; head/loss nodes stay saved."""
        joint, _ = _run_ac(self.model, self.config)
        gm = joint.graph_module

        first_addmm = _forward_policy(gm, "addmm")
        self.assertEqual(first_addmm, CheckpointPolicy.MUST_RECOMPUTE)
        head_addmm = _forward_policy(gm, "addmm_4")
        self.assertIsNone(head_addmm, "the head must stay outside the AC scope")
        for node in gm.graph.nodes:
            if node.op == "call_function" and not node.meta.get("autograd_backward"):
                stack = [
                    fqn for fqn, _ in (node.meta.get("nn_module_stack") or {}).values()
                ]
                under_block = any(fqn.startswith("layers.") for fqn in stack)
                if not under_block:
                    self.assertNotIn(
                        "recompute",
                        node.meta,
                        f"node {node.name} outside the blocks must not be tagged",
                    )

    def test_selective_mode_recomputes_only_declared_targets(self):
        """``selective`` recomputes the declared block targets and saves the rest."""
        config = _selective_config(["layers.0"])
        joint, _ = _run_ac(self.model, config)
        gm = joint.graph_module

        inner = _forward_policies_under(gm, "layers.0.fc1")
        self.assertTrue(inner, "the target block must contribute forward nodes")
        self.assertTrue(
            all(policy == CheckpointPolicy.MUST_RECOMPUTE for policy in inner),
            "interior nodes of the target block must be recompute candidates",
        )
        for fqn in ("layers.1.fc1", "layers.1.fc2", "head"):
            policies = _forward_policies_under(gm, fqn)
            self.assertTrue(
                all(policy is None for policy in policies),
                f"non-target {fqn} must stay saved",
            )

    def test_selective_targets_resolve_to_module_fqns(self):
        """Layer-relative and model-root selective targets resolve to FQNs."""
        config = PassConfig(
            ac_mode="selective",
            ac_selective_recompute_regions=[("fc1", [0, 1])],
            ac_selective_recompute_modules=["head"],
            enable_overlap=False,
            fsdp_enabled=False,
        )
        scope = prepare_activation_checkpoint(self.model, config)
        self.assertEqual(
            scope.region_fqns,
            frozenset({"layers.0.fc1", "layers.1.fc1", "head"}),
        )

    def test_rng_op_is_saved_and_not_duplicated(self):
        """Dropout (seeded RNG) stays saved, so remat never replays it."""
        torch.manual_seed(1)
        model = _DropModel()
        config = PassConfig(ac_mode="full", enable_overlap=False, fsdp_enabled=False)
        joint, _ = _run_ac(model, config)
        gm = joint.graph_module

        rng_nodes = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function"
            and torch.Tag.nondeterministic_seeded in getattr(node.target, "tags", set())
        ]
        self.assertTrue(rng_nodes, "the traced dropout must expose a seeded op")
        for node in rng_nodes:
            self.assertEqual(node.meta.get("recompute"), CheckpointPolicy.MUST_SAVE)
        for duplicate in _duplicates(gm):
            self.assertNotIn(
                torch.Tag.nondeterministic_seeded,
                getattr(duplicate.target, "tags", set()),
                "RNG ops must never be duplicated",
            )

    def test_communication_ops_and_all_gather_switch(self):
        """Collectives are saved; the switch chooses all-gather save vs replay."""
        all_to_all = _functional_op("all_to_all_single")
        all_gather = _functional_op("all_gather_into_tensor")
        self.assertIn(all_to_all, _GRAPH_AC_COMM_OPS)
        self.assertIn(all_gather, _GRAPH_FORWARD_ALL_GATHER_OPS)

        cases = (
            (all_to_all, ([1], [1], "0"), None, CheckpointPolicy.MUST_SAVE),
            (all_gather, (1, "0"), True, CheckpointPolicy.MUST_SAVE),
            (all_gather, (1, "0"), False, CheckpointPolicy.MUST_RECOMPUTE),
        )
        scope = _CheckpointScope(
            region_fqns=frozenset({"layers.0"}), unit_fqns=frozenset({"layers.0"})
        )
        for target, args_tail, save, expected in cases:
            graph = torch.fx.Graph()
            placeholder = graph.placeholder("x")
            comm = graph.call_function(target, args=(placeholder, *args_tail))
            graph.output((comm,))
            comm.meta["nn_module_stack"] = {
                "": ("root", type(nn.Module)),
                "layers.0": ("layers.0", type(nn.Module)),
            }
            comm.meta["val"] = torch.randn(2, 2)
            gm = torch.fx.GraphModule(nn.Module(), graph)

            kwargs = {} if save is None else {"ac_save_forward_all_gather": save}
            _tag_sac_policy(gm, PassConfig(ac_mode="full", **kwargs), scope)

            self.assertEqual(comm.meta["recompute"], expected)

    def test_inplace_mutation_and_consumers_are_saved(self):
        """Mutable ops and their forward consumers are forced to MUST_SAVE."""
        graph = torch.fx.Graph()
        placeholder = graph.placeholder("x")
        mul_ = graph.call_function(
            torch.ops.aten.mul_.Tensor, args=(placeholder, placeholder)
        )
        relu = graph.call_function(torch.ops.aten.relu.default, args=(mul_,))
        graph.output((relu,))
        stack = {
            "": ("root", type(nn.Module)),
            "layers.0": ("layers.0", type(nn.Module)),
        }
        for node in (mul_, relu):
            node.meta["nn_module_stack"] = stack
            node.meta["val"] = torch.randn(2, 2)
        gm = torch.fx.GraphModule(nn.Module(), graph)

        scope = _CheckpointScope(
            region_fqns=frozenset({"layers.0"}), unit_fqns=frozenset({"layers.0"})
        )
        _tag_sac_policy(gm, PassConfig(ac_mode="full"), scope)

        self.assertEqual(mul_.meta["recompute"], CheckpointPolicy.MUST_SAVE)
        self.assertEqual(relu.meta["recompute"], CheckpointPolicy.MUST_SAVE)

    def test_readers_of_mutated_base_are_saved(self):
        """Forward readers of a mutated base tensor must not be recomputed."""
        graph = torch.fx.Graph()
        placeholder = graph.placeholder("x")
        producer = graph.call_function(torch.ops.aten.neg.default, args=(placeholder,))
        before = graph.call_function(torch.ops.aten.relu.default, args=(producer,))
        mul_ = graph.call_function(
            torch.ops.aten.mul_.Tensor, args=(producer, placeholder)
        )
        after = graph.call_function(torch.ops.aten.relu.default, args=(producer,))
        graph.output((after,))
        stack = {
            "": ("root", type(nn.Module)),
            "layers.0": ("layers.0", type(nn.Module)),
        }
        for node in (producer, before, mul_, after):
            node.meta["nn_module_stack"] = stack
            node.meta["val"] = torch.randn(2, 2)
        gm = torch.fx.GraphModule(nn.Module(), graph)

        scope = _CheckpointScope(
            region_fqns=frozenset({"layers.0"}), unit_fqns=frozenset({"layers.0"})
        )
        _tag_sac_policy(gm, PassConfig(ac_mode="full"), scope)

        self.assertEqual(mul_.meta["recompute"], CheckpointPolicy.MUST_SAVE)
        self.assertEqual(
            before.meta["recompute"],
            CheckpointPolicy.MUST_SAVE,
            "a reader before the mutation would recompute against the "
            "post-mutation original",
        )
        self.assertEqual(
            after.meta["recompute"],
            CheckpointPolicy.MUST_SAVE,
            "a reader after the mutation would recompute against the "
            "pre-mutation producer duplicate",
        )
        self.assertEqual(
            producer.meta["recompute"],
            CheckpointPolicy.MUST_SAVE,
            "the producer owns mutated storage and backward may need its final value",
        )

    def test_alias_chain_readers_are_saved(self):
        """Readers through sibling views of mutated storage must not recompute."""
        graph = torch.fx.Graph()
        placeholder = graph.placeholder("x")
        producer = graph.call_function(torch.ops.aten.neg.default, args=(placeholder,))
        view = graph.call_function(torch.ops.aten.view.default, args=(producer, [-1]))
        before = graph.call_function(torch.ops.aten.relu.default, args=(view,))
        sibling = graph.call_function(
            torch.ops.aten.transpose.int, args=(view, 0, 1)
        )
        mul_ = graph.call_function(
            torch.ops.aten.mul_.Tensor, args=(sibling, placeholder)
        )
        after = graph.call_function(torch.ops.aten.relu.default, args=(view,))
        graph.output((after,))
        stack = {
            "": ("root", type(nn.Module)),
            "layers.0": ("layers.0", type(nn.Module)),
        }
        for node in (producer, view, before, sibling, mul_, after):
            node.meta["nn_module_stack"] = stack
            node.meta["val"] = torch.randn(2, 2)
        gm = torch.fx.GraphModule(nn.Module(), graph)

        scope = _CheckpointScope(
            region_fqns=frozenset({"layers.0"}), unit_fqns=frozenset({"layers.0"})
        )
        _tag_sac_policy(gm, PassConfig(ac_mode="full"), scope)

        for node in (view, sibling, before, after):
            self.assertEqual(
                node.meta["recompute"],
                CheckpointPolicy.MUST_SAVE,
                f"{node.name} shares or reads mutated storage and cannot recompute",
            )
        self.assertEqual(
            producer.meta["recompute"],
            CheckpointPolicy.MUST_SAVE,
            "the base producer must retain writes performed through sibling views",
        )

    def test_out_of_scope_mutation_saves_scoped_alias_readers(self):
        """A mutation outside the scope still invalidates scoped alias readers."""
        graph = torch.fx.Graph()
        placeholder = graph.placeholder("x")
        producer = graph.call_function(torch.ops.aten.neg.default, args=(placeholder,))
        view = graph.call_function(torch.ops.aten.view.default, args=(producer, [-1]))
        before = graph.call_function(torch.ops.aten.relu.default, args=(view,))
        mutation = graph.call_function(
            torch.ops.aten.add_.Tensor, args=(producer, 1.0)
        )
        graph.output((before, mutation))
        stack = {
            "": ("root", type(nn.Module)),
            "layers.0": ("layers.0", type(nn.Module)),
        }
        for node in (producer, view, before):
            node.meta["nn_module_stack"] = stack
            node.meta["val"] = torch.randn(2, 2)
        mutation.meta["val"] = torch.randn(2, 2)
        gm = torch.fx.GraphModule(nn.Module(), graph)

        scope = _CheckpointScope(
            region_fqns=frozenset({"layers.0"}), unit_fqns=frozenset({"layers.0"})
        )
        _tag_sac_policy(gm, PassConfig(ac_mode="full"), scope)

        self.assertNotIn("recompute", mutation.meta)
        self.assertEqual(view.meta["recompute"], CheckpointPolicy.MUST_SAVE)
        self.assertEqual(before.meta["recompute"], CheckpointPolicy.MUST_SAVE)


class TestRemat(unittest.TestCase):
    """Recompute duplicates preserve numerics and carry the phase tag."""

    @classmethod
    def setUpClass(cls) -> None:
        """Fix the model and inputs shared by the remat tests."""
        torch.manual_seed(2)
        cls.model = _Model()
        cls.x = torch.randn(4, 8)
        cls.y = torch.randn(4, 8)
        cls.config = PassConfig(
            ac_mode="full", enable_overlap=False, fsdp_enabled=False
        )

    def test_numerics_match_eager(self):
        """Full and selective both reproduce eager loss and gradients."""
        ref_loss, ref_grads = _eager_reference(self.model, self.x, self.y)
        configs = (
            self.config,
            _selective_config(["layers.0", "layers.1"]),
        )
        for config in configs:
            with self.subTest(ac_mode=config.ac_mode):
                joint, model = _run_ac(self.model, config)
                loss, grads, _ = run_traced_graph(
                    joint, model, {"x": self.x, "y": self.y}
                )
                self.assertTrue(torch.allclose(loss, ref_loss, atol=1e-10))
                for grad, ref_grad in zip(grads, ref_grads):
                    self.assertTrue(torch.allclose(grad, ref_grad, atol=1e-10))

    def test_remat_recompiles_without_raise_getitems(self):
        """The fallback still recompiles when torch lacks raise_getitems."""
        joint = _trace(self.model, self.x, self.y)
        with patch(
            "hyper_parallel.compile.passes.activation_checkpoint."
            "compile_utils.raise_getitems",
            None,
            create=True,
        ):
            ActivationCheckpointPass().run(
                joint.graph_module, self.config, model=self.model
            )
        loss, grads, _ = run_traced_graph(joint, self.model, {"x": self.x, "y": self.y})
        ref_loss, ref_grads = _eager_reference(self.model, self.x, self.y)
        self.assertTrue(torch.allclose(loss, ref_loss, atol=1e-10))
        for grad, ref_grad in zip(grads, ref_grads):
            self.assertTrue(torch.allclose(grad, ref_grad, atol=1e-10))

    def test_duplicates_are_backward_tagged_and_selective_is_subset(self):
        """Duplicates are backward-tagged; selective duplicates fewer than full."""
        full_joint, _ = _run_ac(self.model, self.config)
        selective_joint, _ = _run_ac(self.model, _selective_config(["layers.0"]))
        duplicates = _duplicates(full_joint.graph_module)

        self.assertTrue(duplicates, "full mode must duplicate recompute nodes")
        self.assertTrue(all(d.meta.get("autograd_backward", False) for d in duplicates))
        self.assertLess(
            len(_duplicates(selective_joint.graph_module)),
            len(duplicates),
        )

    def test_alias_view_readers_match_eager(self):
        """A view read before an in-place write keeps the correct sin gradient."""
        torch.manual_seed(8)
        x = torch.randn(4, 8)
        y = torch.randn(4, 8)
        configs = (
            PassConfig(ac_mode="full", enable_overlap=False, fsdp_enabled=False),
            _selective_config(["layers.0"]),
        )
        for config in configs:
            with self.subTest(ac_mode=config.ac_mode):
                model = _AliasModel()
                ref_loss, ref_grads = _eager_reference(model, x, y)
                joint = _trace(model, x, y)
                ActivationCheckpointPass().run(
                    joint.graph_module, config, model=model
                )
                loss, grads, _ = run_traced_graph(joint, model, {"x": x, "y": y})

                self.assertTrue(torch.allclose(loss, ref_loss, atol=1e-10))
                for grad, ref_grad in zip(grads, ref_grads):
                    self.assertTrue(torch.allclose(grad, ref_grad, atol=1e-10))

    def test_saved_buffer_written_through_view_matches_original(self) -> None:
        """A saved precomputed gradient must retain writes through its view.

        Description: Write into a gradient buffer through a view before backward.
        Expectation: Full and selective recompute preserve the original result.
        Feature: Static-graph activation checkpointing with aliased mutations.
        """
        for config in (
            PassConfig(ac_mode="full"),
            _selective_config(["layers.0"]),
        ):
            with self.subTest(ac_mode=config.ac_mode):
                graph = torch.fx.Graph()
                value = graph.placeholder("value")
                buffer = graph.call_function(torch.ops.aten.zeros_like.default, args=(value,))
                view = graph.call_function(torch.ops.aten.view.default, args=(buffer, [-1]))
                source = graph.call_function(torch.ops.aten.view.default, args=(value, [-1]))
                write = graph.call_function(torch.ops.aten.copy_.default, args=(view, source))
                backward = graph.call_function(torch.ops.aten.mul.Tensor, args=(buffer, value))
                backward.meta["autograd_backward"] = True
                graph.output(backward)
                for node in (buffer, view, source, write):
                    node.meta["nn_module_stack"] = {"block": ("layers.0", nn.Module)}
                    node.meta["val"] = torch.randn(2, 2)
                module = torch.fx.GraphModule(nn.Module(), graph)
                inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
                expected = module(inputs)
                scope = _CheckpointScope(
                    region_fqns=frozenset({"layers.0"}),
                    unit_fqns=frozenset({"layers.0"}),
                )
                ActivationCheckpointPass().run(module, config, ac_scope=scope)
                torch.testing.assert_close(module(inputs), expected, rtol=0, atol=0)


class TestLayerSelection(unittest.TestCase):
    """Layer selectors bound the recompute scope to the chosen blocks."""

    @classmethod
    def setUpClass(cls) -> None:
        """Fix the model and inputs shared by the selection tests."""
        torch.manual_seed(3)
        cls.model = _Model()

    def _duplicate_units(self, pass_config: PassConfig) -> set:
        """Run the pass and return the innermost block FQNs of the duplicates."""
        joint, _ = _run_ac(self.model, pass_config)
        units = set()
        for duplicate in _duplicates(joint.graph_module):
            stack = [
                fqn for fqn, _ in (duplicate.meta.get("nn_module_stack") or {}).values()
            ]
            units.add(next(fqn for fqn in stack if fqn.startswith("layers.")))
        return units

    def test_layer_selectors_bound_recompute_scope(self):
        """Prefix count and explicit indices select the expected blocks."""
        cases = (
            ({"ac_layer_count": 1}, {"layers.0"}),
            ({"ac_layer_indices": [1]}, {"layers.1"}),
        )
        for kwargs, expected in cases:
            with self.subTest(**kwargs):
                config = PassConfig(
                    ac_mode="full",
                    enable_overlap=False,
                    fsdp_enabled=False,
                    **kwargs,
                )
                self.assertEqual(self._duplicate_units(config), expected)

    def test_missing_block_marker_raises(self):
        """A model without the HF checkpointing marker cannot resolve layers."""
        model = _Model()
        del model.gradient_checkpointing
        config = PassConfig(ac_mode="full", enable_overlap=False, fsdp_enabled=False)
        with self.assertRaises(ValueError):
            ActivationCheckpointPass().run(
                torch.fx.symbolic_trace(model), config, model=model
            )

    def test_selector_without_model_raises(self):
        """Layer selectors require the live model kwarg."""
        config = PassConfig(
            ac_mode="full",
            ac_layer_count=1,
            enable_overlap=False,
            fsdp_enabled=False,
        )
        with self.assertRaises(ValueError):
            ActivationCheckpointPass().run(torch.fx.symbolic_trace(_Model()), config)


class TestEdgeCases(unittest.TestCase):
    """Boundary behaviours found in review: zero layers, rerun, no targets."""

    @classmethod
    def setUpClass(cls) -> None:
        """Fix the model and inputs shared by the edge-case tests."""
        torch.manual_seed(6)
        cls.model = _Model()
        cls.x = torch.randn(4, 8)
        cls.y = torch.randn(4, 8)

    def test_zero_layer_count_is_a_noop(self):
        """Zero selected layers no-ops."""
        config = PassConfig(
            ac_mode="full",
            ac_layer_count=0,
            enable_overlap=False,
            fsdp_enabled=False,
        )
        joint, model = _run_ac(self.model, config)

        self.assertFalse(_duplicates(joint.graph_module))
        self.assertTrue(
            all("recompute" not in node.meta for node in joint.graph_module.graph.nodes)
        )
        loss, grads, _ = run_traced_graph(joint, model, {"x": self.x, "y": self.y})
        ref_loss, ref_grads = _eager_reference(self.model, self.x, self.y)
        self.assertTrue(torch.allclose(loss, ref_loss, atol=1e-10))
        for grad, ref_grad in zip(grads, ref_grads):
            self.assertTrue(torch.allclose(grad, ref_grad, atol=1e-10))

    def test_selective_model_target_without_block_marker(self):
        """Model-root selective targets do not need transformer-block discovery."""
        model = _Model()
        del model.gradient_checkpointing
        joint, _ = _run_ac(model, _selective_config(["head"]))
        self.assertTrue(_duplicates(joint.graph_module))

    def test_second_run_does_not_duplicate_duplicates(self):
        """Rerunning the pass does not nest recompute copies."""
        joint, _ = _run_ac(self.model, PassConfig(ac_mode="full"))
        gm = joint.graph_module
        first = len(_duplicates(gm))

        ActivationCheckpointPass().run(
            gm,
            PassConfig(ac_mode="full", enable_overlap=False, fsdp_enabled=False),
            model=self.model,
        )

        self.assertEqual(len(_duplicates(gm)), first)
        self.assertFalse(
            any(
                node.meta.get("ac_recomputed_from")
                in {dup.name for dup in _duplicates(gm)}
                for node in _duplicates(gm)
            ),
            "a duplicate must not be produced from another duplicate",
        )


class TestFsdpComposition(unittest.TestCase):
    """AC runs before FSDPPass and keeps the sharded graph correct."""

    def test_full_ac_then_fsdp_simulated_rank_matches_eager(self):
        """Simulated single-rank FSDP execution of the AC graph is exact."""
        torch.manual_seed(4)
        reference = _Model().double()
        x = torch.randn(4, 8, dtype=torch.float64)
        y = torch.randn(4, 8, dtype=torch.float64)

        joint = _trace(reference, x, y)
        graph = joint.graph_module
        live = copy.deepcopy(reference)
        ac_config = PassConfig(ac_mode="full", enable_overlap=False, fsdp_enabled=False)
        ActivationCheckpointPass().run(graph, ac_config, model=reference)

        fsdp_config = PassConfig(
            fsdp_enabled=True,
            fsdp_degree=_DEGREE,
            fsdp_reshard_after_forward=True,
            ac_mode="full",
        )
        plan = GraphParallelPlan().fsdp_mark_pattern("*")
        fsdp_pass = FSDPPass(parallel_plan=plan)
        with _patch_dist(world_size=_DEGREE, rank=0):
            fsdp_pass.run(graph, fsdp_config, model=live, fsdp_group_name="fsdp")
        graph.graph.lint()

        id_to_full = {
            id(shard): full
            for shard, full in zip(live.parameters(), reference.parameters())
        }
        state = extract_module_state(live)
        outputs = _SimulatedRank(graph, id_to_full, _DEGREE).run(
            *list(state.values()), x, y
        )
        loss, grads = outputs[0], list(outputs[1:])

        ref_loss = _train_fn(reference, x=x, y=y)
        ref_grads = torch.autograd.grad(ref_loss, list(reference.parameters()))
        self.assertTrue(torch.allclose(loss, ref_loss, atol=1e-10))
        for grad, ref_grad in zip(grads, ref_grads):
            expected = ref_grad.chunk(_DEGREE, dim=0)[0]
            self.assertTrue(torch.allclose(grad, expected, atol=1e-10))


class TestPassConfigValidation(unittest.TestCase):
    """``PassConfig`` validates the new AC fields."""

    def test_invalid_fields_raise(self):
        """Invalid modes and layer selectors are rejected."""
        invalid = (
            {"ac_mode": "partial"},
            {"ac_mode": "full", "ac_layer_count": 1, "ac_layer_indices": [0]},
            {"ac_layer_count": -1},
            {"ac_layer_indices": []},
            {"ac_layer_indices": [1, 1]},
            {"ac_mode": "selective"},
            {"ac_mode": "selective", "ac_selective_recompute_regions": [("fc1", [])]},
            {"ac_selective_recompute_modules": ["head"]},
            {
                "ac_mode": "selective",
                "ac_layer_count": 1,
                "ac_selective_recompute_modules": ["head"],
            },
        )
        for kwargs in invalid:
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                PassConfig(**kwargs)


class TestTrainerConfigProjection(unittest.TestCase):
    """The dynamic ``ActivationCheckpointConfig`` projects onto ``PassConfig``.

    ``full`` is a real ``FullCheckpointConfig`` so the projection exercises the
    ``recompute_layer`` -> ``recompute_layers`` normalization. The top-level
    object stays duck-typed because ``ActivationCheckpointConfig`` validates
    mode-specific sections (and rejects ``mode=selective`` without a selective
    block) that these projection tests deliberately leave out.
    """

    @staticmethod
    def _ac_config(
        mode: Any = "off",
        *,
        recompute_layer: Any = None,
        exclude_region: Any = None,
        swap_inputs: bool = False,
        allow_cache_entry_mutation: bool = False,
        selective: Any = None,
    ) -> Any:
        """Build the ``ActivationCheckpointConfig`` field surface."""
        full = (
            None
            if recompute_layer is None
            else FullCheckpointConfig(
                recompute_layer=list(recompute_layer),
                exclude_region=exclude_region,
            )
        )
        return SimpleNamespace(
            mode=mode,
            full=full,
            selective=selective,
            swap_inputs=swap_inputs,
            allow_cache_entry_mutation=allow_cache_entry_mutation,
        )

    @staticmethod
    def _trainer_config(activation_checkpoint: Any) -> Any:
        """Build the minimal trainer-config surface the projection reads."""
        return SimpleNamespace(
            fsdp_config=SimpleNamespace(dp_shard_size=1, edp_shard_size=1),
            activation_checkpoint=activation_checkpoint,
        )

    def test_mode_and_selectors_project(self):
        """Mode, full-layer selectors, defaults, and the missing section."""
        config = build_pass_config_from_trainer_config(
            self._trainer_config(self._ac_config())
        )
        self.assertEqual(config.ac_mode, "off")

        config = build_pass_config_from_trainer_config(
            self._trainer_config(
                self._ac_config(
                    mode="selective",
                    selective=SelectiveCheckpointConfig(recompute={"fc1": [0, 1]}),
                )
            )
        )
        self.assertEqual((config.ac_mode, config.ac_layer_count), ("selective", None))
        self.assertIsNone(config.ac_layer_indices)
        self.assertEqual(
            config.ac_selective_recompute_regions,
            [("fc1", [0, 1])],
        )
        self.assertIsNone(config.ac_selective_recompute_modules)

        config = build_pass_config_from_trainer_config(
            self._trainer_config(self._ac_config(mode="full", recompute_layer=[0, 2]))
        )
        self.assertEqual(config.ac_mode, "full")
        self.assertIsNone(config.ac_layer_count)
        self.assertEqual(config.ac_layer_indices, [0, 2])

        config = build_pass_config_from_trainer_config(
            self._trainer_config(self._ac_config(mode="full", recompute_layer=["1-3"]))
        )
        self.assertEqual(config.ac_layer_indices, [1, 2, 3])

        config = build_pass_config_from_trainer_config(
            self._trainer_config(self._ac_config(mode="full"))
        )
        self.assertEqual(config.ac_mode, "full")
        self.assertIsNone(config.ac_layer_count)
        self.assertIsNone(config.ac_layer_indices)

        config = build_pass_config_from_trainer_config(
            SimpleNamespace(
                fsdp_config=SimpleNamespace(dp_shard_size=1, edp_shard_size=1)
            )
        )
        self.assertEqual(config.ac_mode, "off")

    def test_full_raw_recompute_layer_projects(self):
        """A duck-typed full config may expose only the raw integer selector."""
        config = build_pass_config_from_trainer_config(
            self._trainer_config(
                SimpleNamespace(
                    mode="full",
                    full=SimpleNamespace(recompute_layer=[0, 2]),
                )
            )
        )
        self.assertEqual(config.ac_layer_indices, [0, 2])

    def test_full_raw_recompute_layer_ranges_are_rejected(self):
        """Unnormalized ranges must not silently degrade to all layers."""
        config = SimpleNamespace(
            mode="full",
            full=SimpleNamespace(recompute_layer=["1-3"]),
        )
        with self.assertRaisesRegex(ValueError, "recompute_layers is normalized"):
            build_pass_config_from_trainer_config(self._trainer_config(config))

    def test_hybrid_mode_is_rejected(self):
        """Graph mode has no ``hybrid`` equivalent and fails loudly."""
        trainer_config = self._trainer_config(
            self._ac_config(mode="hybrid", recompute_layer=[0])
        )
        with self.assertRaisesRegex(ValueError, "hybrid.*not supported"):
            build_pass_config_from_trainer_config(trainer_config)

    def test_selective_model_targets_project(self):
        """Model-root selective recompute targets project as module paths."""
        config = build_pass_config_from_trainer_config(
            self._trainer_config(
                self._ac_config(
                    mode="selective",
                    selective=SelectiveCheckpointConfig(recompute=["head"]),
                )
            )
        )
        self.assertEqual(config.ac_selective_recompute_regions, None)
        self.assertEqual(config.ac_selective_recompute_modules, ["head"])

    def test_selective_swap_warns_when_ignored(self):
        """Selective swap targets are not supported in graph mode yet."""
        selective = SelectiveCheckpointConfig(
            recompute={"fc1": [0]},
            swap={"fc2": [0]},
        )
        trainer_config = self._trainer_config(
            self._ac_config(mode="selective", selective=selective)
        )
        with self.assertLogs("hyper_parallel.compile.pass_config", level="WARNING"):
            config = build_pass_config_from_trainer_config(trainer_config)
        self.assertEqual(config.ac_selective_recompute_regions, [("fc1", [0])])

    def test_selective_swap_only_is_rejected(self):
        """Graph mode cannot express a swap-only selective config yet."""
        selective = SelectiveCheckpointConfig(swap={"fc2": [0]})
        trainer_config = self._trainer_config(
            self._ac_config(mode="selective", selective=selective)
        )
        with self.assertRaisesRegex(ValueError, "requires at least one recompute"):
            build_pass_config_from_trainer_config(trainer_config)

    def test_full_exclusions_warn_when_ignored(self):
        """Full-mode exclusion regions are reported, not applied."""
        exclude_region = FullExcludeRegionConfig(
            save={"layers.0.fc1": [0]},
        )
        trainer_config = self._trainer_config(
            self._ac_config(
                mode="full",
                recompute_layer=[0, 1],
                exclude_region=exclude_region,
            )
        )
        with self.assertLogs("hyper_parallel.compile.pass_config", level="WARNING"):
            config = build_pass_config_from_trainer_config(trainer_config)
        self.assertEqual((config.ac_mode, config.ac_layer_indices), ("full", [0, 1]))

    def test_swap_inputs_is_ignored_with_warning(self):
        """``swap_inputs`` has no graph-mode counterpart and warns."""
        trainer_config = self._trainer_config(
            self._ac_config(mode="full", swap_inputs=True)
        )
        with self.assertLogs("hyper_parallel.compile.pass_config", level="WARNING"):
            config = build_pass_config_from_trainer_config(trainer_config)
        self.assertEqual(config.ac_mode, "full")


class TestPipelineIntegration(unittest.TestCase):
    """The pipeline selects the AC pass between cleanup and FSDP."""

    def test_pipeline_selection(self):
        """AC sits between canonicalize and FSDP; off keeps it out."""
        names = [
            type(pass_obj).__name__
            for pass_obj in PassPipeline(PassConfig(ac_mode="full")).build().passes
        ]
        self.assertIn("ActivationCheckpointPass", names)
        self.assertLess(
            names.index("ActivationCheckpointPass"),
            names.index("FSDPPass"),
            "the AC pass must run before FSDPPass",
        )
        self.assertGreater(
            names.index("ActivationCheckpointPass"),
            names.index("CanonicalizeGraphPass"),
        )

        off_names = [
            type(pass_obj).__name__
            for pass_obj in PassPipeline(PassConfig(ac_mode="off")).build().passes
        ]
        self.assertNotIn("ActivationCheckpointPass", off_names)


class TestGraphCompilerEndToEnd(unittest.TestCase):
    """``GraphCompiler`` drives the pass and matches eager training."""

    def test_compiler_with_full_ac_matches_eager(self):
        """Compiled forward_backward equals eager loss and gradients."""
        torch.manual_seed(5)
        model = _Model()
        x = torch.randn(4, 8)
        y = torch.randn(4, 8)
        ref_loss, ref_grads = _eager_reference(model, x, y)

        compiler = GraphCompiler(
            model=model,
            train_fn=_train_fn,
            pass_config=PassConfig(
                ac_mode="full", fsdp_enabled=False, enable_overlap=False
            ),
        )
        compiler.compile(x=x, y=y)
        loss, _ = compiler.forward_backward(x=x, y=y)

        self.assertTrue(torch.allclose(loss, ref_loss, atol=1e-10))
        for parameter, ref_grad in zip(model.parameters(), ref_grads):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.allclose(parameter.grad, ref_grad, atol=1e-10))


class TestHOPSubgraphAttrs(unittest.TestCase):
    """Recompute duplicates get private HOP subgraph ``get_attr`` nodes."""

    def test_duplicate_owns_a_fresh_get_attr(self):
        """The duplicate reads a new get_attr instead of the shared subgraph."""

        class _Subgraph(nn.Module):
            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                """Return the shifted input so the subgraph is not an identity."""
                return inputs + 1

        subgraph = torch.fx.symbolic_trace(_Subgraph())
        graph = torch.fx.Graph()
        inputs = graph.placeholder("x")
        sub_attr = graph.get_attr("sub")
        sub_attr.meta["marker"] = "shared"
        forward = graph.call_function(
            torch.ops.aten.add.Tensor, args=(inputs, sub_attr)
        )
        forward.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE
        backward = graph.call_function(
            torch.ops.aten.mul.Tensor, args=(forward, forward)
        )
        backward.meta["autograd_backward"] = True
        graph.output((backward,))
        root = nn.Module()
        root.sub = subgraph
        graph_module = torch.fx.GraphModule(root, graph)

        _remat(graph_module)

        attributes = [
            node
            for node in graph_module.graph.nodes
            if node.op == "get_attr" and node.target == "sub"
        ]
        self.assertEqual(len(attributes), 2)
        duplicate = _duplicates(graph_module)
        self.assertEqual(len(duplicate), 1)
        duplicate_attr = next(
            node for node in duplicate[0].all_input_nodes if node.op == "get_attr"
        )
        self.assertIsNot(duplicate_attr, sub_attr)
        self.assertEqual(duplicate_attr.meta.get("marker"), "shared")
        self.assertIn(duplicate[0], backward.all_input_nodes)

    def test_duplicate_owns_a_fresh_get_attr_for_dotted_target(self):
        """A nested (dotted) HOP get_attr is duplicated too."""

        class _Subgraph(nn.Module):
            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                """Return the shifted input so the subgraph is not an identity."""
                return inputs + 1

        subgraph = torch.fx.symbolic_trace(_Subgraph())
        graph = torch.fx.Graph()
        inputs = graph.placeholder("x")
        sub_attr = graph.get_attr("layers.0.sub")
        sub_attr.meta["marker"] = "shared"
        forward = graph.call_function(
            torch.ops.aten.add.Tensor, args=(inputs, sub_attr)
        )
        forward.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE
        backward = graph.call_function(
            torch.ops.aten.mul.Tensor, args=(forward, forward)
        )
        backward.meta["autograd_backward"] = True
        graph.output((backward,))
        root = nn.Module()
        root.layers = nn.ModuleList([nn.Module()])
        root.layers[0].sub = subgraph
        graph_module = torch.fx.GraphModule(root, graph)

        _remat(graph_module)

        attributes = [
            node
            for node in graph_module.graph.nodes
            if node.op == "get_attr" and node.target == "layers.0.sub"
        ]
        self.assertEqual(len(attributes), 2)
        duplicate = _duplicates(graph_module)
        self.assertEqual(len(duplicate), 1)
        duplicate_attr = next(
            node for node in duplicate[0].all_input_nodes if node.op == "get_attr"
        )
        self.assertIsNot(duplicate_attr, sub_attr)
        self.assertEqual(duplicate_attr.meta.get("marker"), "shared")


class TestEagerWrappedModel(unittest.TestCase):
    """A live model already wrapped by eager AC is reported, not silent."""

    @classmethod
    def setUpClass(cls) -> None:
        """Keep eager recomputation on CPU instead of probing an accelerator."""
        cls._default_device_type = DefaultDeviceType.get_device_type()
        DefaultDeviceType.set_device_type("cpu")

    @classmethod
    def tearDownClass(cls) -> None:
        """Restore the process-wide checkpoint device type."""
        DefaultDeviceType.set_device_type(cls._default_device_type)

    def test_eager_wrapped_graph_warns_when_nothing_recomputed(self):
        """Graph AC warns when the model was already eagerly checkpointed."""
        torch.manual_seed(7)
        model = _Model()
        wrapped = _apply_activation_checkpointing(model, "full", enable_compile=False)
        joint = _trace(wrapped, torch.randn(4, 8), torch.randn(4, 8))

        with self.assertLogs(
            "hyper_parallel.compile.passes.activation_checkpoint", level="WARNING"
        ):
            ActivationCheckpointPass().run(
                joint.graph_module,
                PassConfig(ac_mode="full", enable_overlap=False, fsdp_enabled=False),
                model=wrapped,
            )

        self.assertFalse(_duplicates(joint.graph_module))


if __name__ == "__main__":
    unittest.main()
