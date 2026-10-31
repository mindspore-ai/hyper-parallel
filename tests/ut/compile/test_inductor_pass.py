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
"""Unit tests for ``hyper_parallel.compile.passes.inductor_pass.InductorPass``.

Covers the terminal TorchInductor compilation pass:

1. ``full`` / ``regional`` modes end-to-end through ``GraphCompiler``: the
   compiled joint graph reproduces the uncompiled loss/gradients and keeps
   the positional calling convention of ``run_traced_graph``.
2. Regional tagging on an FSDP-shaped graph (hand-inserted collectives and
   the reshard storage-free op): communication and side-effectful nodes stay
   eager, val-less inserted nodes get fake values, compute is absorbed into a
   compiled region, and full mode functionalizes the collectives.
3. Pipeline-parallel support: a ``PpPass`` schedule stub is not compiled
   itself; every per-stage forward/backward subgraph is compiled instead
   (both ``regional`` and ``full``), while unknown ``call_module`` stubs are
   still rejected.
4. Guards: missing ``example_inputs``, mode canonicalization, pipeline
   placement (terminal), and that inductor settings stay PassConfig-owned
   (PP-style) instead of being projected from ``CompileConfig``.
"""

# The backend env var must be set before torch / hyper_parallel are imported,
# heavy test-only deps are imported lazily inside helpers, and the tests
# intentionally inspect FX internals (codegen convention).
# pylint: disable=wrong-import-position,import-outside-toplevel,protected-access

import os
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import torch
from torch import fx, nn

from tests.ut.compile.inductor_stub import install_regional_inductor_stub

# torch 2.6/2.7/2.9 do not ship regional_inductor; install the executable test
# double so these UT exercise the real compile path instead of being skipped.
install_regional_inductor_stub()

from hyper_parallel.compile.compiler import GraphCompiler
from hyper_parallel.compile.pass_config import (
    PassConfig,
    build_pass_config_from_trainer_config,
)
from hyper_parallel.compile.passes.inductor_pass import (
    InductorPass,
    _propagate_fake_vals,
)
from hyper_parallel.compile.passes.parallel.pp_schedule import PipelineScheduleBase
from hyper_parallel.compile.passes.pipeline import PassPipeline
from hyper_parallel.compile.tracer.graph_tracer import trace_model_graph


def _make_model() -> nn.Sequential:
    """A tiny CPU model; tracing and Inductor compilation stay cheap."""
    return nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 4))


def _train_fn(model, *, x, y) -> torch.Tensor:
    """Training function: mean-squared error between prediction and target."""
    return ((model(x) - y) ** 2).mean()


def _fixed_batch():
    """A deterministic ``(input, label)`` batch (same tensors every call)."""
    torch.manual_seed(7)
    return torch.randn(2, 4), torch.randn(2, 4)


def _linear_stage(x, w):
    """A tiny compute-only stage used to build a fake PP schedule."""
    return torch.nn.functional.linear(x, w)  # pylint: disable=not-callable


def _make_fake_stage_gm(*shapes):
    """Trace ``_linear_stage`` into an FX graph with fake ``meta['val']``."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx

    with FakeTensorMode():
        args = [torch.randn(*shape) for shape in shapes]
        return make_fx(_linear_stage)(*args)


def _make_pp_stub():
    """Build a PpPass-shaped stub around a one-stage schedule.

    Mirrors ``PpPass._install_stub``: the outer graph only has the
    ``call_module`` dispatch node; the compute lives in ``fwd_gm`` / ``bwd_gm``.
    """
    fwd_gm = _make_fake_stage_gm((2, 4), (4, 4))
    bwd_gm = _make_fake_stage_gm((2, 4), (4, 4))
    schedule = PipelineScheduleBase(
        fwd_gm,
        bwd_gm,
        stage_idx=0,
        pp_degree=2,
        num_state=2,
        num_trainable=0,
        num_send=0,
        grad_send_count=0,
        microbatch_size=1,
    )
    stub = fx.GraphModule(nn.Module(), fx.Graph())
    stub.add_module("pp_schedule", schedule)
    call = stub.graph.call_module("pp_schedule")
    stub.graph.output(call)
    stub.recompile()
    return stub, schedule


class TestInductorEndToEnd(unittest.TestCase):
    """``full`` / ``regional`` compilation through ``GraphCompiler``."""

    def test_compiled_graph_matches_uncompiled(self):
        """Test both compiling modes reproduce the uncompiled graph exactly."""
        for inductor_mode in ("full", "regional"):
            with self.subTest(inductor_mode=inductor_mode):
                torch.manual_seed(3)
                model = _make_model()
                baseline = GraphCompiler(
                    model=model,
                    train_fn=_train_fn,
                    pass_config=PassConfig(fsdp_enabled=False),
                    device=torch.device("cpu"),
                )
                x, y = _fixed_batch()
                loss_ref, _ = baseline.forward_backward(x=x, y=y)
                grads_ref = [p.grad.clone() for p in model.parameters()]

                compiled_model = _make_model()
                compiled_model.load_state_dict(model.state_dict())
                compiled = GraphCompiler(
                    model=compiled_model,
                    train_fn=_train_fn,
                    pass_config=PassConfig(
                        fsdp_enabled=False, inductor_mode=inductor_mode
                    ),
                    device=torch.device("cpu"),
                )
                loss, _ = compiled.forward_backward(x=x, y=y)

                joint_gm = compiled._joint_graph.graph_module
                self.assertEqual(joint_gm.meta.get("inductor_compiled"), inductor_mode)
                # Positional calling convention restored: run_traced_graph
                # feeds the flat state+inputs list positionally, so boxed
                # codegen must be gone.
                self.assertNotIsInstance(
                    joint_gm.graph._codegen, fx.graph._BoxedCodeGen
                )
                self.assertEqual(float(loss), float(loss_ref))
                for param, grad_ref in zip(compiled_model.parameters(), grads_ref):
                    self.assertTrue(
                        torch.allclose(param.grad, grad_ref, atol=1e-6),
                        f"{inductor_mode} gradients must match the uncompiled graph",
                    )


class TestInductorRegionalGraphShape(unittest.TestCase):
    """Regional tagging on an FSDP-shaped graph (no real communication).

    Inductor resolves a collective's process group by name at region-compile
    time, so these tests bring up a single-process gloo world (no accelerator
    backend) and register it under the ``"fsdp"`` group name. The graph is
    still never executed; teardown fully destroys the group so later tests
    see the same no-distributed state they would without this class.
    """

    _pg = None
    _prev_fsdp_pg = None
    _owned_dist = False

    @classmethod
    def setUpClass(cls) -> None:
        """Bring up the one-rank gloo world and register the ``fsdp`` group."""
        import socket

        import torch.distributed as dist
        from torch.distributed.distributed_c10d import (
            _register_process_group,
            _resolve_process_group,
        )

        if dist.is_initialized():
            cls._owned_dist = False
        else:
            # Bind port 0 to let the OS pick a free port; a fixed port races
            # with other tests / parallel CI jobs.
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", 0))
                free_port = sock.getsockname()[1]
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", str(free_port))
            dist.init_process_group(backend="gloo", rank=0, world_size=1)
            cls._owned_dist = True
        cls._pg = dist.new_group([0])
        # Remember any pre-existing ``fsdp`` name so teardown can restore the
        # global registry instead of leaking this class's group to later tests.
        try:
            cls._prev_fsdp_pg = _resolve_process_group("fsdp")
        except RuntimeError:
            cls._prev_fsdp_pg = None
        _register_process_group("fsdp", cls._pg)

    @classmethod
    def tearDownClass(cls) -> None:
        """Tear the world down so other UT files see no distributed state."""
        import torch.distributed as dist
        from torch.distributed.distributed_c10d import (
            _register_process_group,
            _unregister_process_group,
        )

        if cls._prev_fsdp_pg is not None:
            _register_process_group("fsdp", cls._prev_fsdp_pg)
        else:
            _unregister_process_group("fsdp")
        cls._prev_fsdp_pg = None
        if cls._pg is not None:
            dist.destroy_process_group(cls._pg)
            cls._pg = None
        if cls._owned_dist:
            dist.destroy_process_group()
            cls._owned_dist = False

    def _trace_with_collectives(self):
        """Trace a joint graph and mimic FSDPPass insertions on its weight.

        Inserts an all_gather + wait pair, the reshard storage-free op, and a
        val-less pad-narrow view — the node kinds real FSDP graphs carry —
        without executing anything.
        """
        from torch.ops import _c10d_functional

        from hyper_parallel.compile.passes.parallel.fsdp_pass import (
            _free_tensor_storage,
        )

        torch.manual_seed(5)
        model = nn.Linear(4, 4)
        joint = trace_model_graph(
            model, _train_fn, dict(zip(("x", "y"), _fixed_batch()))
        )
        gm = joint.graph_module
        weight_ph = gm.graph.find_nodes(op="placeholder")[2]

        out_node = gm.graph.find_nodes(op="output")[0]
        with gm.graph.inserting_before(out_node):
            ag_node = gm.graph.call_function(
                _c10d_functional.all_gather_into_tensor,
                args=(weight_ph, self._pg.size(), "fsdp"),
            )
            wait_node = gm.graph.call_function(_c10d_functional.wait_tensor, (ag_node,))
            free_node = gm.graph.call_function(_free_tensor_storage, (wait_node,))
            narrow = gm.graph.call_function(
                torch.ops.aten.slice.Tensor, (wait_node, 0, 0, 2)
            )
        gm.graph.lint()
        gm.recompile()
        return joint, ag_node, wait_node, free_node, narrow

    def test_propagate_fills_vals_on_inserted_nodes(self):
        """Test fake propagation fills ``val`` on pass-inserted nodes."""
        joint, _, wait_node, _, narrow = self._trace_with_collectives()
        self.assertIsNone(wait_node.meta.get("val"))
        _propagate_fake_vals(joint.graph_module)
        self.assertIsNotNone(wait_node.meta.get("val"))
        # The narrow slices the gathered weight back down: propagation must
        # produce a shape differing from its input's, proving it ran.
        self.assertEqual(tuple(narrow.meta["val"].shape), (2, 4))

    def test_regional_keeps_collectives_and_free_eager(self):
        """Test regional mode compiles compute but keeps comms/free untouched."""
        joint, ag_node, wait_node, free_node, _ = self._trace_with_collectives()
        InductorPass().run(
            joint.graph_module,
            PassConfig(inductor_mode="regional"),
            example_inputs=joint.example_inputs,
        )

        graph = joint.graph_module.graph
        targets = [n.target for n in graph.nodes if n.op == "call_function"]
        self.assertIn(ag_node.target, targets)
        self.assertIn(wait_node.target, targets)
        self.assertIn(free_node.target, targets)
        # All functional compute (aten.addmm/gelu/...) was absorbed into the
        # compiled region: no aten op remains at the top level.
        self.assertFalse([t for t in targets if str(t).startswith("aten.")])
        # The compiled region call exists and the graph is still positional.
        self.assertNotIsInstance(graph._codegen, fx.graph._BoxedCodeGen)

    def test_full_mode_compiles_collectives_into_region(self):
        """Test full mode also compiles the c10d functional collectives."""
        joint, _, _, _, _ = self._trace_with_collectives()
        gm = joint.graph_module
        InductorPass().run(
            gm, PassConfig(inductor_mode="full"), example_inputs=joint.example_inputs
        )
        self.assertEqual(gm.meta.get("inductor_compiled"), "full")
        # The all-gather launch was functionalized into the compiled region;
        # Inductor re-materializes only the async wait outside the kernel.
        targets = [str(n.target) for n in gm.graph.nodes if n.op == "call_function"]
        self.assertFalse([t for t in targets if "all_gather_into_tensor" in t])
        self.assertTrue([t for t in targets if "_c10d_functional.wait_tensor" in t])


class TestInductorGuards(unittest.TestCase):
    """Validation and pipeline-placement guards."""

    def _traced(self):
        torch.manual_seed(11)
        model = nn.Linear(4, 4)
        return trace_model_graph(
            model, _train_fn, dict(zip(("x", "y"), _fixed_batch()))
        )

    def test_pp_schedule_compiles_each_stage_graph(self):
        """Test PP stage subgraphs compile in both modes; the stub stays outer."""
        for mode in ("regional", "full"):
            with self.subTest(mode=mode):
                stub, schedule = _make_pp_stub()
                InductorPass().run(stub, PassConfig(inductor_mode=mode))

                self.assertEqual(stub.meta.get("inductor_compiled"), mode)
                # The outer stub is still the schedule dispatch.
                self.assertTrue(stub.graph.find_nodes(op="call_module"))
                for stage_gm in (schedule.fwd_gm, schedule.bwd_gm):
                    self.assertEqual(stage_gm.meta.get("inductor_compiled"), mode)
                    self.assertNotIsInstance(
                        stage_gm.graph._codegen, fx.graph._BoxedCodeGen
                    )
                    targets = [
                        str(n.target)
                        for n in stage_gm.graph.nodes
                        if n.op == "call_function"
                    ]
                    self.assertFalse(
                        [t for t in targets if t.startswith("aten.")],
                        f"{mode} must absorb compute into the compiled region",
                    )
                # The positional calling convention works after compilation.
                out = schedule.fwd_gm(torch.randn(2, 4), torch.randn(4, 4))
                self.assertEqual(tuple(out.shape), (2, 4))

    def test_rejects_unknown_call_module_stub(self):
        """Test a ``call_module`` graph that is not a PP schedule is rejected."""
        joint = self._traced()
        gm = joint.graph_module
        gm.add_module("not_a_schedule", nn.Linear(4, 4))
        with gm.graph.inserting_before(gm.graph.find_nodes(op="output")[0]):
            gm.graph.call_module("not_a_schedule")
        gm.graph.lint()
        gm.recompile()
        with self.assertRaisesRegex(ValueError, "map_stage_graphs"):
            InductorPass().run(
                gm,
                PassConfig(inductor_mode="full"),
                example_inputs=joint.example_inputs,
            )

    def test_missing_example_inputs_rejected(self):
        """Test the pass refuses to size regions without example inputs."""
        joint = self._traced()
        with self.assertRaisesRegex(ValueError, "example_inputs"):
            InductorPass().run(joint.graph_module, PassConfig(inductor_mode="full"))

    def test_unavailable_regional_inductor_is_noop(self):
        """Test a torch without regional_inductor degrades to a no-op."""
        joint = self._traced()
        graph_module = joint.graph_module
        graph_before = str(graph_module.graph)
        with mock.patch(
            "hyper_parallel.compile.passes.inductor_pass._regional_inductor_available",
            return_value=False,
        ):
            returned = InductorPass().run(
                graph_module,
                PassConfig(inductor_mode="full"),
                example_inputs=joint.example_inputs,
            )
        self.assertIs(returned, graph_module)
        self.assertEqual(str(graph_module.graph), graph_before)
        self.assertEqual(graph_module.meta.get("inductor_compiled"), "unavailable")

    def test_off_mode_is_noop(self):
        """Test ``inductor_mode='off'`` leaves the graph untouched."""
        joint = self._traced()
        graph_before = str(joint.graph_module.graph)
        returned = InductorPass().run(
            joint.graph_module,
            PassConfig(inductor_mode="off"),
            example_inputs=joint.example_inputs,
        )
        self.assertIs(returned, joint.graph_module)
        self.assertEqual(str(joint.graph_module.graph), graph_before)

    def test_pass_config_validates_and_canonicalizes_mode(self):
        """Test PassConfig rejects bad values and lowercases the mode."""
        with self.assertRaisesRegex(ValueError, "inductor_mode"):
            PassConfig(inductor_mode="eager")
        with self.assertRaisesRegex(ValueError, "inductor_configs"):
            PassConfig(inductor_configs=["reorder_for_peak_memory"])
        config = PassConfig(
            inductor_mode="FULL", inductor_configs={"reorder_for_peak_memory": True}
        )
        self.assertEqual(config.inductor_mode, "full")

    def test_pipeline_places_inductor_terminal(self):
        """Test InductorPass is appended last and only when enabled."""
        pipeline = PassPipeline.from_config(PassConfig(inductor_mode="regional"))
        self.assertIsInstance(pipeline.passes[-1], InductorPass)
        default_pipeline = PassPipeline.from_config(PassConfig())
        self.assertNotIsInstance(default_pipeline.passes[-1], InductorPass)

    def test_pipeline_allows_pp_with_inductor(self):
        """Test PP + Inductor pipelines without an early rejection."""
        from hyper_parallel.compile.passes.parallel.pp_pass import PpPass

        pipeline = PassPipeline.from_config(
            PassConfig(pp_enabled=True, fsdp_enabled=False, inductor_mode="regional")
        )
        self.assertIsInstance(pipeline.passes[-2], PpPass)
        self.assertIsInstance(pipeline.passes[-1], InductorPass)
        # No early guard method survives: PP is now compiled per stage.
        self.assertFalse(hasattr(pipeline, "_reject_inductor_with_pipeline_parallel"))
        # With a single-rank world PpPass is a no-op; Inductor then compiles
        # the joint graph exactly like the non-PP path.
        joint = self._traced()
        pipeline.run(joint.graph_module, example_inputs=joint.example_inputs)
        self.assertEqual(joint.graph_module.meta.get("inductor_compiled"), "regional")

    def test_trainer_config_projection_strategy(self):
        """Test inductor config is PassConfig-owned (PP-style), not projected."""
        trainer_config = SimpleNamespace(
            accelerator=SimpleNamespace(
                tp_size=1, sequence_parallel=False, loss_parallel=False
            ),
            fsdp_config=SimpleNamespace(
                dp_shard_size=1, edp_shard_size=1, reshard_after_forward=True
            ),
            compile=SimpleNamespace(
                inductor_mode="regional",
                inductor_configs={"reorder_for_peak_memory": True},
            ),
        )
        config = build_pass_config_from_trainer_config(trainer_config)
        # Like PP, inductor settings live on PassConfig only; the projected
        # config keeps the defaults and ignores CompileConfig-level overrides.
        self.assertEqual(config.inductor_mode, "off")
        self.assertIsNone(config.inductor_configs)


if __name__ == "__main__":
    unittest.main()
