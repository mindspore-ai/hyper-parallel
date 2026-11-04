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
"""CPU contracts for ACLGraph partitioning, replay guards and compiler wiring."""

import copy
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.node import _side_effectful_functions
from torch.utils._pytree import tree_leaves

from hyper_parallel.compile import GraphCompiler, GraphTrainer, PassConfig
from hyper_parallel.compile.passes import aclgraph_pass as passes
from hyper_parallel.compile.passes import aclgraph_runtime as runtime
from hyper_parallel.compile.passes import graph_metadata
from hyper_parallel.compile.passes.inductor_pass import InductorPass
from hyper_parallel.compile.passes.pipeline import PassPipeline
from hyper_parallel.compile.passes.parallel.fsdp_pass import _free_tensor_storage
from hyper_parallel.models.build_options import CompileConfig
from tests.ut.compile.inductor_stub import _fake_regional_inductor, install_regional_inductor_stub


class _Capture:
    """Simulate only native graph recording; Python/FX still execute on CPU."""

    def __init__(self) -> None:
        """Track mock recording boundaries and captured callable state."""
        self.current = None
        self.recordings = []
        self.npu = SimpleNamespace(
            NPUGraph=self._new_graph, graph=self._record, graph_pool_handle=object,
            synchronize=Mock(), get_task_queue_enable=lambda: 1,
        )

    @staticmethod
    def _new_graph():
        return SimpleNamespace(replay=lambda: None)

    @contextmanager
    def _record(self, graph, *, pool):
        del pool
        self.current = graph
        self.recordings.append(graph)
        try:
            yield
        finally:
            self.current = None

    def _run(self, function, inputs):
        """Use captured buffers when replaying an FX region on CPU."""
        saved = function(*inputs)
        if self.current is not None:
            # CPU executes during recording; native NPU recording runs on replay.
            initial = True

            def _replay():
                nonlocal initial
                if initial:
                    initial = False
                    return
                current = function(*inputs)
                for destination, value in zip(tree_leaves(saved), tree_leaves(current), strict=True):
                    if isinstance(destination, torch.Tensor):
                        destination.copy_(value)

            self.current.replay = _replay
        return saved


class TestACLGraphRuntime(unittest.TestCase):
    """Exercise guards and ownership without importing the optional backend."""

    def setUp(self) -> None:
        """Install a CPU simulation of the optional native backend."""
        self.capture = _Capture()
        self.patch = patch.object(runtime, "_get_torch_npu", return_value=SimpleNamespace(npu=self.capture.npu))
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def _wrap(self, function, *, static=(), mode="static"):
        """Construct a wrapper whose captured callable uses CPU replay."""
        coordinator = runtime._RecaptureCoordinator(coordinate_recapture=mode == "dynamic")
        wrapper = runtime.ACLGraphWrapper(
            function, static_input_indices=static,
            capture_runnable=lambda *args: self.capture._run(function, args),
            recapture_coordinator=coordinator,
        )
        coordinator.wrappers.append(wrapper)
        self.addCleanup(coordinator.teardown)
        self.addCleanup(wrapper.teardown)
        return wrapper

    def test_replay_preserves_state_and_previous_gradients(self):
        """Live state is updated in place; old returned gradients remain valid."""
        state = torch.zeros(())

        def _step(value, accumulator):
            accumulator.add_(value.sum())
            return {"state": accumulator, "gradient": value * 2}

        wrapper = self._wrap(_step, static=(1,))
        retained = []
        for index in range(1, 5):
            result = wrapper(torch.full((4,), float(index)), state)
            self.assertIs(result["state"], state)
            retained.append((result["gradient"], torch.full((4,), float(2 * index))))
            for actual, expected in retained:
                torch.testing.assert_close(actual, expected)
            self.assertEqual(state.item(), 4 * index * (index + 1) / 2)
        self.assertEqual(len(self.capture.recordings), 1)
        wrapper.teardown()
        self.assertIsNone(wrapper.graph)
        self.assertIsNone(wrapper.args)

    def test_signature_changes_are_rejected_before_replay(self):
        """Shape, stride, dtype, scalar type/value and static address are guards."""
        cases = (
            (torch.ones(3, 2), 1, "shape"),
            (torch.ones(2, 2).t(), 1, "stride"),
            (torch.ones(2, 2, dtype=torch.float64), 1, "dtype"),
            (torch.ones(2, 2), 2, "scalar"),
            (torch.ones(2, 2), True, "scalar"),
        )
        for value, scale, reason in cases:
            with self.subTest(reason=reason, scale=scale):
                wrapper = self._wrap(lambda x, multiplier: x * multiplier)
                wrapper(torch.ones(2, 2), 1)
                wrapper(torch.ones(2, 2), 1)
                with self.assertRaisesRegex(ValueError, reason):
                    wrapper(value, scale)
        wrapper = self._wrap(lambda x: x + 1, static=(0,))
        value = torch.ones(2)
        wrapper(value)
        wrapper(value)
        with self.assertRaisesRegex(ValueError, "static address"):
            wrapper(value.clone())

    def test_nested_tensor_guards_do_not_compare_tensor_values(self):
        """Nested state is borrowed and guarded by layout/address, never equality."""
        value = torch.ones(3)
        payload = {"values": [value], "flag": True}
        wrapper = self._wrap(lambda data: data["values"][0] * 2, static=(0,))
        wrapper(payload)
        wrapper(payload)
        value.fill_(3)
        torch.testing.assert_close(wrapper({"values": [value], "flag": True}), torch.full((3,), 6.0))
        with self.assertRaisesRegex(ValueError, "static address"):
            wrapper({"values": [value.clone()], "flag": True})
        with self.assertRaisesRegex(ValueError, "container structure"):
            wrapper({"values": (value,), "flag": True})

    def test_non_dense_dynamic_clone_is_rejected(self):
        """A clone that changes strides cannot become a replay input buffer."""
        wrapper = self._wrap(lambda value: value * 2)
        value = torch.ones(8)[::2]
        wrapper(value)
        with self.assertRaisesRegex(ValueError, "preserve strides"):
            wrapper(value)

    def test_dynamic_mode_recaptures_changed_layout_and_state(self):
        """Shape and state-address changes trigger new recordings, then replay."""
        wrapper = self._wrap(lambda value, state: value + state, static=(1,), mode="dynamic")
        state = torch.tensor(1.0)
        for size in (2, 2, 2, 4, 4):
            torch.testing.assert_close(wrapper(torch.ones(size), state), torch.full((size,), 2.0))
        torch.testing.assert_close(wrapper(torch.ones(4), state.clone()), torch.full((4,), 2.0))
        self.assertEqual(len(self.capture.recordings), 3)

    def test_remote_input_change_coordinates_recapture(self):
        """All ranks enter recording even when only another rank changes input."""
        wrapper = self._wrap(lambda value: value * 3, mode="dynamic")
        coordinator = wrapper.recapture_coordinator
        coordinator.initialized = True
        coordinator.group = object()
        with patch.object(runtime.torch.distributed, "all_reduce") as reduce:
            wrapper(torch.ones(2))
            wrapper(torch.ones(2))
            first_graph = wrapper.graph
            reduce.side_effect = lambda flags, **kwargs: flags.__setitem__(0, 1)
            wrapper(torch.ones(2))
            self.assertIsNot(wrapper.graph, first_graph)
            self.assertEqual(len(self.capture.recordings), 2)
            coordinator.group = None

    def test_task_queue_switch_and_missing_api(self):
        """Queue mode 2 is switched before recording, or rejected explicitly."""
        self.capture.npu.get_task_queue_enable = lambda: 2
        with self.assertRaisesRegex(ValueError, "task-queue API"):
            runtime.prepare_aclgraph_runtime()
        self.capture.npu.set_task_queue_enable = Mock()
        runtime.prepare_aclgraph_runtime()
        self.capture.npu.set_task_queue_enable.assert_called_once_with(1)


class TestACLGraphPass(unittest.TestCase):
    """Run the actual partition pass and FX interpreter against CPU tensors."""

    def setUp(self) -> None:
        """Install a CPU simulation of the optional native backend."""
        self.capture = _Capture()
        self.addCleanup(patch.stopall)
        patch.object(runtime, "_get_torch_npu", return_value=SimpleNamespace(npu=self.capture.npu)).start()
        patch.object(passes, "prepare_aclgraph_runtime").start()
        # CUDA FakeTensors are available in CPU PyTorch builds, including those
        # with no registered private NPU backend. Only metadata uses this label.
        patch.object(passes, "_is_npu_tensor", side_effect=lambda value: value.device.type == "cuda").start()
        original = torch.fx.Interpreter.boxed_run

        def _boxed_run(interpreter, args):
            values = list(args)
            return self.capture._run(lambda *inputs: original(interpreter, list(inputs)), values)

        patch.object(torch.fx.Interpreter, "boxed_run", _boxed_run).start()

    @staticmethod
    def _graph():
        """Create device work separated by an explicit CPU scalar read."""
        mode = FakeTensorMode()
        def _fake(shape):
            return FakeTensor(mode, torch.empty(shape, device="meta"), torch.device("cuda:0"))
        graph = torch.fx.Graph()
        value = graph.placeholder("value")
        state = graph.placeholder("state")
        value.meta["val"] = _fake((2, 2))
        state.meta["val"] = _fake((2, 2))
        product = graph.call_function(torch.ops.aten.mm.default, (value, state))
        product.meta["val"] = _fake((2, 2))
        total = graph.call_function(torch.ops.aten.sum.default, (product,))
        total.meta["val"] = _fake(())
        host = graph.call_function(torch.ops.aten._to_copy.default, (total,), {"device": torch.device("cpu")})
        host.meta["val"] = torch.empty(())
        scalar = graph.call_function(torch.ops.aten._local_scalar_dense.default, (host,))
        scalar.meta["val"] = 1.0
        result = graph.call_function(torch.ops.aten.mul.Scalar, (product, scalar))
        result.meta["val"] = _fake((2, 2))
        graph.output(result)
        module = torch.fx.GraphModule({}, graph)
        module.loss_dict_keys = ["loss"]
        return module

    def test_host_partition_replay_and_escaping_outputs(self):
        """Host reads run each step; captured outputs track changing batch values."""
        gm = passes.ACLGraphPass().run(self._graph(), PassConfig(aclgraph_mode="static"), static_input_indices=(1,))
        self.addCleanup(runtime.teardown_aclgraph, gm)
        self.assertEqual([entry["host"] for entry in gm.meta["aclgraph_partitions"]], [False, True, False])
        self.assertEqual(gm.loss_dict_keys, ["loss"])
        state = torch.eye(2)
        retained = []
        # Keep host scalar constant so static replay remains valid.
        for data in ([[1., 2.], [3., 4.]], [[4., 3.], [2., 1.]], [[3., 1.], [2., 4.]]):
            source = torch.tensor(data)
            actual = gm(source, state)
            retained.append((actual, source * source.sum()))
            for output, expected in retained:
                torch.testing.assert_close(output, expected)
        self.assertEqual(len(self.capture.recordings), 2)
        runtime.teardown_aclgraph(gm)
        self.assertTrue(all(wrapper.graph is None for wrapper in gm._aclgraph_wrappers))

    def test_device_wait_is_captured_even_when_registered_as_effectful(self):
        """Inductor's side-effect registry must not move HCCL waits onto host."""
        gm = self._graph()
        value = list(gm.graph.find_nodes(op="placeholder"))[0]
        with gm.graph.inserting_after(value):
            wait = gm.graph.call_function(torch.ops._c10d_functional.wait_tensor.default, (value,))
        wait.meta["val"] = value.meta["val"]
        existed = wait.target in _side_effectful_functions
        _side_effectful_functions.add(wait.target)
        try:
            self.assertFalse(passes._host_boundary(wait))
        finally:
            if not existed:
                _side_effectful_functions.remove(wait.target)

    def test_compilation_does_not_merge_across_storage_release(self):
        """A parameter read must stay before the FSDP storage-release boundary."""
        gm = self._graph()
        state = list(gm.graph.find_nodes(op="placeholder"))[1]
        product = next(node for node in gm.graph.nodes if node.target == torch.ops.aten.mm.default)
        total = next(node for node in gm.graph.nodes if node.target == torch.ops.aten.sum.default)
        with gm.graph.inserting_after(product):
            released = gm.graph.call_function(_free_tensor_storage, (state,))
        InductorPass._tag_regional(gm, {}, preserve_host_work=True)
        before = product.meta["custom"]["compile_with_inductor"]["inductor_region"]
        after = total.meta["custom"]["compile_with_inductor"]["inductor_region"]
        self.assertNotEqual(before, after)
        self.assertNotIn("compile_with_inductor", released.meta.get("custom", {}))

    def test_regional_compiled_calls_are_captured_with_host_boundaries(self):
        """Compiled functions stay inside capture while scalar reads stay eager."""
        install_regional_inductor_stub()
        source = self._graph()
        config = PassConfig(fsdp_enabled=False, inductor_mode="regional", aclgraph_mode="static")
        examples = tuple(node.meta["val"] for node in source.graph.find_nodes(op="placeholder"))
        with patch("hyper_parallel.compile.passes.inductor_pass._propagate_fake_vals", return_value=source), \
             patch("torch.fx.passes.regional_inductor.regional_inductor", side_effect=_fake_regional_inductor):
            InductorPass().run(source, config, example_inputs=examples)
        with patch.object(graph_metadata, "FakeTensorProp") as propagation:
            gm = passes.ACLGraphPass().run(source, config, example_inputs=examples, static_input_indices=(1,))
            propagation.assert_not_called()
        self.addCleanup(runtime.teardown_aclgraph, gm)
        captured = {entry["name"] for entry in gm.meta["aclgraph_partitions"] if entry["capture"]}
        compiled = [node for name, region in gm.named_children() if name in captured
                    for node in region.graph.nodes if node.meta.get(graph_metadata.COMPILED_REGION_META)]
        self.assertTrue(compiled)
        self.assertEqual([entry["host"] for entry in gm.meta["aclgraph_partitions"]], [False, True, False])
        state = torch.eye(2)
        for data in ([[1., 2.], [3., 4.]], [[4., 3.], [2., 1.]], [[3., 1.], [2., 4.]]):
            value = torch.tensor(data)
            torch.testing.assert_close(gm(value, state), value * value.sum())
        self.assertEqual(len(self.capture.recordings), 2)

    def test_opaque_mutations_keep_original_input_storage(self):
        """A compiled in-place update must not be staged as an ordinary input."""
        mode = FakeTensorMode()
        value = FakeTensor(mode, torch.empty(2, 2, device="meta"), torch.device("cuda:0"))
        graph = torch.fx.Graph()
        state = graph.placeholder("state")
        state.meta["val"] = value
        update = graph.call_function(torch.ops.aten.add_.Tensor, (state, state))
        update.meta["val"] = value
        graph.output(update)
        gm = torch.fx.GraphModule({}, graph)
        examples = (value,)
        config = PassConfig(fsdp_enabled=False, inductor_mode="regional", aclgraph_mode="static")
        install_regional_inductor_stub()
        with patch("hyper_parallel.compile.passes.inductor_pass._propagate_fake_vals", return_value=gm), \
             patch("torch.fx.passes.regional_inductor.regional_inductor", side_effect=_fake_regional_inductor):
            InductorPass().run(gm, config, example_inputs=examples)
        self.assertEqual(passes._mutated_input_indices(gm), {0})
        gm = passes.ACLGraphPass().run(gm, config, example_inputs=examples)
        self.addCleanup(runtime.teardown_aclgraph, gm)
        actual = torch.ones(2, 2)
        for index in range(1, 4):
            result = gm(actual)
            self.assertIs(result, actual)
            torch.testing.assert_close(actual, torch.full((2, 2), float(2 ** index)))

    def test_cpu_and_disabled_graphs_do_not_load_backend(self):
        """CPU use and defaults retain normal FX execution."""
        gm = make_fx(lambda value: value.sin())(torch.ones(2))
        with patch.object(passes, "prepare_aclgraph_runtime") as prepare:
            self.assertIs(passes.ACLGraphPass().run(gm, PassConfig(aclgraph_mode="static")), gm)
            device_graph = self._graph()
            self.assertIs(passes.ACLGraphPass().run(device_graph, PassConfig()), device_graph)
            prepare.assert_not_called()

    def test_mutation_alias_is_static(self):
        """Mutation through a view is traced back to the original input storage."""
        gm = make_fx(lambda value: value.view(-1).add_(1), tracing_mode="fake")(torch.ones(2, 2))
        self.assertEqual(passes._mutated_input_indices(gm), {0})

    def test_metadata_uses_live_shards_instead_of_full_trace_examples(self):
        """Post-FSDP propagation must size collectives from physical shards."""
        model = torch.nn.Linear(4, 4, bias=False)
        gm = make_fx(lambda weight, value: value.mm(weight.t()), tracing_mode="fake")(
            torch.ones(4, 4), torch.ones(3, 4)
        )
        placeholders = list(gm.graph.find_nodes(op="placeholder"))
        examples = tuple(node.meta["val"] for node in placeholders)
        gm.state_fqns = ["weight"]
        weight = placeholders[0]
        users = list(weight.users)
        with gm.graph.inserting_after(weight):
            gathered = gm.graph.call_function(torch.ops.aten.repeat.default, (weight, (2, 1)))
        for user in users:
            user.replace_input_with(weight, gathered)
        gm.recompile()
        model.weight.data = torch.ones(2, 4)
        passes._propagate_metadata(gm, examples, model)
        self.assertEqual(tuple(weight.meta["val"].shape), (2, 4))
        self.assertEqual(tuple(gathered.meta["val"].shape), (4, 4))
        output = next(node for node in gm.graph.nodes if node.op == "output")
        self.assertEqual(tuple(output.args[0].meta["val"].shape), (3, 4))

    def test_python_helpers_run_on_every_replay(self):
        """Python helpers remain eager even when all their metadata is on device."""
        calls = []
        module = self._graph()
        first = next(node for node in module.graph.nodes if node.op == "call_function")
        value = list(module.graph.find_nodes(op="placeholder"))[0]

        def _callback(tensor):
            calls.append(1)
            return tensor + 1

        with module.graph.inserting_before(first):
            callback = module.graph.call_function(_callback, (value,))
        callback.meta["val"] = first.meta["val"]
        first.replace_input_with(value, callback)
        module.recompile()
        gm = passes.ACLGraphPass().run(module, PassConfig(aclgraph_mode="dynamic"), static_input_indices=(1,))
        self.addCleanup(runtime.teardown_aclgraph, gm)
        state = torch.eye(2)
        for index in range(3):
            source = torch.full((2, 2), float(index))
            expected = (source + 1) * (source + 1).sum()
            torch.testing.assert_close(gm(source, state), expected)
        self.assertEqual(len(calls), 3)

    def test_pipeline_selects_capture_after_scheduling(self):
        """Only explicit capture modes append the pass to the existing pipeline."""
        for mode in ("off", "static", "dynamic"):
            pipeline = PassPipeline.from_config(PassConfig(aclgraph_mode=mode))
            names = [entry.name for entry in pipeline.passes]
            self.assertEqual("aclgraph" in names, mode != "off")
            if mode != "off":
                self.assertIsInstance(pipeline.passes[-1], passes.ACLGraphPass)


class TestACLGraphIntegration(unittest.TestCase):
    """Check configuration projection, replacement graphs and cleanup owners."""

    def test_config_validation_and_projection(self):
        """Capture is opt-in, YAML modes are validated, and projection preserves it."""
        self.assertEqual(PassConfig().aclgraph_mode, "off")
        for mode in (None, "unknown", True):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                PassConfig(aclgraph_mode=mode)
        for options in ({"pp_enabled": True}, {"inductor_mode": "full"}):
            with self.assertRaisesRegex(ValueError, "ACLGraph currently requires"):
                PassConfig(aclgraph_mode="static", **options)
        with self.assertRaisesRegex(ValueError, "use_joint_graph"):
            CompileConfig(aclgraph_mode="static")
        config = SimpleNamespace(
            compile=CompileConfig(enabled=True, use_joint_graph=True, aclgraph_mode="dynamic"),
            fsdp_config=SimpleNamespace(dp_shard_size=1, edp_shard_size=1),
        )
        compiler = GraphCompiler(torch.nn.Linear(2, 2), lambda model, value: model(value).sum(),
                                 trainer_config=config, device=torch.device("cpu"))
        self.assertEqual(compiler.pass_config.aclgraph_mode, "dynamic")

    def test_regional_compilation_precedes_capture(self):
        """The pass pipeline accepts and orders regional compilation then capture."""
        config = PassConfig(fsdp_enabled=False, inductor_mode="regional", aclgraph_mode="static")
        pipeline = PassPipeline.from_config(config)
        self.assertEqual([entry.name for entry in pipeline.passes[-2:]], ["inductor_compilation", "aclgraph"])

    def test_real_compiled_cpu_metadata_is_not_reexecuted(self):
        """Retain the region metadata produced by the real CPU compiler."""
        install_regional_inductor_stub()
        source = make_fx(lambda value: (value.sin() * value).sum(), tracing_mode="fake")(torch.ones(2, 4))
        examples = tuple(node.meta["val"] for node in source.graph.find_nodes(op="placeholder"))
        InductorPass().run(source, PassConfig(fsdp_enabled=False, inductor_mode="regional"), example_inputs=examples)
        self.assertTrue(any(node.meta.get(graph_metadata.COMPILED_REGION_META) for node in source.graph.nodes))
        with patch.object(graph_metadata, "FakeTensorProp") as propagation:
            result = passes.ACLGraphPass().run(
                source, PassConfig(fsdp_enabled=False, inductor_mode="regional", aclgraph_mode="static"),
                example_inputs=examples,
            )
            propagation.assert_not_called()
        self.assertIs(result, source)
        for value in (torch.ones(2, 4), torch.full((2, 4), 0.5)):
            torch.testing.assert_close(source(value), (value.sin() * value).sum())

    def test_compiler_executes_replacement_graph(self):
        """The compiler must use the graph returned by a partitioning pass."""
        model = torch.nn.Linear(2, 2)
        compiler = GraphCompiler(model, lambda module, value: module(value).sum(),
                                 pass_config=PassConfig(fsdp_enabled=False), device=torch.device("cpu"))
        transformed = []

        def _replace(graph, **kwargs):
            del kwargs
            result = copy.deepcopy(graph)
            result.meta["replacement"] = True
            transformed.append(result)
            return result

        with patch("hyper_parallel.compile.compiler.PassPipeline.from_config") as factory:
            factory.return_value.run.side_effect = _replace
            compiler.forward_backward(value=torch.ones(2, 2))
        self.assertIs(compiler._joint_graph.graph_module, transformed[0])
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
        with patch("hyper_parallel.compile.compiler.teardown_aclgraph") as teardown:
            compiler.close()
            teardown.assert_called_once_with(transformed[0])
        trainer = GraphTrainer.__new__(GraphTrainer)
        trainer._compiler = Mock()
        trainer.close()
        trainer._compiler.close.assert_called_once_with()
