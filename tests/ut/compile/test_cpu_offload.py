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
"""Unit tests for graph-mode CPU offload (attention activation swap).

Covers config reuse (``activation_swap`` projection / validation), pipeline
placement, attention-scope selection, budget ordering, view redirection, and
numerical parity using CPU-safe stand-ins for the upstream ``ao::`` ops.
"""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import fx, nn

from hyper_parallel.compile.pass_config import (
    PassConfig,
    build_pass_config_from_trainer_config,
)
from hyper_parallel.compile import GraphCompiler
from hyper_parallel.compile.passes.cpu_offload import CpuOffloadPass
from hyper_parallel.compile.passes import cpu_offload as offload_module
from hyper_parallel.compile.passes.pipeline import PassPipeline
from hyper_parallel.compile.tracer.graph_tracer import (
    run_traced_graph,
    trace_model_graph,
)
from hyper_parallel.models._transformers.model_builder import (
    _apply_activation_features,
)
from hyper_parallel.models.build_options import CompileConfig


class _Attention(nn.Module):
    """Small multi-head-free attention with saved q/k/v and softmax values."""

    def __init__(self, dim: int) -> None:
        """Create the three projections."""
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run scaled dot-product attention."""
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)
        # ``@`` lowers through the composite ``aten.matmul`` kernel on older
        # torch releases, which specializes symbolic sizes while broadcasting
        # and defeats the dynamic-shape trace. Use the explicit ``bmm`` here so
        # this test exercises CPU offload, not that kernel's shape handling.
        scores = torch.softmax(torch.bmm(query, key.transpose(-1, -2)), dim=-1)
        return torch.bmm(scores, value)


class _Block(nn.Module):
    """Transformer-like block with an offload scope and an out-of-scope MLP."""

    def __init__(self, dim: int) -> None:
        """Create attention and the MLP."""
        super().__init__()
        self.self_attn = _Attention(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run attention then the residual MLP."""
        return self.mlp(self.self_attn(x)) + x


class _Model(nn.Module):
    """Two-block model with the attention module named ``self_attn``."""

    def __init__(self, dim: int = 64, num_blocks: int = 2) -> None:
        """Register the repeated blocks."""
        super().__init__()
        self.layers = nn.ModuleList(_Block(dim) for _ in range(num_blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run every block."""
        for layer in self.layers:
            x = layer(x)
        return x


class _NoAttentionModel(nn.Module):
    """Model without the eager attention-swap discovery marker."""

    def __init__(self, dim: int = 64) -> None:
        """Create a plain MLP stack."""
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(dim, dim) for _ in range(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the linear stack."""
        for layer in self.layers:
            x = layer(x)
        return x


def _train_fn(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return a scalar loss for tracing."""
    return model(x).float().pow(2).mean()


def _trace(model: nn.Module, x: torch.Tensor):
    """Trace the joint forward/backward graph."""
    return trace_model_graph(model, _train_fn, {"x": x})


def _ao_name(target):
    """Normalize real ``ao::`` overloads and CPU-safe stand-ins to a short name."""
    if target is _fake_offload or str(target) == "ao.offload.default":
        return "ao.offload"
    if target is _fake_reload or str(target) == "ao.reload.default":
        return "ao.reload"
    if (
        target in (_fake_wait, _fake_wait_free)
        or str(target) == "ao.wait_tensor.default"
    ):
        return "ao.wait"
    return None


def _ao_nodes(graph_module):
    """Return inserted ``ao.*`` nodes (real or stand-in) in graph order."""
    return [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function" and _ao_name(node.target) is not None
    ]


def _offload_sources(graph_module):
    """Return the producer nodes wrapped by ``ao.offload``."""
    return [
        node.args[0]
        for node in _ao_nodes(graph_module)
        if _ao_name(node.target) == "ao.offload"
    ]


def _fake_offload(tensor: torch.Tensor) -> torch.Tensor:
    """CPU-safe ``ao::offload`` stand-in."""
    return tensor.detach().to(device="cpu").clone()


def _fake_reload(tensor, device, original_size=None, original_stride=None):
    """CPU-safe ``ao::reload`` stand-in."""
    size = original_size if original_size is not None else list(tensor.shape)
    stride = original_stride if original_stride is not None else list(tensor.stride())
    return torch.empty_strided(
        tuple(size), tuple(stride), dtype=tensor.dtype, device=device
    ).copy_(tensor)


def _fake_wait(tensor, keepalive=None, last_use_of_storage=None):  # pylint: disable=unused-argument
    """CPU-safe ``ao::wait_tensor`` stand-in."""
    return tensor


def _fake_wait_free(tensor, keepalive=None, last_use_of_storage=None):  # pylint: disable=unused-argument
    """``ao::wait_tensor`` stand-in that frees ``keepalive`` storage like the real op.

    The real op calls ``keepalive.untyped_storage().resize_(0)`` after the async
    transfer. Emulating that here makes a premature wait observable instead of
    being hidden by a no-op clone.
    """
    if keepalive is not None:
        keepalive.untyped_storage().resize_(0)
    return tensor


def _patch_fake_ao(free_waits: bool = False):
    """Patch the pass onto CPU-safe stand-ins, independent of the torch build."""
    wait = _fake_wait_free if free_waits else _fake_wait
    return patch.multiple(
        offload_module,
        _AO_OFFLOAD=_fake_offload,
        _AO_RELOAD=_fake_reload,
        _AO_WAIT=wait,
        _AO_OPS=frozenset((_fake_offload, _fake_reload, wait)),
        _AO_AVAILABLE=True,
    )


class TestCpuOffloadConfig(unittest.TestCase):
    """The dynamic ``activation_swap`` field drives graph-mode offload."""

    def test_defaults_are_off(self):
        """Default config keeps offload disabled and uses finite budgets."""
        cfg = PassConfig()
        self.assertEqual(cfg.activation_swap, "none")
        self.assertEqual(cfg.offload_budget_gb, 64.0)
        self.assertEqual(cfg.offload_prefetch_n_layers, 1)

    def test_mode_acceptance_and_combination_bans(self):
        """Accept ``attention`` and reject unknown modes / forbidden combos."""
        cfg = PassConfig(activation_swap="attention")
        self.assertEqual(cfg.activation_swap, "attention")
        with self.assertRaisesRegex(ValueError, "activation_swap"):
            PassConfig(activation_swap="all")
        with self.assertRaisesRegex(ValueError, "activation checkpointing"):
            PassConfig(activation_swap="attention", ac_mode="full")
        with self.assertRaisesRegex(ValueError, "pipeline"):
            PassConfig(activation_swap="attention", pp_enabled=True)

    def test_budget_and_prefetch_validation(self):
        """Graph-only knobs still reject nonsense values."""
        for invalid in (-1, True, "100", float("nan")):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "offload_budget_gb"):
                    PassConfig(activation_swap="attention", offload_budget_gb=invalid)
        for invalid in (-1, True, 1.5):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "offload_prefetch_n_layers"):
                    PassConfig(
                        activation_swap="attention",
                        offload_prefetch_n_layers=invalid,
                    )

    def test_trainer_projection_reuses_activation_swap(self):
        """``TrainerConfig.activation_swap`` lands on ``PassConfig``."""
        config = SimpleNamespace(
            activation_swap="attention",
            fsdp_config=SimpleNamespace(dp_shard_size=1, edp_shard_size=1),
            compile=CompileConfig(),
        )
        result = build_pass_config_from_trainer_config(config)
        self.assertEqual(result.activation_swap, "attention")


class TestGraphModeSwapWiring(unittest.TestCase):
    """Graph mode must leave the live model unwrapped for the pass."""

    def test_graph_mode_controls_eager_wrapper(self):
        """Graph mode skips the eager wrapper; the default path applies it."""
        model = nn.Linear(4, 4)
        with patch(
            "hyper_parallel.models._transformers.model_builder.apply_attention_swap"
        ) as apply_swap:
            result = _apply_activation_features(
                model,
                "off",
                "attention",
                False,
                SimpleNamespace(pp_size=1),
                apply_swap=False,
            )
        self.assertIs(result, model)
        apply_swap.assert_not_called()

        with patch(
            "hyper_parallel.models._transformers.model_builder.apply_attention_swap",
            return_value=model,
        ) as apply_swap:
            result = _apply_activation_features(
                model, "off", "attention", False, SimpleNamespace(pp_size=1)
            )
        self.assertIs(result, model)
        apply_swap.assert_called_once_with(model, "attention")


class TestCpuOffloadPipeline(unittest.TestCase):
    """Pipeline placement follows AC and precedes FSDP."""

    def test_pass_is_added_only_for_attention_swap(self):
        """``attention`` adds exactly one pass; the default adds none."""
        enabled = PassPipeline(
            PassConfig(
                fsdp_enabled=False,
                enable_overlap=False,
                activation_swap="attention",
            )
        ).build()
        enabled_names = [type(p).__name__ for p in enabled.passes]
        self.assertEqual(enabled_names.count("CpuOffloadPass"), 1)

        disabled = PassPipeline(
            PassConfig(fsdp_enabled=False, enable_overlap=False)
        ).build()
        disabled_names = [type(p).__name__ for p in disabled.passes]
        self.assertNotIn("CpuOffloadPass", disabled_names)


class TestCpuOffloadPass(unittest.TestCase):
    """The pass selects attention activations and rewrites the joint graph."""

    def setUp(self) -> None:
        """Trace a representative two-block attention model."""
        torch.manual_seed(0)
        fake_ao = _patch_fake_ao()
        fake_ao.start()
        self.addCleanup(fake_ao.stop)
        self.model = _Model(dim=512, num_blocks=2)
        self.x = torch.randn(16, 32, 512)
        self.joint = _trace(self.model, self.x)

    def _run(self, **kwargs):
        """Run the pass with the requested graph-only knobs."""
        config = PassConfig(
            fsdp_enabled=False,
            enable_overlap=False,
            activation_swap="attention",
            **kwargs,
        )
        return CpuOffloadPass().run(self.joint.graph_module, config, model=self.model)

    def test_rewrites_attention_activations(self):
        """Offload/reload/wait are paired, attention-scoped, and waited."""
        graph_module = self._run()
        counts = {}
        for node in _ao_nodes(graph_module):
            name = _ao_name(node.target)
            counts[name] = counts.get(name, 0) + 1
        self.assertGreater(counts.get("ao.offload", 0), 0)
        self.assertEqual(counts.get("ao.offload"), counts.get("ao.reload"))
        self.assertEqual(counts.get("ao.offload") * 2, counts.get("ao.wait"))

        sources = _offload_sources(graph_module)
        self.assertTrue(sources, "expected attention activations to be offloaded")
        for source in sources:
            stack = source.meta.get("nn_module_stack") or {}
            fqns = [fqn for fqn, _ in stack.values()]
            self.assertTrue(
                any(fqn.rsplit(".", 1)[-1] == "self_attn" for fqn in fqns),
                f"offloaded a non-attention activation: {source.name} {fqns}",
            )

        reloads = [
            node
            for node in _ao_nodes(graph_module)
            if _ao_name(node.target) == "ao.reload"
        ]
        self.assertTrue(reloads)
        for reload in reloads:
            self.assertTrue(reload.users)
            self.assertTrue(
                all(_ao_name(user.target) == "ao.wait" for user in reload.users),
                f"reload {reload.name} is not waited before use",
            )

        # Every forward wait must pin the storage free to the last forward read
        # through ``last_use_of_storage`` so a reorder pass cannot hoist it.
        forward_waits = [
            node
            for node in _ao_nodes(graph_module)
            if _ao_name(node.target) == "ao.wait" and len(node.args) == 3
        ]
        self.assertTrue(forward_waits)
        for wait in forward_waits:
            self.assertIsNotNone(
                wait.args[2],
                f"forward wait {wait.name} has no last-use storage edge",
            )

    def test_budget_selects_largest_first(self):
        """A small budget selects fewer activations than the default."""
        self.joint = _trace(self.model, self.x)
        unlimited = self._run(offload_budget_gb=0.0)
        self.joint = _trace(self.model, self.x)
        limited = self._run(offload_budget_gb=0.003)
        self.assertGreater(len(_offload_sources(limited)), 0)
        self.assertLess(
            len(_offload_sources(limited)), len(_offload_sources(unlimited))
        )

    def test_missing_attention_raises_like_eager(self):
        """The eager contract requires ``self_attn``; graph mode matches it."""
        model = _NoAttentionModel(dim=512)
        joint = _trace(model, torch.randn(16, 32, 512))
        with self.assertRaisesRegex(ValueError, "self_attn"):
            CpuOffloadPass().run(
                joint.graph_module,
                PassConfig(
                    fsdp_enabled=False,
                    enable_overlap=False,
                    activation_swap="attention",
                ),
                model=model,
            )

    def test_unavailable_ao_ops_raise(self):
        """Requesting offload without the ``ao::`` ops fails loudly."""
        with patch.object(offload_module, "_AO_AVAILABLE", False):
            with self.assertRaisesRegex(RuntimeError, "offload_ops"):
                self._run()

    def test_prefetch_moves_reloads_earlier(self):
        """Prefetch pulls at least one reload toward an earlier region."""
        self.joint = _trace(self.model, self.x)
        config = PassConfig(
            fsdp_enabled=False,
            enable_overlap=False,
            activation_swap="attention",
            offload_prefetch_n_layers=0,
        )
        graph_module = CpuOffloadPass().run(
            self.joint.graph_module, config, model=self.model
        )
        roots = offload_module._attention_roots(graph_module, self.model)
        moved = offload_module._prefetch_reloads(graph_module, 1, roots)
        self.assertGreater(moved, 0, f"expected prefetched reloads, moved={moved}")

    def test_dynamic_shapes_are_supported(self):
        """Batch/sequence-dynamic trace: offload runs at another size, adds no guard."""
        dynamic_joint = trace_model_graph(
            self.model, _train_fn, {"x": self.x}, dynamic_arg_dims={"x": [0, 1]}
        )
        config = PassConfig(
            fsdp_enabled=False,
            enable_overlap=False,
            activation_swap="attention",
            offload_prefetch_n_layers=1,
        )
        guards_before = len(dynamic_joint.input_guards.shape_env.guards)
        graph_module = CpuOffloadPass().run(
            dynamic_joint.graph_module, config, model=self.model
        )
        self.assertGreater(len(_offload_sources(graph_module)), 0)
        self.assertEqual(
            len(dynamic_joint.input_guards.shape_env.guards),
            guards_before,
            "CPU offload must not specialize the dynamic graph with a shape guard",
        )
        # Symbolic sizes must never be baked into the graph: the reload derives
        # its runtime shape/stride from the offloaded host tensor.
        for node in _ao_nodes(graph_module):
            if _ao_name(node.target) == "ao.reload":
                self.assertIsNone(node.args[2])
                self.assertIsNone(node.args[3])

        x_smaller = torch.randn(8, 32, 512)
        reference = copy.deepcopy(self.model)
        reference_loss = _train_fn(reference, x_smaller)
        reference_loss.backward()
        expected_grads = {
            name: param.grad.clone() for name, param in reference.named_parameters()
        }

        # ``refresh`` picks up any guard a pass added; the new shape must pass.
        dynamic_joint.input_guards.refresh()
        dynamic_joint.input_guards.validate([x_smaller])
        loss, grads, _ = run_traced_graph(dynamic_joint, self.model, {"x": x_smaller})
        self.assertLess((loss - reference_loss).abs().item(), 1e-6)
        actual_grads = dict(zip(dict(self.model.named_parameters()), grads))
        for name, expected in expected_grads.items():
            torch.testing.assert_close(
                actual_grads[name], expected, rtol=1e-5, atol=1e-6
            )

    def test_numerical_parity_with_fake_ao_ops(self):
        """Offload/reload preserves loss and gradients (CPU-safe stand-ins)."""
        reference = copy.deepcopy(self.model)
        reference_loss = _train_fn(reference, self.x)
        reference_loss.backward()
        expected_grads = {
            name: param.grad.clone() for name, param in reference.named_parameters()
        }

        with _patch_fake_ao():
            self._run(offload_prefetch_n_layers=1)
            loss, grads, _ = run_traced_graph(self.joint, self.model, {"x": self.x})

        self.assertLess((loss - reference_loss).abs().item(), 1e-6)
        actual_grads = dict(zip(dict(self.model.named_parameters()), grads))
        for name, expected in expected_grads.items():
            torch.testing.assert_close(
                actual_grads[name], expected, rtol=1e-5, atol=1e-6
            )

    def test_graph_compiler_integration(self):
        """``GraphCompiler`` forwards the live model to the offload pass."""
        reference = copy.deepcopy(self.model)
        reference_loss = _train_fn(reference, self.x)

        with _patch_fake_ao():
            compiler = GraphCompiler(
                self.model,
                _train_fn,
                pass_config=PassConfig(
                    fsdp_enabled=False,
                    enable_overlap=False,
                    activation_swap="attention",
                ),
                device=torch.device("cpu"),
            )
            loss, _ = compiler.forward_backward(x=self.x)
            offloaded = any(
                node.target is _fake_offload
                for node in compiler._joint_graph.graph_module.graph.nodes
            )

        self.assertLess((loss - reference_loss).abs().item(), 1e-6)
        self.assertTrue(offloaded)


class TestInPlaceWriteSafety(unittest.TestCase):
    """In-place writes with no tensor return must still poison their storage."""

    @staticmethod
    def _mutating_model(dim: int = 512) -> nn.Module:
        """Build a model whose attention mutates a saved activation in place."""

        class _MutatingAttention(nn.Module):
            """Mutate a large activation through ``_foreach_add_`` (returns ``()``)."""

            def __init__(self) -> None:
                """Create the elementwise scale used to form the activation."""
                super().__init__()
                self.weight = nn.Parameter(torch.full((dim,), 0.7))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """Read the activation through an alias after mutating it."""
                hidden = x * self.weight
                view = hidden.view_as(hidden)
                torch._foreach_add_([hidden], 0.5)
                return view.sin().square().mean(dim=-1)

        class _MutatingModel(nn.Module):
            """Single attention block exposing the ``self_attn`` marker."""

            def __init__(self) -> None:
                """Register the mutating attention."""
                super().__init__()
                self.self_attn = _MutatingAttention()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """Run the mutating attention."""
                return self.self_attn(x)

        return _MutatingModel()

    def test_foreach_write_without_return_is_not_offloaded(self):
        """Schema write args skip offload even when the op returns no tensor."""
        torch.manual_seed(0)
        model = self._mutating_model()
        x = torch.randn(16, 32, 512)
        joint = _trace(model, x)
        graph = joint.graph_module.graph
        order = {node: index for index, node in enumerate(graph.nodes)}

        write = next(
            (
                node
                for node in graph.nodes
                if node.op == "call_function"
                and node.target is torch.ops.aten._foreach_add_.Scalar
            ),
            None,
        )
        if write is None:
            self.skipTest("this torch build decomposes the foreach in-place add")
        self.assertIsNone(write.meta.get("val"), "foreach write must return no tensor")
        producer = next(
            node
            for node in graph.nodes
            if node.op == "call_function"
            and node.target is torch.ops.aten.mul.Tensor
            and not node.meta.get("autograd_backward", False)
        )

        with _patch_fake_ao():
            storage = offload_module._StorageIndex.build(graph, order)
            self.assertIn(
                offload_module._storage_key(producer.meta["val"]),
                storage.unsafe,
                "storage written by the no-return foreach op must be unsafe",
            )

            graph_module = CpuOffloadPass().run(
                joint.graph_module,
                PassConfig(
                    fsdp_enabled=False,
                    enable_overlap=False,
                    activation_swap="attention",
                ),
                model=model,
            )
            sources = _offload_sources(graph_module)
            self.assertTrue(sources, "expected other attention activations to offload")
            self.assertNotIn(
                producer, sources, "written activation must stay on device"
            )

            reference = copy.deepcopy(model)
            reference_loss = _train_fn(reference, x)
            reference_loss.backward()
            expected_grads = {
                name: param.grad.clone() for name, param in reference.named_parameters()
            }
            loss, grads, _ = run_traced_graph(joint, model, {"x": x})

        self.assertLess((loss - reference_loss).abs().item(), 1e-6)
        actual_grads = dict(zip(dict(model.named_parameters()), grads))
        for name, expected in expected_grads.items():
            torch.testing.assert_close(
                actual_grads[name], expected, rtol=1e-5, atol=1e-6
            )


class TestWriteDetectionFallbacks(unittest.TestCase):
    """Unprovable writes must fall back to marking every tensor argument."""

    @staticmethod
    def _placeholder(graph, name, shape=(4, 4)):
        """Create a tensor placeholder carrying a fake value."""
        node = graph.placeholder(name)
        node.meta["val"] = torch.empty(*shape)
        return node

    def test_unstable_storage_identity_is_unknown(self):
        """A storage without ``_cdata`` yields ``None`` instead of ``id()``."""

        class _StorageWithoutCData:
            """Storage stand-in missing the stable ``_cdata`` handle."""

        class _ValueWithoutCData:
            """Tensor stand-in returning the unstable storage."""

            def untyped_storage(self):
                """Return a storage object without ``_cdata``."""
                return _StorageWithoutCData()

        self.assertIsNone(offload_module._storage_key(_ValueWithoutCData()))

    def test_node_without_schema_marks_tensor_arguments(self):
        """A schema-less callable cannot be proven pure, so args are unsafe."""
        graph = fx.Graph()
        x = self._placeholder(graph, "x")

        def unknown(*args):  # pylint: disable=unused-argument
            """Schema-less callable stand-in with unknown in-place effects."""

        node = graph.call_function(unknown, args=(x,))
        self.assertEqual(
            offload_module._written_storage_keys(node),
            [offload_module._storage_key(x.meta["val"])],
        )

    def test_unresolved_write_argument_falls_back_to_tensor_arguments(self):
        """An out-of-range write position marks every tensor argument."""
        graph = fx.Graph()
        read = self._placeholder(graph, "read")

        def mutating(*args):  # pylint: disable=unused-argument
            """Callable carrying a schema whose write position cannot resolve."""

        mutating._schema = SimpleNamespace(
            is_mutable=True,
            arguments=[
                SimpleNamespace(name="read", kwarg_only=False, alias_info=None),
                SimpleNamespace(
                    name="write",
                    kwarg_only=False,
                    alias_info=SimpleNamespace(is_write=True),
                ),
            ],
        )
        # Only the read argument is supplied, so the write index cannot resolve.
        node = graph.call_function(mutating, args=(read,))
        self.assertEqual(
            offload_module._written_storage_keys(node),
            [offload_module._storage_key(read.meta["val"])],
        )


class TestOffloadOrderingSafety(unittest.TestCase):
    """Storage-free ordering must survive reordering passes."""

    def test_multi_output_last_use_is_skipped(self):
        """A tuple-valued last forward consumer has no tensor edge; skip it."""
        graph = fx.Graph()
        x = graph.placeholder("x")
        producer = graph.call_function(torch.ops.aten.relu.default, (x,))
        backward = graph.call_function(torch.ops.aten.mul.Tensor, (producer, x))
        backward.meta["autograd_backward"] = True
        tuple_consumer = graph.call_function(torch.ops.aten.max.dim, (producer, 0))
        tuple_consumer.meta["val"] = (object(), object())
        graph.output((tuple_consumer, backward))
        fx.GraphModule(torch.nn.Module(), graph)

        order = {node: index for index, node in enumerate(graph.nodes)}
        self.assertIsNone(
            offload_module._collect_offload_info(producer, order),
            "candidate whose last forward consumer returns a tuple must be skipped",
        )


class _AliasAttention(nn.Module):
    """Attention-like module with a full-size ``_unsafe_view`` alias of a base."""

    def __init__(self, dim: int = 512) -> None:
        """Create the two branch weights."""
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(dim, dim))
        self.w2 = nn.Parameter(torch.randn(dim // 2, dim * 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Use the base and its alias in separate, independently-live branches."""
        base = torch.relu(x)
        alias = torch.ops.aten._unsafe_view.default(
            base, [x.shape[0] // 2, x.shape[1] * 2]
        )
        return (base * self.w1).sum() + (alias * self.w2).sum()


class _AliasModel(nn.Module):
    """Alias model whose offload scope is named ``self_attn``."""

    def __init__(self, dim: int = 512) -> None:
        """Register the attention module."""
        super().__init__()
        self.self_attn = _AliasAttention(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the attention module."""
        return self.self_attn(x)


class TestStorageAliasing(unittest.TestCase):
    """A full-size ``_unsafe_view`` must share one offload and one lifetime.

    ``aten::_unsafe_view`` has no schema ``alias_info`` but its output aliases
    the input's storage, so schema-based view detection alone would offload the
    base and the alias separately and release shared storage too early.
    """

    def setUp(self) -> None:
        """Patch CPU-safe ao stand-ins and trace an aliasing graph."""
        self.patch = _patch_fake_ao()
        self.patch.start()
        torch.manual_seed(0)
        self.model = _AliasModel()
        self.x = torch.randn(512, 512)
        self.joint = _trace(self.model, self.x)

    def tearDown(self) -> None:
        """Restore the real ao ops."""
        self.patch.stop()

    @staticmethod
    def _config() -> PassConfig:
        """Offload config without prefetch, so positions are easy to assert."""
        return PassConfig(
            fsdp_enabled=False,
            enable_overlap=False,
            activation_swap="attention",
            offload_prefetch_n_layers=0,
        )

    def test_alias_is_not_a_second_offload(self):
        """Only the storage owner is offloaded, not its ``_unsafe_view`` alias."""
        graph_module = CpuOffloadPass().run(
            self.joint.graph_module, self._config(), model=self.model
        )
        sources = [
            node.args[0]
            for node in _ao_nodes(graph_module)
            if _ao_name(node.target) == "ao.offload"
        ]
        targets = [str(node.target) for node in sources]
        self.assertEqual(targets.count("aten.relu.default"), 1)
        self.assertNotIn("aten._unsafe_view.default", targets)

    def test_wait_after_every_alias_forward_read(self):
        """The shared-storage free must come after both forward branches."""
        graph_module = CpuOffloadPass().run(
            self.joint.graph_module, self._config(), model=self.model
        )
        order = {node: index for index, node in enumerate(graph_module.graph.nodes)}
        forward_muls = [
            node
            for node in graph_module.graph.nodes
            if node.op == "call_function"
            and str(node.target) == "aten.mul.Tensor"
            and not node.meta.get("autograd_backward")
        ]
        self.assertEqual(len(forward_muls), 2)
        last_read = max(order[node] for node in forward_muls)
        waits = [
            node
            for node in _ao_nodes(graph_module)
            if _ao_name(node.target) == "ao.wait" and len(node.args) == 3
        ]
        self.assertTrue(waits)
        for wait in waits:
            self.assertGreater(
                order[wait], last_read, "forward wait frees storage before alias read"
            )

    def test_freeing_wait_keeps_loss_and_grads(self):
        """Wait that frees storage still yields eager-identical loss/grads."""
        reference = copy.deepcopy(self.model)
        reference_loss = _train_fn(reference, self.x)
        reference_loss.backward()
        expected_grads = {
            name: param.grad.clone() for name, param in reference.named_parameters()
        }

        with _patch_fake_ao(free_waits=True):
            CpuOffloadPass().run(
                self.joint.graph_module, self._config(), model=self.model
            )
            loss, grads, _ = run_traced_graph(self.joint, self.model, {"x": self.x})

        self.assertLess((loss - reference_loss).abs().item(), 1e-6)
        actual_grads = dict(zip(dict(self.model.named_parameters()), grads))
        for name, expected in expected_grads.items():
            torch.testing.assert_close(
                actual_grads[name], expected, rtol=1e-5, atol=1e-6
            )


class TestCollectiveClassification(unittest.TestCase):
    """Functional collectives stay on device in both launch namespaces."""

    @staticmethod
    def _node(namespace):
        """Build a node-like object carrying only a target namespace."""
        return SimpleNamespace(target=SimpleNamespace(namespace=namespace))

    def test_collective_namespaces_are_excluded(self):
        """Both functional namespaces are collectives; ``aten`` is not."""
        for namespace in ("_c10d_functional", "_c10d_functional_autograd"):
            with self.subTest(namespace=namespace):
                self.assertTrue(
                    offload_module._is_collective_or_wait(self._node(namespace))
                )
        self.assertFalse(offload_module._is_collective_or_wait(self._node("aten")))


class TestViewChain(unittest.TestCase):
    """Only views read in backward are replayed onto the reloaded tensor."""

    @staticmethod
    def _graph():
        """Build a base tensor with one backward-read and one forward-only view."""
        graph = fx.Graph()
        x = graph.placeholder("x")
        tensor = graph.call_function(torch.ops.aten.relu.default, (x,))
        read_in_bwd = graph.call_function(torch.ops.aten.t.default, (tensor,))
        forward_only = graph.call_function(torch.ops.aten.t.default, (tensor,))
        forward_use = graph.call_function(torch.ops.aten.clone.default, (forward_only,))
        backward_use = graph.call_function(torch.ops.aten.mul.Tensor, (read_in_bwd, x))
        backward_use.meta["autograd_backward"] = True
        graph.output((forward_use, backward_use))
        return (
            fx.GraphModule(torch.nn.Module(), graph),
            tensor,
            read_in_bwd,
            forward_only,
            backward_use,
        )

    def test_forward_only_view_is_not_replayed(self):
        """Lifetime keeps every view; replay only the backward-read one."""
        _, tensor, read_in_bwd, forward_only, backward_use = self._graph()
        lifetime_views, replay_views, redirects = offload_module._view_chain(tensor)
        self.assertCountEqual(lifetime_views, [read_in_bwd, forward_only])
        self.assertEqual(replay_views, [read_in_bwd])
        self.assertEqual(redirects, [(read_in_bwd, backward_use)])


if __name__ == "__main__":
    unittest.main()
