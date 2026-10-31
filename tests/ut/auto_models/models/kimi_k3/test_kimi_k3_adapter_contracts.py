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
"""Contract tests for Kimi K3 registration and KDA CP injection."""
# pylint: disable=wrong-import-position

import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from hyper_parallel.components.modules import KimiDeltaAttention
from hyper_parallel.distributed._builder.forward_rewriter import (
    _ForwardRewriteRequest,
)
from hyper_parallel.distributed._builder.planner import ShardingPlanner
from hyper_parallel.distributed.context_parallel import kimi_delta_attention_state as state_cp
from hyper_parallel.distributed.recipe_spec import (
    CP,
    INNER_WRAPPER,
    ModuleShardingSpec,
)
from hyper_parallel.core.dtensor.placement_types import Replicate
from hyper_parallel.distributed.tensor_parallel.param_role import (
    ParamRole,
    ParameterClassifier,
)
from hyper_parallel.models.kimi_k3.adapter import load_context_parallel
from hyper_parallel.models.kimi_k3.adapter.distributed import (
    context_parallel as adapter_context_parallel,
)
from hyper_parallel.models.registry import get_model_adapter
from hyper_parallel.trainer.config import entries_to_plan_overrides
from hyper_parallel.trainer.config.parser import parse_training_args
from tests.common.mark_utils import arg_mark
from tests.ut.auto_models.distributed.conftest import FakeDeviceMesh

_WRAPPER_NAMES = (
    "kimi_delta_attention_ulysses_cp_wrapper",
    "kimi_delta_attention_p2p_cp_wrapper",
)
_MESH_FAMILY = frozenset({"mesh", "tp_mesh", "cp_mesh", "ep_mesh"})


class _FakeMesh:
    """Minimal mesh metadata used before any distributed process is initialized."""

    def __init__(self, size: int) -> None:
        """Store one synthetic mesh extent."""
        self._size = size

    def size(self) -> int:
        """Return the configured mesh size."""
        return self._size


class _RecordingExecutor(nn.Module):
    """Executor stub that records construction without touching collectives."""

    def __init__(
        self,
        module: nn.Module,
        mesh: Any,
        *,
        chunk_size: int,
        backend: str,
        **execution_options: Any,
    ) -> None:
        """Record the executor construction arguments."""
        super().__init__()
        self.module = module
        self.mesh = mesh
        self.chunk_size = chunk_size
        self.backend = backend
        self.execution_options = execution_options

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Return inputs so the wrapper remains callable in this metadata test."""
        return args, kwargs


class _TinyKimiModel(nn.Module):
    """Small Kimi-shaped model used to exercise Planner derivation."""

    def __init__(self) -> None:
        """Build one KDA layer under an official Kimi architecture ID."""
        super().__init__()
        self.config = SimpleNamespace(
            architectures=["KimiK3ForConditionalGeneration"]
        )
        self.self_attn = KimiDeltaAttention(
            hidden_size=32,
            num_heads=4,
            head_k_dim=4,
            head_v_dim=4,
            chunk_size=4,
        )


class TestKimiK3AdapterRegistration(unittest.TestCase):
    """Pin model-family discovery and KDA-specific parameter roles."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_top_level_and_text_specs_share_kda_providers(self):
        """Both official Kimi config identities discover the same KDA rules."""
        top_spec = get_model_adapter("KimiK3ForConditionalGeneration")
        text_spec = get_model_adapter("KimiLinearForCausalLM")
        self.assertIsNotNone(top_spec)
        self.assertIsNotNone(text_spec)
        self.assertIs(top_spec.context_parallel, load_context_parallel)
        self.assertIs(text_spec.context_parallel, load_context_parallel)
        self.assertIs(
            top_spec.context_parallel(),
            adapter_context_parallel,
        )
        self.assertIs(
            text_spec.context_parallel(),
            adapter_context_parallel,
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_kda_specific_parameter_roles_override_generic_rules(self):
        """ShortConv, recurrent-state and low-rank gate parameters are explicit."""
        rules = get_model_adapter("kimi_k3").sharding_rules()
        classifier = ParameterClassifier()
        expected = {
            "self_attn.q_conv1d.weight": ParamRole.COLWISE,
            "self_attn.A_log": ParamRole.COLWISE,
            "self_attn.dt_bias": ParamRole.COLWISE,
            "self_attn.f_a_proj.weight": ParamRole.REPLICATED,
            "self_attn.g_a_proj.weight": ParamRole.REPLICATED,
            "self_attn.f_b_proj.weight": ParamRole.COLWISE,
            "self_attn.g_b_proj.weight": ParamRole.COLWISE,
        }
        for name, role in expected.items():
            with self.subTest(parameter=name):
                self.assertIs(classifier.classify_param(name, rules), role)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_cp4_plan_retains_all_kda_params_and_injection(self):
        """Feature: Kimi KDA CP planner integration.

        Description: Derive a CP4 plan with the requested layer wrapper.
        Expectation: KDA parameters stay replicated and wrapper injection is retained.
        """
        override = ModuleShardingSpec(
            inner_target="self",
            inner_wrapper=(
                adapter_context_parallel.kimi_delta_attention_p2p_cp_wrapper
            ),
            region_dispatch=False,
        )
        plan = ShardingPlanner(plan_overrides={"self_attn": override}).plan(
            _TinyKimiModel(),
            FakeDeviceMesh((4,), ("cp",)),
            cp_size=4,
        )
        spec = plan.modules["self_attn"]
        self.assertIs(
            spec.inner_wrapper,
            adapter_context_parallel.kimi_delta_attention_p2p_cp_wrapper,
        )
        self.assertFalse(spec.region_dispatch)
        parameter_names = {
            name for name, _ in _TinyKimiModel().self_attn.named_parameters()
            if not name.startswith("o_norm.")
        }
        self.assertTrue(parameter_names.issubset(spec.params))
        for placements in spec.params.values():
            self.assertEqual(placements[CP], Replicate())

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_yaml_cp_condition_and_wrapper_arguments(self):
        """Feature: Unified KDA CP configuration.

        Description: Parse state_cp_method through the unified wrapper.
        Expectation: The method is preserved without a chunk-size option; CP1 disables the rule.
        """
        yaml_text = """
model:
  _target_: torch.nn.Identity
optimizer:
  _target_: torch.optim.AdamW
  lr: 1.0e-3
plan_overrides:
  - match: "*.self_attn.forward"
    when: cp
    region_dispatch: false
    replace:
      kind: method
      implementation:
        _target_: hyper_parallel.models.kimi_k3.adapter.distributed.context_parallel.kimi_delta_attention_cp_wrapper
        backend: triton
        state_cp_method: grouped_allgather_p2p
        allgather_group_size: 2
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kda_cp.yaml"
            path.write_text(yaml_text, encoding="utf-8")
            config = parse_training_args([str(path)])

        self.assertEqual(len(config.plan_overrides), 1)
        entry = config.plan_overrides[0]
        self.assertEqual(entry.when, "cp")
        self.assertEqual(entry.to_override()[1].inner_target, "self")
        self.assertFalse(entry.region_dispatch)
        self.assertEqual(entry.replace.implementation.backend, "triton")
        self.assertEqual(entry.replace.implementation.state_cp_method, "grouped_allgather_p2p")
        self.assertEqual(entry.replace.implementation.allgather_group_size, 2)
        self.assertFalse(hasattr(entry.replace.implementation, "chunk_size"))
        self.assertEqual(entries_to_plan_overrides([entry], cp_size=1), {})
        active = entries_to_plan_overrides([entry], cp_size=4)
        self.assertEqual(set(active), {"*.self_attn"})
        self.assertIs(active["*.self_attn"].inner_wrapper, entry.replace.implementation)


class TestKimiK3CpWrapperContracts(unittest.TestCase):
    """Pin the new Planner-facing KDA CP wrapper protocol."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_wrappers_declare_context_and_configuration(self):
        """Both modes use @inner_wrapper and expose backend/chunk configuration."""
        for name in _WRAPPER_NAMES:
            wrapper = getattr(adapter_context_parallel, name)
            with self.subTest(wrapper=name):
                meta = getattr(wrapper, "_injection_meta", None)
                self.assertIsNotNone(meta)
                self.assertEqual(meta.kind, INNER_WRAPPER)
                self.assertEqual(meta.context, {"target_module"} | _MESH_FAMILY)
                parameters = inspect.signature(wrapper).parameters
                self.assertEqual(parameters["backend"].default, "eager")
                self.assertEqual(parameters["chunk_size"].default, 64)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_wrappers_return_atomic_request_without_mutating_target(self):
        """The generic rewriter, rather than the family adapter, owns mutation."""
        cp_mesh = _FakeMesh(4)
        target = KimiDeltaAttention(
            hidden_size=32,
            num_heads=4,
            head_k_dim=4,
            head_v_dim=4,
            chunk_size=4,
        )
        original_forward = target.forward
        patch_names = (
            "KimiDeltaAttentionLayerUlyssesCP",
            "KimiDeltaAttentionLayerP2PCP",
        )
        for wrapper_name, executor_name in zip(_WRAPPER_NAMES, patch_names):
            with self.subTest(wrapper=wrapper_name), patch.object(
                adapter_context_parallel,
                executor_name,
                _RecordingExecutor,
            ):
                request = getattr(adapter_context_parallel, wrapper_name)(
                    target,
                    None,
                    None,
                    cp_mesh,
                    None,
                    backend="eager",
                    chunk_size=4,
                )
                self.assertIsInstance(request, _ForwardRewriteRequest)
                self.assertIs(request.target, target)
                self.assertIs(target.forward.__func__, original_forward.__func__)
                self.assertEqual(
                    request.companion_attrs["_hp_kda_cp_config"]["chunk_size"],
                    4,
                )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_wrapper_fail_fast_guards(self):
        """Inactive CP, TPxCP and duplicate application fail before mutation."""
        wrapper = adapter_context_parallel.kimi_delta_attention_p2p_cp_wrapper
        target = KimiDeltaAttention(
            hidden_size=16,
            num_heads=2,
            head_k_dim=4,
            head_v_dim=4,
            chunk_size=4,
        )
        with self.assertRaisesRegex(ValueError, "active CP mesh"):
            wrapper(target, None, None, _FakeMesh(1), None)
        with self.assertRaisesRegex(NotImplementedError, "simultaneous TP and CP"):
            wrapper(target, None, _FakeMesh(2), _FakeMesh(2), None)
        target._hp_kda_cp_config = {"mode": "p2p"}
        with self.assertRaisesRegex(RuntimeError, "already been applied"):
            wrapper(target, None, None, _FakeMesh(2), None)


class TestKDACombinedAdapter(unittest.TestCase):
    """Pin the recipe options for all pairwise and three-way combinations."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_all_combinations_record_effective_options(self):
        """Feature: Unified KDA CP recipe adapter.

        Description: Construct AG, RD and mixed recipes using metadata-only executors.
        Expectation: The forward rewrite preserves protocol, Ulysses and group-width choices.
        """
        parameters = inspect.signature(adapter_context_parallel.kimi_delta_attention_cp_wrapper).parameters
        self.assertNotIn("chunk_size", parameters)
        cases = [("allgather", 1, 1), ("p2p", 2, 1), ("allgather", 2, 1),
                 ("grouped_allgather_p2p", 1, 2), ("grouped_allgather_p2p", 2, 2),
                 ("recursive_doubling", 1, 1), ("recursive_doubling", 2, 1)]
        for protocol, ulysses, width in cases:
            with self.subTest(protocol=protocol, ulysses=ulysses):
                name = "KimiDeltaAttentionLayerCP"
                model = _TinyKimiModel().self_attn
                with patch.object(adapter_context_parallel, name, _RecordingExecutor):
                    request = adapter_context_parallel.kimi_delta_attention_cp_wrapper(
                        model, None, None, _FakeMesh(8), None, state_cp_method=protocol,
                        ulysses_degree=ulysses, allgather_group_size=width)
                config = request.companion_attrs["_hp_kda_cp_config"]
                self.assertEqual(config["state_cp_method"], protocol)
                self.assertEqual(config["allgather_group_size"], width)
                self.assertEqual(config.get("ulysses_degree", 1), ulysses)
                self.assertEqual(config["backend"], "triton")
                self.assertEqual(config["chunk_size"], 64)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_invalid_options_rejected_by_real_executors(self):
        """Feature: Unified KDA CP configuration validation.

        Description: Construct real state and hybrid executors with mocked mesh metadata.
        Expectation: Invalid protocols and group widths fail without rewriting the model or splitting groups.
        """
        cases = [("unknown", 1, "Unknown|supported"),
                 ("p2p", 2, "only valid"),
                 ("allgather", 2, "only valid"),
                 ("grouped_allgather_p2p", 0, "positive integer"),
                 ("grouped_allgather_p2p", True, "positive integer"),
                 ("grouped_allgather_p2p", 1.5, "positive integer"),
                 ("grouped_allgather_p2p", 3, "divid")]
        for ulysses in (1, 2):
            for protocol, width, message in cases:
                with self.subTest(ulysses=ulysses, protocol=protocol, width=width):
                    mesh = MagicMock()
                    mesh.ndim = 1
                    mesh.rank_list = tuple(range(8))
                    mesh.size.return_value = 8
                    mesh.get_local_rank.return_value = 0
                    mesh.get_group.return_value = None
                    model = KimiDeltaAttention(hidden_size=32, num_heads=4)
                    original_forward = model.forward
                    with self.assertRaisesRegex(ValueError, message):
                        adapter_context_parallel.kimi_delta_attention_cp_wrapper(
                            model, None, None, mesh, None, state_cp_method=protocol,
                            ulysses_degree=ulysses, allgather_group_size=width)
                    self.assertFalse(hasattr(model, "_hp_kda_cp_config"))
                    self.assertEqual(model.forward, original_forward)
                    mesh._unflatten.assert_not_called()  # pylint: disable=protected-access

        for attribute, value in (("padding", (0,)), ("padding_mode", "reflect"), ("out_channels", 1024)):
            with self.subTest(short_conv_attribute=attribute):
                model = KimiDeltaAttention(hidden_size=32, num_heads=4)
                setattr(model.q_conv1d, attribute, value)
                with self.assertRaisesRegex(ValueError, "ShortConv"):
                    adapter_context_parallel.kimi_delta_attention_cp_wrapper(
                        model, None, None, mesh, None, state_cp_method="allgather")
                self.assertFalse(hasattr(model, "_hp_kda_cp_config"))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_invalid_contracts_rejected_before_transport(self):
        """Feature: KDA state CP validation.

        Description: Pass incompatible shapes, dtypes, rank orders and options.
        Expectation: Invalid calls fail before transport is invoked.
        """
        state = torch.zeros(1, 2, 128, 128)
        protocols = [state_cp.KDAStateAllGather(None, 0, 4),
                     state_cp.KDAStateAllGatherP2P(None, None, 0, (0, 2, 4, 6), 2)]
        with patch.object(state_cp, "dist") as transport:
            for protocol in protocols:
                with self.assertRaises(ValueError):
                    protocol.forward(state, state.double())
                with self.assertRaises(ValueError):
                    protocol.backward(state, (state[..., :64],))
                for invalid in (state[:0], torch.empty_like(state, device="meta")):
                    with self.assertRaises(ValueError):
                        protocol.forward(invalid, invalid)
            transport.all_gather_into_tensor.assert_not_called()
            transport.irecv.assert_not_called()
        with self.assertRaisesRegex(ValueError, "unique"):
            state_cp.KDAStateAllGatherP2P(None, None, 0, (0, 0, 4, 6), 2)
        with self.assertRaisesRegex(ValueError, "Ascend NPU"):
            state_cp.merge_affine_summaries(state.unsqueeze(0), state.unsqueeze(0), 0, True)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_mesh_split_preserves_world_root_and_noncontiguous_peers(self):
        """Feature: Root-preserving KDA subgroup creation.

        Description: Split an interleaved CP axis and reject a partial root or reordered peers.
        Expectation: Valid siblings share WORLD metadata; invalid meshes fail before group creation.
        """
        mesh = MagicMock()
        mesh.ndim = 1
        mesh.size.return_value = 4
        mesh.rank_list = (0, 2, 4, 6)
        mesh._get_root_mesh.return_value.size.return_value = 8
        with patch.object(state_cp.dist, "get_world_size", return_value=8), \
                patch.object(state_cp.dist, "get_process_group_ranks", return_value=[0, 2, 4, 6]):
            state_cp.split_kda_mesh(mesh, 2, ("state", "ulysses"))
            mesh._unflatten.assert_called_once_with(0, (2, 2), ("state", "ulysses"))
            mesh._unflatten.reset_mock()
            mesh._get_root_mesh.return_value.size.return_value = 4
            with self.assertRaisesRegex(ValueError, "WORLD root"):
                state_cp.split_kda_mesh(mesh, 2, ("state", "ulysses"))
            mesh.rank_list = (2, 0, 4, 6)
            with self.assertRaisesRegex(ValueError, "communication group order"):
                state_cp.split_kda_mesh(mesh, 2, ("state", "ulysses"))
            mesh._unflatten.assert_not_called()


if __name__ == "__main__":
    unittest.main()
