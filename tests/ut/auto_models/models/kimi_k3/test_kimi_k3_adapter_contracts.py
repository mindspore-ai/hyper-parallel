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
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

from torch import nn

from hyper_parallel.components.modules import KimiDeltaAttention
from hyper_parallel.distributed._builder.forward_rewriter import (
    _ForwardRewriteRequest,
)
from hyper_parallel.distributed._builder.planner import ShardingPlanner
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
from hyper_parallel.models.kimi_k3.adapter.distributed import (
    context_parallel as adapter_context_parallel,
)
from hyper_parallel.models.registry import get_model_adapter
from hyper_parallel.trainer.config import entries_to_plan_overrides
from hyper_parallel.trainer.config.manager import parse_training_args
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
    ) -> None:
        """Record the executor construction arguments."""
        super().__init__()
        self.module = module
        self.mesh = mesh
        self.chunk_size = chunk_size
        self.backend = backend

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Return inputs so the wrapper remains callable in this metadata test."""
        return args, kwargs


class _TinyKimiModel(nn.Module):
    """Small Kimi-shaped model used to exercise Planner derivation."""

    def __init__(self) -> None:
        """Build one KDA boundary under an official Kimi architecture ID."""
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
        """Planner derives one replicated CP boundary with the requested wrapper."""
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
        """YAML resolution preserves KDA arguments and gates the rule on CP."""
        yaml_text = """
model:
  _target_: torch.nn.Identity
optimizer:
  _target_: torch.optim.AdamW
  lr: 1.0e-3
plan_overrides:
  - match: "*.self_attn"
    when: cp
    region_dispatch: false
    inner_target: self
    inner_wrapper:
      _target_: hyper_parallel.models.kimi_k3.adapter.distributed.context_parallel.kimi_delta_attention_p2p_cp_wrapper
      backend: triton
      chunk_size: 64
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kda_cp.yaml"
            path.write_text(yaml_text, encoding="utf-8")
            config = parse_training_args([str(path)])

        self.assertEqual(len(config.plan_overrides), 1)
        entry = config.plan_overrides[0]
        self.assertEqual(entry.when, "cp")
        self.assertEqual(entry.inner_target, "self")
        self.assertFalse(entry.region_dispatch)
        self.assertEqual(entry.inner_wrapper.backend, "triton")
        self.assertEqual(entry.inner_wrapper.chunk_size, 64)
        self.assertEqual(entries_to_plan_overrides([entry], cp_size=1), {})
        active = entries_to_plan_overrides([entry], cp_size=4)
        self.assertEqual(set(active), {"*.self_attn"})
        self.assertIs(active["*.self_attn"].inner_wrapper, entry.inner_wrapper)


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


if __name__ == "__main__":
    unittest.main()
