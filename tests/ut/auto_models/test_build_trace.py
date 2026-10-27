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
"""Unit tests for context-local model patch recording."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from hyper_parallel.distributed._builder.applier import _apply_phase_c
from hyper_parallel.distributed._builder.forward_rewriter import (
    _ForwardRewriteRequest,
    _wrap_inner_attention,
)
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec, inner_wrapper
from hyper_parallel.models.build_trace import (
    record_local_compute,
    record_model_patches,
)
from hyper_parallel.models.replacement import (
    ModuleReplacementSpec,
    apply_module_replacements,
    compile_module_replacements,
    module_replacement,
)


class _SourceModule(nn.Module):
    """Parameter-free source module used by replacement tests."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Return the input unchanged."""
        return value


class _ReplacementModule(nn.Module):
    """Source-compatible replacement module."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Return the input incremented by one."""
        return value + 1


@module_replacement
def _replace_source(*, module, module_fqn, context):
    """Return a source-compatible replacement for recorder tests."""
    del module_fqn, context
    replacement = _ReplacementModule()
    replacement.train(module.training)
    return replacement


@inner_wrapper
def _replace_forward(*, target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
    """Return one atomic rewrite request with a companion attribute."""
    del mesh, tp_mesh, cp_mesh, ep_mesh
    original_forward = target_module.forward

    def replacement(value: torch.Tensor) -> torch.Tensor:
        """Delegate to the original forward and increment its result."""
        return original_forward(value) + 1

    return _ForwardRewriteRequest(
        target_module,
        replacement,
        {"trace_marker": True},
    )


class TestModelPatchRecorder(unittest.TestCase):
    """Verify patch events reflect successful builder mutations only."""

    def test_module_replacement_records_final_fqn_and_types(self):
        """Record the factory result only after replacement installation."""
        model = nn.Module()
        model.block = _SourceModule()
        rule = ModuleReplacementSpec(
            match=("block",),
            factory=_replace_source,
            module_type=_SourceModule,
        )
        plan = compile_module_replacements(model, (rule,))

        with record_model_patches() as recorder:
            apply_module_replacements(
                model,
                plan,
                context={"policy": "test"},
            )

        snapshot = recorder.freeze()
        self.assertEqual(len(snapshot.module_replacements), 1)
        record = snapshot.module_replacements[0]
        self.assertEqual(record.module_fqn, "block")
        self.assertIs(record.source_type, _SourceModule)
        self.assertIs(record.replacement_type, _ReplacementModule)
        self.assertIs(record.factory, _replace_source)
        self.assertEqual(record.factory_kwargs["context"]["policy"], "test")

    def test_inner_wrapper_records_normalized_request_after_install(self):
        """Capture the actual forward request and resolved target FQN."""
        module = _SourceModule()
        spec = ModuleShardingSpec(
            inner_target="self",
            inner_wrapper=_replace_forward,
            region_dispatch=False,
        )

        with record_model_patches() as recorder:
            _wrap_inner_attention(
                module,
                None,
                spec=spec,
                module_fqn="model.layers.0.attention",
            )

        snapshot = recorder.freeze()
        self.assertEqual(len(snapshot.inner_wrappers), 1)
        record = snapshot.inner_wrappers[0]
        self.assertEqual(record.module_fqn, "model.layers.0.attention")
        self.assertEqual(record.target_fqn, "model.layers.0.attention")
        self.assertIs(record.wrapper, _replace_forward)
        self.assertIs(record.request.target, module)
        self.assertTrue(record.request.companion_attrs["trace_marker"])
        torch.testing.assert_close(module(torch.tensor(2)), torch.tensor(3))

    def test_local_compute_record_preserves_factory_configuration(self):
        """Freeze local-compute configuration without copying runtime objects."""
        module = _SourceModule()

        def factory() -> None:
            """Stand in for a local-compute factory."""
            return None

        def compute(target: nn.Module, value: torch.Tensor) -> torch.Tensor:
            """Run the supplied module on one value."""
            return target(value)

        runtime_binding = object()
        with record_model_patches() as recorder:
            record_local_compute(
                module_fqn="model.layers.0.mlp",
                module=module,
                factory=factory,
                factory_kwargs={"ep_mesh": runtime_binding, "fix_router": False},
                resolved_compute=compute,
                region_dispatch=False,
            )

        record = recorder.freeze().local_computes[0]
        self.assertIs(record.factory, factory)
        self.assertIs(record.resolved_compute, compute)
        self.assertIs(record.factory_kwargs["ep_mesh"], runtime_binding)
        self.assertFalse(record.factory_kwargs["fix_router"])

    def test_phase_c_records_local_compute_after_forward_install(self):
        """Observe the resolved compute through the actual Phase-C install path."""
        module = _SourceModule()

        def factory() -> None:
            """Stand in for the declared local-compute factory."""
            return None

        def compute(value: torch.Tensor) -> torch.Tensor:
            """Return the local-region input unchanged."""
            return value

        spec = ModuleShardingSpec(
            local_compute_fn=factory,
            region_dispatch=False,
        )
        plan = SimpleNamespace(
            mesh_dim_names=(),
            modules={"model.layers.0.mlp": spec},
        )
        target = "hyper_parallel.distributed._builder.applier"

        with patch(target + "._get_cp_submesh", return_value=None), \
                patch(target + "._get_tp_submesh", return_value=None), \
                patch(target + ".create_tp_collective_lowerer", return_value=None), \
                patch(target + "._resolve_module", return_value=module), \
                patch(target + ".PrecompiledBoundary", return_value=object()), \
                patch(target + "._bind_input_indices"), \
                patch(target + "._require_region_dispatch"), \
                patch(target + "._log_injection_choice"), \
                patch(target + "._wrap_inner_attention"), \
                patch(target + "._resolve_local_compute_fn", return_value=compute), \
                patch(target + "._descendant_boundary_fqns", return_value=[]), \
                patch(target + "._wrap_local_region_forward") as install:
            with record_model_patches() as recorder:
                _apply_phase_c(module, plan, mesh=object(), validate_mode=False)

        install.assert_called_once()
        record = recorder.freeze().local_computes[0]
        self.assertEqual(record.module_fqn, "model.layers.0.mlp")
        self.assertIs(record.factory, factory)
        self.assertIs(record.resolved_compute, compute)

    def test_nested_recorders_are_isolated_and_restore_outer_scope(self):
        """Keep nested build traces independent while restoring the outer one."""
        module = _SourceModule()

        with record_model_patches() as outer:
            record_local_compute(
                module_fqn="outer.before",
                module=module,
                factory=None,
                factory_kwargs=None,
                resolved_compute=module.forward,
                region_dispatch=False,
            )
            with record_model_patches() as inner:
                record_local_compute(
                    module_fqn="inner",
                    module=module,
                    factory=None,
                    factory_kwargs=None,
                    resolved_compute=module.forward,
                    region_dispatch=False,
                )
            record_local_compute(
                module_fqn="outer.after",
                module=module,
                factory=None,
                factory_kwargs=None,
                resolved_compute=module.forward,
                region_dispatch=False,
            )

        self.assertEqual(
            [record.module_fqn for record in outer.freeze().local_computes],
            ["outer.before", "outer.after"],
        )
        self.assertEqual(
            [record.module_fqn for record in inner.freeze().local_computes],
            ["inner"],
        )

    def test_record_helpers_are_noop_without_active_scope(self):
        """Leave ordinary trainer execution untouched when recording is disabled."""
        record_local_compute(
            module_fqn="unused",
            module=_SourceModule(),
            factory=None,
            factory_kwargs=None,
            resolved_compute=lambda value: value,
            region_dispatch=False,
        )


if __name__ == "__main__":
    unittest.main()
