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
"""Generated Phase C binds a live plan without wrapping forward again."""

from types import SimpleNamespace

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from hyper_parallel.codegen.emit.parallel import lower_forward_boundaries
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.distributed._builder.tp_collective_lowering import TPCollectiveLowerer
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.models._generated import phase_c
from tests.common.mark_utils import arg_mark


def future_compute_factory() -> None:
    """Supply a stable emitted helper name for the binding probe."""


def test_generated_boundary_adapts_packed_input_contract() -> None:
    """Generated boundaries adapt packed inputs before local execution."""
    calls = []

    class CompiledBoundary:
        def redistribute_inputs(self, args: tuple, kwargs: dict) -> tuple:
            """Record and return redistributed inputs."""
            calls.append((args, kwargs))
            return args, kwargs

        def redistribute_outputs(self, outputs: object) -> object:
            """Return redistributed outputs."""
            return outputs

    boundary = phase_c._GeneratedBoundary(  # pylint: disable=protected-access
        CompiledBoundary(),
    )

    args, kwargs = boundary.redistribute_inputs((("value",), {"flag": True}))

    assert args == ("value",)
    assert kwargs == {"flag": True}
    assert calls == [(("value",), {"flag": True})]


def test_generated_phase_c_binds_state_without_rewriting_forward(monkeypatch) -> None:
    """Generated phase C binds state without replacing forward."""
    def _original_forward(value):
        return value

    def _compute(value):
        return value

    module = SimpleNamespace(
        forward=_original_forward,
        ep_enable=False,
        cp_enable=False,
        tp_enable=False,
    )
    spec = SimpleNamespace(
        is_boundary=True,
        local_compute_fn=future_compute_factory,
        inner_wrapper="future_cp",
        _deferred_bias_params=(),
    )
    plan = SimpleNamespace(mesh_dim_names=("tp", "cp"), modules={"future": spec})
    lowerer = SimpleNamespace()
    boundary = object()

    monkeypatch.setattr(phase_c, "_get_cp_submesh", lambda mesh, names: object())
    monkeypatch.setattr(phase_c, "_get_tp_submesh", lambda mesh, names: object())
    monkeypatch.setattr(phase_c, "create_tp_collective_lowerer", lambda mesh, names: lowerer)
    monkeypatch.setattr(phase_c, "_resolve_module", lambda model, fqn: module)
    monkeypatch.setattr(phase_c, "PrecompiledBoundary", lambda *args, **kwargs: boundary)
    monkeypatch.setattr(phase_c, "_bind_input_indices", lambda bound, target: None)
    monkeypatch.setattr(phase_c, "_resolve_local_compute_fn", lambda *args: _compute)
    monkeypatch.setattr(phase_c, "_resolve_inner_target", lambda target, target_spec: target)

    def install_inner(target: SimpleNamespace, *args: object, **kwargs: object) -> None:
        """Simulate native companion-state installation and forward replacement."""
        del args, kwargs
        target.forward = _compute

    monkeypatch.setattr(phase_c, "_wrap_inner_attention", install_inner)

    phase_c.apply_generated_phase_c(object(), plan, object(), False)

    assert module.forward is _original_forward
    assert module._hp_boundary.compiled is boundary
    assert module.future_compute_factory is _compute
    assert module.ep_enable is True
    assert module.cp_enable is True
    assert module.tp_enable is True


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
def test_generated_local_shard_forward_backward(monkeypatch) -> None:
    """Feature: Generated TP local slices on input and output boundaries.

    Description: Generate source, bind Phase C and execute both logical TP ranks.
    Expectation: Eager and FX outputs and gradients exactly match native slicing.
    """
    source = """
import torch
from torch import nn

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        return torch.sin(x * self.weight)
"""
    monkeypatch.setattr(phase_c, "_get_cp_submesh", lambda *args: None)
    monkeypatch.setattr(phase_c, "_get_tp_submesh", lambda *args: None)
    for side in ("in", "out"):
        name = "x" if side == "in" else "output"
        entry = {f"{side}_src": {name: {"tp": "R"}}, f"{side}_dst": {name: {"tp": "S(1)"}}}
        frozen = {"param_plan": {"block": entry}, "injections": [], "mesh_dim_names": ("tp",)}
        rendered = lower_forward_boundaries(source, frozen, boundary_classes={"block": "Block"})
        assert "self._hp_tp.local_shard(" in rendered
        namespace = {}
        exec(compile(rendered, "<generated-local-shard>", "exec"), namespace)  # pylint: disable=exec-used
        spec = ModuleShardingSpec(**{
            f"{side}_src": {name: {"tp": Replicate()}}, f"{side}_dst": {name: {"tp": Shard(1)}},
        })
        plan = SimpleNamespace(mesh_dim_names=("tp",), modules={"block": spec})
        for rank in (0, 1):
            lowerer = TPCollectiveLowerer(("tp",), object(), 2, rank, "hccl")
            monkeypatch.setattr(phase_c, "create_tp_collective_lowerer", lambda *args, op=lowerer: op)
            for trace in (False, True):
                module = namespace["Block"]()
                phase_c.apply_generated_phase_c(SimpleNamespace(block=module), plan, object(), False)
                value = torch.arange(24.0).reshape(3, 8).requires_grad_()
                reference = value.detach().clone().requires_grad_()
                weight = module.weight.detach().clone().requires_grad_()
                local = reference.chunk(2, dim=1)[rank] if side == "in" else reference
                expected = torch.sin(local * weight)
                if side == "out":
                    expected = expected.chunk(2, dim=1)[rank]
                actual = (make_fx(module)(value) if trace else module)(value)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                gradients = torch.autograd.grad(actual.sum(), (value, module.weight))
                expected_gradients = torch.autograd.grad(expected.sum(), (reference, weight))
                for actual_grad, expected_grad in zip(gradients, expected_gradients):
                    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
