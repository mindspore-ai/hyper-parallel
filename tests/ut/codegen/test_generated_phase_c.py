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

from hyper_parallel.models._generated import phase_c


def future_compute_factory() -> None:
    """Supply a stable emitted helper name for the binding probe."""


def test_generated_boundary_adapts_packed_input_contract(monkeypatch) -> None:
    calls = []

    class CompiledBoundary:
        def redistribute_inputs(self, args: tuple, kwargs: dict) -> tuple:
            """Record and return redistributed inputs."""
            calls.append((args, kwargs))
            return args, kwargs

        def redistribute_outputs(self, outputs: object) -> object:
            """Return redistributed outputs."""
            return outputs

    monkeypatch.setattr(
        phase_c,
        "_rewrap_local_outputs",
        lambda outputs, spec, mesh, names, label: outputs,
    )
    boundary = phase_c._GeneratedBoundary(  # pylint: disable=protected-access
        CompiledBoundary(), object(), object(), ("tp",), "future.block",
    )

    args, kwargs = boundary.redistribute_inputs((("value",), {"flag": True}))

    assert args == ("value",)
    assert kwargs == {"flag": True}
    assert calls == [(("value",), {"flag": True})]


def test_generated_phase_c_binds_state_without_rewriting_forward(monkeypatch) -> None:
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
