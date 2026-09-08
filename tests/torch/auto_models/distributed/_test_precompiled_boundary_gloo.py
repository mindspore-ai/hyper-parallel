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
"""CPU Gloo integration worker for PrecompiledBoundary / local_region.

These cases initialize a real (single-rank) Gloo process group and build real
DTensors, so they must not live in ``tests/ut`` (Gate-1 forbids real process
groups there). They are launched via torchrun by
``test_precompiled_boundary_gloo.py`` in the same directory.

The cases were moved verbatim from
``tests/ut/dual_mode_dtensor/test_precompiled_boundary.py``; they keep their
original world_size=1 semantics (at world_size=1 Replicate and any Shard are
equivalent, which suffices for wrap/unwrap/identity logic).
"""
import os

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

# pylint: disable=wrong-import-position
import logging

import pytest
import torch
from torch import nn  # noqa: F401  (kept for parity with the original suite)

from hyper_parallel.distributed._builder.local_region import local_region
from hyper_parallel.distributed._builder.precompiled_boundary import (
    PrecompiledBoundary,
)
from hyper_parallel.distributed.recipe_spec import (
    ModuleShardingSpec,
    TP,
)
from hyper_parallel.distributed._builder.tp_collective_lowering import (
    create_tp_collective_lowerer,
)
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import (
    Partial,
    Replicate,
    Shard,
)
from tests.torch.utils import init_dist_gloo


class _FakeTPMesh:
    """Minimal TP mesh used to verify production lowering without extra groups."""

    mesh_dim_names = ("tp",)
    rank_list = (0, 1)

    def __init__(self):
        """Create a distinct fake process group."""
        self.group = object()

    def get_group(self):
        """Return the fake process group."""
        return self.group

    @staticmethod
    def size():
        """Return the fake TP group size."""
        return 2

    @staticmethod
    def get_local_rank():
        """Return the fake rank inside the TP group."""
        return 1


def _set_fake_group_ranks(monkeypatch, ranks=(0, 1)):
    monkeypatch.setattr(
        "hyper_parallel.distributed._builder.tp_collective_lowering."
        "platform.get_process_group_ranks",
        lambda _group: list(ranks),
    )


@pytest.fixture(scope="module")
def mesh():
    """Single-rank TP mesh backed by the real Gloo process group."""
    init_dist_gloo()
    return init_device_mesh("cpu", (1,), mesh_dim_names=("tp",))


def test_redistribute_io(mesh, monkeypatch, caplog):
    """Redistribute IO family: runtime input/output redistribution (identity, skip, DTensor, execution paths)."""
    # --- case: inputs_kwargs_hit ---
    spec = ModuleShardingSpec(
        in_src={"x": {TP: Replicate()}}, in_dst={"x": {TP: Replicate()}})
    b = PrecompiledBoundary(spec, mesh, ("tp",))
    t = torch.tensor([1.0])
    _, kwargs = b.redistribute_inputs((), {"x": t})
    assert kwargs["x"] is t, "case: inputs_kwargs_hit"  # identity pass-through

    # --- case: inputs_missing_arg_skipped ---
    # arg not found (None) -> skipped; no None injected into kwargs.
    spec = ModuleShardingSpec(
        in_src={"input": {TP: Replicate()}}, in_dst={"input": {TP: Replicate()}})
    b = PrecompiledBoundary(spec, mesh, ("tp",))
    _, kwargs = b.redistribute_inputs((torch.tensor([1]),), {})
    assert "input" not in kwargs, "case: inputs_missing_arg_skipped"

    # --- case: outputs_single ---
    spec = ModuleShardingSpec(
        out_src={"output": {TP: Shard(1)}}, out_dst={"output": {TP: Shard(1)}})
    b = PrecompiledBoundary(spec, mesh, ("tp",))
    # identity out_plan is empty -> returned as-is
    t = torch.tensor([1.0])
    assert b.redistribute_outputs(t) is t, "case: outputs_single"

    # --- case: outputs_tuple_order_preserved ---
    spec = ModuleShardingSpec(
        out_src={"a": {TP: Shard(1)}, "b": {TP: Replicate()}},
        out_dst={"a": {TP: Shard(1)}, "b": {TP: Replicate()}},
        out_names=["a", "b"],
    )
    b = PrecompiledBoundary(spec, mesh, ("tp",))
    ta, tb = torch.tensor([1.0]), torch.tensor([2.0])
    out = b.redistribute_outputs((ta, tb))
    assert isinstance(out, tuple) and out[0] is ta and out[1] is tb, \
        "case: outputs_tuple_order_preserved"

    # --- case: outputs_index_out_of_range_warns_and_skips ---
    # Skip tuple outputs absent from the runtime result and log a warning.
    spec = ModuleShardingSpec(
        out_src={"a": {TP: Shard(1)}, "b": {TP: Partial()}},
        out_dst={"a": {TP: Replicate()}, "b": {TP: Replicate()}},
        out_names=["a", "b"],
    )
    b = PrecompiledBoundary(spec, mesh, ("tp",))
    with caplog.at_level(logging.WARNING):
        out = b.redistribute_outputs((torch.tensor([[1.0]]),))
    assert "Skipping" in caplog.text, "case: outputs_index_out_of_range_warns_and_skips"
    assert len(out) == 1, "case: outputs_index_out_of_range_warns_and_skips"

    # --- case: inputs_as_dtensor ---
    spec = ModuleShardingSpec(
        in_src={"x": {TP: Replicate()}}, in_dst={"x": {TP: Replicate()}})
    b = PrecompiledBoundary(spec, mesh, ("tp",))
    _, kwargs = b.redistribute_inputs(
        (), {"x": torch.tensor([1.0])}, as_dtensor=True
    )
    assert isinstance(kwargs["x"], DTensor), "case: inputs_as_dtensor"

    # --- case: execution_tp_path_does_not_construct_dtensor ---
    # Explicit TP execution communicates directly on the local tensor.
    _set_fake_group_ranks(monkeypatch)
    fake_mesh = _FakeTPMesh()
    spec = ModuleShardingSpec(
        in_src={"x": {TP: Shard(1)}},
        in_dst={"x": {TP: Replicate()}},
    )
    boundary = PrecompiledBoundary(
        spec,
        fake_mesh,
        ("tp",),
        op_lowerer=create_tp_collective_lowerer(
            fake_mesh, ("tp",), collective_backend="hccl"
        ),
    )
    tensor = torch.randn(2, 3)
    gathered = torch.randn(2, 6)
    monkeypatch.setattr(
        "hyper_parallel.distributed._builder.tp_collective_lowering."
        "platform.differentiable_all_gather_concat",
        lambda *_args, **_kwargs: gathered,
    )
    monkeypatch.setattr(
        DTensor,
        "from_local",
        lambda *_args, **_kwargs: pytest.fail("execution TP path constructed a DTensor"),
    )
    _, kwargs = boundary.redistribute_inputs((), {"x": tensor})
    assert kwargs["x"] is gathered, "case: execution_tp_path_does_not_construct_dtensor"


@pytest.fixture(scope="module")
def mesh__local_region():
    """Single-rank mesh__local_region: at world_size=1, Replicate and any Shard are
    semantically equivalent, which suffices to verify wrap/unwrap/autograd logic."""
    init_dist_gloo()
    return init_device_mesh("cpu", (1,), mesh_dim_names=("tp",))


def _make_dtensor(mesh__local_region, data, requires_grad=False):
    local = torch.tensor(data, dtype=torch.float32, requires_grad=requires_grad)
    return DTensor.from_local(local, mesh__local_region, [Replicate()])


def test_local_region_error_paths(mesh__local_region):
    """local_region error family: fail-fast raises for local_region contract violations."""
    # --- case: out_placements_count_mismatch ---
    # Reject placement counts that differ from the runtime output count.
    def fn_count_mismatch(x):
        return x, x

    wrapped = local_region(
        fn_count_mismatch, device_mesh=mesh__local_region,
        in_placements={"x": (Replicate(),)},
        out_placements=((Replicate(),), (Replicate(),), (Replicate(),)),
    )
    with pytest.raises(ValueError, match="does not match"):
        wrapped(_make_dtensor(mesh__local_region, [1.0]))

    # --- case: flat_out_placements_rejected_for_multi_output ---
    # Reject a flat placement declaration for multiple tensor outputs.
    def fn_flat(x):
        return x, x

    wrapped = local_region(
        fn_flat, device_mesh=mesh__local_region,
        in_placements={"x": (Replicate(),)},
        out_placements=(Replicate(),),  # the flat form is only allowed for a single output
    )
    with pytest.raises(ValueError, match="single-output"):
        wrapped(_make_dtensor(mesh__local_region, [1.0]))

    # --- case: tensor_output_with_none_placement_raises ---
    # Reject a None placement declaration for a tensor output.
    def fn_none_placement(x):
        return x

    wrapped = local_region(
        fn_none_placement, device_mesh=mesh__local_region,
        in_placements={"x": (Replicate(),)},
        out_placements=(None,),
    )
    with pytest.raises(TypeError, match="non-None out_placements"):
        wrapped(_make_dtensor(mesh__local_region, [1.0]))


def test_local_region_wrap_unwrap(mesh__local_region):
    """local_region wrap/unwrap family: DTensor unwrap/rewrap, pass-through, tuple output, contract defaults."""
    # --- case: kwargs_input_and_output_wrap ---
    # Unwrap keyword DTensor input and wrap the tensor output.
    def fn_kwargs(hidden_states, scale=None):
        assert not isinstance(hidden_states, DTensor)  # local tensor inside the region
        return hidden_states * (scale or 2.0)

    wrapped = local_region(
        fn_kwargs, device_mesh=mesh__local_region,
        in_placements={"hidden_states": (Replicate(),)},
        out_placements=(Replicate(),),
    )
    dt = _make_dtensor(mesh__local_region, [1.0, 2.0, 3.0])
    out = wrapped(hidden_states=dt, scale=3.0)
    assert isinstance(out, DTensor), "case: kwargs_input_and_output_wrap"
    assert tuple(out.placements) == (Replicate(),), "case: kwargs_input_and_output_wrap"
    assert torch.allclose(out.to_local(), torch.tensor([3.0, 6.0, 9.0])), \
        "case: kwargs_input_and_output_wrap"

    # --- case: positional_input_via_signature_binding ---
    # Bind and unwrap positional DTensor inputs using the callable signature.
    def fn_positional(x, y):
        return x + y

    wrapped = local_region(
        fn_positional, device_mesh=mesh__local_region,
        in_placements={"x": (Replicate(),), "y": (Replicate(),)},
        out_placements=(Replicate(),),
    )
    dt_x = _make_dtensor(mesh__local_region, [1.0, 2.0])
    dt_y = _make_dtensor(mesh__local_region, [10.0, 20.0])
    out = wrapped(dt_x, dt_y)
    assert isinstance(out, DTensor), "case: positional_input_via_signature_binding"
    assert torch.allclose(out.to_local(), torch.tensor([11.0, 22.0])), \
        "case: positional_input_via_signature_binding"

    # --- case: plain_tensor_passthrough_no_wrap ---
    # all inputs non-DTensor (production scenario with parameters already unwrapped) -> output not wrapped.
    def fn_plain(x):
        return x * 2.0

    wrapped = local_region(
        fn_plain, device_mesh=mesh__local_region,
        in_placements={"x": (Replicate(),)},
        out_placements=(Replicate(),),
    )
    out = wrapped(torch.tensor([1.0, 2.0]))
    assert not isinstance(out, DTensor), "case: plain_tensor_passthrough_no_wrap"
    assert torch.allclose(out, torch.tensor([2.0, 4.0])), \
        "case: plain_tensor_passthrough_no_wrap"

    # --- case: mixed_dtensor_and_plain_args ---
    # Unwrap DTensor inputs while passing ordinary tensor inputs through.
    def fn_mixed(x, bias):
        return x + bias

    wrapped = local_region(
        fn_mixed, device_mesh=mesh__local_region,
        in_placements={"x": (Replicate(),)},
        out_placements=(Replicate(),),
    )
    dt = _make_dtensor(mesh__local_region, [1.0, 2.0])
    out = wrapped(dt, torch.tensor([100.0, 100.0]))
    assert isinstance(out, DTensor), "case: mixed_dtensor_and_plain_args"
    assert torch.allclose(out.to_local(), torch.tensor([101.0, 102.0])), \
        "case: mixed_dtensor_and_plain_args"

    # --- case: tuple_output_with_none_placeholder ---
    # Wrap tensor tuple entries and preserve metadata placeholders.
    def fn_tuple(x):
        return x * 2.0, "meta"

    wrapped = local_region(
        fn_tuple, device_mesh=mesh__local_region,
        in_placements={"x": (Replicate(),)},
        out_placements=((Replicate(),), None),
    )
    dt = _make_dtensor(mesh__local_region, [1.0, 2.0])
    out_tensor, meta = wrapped(dt)
    assert isinstance(out_tensor, DTensor), "case: tuple_output_with_none_placeholder"
    assert meta == "meta", "case: tuple_output_with_none_placeholder"

    # --- case: output_already_dtensor_not_rewrapped ---
    # Return an existing DTensor output without a second wrapper.
    def fn_already_dt(x):
        return DTensor.from_local(x * 2.0, mesh__local_region, [Replicate()])

    wrapped = local_region(
        fn_already_dt, device_mesh=mesh__local_region,
        in_placements={"x": (Replicate(),)},
        out_placements=(Replicate(),),
    )
    out = wrapped(_make_dtensor(mesh__local_region, [1.0]))
    assert isinstance(out, DTensor), "case: output_already_dtensor_not_rewrapped"
    assert torch.allclose(out.to_local(), torch.tensor([2.0])), \
        "case: output_already_dtensor_not_rewrapped"

    # --- case: out_placements_none_returns_raw ---
    # Return a raw local tensor when output placements are unspecified.
    def fn_raw(x):
        return x * 2.0

    wrapped = local_region(
        fn_raw, device_mesh=mesh__local_region,
        in_placements={"x": (Replicate(),)},
        out_placements=None,
    )
    out = wrapped(_make_dtensor(mesh__local_region, [1.0]))
    assert not isinstance(out, DTensor), "case: out_placements_none_returns_raw"
    assert torch.allclose(out, torch.tensor([2.0])), \
        "case: out_placements_none_returns_raw"
