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
"""Unit tests for Torch Adam/AdamW swap optimizer adapters."""

from __future__ import annotations

import copy
import os
from types import SimpleNamespace
from unittest import mock

import pytest

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import torch

from hyper_parallel.core.optimizer.swap_optimizer_base import SwapSlot, UpdateUnit
from hyper_parallel.platform.torch.swap_optimizer import adapters as module


class _Runtime:
    """Small runtime double exposing the methods used by the adapter."""

    def __init__(self, *, packed=False, swappable=False, packable=None):
        self.packed_enabled = packed
        self.swappable = swappable
        self.packable = swappable if packable is None else packable
        self.released = []
        self.prepared = []
        self.synchronized = []

    def is_swappable_tensor(self, tensor, min_numel):
        del tensor, min_numel
        return self.swappable

    def is_packable_template(self, tensor, min_numel):
        del tensor, min_numel
        return self.packable

    @staticmethod
    def is_distributed_tensor(tensor):
        return callable(getattr(tensor, "to_local", None))

    @staticmethod
    def populate_slot_metadata(slot, tensor):
        slot.shape = tuple(tensor.shape)
        slot.dtype = tensor.dtype
        slot.device = tensor.device
        slot.numel = int(tensor.numel())
        slot.storage_nbytes = slot.numel * int(tensor.element_size())

    def make_zero_cpu_tensor_like(self, tensor):
        return torch.zeros_like(tensor, device="cpu")

    def make_cpu_tensor(self, tensor):
        return tensor.detach().to(device="cpu").clone()

    @staticmethod
    def make_empty_device_tensor_like(tensor, source):
        return torch.empty_like(tensor, device=tensor.device, dtype=source.dtype)

    def release_device_storage(self, slot):
        self.released.append(slot)

    def prepare_packed_host(self, slots):
        self.prepared.append(tuple(slots))

    def synchronize_cpu_mirrors(self, slots):
        self.synchronized.append(tuple(slots))


def _config(**overrides):
    values = {"min_numel": 1, "state_keys": None}
    values.update(overrides)
    return SimpleNamespace(**values)


def _adapter(optimizer, runtime=None, **config):
    runtime = runtime or _Runtime()
    return module.TorchAdamBaseAdapter(optimizer, _config(**config), runtime)


@pytest.mark.parametrize(
    ("optimizer_type", "adapter_type"),
    ((torch.optim.Adam, module.TorchNativeAdamAdapter),
     (torch.optim.AdamW, module.TorchNativeAdamWAdapter)),
)
def test_native_adam_adapters_match_only_their_optimizer_family(optimizer_type, adapter_type):
    """Ensure each native adapter rejects the other Adam optimizer family."""
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = optimizer_type([parameter])

    assert adapter_type.matches(optimizer)
    other = torch.optim.AdamW([torch.nn.Parameter(torch.ones(2))]) if optimizer_type is torch.optim.Adam else (
        torch.optim.Adam([torch.nn.Parameter(torch.ones(2))])
    )
    assert not adapter_type.matches(other)


@pytest.mark.parametrize(
    ("group", "message"),
    (({"foreach": True}, "foreach=True"),
     ({"fused": True}, "fused=True"),
     ({"differentiable": True}, "differentiable=True"),
     ({"capturable": True}, "capturable=True")),
)
def test_validate_rejects_unsupported_torch_flags(group, message):
    """Reject Torch optimizer flags unsupported by the swap adapter."""
    adapter = object.__new__(module.TorchAdamBaseAdapter)
    adapter.optimizer = SimpleNamespace(param_groups=[group])

    with pytest.raises(ValueError, match=message):
        adapter.validate()


def test_adamw_validate_allows_fused_but_adam_does_not():
    """Allow fused mode for native AdamW but reject it for native Adam."""
    adamw = object.__new__(module.TorchNativeAdamWAdapter)
    adamw.optimizer = SimpleNamespace(param_groups=[{"fused": True}])
    adamw.validate()

    adam = object.__new__(module.TorchNativeAdamAdapter)
    adam.optimizer = SimpleNamespace(param_groups=[{"fused": True}])
    with pytest.raises(ValueError, match="fused=True"):
        adam.validate()


def test_prepare_step_rejects_extra_arguments_and_sparse_gradients():
    """Reject closures, extra step arguments, and sparse gradients."""
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.Adam([parameter])
    adapter = _adapter(optimizer)

    with pytest.raises(ValueError, match="closure or extra arguments"):
        adapter.prepare_step(lambda: None)

    parameter.grad = torch.sparse_coo_tensor([[0]], [1.0], size=(2,))
    with pytest.raises(ValueError, match="dense Adam/AdamW gradients"):
        adapter.prepare_step()


def test_prepare_step_initializes_lazy_state_and_skips_parameters_without_grad():
    """Initialize Adam state only for parameters that have a dense gradient."""
    parameters = [torch.nn.Parameter(torch.ones(4)), torch.nn.Parameter(torch.ones(4))]
    optimizer = torch.optim.Adam([parameters[0], parameters[1]], amsgrad=True)
    parameters[0].grad = torch.ones_like(parameters[0])
    adapter = _adapter(optimizer, _Runtime(swappable=False))

    context = adapter.prepare_step()

    assert len(context["units"]) == 1
    assert context["units"][0].param is parameters[0]
    assert set(optimizer.state[parameters[0]]) == {"step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"}
    assert [slot.name for slot in context["units"][0].slots] == [
        "exp_avg", "exp_avg_sq", "max_exp_avg_sq"
    ]
    assert parameters[1] not in optimizer.state


def test_prepare_packed_step_keeps_materialized_inactive_state_and_publishes_host_slots():
    """Keep materialized inactive states in the packed layout for later updates."""
    active = torch.nn.Parameter(torch.ones(4))
    inactive = torch.nn.Parameter(torch.ones(4))
    optimizer = torch.optim.Adam([active, inactive])
    optimizer.state[inactive]["exp_avg"] = torch.full_like(inactive, 3)
    optimizer.state[inactive]["exp_avg_sq"] = torch.full_like(inactive, 4)
    active.grad = torch.ones_like(active)
    runtime = _Runtime(packed=True, swappable=True, packable=True)
    adapter = _adapter(optimizer, runtime)

    context = adapter.prepare_step()

    assert [id(unit.param) for unit in context["units"]] == [id(active), id(inactive)]
    assert runtime.prepared
    assert all(slot.packed for slot in adapter.all_slots())


def test_configured_state_keys_control_slots_and_reject_master_param():
    """Honor configured logical state keys and reject unsupported master params."""
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.Adam([parameter])
    adapter = _adapter(optimizer, state_keys=("exp_avg",))
    optimizer.state[parameter]["exp_avg"] = torch.zeros_like(parameter)
    optimizer.state[parameter]["exp_avg_sq"] = torch.zeros_like(parameter)

    assert adapter._state_keys_for_param(parameter) == ("exp_avg",)
    assert adapter._configured_state_keys() == ("exp_avg",)

    invalid = _adapter(optimizer, state_keys=("master_param",))
    with pytest.raises(ValueError, match="master_param"):
        invalid._configured_state_keys()


def test_build_slots_reuses_slot_when_state_tensor_is_unchanged_and_orders_parameters():
    """Reuse stable slots and expose them in optimizer parameter order."""
    first, second = (torch.nn.Parameter(torch.ones(2)) for _ in range(2))
    optimizer = torch.optim.Adam([first, second])
    for parameter in (first, second):
        optimizer.state[parameter]["exp_avg"] = torch.zeros_like(parameter)
        optimizer.state[parameter]["exp_avg_sq"] = torch.zeros_like(parameter)
    adapter = _adapter(optimizer)

    first_slots = adapter._build_slots(first, optimizer.state[first])
    second_slots = adapter._build_slots(second, optimizer.state[second])

    assert adapter._build_slots(first, optimizer.state[first])[0] is first_slots[0]
    assert [id(slot) for slot in adapter.all_slots()] == [
        id(slot) for slot in first_slots + second_slots
    ]


def test_step_batch_skips_units_without_gradients():
    """Avoid invoking the functional optimizer when a batch has no gradients."""
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.Adam([parameter])
    adapter = _adapter(optimizer)
    unit = UpdateUnit(0, parameter, None, [])

    with mock.patch.object(torch.optim._functional, "adam") as functional:
        adapter.step_batch([unit], {})

    functional.assert_not_called()


@pytest.mark.parametrize("supports_decoupled_weight_decay", (False, True))
def test_step_batch_only_passes_supported_decoupled_weight_decay(
        supports_decoupled_weight_decay,
):
    """Pass decoupled weight decay only to functional Adam versions that support it."""
    parameter = torch.nn.Parameter(torch.ones(2))
    parameter.grad = torch.ones_like(parameter)
    optimizer = torch.optim.Adam([parameter])
    adapter = _adapter(optimizer)
    adapter._init_param_state(parameter, parameter.grad, optimizer.param_groups[0])
    slots = adapter._build_slots(parameter, optimizer.state[parameter])
    unit = UpdateUnit(0, parameter, parameter.grad, slots)
    call_kwargs = {}
    missing = object()

    if supports_decoupled_weight_decay:
        def functional_adam(*args, decoupled_weight_decay=missing, **kwargs):
            del args, kwargs
            if decoupled_weight_decay is not missing:
                call_kwargs["decoupled_weight_decay"] = decoupled_weight_decay

    else:
        def functional_adam(*args, **kwargs):
            del args
            call_kwargs.update(kwargs)

    with mock.patch.object(torch.optim._functional, "adam", functional_adam):
        adapter.step_batch([unit], {})

    if supports_decoupled_weight_decay:
        assert call_kwargs["decoupled_weight_decay"] is False
    else:
        assert "decoupled_weight_decay" not in call_kwargs


def test_export_swappable_state_prefers_host_mirror_and_deep_copies_metadata():
    """Export host mirrors for swapped tensors and copy all checkpoint data."""
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.Adam([parameter])
    optimizer.state[parameter]["step"] = torch.tensor(1.0)
    optimizer.state[parameter]["exp_avg"] = torch.full_like(parameter, 9)
    optimizer.state[parameter]["exp_avg_sq"] = torch.full_like(parameter, 7)
    runtime = _Runtime(swappable=True)
    adapter = _adapter(optimizer, runtime)
    host = torch.full_like(parameter, 2, device="cpu")
    adapter._slots[(id(parameter), "exp_avg")] = SwapSlot(
        name="exp_avg", tensor=torch.empty_like(parameter), cpu_tensor=host, swappable=True, state="host"
    )
    state_dict = optimizer.state_dict()
    exported = adapter.export_swappable_state(state_dict)

    assert torch.equal(exported["state"][0]["exp_avg"], host)
    assert exported["state"][0]["exp_avg"] is not host
    assert exported["state"][0]["exp_avg_sq"] is not state_dict["state"][0]["exp_avg_sq"]
    assert exported["param_groups"] == copy.deepcopy(state_dict["param_groups"])


def test_strip_swappable_state_does_not_mutate_checkpoint():
    """Remove configured swap tensors from a copied checkpoint state only."""
    checkpoint = {
        "state": {0: {"step": torch.tensor(2.0), "exp_avg": torch.ones(2), "other": "value"}},
        "param_groups": [{"params": [0]}],
    }
    adapter = object.__new__(module.TorchAdamBaseAdapter)
    adapter.config = _config()

    stripped, removed = adapter.strip_swappable_state(checkpoint)

    assert set(stripped["state"][0]) == {"step", "other"}
    assert torch.equal(removed[0]["exp_avg"], checkpoint["state"][0]["exp_avg"])
    assert "exp_avg" in checkpoint["state"][0]
    assert stripped is not checkpoint


def test_load_checkpoint_state_dict_delegates_stripped_and_removed_state():
    """Load ordinary state through Torch and route removed tensors to the adapter."""
    adapter = object.__new__(module.TorchAdamBaseAdapter)
    adapter.optimizer = mock.Mock()
    original = {"state": {0: {"exp_avg": torch.ones(2)}}, "param_groups": [{"params": [0]}]}
    stripped = {"state": {0: {}}, "param_groups": [{"params": [0]}]}
    removed = {0: {"exp_avg": original["state"][0]["exp_avg"]}}
    adapter.strip_swappable_state = mock.Mock(return_value=(stripped, removed))
    adapter.load_swappable_state = mock.Mock()

    adapter.load_checkpoint_state_dict(original)

    adapter.optimizer.load_state_dict.assert_called_once_with(stripped)
    adapter.load_swappable_state.assert_called_once_with(original, removed)


def test_state_key_and_tensor_cast_validation_errors_are_explicit():
    """Report missing configured state and invalid checkpoint tensor inputs clearly."""
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.Adam([parameter])
    adapter = _adapter(optimizer, state_keys=("exp_avg_sq",))

    with pytest.raises(ValueError, match="is not present"):
        adapter._state_keys_for_param(parameter)
    with pytest.raises(ValueError, match="Expected torch.Tensor"):
        adapter._cast_swappable_tensor_to_cpu(parameter, object())
    with pytest.raises(ValueError, match="without a tensor or template"):
        adapter._make_slot("exp_avg", None)


def test_load_swappable_state_packed_restores_cpu_mirror_and_slot():
    """Restore packed checkpoint tensors as host-backed swap slots."""
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = SimpleNamespace(param_groups=[{"params": [parameter]}], state={parameter: {}})
    runtime = _Runtime(packed=True, packable=True)
    adapter = object.__new__(module.TorchAdamBaseAdapter)
    adapter.optimizer = optimizer
    adapter.runtime = runtime
    adapter.config = _config()
    adapter._slots = {}
    saved = torch.arange(2, dtype=torch.float64)

    adapter.load_swappable_state({"param_groups": [{"params": [0]}]}, {0: {"exp_avg": saved}})

    slot = adapter._slots[(id(parameter), "exp_avg")]
    assert slot.state == "host"
    assert slot.packed
    assert slot.tensor is slot.cpu_tensor
    assert torch.equal(slot.cpu_tensor, saved.to(dtype=parameter.dtype))
    assert optimizer.state[parameter]["exp_avg"] is slot.cpu_tensor
    assert runtime.prepared


def test_checkpoint_state_dict_synchronizes_and_exports_optimizer_state():
    """Synchronize mirrors before exporting the wrapped optimizer checkpoint."""
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.Adam([parameter])
    optimizer.state[parameter]["exp_avg"] = torch.zeros_like(parameter)
    optimizer.state[parameter]["exp_avg_sq"] = torch.zeros_like(parameter)
    runtime = _Runtime()
    adapter = _adapter(optimizer, runtime)

    result = adapter.checkpoint_state_dict()

    assert runtime.synchronized and runtime.synchronized[0] == adapter.all_slots()
    assert set(result) == {"state", "param_groups"}
