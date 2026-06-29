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
"""Unit tests for MindSpore swap optimizer adapter helpers."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import Any

import pytest

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

from tests.ut.platform.mindspore._ensure_mindspore_platform import (  # noqa: E402
    ensure_mindspore_platform_default,
)

ensure_mindspore_platform_default()

from hyper_parallel.platform.mindspore.swap_optimizer import swap_optimizer  # noqa: E402
from hyper_parallel.platform.mindspore.swap_optimizer import adapters as adapters_module  # noqa: E402


def _load_adapters_module():
    """Return the adapter module from the installed hyper-parallel package."""
    return adapters_module


def _load_runtime_module():
    return swap_optimizer


class _FakeStorage:
    def __init__(self, size):
        self._size = size

    def size(self) -> int:
        """Return the fake storage size."""
        return self._size


class _FakeTensor:
    """Minimal contiguous tensor model used by storage eligibility tests."""

    dtype = "Float32"
    itemsize = 4

    def __init__(self, device, *, contiguous=True):
        self.device = device
        self._contiguous = contiguous

    def numel(self) -> int:
        """Return the fake element count."""
        return 8

    def untyped_storage(self) -> _FakeStorage:
        """Return a full-size fake storage."""
        return _FakeStorage(self.numel() * self.itemsize)

    def is_contiguous(self) -> bool:
        """Return whether the fake tensor is contiguous."""
        return self._contiguous


class _FakeDTensor:
    """Minimal distributed tensor exposing one local storage shard."""

    def __init__(self, local_tensor, global_numel):
        self._local_tensor = local_tensor
        self._global_numel = global_numel

    def numel(self) -> int:
        """Return the global element count."""
        return self._global_numel

    def to_local(self) -> _FakeTensor:
        """Return the local tensor shard."""
        return self._local_tensor


class _FakeParameter:
    def __init__(self, data):
        self.data = data


class _FailingStorage:
    def size(self) -> int:
        """Return the fake storage size."""
        return 8

    def resize_(self, size: int) -> None:
        """Simulate a failed storage resize."""
        del size
        raise RuntimeError("resize failed")


class _FailingTensor:
    device = "Ascend:0"

    def untyped_storage(self) -> _FailingStorage:
        """Return storage whose resize always fails."""
        return _FailingStorage()


def _make_runtime_with_fake_mindspore(module):
    runtime = module.MindSporeSwapRuntime.__new__(module.MindSporeSwapRuntime)
    runtime.ms = SimpleNamespace(Parameter=type("FakeParameter", (), {}))
    return runtime


def test_checkpoint_swappable_slot_allows_cpu_tensor():
    """Checkpoint promotion should catch fresh optimizer state before runtime placement."""
    module = _load_adapters_module()
    adapter = module.MindSporeAdamBaseAdapter.__new__(module.MindSporeAdamBaseAdapter)
    adapter.config = SimpleNamespace(min_numel=1)

    cpu_slot = module.SwapSlot(name="exp_avg", tensor=_FakeTensor("cpu"), swappable=False)
    device_slot = module.SwapSlot(name="exp_avg", tensor=_FakeTensor("Ascend:0"), swappable=False)

    assert adapter._is_checkpoint_swappable_slot(cpu_slot)
    assert adapter._is_checkpoint_swappable_slot(device_slot)


def test_checkpoint_swappable_slot_rejects_unknown_logical_key():
    """Checkpoint promotion should only accept supported logical state keys."""
    module = _load_adapters_module()
    adapter = module.MindSporeAdamBaseAdapter.__new__(module.MindSporeAdamBaseAdapter)
    adapter.config = SimpleNamespace(min_numel=1)
    slot = module.SwapSlot(name="unknown", tensor=_FakeTensor("Ascend:0"), swappable=False)

    assert not adapter._is_checkpoint_swappable_slot(slot)


def test_checkpoint_swappable_slot_uses_dtensor_local_shard(monkeypatch):
    """Checkpoint eligibility should use the local storage represented by a DTensor."""
    module = _load_adapters_module()
    monkeypatch.setattr(module, "ms", SimpleNamespace(Parameter=_FakeParameter))
    adapter = module.MindSporeAdamBaseAdapter.__new__(module.MindSporeAdamBaseAdapter)
    adapter.config = SimpleNamespace(min_numel=8)

    local_tensor = _FakeTensor("cpu")
    dtensor = _FakeDTensor(local_tensor, global_numel=32)
    slot = module.SwapSlot(
        name="exp_avg",
        tensor=_FakeParameter(dtensor),
        swappable=False,
    )

    assert adapter._is_checkpoint_swappable_slot(slot)

    adapter.config.min_numel = 9
    assert not adapter._is_checkpoint_swappable_slot(slot)


def test_checkpoint_swappable_slot_rejects_noncontiguous_tensor():
    """Checkpoint promotion should reject non-contiguous optimizer state."""
    module = _load_adapters_module()
    adapter = module.MindSporeAdamBaseAdapter.__new__(module.MindSporeAdamBaseAdapter)
    adapter.config = SimpleNamespace(min_numel=1)
    slot = module.SwapSlot(
        name="exp_avg",
        tensor=_FakeTensor("cpu", contiguous=False),
        swappable=False,
    )

    assert not adapter._is_checkpoint_swappable_slot(slot)


def test_refresh_swappable_slots_only_promotes_optimizer_state(monkeypatch):
    """Dynamic promotion should not treat a master parameter as an optimizer state tensor."""
    module = _load_runtime_module()
    runtime = module.MindSporeSwapRuntime.__new__(module.MindSporeSwapRuntime)
    runtime.config = SimpleNamespace(min_numel=1)
    state_tensor = object()
    master_tensor = object()
    state_slot = module.SwapSlot(name="exp_avg", tensor=state_tensor, swappable=False)
    master_slot = module.SwapSlot(name="master_param", tensor=master_tensor, swappable=False)
    checked_tensors = []
    monkeypatch.setattr(
        runtime,
        "is_swappable_tensor",
        lambda tensor, min_numel: checked_tensors.append((tensor, min_numel)) or True,
    )
    monkeypatch.setattr(runtime, "storage_nbytes", lambda _tensor: 32)

    runtime.refresh_swappable_slots([SimpleNamespace(slots=[state_slot, master_slot])])

    assert state_slot.swappable
    assert state_slot.storage_nbytes == 32
    assert not master_slot.swappable
    assert checked_tensors == [(state_tensor, 1)]


def test_make_slot_reuses_cached_swappable_slot():
    """A stable swappable tensor should not repeat storage eligibility checks every step."""
    module = _load_adapters_module()
    calls = {"is_swappable": 0, "storage_nbytes": 0}

    class _FakeRuntime:
        """Record tensor eligibility and storage-size queries."""

        @staticmethod
        def is_swappable_tensor(tensor: Any, min_numel: int) -> bool:
            """Record and accept a tensor eligibility check."""
            del tensor, min_numel
            calls["is_swappable"] += 1
            return True

        @staticmethod
        def storage_nbytes(tensor: Any) -> int:
            """Record a storage-size query."""
            del tensor
            calls["storage_nbytes"] += 1
            return 32

    adapter = module.MindSporeAdamBaseAdapter.__new__(module.MindSporeAdamBaseAdapter)
    adapter.runtime = _FakeRuntime()
    adapter.config = SimpleNamespace(min_numel=1)
    adapter._slots = {}
    tensor = _FakeTensor("Ascend:0")

    first = adapter._make_slot(0, "exp_avg", tensor)
    second = adapter._make_slot(0, "exp_avg", tensor)

    assert second is first
    assert isinstance(first, module.SwapSlot)
    assert not hasattr(first, "parameter")
    assert calls == {"is_swappable": 1, "storage_nbytes": 1}


def test_checkpoint_slot_map_resolves_name_from_optimizer_parameter():
    """Checkpoint names remain available after a packed slot switches to a host view."""
    module = _load_adapters_module()
    parameter = SimpleNamespace(name="moment1.weight")
    host_view = object()
    slot = module.SwapSlot(name="exp_avg", tensor=host_view, cpu_tensor=host_view, packed=True)
    adapter = module.MindSporeAdamBaseAdapter.__new__(module.MindSporeAdamBaseAdapter)
    adapter._slots = {(0, "exp_avg"): slot}
    adapter._checkpoint_slots = lambda: (slot,)
    adapter._state_parameter = lambda _index, _key: parameter

    assert adapter._checkpoint_slot_map() == {parameter.name: slot}


def test_publish_packed_state_rebinds_optimizer_parameter_only_once():
    """The adapter publishes a persistent CPU mirror without storing the Parameter on the slot."""
    module = _load_adapters_module()
    published = []
    parameter = SimpleNamespace(set_data=published.append)
    cpu_mirror = object()
    slot = module.SwapSlot(
        name="exp_avg",
        tensor=cpu_mirror,
        cpu_tensor=cpu_mirror,
        packed=True,
        state="host",
    )
    adapter = module.MindSporeAdamBaseAdapter.__new__(module.MindSporeAdamBaseAdapter)
    adapter.runtime = SimpleNamespace(packed_enabled=True)
    adapter._slots = {(0, "exp_avg"): slot}
    adapter._state_parameter = lambda _index, _key: parameter

    adapter.publish_packed_state()

    assert published == [cpu_mirror]
    assert all(value is not parameter for value in vars(slot).values())


@pytest.mark.parametrize(
    ("adapter_name", "optimizer", "expected"),
    (
        (
            "MindSporeNativeAdamAdapter",
            SimpleNamespace(moment1=["moment1"], moment2=["moment2"], vhat=["vhat"]),
            {"exp_avg": "moment1", "exp_avg_sq": "moment2", "max_exp_avg_sq": "vhat"},
        ),
        (
            "MindSporeNativeAdamWAdapter",
            SimpleNamespace(moments1=["moments1"], moments2=["moments2"]),
            {"exp_avg": "moments1", "exp_avg_sq": "moments2"},
        ),
        (
            "MindFormersAdamWAdapter",
            SimpleNamespace(
                exp_avg=["exp_avg"],
                exp_avg_sq=["exp_avg_sq"],
                max_exp_avg_sq=["max_exp_avg_sq"],
                fp32_params=["master_param"],
            ),
            {
                "exp_avg": "exp_avg",
                "exp_avg_sq": "exp_avg_sq",
                "max_exp_avg_sq": "max_exp_avg_sq",
                "master_param": "master_param",
            },
        ),
    ),
)
def test_state_parameter_resolves_optimizer_owned_collections(adapter_name, optimizer, expected):
    """Every adapter resolves logical slot keys through its optimizer-owned Parameter collections."""
    module = _load_adapters_module()
    adapter_type = getattr(module, adapter_name)
    adapter = adapter_type.__new__(adapter_type)
    adapter.optimizer = optimizer

    assert {key: adapter._state_parameter(0, key) for key in expected} == expected


def test_state_dict_requires_mindspore_state_dict_support():
    """Optimizer checkpointing should fail clearly when state_dict is unavailable."""
    module = _load_adapters_module()
    adapter = module.MindSporeAdamBaseAdapter.__new__(module.MindSporeAdamBaseAdapter)
    adapter.optimizer = SimpleNamespace(parameters_dict=lambda: {"unused": object()})

    with pytest.raises(
        RuntimeError,
        match="The installed MindSpore version does not support optimizer\\.state_dict\\(\\)\\.",
    ):
        adapter._state_dict()


def test_load_state_dict_requires_mindspore_load_state_dict_support():
    """Optimizer checkpoint loading should fail clearly when load_state_dict is unavailable."""
    module = _load_adapters_module()
    adapter = module.MindSporeAdamBaseAdapter.__new__(module.MindSporeAdamBaseAdapter)
    adapter.optimizer = SimpleNamespace()

    with pytest.raises(
        RuntimeError,
        match="The installed MindSpore version does not support optimizer\\.load_state_dict\\(\\)\\.",
    ):
        adapter._load_state_dict({})


def test_load_packed_checkpoint_keeps_slot_on_cpu_and_republishes_parameter():
    """Packed checkpoint load updates the persistent mirror and republishes optimizer state."""
    module = _load_adapters_module()
    copy_calls = []
    publish_calls = []

    cpu_mirror = object()
    loaded_cpu_tensor = object()
    slot = module.SwapSlot(
        name="exp_avg",
        tensor=cpu_mirror,
        cpu_tensor=cpu_mirror,
        swappable=True,
        state="host",
        packed=True,
    )
    adapter = module.MindSporeAdamBaseAdapter.__new__(module.MindSporeAdamBaseAdapter)
    adapter.runtime = SimpleNamespace(
        make_cpu_tensor=lambda _tensor: loaded_cpu_tensor,
        copy_cpu_tensor=lambda target, source: copy_calls.append((target, source)),
    )
    adapter._checkpoint_slot_map = lambda **_kwargs: {"moment1.weight": slot}
    adapter.publish_packed_state = lambda: publish_calls.append(True)

    adapter.load_checkpoint_state_dict({"moment1.weight": SimpleNamespace(data=object())})

    assert copy_calls == [(cpu_mirror, loaded_cpu_tensor)]
    assert slot.tensor is cpu_mirror
    assert slot.state == "host"
    assert publish_calls == [True]


def test_make_slot_packs_dtensor_local_storage():
    """MindSpore DTensor optimizer states can pack their local contiguous storage."""
    module = _load_adapters_module()

    class _FakeRuntime:
        """Model packed eligibility for distributed optimizer state."""

        packed_enabled = True

        @staticmethod
        def is_swappable_tensor(tensor, min_numel):
            del tensor, min_numel
            return False

        @staticmethod
        def is_packable_tensor(tensor, min_numel):
            del min_numel
            return isinstance(tensor, _FakeDTensor)

        @staticmethod
        def storage_nbytes(tensor):
            del tensor
            return 32

    adapter = module.MindSporeAdamBaseAdapter.__new__(module.MindSporeAdamBaseAdapter)
    adapter.runtime = _FakeRuntime()
    adapter.config = SimpleNamespace(min_numel=1)
    adapter._slots = {}
    tensor = _FakeDTensor(_FakeTensor("Ascend:0"), global_numel=8)

    slot = adapter._make_slot(0, "exp_avg", tensor)

    assert slot.packed
    assert slot.swappable


def test_storage_tensor_prefers_dtensor_backing_local_tensor():
    """Storage operations must target the DTensor backing object, not its wrapper."""
    module = _load_runtime_module()
    runtime = module.MindSporeSwapRuntime.__new__(module.MindSporeSwapRuntime)
    parameter_type = type("FakeParameter", (), {})
    runtime.ms = SimpleNamespace(Parameter=parameter_type)
    local = object()
    wrapper = parameter_type()
    wrapper.data = SimpleNamespace(_local_tensor=object())
    wrapper._local_tensor = local
    wrapper.to_local = object

    assert runtime._storage_tensor(wrapper) is local


def test_make_cpu_tensor_uses_dtensor_backing_local_tensor(monkeypatch):
    """CPU mirrors must be allocated from the concrete local storage tensor."""
    module = _load_runtime_module()
    runtime = module.MindSporeSwapRuntime.__new__(module.MindSporeSwapRuntime)
    parameter_type = type("FakeParameter", (), {})
    runtime.ms = SimpleNamespace(Parameter=parameter_type)
    local = _FakeTensor("Ascend:0")
    local.shape = (8,)
    wrapper = parameter_type()
    wrapper._local_tensor = local
    wrapper.data = SimpleNamespace(device="Ascend:0")
    cpu_tensor = _FakeTensor("CPU")
    allocations = []
    copy_calls = []
    monkeypatch.setattr(
        module,
        "ms",
        SimpleNamespace(
            Parameter=parameter_type,
            mint=SimpleNamespace(
                empty=lambda shape, **kwargs: allocations.append((shape, kwargs)) or cpu_tensor
            ),
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_copy_storage",
        lambda target, source: copy_calls.append((target, source)),
    )

    assert runtime.make_cpu_tensor(wrapper) is cpu_tensor
    assert allocations == [((8,), {"dtype": "Float32", "device": "cpu", "pin_memory": True})]
    assert copy_calls == [(cpu_tensor, local)]


@pytest.mark.parametrize(
    ("adapter_name", "optimizer_attributes", "optimizer_name"),
    (
        ("MindSporeNativeAdamAdapter", {}, "nn.Adam"),
        ("MindSporeNativeAdamWAdapter", {"use_fused_opt": True}, "nn.AdamWeightDecay"),
    ),
)
def test_native_adam_rejects_packed_swap(adapter_name, optimizer_attributes, optimizer_name):
    """Native MindSpore Adam implementations only support per-tensor swap."""
    module = _load_adapters_module()
    adapter_type = getattr(module, adapter_name)
    adapter = adapter_type.__new__(adapter_type)
    adapter.optimizer = SimpleNamespace(use_parallel=False, **optimizer_attributes)
    adapter.config = SimpleNamespace(packed_swap=True)

    with pytest.raises(ValueError, match=rf"{optimizer_name}.*packed_swap=True"):
        adapter.validate()


@pytest.mark.parametrize(
    ("adapter_name", "optimizer_attributes"),
    (
        ("MindSporeNativeAdamAdapter", {"use_lazy": False, "use_offload": False}),
        ("MindSporeNativeAdamWAdapter", {"use_fused_opt": True}),
    ),
)
def test_native_adam_accepts_per_tensor_swap(adapter_name, optimizer_attributes):
    """Native MindSpore Adam implementations remain available with packed swap disabled."""
    module = _load_adapters_module()
    adapter_type = getattr(module, adapter_name)
    adapter = adapter_type.__new__(adapter_type)
    adapter.optimizer = SimpleNamespace(use_parallel=False, **optimizer_attributes)
    adapter.config = SimpleNamespace(packed_swap=False)

    adapter.validate()


def test_mindformers_adamw_preserves_packed_runtime_setting():
    """MindFormers AdamW validation must leave packed swap enabled."""
    module = _load_adapters_module()
    adapter = module.MindFormersAdamWAdapter.__new__(module.MindFormersAdamWAdapter)
    adapter.optimizer = SimpleNamespace(use_parallel=False, enable_cpu_offload=False)
    adapter.config = SimpleNamespace(packed_swap=True)
    adapter.runtime = SimpleNamespace(_packed_enabled=True)

    adapter.validate()

    assert adapter.runtime._packed_enabled


def test_mindformers_packed_layout_units_include_every_parameter():
    """Persistent host layout units should not depend on the current gradient set."""
    module = _load_adapters_module()
    adapter = module.MindFormersAdamWAdapter.__new__(module.MindFormersAdamWAdapter)
    adapter.optimizer = SimpleNamespace(fp32_params=("param-0", "param-1"))
    adapter._build_slots = lambda index: [f"slot-{index}"]

    units = adapter.packed_layout_units()

    assert [unit.adapter_index for unit in units] == [0, 1]
    assert [unit.param for unit in units] == ["param-0", "param-1"]
    assert [unit.grad for unit in units] == [None, None]
    assert [unit.slots for unit in units] == [["slot-0"], ["slot-1"]]


@pytest.mark.parametrize(
    ("packed_enabled", "expected_indices"),
    ((True, [0, 1]), (False, [0])),
)
def test_mindformers_prepare_step_keeps_inactive_units_only_for_packed_layout(
        packed_enabled,
        expected_indices,
):
    """Packed steps should preserve static batches while per-tensor steps skip missing gradients."""
    module = _load_adapters_module()
    optimizer = SimpleNamespace(
        fp32_params=("param-0", "param-1"),
        get_weight_decay=lambda: 0.01,
        get_lr=lambda: 0.1,
        _increase_global_step=lambda: None,
        is_group=False,
        is_group_lr=False,
    )
    adapter = module.MindFormersAdamWAdapter.__new__(module.MindFormersAdamWAdapter)
    adapter.optimizer = optimizer
    adapter.runtime = SimpleNamespace(packed_enabled=packed_enabled)
    adapter._build_slots = lambda index: [f"slot-{index}"]

    context = adapter.prepare_step(("grad-0", None))

    assert [unit.adapter_index for unit in context["units"]] == expected_indices
    assert [unit.grad for unit in context["units"]] == ["grad-0", None][:len(expected_indices)]


def test_prepare_packed_host_skips_per_tensor_initialization(monkeypatch):
    """Per-tensor initialization is owned by the optimizer wrapper."""
    module = _load_runtime_module()
    runtime = module.MindSporeSwapRuntime.__new__(module.MindSporeSwapRuntime)
    runtime._packed_enabled = False
    monkeypatch.setattr(
        runtime,
        "offload_initial_slots",
        lambda slots: pytest.fail(f"unexpected runtime offload for {slots!r}"),
    )

    runtime.prepare_packed_host((object(),))


@pytest.mark.parametrize("packed_enabled", (False, True))
def test_swap_optimizer_offloads_slots_before_packed_host_preparation(monkeypatch, packed_enabled):
    """The wrapper offloads initial slots before optional packed host preparation."""
    module = _load_runtime_module()
    initial_slots = (object(), object())
    calls = []

    class _Runtime:
        """Record wrapper initialization order for packed and per-tensor modes."""

        def __init__(self, config):
            self.packed_enabled = config.packed_swap

        @staticmethod
        def offload_initial_slots(slots):
            calls.append(("offload", slots))

        @staticmethod
        def partition(units):
            calls.append(("partition", units))
            return [list(units)]

        @staticmethod
        def prepare_packed_host(batches):
            calls.append(("prepare", batches))
            return True

    class _Adapter:
        """Expose fixed initial slots for optimizer wrapper construction."""

        def __init__(self, optimizer, config, runtime):
            del optimizer, config, runtime

        @staticmethod
        def matches(optimizer):
            del optimizer
            return True

        @staticmethod
        def validate():
            return None

        @staticmethod
        def initial_slots():
            return iter(initial_slots)

        @staticmethod
        def packed_layout_units():
            calls.append(("layout", initial_slots))
            return list(initial_slots)

        @staticmethod
        def publish_packed_state():
            calls.append(("publish", initial_slots))

    monkeypatch.setattr(module, "MindSporeSwapRuntime", _Runtime)
    monkeypatch.setattr(module.MindSporeSwapOptimizer, "_adapters", (_Adapter,))

    module.MindSporeSwapOptimizer(object(), SimpleNamespace(packed_swap=packed_enabled))

    expected = [("offload", initial_slots)]
    if packed_enabled:
        expected.extend([
            ("layout", initial_slots),
            ("partition", list(initial_slots)),
            ("prepare", [list(initial_slots)]),
        ])
    expected.append(("publish", initial_slots))
    assert calls == expected


def test_prepare_packed_host_builds_persistent_batch_dtype_buffers(monkeypatch):
    """Packed initialization should build one persistent host buffer per batch and dtype."""
    module = _load_runtime_module()
    runtime = module.MindSporeSwapRuntime.__new__(module.MindSporeSwapRuntime)
    runtime._packed_enabled = True
    runtime._host_layout_signature = ()
    runtime._host_buffers = []
    runtime._packed_batch_plans = []
    calls = []
    runtime.wait_event = lambda event, stream: calls.append(("wait_event", event, stream))

    class _View:
        """Fake CPU tensor view backed by replaceable storage."""

        dtype = "Float32"
        device = "cpu"

        def __init__(self, value=None, storage=None):
            self.value = value
            self._storage = storage or _FakeStorage(32)

        def untyped_storage(self):
            return self._storage

        def set_(self, storage, offset, shape, stride):
            calls.append(("set", storage, offset, shape, stride))
            self._storage = storage
            return self

    monkeypatch.setattr(
        module,
        "ms",
        SimpleNamespace(
            mint=SimpleNamespace(empty=lambda *args, **kwargs: _View((args, kwargs))),
            Parameter=type("FakeParameter", (), {}),
        ),
    )
    slots = [
        module.SwapSlot(
            name="exp_avg",
            tensor=object(),
            dtype="Float32",
            numel=4,
            shape=(4,),
            packed=True,
        ),
        module.SwapSlot(
            name="exp_avg_sq",
            tensor=object(),
            dtype="Float32",
            numel=4,
            shape=(4,),
            packed=True,
        ),
    ]

    monkeypatch.setattr(
        runtime,
        "_copy_cpu_tensor",
        lambda target, source: calls.append(("copy", source)),
    )
    for index, slot in enumerate(slots):
        slot.cpu_tensor = f"mirror-{index}"
        slot.state = "host"
        slot.event = f"copy-event-{index}"
    batches = [[SimpleNamespace(adapter_index=0, slots=slots)]]

    assert runtime.prepare_packed_host(batches)

    assert [call[1] for call in calls if call[0] == "copy"] == ["mirror-0", "mirror-1"]
    assert [call[1:] for call in calls if call[0] == "wait_event"] == [
        ("copy-event-0", None),
        ("copy-event-1", None),
    ]
    assert all(slot.event is None for slot in slots)
    assert [(call[2], call[3], call[4]) for call in calls if call[0] == "set"] == [
        (0, (4,), (1,)),
        (4, (4,), (1,)),
    ]
    assert [slot.host_offset for slot in slots] == [0, 4]
    assert all(slot.cpu_tensor.device == "cpu" for slot in slots)
    assert len(runtime._host_buffers) == 1
    assert set(runtime._host_buffers[0]) == {"Float32"}
    assert runtime._packed_batch_plans[0].regions["Float32"].numel == 8
    persistent_buffer = runtime._host_buffers[0]["Float32"]
    calls.clear()

    assert not runtime.prepare_packed_host(batches)
    assert runtime._host_buffers[0]["Float32"] is persistent_buffer
    assert not calls


def test_cpu_storage_view_rejects_device_materialization(monkeypatch):
    """Packed host aliases must never silently materialize on Ascend."""
    module = _load_runtime_module()
    runtime = module.MindSporeSwapRuntime.__new__(module.MindSporeSwapRuntime)

    class _View:
        dtype = "Float32"
        device = "Ascend:0"

        def untyped_storage(self):
            return object()

        def set_(self, storage, offset, shape, stride):
            del storage, offset, shape, stride
            return self

    monkeypatch.setattr(
        module,
        "ms",
        SimpleNamespace(mint=SimpleNamespace(empty=lambda *args, **kwargs: _View())),
    )

    with pytest.raises(RuntimeError, match="must remain on CPU"):
        runtime._make_cpu_storage_view(_View(), 0, (4,))


@pytest.mark.parametrize("device", ("CPU", "CPU:0", "cpu:0"))
def test_cpu_tensor_accepts_mindspore_host_device_suffix(device):
    """Host tensors may expose a logical CPU device index."""
    module = _load_runtime_module()
    tensor = SimpleNamespace(device=device)

    assert module._is_cpu_tensor(tensor)


def test_end_packed_step_detaches_slots_and_drops_staging_arenas():
    """Step teardown must leave no slot or runtime reference to NPU staging storage."""
    module = _load_runtime_module()
    runtime = module.MindSporeSwapRuntime.__new__(module.MindSporeSwapRuntime)

    class _ResizableStorage(_FakeStorage):
        def resize_(self, size):
            self._size = size

    class _RawBuffer:
        def __init__(self):
            self.storage = _ResizableStorage(128)

        def untyped_storage(self):
            return self.storage

    cpu_tensor = object()
    slot = module.SwapSlot(name="exp_avg", tensor=object(), cpu_tensor=cpu_tensor, state="host")
    region = SimpleNamespace(slots=[slot])
    persistent_plans = [SimpleNamespace(regions={"Float32": region})]
    runtime._packed_batch_plans = persistent_plans
    runtime._packed_ready_events = {}
    runtime._packed_offload_events = {}
    raw_buffers = [_RawBuffer(), _RawBuffer()]
    arenas = [module._StagingArena(raw_buffer, {"Float32": object()}) for raw_buffer in raw_buffers]
    runtime._staging_arenas = arenas

    runtime.end_packed_step()

    assert slot.tensor is cpu_tensor
    assert runtime._packed_batch_plans is persistent_plans
    assert runtime._staging_arenas == [None, None]
    assert all(arena.dtype_views == {} for arena in arenas)
    assert all(arena.raw_buffer is None for arena in arenas)
    assert all(raw_buffer.untyped_storage().size() == 0 for raw_buffer in raw_buffers)


def test_mindspore_packed_results_are_dropped_before_teardown():
    """MindSpore update outputs must not retain PyNative staging inputs."""
    module = _load_runtime_module()
    runtime = module.MindSporeSwapRuntime.__new__(module.MindSporeSwapRuntime)
    results = [object(), object()]

    runtime.release_packed_step_results(results)

    assert not results


def test_packed_host_offsets_restart_for_each_batch_and_dtype(monkeypatch):
    """Every persistent batch/dtype host buffer should use its own zero-based offsets."""
    module = _load_runtime_module()
    runtime = module.MindSporeSwapRuntime.__new__(module.MindSporeSwapRuntime)
    runtime._packed_enabled = True
    runtime._host_layout_signature = ()
    runtime._host_buffers = []
    runtime._packed_batch_plans = []
    allocations = []

    class _View:
        """Fake typed host view used to verify per-batch offset layouts."""

        device = "CPU"

        def __init__(self, dtype="Float32", storage=None):
            self.dtype = dtype
            self._storage = storage or _FakeStorage(64)

        def untyped_storage(self):
            return self._storage

        def set_(self, storage, offset, shape, stride):
            del offset, shape, stride
            self._storage = storage
            return self

    def _empty(*args, **kwargs):
        allocations.append((args, kwargs))
        return _View(kwargs.get("dtype", "Float32"))

    monkeypatch.setattr(
        module,
        "ms",
        SimpleNamespace(
            mint=SimpleNamespace(empty=_empty),
            Parameter=type("FakeParameter", (), {}),
        ),
    )
    monkeypatch.setattr(runtime, "_copy_cpu_tensor", lambda _target, _source: None)

    def _slot(dtype, numel):
        return module.SwapSlot(
            name="exp_avg",
            tensor=object(),
            cpu_tensor=object(),
            dtype=dtype,
            numel=numel,
            shape=(numel,),
            swappable=True,
            packed=True,
            state="host",
        )

    slots = [
        _slot("Float32", 4),
        _slot("Float32", 4),
        _slot("BFloat16", 2),
        _slot("Float32", 4),
        _slot("Float32", 4),
        _slot("BFloat16", 2),
    ]
    batches = [
        [SimpleNamespace(adapter_index=0, slots=slots[:3])],
        [SimpleNamespace(adapter_index=1, slots=slots[3:])],
    ]

    runtime.prepare_packed_host(batches)

    assert [slot.host_offset for slot in slots] == [0, 4, 0, 0, 4, 0]
    assert len(runtime._host_buffers) == 2
    assert all(
        set(batch_buffers) == {"Float32", "BFloat16"}
        for batch_buffers in runtime._host_buffers
    )
    assert [plan.regions["Float32"].numel for plan in runtime._packed_batch_plans] == [8, 8]
    assert [plan.regions["BFloat16"].numel for plan in runtime._packed_batch_plans] == [2, 2]
    assert len([call for call in allocations if call[0] == ((8,),)]) == 2
    assert len([call for call in allocations if call[0] == ((2,),)]) == 2


def test_packed_copies_use_persistent_host_buffers_without_bounce(monkeypatch):
    """Steady-state packed transfers should copy directly to and from batch host buffers."""
    module = _load_runtime_module()
    runtime = module.MindSporeSwapRuntime.__new__(module.MindSporeSwapRuntime)
    host_buffer = object()
    staging_view = object()
    copies = []

    class _DtypeView:
        @staticmethod
        def narrow(dim, offset, numel):
            assert (dim, offset, numel) == (0, 0, 8)
            return staging_view

    region = module._PackedBatchRegion("Float32", 8, [])
    runtime._host_buffers = [{"Float32": host_buffer}]
    runtime._packed_batch_plans = [module._PackedBatchPlan({"Float32": region})]
    runtime._staging_arenas = [module._StagingArena(object(), {"Float32": _DtypeView()}), None]
    monkeypatch.setattr(runtime, "_copy_tensor", lambda target, source: copies.append((target, source)))

    runtime._copy_packed_to_device(0, 0)
    runtime._copy_packed_to_host(0, 0)

    assert copies == [(staging_view, host_buffer), (host_buffer, staging_view)]


def test_packed_pipeline_rejects_unpacked_slot_without_cpu_mirror():
    """Packed pipeline selection should consistently reject an ineligible slot."""
    module = _load_runtime_module()
    runtime = module.MindSporeSwapRuntime.__new__(module.MindSporeSwapRuntime)
    runtime._packed_enabled = True
    slot = module.SwapSlot(
        name="exp_avg",
        tensor=object(),
        cpu_tensor=None,
        dtype="Float32",
        numel=4,
        packed=False,
    )
    batches = [[SimpleNamespace(slots=[slot])]]

    assert not runtime.supports_packed_pipeline(batches)
    assert not runtime.supports_packed_pipeline(batches)


def test_end_packed_step_drops_staging_arena_owners():
    """No raw staging tensor owner may survive into the next forward pass."""
    module = _load_runtime_module()
    runtime = module.MindSporeSwapRuntime.__new__(module.MindSporeSwapRuntime)

    class _Storage(_FakeStorage):
        def resize_(self, size):
            self._size = size

    raw_buffer = SimpleNamespace(untyped_storage=lambda: _Storage(128))
    arena = module._StagingArena(raw_buffer, dtype_views={"Float32": object()})
    runtime._staging_arenas = [arena, None]
    runtime._packed_batch_plans = []
    runtime._packed_ready_events = {}
    runtime._packed_offload_events = {}

    runtime.end_packed_step()

    assert runtime._staging_arenas == [None, None]
    assert arena.dtype_views == {}
    assert arena.raw_buffer is None


def test_copy_to_cpu_reuses_existing_cpu_mirror(monkeypatch):
    """Repeated offloads should overwrite the pinned mirror instead of allocating it again."""
    module = _load_runtime_module()
    runtime = module.MindSporeSwapRuntime.__new__(module.MindSporeSwapRuntime)
    device_tensor = object()
    cpu_tensor = object()
    make_calls = []
    copy_calls = []
    monkeypatch.setattr(runtime, "make_cpu_tensor", lambda tensor: make_calls.append(tensor) or cpu_tensor)
    monkeypatch.setattr(runtime, "_copy_storage", lambda target, source: copy_calls.append((target, source)))
    slot = module.SwapSlot(name="exp_avg", tensor=device_tensor)

    runtime.copy_to_cpu(slot)
    slot.state = "device"
    runtime.copy_to_cpu(slot)

    assert slot.cpu_tensor is cpu_tensor
    assert slot.state == "d2h"
    assert make_calls == [device_tensor]
    assert copy_calls == [(cpu_tensor, device_tensor)]


def test_make_cpu_tensor_does_not_use_ascend_placing_copy_for_cpu_source(monkeypatch):
    """CPU state copies must stay on host under an Ascend execution context."""
    module = _load_runtime_module()
    runtime = module.MindSporeSwapRuntime.__new__(module.MindSporeSwapRuntime)
    source = SimpleNamespace(shape=(4,), dtype="Float32", device="CPU")
    target = SimpleNamespace(device="CPU")
    allocations = []
    copies = []

    def _empty(shape, **kwargs):
        allocations.append((shape, kwargs))
        return target

    monkeypatch.setattr(
        module,
        "ms",
        SimpleNamespace(
            Parameter=type("FakeParameter", (), {}),
            mint=SimpleNamespace(empty=_empty),
        ),
    )
    monkeypatch.setattr(runtime, "_copy_cpu_tensor", lambda dst, src: copies.append((dst, src)))
    monkeypatch.setattr(runtime, "_copy_storage", lambda dst, src: pytest.fail("CPU source used device copy"))

    assert runtime.make_cpu_tensor(source) is target
    assert allocations == [((4,), {"dtype": "Float32", "device": "cpu", "pin_memory": True})]
    assert copies == [(target, source)]


def test_restore_device_storage_surfaces_resize_failure():
    """Device storage restore failures must stop the swap state machine."""
    module = _load_runtime_module()
    runtime = _make_runtime_with_fake_mindspore(module)
    slot = module.SwapSlot(name="exp_avg", tensor=_FailingTensor(), storage_nbytes=16, state="host")

    with pytest.raises(RuntimeError, match="Failed to restore device storage for swap slot 'exp_avg'"):
        runtime.restore_device_storage(slot)


def test_wait_offload_slot_preserves_state_when_release_fails():
    """A failed device storage release must not mark the slot as host-resident."""
    module = _load_runtime_module()
    runtime = _make_runtime_with_fake_mindspore(module)
    slot = module.SwapSlot(name="exp_avg", tensor=_FailingTensor(), state="d2h")

    with pytest.raises(RuntimeError, match="Failed to release device storage for swap slot 'exp_avg'"):
        runtime.wait_offload_slot(slot)

    assert slot.state == "d2h"


def test_mindformers_adamw_step_batch_skips_dtensor_dispatch(monkeypatch):
    """Swap adapter should preserve MindFormers AdamW's local optimizer-update dispatch mode."""
    module = _load_adapters_module()
    active = {"skip": False}
    calls = []

    class _RecordingSkipDTensorDispatch:
        def __enter__(self):
            active["skip"] = True

        def __exit__(self, exc_type, exc_val, exc_tb):
            active["skip"] = False

    fake_optimizer_module = SimpleNamespace()

    def _run_adamw_opt(*args):
        calls.append(("adamw", active["skip"], args))
        return "ok"

    fake_optimizer_module._run_adamw_opt = _run_adamw_opt
    monkeypatch.setattr(module, "SkipDTensorDispatch", _RecordingSkipDTensorDispatch)
    monkeypatch.setitem(sys.modules, "fake_mindformers_adamw", fake_optimizer_module)

    FakeOptType = type("FakeOpt", (), {"__module__": "fake_mindformers_adamw"})
    optimizer = FakeOptType()
    optimizer.enable_fused_opt = False
    optimizer.beta1 = 0.9
    optimizer.beta2 = 0.999
    optimizer.global_step = 1
    optimizer.eps = 1e-8
    optimizer.exp_avg = ["exp_avg"]
    optimizer.exp_avg_sq = ["exp_avg_sq"]
    optimizer.optim_filter = [True]
    optimizer.one_minus_beta2 = 0.001

    adapter = module.MindFormersAdamWAdapter.__new__(module.MindFormersAdamWAdapter)
    adapter.optimizer = optimizer
    adapter._sync_batch_master_params = lambda batch: calls.append(("sync", active["skip"], batch))
    unit = module.UpdateUnit(
        adapter_index=0,
        param="param",
        grad="grad",
        slots=[],
    )
    inactive_unit = module.UpdateUnit(
        adapter_index=0,
        param="inactive-param",
        grad=None,
        slots=[],
    )

    batch = [inactive_unit, unit]
    assert adapter.step_batch(batch, {"lr": 0.1, "weight_decay": 0.01}) == ("ok",)
    assert calls[0][0:2] == ("adamw", True)
    assert calls[1][0:2] == ("sync", True)
    assert calls[1][2] == batch


def test_mindformers_adamw_finish_step_skips_dtensor_dispatch(monkeypatch):
    """Final master-param sync should also use the local optimizer-update dispatch mode."""
    module = _load_adapters_module()
    active = {"skip": False}
    calls = []

    class _RecordingSkipDTensorDispatch:
        def __enter__(self):
            active["skip"] = True

        def __exit__(self, exc_type, exc_val, exc_tb):
            active["skip"] = False

    class _FakeOptimizer:
        def _copy_main_params_to_model_params(self):
            calls.append(active["skip"])

    monkeypatch.setattr(module, "SkipDTensorDispatch", _RecordingSkipDTensorDispatch)

    adapter = module.MindFormersAdamWAdapter.__new__(module.MindFormersAdamWAdapter)
    adapter.optimizer = _FakeOptimizer()
    adapter.config = SimpleNamespace(include_master_params=False)

    adapter.finish_step({})
    assert calls == [True]
