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
"""MindSpore swap optimizer ST cases."""
import gc
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import mindspore as ms
import numpy as np
from mindspore.graph.api import _no_grad
from mindspore import Parameter, Tensor, nn, ops
from mindspore.communication import get_group_size, get_rank, init

from hyper_parallel import SkipDTensorDispatch, init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.core.optimizer import SwapOptimizerConfig, swap_optimizer
from hyper_parallel.core.optimizer.swap_optimizer_base import MASTER_PARAM_KEY
from hyper_parallel.platform.mindspore.autograd_compat import enable_mindspore_backward_compat
from tests.mindspore.st.swap_optimizer.mf_adamw import AdamW as MindFormersAdamW

enable_mindspore_backward_compat()


# The production adapter identifies MindFormers AdamW by module name and imports
# its update helpers from that module. Register the vendored test implementation
# under the production name so the ST exercises the same adapter path.
_MINDFORMERS_ADAMW_MODULE = "mindformers.pynative.optimizer.adamw"
sys.modules[_MINDFORMERS_ADAMW_MODULE] = sys.modules[MindFormersAdamW.__module__]
MindFormersAdamW.__module__ = _MINDFORMERS_ADAMW_MODULE


_TRAIN_STEPS = 4
_RTOL = 1e-6
_ATOL = 1e-6
_MEMORY_PARAM_SHAPE = (4096, 4096)
_MINDFORMERS_ADAMW_PARAM_SHAPE = (64, 64)
_CHECKPOINT_PARAM_SHAPE = (4, 4)
_CHECKPOINT_SMALL_PARAM_SHAPE = (1,)
_SWAP_TIMES = 3
_OPTIMIZER_ADAM = "adam"
_OPTIMIZER_ADAM_WEIGHT_DECAY = "adam_weight_decay"
_OPTIMIZER_MINDFORMERS_ADAMW = "mindformers_adamw"
_CHECKPOINT_OPTIMIZER_NAMES = (_OPTIMIZER_ADAM, _OPTIMIZER_ADAM_WEIGHT_DECAY)
_FULLY_SHARD_BATCH_SIZE = 2
_FULLY_SHARD_INPUT_DIM = 16
_FULLY_SHARD_HIDDEN_DIM = 32
_FULLY_SHARD_OUTPUT_DIM = 16
_ADAM_STATE_COUNT = 2


class _LargeAdamStateNet(nn.Cell):
    """Network with large Adam states for state and memory comparison."""

    def __init__(self):
        super().__init__()
        self.weight0 = Parameter(
            Tensor(np.full(_MEMORY_PARAM_SHAPE, 0.010, dtype=np.float32)),
            name="weight0",
        )
        self.weight1 = Parameter(
            Tensor(np.full(_MEMORY_PARAM_SHAPE, 0.011, dtype=np.float32)),
            name="weight1",
        )
        self.weight2 = Parameter(
            Tensor(np.full(_MEMORY_PARAM_SHAPE, 0.012, dtype=np.float32)),
            name="weight2",
        )

    def construct(self, x):
        x0, x1, x2 = x
        return self.weight0 * x0 + self.weight1 * x1 + self.weight2 * x2


class _MindFormersAdamWNet(nn.Cell):
    """Low-precision network that makes MindFormers AdamW create fp32 master params."""

    def __init__(self):
        super().__init__()
        self.weight0 = Parameter(
            Tensor(np.full(_MINDFORMERS_ADAMW_PARAM_SHAPE, 0.010, dtype=np.float16)),
            name="weight0",
        )
        self.weight1 = Parameter(
            Tensor(np.full(_MINDFORMERS_ADAMW_PARAM_SHAPE, 0.011, dtype=np.float16)),
            name="weight1",
        )
        self.weight2 = Parameter(
            Tensor(np.full(_MINDFORMERS_ADAMW_PARAM_SHAPE, 0.012, dtype=np.float16)),
            name="weight2",
        )

    def construct(self, x):
        x0, x1, x2 = x
        return (
            ops.cast(self.weight0, ms.float32) * x0
            + ops.cast(self.weight1, ms.float32) * x1
            + ops.cast(self.weight2, ms.float32) * x2
        )


class _CheckpointAdamStateNet(nn.Cell):
    """Small network used to exercise swap optimizer checkpoint paths."""

    def __init__(self):
        super().__init__()
        self.weight = Parameter(
            Tensor(np.full(_CHECKPOINT_PARAM_SHAPE, 0.010, dtype=np.float32)),
            name="ckpt_weight",
        )
        self.bias = Parameter(
            Tensor(np.full(_CHECKPOINT_SMALL_PARAM_SHAPE, 0.001, dtype=np.float32)),
            name="ckpt_bias",
        )

    def construct(self, x):
        return self.weight * x + self.bias


class _FullyShardAdamNet(nn.Cell):
    """Small deterministic network for fully_shard optimizer swap checks."""

    def __init__(self):
        super().__init__()
        self.proj0 = nn.Dense(
            _FULLY_SHARD_INPUT_DIM,
            _FULLY_SHARD_HIDDEN_DIM,
            has_bias=False,
            weight_init="zeros",
        )
        self.proj1 = nn.Dense(
            _FULLY_SHARD_HIDDEN_DIM,
            _FULLY_SHARD_OUTPUT_DIM,
            has_bias=False,
            weight_init="zeros",
        )
        weight0 = np.arange(
            _FULLY_SHARD_HIDDEN_DIM * _FULLY_SHARD_INPUT_DIM,
            dtype=np.float32,
        ).reshape(_FULLY_SHARD_HIDDEN_DIM, _FULLY_SHARD_INPUT_DIM)
        weight1 = np.arange(
            _FULLY_SHARD_OUTPUT_DIM * _FULLY_SHARD_HIDDEN_DIM,
            dtype=np.float32,
        ).reshape(_FULLY_SHARD_OUTPUT_DIM, _FULLY_SHARD_HIDDEN_DIM)
        self.proj0.weight.set_data(Tensor(weight0 / (10 * weight0.size), ms.float32))
        self.proj1.weight.set_data(Tensor(weight1 / (10 * weight1.size), ms.float32))
        self.bias = Parameter(
            Tensor(np.linspace(-0.04, 0.04, _FULLY_SHARD_OUTPUT_DIM, dtype=np.float32)),
            name="fully_shard_bias",
        )

    def construct(self, x):
        hidden = ops.relu(self.proj0(x))
        return self.proj1(hidden) + self.bias


def _make_batch_data(step):
    """Return large deterministic data for one training step."""
    x0 = np.full(_MEMORY_PARAM_SHAPE, 0.020 + step * 0.001, dtype=np.float32)
    x1 = np.full(_MEMORY_PARAM_SHAPE, 0.021 + step * 0.001, dtype=np.float32)
    x2 = np.full(_MEMORY_PARAM_SHAPE, 0.022 + step * 0.001, dtype=np.float32)
    y = np.full(_MEMORY_PARAM_SHAPE, 0.01 - step * 0.001, dtype=np.float32)
    return (Tensor(x0), Tensor(x1), Tensor(x2)), Tensor(y)


def _make_mindformers_adamw_batch_data(step):
    """Return deterministic data for one MindFormers AdamW training step."""
    shape = _MINDFORMERS_ADAMW_PARAM_SHAPE
    x0 = np.full(shape, 0.020 + step * 0.001, dtype=np.float32)
    x1 = np.full(shape, 0.021 + step * 0.001, dtype=np.float32)
    x2 = np.full(shape, 0.022 + step * 0.001, dtype=np.float32)
    y = np.full(shape, 0.01 - step * 0.001, dtype=np.float32)
    return (Tensor(x0), Tensor(x1), Tensor(x2)), Tensor(y)


def _make_fully_shard_batch_data(step):
    """Return deterministic fully_shard data for one training step."""
    x_base = np.arange(
        _FULLY_SHARD_BATCH_SIZE * _FULLY_SHARD_INPUT_DIM,
        dtype=np.float32,
    ).reshape(_FULLY_SHARD_BATCH_SIZE, _FULLY_SHARD_INPUT_DIM)
    y_base = np.arange(
        _FULLY_SHARD_BATCH_SIZE * _FULLY_SHARD_OUTPUT_DIM,
        dtype=np.float32,
    ).reshape(_FULLY_SHARD_BATCH_SIZE, _FULLY_SHARD_OUTPUT_DIM)
    x = x_base / x_base.size + (step + 1) * 0.003
    y = y_base / y_base.size - (step + 1) * 0.004
    return Tensor(x, ms.float32), Tensor(y, ms.float32)


def _to_numpy(value):
    """Convert a MindSpore Tensor/Parameter to numpy."""
    if isinstance(value, DTensor):
        return value.to_local().asnumpy()
    if hasattr(value, "asnumpy"):
        return value.asnumpy()
    return np.asarray(value)


def _is_cpu_value(value):
    """Return whether a tensor-like checkpoint value is host-resident."""
    tensor = getattr(value, "data", value)
    device = getattr(tensor, "device", None)
    return str(device).strip().lower().split(":", 1)[0] == "cpu"


def _assert_allclose(actual, expected, name):
    np.testing.assert_allclose(
        _to_numpy(actual),
        _to_numpy(expected),
        rtol=_RTOL,
        atol=_ATOL,
        err_msg=f"{name} mismatch",
    )


def get_forward_fn(net):
    """Return the loss forward function."""

    def forward_fn(x, y):
        diff = net(x) - y
        return ops.mean(diff * diff)

    return forward_fn


def _reset_peak_memory_stats():
    ms.runtime.empty_cache()
    ms.runtime.reset_peak_memory_stats()


def _device_peak_memory():
    return ms.runtime.max_memory_allocated()


def _fingerprint_value(value):
    """Return a compact, JSON-serializable fingerprint for a tensor-like value."""
    array = np.ascontiguousarray(_to_numpy(value))
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def _filled_tensor_like(value, fill_value):
    """Return a tensor with the same shape and dtype as ``value``."""
    return Tensor(np.full(tuple(value.shape), fill_value, dtype=np.float32), value.dtype)


def _make_optimizer(
        optimizer_name,
        params,
        use_nesterov=False,
        use_amsgrad=False,
        enable_fused_opt=False,
):
    """Build a native MindSpore optimizer for the swap ST."""
    if optimizer_name == _OPTIMIZER_ADAM:
        return nn.Adam(
            params,
            learning_rate=Tensor(0.01, ms.float32),
            beta1=0.8,
            beta2=0.9,
            eps=1e-6,
            use_nesterov=use_nesterov,
            use_amsgrad=use_amsgrad,
        )
    if optimizer_name == _OPTIMIZER_ADAM_WEIGHT_DECAY:
        if use_nesterov or use_amsgrad:
            raise ValueError("AdamWeightDecay does not support use_nesterov/use_amsgrad in this ST.")
        return nn.AdamWeightDecay(
            params,
            learning_rate=Tensor(0.01, ms.float32),
            beta1=0.8,
            beta2=0.9,
            eps=1e-6,
            weight_decay=0.01,
        )
    if optimizer_name == _OPTIMIZER_MINDFORMERS_ADAMW:
        if use_nesterov or use_amsgrad:
            raise ValueError("MindFormers AdamW does not use native Adam test flags.")
        return MindFormersAdamW(
            params,
            learning_rate=0.01,
            betas=(0.8, 0.9),
            eps=1e-6,
            weight_decay=0.01,
            enable_fused_opt=enable_fused_opt,
        )
    raise ValueError(f"Unsupported optimizer_name: {optimizer_name!r}")


def _checkpoint_primary_state_params(optimizer, optimizer_name):
    """Return the first moment state container used by checkpoint checks."""
    if optimizer_name == _OPTIMIZER_ADAM_WEIGHT_DECAY:
        return optimizer.moments1
    if optimizer_name == _OPTIMIZER_ADAM:
        return optimizer.moment1
    raise ValueError(f"Unsupported optimizer_name: {optimizer_name!r}")


def _train(
        use_swap,
        optimizer_name=_OPTIMIZER_ADAM,
        use_nesterov=False,
        use_amsgrad=False,
        packed_swap=False,
        include_master_params=False,
        enable_fused_opt=False,
):
    """Train a native optimizer or its swap wrapper for a few deterministic steps."""
    if optimizer_name == _OPTIMIZER_MINDFORMERS_ADAMW:
        net = _MindFormersAdamWNet()
        make_batch_data = _make_mindformers_adamw_batch_data
    else:
        net = _LargeAdamStateNet()
        make_batch_data = _make_batch_data
    optimizer = _make_optimizer(
        optimizer_name,
        net.trainable_params(),
        use_nesterov=use_nesterov,
        use_amsgrad=use_amsgrad,
        enable_fused_opt=enable_fused_opt,
    )
    if use_swap:
        optimizer = swap_optimizer(
            optimizer,
            SwapOptimizerConfig(
                swap_times=_SWAP_TIMES,
                min_numel=1,
                include_master_params=include_master_params,
                packed_swap=packed_swap,
            ),
        )

    params = tuple(net.trainable_params())

    losses = []
    peak_mem = 0
    for step in range(_TRAIN_STEPS):
        for param in params:
            param.grad = None
        x, y = make_batch_data(step)
        _reset_peak_memory_stats()
        loss = get_forward_fn(net)(x, y)
        loss.backward()
        grads = tuple(param.grad for param in params)
        with _no_grad():
            optimizer(grads)
        peak_mem = max(_device_peak_memory(), peak_mem)
        losses.append(loss)

    return net, optimizer, losses, peak_mem


def _run_train_once(
        use_swap,
        optimizer_name=_OPTIMIZER_ADAM,
        use_nesterov=False,
        use_amsgrad=False,
        packed_swap=False,
        include_master_params=False,
        enable_fused_opt=False,
):
    """Run one train in an isolated MindSpore context and summarize the result."""
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_seed(1)
    try:
        net, optimizer, losses, memory = _train(
            use_swap=use_swap,
            optimizer_name=optimizer_name,
            use_nesterov=use_nesterov,
            use_amsgrad=use_amsgrad,
            packed_swap=packed_swap,
            include_master_params=include_master_params,
            enable_fused_opt=enable_fused_opt,
        )
        param_items = tuple(net.parameters_and_names())
        state_dict = optimizer.state_dict()
        state_names = tuple(_optimizer_state_names(optimizer_name, optimizer))
        slots = tuple(optimizer.adapter.all_slots()) if use_swap else ()
        return {
            "losses": [float(loss.asnumpy()) for loss in losses],
            "optimizer_name": optimizer_name,
            "use_fused_opt": bool(getattr(optimizer, "use_fused_opt", False)),
            "enable_fused_opt": bool(getattr(optimizer, "enable_fused_opt", False)),
            "use_nesterov": bool(getattr(optimizer, "use_nesterov", False)),
            "use_amsgrad": bool(getattr(optimizer, "use_amsgrad", False)),
            "is_swap_optimizer": bool(getattr(optimizer, "_is_swap_optimizer", False)),
            "include_master_params": bool(
                getattr(getattr(optimizer, "config", None), "include_master_params", False)
            ),
            "packed_swap": bool(getattr(getattr(optimizer, "config", None), "packed_swap", False)),
            "packed_enabled": bool(getattr(getattr(optimizer, "runtime", None), "packed_enabled", False)),
            "model_names": [name for name, _ in param_items],
            "model_fingerprints": {name: _fingerprint_value(param) for name, param in param_items},
            "state_names": list(state_names),
            "state_fingerprints": {name: _fingerprint_value(state_dict[name]) for name in state_names},
            "slot_count": len(slots),
            "master_slot_count": len([slot for slot in slots if slot.name == MASTER_PARAM_KEY]),
            "swappable_master_slot_count": len([
                slot for slot in slots if slot.name == MASTER_PARAM_KEY and slot.swappable
            ]),
            "swappable_slot_count": len([slot for slot in slots if slot.swappable]),
            "host_slot_count": len([slot for slot in slots if slot.swappable and slot.state == "host"]),
            "packed_slot_count": len([slot for slot in slots if slot.packed]),
            "memory": memory,
        }
    finally:
        gc.collect()
        ms.runtime.empty_cache()


def _build_fully_shard_adam_net(mesh):
    """Build the fully_shard network used by optimizer swap alignment."""
    mp_policy = MixedPrecisionPolicy(
        param_dtype=ms.float32,
        reduce_dtype=ms.float32,
        output_dtype=ms.float32,
        cast_forward_inputs=False,
    )
    net = _FullyShardAdamNet()
    fully_shard(net.proj0, mesh=mesh, mp_policy=mp_policy)
    fully_shard(net.proj1, mesh=mesh, mp_policy=mp_policy)
    fully_shard(net, mesh=mesh, mp_policy=mp_policy)
    return net


def _fully_shard_adam_grads(net):
    """Collect fully_shard DTensor gradients in optimizer parameter order."""
    grads = []
    for index, param in enumerate(net.trainable_params()):
        grad = param.grad
        assert grad is not None, f"Parameter grad {index} is None"
        assert isinstance(grad, DTensor), f"Parameter grad {index} is not a DTensor"
        assert grad.shape == param.shape, (
            f"Gradient global shape mismatch at index {index}: "
            f"expected {param.shape}, got {grad.shape}"
        )
        assert grad.local_shape == param.local_shape, (
            f"Gradient local shape mismatch at index {index}: "
            f"expected {param.local_shape}, got {grad.local_shape}"
        )
        grads.append(grad)
    return tuple(grads)


def _fully_shard_optimizer_state_dict(optimizer):
    """Return checkpoint-visible state with swapped slots sourced from CPU mirrors."""
    return optimizer.state_dict()


def _run_fully_shard_adam_train_once(
        use_swap,
        mesh,
        optimizer_name=_OPTIMIZER_ADAM,
        packed_swap=True,
):
    """Train a fully_shard optimizer or its swap wrapper and summarize local state."""
    net = _build_fully_shard_adam_net(mesh)
    optimizer = _make_optimizer(optimizer_name, net.trainable_params())
    if use_swap:
        optimizer = swap_optimizer(
            optimizer,
            SwapOptimizerConfig(
                swap_times=_SWAP_TIMES,
                min_numel=1,
                packed_swap=packed_swap,
            ),
        )

    losses = []
    for step in range(_TRAIN_STEPS):
        net.zero_grad()
        x, y = _make_fully_shard_batch_data(step)
        loss = get_forward_fn(net)(x, y)
        loss.backward()
        grads = _fully_shard_adam_grads(net)
        with SkipDTensorDispatch(), _no_grad():
            optimizer(grads)
        losses.append(loss)

    param_items = tuple(net.parameters_and_names())
    state_dict = _fully_shard_optimizer_state_dict(optimizer) if use_swap else optimizer.state_dict()
    state_names = tuple(_optimizer_state_names(optimizer_name, optimizer))
    slots = tuple(optimizer.adapter.all_slots()) if use_swap else ()
    return {
        "losses": [float(loss.asnumpy()) for loss in losses],
        "optimizer_name": optimizer_name,
        "use_fused_opt": bool(getattr(optimizer, "use_fused_opt", False)),
        "enable_fused_opt": bool(getattr(optimizer, "enable_fused_opt", False)),
        "use_nesterov": bool(getattr(optimizer, "use_nesterov", False)),
        "use_amsgrad": bool(getattr(optimizer, "use_amsgrad", False)),
        "is_swap_optimizer": bool(getattr(optimizer, "_is_swap_optimizer", False)),
        "include_master_params": bool(
            getattr(getattr(optimizer, "config", None), "include_master_params", False)
        ),
        "packed_swap": bool(getattr(getattr(optimizer, "config", None), "packed_swap", False)),
        "packed_enabled": bool(getattr(getattr(optimizer, "runtime", None), "packed_enabled", False)),
        "rank": int(get_rank()),
        "world_size": int(get_group_size()),
        "model_names": [name for name, _ in param_items],
        "model_fingerprints": {name: _fingerprint_value(param) for name, param in param_items},
        "state_names": list(state_names),
        "state_fingerprints": {name: _fingerprint_value(state_dict[name]) for name in state_names},
        "param_count": len(param_items),
        "slot_count": len(slots),
        "master_slot_count": len([slot for slot in slots if slot.name == MASTER_PARAM_KEY]),
        "swappable_master_slot_count": len([
            slot for slot in slots if slot.name == MASTER_PARAM_KEY and slot.swappable
        ]),
        "swappable_slot_count": len([slot for slot in slots if slot.swappable]),
        "host_slot_count": len([slot for slot in slots if slot.swappable and slot.state == "host"]),
        "packed_slot_count": len([slot for slot in slots if slot.packed]),
    }


def _release_mindspore_memory():
    """Release host references and cached device memory between in-process runs."""
    gc.collect()
    ms.runtime.empty_cache()


def _run_train_once_in_subprocess(
        use_swap,
        optimizer_name=_OPTIMIZER_ADAM,
        use_nesterov=False,
        use_amsgrad=False,
        packed_swap=False,
        include_master_params=False,
        enable_fused_opt=False,
):
    """Run one train in a clean subprocess to avoid cross-run cache effects."""
    project_root = Path(__file__).resolve().parents[4]
    command = [
        sys.executable,
        "-c",
        (
            "import json; "
            "from tests.mindspore.st.swap_optimizer.swap_optimizer import _run_train_once; "
            f"result = _run_train_once(use_swap={use_swap!r}, optimizer_name={optimizer_name!r}, "
            f"use_nesterov={use_nesterov!r}, "
            f"use_amsgrad={use_amsgrad!r}, "
            f"packed_swap={packed_swap!r}, "
            f"include_master_params={include_master_params!r}, "
            f"enable_fused_opt={enable_fused_opt!r}); "
            "print('__SWAP_OPT_RESULT__' + json.dumps(result, sort_keys=True))"
        ),
    ]

    completed = subprocess.run(
        command,
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    marker = "__SWAP_OPT_RESULT__"
    for line in reversed(completed.stdout.splitlines()):
        if marker in line:
            return json.loads(line[line.find(marker) + len(marker):])

    raise RuntimeError(
        f"Train use_swap={use_swap!r} did not produce a result marker.\n"
        f"STDOUT:\n{completed.stdout}\n"
        f"STDERR:\n{completed.stderr}"
    )


def _train_checkpoint_swap_adam_once(optimizer_name=_OPTIMIZER_ADAM):
    """Train one small swap optimizer step for checkpoint behavior checks."""
    net = _CheckpointAdamStateNet()
    optimizer = _make_optimizer(optimizer_name, net.trainable_params())
    optimizer = swap_optimizer(
        optimizer,
        SwapOptimizerConfig(swap_times=1, min_numel=8, packed_swap=False),
    )

    params = tuple(net.trainable_params())
    x = Tensor(np.full(_CHECKPOINT_PARAM_SHAPE, 0.020, dtype=np.float32))
    y = Tensor(np.full(_CHECKPOINT_PARAM_SHAPE, 0.005, dtype=np.float32))
    loss = get_forward_fn(net)(x, y)
    loss.backward()
    optimizer(tuple(param.grad for param in params))
    return net, optimizer


def _make_mindformers_adamw_packed_checkpoint_optimizer():
    """Build a packed MindFormers AdamW swap optimizer with fp32 master slots."""
    net = _MindFormersAdamWNet()
    optimizer = _make_optimizer(_OPTIMIZER_MINDFORMERS_ADAMW, net.trainable_params())
    optimizer = swap_optimizer(
        optimizer,
        SwapOptimizerConfig(
            swap_times=_SWAP_TIMES,
            min_numel=1,
            include_master_params=True,
            packed_swap=True,
        ),
    )
    return net, optimizer


def _train_mindformers_adamw_packed_checkpoint_once():
    """Train one packed MindFormers AdamW step for checkpoint checks."""
    net, optimizer = _make_mindformers_adamw_packed_checkpoint_optimizer()
    params = tuple(net.trainable_params())
    x, y = _make_mindformers_adamw_batch_data(0)
    loss = get_forward_fn(net)(x, y)
    loss.backward()
    with _no_grad():
        optimizer(tuple(param.grad for param in params))
    return net, optimizer


def _run_checkpoint_roundtrip_once(optimizer_name=_OPTIMIZER_ADAM):
    """Exercise checkpoint save/load with swappable and non-swappable optimizer states."""
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_seed(1)
    original_load_state_dict = None
    try:
        net, optimizer = _train_checkpoint_swap_adam_once(optimizer_name)
        original_load_state_dict = optimizer.optimizer.load_state_dict
        model_param_names = {param.name for _, param in net.parameters_and_names()}
        model_name = sorted(model_param_names)[0]
        slots = tuple(optimizer.adapter.all_slots())
        swappable_slot = next(slot for slot in slots if slot.swappable)
        non_swappable_slot = next(slot for slot in slots if not slot.swappable)
        swappable_name = swappable_slot.tensor.name
        non_swappable_name = non_swappable_slot.tensor.name
        original_state = dict(optimizer.optimizer.state_dict())

        optimizer.runtime.restore_device_storage(swappable_slot)
        save_tensor = _filled_tensor_like(swappable_slot.tensor, 7.0)
        optimizer.runtime.load_into_tensor(swappable_slot.tensor, save_tensor)
        swappable_slot.state = "device"

        checkpoint_state = optimizer.state_dict()
        cpu_mirror_fingerprint = _fingerprint_value(swappable_slot.cpu_tensor)
        checkpoint_swappable_fingerprint = _fingerprint_value(checkpoint_state[swappable_name])

        load_tensor = _filled_tensor_like(swappable_slot.tensor, 3.0)
        load_non_swappable_tensor = _filled_tensor_like(non_swappable_slot.tensor, 0.321)
        load_model_tensor = _filled_tensor_like(original_state[model_name], 0.456)
        load_state = {
            swappable_name: Parameter(load_tensor, name=swappable_name),
            non_swappable_name: Parameter(load_non_swappable_tensor, name=non_swappable_name),
            model_name: Parameter(load_model_tensor, name=model_name),
        }
        load_state_dict_calls = []

        def capture_load_state_dict(state_dict, *args, **kwargs):
            load_state_dict_calls.append({
                "keys": sorted(state_dict.keys()),
                "strict": kwargs.get("strict"),
            })
            return original_load_state_dict(state_dict, *args, **kwargs)

        object.__setattr__(optimizer.optimizer, "load_state_dict", capture_load_state_dict)
        optimizer.load_state_dict(load_state)
        loaded_names = [name for call in load_state_dict_calls for name in call["keys"]]

        return {
            "optimizer_name": optimizer_name,
            "swappable_name": swappable_name,
            "non_swappable_name": non_swappable_name,
            "checkpoint_swappable_matches_cpu_mirror": checkpoint_swappable_fingerprint == cpu_mirror_fingerprint,
            "checkpoint_swappable_matches_latest": checkpoint_swappable_fingerprint == _fingerprint_value(save_tensor),
            "checkpoint_swappable_is_original": checkpoint_state[swappable_name] is original_state[swappable_name],
            "checkpoint_non_swappable_is_original": (
                checkpoint_state[non_swappable_name] is original_state[non_swappable_name]
            ),
            "checkpoint_model_param_is_original": checkpoint_state[model_name] is original_state[model_name],
            "checkpoint_keys_match_optimizer_state": list(checkpoint_state.keys()) == list(original_state.keys()),
            "checkpoint_includes_model_params": set(model_param_names).issubset(checkpoint_state),
            "load_swappable_matches_cpu_mirror": _fingerprint_value(swappable_slot.cpu_tensor)
                                           == _fingerprint_value(load_tensor),
            "load_swappable_state": swappable_slot.state,
            "load_state_dict_names": loaded_names,
            "load_state_dict_strict_values": [call["strict"] for call in load_state_dict_calls],
            "loaded_non_swappable_matches": _fingerprint_value(original_state[non_swappable_name])
                                           == _fingerprint_value(load_non_swappable_tensor),
            "loaded_model_param_matches": _fingerprint_value(original_state[model_name])
                                          == _fingerprint_value(load_model_tensor),
        }
    finally:
        if original_load_state_dict is not None:
            object.__setattr__(optimizer.optimizer, "load_state_dict", original_load_state_dict)
        gc.collect()
        ms.runtime.empty_cache()


def _run_checkpoint_fresh_load_once(optimizer_name=_OPTIMIZER_ADAM):
    """Load checkpoint state into a fresh swap optimizer with initially offloaded slots."""
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_seed(1)
    original_load_state_dict = None
    try:
        net = _CheckpointAdamStateNet()
        optimizer = _make_optimizer(optimizer_name, net.trainable_params())
        optimizer = swap_optimizer(
            optimizer,
            SwapOptimizerConfig(swap_times=1, min_numel=8, packed_swap=False),
        )
        original_load_state_dict = optimizer.optimizer.load_state_dict
        slot_count_before_load = len(tuple(optimizer.adapter.all_slots()))
        state_params = _checkpoint_primary_state_params(optimizer.optimizer, optimizer_name)
        swappable_name = state_params[0].name
        non_swappable_name = state_params[1].name
        load_tensor = _filled_tensor_like(state_params[0], 5.0)
        load_non_swappable_tensor = _filled_tensor_like(state_params[1], 0.123)
        load_state = {
            swappable_name: Parameter(load_tensor, name=swappable_name),
            non_swappable_name: Parameter(load_non_swappable_tensor, name=non_swappable_name),
        }
        load_state_dict_calls = []

        def capture_load_state_dict(state_dict, *args, **kwargs):
            load_state_dict_calls.append({
                "keys": sorted(state_dict.keys()),
                "strict": kwargs.get("strict"),
            })
            return original_load_state_dict(state_dict, *args, **kwargs)

        object.__setattr__(optimizer.optimizer, "load_state_dict", capture_load_state_dict)
        optimizer.load_state_dict(load_state)
        loaded_names = [name for call in load_state_dict_calls for name in call["keys"]]
        slots = tuple(optimizer.adapter.all_slots())
        swappable_slot = next(slot for slot in slots if getattr(slot.tensor, "name", None) == swappable_name)
        current_state = dict(optimizer.optimizer.state_dict())

        return {
            "optimizer_name": optimizer_name,
            "slot_count_before_load": slot_count_before_load,
            "slot_count_after_load": len(slots),
            "swappable_name": swappable_name,
            "non_swappable_name": non_swappable_name,
            "load_swappable_matches_cpu_mirror": _fingerprint_value(swappable_slot.cpu_tensor)
                                           == _fingerprint_value(load_tensor),
            "load_swappable_state": swappable_slot.state,
            "load_state_dict_names": loaded_names,
            "load_state_dict_strict_values": [call["strict"] for call in load_state_dict_calls],
            "loaded_non_swappable_matches": _fingerprint_value(current_state[non_swappable_name])
                                           == _fingerprint_value(load_non_swappable_tensor),
        }
    finally:
        if original_load_state_dict is not None:
            object.__setattr__(optimizer.optimizer, "load_state_dict", original_load_state_dict)
        gc.collect()
        ms.runtime.empty_cache()


def _run_mindformers_adamw_packed_checkpoint_roundtrip_once():
    """Save packed MindFormers AdamW state and load it into a fresh packed optimizer."""
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_seed(1)
    original_load_state_dict = None
    try:
        _, source_optimizer = _train_mindformers_adamw_packed_checkpoint_once()
        # The test validates the adapter's private checkpoint-slot mapping.
        source_slot_map = source_optimizer.adapter._checkpoint_slot_map()  # pylint: disable=protected-access
        source_optimizer_state = dict(source_optimizer.optimizer.state_dict())
        checkpoint_state = source_optimizer.state_dict()
        checkpoint_fingerprints = {
            name: _fingerprint_value(value) for name, value in checkpoint_state.items()
        }
        packed_names = set(source_slot_map)
        non_packed_names = set(checkpoint_state) - packed_names

        _, target_optimizer = _make_mindformers_adamw_packed_checkpoint_optimizer()
        # The test verifies that loading preserves the preallocated slot mapping.
        target_slot_map = target_optimizer.adapter._checkpoint_slot_map()  # pylint: disable=protected-access
        target_cpu_tensor_ids = {
            name: id(slot.cpu_tensor) for name, slot in target_slot_map.items()
        }
        original_load_state_dict = target_optimizer.optimizer.load_state_dict
        load_state_dict_calls = []

        def capture_load_state_dict(state_dict, *args, **kwargs):
            load_state_dict_calls.append({
                "keys": sorted(state_dict.keys()),
                "strict": kwargs.get("strict"),
            })
            return original_load_state_dict(state_dict, *args, **kwargs)

        object.__setattr__(target_optimizer.optimizer, "load_state_dict", capture_load_state_dict)
        target_optimizer.load_state_dict(checkpoint_state)
        loaded_names = {
            name for call in load_state_dict_calls for name in call["keys"]
        }
        loaded_state = target_optimizer.state_dict()

        return {
            "packed_slot_count": len(source_slot_map),
            "checkpoint_keys_match_optimizer_state": (
                list(checkpoint_state) == list(source_optimizer_state)
            ),
            "checkpoint_packed_matches_cpu_mirrors": all(
                checkpoint_fingerprints[name] == _fingerprint_value(slot.cpu_tensor)
                for name, slot in source_slot_map.items()
            ),
            "checkpoint_packed_values_are_replacements": all(
                checkpoint_state[name] is not source_optimizer_state[name]
                for name in packed_names
            ),
            "checkpoint_packed_values_are_cpu": all(
                _is_cpu_value(checkpoint_state[name]) for name in packed_names
            ),
            "checkpoint_non_packed_values_are_original": all(
                checkpoint_state[name] is source_optimizer_state[name]
                for name in non_packed_names
            ),
            "source_packed_slots_are_host_resident": all(
                slot.packed and slot.swappable and slot.state == "host" and slot.cpu_tensor is not None
                for slot in source_slot_map.values()
            ),
            "fresh_slot_names_match": set(target_slot_map) == packed_names,
            "loaded_packed_matches_checkpoint": all(
                _fingerprint_value(slot.cpu_tensor) == checkpoint_fingerprints[name]
                for name, slot in target_slot_map.items()
            ),
            "loaded_packed_cpu_views_are_reused": all(
                id(slot.cpu_tensor) == target_cpu_tensor_ids[name]
                for name, slot in target_slot_map.items()
            ),
            "loaded_packed_slots_are_host_resident": all(
                slot.packed and slot.swappable and slot.state == "host" and slot.cpu_tensor is not None
                for slot in target_slot_map.values()
            ),
            "load_delegates_only_non_packed_names": loaded_names == non_packed_names,
            "load_state_dict_strict_values": [call["strict"] for call in load_state_dict_calls],
            "loaded_state_matches_checkpoint": (
                list(loaded_state) == list(checkpoint_state)
                and all(
                    _fingerprint_value(loaded_state[name]) == checkpoint_fingerprints[name]
                    for name in checkpoint_state
                )
            ),
        }
    finally:
        if original_load_state_dict is not None:
            object.__setattr__(target_optimizer.optimizer, "load_state_dict", original_load_state_dict)
        gc.collect()
        ms.runtime.empty_cache()


def _run_checkpoint_roundtrip_once_in_subprocess(optimizer_name=_OPTIMIZER_ADAM):
    """Run checkpoint behavior checks in a clean subprocess."""
    project_root = Path(__file__).resolve().parents[4]
    command = [
        sys.executable,
        "-c",
        (
            "import json; "
            "from tests.mindspore.st.swap_optimizer.swap_optimizer import _run_checkpoint_roundtrip_once; "
            f"result = _run_checkpoint_roundtrip_once(optimizer_name={optimizer_name!r}); "
            "print('__SWAP_OPT_CHECKPOINT_RESULT__' + json.dumps(result, sort_keys=True))"
        ),
    ]

    completed = subprocess.run(
        command,
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    marker = "__SWAP_OPT_CHECKPOINT_RESULT__"
    for line in reversed(completed.stdout.splitlines()):
        if marker in line:
            return json.loads(line[line.find(marker) + len(marker):])

    raise RuntimeError(
        "Checkpoint roundtrip did not produce a result marker.\n"
        f"STDOUT:\n{completed.stdout}\n"
        f"STDERR:\n{completed.stderr}"
    )


def _run_checkpoint_fresh_load_once_in_subprocess(optimizer_name=_OPTIMIZER_ADAM):
    """Run fresh checkpoint load behavior checks in a clean subprocess."""
    project_root = Path(__file__).resolve().parents[4]
    command = [
        sys.executable,
        "-c",
        (
            "import json; "
            "from tests.mindspore.st.swap_optimizer.swap_optimizer import _run_checkpoint_fresh_load_once; "
            f"result = _run_checkpoint_fresh_load_once(optimizer_name={optimizer_name!r}); "
            "print('__SWAP_OPT_FRESH_LOAD_RESULT__' + json.dumps(result, sort_keys=True))"
        ),
    ]

    completed = subprocess.run(
        command,
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    marker = "__SWAP_OPT_FRESH_LOAD_RESULT__"
    for line in reversed(completed.stdout.splitlines()):
        if marker in line:
            return json.loads(line[line.find(marker) + len(marker):])

    raise RuntimeError(
        "Fresh checkpoint load did not produce a result marker.\n"
        f"STDERR:\n{completed.stderr}"
    )


def _run_mindformers_adamw_packed_checkpoint_roundtrip_once_in_subprocess():
    """Run packed MindFormers AdamW checkpoint checks in a clean subprocess."""
    project_root = Path(__file__).resolve().parents[4]
    command = [
        sys.executable,
        "-c",
        (
            "import json; "
            "from tests.mindspore.st.swap_optimizer.swap_optimizer import "
            "_run_mindformers_adamw_packed_checkpoint_roundtrip_once; "
            "result = _run_mindformers_adamw_packed_checkpoint_roundtrip_once(); "
            "print('__SWAP_OPT_MF_PACKED_CHECKPOINT_RESULT__' + json.dumps(result, sort_keys=True))"
        ),
    ]

    completed = subprocess.run(
        command,
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    marker = "__SWAP_OPT_MF_PACKED_CHECKPOINT_RESULT__"
    for line in reversed(completed.stdout.splitlines()):
        if marker in line:
            return json.loads(line[line.find(marker) + len(marker):])

    raise RuntimeError(
        "Packed MindFormers AdamW checkpoint roundtrip did not produce a result marker.\n"
        f"STDOUT:\n{completed.stdout}\n"
        f"STDERR:\n{completed.stderr}"
    )


def _optimizer_state_names(optimizer_name, optimizer):
    """Return optimizer state parameter names used for state-alignment checks."""
    if optimizer_name == _OPTIMIZER_MINDFORMERS_ADAMW:
        # MindFormers exposes this precision mask only as an internal attribute.
        low_precision_flags = optimizer._is_low_precision_param  # pylint: disable=protected-access
        return (
            "global_step",
            "learning_rate",
            *(param.name for param, is_low_precision in zip(
                optimizer.fp32_params, low_precision_flags
            ) if is_low_precision),
            *(param.name for param in optimizer.exp_avg),
            *(param.name for param in optimizer.exp_avg_sq),
        )
    if optimizer_name == _OPTIMIZER_ADAM_WEIGHT_DECAY:
        return tuple(
            [
                *(param.name for param in optimizer.moments1),
                *(param.name for param in optimizer.moments2),
            ]
        )
    return _adam_state_names(optimizer)


def _adam_state_names(optimizer):
    state_names = [
        "beta1_power",
        "beta2_power",
        *(param.name for param in optimizer.moment1),
        *(param.name for param in optimizer.moment2),
    ]
    if getattr(optimizer, "use_amsgrad", False):
        state_names.extend(param.name for param in optimizer.vhat)
    return tuple(state_names)


def _assert_train_results_align(base_result, swap_result):
    """Assert native Adam and swap Adam produce aligned training state."""
    assert base_result["optimizer_name"] == swap_result["optimizer_name"]
    assert base_result["use_fused_opt"] == swap_result["use_fused_opt"]
    assert base_result["enable_fused_opt"] == swap_result["enable_fused_opt"]
    assert base_result["use_nesterov"] == swap_result["use_nesterov"]
    assert base_result["use_amsgrad"] == swap_result["use_amsgrad"]
    assert len(base_result["losses"]) == len(swap_result["losses"]) == _TRAIN_STEPS
    for step, (base_loss, swap_loss) in enumerate(zip(base_result["losses"], swap_result["losses"])):
        _assert_allclose(swap_loss, base_loss, f"loss step {step}")

    assert base_result["model_names"] == swap_result["model_names"]
    assert len(swap_result["model_names"]) == _SWAP_TIMES
    for name in base_result["model_names"]:
        assert base_result["model_fingerprints"][name] == swap_result["model_fingerprints"][name], name

    assert base_result["state_names"] == swap_result["state_names"]
    for name in base_result["state_names"]:
        assert base_result["state_fingerprints"][name] == swap_result["state_fingerprints"][name], name


def _assert_adam_state_names(result):
    """Assert native Adam state names include moment1/moment2 and optional vhat."""
    assert len([name for name in result["state_names"] if name.startswith("moment")]) == 2 * _SWAP_TIMES
    assert len([name for name in result["state_names"] if name.startswith("vhat")]) == (
        _SWAP_TIMES if result["use_amsgrad"] else 0
    )


def _assert_adam_weight_decay_state_names(result):
    """Assert AdamWeightDecay state names include fused AdamW moment1/moment2 tensors."""
    assert result["use_fused_opt"] is True
    assert len([name for name in result["state_names"] if name.startswith("adam_m.")]) == _SWAP_TIMES
    assert len([name for name in result["state_names"] if name.startswith("adam_v.")]) == _SWAP_TIMES


def _assert_mindformers_adamw_swap_result(result, packed_swap):
    """Assert MindFormers AdamW registered state and used the requested swap path."""
    expected_slot_count = 3 * _SWAP_TIMES
    expected_swappable_count = expected_slot_count if packed_swap else 2 * _SWAP_TIMES
    assert result["enable_fused_opt"] is False
    assert result["is_swap_optimizer"] is True
    assert result["include_master_params"] is True
    assert result["packed_swap"] is packed_swap
    assert result["packed_enabled"] is packed_swap
    assert result["slot_count"] == expected_slot_count
    assert result["master_slot_count"] == _SWAP_TIMES
    assert result["swappable_master_slot_count"] == (_SWAP_TIMES if packed_swap else 0)
    assert result["swappable_slot_count"] == expected_swappable_count
    assert result["host_slot_count"] == expected_swappable_count
    assert result["packed_slot_count"] == (expected_slot_count if packed_swap else 0)
    assert len([name for name in result["state_names"] if name.startswith("main_param.")]) == _SWAP_TIMES
    assert len([name for name in result["state_names"] if name.startswith("adam_m.")]) == _SWAP_TIMES
    assert len([name for name in result["state_names"] if name.startswith("adam_v.")]) == _SWAP_TIMES


def _assert_mindformers_adamw_fused_swap_result(result, packed_swap):
    """Assert fused MindFormers AdamW swaps moments but not fp32 master params."""
    expected_slot_count = 2 * _SWAP_TIMES
    assert result["enable_fused_opt"] is True
    assert result["is_swap_optimizer"] is True
    assert result["include_master_params"] is False
    assert result["packed_swap"] is packed_swap
    assert result["packed_enabled"] is packed_swap
    assert result["slot_count"] == expected_slot_count
    assert result["master_slot_count"] == 0
    assert result["swappable_master_slot_count"] == 0
    assert result["swappable_slot_count"] == expected_slot_count
    assert result["host_slot_count"] == expected_slot_count
    assert result["packed_slot_count"] == (expected_slot_count if packed_swap else 0)
    assert len([name for name in result["state_names"] if name.startswith("main_param.")]) == _SWAP_TIMES
    assert len([name for name in result["state_names"] if name.startswith("adam_m.")]) == _SWAP_TIMES
    assert len([name for name in result["state_names"] if name.startswith("adam_v.")]) == _SWAP_TIMES


def _assert_fully_shard_train_results_align(base_result, swap_result):
    """Assert fully_shard baseline and swap optimizers produce aligned local state."""
    assert base_result["optimizer_name"] == swap_result["optimizer_name"]
    assert base_result["use_fused_opt"] == swap_result["use_fused_opt"]
    assert base_result["enable_fused_opt"] == swap_result["enable_fused_opt"]
    assert base_result["use_nesterov"] == swap_result["use_nesterov"]
    assert base_result["use_amsgrad"] == swap_result["use_amsgrad"]
    assert base_result["rank"] == swap_result["rank"]
    assert base_result["world_size"] == swap_result["world_size"]
    assert len(base_result["losses"]) == len(swap_result["losses"]) == _TRAIN_STEPS
    for step, (base_loss, swap_loss) in enumerate(zip(base_result["losses"], swap_result["losses"])):
        _assert_allclose(swap_loss, base_loss, f"fully_shard loss step {step}")

    assert base_result["model_names"] == swap_result["model_names"]
    assert base_result["param_count"] == swap_result["param_count"]
    for name in base_result["model_names"]:
        assert base_result["model_fingerprints"][name] == swap_result["model_fingerprints"][name], name

    assert base_result["state_names"] == swap_result["state_names"]
    for name in base_result["state_names"]:
        assert base_result["state_fingerprints"][name] == swap_result["state_fingerprints"][name], name


def _assert_fully_shard_adam_state_names(result):
    """Assert fully_shard Adam state names include moment1/moment2 for each local shard."""
    expected_param_count = result["param_count"]
    assert len([name for name in result["state_names"] if name.startswith("moment")]) == (
        _ADAM_STATE_COUNT * expected_param_count
    )


def _assert_fully_shard_swap_slots_registered(result):
    """Assert swap Adam registered Adam state slots for fully_shard parameters."""
    assert result["is_swap_optimizer"] is True
    assert result["slot_count"] == _ADAM_STATE_COUNT * result["param_count"]
    assert result["host_slot_count"] == result["swappable_slot_count"]


def _assert_fully_shard_mindformers_adamw_swap_result(result, packed_swap):
    """Assert fully_shard MindFormers AdamW used the requested swap path."""
    expected_param_count = result["param_count"]
    expected_slot_count = _ADAM_STATE_COUNT * expected_param_count
    assert result["enable_fused_opt"] is False
    assert result["is_swap_optimizer"] is True
    assert result["include_master_params"] is False
    assert result["packed_swap"] is packed_swap
    assert result["packed_enabled"] is packed_swap
    assert result["slot_count"] == expected_slot_count
    assert result["master_slot_count"] == 0
    assert result["swappable_master_slot_count"] == 0
    assert result["swappable_slot_count"] == expected_slot_count
    assert result["host_slot_count"] == expected_slot_count
    assert result["packed_slot_count"] == (expected_slot_count if packed_swap else 0)
    assert not [name for name in result["state_names"] if name.startswith("main_param.")]
    assert len([name for name in result["state_names"] if name.startswith("adam_m.")]) == expected_param_count
    assert len([name for name in result["state_names"] if name.startswith("adam_v.")]) == expected_param_count


def test_native_adam_swap_optimizer_state_align():
    """
    Feature: MindSpore native Adam swap optimizer.
    Description: Train with native Adam and native Adam wrapped by swap optimizer for several steps.
    Expectation: Parameters, moment1, moment2, beta powers are aligned; native Adam
        uses more device memory than swap Adam during forward and backward.
    """
    base_result = _run_train_once_in_subprocess(use_swap=False)
    swap_result = _run_train_once_in_subprocess(use_swap=True)

    _assert_train_results_align(base_result, swap_result)
    _assert_adam_state_names(swap_result)


def test_native_adam_nesterov_swap_optimizer_state_align():
    """
    Feature: MindSpore native Adam Nesterov swap optimizer.
    Description: Train with native Adam(use_nesterov=True) and the same optimizer wrapped by swap optimizer.
    Expectation: Parameters, moment1, moment2, beta powers are aligned; swap Adam uses less device memory.
    """
    base_result = _run_train_once_in_subprocess(use_swap=False, use_nesterov=True)
    swap_result = _run_train_once_in_subprocess(use_swap=True, use_nesterov=True)

    _assert_train_results_align(base_result, swap_result)
    _assert_adam_state_names(swap_result)


def test_native_adam_amsgrad_swap_optimizer_state_align():
    """
    Feature: MindSpore native Adam AMSGrad swap optimizer.
    Description: Train with native Adam(use_amsgrad=True) and the same optimizer wrapped by swap optimizer.
    Expectation: Parameters, moment1, moment2, vhat, beta powers are aligned; swap Adam uses less device memory.
    """
    base_result = _run_train_once_in_subprocess(use_swap=False, use_amsgrad=True)
    swap_result = _run_train_once_in_subprocess(use_swap=True, use_amsgrad=True)

    _assert_train_results_align(base_result, swap_result)
    _assert_adam_state_names(swap_result)


def test_native_adam_weight_decay_swap_optimizer_state_align():
    """
    Feature: MindSpore native AdamWeightDecay swap optimizer.
    Description: Train with native AdamWeightDecay(default use_fused_opt=True) and the same optimizer wrapped by
        swap optimizer for several steps.
    Expectation: Parameters, moments1 and moments2 are aligned; swap AdamWeightDecay uses less device memory.
    """
    base_result = _run_train_once_in_subprocess(
        use_swap=False,
        optimizer_name=_OPTIMIZER_ADAM_WEIGHT_DECAY,
    )
    swap_result = _run_train_once_in_subprocess(
        use_swap=True,
        optimizer_name=_OPTIMIZER_ADAM_WEIGHT_DECAY,
    )

    _assert_train_results_align(base_result, swap_result)
    _assert_adam_weight_decay_state_names(swap_result)


def test_mindformers_adamw_non_fused_swap_optimizer_state_align():
    """
    Feature: MindFormers AdamW non-fused swap optimizer.
    Description: Compare no-swap AdamW with per-tensor and packed swap using fp32 master parameter swapping.
    Expectation: Per-step losses, model parameters, optimizer moments, global step and fp32 masters are aligned.
    """
    base_result = _run_train_once_in_subprocess(
        use_swap=False,
        optimizer_name=_OPTIMIZER_MINDFORMERS_ADAMW,
    )

    for packed_swap in (False, True):
        swap_result = _run_train_once_in_subprocess(
            use_swap=True,
            optimizer_name=_OPTIMIZER_MINDFORMERS_ADAMW,
            packed_swap=packed_swap,
            include_master_params=True,
        )

        _assert_train_results_align(base_result, swap_result)
        _assert_mindformers_adamw_swap_result(swap_result, packed_swap)


def test_mindformers_adamw_fused_swap_optimizer_state_align():
    """
    Feature: MindFormers AdamW fused swap optimizer.
    Description: Compare no-swap fused AdamW with per-tensor and packed swap without fp32 master swapping.
    Expectation: Per-step losses, model parameters, optimizer moments, global step and fp32 masters are aligned.
    """
    base_result = _run_train_once_in_subprocess(
        use_swap=False,
        optimizer_name=_OPTIMIZER_MINDFORMERS_ADAMW,
        enable_fused_opt=True,
    )

    for packed_swap in (False, True):
        swap_result = _run_train_once_in_subprocess(
            use_swap=True,
            optimizer_name=_OPTIMIZER_MINDFORMERS_ADAMW,
            packed_swap=packed_swap,
            include_master_params=False,
            enable_fused_opt=True,
        )

        _assert_train_results_align(base_result, swap_result)
        _assert_mindformers_adamw_fused_swap_result(swap_result, packed_swap)


def test_native_adam_fully_shard_swap_optimizer_state_align_worker():
    """
    Feature: MindSpore fully_shard native Adam swap optimizer worker.
    Description: Train a fully_shard model with native Adam and with the same optimizer wrapped by swap optimizer.
    Expectation: Per-step losses, final local parameter shards and Adam optimizer states are aligned.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_seed(1)
    ms.set_deterministic(True)
    init()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "ep"))

    base_result = _run_fully_shard_adam_train_once(use_swap=False, mesh=mesh)

    swap_result = _run_fully_shard_adam_train_once(use_swap=True, mesh=mesh, packed_swap=False)

    _assert_fully_shard_train_results_align(base_result, swap_result)
    _assert_fully_shard_adam_state_names(swap_result)
    _assert_fully_shard_swap_slots_registered(swap_result)

    if get_rank() == 0:
        print(
            "fully_shard Adam swap optimizer align passed: "
            f"param_count={swap_result['param_count']}, "
            f"slot_count={swap_result['slot_count']}, "
            f"swappable_slot_count={swap_result['swappable_slot_count']}"
        )


def test_mindformers_adamw_fully_shard_swap_optimizer_state_align_worker():
    """
    Feature: MindSpore fully_shard MindFormers AdamW swap optimizer worker.
    Description: Compare fully_shard MindFormers AdamW baseline with per-tensor and packed swap modes.
    Expectation: Losses, local parameter shards and optimizer states are aligned.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_seed(1)
    ms.set_deterministic(True)
    init()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "ep"))
    base_result = _run_fully_shard_adam_train_once(
        use_swap=False,
        mesh=mesh,
        optimizer_name=_OPTIMIZER_MINDFORMERS_ADAMW,
    )
    _release_mindspore_memory()

    for packed_swap in (False, True):
        swap_result = _run_fully_shard_adam_train_once(
            use_swap=True,
            mesh=mesh,
            optimizer_name=_OPTIMIZER_MINDFORMERS_ADAMW,
            packed_swap=packed_swap,
        )

        _assert_fully_shard_train_results_align(base_result, swap_result)
        _assert_fully_shard_mindformers_adamw_swap_result(swap_result, packed_swap)
        _release_mindspore_memory()

    if get_rank() == 0:
        print("fully_shard MindFormers AdamW per-tensor and packed swap optimizer align passed")


def test_native_adam_swap_optimizer_checkpoint_cpu_mirror_roundtrip():
    """
    Feature: MindSpore native Adam/AdamWeightDecay swap optimizer checkpoint.
    Description: Save and load checkpoint state with swappable optimizer moments and non-swappable states.
    Expectation: Checkpoint save uses CPU mirrors for swappable tensors and original optimizer values for
        non-swappable states; checkpoint load writes swappable tensors to CPU mirrors and delegates
        non-swappable states to optimizer.load_state_dict.
    """
    for optimizer_name in _CHECKPOINT_OPTIMIZER_NAMES:
        result = _run_checkpoint_roundtrip_once_in_subprocess(optimizer_name)

        assert result["optimizer_name"] == optimizer_name
        for key in (
                "checkpoint_swappable_matches_cpu_mirror",
                "checkpoint_swappable_matches_latest",
                "checkpoint_non_swappable_is_original",
                "checkpoint_model_param_is_original",
                "checkpoint_keys_match_optimizer_state",
                "checkpoint_includes_model_params",
                "load_swappable_matches_cpu_mirror",
                "loaded_non_swappable_matches",
                "loaded_model_param_matches",
        ):
            assert result[key], (optimizer_name, key)
        assert not result["checkpoint_swappable_is_original"], optimizer_name
        assert result["load_swappable_state"] == "host", optimizer_name
        assert result["load_state_dict_strict_values"] == [False], optimizer_name
        assert result["swappable_name"] not in result["load_state_dict_names"], optimizer_name
        assert result["non_swappable_name"] in result["load_state_dict_names"], optimizer_name


def test_native_adam_swap_optimizer_checkpoint_fresh_load_builds_slots():
    """
    Feature: MindSpore native Adam/AdamWeightDecay swap optimizer checkpoint load.
    Description: Load swappable Adam state into a fresh swap optimizer before the first training step.
    Expectation: Checkpoint load reuses initial slots from optimizer state containers and keeps swappable state on CPU.
    """
    for optimizer_name in _CHECKPOINT_OPTIMIZER_NAMES:
        result = _run_checkpoint_fresh_load_once_in_subprocess(optimizer_name)

        assert result["optimizer_name"] == optimizer_name
        assert result["slot_count_before_load"] > 0, optimizer_name
        assert result["slot_count_after_load"] == result["slot_count_before_load"], optimizer_name
        assert result["load_swappable_matches_cpu_mirror"], optimizer_name
        assert result["load_swappable_state"] == "host", optimizer_name
        assert result["load_state_dict_strict_values"] == [False], optimizer_name
        assert result["swappable_name"] not in result["load_state_dict_names"], optimizer_name
        assert result["non_swappable_name"] in result["load_state_dict_names"], optimizer_name
        assert result["loaded_non_swappable_matches"], optimizer_name


def test_mindformers_adamw_packed_swap_optimizer_checkpoint_roundtrip():
    """
    Feature: MindFormers AdamW packed swap optimizer checkpoint.
    Description: Save packed moments and fp32 master params, then load them into a fresh packed optimizer.
    Expectation: Packed slots use checkpoint CPU copies and are restored into existing packed CPU views;
        non-packed state is delegated to optimizer.load_state_dict with strict=False.
    """
    result = _run_mindformers_adamw_packed_checkpoint_roundtrip_once_in_subprocess()

    assert result["packed_slot_count"] == 3 * _SWAP_TIMES
    for key in (
            "checkpoint_keys_match_optimizer_state",
            "checkpoint_packed_matches_cpu_mirrors",
            "checkpoint_packed_values_are_replacements",
            "checkpoint_packed_values_are_cpu",
            "checkpoint_non_packed_values_are_original",
            "source_packed_slots_are_host_resident",
            "fresh_slot_names_match",
            "loaded_packed_matches_checkpoint",
            "loaded_packed_cpu_views_are_reused",
            "loaded_packed_slots_are_host_resident",
            "load_delegates_only_non_packed_names",
            "loaded_state_matches_checkpoint",
    ):
        assert result[key], key
    assert result["load_state_dict_strict_values"] == [False]
