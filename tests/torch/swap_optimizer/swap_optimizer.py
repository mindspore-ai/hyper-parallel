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
"""Torch Adam swap optimizer ST cases."""

import gc
import inspect
from typing import Any, Dict, List, Optional, Tuple

import torch

from hyper_parallel import DTensor, SkipDTensorDispatch, init_device_mesh
from hyper_parallel.core.optimizer import SwapOptimizerConfig, swap_optimizer
from hyper_parallel.core.optimizer.adamw import AdamW as NewAdamW
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from tests.torch.utils import init_dist


_TRAIN_STEPS = 8
_SWAP_TIMES = 3
_RTOL = 1e-6
_ATOL = 1e-6
_BATCH_SIZE = 2
_INPUT_DIM = 2048
_HIDDEN_DIM = 4096
_OUTPUT_DIM = 2048
_PARAM_COUNT = 3
_DEFAULT_STATE_COUNT = 2
_AMSGRAD_STATE_COUNT = 3
_MEASURE_MEMORY_STEP = 1
_SWAPPABLE_STATE_KEYS = ("exp_avg", "exp_avg_sq")
_FSDP_INPUT_DIM = 16
_FSDP_HIDDEN_DIM = 32
_FSDP_OUTPUT_DIM = 16


class _AdamStateNet(torch.nn.Module):
    """Small deterministic network with three Adam state-bearing parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.weight0 = torch.nn.Parameter(torch.full((_INPUT_DIM, _HIDDEN_DIM), 0.010, dtype=torch.float32))
        self.weight1 = torch.nn.Parameter(torch.full((_HIDDEN_DIM, _OUTPUT_DIM), 0.011, dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.linspace(-0.020, 0.020, _OUTPUT_DIM, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward network."""
        hidden = torch.relu(torch.matmul(x, self.weight0))
        return torch.matmul(hidden, self.weight1) + self.bias


class _FullyShardAdamWNet(torch.nn.Module):
    """Small deterministic network for fully_shard + AdamW swap alignment."""

    def __init__(self) -> None:
        super().__init__()
        self.proj0 = torch.nn.Linear(_FSDP_INPUT_DIM, _FSDP_HIDDEN_DIM, bias=False)
        self.proj1 = torch.nn.Linear(_FSDP_HIDDEN_DIM, _FSDP_OUTPUT_DIM, bias=False)
        self.bias = torch.nn.Parameter(torch.linspace(-0.04, 0.04, _FSDP_OUTPUT_DIM, dtype=torch.float32))
        with torch.no_grad():
            weight0 = torch.arange(_FSDP_INPUT_DIM * _FSDP_HIDDEN_DIM, dtype=torch.float32)
            weight1 = torch.arange(_FSDP_HIDDEN_DIM * _FSDP_OUTPUT_DIM, dtype=torch.float32)
            self.proj0.weight.copy_(weight0.reshape(_FSDP_HIDDEN_DIM, _FSDP_INPUT_DIM) / (10 * weight0.numel()))
            self.proj1.weight.copy_(weight1.reshape(_FSDP_OUTPUT_DIM, _FSDP_HIDDEN_DIM) / (10 * weight1.numel()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward network."""
        hidden = torch.relu(self.proj0(x))
        return self.proj1(hidden) + self.bias


def _optimizer_kwargs(
        optimizer_cls: type[torch.optim.Optimizer],
        extra_kwargs: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Return optimizer options shared by the baseline and swap optimizer."""
    kwargs = {
        "lr": 0.01,
        "betas": (0.8, 0.9),
        "eps": 1e-6,
        "weight_decay": 0.01,
    }
    parameters = inspect.signature(optimizer_cls).parameters
    if "foreach" in parameters:
        kwargs["foreach"] = False
    if "fused" in parameters:
        kwargs["fused"] = False
    if extra_kwargs is not None:
        kwargs.update(extra_kwargs)
    return kwargs


def _make_batch(step: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic input and target tensors for one training step."""
    x_base = torch.arange(_BATCH_SIZE * _INPUT_DIM, dtype=torch.float32).reshape(
        _BATCH_SIZE,
        _INPUT_DIM,
    )
    target_base = torch.arange(_BATCH_SIZE * _OUTPUT_DIM, dtype=torch.float32).reshape(
        _BATCH_SIZE,
        _OUTPUT_DIM,
    )
    x = x_base / x_base.numel() + (step + 1) * 0.001
    target = target_base / target_base.numel() - (step + 1) * 0.002
    return x.npu(), target.npu()


def _make_fully_shard_batch(step: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic fully_shard input and target tensors for one step."""
    x_base = torch.arange(_BATCH_SIZE * _FSDP_INPUT_DIM, dtype=torch.float32).reshape(
        _BATCH_SIZE,
        _FSDP_INPUT_DIM,
    )
    target_base = torch.arange(_BATCH_SIZE * _FSDP_OUTPUT_DIM, dtype=torch.float32).reshape(
        _BATCH_SIZE,
        _FSDP_OUTPUT_DIM,
    )
    x = x_base / x_base.numel() + (step + 1) * 0.003
    target = target_base / target_base.numel() - (step + 1) * 0.004
    return x.npu(), target.npu()


def _release_device_memory() -> None:
    """Release cached device memory after dropping Python references."""
    gc.collect()
    torch.npu.empty_cache()


def _reset_peak_memory() -> None:
    """Reset peak memory stats for a measured step."""
    torch.npu.reset_peak_memory_stats()
    torch.npu.empty_cache()


def _peak_memory() -> int:
    """Return device peak memory in bytes."""
    return int(torch.npu.max_memory_allocated())


def _train_step(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
) -> Tuple[torch.Tensor, int]:
    """Run one training step and return peak memory."""
    optimizer.zero_grad(set_to_none=True)
    _reset_peak_memory()
    x, target = _make_batch(step)
    loss = torch.nn.functional.mse_loss(model(x), target)
    loss.backward()
    optimizer.step()
    return loss.detach().cpu().clone(), _peak_memory()


def _train(
        use_swap: bool,
        optimizer_cls: type[torch.optim.Optimizer],
        optimizer_extra_kwargs: Optional[Dict[str, object]] = None,
        group_step_history: Optional[List[int]] = None,
        eager_state: bool = False,
        measure_memory_step: int = _MEASURE_MEMORY_STEP,
        packed_swap: bool = True,
) -> Tuple[Dict[str, torch.Tensor], torch.optim.Optimizer, List[torch.Tensor], int]:
    """Train a native optimizer or its swap wrapper for several deterministic steps."""
    model = _AdamStateNet().npu()
    optimizer = optimizer_cls(
        model.parameters(),
        **_optimizer_kwargs(optimizer_cls, optimizer_extra_kwargs),
    )
    if eager_state:
        _materialize_optimizer_state(optimizer)
    if use_swap:
        optimizer = swap_optimizer(
            optimizer,
            SwapOptimizerConfig(swap_times=_SWAP_TIMES, min_numel=1, packed_swap=packed_swap),
        )

    losses = []
    measured_peak_memory = 0
    for step in range(_TRAIN_STEPS):
        loss, peak_memory = _train_step(
            model,
            optimizer,
            step,
        )
        if step >= measure_memory_step:
            measured_peak_memory = max(measured_peak_memory, peak_memory)
        if group_step_history is not None:
            group_step_history.append(int(optimizer.param_groups[0]["step"]))
        losses.append(loss)
    params = _named_parameters_on_cpu(model)
    return params, optimizer, losses, measured_peak_memory


def _materialize_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    """Materialize zero-valued Adam state without changing parameters or logical step."""
    assert not optimizer.state
    scalar_dtype = torch.float64 if torch.get_default_dtype() == torch.float64 else torch.float32
    for group in optimizer.param_groups:
        for param in group["params"]:
            if not param.requires_grad:
                continue
            state_device = (
                param.device
                if group.get("capturable", False) or group.get("fused", False)
                else torch.device("cpu")
            )
            state = optimizer.state[param]
            state["step"] = torch.zeros((), dtype=scalar_dtype, device=state_device)
            state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            if group.get("amsgrad", False):
                state["max_exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)

    assert len(optimizer.state) == _PARAM_COUNT
    assert all(float(state["step"].item()) == 0.0 for state in optimizer.state.values())


def _grouped_adam_param_groups(model: _AdamStateNet) -> List[Dict[str, object]]:
    """Return Adam param groups with intentionally different hyperparameters."""
    return [
        {
            "params": [model.weight0],
            "lr": 0.010,
            "weight_decay": 0.010,
            "betas": (0.8, 0.9),
        },
        {
            "params": [model.weight1],
            "lr": 0.006,
            "weight_decay": 0.030,
            "betas": (0.7, 0.95),
        },
        {
            "params": [model.bias],
            "lr": 0.020,
            "weight_decay": 0.0,
            "betas": (0.6, 0.99),
        },
    ]


def _train_grouped_adam(
        use_swap: bool,
        record_pipeline_batches: bool = False,
) -> Tuple[Dict[str, torch.Tensor], torch.optim.Optimizer, List[torch.Tensor], List[List[int]], int]:
    """Train grouped Adam or its swap wrapper and collect peak device memory."""
    model = _AdamStateNet().npu()
    optimizer = torch.optim.Adam(
        _grouped_adam_param_groups(model),
        **_optimizer_kwargs(torch.optim.Adam),
    )
    if use_swap:
        optimizer = swap_optimizer(
            optimizer,
            SwapOptimizerConfig(swap_times=_SWAP_TIMES, min_numel=1),
        )

    pipeline_batches = []
    original_run_pipeline = None
    if use_swap and record_pipeline_batches:
        original_run_pipeline = optimizer.runtime.run_pipeline

        def _record_run_pipeline(batches, step_context, step_batch):
            batch_lists = [list(batch) for batch in batches]
            pipeline_batches.append([len(batch) for batch in batch_lists])
            return original_run_pipeline(batch_lists, step_context, step_batch)

        optimizer.runtime.run_pipeline = _record_run_pipeline

    losses = []
    measured_peak_memory = 0
    try:
        for step in range(_TRAIN_STEPS):
            loss, peak_memory = _train_step(
                model,
                optimizer,
                step,
            )
            if step >= _MEASURE_MEMORY_STEP:
                measured_peak_memory = max(measured_peak_memory, peak_memory)
            losses.append(loss)
    finally:
        if original_run_pipeline is not None:
            optimizer.runtime.run_pipeline = original_run_pipeline

    params = _named_parameters_on_cpu(model)
    return params, optimizer, losses, pipeline_batches, measured_peak_memory


def _to_cpu_tensor(tensor: Any) -> torch.Tensor:
    """Return a detached CPU clone for Tensor or DTensor values."""
    if isinstance(tensor, DTensor):
        tensor = tensor.to_local()
    return tensor.detach().cpu().clone()


def _to_cpu_nested(value: Any) -> Any:
    """Recursively clone tensor leaves to CPU for stable comparisons."""
    if isinstance(value, DTensor):
        return _to_cpu_tensor(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu_nested(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu_nested(item) for item in value)
    return value


def _named_parameters_on_cpu(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Return detached CPU copies of model parameters."""
    return {
        name: _to_cpu_tensor(param)
        for name, param in model.named_parameters()
    }


def _optimizer_state_on_cpu(optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    """Return optimizer state in parameter order with tensor leaves on CPU."""
    state_dict = optimizer.state_dict()
    param_groups = []
    ordered_state = []
    for group in state_dict["param_groups"]:
        group_copy = {
            key: _to_cpu_nested(value)
            for key, value in group.items()
            if key != "params"
        }
        group_copy["param_count"] = len(group["params"])
        param_groups.append(group_copy)
        for param_id in group["params"]:
            ordered_state.append(_to_cpu_nested(state_dict["state"][param_id]))
    return {"param_groups": param_groups, "state": ordered_state}


def _build_fully_shard_adamw_model(mesh) -> _FullyShardAdamWNet:
    """Build a fully_shard model with mixed precision policy."""
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.float16,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
        cast_forward_inputs=True,
    )
    model = _FullyShardAdamWNet().npu()
    for layer in (model.proj0, model.proj1):
        fully_shard(
            layer,
            mesh=mesh,
            reshard_after_forward=True,
            mp_policy=mp_policy,
        )
    fully_shard(
        model,
        mesh=mesh,
        reshard_after_forward=True,
        mp_policy=mp_policy,
    )
    model.set_reduce_op_type("sum")
    return model


def _train_fully_shard_adamw(
        use_swap: bool,
        mesh,
        optimizer_cls: type[torch.optim.Optimizer],
        packed_swap: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], Optional[torch.optim.Optimizer], List[torch.Tensor], int, int]:
    """Train fully_shard AdamW or its swap wrapper and return CPU data plus peak memory."""
    model = _build_fully_shard_adamw_model(mesh)
    optimizer = optimizer_cls(
        model.parameters(),
        **_optimizer_kwargs(optimizer_cls),
    )
    if use_swap:
        optimizer = swap_optimizer(
            optimizer,
            SwapOptimizerConfig(swap_times=_SWAP_TIMES, min_numel=1, packed_swap=packed_swap),
        )
    losses = []
    measured_peak_memory = 0
    for step in range(_TRAIN_STEPS):
        _reset_peak_memory()
        x, target = _make_fully_shard_batch(step)
        loss = torch.nn.functional.mse_loss(model(x), target)
        loss.backward(torch.tensor(1.0 / mesh.size(), device=x.device))
        with SkipDTensorDispatch():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if step >= _MEASURE_MEMORY_STEP:
            measured_peak_memory = max(measured_peak_memory, _peak_memory())
        losses.append(_to_cpu_tensor(loss))

    params = _named_parameters_on_cpu(model)
    optimizer_state = _optimizer_state_on_cpu(optimizer)
    param_count = sum(1 for _ in model.parameters())
    return params, optimizer_state, optimizer if use_swap else None, losses, param_count, measured_peak_memory


def _assert_losses_align(base_losses: List[torch.Tensor], swap_losses: List[torch.Tensor]) -> None:
    """Assert per-step losses align."""
    assert len(base_losses) == len(swap_losses) == _TRAIN_STEPS
    for step, (base_loss, swap_loss) in enumerate(zip(base_losses, swap_losses)):
        torch.testing.assert_close(
            swap_loss,
            base_loss,
            rtol=_RTOL,
            atol=_ATOL,
            msg=(
                f"loss step {step} mismatch: base={base_loss.item():.10g}, "
                f"swap={swap_loss.item():.10g}, abs_diff={abs(base_loss.item() - swap_loss.item()):.10g}"
            ),
        )


def _assert_parameters_align(base_params: Dict[str, torch.Tensor], swap_params: Dict[str, torch.Tensor]) -> None:
    """Assert final parameters align."""
    assert base_params.keys() == swap_params.keys()
    for name, base_param in base_params.items():
        torch.testing.assert_close(
            swap_params[name],
            base_param,
            rtol=_RTOL,
            atol=_ATOL,
            msg=f"parameter {name} mismatch",
        )


def _assert_nested_align(base_value: Any, swap_value: Any, path: str) -> None:
    """Assert two nested optimizer-state values align."""
    if isinstance(base_value, torch.Tensor) and isinstance(swap_value, torch.Tensor):
        torch.testing.assert_close(
            swap_value,
            base_value,
            rtol=_RTOL,
            atol=_ATOL,
            msg=f"{path} mismatch",
        )
        return
    if isinstance(base_value, dict) and isinstance(swap_value, dict):
        assert base_value.keys() == swap_value.keys(), f"{path} keys mismatch"
        for key, base_item in base_value.items():
            _assert_nested_align(base_item, swap_value[key], f"{path}.{key}")
        return
    if isinstance(base_value, list) and isinstance(swap_value, list):
        assert len(base_value) == len(swap_value), f"{path} length mismatch"
        for index, (base_item, swap_item) in enumerate(zip(base_value, swap_value)):
            _assert_nested_align(base_item, swap_item, f"{path}[{index}]")
        return
    if isinstance(base_value, tuple) and isinstance(swap_value, tuple):
        assert len(base_value) == len(swap_value), f"{path} length mismatch"
        for index, (base_item, swap_item) in enumerate(zip(base_value, swap_value)):
            _assert_nested_align(base_item, swap_item, f"{path}[{index}]")
        return
    assert base_value == swap_value, f"{path} mismatch: {base_value!r} != {swap_value!r}"


def _assert_optimizer_state_align(base_state: Dict[str, Any], swap_state: Dict[str, Any]) -> None:
    """Assert optimizer param groups and state tensors align."""
    _assert_nested_align(base_state, swap_state, "optimizer_state")


def _assert_swap_slots_offloaded(
        optimizer: torch.optim.Optimizer,
        expected_state_count: int,
        expected_param_count: int = _PARAM_COUNT,
) -> None:
    """Assert real device swap slots were registered and offloaded."""
    assert bool(getattr(optimizer, "_is_swap_optimizer", False))
    slots = tuple(optimizer.adapter.all_slots())
    assert len(slots) == expected_param_count * expected_state_count
    assert all(slot.swappable for slot in slots)
    assert all(slot.cpu_tensor is not None for slot in slots)
    assert all(slot.state == "host" for slot in slots)
    if optimizer.runtime.packed_enabled:
        assert all(slot.packed for slot in slots)
        assert all(slot.tensor is slot.cpu_tensor for slot in slots)
        slot_names = {slot.name for slot in slots}
        published_state_tensors = [
            state[key]
            for state in optimizer.state.values()
            for key in slot_names
            if key in state
        ]
        assert len(published_state_tensors) == len(slots)
        assert {id(tensor) for tensor in published_state_tensors} == {id(slot.cpu_tensor) for slot in slots}
        assert all(slot.cpu_tensor.is_pinned() for slot in slots)
        for group in optimizer.param_groups:
            for param in group["params"]:
                step = optimizer.state.get(param, {}).get("step")
                if step is None:
                    continue
                expected_step_device = param.device if group.get("fused", False) else torch.device("cpu")
                assert step.device == expected_step_device
        # Verify packed staging storage is released after the optimizer step.
        # pylint: disable=protected-access
        assert all(
            arena is None or arena.raw_buffer.untyped_storage().size() == 0
            for arena in optimizer.runtime._staging_arenas
        )
        # pylint: enable=protected-access


def _assert_checkpoint_swappable_tensors_on_cpu(state_dict: Dict[str, object]) -> None:
    """Assert checkpoint swappable Adam state tensors are CPU resident."""
    assert len(state_dict["state"]) == _PARAM_COUNT
    for saved_state in state_dict["state"].values():
        for key in _SWAPPABLE_STATE_KEYS:
            assert key in saved_state
            assert saved_state[key].device.type == "cpu"


def _assert_peak_memory_reduced(base_peak_memory: int, swap_peak_memory: int) -> None:
    """Assert swap optimizer has lower peak device memory than the native optimizer."""
    print(f"base_memory={base_peak_memory}, swap_memory={swap_peak_memory}")
    assert base_peak_memory > swap_peak_memory, (
        "Expected native optimizer peak memory to be greater than swap optimizer, "
        f"but got base={base_peak_memory} bytes, swap={swap_peak_memory} bytes."
    )


def _run_optimizer_align_case(
        case_name: str,
        optimizer_cls: type[torch.optim.Optimizer],
        optimizer_extra_kwargs: Optional[Dict[str, object]] = None,
        expected_state_count: int = _DEFAULT_STATE_COUNT,
        verify_packed_swap: bool = False,
        packed_swap: bool = True,
) -> None:
    """Run one native-vs-swap optimizer alignment case."""
    print(case_name)
    base_params, base_optimizer, base_losses, base_peak_memory = _train(
        use_swap=False,
        optimizer_cls=optimizer_cls,
        optimizer_extra_kwargs=optimizer_extra_kwargs,
    )
    del base_optimizer
    _release_device_memory()

    swap_params, swap_optimizer_inst, swap_losses, swap_peak_memory = _train(
        use_swap=True,
        optimizer_cls=optimizer_cls,
        optimizer_extra_kwargs=optimizer_extra_kwargs,
        packed_swap=packed_swap,
    )

    _assert_losses_align(base_losses, swap_losses)
    _assert_parameters_align(base_params, swap_params)
    if verify_packed_swap:
        assert swap_optimizer_inst.runtime.packed_enabled
    else:
        assert swap_optimizer_inst.runtime.packed_enabled == packed_swap
    _assert_swap_slots_offloaded(swap_optimizer_inst, expected_state_count)
    _assert_peak_memory_reduced(base_peak_memory, swap_peak_memory)


def test_torch_adam_swap_optimizer_parameter_align() -> None:
    """
    Feature: Torch native Adam swap optimizer.
    Description: Train with torch.optim.Adam and torch.optim.Adam wrapped by swap optimizer for several steps.
    Expectation: Per-step losses and final parameters are aligned; swap Adam uses lower peak memory.
    """
    _run_optimizer_align_case("test_torch_adam_swap_optimizer_parameter_align", torch.optim.Adam)


def test_torch_adamw_swap_optimizer_parameter_align() -> None:
    """
    Feature: Torch native AdamW swap optimizer.
    Description: Train with torch.optim.AdamW and torch.optim.AdamW wrapped by swap optimizer for several steps.
    Expectation: Per-step losses and final parameters are aligned; swap AdamW uses lower peak memory.
    """
    _run_optimizer_align_case("test_torch_adamw_swap_optimizer_parameter_align", torch.optim.AdamW)


def test_torch_fused_adamw_swap_optimizer_parameter_align() -> None:
    """
    Feature: Torch fused AdamW per-tensor and packed swap optimizer.
    Description: Train with torch.optim.AdamW(fused=True) and both swap pipelines.
    Expectation: Both swap modes align with native fused AdamW, offload states, and reduce peak memory.
    """
    for packed_swap in (False, True):
        _run_optimizer_align_case(
            f"test_torch_fused_adamw_swap_optimizer_parameter_align_packed_{packed_swap}",
            torch.optim.AdamW,
            optimizer_extra_kwargs={"fused": True},
            verify_packed_swap=packed_swap,
            packed_swap=packed_swap,
        )


def test_torch_adamw_eager_state_swap_optimizer_parameter_align() -> None:
    """
    Feature: Torch AdamW swap optimizer with eagerly materialized optimizer state.
    Description: Materialize AdamW states before wrapping, then compare native and swap training from the first step.
    Expectation: Per-step results align; eager states are offloaded before step one and reduce its peak device memory.
    """
    print("test_torch_adamw_eager_state_swap_optimizer_parameter_align")
    base_params, base_optimizer, base_losses, base_peak_memory = _train(
        use_swap=False,
        optimizer_cls=torch.optim.AdamW,
        eager_state=True,
        measure_memory_step=0,
    )
    del base_optimizer
    _release_device_memory()

    swap_params, swap_optimizer_inst, swap_losses, swap_peak_memory = _train(
        use_swap=True,
        optimizer_cls=torch.optim.AdamW,
        eager_state=True,
        measure_memory_step=0,
    )

    _assert_losses_align(base_losses, swap_losses)
    _assert_parameters_align(base_params, swap_params)
    _assert_swap_slots_offloaded(swap_optimizer_inst, _DEFAULT_STATE_COUNT)
    _assert_peak_memory_reduced(base_peak_memory, swap_peak_memory)


def test_torch_adam_amsgrad_swap_optimizer_parameter_align() -> None:
    """
    Feature: Torch native Adam AMSGrad swap optimizer.
    Description: Train with torch.optim.Adam(amsgrad=True) and the same optimizer wrapped by swap optimizer.
    Expectation: Per-step losses and final parameters are aligned; swap Adam AMSGrad uses lower peak memory.
    """
    _run_optimizer_align_case(
        "test_torch_adam_amsgrad_swap_optimizer_parameter_align",
        torch.optim.Adam,
        optimizer_extra_kwargs={"amsgrad": True},
        expected_state_count=_AMSGRAD_STATE_COUNT,
    )


def test_new_adamw_swap_optimizer_parameter_align() -> None:
    """
    Feature: HyperParallel AdamW packed swap optimizer.
    Description: Train with HyperParallel AdamW and the same optimizer wrapped in explicitly enabled packed swap mode.
    Expectation: Results align; packed swap storage is used and lowers peak memory.
    """
    print("test_new_adamw_swap_optimizer_parameter_align")
    base_group_steps = []
    swap_group_steps = []
    base_params, base_optimizer, base_losses, base_peak_memory = _train(
        use_swap=False,
        optimizer_cls=NewAdamW,
        group_step_history=base_group_steps,
    )
    del base_optimizer
    _release_device_memory()

    swap_params, swap_optimizer_inst, swap_losses, swap_peak_memory = _train(
        use_swap=True,
        optimizer_cls=NewAdamW,
        group_step_history=swap_group_steps,
        packed_swap=True,
    )

    expected_group_steps = list(range(1, _TRAIN_STEPS + 1))
    assert base_group_steps == expected_group_steps
    assert swap_group_steps == expected_group_steps
    _assert_losses_align(base_losses, swap_losses)
    _assert_parameters_align(base_params, swap_params)
    assert swap_optimizer_inst.runtime.packed_enabled
    _assert_swap_slots_offloaded(swap_optimizer_inst, _DEFAULT_STATE_COUNT)
    _assert_peak_memory_reduced(base_peak_memory, swap_peak_memory)


def test_new_adamw_amsgrad_swap_optimizer_parameter_align() -> None:
    """
    Feature: HyperParallel AdamW AMSGrad packed swap optimizer.
    Description: Train with HyperParallel AdamW(amsgrad=True) and the same optimizer in explicitly enabled packed
        swap mode.
    Expectation: Results align; packed swap storage is used and lowers peak memory.
    """
    _run_optimizer_align_case(
        "test_new_adamw_amsgrad_swap_optimizer_parameter_align",
        NewAdamW,
        optimizer_extra_kwargs={"amsgrad": True},
        expected_state_count=_AMSGRAD_STATE_COUNT,
        verify_packed_swap=True,
    )


def test_torch_adam_swap_optimizer_multi_param_group_align() -> None:
    """
    Feature: Torch native Adam swap optimizer with multiple param groups.
    Description: Train torch.optim.Adam with different lr, weight_decay and betas per param group; compare with
        the same optimizer wrapped by swap optimizer after pipeline batch partitioning.
    Expectation: Per-step losses and final parameters are aligned, swap batches are split across param groups, and
        swap optimizer peak device memory is lower than the native optimizer.
    """
    print("test_torch_adam_swap_optimizer_multi_param_group_align")
    base_params, base_optimizer, base_losses, _, base_peak_memory = _train_grouped_adam(use_swap=False)
    del base_optimizer
    _release_device_memory()

    swap_params, swap_optimizer_inst, swap_losses, pipeline_batches, swap_peak_memory = _train_grouped_adam(
        use_swap=True,
        record_pipeline_batches=True,
    )

    assert len(pipeline_batches) == _TRAIN_STEPS
    assert all(batch_sizes == [1, 1, 1] for batch_sizes in pipeline_batches)
    _assert_losses_align(base_losses, swap_losses)
    _assert_parameters_align(base_params, swap_params)
    _assert_swap_slots_offloaded(swap_optimizer_inst, _DEFAULT_STATE_COUNT)
    _assert_peak_memory_reduced(base_peak_memory, swap_peak_memory)


def test_fully_shard_adamw_mixed_precision_swap_optimizer_parameter_align() -> None:
    """
    Feature: fully_shard AdamW mixed precision swap optimizer.
    Description: Train a fully_shard model with AdamW optimizers and with the same optimizer wrapped by swap optimizer
        under MixedPrecisionPolicy.
    Expectation: Per-step losses, final local parameter shards and AdamW optimizer states are aligned.
    """
    print("test_fully_shard_adamw_mixed_precision_swap_optimizer_parameter_align")
    init_dist()
    mesh = init_device_mesh(device_type="npu", mesh_shape=(4,), mesh_dim_names=("dp",))

    fully_shard_adamw_cases = (
        ("torch_adamw", torch.optim.AdamW),
        ("new_adamw", NewAdamW),
    )

    for case_name, optimizer_cls in fully_shard_adamw_cases:
        print(case_name)
        base_params, base_optimizer_state, _, base_losses, _, _ = _train_fully_shard_adamw(
            use_swap=False,
            mesh=mesh,
            optimizer_cls=optimizer_cls,
        )
        _release_device_memory()

        swap_params, swap_optimizer_state, swap_optimizer_inst, swap_losses, param_count, _ = _train_fully_shard_adamw(
            use_swap=True,
            mesh=mesh,
            optimizer_cls=optimizer_cls,
        )

        _assert_losses_align(base_losses, swap_losses)
        _assert_parameters_align(base_params, swap_params)
        _assert_optimizer_state_align(base_optimizer_state, swap_optimizer_state)
        _assert_swap_slots_offloaded(swap_optimizer_inst, _DEFAULT_STATE_COUNT, expected_param_count=param_count)
        _release_device_memory()


def test_fully_shard_optimizer_swap_adamw_4card_parameter_align() -> None:
    """
    Feature: Four-card fully_shard with packed and non-packed swap AdamW optimizers.
    Description: Train a four-card fully_shard model with native AdamW and both swap modes.
    Expectation: Both swap modes align with the baseline, use the expected storage mode, and use lower peak device
        memory than native AdamW.
    """
    print("test_fully_shard_optimizer_swap_adamw_4card_parameter_align")
    init_dist()
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "op"))

    base_params, base_optimizer_state, _, base_losses, _, base_peak_memory = _train_fully_shard_adamw(
        use_swap=False,
        mesh=mesh,
        optimizer_cls=torch.optim.AdamW,
    )
    _release_device_memory()

    (
        non_packed_params,
        non_packed_optimizer_state,
        non_packed_optimizer,
        non_packed_losses,
        param_count,
        non_packed_peak_memory,
    ) = (
        _train_fully_shard_adamw(
            use_swap=True,
            mesh=mesh,
            optimizer_cls=torch.optim.AdamW,
            packed_swap=False,
        )
    )

    assert not non_packed_optimizer.runtime.packed_enabled
    _assert_losses_align(base_losses, non_packed_losses)
    _assert_parameters_align(base_params, non_packed_params)
    _assert_optimizer_state_align(base_optimizer_state, non_packed_optimizer_state)
    _assert_swap_slots_offloaded(
        non_packed_optimizer,
        _DEFAULT_STATE_COUNT,
        expected_param_count=param_count,
    )
    _assert_peak_memory_reduced(base_peak_memory, non_packed_peak_memory)
    del non_packed_optimizer
    _release_device_memory()

    swap_params, swap_optimizer_state, swap_optimizer_inst, swap_losses, param_count, swap_peak_memory = (
        _train_fully_shard_adamw(
            use_swap=True,
            mesh=mesh,
            optimizer_cls=torch.optim.AdamW,
            packed_swap=True,
        )
    )

    assert swap_optimizer_inst.runtime.packed_enabled
    _assert_losses_align(base_losses, swap_losses)
    _assert_parameters_align(base_params, swap_params)
    _assert_optimizer_state_align(base_optimizer_state, swap_optimizer_state)
    _assert_swap_slots_offloaded(
        swap_optimizer_inst,
        _DEFAULT_STATE_COUNT,
        expected_param_count=param_count,
    )
    _assert_peak_memory_reduced(base_peak_memory, swap_peak_memory)

    checkpoint = swap_optimizer_inst.state_dict()
    loaded_model = _build_fully_shard_adamw_model(mesh)
    loaded_optimizer = swap_optimizer(
        torch.optim.AdamW(
            loaded_model.parameters(),
            **_optimizer_kwargs(torch.optim.AdamW),
        ),
        SwapOptimizerConfig(swap_times=_SWAP_TIMES, min_numel=1, packed_swap=True),
    )
    loaded_optimizer.load_state_dict(checkpoint)
    loaded_slots = tuple(loaded_optimizer.adapter.all_slots())
    assert loaded_slots
    assert all(isinstance(slot.logical_tensor, DTensor) for slot in loaded_slots)

    x, target = _make_fully_shard_batch(_TRAIN_STEPS)
    loss = torch.nn.functional.mse_loss(loaded_model(x), target)
    loss.backward(torch.tensor(1.0 / mesh.size(), device=x.device))
    with SkipDTensorDispatch():
        loaded_optimizer.step()
        loaded_optimizer.zero_grad(set_to_none=True)
    _assert_swap_slots_offloaded(
        loaded_optimizer,
        _DEFAULT_STATE_COUNT,
        expected_param_count=param_count,
    )
    _release_device_memory()


def _run_optimizer_checkpoint_host_state_case(
        case_name: str,
        optimizer_cls: type[torch.optim.Optimizer],
        optimizer_extra_kwargs: Optional[Dict[str, object]] = None,
        expected_state_count: int = _DEFAULT_STATE_COUNT,
) -> None:
    """Run one optimizer checkpoint case with host-resident swappable state."""
    print(case_name)
    model = _AdamStateNet().npu()
    optimizer = swap_optimizer(
        optimizer_cls(
            model.parameters(),
            **_optimizer_kwargs(optimizer_cls, optimizer_extra_kwargs),
        ),
        SwapOptimizerConfig(swap_times=_SWAP_TIMES, min_numel=1),
    )
    _train_step(model, optimizer, step=0)
    _assert_swap_slots_offloaded(optimizer, expected_state_count)

    save_calls = {"copy_to_cpu": 0, "copy_to_device": 0, "restore_device_storage": 0}
    original_copy_to_cpu = optimizer.runtime.copy_to_cpu
    original_copy_to_device = optimizer.runtime.copy_to_device
    original_restore_device_storage = optimizer.runtime.restore_device_storage

    def _record_save_copy_to_cpu(slot):
        save_calls["copy_to_cpu"] += 1
        return original_copy_to_cpu(slot)

    def _record_save_copy_to_device(slot):
        save_calls["copy_to_device"] += 1
        return original_copy_to_device(slot)

    def _record_save_restore_device_storage(slot):
        save_calls["restore_device_storage"] += 1
        return original_restore_device_storage(slot)

    optimizer.runtime.copy_to_cpu = _record_save_copy_to_cpu
    optimizer.runtime.copy_to_device = _record_save_copy_to_device
    optimizer.runtime.restore_device_storage = _record_save_restore_device_storage
    try:
        state_dict = optimizer.state_dict()
    finally:
        optimizer.runtime.copy_to_cpu = original_copy_to_cpu
        optimizer.runtime.copy_to_device = original_copy_to_device
        optimizer.runtime.restore_device_storage = original_restore_device_storage

    assert save_calls == {"copy_to_cpu": 0, "copy_to_device": 0, "restore_device_storage": 0}
    _assert_checkpoint_swappable_tensors_on_cpu(state_dict)

    loaded_model = _AdamStateNet().npu()
    loaded_optimizer = swap_optimizer(
        optimizer_cls(
            loaded_model.parameters(),
            **_optimizer_kwargs(optimizer_cls, optimizer_extra_kwargs),
        ),
        SwapOptimizerConfig(swap_times=_SWAP_TIMES, min_numel=1),
    )

    load_calls = {"full_h2d": 0, "copy_to_device": 0, "restore_device_storage": 0}
    original_make_device_tensor_like = loaded_optimizer.runtime.make_device_tensor_like
    original_load_copy_to_device = loaded_optimizer.runtime.copy_to_device
    original_load_restore_device_storage = loaded_optimizer.runtime.restore_device_storage

    def _record_make_device_tensor_like(param, saved_tensor):
        load_calls["full_h2d"] += 1
        return original_make_device_tensor_like(param, saved_tensor)

    def _record_load_copy_to_device(slot):
        load_calls["copy_to_device"] += 1
        return original_load_copy_to_device(slot)

    def _record_load_restore_device_storage(slot):
        load_calls["restore_device_storage"] += 1
        return original_load_restore_device_storage(slot)

    loaded_optimizer.runtime.make_device_tensor_like = _record_make_device_tensor_like
    loaded_optimizer.runtime.copy_to_device = _record_load_copy_to_device
    loaded_optimizer.runtime.restore_device_storage = _record_load_restore_device_storage
    try:
        loaded_optimizer.load_state_dict(state_dict)
    finally:
        loaded_optimizer.runtime.make_device_tensor_like = original_make_device_tensor_like
        loaded_optimizer.runtime.copy_to_device = original_load_copy_to_device
        loaded_optimizer.runtime.restore_device_storage = original_load_restore_device_storage

    assert load_calls == {"full_h2d": 0, "copy_to_device": 0, "restore_device_storage": 0}
    _assert_swap_slots_offloaded(loaded_optimizer, expected_state_count)
    assert all(slot.tensor is slot.cpu_tensor for slot in loaded_optimizer.adapter.all_slots())

    prefetch_calls = {
        "copy_to_device": 0,
        "restore_device_storage": 0,
        "packed_h2d": 0,
        "packed_d2h": 0,
    }
    original_prefetch_copy_to_device = loaded_optimizer.runtime.copy_to_device
    original_prefetch_restore_device_storage = loaded_optimizer.runtime.restore_device_storage
    # Instrument the internal packed copy chain to verify its checkpoint path.
    # pylint: disable=protected-access
    original_packed_h2d = loaded_optimizer.runtime._copy_packed_to_device
    original_packed_d2h = loaded_optimizer.runtime._copy_packed_to_host

    def _record_prefetch_copy_to_device(slot):
        prefetch_calls["copy_to_device"] += 1
        return original_prefetch_copy_to_device(slot)

    def _record_prefetch_restore_device_storage(slot):
        prefetch_calls["restore_device_storage"] += 1
        return original_prefetch_restore_device_storage(slot)

    def _record_packed_h2d(batch_index, staging_index):
        prefetch_calls["packed_h2d"] += 1
        return original_packed_h2d(batch_index, staging_index)

    def _record_packed_d2h(batch_index, staging_index):
        prefetch_calls["packed_d2h"] += 1
        return original_packed_d2h(batch_index, staging_index)

    loaded_optimizer.runtime.copy_to_device = _record_prefetch_copy_to_device
    loaded_optimizer.runtime.restore_device_storage = _record_prefetch_restore_device_storage
    loaded_optimizer.runtime._copy_packed_to_device = _record_packed_h2d
    loaded_optimizer.runtime._copy_packed_to_host = _record_packed_d2h
    try:
        _train_step(loaded_model, loaded_optimizer, step=1)
    finally:
        loaded_optimizer.runtime.copy_to_device = original_prefetch_copy_to_device
        loaded_optimizer.runtime.restore_device_storage = original_prefetch_restore_device_storage
        loaded_optimizer.runtime._copy_packed_to_device = original_packed_h2d
        loaded_optimizer.runtime._copy_packed_to_host = original_packed_d2h
    # pylint: enable=protected-access

    expected_batch_count = min(_SWAP_TIMES, _PARAM_COUNT)
    assert prefetch_calls == {
        "copy_to_device": 0,
        "restore_device_storage": 0,
        "packed_h2d": expected_batch_count,
        "packed_d2h": expected_batch_count,
    }
    _assert_swap_slots_offloaded(loaded_optimizer, expected_state_count)


def test_torch_adam_swap_optimizer_checkpoint_host_state() -> None:
    """
    Feature: Torch Adam/AdamW swap optimizer checkpoint with host-resident state.
    Description: Save and load checkpoint when all swappable Adam/AdamW states are offloaded to host.
    Expectation: Saving returns CPU swappable tensors without full H2D; loading keeps states host-resident until the
        next step performs on-demand H2D prefetch.
    """
    checkpoint_cases = (
        ("test_torch_adam_swap_optimizer_checkpoint_host_state", torch.optim.Adam),
        ("test_torch_adamw_swap_optimizer_checkpoint_host_state", torch.optim.AdamW),
        ("test_new_adamw_swap_optimizer_checkpoint_host_state", NewAdamW),
    )
    for case_name, optimizer_cls in checkpoint_cases:
        _run_optimizer_checkpoint_host_state_case(case_name, optimizer_cls)
        _release_device_memory()
