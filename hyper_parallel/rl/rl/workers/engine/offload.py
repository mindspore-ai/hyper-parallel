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
"""Stage-boundary residency using existing HSDP model conversion contracts."""

from typing import Any, Iterator

import torch

from hyper_parallel.core.optimizer.dtensor_compat import to_local_if_dtensor


def _state_tensors(value: Any) -> Iterator[torch.Tensor]:
    """Visit tensor leaves without copying or changing optimizer state structure."""
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _state_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _state_tensors(item)


def _tensor_bytes(tensor: torch.Tensor) -> int:
    """Count local storage payload without entering distributed operator dispatch."""
    local = to_local_if_dtensor(tensor)
    return local.numel() * local.element_size()


def _transfer_bytes(tensors: dict[str, torch.Tensor], devices: dict[str, torch.device]) -> int:
    """Count tied parameters once, as module conversion preserves their identity."""
    seen = set()
    total = 0
    for name, tensor in tensors.items():
        if id(tensor) not in seen and tensor.device != devices[name]:
            total += _tensor_bytes(tensor)
        seen.add(id(tensor))
    return total


def _model_tensors(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Name parameters and buffers separately for mixed-device restoration."""
    return {**{f"parameter:{name}": tensor for name, tensor in model.named_parameters(remove_duplicate=False)},
            **{f"buffer:{name}": tensor for name, tensor in model.named_buffers(remove_duplicate=False)}}


class TrainingStateOffload:
    """Move idle training state to CPU and restore its original placement.

    This does not change HSDP's CPUOffloadPolicy or the optimizer implementation.
    Callers must reshard and synchronize before offload, and sleep inference
    before restore. Copies are blocking; no unsynchronized host reads are used.
    A failed conversion is fatal to the enclosing distributed training step.

    Args:
        roles: Training models and their optional optimizers.
    """

    def __init__(self, roles: list[tuple[torch.nn.Module, Any]]) -> None:
        """Track role residency without changing the configured training offload policy."""
        self.roles = roles
        self.offloaded = False
        self._model_devices: list[dict[str, torch.device]] = []
        self._optimizer_tensors: list[tuple[torch.Tensor, torch.device]] = []

    def _validate_idle(self) -> None:
        """Reject live gradients instead of silently discarding an update."""
        for model, optimizer in self.roles:
            parameters = list(model.parameters())
            if optimizer is not None:
                for component in getattr(optimizer, "chained_optimizers", (optimizer,)):
                    parameters.extend(p for group in component.param_groups for p in group["params"])
            if any(p.grad is not None or getattr(p, "main_grad", None) is not None for p in parameters):
                raise RuntimeError("Rollout offload requires cleared gradients after the optimizer update")

    @torch.no_grad()
    def offload(self) -> int:
        """Offload once, preserving Parameter identity; return copied tensor bytes."""
        if self.offloaded:
            return 0
        self._validate_idle()
        self._model_devices = []
        self._optimizer_tensors = []
        seen = set()
        copied_bytes = 0
        for model, optimizer in self.roles:
            tensors = _model_tensors(model)
            self._model_devices.append({name: tensor.device for name, tensor in tensors.items()})
            parameters = tuple(model.parameters())
            seen.update(id(p) for p in parameters)
            copied_bytes += _transfer_bytes(tensors, {name: torch.device("cpu") for name in tensors})
            # HSDPModule._apply reshards and resets its padded/local storage bookkeeping.
            # Do not replace parameters or rebuild the optimizer around CPU tensors.
            del tensors
            model.to(device="cpu")
            if tuple(map(id, model.parameters())) != tuple(map(id, parameters)):
                raise RuntimeError("Model conversion replaced optimizer-bound Parameter objects")
            if optimizer is None:
                continue
            for component in getattr(optimizer, "chained_optimizers", (optimizer,)):
                values = [p for group in component.param_groups for p in group["params"]]
                values.extend(_state_tensors(component.state))
                for tensor in values:
                    if id(tensor) in seen:
                        continue
                    seen.add(id(tensor))
                    if tensor.device.type == "cpu":
                        continue
                    self._optimizer_tensors.append((tensor, tensor.device))
                    copied_bytes += _tensor_bytes(tensor)
                    tensor.data = tensor.detach().to(device="cpu")
        self.offloaded = True
        return copied_bytes

    @torch.no_grad()
    def restore(self) -> int:
        """Restore each tensor's prior device, including CPU optimizer counters."""
        if not self.offloaded:
            return 0
        copied_bytes = 0
        for (model, _), devices in zip(self.roles, self._model_devices):
            tensors = _model_tensors(model)
            if tensors.keys() != devices.keys():
                raise RuntimeError("Model registrations changed while training state was offloaded")
            targets = {id(tensor): devices[name] for name, tensor in tensors.items()}
            copied_bytes += _transfer_bytes(tensors, devices)
            parameters = tuple(model.parameters())
            del tensors
            # Per-tensor targets preserve CPU-offloaded parameters alongside device buffers.
            # Dispatch through HSDP's existing conversion hook instead of editing its state.
            model._apply(  # pylint: disable=protected-access
                lambda tensor, targets=targets: tensor.to(device=targets[id(tensor)]),
            )
            if tuple(map(id, model.parameters())) != tuple(map(id, parameters)):
                raise RuntimeError("Model restoration replaced optimizer-bound Parameter objects")
        for tensor, device in self._optimizer_tensors:
            copied_bytes += _tensor_bytes(tensor)
            tensor.data = tensor.detach().to(device=device)
        self._model_devices.clear()
        self._optimizer_tensors.clear()
        self.offloaded = False
        return copied_bytes
