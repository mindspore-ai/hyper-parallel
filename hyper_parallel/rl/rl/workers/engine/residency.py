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
"""Training state residency at generation, update and publication boundaries."""
import logging
import time
from typing import Any, Callable, Mapping, Optional
import torch
import torch.distributed as dist
from hyper_parallel import hsdp_sync_stream
from rl.workers.engine.model import iter_hsdp_roots

logger = logging.getLogger(__name__)

def _iter_state_tensors(value: Any):
    """Yield tensors recursively from one optimizer state value."""
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_state_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_state_tensors(item)


class TrainingResidency:
    """Coordinate model shards and optimizer storage without owning training policy."""

    def __init__(self, *, actor: Any, reference_actor: Any, critic: Any,
                 device: torch.device, run_synchronized: Callable,
                 colocated: bool, cpu_offload: bool, offload_manager: Any = None) -> None:
        """Bind only storage owners and the configured residency policy."""
        self.actor = actor
        self.reference_actor = reference_actor
        self.critic = critic
        self.device = device
        self.run_synchronized = run_synchronized
        self.colocated = colocated
        self.cpu_offload = cpu_offload
        self.offload_manager = offload_manager
        self.metrics: dict[str, float] = {}

    @property
    def optimizer(self) -> Any:
        """Read the Actor's authoritative optimizer binding."""
        return self.actor.optimizer

    @staticmethod
    def reshard_model(model: Optional[Any]) -> None:
        """Initialize loaded shards and release nested full FSDP parameter allocations."""
        if model is None:
            return
        for hsdp_root in iter_hsdp_roots(model):
            # Meta builds defer CPU offload initialization until the first forward,
            # while colocated rollout can allocate its weights before that forward.
            hsdp_root.hsdp_scheduler.hsdp_state.lazy_init()
            hsdp_root.reshard()


    @staticmethod
    def validate_optimizer_cpu_residency(optimizer: Optional[Any], role: str) -> None:
        """Fail if a colocated optimizer retains tensor state on the NPU."""
        if optimizer is None:
            return
        optimizers = getattr(optimizer, "chained_optimizers", (optimizer,))
        device_states = []
        for component in optimizers:
            for state in component.state.values():
                for tensor in _iter_state_tensors(state):
                    device = str(tensor.device)
                    if not device.startswith("cpu"):
                        device_states.append(device)
        if device_states:
            raise RuntimeError(
                f"Colocated {role} optimizer state must be CPU resident, got devices={sorted(set(device_states))}"
            )


    @staticmethod
    def validate_optimizer_device_residency(optimizer: Any, device: torch.device) -> None:
        """Keep optimizer parameters and moments on device; allow CPU scalar step counters."""
        for component in getattr(optimizer, "chained_optimizers", (optimizer,)):
            for group in component.param_groups:
                for parameter in group["params"]:
                    if parameter.device != device:
                        raise RuntimeError(
                            f"Resident Actor optimizer parameter is on {parameter.device}, expected {device}"
                        )
            for state in component.state.values():
                for name, value in state.items():
                    for tensor in _iter_state_tensors(value):
                        if name == "step" and tensor.device.type == "cpu" and tensor.ndim == 0:
                            continue
                        if tensor.device != device:
                            raise RuntimeError(
                                f"Resident Actor optimizer state is on {tensor.device}, expected {device}"
                            )


    def move(self, *, offload: bool) -> None:
        """Move idle training state at a phase boundary after inference residency is safe."""
        manager = self.offload_manager
        if manager is None:
            return
        direction = "offload" if offload else "restore"

        def move() -> dict[str, float]:
            """Measure completed copies and allocator release on the owning device."""
            hsdp_sync_stream()
            device = torch.get_device_module()
            device.synchronize()
            started = time.perf_counter()
            before = device.memory_allocated()
            copied_bytes = manager.offload() if offload else manager.restore()
            device.synchronize()
            if offload:
                device.empty_cache()
            device.synchronize()
            metrics = {
                f"residency/{direction}_seconds": time.perf_counter() - started,
                f"residency/{direction}_bytes": float(copied_bytes),
                f"residency/{direction}_allocated_before": float(before),
                f"residency/{direction}_allocated_after": float(device.memory_allocated()),
            }
            logger.info("rank=%d training_state_%s %s", dist.get_rank(), direction, metrics)
            return metrics

        self.metrics.update(self.run_synchronized(f"training-state {direction}", move))


    def release(self) -> None:
        """Reshard FSDP state and release allocator cache before waking vLLM."""
        if not self.colocated:
            return

        def release_training_state() -> None:
            """Release rank-local FSDP and optimizer residency."""
            hsdp_sync_stream()
            self.reshard_model(self.actor.actor_model)
            self.reshard_model(
                None
                if self.reference_actor is None
                else self.reference_actor.actor_model
            )
            self.reshard_model(
                None if self.critic is None else self.critic.critic_model
            )
            cpu_offload = self.cpu_offload
            if cpu_offload:
                self.validate_optimizer_cpu_residency(self.optimizer, "actor")
                self.validate_optimizer_cpu_residency(
                    None if self.critic is None else self.critic.optimizer, "critic",
                )
            else:
                self.validate_optimizer_device_residency(self.optimizer, self.device)
            torch.get_device_module().current_stream().synchronize()

        self.run_synchronized("training-state release", release_training_state)

        def release_allocator_cache() -> None:
            """Release rank-local allocator cache after all models are sharded."""
            device_handle = torch.get_device_module()
            empty_cache = getattr(device_handle, "empty_cache", None)
            if callable(empty_cache):
                empty_cache()
            torch.get_device_module().current_stream().synchronize()

        self.run_synchronized("allocator-cache release", release_allocator_cache)
