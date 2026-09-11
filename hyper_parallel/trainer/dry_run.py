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
"""Standalone configurable FakeTensor dry-run for the HyperModels LLM Trainer."""
# This runner is intentionally Torch-only; MindSpore has no FakeTensor mode.
# pylint: disable=forbidden-backend-import,no-member,unsupported-binary-operation

import logging
import os
import re
from copy import copy
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Iterator, Optional

import torch
import torch.distributed as dist
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.distributed._tools.mem_tracker import MemTracker

from hyper_parallel.models._transformers.model_builder import (
    model_build_context,
)
from hyper_parallel.data.batching.build_dataloader import calculate_num_micro_batches
from hyper_parallel.distributed.mesh import DistributedSetup, MeshContext
from hyper_parallel.models.build_options import get_device_type
from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.config import (
    DryRunConfig,
    TrainerConfig,
    normalize_distributed_setup_overrides,
)
from hyper_parallel.trainer.dry_run_data import DryRunDataProbe, PreparedDryRunBatch
from hyper_parallel.trainer.runtime.distributed import destroy_process_group as destroy_distributed_runtime
from hyper_parallel.trainer.runtime.random import set_seed
from hyper_parallel import SkipDTensorDispatch, hsdp_sync_stream
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.fully_shard import api as fully_shard_api
from hyper_parallel.core.tensor_parallel import loss_parallel
from hyper_parallel.core.utils import clip_grad_norm_
from hyper_parallel.platform.torch import dry_run as torch_dry_run

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _DryRunTrainingBatch:
    """Value-free normal-training inputs and their CPU-derived statistics."""

    model_inputs: dict[str, Any]
    loss_inputs: dict[str, Any]
    token_counts: dict[str, int]
    valid_token_count: int
    target_tokens_per_rank: tuple[int, ...]



class HyperModelsDryRunRunner:
    """Execute one shape-accurate fake LLM training step and report memory."""

    def __init__(
            self,
            config: TrainerConfig,
            runtime: Optional[torch_dry_run.DryRunRuntime] = None,
    ) -> None:
        """Store configuration without changing distributed or device state."""
        self.config = config
        self._runtime = runtime
        self._stage = "validation"
        self._target_device = "cpu"
        self._simulation_device = "cpu"
        self._resolved_dtype = torch.get_default_dtype()

    @staticmethod
    def _check_torch_version() -> None:
        """Require the FakeTensor collective tracking available in PyTorch 2.7."""
        match = re.match(r"^(\d+)\.(\d+)", torch.__version__)
        version = tuple(int(part) for part in match.groups()) if match else (0, 0)
        if version < (2, 7):
            raise RuntimeError(
                "HyperModelsDryRunRunner requires PyTorch >= 2.7, found "
                f"{torch.__version__}"
            )

    def _get_runtime(self) -> torch_dry_run.DryRunRuntime:
        """Return the validated torchrun identity."""
        if self._runtime is None:
            self._runtime = torch_dry_run.DryRunRuntime.from_torchrun_env()
        return self._runtime

    def _validate_config(self) -> DryRunConfig:
        """Validate static Dry-run and unsupported-feature boundaries."""
        self._check_torch_version()
        dry_run = self.config.dry_run
        if dry_run is None:
            raise ValueError("TrainerConfig.dry_run must be configured")
        if not dry_run.enabled:
            raise ValueError("TrainerConfig.dry_run.enabled must be true")
        if not isinstance(dry_run.output_dir, str) or not dry_run.output_dir.strip():
            raise ValueError("dry_run.output_dir must be a non-empty string")
        if self.config.training.micro_batch_size < 1:
            raise ValueError("training.micro_batch_size must be >= 1")
        activation_checkpoint = self.config.activation_checkpoint.mode
        if activation_checkpoint not in ("off", "none"):
            raise NotImplementedError("LLM Dry-run does not support activation checkpointing")
        if getattr(
                self.config.fsdp_config,
                "activation_checkpointing",
                None,
        ) not in (False, None, "off", "none"):
            raise NotImplementedError("LLM Dry-run does not support FSDP activation checkpointing")
        if self.config.fsdp_config.enable_offload:
            raise NotImplementedError("LLM Dry-run does not support CPU offload")
        if self.config.peft is not None:
            raise NotImplementedError("LLM Dry-run does not support PEFT model mutation")
        if self.config.compile.enabled:
            raise NotImplementedError("LLM Dry-run does not support torch.compile")
        torch_dry_run._DryRunValueProfile(dry_run)  # pylint: disable=protected-access
        if self.config.accelerator.pp_size > 1:
            raise NotImplementedError(
                "Pipeline parallel dry-run is not supported by the base implementation; "
                "use the PP adapter implementation until Trainer PP integration is available"
            )
        self._validate_topology()
        return dry_run

    def _validate_topology(self) -> None:
        """Validate the configured topology against the launcher world size."""
        runtime = self._get_runtime()
        accelerator = self.config.accelerator
        topology_sizes = {
            "tp_size": accelerator.tp_size,
            "cp_size": accelerator.cp_size,
            "ep_size": accelerator.ep_size,
            "dp_shard_size": self.config.fsdp_config.dp_shard_size,
            "edp_shard_size": self.config.fsdp_config.edp_shard_size,
        }
        invalid_sizes = {
            name: size
            for name, size in topology_sizes.items()
            if not isinstance(size, int) or isinstance(size, bool) or size < 1
        }
        if invalid_sizes:
            raise ValueError(
                "Dry-run parallel sizes must be positive integers, got "
                f"{invalid_sizes}"
            )
        non_dp_size = accelerator.tp_size * accelerator.cp_size
        if runtime.world_size % non_dp_size:
            raise ValueError(
                f"WORLD_SIZE {runtime.world_size} is not divisible by TP*CP "
                f"size {non_dp_size}"
            )
        dp_size = runtime.world_size // non_dp_size
        fsdp_domain_size = dp_size * max(1, accelerator.cp_size)
        if fsdp_domain_size % self.config.fsdp_config.dp_shard_size:
            raise ValueError(
                f"DP+CP size {fsdp_domain_size} is not divisible by "
                f"dp_shard_size {self.config.fsdp_config.dp_shard_size}"
            )
        expert_domain_size = dp_size * accelerator.cp_size * accelerator.tp_size
        if expert_domain_size % accelerator.ep_size:
            raise ValueError(
                f"expert domain size {expert_domain_size} is not divisible by "
                f"ep_size {accelerator.ep_size}"
            )
        expert_dp_size = expert_domain_size // accelerator.ep_size
        if expert_dp_size % self.config.fsdp_config.edp_shard_size:
            raise ValueError(
                f"expert DP size {expert_dp_size} is not divisible by "
                f"edp_shard_size {self.config.fsdp_config.edp_shard_size}"
            )


    @staticmethod
    def _prewarm_standard_meshes(mesh_context: MeshContext) -> None:
        """Materialize child meshes used by the non-pipeline training path."""
        meshes = [mesh_context.device_mesh, mesh_context.fsdp_non_moe_mesh]
        for axis in ("dp", "cp", "tp"):
            meshes.append(mesh_context.device_mesh[axis])
        active_names = tuple(
            axis
            for axis, size in (("cp", mesh_context.cp_size), ("tp", mesh_context.tp_size))
            if size > 1
        )
        if active_names:
            active_mesh = mesh_context.device_mesh[active_names]
            meshes.append(active_mesh)
            meshes.extend(active_mesh[axis] for axis in active_names)
        if mesh_context.dp_cp_mesh is not None:
            meshes.append(mesh_context.dp_cp_mesh)
        dense_selector: str | tuple[str, str] = "fsdp_shard"
        if mesh_context.dp_replicate_size > 1:
            dense_selector = ("fsdp_replicate", "fsdp_shard")
        meshes.append(mesh_context.fsdp_non_moe_mesh[dense_selector])
        if mesh_context.fsdp_moe_mesh is not None:
            meshes.extend((mesh_context.fsdp_moe_mesh, mesh_context.fsdp_moe_mesh["ep"]))
            expert_selector: str | tuple[str, str] = "edp_shard"
            if "edp_replicate" in mesh_context.fsdp_moe_mesh.mesh_dim_names:
                expert_selector = ("edp_replicate", "edp_shard")
            meshes.append(mesh_context.fsdp_moe_mesh[expert_selector])
        for mesh in meshes:
            for mesh_dim in range(mesh.ndim):
                mesh.get_group(mesh_dim)

    def _build_distributed_setup(
            self,
            runtime: torch_dry_run.DryRunRuntime,
            device_type: str,
    ) -> DistributedSetup:
        """Build the standard non-pipeline mesh and sharding setup."""
        accelerator = self.config.accelerator
        tp_size = accelerator.tp_size
        cp_size = accelerator.cp_size
        dp_size = runtime.world_size // (tp_size * cp_size)
        dp_shard_size = self.config.fsdp_config.dp_shard_size
        dp_replicate_size = dp_size * cp_size // dp_shard_size
        mesh_context = MeshContext(
            dp_size=dp_size,
            dp_replicate_size=dp_replicate_size,
            dp_shard_size=dp_shard_size,
            edp_shard_size=self.config.fsdp_config.edp_shard_size,
            tp_size=tp_size,
            cp_size=cp_size,
            pp_size=1,
            ep_size=accelerator.ep_size,
            sequence_parallel=bool(accelerator.sequence_parallel),
            loss_parallel=bool(accelerator.loss_parallel),
        )
        mesh_context.build_meshs(device_type, runtime.world_size)
        mesh_context.dp_rank = mesh_context.device_mesh.get_local_rank("dp")
        mesh_context.tp_rank = mesh_context.device_mesh.get_local_rank("tp")
        mesh_context.cp_rank = mesh_context.device_mesh.get_local_rank("cp")
        mesh_context.ep_rank = (
            mesh_context.fsdp_moe_mesh.get_local_rank("ep")
            if mesh_context.fsdp_moe_mesh is not None
            else 0
        )
        mesh_context.pp_rank = 0
        self._prewarm_standard_meshes(mesh_context)
        fsdp_enabled = (
            dp_shard_size > 1
            or dp_replicate_size > 1
            or self.config.fsdp_config.edp_shard_size > 1
        )
        return DistributedSetup(
            mesh_context=mesh_context,
            strategy_config=self.config.fsdp_config if fsdp_enabled else None,
            plan_overrides=self.config.plan_overrides,
            low_precision_config=getattr(self.config.training, "low_precision", None),
            fp32_main_params=self.config.optimizer.fp32_main_params,
        )

    def _select_simulation_device(self, target_device: str) -> str:
        """Select a registered FakeTensor device without probing real hardware."""
        backend_registered = (
            torch.version.cuda is not None
            if target_device == "cuda"
            else hasattr(torch, "npu")
        )
        if not backend_registered:
            logger.info(
                "PyTorch has no %s device guard; using CPU FakeTensor simulation",
                target_device,
            )
            return "cpu"
        try:
            target = torch.device(target_device)
            with FakeTensorMode():
                torch.empty((), device=target)
            return target_device
        except (AttributeError, RuntimeError):
            logger.info(
                "FakeTensor device %s is not registered; using CPU simulation",
                target_device,
            )
        return "cpu"

    def _resolve_target_device(self) -> str:
        """Read the normal-training accelerator type without binding a device."""
        target_device = get_device_type()
        if target_device not in ("cuda", "npu"):
            raise RuntimeError(
                "HyperModels Dry-run requires an available CUDA or NPU runtime; "
                f"detected {target_device!r}"
            )
        self._target_device = target_device
        return target_device

    @staticmethod
    def _init_cpu_data_process_group(runtime: torch_dry_run.DryRunRuntime) -> None:
        """Initialize a temporary Gloo group for the normal data pipeline."""
        if dist.is_initialized():
            raise RuntimeError("Dry-run data probing must start before process-group initialization")
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
        kwargs = {
            "backend": "gloo",
            "world_size": runtime.world_size,
            "rank": runtime.rank,
        }
        if runtime.world_size == 1:
            kwargs["store"] = dist.HashStore()
        dist.init_process_group(**kwargs)

    @staticmethod
    def _init_fake_process_group(runtime: torch_dry_run.DryRunRuntime) -> None:
        """Initialize PyTorch's fake backend using the torchrun identity."""
        if dist.is_initialized():
            raise RuntimeError("Dry-run must start before any process group is initialized")
        # Import registers the version-matched fake backend factory.
        from torch.testing._internal.distributed import fake_pg  # pylint: disable=C0415,unused-import

        dist.init_process_group(
            backend=dist.Backend.FAKE,
            store=dist.HashStore(),
            world_size=runtime.world_size,
            rank=runtime.rank,
        )

    def _prepare_data_base(
            self,
            setup: DistributedSetup,
            model: nn.Module,
            device: Optional[torch.device] = None,
    ) -> BaseTrainer:
        """Create the BaseTrainer state consumed by shared text-data setup."""
        runtime = self._get_runtime()
        base = BaseTrainer.__new__(BaseTrainer)
        base.config = self.config
        base.local_rank = runtime.local_rank
        base.global_rank = runtime.rank
        base.world_size = runtime.world_size
        base.device = torch.device("cpu") if device is None else device
        base.distributed_setup = setup
        base.mesh = setup.mesh_context
        base.device_mesh = base.mesh.device_mesh
        base.dp_cp_mesh = base.mesh.dp_cp_mesh
        base.model_config = model.config
        if base.config.training.seed is None:
            base.config.training.seed = base.default_seed
        set_seed(
            base.config.training.seed,
            base.config.training.enable_full_determinism,
        )
        return base

    def _read_training_batch(
            self,
            setup: DistributedSetup,
            model: nn.Module,
            device: Optional[torch.device] = None,
    ) -> PreparedDryRunBatch:
        """Read one CPU batch through the normal TextTrainer data lifecycle."""
        data_runtime = DryRunDataProbe(self._prepare_data_base(setup, model, device))
        try:
            data_runtime.build()
            return data_runtime.read_first_batch()
        finally:
            data_runtime.close()

    def _mock_training_batch(
            self,
            batch: PreparedDryRunBatch,
            fake_mode: FakeTensorMode,
            model: nn.Module,
            tp_size: int,
    ) -> _DryRunTrainingBatch:
        """Erase CPU values and retain only the normal batch structure."""
        valid_count, owned = torch_dry_run.derive_tp_target_counts(
            batch.loss_inputs,
            int(model.config.vocab_size),
            tp_size,
        )
        mocker = torch_dry_run.DryRunBatchMocker(
            fake_mode,
            self._simulation_torch_device(),
        )
        return _DryRunTrainingBatch(
            model_inputs=mocker.mock(batch.model_inputs),
            loss_inputs=mocker.mock(batch.loss_inputs),
            token_counts=dict(batch.token_counts),
            valid_token_count=valid_count,
            target_tokens_per_rank=owned,
        )

    @contextmanager
    def _dry_run_fsdp_device(self) -> Iterator[None]:
        """Temporarily resolve FSDP storage to the simulation device."""
        original = fully_shard_api._get_device_from_mesh  # pylint: disable=protected-access
        original_concatenate = DeviceMesh.concatenate
        simulation_device = self._simulation_torch_device()

        def _resolve_device(_mesh: Any) -> torch.device:
            return simulation_device

        def _concatenate_meshes(meshes: Any) -> DeviceMesh:
            # Mesh rank maps are host metadata and must never become FakeTensors.
            with unset_fake_temporarily():
                return original_concatenate(meshes)

        fully_shard_api._get_device_from_mesh = _resolve_device  # pylint: disable=protected-access
        DeviceMesh.concatenate = staticmethod(_concatenate_meshes)
        try:
            yield
        finally:
            fully_shard_api._get_device_from_mesh = original  # pylint: disable=protected-access
            DeviceMesh.concatenate = staticmethod(original_concatenate)

    def _simulation_torch_device(self) -> torch.device:
        """Return the rank-local logical device used by FakeTensor execution."""
        if self._simulation_device == "cpu":
            return torch.device("cpu")
        return torch.device(
            self._simulation_device,
            self._get_runtime().local_rank,
        )

    def _build_target_model(self, setup: DistributedSetup) -> nn.Module:
        """Build the configured model through the normal deferred Target path."""
        with model_build_context():
            model = self.config.model.build(
                distributed_setup=setup,
                peft_config=self.config.peft,
                activation_checkpoint=self.config.activation_checkpoint.mode,
                swap_inputs=getattr(self.config.activation_checkpoint, "swap_inputs", False),
                activation_swap=self.config.activation_swap,
                compile_config=self.config.compile,
                model_init_dtype=self.config.model_init_dtype,
            )
        return model

    def _record_model_dtype(self, model: nn.Module) -> None:
        """Record the effective floating dtype from the normally built model."""
        self._resolved_dtype = next(
            (tensor.dtype for tensor in model.parameters() if tensor.is_floating_point()),
            torch.get_default_dtype(),
        )

    def _prepare_base(
            self,
            setup: DistributedSetup,
            model: nn.Module,
    ) -> BaseTrainer:
        """Create only the BaseTrainer state needed by one fake training step."""
        runtime = self._get_runtime()
        base = BaseTrainer.__new__(BaseTrainer)
        base.config = self.config
        base.local_rank = runtime.local_rank
        base.global_rank = runtime.rank
        base.world_size = runtime.world_size
        base.device = self._simulation_torch_device()
        base.distributed_setup = setup
        base.mesh = setup.mesh_context
        base.device_mesh = base.mesh.device_mesh
        base.dp_cp_mesh = base.mesh.dp_cp_mesh
        base.model = model
        base.model_config = model.config
        base.model_parts = [model]
        base.hsdp_model_parts = [
            module for module in model.modules()
            if hasattr(module, "hsdp_scheduler")
        ]
        BaseTrainer._build_loss(base)
        BaseTrainer._build_optimizer(base)
        BaseTrainer._build_training_context(base)
        return base

    def _materialize_fake_model(self, model: nn.Module) -> nn.Module:
        """Materialize meta state as FakeTensors without assigning values."""
        model.to_empty(device=self._simulation_torch_device())
        return model

    @staticmethod
    def _initialize_flat_buffers(model: nn.Module) -> None:
        """Materialize enabled zero-copy FSDP flat shards before tracking."""
        visited_states = set()
        for module in model.modules():
            scheduler = getattr(module, "hsdp_scheduler", None)
            state = getattr(scheduler, "hsdp_state", None)
            if state is None or id(state) in visited_states:
                continue
            visited_states.add(id(state))
            # Match the real forward-pre-hook order: dtype attributes must be
            # initialized before communication buckets choose their dtype.
            state.lazy_init()
            param_group = getattr(state, "param_group", None)
            if param_group is not None and param_group.enable_zero_copy:
                param_group._init_all_gather_buckets()  # pylint: disable=protected-access
                for bucket in param_group.all_gather_buckets:
                    bucket.init_flat_param_buffer(param_group.device)

    @staticmethod
    def _configure_gradient_sync(base: BaseTrainer) -> None:
        """Mark the single simulated micro-step as the final backward."""
        for model_part in base.hsdp_model_parts:
            model_part.set_requires_gradient_sync(True)
            model_part.set_is_last_backward(True)
            if base.mesh.dp_replicate_size > 1:
                model_part.set_requires_all_reduce(True)

    @staticmethod
    def _loss_context(base: BaseTrainer) -> Any:
        """Return the real loss-parallel context when configured."""
        if not base.mesh.loss_parallel:
            return nullcontext()
        return loss_parallel(mesh=base.mesh.device_mesh["tp"])

    def _execute_micro_step(
            self,
            base: BaseTrainer,
            batch: _DryRunTrainingBatch,
            value_dependencies: Any,
            tracker: Any,
            operator_trace: Any,
    ) -> None:
        """Run one fake forward/backward in a micro-step-local scope."""
        model_batch = dict(batch.model_inputs)

        with base.model_fwd_context:
            outputs = base.model(**model_batch, use_cache=False)
        with value_dependencies.logical_scope("loss"):
            loss_value = base.loss_fn(
                model_output=outputs,
                labels=batch.loss_inputs.get("labels"),
            )
        del outputs
        if isinstance(loss_value, dict):
            loss_value = torch.stack(list(loss_value.values())).sum()
        with base.model_bwd_context, value_dependencies.logical_scope("backward"):
            loss_value.backward()

        gradient_tensors = tracker.refresh_parameter_gradients(base.model)
        operator_trace.refresh_tensor_roles(gradient_tensors)

    def _execute_step(
            self,
            base: BaseTrainer,
            batch: _DryRunTrainingBatch,
            profile: Any,
            value_dependencies: Any,
    ) -> dict[str, Any]:
        """Track accumulated micro-batches and one optimizer update."""
        self._initialize_flat_buffers(base.model)
        tracker = torch_dry_run._create_indexed_mem_tracker(MemTracker)  # pylint: disable=protected-access
        tracked_optimizers = getattr(
            base.optimizer,
            "chained_optimizers",
            [base.optimizer],
        )
        tracker.track_external(
            base.model,
            *tracked_optimizers,
            *self._training_batch_tensors(batch),
        )
        operator_trace = torch_dry_run._create_operator_trace_mode(  # pylint: disable=protected-access
            tracker,
            self._simulation_device,
        )
        self._configure_gradient_sync(base)
        num_micro_batches = calculate_num_micro_batches(
            self.config.training.global_batch_size,
            self.config.training.micro_batch_size,
            dp_world_size=base.mesh.dp_size,
        )
        with (
                tracker,
                operator_trace,
                value_dependencies.fake_step_context(base),
                self._loss_context(base),
        ):
            for _micro_index in range(num_micro_batches):
                self._execute_micro_step(
                    base,
                    batch,
                    value_dependencies,
                    tracker,
                    operator_trace,
                )
            hsdp_sync_stream()
            gradient_tensors = tracker.refresh_parameter_gradients(base.model)
            operator_trace.refresh_tensor_roles(gradient_tensors)
            del gradient_tensors
            clip_grad_norm_(base.model, self.config.training.max_grad_norm)
            optimizers = base.optimizer if isinstance(base.optimizer, list) else [base.optimizer]
            with value_dependencies.logical_scope("optimizer"):
                for optimizer in optimizers:
                    with SkipDTensorDispatch():
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            tracker.clear_fake_dtensor_grad_bridges()
        operator_trace.finalize()
        return torch_dry_run.build_memory_report(
            tracker,
            self._metadata(base, batch, profile, num_micro_batches=num_micro_batches),
            self._simulation_device,
            operator_trace.memory_blocks,
            report_device_type=self._target_device,
        )

    def _metadata(
            self,
            base: BaseTrainer,
            batch: _DryRunTrainingBatch,
            profile: Any,
            num_micro_batches: Optional[int] = None,
    ) -> dict[str, Any]:
        """Build reproducibility metadata for the memory report."""
        runtime = self._get_runtime()
        metadata = {
            "model": type(base.model).__name__,
            "torch_version": torch.__version__,
            "rank": runtime.rank,
            "world_size": runtime.world_size,
            "local_rank": runtime.local_rank,
            "target_device": self._target_device,
            "simulation_device": self._simulation_device,
            "initialization_device": "meta",
            "weights_loaded": False,
            "model_dtype": str(self._resolved_dtype),
            "batch_shapes": {
                "model_inputs": self._batch_shape_metadata(batch.model_inputs),
                "loss_inputs": self._batch_shape_metadata(batch.loss_inputs),
            },
            "token_counts": dict(batch.token_counts),
            "parallel": {
                "dp": base.mesh.dp_size,
                "dp_replicate": base.mesh.dp_replicate_size,
                "dp_shard": base.mesh.dp_shard_size,
                "tp": base.mesh.tp_size,
                "cp": base.mesh.cp_size,
                "ep": base.mesh.ep_size,
                "pp": base.mesh.pp_size,
            },
            "reshard_after_forward": self.config.fsdp_config.reshard_after_forward,
            "comm_fusion": self.config.fsdp_config.comm_fusion,
            "value_dependencies": profile.metadata(),
        }
        if num_micro_batches is not None:
            metadata["num_micro_batches"] = num_micro_batches
        return metadata

    @classmethod
    def _training_batch_tensors(cls, batch: _DryRunTrainingBatch) -> tuple[torch.Tensor, ...]:
        """Return deduplicated tensor leaves from model and loss inputs."""
        tensors = []
        seen = set()
        for value in (batch.model_inputs, batch.loss_inputs):
            for tensor in cls._tensor_leaves(value):
                if id(tensor) not in seen:
                    seen.add(id(tensor))
                    tensors.append(tensor)
        return tuple(tensors)

    @classmethod
    def _tensor_leaves(cls, value: Any) -> tuple[torch.Tensor, ...]:
        """Recursively collect tensor leaves from a mocked batch value."""
        if isinstance(value, torch.Tensor):
            return (value,)
        if isinstance(value, Mapping):
            return tuple(
                tensor
                for item in value.values()
                for tensor in cls._tensor_leaves(item)
            )
        if isinstance(value, (list, tuple)):
            return tuple(
                tensor
                for item in value
                for tensor in cls._tensor_leaves(item)
            )
        if is_dataclass(value) and not isinstance(value, type):
            return tuple(
                tensor
                for field in fields(value)
                for tensor in cls._tensor_leaves(getattr(value, field.name))
            )
        return ()

    @classmethod
    def _batch_shape_metadata(cls, value: Any) -> Any:
        """Serialize nested batch tensor shapes without retaining values."""
        if isinstance(value, torch.Tensor):
            return {
                "shape": [int(size) for size in value.shape],
                "dtype": str(value.dtype),
                "layout": str(value.layout),
            }
        if isinstance(value, Mapping):
            return {name: cls._batch_shape_metadata(item) for name, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._batch_shape_metadata(item) for item in value]
        if is_dataclass(value) and not isinstance(value, type):
            return {
                field.name: cls._batch_shape_metadata(getattr(value, field.name))
                for field in fields(value)
            }
        return value

    def run(self) -> dict[str, Any]:
        """Execute one fake step, write the rank-local CSV, and return its report."""
        runtime = self._get_runtime()
        dry_run = self._validate_config()
        csv_path = os.path.join(
            dry_run.output_dir,
            f"rank_{runtime.rank}",
            f"rank_{runtime.rank}_memory.csv",
        )
        json_path = os.path.splitext(csv_path)[0] + ".json"
        target_device = self._resolve_target_device()
        self._simulation_device = self._select_simulation_device(target_device)
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)

        self._stage = "data_probe"
        data_group_initialized = False
        cpu_setup = None
        try:
            self._init_cpu_data_process_group(runtime)
            data_group_initialized = True
            cpu_setup = self._build_distributed_setup(runtime, "cpu")
            normalize_distributed_setup_overrides(cpu_setup, self.config)
            probe_setup = copy(cpu_setup)
            probe_setup.strategy_config = None
            probe_model = self._build_target_model(probe_setup)
            cpu_batch = self._read_training_batch(cpu_setup, probe_model)
            batch = self._mock_training_batch(
                cpu_batch,
                fake_mode,
                probe_model,
                cpu_setup.mesh_context.tp_size,
            )
            del cpu_batch
            del probe_model
        finally:
            cpu_setup = None
            if data_group_initialized:
                destroy_distributed_runtime()

        initialized_here = False
        try:
            self._stage = "distributed_setup"
            self._init_fake_process_group(runtime)
            initialized_here = True
            setup = self._build_distributed_setup(
                runtime,
                self._simulation_device,
            )
            normalize_distributed_setup_overrides(setup, self.config)

            self._stage = "model_build"
            profile = torch_dry_run._DryRunValueProfile(dry_run)  # pylint: disable=protected-access
            with fake_mode:
                with self._dry_run_fsdp_device():
                    model = self._build_target_model(setup)
                self._record_model_dtype(model)
                profile.bind_model(model)
                value_dependencies = torch_dry_run.ValueDependencyManager(
                    profile, model, runtime,
                )
                self._stage = "materialize"
                model = self._materialize_fake_model(model)
                base = self._prepare_base(setup, model)
                profile.configure_tp_cross_entropy_counts(
                    batch.valid_token_count,
                    batch.target_tokens_per_rank,
                )
                self._stage = "fake_step"
                report = self._execute_step(base, batch, profile, value_dependencies)

            self._stage = "report_generation"
            torch_dry_run.write_memory_report(report, json_path)
            torch_dry_run.write_memory_csv(report, csv_path)
            logger.info("HyperModels Dry-run memory report: %s", csv_path)
            return report
        except torch_dry_run.UnconfiguredValueDependencyError:
            raise
        except Exception as error:
            raise RuntimeError(
                f"HyperModels Dry-run failed during {self._stage}: {error}"
            ) from error
        finally:
            if initialized_here:
                destroy_distributed_runtime()


__all__ = ["HyperModelsDryRunRunner"]
