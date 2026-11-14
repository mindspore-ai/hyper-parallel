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
# pylint: disable=no-member,unsupported-binary-operation

import logging
import os
import re
from copy import copy
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import
from torch._subclasses.fake_tensor import FakeTensorMode  # pylint: disable=forbidden-backend-import
from torch._subclasses.fake_tensor import unset_fake_temporarily  # pylint: disable=forbidden-backend-import

from hyper_parallel.data.batching.build_dataloader import calculate_num_micro_batches
from hyper_parallel.distributed.mesh import DistributedSetup, MeshContext
from hyper_parallel.models.build_options import get_device_type
from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.config import TrainerConfig, normalize_distributed_setup_overrides
from hyper_parallel.trainer.runtime import fsdp as fsdp_runtime
from hyper_parallel.trainer.runtime import memory_profiler as memory_profiler_module
from hyper_parallel.trainer.runtime.device import simulate_device_runtime
from hyper_parallel.trainer.runtime.distributed import destroy_process_group as destroy_distributed_runtime
from hyper_parallel.trainer.runtime.random import set_seed
from hyper_parallel.trainer.text_trainer import TextTrainer
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.fully_shard import api as fully_shard_api
from hyper_parallel.tools.dryrun import runtime as torch_dry_run
from hyper_parallel.tools.dryrun.compatibility import (
    install_dryrun_compatibility,
    model_build_context,
)
from hyper_parallel.tools.dryrun.config import DryRunConfig, DryRunSnapshotConfig
from hyper_parallel.tools.dryrun.data import DryRunDataProbe, PreparedDryRunBatch
from hyper_parallel.tools.dryrun.memory import LogicalMemoryTracker, write_memory_snapshot
from hyper_parallel.tools.dryrun.operator_support import (
    register_builtin_dryrun_ops,
    unsupported_operator_hint,
)

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
        install_dryrun_compatibility()

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
            dry_run = self.config.dry_run
            virtual_size = dry_run.virtual_world_size if isinstance(dry_run, DryRunConfig) else None
            torchrun_env = {"RANK", "WORLD_SIZE", "LOCAL_RANK"}
            if torchrun_env <= os.environ.keys():
                self._runtime = torch_dry_run.DryRunRuntime.from_torchrun_env()
                if virtual_size is not None and virtual_size != self._runtime.world_size:
                    raise ValueError(
                        "dry_run.virtual_world_size must match torchrun WORLD_SIZE: "
                        f"{virtual_size} versus {self._runtime.world_size}"
                    )
            else:
                self._runtime = torch_dry_run.DryRunRuntime(
                    rank=0,
                    world_size=virtual_size or 1,
                    local_rank=0,
                )
        return self._runtime

    def _validate_config(self) -> DryRunConfig:
        """Validate static Dry-run and unsupported-feature boundaries."""
        self._check_torch_version()
        raw_dry_run = self.config.dry_run
        if raw_dry_run is None:
            raise ValueError("TrainerConfig.dry_run must be configured")
        if isinstance(raw_dry_run, Mapping):
            try:
                raw_dry_run = DryRunConfig.from_mapping(raw_dry_run)
            except TypeError as error:
                raise ValueError(f"Invalid dry_run configuration: {error}") from error
        if not isinstance(raw_dry_run, DryRunConfig):
            raise TypeError("TrainerConfig.dry_run must resolve to a DryRunConfig")
        dry_run = raw_dry_run
        self.config.dry_run = dry_run
        if not dry_run.enabled:
            raise ValueError("TrainerConfig.dry_run.enabled must be true")
        if not isinstance(dry_run.snapshot, DryRunSnapshotConfig):
            raise TypeError("dry_run.snapshot must resolve to DryRunSnapshotConfig")
        if dry_run.snapshot.enabled and (
                not isinstance(dry_run.snapshot.path, str) or not dry_run.snapshot.path.strip()
        ):
            raise ValueError("dry_run.snapshot.path must be a non-empty string")
        if not isinstance(dry_run.snapshot.capture_stacks, bool):
            raise TypeError("dry_run.snapshot.capture_stacks must be a boolean")
        if (
                isinstance(dry_run.snapshot.stack_depth, bool)
                or not isinstance(dry_run.snapshot.stack_depth, int)
                or not 1 <= dry_run.snapshot.stack_depth <= 256
        ):
            raise ValueError("dry_run.snapshot.stack_depth must be an integer in [1, 256]")
        if dry_run.ranks is not None:
            if (
                    not isinstance(dry_run.ranks, list)
                    or not dry_run.ranks
                    or any(
                        isinstance(rank, bool) or not isinstance(rank, int) or rank < 0
                        for rank in dry_run.ranks
                    )
                    or len(set(dry_run.ranks)) != len(dry_run.ranks)
            ):
                raise ValueError("dry_run.ranks must be a non-empty list of unique non-negative integers")
        if dry_run.virtual_world_size is not None and (
                isinstance(dry_run.virtual_world_size, bool)
                or not isinstance(dry_run.virtual_world_size, int)
                or dry_run.virtual_world_size < 1
        ):
            raise ValueError("dry_run.virtual_world_size must be a positive integer")
        if self.config.training.micro_batch_size < 1:
            raise ValueError("training.micro_batch_size must be >= 1")
        activation_checkpoint = self.config.activation_checkpoint.mode
        if activation_checkpoint not in ("off", "none", "full", "selective"):
            raise NotImplementedError(
                "LLM Dry-run does not support activation checkpoint "
                f"mode {activation_checkpoint!r}"
            )
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
        if self.config.activation_swap != "none":
            raise NotImplementedError(
                "LLM Dry-run does not simulate activation swap because FakeTensor "
                "cannot reproduce asynchronous host offload and stream lifetimes"
            )
        torch_dry_run._DryRunValueProfile(dry_run)  # pylint: disable=protected-access
        self._validate_topology()
        if self.config.accelerator.pp_size != 1:
            raise NotImplementedError("Dry-run does not support pipeline parallelism")
        return dry_run

    def _validate_topology(self) -> None:
        """Validate the configured topology against the launcher world size."""
        runtime = self._get_runtime()
        if self.config.dry_run.ranks is not None and any(
                rank >= runtime.world_size for rank in self.config.dry_run.ranks
        ):
            raise ValueError(
                f"dry_run.ranks must be smaller than virtual world size {runtime.world_size}"
            )
        accelerator = self.config.accelerator
        topology_sizes = {
            "tp_size": accelerator.tp_size,
            "cp_size": accelerator.cp_size,
            "ep_size": accelerator.ep_size,
            "pp_size": accelerator.pp_size,
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
        non_dp_size = accelerator.tp_size * accelerator.cp_size * accelerator.pp_size
        if runtime.world_size % non_dp_size:
            raise ValueError(
                f"WORLD_SIZE {runtime.world_size} is not divisible by TP*CP*PP "
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
        if self.config.dry_run.snapshot.enabled:
            self._snapshot_path(self.config.dry_run, runtime)

    @staticmethod
    def _snapshot_path(
            dry_run: DryRunConfig,
            runtime: torch_dry_run.DryRunRuntime,
    ) -> Optional[str]:
        """Create the rank directory and fixed pickle name under the output path."""
        if not dry_run.snapshot.enabled:
            return None
        return str(
            Path(dry_run.snapshot.path).expanduser()
            / f"rank_{runtime.rank}"
            / "memory.pickle"
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
            # A backend with zero visible devices aborts inside its lazy
            # device binding the first time any op touches a device-typed
            # tensor; simulate on CPU instead of crashing mid-step.
            device_module = getattr(torch, target_device)
            if int(device_module.device_count()) < 1:
                logger.info(
                    "No visible %s device; using CPU FakeTensor simulation",
                    target_device,
                )
                self._neutralize_device_capability_probes(target_device)
                return "cpu"
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

    @staticmethod
    def _neutralize_device_capability_probes(target_device: str) -> None:
        """Answer global capability queries without initializing the backend.

        ``torch.utils.checkpoint`` recompute preserves per-backend autocast
        state, and ``torch_npu.npu.is_bf16_supported`` unconditionally
        lazy-initializes the NPU runtime — which aborts when zero devices are
        visible. The answer is a property of the *target* accelerator, not of
        this host, so pin it to the simulated target's capability.
        """
        if target_device != "npu" or not hasattr(torch, "npu"):
            return
        torch.npu.is_bf16_supported = (  # type: ignore[attr-defined]
            lambda including_emulation=False: True
        )

    def _resolve_target_device(self) -> str:
        """Resolve the normal-training accelerator type without binding a device.

        ``dry_run.target_device`` states the accelerator the simulated plan is
        for; environment detection is only a fallback so capacity planning can
        run on hosts without that accelerator (or with none visible at all).
        """
        target_device = getattr(self.config.dry_run, "target_device", None)
        if target_device not in ("cuda", "npu"):
            target_device = get_device_type()
        if target_device not in ("cuda", "npu"):
            raise RuntimeError(
                "HyperModels Dry-run requires dry_run.target_device or an "
                f"available CUDA or NPU runtime; detected {target_device!r}"
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
        # pylint: disable-next=C0415,unused-import,forbidden-backend-import
        from torch.testing._internal.distributed import fake_pg

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

        def _resolve_device(mesh: Any) -> torch.device:
            del mesh
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

    def _build_target_model(
            self,
            setup: DistributedSetup,
    ) -> nn.Module:
        """Build the configured model through the normal deferred Target path."""
        with model_build_context():
            model = self.config.model.build(
                distributed_setup=setup,
                peft_config=self.config.peft,
                activation_checkpoint=self.config.activation_checkpoint.mode,
                activation_checkpoint_selection=self.config.activation_checkpoint.selection,
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
        """Build the real Trainer state for one fake step via its own stages.

        Every stage below is the unmodified BaseTrainer/TextTrainer code; only
        the environment preparation that ``_setup()`` performs (process group,
        device binding) is owned by dry-run, which has already installed the
        fake process group and enabled the simulated device runtime. Driving
        the real initialization keeps dry-run's execution identical to a
        normal step by construction instead of by imitation.
        """
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
        # The model was already built and parallelized by dry-run's deferred
        # path; adopt it and run the real _build_model tail (parts, HSDP list,
        # integration session, diagnostics) through the same statements.
        base.model = model
        base.model_config = model.config
        model_parts = getattr(model, "parts", None)
        base.model_parts = list(model_parts) if model_parts is not None else [model]
        base.hsdp_model_parts = [
            module for module in model.modules()
            if hasattr(module, "hsdp_scheduler")
        ]
        from hyper_parallel.trainer.runtime import model_integration as model_integration_runtime  # pylint: disable=C0415
        base.model_integration = model_integration_runtime.build_model_integration_session(
            base.model, base.config, base.global_rank, base.mesh,
        )
        base.fsdp_runtime_diagnostics = (
            fsdp_runtime.build_fsdp_runtime_diagnostics(base.hsdp_model_parts)
            if getattr(self.config.debug, "check_fsdp_runtime", False)
            else None
        )
        base._build_loss()  # pylint: disable=protected-access
        base._build_optimizer()  # pylint: disable=protected-access
        base.model_integration.attach_optimizer(base.optimizer)
        base.num_micro_batches = calculate_num_micro_batches(
            global_batch_size=self.config.training.global_batch_size,
            micro_batch_size=self.config.training.micro_batch_size,
            dp_world_size=base.mesh.dp_size,
        )
        base._compute_train_iters()  # pylint: disable=protected-access
        if self.config.lr_scheduler is not None:
            base._build_lr_scheduler()  # pylint: disable=protected-access
        else:
            base.lr_scheduler = None
        base._build_training_context()  # pylint: disable=protected-access
        base.graph_compiler = None
        base._init_callbacks()  # pylint: disable=protected-access
        return base

    def _prepare_text_trainer(self, base: BaseTrainer, batch: _DryRunTrainingBatch) -> TextTrainer:
        """Attach the real TextTrainer facade and the replayed data runtime."""
        trainer = TextTrainer.__new__(TextTrainer)
        trainer.base = base
        base.train_dataloader = None
        base.current_token_counts = {}

        def replay_get_batch(iterator: Any) -> tuple:
            """Serve the mocked micro-batch through the get_batch contract."""
            return next(iterator)

        base.get_batch = replay_get_batch
        base.model_integration.attach_data_pipeline(
            runtime_adapter=None,
            get_batch=base.get_batch,
            model_assets=(base.model_config,),
        )
        del batch
        return trainer

    def _materialize_fake_model(self, model: nn.Module) -> nn.Module:
        """Materialize meta state as FakeTensors without assigning values."""
        fake_parameter = next((
            parameter
            for parameter in model.parameters()
            if hasattr(parameter, "fake_mode")
        ), None)
        if fake_parameter is None:
            # FSDP may move every parameter into its internal state.
            model.to_empty(device=self._simulation_torch_device())
            return model
        converter = fake_parameter.fake_mode.fake_tensor_converter.meta_converter
        tensor_memo = converter.tensor_memo
        tensor_memo.clear()
        converter.tensor_memo = {}
        try:
            model.to_empty(device=self._simulation_torch_device())
        finally:
            converter.tensor_memo = tensor_memo
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

    def _execute_step(
            self,
            base: BaseTrainer,
            batch: _DryRunTrainingBatch,
            profile: Any,
            value_dependencies: Any,
    ) -> dict[str, Any]:
        """Drive one unmodified TextTrainer.train_step over the fake state."""
        self._initialize_flat_buffers(base.model)
        tracker = LogicalMemoryTracker(
            self._simulation_device,
            self._get_runtime().local_rank,
            capture_stacks=self.config.dry_run.snapshot.capture_stacks,
            stack_depth=self.config.dry_run.snapshot.stack_depth,
        )
        trainer = self._prepare_text_trainer(base, batch)
        iterator = iter(torch_dry_run.DryRunBatchIterator(
            batch.model_inputs,
            batch.loss_inputs,
        ))
        scalar_shim = torch_dry_run.FakeHostScalarShim()
        # The probe observed the real token counts; pin them so the
        # gradient-normalization denominator reads its true host value.
        for name, count in batch.token_counts.items():
            tensor = batch.loss_inputs.get(name)
            if tensor is not None and hasattr(tensor, "fake_mode"):
                scalar_shim.register(tensor, count)
        num_micro_batches = base.num_micro_batches
        with (
                tracker,
                value_dependencies.fake_step_context(base),
                scalar_shim,
        ):
            tracker.track(self._existing_tensors(base, batch))
            self._register_token_count_values(base, batch, scalar_shim)
            trainer.train_step(iterator)
        snapshot = tracker.snapshot()
        snapshot["metadata"] = self._metadata(base, batch, profile, num_micro_batches=num_micro_batches)
        return snapshot

    @staticmethod
    def _register_token_count_values(
            base: BaseTrainer,
            batch: _DryRunTrainingBatch,
            scalar_shim: Any,
    ) -> None:
        """Pin probe-derived token counts onto the counts the step will read.

        ``count_loss_token`` recomputes token counts from the mocked labels,
        producing fake scalar tensors. Wrap it so every count tensor it emits
        is registered with the probe-known true value before ``.item()`` runs.
        """
        from hyper_parallel.trainer.runtime import loss_aggregation  # pylint: disable=C0415
        from hyper_parallel.trainer import text_trainer as text_trainer_module  # pylint: disable=C0415
        original = loss_aggregation.count_loss_token
        known = dict(batch.token_counts)

        def counted(batches: Any) -> dict:
            """Count tokens normally, then pin probe-known true values."""
            result = original(batches)
            for name, tensor in result.items():
                if hasattr(tensor, "fake_mode") and name in known:
                    scalar_shim.register(tensor, known[name])
            return result

        # The step loop reads count_loss_token through the modules that
        # imported it; patch both import sites for the step's duration. The
        # fake step is single-shot per process, so no restore is needed before
        # process exit, but keep the originals on the base for debuggability.
        text_trainer_module.count_loss_token = counted
        import hyper_parallel.trainer.base as base_module  # pylint: disable=C0415
        base_module.count_loss_token = counted
        base._dryrun_original_count_loss_token = original  # pylint: disable=protected-access

    @classmethod
    def _existing_tensors(cls, base: BaseTrainer, batch: _DryRunTrainingBatch) -> Iterator[torch.Tensor]:
        """Seed storage allocated before the tracked step, without assigning roles."""
        yield from base.model.parameters()
        yield from base.model.buffers()
        yield from cls._training_batch_tensors(batch)
        optimizers = getattr(base.optimizer, "chained_optimizers", None)
        if optimizers is None:
            optimizers = base.optimizer if isinstance(base.optimizer, list) else [base.optimizer]
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                yield from group["params"]
            yield from cls._tensor_leaves(optimizer.state)

    def _metadata(
            self,
            base: BaseTrainer,
            batch: _DryRunTrainingBatch,
            profile: Any,
            num_micro_batches: Optional[int] = None,
    ) -> dict[str, Any]:
        """Build reproducibility metadata for the memory snapshot."""
        runtime = self._get_runtime()
        metadata = {
            "model": type(base.model).__name__,
            "torch_version": str(torch.__version__),
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
            "value_dependency_replay": profile.metadata(),
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

    def _probe_training_batch(
            self,
            runtime: torch_dry_run.DryRunRuntime,
            fake_mode: FakeTensorMode,
    ) -> _DryRunTrainingBatch:
        """Read one CPU batch and erase its tensor values for simulation."""
        self._stage = "data_probe"
        data_group_initialized = False
        cpu_setup = None
        try:
            if self.config.dry_run.virtual_world_size is None:
                self._init_cpu_data_process_group(runtime)
            else:
                self._init_fake_process_group(runtime)
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
            return batch
        finally:
            cpu_setup = None
            if data_group_initialized:
                destroy_distributed_runtime()

    def _run_simulation(
            self,
            runtime: torch_dry_run.DryRunRuntime,
            dry_run: DryRunConfig,
            batch: _DryRunTrainingBatch,
            fake_mode: FakeTensorMode,
            snapshot_path: Optional[str],
    ) -> dict[str, Any]:
        """Execute one fake distributed step and persist its memory snapshot."""
        initialized_here = False
        try:
            self._stage = "distributed_setup"
            self._init_fake_process_group(runtime)
            initialized_here = True
            setup = self._build_distributed_setup(runtime, self._simulation_device)
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
                profile.configure_tp_cross_entropy_counts(
                    batch.valid_token_count,
                    batch.target_tokens_per_rank,
                )
                base = self._prepare_base(setup, model)
                self._stage = "fake_step"
                snapshot = self._execute_step(base, batch, profile, value_dependencies)

            if snapshot_path is not None:
                self._stage = "snapshot_generation"
                write_memory_snapshot(snapshot, snapshot_path)
                logger.info("HyperModels Dry-run memory snapshot: %s", snapshot_path)
            return snapshot
        except Exception as error:
            plugin_hint = unsupported_operator_hint(error)
            if plugin_hint is not None:
                raise RuntimeError(
                    f"HyperModels Dry-run failed during {self._stage}: {plugin_hint}"
                ) from error
            raise RuntimeError(
                f"HyperModels Dry-run failed during {self._stage}: {error}"
            ) from error
        finally:
            if initialized_here:
                destroy_distributed_runtime()

    def run(self) -> dict[str, Any]:
        """Execute one fake step, write the rank-local pickle, and return its snapshot."""
        dry_run = self._validate_config()
        runtime = self._get_runtime()
        # The real Trainer step loop runs over FakeTensors: silence every
        # device-side effect (set_device/synchronize/empty_cache and allocator
        # history) so simulation never binds or initializes an accelerator.
        simulate_device_runtime(True)
        memory_profiler_module.memory_profiler.reset(
            None, global_rank=runtime.rank, tp_rank=0, dp_rank=0,
        )
        # Register shape-only Meta kernels for Ascend vendor operators whose
        # backward passes ship without them, so the fused kernels stay on the
        # execution path instead of falling back to reference implementations.
        register_builtin_dryrun_ops()
        snapshot_path = self._snapshot_path(dry_run, runtime)
        target_device = self._resolve_target_device()
        self._simulation_device = self._select_simulation_device(target_device)
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
        batch = self._probe_training_batch(runtime, fake_mode)
        return self._run_simulation(runtime, dry_run, batch, fake_mode, snapshot_path)


__all__ = ["HyperModelsDryRunRunner"]
