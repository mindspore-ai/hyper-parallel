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
"""Core HyperParallel utilities for LlamaFactory integration."""
# Torch is the only supported backend; the legacy platform-agnostic pylint rule is not applicable here.
# pylint: disable=forbidden-backend-import,unsupported-binary-operation

import json
import logging
import os
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from hyper_parallel import SkipDTensorDispatch
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer
from hyper_parallel.distributed.mesh import DistributedSetup, MeshContext
from hyper_parallel.models import FSDP2Config, FSDP2MixedPrecisionConfig
from hyper_parallel.models.build_options import get_device_id, get_device_type
from hyper_parallel.trainer.config.parallelism import (
    PlanOverride,
    entries_to_module_replacements,
    entries_to_plan_overrides,
)
from hyper_parallel.trainer.config.resolver import resolve_component

logger = logging.getLogger(__name__)

__all__ = [
    "HSDP_MODEL_NAME",
    "HSDP_OPTIMIZER_NAME",
    "HyperParallelArguments",
    "export_to_hf_format",
    "load_hsdp_model",
    "load_hsdp_optimizer_and_scheduler",
    "parallelize_model",
    "save_hsdp_checkpoint",
    "wrap_optimizer_with_skip_dtensor_dispatch",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_VALID_DTYPES = {"float32", "float16", "bfloat16", "fp32", "fp16", "bf16"}
_CANONICAL_DTYPES = {
    "float32": "float32",
    "fp32": "float32",
    "float16": "float16",
    "fp16": "float16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
}
_VALID_TOKEN_DISPATCHERS = {"all_to_all", "deredundency"}

HSDP_MODEL_NAME = "hsdp_model"
HSDP_OPTIMIZER_NAME = "optimizer"


@dataclass
class HyperParallelArguments:
    """Minimal HyperParallel configuration needed by the trainer backend."""

    tp_size: int = 1
    cp_size: int = 1
    ep_size: int = 1
    efsdp_size: int | None = None
    token_dispatcher: str = "all_to_all"
    device_type: str = "auto"
    param_dtype: str | None = None
    reduce_dtype: str | None = None
    reshard_after_forward: bool | None = None
    fsdp_size: int | None = None
    plan_overrides: list[dict] | None = None

    activation_mode: str = "none"
    activation_swap_inputs: bool = True

    @staticmethod
    def _validate_positive_int(name: str, value: int) -> None:
        """Validate a required parallel size."""
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")

    @staticmethod
    def _validate_optional_positive_int(name: str, value: int | None, type_name: str) -> None:
        """Validate an optional parallel size."""
        if value is None:
            return
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive {type_name} when provided, got {value!r}.")

    def _validate_parallel_sizes(self) -> None:
        """Validate configured parallel dimensions."""
        if self.tp_size != 1:
            raise ValueError(
                "Current trainer backend only supports replacing FSDP/fully_shard. "
                f"Expected tp_size=1, got {self.tp_size}."
            )
        self._validate_positive_int("cp_size", self.cp_size)
        self._validate_positive_int("ep_size", self.ep_size)
        self._validate_optional_positive_int("efsdp_size", self.efsdp_size, "integer")
        self._validate_optional_positive_int("fsdp_size", self.fsdp_size, "int")

    def _validate_world_size(self) -> None:
        """Validate that parallel dimensions divide the runtime world size."""
        if self.cp_size == 1 and self.ep_size == 1 and self.efsdp_size is None:
            return
        world_size = dist.get_world_size()
        if self.cp_size > 1 and world_size % self.cp_size != 0:
            raise ValueError(f"world_size ({world_size}) must be divisible by cp_size ({self.cp_size}).")
        if self.ep_size == 1 and self.efsdp_size is None:
            return
        if world_size % self.ep_size != 0:
            raise ValueError(f"world_size ({world_size}) must be divisible by ep_size ({self.ep_size}).")
        edp_size = world_size // self.ep_size
        if self.efsdp_size is not None and edp_size % self.efsdp_size != 0:
            raise ValueError(
                "world_size / ep_size must be divisible by efsdp_size, got "
                f"({world_size} / {self.ep_size}) % {self.efsdp_size} != 0."
            )

    def _validate_runtime_options(self) -> None:
        """Validate dispatcher, dtype, device, and activation settings."""
        if self.token_dispatcher not in _VALID_TOKEN_DISPATCHERS:
            raise ValueError(
                "token_dispatcher must be one of "
                f"{sorted(_VALID_TOKEN_DISPATCHERS)}, got {self.token_dispatcher!r}."
            )
        if self.param_dtype is not None and self.param_dtype not in _VALID_DTYPES:
            raise ValueError(
                f"param_dtype must be one of {sorted(_VALID_DTYPES)}, got {self.param_dtype!r}."
            )
        if self.reduce_dtype is not None and self.reduce_dtype not in _VALID_DTYPES:
            raise ValueError(
                f"reduce_dtype must be one of {sorted(_VALID_DTYPES)}, got {self.reduce_dtype!r}."
            )
        if self.device_type not in {"auto", "npu", "cuda", "cpu"}:
            raise ValueError(
                f"device_type must be one of ['auto', 'cpu', 'cuda', 'npu'], got {self.device_type!r}."
            )
        if self.reshard_after_forward is not None and not isinstance(
            self.reshard_after_forward, bool
        ):
            raise ValueError(
                "reshard_after_forward must be a bool when provided, "
                f"got {type(self.reshard_after_forward).__name__}."
            )
        valid_activation_modes = {"none", "recompute", "swap"}
        if self.activation_mode not in valid_activation_modes:
            raise ValueError(
                f"activation_mode must be one of {sorted(valid_activation_modes)}, "
                f"got {self.activation_mode!r}."
            )
        if self.plan_overrides is not None and not isinstance(self.plan_overrides, list):
            raise ValueError("plan_overrides must be a list when provided.")

    def validate(self) -> None:
        """Validate supported argument values."""
        self._validate_parallel_sizes()
        self._validate_world_size()
        self._validate_runtime_options()

    def build_distributed_setup(self) -> DistributedSetup:
        """Build the unified Hyper topology used by the model-construction path."""
        self.validate()
        world_size = dist.get_world_size()
        dp_size = world_size // self.cp_size
        dp_shard_size = self.fsdp_size or world_size
        if world_size % dp_shard_size != 0:
            raise ValueError(
                f"world_size ({world_size}) must be divisible by fsdp_size ({dp_shard_size})."
            )

        expert_dp_size = world_size // self.ep_size
        edp_shard_size = self.efsdp_size or expert_dp_size
        mesh_context = MeshContext(
            dp_size=dp_size,
            dp_replicate_size=world_size // dp_shard_size,
            dp_shard_size=dp_shard_size,
            edp_shard_size=edp_shard_size,
            cp_size=self.cp_size,
            ep_size=self.ep_size,
        )
        mesh_context.build_meshs(
            get_device_type() if self.device_type == "auto" else self.device_type,
            world_size,
        )

        param_dtype = _CANONICAL_DTYPES.get(self.param_dtype) if self.param_dtype is not None else None
        reduce_dtype = _CANONICAL_DTYPES.get(self.reduce_dtype) if self.reduce_dtype is not None else None
        strategy_config = FSDP2Config(
            dp_shard_size=dp_shard_size,
            edp_shard_size=edp_shard_size,
            mix_precision=FSDP2MixedPrecisionConfig(
                param_dtype=param_dtype,
                reduce_dtype=reduce_dtype,
                output_dtype=param_dtype,
            ),
            reshard_after_forward=True if self.reshard_after_forward is None else self.reshard_after_forward,
        )
        default_plan_overrides = [
            {
                "match": "*.self_attn",
                "when": "cp",
                "inner_target": "self",
                "inner_wrapper": "sdpa_hf",
                "region_dispatch": False,
            }
        ]

        raw_plan_overrides = [
            *default_plan_overrides,
            *(self.plan_overrides or []),
        ]
        entries = resolve_component(
            raw_plan_overrides,
            annotation=list[PlanOverride],
            path="hyper_parallel_args.plan_overrides",
        )
        return DistributedSetup(
            mesh_context=mesh_context,
            strategy_config=strategy_config,
            plan_overrides=entries_to_plan_overrides(
                entries,
                cp_size=self.cp_size,
                ep_size=self.ep_size,
            ),
            module_replacements=entries_to_module_replacements(entries),
        )

    @classmethod
    def from_dict(cls, config: dict) -> "HyperParallelArguments":
        """Build arguments from a plain dict."""
        known_fields = set(cls.__dataclass_fields__)  # pylint: disable=no-member
        parallel_args = cls(
            **{key: value for key, value in config.items() if key in known_fields}
        )
        parallel_args.validate()
        return parallel_args

    @classmethod
    def from_finetuning_args(cls, finetuning_args: Any) -> "HyperParallelArguments":
        """Extract HyperParallel arguments from LlamaFactory finetuning args."""
        raw = getattr(finetuning_args, "hyper_parallel_args", None)
        if raw is None:
            parallel_args = cls()
            parallel_args.validate()
            return parallel_args
        if isinstance(raw, str):
            with open(raw, "r", encoding="utf-8") as file:
                raw = json.load(file)
        if not isinstance(raw, dict):
            raise TypeError(
                "finetuning_args.hyper_parallel_args must be a dict or JSON file path, "
                f"got {type(raw).__name__}."
            )
        return cls.from_dict(raw)


# ---------------------------------------------------------------------------
# LlamaFactory model construction
# ---------------------------------------------------------------------------


def parallelize_model(
    model: nn.Module,
    distributed_setup: DistributedSetup,
    *,
    pretrained_path: str | None = None,
    device: torch.device | None = None,
    activation_checkpoint: Any | None = None,
    validate_placement: bool = False,
    model_init_dtype: str | None = None,
) -> nn.Module:
    """Apply Hyper's unified path to a LlamaFactory-built meta model."""
    from hyper_parallel.models._transformers.model_builder import (  # pylint: disable=C0415
        apply_model_infrastructure,
        instantiate_infrastructure,
    )

    if not any(tensor.is_meta for tensor in (*model.parameters(), *model.buffers())):
        raise ValueError("parallelize_model requires a model containing meta tensors")

    if device is None:
        device_type = get_device_type()
        device = (
            torch.device("cpu")
            if device_type == "cpu"
            else torch.device(device_type, get_device_id())
        )

    sharding_planner, fsdp2_manager = instantiate_infrastructure(
        distributed_setup=distributed_setup,
        device=device,
    )
    return apply_model_infrastructure(
        model,
        mesh=distributed_setup.mesh_context,
        sharding_planner=sharding_planner,
        fsdp2_manager=fsdp2_manager,
        is_meta_device=True,
        is_hf_model=True,
        device=device,
        load_base_model=pretrained_path is not None,
        pretrained_path=pretrained_path,
        validate_placement=validate_placement,
        distributed_setup=distributed_setup,
        activation_checkpoint=activation_checkpoint,
        model_init_dtype=model_init_dtype,
    )


# ---------------------------------------------------------------------------
# Checkpoint and export helpers
# ---------------------------------------------------------------------------


def _localize_optimizer_state(value):
    """Recursively convert optimizer DTensors and tensors to local CPU tensors."""
    if isinstance(value, DTensor):
        return value.to_local().detach().cpu()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _localize_optimizer_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_localize_optimizer_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_localize_optimizer_state(item) for item in value)
    return value


def _get_chained_optimizer(optimizer) -> ChainedOptimizer | None:
    """Unwrap an Accelerate optimizer when it contains a Hyper optimizer chain."""
    seen = set()
    while optimizer is not None and id(optimizer) not in seen:
        if isinstance(optimizer, ChainedOptimizer):
            return optimizer
        seen.add(id(optimizer))
        optimizer = getattr(optimizer, "optimizer", None)
    return None


def _get_optimizer_param_by_idx(optimizer) -> dict[int, torch.nn.Parameter]:
    """Map optimizer state indices to the current optimizer parameters."""
    param_by_idx: dict[int, torch.nn.Parameter] = {}
    param_idx = 0
    for group in optimizer.param_groups:
        for param in group["params"]:
            param_by_idx[param_idx] = param
            param_idx += 1
    return param_by_idx


def _get_optimizer_param_device(param):
    """Return the target device for a tensor restored into an optimizer state."""
    if isinstance(param, DTensor):
        return param.to_local().device
    return param.device


def _restore_optimizer_state_value(current_state, key, param, saved_val) -> None:
    """Restore one optimizer state entry while preserving existing tensor objects."""
    current_val = current_state.get(key)
    if current_val is None:
        if isinstance(saved_val, torch.Tensor):
            current_state[key] = saved_val.to(_get_optimizer_param_device(param))
        else:
            current_state[key] = saved_val
        return

    if isinstance(current_val, DTensor):
        local = current_val.to_local()
        local.copy_(saved_val.to(local.device))
    elif isinstance(current_val, torch.Tensor):
        current_val.copy_(saved_val.to(current_val.device))
    else:
        current_state[key] = saved_val


def _restore_optimizer_param_groups(optimizer, saved_sd: dict) -> None:
    """Restore non-parameter optimizer group options from a saved state dict."""
    for saved_group, current_group in zip(
        saved_sd.get("param_groups", []), optimizer.param_groups
    ):
        for key, val in saved_group.items():
            if key != "params":
                current_group[key] = val


def _load_local_optimizer_state(optimizer, saved_sd: dict) -> None:
    """Copy saved local optimizer state into the optimizer's current state."""
    param_by_idx = _get_optimizer_param_by_idx(optimizer)

    for param_idx, saved_state in saved_sd.get("state", {}).items():
        param_idx = int(param_idx) if isinstance(param_idx, str) else param_idx
        param = param_by_idx.get(param_idx)
        if param is None or param not in optimizer.state:
            continue
        current_state = optimizer.state[param]
        for key, saved_val in saved_state.items():
            _restore_optimizer_state_value(current_state, key, param, saved_val)

    _restore_optimizer_param_groups(optimizer, saved_sd)


# ---------------------------------------------------------------------------
# Optimizer wiring
# ---------------------------------------------------------------------------


def wrap_optimizer_with_skip_dtensor_dispatch(optimizer: Any) -> None:
    """Wrap chain and leaf optimizer steps so DTensor updates use local shards."""
    wrapped_attribute = "_distributed_step_wrapped"
    optimizers = list(getattr(optimizer, "optimizers_dict", {}).values()) + [optimizer]
    for current_optimizer in optimizers:
        if getattr(current_optimizer, wrapped_attribute, False):
            continue

        original_step = current_optimizer.step

        def _distributed_step(bound_optimizer, *args, _original_step=original_step, **kwargs):
            del bound_optimizer
            with SkipDTensorDispatch():
                return _original_step(*args, **kwargs)

        current_optimizer.step = types.MethodType(_distributed_step, current_optimizer)
        setattr(current_optimizer, wrapped_attribute, True)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def export_to_hf_format(model: nn.Module, tokenizer: Any, save_dir: str) -> None:
    """Gather full state dict via HyperParallel and save in HuggingFace-compatible format."""
    from torch.distributed.checkpoint.state_dict import (  # pylint: disable=import-outside-toplevel
        StateDictOptions,
    )

    from hyper_parallel.core.fully_shard.api import (  # pylint: disable=C0415
        get_model_state_dict as get_distributed_model_state_dict,
    )

    export_dir = Path(save_dir)
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    state_dict = get_distributed_model_state_dict(model, options=options)

    if dist.get_rank() == 0:
        export_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(model, "save_pretrained"):
            model.save_pretrained(str(export_dir), state_dict=state_dict)
        else:
            torch.save(state_dict, export_dir / "pytorch_model.bin")

        if tokenizer is not None:
            tokenizer.save_pretrained(str(export_dir))

    if dist.get_world_size() > 1:
        dist.barrier()


def save_hsdp_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    lr_scheduler: Any,
    output_dir: str,
    should_save_scheduler: bool = True,
) -> None:
    """Save HSDP model/optimizer shards per-rank and scheduler."""
    from hyper_parallel.core.distributed_checkpoint.api import (  # pylint: disable=import-outside-toplevel
        save as checkpoint_save,
    )

    os.makedirs(output_dir, exist_ok=True)
    rank = dist.get_rank()

    model_dir = os.path.join(output_dir, f"{HSDP_MODEL_NAME}_0")
    os.makedirs(model_dir, exist_ok=True)
    logger.info("Saving HSDP model shards to %s (rank %d)", model_dir, rank)
    model_sd = model.state_dict()
    checkpoint_save(model_sd, checkpoint_id=model_dir, use_collectives=False)

    if optimizer is not None:
        optim_file = os.path.join(output_dir, f"{HSDP_OPTIMIZER_NAME}_rank{rank}.pt")
        logger.info("Saving optimizer shard to %s", optim_file)
        local_optim_sd = _localize_optimizer_state(optimizer.state_dict())
        torch.save(local_optim_sd, optim_file)

    if should_save_scheduler and lr_scheduler is not None:
        torch.save(lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))


def load_hsdp_model(model: nn.Module, checkpoint_dir: str) -> bool:
    """Load a model from an HSDP sharded checkpoint."""
    from hyper_parallel.core.distributed_checkpoint.api import (  # pylint: disable=import-outside-toplevel
        load as checkpoint_load,
    )

    model_dir = os.path.join(checkpoint_dir, f"{HSDP_MODEL_NAME}_0")

    if not os.path.isdir(model_dir):
        return False

    logger.info("Loading HSDP model shards from %s", model_dir)
    state_dict = model.state_dict()
    checkpoint_load(state_dict, checkpoint_id=model_dir, use_collectives=False)
    model.load_state_dict(state_dict)
    return True


def load_hsdp_optimizer_and_scheduler(
    optimizer: torch.optim.Optimizer | None,
    lr_scheduler: Any,
    checkpoint_dir: str,
) -> None:
    """Load optimizer/scheduler from per-rank checkpoint files."""
    if checkpoint_dir is None:
        return

    rank = dist.get_rank()
    optim_file = os.path.join(checkpoint_dir, f"{HSDP_OPTIMIZER_NAME}_rank{rank}.pt")

    if os.path.isfile(optim_file) and optimizer is not None:
        logger.info("Loading optimizer shard from %s", optim_file)
        saved_sd = torch.load(optim_file, map_location="cpu", weights_only=True)
        if _get_chained_optimizer(optimizer) is not None:
            optimizer.load_state_dict(saved_sd)
        else:
            _load_local_optimizer_state(optimizer, saved_sd)

    scheduler_file = os.path.join(checkpoint_dir, "scheduler.pt")
    if os.path.isfile(scheduler_file) and lr_scheduler is not None:
        lr_scheduler.load_state_dict(
            torch.load(scheduler_file, map_location="cpu", weights_only=True)
        )
