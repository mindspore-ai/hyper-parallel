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
"""Scoped external contracts for testing real adapters without installing VeRL.

Only external data types and interfaces are replaced. Native configuration,
execution, checkpoint, conversion and policy arithmetic come from production
modules. No substitute module remains installed between tests or at collection.
"""

import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib.machinery import ModuleSpec
from importlib.util import find_spec, module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import torch


def _unprovided_boundary(*args, **kwargs):
    raise AssertionError("Supply and assert this external boundary explicitly in the test")


@dataclass
class CheckpointConfig:
    """External checkpoint options accepted by the adapter."""

    save_contents: list[str] = field(default_factory=lambda: ["model", "optimizer", "extra"])
    load_contents: list[str] = field(default_factory=lambda: ["model", "optimizer", "extra"])
    async_save: bool = False
    strict: bool = True
    save_lora_only: bool = False


@dataclass
class ActorConfig:
    """External actor construction fields used by policy validation."""

    strategy: str = ""
    rollout_n: int = 1
    use_dynamic_bsz: bool = True

    def __post_init__(self) -> None:
        """Leave adapter-specific policy validation to the real subclass."""


@dataclass
class EngineConfig:
    """External engine options consumed by native configuration adaptation."""

    _mutable_fields = set()
    strategy: str = ""
    router_replay: SimpleNamespace = field(default_factory=lambda: SimpleNamespace(mode="disabled"))
    forward_only: bool = False
    use_fused_kernels: bool = False
    use_remove_padding: bool = True
    param_offload: bool = False
    optimizer_offload: bool = False
    seed: int = 42
    full_determinism: bool = False

    def __post_init__(self) -> None:
        """Leave native recipe resolution to the real adapter subclass."""


class BaseEngine:
    """External engine boundary; optimizer lifecycle is tested on native code."""

    def to(self, device, model=True, optimizer=True, grad=True) -> None:
        """Accept a residency request without implementing framework policy."""


class BaseEngineCtx:
    """External context interface with no transfer or optimizer implementation."""

    def __init__(self, engine, mode, zero_grad_on_exit=True, **kwargs):
        del kwargs
        self.engine = engine
        self.mode = mode
        self.zero_grad_on_exit = zero_grad_on_exit

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return None


class BaseCheckpointManager:
    """External checkpoint bindings; state handoff uses the real adapter."""

    def __init__(self, model, optimizer, lr_scheduler, processing_class=None, **kwargs):
        del kwargs
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.processing_class = processing_class
        self.rank = torch.distributed.get_rank()

    get_rng_state = load_rng_state = register_checkpoint = _unprovided_boundary


class TensorBatch(dict):
    """Minimal batch fixture with PyTorch tensors and separate metadata."""

    def __init__(self, tensors: dict[str, torch.Tensor], batch_size: int) -> None:
        super().__init__(tensors)
        self.batch_size = batch_size
        self.metadata = {}

    def __len__(self) -> int:
        """Return the sample count expected by the external batch interface."""
        return self.batch_size

    def to(self, device: torch.device) -> "TensorBatch":
        """Move real tensors and preserve the batch's non-tensor metadata."""
        batch = TensorBatch({name: tensor.to(device) for name, tensor in self.items()}, self.batch_size)
        batch.metadata.update(self.metadata)
        return batch


def list_of_dict_to_tensordict(rows):
    """Build CPU batch fixtures with real PyTorch jagged token sequences."""
    columns = {}
    for name in rows[0]:
        values = [row[name] for row in rows]
        columns[name] = (torch.stack(values) if values[0].ndim == 0
                         else torch.nested.nested_tensor(values, layout=torch.jagged))
    return TensorBatch(columns, batch_size=len(rows))


def assign_non_tensor(data, **values):
    """Attach metadata without importing external batch libraries."""
    data.metadata.update(values)


def get_non_tensor_data(data, key, default=None):
    """Read non-tensor metadata from a batch fixture."""
    return data.metadata.get(key, default)


def _detach_tree(value):
    if isinstance(value, dict):
        return {name: _detach_tree(item) for name, item in value.items()}
    return value.detach() if isinstance(value, torch.Tensor) else value


def _module(name, **attributes):
    module = ModuleType(name)
    module.__spec__ = ModuleSpec(name, loader=None, is_package=True)
    module.__path__ = []
    vars(module).update(attributes)
    return module


def _external_modules():
    """Build the external interface substitutes used by adapter unit tests."""
    modules = {}

    def add(name, **attributes):
        parts = name.split(".")
        for end in range(1, len(parts) + 1):
            package = ".".join(parts[:end])
            if package not in modules:
                modules[package] = _module(package)
        vars(modules[name]).update(attributes)

    add("verl.trainer.config", CheckpointConfig=CheckpointConfig)
    add("verl.workers.config.actor", ActorConfig=ActorConfig)
    add("verl.workers.config.critic", CriticConfig=object)
    add("verl.workers.config.engine", EngineConfig=EngineConfig, EngineRouterReplayConfig=SimpleNamespace)
    add("verl.workers.config.optimizer", OptimizerConfig=SimpleNamespace)
    add("verl.utils.device", get_device_id=lambda: 0, get_device_name=lambda: "cpu",
        get_torch_device=lambda: torch.cpu)
    add("verl.utils.checkpoint.checkpoint_manager", BaseCheckpointManager=BaseCheckpointManager)
    add("verl.utils.tensordict_utils", assign_non_tensor=assign_non_tensor,
        get_non_tensor_data=get_non_tensor_data, unwrap_non_tensor_data=lambda item: item,
        list_of_dict_to_tensordict=list_of_dict_to_tensordict)
    add("verl.utils.dataset.dataset_utils", DatasetPadMode=SimpleNamespace(NO_PADDING="no_padding"))
    add("verl.utils.fsdp_utils", load_fsdp_optimizer=_unprovided_boundary,
        offload_fsdp_optimizer=_unprovided_boundary)
    add("verl.utils.torch_functional", logprobs_from_logits=_unprovided_boundary)
    add("verl.workers.engine.base", BaseEngine=BaseEngine, BaseEngineCtx=BaseEngineCtx,
        EngineRegistry=SimpleNamespace(register=_unprovided_boundary))
    add("verl.workers.engine.utils", detach_tree=_detach_tree,
        postprocess_batch_func=_unprovided_boundary, prepare_micro_batches=_unprovided_boundary)
    callbacks = {
        name: _unprovided_boundary for name in (
            "apply_kl_penalty", "compute_advantage_for_multi_trajectories",
            "compute_rollout_correction_and_add_to_batch", "response_to_nested",
            "get_metric_data_with_optional_routed_experts", "compute_moe_lb_metrics",
            "compute_data_metrics", "compute_timing_metrics", "compute_throughout_metrics",
            "compute_variance_proxy_metrics", "compute_spec_decode_metrics",
        )
    }
    add("verl.trainer.ppo.v1.trainer_base", PPOTrainer=type("PPOTrainer", (), {}),
        KVBatchMeta=SimpleNamespace, DataProto=SimpleNamespace, TensorDict=TensorBatch,
        np=np, tq=SimpleNamespace(), **callbacks)
    return modules


@contextmanager
def contract_modules():
    """Publish only these contract bindings and restore every affected import."""
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in MODULES}
    parents = []
    sys.modules.update(MODULES)
    for name, module in MODULES.items():
        parent_name, _, attribute = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None:
            parents.append((parent, attribute, getattr(parent, attribute, missing)))
            setattr(parent, attribute, module)
    try:
        yield
    finally:
        for parent, attribute, value in reversed(parents):
            if value is missing:
                delattr(parent, attribute)
            else:
                setattr(parent, attribute, value)
        for name, value in previous.items():
            if value is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


SOURCE_ROOT = Path(find_spec("hyper_parallel.integration.verl").origin).parent
MODULES = _external_modules()
MODULES["hyper_parallel.integration.verl.patch"] = _module("hyper_parallel.integration.verl.patch")
SOURCES = (
    "config", "engine.trainer", "engine.checkpoint", "engine.weight_export", "engine.transformer_impl",
    "patch.engine", "patch.fix_v1_optional_fields", "patch.resolver",
)
for source in SOURCES:
    module_name = "hyper_parallel.integration.verl." + source
    module_spec = spec_from_file_location(module_name, SOURCE_ROOT / (source.replace(".", "/") + ".py"))
    MODULES[module_name] = module_from_spec(module_spec)
with contract_modules():
    for source in SOURCES:
        loaded = MODULES["hyper_parallel.integration.verl." + source]
        loaded.__spec__.loader.exec_module(loaded)

ADAPTERS = SimpleNamespace(
    config=MODULES["hyper_parallel.integration.verl.config"],
    trainer=MODULES["hyper_parallel.integration.verl.engine.trainer"],
    checkpoint=MODULES["hyper_parallel.integration.verl.engine.checkpoint"],
    weight_export=MODULES["hyper_parallel.integration.verl.engine.weight_export"],
    engine=MODULES["hyper_parallel.integration.verl.engine.transformer_impl"],
    registration=MODULES["hyper_parallel.integration.verl.patch.engine"],
    optional_fields=MODULES["hyper_parallel.integration.verl.patch.fix_v1_optional_fields"],
    resolver=MODULES["hyper_parallel.integration.verl.patch.resolver"],
)
