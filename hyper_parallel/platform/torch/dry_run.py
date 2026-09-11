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
"""Configurable FakeTensor LLM dry-run support for the HyperModels Trainer."""
# pylint: disable=protected-access
import copy
import functools
import inspect
import json
import logging
import os
import sys
import tempfile
import weakref
from collections.abc import Mapping
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, fields, is_dataclass, replace
from decimal import Decimal
from enum import Enum
from fnmatch import fnmatchcase
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, Callable, ContextManager, Dict, Iterator, Optional, TypeVar

from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.platform.torch.memory_report import (  # pylint: disable=unused-import
    build_memory_csv_rows,
    write_memory_csv,
)


_SCHEMA_VERSION = 3
_GIB = 2 ** 30
_PERSISTENT_END_INDEX = 2 ** 63 - 1
_SMALL_ALLOCATION_BYTES = 2 ** 20
_VIRTUAL_ADDRESS_BASE = 0xF00000000000
_VIRTUAL_ADDRESS_ALIGNMENT = 512
_FAKE_MEMORY_POOL_TYPE = "FakeTensorLogicalMemoryPool"
logger = logging.getLogger(__name__)
_LIMITATIONS = [
    "Reports logical live tensor bytes, not allocator reserved memory or fragmentation.",
    "Kernel workspaces and opaque third-party fused-operator allocations are not tracked.",
    "Checkpoint loading, dataloaders, callbacks, and gradient accumulation across training steps are not simulated.",
    "Pipeline parallel dry-run is not supported by the base implementation.",
    "Unconfigured value dependencies fail with an actionable diagnostic.",
]


@dataclass(frozen=True)
class DryRunRuntime:
    """Distributed identity supplied by ``torchrun`` for one dry-run worker."""

    rank: int
    world_size: int
    local_rank: int

    @classmethod
    def from_torchrun_env(cls) -> "DryRunRuntime":
        """Build and validate the current worker identity.

        Returns:
            The validated dry-run runtime identity.

        Raises:
            RuntimeError: If the process was not launched by ``torchrun``.
            ValueError: If a launcher variable is outside its valid range.
        """
        required_names = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
        missing_names = [name for name in required_names if name not in os.environ]
        if missing_names:
            raise RuntimeError(
                "Multi-rank HyperDryRun must be launched with torchrun; "
                f"missing environment variables: {missing_names}"
            )
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        if world_size < 1:
            raise ValueError(f"WORLD_SIZE must be >= 1, got {world_size}")
        if rank < 0 or rank >= world_size:
            raise ValueError(
                f"RANK must be in [0, {world_size}), got {rank}"
            )
        if local_rank < 0:
            raise ValueError(f"LOCAL_RANK must be >= 0, got {local_rank}")
        return cls(rank=rank, world_size=world_size, local_rank=local_rank)


class DryRunBatchMocker:
    """Erase one CPU training batch while preserving its complete structure."""

    def __init__(self, fake_mode: Any, device: Any) -> None:
        """Store the FakeTensor mode and logical simulation device."""
        self.fake_mode = fake_mode
        self.device = device

    def mock(self, value: Any) -> Any:
        """Recursively replace tensor leaves with value-free FakeTensors."""
        import torch  # pylint: disable=C0415

        if isinstance(value, torch.Tensor):
            with self.fake_mode:
                return torch.empty_like(
                    value,
                    device=self.device,
                    memory_format=torch.preserve_format,
                )
        if isinstance(value, Mapping):
            return self._mock_mapping(value)
        if isinstance(value, (list, tuple)):
            return self._mock_sequence(value)
        if is_dataclass(value) and not isinstance(value, type):
            return self._mock_dataclass(value)
        if hasattr(value, "__dict__") and self._contains_tensor(vars(value)):
            raise ValueError(
                "Dry-run cannot mock tensor-bearing batch carrier "
                f"{type(value).__module__}.{type(value).__qualname__}"
            )
        return value

    def _mock_mapping(self, value: Mapping[Any, Any]) -> Mapping[Any, Any]:
        """Mock mapping values while preserving the mapping carrier type."""
        mocked_items = [(key, self.mock(item)) for key, item in value.items()]
        if type(value) is dict:  # pylint: disable=unidiomatic-typecheck
            return dict(mocked_items)
        try:
            mocked_mapping = copy.copy(value)
            mocked_mapping.clear()
            mocked_mapping.update(mocked_items)
            return mocked_mapping
        except (AttributeError, TypeError):
            try:
                return type(value)(mocked_items)
            except TypeError as exc:
                raise ValueError(
                    "Dry-run cannot preserve batch mapping carrier "
                    f"{type(value).__module__}.{type(value).__qualname__}"
                ) from exc

    def _mock_sequence(self, value: Any) -> Any:
        """Mock list and tuple values while preserving named tuples."""
        mocked = tuple(self.mock(item) for item in value)
        if isinstance(value, list):
            return list(mocked)
        if hasattr(value, "_fields"):
            return type(value)(*mocked)
        return mocked

    def _mock_dataclass(self, value: Any) -> Any:
        """Mock initialized and deferred dataclass fields."""
        updates = {
            field.name: self.mock(getattr(value, field.name))
            for field in fields(value)
            if field.init
        }
        mocked_dataclass = replace(value, **updates)
        for field in fields(value):
            if not field.init:
                object.__setattr__(
                    mocked_dataclass,
                    field.name,
                    self.mock(getattr(value, field.name)),
                )
        return mocked_dataclass

    def _contains_tensor(self, value: Any) -> bool:
        """Return whether an unsupported object recursively owns a tensor."""
        import torch  # pylint: disable=C0415

        if isinstance(value, torch.Tensor):
            return True
        if isinstance(value, Mapping):
            return any(self._contains_tensor(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(self._contains_tensor(item) for item in value)
        return False


def derive_tp_target_counts(
        loss_inputs: Mapping[str, Any],
        vocab_size: int,
        tp_size: int,
) -> tuple[int, tuple[int, ...]]:
    """Derive valid-label ownership using the real loss-parallel vocab split."""
    import torch  # pylint: disable=C0415

    labels = loss_inputs.get("shift_labels")
    if labels is None:
        labels = loss_inputs.get("labels")
        if labels is None:
            raise ValueError("Dry-run loss_inputs must contain labels")
        labels = labels[..., 1:]
    if not isinstance(labels, torch.Tensor):
        raise ValueError("Dry-run labels must be a torch.Tensor")
    valid_mask = labels.ne(IGNORE_INDEX)
    loss_mask = loss_inputs.get("loss_mask")
    if isinstance(loss_mask, torch.Tensor):
        if loss_mask.shape != labels.shape:
            loss_mask = loss_mask[..., -labels.shape[-1]:]
        valid_mask = valid_mask & loss_mask.reshape_as(labels).bool()
    valid_labels = labels[valid_mask]
    chunk_size = (vocab_size + tp_size - 1) // tp_size
    owned = tuple(
        int(((valid_labels >= rank * chunk_size) &
             (valid_labels < min((rank + 1) * chunk_size, vocab_size))).sum().item())
        for rank in range(tp_size)
    )
    valid_count = int(valid_labels.numel())
    if sum(owned) != valid_count:
        raise ValueError("Dry-run labels contain token IDs outside the configured vocabulary")
    return valid_count, owned


@dataclass(frozen=True)
class _DryRunMoERoutingPlan:
    """Aggregate routed-token load for one MoE dry-run invocation."""

    source_expert_loads: tuple[tuple[int, ...], ...]
    local_expert_loads: tuple[int, ...]
    received_expert_loads: tuple[int, ...]
    input_split_sizes: tuple[int, ...]
    output_split_sizes: tuple[int, ...]
    outgoing_tokens: int


@dataclass(frozen=True)
class _DryRunMoEDispatchContext:
    """Shape-only state retained between dry-run dispatch and combine."""

    rank_major_shape: tuple[int, ...]
    permuted_indices: Any
    input_split_sizes: tuple[int, ...]
    output_split_sizes: tuple[int, ...]


@dataclass(frozen=True)
class _ValueDependencyRule:
    """One validated but not yet expanded value-dependency rule."""

    index: int
    match: str
    handler: str
    path: str
    inputs: Dict[str, Any]
    optional: bool


class ValueDependencyStage(Enum):
    """Lifecycle stages at which a value-dependency handler may install state."""

    PARALLELIZE = "parallelize"
    FAKE_STEP = "fake_step"


class ValueDependencyHandler:
    """Base interface for configured dry-run value-dependency handlers."""

    name = ""
    paths: tuple[str, ...] = ()

    def discover(self, model: Any, context: Any) -> list[str]:
        """Return logical targets supported by this handler."""
        del model, context
        return []

    def recognizes(self, target: str, module: Any) -> bool:
        """Return whether this handler can explain an unconfigured target."""
        del target, module
        return False

    def validate(self, target: str, path: str, inputs: Dict[str, Any], context: Any) -> None:
        """Validate one expanded rule."""
        del target, inputs, context
        if self.paths and path not in self.paths:
            raise ValueError(
                f"value dependency handler {self.name!r} path must be one of "
                f"{self.paths}, got {path!r}"
            )

    def install(
            self, stage: ValueDependencyStage, target: str, path: str,
            inputs: Dict[str, Any], context: Any,
    ) -> ContextManager[Any]:
        """Return scoped runtime state for one expanded rule."""
        del stage, target, path, inputs, context
        return nullcontext()

    def prepare_batch(
            self, target: str, path: str, inputs: Dict[str, Any],
            batch: Dict[str, Any], context: Any,
    ) -> None:
        """Apply shape-only batch changes for one expanded rule."""
        del target, path, inputs, batch, context

    def resolved_metadata(self) -> Dict[str, Any]:
        """Return handler-specific resolved report metadata."""
        return {}


_VALUE_DEPENDENCY_HANDLER_FACTORIES: Dict[str, Callable[[], ValueDependencyHandler]] = {}


def register_value_dependency_handler(
        name: str, factory: Callable[[], ValueDependencyHandler],
) -> None:
    """Register one dry-run value-dependency handler factory.

    Args:
        name: Unique handler name used by YAML rules.
        factory: Zero-argument factory returning a fresh handler instance.

    Raises:
        ValueError: If the name is empty, already registered, or inconsistent
            with the constructed handler.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("value dependency handler name must be a non-empty string")
    if name in _VALUE_DEPENDENCY_HANDLER_FACTORIES:
        raise ValueError(f"value dependency handler {name!r} is already registered")
    handler = factory()
    if not isinstance(handler, ValueDependencyHandler) or handler.name != name:
        raise ValueError(
            f"value dependency handler factory for {name!r} must return "
            f"ValueDependencyHandler(name={name!r})"
        )
    _VALUE_DEPENDENCY_HANDLER_FACTORIES[name] = factory


def _is_supported_moe_module(module: Any) -> bool:
    """Return whether a module exposes a supported routed-expert layout."""
    experts = getattr(module, "experts", None)
    if experts is None or (
            getattr(module, "gate", None) is None
            and getattr(module, "router", None) is None
    ):
        return False
    layouts = (
        ("w1", "w2", "w3"),
        ("gate_up_proj", "down_proj"),
        ("gate_proj", "up_proj", "down_proj"),
    )
    return any(all(hasattr(experts, name) for name in layout) for layout in layouts)


class MoERoutingValueDependencyHandler(ValueDependencyHandler):
    """Validate and describe configured routed-expert replay."""

    name = "moe_routing"
    paths = ("balanced", "hotspot", "explicit")

    def discover(self, model: Any, context: Any) -> list[str]:
        """Return supported routed-expert module FQNs."""
        del context
        return [name for name, module in model.named_modules() if name and _is_supported_moe_module(module)]

    def recognizes(self, target: str, module: Any) -> bool:
        """Recognize supported routed-expert modules."""
        del target
        return _is_supported_moe_module(module)

    def validate(self, target: str, path: str, inputs: Dict[str, Any], context: Any) -> None:
        """Validate path-specific MoE rule inputs."""
        super().validate(target, path, inputs, context)
        allowed = {
            "balanced": set(),
            "hotspot": {"hotspot_expert"},
            "explicit": {"source_expert_loads"},
        }[path]
        unknown = sorted(set(inputs) - allowed)
        if unknown:
            raise ValueError(f"MoE rule for {target!r} contains unsupported inputs: {unknown}")
        if path == "hotspot" and (
                not isinstance(inputs.get("hotspot_expert"), int)
                or isinstance(inputs.get("hotspot_expert"), bool)
                or inputs["hotspot_expert"] < 0
        ):
            raise ValueError(f"MoE hotspot rule for {target!r} requires non-negative hotspot_expert")
        if path == "explicit" and not isinstance(inputs.get("source_expert_loads"), list):
            raise ValueError(f"MoE explicit rule for {target!r} requires source_expert_loads")

    def install(
            self, stage: ValueDependencyStage, target: str, path: str,
            inputs: Dict[str, Any], context: Any,
    ) -> ContextManager[Any]:
        """Install routed-expert replay once for the configured fake step."""
        del target, path, inputs
        if stage is ValueDependencyStage.FAKE_STEP:
            return _MoERoutingValueDependencyRuntime(context.profile, context.base)
        return nullcontext()


class BranchValueDependencyHandler(ValueDependencyHandler):
    """Provide explicit values to instrumented Python branch decisions."""

    name = "branch"

    def discover(self, model: Any, context: Any) -> list[str]:
        """Expose module FQNs and logical execution regions."""
        del context
        return [name for name, _ in model.named_modules() if name] + ["loss", "backward", "optimizer"]

    def validate(self, target: str, path: str, inputs: Dict[str, Any], context: Any) -> None:
        """Validate named scalar branch decisions."""
        del path, context
        decisions = inputs.get("decisions")
        if set(inputs) != {"decisions"} or not isinstance(decisions, dict) or not decisions:
            raise ValueError(f"branch rule for {target!r} requires a non-empty decisions mapping")
        for name, value in decisions.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"branch rule for {target!r} contains an invalid decision name")
            if not isinstance(value, (bool, int, float, str)) and value is not None:
                raise ValueError(
                    f"branch decision {name!r} for {target!r} must be a YAML scalar"
                )


class OperatorDebugValueDependencyHandler(ValueDependencyHandler):
    """Validate emergency source-anchored ATen output mocks."""

    name = "operator_debug"

    def discover(self, model: Any, context: Any) -> list[str]:
        """Expose module FQNs and logical execution regions."""
        del context
        return [name for name, _ in model.named_modules() if name] + ["loss", "backward", "optimizer"]

    def validate(self, target: str, path: str, inputs: Dict[str, Any], context: Any) -> None:
        """Validate source anchors and mock return envelopes."""
        del path, context
        mocks = inputs.get("mocks")
        if set(inputs) != {"mocks"} or not isinstance(mocks, list) or not mocks:
            raise ValueError(f"operator_debug rule for {target!r} requires a non-empty mocks list")
        selectors = set()
        for index, mock in enumerate(mocks):
            location = f"operator_debug mock {index} for {target!r}"
            selector = _operator_debug_selector(mock, location)
            if selector in selectors:
                raise ValueError(f"{location} duplicates an earlier selector")
            selectors.add(selector)
            _validate_operator_return(mock["return"], location)


def _operator_debug_selector(mock: Any, location: str) -> tuple[Any, ...]:
    """Validate and return the stable selector for one operator mock."""
    if not isinstance(mock, dict) or set(mock) != {"source", "op", "occurrence", "return"}:
        raise ValueError(f"{location} must contain source, op, occurrence, and return")
    source = mock["source"]
    if not isinstance(source, dict) or set(source) != {"file", "function", "line"}:
        raise ValueError(f"{location}.source must contain file, function, and line")
    if (
            not isinstance(source["file"], str)
            or not isinstance(source["function"], str)
            or not isinstance(source["line"], int)
            or isinstance(source["line"], bool)
            or source["line"] < 1
    ):
        raise ValueError(f"{location} has an invalid source anchor")
    if not isinstance(mock["op"], str) or not mock["op"].startswith("aten."):
        raise ValueError(f"{location}.op must be an ATen overload name")
    occurrence = mock["occurrence"]
    if not isinstance(occurrence, int) or isinstance(occurrence, bool) or occurrence < 0:
        raise ValueError(f"{location}.occurrence must be a non-negative integer")
    return source["file"], source["function"], source["line"], mock["op"], occurrence


def _validate_operator_return(spec: Any, location: str) -> None:
    """Validate one recursive operator-debug return specification."""
    if not isinstance(spec, dict) or len(spec) != 1:
        raise ValueError(f"{location}.return must contain exactly one return kind")
    kind, value = next(iter(spec.items()))
    if kind == "scalar":
        if not isinstance(value, (bool, int, float)):
            raise ValueError(f"{location}.return.scalar must be bool, int, or float")
        return
    if kind == "tensor":
        _validate_operator_tensor_return(value, location)
        return
    if kind in ("tuple", "list"):
        _validate_operator_sequence_return(value, kind, location)
        return
    if kind == "by_global_rank":
        _validate_operator_rank_return(value, location)
        return
    raise ValueError(f"{location}.return uses unsupported kind {kind!r}")


def _validate_operator_tensor_return(value: Any, location: str) -> None:
    """Validate a tensor-shaped operator-debug return."""
    if not isinstance(value, dict) or set(value) != {"shape", "dtype"}:
        raise ValueError(f"{location}.return.tensor must contain shape and dtype")
    shape = value["shape"]
    if not isinstance(shape, list) or any(
            not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in shape
    ):
        raise ValueError(f"{location}.return.tensor.shape must contain non-negative integers")
    if not isinstance(value["dtype"], str) or not value["dtype"]:
        raise ValueError(f"{location}.return.tensor.dtype must be a non-empty string")


def _validate_operator_sequence_return(value: Any, kind: str, location: str) -> None:
    """Validate a tuple- or list-shaped operator-debug return."""
    if not isinstance(value, list):
        raise ValueError(f"{location}.return.{kind} must be a list")
    for index, item in enumerate(value):
        _validate_operator_return(item, f"{location}.return.{kind}[{index}]")


def _validate_operator_rank_return(value: Any, location: str) -> None:
    """Validate a rank-indexed operator-debug return."""
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{location}.return.by_global_rank must be a non-empty mapping")
    for rank, item in value.items():
        if not isinstance(rank, (int, str)) or not str(rank).isdigit():
            raise ValueError(f"{location}.return.by_global_rank keys must be non-negative ranks")
        _validate_operator_return(item, f"{location}.return.by_global_rank[{rank}]")


for _builtin_handler in (
        MoERoutingValueDependencyHandler,
        BranchValueDependencyHandler,
        OperatorDebugValueDependencyHandler,
):
    _VALUE_DEPENDENCY_HANDLER_FACTORIES[_builtin_handler.name] = _builtin_handler


class _DryRunValueProfile:
    """Parse, expand, and report unified value-dependency rules."""

    def __init__(self, config: Any) -> None:
        """Parse statically valid rules from ``dry_run.value_dependencies``."""
        self._config = config
        self._rules = self._parse_rules(config.value_dependencies)
        self._handlers = {
            name: factory()
            for name, factory in _VALUE_DEPENDENCY_HANDLER_FACTORIES.items()
        }
        self._resolved: Dict[str, Dict[str, _ValueDependencyRule]] = {}
        self._target_modules: Dict[str, Any] = {}
        self._resolved_moe_plans: Dict[str, Dict[str, Any]] = {}
        self._runtime_metadata: Dict[str, Any] = {}
        self._tp_cross_entropy_counts: Optional[tuple[int, tuple[int, ...]]] = None

    @staticmethod
    def _parse_rules(value_dependencies: Any) -> tuple[_ValueDependencyRule, ...]:
        """Validate the common rule envelope before model discovery."""
        if not isinstance(value_dependencies, dict):
            raise ValueError("dry_run.value_dependencies must be a mapping")
        unknown = sorted(set(value_dependencies) - {"rules"})
        if unknown:
            raise ValueError(f"dry_run.value_dependencies contains unknown fields: {unknown}")
        raw_rules = value_dependencies.get("rules", [])
        if not isinstance(raw_rules, list):
            raise ValueError("dry_run.value_dependencies.rules must be a list")
        rules = []
        for index, raw_rule in enumerate(raw_rules):
            location = f"dry_run.value_dependencies.rules[{index}]"
            if not isinstance(raw_rule, dict):
                raise ValueError(f"{location} must be a mapping")
            unknown_rule_fields = sorted(
                set(raw_rule) - {"match", "handler", "path", "inputs", "optional"}
            )
            if unknown_rule_fields:
                raise ValueError(f"{location} contains unknown fields: {unknown_rule_fields}")
            match = raw_rule.get("match")
            handler = raw_rule.get("handler")
            path = raw_rule.get("path")
            inputs = raw_rule.get("inputs", {})
            optional = raw_rule.get("optional", False)
            if not isinstance(match, str) or not match.strip():
                raise ValueError(f"{location}.match must be a non-empty string")
            if not isinstance(handler, str) or not handler.strip():
                raise ValueError(f"{location}.handler must be a non-empty string")
            if not isinstance(path, str) or not path.strip():
                raise ValueError(f"{location}.path must be a non-empty string")
            if not isinstance(inputs, dict):
                raise ValueError(f"{location}.inputs must be a mapping")
            if not isinstance(optional, bool):
                raise ValueError(f"{location}.optional must be a boolean")
            rules.append(_ValueDependencyRule(index, match, handler, path, inputs, optional))
        return tuple(rules)

    def bind_model(self, model: Any, context: Any = None) -> None:
        """Expand all configured rules against one model and logical scopes."""
        self._target_modules = dict(model.named_modules())
        candidates: Dict[tuple[str, str], list[tuple[int, _ValueDependencyRule]]] = {}
        for rule in self._rules:
            handler = self._handlers.get(rule.handler)
            if handler is None:
                raise ValueError(
                    f"value dependency rule {rule.index} uses unknown handler {rule.handler!r}; "
                    f"registered handlers: {sorted(self._handlers)}"
                )
            supported_targets = handler.discover(model, context)
            matches = [target for target in supported_targets if fnmatchcase(target, rule.match)]
            if not matches and not rule.optional:
                raise ValueError(
                    f"value dependency rule {rule.index} match {rule.match!r} did not match "
                    f"any target supported by handler {rule.handler!r}"
                )
            exact = int(rule.match in matches)
            for target in matches:
                candidates.setdefault((rule.handler, target), []).append((exact, rule))

        resolved: Dict[str, Dict[str, _ValueDependencyRule]] = {}
        for (handler_name, target), target_candidates in candidates.items():
            best_priority = max(priority for priority, _ in target_candidates)
            best_rules = [rule for priority, rule in target_candidates if priority == best_priority]
            if len(best_rules) > 1:
                indexes = [rule.index for rule in best_rules]
                raise ValueError(
                    f"value dependency rules {indexes} conflict for target {target!r} "
                    f"and handler {handler_name!r}"
                )
            rule = best_rules[0]
            self._handlers[handler_name].validate(target, rule.path, rule.inputs, context)
            resolved.setdefault(handler_name, {})[target] = rule
        self._resolved = resolved

    def project_model(self, model: Any, context: Any = None) -> None:
        """Keep validated value-dependency rules owned by one pipeline stage.

        ``bind_model`` must run on the complete model before this projection.
        Rules are intentionally not glob-expanded again: a valid global target
        may be owned by a different pipeline rank.
        """
        self._target_modules = dict(model.named_modules())
        projected = {}
        for handler_name, targets in self._resolved.items():
            handler = self._handlers[handler_name]
            supported_targets = set(handler.discover(model, context))
            local_targets = {}
            for target, rule in targets.items():
                for local_target in supported_targets:
                    if local_target == target or local_target.endswith(f".{target}"):
                        local_targets[local_target] = rule
            if local_targets:
                projected[handler_name] = local_targets
        self._resolved = projected

    def resolved_rules(self, handler: str) -> Dict[str, _ValueDependencyRule]:
        """Return expanded rules for one handler."""
        return dict(self._resolved.get(handler, {}))

    def handler(self, name: str) -> ValueDependencyHandler:
        """Return one fresh profile-owned handler instance."""
        return self._handlers[name]

    @property
    def moe_layers(self) -> dict[str, Any]:
        """Return per-layer routing overrides."""
        return {
            target: {"routing_policy": rule.path, **rule.inputs}
            for target, rule in self._resolved.get("moe_routing", {}).items()
        }

    @property
    def moe_enabled(self) -> bool:
        """Return whether the MoE adapter is enabled."""
        return bool(self._resolved.get("moe_routing"))

    def metadata(self) -> Dict[str, Any]:
        """Return a JSON-safe profile summary for the memory report."""
        return {
            "rules": [
                {
                    "index": rule.index,
                    "match": rule.match,
                    "handler": handler_name,
                    "path": rule.path,
                    "inputs": rule.inputs,
                    "resolved_target": target,
                }
                for handler_name, targets in self._resolved.items()
                for target, rule in targets.items()
            ],
            "handlers": {
                name: handler.resolved_metadata()
                for name, handler in self._handlers.items()
                if self._resolved.get(name)
            },
            "moe_routing": {"resolved_layers": self._resolved_moe_plans},
            "runtime": self._runtime_metadata,
        }

    def set_runtime_metadata(self, metadata: Dict[str, Any]) -> None:
        """Store JSON-safe runtime consumption details for the final report."""
        self._runtime_metadata = metadata

    def _moe_routing_matrix(
            self,
            layer: Dict[str, Any],
            ep_size: int,
            num_experts: int,
            expected: int,
            module_name: str,
    ) -> tuple[tuple[int, ...], ...]:
        """Build the configured or synthetic source-to-expert load matrix."""
        policy = layer["routing_policy"]
        if policy == "explicit":
            return self._explicit_moe_matrix(
                layer.get("source_expert_loads"),
                ep_size,
                num_experts,
                expected,
                module_name,
            )

        hotspot = int(layer.get("hotspot_expert", 0))
        if policy == "hotspot" and not 0 <= hotspot < num_experts:
            raise ValueError(f"MoE hotspot_expert for {module_name!r} must be in [0, {num_experts})")
        if policy == "hotspot":
            row = tuple(expected if index == hotspot else 0 for index in range(num_experts))
        else:
            base, remainder = divmod(expected, num_experts)
            row = tuple(base + int(index < remainder) for index in range(num_experts))
        return tuple(row for _ in range(ep_size))

    @staticmethod
    def _local_expert_start(
            module_name: str,
            ep_size: int,
            ep_rank: int,
            num_experts: int,
            local_expert_count: int,
    ) -> int:
        """Validate the expert layout and return this rank's first expert."""
        if local_expert_count < 1:
            raise ValueError(f"MoE {module_name!r} has no local experts")
        if local_expert_count == num_experts:
            return 0
        if num_experts % ep_size or local_expert_count != num_experts // ep_size:
            raise ValueError(f"MoE {module_name!r} local expert layout is incompatible with EP={ep_size}")
        return ep_rank * local_expert_count

    @staticmethod
    def _expert_ranges(
            ep_size: int,
            num_experts: int,
            local_expert_count: int,
    ) -> tuple[tuple[int, int], ...]:
        """Return the global expert range owned by each destination rank."""
        if local_expert_count == num_experts:
            return ((0, num_experts),)
        return tuple(
            (rank * local_expert_count, (rank + 1) * local_expert_count)
            for rank in range(ep_size)
        )

    def moe_routing_plan(
            self, module_name: str, module: Any, ep_size: int, ep_rank: int,
            local_tokens: int, local_expert_count: int,
    ) -> _DryRunMoERoutingPlan:
        """Build aggregate EP source-to-expert load for one MoE layer."""
        num_experts = int(module.num_experts)
        expected = local_tokens * int(module.top_k)
        layer = self.moe_layers.get(module_name)
        if layer is None:
            raise ValueError(f"MoE module {module_name!r} has no value-dependency rule")
        matrix = self._moe_routing_matrix(layer, ep_size, num_experts, expected, module_name)
        start = self._local_expert_start(
            module_name,
            ep_size,
            ep_rank,
            num_experts,
            local_expert_count,
        )
        local_loads = tuple(
            sum(row[index] for row in matrix)
            for index in range(start, start + local_expert_count)
        )
        received_loads = tuple(
            matrix[source_rank][expert_index]
            for source_rank in range(ep_size)
            for expert_index in range(start, start + local_expert_count)
        )
        expert_ranges = self._expert_ranges(ep_size, num_experts, local_expert_count)
        input_splits = tuple(
            sum(matrix[ep_rank][start_index:end_index])
            for start_index, end_index in expert_ranges
        )
        output_splits = tuple(
            sum(matrix[source_rank][start:start + local_expert_count])
            for source_rank in range(ep_size)
        )
        plan = _DryRunMoERoutingPlan(
            matrix,
            local_loads,
            received_loads,
            input_splits,
            output_splits,
            sum(matrix[ep_rank]),
        )
        self._resolved_moe_plans[module_name] = {
            "source_expert_loads": [list(row) for row in plan.source_expert_loads],
            "local_expert_loads": list(plan.local_expert_loads),
            "received_expert_loads": list(plan.received_expert_loads),
            "input_split_sizes": list(plan.input_split_sizes),
            "output_split_sizes": list(plan.output_split_sizes),
            "outgoing_tokens": plan.outgoing_tokens,
        }
        return plan

    @staticmethod
    def _explicit_moe_matrix(
            source_loads: Any, ep_size: int, num_experts: int,
            expected: int, module_name: str,
    ) -> tuple[tuple[int, ...], ...]:
        """Validate the configured aggregate EP traffic matrix."""
        if not isinstance(source_loads, list) or len(source_loads) != ep_size:
            raise ValueError(f"MoE explicit source_expert_loads for {module_name!r} must have {ep_size} rows")
        matrix = []
        for source_rank, row in enumerate(source_loads):
            if not isinstance(row, list) or len(row) != num_experts:
                raise ValueError(
                    f"MoE explicit source_expert_loads row {source_rank} for {module_name!r} "
                    f"must have {num_experts} entries"
                )
            if any(
                    not isinstance(count, int) or isinstance(count, bool) or count < 0
                    for count in row
            ):
                raise ValueError("MoE explicit source_expert_loads entries must be non-negative integers")
            if sum(row) != expected:
                raise ValueError(
                    f"MoE explicit source_expert_loads row {source_rank} for {module_name!r} must sum to "
                    f"local_tokens * top_k ({expected}), got {sum(row)}"
                )
            matrix.append(tuple(row))
        return tuple(matrix)

    def tp_cross_entropy_target_count(self, target_count: int, tp_size: int, tp_rank: int) -> tuple[int, int]:
        """Return CPU-derived valid and local-owned target counts for TP CE."""
        counts = self._tp_cross_entropy_counts
        if counts is None:
            raise ValueError("TP cross-entropy counts were not derived from the training batch")
        valid, owned = counts
        if len(owned) != tp_size or not 0 <= tp_rank < tp_size:
            raise ValueError(
                f"CPU-derived TP target counts contain {len(owned)} ranks, expected {tp_size}"
            )
        if valid > target_count:
            raise ValueError(
                f"CPU-derived valid token count {valid} exceeds FakeTensor target size {target_count}"
            )
        return valid, owned[tp_rank]

    def configure_tp_cross_entropy_counts(
            self,
            valid_token_count: int,
            target_tokens_per_rank: tuple[int, ...],
    ) -> None:
        """Install token ownership computed from the real CPU training batch."""
        if valid_token_count < 0 or sum(target_tokens_per_rank) != valid_token_count:
            raise ValueError("CPU-derived TP target counts must be non-negative and sum to the valid count")
        self._tp_cross_entropy_counts = (valid_token_count, target_tokens_per_rank)


_T = TypeVar("_T")
_ACTIVE_VALUE_DEPENDENCY_MANAGER: ContextVar[Optional["ValueDependencyManager"]] = ContextVar(
    "active_value_dependency_manager", default=None,
)
_VALUE_DEPENDENCY_TARGET_STACK: ContextVar[tuple[str, ...]] = ContextVar(
    "value_dependency_target_stack", default=(),
)


def value_dependency_decision(name: str, default_factory: Callable[[], _T]) -> _T:
    """Resolve one instrumented Python branch decision.

    Args:
        name: Decision name configured by a ``branch`` handler rule.
        default_factory: Lazy real-execution computation used when no matching
            dry-run decision is active.

    Returns:
        The configured dry-run decision or the lazily computed real value.

    Raises:
        ValueError: If ``name`` is empty or ``default_factory`` is not callable.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("value dependency decision name must be a non-empty string")
    if not callable(default_factory):
        raise ValueError("value dependency decision default_factory must be callable")
    manager = _ACTIVE_VALUE_DEPENDENCY_MANAGER.get()
    if manager is not None:
        configured, value = manager.resolve_decision(name)
        if configured:
            return value
    return default_factory()


class UnconfiguredValueDependencyError(RuntimeError):
    """Actionable error for an unconfigured FakeTensor value dependency."""


class ValueDependencyManager(AbstractContextManager):
    """Install unified semantic, branch, and operator-debug dry-run behavior."""

    _BUILTIN_RUNTIME_HANDLERS = {"moe_routing", "branch", "operator_debug"}
    _SEMANTIC_HANDLERS = {"moe_routing"}

    def __init__(self, profile: _DryRunValueProfile, model: Any, runtime: DryRunRuntime) -> None:
        """Bind runtime tracking to one expanded value-dependency profile."""
        self._profile = profile
        self._model = model
        self._runtime = runtime
        self._base = None
        self._stack = ExitStack()
        self._hook_handles = []
        self._active_token = None
        self._consumed_decisions = set()
        self._operator_counts: Dict[tuple[Any, ...], int] = {}
        self._consumed_operator_mocks = set()
        self._module_by_name = dict(model.named_modules())

    @contextmanager
    def parallelize_context(self) -> Iterator[None]:
        """Install value dependencies required while parallelization is applied."""
        with ExitStack() as stack:
            self._install_handler_contexts(
                stack,
                ValueDependencyStage.PARALLELIZE,
                SimpleNamespace(profile=self._profile, model=self._model, runtime=self._runtime),
            )
            yield

    def fake_step_context(self, base: Any) -> "ValueDependencyManager":
        """Prepare this manager to enter the complete fake-step scope."""
        self._base = base
        return self

    def __enter__(self) -> "ValueDependencyManager":
        """Install semantic overrides, module scopes, and dispatch interception."""
        if self._base is None:
            raise RuntimeError("ValueDependencyManager.fake_step_context must be configured before entry")
        self._install_handler_contexts(
            self._stack,
            ValueDependencyStage.FAKE_STEP,
            SimpleNamespace(profile=self._profile, base=self._base, runtime=self._runtime),
        )
        mesh = getattr(self._base, "mesh", None)
        if bool(getattr(mesh, "loss_parallel", False)):
            self._stack.enter_context(_TPCrossEntropyValueDependencyRuntime(self._profile))
        self._install_module_hooks()
        self._active_token = _ACTIVE_VALUE_DEPENDENCY_MANAGER.set(self)
        self._stack.enter_context(self._build_operator_debug_mode())
        return self

    def _install_handler_contexts(
            self, stack: ExitStack, stage: ValueDependencyStage, context: Any,
    ) -> None:
        """Install semantic handlers once and external handlers per expanded target."""
        for handler_name, rules in self._profile._resolved.items():
            if handler_name in ("branch", "operator_debug") or not rules:
                continue
            handler = self._profile.handler(handler_name)
            selected_rules = [next(iter(rules.items()))] if handler_name in self._SEMANTIC_HANDLERS else rules.items()
            for target, rule in selected_rules:
                stack.enter_context(handler.install(stage, target, rule.path, rule.inputs, context))

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        """Restore every scope and reject unused configured decisions/mocks."""
        del traceback
        try:
            if exc_type is None:
                self._validate_consumption()
        finally:
            self._profile.set_runtime_metadata({
                "branch": {
                    "consumed_decisions": sorted(
                        f"{target}:{name}" for target, name in self._consumed_decisions
                    ),
                },
                "operator_debug": {
                    "unsafe_operator_mock": bool(self._profile.resolved_rules("operator_debug")),
                    "consumed_mock_count": len(self._consumed_operator_mocks),
                },
            })
            if self._active_token is not None:
                _ACTIVE_VALUE_DEPENDENCY_MANAGER.reset(self._active_token)
                self._active_token = None
            while self._hook_handles:
                self._hook_handles.pop().remove()
            self._stack.close()
        return False

    @contextmanager
    def logical_scope(self, target: str) -> Iterator[None]:
        """Activate one non-module execution target such as loss or backward."""
        stack = _VALUE_DEPENDENCY_TARGET_STACK.get()
        token = _VALUE_DEPENDENCY_TARGET_STACK.set((*stack, target))
        self._clear_target_operator_counts(target)
        try:
            yield
        finally:
            _VALUE_DEPENDENCY_TARGET_STACK.reset(token)

    def prepare_batch(self, batch: Dict[str, Any], context: Any) -> None:
        """Run configured handler batch preparation hooks."""
        for handler_name, rules in self._profile._resolved.items():
            handler = self._profile.handler(handler_name)
            for target, rule in rules.items():
                handler.prepare_batch(target, rule.path, rule.inputs, batch, context)

    def resolve_decision(self, name: str) -> tuple[bool, Any]:
        """Resolve a configured branch decision for the active innermost target."""
        target = self._current_target()
        if target is None:
            return False, None
        rule = self._profile.resolved_rules("branch").get(target)
        if rule is None or name not in rule.inputs["decisions"]:
            return False, None
        self._consumed_decisions.add((target, name))
        return True, rule.inputs["decisions"][name]

    def _install_module_hooks(self) -> None:
        """Track the innermost executing module with exception-safe hooks."""
        for module_name, module in self._model.named_modules():
            if not module_name:
                continue

            def _pre_hook(current_module: Any, args: Any, name: str = module_name) -> None:
                del current_module, args
                stack = _VALUE_DEPENDENCY_TARGET_STACK.get()
                _VALUE_DEPENDENCY_TARGET_STACK.set((*stack, name))
                self._clear_target_operator_counts(name)

            def _post_hook(
                    current_module: Any, args: Any, output: Any,
                    name: str = module_name,
            ) -> None:
                del current_module, args, output
                stack = _VALUE_DEPENDENCY_TARGET_STACK.get()
                if stack and stack[-1] == name:
                    _VALUE_DEPENDENCY_TARGET_STACK.set(stack[:-1])

            self._hook_handles.append(module.register_forward_pre_hook(_pre_hook))
            try:
                handle = module.register_forward_hook(_post_hook, always_call=True)
            except TypeError:
                handle = module.register_forward_hook(_post_hook)
            self._hook_handles.append(handle)

    def _clear_target_operator_counts(self, target: str) -> None:
        """Reset occurrence counters for one new target invocation."""
        self._operator_counts = {
            key: value for key, value in self._operator_counts.items() if key[0] != target
        }

    @staticmethod
    def _current_target() -> Optional[str]:
        """Return the active innermost module or logical region."""
        stack = _VALUE_DEPENDENCY_TARGET_STACK.get()
        return stack[-1] if stack else None

    @staticmethod
    def _source_anchor() -> Dict[str, Any]:
        """Resolve the first user-code frame for the current ATen dispatch."""
        this_file = os.path.abspath(__file__)
        for frame_info in inspect.stack()[2:]:
            filename = os.path.abspath(frame_info.filename)
            if filename == this_file or f"{os.sep}site-packages{os.sep}torch{os.sep}" in filename:
                continue
            owner = frame_info.frame.f_locals.get("self")
            function = frame_info.function
            if owner is not None:
                function = f"{type(owner).__qualname__}.{function}"
            return {"file": frame_info.filename, "function": function, "line": frame_info.lineno}
        return {"file": "<unknown>", "function": "<unknown>", "line": 0}

    @staticmethod
    def _source_matches(configured: Dict[str, Any], actual: Dict[str, Any]) -> bool:
        """Return whether an actual frame matches a strict configured anchor."""
        return (
            actual["file"].endswith(configured["file"])
            and actual["function"] == configured["function"]
            and actual["line"] == configured["line"]
        )

    def _build_operator_debug_mode(self) -> ContextManager[Any]:
        """Create a Torch dispatch mode bound to this manager."""
        from torch.utils._python_dispatch import TorchDispatchMode  # pylint: disable=C0415

        manager = self

        class _OperatorDebugMode(TorchDispatchMode):
            def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
                del types
                return manager._operator_dispatch(func, args, kwargs or {})

        return _OperatorDebugMode()

    def _operator_dispatch(self, func: Any, args: Any, kwargs: Dict[str, Any]) -> Any:
        """Apply one exact source-anchored mock or enrich FakeTensor failures."""
        from torch._subclasses.fake_tensor import (  # pylint: disable=C0415
            DataDependentOutputException,
            DynamicOutputShapeException,
        )

        target = self._current_target()
        op_name = str(func)
        rule = self._profile.resolved_rules("operator_debug").get(target or "")
        if rule is not None:
            matching_op_mocks = [mock for mock in rule.inputs["mocks"] if mock["op"] == op_name]
            if matching_op_mocks:
                source = self._source_anchor()
                anchored = [
                    mock for mock in matching_op_mocks
                    if self._source_matches(mock["source"], source)
                ]
                if anchored:
                    source_key = (source["file"], source["function"], source["line"])
                    count_key = (target, *source_key, op_name)
                    occurrence = self._operator_counts.get(count_key, 0)
                    self._operator_counts[count_key] = occurrence + 1
                    selected = [mock for mock in anchored if mock["occurrence"] == occurrence]
                    if len(selected) != 1:
                        raise ValueError(
                            f"operator_debug target {target!r} op {op_name} source {source} "
                            f"has no unique mock for occurrence {occurrence}"
                        )
                    mock = selected[0]
                    self._consumed_operator_mocks.add((target, id(mock)))
                    return self._build_operator_return(mock["return"], args, kwargs)
        try:
            return func(*args, **kwargs)
        except (DataDependentOutputException, DynamicOutputShapeException) as error:
            source = self._source_anchor()
            count_key = (target, source["file"], source["function"], source["line"], op_name)
            occurrence = self._operator_counts.get(count_key, 0)
            self._operator_counts[count_key] = occurrence + 1
            raise self._unconfigured_error(target, op_name, args, source, occurrence) from error

    def _build_operator_return(self, spec: Dict[str, Any], args: Any, kwargs: Dict[str, Any]) -> Any:
        """Construct one recursive operator-debug return value."""
        import torch  # pylint: disable=C0415

        kind, value = next(iter(spec.items()))
        if kind == "scalar":
            return value
        if kind == "by_global_rank":
            rank_key = str(self._runtime.rank)
            selected = value.get(self._runtime.rank, value.get(rank_key))
            if selected is None:
                raise ValueError(
                    f"operator_debug return has no by_global_rank entry for rank {self._runtime.rank}"
                )
            return self._build_operator_return(selected, args, kwargs)
        if kind in ("tuple", "list"):
            built = [self._build_operator_return(item, args, kwargs) for item in value]
            return tuple(built) if kind == "tuple" else built
        dtype = getattr(torch, value["dtype"], None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"operator_debug uses unknown torch dtype {value['dtype']!r}")
        device = self._first_tensor_device((args, kwargs))
        return torch.empty(tuple(value["shape"]), dtype=dtype, device=device)

    @staticmethod
    def _first_tensor_device(value: Any) -> Any:
        """Return the first tensor device in a nested operator argument tree."""
        import torch  # pylint: disable=C0415

        if isinstance(value, torch.Tensor):
            return value.device
        if isinstance(value, dict):
            values = value.values()
        elif isinstance(value, (tuple, list)):
            values = value
        else:
            return torch.device("cpu")
        for item in values:
            device = ValueDependencyManager._first_tensor_device(item)
            if device.type != "cpu" or isinstance(item, torch.Tensor):
                return device
        return torch.device("cpu")

    def _unconfigured_error(
            self, target: Optional[str], op_name: str, args: Any,
            source: Dict[str, Any], occurrence: int,
    ) -> Exception:
        """Build an actionable failure with semantic handler suggestions."""
        module = self._module_by_name.get(target or "")
        available = []
        if target is not None:
            for name, handler in self._profile._handlers.items():
                if name not in ("branch", "operator_debug") and handler.recognizes(target, module):
                    available.append(f"  - {name}: {', '.join(handler.paths)}")
        tensor_metadata = []
        for arg in args:
            if hasattr(arg, "shape") and hasattr(arg, "dtype"):
                tensor_metadata.append(f"shape={tuple(arg.shape)}, dtype={arg.dtype}")
        lines = [
            "Unconfigured value dependency detected",
            f"Target: {target or '<outside configured scope>'}",
            f"Operator: {op_name}",
            f"Occurrence: {occurrence}",
            f"Source: {source['file']}:{source['line']} ({source['function']})",
            f"Rank: {self._runtime.rank}",
            f"Inputs: {tensor_metadata}",
        ]
        if available:
            lines.extend(["Available handlers:", *available, "Add a dry_run.value_dependencies rule for this target."])
        else:
            lines.extend([
                "No semantic handler recognizes this dependency.",
                "Register a ValueDependencyHandler, instrument a branch decision, "
                "or use source-anchored operator_debug.",
            ])
        return UnconfiguredValueDependencyError("\n".join(lines))

    def _validate_consumption(self) -> None:
        """Fail closed when configured branch decisions or operator mocks were unused."""
        unused_decisions = []
        for target, rule in self._profile.resolved_rules("branch").items():
            for name in rule.inputs["decisions"]:
                if (target, name) not in self._consumed_decisions:
                    unused_decisions.append(f"{target}:{name}")
        unused_mocks = []
        for target, rule in self._profile.resolved_rules("operator_debug").items():
            for index, mock in enumerate(rule.inputs["mocks"]):
                if (target, id(mock)) not in self._consumed_operator_mocks:
                    unused_mocks.append(f"{target}:mock[{index}]")
        if unused_decisions or unused_mocks:
            raise ValueError(
                "unused value-dependency configuration: "
                f"decisions={unused_decisions}, operator_mocks={unused_mocks}"
            )


class _TPCrossEntropyValueDependencyRuntime(AbstractContextManager):
    """Scoped TP cross-entropy replay owned by its semantic handler."""

    def __init__(self, profile: _DryRunValueProfile) -> None:
        """Store the resolved ownership profile."""
        self._profile = profile
        self._loss_parallel_ops = None
        self._original_cross_entropy_function = None

    def __enter__(self) -> "_TPCrossEntropyValueDependencyRuntime":
        """Patch the complete CE autograd Function for this dry-run scope."""
        from hyper_parallel.platform.torch import loss_parallel_ops  # pylint: disable=C0415
        import torch  # pylint: disable=C0415

        profile = self._profile

        class DryRunDistributedCrossEntropyFunction(torch.autograd.Function):
            """CE Function that preserves tensor paths without inspecting labels."""

            @staticmethod
            def forward(ctx: Any, input_local: Any, target: Any, weight: Any, ignore_index: int,
                        reduction: str, vocab_size: int, mesh: Any, mesh_dim: int) -> Any:
                """Run distributed log-softmax and allocate configured local NLL state."""
                del weight, ignore_index, vocab_size
                tp_size = int(mesh.size(mesh_dim))
                tp_rank = int(mesh.get_local_rank(mesh_dim))
                valid_count, local_count = profile.tp_cross_entropy_target_count(
                    int(target.numel()), tp_size, tp_rank,
                )
                log_probs = loss_parallel_ops.distributed_log_softmax(
                    input_local, dim=-1, mesh=mesh, mesh_dim=mesh_dim,
                )
                selected = torch.empty((local_count,), dtype=log_probs.dtype, device=log_probs.device)
                anchor = log_probs.sum() * 0.0 + selected.sum() * 0.0
                ctx.save_for_backward(log_probs)
                ctx.reduction = reduction
                ctx.valid_count = valid_count
                ctx.local_count = local_count
                if reduction == "none":
                    return torch.empty_like(target, dtype=log_probs.dtype) + anchor
                total = loss_parallel_ops.platform.differentiable_all_reduce(
                    anchor, op="sum", group=mesh.get_group(mesh_dim),
                )
                return total / max(valid_count, 1) if reduction == "mean" else total

            @staticmethod
            def backward(ctx: Any, grad_output: Any) -> tuple[Any, ...]:
                """Retain local softmax-gradient allocations without label indexing."""
                (log_probs,) = ctx.saved_tensors
                if ctx.reduction == "none":
                    grad_scale = grad_output.reshape(-1, 1)
                elif ctx.reduction == "mean":
                    grad_scale = grad_output / max(ctx.valid_count, 1)
                else:
                    grad_scale = grad_output
                selected_grad = torch.empty(
                    (ctx.local_count,), dtype=log_probs.dtype, device=log_probs.device,
                )
                grad_input = log_probs.exp() * grad_scale + selected_grad.sum() * 0.0
                return grad_input, None, None, None, None, None, None, None

        self._loss_parallel_ops = loss_parallel_ops
        self._original_cross_entropy_function = loss_parallel_ops.DistributedCrossEntropyFunction
        loss_parallel_ops.DistributedCrossEntropyFunction = DryRunDistributedCrossEntropyFunction
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        """Restore the real TP cross-entropy autograd Function."""
        del exc_type, exc_value, traceback
        if self._loss_parallel_ops is not None:
            self._loss_parallel_ops.DistributedCrossEntropyFunction = self._original_cross_entropy_function
            self._loss_parallel_ops = None
            self._original_cross_entropy_function = None
        return False


class _MoERoutingValueDependencyRuntimeBase:
    """Scoped routed-expert replay used by the MoE semantic handler."""

    def __init__(self, profile: _DryRunValueProfile, base: Any) -> None:
        """Create a scope for one prepared trainer."""
        self._profile = profile
        self._base = base
        self._patched_moe_modules = []
        self._moe_module_names: Dict[int, str] = {}
        self._ep_compute_module = None
        self._original_ep_compute = None

    def __enter__(self) -> "_MoERoutingValueDependencyRuntime":
        """Install configured MoE replay."""
        try:
            if self._profile.moe_enabled:
                self._patch_moe()
        except Exception:
            self._restore_moe()
            raise
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        """Restore every patched symbol even if the simulated step failed."""
        del exc_type, exc_value, traceback
        self._restore_moe()
        return False


class _MoERoutingValueDependencyRuntime(_MoERoutingValueDependencyRuntimeBase):
    """Complete shape and memory replay for configured routed-expert modules."""

    def _patch_moe(self) -> None:
        """Patch supported MoE execution modules without changing their classes."""
        module_names = set()
        configured_names = set(self._profile.moe_layers)
        for module_name, module in self._base.model.named_modules():
            if module_name in configured_names and self._is_supported_moe_module(module):
                module_names.add(module_name)
                self._moe_module_names[id(module)] = module_name
                if self._base.mesh.ep_size > 1:
                    continue
                had_forward = "forward" in module.__dict__
                original_forward = module.__dict__.get("forward")

                def dry_run_forward(current_module: Any, x: Any, name: str = module_name) -> Any:
                    """Execute the configured value-independent MoE path."""
                    return self._run_moe_forward(name, current_module, x)

                module.forward = MethodType(dry_run_forward, module)
                self._patched_moe_modules.append((module, had_forward, original_forward))
        if self._base.mesh.ep_size > 1 and module_names:
            from hyper_parallel.distributed.expert_parallel import recipes as ep_compute  # pylint: disable=C0415

            self._ep_compute_module = ep_compute
            self._original_ep_compute = ep_compute.ep_routed_forward

            def dry_run_ep_compute(
                    module: Any,
                    hidden_states: Any,
                    *,
                    router_fn: Any,
                    ep_group: Any,
            ) -> Any:
                """Execute configured EP shapes inside the existing local region."""
                return self._run_hf_native_ep_compute(
                    module,
                    hidden_states,
                    router_fn=router_fn,
                    ep_group=ep_group,
                    tp_group=None,
                )

            ep_compute.ep_routed_forward = dry_run_ep_compute
        unknown_names = configured_names - module_names
        if unknown_names:
            self._restore_moe()
            raise ValueError(
                "moe_routing rules contain unsupported module FQNs: "
                f"{sorted(unknown_names)}; supported MoE module FQNs discovered "
                f"in the parallelized model: {sorted(module_names)}"
            )

    @staticmethod
    def _is_supported_moe_module(module: Any) -> bool:
        """Return whether a module exposes one of the supported expert layouts."""
        return _is_supported_moe_module(module)

    @staticmethod
    def _moe_dimensions(module: Any) -> tuple[int, int]:
        """Return global expert count and top-k for a supported MoE module."""
        experts = module.experts
        config = getattr(module, "config", None)
        num_experts = (
            getattr(experts, "num_experts", None)
            or getattr(module, "num_experts", None)
            or getattr(config, "num_experts", None)
            or getattr(config, "n_routed_experts", None)
        )
        top_k = (
            getattr(module, "top_k", None)
            or getattr(getattr(module, "gate", None), "top_k", None)
            or getattr(config, "num_experts_per_tok", None)
            or getattr(config, "top_k", None)
        )
        if num_experts is None or top_k is None:
            raise ValueError(
                f"Cannot resolve expert count/top-k for {type(module).__name__}"
            )
        return int(num_experts), int(top_k)

    @staticmethod
    def _local_weight(weight: Any) -> Any:
        """Return a local expert-weight shard while preserving FakeTensor state."""
        return weight.to_local() if hasattr(weight, "to_local") else weight

    def _expert_weight_layout(self, module: Any) -> tuple[str, tuple[Any, ...]]:
        """Extract local expert weights from a supported MoE module.

        The framework ``MoE`` stores independent SwiGLU matrices ``w1/w2/w3``.
        Qwen3.5 packs the gate and up matrices together in ``gate_up_proj``.
        Both layouts retain the EP-sharded leading expert dimension.
        """
        experts = module.experts
        if all(hasattr(experts, name) for name in ("w1", "w2", "w3")):
            return "three_projection", tuple(
                self._local_weight(getattr(experts, name)) for name in ("w1", "w2", "w3")
            )
        if all(hasattr(experts, name) for name in ("gate_up_proj", "down_proj")):
            return "packed_gate_up", (
                self._local_weight(experts.gate_up_proj),
                self._local_weight(experts.down_proj),
            )
        if all(
                hasattr(experts, name)
                for name in ("gate_proj", "up_proj", "down_proj")
        ):
            return "three_projection", (
                self._local_weight(experts.gate_proj),
                self._local_weight(experts.down_proj),
                self._local_weight(experts.up_proj),
            )
        raise ValueError(
            f"MoE module {type(module).__name__!r} does not expose a supported expert-weight layout"
        )

    def _run_moe_forward(self, module_name: str, module: Any, x: Any) -> Any:
        """Run a value-independent analogue of the real routed MoE path."""
        import torch  # pylint: disable=C0415

        batch_size, sequence_length, hidden_size = x.shape
        ep_size, ep_rank = self._ep_group_identity()
        layout, weights = self._expert_weight_layout(module)
        num_experts, top_k = self._moe_dimensions(module)
        routing_module = SimpleNamespace(num_experts=num_experts, top_k=top_k)
        plan = self._profile.moe_routing_plan(
            module_name, routing_module, ep_size, ep_rank,
            int(batch_size * sequence_length), int(weights[0].shape[0]),
        )
        x_flat = x.view(-1, hidden_size)
        shared_output = None
        if getattr(module, "shared_expert", None) is not None:
            shared_output = module.shared_expert(x_flat)
        routed_input, top_weights, permutation, inverse_permutation, token_counts = self._dry_run_route(
            module_name, module, x_flat, plan,
        )
        expert_input, dispatch_context = self._dry_run_dispatch(
            routed_input, token_counts, plan, ep_size, int(weights[0].shape[0]),
        )
        expert_output = self._run_grouped_experts(
            layout, weights, expert_input, plan.local_expert_loads,
        )
        combined = self._dry_run_combine(expert_output, dispatch_context)
        routed_output = self._dry_run_weight_and_unpermute(
            module, x_flat, combined, top_weights, permutation, inverse_permutation,
        )
        output = routed_output
        if shared_output is not None:
            shared_gate = torch.sigmoid(module.shared_expert_gate(x_flat))
            output = output + shared_gate * shared_output
        if hasattr(module, "last_aux_loss"):
            module.last_aux_loss = None
        return output.view(batch_size, sequence_length, hidden_size)

    def _run_hf_native_ep_compute(
            self,
            module: Any,
            hidden_states: Any,
            *,
            router_fn: Any,
            ep_group: Any,
            tp_group: Any,
    ) -> Any:
        """Replay the new Trainer's EP lifecycle with configured split sizes."""
        import torch  # pylint: disable=C0415
        import torch.distributed as dist  # pylint: disable=C0415

        module_name = self._moe_module_names.get(id(module))
        if module_name is None:
            raise ValueError(
                f"Dry-run EP compute received an unregistered {type(module).__name__}"
            )
        ep_size = int(ep_group.size())
        ep_rank = int(dist.get_rank(group=ep_group))
        num_experts, top_k = self._moe_dimensions(module)
        local_expert_count = int(module.experts.local_expert_count)
        batch_size, sequence_length, hidden_size = hidden_states.shape
        local_tokens = int(batch_size * sequence_length)
        routing_module = SimpleNamespace(num_experts=num_experts, top_k=top_k)
        plan = self._profile.moe_routing_plan(
            module_name,
            routing_module,
            ep_size,
            ep_rank,
            local_tokens,
            local_expert_count,
        )

        flattened_states = hidden_states.reshape(-1, hidden_size)
        topk_indices, topk_weights = router_fn(module, hidden_states)
        flattened_weights = topk_weights.reshape(-1).to(flattened_states.dtype)
        source_indices = torch.arange(
            local_tokens,
            device=flattened_states.device,
        ).repeat_interleave(top_k)
        dispatch_order = topk_indices.reshape(-1).argsort()
        dispatched_states = flattened_states[source_indices[dispatch_order]].contiguous()
        received_states = self._differentiable_all_to_all(
            dispatched_states,
            plan.input_split_sizes,
            plan.output_split_sizes,
        )
        received_expert_indices = torch.empty(
            (sum(plan.output_split_sizes),),
            dtype=torch.int64,
            device=hidden_states.device,
        )

        experts = module.experts
        had_forward = "forward" in experts.__dict__
        original_forward = experts.__dict__.get("forward")

        def grouped_forward(
                current_experts: Any,
                dispatched: Any,
                local_indices: Any,
        ) -> Any:
            """Execute grouped experts with the simulated token distribution.

            Args:
                current_experts: Local expert module collection.
                dispatched: Tokens received by the local experts.
                local_indices: Runtime expert indices, unused by the simulation.

            Returns:
                The grouped expert outputs.
            """
            del local_indices
            layout, weights = self._expert_weight_layout(
                SimpleNamespace(experts=current_experts)
            )
            return self._run_grouped_experts(
                layout,
                weights,
                dispatched,
                plan.local_expert_loads,
            )

        experts.forward = MethodType(grouped_forward, experts)
        try:
            local_outputs = experts(received_states, received_expert_indices)
        finally:
            if had_forward:
                experts.forward = original_forward
            else:
                del experts.forward

        combined = self._differentiable_all_to_all(
            local_outputs.contiguous(),
            plan.output_split_sizes,
            plan.input_split_sizes,
        )
        flattened_outputs = torch.zeros_like(combined)
        flattened_outputs[dispatch_order] = combined
        output = (
            flattened_outputs * flattened_weights.unsqueeze(-1)
        ).view(local_tokens, top_k, hidden_size).sum(dim=1).view(
            batch_size,
            sequence_length,
            hidden_size,
        )
        shared = getattr(module, "shared_experts", None)
        if shared is not None:
            if tp_group is None:
                raise ValueError("MoE shared_experts requires a TP group")
            shared_output = shared(hidden_states)
            dist.all_reduce(shared_output, group=tp_group)
            output = output + shared_output
        return output

    @staticmethod
    def _dry_run_route(
            module_name: str, module: Any, x_flat: Any,
            plan: _DryRunMoERoutingPlan,
    ) -> tuple[Any, Any, Any, Any, Any]:
        """Replay value-independent router, sort, and histogram operators."""
        import torch  # pylint: disable=C0415
        from torch.nn import functional  # pylint: disable=C0415

        gate = getattr(module, "gate", None)
        gate_weight = getattr(gate, "weight", None)
        if gate_weight is None:
            raise ValueError(
                f"MoE module {module_name!r} must expose gate.weight for configured dry-run routing"
            )
        router_logits = x_flat @ gate_weight.transpose(0, 1)
        router_probs = functional.softmax(router_logits, dim=-1, dtype=torch.float32)
        top_weights, top_indices = torch.topk(
            router_probs, int(module.top_k), dim=-1,
        )
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        top_weights = top_weights.to(x_flat.dtype)
        module.router_logits = router_logits

        token_indices = torch.arange(
            x_flat.shape[0], device=x_flat.device,
        ).unsqueeze(1).expand(-1, int(module.top_k)).reshape(-1)
        expert_indices = top_indices.reshape(-1)
        permutation = torch.argsort(expert_indices, stable=True)
        inverse_permutation = torch.empty_like(permutation)
        inverse_permutation[permutation] = torch.arange(
            plan.outgoing_tokens, device=x_flat.device,
        )
        routed_input = x_flat[token_indices[permutation]]
        histc_input = expert_indices.float() if x_flat.device.type == "cpu" else expert_indices.int()
        token_counts = torch.histc(
            histc_input,
            bins=int(module.num_experts),
            min=0,
            max=int(module.num_experts) - 1,
        ).to(torch.int64)
        return routed_input, top_weights, permutation, inverse_permutation, token_counts

    def _dry_run_dispatch(
            self, routed_input: Any, token_counts: Any,
            plan: _DryRunMoERoutingPlan,
            ep_size: int, local_expert_count: int,
    ) -> tuple[Any, _DryRunMoEDispatchContext]:
        """Replay counts exchange, token dispatch, and expert permutation."""
        counts_output = self._dry_run_exchange_counts(
            token_counts, ep_size,
        )
        dispatched = self._differentiable_all_to_all(
            routed_input,
            plan.input_split_sizes,
            plan.output_split_sizes,
        )
        rank_major_shape = tuple(dispatched.shape)
        permutation = self._dry_run_rank_to_expert_indices(
            counts_output,
            sum(plan.output_split_sizes),
            ep_size,
            local_expert_count,
        )
        expert_input = dispatched[permutation]
        context = _DryRunMoEDispatchContext(
            rank_major_shape=rank_major_shape,
            permuted_indices=permutation,
            input_split_sizes=plan.input_split_sizes,
            output_split_sizes=plan.output_split_sizes,
        )
        return expert_input, context

    def _dry_run_combine(
            self, expert_output: Any,
            context: _DryRunMoEDispatchContext,
    ) -> Any:
        """Replay expert unpermutation and reverse token all-to-all."""
        rank_major_output = expert_output.new_zeros(*context.rank_major_shape)
        rank_major_output[context.permuted_indices] = expert_output
        return self._differentiable_all_to_all(
            rank_major_output,
            context.output_split_sizes,
            context.input_split_sizes,
        )

    @staticmethod
    def _dry_run_weight_and_unpermute(
            module: Any, x_flat: Any, combined: Any, top_weights: Any,
            permutation: Any, inverse_permutation: Any,
    ) -> Any:
        """Apply routing weights and restore the original token order."""
        import torch  # pylint: disable=C0415

        sorted_weights = top_weights.reshape(-1)[permutation]
        use_fp32_combine = (
            getattr(module, "_hp_moe_tp_enabled", False)
            or getattr(module, "_hp_moe_ep_fp32_routing", False)
        )
        if use_fp32_combine:
            weighted = (
                combined.to(torch.float32) * sorted_weights.to(torch.float32).unsqueeze(-1)
            ).to(combined.dtype)
            unsorted = weighted[inverse_permutation]
            return unsorted.view(
                x_flat.shape[0], int(module.top_k), x_flat.shape[-1],
            ).sum(dim=1, dtype=torch.float32).to(x_flat.dtype)
        weighted = combined * sorted_weights.unsqueeze(-1)
        unsorted = weighted[inverse_permutation]
        return unsorted.view(
            x_flat.shape[0], int(module.top_k), x_flat.shape[-1],
        ).sum(dim=1).to(x_flat.dtype)

    def _dry_run_exchange_counts(
            self, counts_input: Any, ep_size: int,
    ) -> Any:
        """Replay the non-differentiable count all-to-all at a fixed shape."""
        from hyper_parallel.platform import get_platform  # pylint: disable=C0415

        if ep_size == 1:
            return counts_input.clone()
        try:
            ep_group = self._ep_mesh().get_group()
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError("Configured MoE dry-run requires an accessible EP process group") from error
        counts_output, handle = get_platform().all_to_all_single(
            counts_input,
            output_shape=[counts_input.shape[0]],
            group=ep_group,
            async_op=True,
        )
        if handle is not None:
            handle.wait()
        return counts_output

    @staticmethod
    def _dry_run_rank_to_expert_indices(
            counts: Any, total_tokens: int,
            ep_size: int, local_expert_count: int,
    ) -> Any:
        """Replay rank-major to expert-major index construction at fixed shape."""
        import torch  # pylint: disable=C0415

        counts_2d = counts.view(ep_size, local_expert_count)
        source_offsets = counts.cumsum(0) - counts
        expert_major_offsets = source_offsets.view(
            ep_size, local_expert_count,
        ).transpose(0, 1).contiguous().view(-1)
        expert_major_counts = counts_2d.transpose(0, 1).contiguous().view(-1)
        block_source_starts = torch.repeat_interleave(
            expert_major_offsets,
            expert_major_counts,
            output_size=total_tokens,
        )
        destination_offsets = expert_major_counts.cumsum(0) - expert_major_counts
        destination_starts = torch.repeat_interleave(
            destination_offsets,
            expert_major_counts,
            output_size=total_tokens,
        )
        intra_block_offsets = torch.arange(
            total_tokens, device=counts.device,
        ) - destination_starts
        return (block_source_starts + intra_block_offsets).long()

    @staticmethod
    def _dry_run_grouped_matmul(
            input_tensor: Any, weight: Any,
            expert_loads: tuple[int, ...],
    ) -> Any:
        """Replay grouped matmul events with one output and one weight gradient."""
        import torch  # pylint: disable=C0415

        class _DryRunGroupedMatmul(torch.autograd.Function):
            """Grouped matmul analogue with preallocated forward/backward buffers."""

            @staticmethod
            def forward(
                    ctx: Any, value: Any, grouped_weight: Any,
                    loads: tuple[int, ...],
            ) -> Any:
                """Compute packed expert outputs into one logical allocation."""
                ctx.save_for_backward(value, grouped_weight)
                ctx.expert_loads = loads
                output = torch.zeros(
                    value.shape[0], grouped_weight.shape[-1],
                    dtype=value.dtype, device=value.device,
                )
                start = 0
                for expert_index, load in enumerate(loads):
                    end = start + load
                    if load:
                        torch.mm(
                            value[start:end], grouped_weight[expert_index],
                            out=output[start:end],
                        )
                    start = end
                return output

            @staticmethod
            def backward(ctx: Any, grad_output: Any) -> tuple[Any, Any, None]:
                """Compute packed input and expert-weight gradients."""
                value, grouped_weight = ctx.saved_tensors
                grad_input = torch.zeros_like(value)
                grad_weight = torch.zeros_like(grouped_weight)
                start = 0
                for expert_index, load in enumerate(ctx.expert_loads):
                    end = start + load
                    if load:
                        torch.mm(
                            grad_output[start:end], grouped_weight[expert_index].transpose(0, 1),
                            out=grad_input[start:end],
                        )
                        torch.mm(
                            value[start:end].transpose(0, 1), grad_output[start:end],
                            out=grad_weight[expert_index],
                        )
                    start = end
                return grad_input, grad_weight, None

        if len(expert_loads) != int(weight.shape[0]):
            raise ValueError(
                "MoE dry-run expert load count must match the local expert weight count"
            )
        if sum(expert_loads) != int(input_tensor.shape[0]):
            raise ValueError(
                "MoE dry-run local expert loads must sum to the dispatched token count"
            )
        return _DryRunGroupedMatmul.apply(input_tensor, weight, expert_loads)

    def _run_grouped_experts(
            self, layout: str, weights: tuple[Any, ...], expert_input: Any,
            expert_loads: tuple[int, ...],
    ) -> Any:
        """Execute the supported expert layout without per-expert slicing."""
        from torch.nn import functional  # pylint: disable=C0415

        if layout == "three_projection":
            w1, w2, w3 = weights
            gate = self._dry_run_grouped_matmul(
                expert_input, w1.transpose(-2, -1), expert_loads,
            )
            up = self._dry_run_grouped_matmul(
                expert_input, w3.transpose(-2, -1), expert_loads,
            )
            hidden = functional.silu(gate) * up
            return self._dry_run_grouped_matmul(
                hidden, w2.transpose(-2, -1), expert_loads,
            )
        gate_up, down = weights
        gate_up_output = self._dry_run_grouped_matmul(
            expert_input, gate_up.transpose(-2, -1), expert_loads,
        )
        gate, up = gate_up_output.chunk(2, dim=-1)
        hidden = functional.silu(gate) * up
        return self._dry_run_grouped_matmul(
            hidden, down.transpose(-2, -1), expert_loads,
        )

    def _differentiable_all_to_all(
            self, input_tensor: Any,
            input_splits: tuple[int, ...], output_splits: tuple[int, ...],
    ) -> Any:
        """Execute the real differentiable EP collective with configured splits."""
        from hyper_parallel.platform import get_platform  # pylint: disable=C0415

        try:
            ep_group = self._ep_mesh().get_group()
        except (KeyError, TypeError, AttributeError) as error:
            if len(input_splits) == 1 and input_splits == output_splits:
                return input_tensor.clone()
            raise ValueError(
                "Configured MoE dry-run requires an accessible EP process group"
            ) from error
        return get_platform().differentiable_all_to_all_single(
            input_tensor,
            list(input_splits),
            list(output_splits),
            group=ep_group,
        )

    def _ep_group_identity(self) -> tuple[int, int]:
        """Return EP group size/rank, falling back to the configured topology."""
        try:
            ep_mesh = self._ep_mesh()
            return int(ep_mesh.size()), int(ep_mesh.get_local_rank())
        except (KeyError, TypeError, AttributeError):
            return int(self._base.mesh.ep_size), int(self._base.mesh.ep_rank)

    def _ep_mesh(self) -> Any:
        """Return the expert-parallel child mesh from the HyperModels mesh context."""
        expert_mesh = getattr(self._base.mesh, "fsdp_moe_mesh", None)
        if expert_mesh is None:
            raise ValueError("Configured MoE dry-run requires an expert mesh")
        return expert_mesh["ep"]

    def _restore_moe(self) -> None:
        """Restore every MoE instance forward replaced by this scope."""
        if self._ep_compute_module is not None:
            self._ep_compute_module.ep_routed_forward = self._original_ep_compute
            self._ep_compute_module = None
            self._original_ep_compute = None
        self._moe_module_names.clear()
        while self._patched_moe_modules:
            module, had_forward, original_forward = self._patched_moe_modules.pop()
            if had_forward:
                module.forward = original_forward
            else:
                del module.forward


def _report_limitations(metadata: Optional[Dict[str, Any]] = None) -> list[str]:
    """Return static limitations plus run-specific simulation caveats."""
    limitations = list(_LIMITATIONS)
    metadata = metadata or {}
    target = metadata.get("target_device")
    simulation = metadata.get("simulation_device")
    if target and simulation and target != simulation:
        limitations.append(
            f"The logical {target} run was simulated with {simulation} FakeTensors; "
            "target-backend custom operators, device guards, and allocator behavior "
            "were not exercised."
        )
    return limitations


def _category_name(category: Any) -> str:
    """Return a stable JSON key for a MemTracker category."""
    if isinstance(category, str):
        return category
    value = getattr(category, "value", None)
    return str(value if value is not None else category)


def _normalize_snapshot(snapshot: Dict[Any, Dict[Any, int]]) -> Dict[str, Dict[str, int]]:
    """Convert device/category objects in a MemTracker snapshot to JSON keys."""
    normalized = {}
    for device, categories in snapshot.items():
        normalized[str(device)] = {
            _category_name(category): int(value)
            for category, value in categories.items()
        }
    return normalized


def _snapshot_total(snapshot: Dict[str, Dict[str, int]], device_type: str) -> int:
    """Sum total bytes for devices matching ``device_type``."""
    return sum(
        categories.get("Total", 0)
        for device, categories in snapshot.items()
        if device.split(":", maxsplit=1)[0] == device_type
    )


def _snapshot_breakdown(snapshot: Dict[str, Dict[str, int]], device_type: str) -> Dict[str, int]:
    """Aggregate category bytes for devices matching ``device_type``."""
    breakdown: Dict[str, int] = {}
    for device, categories in snapshot.items():
        if device.split(":", maxsplit=1)[0] != device_type:
            continue
        for category, value in categories.items():
            if category == "Total":
                continue
            breakdown[category] = breakdown.get(category, 0) + int(value)
    return dict(sorted(breakdown.items()))


def _module_local_peak(module_stats: Any, device_type: str) -> int:
    """Return a module's largest local peak on the target device type."""
    peaks = [
        int(value)
        for device, value in getattr(module_stats, "local_peak", {}).items()
        if str(device).split(":", maxsplit=1)[0] == device_type
    ]
    return max(peaks, default=0)


def _serialize_module(module_stats: Any, device_type: str) -> Dict[str, Any]:
    """Serialize one MemTracker module-stat object without private enum types."""
    snapshots = {}
    for state, snapshot_list in getattr(module_stats, "snapshots", {}).items():
        snapshots[_category_name(state)] = [
            _normalize_snapshot(snapshot)
            for snapshot in snapshot_list
        ]
    return {
        "fqn": str(module_stats.mod_fqn),
        "parameter_bytes": int(getattr(module_stats, "parameter_mem", 0)),
        "buffer_bytes": int(getattr(module_stats, "buffer_mem", 0)),
        "input_bytes": int(getattr(module_stats, "input_mem", 0)),
        "output_bytes": int(getattr(module_stats, "output_mem", 0)),
        "local_peak_bytes": _module_local_peak(module_stats, device_type),
        "snapshots": snapshots,
    }


class _PythonStackCache:
    """Capture and intern operator allocation stacks for one trace mode."""

    def __init__(self) -> None:
        """Initialize an empty cache keyed by complete filtered frame tuples."""
        self._cache: Dict[tuple[tuple[str, int, str], ...], Dict[str, Any]] = {}

    @staticmethod
    def _is_internal_frame(filename: str, function_name: str) -> bool:
        """Return whether a frame was excluded by the previous stack collector."""
        normalized_path = filename.replace("\\", "/")
        if normalized_path == __file__ and function_name in (
                "__torch_dispatch__",
                "capture",
                "_record_storage_resize",
                "resize_",
        ):
            return True
        return normalized_path.endswith(("/torch/_compile.py", "/torch/_dynamo/eval_frame.py"))

    def capture(self) -> Dict[str, Any]:
        """Return cached stack text and leaf location for the current caller."""
        frame = sys._getframe()  # pylint: disable=W0212
        frames = []
        while frame is not None and len(frames) < 64:
            code = frame.f_code
            frames.append((code.co_filename, frame.f_lineno, code.co_name))
            frame = frame.f_back
        frames.reverse()
        stack_key = tuple(
            frame_info
            for frame_info in frames
            if not self._is_internal_frame(frame_info[0], frame_info[2])
        )
        cached = self._cache.get(stack_key)
        if cached is not None:
            return cached
        python_stack = "|".join(
            f"File:{filename};Line:{line_num};Function:{function_name}"
            for filename, line_num, function_name in stack_key
        )
        leaf_frame = stack_key[-1] if stack_key else ("", 0, "")
        captured = {
            "python_stack": python_stack,
            "file_name": leaf_frame[0],
            "line_num": leaf_frame[1],
        }
        self._cache[stack_key] = captured
        return captured


def _create_indexed_mem_tracker(mem_tracker_type: Any) -> Any:
    """Create a dry-run-only MemTracker with indexed module peaks.
    """

    class _IndexedMemTracker(mem_tracker_type):
        """Dry-run tracker with an ordered FQN-to-module-stat index."""

        class _NoOpHookHandle:
            """Minimal removable handle for an intentionally absent fake hook."""

            def remove(self) -> None:
                """Match PyTorch's hook-handle removal protocol."""

        @staticmethod
        def _is_fake_dtensor_nonleaf(parameter: Any) -> bool:
            """Return whether a failed hook belongs to a fake DTensor view."""
            local_tensor = getattr(parameter, "_local_tensor", None)
            return bool(
                getattr(parameter, "_is_fake_wrapper", False)
                and local_tensor is not None
                and not bool(getattr(local_tensor, "is_leaf", True))
            )

        def _track_module_params_and_buffers(
                self, module: Any, install_grad_hooks: bool = True,
        ) -> tuple[int, int]:
            """Track module state with a scoped fake-DTensor hook exception."""
            # MemTracker does not expose a narrower hook-registration method.
            # Keep this copy of its small registration loop local to the dryrun
            # subclass so the exception cannot affect production tracking.
            from torch.distributed._tools.mem_tracker import _MemRefType  # pylint: disable=C0415

            def _grad_hook(gradient: Any) -> None:
                self._update_and_maybe_create_winfos(gradient, _MemRefType.GRAD)

            parameter_memory = 0
            for parameter in module.parameters():
                winfos = self._update_and_maybe_create_winfos(parameter, _MemRefType.PARAM)
                parameter_memory += sum(winfo.mem_consumed for winfo in winfos)
                parameter_gradient = next(self._gradient_candidates(parameter), None)
                if parameter_gradient is not None:
                    self._update_and_maybe_create_winfos(parameter_gradient, _MemRefType.GRAD)
                if self._param_to_grad_hook_handles.get(parameter) is not None or not install_grad_hooks:
                    continue
                grad_hook_handle = parameter.register_hook(_grad_hook)
                try:
                    post_accumulate_hook_handle = parameter.register_post_accumulate_grad_hook(
                        lambda param: _grad_hook(param.grad),
                    )
                except RuntimeError as error:
                    if "non-leaf" not in str(error) or not self._is_fake_dtensor_nonleaf(parameter):
                        grad_hook_handle.remove()
                        raise
                    post_accumulate_hook_handle = self._NoOpHookHandle()
                self._param_to_grad_hook_handles[parameter] = (
                    grad_hook_handle, post_accumulate_hook_handle,
                )

            buffer_memory = 0
            for buffer in module.buffers():
                winfos = self._update_and_maybe_create_winfos(buffer, _MemRefType.BUFFER)
                buffer_memory += sum(winfo.mem_consumed for winfo in winfos)
            return parameter_memory, buffer_memory

        def __init__(self) -> None:
            """Initialize base tracking state and empty module indexes."""
            super().__init__()
            self._module_stats_by_fqn: Dict[str, Dict[Any, Any]] = {}
            self._module_fqn_by_module: weakref.WeakKeyDictionary[Any, str] = weakref.WeakKeyDictionary()
            self._module_registration_order: weakref.WeakKeyDictionary[Any, int] = weakref.WeakKeyDictionary()
            self._next_module_registration_order = 0
            self._fake_dtensor_grad_bridge_refs: Dict[int, Any] = {}

        @staticmethod
        def _gradient_candidates(parameter: Any) -> Iterator[Any]:
            """Yield every supported final-gradient owner for one parameter."""
            import torch  # pylint: disable=C0415

            seen = set()
            if getattr(parameter, "_is_fake_wrapper", False):
                # Bypass the DTensor grad override and inspect wrapper storage.
                gradient = torch.Tensor.grad.__get__(parameter, type(parameter))  # pylint: disable=C2801
            else:
                gradient = getattr(parameter, "grad", None)
            if gradient is not None:
                seen.add(id(gradient))
                yield gradient
            for attribute in ("main_grad", "_fake_pending_grad", "_dry_run_grad_bridge"):
                gradient = getattr(parameter, attribute, None)
                if gradient is None or id(gradient) in seen:
                    continue
                seen.add(id(gradient))
                yield gradient
            local_tensor = getattr(parameter, "_local_tensor", None)
            local_gradient = (
                getattr(local_tensor, "grad", None)
                if local_tensor is not None and local_tensor.is_leaf
                else None
            )
            if local_gradient is not None and id(local_gradient) not in seen:
                yield local_gradient

        @staticmethod
        def _module_gradient_parameters(module: Any) -> Iterator[Any]:
            """Yield module and FSDP-managed parameters without duplicate identities."""
            seen = set()
            for parameter in module.parameters():
                if id(parameter) not in seen:
                    seen.add(id(parameter))
                    yield parameter
            for submodule in module.modules():
                scheduler = getattr(submodule, "hsdp_scheduler", None)
                state = getattr(scheduler, "hsdp_state", None)
                for hsdp_param in getattr(state, "hsdp_params", ()):
                    for attribute in ("sharded_param", "unsharded_param"):
                        parameter = getattr(hsdp_param, attribute, None)
                        if parameter is not None and id(parameter) not in seen:
                            seen.add(id(parameter))
                            yield parameter

        def refresh_parameter_gradients(self, module: Any) -> list[Any]:
            """Classify all currently retained local parameter gradients."""
            import torch  # pylint: disable=C0415
            from torch.distributed._tools.mem_tracker import _MemRefType  # pylint: disable=C0415

            gradients = []
            seen = set()
            module_parameters = list(module.parameters())
            for parameter in module_parameters:
                has_gradient = next(self._gradient_candidates(parameter), None) is not None
                local_tensor = getattr(parameter, "_local_tensor", None)
                if (
                        not has_gradient
                        and getattr(parameter, "_is_fake_wrapper", False)
                        and local_tensor is not None
                        and parameter.requires_grad
                ):
                    with torch.no_grad():
                        parameter._dry_run_grad_bridge = torch.empty_like(
                            local_tensor,
                            memory_format=torch.preserve_format,
                        )
                    self._fake_dtensor_grad_bridge_refs[id(parameter)] = weakref.ref(parameter)
            for parameter in self._module_gradient_parameters(module):
                for gradient in self._gradient_candidates(parameter):
                    if id(gradient) in seen:
                        continue
                    seen.add(id(gradient))
                    self._update_and_maybe_create_winfos(gradient, _MemRefType.GRAD)
                    gradients.append(gradient)
            return gradients

        def clear_fake_dtensor_grad_bridges(self) -> None:
            """Release gradients retained solely by FakeTensor DTensor bridges."""
            stale_parameter_ids = []
            for parameter_id, parameter_ref in self._fake_dtensor_grad_bridge_refs.items():
                parameter = parameter_ref()
                if parameter is None:
                    stale_parameter_ids.append(parameter_id)
                    continue
                parameter._dry_run_grad_bridge = None
            for parameter_id in stale_parameter_ids:
                self._fake_dtensor_grad_bridge_refs.pop(parameter_id, None)

        def _remove_from_fqn(self, module_fqn: str, module: Any) -> None:
            """Remove a module from one FQN bucket and prune an empty bucket."""
            bucket = self._module_stats_by_fqn.get(module_fqn)
            if bucket is None:
                return
            bucket.pop(module, None)
            if not bucket:
                self._module_stats_by_fqn.pop(module_fqn, None)

        def _rebind_module_stats(self, module: Any) -> None:
            """Bind a module's latest stats FQN while preserving first-seen order."""
            module_stats = self.memory_tracking.get(module)
            if module_stats is None:
                return
            new_fqn = str(module_stats.mod_fqn)
            old_fqn = self._module_fqn_by_module.get(module)
            if old_fqn == new_fqn:
                return
            if old_fqn is not None:
                self._remove_from_fqn(old_fqn, module)
            if module not in self._module_registration_order:
                self._module_registration_order[module] = self._next_module_registration_order
                self._next_module_registration_order += 1
            self._module_stats_by_fqn.setdefault(new_fqn, {})[module] = module_stats
            self._module_fqn_by_module[module] = new_fqn

        def _pre_fw_hook(self, module: Any, inputs: Any) -> None:
            """Track repeated forwards without resetting global peaks."""
            from torch.distributed._tools.mem_tracker import (  # pylint: disable=C0415
                _ModMemStats,
                _ModState,
                _TOTAL_KEY,
            )

            module_fqn = self._mod_tracker.get_known_fqn(module)
            if module_fqn is None:
                raise RuntimeError("MemTracker could not resolve a module FQN")
            previous_peaks = {}
            if module not in self.memory_tracking:
                module_stats = _ModMemStats(module_fqn)
                parameter_mem, buffer_mem = self._track_module_params_and_buffers(
                    module,
                    install_grad_hooks=True,
                )
                module_stats.parameter_mem = parameter_mem
                module_stats.buffer_mem = buffer_mem
                module_stats.input_mem = self._track_inputs_or_outputs(inputs)
                self.memory_tracking[module] = module_stats
                state = _ModState.PRE_FW
            elif self._mod_tracker.is_bw:
                module_stats = self.memory_tracking[module]
                state = _ModState.PRE_FW_AC
                if getattr(self, "_ac_mod", None) is None:
                    self._ac_mod = weakref.ref(module)
                    self._in_ac = True
            else:
                module_stats = self.memory_tracking[module]
                state = _ModState.PRE_FW
                previous_peaks = dict(module_stats.local_peak)
                module_stats.mod_fqn = module_fqn
                module_stats.input_mem = self._track_inputs_or_outputs(inputs)

            memory_snapshot = self.get_tracker_snapshot()
            if state == _ModState.PRE_FW:
                module_stats.local_peak = {
                    device: max(
                        previous_peaks.get(device, 0),
                        device_snapshot[_TOTAL_KEY],
                    )
                    for device, device_snapshot in memory_snapshot.items()
                }
                module_stats.snapshots.setdefault(_ModState.PEAK_FW, []).append(
                    memory_snapshot
                )
            module_stats.snapshots.setdefault(state, []).append(
                copy.deepcopy(memory_snapshot)
            )
            self._rebind_module_stats(module)

        def reset_mod_stats(self) -> None:
            """Clear module statistics and all indexes derived from them."""
            super().reset_mod_stats()
            self._module_stats_by_fqn.clear()
            self._module_fqn_by_module.clear()
            self._module_registration_order.clear()
            self._next_module_registration_order = 0
            self._fake_dtensor_grad_bridge_refs.clear()

        def _update_peak_stats(self, peak_state: Any) -> None:
            """Update active module peaks in registration order and global peak."""
            active_modules = {}
            for module_fqn in self._mod_tracker.parents:
                for module, module_stats in self._module_stats_by_fqn.get(str(module_fqn), {}).items():
                    active_modules[module] = module_stats
            ordered_modules = sorted(
                active_modules.items(),
                key=lambda item: self._module_registration_order[item[0]],
            )
            current_snapshot = self._curr_mem_snap
            for _, module_stats in ordered_modules:
                if peak_state not in module_stats.snapshots:
                    continue
                for device, device_snapshot in current_snapshot.items():
                    if module_stats.local_peak.get(device, 0) < device_snapshot["Total"]:
                        module_stats.local_peak[device] = device_snapshot["Total"]
                        module_stats.snapshots[peak_state][-1][device] = copy.deepcopy(device_snapshot)

            for device, device_snapshot in current_snapshot.items():
                if self._peak_mem.get(device, 0) < device_snapshot["Total"]:
                    self._peak_mem[device] = device_snapshot["Total"]
                    self._peak_mem_snap[device] = copy.deepcopy(device_snapshot)

        def get_device_memory_totals(self, device_name: str) -> tuple[int, int]:
            """Read current and peak totals for one device without copying snapshots."""
            current_total = 0
            for device, device_snapshot in self._curr_mem_snap.items():
                if str(device) == device_name:
                    current_total = int(device_snapshot.get("Total", 0))
                    break
            recorded_peak_total = 0
            for device, device_snapshot in self._peak_mem_snap.items():
                if str(device) == device_name:
                    recorded_peak_total = int(device_snapshot.get("Total", 0))
                    break
            return current_total, max(current_total, recorded_peak_total)

    return _IndexedMemTracker()


def _operator_phase(tracker: Any) -> str:
    """Resolve the current training phase from MemTracker state."""
    if getattr(tracker, "_in_opt", False):
        return "optimizer"
    from hyper_parallel.core.activation_checkpoint.recompute_state import (  # pylint: disable=C0415
        is_recomputing,
    )
    if is_recomputing():
        return "recompute"
    module_tracker = getattr(tracker, "_mod_tracker", None)
    if getattr(module_tracker, "is_bw", False):
        return "backward"
    return "forward"


def _active_module_fqn(tracker: Any) -> str:
    """Return the deepest active module as optional operator context."""
    module_tracker = getattr(tracker, "_mod_tracker", None)
    parents = getattr(module_tracker, "parents", ())
    candidates = [str(parent) for parent in parents if str(parent) not in ("", "Global")]
    return max(candidates, key=lambda fqn: (fqn.count("."), len(fqn)), default="")


def _create_operator_trace_mode(tracker: Any, device_type: str) -> Any:
    """Create a lazy TorchDispatchMode that records logical memory blocks."""
    import torch  # pylint: disable=C0415
    from torch.distributed._tools.mem_tracker import get_untyped_storages  # pylint: disable=C0415
    from torch.utils._python_dispatch import TorchDispatchMode  # pylint: disable=C0415
    from torch.utils._pytree import tree_flatten  # pylint: disable=C0415

    def _storage_map(values: Any) -> Dict[int, Dict[str, Any]]:
        """Return storage identifiers, logical devices, and byte sizes."""
        flat_values, _ = tree_flatten(values)
        storages = {}
        for value in flat_values:
            if not isinstance(value, torch.Tensor):
                continue
            try:
                tensor_storages = get_untyped_storages(value)
            except (RuntimeError, TypeError):
                continue
            for storage in tensor_storages:
                storage_key = int(getattr(storage, "_cdata", id(storage)))
                storages[storage_key] = {
                    "device": str(value.device),
                    "size": int(storage.size()),
                    "storage": storage,
                }
        return storages

    def _lookup_tracked_storage(storage: Any) -> Optional[Dict[str, Any]]:
        """Read MemTracker metadata for one storage without scanning all entries."""
        entry = tracker._WINFO.get(storage)
        if entry is None:
            return None
        winfo, storage_ref = entry
        return {
            "device": str(winfo.device),
            "size": int(winfo.mem_consumed),
            "type": _category_name(winfo.reftype),
            "storage_ref": storage_ref,
        }

    def _tracked_output_info(
            storage_keys: set[int],
            output_storages: Dict[int, Dict[str, Any]],
    ) -> Dict[int, Dict[str, Any]]:
        """Read MemTracker metadata only for selected operator outputs."""
        tracked = {}
        for storage_key in storage_keys:
            tracked_storage = _lookup_tracked_storage(
                output_storages[storage_key]["storage"]
            )
            if tracked_storage is not None:
                tracked[storage_key] = tracked_storage
        return tracked

    class _OperatorTraceMode(TorchDispatchMode):
        """Track producer, consumer, and lifetime data for fake storages."""

        def __init__(self) -> None:
            """Initialize ordered logical task and storage state."""
            super().__init__()
            self.memory_blocks: list[Dict[str, Any]] = []
            self._active_blocks: Dict[int, Dict[str, Any]] = {}
            self._active_storage_refs: Dict[int, Any] = {}
            self._active_storage_sizes: Dict[int, int] = {}
            self._released_storage_keys: set[int] = set()
            self._next_task_index = 0
            self._next_lifecycle_event_index = 0
            self._next_virtual_address = _VIRTUAL_ADDRESS_BASE
            self._previous_storage_resize = None
            self._stack_cache = _PythonStackCache()

        def __enter__(self) -> Any:
            """Enter dispatch mode and observe MemTracker's storage resize hook."""
            super().__enter__()
            try:
                self._install_storage_resize_hook()
                self._seed_existing_state()
            except Exception:
                super().__exit__(None, None, None)
                raise
            return self

        def __exit__(self, *args: Any) -> Any:
            """Restore MemTracker's storage resize hook before leaving dispatch mode."""
            self._restore_storage_resize_hook()
            return super().__exit__(*args)

        def _install_storage_resize_hook(self) -> None:
            """Wrap the active storage resize method to record capacity lifetimes."""
            previous_resize = torch.UntypedStorage.resize_
            self._previous_storage_resize = previous_resize

            @functools.wraps(previous_resize)
            def resize_(storage: Any, size: int) -> Any:
                """Resize storage through MemTracker, then record its new lifetime."""
                old_size = int(storage.size())
                result = previous_resize(storage, size)
                new_size = int(storage.size())
                if old_size != new_size:
                    self._record_storage_resize(storage, old_size, new_size)
                return result

            torch.UntypedStorage.resize_ = resize_  # type: ignore[method-assign, assignment]

        def _restore_storage_resize_hook(self) -> None:
            """Restore the resize method that was active when this mode entered."""
            if self._previous_storage_resize is not None:
                torch.UntypedStorage.resize_ = self._previous_storage_resize  # type: ignore[method-assign, assignment]
            self._previous_storage_resize = None

        def _seed_existing_state(self) -> None:
            """Seed Parameter and Buffer storages allocated before tracing."""
            seed_roles = {"Parameter", "Buffer"}
            output_storages = {}
            tracked_storages = {}
            for storage, _ in list(tracker._WINFO.items()):
                tracked_storage = _lookup_tracked_storage(storage)
                if (
                        tracked_storage is None
                        or tracked_storage["type"] not in seed_roles
                        or tracked_storage["device"].split(":", maxsplit=1)[0] != device_type
                        or tracked_storage["size"] <= 0
                ):
                    continue
                storage_key = int(getattr(storage, "_cdata", id(storage)))
                output_storages[storage_key] = {
                    "device": tracked_storage["device"],
                    "size": tracked_storage["size"],
                    "storage": storage,
                }
                tracked_storages[storage_key] = tracked_storage
            if not output_storages:
                return
            task_index = self._next_task_index
            self._next_task_index += 1
            devices = {storage["device"] for storage in output_storages.values()}
            self._track_new_outputs(
                set(output_storages),
                output_storages,
                tracked_storages,
                {
                    "task_index": task_index,
                    "device_totals": {
                        device: tracker.get_device_memory_totals(device)
                        for device in devices
                    },
                    "stack": {"python_stack": "", "file_name": "", "line_num": 0},
                    "phase": "initialization",
                    "operator_name": "dry_run.seed_external",
                    "module_fqn": "",
                },
            )

        def refresh_tensor_roles(self, tensors: Any) -> None:
            """Apply MemTracker role changes to active CSV storage blocks."""
            for storage_key, storage_info in _storage_map(tensors).items():
                block = self._active_blocks.get(storage_key)
                if block is None:
                    continue
                tracked_storage = _lookup_tracked_storage(storage_info["storage"])
                if tracked_storage is not None:
                    block["type"] = tracked_storage["type"]

        def _close_blocks(self, storage_keys: set[int], end_index: int) -> None:
            """Close active blocks and discard their lifetime bookkeeping."""
            for storage_key in sorted(storage_keys):
                block = self._active_blocks.pop(storage_key, None)
                if block is None:
                    continue
                self._active_storage_refs.pop(storage_key, None)
                self._active_storage_sizes.pop(storage_key, None)
                self._released_storage_keys.discard(storage_key)
                block["end_time_stamp"] = end_index
                block["_end_lifecycle_event_index"] = self._next_lifecycle_event_index
                self._next_lifecycle_event_index += 1

        def _close_released_blocks(self, end_index: int) -> None:
            """Close blocks whose storage weak refs disappeared."""
            released_keys = self._released_storage_keys
            self._released_storage_keys = set()
            self._close_blocks(released_keys, end_index)

        def _allocate_virtual_address(self, size: int) -> str:
            """Return a deterministic aligned address in a reserved fake range."""
            address = self._next_virtual_address
            aligned_size = max(
                _VIRTUAL_ADDRESS_ALIGNMENT,
                (
                    (size + _VIRTUAL_ADDRESS_ALIGNMENT - 1)
                    // _VIRTUAL_ADDRESS_ALIGNMENT
                ) * _VIRTUAL_ADDRESS_ALIGNMENT,
            )
            self._next_virtual_address += aligned_size
            return f"0x{address:x}"

        def _assign_plot_coordinates(self) -> None:
            """Assign profiler-style fractional coordinates to lifecycle events.

            Logical task indices intentionally remain unchanged. Release and
            allocation events sharing one task are ordered by their observation
            sequence and spread inside that task's ``[task, task + 1)`` range.
            """
            events_by_task: Dict[int, list[tuple[int, Dict[str, Any], str]]] = {}
            for block in self.memory_blocks:
                start_task = int(block["start_time_stamp"])
                events_by_task.setdefault(start_task, []).append((
                    int(block.pop("_start_lifecycle_event_index")),
                    block,
                    "start_plot_time_stamp",
                ))
                end_event_index = block.pop("_end_lifecycle_event_index", None)
                if end_event_index is None:
                    continue
                end_task = int(block["end_time_stamp"])
                events_by_task.setdefault(end_task, []).append((
                    int(end_event_index),
                    block,
                    "end_plot_time_stamp",
                ))

            for task_index, task_events in events_by_task.items():
                ordered_events = sorted(task_events, key=lambda event: event[0])
                denominator = Decimal(len(ordered_events) + 1)
                for event_offset, (_, block, field) in enumerate(ordered_events, start=1):
                    coordinate = Decimal(task_index) + Decimal(event_offset) / denominator
                    block[field] = format(coordinate, ".12f")

        def finalize(self) -> None:
            """Close released blocks and mark remaining storages persistent."""
            self._close_released_blocks(self._next_task_index)
            for storage_key, block in self._active_blocks.items():
                storage_ref = self._active_storage_refs.get(storage_key)
                storage = storage_ref() if storage_ref is not None else None
                tracked_storage = (
                    _lookup_tracked_storage(storage)
                    if storage is not None
                    else None
                )
                if tracked_storage is None or tracked_storage["size"] <= 0:
                    block["end_time_stamp"] = self._next_task_index
                    block["_end_lifecycle_event_index"] = self._next_lifecycle_event_index
                    self._next_lifecycle_event_index += 1
                    continue
                block["type"] = tracked_storage["type"]
                block["end_time_stamp"] = _PERSISTENT_END_INDEX
                block["is_persistent"] = 1
            self._active_blocks.clear()
            self._active_storage_refs.clear()
            self._active_storage_sizes.clear()
            self._released_storage_keys.clear()
            self._assign_plot_coordinates()

        def _classify_output_keys(
                self,
                input_storages: Dict[int, Dict[str, Any]],
                output_storages: Dict[int, Dict[str, Any]],
                active_input_keys: set[int],
        ) -> tuple[set[int], set[int]]:
            """Return resized inputs and newly allocated target-device outputs."""
            resized_output_keys = {
                storage_key
                for storage_key in active_input_keys & set(output_storages)
                if output_storages[storage_key]["size"] != self._active_storage_sizes[storage_key]
            }
            new_output_keys = {
                storage_key
                for storage_key, storage in output_storages.items()
                if (
                    storage_key not in input_storages
                    and storage_key not in self._active_blocks
                    and storage["device"].split(":", maxsplit=1)[0] == device_type
                    and storage["size"] > 0
                )
            }
            new_output_keys.update(
                storage_key
                for storage_key in resized_output_keys
                if (
                    output_storages[storage_key]["device"].split(":", maxsplit=1)[0] == device_type
                    and output_storages[storage_key]["size"] > 0
                )
            )
            return resized_output_keys, new_output_keys

        def _record_input_users(self, active_input_keys: set[int], task_index: int) -> None:
            """Append the task index to every active input storage's user list."""
            for storage_key in active_input_keys:
                user_tasks = self._active_blocks[storage_key]["user_tasks"]
                if not user_tasks or user_tasks[-1] != task_index:
                    user_tasks.append(task_index)

        def _track_new_outputs(
                self,
                output_keys: set[int],
                output_storages: Dict[int, Dict[str, Any]],
                tracked_storages: Dict[int, Dict[str, Any]],
                context: Dict[str, Any],
        ) -> None:
            """Create logical memory blocks and weak lifetime refs for outputs."""
            for storage_key in sorted(output_keys):
                output_storage = output_storages[storage_key]
                tracked_storage = tracked_storages.get(storage_key, {})
                device = output_storage["device"]
                size = int(tracked_storage.get("size", output_storage["size"]))
                current_total, peak_total = context["device_totals"].get(device, (0, 0))
                block = {
                    "start_time_stamp": context["task_index"],
                    "end_time_stamp": _PERSISTENT_END_INDEX,
                    "_start_lifecycle_event_index": self._next_lifecycle_event_index,
                    "device_addr": self._allocate_virtual_address(size),
                    "stream_id": 0,
                    "pool_type": _FAKE_MEMORY_POOL_TYPE,
                    "size": size,
                    "actual_used_memory": current_total,
                    "actual_peak_memory": peak_total,
                    "file_name": context["stack"]["file_name"],
                    "line_num": context["stack"]["line_num"],
                    "type": tracked_storage.get("type", "Activation"),
                    "producer_task": context["task_index"],
                    "task_name": context["phase"],
                    "node_name": context["operator_name"],
                    "graph_name": context["module_fqn"],
                    "user_tasks": [],
                    "python_stack": context["stack"]["python_stack"],
                    "is_persistent": 0,
                    "is_small": int(size < _SMALL_ALLOCATION_BYTES),
                }
                self._next_lifecycle_event_index += 1
                self.memory_blocks.append(block)
                self._active_blocks[storage_key] = block
                self._active_storage_sizes[storage_key] = output_storage["size"]
                storage_ref = tracked_storage.get("storage_ref")
                storage = storage_ref() if storage_ref is not None else None
                if storage is None:
                    storage = output_storage["storage"]
                self._active_storage_refs[storage_key] = weakref.ref(
                    storage,
                    lambda _, key=storage_key: self._released_storage_keys.add(key),
                )

        def _record_storage_resize(self, storage: Any, old_size: int, new_size: int) -> None:
            """Record one tracked storage capacity transition as a block lifetime."""
            storage_key = int(getattr(storage, "_cdata", id(storage)))
            tracked_storage = _lookup_tracked_storage(storage)
            if tracked_storage is None:
                return
            device = tracked_storage["device"]
            if device.split(":", maxsplit=1)[0] != device_type:
                return

            if old_size > 0 or storage_key in self._active_blocks:
                self._close_blocks({storage_key}, self._next_task_index)
            if new_size <= 0:
                return

            task_index = self._next_task_index
            self._next_task_index += 1
            output_storages = {
                storage_key: {
                    "device": device,
                    "size": new_size,
                    "storage": storage,
                },
            }
            tracked_storages = {storage_key: tracked_storage}
            self._track_new_outputs(
                {storage_key},
                output_storages,
                tracked_storages,
                {
                    "task_index": task_index,
                    "device_totals": {device: tracker.get_device_memory_totals(device)},
                    "stack": self._stack_cache.capture(),
                    "phase": _operator_phase(tracker),
                    "operator_name": "torch.UntypedStorage.resize_",
                    "module_fqn": _active_module_fqn(tracker),
                },
            )

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            del types
            self._close_released_blocks(self._next_task_index)
            input_storages = _storage_map((args, kwargs))
            active_input_keys = {
                storage_key
                for storage_key in input_storages
                if storage_key in self._active_blocks
            }
            result = func(*args, **(kwargs or {}))
            output_storages = _storage_map(result)
            resized_output_keys, new_output_keys = self._classify_output_keys(
                input_storages,
                output_storages,
                active_input_keys,
            )
            self._close_released_blocks(self._next_task_index)
            if not active_input_keys and not new_output_keys and not resized_output_keys:
                return result
            task_index = self._next_task_index
            self._next_task_index += 1
            self._record_input_users(active_input_keys, task_index)
            self._close_blocks(resized_output_keys, task_index)
            if not new_output_keys:
                return result
            tracked_storages = _tracked_output_info(
                new_output_keys,
                output_storages,
            )
            output_devices = {
                output_storages[storage_key]["device"]
                for storage_key in new_output_keys
            }
            device_totals = {
                device: tracker.get_device_memory_totals(device)
                for device in output_devices
            }
            self._track_new_outputs(
                new_output_keys,
                output_storages,
                tracked_storages,
                {
                    "task_index": task_index,
                    "device_totals": device_totals,
                    "stack": self._stack_cache.capture(),
                    "phase": _operator_phase(tracker),
                    "operator_name": str(func),
                    "module_fqn": _active_module_fqn(tracker),
                },
            )
            return result

    return _OperatorTraceMode()


def _report_device_name(device: str, device_type: str, report_device_type: Optional[str]) -> str:
    """Replace a simulation-device prefix with its logical report prefix."""
    if report_device_type is None:
        return device
    prefix, separator, suffix = device.partition(":")
    if prefix != device_type:
        return device
    return report_device_type + (separator + suffix if separator else "")


def _rewrite_module_snapshot_devices(
        modules: list[Dict[str, Any]],
        device_type: str,
        report_device_type: Optional[str],
) -> None:
    """Rewrite device keys in every serialized module snapshot in place."""
    if report_device_type is None or report_device_type == device_type:
        return
    for module in modules:
        for snapshots in module["snapshots"].values():
            for snapshot in snapshots:
                renamed = {
                    _report_device_name(device, device_type, report_device_type): values
                    for device, values in snapshot.items()
                }
                snapshot.clear()
                snapshot.update(renamed)


def build_memory_report(
        tracker: Any,
        metadata: Dict[str, Any],
        device_type: str,
        memory_blocks: Optional[list[Dict[str, Any]]] = None,
        report_device_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the stable JSON report from a completed MemTracker run.

    Args:
        tracker: Completed memory-tracker instance.
        metadata: Run metadata to include verbatim.
        device_type: Device type used for FakeTensor simulation. This is the
            logical accelerator type when its backend is available, otherwise
            ``"cpu"`` for the no-backend fallback.
        memory_blocks: Optional logical output-storage lifetime records.
        report_device_type: Logical target device label used in the report.

    Returns:
        JSON-serializable report dictionary.

    Raises:
        ValueError: If the tracker observed no memory on the simulation device.
    """
    peak = _normalize_snapshot(tracker.get_tracker_snapshot("peak"))
    current = _normalize_snapshot(tracker.get_tracker_snapshot("current"))
    peak_bytes = _snapshot_total(peak, device_type)
    if peak_bytes <= 0:
        raise ValueError(
            f"MemTracker observed no {device_type!r} tensor memory; "
            f"tracked devices are {sorted(peak)}"
        )

    modules = [
        _serialize_module(module_stats, device_type)
        for module_stats in tracker.memory_tracking.values()
    ]
    modules.sort(key=lambda item: (-item["local_peak_bytes"], item["fqn"]))

    devices = {}
    for device in sorted(set(peak) | set(current)):
        report_device = _report_device_name(device, device_type, report_device_type)
        devices[report_device] = {
            "peak": peak.get(device, {}),
            "current": current.get(device, {}),
        }
    _rewrite_module_snapshot_devices(modules, device_type, report_device_type)
    return {
        "schema_version": _SCHEMA_VERSION,
        "status": "ok",
        "metadata": metadata,
        "summary": {
            "peak_bytes": peak_bytes,
            "current_bytes": _snapshot_total(current, device_type),
            "peak_gib": peak_bytes / _GIB,
            "peak_breakdown_bytes": _snapshot_breakdown(peak, device_type),
            "current_breakdown_bytes": _snapshot_breakdown(current, device_type),
        },
        "devices": devices,
        "modules": modules,
        "memory_blocks": memory_blocks or [],
        "limitations": _report_limitations(metadata),
    }


def write_memory_report(report: Dict[str, Any], output_path: str) -> str:
    """Atomically write the complete JSON report including metadata."""
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as report_file:
            json.dump(report, report_file, indent=2, sort_keys=True)
            report_file.write("\n")
        os.replace(temporary_path, destination)
    except Exception:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise
    return str(destination)
