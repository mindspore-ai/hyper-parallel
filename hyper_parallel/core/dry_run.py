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
import os
import sys
from collections.abc import Mapping
from contextlib import AbstractContextManager, ExitStack, nullcontext
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from fnmatch import fnmatchcase
from types import MethodType, SimpleNamespace
from typing import Any, Callable, ContextManager, Dict, Optional

from hyper_parallel.data.constants import IGNORE_INDEX


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
        import torch  # pylint: disable=C0415,forbidden-backend-import

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
        import torch  # pylint: disable=C0415,forbidden-backend-import

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
    import torch  # pylint: disable=C0415,forbidden-backend-import

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


_VALUE_DEPENDENCY_HANDLER_FACTORIES[MoERoutingValueDependencyHandler.name] = MoERoutingValueDependencyHandler


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
        self._resolved_moe_plans: Dict[str, Dict[str, Any]] = {}
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
        candidates: Dict[tuple[str, str], list[tuple[int, _ValueDependencyRule]]] = {}
        for rule in self._rules:
            handler = self._handlers.get(rule.handler)
            if handler is None:
                raise ValueError(
                    f"value dependency rule {rule.index} uses unknown handler {rule.handler!r}; "
                    f"registered handlers: {sorted(self._handlers)}"
                )
            supported_targets = handler.discover(model, context)
            # Activation wrappers can expose a name through more than one tree
            # position; keep the first occurrence so one target resolves once.
            matches = list(dict.fromkeys(
                target for target in supported_targets if fnmatchcase(target, rule.match)
            ))
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
        }

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


class ValueDependencyManager(AbstractContextManager):
    """Install configured semantic replay for one fake training step."""

    def __init__(self, profile: _DryRunValueProfile, model: Any, runtime: DryRunRuntime) -> None:
        """Store profile state without changing model or distributed runtime."""
        del model, runtime
        self._profile = profile
        self._base = None
        self._stack = ExitStack()

    def fake_step_context(self, base: Any) -> "ValueDependencyManager":
        """Bind the prepared Trainer state used by semantic handlers."""
        self._base = base
        return self

    def __enter__(self) -> "ValueDependencyManager":
        """Install configured MoE and TP cross-entropy replay."""
        if self._base is None:
            raise RuntimeError("ValueDependencyManager.fake_step_context must be configured before entry")
        for handler_name, rules in self._profile._resolved.items():
            if not rules:
                continue
            handler = self._profile.handler(handler_name)
            target, rule = next(iter(rules.items()))
            self._stack.enter_context(handler.install(
                ValueDependencyStage.FAKE_STEP,
                target,
                rule.path,
                rule.inputs,
                SimpleNamespace(profile=self._profile, base=self._base),
            ))
        mesh = getattr(self._base, "mesh", None)
        if bool(getattr(mesh, "loss_parallel", False)):
            self._stack.enter_context(_TPCrossEntropyValueDependencyRuntime(self._profile))
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        """Restore every scoped semantic override."""
        del exc_type, exc_value, traceback
        self._stack.close()
        return False


class _TPCrossEntropyValueDependencyRuntime(AbstractContextManager):
    """Scoped TP cross-entropy replay owned by its semantic handler."""

    def __init__(self, profile: _DryRunValueProfile) -> None:
        """Store the resolved ownership profile."""
        self._profile = profile
        self._loss_parallel_ops = None
        self._original_cross_entropy_function = None

    def __enter__(self) -> "_TPCrossEntropyValueDependencyRuntime":
        """Patch the complete CE autograd Function for this dry-run scope."""
        from hyper_parallel.components.losses import (  # pylint: disable=C0415
            _vocab_parallel_cross_entropy as loss_parallel_ops,
        )
        import torch  # pylint: disable=C0415,forbidden-backend-import

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
                total = loss_parallel_ops._differentiable_all_reduce(
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
        self._ep_compute_modules: list[Any] = []
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

    def _patch_moe(self) -> None:
        """Install model-specific MoE replay in a concrete runtime."""
        raise NotImplementedError

    def _restore_moe(self) -> None:
        """Restore model-specific MoE symbols in a concrete runtime."""
        raise NotImplementedError


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

                def dry_run_forward(
                        current_module: Any,
                        x: Any,
                        *args: Any,
                        name: str = module_name,
                        **kwargs: Any,
                ) -> Any:
                    """Execute the configured value-independent MoE path."""
                    del args, kwargs
                    return self._run_moe_forward(name, current_module, x)

                module.forward = MethodType(dry_run_forward, module)
                self._patched_moe_modules.append((module, had_forward, original_forward))
        if self._base.mesh.ep_size > 1 and module_names:
            from hyper_parallel.distributed.expert_parallel import experts as ep_experts  # pylint: disable=C0415

            self._original_ep_compute = ep_experts.ep_routed_forward

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
                )

            # ``ep_routed_forward`` is re-exported through several modules
            # (``recipes``, model adapters) that each bind the symbol into
            # their own namespace at import time. Patch every live binding so
            # the configured replay runs regardless of the entry point.
            for loaded_module in list(sys.modules.values()):
                if loaded_module is None:
                    continue
                # Inspect the module ``__dict__`` directly: ``getattr`` would
                # trigger PEP 562 lazy-import hooks (e.g. ``transformers``) that
                # pull optional imaging dependencies this path never uses.
                if (
                        loaded_module.__dict__.get("ep_routed_forward")
                        is self._original_ep_compute
                ):
                    loaded_module.ep_routed_forward = dry_run_ep_compute
                    self._ep_compute_modules.append(loaded_module)
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
        import torch  # pylint: disable=C0415,forbidden-backend-import

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
            module_name, module, x_flat, plan, num_experts, top_k,
        )
        expert_input, dispatch_context = self._dry_run_dispatch(
            routed_input, token_counts, plan, ep_size, int(weights[0].shape[0]),
        )
        expert_output = self._run_grouped_experts(
            layout, weights, expert_input, plan.local_expert_loads,
        )
        combined = self._dry_run_combine(expert_output, dispatch_context)
        routed_output = self._dry_run_weight_and_unpermute(
            module, x_flat, combined, top_weights, permutation, inverse_permutation, top_k,
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
    ) -> Any:
        """Replay the new Trainer's EP lifecycle with configured split sizes."""
        import torch  # pylint: disable=C0415,forbidden-backend-import
        import torch.distributed as dist  # pylint: disable=C0415,forbidden-backend-import

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
        # ep_routed_forward owns routed experts only. Model adapters execute
        # their shared-expert branch after this function returns.
        return output

    @staticmethod
    def _dry_run_route(
            module_name: str, module: Any, x_flat: Any,
            plan: _DryRunMoERoutingPlan,
            num_experts: int,
            top_k: int,
    ) -> tuple[Any, Any, Any, Any, Any]:
        """Replay value-independent router, sort, and histogram operators."""
        import torch  # pylint: disable=C0415,forbidden-backend-import
        from torch.nn import functional  # pylint: disable=C0415,forbidden-backend-import

        gate = getattr(module, "gate", None)
        gate_weight = getattr(gate, "weight", None)
        if gate_weight is None:
            raise ValueError(
                f"MoE module {module_name!r} must expose gate.weight for configured dry-run routing"
            )
        router_logits = x_flat @ gate_weight.transpose(0, 1)
        router_probs = functional.softmax(router_logits, dim=-1, dtype=torch.float32)
        top_weights, top_indices = torch.topk(
            router_probs, top_k, dim=-1,
        )
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        top_weights = top_weights.to(x_flat.dtype)
        module.router_logits = router_logits

        token_indices = torch.arange(
            x_flat.shape[0], device=x_flat.device,
        ).unsqueeze(1).expand(-1, top_k).reshape(-1)
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
            bins=num_experts,
            min=0,
            max=num_experts - 1,
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
            top_k: int,
    ) -> Any:
        """Apply routing weights and restore the original token order."""
        import torch  # pylint: disable=C0415,forbidden-backend-import

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
                x_flat.shape[0], top_k, x_flat.shape[-1],
            ).sum(dim=1, dtype=torch.float32).to(x_flat.dtype)
        weighted = combined * sorted_weights.unsqueeze(-1)
        unsorted = weighted[inverse_permutation]
        return unsorted.view(
            x_flat.shape[0], top_k, x_flat.shape[-1],
        ).sum(dim=1).to(x_flat.dtype)

    def _dry_run_exchange_counts(
            self, counts_input: Any, ep_size: int,
    ) -> Any:
        """Replay the non-differentiable count all-to-all at a fixed shape."""
        from hyper_parallel.core.context_parallel.utils import (  # pylint: disable=C0415
            all_to_all_single,
        )

        if ep_size == 1:
            return counts_input.clone()
        try:
            ep_group = self._ep_mesh().get_group()
        except (KeyError, TypeError, AttributeError, ValueError) as error:
            raise ValueError("Configured MoE dry-run requires an accessible EP process group") from error
        counts_output, handle = all_to_all_single(
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
        import torch  # pylint: disable=C0415,forbidden-backend-import

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
        import torch  # pylint: disable=C0415,forbidden-backend-import

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
        from torch.nn import functional  # pylint: disable=C0415,forbidden-backend-import

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
        from hyper_parallel.core.utils.communication import (  # pylint: disable=C0415
            differentiable_all_to_all_single,
        )

        try:
            ep_group = self._ep_mesh().get_group()
        except (KeyError, TypeError, AttributeError, ValueError) as error:
            if len(input_splits) == 1 and input_splits == output_splits:
                return input_tensor.clone()
            raise ValueError(
                "Configured MoE dry-run requires an accessible EP process group"
            ) from error
        return differentiable_all_to_all_single(
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
        except (KeyError, TypeError, AttributeError, ValueError):
            return int(self._base.mesh.ep_size), int(self._base.mesh.ep_rank)

    def _ep_mesh(self) -> Any:
        """Return the expert-parallel child mesh from the HyperModels mesh context."""
        expert_mesh = getattr(self._base.mesh, "fsdp_moe_mesh", None)
        if expert_mesh is None:
            raise ValueError("Configured MoE dry-run requires an expert mesh")
        return expert_mesh["ep"]

    def _restore_moe(self) -> None:
        """Restore every MoE instance forward replaced by this scope."""
        if self._original_ep_compute is not None:
            while self._ep_compute_modules:
                patched_module = self._ep_compute_modules.pop()
                patched_module.ep_routed_forward = self._original_ep_compute
            self._original_ep_compute = None
        self._moe_module_names.clear()
        while self._patched_moe_modules:
            module, had_forward, original_forward = self._patched_moe_modules.pop()
            if had_forward:
                module.forward = original_forward
            else:
                del module.forward
