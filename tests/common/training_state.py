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
"""Full-state migration helpers for checkpoint and distributed continuation acceptance tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributed.checkpoint.state_dict import StateDictOptions

from hyper_parallel.components.checkpoint.conversion_ops import ConcatenateWithSections, Split
from hyper_parallel.components.checkpoint.weight_conversion import Transpose
from hyper_parallel.components.modules.module_family import collect_family_modules
from hyper_parallel.components.optim.mixed_precision_optimizer import (
    FP32_MAIN_PARAM_STATE_KEY, MIXED_PRECISION_OPTIMIZER_STATE_KEY, MixedPrecisionOptimizer,
)
from hyper_parallel.core.dtensor.dtensor import DTensor, distribute_tensor
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer
from hyper_parallel.core.fully_shard.state_dict_utils import (
    get_model_state_dict, get_optim_state_dict, set_model_state_dict, set_optim_state_dict,
)



_SCHEMA_VERSION = 1
_ELEMENT_STATES = frozenset(("exp_avg", "exp_avg_sq", "max_exp_avg_sq", "momentum_buffer"))


@dataclass(frozen=True)
class TensorStateTransform:
    """An explicit reversible index rearrangement for parameters and element states."""

    sources: tuple[str, ...]
    targets: tuple[str, ...]
    operations: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if not self.sources or not self.targets or len(set(self.sources)) != len(self.sources):
            raise ValueError("state transforms require distinct explicit source and target names")
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("state transform targets must be unique")
        if not self.operations and len(self.sources) != len(self.targets):
            raise ValueError("a rename must preserve the number of tensors")
        if any(not isinstance(op, (ConcatenateWithSections, Split, Transpose)) for op in self.operations):
            raise ValueError("optimizer migration accepts only reversible concat/split/transpose or renaming")

    def apply(self, values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply the same coordinate transform without casting or modifying inputs.

        Args:
            values: Full logical tensors keyed by the explicit source FQNs.
        """
        tensors = {name: values[name] for name in self.sources}
        if any(isinstance(value, DTensor) or value.is_meta for value in tensors.values()):
            raise ValueError("migration requires materialized full logical tensors, not local/DTensor shards")
        if len({value.dtype for value in tensors.values()}) != 1:
            raise ValueError("merged tensors must have the same dtype")
        if not self.operations:
            return {target: tensors[source].clone() for source, target in zip(self.sources, self.targets)}
        for operation in self.operations:
            tensors = operation.convert(tensors, list(tensors), list(self.targets))
        if set(tensors) != set(self.targets):
            raise ValueError("conversion produced unexpected target names")
        return {name: tensor.clone() for name, tensor in tensors.items()}

    def inverse(self) -> TensorStateTransform:
        """Return the exact inverse mapping, including optimizer-state coordinates."""
        return TensorStateTransform(self.targets, self.sources,
                                    tuple(op.reverse_op for op in reversed(self.operations)))


def _optimizer_view(model, optimizer):
    """Separate the base optimizer from optional FP32 master-parameter state."""
    base = optimizer.optimizer if isinstance(optimizer, MixedPrecisionOptimizer) else optimizer
    if isinstance(base, ChainedOptimizer):
        if len(base.chained_optimizers) != 1 or base.flatten:
            raise ValueError("full migration currently supports one non-flattened optimizer algorithm")
        base = base.chained_optimizers[0]
    parameters = {}
    for name, parameter in model.named_parameters():
        parameters[parameter] = name
        main = getattr(parameter, "main_param", None)
        if main is not None:
            parameters[main] = name
    view = nn.Module()
    for group in base.param_groups:
        for parameter in group["params"]:
            if parameter not in parameters:
                raise ValueError("optimizer parameters must be identifiable in the final model")
            path = parameters[parameter].split(".")
            owner = view
            for part in path[:-1]:
                if part not in owner._modules:  # pylint: disable=protected-access
                    owner.add_module(part, nn.Module())
                owner = owner.get_submodule(part)
            owner.register_parameter(path[-1], parameter)
    return view, base


def _aliases(model):
    groups = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        groups.setdefault(id(parameter), []).append(name)
    return sorted(sorted(names) for names in groups.values() if len(names) > 1)


def family_state_contracts(model: nn.Module) -> list[dict[str, Any]]:
    """Return semantic state identity independently of direct/source provenance.

    Args:
        model: Final model whose families declare representation and numerical semantics.
    """
    records = []
    for boundary in collect_family_modules(model):
        semantic = getattr(boundary.module, "state_contract", None)
        records.append({"fqns": list(boundary.fqns), "family": boundary.info.resolved.family.name,
                        "semantic": semantic() if semantic is not None else {}})
    return records


def _capture_rng(model):
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    devices = {parameter.device.type for parameter in model.parameters()} - {"cpu", "meta"}
    for device in devices:
        state[device] = torch.get_device_module(device).get_rng_state()
    return state


def capture_full_training_state(
    model: nn.Module, optimizer: Any, *, global_step: int, data_progress: Any, micro_step: int = 0,
) -> dict[str, Any]:
    """Export a complete step-boundary snapshot on every participating rank.

    Args:
        model: Final model after FSDP2; global layout comes from HP state exports.
        optimizer: Adam/AdamW/SGD or the HP fp32-main wrapper around those optimizers.
        global_step: Number of completed optimizer updates.
        data_progress: Caller-owned resumable sampler/dataset progress.
        micro_step: Must be zero; unfinished accumulated gradients are unsupported.
    """
    if global_step < 0 or micro_step != 0 or _has_pending_gradients(model):
        raise ValueError("full migration snapshots require a completed step and cleared gradients")
    view, base = _optimizer_view(model, optimizer)
    if type(base).__name__ not in ("Adam", "AdamW", "SGD"):
        raise ValueError("complete migration supports Adam, AdamW and SGD optimizer algorithms")
    options = StateDictOptions(full_state_dict=True)
    masters = {}
    if isinstance(optimizer, MixedPrecisionOptimizer):
        masters = optimizer.state_dict()[MIXED_PRECISION_OPTIMIZER_STATE_KEY][FP32_MAIN_PARAM_STATE_KEY]
        masters = {name: value.full_tensor() if isinstance(value, DTensor) else value
                   for name, value in masters.items()}
    model_state = get_model_state_dict(model, options=options)
    if any(not isinstance(value, torch.Tensor) for value in model_state.values()):
        raise ValueError("non-Tensor module state requires a dedicated versioned migration converter")
    payload = {
        "model": model_state,
        "optimizer": get_optim_state_dict(view, base, options=options),
        "master_weights": masters,
        "extra_state": {
            "training_state_version": _SCHEMA_VERSION, "global_step": global_step,
            "data_progress": data_progress, "micro_step": 0, "rng": _capture_rng(model),
            "requires_grad": {name: parameter.requires_grad for name, parameter in model.named_parameters(
                remove_duplicate=False)},
            "aliases": _aliases(model),
            "family_contracts": family_state_contracts(model),
            "logical_shapes": {name: tuple(value.shape) for name, value in model_state.items()},
            "optimizer_type": type(base).__name__,
        },
    }
    return deepcopy(payload)


def _has_pending_gradients(model):
    return any(parameter.grad is not None or getattr(parameter, "main_grad", None) is not None
               for parameter in model.parameters())


def _validate_snapshot_tensors(state):
    """Check full-shape provenance before any coordinate conversion."""
    extra = state["extra_state"]
    if set(extra["logical_shapes"]) != set(state["model"]):
        raise ValueError("full snapshot must declare global shapes for every model tensor")
    for name, shape in extra["logical_shapes"].items():
        value = state["model"][name]
        if isinstance(value, DTensor) or value.is_meta or tuple(value.shape) != tuple(shape):
            raise ValueError(f"snapshot must contain full materialized logical tensor {name!r}")
    for name, values in state["optimizer"]["state"].items():
        shape = state["model"][name].shape
        for key, value in values.items():
            if key in _ELEMENT_STATES and (not isinstance(value, torch.Tensor) or value.shape != shape):
                raise ValueError(f"optimizer element state {name!r}/{key!r} has an invalid logical shape")
    _validate_alias_values(state)


def _validate_alias_values(state):
    for group in state["extra_state"]["aliases"]:
        first = state["model"][group[0]]
        for name in group[1:]:
            value = state["model"][name]
            if value.dtype != first.dtype or value.shape != first.shape or not torch.equal(value, first):
                raise ValueError("shared parameter aliases require identical converted values and representation")


def _validate_snapshot(state):
    """Check the saved training-state schema before restoring tensors and metadata."""
    extra = state["extra_state"]
    if extra.get("training_state_version") != _SCHEMA_VERSION or extra.get("micro_step") != 0:
        raise ValueError("unsupported training-state version or unfinished gradient accumulation")
    groups = state["optimizer"]["param_groups"]
    names = [name for group in groups for name in group["params"]]
    if any(not isinstance(name, str) for name in names) or len(set(names)) != len(names):
        raise ValueError("optimizer state must use proven, unique parameter FQNs, not unbound saved IDs")
    if set(state["optimizer"]["state"]) - set(names):
        raise ValueError("optimizer state contains parameters outside its param groups")
    if extra.get("optimizer_type") not in ("Adam", "AdamW", "SGD"):
        raise ValueError("complete migration supports Adam, AdamW and SGD optimizer algorithms")
    _validate_snapshot_tensors(state)


def _group_index(groups):
    return {name: index for index, group in enumerate(groups) for name in group["params"]}


def _matching_steps(first, states):
    return all(torch.equal(torch.as_tensor(first), torch.as_tensor(item["step"])) for item in states)


def _convert_moments(state, transform):
    """Transform optimizer moments into the target logical parameter representation."""
    states = [state.get(name, {}) for name in transform.sources]
    if all(not item for item in states):
        return {name: {} for name in transform.targets}
    keys = set(states[0])
    if any(set(item) != keys for item in states):
        raise ValueError("merged parameters must have identical initialized optimizer-state fields")
    unknown = keys - _ELEMENT_STATES - {"step"}
    if unknown:
        raise ValueError(f"unsupported/factored optimizer states {sorted(unknown)}")
    converted = {name: {} for name in transform.targets}
    for key in keys:
        if key == "step":
            first = states[0][key]
            if not _matching_steps(first, states):
                raise ValueError("merged parameters have different optimizer steps")
            for target in transform.targets:
                converted[target][key] = deepcopy(first)
        else:
            values = transform.apply({name: state[name][key] for name in transform.sources})
            for target, value in values.items():
                converted[target][key] = value
    return converted


def _rewrite_groups(groups, transform):
    """Remap optimizer group membership and ordering to converted parameters."""
    index = _group_index(groups)
    owners = {index.get(name) for name in transform.sources}
    if owners == {None}:
        return
    if None in owners or len(owners) != 1:
        raise ValueError("cannot merge frozen/unoptimized parameters or parameters from different optimizer groups")
    group = groups[next(iter(owners))]
    named = group.get("param_names")
    if named is not None and named != group["params"]:
        raise ValueError("named optimizer groups must use their canonical parameter FQNs")
    replacement = []
    installed = False
    for name in group["params"]:
        if name in transform.sources:
            if not installed:
                replacement.extend(transform.targets)
                installed = True
        else:
            replacement.append(name)
    group["params"] = replacement
    if named is not None:
        group["param_names"] = replacement.copy()


def _convert_parameter_metadata(extra, transform):
    """Convert parameter metadata alongside the corresponding model weights."""
    frozen = extra["requires_grad"]
    flags = {frozen[name] for name in transform.sources if name in frozen}
    if len(flags) > 1:
        raise ValueError("cannot merge parameters with different requires_grad states")
    for name in transform.sources:
        frozen.pop(name, None)
    if flags:
        frozen.update(dict.fromkeys(transform.targets, flags.pop()))
    aliases = []
    for group in extra["aliases"]:
        if set(group) & set(transform.sources):
            if len(transform.sources) != 1 or len(transform.targets) != 1:
                raise ValueError("packed transforms of shared parameters require an explicit alias migration")
            group = [transform.targets[0] if name == transform.sources[0] else name for name in group]
        aliases.append(group)
    extra["aliases"] = sorted(sorted(group) for group in aliases)


def convert_full_training_state(
    state: Mapping[str, Any], transforms: Sequence[TensorStateTransform],
) -> dict[str, Any]:
    """Convert weights, moments, main weights and metadata without touching the source.

    Args:
        state: Full logical, FQN-keyed checkpoint at an optimizer-step boundary.
        transforms: Ordered explicit coordinate transforms; source keys must be disjoint.
    """
    _validate_snapshot(state)
    result = deepcopy(dict(state))
    consumed = set()
    for transform in transforms:
        if consumed & set(transform.sources):
            raise ValueError("a parameter cannot be consumed by two training-state transforms")
        consumed.update(transform.sources)
        collisions = set(transform.targets) & (set(result["model"]) - set(transform.sources))
        if collisions:
            raise ValueError(f"converted parameter keys collide with existing keys {sorted(collisions)}")
        model_values = transform.apply(result["model"])
        moments = _convert_moments(result["optimizer"]["state"], transform)
        masters = result["master_weights"]
        present = set(transform.sources) & set(masters)
        if present and present != set(transform.sources):
            raise ValueError("every merged parameter must have the same fp32-main-weight contract")
        master_values = transform.apply(masters) if present else {}
        _rewrite_groups(result["optimizer"]["param_groups"], transform)
        _convert_parameter_metadata(result["extra_state"], transform)
        for name in transform.sources:
            result["model"].pop(name)
            result["optimizer"]["state"].pop(name, None)
            masters.pop(name, None)
        result["model"].update(model_values)
        shapes = result["extra_state"]["logical_shapes"]
        for name in transform.sources:
            shapes.pop(name)
        shapes.update({name: tuple(value.shape) for name, value in model_values.items()})
        result["optimizer"]["state"].update({name: value for name, value in moments.items() if value})
        masters.update(master_values)
    _validate_snapshot(result)
    return result


def _validate_restore_model(model, state):
    """Reject representation, shape, freeze and alias mismatches before mutation."""
    extra = state["extra_state"]
    actual = {name: parameter.requires_grad for name, parameter in model.named_parameters(remove_duplicate=False)}
    if actual != extra["requires_grad"] or _aliases(model) != extra["aliases"]:
        raise ValueError("target model freeze/alias contract differs from the converted checkpoint")
    if family_state_contracts(model) != extra["family_contracts"]:
        raise ValueError("target family representation/probability semantics differ from the checkpoint")
    model_state = model.state_dict()
    if set(model_state) != set(state["model"]):
        raise ValueError("target model has different parameter/buffer keys")
    for name, parameter in model_state.items():
        source = state["model"][name]
        if parameter.shape != source.shape or parameter.dtype != source.dtype:
            raise ValueError(f"target logical shape/dtype mismatch at {name!r}")


def _validate_restore_optimizer(model, optimizer, state):
    """Resolve the current identity map and validate group/main-weight contracts."""
    extra = state["extra_state"]
    view, base = _optimizer_view(model, optimizer)
    if type(base).__name__ != extra["optimizer_type"]:
        raise ValueError("optimizer algorithm differs from the converted checkpoint")
    expected_groups = get_optim_state_dict(view, base)["param_groups"]
    actual_groups = [set(group["params"]) for group in state["optimizer"]["param_groups"]]
    if [set(group["params"]) for group in expected_groups] != actual_groups:
        raise ValueError("target optimizer parameter groups differ from the converted checkpoint")
    expected_masters = (optimizer.state_dict()[MIXED_PRECISION_OPTIMIZER_STATE_KEY][FP32_MAIN_PARAM_STATE_KEY]
                        if isinstance(optimizer, MixedPrecisionOptimizer) else {})
    if set(expected_masters) != set(state["master_weights"]):
        raise ValueError("target fp32-main-weight contract differs from the converted checkpoint")
    for name, master in expected_masters.items():
        source = state["master_weights"][name]
        if source.shape != master.shape or source.dtype != torch.float32:
            raise ValueError(f"invalid fp32 main weight at {name!r}")
    return view, base, expected_masters


def _restore_master_weights(optimizer, state, expected_masters):
    """Shard full fp32 weights into the wrapper's existing main parameters."""
    if state["master_weights"]:
        raw = optimizer.optimizer.state_dict()
        converted_masters = {
            name: distribute_tensor(value, expected_masters[name].device_mesh, expected_masters[name].placements)
            if isinstance(expected_masters[name], DTensor) else value
            for name, value in state["master_weights"].items()
        }
        raw[MIXED_PRECISION_OPTIMIZER_STATE_KEY] = {FP32_MAIN_PARAM_STATE_KEY: converted_masters}
        optimizer.load_state_dict(raw)


def restore_full_training_state(model: nn.Module, optimizer: Any, state: Mapping[str, Any]) -> dict[str, Any]:
    """Restore through the main HP FSDP2 state APIs after all validation succeeds.

    Args:
        model: Final model on the target mesh, after FSDP2.
        optimizer: Optimizer created from final parameters.
        state: Converted full snapshot; must be available on all participating ranks.
    """
    _validate_snapshot(state)
    extra = state["extra_state"]
    _validate_restore_model(model, state)
    view, base, expected_masters = _validate_restore_optimizer(model, optimizer, state)
    options = StateDictOptions(full_state_dict=True)
    set_model_state_dict(model, state["model"], options=options, assign=False)
    set_optim_state_dict(view, base, state["optimizer"], options=options)
    _restore_master_weights(optimizer, state, expected_masters)
    rng = extra["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"].cpu())
    for device in set(rng) - {"python", "numpy", "torch"}:
        torch.get_device_module(device).set_rng_state(rng[device].cpu())
    return deepcopy(extra)


def save_full_training_state(checkpointer: Any, path: str, state: Mapping[str, Any]) -> None:
    """Persist a migration result using the existing HP DCP writer.

    Args:
        checkpointer: Existing DistributedCheckpointer instance.
        path: New checkpoint directory; callers keep the source checkpoint read-only.
        state: Complete converted snapshot, including rank-local progress and RNG.
    """
    _validate_snapshot(state)
    checkpointer.save(path, dict(state), global_step=state["extra_state"]["global_step"])


def load_full_training_state(checkpointer: Any, path: str, skeleton: Mapping[str, Any]) -> dict[str, Any]:
    """Read a full migration result into an explicit tensor/state skeleton.

    Args:
        checkpointer: Existing DistributedCheckpointer instance.
        path: Complete converted checkpoint directory.
        skeleton: Expected full model/optimizer/master tensor schema, with initialized state slots.
    """
    state = checkpointer.load(path, deepcopy(dict(skeleton)), extra_state_skeleton=skeleton["extra_state"])
    _validate_snapshot(state)
    return state
