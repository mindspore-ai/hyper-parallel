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
"""Optimizer construction, local views and checkpoint protocol for VeOmni."""

import math
from copy import copy
from functools import wraps
from importlib import import_module
from inspect import Parameter, signature
from typing import Any

import torch
from torch.distributed.checkpoint.state_dict import _flatten_optim_state_dict
from torch.distributed.tensor import DTensor as TorchDTensor
from torch.utils._pytree import tree_map

from hyper_parallel import DTensor, SkipDTensorDispatch
from hyper_parallel.core.fully_shard.api import HSDPModule
from hyper_parallel.core.fully_shard.state_dict_utils import (
    get_optim_state_dict,
    set_optim_state_dict,
)
from hyper_parallel.integration.torch.dtensor import from_torch_view, to_torch_view
from hyper_parallel.integration.torch.fsdp import _reshard_for_checkpoint

_MUON_NS_KEYS = (
    "ns_implementation",
    "ns_coefficients",
    "ns_epsilon",
    "ns_steps",
    "ns_variant",
    "gram_ns_reset_iterations",
)
_MISSING_CHECKPOINT_STRING = "__hyper_muon_not_loaded__"
_MISSING_CHECKPOINT_INTEGER = -(2**63)


def _missing_checkpoint_option(value: Any) -> Any:
    """Preserve the DCP template structure without supplying usable defaults."""
    if isinstance(value, (list, tuple)):
        return type(value)(_missing_checkpoint_option(item) for item in value)
    if isinstance(value, float):
        return float("nan")
    if isinstance(value, int):
        return _MISSING_CHECKPOINT_INTEGER
    return _MISSING_CHECKPOINT_STRING


def _is_missing_checkpoint_option(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return any(_is_missing_checkpoint_option(item) for item in value)
    return (
        value == _MISSING_CHECKPOINT_STRING
        or value == _MISSING_CHECKPOINT_INTEGER
        or isinstance(value, float)
        and math.isnan(value)
    )


def build_optimizer_state_binding(native_class: type) -> tuple:
    """Keep VeOmni's Stateful protocol while marking load-only option templates."""

    class OptimizerState(native_class):
        """Distinguish DCP-loaded Muon settings from missing template entries."""

        def state_dict(self) -> dict:
            state = super().state_dict()
            if self._load and isinstance(self.optimizer, HyperOptimizerAdapter):
                return self.optimizer.dcp_load_template(state)
            return state

    return native_class, OptimizerState


def uses_hyper(model: torch.nn.Module) -> bool:
    """Recognize actual Hyper parameters rather than a global backend flag."""
    return any(isinstance(module, HSDPModule) for module in model.modules()) or any(
        isinstance(parameter, DTensor) for parameter in model.parameters()
    )


def _canonical_muon_option(value: Any) -> Any:
    """Compare checkpoint sequences independently of list/tuple transport types."""
    if isinstance(value, (list, tuple)):
        return tuple(_canonical_muon_option(item) for item in value)
    return value


def _fill_checkpoint_templates(state: dict, optimizer: Any, parameters: dict) -> None:
    """Describe lazily created states without inserting them into the optimizer."""
    for group in state["param_groups"]:
        if getattr(optimizer, "is_muon", False):
            # Native DCP loads only fields present in the destination template.
            group.setdefault("step", 0)
        for name in group["params"]:
            if name not in state["state"]:
                state["state"][name] = _empty_optimizer_state(
                    optimizer, parameters[name], group
                )


def _empty_optimizer_state(optimizer: Any, parameter: Any, group: dict) -> dict:
    """Create a temporary Muon or AdamW checkpoint destination."""
    local = (
        parameter.to_local()
        if isinstance(parameter, (DTensor, TorchDTensor))
        else parameter
    )
    if getattr(optimizer, "is_muon", False):
        return {"momentum_buffer": torch.zeros_like(local)}
    if isinstance(optimizer, torch.optim.AdamW):
        result = {
            "step": torch.zeros((), device=local.device),
            "exp_avg": torch.zeros_like(local),
            "exp_avg_sq": torch.zeros_like(local),
        }
        if group.get("amsgrad", False):
            result["max_exp_avg_sq"] = torch.zeros_like(local)
        return result
    raise NotImplementedError("No checkpoint template for this optimizer.")


def _state_tensor_with_parameter_layout(value: Any, parameter: Any) -> Any:
    """Attach the owning parameter's layout to an optimizer state tensor."""
    if not isinstance(value, torch.Tensor) or value.ndim == 0:
        return value
    if isinstance(parameter, TorchDTensor):
        local = value.to_local() if isinstance(value, TorchDTensor) else value
        return TorchDTensor.from_local(
            local,
            parameter.device_mesh,
            parameter.placements,
            run_check=False,
            shape=parameter.shape,
            stride=parameter.stride(),
        )
    if not isinstance(parameter, DTensor):
        return value
    return DTensor.from_local(
        value,
        parameter.device_mesh,
        parameter.placements,
        shape=parameter.shape,
        stride=parameter.layout.tensor_stride,
    )


def _optimizer_leaf_from_flat(state: dict, optimizer: Any, parameters: dict) -> dict:
    """Select one leaf from VeOmni's flattened FQN checkpoint protocol.

    Use the live groups rather than regrouping by hyperparameter values: Muon
    and AdamW can share learning rates without sharing group semantics.
    """
    names = {id(parameter): name for name, parameter in parameters.items()}
    entries, groups = {}, []
    for group in optimizer.param_groups:
        fqns = [names[id(parameter)] for parameter in group["params"]]
        restored_group = {**group, "params": fqns}
        for fqn in fqns:
            prefix = f"state.{fqn}."
            values = {
                key[len(prefix) :]: value
                for key, value in state.items()
                if key.startswith(prefix)
            }
            if values:
                entries[fqn] = values
        if fqns:
            prefix = f"param_groups.{fqns[0]}."
            restored_group.update(
                {
                    key[len(prefix) :]: value
                    for key, value in state.items()
                    if key.startswith(prefix)
                }
            )
        groups.append(restored_group)
    return {"state": entries, "param_groups": groups}


def _restore_hyper_state_view(value: Any, parameter: DTensor) -> Any:
    """Reuse the live mesh for an already loaded, same-layout DCP shard."""
    return from_torch_view(value, reference=parameter)


def build_optimizer_binding(native_module: Any, *, use_hyper_optimizer: bool) -> tuple:
    """Translate the VeOmni factory call once, then choose the owning optimizer."""
    original_optimizer = getattr(native_module, "build_optimizer", None)
    try:
        optimizer_signature = signature(original_optimizer)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "Unsupported VeOmni veomni.optim.optimizer.build_optimizer callable."
        ) from error
    required = {"model", "optimizer_type", "param_groups"}
    parameters = optimizer_signature.parameters
    if not required.issubset(parameters) or any(
        parameters[name].kind
        not in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
        for name in required
    ):
        raise RuntimeError(
            "Unsupported VeOmni veomni.optim.optimizer.build_optimizer signature: "
            "requires keyword-compatible model, optimizer_type and param_groups parameters."
        )

    @wraps(original_optimizer)
    def build_optimizer(*args: Any, **kwargs: Any) -> Any:
        """Reuse native construction unless actual Hyper parameters are present."""
        bound = optimizer_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = bound.arguments
        model = arguments["model"]
        hyper_parameters = uses_hyper(model)
        if arguments["optimizer_type"] != "muon":
            if not hyper_parameters:
                return original_optimizer(*args, **kwargs)
            if arguments["optimizer_type"] != "adamw":
                raise NotImplementedError(
                    "The adapter currently supports AdamW and Hyper Muon."
                )
            return _build_native_adamw_adapter(
                model, arguments, native_module, original_optimizer
            )
        config = arguments.get("optimizer_config")
        select_hyper = getattr(config, "use_hyper_optimizer", use_hyper_optimizer)
        if not select_hyper:
            if not hyper_parameters:
                return original_optimizer(*args, **kwargs)
            return _build_native_muon_adapter(model, arguments, original_optimizer)
        if arguments["param_groups"] is not None:
            native_module.logger.warning_rank0(
                "build_optimizer(optimizer_type='muon') ignores the provided "
                "param_groups argument; all parameters are re-split by "
                "split_muon_adamw_params."
            )
        return _build_hyper_muon_adapter(model, arguments, native_module)

    return original_optimizer, build_optimizer


class HyperOptimizerAdapter:
    """Expose the existing multi-optimizer/scheduler/DCP protocol around leaves."""

    _is_multi_optimizer = True

    def __init__(self, model: torch.nn.Module, optimizers: dict) -> None:
        """Retain native parameter groups and Hyper-owned optimizer execution."""
        self.model = model
        self.optimizers_dict = optimizers
        self.key_names = list(optimizers)
        self._legacy_muon_checkpoint_options = {}

    def confirm_legacy_muon_checkpoint_options(self, **options: Any) -> None:
        """Explicitly supply historical NS settings absent from an old checkpoint.

        Only missing checkpoint fields use these values. Saved fields always
        take precedence, and every value must match the active Muon configuration.
        The caller must know the historical settings; this API does not infer them.
        """
        if options.keys() - set(_MUON_NS_KEYS):
            raise ValueError(
                "Legacy Muon confirmation accepts NS configuration keys only."
            )
        self._legacy_muon_checkpoint_options = {
            key: _canonical_muon_option(value) for key, value in options.items()
        }

    def dcp_load_template(self, state: dict) -> dict:
        """Poison invariant defaults so partial DCP loads cannot impersonate saved values."""
        result = dict(state)
        for key, value in state.items():
            if key.startswith("param_groups.") and key.rsplit(".", 1)[-1] in (
                *_MUON_NS_KEYS,
                "head_blocks",
                "adjust_lr_fn",
            ):
                result[key] = _missing_checkpoint_option(value)
        return result

    @property
    def param_groups(self) -> list:
        """Expose existing leaf groups for LR reporting."""
        return [
            group
            for optimizer in self.optimizers_dict.values()
            for group in optimizer.param_groups
        ]

    def step(self, *args: Any, **kwargs: Any) -> None:
        """Use Hyper's existing local-dispatch boundary without altering steps."""
        with SkipDTensorDispatch():
            for optimizer in self.optimizers_dict.values():
                optimizer.step(*args, **kwargs)

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients through the owning optimizer implementations."""
        for optimizer in self.optimizers_dict.values():
            optimizer.zero_grad(set_to_none=set_to_none)

    def _checkpoint_state_dict(self, include_uninitialized: bool) -> dict:
        """Serialize leaves with transport layouts and optional loading templates."""
        _reshard_for_checkpoint(self.model)
        result = {}
        parameters = dict(self.model.named_parameters())
        for optimizer in self.optimizers_dict.values():
            synchronize = getattr(optimizer, "_broadcast_state_fused_for_ckpt", None)
            if synchronize is not None:
                synchronize()
            try:
                state = get_optim_state_dict(self.model, optimizer)
                if include_uninitialized:
                    _fill_checkpoint_templates(state, optimizer, parameters)
                for fqn, entries in state["state"].items():
                    parameter = parameters[fqn]
                    for key, value in tuple(entries.items()):
                        entries[key] = _state_tensor_with_parameter_layout(
                            value, parameter
                        )
                # The native OptimizerState restores EP dimensions on flattened
                # FQN keys. Reuse Torch's serializer instead of nesting leaves.
                flattened = _flatten_optim_state_dict(tree_map(to_torch_view, state))
                overlap = result.keys() & flattened.keys()
                if overlap:
                    raise ValueError(
                        f"Optimizer checkpoint keys overlap: {sorted(overlap)}"
                    )
                result.update(flattened)
            finally:
                if synchronize is not None:
                    optimizer.cleanup_synced_state()
        return result

    def state_dict(self) -> dict:
        """Provide dense transient checkpoint views without changing parameters."""
        return self._checkpoint_state_dict(include_uninitialized=True)

    def _sparse_state_dict(self) -> dict:
        """Serialize only initialized state for native checkpoint discovery."""
        return self._checkpoint_state_dict(include_uninitialized=False)

    def _validate_muon_checkpoint(
        self, state_dict: dict, configured: dict, prefix: str
    ) -> None:
        """Complete historical metadata before validating the restore payload."""
        for key, default in (("head_blocks", 1), ("adjust_lr_fn", "match_rms_adamw")):
            if _is_missing_checkpoint_option(state_dict.get(prefix + key)):
                state_dict[prefix + key] = default
        self._complete_muon_ns_options(state_dict, configured, prefix)
        if configured.get("head_blocks", 1) != state_dict.get(
            prefix + "head_blocks", 1
        ):
            raise ValueError(
                "Muon checkpoint head_blocks differs from the configured head splitting."
            )
        if configured.get("adjust_lr_fn", "match_rms_adamw") != state_dict.get(
            prefix + "adjust_lr_fn", "match_rms_adamw"
        ):
            raise ValueError(
                "Muon checkpoint LR policy differs from the configured policy."
            )
        if "ns_implementation" not in configured:
            return
        for key in _MUON_NS_KEYS:
            if prefix + key not in state_dict:
                raise ValueError(f"Muon checkpoint is missing NS configuration: {key}.")
            if _canonical_muon_option(configured[key]) != _canonical_muon_option(
                state_dict[prefix + key]
            ):
                raise ValueError(f"Muon checkpoint NS configuration differs: {key}.")

    def _complete_muon_ns_options(
        self, state_dict: dict, configured: dict, prefix: str
    ) -> None:
        """Use explicit legacy confirmation only for absent checkpoint settings."""
        for key in _MUON_NS_KEYS:
            if prefix + key in state_dict and not _is_missing_checkpoint_option(
                state_dict[prefix + key]
            ):
                continue
            if key in self._legacy_muon_checkpoint_options:
                state_dict[prefix + key] = self._legacy_muon_checkpoint_options[key]
            elif "ns_implementation" in configured:
                raise ValueError(
                    f"Muon checkpoint is missing NS configuration: {key}. "
                    "Explicit historical settings are required for legacy restore."
                )

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore Hyper optimizer states from native DCP transport views."""
        if any(not key.startswith(("state.", "param_groups.")) for key in state_dict):
            raise ValueError("Expected VeOmni flattened optimizer checkpoint keys.")
        state_dict = dict(state_dict)
        parameters = dict(self.model.named_parameters())
        parameter_names = {
            id(parameter): name for name, parameter in parameters.items()
        }
        for optimizer in self.optimizers_dict.values():
            if getattr(optimizer, "is_muon", False):
                for configured in optimizer.param_groups:
                    for parameter in configured["params"]:
                        fqn = parameter_names[id(parameter)]
                        prefix = f"param_groups.{fqn}."
                        self._validate_muon_checkpoint(state_dict, configured, prefix)
            # Construct the payload only after confirmed values have replaced
            # missing-field markers, so validation and restore use the same data.
            payload = _optimizer_leaf_from_flat(state_dict, optimizer, parameters)
            restored = {}
            for fqn, entries in payload["state"].items():
                parameter = parameters[fqn]
                if isinstance(parameter, DTensor):
                    restored[fqn] = tree_map(
                        lambda value, target=parameter: _restore_hyper_state_view(
                            value, target
                        ),
                        entries,
                    )
                elif isinstance(parameter, TorchDTensor):
                    restored[fqn] = entries
                else:
                    restored[fqn] = tree_map(
                        lambda value: (
                            value.to_local()
                            if isinstance(value, TorchDTensor)
                            else value
                        ),
                        entries,
                    )
            set_optim_state_dict(self.model, optimizer, {**payload, "state": restored})
            # DCP materializes transport state on replica peers as well. Keep
            # Hyper's owner-only state contract after restore, as after save.
            cleanup = getattr(optimizer, "cleanup_synced_state", None)
            if cleanup is not None:
                cleanup()


class _NativeMuonAdapter:
    """Run the unchanged native optimizer on shared-storage Torch parameter views."""

    _is_multi_optimizer = True

    def __init__(
        self, model: torch.nn.Module, view_model: torch.nn.Module, optimizer: Any
    ) -> None:
        self.model = model
        self.view_model = view_model
        self.optimizer = optimizer
        self.optimizers_dict = optimizer.optimizers_dict
        self.key_names = optimizer.key_names
        self.parameter_pairs = list(zip(model.parameters(), view_model.parameters()))

    @property
    def param_groups(self) -> list:
        """Expose native groups for the existing LR scheduler."""
        return [
            group
            for leaf in self.optimizers_dict.values()
            for group in leaf.param_groups
        ]

    def step(self) -> None:
        """Refresh views and gradients without replacing live Hyper parameters."""
        for parameter, view in self.parameter_pairs:
            view.data = to_torch_view(parameter)
            grad = getattr(parameter, "main_grad", None)
            if grad is None:
                grad = parameter.grad
            grad = to_torch_view(grad)
            if grad is not None and not isinstance(grad, TorchDTensor):
                grad = TorchDTensor.from_local(
                    grad,
                    view.device_mesh,
                    view.placements,
                    run_check=False,
                    shape=view.shape,
                    stride=view.stride(),
                )
            view.grad = grad
        self.optimizer.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear the live gradients as well as the optimizer-only view gradients."""
        self.model.zero_grad(set_to_none=set_to_none)
        self.optimizer.zero_grad()

    def state_dict(self) -> dict:
        """Reuse the native multi-optimizer DCP protocol and parameter names."""
        return self.optimizer.state_dict()

    def _sparse_state_dict(self) -> dict:
        """Reuse native sparse discovery; no public equivalent is available."""
        return self.optimizer._sparse_state_dict()  # pylint: disable=protected-access

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore the native optimizer state without changing training parameters."""
        self.optimizer.load_state_dict(state_dict)


def _build_native_muon_adapter(
    model: torch.nn.Module, factory_options: dict, factory: Any
) -> _NativeMuonAdapter:
    """Preserve native Muon algorithms; bridge only its Torch DTensor inputs."""
    if getattr(model, "_fqn2spec_info", None):
        raise NotImplementedError(
            "Native Muon with Hyper ExtraParallel layouts needs separate validation."
        )
    memo = {}
    parameter_views = {}

    # Registration dictionaries preserve tied parameters and None placeholders;
    # public iterators omit these entries and cannot construct the same view tree.
    def clone_module(module: torch.nn.Module) -> torch.nn.Module:
        """Clone registrations while retaining shared-storage parameter views."""
        # pylint: disable=protected-access
        if id(module) in memo:
            return memo[id(module)]
        result = copy(module)
        memo[id(module)] = result
        result._parameters = {}
        for name, parameter in module._parameters.items():
            if parameter is None:
                result._parameters[name] = None
                continue
            if id(parameter) not in parameter_views:
                view = torch.nn.Parameter(
                    to_torch_view(parameter), requires_grad=parameter.requires_grad
                )
                for attribute in ("model_name", "_is_ep_replicated_lora"):
                    if hasattr(parameter, attribute):
                        setattr(view, attribute, getattr(parameter, attribute))
                parameter_views[id(parameter)] = view
            result._parameters[name] = parameter_views[id(parameter)]
        result._modules = {
            name: clone_module(child) if child is not None else None
            for name, child in module._modules.items()
        }
        return result

    view_model = clone_module(model)
    options = dict(factory_options)
    options["model"] = view_model
    if options.get("param_groups") is not None:
        options["param_groups"] = [
            dict(group, params=[parameter_views[id(p)] for p in group["params"]])
            for group in options["param_groups"]
        ]
    optimizer = factory(**options)
    return _NativeMuonAdapter(model, view_model, optimizer)


def _extra_parallel_adamw_groups(model, source, names, factory_options, native_module):
    """Partition one input group without duplicating native decay rules."""
    buckets = {name: [] for name in (*names, "non_extra_parallel")}
    for parameter in source["params"]:
        if not parameter.requires_grad:
            continue
        mesh = getattr(parameter, "device_mesh", None)
        dimensions = getattr(mesh, "mesh_dim_names", ()) or ()
        owner = next(
            (name for name in names if f"{name}_fsdp" in dimensions),
            "non_extra_parallel",
        )
        buckets[owner].append(parameter)
    groups = {}
    for name, parameters in buckets.items():
        if not parameters:
            continue
        # VeOmni exposes no public subset factory; reuse its decay grouping.
        subgroups = native_module._make_param_groups_for_subset(  # pylint: disable=protected-access
            model,
            parameters,
            factory_options["weight_decay"],
            factory_options["no_decay_modules"],
            factory_options["no_decay_params"],
        )
        overrides = {
            key: value
            for key, value in source.items()
            if key not in ("params", "weight_decay")
        }
        for group in subgroups:
            group.update(overrides)
        groups[name] = subgroups
    return groups


def _build_native_adamw_adapter(
    model: torch.nn.Module, factory_options: dict, native_module: Any, factory: Any
) -> HyperOptimizerAdapter:
    """Read live mesh metadata and reuse VeOmni's weight-decay grouping rules."""
    # Reuse the native EP eligibility test instead of maintaining a second one.
    if not native_module._should_build_extra_parallel_aware(model):  # pylint: disable=protected-access
        optimizer = factory(**factory_options)
        leaves = (
            optimizer.optimizers_dict
            if getattr(optimizer, "_is_multi_optimizer", False)
            else {"adamw": optimizer}
        )
        return HyperOptimizerAdapter(model, leaves)
    names = native_module.get_parallel_state().extra_parallel_names
    groups = {name: [] for name in (*names, "non_extra_parallel")}
    provided = factory_options.get("param_groups")
    if provided is not None:
        provided = native_module.filter_empty_param_groups(provided)
        if not provided:
            raise ValueError(
                "All optimizer param groups are empty; no trainable parameters to optimize."
            )
    sources = (
        provided if provided is not None else [{"params": list(model.parameters())}]
    )
    for source in sources:
        for name, subgroups in _extra_parallel_adamw_groups(
            model, source, names, factory_options, native_module
        ).items():
            groups[name].extend(subgroups)
    fused = factory_options["fused"]
    leaves = {
        name: torch.optim.AdamW(
            param_groups,
            lr=factory_options["lr"],
            betas=factory_options["betas"],
            eps=factory_options["eps"],
            weight_decay=factory_options["weight_decay"],
            fused=fused,
            foreach=not fused,
        )
        for name, param_groups in groups.items()
        if param_groups
    }
    extra_parallel_groups = {
        name: [parameter for group in param_groups for parameter in group["params"]]
        for name, param_groups in groups.items()
    }
    setattr(model, "_extra_parallel_param_groups", extra_parallel_groups)
    setattr(model, "_ep_replicated_lora_param_ids", _shared_lora_parameter_ids(model))
    return HyperOptimizerAdapter(model, leaves)


def _shared_lora_parameter_ids(model: torch.nn.Module) -> set[int]:
    """Identify shared EP LoRA without imposing Torch's tensor representation."""
    result = set()
    for module in model.modules():
        if not any(
            base.__name__ == "LoraSharedExperts" for base in type(module).__mro__
        ):
            continue
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            if not isinstance(parameter, (DTensor, TorchDTensor)):
                raise TypeError(
                    f"Shared LoRA parameter {name!r} must be sharded before optimizer construction."
                )
            result.add(id(parameter))
    return result


def _infer_muon_head_blocks(model, options, native_muon):
    """Validate head grouping and reuse native head-count inference."""
    head_group_size = options.get("head_group_size", 0)
    if (
        isinstance(head_group_size, bool)
        or not isinstance(head_group_size, int)
        or head_group_size < 0
    ):
        raise ValueError("Muon head_group_size must be a non-negative integer.")
    modules = options.get("head_split_modules") or ()
    if head_group_size and not modules:
        raise ValueError("Muon head splitting requires head_split_modules.")
    return (
        native_muon.infer_head_block_counts(model, head_group_size, modules)
        if head_group_size
        else {}
    )


def _muon_layout_options(
    model: torch.nn.Module, parameters: list, options: dict, native_muon: Any
) -> tuple:
    """Translate head blocks and LR policy into existing Hyper extension points."""
    policy = options.get("adjust_lr_fn", "match_rms_adamw")
    if policy not in ("original", "match_rms_adamw"):
        raise ValueError(f"Unsupported Muon learning-rate policy: {policy!r}.")
    inferred = _infer_muon_head_blocks(model, options, native_muon)
    blocks_by_name = {
        parameter.model_name: inferred.get(parameter.model_name, 1)
        for parameter in parameters
    }
    groups = {}
    singleton_experts = set()
    for parameter in parameters:
        blocks = blocks_by_name[parameter.model_name]
        shape = tuple(parameter.shape)
        if blocks > 1 and (len(shape) != 2 or shape[0] % blocks):
            raise ValueError(
                f"Muon head blocks must evenly divide a 2D parameter: {parameter.model_name}."
            )
        rows, columns = shape[-2:]
        rows //= blocks
        if len(shape) == 3 and shape[1] == 1:
            singleton_experts.add(parameter.model_name)
        rms = 0.2
        if policy == "original":
            # Hyper multiplies by sqrt(max(logical_dims)) * matched_adamw_rms.
            # Encode the native original scale in the group, keeping LR and
            # decoupled weight decay unchanged, including scheduler updates.
            rms = math.sqrt(max(1.0, rows / columns)) / math.sqrt(max(rows, columns))
        group = groups.setdefault(
            (blocks, rms),
            {"params": [], "matched_adamw_rms": rms, "adjust_lr_fn": policy},
        )
        group["params"].append(parameter)
        if blocks > 1:
            group["head_blocks"] = blocks

    def reshape_heads(name: str, matrix: torch.Tensor) -> list:
        if name in singleton_experts:
            # A list of 2D views avoids Hyper's native [E, 1, H] convolution
            # interpretation and preserves VeOmni's E independent matrices.
            return list(matrix.unbind(dim=0))
        blocks = blocks_by_name.get(name, 1)
        return list(matrix.chunk(blocks, dim=0)) if blocks > 1 else [matrix]

    return list(groups.values()), reshape_heads if singleton_experts or any(
        value > 1 for value in blocks_by_name.values()
    ) else None


def _build_muon_adamw_leaves(model, parameters, options, native_module):
    """Keep persistent Torch tables separate from Hyper foreach updates."""
    # The native subset helper owns VeOmni's weight-decay rules.
    groups = native_module._make_param_groups_for_subset(  # pylint: disable=protected-access
        model,
        parameters,
        options["weight_decay"],
        options["no_decay_modules"],
        options["no_decay_params"],
    )
    leaves = {}
    for torch_owned in (False, True):
        owned_groups = []
        for group in groups:
            params = [
                parameter
                for parameter in group["params"]
                if isinstance(parameter, TorchDTensor) == torch_owned
            ]
            if params:
                owned_groups.append({**group, "params": params})
        if owned_groups:
            key = "adamw_torch" if torch_owned else "adamw"
            leaves[key] = torch.optim.AdamW(
                owned_groups,
                lr=options["lr"],
                betas=options["betas"],
                eps=options["eps"],
                fused=options["fused"],
                foreach=not options["fused"],
            )
    return leaves


def _build_hyper_muon_adapter(
    model: torch.nn.Module, optimizer_options: dict, native_optimizer_module: Any
) -> HyperOptimizerAdapter:
    """Construct Hyper Muon using VeOmni's grouping and selected NS callback."""
    # Muon is optional and is loaded only after the real parameter layout is
    # known, following Hyper's existing optimizer configuration boundary.
    optimizer_utils = import_module("hyper_parallel.core.optimizer.utils")
    # Hyper's optional optimizer initializes its logger through this helper.
    optimizer_utils._install_logger_methods()  # pylint: disable=protected-access
    detect_dtensor_backend = import_module(
        "hyper_parallel.core.optimizer.dtensor_compat"
    ).detect_dtensor_backend
    muon_class = import_module("hyper_parallel.core.optimizer.muon").Muon
    for name, parameter in model.named_parameters():
        parameter.model_name = name
    muon_options = dict(optimizer_options.get("muon_kwargs") or {})
    if optimizer_options.get("optimizer_config") is not None:
        muon_options = {
            # Reuse native config translation; no public equivalent exists.
            **native_optimizer_module._collect_muon_kwargs(  # pylint: disable=protected-access
                optimizer_options["optimizer_config"]
            ),
            **muon_options,
        }
    native_muon_module = import_module("veomni.optim.muon")
    muon_params, adamw_params, _, _ = native_muon_module.split_muon_adamw_params(
        model,
        optimizer_options["no_decay_modules"],
        optimizer_options["no_decay_params"],
    )
    if not muon_params:
        raise ValueError("No eligible Muon parameters.")
    muon_groups, reshape_fn = _muon_layout_options(
        model, muon_params, muon_options, native_muon_module
    )
    detect_dtensor_backend([], [{"params": muon_params}])
    coefficients = muon_options.get("ns_coefficients", (3.4445, -4.775, 2.0315))
    steps = muon_options.get("ns_steps", 5)
    callback = None
    implementation = muon_options.get("ns_implementation", "gram_quack")
    if implementation not in ("std", "gram", "gram_quack"):
        raise ValueError(f"Unsupported Muon NS implementation: {implementation!r}.")
    for group in muon_groups:
        group["ns_implementation"] = implementation
        group["gram_ns_reset_iterations"] = tuple(
            muon_options.get("gram_ns_reset_iterations", (2,))
        )
    if implementation != "std":
        run_newton_schulz = native_muon_module.run_newton_schulz

        def run_zeropower(matrix: torch.Tensor, steps: int) -> torch.Tensor:
            """Invoke the selected existing NS kernel with Hyper's callback signature."""
            return run_newton_schulz(
                matrix,
                ns_steps=steps,
                ns_coefficients=coefficients,
                eps=muon_options.get("eps", 1e-7),
                ns_implementation=implementation,
                gram_ns_reset_iterations=muon_options.get(
                    "gram_ns_reset_iterations", (2,)
                ),
            )

        callback = run_zeropower

    muon_lr = (
        optimizer_options["lr"]
        * (25.0 if muon_options.get("adjust_lr_fn") == "original" else 1.0)
        if muon_options.get("lr") is None
        else muon_options["lr"]
    )
    optimizers = {
        "muon": muon_class(
            muon_groups,
            lr=muon_lr,
            momentum=muon_options.get("momentum", 0.95),
            nesterov=muon_options.get("nesterov", True),
            weight_decay=muon_options.get("weight_decay", 0.0),
            ns_variant="custom",
            ns_steps=steps,
            ns_coefficients=[coefficients] * steps,
            ns_epsilon=muon_options.get("eps", 1e-7),
            zeropower_fn=callback,
            reshape_fn=reshape_fn,
        )
    }
    if adamw_params:
        optimizers.update(
            _build_muon_adamw_leaves(
                model, adamw_params, optimizer_options, native_optimizer_module
            )
        )
    return HyperOptimizerAdapter(model, optimizers)
