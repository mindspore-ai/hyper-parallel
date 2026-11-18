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
"""Opt-in, reversible VeOmni bindings owned entirely by HyperParallel.

Only VeOmni module attributes are rebound. Torch exports, tensor types and the
training loop are never replaced. Apply patches before model construction and
remove them after training; concurrent patch changes are unsupported.
"""

import sys
from collections.abc import Callable
from functools import wraps
from importlib import import_module
from inspect import signature
from typing import Any, Optional

import torch

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.fully_shard.api import HSDPModule
from hyper_parallel.integration.torch import (
    DTensorAdapter,
    fully_shard,
    get_model_state_dict,
    set_model_state_dict,
)
from hyper_parallel.integration.veomni.extra_parallel import (
    bind_gradient_divisor,
    native_clipping_view,
)
from hyper_parallel.integration.veomni.optimizer import (
    build_optimizer_binding,
    build_optimizer_state_binding,
    uses_hyper,
)


def _loaded_veomni_modules() -> list:
    """Snapshot loaded VeOmni modules without running lazy module attributes."""
    return [
        module
        for name, module in tuple(sys.modules.items())
        if (name == "veomni" or name.startswith("veomni.")) and module is not None
    ]


@torch.no_grad()
def _copy_parameter(
    parameter: DTensor,
    tensor: torch.Tensor,
    *,
    full_param_name: str,
    dtensor_factory: Callable | None = None,
    parallel_plan: Any = None,
    dtensor_to_cpu: bool = False,
    tensor_is_local_dtensor_shard: bool = False,
) -> None:
    """Use Hyper storage operations inside VeOmni's existing weight reader loop."""
    del dtensor_factory
    adapter = DTensorAdapter(parameter)
    local = adapter.to_local()
    if dtensor_to_cpu and local.device.type != "cpu":
        raise ValueError(
            "Materialize Hyper parameters on CPU before requesting CPU weight loading."
        )
    if parallel_plan is not None:
        tensor = parallel_plan.shard_tensor(tensor, full_param_name, parameter.shape)
    tensor = tensor.to(device=local.device, dtype=local.dtype)
    if tensor_is_local_dtensor_shard:
        adapter.copy_local_(tensor)
    else:
        adapter.copy_from_full_(tensor)


def _build_construction_bindings(
    *, fsdp: bool = True, muon: bool = True, fsdp_config: Any = None
) -> list:
    """Resolve the optional framework APIs and assemble independent bindings."""
    native_optimizer_module = import_module("veomni.optim.optimizer")
    bindings = [
        build_optimizer_binding(native_optimizer_module, hyper_muon=muon)
    ]
    if fsdp or muon:
        checkpoint = import_module("veomni.checkpoint.dcp_checkpointer")
        bindings.append(build_optimizer_state_binding(checkpoint.OptimizerState))
    if not fsdp:
        return bindings
    parallel = import_module("veomni.distributed.torch_parallelize")
    options = getattr(fsdp_config, "hyper_parallel", None)
    if options is not None:
        if fsdp_config.low_precision_reduce_scatter_comm:
            raise ValueError(
                "low_precision_reduce_scatter_comm is not supported by Hyper FSDP."
            )

    vit_policy = (
        _ViTFusionPolicy(fsdp_config)
        if options is not None and options.comm_fusion_vit is not None
        else None
    )
    bindings.append(_build_sharding_binding(parallel, options, vit_policy))
    bindings.append((parallel.FSDPModule, (parallel.FSDPModule, HSDPModule)))
    bindings.extend(_build_checkpoint_bindings(checkpoint))
    if options is not None:
        bindings.append(_build_parallelize_binding(parallel, fsdp_config, vit_policy))
    return bindings


class _ViTFusionPolicy:
    """Track visual ownership only while VeOmni constructs one FSDP tree."""

    def __init__(self, config: Any) -> None:
        self.comm_fusion = config.hyper_parallel.comm_fusion_vit
        self.reshard_after_forward = config.reshard_after_forward
        self.model = None
        self.visual = None
        self.modules = set()

    def prepare(self, model: torch.nn.Module) -> None:
        """Reuse the VLM trainer's tower resolver, including nested HF layouts."""
        vlm = import_module("veomni.trainer.vlm_trainer")
        # VeOmni has no public tower resolver; keep this VLM-specific coupling here.
        visual = vlm._get_vlm_visual_module(getattr(model, "thinker", model))  # pylint: disable=protected-access
        if visual is None:
            raise ValueError(
                "comm_fusion_vit requires a VLM with a recognized visual tower."
            )
        self.model = model
        self.visual = visual
        self.modules = set(visual.modules())

    def clear(self) -> None:
        """Release construction-only references on success or failure."""
        self.model = None
        self.visual = None
        self.modules.clear()


def _build_sharding_binding(
    parallel: Any, options: Any, vit_policy: Optional[_ViTFusionPolicy] = None
) -> tuple:
    """Keep VeOmni's root lifetime and translate only Hyper-specific options."""
    original_shard = parallel.fully_shard

    @wraps(original_shard)
    def shard(module: torch.nn.Module, **kwargs: Any) -> torch.nn.Module:
        """Retain VeOmni's root lifetime while using generic Hyper descriptors."""
        if (
            vit_policy is not None
            and module is vit_policy.model
            and not isinstance(vit_policy.visual, HSDPModule)
        ):
            # Isolate patch embedding / merger leftovers before the mixed VLM root
            # claims them. Existing visual block wrappers remain unchanged.
            shard(
                vit_policy.visual,
                **{**kwargs, "reshard_after_forward": vit_policy.reshard_after_forward},
            )
        # VeOmni omits this argument for the root to request Torch's automatic
        # root lifetime. Hyper must keep that same lifetime for fused loss users.
        kwargs.setdefault("reshard_after_forward", False)
        if options is not None:
            kwargs["comm_fusion"] = options.comm_fusion
            if vit_policy is not None and module in vit_policy.modules:
                kwargs["comm_fusion"] = vit_policy.comm_fusion
        result = fully_shard(module, **kwargs)
        if isinstance(result, HSDPModule) and kwargs.get("mesh") is not None:
            bind_gradient_divisor(result, kwargs["mesh"].size())
        return result

    return original_shard, shard


def _build_checkpoint_bindings(checkpoint: Any) -> list:
    """Select state transport from the actual parameter representation."""
    old_get, old_set = checkpoint.get_model_state_dict, checkpoint.set_model_state_dict

    @wraps(old_get)
    def get_model(model: torch.nn.Module, **kwargs: Any) -> dict:
        """Select the checkpoint boundary from actual parameter representation."""
        if not uses_hyper(model):
            return old_get(model, **kwargs)
        return get_model_state_dict(model, **kwargs)

    @wraps(old_set)
    def set_model(model: torch.nn.Module, model_state_dict: dict, **kwargs: Any) -> Any:
        """Restore native or Hyper state without replacing training parameters."""
        if not uses_hyper(model):
            return old_set(model, model_state_dict=model_state_dict, **kwargs)
        return set_model_state_dict(model, model_state_dict=model_state_dict, **kwargs)

    return [(old_get, get_model), (old_set, set_model)]


def _build_parallelize_binding(
    parallel: Any, config: Any, vit_policy: Optional[_ViTFusionPolicy] = None
) -> tuple:
    """Reuse native wrap discovery and loading, then configure native Hyper handles."""
    original = parallel.parallelize_model_fsdp2
    call_signature = signature(original)

    @wraps(original)
    def parallelize(*args: Any, **kwargs: Any) -> torch.nn.Module:
        bound = call_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = bound.arguments
        # The original routine's optional manual prefetch is one block deep.
        # Configure Hyper handles once below instead of stacking two policies.
        arguments["kwargs"]["enable_forward_prefetch"] = False
        if vit_policy is not None:
            vit_policy.prepare(arguments["model"])
        try:
            result = original(*bound.args, **bound.kwargs)
        finally:
            if vit_policy is not None:
                vit_policy.clear()
        result.set_reshard_after_backward(config.reshard_after_backward)
        interval = config.hyper_parallel.reduce_comm_interval
        for module in result.modules():
            if isinstance(module, HSDPModule):
                module.set_reduce_comm_interval(interval)
        targets = {
            name: module
            for name, module in result.named_modules()
            if "_fsdp_modules" in vars(module)
        }
        blocks = [
            targets[name]
            for name in parallel.sort_fqn_by_submodule_first(list(targets))
        ]
        _configure_block_prefetch(blocks, config)
        return result

    return original, parallelize


def _configure_block_prefetch(blocks: list, config: Any) -> None:
    """Use VeOmni's existing per-block managed units for Hyper prefetch targets."""
    depth = config.hyper_parallel.forward_prefetch_depth
    backward_depth = config.hyper_parallel.backward_prefetch_depth

    def units(targets: list) -> list:
        """Keep the model builder's managed units in their existing order."""
        result = []
        for block in targets:
            # VeOmni's block builder exposes no public managed-unit accessor.
            for module in reversed(block._fsdp_modules):  # pylint: disable=protected-access
                if not any(module is existing for existing in result):
                    result.append(module)
        return result

    for index, block in enumerate(blocks):
        forward = (
            units(blocks[index + 1 : index + depth + 1])
            if config.forward_prefetch
            else []
        )
        block.set_modules_to_forward_prefetch(forward)
        block.set_modules_to_backward_prefetch(
            units(blocks[max(0, index - backward_depth) : index][::-1])
        )


class VeOmniAdapter:
    """Own explicit function bindings and restore aliases imported while active."""

    def __init__(
        self, *, fsdp: bool = True, muon: bool = True, fsdp_config: Any = None
    ) -> None:
        """Create an inactive adapter without importing VeOmni."""
        self._replacements: list[tuple[Any, Any]] = []
        self._bound_targets: tuple[str, ...] = ()
        self.fsdp = fsdp
        self.muon = muon
        self.fsdp_config = fsdp_config

    @property
    def is_patched(self) -> bool:
        """Return whether this instance owns active VeOmni bindings."""
        return bool(self._replacements)

    @property
    def bound_targets(self) -> tuple[str, ...]:
        """Return the VeOmni module attributes replaced by this adapter."""
        return self._bound_targets

    def patch(self) -> "VeOmniAdapter":
        """Apply Hyper construction, loading, clipping and checkpoint bindings.

        Returns:
            This adapter. Repeated application is idempotent.

        Raises:
            RuntimeError: Another adapter or an unsupported VeOmni API is present.
        """
        if self.is_patched:
            return self
        if not (self.fsdp or self.muon):
            return self
        replacements = (
            [self._build_loading_binding(), self._build_clipping_binding()]
            if self.fsdp
            else []
        )
        replacements.extend(
            _build_construction_bindings(
                fsdp=self.fsdp, muon=self.muon, fsdp_config=self.fsdp_config
            )
        )
        for original, _ in replacements:
            self._check_binding_owner(original)
        for _, wrapper in replacements:
            if callable(wrapper):
                setattr(wrapper, "_hyper_veomni_adapter", self)
        # Build and validate bindings before the first mutation. Already-imported
        # aliases and subsequent imports from these definitions see one wrapper.
        self._bound_targets = self._replace_module_references(replacements)
        self._replacements = replacements
        return self

    def _build_loading_binding(self) -> tuple:
        """Adapt only Hyper shard writes inside the original loading pipeline."""
        # VeOmni has no public per-shard write hook; keep this coupling local.
        # pylint: disable=protected-access
        loading = import_module("veomni.models.checkpoint.weights")
        original_load = loading._dispatch_parameter
        load_signature = signature(original_load)
        if not {"module", "name", "tensor"}.issubset(load_signature.parameters):
            raise RuntimeError("Unsupported VeOmni parameter dispatcher signature.")

        @wraps(original_load)
        def dispatch_parameter(*args: Any, **kwargs: Any) -> Any:
            if not self.is_patched:
                return original_load(*args, **kwargs)
            bound = load_signature.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = bound.arguments
            module, local_name = loading._find_submodule(
                arguments["module"], arguments["name"]
            )
            parameter = module._parameters[local_name]
            if not isinstance(parameter, DTensor):
                return original_load(*args, **kwargs)
            _copy_parameter(
                parameter,
                arguments["tensor"],
                full_param_name=arguments["name"],
                dtensor_factory=arguments.get("dtensor_factory"),
                parallel_plan=arguments.get("parallel_plan"),
                dtensor_to_cpu=arguments.get("dtensor_to_cpu", False),
                tensor_is_local_dtensor_shard=arguments.get(
                    "tensor_is_local_dtensor_shard", False
                ),
            )
            return None

        return original_load, dispatch_parameter

    def _build_clipping_binding(self) -> tuple:
        """Reuse VeOmni norm reduction and scaling on local Hyper gradient views."""
        clipping = import_module("veomni.distributed.fsdp2.clip_grad_norm")
        original_clip = clipping.clip_grad_norm
        clip_signature = signature(original_clip)
        if not {"model", "max_norm"}.issubset(clip_signature.parameters):
            raise RuntimeError("Unsupported VeOmni gradient clipping signature.")

        @wraps(original_clip)
        def clip_grad_norm(*args: Any, **kwargs: Any) -> Any:
            if not self.is_patched:
                return original_clip(*args, **kwargs)
            bound = clip_signature.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = bound.arguments
            parameters = arguments["model"].parameters()
            if not any(isinstance(parameter, DTensor) for parameter in parameters):
                return original_clip(*args, **kwargs)
            return original_clip(
                native_clipping_view(arguments["model"]),
                arguments["max_norm"],
                arguments.get("norm_type", 2.0),
                arguments.get("error_if_nonfinite", False),
                arguments.get("foreach"),
            )

        return original_clip, clip_grad_norm

    @staticmethod
    def _check_binding_owner(function: Callable) -> None:
        if getattr(function, "_hyper_veomni_adapter", None) is not None:
            raise RuntimeError("A VeOmni Hyper adapter is already active.")

    @staticmethod
    def _replace_module_references(
        replacements: list[tuple[Any, Any]],
    ) -> tuple[str, ...]:
        """Replace identities only inside the framework, never inside Torch."""
        targets = []
        for module in _loaded_veomni_modules():
            for name, value in tuple(vars(module).items()):
                for original, replacement in replacements:
                    if value is original:
                        setattr(module, name, replacement)
                        targets.append(f"{module.__name__}.{name}")
                        break
        return tuple(targets)

    def unpatch(self) -> None:
        """Restore this adapter's bindings, including aliases imported later.

        Binding changes made by other code are not overwritten. References
        captured in application locals are outside the module-binding contract.
        """
        self._replace_module_references(
            [(wrapper, original) for original, wrapper in self._replacements]
        )
        self._replacements = []
        self._bound_targets = ()

    def __enter__(self) -> "VeOmniAdapter":
        """Apply bindings for one non-nested application scope."""
        if self.is_patched:
            raise RuntimeError(
                "The adapter is already patched; nested contexts are unsupported."
            )
        return self.patch()

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Restore framework bindings even when the application raises."""
        self.unpatch()


_ADAPTER = VeOmniAdapter()


def apply_hyper_parallel_patch(
    model_config: Any = None, *, for_inference: bool = False
) -> VeOmniAdapter:
    """Apply Hyper bindings once, before constructing VeOmni runtimes."""
    if model_config is not None:
        config = model_config.accelerator.fsdp_config
        fsdp = config.fsdp_mode == "hyper_fsdp2"
        muon = model_config.optimizer.type == "hyper_muon"
        if for_inference and (fsdp or muon):
            raise ValueError("HyperParallel VeOmni integration supports training only.")
        config = config if fsdp else None
        if _ADAPTER.is_patched and (_ADAPTER.fsdp, _ADAPTER.muon) != (fsdp, muon):
            raise RuntimeError(
                "Remove the existing patch before changing backend selection."
            )
        if _ADAPTER.is_patched and _ADAPTER.fsdp_config != config:
            raise RuntimeError(
                "Remove the existing patch before changing FSDP options."
            )
        _ADAPTER.fsdp, _ADAPTER.muon = fsdp, muon
        _ADAPTER.fsdp_config = config
        if not (fsdp or muon):
            return _ADAPTER
    return _ADAPTER.patch()


def remove_hyper_parallel_patch() -> None:
    """Restore VeOmni bindings owned by the default adapter."""
    _ADAPTER.unpatch()
