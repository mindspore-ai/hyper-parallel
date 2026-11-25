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
"""MoE shared expert module."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from math import isfinite, sqrt
from typing import Any, Literal, Optional

import torch
from torch import nn
from transformers.activations import ACT2FN
from hyper_parallel.components.checkpoint.weight_conversion import (
    WeightConverter,
    WeightRenaming,
)

from hyper_parallel.components.checkpoint import ConcatenateWithSections
from hyper_parallel.models.replacement import module_replacement
from hyper_parallel.components.modules.mlp.swiglu_mlp import (
    SwiGLUMLPModule, _validate_mlp_config, local_swiglu, _activation_settings,
)
from hyper_parallel.components.modules.module_family import ModuleFamily, build_family, register_module_family


class LinearWithMatmul(nn.Linear):
    """Aligned with ColumnParallelLinear (single device, no parallelism).

    name: ColumnParallelLinear
    forward path: LinearWithGradAccumulationAndAsyncCommunication.forward
      → output = torch.matmul(input, weight.t())
      → output = output + bias  (bias explicitly added separately, not addmm fused)

    Used for: linear_fc1, linear_fc2

    skip_bias_add=True: don't add bias, return (output, bias) for external addition.
    skip_bias_add=False: bias explicitly added in forward, return (output, None).

    NPU bf16 note: matmul runs in @custom_fwd autograd Function context,
    NPU automatically selects the same kernel for non-contiguous weight.t() as weight.t().contiguous().
    calling directly in forward does not trigger this behavior, so explicit .contiguous() is needed.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        skip_bias_add: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize the linear parameters and bias-return behavior."""
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.skip_bias_add = skip_bias_add

    def forward(  # pylint: disable=redefined-builtin
        self, input: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply matmul and optionally return bias for external addition.

        Args:
            input: Activations ending in in_features.
        """
        output = torch.matmul(input, self.weight.t().contiguous())
        if self.skip_bias_add:
            return output, self.bias
        if self.bias is not None:
            output = output + self.bias
        return output, None


class SharedExpertModule(nn.Module):
    """Transformers-compatible parameter-owning MoE shared expert.

    Args:
        module: Source Transformers shared-expert module.
        module_fqn: Fully qualified name supplied by the replacement framework.
        context: Additional replacement context.
    """

    @staticmethod
    def _separate_projections(module: nn.Module) -> tuple[nn.Linear, nn.Linear, nn.Linear]:
        """Validate and return separate source projections."""
        source_linears = (module.gate_proj, module.up_proj, module.down_proj)
        if any(not isinstance(layer, nn.Linear) for layer in source_linears):
            raise TypeError("SharedExpertModule source projections must be nn.Linear instances")
        gate_proj, up_proj, down_proj = source_linears
        SwiGLUMLPModule._validate_projection_contract(gate_proj, up_proj, down_proj)  # pylint: disable=protected-access
        if gate_proj.in_features != up_proj.in_features:
            raise ValueError("SharedExpertModule gate and up projections must have the same input size")
        if gate_proj.out_features != up_proj.out_features:
            raise ValueError("SharedExpertModule gate and up projections must have the same output size")
        if down_proj.in_features != gate_proj.out_features:
            raise ValueError("SharedExpertModule down projection input size is incompatible with gate/up")
        if down_proj.out_features != gate_proj.in_features:
            raise ValueError("SharedExpertModule down projection output size is incompatible with gate/up")
        if gate_proj.weight.requires_grad != up_proj.weight.requires_grad:
            raise ValueError("SharedExpertModule gate and up weights must use the same training policy")
        if (gate_proj.bias is None) != (up_proj.bias is None):
            raise ValueError("SharedExpertModule gate and up projections must use the same bias policy")
        if gate_proj.bias is not None and gate_proj.bias.requires_grad != up_proj.bias.requires_grad:
            raise ValueError("SharedExpertModule gate and up biases must use the same training policy")
        return gate_proj, up_proj, down_proj

    def _initialize_fused_source(self, module: nn.Module) -> None:
        """Reuse an already fused source module without converting parameters."""
        self.linear_fc1 = module.linear_fc1
        self.linear_fc2 = module.linear_fc2
        if not all(hasattr(layer, "skip_bias_add") for layer in (self.linear_fc1, self.linear_fc2)):
            raise TypeError("fused SharedExpertModule projections must return output and bias separately")
        if not hasattr(module, "activation_func"):
            raise TypeError("fused SharedExpertModule source module is missing activation_func")
        self.activation_func = module.activation_func
        self._return_bias_tuple = getattr(module, "_return_bias_tuple", True)
        self._gate_size = 0
        self._up_size = 0
        self.train(module.training)

    def _initialize_separate_linears(
        self,
        gate_proj: nn.Linear,
        up_proj: nn.Linear,
        down_proj: nn.Linear,
    ) -> None:
        """Construct fused linear layers from separate source projections."""
        self._gate_size = gate_proj.out_features
        self._up_size = up_proj.out_features
        self.linear_fc1 = LinearWithMatmul(
            gate_proj.in_features,
            self._gate_size + self._up_size,
            bias=gate_proj.bias is not None,
            device=gate_proj.weight.device,
            dtype=gate_proj.weight.dtype,
        )
        self.linear_fc1.weight.requires_grad_(gate_proj.weight.requires_grad)
        if gate_proj.bias is not None:
            self.linear_fc1.bias.requires_grad_(gate_proj.bias.requires_grad)
        self.linear_fc2 = LinearWithMatmul(
            down_proj.in_features,
            down_proj.out_features,
            bias=down_proj.bias is not None,
            device=down_proj.weight.device,
            dtype=down_proj.weight.dtype,
        )
        self.linear_fc2.weight.requires_grad_(down_proj.weight.requires_grad)
        if down_proj.bias is not None:
            self.linear_fc2.bias.requires_grad_(down_proj.bias.requires_grad)
        self.linear_fc1.weight = nn.Parameter(
            torch.cat((gate_proj.weight.detach(), up_proj.weight.detach())),
            requires_grad=gate_proj.weight.requires_grad,
        )
        if gate_proj.bias is not None:
            self.linear_fc1.bias = nn.Parameter(
                torch.cat((gate_proj.bias.detach(), up_proj.bias.detach())),
                requires_grad=gate_proj.bias.requires_grad,
            )
        self.linear_fc2.weight = down_proj.weight
        self.linear_fc2.bias = down_proj.bias

    def _initialize_activation(self, module: nn.Module, config: Any) -> None:
        """Select the source-compatible gated activation implementation."""
        hidden_act = getattr(config, "hidden_act", None)
        if hidden_act is None:
            hidden_act = getattr(config, "hidden_activation", None)
        if hidden_act is None:
            hidden_act = getattr(config, "mlp_hidden_act", None)
        activation_func = getattr(module, "act_fn", None)
        if activation_func is None:
            if hidden_act is None:
                hidden_act = "silu"
            activation_func = ACT2FN[hidden_act]
        if hidden_act == "silu" and bool(getattr(config, "use_fused_swiglu", True)):
            self.activation_func = local_swiglu
            return

        def glu(x: torch.Tensor) -> torch.Tensor:
            """Apply the configured gated linear unit.

            Args:
                x: Concatenated gate/up activations.
            """
            gate, value = torch.chunk(x, 2, dim=-1)
            return activation_func(gate) * value

        self.activation_func = glu

    def __init__(
        self,
        *,
        module: nn.Module | None = None,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
        config: Any = None,
        activation_dtype: str | None = None,
        swiglu_limit: float | None = None,
        preserve_source_layout: bool = False,
    ) -> None:
        """Build the high-performance module from a Transformers shared expert."""
        super().__init__()
        if (config is None) == (module is None):
            raise ValueError("provide exactly one of config or module")
        self._ep_context = None
        del module_fqn
        source_config = config if config is not None else getattr(module, "config", None)
        self.num_layers = (context or {}).get("num_layers", getattr(
            source_config, "num_layers", getattr(source_config, "num_hidden_layers", None)))
        self.init_method_std = getattr(source_config, "init_method_std", 0.02)
        if not isinstance(preserve_source_layout, bool):
            raise TypeError("preserve_source_layout must be bool")
        self._preserve_source_layout = preserve_source_layout
        self.swiglu_limit, self.activation_dtype = _activation_settings(
            config if config is not None else module, limit=swiglu_limit, activation_dtype=activation_dtype)
        self.limit = self.swiglu_limit
        if config is not None:
            self._initialize_from_config(config)
            self.reset_parameters()
            return
        config = getattr(module, "config", None)
        self.config = config
        self._source_is_fused = all(
            hasattr(module, name) for name in ("linear_fc1", "linear_fc2")
        )
        self._source_is_separate = all(
            hasattr(module, name) for name in ("gate_proj", "up_proj", "down_proj")
        )
        if not self._source_is_fused and not self._source_is_separate:
            raise TypeError(
                "SharedExpertModule requires linear_fc1/linear_fc2 or gate_proj/up_proj/down_proj"
            )

        if self._source_is_fused:
            if self._preserve_source_layout:
                raise ValueError("preserve_source_layout requires separate gate/up/down source projections")
            self._initialize_fused_source(module)
            return

        gate_proj, up_proj, down_proj = self._separate_projections(module)
        if self._preserve_source_layout:
            self.gate_proj, self.up_proj, self.down_proj = gate_proj, up_proj, down_proj
            if isinstance(getattr(module, "act_fn", None), nn.Module):
                self.act_fn = module.act_fn
            self._initialize_activation(module, config)
            self._set_clamped_activation(config)
            self._return_bias_tuple = False
            self.train(module.training)
            return
        self._initialize_separate_linears(gate_proj, up_proj, down_proj)
        self._initialize_activation(module, config)
        self._set_clamped_activation(config)
        self._return_bias_tuple = False
        self.train(module.training)

    def _initialize_from_config(self, config: Any) -> None:
        """Create the separate gate, up and down projections for a new shared expert."""
        _validate_mlp_config(config)
        if not isinstance(config.return_bias_tuple, bool):
            raise TypeError("return_bias_tuple must be bool")
        self.config = config
        self._gate_size = self._up_size = config.intermediate_size
        self._source_is_fused = False
        self._source_is_separate = True
        self._return_bias_tuple = config.return_bias_tuple
        self.activation_func = partial(local_swiglu, limit=self.swiglu_limit,
                                       activation_dtype=self.activation_dtype)
        if self._preserve_source_layout:
            if config.return_bias_tuple:
                raise ValueError("separate SharedExpert returns a tensor, not a bias tuple")
            self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.bias)
            self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.bias)
            self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.bias)
            return
        self.linear_fc1 = LinearWithMatmul(config.hidden_size, 2 * config.intermediate_size, bias=config.bias)
        self.linear_fc2 = LinearWithMatmul(config.intermediate_size, config.hidden_size, bias=config.bias)
        return

    def _set_clamped_activation(self, config: Any) -> None:
        if self.swiglu_limit > 0 or self.activation_dtype == "float32":
            hidden_act = getattr(config, "hidden_act", getattr(config, "hidden_activation", None))
            if hidden_act not in (None, "silu", "swiglu"):
                raise ValueError("clamped/FP32 SharedExpert requires SwiGLU activation")
            self.activation_func = partial(local_swiglu, limit=self.swiglu_limit,
                                           activation_dtype=self.activation_dtype)

    def make_transforms(self) -> list[WeightRenaming | WeightConverter]:
        """Describe reversible source-checkpoint to high-performance conversion."""
        if self._source_is_fused or self._preserve_source_layout:
            return []
        transforms: list[WeightRenaming | WeightConverter] = [
            WeightConverter(
                source_patterns=["gate_proj.weight", "up_proj.weight"],
                target_patterns="linear_fc1.weight",
                operations=[
                    ConcatenateWithSections(
                        (self._gate_size, self._up_size),
                        dim=0,
                    )
                ],
            ),
            WeightRenaming(
                source_patterns="down_proj.weight",
                target_patterns="linear_fc2.weight",
            ),
        ]
        if self.linear_fc1.bias is not None:
            transforms.insert(
                1,
                WeightConverter(
                    source_patterns=["gate_proj.bias", "up_proj.bias"],
                    target_patterns="linear_fc1.bias",
                    operations=[
                        ConcatenateWithSections(
                            (self._gate_size, self._up_size),
                            dim=0,
                        )
                    ],
                ),
            )
        if self.linear_fc2.bias is not None:
            transforms.append(
                WeightRenaming(
                    source_patterns="down_proj.bias",
                    target_patterns="linear_fc2.bias",
                )
            )
        return transforms

    split_forward_supported = True

    def bind_ep_context(self, context: Any) -> None:
        """Borrow the transport owner without registering another module or parameter tree.

        Args:
            context: EP context scheduling this shared expert's FC1 and FC2.
        """
        if self._ep_context is not None and self._ep_context is not context:
            raise ValueError("shared expert already has an EP communication owner")
        self._ep_context = context

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Consume an overlapped result before repeating module/FSDP hooks.

        The two scheduled stages use normal module calls. The later model call
        only consumes their result, so it must not gather the same weights again.
        """
        if self._ep_context is not None and kwargs.get("stage", "full") == "full":
            hidden = args[0] if args else kwargs.get("hidden_states", kwargs.get("x"))
            output = self._ep_context.take_shared_output(hidden)
            if output is not None:
                return output
        return super().__call__(*args, **kwargs)

    def forward_fc1(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the first projection and activation during dispatch.

        Args:
            hidden_states: Original shared-branch tokens [..., hidden_size].

        Returns:
            Activated intermediate [..., intermediate_size], retaining source dtype policy.
        """
        if self._preserve_source_layout:
            if self.swiglu_limit <= 0 and self.activation_dtype == "input" and hasattr(self, "act_fn"):
                return self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
            intermediate = torch.cat((self.gate_proj(hidden_states), self.up_proj(hidden_states)), dim=-1)
            return self.activation_func(intermediate).to(hidden_states.dtype)
        intermediate, bias = self.linear_fc1(hidden_states)
        if bias is not None:
            intermediate = intermediate + bias
        return self.activation_func(intermediate)

    def forward_fc2(self, intermediate: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply the second projection during combine, preserving the output-bias protocol.

        Args:
            intermediate: This invocation's activated FC1 output [..., intermediate_size].

        Returns:
            Shared output [..., hidden_size], or the source's (output, output_bias) pair.
        """
        if self._preserve_source_layout:
            return self.down_proj(intermediate)
        output, output_bias = self.linear_fc2(intermediate)
        if not self._return_bias_tuple:
            if output_bias is not None:
                output = output + output_bias
            return output
        return output, output_bias

    def forward(
        self,
        hidden_states: torch.Tensor | None = None,
        *,
        x: torch.Tensor | None = None,
        stage: Literal["full", "fc1", "fc2"] = "full",
    ) -> torch.Tensor | tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply the shared-expert MLP while preserving the source return contract.

        Args:
            hidden_states: Source token activations; mutually exclusive with x.
            x: Compatibility alias for hidden_states.
            stage: Full MLP, FC1 plus activation, or FC2 using an already activated intermediate.
                Staged callers use module(..., stage=...) so module/FSDP hooks still run.
        """
        if (hidden_states is None) == (x is None):
            raise ValueError("provide exactly one of hidden_states or x")
        hidden_states = x if hidden_states is None else hidden_states
        if stage == "fc1":
            return self.forward_fc1(hidden_states)
        if stage == "fc2":
            return self.forward_fc2(hidden_states)
        if stage != "full":
            raise ValueError("shared expert stage must be full, fc1 or fc2")
        return self.forward_fc2(self.forward_fc1(hidden_states))

    def reset_parameters(self) -> None:
        """Use Megatron's normal FC1 and depth-scaled output-layer initialization."""
        if isinstance(self.num_layers, bool) or not isinstance(self.num_layers, int) or self.num_layers < 1:
            raise ValueError("shared expert initialization requires num_layers in config or replacement context")
        if (isinstance(self.init_method_std, bool) or not isinstance(self.init_method_std, (int, float))
                or not isfinite(self.init_method_std) or self.init_method_std < 0):
            raise ValueError("init_method_std must be finite and nonnegative")
        output_std = self.init_method_std / sqrt(2.0 * self.num_layers)
        if self._preserve_source_layout:
            projections = ((self.gate_proj, self.init_method_std), (self.up_proj, self.init_method_std),
                           (self.down_proj, output_std))
        else:
            projections = ((self.linear_fc1, self.init_method_std), (self.linear_fc2, output_std))
        for projection, std in projections:
            nn.init.normal_(projection.weight, mean=0.0, std=std)
            if projection.bias is not None:
                nn.init.zeros_(projection.bias)


@module_replacement
def SharedExpert(  # pylint: disable=invalid-name
    *, config: Any = None, module: nn.Module | None = None,
    module_fqn: str = "", context: Mapping[str, Any] | None = None, mesh: Any = None,
    activation_dtype: str | None = None, swiglu_limit: float | None = None,
    preserve_source_layout: bool = False,
) -> nn.Module:
    """Construct a tracked shared expert preserving its source return protocol."""
    return build_family(SharedExpert, config=config, module=module, module_fqn=module_fqn,
                        context=context, mesh=mesh, activation_dtype=activation_dtype,
                        swiglu_limit=swiglu_limit, preserve_source_layout=preserve_source_layout)


register_module_family(ModuleFamily(
    name="shared_expert", baseline=SharedExpert,
    variants={"local": "hyper_parallel.components.modules.moe.shared_expert.SharedExpertModule"},
    default_variant="local",
))
