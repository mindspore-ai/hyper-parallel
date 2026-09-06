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
"""NPU grouped routed-expert module."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import
from transformers.activations import ACT2FN
from hyper_parallel.components.checkpoint.weight_conversion import (
    Transpose,
    WeightConverter,
    WeightRenaming,
)

from hyper_parallel.models.replacement import module_replacement
from hyper_parallel.components.functional import (
    grouped_matmul,
    moe_token_permute,
    moe_token_unpermute,
    swiglu,
)


@module_replacement
class GroupedExperts(nn.Module):
    """Transformers-compatible NPU grouped routed experts.

    Holds gate_up_proj/up_proj, down_proj (and optional bias1, bias2) as nn.Parameters.
    Computation logic is handled by _grouped_gemm_expert_forward.

    Args:
        module: Source Transformers experts module.
        module_fqn: Fully qualified name supplied by the replacement framework.
        context: Additional replacement context.
    """

    @staticmethod
    def _source_parameters(module: nn.Module, source_fc1_name: str) -> tuple[nn.Parameter, nn.Parameter]:
        """Validate and return source expert weight parameters."""
        missing = [name for name in (source_fc1_name, "down_proj") if not hasattr(module, name)]
        if missing:
            raise TypeError(f"GroupedExperts source module is missing: {missing}")
        source_gate_up = getattr(module, source_fc1_name)
        if not isinstance(source_gate_up, nn.Parameter):
            raise TypeError(f"GroupedExperts requires {source_fc1_name} to be an nn.Parameter")
        if not isinstance(module.down_proj, nn.Parameter):
            raise TypeError("GroupedExperts requires down_proj to be an nn.Parameter")
        return source_gate_up, module.down_proj

    @staticmethod
    def _source_dimensions(
        module: nn.Module,
        config: Any,
        source_gate_up: nn.Parameter,
        source_down: nn.Parameter,
        gated_linear_unit: bool,
    ) -> tuple[int, int, int]:
        """Infer expert count and projection dimensions from the source module."""
        num_experts = getattr(module, "num_local_experts", getattr(module, "num_experts", None))
        if num_experts is None and config is not None:
            num_experts = getattr(config, "n_routed_experts", getattr(config, "num_experts", None))
        hidden_size = getattr(module, "hidden_size", getattr(module, "hidden_dim", None))
        if hidden_size is None and config is not None:
            hidden_size = getattr(config, "hidden_size", None)
        intermediate_size = getattr(module, "intermediate_size", getattr(module, "intermediate_dim", None))
        if intermediate_size is None and config is not None:
            intermediate_size = getattr(config, "moe_intermediate_size", getattr(config, "intermediate_size", None))
        source_is_transposed = getattr(module, "is_transposed", None)
        if not gated_linear_unit and source_gate_up.dim() == 3:
            hidden_size = GroupedExperts._infer_ungated_hidden_size(
                source_gate_up, source_down, source_is_transposed, intermediate_size, hidden_size
            )
        if num_experts is None or hidden_size is None or intermediate_size is None:
            raise ValueError("GroupedExperts requires expert count, hidden size, and intermediate size")
        return num_experts, hidden_size, intermediate_size

    @staticmethod
    def _infer_ungated_hidden_size(
        source_gate_up: nn.Parameter,
        source_down: nn.Parameter,
        source_is_transposed: bool | None,
        intermediate_size: int | None,
        default: int | None,
    ) -> int | None:
        """Infer hidden size for ungated three-dimensional source weights."""
        if source_is_transposed is False:
            return source_gate_up.shape[-1]
        if source_is_transposed is True:
            return source_gate_up.shape[-2]
        if intermediate_size is None:
            return default
        if source_gate_up.shape[-2] == intermediate_size and source_down.shape[-1] == intermediate_size:
            return source_gate_up.shape[-1]
        if source_gate_up.shape[-1] == intermediate_size and source_down.shape[-2] == intermediate_size:
            return source_gate_up.shape[-2]
        return default

    @staticmethod
    def _activation(module: nn.Module, config: Any, gated_linear_unit: bool) -> Any:
        """Resolve the source-compatible expert activation."""
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
        source_apply_gate = getattr(module, "_apply_gate", None)
        source_apply_gate_func = getattr(source_apply_gate, "__func__", source_apply_gate)
        if (
            gated_linear_unit
            and source_apply_gate_func is not None
            and getattr(source_apply_gate_func, "__name__", "") != "_default_apply_gate"
        ):
            raise ValueError("GroupedExperts custom _apply_gate requires a dedicated replacement")
        if not gated_linear_unit:
            return activation_func
        if hidden_act == "silu" and bool(getattr(config, "use_fused_swiglu", True)):
            return partial(swiglu, dim=-1)

        def glu(x: torch.Tensor) -> torch.Tensor:
            """Apply the configured gated linear unit."""
            gate, value = torch.chunk(x, 2, dim=-1)
            return activation_func(gate) * value

        return glu

    @staticmethod
    def _source_layout(
        source_shapes: tuple[tuple[int, ...], tuple[int, ...]],
        source_is_transposed: bool | None,
        expert_shapes: tuple[tuple[int, ...], tuple[int, ...]],
        transformers_shapes: tuple[tuple[int, ...], tuple[int, ...]],
        npu_shapes: tuple[tuple[int, ...], tuple[int, ...]],
    ) -> tuple[bool, bool]:
        """Return whether experts are two-dimensional and need transposition."""
        if source_shapes == expert_shapes:
            return True, False
        if source_is_transposed is False:
            if source_shapes != transformers_shapes:
                raise ValueError("GroupedExperts source weights do not match their declared layout")
            return False, True
        if source_is_transposed is True:
            if source_shapes != npu_shapes:
                raise ValueError("GroupedExperts source weights do not match their declared layout")
            return False, False
        if source_shapes == transformers_shapes:
            return False, True
        if source_shapes == npu_shapes:
            return False, False
        raise ValueError("GroupedExperts source weight shapes are incompatible with the configured dimensions")

    def _initialize_weights(
        self,
        source_gate_up: nn.Parameter,
        source_down: nn.Parameter,
        npu_gate_up_shape: tuple[int, ...],
        npu_down_shape: tuple[int, ...],
        gated_linear_unit: bool,
    ) -> None:
        """Create or reuse expert parameters in the target NPU layout."""
        if self._transpose_source_weights:
            target_gate_up = nn.Parameter(
                torch.empty(npu_gate_up_shape, device=source_gate_up.device, dtype=source_gate_up.dtype),
                requires_grad=source_gate_up.requires_grad,
            )
            self.down_proj = nn.Parameter(
                torch.empty(npu_down_shape, device=source_down.device, dtype=source_down.dtype),
                requires_grad=source_down.requires_grad,
            )
        else:
            target_gate_up = source_gate_up
            self.down_proj = source_down
        if gated_linear_unit:
            self.gate_up_proj = target_gate_up
        else:
            self.up_proj = target_gate_up

    def _initialize_biases(self, module: nn.Module) -> None:
        """Validate and materialize optional expert bias parameters."""
        self._source_bias_names = (
            "bias1" if hasattr(module, "bias1") else f"{self._source_fc1_name}_bias",
            "bias2" if hasattr(module, "bias2") else "down_proj_bias",
        )
        source_bias1 = getattr(module, self._source_bias_names[0], None)
        source_bias2 = getattr(module, self._source_bias_names[1], None)
        if (source_bias1 is None) != (source_bias2 is None):
            raise ValueError("GroupedExperts source projections must use the same bias policy")
        self.add_bias = source_bias1 is not None
        if not self.add_bias:
            return
        if not isinstance(source_bias1, nn.Parameter) or not isinstance(source_bias2, nn.Parameter):
            raise TypeError("GroupedExperts source biases must be nn.Parameter instances")
        target_biases = []
        for source_name, target_name, source_bias in zip(
            self._source_bias_names, ("bias1", "bias2"), (source_bias1, source_bias2)
        ):
            if source_name == target_name:
                target_biases.append(source_bias)
            else:
                target_biases.append(
                    nn.Parameter(torch.empty_like(source_bias), requires_grad=source_bias.requires_grad)
                )
        # astroid cannot infer the length of a list built by appends in a
        # loop; the zip above always yields exactly two entries.
        self.bias1, self.bias2 = target_biases  # pylint: disable=W0632

    def __init__(
        self,
        *,
        module: nn.Module,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """Build the high-performance module from a Transformers experts module."""
        super().__init__()
        del module_fqn, context
        config = getattr(module, "config", None)
        self.config = config
        self.initializer_range = getattr(config, "initializer_range", None)
        if bool(getattr(module, "has_bias", False)):
            raise ValueError("GroupedExperts does not support source modules with has_bias=True")
        gated_linear_unit = bool(
            getattr(
                module,
                "has_gate",
                getattr(config, "gated_linear_unit", True),
            )
        )
        self._source_fc1_name = "gate_up_proj" if gated_linear_unit else "up_proj"
        source_gate_up, source_down = self._source_parameters(module, self._source_fc1_name)
        num_experts, hidden_size, intermediate_size = self._source_dimensions(
            module, config, source_gate_up, source_down, gated_linear_unit
        )
        source_down = module.down_proj
        source_is_transposed = getattr(module, "is_transposed", None)

        self.num_local_experts = num_experts
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.router_gating_in_fp32 = bool(
            getattr(module, "router_gating_in_fp32", True)
        )
        self.activation_func = self._activation(module, config, gated_linear_unit)

        # GroupGemm state (NPU-only)
        self._group_list = None
        self._tokens_per_expert_gmm = None

        fc1_output_size = intermediate_size
        if gated_linear_unit:
            fc1_output_size *= 2
        fc1_output_size_per_partition = fc1_output_size

        fc2_input_size = intermediate_size
        fc2_input_size_per_partition = fc2_input_size

        npu_gate_up_shape = (
            self.num_local_experts,
            hidden_size,
            fc1_output_size_per_partition,
        )
        npu_down_shape = (
            self.num_local_experts,
            fc2_input_size_per_partition,
            hidden_size,
        )
        transformers_gate_up_shape = (
            self.num_local_experts,
            fc1_output_size_per_partition,
            hidden_size,
        )
        transformers_down_shape = (
            self.num_local_experts,
            hidden_size,
            fc2_input_size_per_partition,
        )
        experts_2d_gate_up_shape = (
            self.num_local_experts * hidden_size,
            fc1_output_size_per_partition,
        )
        experts_2d_down_shape = (
            self.num_local_experts * fc2_input_size_per_partition,
            hidden_size,
        )
        source_shapes = (tuple(source_gate_up.shape), tuple(source_down.shape))
        self.use_2d_experts, self._transpose_source_weights = self._source_layout(
            source_shapes,
            source_is_transposed,
            (experts_2d_gate_up_shape, experts_2d_down_shape),
            (transformers_gate_up_shape, transformers_down_shape),
            (npu_gate_up_shape, npu_down_shape),
        )
        self._initialize_weights(
            source_gate_up, source_down, npu_gate_up_shape, npu_down_shape, gated_linear_unit
        )
        self._initialize_biases(module)

        self.has_gate = gated_linear_unit
        self.has_bias = self.add_bias
        self.is_concatenated = bool(getattr(module, "is_concatenated", True))
        if self.has_gate and not self.is_concatenated:
            raise ValueError("GroupedExperts requires concatenated gate/up expert weights")
        self.is_transposed = True
        self.train(module.training)

    def reset_parameters(self) -> None:
        """Initialize grouped weights with the source model's configured standard deviation."""
        if self.initializer_range is None:
            raise ValueError(
                "GroupedExperts random initialization requires config.initializer_range"
            )
        target_fc1 = self.gate_up_proj if self.has_gate else self.up_proj
        nn.init.normal_(target_fc1, mean=0.0, std=self.initializer_range)
        nn.init.normal_(self.down_proj, mean=0.0, std=self.initializer_range)
        if self.add_bias:
            nn.init.zeros_(self.bias1)
            nn.init.zeros_(self.bias2)

    def make_transforms(self) -> list[WeightRenaming | WeightConverter]:
        """Describe reversible source-checkpoint to high-performance conversion."""
        transforms: list[WeightRenaming | WeightConverter] = []
        if self._transpose_source_weights:
            transforms.extend(
                [
                    WeightConverter(
                        source_patterns=self._source_fc1_name,
                        target_patterns=self._source_fc1_name,
                        operations=[Transpose(dim0=-2, dim1=-1)],
                    ),
                    WeightConverter(
                        source_patterns="down_proj",
                        target_patterns="down_proj",
                        operations=[Transpose(dim0=-2, dim1=-1)],
                    ),
                ]
            )
        if self.add_bias:
            for source_name, target_name in zip(
                self._source_bias_names,
                ("bias1", "bias2"),
            ):
                if source_name == target_name:
                    continue
                transforms.append(
                    WeightRenaming(
                        source_patterns=source_name,
                        target_patterns=target_name,
                    )
                )
        return transforms

    def _grouped_gemm_expert_forward(
        self, gate_up_proj, down_proj, permuted, tokens_per_expert, permuted_probs
    ):
        """GroupGemm expert computation (NPU-only)."""
        self._tokens_per_expert_gmm = tokens_per_expert.to(device=permuted.device)
        self._group_list = self._tokens_per_expert_gmm.cumsum(dim=0)

        if self.use_2d_experts:
            gate_up_proj = gate_up_proj.view(self.num_local_experts, self.hidden_size, -1)

        # up
        if permuted.nelement() != 0:
            fc1_output = grouped_matmul(
                permuted, gate_up_proj, bias=None, group_list=self._group_list, group_type=0, group_list_type=0,
            )
            if self.add_bias:
                b1 = self.bias1.view(self.num_local_experts, 1, -1)
                fc1_output = fc1_output + torch.repeat_interleave(b1, self._tokens_per_expert_gmm, dim=0)
        else:
            gate_up_proj_2d = gate_up_proj.view(self.hidden_size, -1)
            fc1_output = torch.matmul(permuted, gate_up_proj_2d)

        # down
        if not self.router_gating_in_fp32 and permuted_probs is not None:
            fc1_output = (
                self.activation_func(fc1_output)
                * permuted_probs.reshape(*fc1_output.shape[:-1], 1)
            )
        else:
            fc1_output = self.activation_func(fc1_output)

        if self.use_2d_experts:
            down_proj = down_proj.view(self.num_local_experts, -1, self.hidden_size)

        if fc1_output.nelement() != 0:
            fc2_output = grouped_matmul(
                fc1_output, down_proj, bias=None, group_list=self._group_list, group_type=0, group_list_type=0,
            )
            if self.add_bias:
                b2 = self.bias2.view(self.num_local_experts, 1, -1)
                fc2_output = fc2_output + torch.repeat_interleave(
                    b2, self._tokens_per_expert_gmm, dim=0,
                )
        else:
            down_proj_2d = down_proj.view(-1, self.hidden_size)
            fc2_output = torch.matmul(fc1_output, down_proj_2d)

        return fc2_output

    def forward_expert_major(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run grouped experts on expert-major routed tokens.

        Args:
            x: Expert-major input with shape ``[tokens, hidden_size]``.
            num_tokens_per_expert: Token count for every local expert.
            scores: Optional routing weight for every routed token.

        Returns:
            Expert-major output with shape ``[tokens, hidden_size]``.
        """
        gate_up_proj = self.gate_up_proj if self.has_gate else self.up_proj
        down_proj = self.down_proj
        self.num_local_experts = num_tokens_per_expert.shape[0]

        return self._grouped_gemm_expert_forward(
            gate_up_proj, down_proj, x, num_tokens_per_expert, scores
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Run grouped experts with the Transformers Experts interface."""
        hidden_shape = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_states.shape[-1])
        permuted_tokens, sorted_indices = moe_token_permute(
            hidden_states_flat,
            top_k_index,
        )
        flatten_probs = top_k_weights.view(-1)
        permuted_probs = flatten_probs.index_select(0, sorted_indices)
        tokens_per_expert = torch.histc(
            top_k_index.view(-1).float(),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        ).long()
        expert_outputs = self.forward_expert_major(
            permuted_tokens,
            tokens_per_expert,
            permuted_probs,
        )
        output = moe_token_unpermute(
            expert_outputs,
            sorted_indices,
            top_k_weights,
        )
        return output.view(hidden_shape)
