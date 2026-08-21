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
from transformers.core_model_loading import Transpose, WeightConverter, WeightRenaming

from hyper_models.components.model_transform import module_replacement
from hyper_models.ops import (
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
        required = (self._source_fc1_name, "down_proj")
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            raise TypeError(f"GroupedExperts source module is missing: {missing}")
        source_gate_up = getattr(module, self._source_fc1_name)
        if not isinstance(source_gate_up, nn.Parameter):
            raise TypeError(
                f"GroupedExperts requires {self._source_fc1_name} to be an nn.Parameter"
            )
        if not isinstance(module.down_proj, nn.Parameter):
            raise TypeError("GroupedExperts requires down_proj to be an nn.Parameter")

        num_experts = getattr(
            module,
            "num_local_experts",
            getattr(module, "num_experts", None),
        )
        if num_experts is None and config is not None:
            num_experts = getattr(
                config,
                "n_routed_experts",
                getattr(config, "num_experts", None),
            )
        hidden_size = getattr(
            module,
            "hidden_size",
            getattr(module, "hidden_dim", None),
        )
        if hidden_size is None and config is not None:
            hidden_size = getattr(config, "hidden_size", None)
        intermediate_size = getattr(
            module,
            "intermediate_size",
            getattr(module, "intermediate_dim", None),
        )
        if intermediate_size is None and config is not None:
            intermediate_size = getattr(
                config,
                "moe_intermediate_size",
                getattr(config, "intermediate_size", None),
            )
        source_down = module.down_proj
        source_is_transposed = getattr(module, "is_transposed", None)
        if not gated_linear_unit and source_gate_up.dim() == 3:
            if source_is_transposed is False:
                hidden_size = source_gate_up.shape[-1]
            elif source_is_transposed is True:
                hidden_size = source_gate_up.shape[-2]
            elif intermediate_size is not None:
                if (
                    source_gate_up.shape[-2] == intermediate_size
                    and source_down.shape[-1] == intermediate_size
                ):
                    hidden_size = source_gate_up.shape[-1]
                elif (
                    source_gate_up.shape[-1] == intermediate_size
                    and source_down.shape[-2] == intermediate_size
                ):
                    hidden_size = source_gate_up.shape[-2]
        if num_experts is None or hidden_size is None or intermediate_size is None:
            raise ValueError(
                "GroupedExperts requires expert count, hidden size, and intermediate size"
            )

        self.num_local_experts = num_experts
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.router_gating_in_fp32 = bool(
            getattr(module, "router_gating_in_fp32", True)
        )
        hidden_act = getattr(config, "hidden_act", None)
        if hidden_act is None:
            hidden_act = getattr(config, "hidden_activation", None)
        if hidden_act is None:
            hidden_act = getattr(config, "mlp_hidden_act", None)
        use_fused_swiglu = bool(getattr(config, "use_fused_swiglu", True))
        _activation_func = getattr(module, "act_fn", None)
        if _activation_func is None:
            if hidden_act is None:
                hidden_act = "silu"
            _activation_func = ACT2FN[hidden_act]
        source_apply_gate = getattr(module, "_apply_gate", None)
        source_apply_gate_func = getattr(source_apply_gate, "__func__", source_apply_gate)
        if (
            gated_linear_unit
            and source_apply_gate_func is not None
            and getattr(source_apply_gate_func, "__name__", "") != "_default_apply_gate"
        ):
            raise ValueError(
                "GroupedExperts custom _apply_gate requires a dedicated replacement"
            )

        # Activation function for expert computation
        if gated_linear_unit:
            if hidden_act == "silu" and use_fused_swiglu:
                self.activation_func = partial(swiglu, dim=-1)
            else:
                def glu(x: torch.Tensor) -> torch.Tensor:
                    """Apply the configured gated linear unit."""
                    x = torch.chunk(x, 2, dim=-1)
                    return _activation_func(x[0]) * x[1]
                self.activation_func = glu
        else:
            self.activation_func = _activation_func

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
        if source_shapes == (experts_2d_gate_up_shape, experts_2d_down_shape):
            self.use_2d_experts = True
            self._transpose_source_weights = False
        elif source_is_transposed is False:
            if source_shapes != (transformers_gate_up_shape, transformers_down_shape):
                raise ValueError("GroupedExperts source weights do not match their declared layout")
            self.use_2d_experts = False
            self._transpose_source_weights = True
        elif source_is_transposed is True:
            if source_shapes != (npu_gate_up_shape, npu_down_shape):
                raise ValueError("GroupedExperts source weights do not match their declared layout")
            self.use_2d_experts = False
            self._transpose_source_weights = False
        elif source_shapes == (transformers_gate_up_shape, transformers_down_shape):
            self.use_2d_experts = False
            self._transpose_source_weights = True
        elif source_shapes == (npu_gate_up_shape, npu_down_shape):
            self.use_2d_experts = False
            self._transpose_source_weights = False
        else:
            raise ValueError(
                "GroupedExperts source weight shapes are incompatible with the configured dimensions"
            )

        if self._transpose_source_weights:
            target_gate_up = nn.Parameter(
                torch.empty(
                    npu_gate_up_shape,
                    device=source_gate_up.device,
                    dtype=source_gate_up.dtype,
                ),
                requires_grad=source_gate_up.requires_grad,
            )
            self.down_proj = nn.Parameter(
                torch.empty(
                    npu_down_shape,
                    device=source_down.device,
                    dtype=source_down.dtype,
                ),
                requires_grad=source_down.requires_grad,
            )
        else:
            target_gate_up = source_gate_up
            self.down_proj = source_down
        if gated_linear_unit:
            self.gate_up_proj = target_gate_up
        else:
            self.up_proj = target_gate_up

        self._source_bias_names = (
            "bias1" if hasattr(module, "bias1") else f"{self._source_fc1_name}_bias",
            "bias2" if hasattr(module, "bias2") else "down_proj_bias",
        )
        source_bias1 = getattr(module, self._source_bias_names[0], None)
        source_bias2 = getattr(module, self._source_bias_names[1], None)
        if (source_bias1 is None) != (source_bias2 is None):
            raise ValueError("GroupedExperts source projections must use the same bias policy")
        self.add_bias = source_bias1 is not None
        if self.add_bias:
            if not isinstance(source_bias1, nn.Parameter) or not isinstance(source_bias2, nn.Parameter):
                raise TypeError("GroupedExperts source biases must be nn.Parameter instances")
            target_biases = []
            for source_name, target_name, source_bias in zip(
                self._source_bias_names,
                ("bias1", "bias2"),
                (source_bias1, source_bias2),
            ):
                if source_name == target_name:
                    target_biases.append(source_bias)
                else:
                    target_biases.append(
                        nn.Parameter(
                            torch.empty_like(source_bias),
                            requires_grad=source_bias.requires_grad,
                        )
                    )
            self.bias1, self.bias2 = target_biases

        self.has_gate = gated_linear_unit
        self.has_bias = self.add_bias
        self.is_concatenated = bool(getattr(module, "is_concatenated", True))
        if self.has_gate and not self.is_concatenated:
            raise ValueError("GroupedExperts requires concatenated gate/up expert weights")
        self.is_transposed = True
        self.train(module.training)

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

    def _forward_expert_major(
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
        expert_outputs = self._forward_expert_major(
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
