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
"""Separate-matrix SiTU MLP with source-owned names and independent Muon NS boundaries."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from hyper_parallel.components.functional.activation_situ import situ
from hyper_parallel.components.modules.module_family import ModuleFamily, build_family, register_module_family
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.models.replacement import module_replacement


class SiTUActivation(nn.Module):
    """Evaluate the public FP32 SiTU equation on a gate-first packed activation."""

    def __init__(self, beta: float = 1.0, linear_beta: float | None = None) -> None:
        """Declare gate and optional up-projection saturation scales."""
        super().__init__()
        if beta <= 0 or (linear_beta is not None and linear_beta <= 0):
            raise ValueError("SiTU saturation scales must be positive")
        self.beta, self.linear_beta = beta, linear_beta

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        """Preserve the source activation's slicing and FP32 evaluation boundaries."""
        width = packed.shape[-1] // 2
        return situ(packed[..., :width], packed[..., width:], beta=self.beta, linear_beta=self.linear_beta)


class SiTUMLPModule(nn.Module):
    """Own three independent matrices with gate/up/down or w1/w3/w2 source keys."""

    def __init__(
        self,
        *,
        config: Any = None,
        module: nn.Module | None = None,
        module_fqn: str = "",
        context: Any = None,
        state_layout: str = "gate_up_down",
    ) -> None:
        """Construct directly or adopt the exact source parameter identities without packing weights."""
        super().__init__()
        del module_fqn, context
        if (config is None) == (module is None):
            raise ValueError("provide exactly one of config or module")
        if module is not None:
            config = module.config
            state_layout = "w1_w3_w2" if all(hasattr(module, name) for name in ("w1", "w3", "w2")) else "gate_up_down"
        self.config = config
        if state_layout not in ("gate_up_down", "w1_w3_w2"):
            raise ValueError("SiTU MLP supports gate_up_down or w1_w3_w2 source state")
        self.state_layout = state_layout
        self.projection_names = (
            ("gate_proj", "up_proj", "down_proj") if state_layout == "gate_up_down" else ("w1", "w3", "w2")
        )
        beta = getattr(config, "activation_situ_beta", None) or 1.0
        linear_beta = getattr(config, "activation_situ_linear_beta", None)
        if getattr(config, "hidden_act", None) != "situ":
            raise ValueError("SiTUMLP requires a SiTU source activation contract")
        if module is not None:
            self._adopt_source(module, beta, linear_beta)
        else:
            self.hidden_size, self.intermediate_size = config.hidden_size, config.intermediate_size
            registration_order = ("w1", "w2", "w3") if state_layout == "w1_w3_w2" else self.projection_names
            for name in registration_order:
                source, destination = (
                    (self.intermediate_size, self.hidden_size)
                    if name == self.projection_names[2]
                    else (self.hidden_size, self.intermediate_size)
                )
                self.add_module(name, nn.Linear(source, destination, bias=getattr(config, "mlp_bias", False)))
            self.act_fn = SiTUActivation(beta, linear_beta)

    def _adopt_source(self, module: nn.Module, beta: float, linear_beta: float | None) -> None:
        """Validate and retain separate SiTU matrices and the source activation contract."""
        expected = set(self.projection_names) | {"act_fn"}
        if set(dict(module.named_children())) != expected or list(module.named_parameters(recurse=False)):
            raise ValueError("Unsupported separate-matrix SiTU source structure")
        gate, up, down = (getattr(module, name) for name in self.projection_names)
        if not all(type(item) is nn.Linear for item in (gate, up, down)):  # pylint: disable=unidiomatic-typecheck
            raise ValueError("SiTU source projections must be plain Linear modules")
        if ((gate.in_features, gate.out_features, down.in_features, down.out_features)
                != (up.in_features, up.out_features, gate.out_features, gate.in_features)):
            raise ValueError("SiTU source projection dimensions do not compose")
        if len({(p.device, p.dtype) for item in (gate, up, down) for p in item.parameters()}) != 1:
            raise ValueError("SiTU source projections must share device and dtype")
        if module.act_fn.beta != beta or module.act_fn.linear_beta != linear_beta:
            raise ValueError("SiTU source activation and config disagree")
        self.hidden_size, self.intermediate_size = gate.in_features, gate.out_features
        for name, child in module.named_children():
            self.add_module(name, SiTUActivation(beta, linear_beta) if name == "act_fn" else child)
        self.get_submodule("act_fn").training = module.act_fn.training
        self.training = module.training

    def make_transforms(self) -> list:
        """Keep every independent source matrix and state key unchanged."""
        return []

    def forward(self, hidden_states: torch.Tensor | None = None, *, x: torch.Tensor | None = None) -> torch.Tensor:
        """Project gate/up separately, evaluate public SiTU and project back to hidden width."""
        if hidden_states is not None and x is not None:
            raise ValueError("Provide SiTU inputs through hidden_states or x, not both")
        hidden_states = hidden_states if x is None else x
        if hidden_states is None:
            raise ValueError("SiTU requires input activations")
        gate, up, down = (getattr(self, name) for name in self.projection_names)
        return down(self.act_fn(torch.cat((gate(hidden_states), up(hidden_states)), dim=-1)))


def situ_sharding(*, module: nn.Module, module_fqn: str, member: Any, context: Any) -> ModuleShardingSpec:
    """Own every source matrix while keeping the currently local computation explicit."""
    del module_fqn, member
    placements = {axis: Replicate() for axis in context.mesh_dim_names if axis in context.axis_sizes}
    tokens = dict(placements)
    if context.axis_sizes["cp"] > 1:
        tokens["cp"] = Shard(1)
    return ModuleShardingSpec(
        params={name: dict(placements) for name, _ in module.named_parameters()},
        in_src={"hidden_states": dict(tokens)},
        in_dst={"hidden_states": dict(tokens)},
        out_src={"output": dict(tokens)},
        out_dst={"output": dict(tokens)},
        is_boundary=True,
    )


def validate_situ(*, config: Any, module: nn.Module | None, context: Any, member: Any) -> None:
    """Reject TP until the separate-matrix SiTU layout has an accepted parallel member."""
    del config, module, context
    if member.axis_sizes["tp"] > 1:
        raise NotImplementedError("Separate-matrix SiTU currently requires TP1")


@module_replacement
def SiTUMLP(  # pylint: disable=invalid-name
    *,
    config: Any = None,
    module: nn.Module | None = None,
    module_fqn: str = "",
    context: Any = None,
    mesh: Any = None,
    state_layout: str = "gate_up_down",
) -> nn.Module:
    """Build a separate-matrix SiTU MLP without changing logical optimizer matrices."""
    return build_family(
        SiTUMLP,
        config=config,
        module=module,
        module_fqn=module_fqn,
        context=context,
        mesh=mesh,
        state_layout=state_layout,
    )


register_module_family(
    ModuleFamily(
        name="situ_mlp",
        baseline=SiTUMLP,
        variants={"local": "hyper_parallel.components.modules.common.situ_mlp.SiTUMLPModule"},
        default_variant="local",
        sharding=situ_sharding,
        validate=validate_situ,
    )
)
