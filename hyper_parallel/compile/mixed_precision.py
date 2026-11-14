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
"""Lower the shared FSDP precision policy into the joint forward/backward graph."""

from contextlib import ExitStack, contextmanager
from functools import partial
from typing import Any, Iterator

import torch
from torch import nn

from hyper_parallel.core.fully_shard.utils import (
    MixedPrecisionPolicy,
    apply_to_tensors,
    cast_fp_tensor,
    resolve_param_dtypes,
)


class GraphMixedPrecision:
    """Keep storage, computation, communication and accumulation dtypes distinct."""

    def __init__(self, model: nn.Module, policy: MixedPrecisionPolicy, modules: tuple[nn.Module, ...]) -> None:
        """Bind the shared policy to source parameter identities and unit boundaries.

        Args:
            model: Final model before graph FSDP slices local storage.
            policy: Resolved FSDP policy, including Parameter-specific overrides.
            modules: The same module boundaries selected by eager FSDP.
        """
        self.policy = policy
        self.modules = modules
        self.parameter_dtypes = {}
        for name, parameter in model.named_parameters(remove_duplicate=False):
            compute, reduce = resolve_param_dtypes(policy, id(parameter), parameter.dtype)
            self.parameter_dtypes[name] = (compute or parameter.dtype, reduce)

    @contextmanager
    def module_casts(self) -> Iterator[None]:
        """Record eager-compatible module input/output casts during capture only."""
        with ExitStack() as stack:
            for module in self.modules:
                if self.policy.cast_forward_inputs and self.policy.param_dtype is not None:
                    handle = module.register_forward_pre_hook(self._cast_inputs, with_kwargs=True)
                    stack.callback(handle.remove)
                if self.policy.output_dtype is not None:
                    handle = module.register_forward_hook(self._cast_outputs)
                    stack.callback(handle.remove)
            yield

    def cast_state(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Create compute views whose gradients stay in compute dtype until reduction."""
        return {
            name: cast_fp_tensor(self.parameter_dtypes[name][0], value)
            if name in self.parameter_dtypes else value
            for name, value in state.items()
        }

    def cast_gradient(self, name: str, gradient: torch.Tensor) -> torch.Tensor:
        """Apply the reduction dtype even when no collective is needed.

        Args:
            name: Source parameter FQN in the joint graph's trainable-state order.
            gradient: Compute gradient, or its zero substitute for an unused parameter.
        """
        dtype = self.parameter_dtypes[name][1]
        return cast_fp_tensor(dtype, gradient) if dtype is not None else gradient

    def _cast_inputs(self, module: nn.Module, args: tuple, kwargs: dict) -> tuple:
        """Mirror the eager FSDP pre-forward cast for floating input leaves."""
        del module
        cast = partial(cast_fp_tensor, self.policy.param_dtype)
        return apply_to_tensors(cast, args), apply_to_tensors(cast, kwargs)

    def _cast_outputs(self, module: nn.Module, args: tuple, outputs: Any) -> Any:
        """Mirror the eager FSDP post-forward output cast."""
        del module, args
        return apply_to_tensors(partial(cast_fp_tensor, self.policy.output_dtype), outputs)
