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
"""Capture the existing FSDP precision policy in a joint graph."""

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
    """Describe casts without changing the live parameter storage."""

    def __init__(self, model: nn.Module, policy: MixedPrecisionPolicy, modules: tuple[nn.Module, ...]) -> None:
        """Bind the resolved eager policy before the graph pass slices parameters."""
        self.policy = policy
        self.modules = modules
        self.parameter_dtypes = {}
        for name, parameter in model.named_parameters(remove_duplicate=False):
            if not parameter.is_floating_point():
                continue
            compute, reduce = resolve_param_dtypes(policy, id(parameter), parameter.dtype)
            # Autograd returns the gradient in the compute weight's dtype.
            if reduce == (compute or parameter.dtype):
                reduce = None
            self.parameter_dtypes[name] = (compute, reduce)

    @property
    def has_param_or_grad_casts(self) -> bool:
        """Whether FSDP parameters or their gradients need a different dtype."""
        return any(compute is not None or reduce is not None for compute, reduce in self.parameter_dtypes.values())

    def cast_state(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Differentiate the same compute-precision weights that forward reads."""
        compute_state = state.copy()
        for name, (dtype, _) in self.parameter_dtypes.items():
            if dtype is not None and name in state:
                compute_state[name] = cast_fp_tensor(dtype, state[name])
        return compute_state

    def cast_gradient(self, name: str, gradient: torch.Tensor) -> torch.Tensor:
        """Convert to the reduction dtype even if no collective is needed."""
        _, dtype = self.parameter_dtypes.get(name, (None, None))
        return cast_fp_tensor(dtype, gradient) if dtype is not None else gradient

    @contextmanager
    def module_casts(self) -> Iterator[None]:
        """Install input/output conversions only while the graph is captured."""
        with ExitStack() as stack:
            for module in self.modules:
                if self.policy.cast_forward_inputs and self.policy.param_dtype is not None:
                    handle = module.register_forward_pre_hook(self._cast_inputs, with_kwargs=True)
                    stack.callback(handle.remove)
                if self.policy.output_dtype is not None:
                    handle = module.register_forward_hook(self._cast_outputs)
                    stack.callback(handle.remove)
            yield

    def _cast_inputs(self, module: nn.Module, args: tuple, kwargs: dict) -> tuple:
        """Capture casts of the floating arguments at the eager unit boundary."""
        del module
        cast = partial(cast_fp_tensor, self.policy.param_dtype)
        return apply_to_tensors(cast, args), apply_to_tensors(cast, kwargs)

    def _cast_outputs(self, module: nn.Module, args: tuple, output: Any) -> Any:
        """Capture the eager unit's output conversion."""
        del module, args
        return apply_to_tensors(partial(cast_fp_tensor, self.policy.output_dtype), output)
