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
"""Run with python -m hyper_parallel.compile.examples.size_specialization."""

import json

import torch

from hyper_parallel.compile import GraphCompiler, PassConfig


def train_fn(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute a scalar loss for graph and eager execution.

    Args:
        model: Model being trained.
        x: Input features with variable batch and sequence dimensions.
        y: Targets with matching batch and sequence dimensions.

    Returns:
        Scalar mean squared error.
    """
    return (model(x) - y).square().mean()


def main() -> None:
    """Show warmup, lazy generation, cache hits and general fallback on CPU."""
    torch.manual_seed(42)
    model = torch.nn.Linear(4, 3)
    compiler = GraphCompiler(
        model, train_fn, pass_config=PassConfig(fsdp_enabled=False), device=torch.device("cpu"),
        dynamic_arg_dims={"x": [0, 1], "y": [0, 1]},
        compile_sizes=[5, 7], compile_size_input="x", compile_size_dim=1,
    )
    for length in (5, 7, 7, 9, 5, 5):
        x, y = torch.randn(2, length, 4), torch.randn(2, length, 3)
        model.zero_grad()
        loss = compiler.forward_backward(x=x, y=y)
        expected = train_fn(model, x, y)
        torch.testing.assert_close(loss, expected)
        for parameter, gradient in zip(model.parameters(), torch.autograd.grad(expected, tuple(model.parameters()))):
            torch.testing.assert_close(parameter.grad, gradient)
        print(json.dumps({"length": length, **compiler.specialization_stats}))


if __name__ == "__main__":
    main()
