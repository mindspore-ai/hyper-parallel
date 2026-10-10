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
"""Run a single joint graph with changing batch and sequence dimensions.

Run from the repository root: python -m hyper_parallel.compile.examples.dynamic_shapes
"""

import torch

from hyper_parallel.compile import GraphCompiler, PassConfig


def train_fn(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return a scalar training loss.

    Args:
        model: Model being trained.
        x: Input features with variable batch and sequence dimensions.
        y: Targets with matching batch and sequence dimensions.

    Returns:
        Scalar mean squared error.
    """
    return (model(x) - y).square().mean()


def main() -> None:
    """Compare dynamic graph loss and parameter gradients with eager execution."""
    torch.manual_seed(42)
    model = torch.nn.Linear(4, 3)
    compiler = GraphCompiler(
        model, train_fn,
        pass_config=PassConfig(fsdp_enabled=False),
        dynamic_arg_dims={"x": [0, 1], "y": [0, 1]},
        device=torch.device("cpu"),
    )
    for batch, sequence in [(2, 5), (3, 7), (4, 3)]:
        x, y = torch.randn(batch, sequence, 4), torch.randn(batch, sequence, 3)
        model.zero_grad()
        loss = compiler.forward_backward(x=x, y=y)
        expected_loss = train_fn(model, x, y)
        expected_grads = torch.autograd.grad(expected_loss, tuple(model.parameters()))
        torch.testing.assert_close(loss, expected_loss)
        for parameter, expected in zip(model.parameters(), expected_grads):
            torch.testing.assert_close(parameter.grad, expected)
        print(f"batch={batch}, sequence={sequence}: loss and gradients match eager")


if __name__ == "__main__":
    main()
