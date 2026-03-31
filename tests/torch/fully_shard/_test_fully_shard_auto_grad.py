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
"""test fully_shard with chunked output layer and autograd"""
# pylint: disable=W0611,C0413,C0412
import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
import torch
import torch_npu
from torch import nn, optim
from hyper_parallel import init_device_mesh, SkipDTensorDispatch
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from tests.torch.utils import init_dist


class OutputLayer(nn.Module):
    """A simple linear output layer for chunked forward testing.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return self.linear(x)

class MLPAndChunkedOutputLayer(nn.Module):
    """MLP followed by a chunked output layer.

    The forward pass runs a simple MLP, then chunks the hidden states and
    calls the same OutputLayer in a for-loop for each chunk — mimicking
    the memory-efficient LLM output head pattern.

    Args:
        hidden_size: Dimension of input and MLP hidden layer.
        output_size: Dimension of the output projection.
        num_chunks: Number of chunks to split hidden states into.
    """

    def __init__(self, hidden_size: int, output_size: int, num_chunks: int = 2):
        super().__init__()
        self.mlp = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_layer = OutputLayer(hidden_size, output_size)
        self.num_chunks = num_chunks

    def forward(self, x):
        hidden_states = torch.relu(self.mlp(x))
        chunks = torch.chunk(hidden_states, self.num_chunks, dim=0)
        results = []
        for chunk in chunks:
            results.append(self.output_layer(chunk))
        output = torch.cat(results, dim=0)
        return output

def test_chunked_output_fully_shard():
    """Test fully_shard with chunked input and looped OutputLayer forward.

    Feature: fully_shard autograd with chunked output
    Description: Verify that a fully_shard-wrapped OutputLayer can be called
        multiple times in a for-loop (once per input chunk), with results
        concatenated and a single backward pass. This pattern is common in
        LLM output heads for memory-efficient loss computation.
    Expectation: Training completes, loss is finite, and gradients are present.
    """
    init_dist()
    hidden_size = 16
    output_size = 8
    batch_size = 8
    num_chunks = 4
    steps = 1

    mesh = init_device_mesh(device_type="npu", mesh_shape=(4,), mesh_dim_names=("dp",))

    torch.manual_seed(42)
    model = MLPAndChunkedOutputLayer(hidden_size, output_size, num_chunks).npu()
    fully_shard(model.output_layer, mesh=mesh)
    fully_shard(model, mesh=mesh)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    input_data = torch.randn(batch_size, hidden_size).npu()

    with SkipDTensorDispatch():
        for step_idx in range(steps):
            output = model(input_data)
            loss = torch.sum(output)
            loss.backward()

            assert torch.isfinite(loss), \
                (f"Loss is not finite at step {step_idx}: "
                 f"loss={loss.item()}")
            for name, p in model.named_parameters():
                assert p.grad is not None, \
                    f"Missing gradient for {name} at step {step_idx}"

            optimizer.step()
            optimizer.zero_grad()
