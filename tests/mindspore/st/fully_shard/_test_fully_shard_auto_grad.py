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
"""test fully_shard with chunked output layer and autograd (MindSpore)"""
import mindspore as ms
from mindspore import mint, nn, Tensor
from mindspore.communication import get_rank, init

from hyper_parallel import SkipDTensorDispatch, init_device_mesh, DTensor
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy

ms.set_seed(42)

def get_backward_grads(net, expected_dtype=None):
    """Collect DTensor gradients from fully_shard-managed DTensor params."""
    grads = []
    for idx, param in enumerate(net.trainable_params()):
        grad = param.grad
        assert isinstance(grad, DTensor), f"Parameter grad {idx} is not a DTensor"
        assert grad.shape == param.shape, (
            f"Gradient global shape mismatch at index {idx}: "
            f"Expected {param.shape}, got {grad.shape}"
        )
        assert grad.local_shape == param.local_shape, (
            f"Gradient local shape mismatch at index {idx}: "
            f"Expected {param.local_shape}, got {grad.local_shape}"
        )
        if expected_dtype is not None:
            assert grad.dtype == expected_dtype, (
                f"Returned grad {idx} dtype mismatch: expected {expected_dtype}, got {grad.dtype}"
            )
        grads.append(grad)
    return tuple(grads)


class OutputLayer(nn.Cell):
    """A simple linear output layer for chunked forward testing.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Dense(in_features, out_features, has_bias=False)

    def construct(self, x):
        return self.linear(x)


class MLPAndChunkedOutputLayer(nn.Cell):
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
        self.mlp = nn.Dense(hidden_size, hidden_size, has_bias=False)
        self.output_layer = OutputLayer(hidden_size, output_size)
        self.num_chunks = num_chunks

    def construct(self, x):
        hidden_states = mint.nn.functional.relu(self.mlp(x))
        chunks = mint.chunk(hidden_states, self.num_chunks, dim=0)
        results = []
        for chunk in chunks:
            results.append(self.output_layer(chunk))
        output = mint.cat(results, dim=0)
        return output


def test_chunked_output_fully_shard():
    """Test fully_shard with chunked input and looped OutputLayer forward.

    Feature: fully_shard autograd with chunked output (MindSpore)
    Description: Verify that a fully_shard-wrapped OutputLayer can be called
        multiple times in a for-loop (once per input chunk), with results
        concatenated and a single backward pass. This pattern is common in
        LLM output heads for memory-efficient loss computation.
    Expectation: Training completes, loss is finite, and gradients are present.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    init()
    rank_id = get_rank()

    hidden_size = 16
    output_size = 8
    batch_size = 8
    num_chunks = 4
    steps = 1

    mesh = init_device_mesh(device_type="npu", mesh_shape=(4,), mesh_dim_names=("dp",))
    mp_policy = MixedPrecisionPolicy()

    model = MLPAndChunkedOutputLayer(hidden_size, output_size, num_chunks)
    fully_shard(model.output_layer, mesh=mesh, mp_policy=mp_policy)
    fully_shard(model, mesh=mesh, mp_policy=mp_policy)

    optimizer = nn.Adam(model.trainable_params(), learning_rate=0.01)

    input_data = Tensor(ms.numpy.randn(batch_size, hidden_size).astype(ms.float32))

    for step_idx in range(steps):
        model.zero_grad()
        output = model(input_data)
        loss = mint.sum(output)
        loss.backward()
        grads = get_backward_grads(model)
        with SkipDTensorDispatch():
            optimizer(grads)

        if rank_id == 0:
            print(f"rank: {rank_id} step: {step_idx}, loss: {float(loss.asnumpy())}")
