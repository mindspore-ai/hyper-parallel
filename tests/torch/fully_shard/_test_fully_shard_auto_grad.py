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
"""Test fully_shard autograd scheduling and standalone parity."""
# pylint: disable=W0611,C0413,C0412
from copy import deepcopy
import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
import torch
import torch_npu
from torch import nn, optim
from hyper_parallel import DTensor, init_device_mesh, SkipDTensorDispatch
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from tests.torch.common_net import SimpleTransformer
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

    def forward(self, input_tensor):
        return self.linear(input_tensor)


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

    def forward(self, input_tensor):
        hidden_states = torch.relu(self.mlp(input_tensor))
        hidden_state_chunks = torch.chunk(hidden_states, self.num_chunks, dim=0)
        chunk_outputs = []
        for hidden_state_chunk in hidden_state_chunks:
            chunk_outputs.append(self.output_layer(hidden_state_chunk))
        return torch.cat(chunk_outputs, dim=0)


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
    world_size = torch.distributed.get_world_size()
    hidden_size = 16
    output_size = 8
    batch_size = 8
    num_chunks = 4

    mesh = init_device_mesh(device_type="npu", mesh_shape=(world_size,), mesh_dim_names=("dp",))

    torch.manual_seed(42)
    model = MLPAndChunkedOutputLayer(hidden_size, output_size, num_chunks).npu()
    fully_shard(model.output_layer, mesh=mesh)
    fully_shard(model, mesh=mesh)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    model_input = torch.randn(batch_size, hidden_size).npu()

    with SkipDTensorDispatch():
        model_output = model(model_input)
        loss = torch.sum(model_output)
        loss.backward()

        assert torch.isfinite(loss), f"Loss is not finite: loss={loss.item()}"
        for parameter_name, parameter in model.named_parameters():
            assert parameter.grad is not None, f"Missing gradient for {parameter_name}: grad={parameter.grad}"

        optimizer.step()
        optimizer.zero_grad()


def _apply_fully_shard_with_prefetch(model, mesh) -> None:
    """Shard the model in-place and configure prefetch in execution order.

    Each transformer block is an independent fully_shard unit. The root owns
    the embedding, final normalization, and output head. Forward and backward
    prefetch follow the same root-to-leaf and leaf-to-root order as autograd.
    """
    mixed_precision_policy = MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
        cast_forward_inputs=False,
    )
    for transformer_layer in model.layers:
        fully_shard(transformer_layer, mesh=mesh, mp_policy=mixed_precision_policy)
    fully_shard(model, mesh=mesh, mp_policy=mixed_precision_policy)
    model.set_reduce_op_type("sum")

    model.set_modules_to_forward_prefetch([model.layers[0]])
    for transformer_layer, next_transformer_layer in zip(model.layers[:-1], model.layers[1:]):
        transformer_layer.set_modules_to_forward_prefetch([next_transformer_layer])

    model.set_modules_to_backward_prefetch([model.layers[-1]])
    for transformer_layer, previous_transformer_layer in zip(
            reversed(model.layers[1:]), reversed(model.layers[:-1])):
        transformer_layer.set_modules_to_backward_prefetch([previous_transformer_layer])


def _assert_accumulated_gradients_match(standalone_model, sharded_model, rank: int, training_step: int) -> None:
    """Compare every accumulated full gradient without changing model state."""
    standalone_parameters = dict(standalone_model.named_parameters())
    sharded_parameters = dict(sharded_model.named_parameters())
    assert standalone_parameters.keys() == sharded_parameters.keys(), (
        f"Rank {rank}, step {training_step}: parameter names differ, "
        f"standalone={standalone_parameters.keys()}, sharded={sharded_parameters.keys()}"
    )

    for parameter_name, standalone_parameter in standalone_parameters.items():
        sharded_parameter = sharded_parameters[parameter_name]
        assert standalone_parameter.grad is not None and sharded_parameter.grad is not None, (
            f"Rank {rank}, step {training_step}: missing gradient for {parameter_name}, "
            f"standalone={standalone_parameter.grad}, sharded={sharded_parameter.grad}"
        )
        assert isinstance(sharded_parameter.grad, DTensor), (
            f"Rank {rank}, step {training_step}: expected DTensor gradient for {parameter_name}, "
            f"standalone={type(standalone_parameter.grad)}, sharded={type(sharded_parameter.grad)}"
        )
        sharded_full_gradient = sharded_parameter.grad.full_tensor()
        assert standalone_parameter.grad.shape == sharded_full_gradient.shape, (
            f"Rank {rank}, step {training_step}: gradient shape differs for {parameter_name}, "
            f"standalone={standalone_parameter.grad.shape}, sharded={sharded_full_gradient.shape}"
        )
        assert torch.allclose(standalone_parameter.grad, sharded_full_gradient, rtol=1e-4, atol=1e-5), (
            f"Rank {rank}, step {training_step}: accumulated gradient differs for {parameter_name}, "
            f"standalone={standalone_parameter.grad}, sharded={sharded_full_gradient}"
        )


def _run_fully_shard_autograd_parity(mesh_shape, mesh_dim_names, forward_pass_count: int) -> None:
    """Run a shared standalone-parity flow for FSDP, HSDP, and FFBB."""
    rank, _ = init_dist()
    world_size = torch.distributed.get_world_size()
    mesh = init_device_mesh(device_type="npu", mesh_shape=mesh_shape, mesh_dim_names=mesh_dim_names)

    vocabulary_size = 32
    model_dimension = 16
    transformer_depth = 2
    local_batch_size = 2
    sequence_length = 4
    training_steps = 2
    global_batch_size = world_size * local_batch_size

    torch.manual_seed(2026)
    base_model = SimpleTransformer(
        vocab_size=vocabulary_size,
        dim=model_dimension,
        depth=transformer_depth,
    )
    standalone_model = deepcopy(base_model).npu()
    sharded_model = deepcopy(base_model).npu()
    _apply_fully_shard_with_prefetch(sharded_model, mesh)

    standalone_optimizer = optim.Adam(standalone_model.parameters(), lr=1e-3)
    sharded_optimizer = optim.Adam(sharded_model.parameters(), lr=1e-3)
    input_generator = torch.Generator().manual_seed(84)
    local_batch_start = rank * local_batch_size
    total_samples = global_batch_size * forward_pass_count

    for training_step in range(training_steps):
        global_token_batches = [
            torch.randint(
                vocabulary_size,
                (global_batch_size, sequence_length),
                generator=input_generator,
            ).npu()
            for _ in range(forward_pass_count)
        ]
        local_token_batches = [
            token_batch.narrow(0, local_batch_start, local_batch_size)
            for token_batch in global_token_batches
        ]

        with SkipDTensorDispatch():
            standalone_outputs = [standalone_model(token_batch) for token_batch in global_token_batches]
            sharded_outputs = [sharded_model(token_batch) for token_batch in local_token_batches]

            for forward_index, (standalone_output, sharded_output) in enumerate(
                    zip(standalone_outputs, sharded_outputs)):
                expected_local_output = standalone_output.narrow(0, local_batch_start, local_batch_size)
                assert torch.allclose(expected_local_output, sharded_output, rtol=1e-5, atol=1e-6), (
                    f"Rank {rank}, step {training_step}, forward {forward_index}: output differs, "
                    f"standalone={expected_local_output}, sharded={sharded_output}"
                )

            for standalone_output in standalone_outputs:
                (standalone_output.sum() / total_samples).backward()
            for sharded_output in sharded_outputs:
                (sharded_output.sum() / total_samples).backward()

        _assert_accumulated_gradients_match(standalone_model, sharded_model, rank, training_step)

        with SkipDTensorDispatch():
            standalone_optimizer.step()
            sharded_optimizer.step()
            standalone_optimizer.zero_grad(set_to_none=True)
            sharded_optimizer.zero_grad(set_to_none=True)

    sharded_model.reset_iter_state()


def test_single_rank_fsdp_autograd_parity():
    """Verify a one-rank FSDP mesh matches standalone forward and backward."""
    _run_fully_shard_autograd_parity((1,), ("dp",), forward_pass_count=1)


def test_single_rank_hsdp_autograd_parity():
    """Verify a one-rank HSDP mesh matches standalone forward and backward."""
    _run_fully_shard_autograd_parity((1, 1), ("replicate", "shard"), forward_pass_count=1)


def test_hsdp_ffbb_autograd_parity():
    """Verify four-rank HSDP parity for forward-forward-backward-backward."""
    _run_fully_shard_autograd_parity((2, 2), ("replicate", "shard"), forward_pass_count=2)
