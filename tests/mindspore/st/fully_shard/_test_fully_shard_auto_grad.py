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
import copy

import numpy as np

import mindspore as ms
from mindspore import mint, nn, Tensor
from mindspore.communication import get_group_size, get_rank, init

from hyper_parallel import SkipDTensorDispatch, init_device_mesh, DTensor
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.platform.mindspore.autograd_compat import enable_mindspore_backward_compat

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

    ws = get_group_size()
    mesh = init_device_mesh(device_type="npu", mesh_shape=(ws,), mesh_dim_names=("dp",))
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


# --------------------------------------------------------------------------- #
# input-requires-grad fully_shard vs single-card standalone grad parity
# --------------------------------------------------------------------------- #

_D_HID = 8
_GRAD_RTOL = 1e-4
_GRAD_ATOL = 1e-5


class _ReplicatedFFN(nn.Cell):
    """Sub-layer kept replicated under fully_shard (pure all-reduce grad path)."""

    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Dense(dim, dim)

    def construct(self, x):
        return self.lin(x)


class _MixedBlock(nn.Cell):
    """A fully_shard unit mixing a sharded Dense and a replicate_params sub-layer.

    ``net1`` follows the normal reduce-scatter (sharded) path; ``rep`` is passed as
    ``replicate_params`` so its grad only goes through all-reduce. When this block is
    fed a grad-requiring activation, its input ``PostBackwardFunction`` drives
    ``scheduler_state == BACKWARD`` -- the boundary case that the root backward hook
    must still drain.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.net1 = nn.Dense(dim, dim)
        self.rep = _ReplicatedFFN(dim)
        self.relu = nn.ReLU()

    def construct(self, x):
        return self.rep(self.relu(self.net1(x)))


class _BlockStack(nn.Cell):
    """Root container (left unwrapped) running fully_shard-wrapped blocks in sequence."""

    def __init__(self, dim: int, num_blocks: int):
        super().__init__()
        self.blocks = nn.CellList([_MixedBlock(dim) for _ in range(num_blocks)])

    def construct(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def _wrap_blocks(model: _BlockStack, mesh) -> _BlockStack:
    """fully_shard each block (root stays unwrapped), keep ``rep`` replicated, sum-reduce.

    Wrapping only inner blocks (not the root) plus a grad-requiring input is exactly the
    configuration in which a wrapped unit becomes its own root yet runs ``post_backward``
    via autograd -- the scenario whose last reduce-scatter the root backward hook must
    still drain. ``set_reduce_op_type("sum")`` makes dp grads sum-reduced so they match the
    single-card baseline exactly under the GBS-equivalent input split.
    """
    mp_policy = MixedPrecisionPolicy(
        param_dtype=ms.float32,
        reduce_dtype=ms.float32,
        output_dtype=ms.float32,
        cast_forward_inputs=False,
    )
    for block in model.blocks:
        replicate_params = {p for _, p in block.rep.parameters_and_names()}
        fully_shard(block, mesh=mesh, reshard_after_forward=False,
                    mp_policy=mp_policy, replicate_params=replicate_params)
        block.set_reduce_op_type("sum")
        block.set_requires_gradient_sync(True)
    return model


def _full_grad_numpy(name: str, grad) -> np.ndarray:
    """Materialize a full-shape grad as numpy for FSDP-vs-standalone comparison.

    replicate_params already hold the full grad on every rank (compare directly via
    ``to_local``); sharded params are all-gathered with ``full_tensor`` first; the
    single-card baseline holds plain (already full) tensors.
    """
    assert grad is not None, f"grad for {name} is None"
    if not isinstance(grad, DTensor):
        return grad.asnumpy()
    if ".rep." in name:
        return grad.to_local().asnumpy()
    return grad.full_tensor().asnumpy()


def test_input_requires_grad_fully_shard_grad_parity():
    """fully_shard with a grad-requiring input: non-None grads matching single-card.

    Feature: fully_shard autograd, grad-requiring input boundary (MindSpore).
    Description: 4-card dp mesh. fully_shard wraps only inner blocks (root unwrapped) and
        the wrapped block's input requires grad, so its ``PostBackwardFunction`` drives
        ``scheduler_state == BACKWARD``. Each block mixes a sharded Dense (reduce-scatter
        path) and a replicate_params sub-layer (all-reduce path). A single base model is
        deepcopied into an unwrapped standalone baseline and an fsdp copy; the global batch
        is split so each rank eats ``micro_batch`` rows while the baseline eats all
        ``world_size * micro_batch`` rows (GBS-equivalent), with ``sum`` loss + sum-reduced
        grads for exact parity. Sharded grads are all-gathered, replicate grads compared
        directly.
    Expectation: every parameter has a non-None grad, and each fully_shard grad matches the
        single-card standalone grad.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_deterministic(True)
    init()
    enable_mindspore_backward_compat()
    rank = get_rank()
    world_size = get_group_size()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(world_size,), mesh_dim_names=("dp",))

    micro_batch = 2
    num_blocks = 2

    # one base model -> identical init weights for both paths via deepcopy
    ms.set_seed(42)
    base_model = _BlockStack(_D_HID, num_blocks)
    ref_model = copy.deepcopy(base_model)
    fsdp_model = _wrap_blocks(copy.deepcopy(base_model), mesh)

    # GBS equivalence: world_size ranks * micro_batch == single-card global batch
    global_batch = world_size * micro_batch
    rng = np.random.default_rng(2026)
    global_inputs = rng.standard_normal((global_batch, _D_HID)).astype(np.float32)

    fsdp_input = Tensor(global_inputs[rank * micro_batch:(rank + 1) * micro_batch])
    ref_input = Tensor(global_inputs)
    # the differentiable boundary: the wrapped block's input requires grad.
    fsdp_input.requires_grad = True

    fsdp_loss = mint.sum(fsdp_model(fsdp_input))
    fsdp_loss.backward()
    ref_loss = mint.sum(ref_model(ref_input))
    ref_loss.backward()

    fsdp_named = list(fsdp_model.parameters_and_names())
    ref_named = list(ref_model.parameters_and_names())
    assert fsdp_named, "no trainable params under fully_shard"
    assert len(fsdp_named) == len(ref_named), (
        f"rank {rank}: param count mismatch fsdp={len(fsdp_named)} ref={len(ref_named)}"
    )

    # core check: every parameter gradient is populated (non-None)
    none_grads = [name for name, p in fsdp_named if p.grad is None]
    assert not none_grads, f"rank {rank}: params with None grad after backward: {none_grads}"

    # accuracy check: fully_shard grads match the single-card standalone grads
    for (name, fsdp_p), (ref_name, ref_p) in zip(fsdp_named, ref_named):
        fsdp_full = _full_grad_numpy(name, fsdp_p.grad)
        ref_full = _full_grad_numpy(ref_name, ref_p.grad)
        assert fsdp_full.shape == ref_full.shape, (
            f"rank {rank}, {name}: grad shape {fsdp_full.shape} vs standalone {ref_full.shape}"
        )
        assert np.allclose(fsdp_full, ref_full, rtol=_GRAD_RTOL, atol=_GRAD_ATOL), (
            f"rank {rank}, {name}: fully_shard grad != standalone grad\n"
            f"fsdp={fsdp_full}\nstandalone={ref_full}"
        )

    if rank == 0:
        print(f"input-requires-grad fully_shard grad parity passed: world_size={world_size}, "
              f"blocks={num_blocks}, sharded+replicate vs standalone (GBS={global_batch}).")
