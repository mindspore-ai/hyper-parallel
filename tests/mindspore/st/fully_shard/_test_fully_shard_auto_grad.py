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
from hyper_parallel.core.fully_shard.hsdp_utils import get_hsdp_state
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


class _DoubleLinear(nn.Cell):
    """Return two independent linear outputs so the second may be unused."""

    def __init__(self, dim: int):
        super().__init__()
        self.lin1 = nn.Dense(dim, dim)
        self.lin2 = nn.Dense(dim, dim)

    def construct(self, x):
        return (
            mint.nn.functional.relu(self.lin1(x)),
            mint.nn.functional.relu(self.lin2(x)),
        )


def _to_local_numpy(tensor) -> np.ndarray:
    """Convert a plain Tensor or size-one DTensor to a local numpy array."""
    if isinstance(tensor, DTensor):
        tensor = tensor.to_local()
    return tensor.asnumpy()


def _assert_initial_sharded_storage_invariants(model: _DoubleLinear) -> dict:
    """Check initial communication storage before lazy-init and remember its identity."""
    initial_sharded_data = {}
    for module_name, module in (("lin1", model.lin1), ("root", model)):
        state = get_hsdp_state(module)
        assert state is not None, f"{module_name}: expected an HSDP state, got={state}"
        assert not state._reset_sharded_params, (  # pylint: disable=protected-access
            f"{module_name}: expected lazy-init reset=False, "
            f"got={state._reset_sharded_params}"  # pylint: disable=protected-access
        )
        for hsdp_param in state.hsdp_params:
            local_tensor = hsdp_param.sharded_param._local_tensor  # pylint: disable=protected-access
            sharded_data = hsdp_param._sharded_param_data  # pylint: disable=protected-access
            assert not sharded_data.requires_grad and sharded_data.is_leaf, (
                f"{module_name}: initial communication storage expected "
                f"requires_grad=False/is_leaf=True, got requires_grad={sharded_data.requires_grad}/"
                f"is_leaf={sharded_data.is_leaf}"
            )
            assert local_tensor.requires_grad and local_tensor.is_leaf, (
                f"{module_name}: initial local shard expected requires_grad=True/is_leaf=True, "
                f"got requires_grad={local_tensor.requires_grad}/is_leaf={local_tensor.is_leaf}"
            )
            assert sharded_data.untyped_storage().data_ptr() == local_tensor.untyped_storage().data_ptr(), (
                f"{module_name}: detached communication storage should alias the logical shard, "
                f"got sharded_data_ptr={sharded_data.untyped_storage().data_ptr()}, "
                f"local_tensor_ptr={local_tensor.untyped_storage().data_ptr()}"
            )
            initial_sharded_data[id(hsdp_param)] = sharded_data
    return initial_sharded_data


def _assert_size_one_leaf_invariants(
    model: _DoubleLinear,
    iteration: int,
    initial_sharded_data: dict,
) -> None:
    """Check the lazy-init storage and logical parameter autograd contract."""
    for module_name, module in (("lin1", model.lin1), ("root", model)):
        state = get_hsdp_state(module)
        assert state is not None, (
            f"Iteration {iteration}, {module_name}: expected an HSDP state, got={state}"
        )
        assert state._reset_sharded_params, (  # pylint: disable=protected-access
            f"Iteration {iteration}, {module_name}: lazy-init reset expected=True, "
            f"got={state._reset_sharded_params}"  # pylint: disable=protected-access
        )
        assert state.hsdp_params, (
            f"Iteration {iteration}, {module_name}: expected managed parameters, got={state.hsdp_params}"
        )
        for hsdp_param in state.hsdp_params:
            sharded_param = hsdp_param.sharded_param
            local_tensor = sharded_param._local_tensor  # pylint: disable=protected-access
            sharded_data = hsdp_param._sharded_param_data  # pylint: disable=protected-access
            unsharded_param = hsdp_param._unsharded_param  # pylint: disable=protected-access
            if iteration == 0:
                assert sharded_data is initial_sharded_data[id(hsdp_param)], (
                    f"{module_name}: first lazy-init unexpectedly refreshed communication storage"
                )
            assert hsdp_param.shard_world_size == 1, (
                f"Iteration {iteration}, {module_name}: expected shard_world_size=1, "
                f"got={hsdp_param.shard_world_size}"
            )
            assert hsdp_param.unsharded_param_buffers[0] is sharded_data, (
                f"Iteration {iteration}, {module_name}: expected the size-one all-gather buffer "
                f"to be sharded_data, got buffer_id={id(hsdp_param.unsharded_param_buffers[0])}, "
                f"sharded_data_id={id(sharded_data)}"
            )
            assert not sharded_data.requires_grad and sharded_data.is_leaf, (
                f"Iteration {iteration}, {module_name}: communication storage expected "
                f"requires_grad=False/is_leaf=True, got requires_grad={sharded_data.requires_grad}/"
                f"is_leaf={sharded_data.is_leaf}"
            )
            assert local_tensor.requires_grad and local_tensor.is_leaf, (
                f"Iteration {iteration}, {module_name}: local shard expected "
                f"requires_grad=True/is_leaf=True, got requires_grad={local_tensor.requires_grad}/"
                f"is_leaf={local_tensor.is_leaf}"
            )
            assert unsharded_param is not None, (
                f"Iteration {iteration}, {module_name}: expected an unsharded parameter, got={unsharded_param}"
            )
            assert unsharded_param.requires_grad and unsharded_param.is_leaf, (
                f"Iteration {iteration}, {module_name}: unsharded parameter expected "
                f"requires_grad=True/is_leaf=True, got requires_grad={unsharded_param.requires_grad}/"
                f"is_leaf={unsharded_param.is_leaf}"
            )


def _step_optimizer_if_grad(optimizer) -> None:
    """Step one Dense optimizer, skipping it when that forward output was unused."""
    grads = tuple(param.grad for param in optimizer.parameters)
    has_grad = tuple(grad is not None for grad in grads)
    if not any(has_grad):
        return
    assert all(has_grad), (
        f"Expected either all or no gradients in one Dense optimizer, got has_grad={has_grad}"
    )
    optimizer(grads)


def _zero_parameter_grads(model: nn.Cell) -> None:
    """Clear eager gradients on both plain and fully_shard-managed parameters."""
    for parameter in model.trainable_params():
        parameter.grad = None


def _assert_single_rank_parity(
    reference_model: _DoubleLinear,
    sharded_model: _DoubleLinear,
    iteration: int,
) -> None:
    """Compare parameter gradients and values for standalone and size-one FSDP."""
    reference_params = list(reference_model.parameters_and_names())
    sharded_params = list(sharded_model.parameters_and_names())
    assert [name for name, _ in reference_params] == [name for name, _ in sharded_params], (
        f"Iteration {iteration}: parameter names differ, "
        f"reference={[name for name, _ in reference_params]}, "
        f"sharded={[name for name, _ in sharded_params]}"
    )
    for (param_name, reference_param), (_, sharded_param) in zip(reference_params, sharded_params):
        assert (reference_param.grad is None) == (sharded_param.grad is None), (
            f"Iteration {iteration}, {param_name}: gradient presence differs, "
            f"reference_grad={reference_param.grad}, sharded_grad={sharded_param.grad}"
        )
        if reference_param.grad is not None:
            reference_grad = _to_local_numpy(reference_param.grad)
            sharded_grad = _to_local_numpy(sharded_param.grad)
            assert np.allclose(reference_grad, sharded_grad, rtol=1e-4, atol=1e-5), (
                f"Iteration {iteration}, {param_name}: gradients differ, "
                f"reference={reference_grad}, sharded={sharded_grad}"
            )


def _assert_single_rank_parameters_match(
    reference_model: _DoubleLinear,
    sharded_model: _DoubleLinear,
    iteration: int,
) -> None:
    """Compare parameter values after the optimizer step."""
    reference_params = list(reference_model.parameters_and_names())
    sharded_params = list(sharded_model.parameters_and_names())
    for (param_name, reference_param), (_, sharded_param) in zip(reference_params, sharded_params):
        reference_value = _to_local_numpy(reference_param)
        sharded_value = _to_local_numpy(sharded_param)
        assert np.allclose(reference_value, sharded_value, rtol=1e-4, atol=1e-5), (
            f"Iteration {iteration}, {param_name}: parameter values differ, "
            f"reference={reference_value}, sharded={sharded_value}"
        )


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


def test_single_rank_unused_forward_output_autograd():
    """Port PyTorch's unused-forward-output autograd case to size-one FSDP.

    Feature: size-one fully_shard autograd with an unused forward output.
    Description: Wrap the first Dense and the root independently, then train for
        ten iterations. The first three losses consume both outputs while the
        remaining losses consume only the first output. Every iteration checks
        standalone parity and the lazy-init leaf/storage invariants.
    Expectation: Losses, gradients, and updated parameters match standalone;
        repeated unshard/backward does not reuse an autograd graph or rebase a view.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_deterministic(True)
    enable_mindspore_backward_compat()
    init()
    rank = get_rank()
    world_size = get_group_size()
    assert world_size == 1, (
        f"This regression requires shard_world_size=1, expected world_size=1, got={world_size}"
    )
    mesh = init_device_mesh(device_type="npu", mesh_shape=(1,), mesh_dim_names=("dp",))
    mp_policy = MixedPrecisionPolicy(
        param_dtype=ms.float32,
        reduce_dtype=ms.float32,
        output_dtype=ms.float32,
        cast_forward_inputs=False,
    )

    ms.set_seed(42)
    base_model = _DoubleLinear(dim=24)
    reference_model = copy.deepcopy(base_model)
    sharded_model = copy.deepcopy(base_model)
    fully_shard(sharded_model.lin1, mesh=mesh, mp_policy=mp_policy)
    fully_shard(sharded_model, mesh=mesh, mp_policy=mp_policy)
    sharded_model.set_reduce_op_type("sum")
    initial_sharded_data = _assert_initial_sharded_storage_invariants(sharded_model)

    reference_optimizers = (
        nn.Adam(reference_model.lin1.trainable_params(), learning_rate=1e-2),
        nn.Adam(reference_model.lin2.trainable_params(), learning_rate=1e-2),
    )
    sharded_optimizers = (
        nn.Adam(sharded_model.lin1.trainable_params(), learning_rate=1e-2),
        nn.Adam(sharded_model.lin2.trainable_params(), learning_rate=1e-2),
    )

    local_batch_size = 2
    global_batch_size = world_size * local_batch_size
    ms.set_seed(1)
    for iteration in range(10):
        _zero_parameter_grads(reference_model)
        _zero_parameter_grads(sharded_model)
        global_input = mint.rand((global_batch_size, 24), dtype=ms.float32)
        local_input = mint.narrow(
            global_input,
            0,
            rank * local_batch_size,
            local_batch_size,
        ).detach()

        sharded_out1, sharded_out2 = sharded_model(local_input)
        _assert_size_one_leaf_invariants(sharded_model, iteration, initial_sharded_data)
        sharded_loss = (
            mint.sum(mint.mul(sharded_out1, sharded_out2))
            if iteration < 3
            else mint.sum(sharded_out1)
        )
        sharded_loss.backward()

        reference_out1, reference_out2 = reference_model(global_input)
        reference_loss = (
            mint.sum(mint.mul(reference_out1, reference_out2))
            if iteration < 3
            else mint.sum(reference_out1)
        )
        reference_loss.backward()

        reference_loss_value = reference_loss.asnumpy()
        sharded_loss_value = sharded_loss.asnumpy()
        assert np.allclose(reference_loss_value, sharded_loss_value, rtol=1e-5, atol=1e-6), (
            f"Iteration {iteration}: loss differs, "
            f"reference={reference_loss_value}, sharded={sharded_loss_value}"
        )
        _assert_single_rank_parity(reference_model, sharded_model, iteration)

        for optimizer in reference_optimizers:
            _step_optimizer_if_grad(optimizer)
        with SkipDTensorDispatch():
            for optimizer in sharded_optimizers:
                _step_optimizer_if_grad(optimizer)
        _assert_single_rank_parameters_match(reference_model, sharded_model, iteration)

    if rank == 0:
        print("size-one fully_shard unused-output autograd parity passed: iterations=10")


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
