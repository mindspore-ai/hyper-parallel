# Copyright 2025-2026 Huawei Technologies Co., Ltd
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

"""
Test mindspore custom pass used in zero3.
"""
import inspect
import os
import tempfile
from pathlib import Path
import numpy as np
import pytest
from tests.common.mark_utils import arg_mark
from tests.mindspore.st.utils import skip_if_ms_version_lt, skip_if_ms_plugin_not_exist
import mindspore as ms
from mindspore import nn, ops
from mindspore.ops import all_gather_into_tensor
from mindspore.ops import _add_attr
import hyper_parallel


RANK_SIZE = 8
RANK_ID = 7


@pytest.fixture(autouse=True)
def setup_env():
    """
    Set environment before running test cases.
    """
    original_mode = ms.get_context("mode")
    original_sim_level = os.environ.get("MS_SIMULATION_LEVEL")
    original_rank_size = os.environ.get("RANK_SIZE")
    original_rank_id = os.environ.get("RANK_ID")
    original_save_graphs = os.environ.get("MS_DEV_SAVE_GRAPHS")
    original_save_graphs_path = os.environ.get("MS_DEV_SAVE_GRAPHS_PATH")

    os.environ["MS_SIMULATION_LEVEL"] = "1"
    os.environ["RANK_SIZE"] = str(RANK_SIZE)
    os.environ["RANK_ID"] = str(RANK_ID)
    os.environ["MS_DEV_SAVE_GRAPHS"] = "3"

    yield

    ms.context.set_context(mode=original_mode)

    if original_sim_level is not None:
        os.environ["MS_SIMULATION_LEVEL"] = original_sim_level
    else:
        os.environ.pop("MS_SIMULATION_LEVEL", None)

    if original_rank_size is not None:
        os.environ["RANK_SIZE"] = original_rank_size
    else:
        os.environ.pop("RANK_SIZE", None)

    if original_rank_id is not None:
        os.environ["RANK_ID"] = original_rank_id
    else:
        os.environ.pop("RANK_ID", None)

    if original_save_graphs is not None:
        os.environ["MS_DEV_SAVE_GRAPHS"] = original_save_graphs
    else:
        os.environ.pop("MS_DEV_SAVE_GRAPHS", None)

    if original_save_graphs_path is not None:
        os.environ["MS_DEV_SAVE_GRAPHS_PATH"] = original_save_graphs_path
    else:
        os.environ.pop("MS_DEV_SAVE_GRAPHS_PATH", None)


INPUT_SIZE = 256
HIDDEN_SIZE = 128
OUTPUT_SIZE = 64
BATCH_SIZE = 4


data = ms.Tensor(np.ones((BATCH_SIZE, INPUT_SIZE)), dtype=ms.float32)


class SequentialParamNet4(nn.Cell):
    """A sequential neural network with four parameterized layers."""

    def __init__(self) -> None:
        super().__init__()
        self.weight1 = ms.Parameter(ms.Tensor(np.random.randn(
            HIDDEN_SIZE, INPUT_SIZE), dtype=ms.float32).T, name="weight1")
        self.bias1 = ms.Parameter(ms.Tensor(np.random.randn(
            HIDDEN_SIZE), dtype=ms.float32), name="bias1")
        self.weight2 = ms.Parameter(ms.Tensor(np.random.randn(
            HIDDEN_SIZE, HIDDEN_SIZE), dtype=ms.float32).T, name="weight2")
        self.bias2 = ms.Parameter(ms.Tensor(np.random.randn(
            HIDDEN_SIZE), dtype=ms.float32), name="bias2")
        self.weight3 = ms.Parameter(ms.Tensor(np.random.randn(
            HIDDEN_SIZE, HIDDEN_SIZE), dtype=ms.float32).T, name="weight3")
        self.bias3 = ms.Parameter(ms.Tensor(np.random.randn(
            HIDDEN_SIZE), dtype=ms.float32), name="bias3")
        self.weight4 = ms.Parameter(ms.Tensor(np.random.randn(
            OUTPUT_SIZE, HIDDEN_SIZE), dtype=ms.float32).T, name="weight4")
        self.bias4 = ms.Parameter(ms.Tensor(np.random.randn(
            OUTPUT_SIZE), dtype=ms.float32), name="bias4")

        self.matmul = ops.MatMul()
        self.bias_add = ops.BiasAdd()

    def construct(self, x):
        o1 = self.bias_add(self.matmul(x, self.weight1), self.bias1)
        o2 = self.bias_add(self.matmul(o1, self.weight2), self.bias2)
        o3 = self.bias_add(self.matmul(o2, self.weight3), self.bias3)
        o4 = self.bias_add(self.matmul(o3, self.weight4), self.bias4)
        return o4


class ParamNetWithReusedLayer(nn.Cell):
    """A neural network that reuses one of its internal layers."""

    def __init__(self) -> None:
        super().__init__()
        self.weight1 = ms.Parameter(ms.Tensor(np.random.randn(
            HIDDEN_SIZE, INPUT_SIZE), dtype=ms.float32).T, name="weight1")
        self.bias1 = ms.Parameter(ms.Tensor(np.random.randn(
            HIDDEN_SIZE), dtype=ms.float32), name="bias1")
        self.weight2 = ms.Parameter(ms.Tensor(np.random.randn(
            HIDDEN_SIZE, HIDDEN_SIZE), dtype=ms.float32).T, name="weight2")
        self.bias2 = ms.Parameter(ms.Tensor(np.random.randn(
            HIDDEN_SIZE), dtype=ms.float32), name="bias2")
        self.weight3 = ms.Parameter(ms.Tensor(np.random.randn(
            OUTPUT_SIZE, HIDDEN_SIZE), dtype=ms.float32).T, name="weight3")
        self.bias3 = ms.Parameter(ms.Tensor(np.random.randn(
            OUTPUT_SIZE), dtype=ms.float32), name="bias3")

        self.matmul = ops.MatMul()
        self.bias_add = ops.BiasAdd()

    @ms.jit(backend="ms_backend")
    def construct(self, x):
        o1 = self.bias_add(self.matmul(x, self.weight1), self.bias1)
        o2 = self.bias_add(self.matmul(o1, self.weight2), self.bias2)
        # weight2 and bias2 are reused
        o3 = self.bias_add(self.matmul(o2, self.weight2), self.bias2)
        o4 = self.bias_add(self.matmul(o3, self.weight3), self.bias3)
        return o4


class SequentialDenseNet4(nn.Cell):
    """A sequential neural network built using four Dense layers."""

    def __init__(self) -> None:
        super().__init__()
        self.dense1 = nn.Dense(INPUT_SIZE, HIDDEN_SIZE,
                               has_bias=True, dtype=ms.float32)
        self.dense2 = nn.Dense(HIDDEN_SIZE, HIDDEN_SIZE,
                               has_bias=True, dtype=ms.float32)
        self.dense3 = nn.Dense(HIDDEN_SIZE, HIDDEN_SIZE,
                               has_bias=True, dtype=ms.float32)
        self.dense4 = nn.Dense(HIDDEN_SIZE, OUTPUT_SIZE,
                               has_bias=True, dtype=ms.float32)

    def construct(self, x):
        o1 = self.dense1(x)
        o2 = self.dense2(o1)
        o3 = self.dense3(o2)
        o4 = self.dense4(o3)
        return o4


class DenseNetWithReusedLayer(nn.Cell):
    """A DenseNet where one Dense layer is reused in the forward pass."""

    def __init__(self) -> None:
        super().__init__()
        self.dense1 = nn.Dense(INPUT_SIZE, HIDDEN_SIZE,
                               has_bias=True, dtype=ms.float32)
        self.dense2 = nn.Dense(HIDDEN_SIZE, HIDDEN_SIZE,
                               has_bias=True, dtype=ms.float32)
        self.dense3 = nn.Dense(HIDDEN_SIZE, OUTPUT_SIZE,
                               has_bias=True, dtype=ms.float32)

    def construct(self, x):
        o1 = self.dense1(x)
        o2 = self.dense2(o1)
        o3 = self.dense2(o2)  # dense2 is reused
        o4 = self.dense3(o3)
        return o4


class DenseNetWithCtrlFlow(nn.Cell):
    """A DenseNet that incorporates control flow (conditional logic) in its forward pass."""

    def __init__(self) -> None:
        super().__init__()
        self.dense1 = nn.Dense(INPUT_SIZE, HIDDEN_SIZE,
                               has_bias=True, dtype=ms.float32)
        self.dense2 = nn.Dense(HIDDEN_SIZE, HIDDEN_SIZE,
                               has_bias=True, dtype=ms.float32)
        self.dense3 = nn.Dense(HIDDEN_SIZE, OUTPUT_SIZE,
                               has_bias=True, dtype=ms.float32)

    def construct(self, x):
        """construct with control flow."""
        if x.sum() > 0:
            o1 = self.dense1(x)
            o2 = self.dense2(o1)
            o3 = self.dense2(o2)  # dense2 is reused
            out = self.dense3(o3)
        else:
            o1 = self.dense1(x)
            o2 = self.dense2(o1)
            out = self.dense3(o2)
        return out


class InnerBlock(nn.Cell):
    """An internal building block containing two linear layers."""

    def __init__(self) -> None:
        super().__init__()
        self.weight1 = ms.Parameter(ms.Tensor(np.random.randn(
            HIDDEN_SIZE, INPUT_SIZE), dtype=ms.float32).T, name="weight1")
        self.bias1 = ms.Parameter(ms.Tensor(np.random.randn(
            HIDDEN_SIZE), dtype=ms.float32), name="bias1")
        self.weight2 = ms.Parameter(ms.Tensor(np.random.randn(
            HIDDEN_SIZE, HIDDEN_SIZE), dtype=ms.float32).T, name="weight2")
        self.bias2 = ms.Parameter(ms.Tensor(np.random.randn(
            HIDDEN_SIZE), dtype=ms.float32), name="bias2")

        self.matmul = ops.MatMul()
        self.bias_add = ops.BiasAdd()

    def construct(self, x):
        o1 = self.bias_add(self.matmul(x, self.weight1), self.bias1)
        o2 = self.bias_add(self.matmul(o1, self.weight2), self.bias2)
        return o2


class NestedNet(nn.Cell):
    """A neural network that contains an instance of InnerBlock as a sub-module."""

    def __init__(self) -> None:
        super().__init__()
        self.nested_net_inner = InnerBlock()
        self.weight3 = ms.Parameter(ms.Tensor(np.random.randn(
            HIDDEN_SIZE, HIDDEN_SIZE), dtype=ms.float32).T, name="weight3")
        self.bias3 = ms.Parameter(ms.Tensor(np.random.randn(
            HIDDEN_SIZE), dtype=ms.float32), name="bias3")
        self.weight4 = ms.Parameter(ms.Tensor(np.random.randn(
            OUTPUT_SIZE, HIDDEN_SIZE), dtype=ms.float32).T, name="weight4")
        self.bias4 = ms.Parameter(ms.Tensor(np.random.randn(
            OUTPUT_SIZE), dtype=ms.float32), name="bias4")

        self.matmul = ops.MatMul()
        self.bias_add = ops.BiasAdd()

    def construct(self, x):
        o2 = self.nested_net_inner(x)
        o3 = self.bias_add(self.matmul(o2, self.weight3), self.bias3)
        o4 = self.bias_add(self.matmul(o3, self.weight4), self.bias4)
        return o4


class CrossBlockWeightSharingNet(nn.Cell):
    """A network that shares parameters across different blocks (cross-layer sharing)."""

    def __init__(self) -> None:
        super().__init__()
        self.nested_net_inner = InnerBlock()
        self.weight3 = ms.Parameter(ms.Tensor(np.random.randn(
            HIDDEN_SIZE, HIDDEN_SIZE), dtype=ms.float32).T, name="weight3")
        self.bias3 = ms.Parameter(ms.Tensor(np.random.randn(
            HIDDEN_SIZE), dtype=ms.float32), name="bias3")
        self.weight4 = ms.Parameter(ms.Tensor(np.random.randn(
            OUTPUT_SIZE, HIDDEN_SIZE), dtype=ms.float32).T, name="weight4")
        self.bias4 = ms.Parameter(ms.Tensor(np.random.randn(
            OUTPUT_SIZE), dtype=ms.float32), name="bias4")

        self.matmul = ops.MatMul()
        self.bias_add = ops.BiasAdd()

    def construct(self, x):
        o2 = self.nested_net_inner(x)
        # Cross-layer reuse of weight2/bias2 from shared_block
        o3 = self.bias_add(self.matmul(
            o2, self.nested_net_inner.weight2), self.nested_net_inner.bias2)
        o4 = self.bias_add(self.matmul(o3, self.weight3), self.bias3)
        o5 = self.bias_add(self.matmul(o4, self.weight4), self.bias4)
        return o5


@ms.jit_class
class ParamHandler:
    """Handles parameter sharding and unsharding operations for distributed training."""

    def __init__(self, param) -> None:
        self.param = param
        self.global_data = param.numpy()
        slice_size = self.global_data.shape[0] // RANK_SIZE
        self.local_data = ms.Tensor(
            self.global_data[RANK_ID * slice_size: (RANK_ID + 1) * slice_size])
        self.__set_sharded__ = False

    def set_sharded_param(self):
        if not self.__set_sharded__:
            self.param.set_data(self.local_data)
            self.__set_sharded__ = True

    def set_unshared_param(self):
        if self.__set_sharded__:
            self.param.set_data(self.global_data)
            self.__set_sharded__ = False


def shard(net):
    """Applies sharding to all parameters in the network."""
    for _, param in net.parameters_and_names():
        param_handler = ParamHandler(param)
        param_handler.set_sharded_param()
        setattr(param, "handler", param_handler)
    return net


def unshard(net):
    """Restores all parameters in the network to unsharded state."""
    for _, param in net.parameters_and_names():
        if hasattr(param, 'handler'):
            handler = getattr(param, 'handler')
            handler.set_unshared_param()
    return net


def get_parameter_forward_hook(hsdp_forward_hook, hsdp_grad_hook):
    """Creates a forward hook with gradient insertion for parameter operations."""
    class ForwardHookNet(nn.Cell):
        def __init__(self, hsdp_forward_hook) -> None:
            super().__init__()
            self.hsdp_forward_hook = hsdp_forward_hook

        def construct(self, param):
            return self.hsdp_forward_hook(param)

        def bprop(self, param, out, dout):  # pylint: disable=W0613
            return (dout,)

    fwd_hook_net = ForwardHookNet(hsdp_forward_hook)
    insert_grad_of = ops.InsertGradientOf(hsdp_grad_hook)

    def parameter_forward_hook(param):
        return insert_grad_of(fwd_hook_net(param))
    return parameter_forward_hook


class HsdpNetBase(nn.Cell):
    """Base class for HSDP(Hierarchical Sharded Data Parallel) networks.

    Initializes a step counter and wraps the given network with sharding.
    """

    def __init__(self, net) -> None:
        super().__init__()
        self.step = ms.Parameter(ms.Tensor(0, ms.int32), name="step")
        self.sharded_net = shard(net)

    def construct(self, x):
        """Forward pass with step increment."""
        ops.AssignAdd()(self.step, 1)
        return self.sharded_net(x)


class HsdpNetParamListHook(HsdpNetBase):
    """HSDP network that registers a list of parameter-specific hooks via `_register_parameter_forward_hook`.

    Each hook is constructed using parameter-aware hook getters and depends on the optimization level.
    """

    def __init__(self, net, hsdp_forward_hook_getter):
        super().__init__(net)
        param_hooks = []
        for _, param in net.parameters_and_names():
            hsdp_forward_hook = hsdp_forward_hook_getter()
            hsdp_grad_hook = hsdp_grad_hook_simple_getter()
            param_fwd_hook = get_parameter_forward_hook(
                hsdp_forward_hook, hsdp_grad_hook)
            param_hooks.append(
                {"params": [param], "hook": param_fwd_hook}
            )
        net.register_parameter_forward_hook(param_hooks)


def hsdp_forward_hook_simple_getter():
    """Returns a forward hook that performs AllGather operation."""
    allgather = ops.AllGather()
    allgather.add_prim_attr("duplicate_on_multiple_users", True)

    def hsdp_forward_hook(p):
        return allgather(p)
    return hsdp_forward_hook


def hsdp_forward_hook_simple_getter_functional():
    """Returns a functional forward hook that performs AllGather operation."""
    all_gather = _add_attr(all_gather_into_tensor,
                           duplicate_on_multiple_users=True)
    zeros = _add_attr(ops.zeros, duplicate_on_multiple_users=True)

    def hsdp_forward_hook(p):
        local_shape = p.shape
        global_shape = (local_shape[0] * RANK_SIZE,) + local_shape[1:]
        output = zeros(global_shape, p.dtype)
        all_gather(output, p)
        return output
    return hsdp_forward_hook


def hsdp_grad_hook_simple_getter():
    """Returns a gradient hook that performs ReduceScatter operation."""
    def hsdp_grad_hook(grad):
        return ops.ReduceScatter(ops.ReduceOp.SUM)(grad)
    return hsdp_grad_hook


def count_pattern(saved_graphs_path, pattern, ir):
    """Counts occurrences of a pattern in files under the given path."""
    def count_in_file(file_path, pattern):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for line in f if pattern in line)
    return sum(count_in_file(fp, pattern) for fp in Path(saved_graphs_path).rglob(ir))


@skip_if_ms_version_lt("2.8.1")
@skip_if_ms_plugin_not_exist()
@arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level0',
    card_mark='onecard',
    essential_mark='essential')
@pytest.mark.parametrize("net_cls, expected_num_allgather", [
    (SequentialParamNet4, 12),
    (ParamNetWithReusedLayer, 12),
    (SequentialDenseNet4, 12),
    (DenseNetWithReusedLayer, 12),
    (DenseNetWithCtrlFlow, 21),
    (NestedNet, 12),
    pytest.param(CrossBlockWeightSharingNet, 15, marks=pytest.mark.skip(
        reason="Not supported: cross-cell direct parameter reuse (accessing inner block's weight2/bias2 externally)"))
])
@pytest.mark.parametrize("hsdp_forward_hook_getter", [
    hsdp_forward_hook_simple_getter,
    hsdp_forward_hook_simple_getter_functional,
])
def test_custom_hook_basic(net_cls, expected_num_allgather, hsdp_forward_hook_getter):
    """
    Feature: Hyper Parallel MindSpore Zero3 custom pass with parameter sharding hooks.
    Description: 
        Test the integration of custom forward hooks with HSDP (Hybrid Sharded Data Parallel) 
        parameter sharding strategy. Validates that:
        1. Forward/backward computation completes successfully with custom hooks
        2. Correct number of AllGather operations are inserted during gradient computation
        3. Parameter gradients match local parameter shapes after sharding
        4. Both imperative (nn.Cell) and functional hook implementations work correctly
    Expectation: 
        - Forward output shape matches (BATCH_SIZE, OUTPUT_SIZE)
        - Input gradient shape matches input data shape
        - Weight gradients match local parameter shapes (after sharding)
        - AllGather operation count matches expected values per network topology:
          * 12 for linear topologies without control flow
          * 21 for networks with conditional control flow (requires extra sync points)
          * 15 for cross-block weight sharing (currently unsupported, skipped)
    """
    saved_graphs_path = tempfile.mkdtemp(
        prefix=f"{inspect.stack()[0].function}-", suffix=f"-[{net_cls.__name__}]")
    os.environ["MS_DEV_SAVE_GRAPHS_PATH"] = saved_graphs_path

    so_path = os.path.join(
        os.path.dirname(hyper_parallel.__file__),
        "platform/mindspore/custom_pass/libhyper_parallel_mindspore.so")

    success = ms.graph.register_custom_pass(
        pass_name="DuplicatePrimOnMultiUsersPass",
        plugin_so_path=str(so_path),
        device="cpu",
        pass_type=ms.graph.CustomPassType.FULL_GRAPH)

    if not success:
        assert False, (f"Failed to register custom pass 'hyper_parallel_mindspore' from {so_path}. "
                       "Possible causes: "
                       "1) SO file built for incompatible architecture "
                       "2) Missing dependencies (e.g., libmindspore_core.so) "
                       "3) API version mismatch between SO and runtime")

    net = net_cls()
    hsdp_net = HsdpNetParamListHook(
        net, hsdp_forward_hook_getter)
    grad_op = ops.GradOperation(
        get_all=True, get_by_list=True, sens_param=True)
    opt = nn.Adam(params=net.trainable_params(), learning_rate=0.01)
    weights = ms.ParameterTuple(net.trainable_params())

    @ms.jit(backend="ms_backend")
    def fwd_and_grad(x):
        fwd_out = hsdp_net(x)
        sens = ops.ones_like(fwd_out)
        input_grad, weight_grads = grad_op(hsdp_net, weights)(x, sens)
        return (fwd_out, (input_grad, weight_grads))

    @ms.jit(backend="ms_backend")
    def optimize(weight_grads):
        return opt(weight_grads)

    fwd_out, (input_grad, weight_grads) = fwd_and_grad(data)
    optimize(weight_grads)

    assert fwd_out.shape == (BATCH_SIZE, OUTPUT_SIZE)
    assert input_grad[0].shape == data.shape
    for idx, (_, param) in enumerate(net.parameters_and_names()):
        assert weight_grads[idx].shape == param.handler.local_data.shape

    if hsdp_forward_hook_getter is hsdp_forward_hook_simple_getter:
        pattern = "AllGather(%"
    elif hsdp_forward_hook_getter is hsdp_forward_hook_simple_getter_functional:
        pattern = "DistCommAllGatherIntoTensor(%"
    else:
        assert False, f"Unknown forward hook getter: {hsdp_forward_hook_getter}"

    num_all_gather = count_pattern(
        saved_graphs_path, pattern, "*validate*.ir")
    assert num_all_gather == expected_num_allgather, (
        f"Expected {expected_num_allgather} AllGather ops but found {num_all_gather} "
        f"in graph IRs under {saved_graphs_path}. "
        f"Check network topology: {net_cls.__name__} may have unexpected parameter sharing patterns."
    )
