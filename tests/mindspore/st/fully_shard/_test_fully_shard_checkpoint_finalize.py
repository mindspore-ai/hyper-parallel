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
"""NPU regression for both checkpoint hook orders and nested HSDP finalization."""
from contextlib import contextmanager
import importlib
from typing import Any
from unittest.mock import patch

import mindspore as ms
from mindspore import Tensor, nn
from mindspore.communication import get_rank, init
import numpy as np
import pytest

from hyper_parallel import SkipDTensorDispatch, init_device_mesh
from hyper_parallel.core.activation_checkpoint import checkpoint_wrapper
from hyper_parallel.core.fully_shard.api import HSDPModule, fully_shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.platform.mindspore.autograd_compat import enable_mindspore_backward_compat
from hyper_parallel.platform.mindspore.fully_shard.scheduler import MindSporeHSDPSchedulerV2
from hyper_parallel.platform.mindspore.fully_shard.state import MindSporeHSDPStateV2


class _Block(nn.Cell):
    """The child projection models an expert FSDP unit inside a checkpoint."""

    def __init__(self, generator: np.random.Generator) -> None:
        """Initialize deterministic projection weights."""
        super().__init__()
        self.fc1 = nn.Dense(32, 32, has_bias=False,
                            weight_init=Tensor(generator.normal(0, 0.1, (32, 32)).astype(np.float32)))
        self.fc2 = nn.Dense(32, 32, has_bias=False,
                            weight_init=Tensor(generator.normal(0, 0.1, (32, 32)).astype(np.float32)))
        self.relu = nn.ReLU()

    def construct(self, inputs: Tensor) -> Tensor:
        """Preserve an input-gradient path across every checkpoint boundary."""
        return inputs + self.fc2(self.relu(self.fc1(inputs)))


class _Model(nn.Cell):
    """Use a stem to give checkpointed layers differentiable inputs."""

    def __init__(self) -> None:
        """Initialize the stem and four nested FSDP blocks."""
        super().__init__()
        generator = np.random.default_rng(17)
        self.stem = nn.Dense(32, 32, has_bias=False,
                             weight_init=Tensor(generator.normal(0, 0.1, (32, 32)).astype(np.float32)))
        self.layers = nn.CellList([_Block(generator) for _ in range(4)])

    def construct(self, inputs: Tensor) -> Tensor:
        """Return a scalar loss."""
        value = self.stem(inputs)
        for layer in self.layers:
            value = layer(value)
        return (value * value).mean()


def _build_model(mesh, wrapper, wrap_root):
    model = _Model()
    if wrapper is not None:
        for index in (0, 1):
            model.layers[index] = wrapper(model.layers[index])
    policy = MixedPrecisionPolicy(param_dtype=ms.float32, reduce_dtype=ms.float32,
                                  output_dtype=ms.float32, apply_grad_on_fp32_main_grad=True)
    options = {"mesh": mesh, "mp_policy": policy, "comm_fusion": False, "reshard_after_forward": True}
    for layer in model.layers:
        raw_layer = getattr(layer, "_wrapped_module", layer)
        fully_shard(raw_layer.fc2, **options)
        fully_shard(layer, **options)
    if wrap_root:
        fully_shard(model, **options)
        roots = [model]
    else:
        fully_shard(model.stem, **options)
        roots = [model.stem, *model.layers]
    for root in roots:
        root.set_reduce_op_type("sum")
    return model, roots


@contextmanager
def _observe_backward(model, reentrant_api):
    schedulers = {module.hsdp_scheduler for _, module in model.cells_and_names()
                  if isinstance(module, HSDPModule)}
    record = {"depth": 0, "completed": set(), "drains": []}
    original_post = MindSporeHSDPSchedulerV2._hsdp_backward_hook
    original_drain = MindSporeHSDPStateV2.delay_apply_reduce_grads.__func__

    def post(scheduler: MindSporeHSDPSchedulerV2, *args: Any) -> None:
        """Record local completion after the actual post-backward work."""
        original_post(scheduler, *args)
        record["completed"].add(scheduler)

    def drain(cls: type) -> None:
        """Record the backward depth and outstanding local finalizers."""
        record["drains"].append((record["depth"], len(schedulers - record["completed"])))
        original_drain(cls)

    with patch.object(MindSporeHSDPSchedulerV2, "_hsdp_backward_hook", post), patch.object(
            MindSporeHSDPStateV2, "delay_apply_reduce_grads", classmethod(drain)):
        if reentrant_api is None:
            yield record
        else:
            original_run = reentrant_api.run_backward

            def nested_run(*args: Any, **kwargs: Any) -> Any:
                """Track the real reentrant engine call without changing its result."""
                record["depth"] += 1
                try:
                    return original_run(*args, **kwargs)
                finally:
                    record["depth"] -= 1

            with patch.object(reentrant_api, "run_backward", nested_run):
                yield record


def _run_steps(model, roots, inputs, reentrant_api, wrap_root):
    results = []
    with _observe_backward(model, reentrant_api) as record:
        for step in range(2):
            for root in roots:
                root.zero_grad()
            for micro in range(2):
                for root in roots:
                    root.set_requires_all_reduce(micro == 1)
                record["completed"].clear()
                record["drains"].clear()
                loss = model(inputs[step * 2 + micro]) / 2
                loss.backward()
                ms.runtime.synchronize()
                assert record["drains"], "The enclosing backward must finalize reductions"
                assert all(depth == 0 for depth, _ in record["drains"]), record["drains"]
                if wrap_root:
                    assert record["drains"] == [(0, 0)], record["drains"]
                assert not MindSporeHSDPStateV2.pending_all_reduce_groups
                for root in roots:
                    ctx = root.hsdp_scheduler.scheduler_ctx
                    assert not ctx.post_backward_final_callback_queued
                    assert not ctx.post_backward_schedulers
            gradients = {}
            for name, parameter in model.parameters_and_names():
                grad = getattr(parameter, "main_grad", None)
                assert grad is not None, name
                local = grad.to_local() if hasattr(grad, "to_local") else grad
                normalized = name.replace("_ckpt_wrapped_module.", "").replace("_wrapped_module.", "")
                gradients[normalized] = local.asnumpy().copy()
            results.append((float(loss.asnumpy()), gradients))
    return results


@pytest.fixture(scope="module", autouse=True)
def _distributed_runtime():
    ms.set_context(mode=ms.PYNATIVE_MODE)
    enable_mindspore_backward_compat()
    init()


@pytest.mark.parametrize("use_reentrant", [False, True])
@pytest.mark.parametrize("wrap_root", [True, False])
def test_checkpoint_finalize_order_and_gradients(use_reentrant, wrap_root):
    """
    Feature: HSDP checkpoint finalization.
    Description: Compare both checkpoint hook orders against an eager HSDP reference,
        with a nested sharded child, optional outer root, and repeated accumulation.
    Expectation: No nested-task drain, no lost terminal gradients, identical gradients.
    """
    reentrant_api = None
    wrapper = checkpoint_wrapper
    if use_reentrant:
        # Reentrant checkpointing is an optional backend on older HyperParallel branches.
        reentrant_api = importlib.import_module(
            "hyper_parallel.platform.mindspore.activation_checkpoint.reentrant_checkpoint"
        )
        wrapper = reentrant_api.reentrant_checkpoint_wrapper
    mesh = init_device_mesh("npu", (2, 2), mesh_dim_names=("replicate", "shard"))
    generator = np.random.default_rng(101 + get_rank())
    inputs = [Tensor(generator.normal(size=(4, 32)).astype(np.float32)) for _ in range(4)]
    with SkipDTensorDispatch():
        reference, reference_roots = _build_model(mesh, None, wrap_root)
        expected = _run_steps(reference, reference_roots, inputs, None, wrap_root)
        model, roots = _build_model(mesh, wrapper, wrap_root)
        actual = _run_steps(model, roots, inputs, reentrant_api, wrap_root)
    for (expected_loss, expected_grads), (actual_loss, actual_grads) in zip(expected, actual):
        assert actual_loss == pytest.approx(expected_loss, rel=1e-5, abs=1e-6)
        assert actual_grads.keys() == expected_grads.keys()
        for name, expected_grad in expected_grads.items():
            np.testing.assert_allclose(actual_grads[name], expected_grad, rtol=1e-5, atol=1e-6, err_msg=name)
