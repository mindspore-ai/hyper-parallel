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
"""MindSpore FSDP + TP-MoE + Muon validation for deferred communication dump."""

# The backend must be selected before importing HyperParallel.
# pylint: disable=wrong-import-position

import csv
import os
from pathlib import Path
import tempfile

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

import mindspore as ms
import mindspore.communication.management as dist
from mindspore import Tensor, nn, ops
from mindspore._c_expression import NoFallbackGuard
from mindspore.ops import communication as comm_ops
import numpy as np

from hyper_parallel import DTensor, SkipDTensorDispatch, init_device_mesh, shard_module
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.core.shard.sharding_plan import ShardingPlan
from hyper_parallel.platform.mindspore.autograd_compat import enable_mindspore_backward_compat
from scripts.mindspore_op_stat_dump import MindSporeOpStatDumpMode

enable_mindspore_backward_compat()


class _TinyTPMoE(nn.Cell):
    """Two-expert soft MoE whose expert output dimensions are tensor parallel."""

    def __init__(self, input_size: int, output_size: int) -> None:
        """Initialize deterministic expert weights."""
        super().__init__()
        rng = np.random.default_rng(2026)
        self.expert_0_weight = ms.Parameter(
            Tensor(rng.standard_normal((input_size, output_size)).astype(np.float32) * 0.05),
            name="expert_0_weight",
        )
        self.expert_1_weight = ms.Parameter(
            Tensor(rng.standard_normal((input_size, output_size)).astype(np.float32) * 0.05),
            name="expert_1_weight",
        )

    def construct(self, inputs: Tensor) -> Tensor:
        """Evaluate both TP experts and combine them with deterministic soft routing."""
        expert_0 = ms.mint.matmul(inputs, self.expert_0_weight)
        expert_1 = ms.mint.matmul(inputs, self.expert_1_weight)
        return expert_0 * 0.375 + expert_1 * 0.625


def _build_model(dp_mesh, tp_mesh) -> tuple[_TinyTPMoE, tuple[Replicate, ...]]:
    """Apply expert tensor parallelism followed by FSDP parameter sharding."""
    model = _TinyTPMoE(input_size=8, output_size=8)
    input_placements = tuple(Replicate() for _ in range(tp_mesh.ndim))
    expert_placements = (Shard(1),) + tuple(Replicate() for _ in range(tp_mesh.ndim - 1))
    model = shard_module(
        model,
        device_mesh=tp_mesh,
        sharding_plan=ShardingPlan(
            plan={
                "expert_0_weight": expert_placements,
                "expert_1_weight": expert_placements,
            },
            input_plan={"input": input_placements},
            output_plan={"output": expert_placements},
        ),
    )
    model = fully_shard(
        model,
        mesh=dp_mesh,
        reshard_after_forward=True,
        mp_policy=MixedPrecisionPolicy(
            param_dtype=ms.float32,
            reduce_dtype=ms.float32,
            output_dtype=ms.float32,
            cast_forward_inputs=False,
        ),
    )
    model.set_reduce_op_type("sum")
    return model, input_placements


def _to_local(tensor):
    """Return the local tensor for either a Tensor or DTensor."""
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _zeropower_via_newton_schulz(gradient: Tensor, steps: int = 3) -> Tensor:
    """Compute the Muon orthogonalized update for one local 2-D gradient shard."""
    mat_x = ops.cast(gradient, ms.float32)
    transposed = mat_x.shape[-2] > mat_x.shape[-1]
    if transposed:
        mat_x = ms.mint.transpose(mat_x, -2, -1)
    mat_x = mat_x / (ms.mint.norm(mat_x) + 1e-7)
    coefficients = (
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
    )
    for coeff_a, coeff_b, coeff_c in coefficients[:steps]:
        mat_a = ms.mint.matmul(mat_x, ms.mint.transpose(mat_x, -2, -1))
        mat_b = coeff_b * mat_a + coeff_c * ms.mint.matmul(mat_a, mat_a)
        mat_x = coeff_a * mat_x + ms.mint.matmul(mat_b, mat_x)
    if transposed:
        mat_x = ms.mint.transpose(mat_x, -2, -1)
    return mat_x


def _muon_step(parameters, learning_rate: float = 0.01, momentum: float = 0.95) -> None:
    """Apply one test-local Muon step to FSDP-local expert parameter shards."""
    for parameter in parameters:
        gradient = _to_local(parameter.grad)
        if gradient is None or gradient.ndim != 2:
            continue
        momentum_buffer = ops.zeros_like(gradient)
        momentum_buffer = momentum * momentum_buffer + gradient
        update = _zeropower_via_newton_schulz(momentum_buffer)
        ops.assign_sub(parameter, ops.cast(learning_rate * update, parameter.dtype))


def _run_training_step(model, tp_mesh, input_placements) -> None:
    """Run FSDP + TP-MoE forward/backward and one Muon parameter update."""
    rank = dist.get_rank()
    rng = np.random.default_rng(1000 + rank // tp_mesh.size())
    local_input = Tensor(rng.standard_normal((4, 8)).astype(np.float32))
    local_target = Tensor(rng.standard_normal((4, 8)).astype(np.float32))
    dist_input = DTensor.from_local(local_input, tp_mesh, input_placements)
    dist_target = DTensor.from_local(local_target, tp_mesh, input_placements)

    model.zero_grad()
    prediction = model(dist_input)
    target_shard = dist_target.redistribute(tp_mesh, prediction.placements)
    loss = ms.mint.sum(ms.mint.square(prediction - target_shard))
    if isinstance(loss, DTensor):
        loss = loss.reduce_partial()
    loss.backward(Tensor(1.0 / tp_mesh.size(), ms.float32))
    with SkipDTensorDispatch(), NoFallbackGuard():
        _muon_step(model.trainable_params())


def _unpack_all_reduce_result(source: Tensor, result) -> tuple[Tensor, object]:
    """Normalize MindSpore's in-place and out-of-place collective returns."""
    if isinstance(result, tuple):
        return result
    return source, result


def _run_sync_async_reference_collectives() -> None:
    """Run equal synchronous and asynchronous AllReduce calls with a unique output shape."""
    rank = dist.get_rank()
    source = np.arange(7, dtype=np.float32) + float(rank)
    sync_input = Tensor(source)
    sync_output, _ = _unpack_all_reduce_result(
        sync_input, comm_ops.all_reduce(sync_input, async_op=False)
    )
    async_input = Tensor(source)
    async_output, async_handle = _unpack_all_reduce_result(
        async_input, comm_ops.all_reduce(async_input, async_op=True)
    )
    assert async_handle is not None, "async AllReduce must return a communication handle"
    async_handle.wait()
    np.testing.assert_array_equal(sync_output.asnumpy(), async_output.asnumpy())


def _assert_reference_dump_matches(csv_path: Path) -> None:
    """Compare synchronous and asynchronous reference collective dump rows."""
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    reference_rows = [
        row for row in rows
        if row["io_type"] == "output"
        and row["shape"] == "(7,)"
        and "allreduce" in row["op_name"].replace("_", "").lower()
    ]
    reference_rows.sort(key=lambda row: int(row["op_index"]))
    assert len(reference_rows) >= 2, (
        f"expected synchronous and asynchronous AllReduce output rows, got {reference_rows}"
    )
    sync_row, async_row = reference_rows[-2:]
    assert sync_row["completion_state"] == "waited"
    assert async_row["completion_state"] == "waited"
    assert sync_row["crc32"] == async_row["crc32"]
    assert sync_row["l2_norm"] == async_row["l2_norm"]


def test_fsdp_tp_moe_muon_async_dump_matches_sync() -> None:
    """
    Feature: Native CommHandle.wait based operator statistics collection.
    Description: Run a 4-rank `(fsdp=2, tp=2)` soft-MoE training step with
        expert TP, outer FSDP, and a test-local Muon update. Then run the same
        uniquely-shaped AllReduce synchronously and asynchronously under the
        dump mode.
    Expectation: Both communication outputs are collected after native wait and
        have identical CRC32 and L2 norm.
    """
    dist.init()
    world_size = dist.get_group_size()
    assert world_size == 4, f"this case requires 4 ranks, got {world_size}"
    root_mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2),
        mesh_dim_names=("fsdp", "tp"),
    )
    dp_mesh = root_mesh["fsdp"]
    tp_mesh = root_mesh["tp"]
    model, input_placements = _build_model(dp_mesh, tp_mesh)

    with tempfile.TemporaryDirectory(prefix=f"hp_op_dump_rank_{dist.get_rank()}_") as output_root:
        dump_mode = MindSporeOpStatDumpMode(output_root, flush_every_op=True)
        dump_mode.set_step(0)
        with dump_mode:
            _run_training_step(model, tp_mesh, input_placements)
            _run_sync_async_reference_collectives()
        dump_mode.close()
        _assert_reference_dump_matches(
            Path(output_root) / "step_0" / f"rank_{dist.get_rank()}.csv"
        )

    print(
        f"[rank{dist.get_rank()}] FSDP+TP-MoE+Muon async dump matches sync: PASS",
        flush=True,
    )
