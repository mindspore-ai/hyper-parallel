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
"""fully_shard simulating 1F1B-style micro-batching, verified against a single-card baseline.

Adapted from PyTorch's ``test_fully_shard_training.test_1f1b_microbatching``. The original
test mirrors the gradient-sync pattern used by a real 1F1B pipeline schedule, but only
exercises the ``fully_shard`` API (no ``PipelineStage``). This test does the same.

Precision strategy (DP=world_size vs single-card):
    * Global input shape ``(world_size * MICRO_BATCH, D_HID)`` generated from a shared seed
      so every rank produces the identical tensor without an explicit broadcast.
    * FSDP path: each rank consumes its own ``MICRO_BATCH`` slice (different inputs per
      rank) and runs ``num_microbatches`` micro-batches; only the last triggers gradient
      sync, mimicking 1F1B. ``set_reduce_op_type("sum")`` makes dp gradients sum-reduced.
    * Baseline path: every rank holds an unwrapped full-model deepcopy and consumes the
      entire global input, micro-batch by micro-batch. Backward accumulates locally; no
      manual all-reduce is needed because the ``mint.sum`` loss is additive.
    * Each rank's stage gradient = global accumulated gradient on both paths (assuming
      HyperParallel does NOT implicitly scale per-microbatch gradients).
"""
# pylint: disable=wrong-import-position
import copy
import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

import mindspore as ms
import mindspore.communication.management as D
import numpy as np
from mindspore import Tensor, nn, mint

from hyper_parallel import init_device_mesh
from hyper_parallel.core.fully_shard.api import fully_shard, HSDPModule
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.platform.mindspore.autograd_compat import enable_mindspore_backward_compat

ms.set_seed(42)
ms.set_deterministic(True)
enable_mindspore_backward_compat()

D_HID = 8
TOTAL_LAYERS = 4
MICRO_BATCH = 2
RTOL = 1e-4
ATOL = 1e-5


class MLPModule(nn.Cell):
    """Two-layer MLP block."""

    def __init__(self, d_hid: int):
        super().__init__()
        self.net1 = nn.Dense(d_hid, d_hid)
        self.net2 = nn.Dense(d_hid, d_hid)
        self.relu = nn.ReLU()

    def construct(self, x):
        return self.net2(self.relu(self.net1(x)))


class FullModel(nn.Cell):
    """Stack of ``TOTAL_LAYERS`` MLP blocks."""

    def __init__(self, layers):
        super().__init__()
        self.layers = nn.CellList(list(layers))

    def construct(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _wrap_with_fsdp(model: FullModel, dp_mesh) -> FullModel:
    """Apply per-layer + module-level fully_shard, with sum-reduce for exact grad parity."""
    mp_policy = MixedPrecisionPolicy(
        param_dtype=ms.float32,
        reduce_dtype=ms.float32,
        output_dtype=ms.float32,
        cast_forward_inputs=False,
    )
    fsdp_model = fully_shard(model, mesh=dp_mesh, reshard_after_forward=False, mp_policy=mp_policy)
    fsdp_model.set_reduce_op_type("sum")
    return fsdp_model


def _global_inputs(num_rows: int) -> Tensor:
    """Generate a deterministic ``(num_rows, D_HID)`` input shared across ranks."""
    rng = np.random.default_rng(2026)
    return Tensor(rng.standard_normal((num_rows, D_HID)).astype(np.float32))


def _split_microbatches(inputs: Tensor, num_microbatches: int) -> list[Tensor]:
    """Split a contiguous batch into ``num_microbatches`` equally-sized micro-batches."""
    return [inputs[i * MICRO_BATCH: (i + 1) * MICRO_BATCH] for i in range(num_microbatches)]


def _to_numpy(value) -> np.ndarray:
    """Convert a Tensor / DTensor to numpy for comparison."""
    if hasattr(value, "to_local"):
        return value.to_local().asnumpy()
    return value.asnumpy()


def _run_fsdp_1f1b(fsdp_model: FullModel, inputs_per_mb: list[Tensor], *,
                   reshard_after_backward: bool, use_explicit_unshard: bool) -> list[Tensor]:
    """Run 1F1B-style micro-batching on a fully_shard-wrapped model.

    Only the last micro-batch triggers gradient sync / final reshard, batching the dp
    communication once per optimizer step (matching how a real 1F1B schedule behaves).
    """
    if use_explicit_unshard:
        for layer in fsdp_model.layers:
            if isinstance(layer, HSDPModule):
                layer.unshard()
        if isinstance(fsdp_model, HSDPModule):
            fsdp_model.unshard()

    losses = []
    last_idx = len(inputs_per_mb) - 1
    for mb_idx, inp in enumerate(inputs_per_mb):
        is_last = mb_idx == last_idx
        fsdp_model.set_requires_gradient_sync(is_last)
        fsdp_model.set_is_last_backward(is_last)
        if not reshard_after_backward:
            fsdp_model.set_reshard_after_backward(is_last)
        loss = mint.sum(fsdp_model(inp))
        loss.backward()
        losses.append(loss)
    return losses


def _run_ref_serial(ref_model: FullModel, inputs_per_mb: list[Tensor]) -> list[Tensor]:
    """Run the same micro-batches on an unwrapped reference; gradients accumulate locally."""
    losses = []
    for inp in inputs_per_mb:
        loss = mint.sum(ref_model(inp))
        loss.backward()
        losses.append(loss)
    return losses


def _assert_grad_parity(case_name: str, rank: int,
                        fsdp_params: tuple, ref_params: tuple,
                        gradient_shard_size) -> None:
    """Verify per-parameter grad equality between the FSDP and single-card baseline."""
    assert len(fsdp_params) == len(ref_params), (
        f"{case_name}, rank {rank}: param count mismatch, "
        f"fsdp={len(fsdp_params)}, ref={len(ref_params)}"
    )
    for idx, (fsdp_p, ref_p) in enumerate(zip(fsdp_params, ref_params)):
        if fsdp_p.grad is None and ref_p.grad is None:
            continue
        ref_gradient_chunk_size = ref_p.grad.shape[0] // gradient_shard_size
        fsdp_grad = _to_numpy(fsdp_p.grad)
        ref_grad = _to_numpy(ref_p.grad[rank * ref_gradient_chunk_size: (rank + 1) * ref_gradient_chunk_size])
        assert np.allclose(fsdp_grad, ref_grad, rtol=RTOL, atol=ATOL), (
            f"{case_name}, rank {rank}, param {idx} ({fsdp_p.name}): "
            f"fsdp_grad={fsdp_grad}, ref_grad={ref_grad}"
        )


def _assert_fully_shard_simu_pp_match_reference(*, case_name: str, num_microbatches: int,
                                                use_explicit_unshard: bool,
                                                reshard_after_backward: bool) -> None:
    """Run fully_shard 1F1B-style micro-batching and compare loss + grad against the single-card baseline."""
    D.init()
    rank = D.get_rank()
    world_size = D.get_group_size()

    dp_mesh = init_device_mesh(device_type="npu", mesh_shape=(world_size,), mesh_dim_names=("dp",))

    base_layers = [MLPModule(D_HID) for _ in range(TOTAL_LAYERS)]
    fsdp_model = _wrap_with_fsdp(FullModel(copy.deepcopy(base_layers)), dp_mesh)
    ref_model = FullModel(copy.deepcopy(base_layers))

    rows_per_rank = num_microbatches * MICRO_BATCH
    total_microbatches = world_size * num_microbatches
    global_inputs = _global_inputs(world_size * rows_per_rank)
    fsdp_inputs_per_mb = _split_microbatches(
        global_inputs[rank * rows_per_rank: (rank + 1) * rows_per_rank],
        num_microbatches,
    )
    ref_inputs_per_mb = _split_microbatches(global_inputs, total_microbatches)

    fsdp_losses = _run_fsdp_1f1b(
        fsdp_model, fsdp_inputs_per_mb,
        reshard_after_backward=reshard_after_backward,
        use_explicit_unshard=use_explicit_unshard,
    )
    ref_losses = _run_ref_serial(ref_model, ref_inputs_per_mb)

    # FSDP rank `i` consumes the same micro-batches as ref's slice [i*N : (i+1)*N].
    fsdp_loss_sum = sum(_to_numpy(loss) for loss in fsdp_losses)
    ref_slice = ref_losses[rank * num_microbatches: (rank + 1) * num_microbatches]
    ref_loss_sum = sum(_to_numpy(loss) for loss in ref_slice)
    assert np.allclose(fsdp_loss_sum, ref_loss_sum, rtol=RTOL, atol=ATOL), (
        f"{case_name}, rank {rank}: fsdp_loss_sum={fsdp_loss_sum}, ref_loss_sum={ref_loss_sum}"
    )

    _assert_grad_parity(
        case_name, rank,
        tuple(fsdp_model.trainable_params()), tuple(ref_model.trainable_params()),
        dp_mesh.size(),
    )

    print(
        f"[Rank {rank}] {case_name} passed with world_size={world_size}, "
        f"num_microbatches={num_microbatches}, use_explicit_unshard={use_explicit_unshard}, "
        f"reshard_after_backward={reshard_after_backward}"
    )


def test_fully_shard_simu_pp_implicit_unshard_reshard():
    """1F1B-style micro-batching: implicit unshard, reshard after every backward."""
    # msrun --worker_num=4 --local_worker_num=4 --join=True --log_dir=log_test_fully_shard_simu_pp_implicit_unshard_reshard pytest -sv _test_fully_shard_simu_pp::test_fully_shard_simu_pp_implicit_unshard_reshard
    _assert_fully_shard_simu_pp_match_reference(
        case_name="fully_shard_simu_pp_implicit_unshard_reshard",
        num_microbatches=4,
        use_explicit_unshard=False,
        reshard_after_backward=True,
    )


def test_fully_shard_simu_pp_explicit_unshard_no_reshard():
    """1F1B-style micro-batching: explicit unshard, only reshard on the last micro-batch."""
    _assert_fully_shard_simu_pp_match_reference(
        case_name="fully_shard_simu_pp_explicit_unshard_no_reshard",
        num_microbatches=4,
        use_explicit_unshard=True,
        reshard_after_backward=False,
    )
