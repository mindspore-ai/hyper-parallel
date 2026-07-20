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
from typing import Optional

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

import mindspore as ms
import mindspore.communication.management as D
import numpy as np
from mindspore import Tensor, nn, mint

from hyper_parallel import init_device_mesh
from hyper_parallel.core.fully_shard.api import fully_shard, HSDPModule
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.platform import get_platform
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


def _wrap_with_fsdp(
    model: FullModel,
    dp_mesh,
    *,
    sharded_accumulated_grad: bool = False,
    sharded_grad_ready_overlap: bool = False,
    comm_fusion: bool = False,
    sharded_accumulated_grad_max_pending: int = 1,
    sharded_grad_reduce_dtype: Optional[ms.Type] = None,
    per_layer_fsdp: bool = False,
) -> FullModel:
    """Apply per-layer + module-level fully_shard, with sum-reduce for exact grad parity."""
    mp_policy = MixedPrecisionPolicy(
        param_dtype=ms.float32,
        reduce_dtype=ms.float32,
        output_dtype=ms.float32,
        cast_forward_inputs=False,
    )
    fsdp_kwargs = {
        "mesh": dp_mesh,
        "reshard_after_forward": False,
        "mp_policy": mp_policy,
        "comm_fusion": comm_fusion,
        "sharded_accumulated_grad": sharded_accumulated_grad,
        "sharded_grad_ready_overlap": sharded_grad_ready_overlap,
        "sharded_accumulated_grad_max_pending": sharded_accumulated_grad_max_pending,
        "sharded_grad_reduce_dtype": sharded_grad_reduce_dtype,
    }
    if per_layer_fsdp:
        for layer in model.layers:
            fully_shard(layer, **fsdp_kwargs)
    fsdp_model = fully_shard(model, **fsdp_kwargs)
    for hsdp_state in _get_hsdp_states(fsdp_model):
        hsdp_state.set_reduce_op_type("sum")
    return fsdp_model


def _get_hsdp_states(fsdp_model: FullModel) -> list:
    """Return every distinct FSDP state managed below ``fsdp_model``."""
    hsdp_states = []
    seen_states = set()
    for _, submod in get_platform().get_cells_and_names(fsdp_model):
        if not isinstance(submod, HSDPModule):
            continue
        hsdp_state = submod.hsdp_scheduler.hsdp_state
        if id(hsdp_state) in seen_states:
            continue
        seen_states.add(id(hsdp_state))
        hsdp_states.append(hsdp_state)
    return hsdp_states


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
                   reshard_after_backward: bool, use_explicit_unshard: bool,
                   explicit_sharded_finalize: bool = False) -> list[Tensor]:
    """Run 1F1B-style micro-batching on a fully_shard-wrapped model.

    The default path syncs only the last micro-batch. The sharded-accumulation
    path keeps every micro-batch in no-sync mode and explicitly finalizes after
    backward, matching PipelineStage's FSDP_REDUCE_GRAD action.
    """
    if use_explicit_unshard:
        for layer in fsdp_model.layers:
            if isinstance(layer, HSDPModule):
                layer.unshard()
        if isinstance(fsdp_model, HSDPModule):
            fsdp_model.unshard()

    hsdp_states = _get_hsdp_states(fsdp_model)
    losses = []
    last_idx = len(inputs_per_mb) - 1
    for mb_idx, inp in enumerate(inputs_per_mb):
        is_last = mb_idx == last_idx and not explicit_sharded_finalize
        fsdp_model.set_requires_gradient_sync(is_last)
        fsdp_model.set_is_last_backward(is_last)
        if not reshard_after_backward:
            fsdp_model.set_reshard_after_backward(is_last)
        loss = mint.sum(fsdp_model(inp))
        loss.backward()
        if explicit_sharded_finalize:
            for hsdp_state in hsdp_states:
                hsdp_state.flush_sharded_accumulation_after_backward()
        losses.append(loss)
    if explicit_sharded_finalize:
        fsdp_model.set_is_last_backward(True)
        fsdp_model.set_reshard_after_backward(True)
        fsdp_model.set_requires_gradient_sync(True)
        fsdp_model.hsdp_scheduler._root_backward_hook(  # pylint: disable=protected-access
            force_reduce=True
        )
        for hsdp_state in hsdp_states:
            hsdp_state.launch_sharded_accumulated_grad_all_reduces()
        for hsdp_state in hsdp_states:
            hsdp_state.wait_sharded_accumulated_grad_all_reduces()
        for hsdp_state in hsdp_states:
            if not hsdp_state.is_shard:
                hsdp_state.shard()
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
                        gradient_shard_size: int,
                        gradient_shard_rank: int,
                        rtol: float = RTOL,
                        atol: float = ATOL) -> None:
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
        start = gradient_shard_rank * ref_gradient_chunk_size
        ref_grad = _to_numpy(ref_p.grad[start: start + ref_gradient_chunk_size])
        assert np.allclose(fsdp_grad, ref_grad, rtol=rtol, atol=atol), (
            f"{case_name}, rank {rank}, param {idx} ({fsdp_p.name}): "
            f"fsdp_grad={fsdp_grad}, ref_grad={ref_grad}"
        )


def _assert_fully_shard_simu_pp_match_reference(*, case_name: str, num_microbatches: int,
                                                use_explicit_unshard: bool,
                                                reshard_after_backward: bool,
                                                sharded_accumulated_grad: bool = False,
                                                sharded_grad_ready_overlap: bool = False,
                                                use_hsdp_mesh: bool = False,
                                                comm_fusion: bool = False,
                                                sharded_accumulated_grad_max_pending: int = 1,
                                                sharded_grad_reduce_dtype: Optional[ms.Type] = None,
                                                grad_rtol: float = RTOL,
                                                grad_atol: float = ATOL,
                                                per_layer_fsdp: bool = False) -> None:
    """Run fully_shard 1F1B-style micro-batching and compare loss + grad against the single-card baseline."""
    D.init()
    rank = D.get_rank()
    world_size = D.get_group_size()

    if use_hsdp_mesh:
        if world_size % 2 != 0:
            raise ValueError(f"HSDP test requires an even world size, but got {world_size}.")
        dp_mesh = init_device_mesh(
            device_type="npu",
            mesh_shape=(world_size // 2, 2),
            mesh_dim_names=("replicate", "shard"),
        )
        gradient_shard_size = 2
        gradient_shard_rank = dp_mesh.get_local_rank("shard")
    else:
        dp_mesh = init_device_mesh(device_type="npu", mesh_shape=(world_size,), mesh_dim_names=("dp",))
        gradient_shard_size = world_size
        gradient_shard_rank = rank

    base_layers = [MLPModule(D_HID) for _ in range(TOTAL_LAYERS)]
    fsdp_model = _wrap_with_fsdp(
        FullModel(copy.deepcopy(base_layers)),
        dp_mesh,
        sharded_accumulated_grad=sharded_accumulated_grad,
        sharded_grad_ready_overlap=sharded_grad_ready_overlap,
        comm_fusion=comm_fusion,
        sharded_accumulated_grad_max_pending=sharded_accumulated_grad_max_pending,
        sharded_grad_reduce_dtype=sharded_grad_reduce_dtype,
        per_layer_fsdp=per_layer_fsdp,
    )
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
        explicit_sharded_finalize=sharded_accumulated_grad,
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
        gradient_shard_size, gradient_shard_rank,
        grad_rtol, grad_atol,
    )

    print(
        f"[Rank {rank}] {case_name} passed with world_size={world_size}, "
        f"num_microbatches={num_microbatches}, use_explicit_unshard={use_explicit_unshard}, "
        f"reshard_after_backward={reshard_after_backward}, "
        f"sharded_accumulated_grad={sharded_accumulated_grad}, "
        f"sharded_grad_ready_overlap={sharded_grad_ready_overlap}, "
        f"comm_fusion={comm_fusion}, "
        f"sharded_accumulated_grad_max_pending={sharded_accumulated_grad_max_pending}, "
        f"sharded_grad_reduce_dtype={sharded_grad_reduce_dtype}, "
        f"per_layer_fsdp={per_layer_fsdp}, "
        f"use_hsdp_mesh={use_hsdp_mesh}"
    )


def test_fully_shard_simu_pp_implicit_unshard_reshard():
    """1F1B-style micro-batching: implicit unshard, reshard after every backward."""
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


def test_fully_shard_simu_pp_sharded_accumulated_grad():
    """Pipeline-style no-sync micro-batches should accumulate reduced gradient shards."""
    _assert_fully_shard_simu_pp_match_reference(
        case_name="fully_shard_simu_pp_sharded_accumulated_grad",
        num_microbatches=4,
        use_explicit_unshard=False,
        reshard_after_backward=False,
        sharded_accumulated_grad=True,
    )


def test_fully_shard_simu_pp_hsdp_sharded_accumulated_grad():
    """Fused HSDP should reduce every micro-batch and finalize local shards."""
    _assert_fully_shard_simu_pp_match_reference(
        case_name="fully_shard_simu_pp_hsdp_sharded_accumulated_grad",
        num_microbatches=3,
        use_explicit_unshard=False,
        reshard_after_backward=False,
        sharded_accumulated_grad=True,
        use_hsdp_mesh=True,
        comm_fusion=True,
    )


def test_fully_shard_simu_pp_hsdp_sharded_accumulated_grad_ready():
    """Fused HSDP grad-ready hooks should reduce each micro-batch exactly once."""
    _assert_fully_shard_simu_pp_match_reference(
        case_name="fully_shard_simu_pp_hsdp_sharded_accumulated_grad_ready",
        num_microbatches=3,
        use_explicit_unshard=False,
        reshard_after_backward=False,
        sharded_accumulated_grad=True,
        sharded_grad_ready_overlap=True,
        use_hsdp_mesh=True,
        comm_fusion=True,
    )


def test_fully_shard_simu_pp_hsdp_sharded_accumulated_grad_bf16_rs():
    """BF16 micro-batch RS should retain FP32 local accumulation with bounded error."""
    _assert_fully_shard_simu_pp_match_reference(
        case_name="fully_shard_simu_pp_hsdp_sharded_accumulated_grad_bf16_rs",
        num_microbatches=3,
        use_explicit_unshard=False,
        reshard_after_backward=False,
        sharded_accumulated_grad=True,
        use_hsdp_mesh=True,
        comm_fusion=True,
        sharded_grad_reduce_dtype=ms.bfloat16,
        grad_rtol=2e-2,
        grad_atol=2e-2,
    )


def test_fully_shard_simu_pp_hsdp_sharded_accumulated_grad_pending_window():
    """Multiple FSDP states should keep a bounded fused RS window without losing gradients."""
    _assert_fully_shard_simu_pp_match_reference(
        case_name="fully_shard_simu_pp_hsdp_sharded_accumulated_grad_pending_window",
        num_microbatches=3,
        use_explicit_unshard=False,
        reshard_after_backward=False,
        sharded_accumulated_grad=True,
        use_hsdp_mesh=True,
        comm_fusion=True,
        sharded_accumulated_grad_max_pending=2,
        per_layer_fsdp=True,
    )
