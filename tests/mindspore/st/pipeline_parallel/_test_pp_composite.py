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
"""End-to-end Pipeline Parallel + fully_shard verification for MindSpore.

The three current schedules (GPipe, 1F1B, Interleaved 1F1B / VPP) and any future
schedule family (e.g. V-shape: DualPipeV, ZeroBubbleV) share one accuracy story:
PP-stage gradients reduced over DP with SUM must equal the full-model gradient
accumulated serially over ``dp_size * num_microbatches`` micro-batches, under an
additive loss. They differ only along three orthogonal axes:

1. **Layer assignment** — how each ``pp_rank`` packs full-model layers into
   virtual stages (one contiguous stage for GPipe/1F1B, loop layout for VPP,
   V layout for future V-shape schedules).
2. **Schedule construction** — which scheduler runs over those stages.
3. **Loss owner** — which ``pp_rank`` holds the virtual stage that emits the
   final loss (for loop layout it is ``pp_size - 1``; V-shape schedules will
   override this).

``PipelineSpec`` captures these three axes; ``_assert_pp_fsdp_match_reference``
is the single driver that consumes a spec.  Adding a new schedule family means
adding one layout function and one ``PipelineSpec`` instance — the driver,
fully_shard wrapping, and grad parity logic are unchanged.
"""
# pylint: disable=wrong-import-position
import copy
import os
from dataclasses import dataclass
from typing import Callable

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

import mindspore as ms
import mindspore.communication.management as D
import numpy as np
from mindspore import Tensor, nn, mint
from mindspore.common import _no_grad
from hyper_parallel import PipelineStage, init_device_mesh
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.core.pipeline_parallel.scheduler import (
    PipelineScheduleRuntime,
    Schedule1F1B,
    ScheduleGPipe,
    ScheduleInterleaved1F1B,
)
from hyper_parallel.platform.mindspore.autograd_compat import enable_mindspore_backward_compat

ms.set_seed(42)
ms.set_deterministic(True)
ms.runtime.launch_blocking()
enable_mindspore_backward_compat()

D_HID = 8
TOTAL_LAYERS = 4
PP_SIZE = 2
RTOL = 1e-4
ATOL = 1e-5
NUM_TRAIN_STEPS = 5
LEARNING_RATE = 0.01


class MLPModule(nn.Cell):
    """Two-layer MLP block, mirroring PyTorch's ``MLPModule``."""

    def __init__(self, d_hid: int):
        super().__init__()
        self.net1 = nn.Dense(d_hid, d_hid)
        self.net2 = nn.Dense(d_hid, d_hid)

    def construct(self, x):
        return self.net2(self.net1(x))


class StageModel(nn.Cell):
    """Wraps a contiguous slice of MLP layers for a single pipeline (virtual) stage."""

    def __init__(self, layers):
        super().__init__()
        self.layers = nn.CellList(list(layers))

    def construct(self, x, prev_frozen=None):  # pylint: disable=W0613
        # prev_frozen: frozen tensor received from previous stage; ignored in computation.
        for layer in self.layers:
            x = layer(x)
        # Emit a pair (activation, frozen) to probe how PP handles requires_grad=False outputs.
        # frozen is a constant zero tensor with no grad history (acts as requires_grad=False).
        frozen = mint.zeros([1], dtype=ms.float32)
        return x, frozen


# ---------------------------------------------------------------------------
# Domain abstraction: ``PipelineSpec``
#
# A pipeline schedule's accuracy test is fully described by three axes:
#   * ``stage_layout`` — for a given (pp_idx, pp_size), return the ordered
#     list of (global_virtual_stage_index, [layer_indices_in_full_model])
#     tuples this pp_rank owns.  GPipe/1F1B return a single tuple per rank;
#     VPP returns ``n_local_stages`` tuples; future V-shape schedules
#     describe their V layout via this same callback.
#   * ``total_virtual_stages`` — total count of virtual stages across the
#     pipeline (``stage_num`` argument to ``PipelineStage``).
#   * ``last_stage_owner_pp_idx`` — which pp_rank owns the final virtual
#     stage that emits losses.  Loop / contiguous layouts use ``pp_size - 1``;
#     V-shape schedules can override.
#   * ``schedule_factory`` — constructs the scheduler.  Extra schedule
#     options (overlap flags, etc.) attach via lambda closure inside the
#     ``PipelineSpec`` instance.
# ---------------------------------------------------------------------------

VirtualStageAssignment = tuple[int, list[int]]
StageLayoutFn = Callable[[int, int], list[VirtualStageAssignment]]
TotalVirtualStagesFn = Callable[[int], int]
LossOwnerFn = Callable[[int], int]
ScheduleFactoryFn = Callable[[list[PipelineStage], int], PipelineScheduleRuntime]


@dataclass(frozen=True)
class PipelineSpec:
    """Schedule-agnostic description of a PP + fully_shard accuracy test.

    Attributes:
        name: Case identifier embedded in assertion messages and logs.
        stage_layout: ``(pp_idx, pp_size) -> [(virtual_stage_idx, [layer_idx, ...]), ...]``;
            lists this pp_rank's owned virtual stages in execution order.
        total_virtual_stages: ``pp_size -> int``; total ``stage_num`` for ``PipelineStage``.
        last_stage_owner_pp_idx: ``pp_size -> int``; pp_rank that runs the final virtual
            stage (and therefore emits losses).
        schedule_factory: ``(stages, micro_batch_num) -> PipelineScheduleRuntime``.
    """
    name: str
    stage_layout: StageLayoutFn
    total_virtual_stages: TotalVirtualStagesFn
    last_stage_owner_pp_idx: LossOwnerFn
    schedule_factory: ScheduleFactoryFn


# ---------------------------------------------------------------------------
# Layer-assignment strategies (one function per schedule family).
# ---------------------------------------------------------------------------


def _contiguous_layout(pp_idx: int, pp_size: int) -> list[VirtualStageAssignment]:
    """One virtual stage per pp_rank, holding a contiguous slice of layers.

    Used by GPipe and 1F1B: virtual_stage_idx == pp_idx, layers are
    ``[pp_idx * L, ..., pp_idx * L + L - 1]`` where ``L = TOTAL_LAYERS // pp_size``.
    """
    layers_per_stage = TOTAL_LAYERS // pp_size
    start = pp_idx * layers_per_stage
    return [(pp_idx, list(range(start, start + layers_per_stage)))]


def _loop_single_layer_layout(pp_idx: int, pp_size: int) -> list[VirtualStageAssignment]:
    """Loop layout for Interleaved 1F1B (VPP): ``n_local`` virtual stages per pp_rank,
    each holding exactly one layer.

    Virtual stage ``v`` is owned by ``pp_rank v % pp_size`` and maps to layer ``v``.
    With ``TOTAL_LAYERS=4, pp_size=2``:
      * pp_rank 0 -> virtual stages [0, 2] -> layers [0, 2]
      * pp_rank 1 -> virtual stages [1, 3] -> layers [1, 3]
    """
    return [
        (virtual_stage_idx, [virtual_stage_idx])
        for virtual_stage_idx in range(pp_idx, TOTAL_LAYERS, pp_size)
    ]


# ---------------------------------------------------------------------------
# Pipeline spec instances (one per schedule).  Adding a new schedule family
# means adding one spec instance — no change to the driver below.
# ---------------------------------------------------------------------------


def _last_stage_is_last_pp_rank(pp_size: int) -> int:
    """Default loss owner for contiguous / loop layouts."""
    return pp_size - 1


GPIPE_SPEC = PipelineSpec(
    name="pp_fsdp_gpipe",
    stage_layout=_contiguous_layout,
    total_virtual_stages=lambda pp_size: pp_size,
    last_stage_owner_pp_idx=_last_stage_is_last_pp_rank,
    schedule_factory=lambda stages, micro_batch_num: ScheduleGPipe(
        stages, micro_batch_num=micro_batch_num,
    ),
)

ONE_F_ONE_B_SPEC = PipelineSpec(
    name="pp_fsdp_1f1b",
    stage_layout=_contiguous_layout,
    total_virtual_stages=lambda pp_size: pp_size,
    last_stage_owner_pp_idx=_last_stage_is_last_pp_rank,
    schedule_factory=lambda stages, micro_batch_num: Schedule1F1B(
        stages, micro_batch_num=micro_batch_num,
    ),
)

VPP_SPEC = PipelineSpec(
    name="pp_fsdp_vpp",
    stage_layout=_loop_single_layer_layout,
    total_virtual_stages=lambda pp_size: TOTAL_LAYERS,
    last_stage_owner_pp_idx=_last_stage_is_last_pp_rank,
    schedule_factory=lambda stages, micro_batch_num: ScheduleInterleaved1F1B(
        stages, micro_batch_num=micro_batch_num,
    ),
)

# ---------------------------------------------------------------------------
# Reference-path helpers (shared across all specs).
# ---------------------------------------------------------------------------


def _build_full_layers() -> list[MLPModule]:
    """Build the full MLP stack; the global seed makes initial weights identical across ranks."""
    return [MLPModule(D_HID) for _ in range(TOTAL_LAYERS)]


def _build_pp_dp_mesh(world_size: int):
    """Build a 2D ``(pp, dp)`` device mesh; return ``None`` if ``PP_SIZE`` does not divide ``world_size``."""
    if world_size % PP_SIZE != 0 or world_size // PP_SIZE <= 1:
        return None, None, None, 0
    dp_size = world_size // PP_SIZE
    root_mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(PP_SIZE, dp_size),
        mesh_dim_names=("pp", "dp"),
    )
    return root_mesh, root_mesh["dp"], root_mesh["pp"], dp_size


def _to_numpy(value) -> np.ndarray:
    """Convert a Tensor / DTensor to numpy for comparison."""
    if hasattr(value, "to_local"):
        return value.to_local().asnumpy()
    return value.asnumpy()


def _global_inputs(dp_size: int, num_microbatches: int, step: int) -> Tensor:
    """Generate the global ``(dp_size * num_microbatches, D_HID)`` input; same on every rank."""
    rng = np.random.default_rng(2026 + step)
    return Tensor(rng.standard_normal((dp_size * num_microbatches, D_HID)).astype(np.float32))


def _build_ref_param_map(ref_layers: list[nn.Cell]) -> dict[str, ms.Parameter]:
    """Map full-model ``{layer_idx}.<name>`` to its parameter."""
    return {
        f"{layer_idx}.{sub_name}": param
        for layer_idx, layer in enumerate(ref_layers)
        for sub_name, param in layer.parameters_and_names()
    }


def _zero_grads(params) -> None:
    for param in params:
        param.grad = None


def _run_ref_serial(ref_layers: list[nn.Cell], global_inputs: Tensor, total_microbatches: int) -> Tensor:
    """Run the full reference model micro-batch by micro-batch; return the summed loss."""
    micro_batch = global_inputs.shape[0] // total_microbatches
    total_loss = None
    for mb_idx in range(total_microbatches):
        start = mb_idx * micro_batch
        x = global_inputs[start: start + micro_batch]
        for layer in ref_layers:
            x = layer(x)
        loss = mint.sum(x)
        loss.backward()
        total_loss = loss if total_loss is None else total_loss + loss
    return total_loss


def _assert_grad_parity(case_name: str, rank: int, step: int, dp_size: int, dp_idx: int,
                        stage_param_map: dict[str, ms.Parameter],
                        ref_param_map: dict[str, ms.Parameter]) -> None:
    """Compare per-parameter gradient between pipeline stages and the full reference model.

    Each stage grad is FSDP-sharded along dim-0 across ``dp_size`` ranks; slice the
    full ref grad to the matching dp shard before comparison.
    """
    for fqn, stage_param in stage_param_map.items():
        if stage_param.grad is None:
            continue
        stage_grad = _to_numpy(stage_param.grad)
        ref_grad_full = _to_numpy(ref_param_map[fqn].grad)
        shard_size = ref_grad_full.shape[0] // dp_size
        ref_grad = ref_grad_full[dp_idx * shard_size: (dp_idx + 1) * shard_size]
        print(f"{fqn} start to compare\n, stage_param_grad: {stage_param.grad},  ref_param_grad: {ref_grad}")
        assert np.allclose(stage_grad, ref_grad, rtol=RTOL, atol=ATOL), (
            f"{case_name}, rank {rank}, step {step}, param '{fqn}': "
            f"stage_grad={stage_grad}, ref_grad={ref_grad}"
        )
        print(f"{fqn} grad compare OK!", flush=True)


def check_param_and_grad(model, hint):
    if isinstance(model, list):
        for idx, l in enumerate(model):
            for name, param in l.parameters_and_names():
                print(f"{hint}: layer.{idx}.{name} param: {param}, grad is_row_tensor: {param.grad}")
        return
    for name, param in model.parameters_and_names():
        print(f"{hint}: {name} param: {param}, grad is_row_tensor: {param.grad}")


# ---------------------------------------------------------------------------
# Spec-driven FSDP-stage construction.
# ---------------------------------------------------------------------------


def _fsdp_mp_policy() -> MixedPrecisionPolicy:
    """Identity-dtype policy: fully_shard reduces grads in fp32, no cast."""
    return MixedPrecisionPolicy(
        param_dtype=ms.float32,
        reduce_dtype=ms.float32,
        output_dtype=ms.float32,
        cast_forward_inputs=False,
    )


def _wrap_stage_with_fsdp(stage_model: StageModel, dp_mesh, *, reduce_op: str) -> StageModel:
    """Apply per-layer + module-level ``fully_shard`` so DP behaves like ``replicate``.

    Single-layer virtual stages (VPP / V-shape) skip the per-layer wrap to avoid
    nesting an FSDP root inside another FSDP root with the same single child.
    """
    mp_policy = _fsdp_mp_policy()
    if len(stage_model.layers) > 1:
        for layer in stage_model.layers:
            fully_shard(layer, mesh=dp_mesh, reshard_after_forward=True, mp_policy=mp_policy)
            layer.set_reduce_op_type(reduce_op)
    fsdp_model = fully_shard(stage_model, mesh=dp_mesh, reshard_after_forward=True, mp_policy=mp_policy)
    fsdp_model.set_reduce_op_type(reduce_op)
    return fsdp_model


def _build_fsdp_stages(spec: PipelineSpec, base_layers: list[MLPModule], pp_idx: int,
                       pp_mesh, dp_mesh) -> tuple[list[StageModel], list[PipelineStage], dict[str, ms.Parameter]]:
    """Realise ``spec.stage_layout`` into FSDP-wrapped ``StageModel`` + ``PipelineStage`` pairs.

    Returns:
        fsdp_stages: ordered list of FSDP-wrapped ``StageModel`` instances owned by this pp_rank.
        pipeline_stages: matching ``PipelineStage`` instances ready for the scheduler.
        stage_param_map: maps each owned parameter's ``{global_layer_idx}.<sub_name>`` to the
            FSDP-wrapped parameter for grad-parity comparison against the reference model.
    """
    assignments = spec.stage_layout(pp_idx, PP_SIZE)
    total_virtual_stages = spec.total_virtual_stages(PP_SIZE)
    layers_copy = copy.deepcopy(base_layers)

    fsdp_stages: list[StageModel] = []
    pipeline_stages: list[PipelineStage] = []
    stage_param_map: dict[str, ms.Parameter] = {}
    for virtual_stage_idx, layer_indices in assignments:
        owned_layers = [layers_copy[layer_idx] for layer_idx in layer_indices]
        fsdp_stage = _wrap_stage_with_fsdp(StageModel(owned_layers), dp_mesh, reduce_op="sum")
        pipeline_stage = PipelineStage(
            fsdp_stage,
            stage_index=virtual_stage_idx,
            stage_num=total_virtual_stages,
            group=pp_mesh.get_group(),
            mesh=pp_mesh,
        )
        fsdp_stages.append(fsdp_stage)
        pipeline_stages.append(pipeline_stage)
        for local_pos, layer_idx in enumerate(layer_indices):
            for sub_name, param in fsdp_stage.layers[local_pos].parameters_and_names():
                stage_param_map[f"{layer_idx}.{sub_name}"] = param
    return fsdp_stages, pipeline_stages, stage_param_map


# ---------------------------------------------------------------------------
# Driver: one function consumes any ``PipelineSpec``.
# ---------------------------------------------------------------------------


def _assert_pp_fsdp_match_reference(spec: PipelineSpec, num_microbatches: int) -> None:
    """Run PP+fully_shard described by ``spec`` and verify loss + grad parity against
    a full-model serial reference for ``NUM_TRAIN_STEPS`` steps."""
    D.init()
    rank = D.get_rank()
    world_size = D.get_group_size()

    root_mesh, dp_mesh, pp_mesh, dp_size = _build_pp_dp_mesh(world_size)
    if root_mesh is None:
        print(f"[Rank {rank}] Skip {spec.name}: world_size={world_size} cannot form (pp={PP_SIZE}, dp) mesh.")
        return

    base_layers = _build_full_layers()
    pp_idx = pp_mesh.get_local_rank()
    dp_idx = dp_mesh.get_local_rank()
    is_loss_owner = pp_idx == spec.last_stage_owner_pp_idx(PP_SIZE)
    total_microbatches = dp_size * num_microbatches

    # FSDP path: each pp_rank's virtual stages, each wrapped with fully_shard (reduce_op=SUM).
    fsdp_stages, pipeline_stages, stage_param_map = _build_fsdp_stages(
        spec, base_layers, pp_idx, pp_mesh, dp_mesh,
    )
    schedule = spec.schedule_factory(pipeline_stages, num_microbatches)
    # Each single-layer virtual stage (VPP / future V-shape) names its sole layer ``layers.0.*``;
    # combining params from multiple virtual stages into one optimizer would produce duplicate
    # names.  Build one optimizer per stage uniformly — for GPipe / 1F1B this degenerates to a
    # single-element list and is behaviourally identical to a single optimizer.
    fsdp_stage_params = [tuple(fsdp_stage.trainable_params()) for fsdp_stage in fsdp_stages]
    fsdp_optimizers = [
        nn.Adam(stage_params, learning_rate=LEARNING_RATE) for stage_params in fsdp_stage_params
    ]
    fsdp_params = tuple(param for stage_params in fsdp_stage_params for param in stage_params)

    # Reference path: every rank holds the full model and serially runs all micro-batches.
    ref_layers = copy.deepcopy(base_layers)
    ref_params = tuple(StageModel(ref_layers).trainable_params())
    ref_optimizer = nn.Adam(ref_params, learning_rate=LEARNING_RATE)
    ref_param_map = _build_ref_param_map(ref_layers)

    for step in range(NUM_TRAIN_STEPS):
        _zero_grads(fsdp_params)
        _zero_grads(ref_params)

        global_inputs = _global_inputs(dp_size, num_microbatches, step)
        fsdp_inputs = global_inputs[dp_idx * num_microbatches: (dp_idx + 1) * num_microbatches]

        fsdp_losses = schedule.run(fsdp_inputs) if pp_idx == 0 else schedule.run()
        ref_total_loss = _run_ref_serial(ref_layers, global_inputs, total_microbatches)

        if is_loss_owner:
            micro_loss_accu = Tensor([0.0])
            for sub_loss in fsdp_losses:
                # StageModel returns (activation, label, frozen); extract the activation.
                activation = sub_loss[0] if isinstance(sub_loss, tuple) else sub_loss
                micro_loss_accu += activation.sum()

            # The loss-owner pp_rank emits per-microbatch losses on its dp shard;
            # all-reduce over the dp group recovers the global summed loss.
            mint.distributed.all_reduce(
                micro_loss_accu, op="sum", group=dp_mesh.get_group(),
            )
            ref_total_value = _to_numpy(ref_total_loss)
            assert np.allclose(micro_loss_accu.asnumpy(), ref_total_value, rtol=RTOL, atol=ATOL), (
                f"{spec.name}, rank {rank}, step {step}: "
                f"fsdp_global_loss={micro_loss_accu.asnumpy()}, "
                f"ref_total_loss={ref_total_value}"
            )
        for local_pos, fsdp_stage in enumerate(fsdp_stages):
            check_param_and_grad(fsdp_stage, f"{spec.name}_stage_{local_pos}")
        check_param_and_grad(ref_layers, "ref_layers")
        _assert_grad_parity(spec.name, rank, step, dp_size, dp_idx, stage_param_map, ref_param_map)
        with _no_grad():
            for stage_optimizer, stage_params in zip(fsdp_optimizers, fsdp_stage_params):
                stage_optimizer(tuple(p.grad for p in stage_params))
            ref_optimizer(tuple(p.grad for p in ref_params))

    print(
        f"[Rank {rank}] {spec.name} passed with mesh={root_mesh.mesh_shape}, "
        f"dp_size={dp_size}, pp_idx={pp_idx}, "
        f"owned_virtual_stages={[v for v, _ in spec.stage_layout(pp_idx, PP_SIZE)]}, "
        f"steps={NUM_TRAIN_STEPS}, num_microbatches={num_microbatches}"
    )


# ---------------------------------------------------------------------------
# Public test entries (invoked from test_pp_composite.py via msrun).
# ---------------------------------------------------------------------------


def test_fully_shard_pp_gpipe():
    """Pipeline parallel + fully_shard (replicate-style) under the GPipe schedule."""
    _assert_pp_fsdp_match_reference(GPIPE_SPEC, num_microbatches=4)


def test_fully_shard_pp_1f1b():
    """Pipeline parallel + fully_shard (replicate-style) under the 1F1B schedule."""
    _assert_pp_fsdp_match_reference(ONE_F_ONE_B_SPEC, num_microbatches=4)


def test_fully_shard_pp_vpp():
    """Pipeline parallel + fully_shard under the Interleaved 1F1B (VPP) schedule."""
    _assert_pp_fsdp_match_reference(VPP_SPEC, num_microbatches=4)
