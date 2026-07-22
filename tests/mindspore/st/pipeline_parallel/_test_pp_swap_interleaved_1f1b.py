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
"""MindSpore real-overlap activation-swap accuracy and memory comparison."""
# pylint: disable=wrong-import-position
import os
import pickle
import tempfile
import threading

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"
os.environ["MS_DEV_RUNTIME_CONF"] = "memory_statistics:True"

import numpy as np
from mindspore import Tensor

from hyper_parallel import DTensor, PipelineStage
from hyper_parallel.core.activation_checkpoint import CheckpointPolicy, swap_wrapper
from hyper_parallel.core.activation_checkpoint.swap import SwapManager
from hyper_parallel.core.pipeline_parallel import (
    CommComputeOverlap,
    MetaStepType,
    ScheduleInterleaved1F1B,
)
from hyper_parallel.core.pipeline_parallel.hook_coordinator import HookRole
from tests.mindspore.st.pipeline_parallel.pp_overlap_moe_poc import (
    BS as _BATCH_SIZE,
    HIDDEN_SIZE as _HIDDEN_SIZE,
    MICRO_BATCH_NUM as _MICRO_BATCH_NUM,
    SEQ_LEN as _SEQ_LEN,
    VIRTUAL_STAGES as _VIRTUAL_STAGES,
    _TinyConfig,
    _build_pipeline,
    _init_pp_ep_mesh,
)
from tests.mindspore.st.pipeline_parallel.pp_swap_test_utils import (
    assert_step_memory_stable,
    current_step_peak_memory_mb,
    format_step_peaks,
    reset_step_peak_memory,
    steady_peak_memory_mb,
)

_BASELINE_PREFIX = os.path.join(tempfile.gettempdir(), "pp_swap_overlap_moe_baseline")
_CHECK_STEPS = int(os.environ.get("PP_SWAP_MOE_CHECK_STEPS", "3"))


def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in ("1", "true", "yes", "on")


def _scenario() -> str:
    return "overlap_b_f_dxdw" if _env_flag("PP_SWAP_MOE_DXDW") else "overlap_b_f"


def _baseline_file(rank: int) -> str:
    tag = os.environ.get("PP_SWAP_INTERLEAVED_BASELINE_TAG", "default")
    return f"{_BASELINE_PREFIX}_{tag}_{_scenario()}_rank{rank}.pkl"


def _save_baseline(
        rank: int,
        losses: np.ndarray,
        grads: list,
        peak_memory_mb: float,
        step_peaks_mb: list[float],
) -> None:
    """Persist one rank's no-swap losses, gradients, and peak memory."""
    tmp_file = _baseline_file(rank) + ".tmp"
    with open(tmp_file, "wb") as file:
        pickle.dump(
            {
                "losses": losses,
                "grads": grads,
                "peak_memory_mb": peak_memory_mb,
                "step_peaks_mb": step_peaks_mb,
            },
            file,
        )
    os.rename(tmp_file, _baseline_file(rank))


def _load_baseline(rank: int) -> dict:
    with open(_baseline_file(rank), "rb") as file:
        return pickle.load(file)


def _clear_grads(stages: list[PipelineStage]) -> None:
    for stage in stages:
        for param in stage.submodule.trainable_params():
            param.grad = None


def _count_nonzero_grads(stages: list[PipelineStage]) -> tuple[int, int]:
    """Return nonzero and total trainable-gradient counts for local stages."""
    nonzero_grads = 0
    total_grads = 0
    for stage in stages:
        for param in stage.submodule.trainable_params():
            total_grads += 1
            if param.grad is None:
                continue
            grad = param.grad.to_local() if isinstance(param.grad, DTensor) else param.grad
            if np.any(grad.asnumpy() != 0):
                nonzero_grads += 1
    return nonzero_grads, total_grads


def _make_overlap_callback(overlap: CommComputeOverlap, stats: dict):
    """Run swap-aware FWD and direct BWD through hook coordination."""
    main_thread_id = threading.get_ident()

    def _callback(step, ctx):
        bwd_step, fwd_step = step.sub_steps
        bwd_stage = ctx.schedule._stage_dict[bwd_step.stage_index]  # pylint: disable=protected-access

        def fwd_fn():
            ctx.schedule.execute_fwd_leaf(fwd_step, ctx.arg_mbs, ctx.kwarg_mbs, ctx.losses)

        def bwd_fn():
            from mindspore.common.api import _pynative_executor  # pylint: disable=C0415

            _pynative_executor.set_enable_grad(True)
            stats["bwd_thread_ids"].add(threading.get_ident())
            ctx.schedule.wait_bwd_recv(bwd_stage.stage_index, bwd_step.micro_index)
            if bwd_step.type == MetaStepType.BWD_INPUT:
                bwd_stage.backward_input_one_chunk(bwd_step.micro_index)
            else:
                bwd_stage.backward_one_chunk(bwd_step.micro_index)
            if overlap.coordinator.is_enabled():
                overlap.coordinator.rendezvous(HookRole.COMPUTE)

        bwd_stage.recompute_one_chunk(bwd_step.micro_index)
        stats["callback_count"] += 1
        overlap.run(fwd_fn=fwd_fn, bwd_fn=bwd_fn)

    stats["main_thread_id"] = main_thread_id
    return _callback


def _losses_to_numpy(losses: list) -> np.ndarray:
    """Copy the last-stage per-microbatch losses to host."""
    if not losses:
        return np.array([], dtype=np.float32)
    return np.array([float(loss.mean().asnumpy()) for loss in losses], dtype=np.float32)


def _grads_to_numpy(stages: list[PipelineStage]) -> list:
    """Copy all local parameter-gradient shards to host."""
    grads = []
    for stage in stages:
        for param in stage.submodule.trainable_params():
            grad = param.grad
            if grad is None:
                grads.append(None)
                continue
            local_grad = grad.to_local() if isinstance(grad, DTensor) else grad
            grads.append(local_grad.asnumpy().copy())
    return grads


def _build_swap_stages(
        pp_rank: int,
        device,
        pp_mesh,
        ep_mesh,
        overlap: CommComputeOverlap,
        enable_swap: bool,
        swap_stats: dict,
) -> list[PipelineStage]:
    """Build the existing MoE overlap model and optionally wrap its layers."""
    chunks, stage_indices = _build_pipeline(
        pp_rank,
        ep_mesh,
        _TinyConfig(),
        use_overlap=True,
        overlap=overlap,
    )
    if enable_swap:
        def policy_fn(tensor):  # pylint: disable=unused-argument
            swap_stats["policy_calls"] += 1
            return CheckpointPolicy.MUST_SWAP

        for chunk in chunks:
            for layer_index, layer in enumerate(chunk.layers):
                chunk.layers[layer_index] = swap_wrapper(layer, policy_fn=policy_fn)
    return [
        PipelineStage(
            chunk,
            stage_index=stage_index,
            stage_num=_VIRTUAL_STAGES,
            device=device,
            mesh=pp_mesh,
        )
        for chunk, stage_index in zip(chunks, stage_indices)
    ]


def _run_phase(enable_swap: bool):
    """Run the existing hook-coordinated PP+EP model for three steps."""
    rank, device, pp_mesh, ep_mesh = _init_pp_ep_mesh()
    pp_rank = pp_mesh.get_local_rank()
    overlap = CommComputeOverlap()
    swap_stats = {"policy_calls": 0}
    stages = _build_swap_stages(
        pp_rank,
        device,
        pp_mesh,
        ep_mesh,
        overlap,
        enable_swap,
        swap_stats,
    )
    schedule = ScheduleInterleaved1F1B(
        stages,
        micro_batch_num=_MICRO_BATCH_NUM,
        overlap_p2p=True,
        overlap_b_f=True,
        enable_dxdw_split=_env_flag("PP_SWAP_MOE_DXDW"),
        swap=enable_swap,
    )
    overlap_stats = {"callback_count": 0, "bwd_thread_ids": set()}
    schedule.register_custom_function(
        MetaStepType.OVERLAP_B_F,
        _make_overlap_callback(overlap, overlap_stats),
    )
    if enable_swap:
        assert any(
            step is not None and step.type.name.startswith("SWAP_")
            for step in schedule.exec_order[pp_rank]
        ), f"rank {rank} has no pipeline-swap control steps"

    losses_by_step = []
    grads_by_step = []
    step_peaks_mb = []
    phase = "swap" if enable_swap else "no-swap"
    previous_policy_calls = 0
    for step in range(_CHECK_STEPS):
        _clear_grads(stages)
        inputs = Tensor(
            np.random.RandomState(100 + pp_rank + step * 1000).randn(  # pylint: disable=no-member
                _BATCH_SIZE,
                _SEQ_LEN,
                _HIDDEN_SIZE,
            ).astype(np.float32),
        )
        reset_step_peak_memory()
        losses = schedule.run(inputs) if pp_rank == 0 else schedule.run()
        step_peaks_mb.append(current_step_peak_memory_mb())
        active_groups = SwapManager().active_group_count()
        nonzero_grads, total_grads = _count_nonzero_grads(stages)
        step_policy_calls = swap_stats["policy_calls"] - previous_policy_calls
        previous_policy_calls = swap_stats["policy_calls"]
        assert active_groups == 0, f"{_scenario()} {phase} leaked swap groups at step {step}"
        assert nonzero_grads == total_grads, (
            f"{_scenario()} {phase} step {step} has missing gradients: "
            f"nonzero_grads={nonzero_grads}/{total_grads}"
        )
        if enable_swap:
            assert step_policy_calls > 0, f"{_scenario()} swap policy was not called at step {step}"
        grads_by_step.append(_grads_to_numpy(stages))
        if losses:
            losses_by_step.append(_losses_to_numpy(losses))

    losses_np = np.concatenate(losses_by_step) if losses_by_step else np.array([], dtype=np.float32)
    peak_memory_mb = steady_peak_memory_mb(step_peaks_mb)
    assert overlap_stats["callback_count"] > 0
    assert overlap_stats["bwd_thread_ids"]
    assert overlap_stats["main_thread_id"] not in overlap_stats["bwd_thread_ids"]
    print(
        f"[rank {rank}] {_scenario()} {phase}: steps={_CHECK_STEPS}, "
        f"step_peaks={format_step_peaks(step_peaks_mb)}, steady_peak={peak_memory_mb:.1f} MB",
        flush=True,
    )
    return rank, losses_np, grads_by_step, peak_memory_mb, step_peaks_mb


def test_moe_overlap_b_f_no_swap():
    """Save the three-step real-overlap no-swap baseline."""
    rank, losses, grads, peak_memory_mb, step_peaks_mb = _run_phase(enable_swap=False)
    assert_step_memory_stable(step_peaks_mb, f"{_scenario()} no-swap")
    _save_baseline(rank, losses, grads, peak_memory_mb, step_peaks_mb)


def test_moe_overlap_b_f_swap():
    """Compare three-step real-overlap swap with the no-swap baseline."""
    rank, losses, grads, peak_memory_mb, step_peaks_mb = _run_phase(enable_swap=True)
    baseline = _load_baseline(rank)
    np.testing.assert_allclose(losses, baseline["losses"], rtol=1e-3, atol=1e-3)
    assert len(grads) == len(baseline["grads"])
    for step, (step_grads, baseline_step_grads) in enumerate(zip(grads, baseline["grads"])):
        assert len(step_grads) == len(baseline_step_grads)
        for grad_index, (grad, baseline_grad) in enumerate(zip(step_grads, baseline_step_grads)):
            if grad is None or baseline_grad is None:
                assert grad is None and baseline_grad is None
                continue
            np.testing.assert_allclose(
                grad,
                baseline_grad,
                rtol=1e-3,
                atol=1e-3,
                err_msg=f"rank {rank}, step {step}, grad {grad_index}",
            )
    baseline_memory_mb = float(baseline["peak_memory_mb"])
    reduction = (baseline_memory_mb - peak_memory_mb) / baseline_memory_mb * 100
    assert_step_memory_stable(step_peaks_mb, f"{_scenario()} swap")
    assert peak_memory_mb < baseline_memory_mb, (
        f"rank {rank} {_scenario()} swap peak ({peak_memory_mb:.1f} MB) should be less than "
        f"no-swap peak ({baseline_memory_mb:.1f} MB)"
    )
    print(
        f"[rank {rank}] {_scenario()} swap comparison passed: "
        f"no_swap={baseline_memory_mb:.1f} MB, swap={peak_memory_mb:.1f} MB, reduction={reduction:.1f}%",
        flush=True,
    )
