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
"""MindSpore Interleaved 1F1B activation swap memory comparison."""
# pylint: disable=wrong-import-position
import os
import tempfile

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"
os.environ["MS_DEV_RUNTIME_CONF"] = "memory_statistics:True"

import numpy as np
import mindspore as ms
import mindspore.communication.management as D
from mindspore import Parameter, Tensor, mint, nn
from mindspore.common.initializer import initializer

from hyper_parallel import PipelineStage
from hyper_parallel.core.activation_checkpoint import swap_wrapper
from hyper_parallel.core.pipeline_parallel import ScheduleInterleaved1F1B
from hyper_parallel.platform.mindspore.autograd_compat import enable_mindspore_backward_compat

_RANK_MEM_FILE_PREFIX = os.path.join(tempfile.gettempdir(), "pp_swap_interleaved_1f1b_mem_no_swap")


def _rank_mem_file(rank: int) -> str:
    tag = os.environ.get("PP_SWAP_INTERLEAVED_MEM_TAG")
    tag_suffix = f"_{tag}" if tag else ""
    return f"{_RANK_MEM_FILE_PREFIX}{tag_suffix}_rank{rank}.txt"


class DeepStage(nn.Cell):
    """A virtual stage with enough layers to make activation memory visible."""

    def __init__(self, hidden_size, num_layers=4):
        super().__init__()
        self.layers = nn.CellList([
            self._build_layer(hidden_size, i) for i in range(num_layers)
        ])

    def _build_layer(self, hidden_size, idx):
        cell = nn.Cell()
        cell.weight = Parameter(
            initializer("ones", [hidden_size, hidden_size], ms.float32),
            name=f"weight_{idx}",
        )
        cell.construct = lambda x, w=cell.weight: mint.nn.functional.relu(mint.matmul(x, w))
        return cell

    def construct(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _cat_losses(losses):
    return mint.cat(tuple(losses), dim=0)


def _get_tokens_per_micro(hidden_size) -> int:
    activation_mb = os.environ.get("PP_SWAP_INTERLEAVED_ACTIVATION_MB", os.environ.get("PP_SWAP_ACTIVATION_MB"))
    if activation_mb is not None:
        target_bytes = int(float(activation_mb) * 1024 * 1024)
        bytes_per_token = hidden_size * np.dtype(np.float32).itemsize
        return max(1, (target_bytes + bytes_per_token - 1) // bytes_per_token)
    return int(os.environ.get("PP_SWAP_TOKENS_PER_MICRO", "1"))


def _make_input(micro_batch_num, hidden_size):
    tokens_per_micro = _get_tokens_per_micro(hidden_size)
    batch = micro_batch_num * tokens_per_micro
    return Tensor(
        np.arange(batch * hidden_size, dtype=np.float32).reshape(batch, hidden_size) / 10000
    )


def _run_standalone(x, hidden_size, num_layers, stage_num):
    """Run a serial reference and return output plus grads."""
    models = [DeepStage(hidden_size, num_layers) for _ in range(stage_num)]
    for model in models:
        for param in model.trainable_params():
            param.grad = None
    out = x
    for model in models:
        out = model(out)
    out.backward(mint.ones(out.shape, dtype=out.dtype))
    ref_grads = []
    for model in models:
        ref_grads.extend(param.grad for param in model.trainable_params())
    return out, ref_grads


def _reset_peak_memory_stats():
    ms.runtime.empty_cache()
    ms.runtime.reset_peak_memory_stats()


def _device_peak_memory_mb() -> float:
    return ms.runtime.max_memory_allocated() / (1024 ** 2)


def _overlap_b_f_enabled() -> bool:
    return os.environ.get("PP_SWAP_INTERLEAVED_OVERLAP_B_F") == "1"


def _local_virtual_stage_indices(rank, world_size, virtual_stages_per_rank):
    return [rank + i * world_size for i in range(virtual_stages_per_rank)]


def _build_stages(rank, world_size, hidden_size, num_layers, enable_swap):
    """Build the local virtual stages for one rank."""
    virtual_stages_per_rank = 2
    stage_num = world_size * virtual_stages_per_rank
    stage_models = []
    stages = []
    for stage_index in _local_virtual_stage_indices(rank, world_size, virtual_stages_per_rank):
        model = DeepStage(hidden_size, num_layers)
        if enable_swap:
            model = swap_wrapper(model)
        stage_models.append((stage_index, model))
        stages.append(PipelineStage(model, stage_index=stage_index, stage_num=stage_num))
    return stage_models, stages, stage_num


def _run_pipeline(enable_swap, record_memory=False, compare_memory=False, check_accuracy=False):
    """Run one Interleaved 1F1B scenario and optionally validate memory or accuracy."""
    ms.set_context(mode=ms.PYNATIVE_MODE)
    enable_mindspore_backward_compat()
    D.init()

    rank = D.get_rank()
    world_size = D.get_group_size()
    assert world_size == 4

    micro_batch_num = 8
    hidden_size = 2048
    num_layers = 4
    x = _make_input(micro_batch_num, hidden_size)
    overlap_b_f = _overlap_b_f_enabled()
    scenario = "interleaved overlap_b_f" if overlap_b_f else "interleaved"

    stage_models, stages, stage_num = _build_stages(rank, world_size, hidden_size, num_layers, enable_swap)
    schedule = ScheduleInterleaved1F1B(
        stages,
        micro_batch_num=micro_batch_num,
        overlap_b_f=overlap_b_f,
        swap=enable_swap,
    )

    _reset_peak_memory_stats()
    if rank == 0:
        pp_losses = schedule.run(x)
    else:
        pp_losses = schedule.run()
    mem_mb = _device_peak_memory_mb()

    if enable_swap:
        print(f"[rank {rank}] {scenario} swap device peak memory: {mem_mb:.1f} MB", flush=True)
    else:
        if record_memory:
            with open(_rank_mem_file(rank), "w", encoding="utf-8") as f:
                f.write(f"{mem_mb:.2f}")
        print(f"[rank {rank}] {scenario} no-swap device peak memory: {mem_mb:.1f} MB", flush=True)
        return

    if check_accuracy:
        ref_out, ref_grads = _run_standalone(x, hidden_size, num_layers, stage_num)
        if rank == world_size - 1:
            np.testing.assert_allclose(
                _cat_losses(pp_losses).asnumpy(), ref_out.asnumpy(), rtol=1e-4, atol=1e-4,
            )
        for stage_index, model in stage_models:
            ref_offset = stage_index * num_layers
            for i, param in enumerate(model.trainable_params()):
                np.testing.assert_allclose(
                    param.grad.asnumpy(), ref_grads[ref_offset + i].asnumpy(), rtol=1e-4, atol=1e-4,
                )
        print(f"[rank {rank}] {scenario} swap accuracy passed", flush=True)

    if compare_memory:
        rank_mem_file = _rank_mem_file(rank)
        assert os.path.exists(rank_mem_file), f"Missing no-swap memory baseline for rank {rank}"
        with open(rank_mem_file, encoding="utf-8") as f:
            mem_no_swap = float(f.read().strip())
            reduction = (mem_no_swap - mem_mb) / mem_no_swap * 100
            print(
                f"[rank {rank}] {scenario} no-swap device peak: {mem_no_swap:.1f} MB, "
                f"swap device peak: {mem_mb:.1f} MB, reduction: {reduction:.1f}%",
                flush=True,
            )
            assert mem_mb < mem_no_swap, (
                f"rank {rank} swap device memory ({mem_mb:.1f} MB) should be less than "
                f"no-swap device memory ({mem_no_swap:.1f} MB)"
            )


def test_interleaved_1f1b_no_swap():
    """Measure Interleaved 1F1B device peak memory without activation swap."""
    _run_pipeline(enable_swap=False, record_memory=True)


def test_interleaved_1f1b_swap_memory():
    """Run Interleaved 1F1B with activation swap and compare per-rank peak memory."""
    _run_pipeline(enable_swap=True, compare_memory=True)


def test_interleaved_1f1b_swap_accuracy():
    """Run Interleaved 1F1B with activation swap and compare outputs/grads."""
    _run_pipeline(enable_swap=True, check_accuracy=True)
