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
"""MindSpore 1F1B activation swap end-to-end test with device memory measurement."""
# pylint: disable=wrong-import-position
import os
import sys
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
from hyper_parallel.core.pipeline_parallel import Schedule1F1B
from hyper_parallel.platform.mindspore.autograd_compat import enable_mindspore_backward_compat

_MEM_FILE = os.path.join(tempfile.gettempdir(), "pp_swap_1f1b_mem_no_swap.txt")
_RANK_MEM_FILE_PREFIX = os.path.join(tempfile.gettempdir(), "pp_swap_1f1b_mem_no_swap")
_NO_SWAP_WINDOW_MEMORY_TOLERANCE_MB = 1.0


def _rank_mem_file(rank: int) -> str:
    return f"{_RANK_MEM_FILE_PREFIX}_rank{rank}.txt"


class DeepStage(nn.Cell):
    """A stage with multiple LinearRelu blocks so activations consume non-trivial device memory."""

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


def _run_standalone(x, hidden_size, num_layers):
    """Run a serial reference and return output plus grads."""
    stage_num = 4
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


def _cat_losses(losses):
    return mint.cat(tuple(losses), dim=0)


def _get_tokens_per_micro(hidden_size) -> int:
    activation_mb = os.environ.get("PP_SWAP_ACTIVATION_MB")
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


def _reset_peak_memory_stats():
    """Reset peak memory stats after setup allocations."""
    ms.runtime.empty_cache()
    ms.runtime.reset_peak_memory_stats()


def _device_peak_memory_mb() -> float:
    """Return device peak memory in MB."""
    return ms.runtime.max_memory_allocated() / (1024 ** 2)


def _has_swap_control_steps(schedule, rank: int) -> bool:
    """Return whether the local schedule has any PP-swap control step."""
    return any(step.type.name.startswith("SWAP_") for step in schedule.exec_order[rank])


def test_1f1b_no_swap():
    """Measure device peak memory without activation swap (baseline)."""
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

    model = DeepStage(hidden_size, num_layers)
    stage = PipelineStage(model, stage_index=rank, stage_num=world_size)
    schedule = Schedule1F1B(stage, micro_batch_num=micro_batch_num, swap=False)

    _reset_peak_memory_stats()
    if rank == 0:
        schedule.run(x)
    else:
        schedule.run()
    mem_mb = _device_peak_memory_mb()

    with open(_rank_mem_file(rank), "w", encoding="utf-8") as f:
        f.write(f"{mem_mb:.2f}")
    if rank == 0:
        with open(_MEM_FILE, "w", encoding="utf-8") as f:
            f.write(f"{mem_mb:.2f}")
    print(f"[rank {rank}] no-swap device peak memory: {mem_mb:.1f} MB", flush=True)


def _run_1f1b_swap(compare_memory: bool, check_accuracy: bool):
    """Run 1F1B with activation swap and optionally check memory/accuracy."""
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

    stage_model = swap_wrapper(DeepStage(hidden_size, num_layers))
    stage = PipelineStage(stage_model, stage_index=rank, stage_num=world_size)
    schedule = Schedule1F1B(stage, micro_batch_num=micro_batch_num, swap=True)
    has_swap_window = _has_swap_control_steps(schedule, rank)

    _reset_peak_memory_stats()
    if rank == 0:
        pp_losses = schedule.run(x)
    else:
        pp_losses = schedule.run()
    mem_swap = _device_peak_memory_mb()
    print(f"[rank {rank}] swap device peak memory: {mem_swap:.1f} MB", flush=True)

    if check_accuracy:
        # Keep the serial reference out of the measured region. Its returned
        # gradients are large live tensors and would pollute the swap peak.
        ref_out, ref_grads = _run_standalone(x, hidden_size, num_layers)

        if rank == world_size - 1:
            np.testing.assert_allclose(
                _cat_losses(pp_losses).asnumpy(), ref_out.asnumpy(), rtol=1e-4, atol=1e-4,
            )

        local_params = list(stage_model.trainable_params())
        ref_offset = rank * num_layers
        for i, param in enumerate(local_params):
            np.testing.assert_allclose(
                param.grad.asnumpy(), ref_grads[ref_offset + i].asnumpy(), rtol=1e-4, atol=1e-4,
            )

    if not compare_memory:
        return

    rank_mem_file = _rank_mem_file(rank)
    assert os.path.exists(rank_mem_file), f"Missing no-swap memory baseline for rank {rank}"
    with open(rank_mem_file, encoding="utf-8") as f:
        mem_no_swap = float(f.read().strip())
    reduction = (mem_no_swap - mem_swap) / mem_no_swap * 100
    print(
        f"[rank {rank}] no-swap device peak: {mem_no_swap:.1f} MB, "
        f"swap device peak: {mem_swap:.1f} MB, reduction: {reduction:.1f}%, "
        f"swap_window={has_swap_window}",
        flush=True,
    )
    if not has_swap_window:
        assert mem_swap <= mem_no_swap + _NO_SWAP_WINDOW_MEMORY_TOLERANCE_MB, (
            f"rank {rank} without a 1F1B swap window should not increase device memory: "
            f"swap ({mem_swap:.1f} MB), no-swap ({mem_no_swap:.1f} MB)"
        )
        return
    assert mem_swap < mem_no_swap, (
        f"rank {rank} swap device memory ({mem_swap:.1f} MB) should be less than "
        f"no-swap device memory ({mem_no_swap:.1f} MB)"
    )


def test_1f1b_swap():
    """Run 1F1B with activation swap on every stage and compare device memory."""
    _run_1f1b_swap(compare_memory=True, check_accuracy=os.environ.get("PP_SWAP_SKIP_REFERENCE") != "1")


def test_1f1b_swap_memory():
    """Run 1F1B activation swap memory comparison without serial reference."""
    _run_1f1b_swap(compare_memory=True, check_accuracy=False)


def test_1f1b_swap_accuracy():
    """Run 1F1B activation swap output/gradient correctness."""
    _run_1f1b_swap(compare_memory=False, check_accuracy=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "no_swap":
        test_1f1b_no_swap()
    else:
        test_1f1b_swap()
