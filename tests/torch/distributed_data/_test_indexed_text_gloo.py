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
"""Real Indexed corpus to parallel training-batch coverage on CPU/Gloo."""

from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from hyper_parallel.data.batching.build_dataloader import build_dataloader
from hyper_parallel.data.batching.get_batch import ParallelBatch
from hyper_parallel.data.indexed.indexed_data_reader import IndexedDataReader
from hyper_parallel.data.indexed.io import IndexedDatasetBuilder
from hyper_parallel.data.text.build_dataset import build_indexed_text_dataset
from hyper_parallel.trainer.runtime.loss_aggregation import count_loss_token
from hyper_parallel.trainer.runtime.metrics import mean_global_loss

_LENGTHS = (1, 3, 2, 2, 3, 1, 2, 1)


class _Tokenizer:
    pad = None
    eod = 99
    special_tokens_dict = {"eod": 99}
    unique_identifiers = {"tokenizer": "indexed_gloo_test"}

    def __len__(self) -> int:
        """Return the test vocabulary size."""
        return 128


def _all_gather_object(value: Any) -> tuple[Any, ...]:
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    return tuple(gathered)


def _attention_loss(weight: torch.Tensor, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Compute summed causal language-model loss with the delivered document mask."""
    hidden = weight[inputs["input_ids"]] + inputs["position_ids"].unsqueeze(-1) / 10
    scores = hidden @ hidden.transpose(-1, -2) / weight.shape[-1] ** 0.5
    scores = scores.masked_fill(~inputs["attention_mask"].squeeze(1), float("-inf"))
    logits = (scores.softmax(dim=-1) @ hidden) @ weight.T
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), inputs["labels"].flatten(), reduction="sum", ignore_index=-100,
    )


def _assert_loss_gradient_parity(
        model_inputs: dict[str, torch.Tensor], loss_inputs: dict[str, torch.Tensor], mesh: Any,
) -> None:
    """Compare DP token-weighted packed training with separate original documents."""
    generator = torch.Generator().manual_seed(17)
    weight = torch.randn(128, 4, generator=generator, dtype=torch.float64, requires_grad=True)
    reference_weight = weight.detach().clone().requires_grad_()
    counts = count_loss_token(loss_inputs)
    local_loss = _attention_loss(weight, model_inputs) / counts["foundation_tokens"]
    context = SimpleNamespace(dp_cp_mesh=mesh["dp"], sequence_parallel=False, dp_size=2, cp_size=1)
    # This Gloo regression deliberately executes the Trainer's loss reduction on CPU.
    with patch("hyper_parallel.trainer.runtime.distributed.get_device_type", return_value="cpu"):
        loss = mean_global_loss(local_loss, counts, counts, context)["foundation_loss"]
    loss.backward()
    dist.all_reduce(weight.grad, group=mesh["dp"].get_group())
    weight.grad.div_(2)

    reference_loss = reference_weight.new_zeros(())
    for token, length in enumerate(_LENGTHS, 1):
        original_inputs = {
            "input_ids": torch.full((1, length), token, dtype=torch.int64),
            "labels": torch.tensor([[token] * (length - 1) + [99]]),
            "position_ids": torch.arange(length).unsqueeze(0),
            "attention_mask": torch.ones(1, 1, length, length, dtype=torch.bool).tril(),
        }
        reference_loss = reference_loss + _attention_loss(reference_weight, original_inputs)
    reference_loss = reference_loss / sum(_LENGTHS)
    reference_loss.backward()
    torch.testing.assert_close(loss, reference_loss, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(weight.grad, reference_weight.grad, rtol=1e-8, atol=1e-10)


def _run_epoch(prefix: str, mesh_context: object, double_buffer: bool) -> None:
    """Check metadata-only Readers, direct Constructor reads, and TP delivery."""
    rank = dist.get_rank()
    config = {
        "seq_length": 8, "split": "1, 0, 0", "mock_data": False,
        "is_dataset_from_mr": False, "simple_blend": "no",
        "data_lazy_load": True, "distributed_walk": False,
        "reset_position_ids": True,
        "packing_stage": "distributed_dataloader",
        "distributed_dataloader": {"dataset_reader_ranks": (1, 3), "double_buffer": double_buffer},
    }
    payload_reads = []
    allow_payload_reads = False
    read_payload = IndexedDataReader.__getitem__

    def tracked_read(reader: IndexedDataReader, index: int) -> object:
        """Record payload reads and reject any performed by metadata-only Readers.

        Args:
            reader: Binary corpus reader being tracked.
            index: Source record selected by the Constructor plan.
        """
        assert allow_payload_reads and rank in (0, 2), (
            f"Payload read before planning or outside a Constructor: allowed={allow_payload_reads}, rank={rank}"
        )
        payload_reads.append(index)
        return read_payload(reader, index)

    with patch.object(IndexedDataReader, "__getitem__", autospec=True, side_effect=tracked_read), \
            patch.object(dist, "all_to_all_single", side_effect=AssertionError("Indexed sidecar used payload A2A")):
        datasets = build_indexed_text_dataset(
            data_path=prefix, data_config=config, tokenizer=_Tokenizer(),
            train_valid_test_num_samples=(1, 0, 0), mesh_context=mesh_context,
        )
        loaders, _ = build_dataloader(
            SimpleNamespace(), datasets=datasets, collate_fn=None, mesh_context=mesh_context,
            training_config=SimpleNamespace(micro_batch_size=1, global_batch_size=2, seed=7),
            data_config=config,
        )
        runtime = ParallelBatch(
            mesh_context=mesh_context, device="cpu", tokenizer=_Tokenizer(),
            data_config=config, pp_shared_data=False, source_type="indexed_source",
        )
        allow_payload_reads = True
        model_inputs, loss_inputs = runtime(loaders[0])
        gathered = _all_gather_object((model_inputs["input_ids"].tolist(), payload_reads))
        assert gathered[0][0] == gathered[1][0] and gathered[2][0] == gathered[3][0], (
            f"TP peers received different token batches: gathered={gathered}"
        )
        read_indices = sorted(index for _, indices in gathered for index in indices)
        assert read_indices == list(range(len(_LENGTHS))), (
            f"Expected each source read once, got indices={read_indices}, expected={list(range(len(_LENGTHS)))}"
        )
        expected_tokens = sorted(token for token, length in enumerate(_LENGTHS, 1) for _ in range(length))
        actual_tokens = sorted(token for token in gathered[0][0][0] + gathered[2][0][0] if token != 0)
        assert actual_tokens == expected_tokens, (
            f"Token conservation: actual={actual_tokens}, expected={expected_tokens}"
        )
        valid_tokens = loss_inputs["loss_mask"].sum().item()
        assert valid_tokens in (7, 8), (
            f"Expected seven or eight valid tokens per DP rank, got count={valid_tokens}"
        )
        _assert_loss_gradient_parity(model_inputs, loss_inputs, mesh_context.device_mesh)
        exhausted = False
        try:
            runtime(loaders[0])
        except StopIteration:
            exhausted = True
        assert exhausted, f"Expected a collective end of the source epoch, got exhausted={exhausted}"
        loaders[0].wait_for_prefetch()


def test_indexed_text_dp2_tp2_gloo() -> None:
    """Run the full provider/loader/ParallelBatch path with real binary and index files."""
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        rank = dist.get_rank()
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
        mesh_context = SimpleNamespace(
            device_mesh=mesh, dp_rank=rank // 2, dp_size=2, tp_rank=rank % 2, tp_size=2, pp_size=1,
        )
        with TemporaryDirectory() as directory:
            prefixes = [None]
            if rank == 0:
                prefix = str(Path(directory) / "corpus")
                builder = IndexedDatasetBuilder(prefix + ".bin")
                for token, length in enumerate(_LENGTHS, 1):
                    builder.add_document(torch.tensor([token] * length + [99]), [length + 1])
                builder.finalize(prefix + ".idx")
                prefixes[0] = prefix
            dist.broadcast_object_list(prefixes, src=0)
            for double_buffer in (False, True):
                _run_epoch(prefixes[0], mesh_context, double_buffer)
            dist.barrier()
    finally:
        dist.destroy_process_group()
