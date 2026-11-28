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
"""Unpacked batches for GDN P2P plus attention KV all-gather CP."""

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.data.batching.get_batch import TextParallelBatch


class GDNCPDataCollator:
    """Right-pad independent batch rows to whole GDN chunks on every CP rank.

    Independent conversations stay in separate rows: attention masking alone
    does not reset the GDN recurrent state or its causal convolution history.
    Padding labels are ignored, and future padding cannot affect real tokens.
    """

    def __init__(self, mesh_context: Any | None = None, chunk_size: int = 64) -> None:
        """Configure global alignment from the CP degree and kernel chunk size."""
        cp_size = int(getattr(mesh_context, "cp_size", 1))
        tp_size = int(getattr(mesh_context, "tp_size", 1))
        if cp_size < 1 or chunk_size < 1:
            raise ValueError("cp_size and chunk_size must be positive")
        if tp_size != 1:
            raise NotImplementedError("The GDN P2P batch path currently requires tp_size=1")
        self.alignment_size = cp_size * chunk_size

    def __call__(self, model_samples: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        """Keep conversations in separate rows, with ignored right-padding labels."""
        if not model_samples:
            raise ValueError("GDN P2P requires at least one conversation")
        for sample in model_samples:
            if sample["input_ids"].ndim != 1 or sample["input_ids"].numel() == 0:
                raise ValueError("GDN P2P requires nonempty one-dimensional input_ids")
            if sample["labels"].shape != sample["input_ids"].shape:
                raise ValueError("GDN P2P requires aligned input_ids and pre-shifted labels")
        length = max(sample["input_ids"].numel() for sample in model_samples)
        length += (-length) % self.alignment_size
        batch = {
            name: torch.stack([
                F.pad(sample[name], (0, length - sample[name].numel()), value=pad_value)
                for sample in model_samples
            ])
            for name, pad_value in (("input_ids", 0), ("labels", IGNORE_INDEX))
        }
        # These boundaries describe physical batch rows for TextParallelBatch;
        # they are not forwarded to the model as packed-sequence metadata.
        batch["cu_seq_lens"] = torch.arange(len(model_samples) + 1, dtype=torch.int32) * length
        return batch


class GDNCPParallelBatch(TextParallelBatch):
    """Use contiguous token shards and delegate causal masking to the CP wrapper.

    This pairs with GDNCPDataCollator. A mask dictionary bypasses HF's
    local-length mask construction; the CP attention implementation owns
    global-offset causal attention. Each row has an independent GDN state.
    Right padding cannot affect real tokens in these causal streams.
    """

    def _build_runtime_inputs(self, parallel_batch: Mapping[str, Any]) -> dict[str, Any]:
        """Reject packing and produce the two HF Qwen3.5 layer-mask entries."""
        if self.attention_runtime.mode != "dense" or not self.causal or self.sliding_window is not None:
            raise ValueError("GDN P2P batches require dense full causal attention without sliding_window")
        if self.parallel_context.tp_world_size != 1:
            raise NotImplementedError("GDN P2P batches currently require tp_size=1")
        batch_size, local_length = parallel_batch["input_ids"].shape
        global_length = local_length * self.parallel_context.cp_world_size
        boundaries = parallel_batch["cu_seq_lens"].tolist()
        if boundaries != list(range(0, (batch_size + 1) * global_length, global_length)):
            raise ValueError("GDN P2P batches require one conversation per row; use GDNCPDataCollator")
        if local_length % 64:
            raise ValueError("GDN Triton CP requires local sequence length divisible by 64")
        return {"attention_mask": {"full_attention": None, "linear_attention": None}}
