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
"""Single-conversation batches for GDN P2P plus attention KV all-gather CP."""

from collections.abc import Mapping, Sequence
from typing import Any

from hyper_parallel.data.batching.build_collate_fn import TextPackingCollator
from hyper_parallel.data.batching.get_batch import TextParallelBatch


class GDNCPDataCollator:
    """Right-pad one conversation so every CP shard contains whole GDN chunks.

    Independent conversations cannot be packed: attention masking alone does
    not reset the GDN recurrent state or its causal convolution history.
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
        self.collator = TextPackingCollator(alignment_size=cp_size * chunk_size)

    def __call__(self, model_samples: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        """Collate exactly one nonempty conversation, including its ignored tail."""
        if len(model_samples) != 1:
            raise ValueError("GDN P2P requires micro_batch_size=1; packed independent samples are unsupported")
        sample = model_samples[0]
        if sample["input_ids"].ndim != 1 or sample["input_ids"].numel() == 0:
            raise ValueError("GDN P2P requires nonempty one-dimensional input_ids")
        if sample["labels"].shape != sample["input_ids"].shape:
            raise ValueError("GDN P2P requires aligned input_ids and pre-shifted labels")
        batch = self.collator(model_samples)
        # Tail padding belongs to the same causal stream, not a new sequence.
        # Keeping one boundary also avoids resetting RoPE positions in the tail.
        batch["cu_seq_lens"] = batch["cu_seq_lens"][[0, -1]]
        return batch


class GDNCPParallelBatch(TextParallelBatch):
    """Use contiguous token shards and delegate causal masking to the CP wrapper.

    This pairs with GDNCPDataCollator and sdpa_hf. A mask dictionary bypasses
    HF's local-length mask construction: sdpa_hf constructs the global-offset
    causal mask after gathering K/V. GDN needs no padding mask for a single
    right-padded causal stream. Neither path supports cached decoding here.
    """

    def _build_runtime_inputs(self, parallel_batch: Mapping[str, Any]) -> dict[str, Any]:
        """Reject packing and produce the two HF Qwen3.5 layer-mask entries."""
        if self.attention_runtime.mode != "dense" or not self.causal or self.sliding_window is not None:
            raise ValueError("GDN P2P batches require dense full causal attention without sliding_window")
        if self.parallel_context.tp_world_size != 1:
            raise NotImplementedError("GDN P2P batches currently require tp_size=1")
        if parallel_batch["input_ids"].shape[0] != 1 or parallel_batch["cu_seq_lens"].numel() != 2:
            raise ValueError("GDN P2P batches require one conversation; use GDNCPDataCollator")
        if parallel_batch["input_ids"].shape[1] % 64:
            raise ValueError("GDN Triton CP requires local sequence length divisible by 64")
        return {"attention_mask": {"full_attention": None, "linear_attention": None}}
