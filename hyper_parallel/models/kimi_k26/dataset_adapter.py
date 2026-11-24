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
"""Keep full media-fusion inputs while distributing next-token targets over decoder CP."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch.nn import functional as F

from hyper_parallel.data.batching.get_batch import OmniParallelBatch


class KimiVLMParallelBatch(OmniParallelBatch):
    """Assign sequence cutting to the decoder after native visual scatter.

    Only targets are partitioned here. Media spans and the input token sequence
    remain complete until the architecture has fused its visual features.
    """

    def __init__(self, *, mesh_context: Any, device: Any, **kwargs: Any) -> None:
        """Require TP1 without sequence parallelism for the replicated visual path."""
        if mesh_context.tp_size != 1 or mesh_context.sequence_parallel:
            raise NotImplementedError("Kimi multimodal CP currently requires TP=1 and sequence_parallel=false")
        super().__init__(mesh_context=mesh_context, device=device, **kwargs)

    def __call__(self, data_iterator: Any, *, external_batch: Mapping[str, Any] | None = None
                 ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Transfer complete fusion inputs and globally shift labels before taking a CP window."""
        if self.batch_transport is not None:
            raise NotImplementedError("Kimi replicated vision CP does not use an independent encoder transport")
        source = None
        if self.parallel_context.build_on_rank():
            source = external_batch if external_batch is not None else next(data_iterator)
            source = self._normalize_source_batch(source)
        batch = self.tp_broadcaster.broadcast(source, broadcast_all_fields=True)
        cp_size, cp_rank = self.parallel_context.cp_world_size, self.parallel_context.cp_rank
        if cp_size == 1:
            model_inputs, loss_inputs = self._split_model_and_loss_inputs(batch)
            model_inputs.update(self._build_runtime_inputs(batch))
            return model_inputs, loss_inputs
        if self.runtime_input_adapter is not None:
            raise NotImplementedError("Kimi deferred CP inputs cannot be combined with another runtime input adapter")
        length = batch["input_ids"].shape[1]
        if length % cp_size:
            raise ValueError("Kimi full multimodal sequence length must be divisible by cp_size")
        boundaries = batch.get("cu_seq_lens")
        if boundaries is not None and boundaries.numel() > 2:
            raise NotImplementedError("Kimi multimodal CP does not yet support packed conversations")
        labels = batch["labels"]
        targets = F.pad(labels, (0, 1), value=-100)[:, 1:]
        mask = F.pad(batch["loss_mask"].bool(), (0, 1), value=False)[:, 1:]
        start, stop = cp_rank * (length // cp_size), (cp_rank + 1) * (length // cp_size)
        targets = targets[:, start:stop].contiguous()
        mask = mask[:, start:stop].contiguous() & targets.ne(-100)
        model_inputs, loss_inputs = self._split_model_and_loss_inputs(batch)
        model_inputs.pop("labels", None)
        model_inputs["kimi_cp_coordinates"] = (cp_size, cp_rank)
        model_inputs["chunk_loss_targets"] = targets
        model_inputs["chunk_loss_mask"] = mask
        # Explicit shifted targets give the trainer the exact local token count,
        # including a supervised next token that crosses a CP boundary.
        loss_inputs["shift_labels"] = targets
        loss_inputs["loss_mask"] = mask
        return model_inputs, loss_inputs
