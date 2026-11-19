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
"""RL next-token coordinates on top of HP's contiguous CP batch sharder."""

from typing import Any

import torch
import torch.distributed as dist
from torch.nn import functional

from hyper_parallel.data.parallel.batch_parallel import CPBatchSharder
from hyper_parallel.data.parallel.dataloader_parallel import DataLoaderParallelContext


def compute_cp_token_log_probs(model: Any, sequences: torch.Tensor, attention_mask: torch.Tensor,
                               cp_mesh: Any) -> torch.Tensor:
    """Shift targets globally before slicing, including each CP boundary target.

    Args:
        model: Actor/reference model with the training CP attention adapter.
        sequences: [B, S] global token IDs, already aligned for TP/SP/CP.
        attention_mask: [B, S] global valid-token mask.
        cp_mesh: HP mesh describing contiguous sequence ownership.

    Returns:
        [B, S/CP] FP32 next-token log-probs on this rank, with autograd intact.
        The final synthetic target and alignment rows must have zero loss mask;
        this function does not discard them or gather full vocabulary logits.
    """
    context = DataLoaderParallelContext(cp_rank=cp_mesh.get_local_rank(), cp_world_size=cp_mesh.size())
    # Shift on the global sequence before sharding: the last query on one rank
    # still predicts the first token on the next rank, rather than losing it.
    batch = CPBatchSharder(context).shard({
        "input_ids": sequences, "labels": functional.pad(sequences[:, 1:], (0, 1)),
    })
    length = batch["input_ids"].shape[1]
    start = context.cp_rank * length
    positions = torch.arange(start, start + length, device=sequences.device).unsqueeze(0)
    # Inputs and positions are local, but the attention adapter needs the global
    # padding mask to pack each query's full document prefix after K/V gather.
    outputs = model(input_ids=batch["input_ids"], attention_mask=attention_mask,
                    position_ids=positions, use_cache=False)
    logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
    if logits.shape[:2] != batch["input_ids"].shape:
        raise ValueError("CP logits must preserve the local sequence coordinates")
    return logits.float().log_softmax(-1).gather(-1, batch["labels"].unsqueeze(-1)).squeeze(-1)


def slice_cp_targets(value: torch.Tensor, cp_mesh: Any, sequence_multiple: int) -> torch.Tensor:
    """Align a [B,S-1] target field with the padded CP input query interval."""
    length = value.shape[1] + 1
    padded_length = length + (-length) % sequence_multiple
    local_length = padded_length // cp_mesh.size()
    # The final next-token target and TP/SP/CP alignment rows carry zero loss.
    padded = functional.pad(value, (0, padded_length - value.shape[1]))
    return padded.narrow(1, cp_mesh.get_local_rank() * local_length, local_length).contiguous()


def gather_cp_log_probs(local: torch.Tensor, cp_mesh: Any) -> torch.Tensor:
    """Restore detached scalar probabilities for old/reference targets, never logits."""
    if local.requires_grad:
        raise ValueError("CP probability restoration is for detached targets; training uses local loss")
    parts = [torch.empty_like(local) for _ in range(cp_mesh.size())]
    dist.all_gather(parts, local.contiguous(), group=cp_mesh.get_group())
    return torch.cat(parts, dim=1)
