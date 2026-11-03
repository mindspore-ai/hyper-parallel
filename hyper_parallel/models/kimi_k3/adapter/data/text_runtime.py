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
"""Validate K3 text ownership and document boundaries after contiguous CP splitting."""
# pylint: disable=forbidden-backend-import

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.models.kimi_k3.adapter.data.runtime import KimiK3PackedRuntime
from hyper_parallel.data.batching.runtime_input import RuntimeInputAdapter, RuntimeInputContext


class KimiLinearRuntimeInputAdapter(RuntimeInputAdapter):
    """Provide a local validity mask while retaining globally pre-shifted targets.

    By default each row is independent. Explicit ``packed=True`` consumes global
    cumulative offsets and supplies the same isolated documents to KDA and MLA.
    If a pad token is declared, the source encoder must mask its target labels
    with -100 before CP splits the globally aligned batch.
    """

    def __init__(self, pad_token_id: int | None = None, *, packed: bool = False) -> None:
        """Select optional explicit right-padding semantics for encoded tokens."""
        if pad_token_id is not None and (not isinstance(pad_token_id, int) or pad_token_id < 0):
            raise ValueError("pad_token_id must be a nonnegative integer or None")
        self.pad_token_id = pad_token_id
        self.packed_runtime = KimiK3PackedRuntime() if packed else None

    def build(self, *, batch: Mapping[str, Any], parallel_context: Any,
              options: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Recover global valid lengths for EOD-followed-by-padding boundaries.

        These integer data facts do not require autograd. Every CP rank receives
        identical lengths before validating right-padded sequence boundaries.

        Args:
            batch: Local text tensors, with global cumulative boundaries retained.
            parallel_context: Parallel coordinates and the chronological CP mesh
                used to sum valid prefix lengths across sequence shards.
            options: Runtime attention options; copied before adding global lengths.

        Returns:
            Additional model inputs from build_runtime_inputs, without mutating batch.
        """
        runtime_options = dict(options or {})
        if self.pad_token_id is not None and self.packed_runtime is None:
            lengths = batch["input_ids"].ne(self.pad_token_id).sum(-1, dtype=torch.int64)
            if parallel_context.cp_world_size > 1:
                cp_mesh = parallel_context.cp_mesh
                if cp_mesh is None or cp_mesh.size() != parallel_context.cp_world_size:
                    raise ValueError("K3 LLM padded CP data requires the chronological CP mesh")
                dist.all_reduce(lengths, op=dist.ReduceOp.SUM, group=cp_mesh.get_group())
            runtime_options["global_valid_lengths"] = lengths
        return super().build(batch=batch, parallel_context=parallel_context, options=runtime_options)

    def runtime_input_fields(self) -> tuple[str, ...]:
        """Declare the only field added to the standard text batch."""
        packed_fields = () if self.packed_runtime is None else self.packed_runtime.runtime_input_fields()
        return ("valid_token_mask",) + packed_fields

    def build_runtime_inputs(self, *, batch: Mapping[str, Any],
                             context: RuntimeInputContext) -> Mapping[str, Any]:
        """Validate row boundaries and produce a sequence-local validity mask.

        Args:
            batch: Local input_ids, labels and position_ids of identical [batch,
                local_sequence] shape, plus optional global cu_seq_lens metadata.
            context: CP rank/size and causal-attention options. TP and PP must be one.

        Returns:
            A valid_token_mask for local input tokens, plus packed-document metadata
            when packed mode is enabled. Labels remain globally pre-shifted.
        """
        if any(int(context.parallel_sizes.get(axis, 1)) != 1 for axis in ("tp", "pp")):
            raise ValueError("K3 LLM text runtime requires TP=PP=1")
        if not context.options.get("causal", True) or context.options.get("sliding_window") is not None:
            raise ValueError("K3 LLM requires causal full attention without sliding windows")
        input_ids = batch["input_ids"]
        if input_ids.ndim != 2 or input_ids.shape[1] <= 0:
            raise ValueError("K3 LLM input_ids must be nonempty [batch, local_sequence]")
        batch_size, local_length = input_ids.shape
        global_length = local_length * int(context.parallel_sizes.get("cp", 1))
        boundaries = batch.get("cu_seq_lens")
        if boundaries is not None and self.packed_runtime is None:
            self._validate_boundaries(boundaries, context, batch_size, global_length)
        for name in ("labels", "position_ids"):
            if name not in batch or batch[name].shape != input_ids.shape:
                raise ValueError(f"K3 LLM {name} must match local input_ids")
        positions = batch["position_ids"]
        start = int(context.parallel_ranks.get("cp", 0)) * local_length
        expected_positions = torch.arange(start, start + local_length, device=positions.device)
        if self.packed_runtime is None and not torch.equal(positions, expected_positions.expand(batch_size, -1)):
            raise ValueError("K3 LLM position_ids must retain contiguous global CP offsets")
        result = {"valid_token_mask": self._valid_token_mask(input_ids, batch["labels"])}
        if self.packed_runtime is not None:
            result.update(self.packed_runtime.build_runtime_inputs(batch=batch, context=context))
        return result

    def _valid_token_mask(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Exclude padding from QB while requiring its next-token targets to be ignored."""
        if self.pad_token_id is None:
            return torch.ones_like(input_ids, dtype=torch.bool)
        valid = input_ids.ne(self.pad_token_id)
        if torch.any(labels.eq(self.pad_token_id)):
            raise ValueError("Mask padded next-token labels with -100 before CP sharding")
        if torch.any(labels.ne(-100) & ~valid):
            raise ValueError("Padded input positions must have ignored next-token targets")
        return valid

    def _validate_boundaries(self, boundaries: torch.Tensor, context: RuntimeInputContext,
                             batch_size: int, global_length: int) -> None:
        """Accept row ends and one optional EOD at the exact valid-prefix end."""
        row_ends = [index * global_length for index in range(batch_size + 1)]
        actual = boundaries.tolist()
        if actual == row_ends:
            return
        lengths = context.options.get("global_valid_lengths")
        if self.pad_token_id is None or lengths is None:
            raise ValueError("K3 LLM packed documents are unsupported; each physical row must be one sequence")
        self._validate_boundary_coverage(boundaries, actual, row_ends)
        lengths = lengths.tolist()
        if len(lengths) != batch_size:
            raise ValueError("K3 LLM global valid lengths must match the batch size")
        for index, (start, end) in enumerate(zip(row_ends, row_ends[1:])):
            interior = [value for value in actual if start < value < end]
            if interior and interior != [start + lengths[index]]:
                raise ValueError("K3 LLM packed documents are unsupported; EOD must end the global valid prefix")


    @staticmethod
    def _validate_boundary_coverage(boundaries: torch.Tensor, actual: list[int], row_ends: list[int]) -> None:
        """Check the complete physical-row partition independently of padding."""
        if (boundaries.ndim != 1 or not actual or actual[0] != 0 or actual[-1] != row_ends[-1]
                or any(right <= left for left, right in zip(actual, actual[1:]))
                or not set(row_ends).issubset(actual)):
            raise ValueError("K3 LLM cumulative boundaries must cover every physical row exactly")


def build_kimi_linear_runtime_input_adapter(pad_token_id: int | None = None, *,
                                            packed: bool = False) -> KimiLinearRuntimeInputAdapter:
    """Build the native text adapter for standard TextParallelBatch.

    Args:
        pad_token_id: Token identifying right padding. None treats every input
            token as valid for QB; ignored labels alone do not define padding.
        packed: Consume global document offsets and emit packed-attention metadata.

    Returns:
        Adapter validating local token ownership and producing validity masks.
    """
    return KimiLinearRuntimeInputAdapter(pad_token_id=pad_token_id, packed=packed)
