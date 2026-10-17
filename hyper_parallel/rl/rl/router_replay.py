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
"""Model-independent, micro-batch router replay state."""
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

import torch
from torch.utils.hooks import RemovableHandle

from rl.dataset.contracts import RouterTrace


@dataclass(frozen=True)
class MicroBatchRouterTargets:
    """Frozen token-major selections shared by forward and checkpoint recomputation."""

    expert_ids: torch.Tensor
    valid_mask: torch.Tensor
    layer_ids: tuple[int, ...]


class RouterReplay:
    """Bind validated rollout selections without depending on a model or algorithm."""

    def __init__(self, layer_ids: tuple[int, ...], num_experts: int, top_k: int) -> None:
        """Define the router contract required from every trace."""
        if not layer_ids or len(set(layer_ids)) != len(layer_ids):
            raise ValueError("Router replay requires unique nonempty layer IDs")
        if num_experts <= 0 or not 0 < top_k <= num_experts:
            raise ValueError("Router replay requires 0 < top_k <= num_experts")
        self.layer_ids = layer_ids
        self.num_experts = num_experts
        self.top_k = top_k
        self.current: Optional[MicroBatchRouterTargets] = None
        self._layer_indices = {layer_id: index for index, layer_id in enumerate(layer_ids)}
        self._handles: list[RemovableHandle] = []
        self._removed = False

    def validate(
        self,
        traces: Optional[tuple[RouterTrace, ...]],
        attention_mask: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> None:
        """Validate all CPU targets and causal coverage before any model execution."""
        if attention_mask.ndim != 2 or action_mask.shape != attention_mask.shape:
            raise ValueError("Router replay masks must have matching [batch, sequence] shapes")
        if traces is None or len(traces) != attention_mask.shape[0]:
            raise ValueError("Router replay requires one trace per batch row")
        attention = attention_mask.detach().to(device="cpu", dtype=torch.bool)
        actions = action_mask.detach().to(device="cpu", dtype=torch.bool)
        for row, trace in enumerate(traces):
            self._validate_trace_layout(trace)
            ids, valid = trace.expert_ids, trace.valid_mask
            length = ids.shape[0]
            if length > attention.shape[1] or bool(valid[~attention[row, :length]].any()):
                raise ValueError("Router trace must align with unpadded Actor input positions")
            self._validate_expert_ids(ids, valid)
            action_positions = actions[row].nonzero().flatten()
            required = int(action_positions[-1]) if action_positions.numel() else 0
            if required > length or not bool(valid[:required].all()):
                raise ValueError("Router trace is missing a causal prefix required by an action")

    def _validate_trace_layout(self, trace: RouterTrace) -> None:
        """Match CPU target tensors to the Actor's layer and top-k contract."""
        if not isinstance(trace, RouterTrace):
            raise ValueError("Router replay requires a RouterTrace for every row")
        ids, valid = trace.expert_ids, trace.valid_mask
        if (ids.device.type != "cpu" or valid.device.type != "cpu" or ids.ndim != 3
                or ids.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
                or valid.dtype != torch.bool or valid.shape != ids.shape[:1]):
            raise ValueError("Router replay requires CPU integer [S,L,K] IDs and bool [S] validity")
        if trace.layer_ids != self.layer_ids or ids.shape[1:] != (len(self.layer_ids), self.top_k):
            raise ValueError("Router trace layers/top-k do not match the Actor router contract")

    def _validate_expert_ids(self, ids: torch.Tensor, valid: torch.Tensor) -> None:
        """Check expert ranges and top-k uniqueness with bounded temporary storage."""
        # Torch sort allocates int64 indices even when only native-dtype values are used.
        for start in range(0, ids.shape[0], 4096):
            selected = ids[start:start + 4096][valid[start:start + 4096]]
            if selected.numel():
                minimum, maximum = torch.aminmax(selected)
                if int(minimum) < 0 or int(maximum) >= self.num_experts:
                    raise ValueError("Router trace contains an out-of-range expert ID")
            ordered = selected.sort(dim=-1).values
            if bool((ordered[..., 1:] == ordered[..., :-1]).any()):
                raise ValueError("Router trace repeats an expert within a top-k selection")

    def prepare(
        self,
        traces: tuple[RouterTrace, ...],
        attention_mask: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> MicroBatchRouterTargets:
        """Pad a validated micro-batch and transfer its compact targets once to the input device."""
        self.validate(traces, attention_mask, action_mask)
        batch_size, sequence_length = attention_mask.shape
        dtype = traces[0].expert_ids.dtype
        if any(trace.expert_ids.dtype != dtype for trace in traces):
            dtype = torch.int64
        ids = torch.zeros((batch_size, sequence_length, len(self.layer_ids), self.top_k), dtype=dtype)
        mask = torch.zeros((batch_size, sequence_length), dtype=torch.bool)
        for row, trace in enumerate(traces):
            length = trace.expert_ids.shape[0]
            ids[row, :length] = trace.expert_ids.masked_fill(~trace.valid_mask[:, None, None], 0)
            mask[row, :length] = trace.valid_mask
        return MicroBatchRouterTargets(
            expert_ids=ids.flatten(0, 1).to(device=attention_mask.device),
            valid_mask=mask.flatten().to(device=attention_mask.device),
            layer_ids=self.layer_ids,
        )

    @contextmanager
    def use(self, targets: MicroBatchRouterTargets) -> Iterator[None]:
        """Keep selections bound through backward and clear them on every exit path."""
        if self._removed or self.current is not None:
            raise RuntimeError("Router replay is removed or already has an active micro-batch")
        if targets.layer_ids != self.layer_ids:
            raise ValueError("Router replay targets belong to a different layer contract")
        self.current = targets
        try:
            yield
        finally:
            self.current = None

    def layer_targets(self, layer_id: int) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Read immutable selections by stable layer ID, including during recomputation."""
        if self.current is None:
            return None
        return self.current.expert_ids[:, self._layer_indices[layer_id]], self.current.valid_mask

    def add_hook(self, handle: RemovableHandle) -> None:
        """Own an installed instance hook so it can be removed explicitly."""
        self._handles.append(handle)

    def remove(self) -> None:
        """Remove all owned hooks and release bound device targets."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.current = None
        self._removed = True
