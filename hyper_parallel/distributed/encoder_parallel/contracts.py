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
"""Model-neutral contracts for visual encoder parallel execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import


@dataclass(frozen=True)
class EncoderWorkItem:
    """One image and the decoder token span that consumes its features."""

    item_id: int
    batch_index: int
    sequence_length: int
    output_rows: int
    decoder_token_start: int
    decoder_token_length: int
    payload_start: int
    payload_stop: int
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class EncoderSegment:
    """One padded image shard in a packed ViT call."""

    item: EncoderWorkItem | None
    local_start: int
    local_valid_length: int
    local_padded_length: int
    packed_offset: int

    @property
    def is_dummy(self) -> bool:
        """A missing image identifies the synchronization-only segment."""
        return self.item is None


EncoderFeature = tuple[EncoderWorkItem, torch.Tensor]
"""One image job paired with its aligned feature rows."""


@dataclass(frozen=True)
class EncoderLocalInputs:
    """Device patches packed in the order of one plan's local segments."""

    patches: torch.Tensor


@dataclass(frozen=True)
class EncoderParallelAdapterSpec:
    """Model-owned visual boundaries and a runtime bridge factory.

    ``bind(model)`` returns an object with ``input_fields``, ``head_counts``,
    ``validate_training``, ``parse_images``, ``pack_payload``, ``payload_field``,
    ``empty_payload_shape``, ``encode_local`` and ``project``.
    The runtime uses these operations without inspecting model-specific state.
    An optional ``project_many(hidden_states, items)`` batches all real/dummy
    slots into one projector invocation, preserving its FSDP parameter lifetime.
    It returns one feature tensor per slot; bridges without it use ``project``.
    An optional ``source_attention_type`` limits a block-wide pattern to its
    actual attention owners. Module replacement is selected by recipe ``replace.kind=module``.
    """

    isolated_subtrees: tuple[str, ...]
    attention_pattern: str
    replacement_attention_type: type[nn.Module]
    bind: Callable[[nn.Module], Any]
    source_attention_type: type[nn.Module] | None = None
