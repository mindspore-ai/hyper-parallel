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
"""K3 raw or globally aligned targets for the standard chunked objective."""

from __future__ import annotations

from typing import Any, Mapping

# Native Torch objective, consistent with the model-owned loss adapters.
import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import
from torch.nn import functional as F  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.losses.chunked_cross_entropy import ChunkedCausalLMLoss


class KimiK3ChunkedCausalLMLoss(ChunkedCausalLMLoss):
    """Shift raw labels once, or consume already aligned media targets.

    When the collator shifts labels, it sets ``labels_are_shifted``.
    Any accompanying ``loss_mask`` must use those same target positions.
    Explicit ``shift_labels`` already follow the common aligned-mask contract.
    """

    def prepare_model_inputs(self, model_inputs: Mapping[str, Any],
                             loss_inputs: Mapping[str, Any]) -> dict[str, Any]:
        """Translate raw image/text labels into aligned next-token targets once.

        Args:
            model_inputs: Expanded model inputs and optional attention/media masks.
            loss_inputs: Raw labels [batch, sequence] with an optional raw loss_mask,
                or an already aligned shift_labels tensor and loss_mask.

        Returns:
            Model inputs populated with the common chunked-objective arguments.
            Existing shift_labels bypass the raw-label shift.

        Raises:
            ValueError: Raw labels or supplied masks have incompatible shapes.
        """
        prepared = dict(loss_inputs)
        if "shift_labels" not in prepared and bool(model_inputs.get("labels_are_shifted", False)):
            prepared["shift_labels"] = prepared.get("labels")
        if "shift_labels" not in prepared:
            labels = prepared.get("labels")
            if not isinstance(labels, torch.Tensor) or labels.ndim != 2:
                raise ValueError("K3 chunk loss requires raw labels [batch,sequence] or shift_labels")
            for name, ignore_true in (("attention_mask", False), ("media_token_mask", True)):
                input_mask = model_inputs.get(name)
                if input_mask is not None:
                    if input_mask.shape != labels.shape:
                        raise ValueError(f"{name} must match raw labels")
                    ignored = input_mask.bool() if ignore_true else ~input_mask.bool()
                    labels = labels.masked_fill(ignored, self.ignore_index)
            prepared["shift_labels"] = F.pad(labels[:, 1:], (0, 1), value=self.ignore_index)
            mask = prepared.get("loss_mask")
            if mask is not None:
                if mask.shape != labels.shape:
                    raise ValueError("loss_mask must match raw labels")
                prepared["loss_mask"] = F.pad(mask[:, 1:].bool(), (0, 1), value=False)
        return super().prepare_model_inputs(model_inputs, prepared)


def bind_chunk_loss(model: nn.Module) -> None:
    """Validate the native chunked-objective entry without changing forward.

    Args:
        model: Registered K3 image/text exposing its native objective contract.

    Raises:
        ValueError: The model family or required native entry is missing.
    """
    if getattr(model.config, "model_type", None) != "kimi_k3" or not hasattr(model, "prepare_embeddings"):
        raise ValueError("K3 chunk loss requires the registered K3 training architecture")
