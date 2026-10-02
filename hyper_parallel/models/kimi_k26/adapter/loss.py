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
"""Family loss adapters for the Kimi-K2.6 multimodal path."""

from __future__ import annotations

from typing import Any, Mapping

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.losses.chunked_cross_entropy import (
    ChunkedCausalLMLoss,
)


class KimiChunkedCausalLMLoss(ChunkedCausalLMLoss):
    """Chunk loss that accepts the multimodal batch's raw ``labels``.

    The framework's :class:`ChunkedCausalLMLoss` consumes pre-shifted
    ``shift_labels``: batch producers that pre-shift (the text path in
    ``data/batching``) publish them directly. The Omni multimodal batch keeps raw
    ``labels`` instead — the family transform pads and truncates one label per
    position — so this subclass derives the next-token targets here and delegates
    the rest of the protocol (``chunk_loss_*`` injection, IO contracts) to the
    framework implementation.

    Only the values handed to ``chunked_cross_entropy`` change; token accounting
    keeps reading the untouched ``loss_inputs``.
    """

    @staticmethod
    def _causal_shift(values: torch.Tensor, fill_value: Any) -> torch.Tensor:
        """Return one next-token target per position, padded at the tail."""
        return torch.nn.functional.pad(
            values, (0, 1), value=fill_value)[..., 1:].contiguous()

    def prepare_model_inputs(
            self,
            model_inputs: Mapping[str, Any],
            loss_inputs: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Inject the ``chunk_loss_*`` protocol, shifting raw labels first.

        Args:
            model_inputs: Batch fields forwarded to the model forward.
            loss_inputs: Loss and token-accounting fields from the batch.

        Returns:
            The model inputs extended with the family ``chunk_loss_*`` protocol.

        Raises:
            ValueError: If the batch carries neither ``shift_labels`` nor raw
                ``labels`` aligned with every local hidden position, or if
                ``loss_mask`` does not match ``labels``.
        """
        prepared_loss_inputs = dict(loss_inputs)
        if not isinstance(prepared_loss_inputs.get("shift_labels"), torch.Tensor):
            labels = prepared_loss_inputs.get("labels")
            if not isinstance(labels, torch.Tensor):
                raise ValueError(
                    "Kimi-K2.6 chunk loss requires raw labels or pre-shifted "
                    "shift_labels aligned with every local hidden position"
                )
            loss_mask = prepared_loss_inputs.get("loss_mask")
            if loss_mask is not None and (
                    not isinstance(loss_mask, torch.Tensor)
                    or loss_mask.shape != labels.shape):
                raise ValueError("loss_mask must be a Tensor with the same shape as labels")
            prepared_loss_inputs["shift_labels"] = self._causal_shift(
                labels, self.ignore_index)
            if loss_mask is not None:
                prepared_loss_inputs["loss_mask"] = self._causal_shift(
                    loss_mask.to(torch.bool), False)
        return super().prepare_model_inputs(model_inputs, prepared_loss_inputs)


__all__ = ["KimiChunkedCausalLMLoss"]
