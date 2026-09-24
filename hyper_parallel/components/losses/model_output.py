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
"""Loss module that reads the loss produced by a model."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

# AutoModels loss components implement the Transformers/PyTorch Trainer API.
# pylint: disable-next=forbidden-backend-import
import torch

from hyper_parallel.data.constants import IGNORE_INDEX


class ModelOutputLoss(torch.nn.Module):
    """Return the loss field from a Transformers-style model output."""

    def __init__(self, *, pass_loss_inputs: bool = False, check_valid_labels: bool = True) -> None:
        """Configure supervision forwarding and causal-label checks.

        Args:
            pass_loss_inputs: Forward supervision under its public batch names.
            check_valid_labels: Zero loss when shifted causal labels are all ignored.
                Disable for complete model-owned objectives that can include auxiliary losses.
        """
        super().__init__()
        self.pass_loss_inputs = pass_loss_inputs
        self.check_valid_labels = check_valid_labels

    def prepare_model_inputs(self, model_inputs: dict, loss_inputs: dict) -> dict:
        """Pass supervision to model-owned objectives without renaming or shifting.

        Args:
            model_inputs: Forward fields from the public batch runtime.
            loss_inputs: Supervision and token-accounting fields from that runtime.

        Returns:
            A new dictionary preserving each supplied tensor by identity.
        """
        result = dict(model_inputs)
        if self.pass_loss_inputs:
            for name, value in loss_inputs.items():
                if name in result and result[name] is not value:
                    raise ValueError(f"Conflicting model and loss input: {name}")
                result[name] = value
        return result

    def forward(  # pylint: disable=unused-argument
        self,
        *,
        model_output: Any,
        labels: Optional[torch.Tensor],
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Read the model-computed loss.

        Args:
            model_output: Model output exposing a ``loss`` attribute.
            labels: Causal targets used by the optional valid-label check.

        Returns:
            The loss tensor or named loss mapping from ``model_output.loss``.
        """
        local_loss = model_output.loss
        if not self.check_valid_labels or labels is None or not isinstance(local_loss, torch.Tensor):
            return local_loss

        # Causal LM loss shifts labels by one position. A CP-local slice may
        # therefore contain no trainable target even when other CP ranks do.
        has_valid_labels = labels[..., 1:].ne(IGNORE_INDEX).any()
        local_loss = torch.where(has_valid_labels, local_loss, torch.zeros_like(local_loss))

        return local_loss


__all__ = ["ModelOutputLoss"]
