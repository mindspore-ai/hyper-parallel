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

from typing import Any, Dict, Optional, Union

# HyperModels loss components implement the Transformers/PyTorch Trainer API.
# pylint: disable-next=forbidden-backend-import
import torch

from hyper_parallel.auto_models.components.utils.constants import IGNORE_INDEX


class ModelOutputLoss(torch.nn.Module):
    """Return the loss field from a Transformers-style model output."""

    def forward(  # pylint: disable=unused-argument
        self,
        *,
        model_output: Any,
        labels: Optional[torch.Tensor],
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Read the model-computed loss.

        Args:
            model_output: Model output exposing a ``loss`` attribute.
            labels: Labels associated with the output. This default loss keeps
                the argument only to share the trainer-facing call signature
                with replaceable loss modules.

        Returns:
            The loss tensor or named loss mapping from ``model_output.loss``.
        """
        local_loss = model_output.loss
        if labels is None or not isinstance(local_loss, torch.Tensor):
            return local_loss

        # Causal LM loss shifts labels by one position. A CP-local slice may
        # therefore contain no trainable target even when other CP ranks do.
        has_valid_labels = labels[..., 1:].ne(IGNORE_INDEX).any()
        local_loss = torch.where(has_valid_labels, local_loss, torch.zeros_like(local_loss))

        return local_loss


__all__ = ["ModelOutputLoss"]
