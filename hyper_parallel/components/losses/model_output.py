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

__all__ = ["ModelOutputLoss"]

from typing import Any, Dict, Optional, Union

# AutoModels loss components implement the Transformers/PyTorch Trainer API.
import torch

from hyper_parallel.data.constants import IGNORE_INDEX


class ModelOutputLoss(torch.nn.Module):
    """Return the loss field from a Transformers-style model output.

    The value the model reports is the mean over its own valid targets, so the
    module declares ``token_mean`` normalization. Standalone evaluation reads
    that declaration to turn the value into a token-weighted numerator; a
    replacement loss that reports a sum or an already-reduced value must
    declare its own normalization instead of inheriting this one silently.
    """

    #: How this loss's reported value is normalized across valid targets.
    loss_normalization = "token_mean"
    #: Whether ``forward`` consumes the Trainer-aligned valid-target count.
    #: Callers pass ``valid_token_count`` when this is true; subclasses inherit
    #: the declaration so the call contract never depends on an exact type.
    accepts_valid_token_count = True

    def forward(  # pylint: disable=unused-argument
        self,
        *,
        model_output: Any,
        labels: Optional[torch.Tensor],
        valid_token_count: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Read the model-computed loss.

        Args:
            model_output: Model output exposing a ``loss`` attribute.
            labels: Unshifted labels used by legacy callers that do not supply
                an aligned token count.
            valid_token_count: Trainer count after target alignment and masking.
                When omitted, preserve the legacy unshifted-label convention.

        Returns:
            The loss tensor or named loss mapping from ``model_output.loss``.
        """
        local_loss = model_output.loss
        if not isinstance(local_loss, torch.Tensor):
            return local_loss

        # Packed inputs may already be shifted: dropping their first target
        # again can erase the only supervised token on a CP shard. Trainer owns
        # target alignment, so use its count whenever it is available.
        if valid_token_count is not None:
            has_valid_labels = valid_token_count > 0
        elif labels is not None:
            has_valid_labels = labels[..., 1:].ne(IGNORE_INDEX).any()
        else:
            return local_loss
        local_loss = torch.where(has_valid_labels, local_loss, torch.zeros_like(local_loss))

        return local_loss
