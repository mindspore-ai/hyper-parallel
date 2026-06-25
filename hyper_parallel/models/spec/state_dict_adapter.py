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
"""``StateDictAdapter`` — per-model HF ↔ hyper weight translator.

Plugged onto :class:`ModelSpec` so ``BaseTrainer._load_weights`` does not need
to know any model name. The adapter owns the HF-specific tensor renaming and
expert-fusion splitting; the trainer keeps the generic wrap-stripping +
shape-validation + ``load_state_dict`` flow.
"""

__all__ = ["StateDictAdapter"]

from typing import Dict, Optional, Protocol, runtime_checkable

import torch


@runtime_checkable
class StateDictAdapter(Protocol):
    """Protocol every model-specific state-dict adapter must satisfy.

    Owns HF ↔ hyper tensor translation so ``BaseTrainer`` stays generic.
    Load-side handles HF → hyper renaming and expert-fusion splitting.
    Save-side handles hyper → HF renaming for checkpoint export.
    """

    def load_hf_state_dict(
        self,
        weights_path: str,
        model_config,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, torch.Tensor]:
        """Read an HF-format checkpoint and return a plain ``state_dict``.

        Args:
            weights_path: Directory containing ``model.safetensors.index.json``
                and the matching shard files.
            model_config: The constructed model's ``.config`` instance. The
                adapter pulls per-architecture geometry (``num_hidden_layers``,
                ``num_experts`` …) from here so the trainer does not have to
                know which fields each architecture exposes.
            dtype: Optional target dtype to cast tensors to during load.

        Returns:
            A dict of HF-named tensor → tensor. Tensors are kept on CPU; the
            trainer-side wrap-stripping + shape-validation pass takes care of
            mapping HF names to hyper module paths and pushing them through
            the FSDP ``load_state_dict`` boundary.
        """

    def save_hf_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        model_config,
    ) -> Dict[str, torch.Tensor]:
        """Convert hyper state_dict to HF format for checkpoint export.

        Args:
            state_dict: The trainer's saved state_dict (hyper-named tensors).
            model_config: The model's ``.config`` instance.

        Returns:
            A dict of HF-named tensors for ``safetensors.save_file()`` or
            ``torch.save()``. Inverse of ``load_hf_state_dict``.
        """
