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
"""Pinned remote Kimi architecture construction with the current HF weight-tying lifecycle."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from hyper_parallel.models.kimi_k3.source_compat import prepare_model_import, prepare_training_mask_protocol


class KimiK3ForConditionalGeneration:
    """Construct the released architecture while retaining model-owned lifecycle compatibility."""

    @classmethod

    def from_config(cls, config: Any, **kwargs: Any) -> nn.Module:
        """Resolve pinned local modeling source and preserve HF's weight-tying arguments.

        This changes construction metadata only. The released architecture owns
        multimodal fusion, language/vision execution and its ordinary outputs.
        """
        del cls
        prepare_model_import()
        mapping = getattr(config, "auto_map", {})
        source_reference = mapping.get("AutoModelForCausalLM") or mapping.get("AutoModel")
        if not source_reference:
            raise ValueError("Kimi K3 config must declare its pinned modeling source")
        source_class = get_class_from_dynamic_module(source_reference, config._name_or_path, local_files_only=True)

        prepare_training_mask_protocol(source_class)

        class CompatibleKimiK3(source_class):
            """Keep current HF construction lifecycle around the released architecture."""

            def _init_weights(self, module: nn.Module) -> None:
                """Initialize training-only bare state omitted by the released inference initializer."""
                super()._init_weights(module)
                if type(module).__name__ == "KimiDeltaAttention":
                    with torch.no_grad():
                        module.A_log.copy_(module.A_log.uniform_(1, 16).log())
                        module.dt_bias.zero_()
                elif type(module).__name__ == "KimiRMSNorm":
                    nn.init.ones_(module.weight)
                elif type(module).__name__ == "Learnable2DInterpPosEmbDivided_fixed":
                    nn.init.normal_(module.weight)
                elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
                    module.reset_parameters()

            def tie_weights(self, *args: Any, **options: Any) -> Any:
                """Delegate current HF tying options to the actual causal language model."""
                return self.language_model.tie_weights(*args, **options)

        kwargs.pop("trust_remote_code", None)
        attention = kwargs.get("attn_implementation", "eager")
        result = CompatibleKimiK3._from_config(config, **kwargs)
        result.language_model.config._attn_implementation = attention
        result.language_model.model._use_flash_attention_2 = attention == "flash_attention_2"
        return result
