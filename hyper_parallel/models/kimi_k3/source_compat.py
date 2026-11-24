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
"""Compatibility for pinned Kimi remote source using moved Transformers output descriptors."""

import inspect
from typing import Any

from torch import nn
from transformers.utils import generic


def prepare_model_import() -> None:
    """Expose the real Transformers descriptor under its historical import location.

    This runs before remote modeling import. It preserves the native output
    capture isinstance protocol and all descriptor fields; it never replaces
    output-capture decorators or model forward methods.
    """
    if hasattr(generic, "OutputRecorder"):
        return
    # This version-dependent module is optional on older supported Transformers installations.
    from transformers.utils.output_capturing import OutputRecorder  # pylint: disable=import-outside-toplevel

    generic.OutputRecorder = OutputRecorder


def prepare_training_mask_protocol(source_class: type[nn.Module]) -> None:
    """Bind the pinned language source's legacy keyword names to the installed HF mask API.

    This compatibility binding is local to the imported model source. It only
    supports cache-free training, where cache_position is redundant with the
    full input span; cached inference must retain a matching native source API.
    """
    namespace = source_class.__init__.__globals__
    language_class = namespace.get("KimiLinearForCausalLM")
    if language_class is None:
        raise ValueError("Pinned K3 source must expose its language architecture")
    model_class = language_class.__init__.__globals__["KimiLinearModel"]
    language_namespace = model_class.__init__.__globals__
    native = language_namespace["create_causal_mask"]
    if "input_embeds" in inspect.signature(native).parameters or getattr(native, "k3_training_protocol", False):
        return

    def training_mask(*, input_embeds: Any, cache_position: Any = None, **kwargs: Any) -> Any:
        """Translate the legacy input name and reject cache states outside the training contract."""
        del cache_position
        if kwargs.get("past_key_values") is not None:
            raise ValueError("K3 legacy mask compatibility supports cache-free training only")
        return native(inputs_embeds=input_embeds, **kwargs)

    training_mask.k3_training_protocol = True
    language_namespace["create_causal_mask"] = training_mask


__all__ = ["prepare_model_import", "prepare_training_mask_protocol"]
