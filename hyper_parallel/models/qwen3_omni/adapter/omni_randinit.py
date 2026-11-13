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
"""Random-initialized Qwen3-Omni model builder (skip weight loading).

Smoke-run helper: builds ``Qwen3OmniMoeForConditionalGeneration`` directly from
its ``config.json`` (via ``HyperAutoModelForMultimodalLM.from_config``) so no
checkpoint ``*.safetensors`` is required. Params are randomly initialised (meta
 init + FSDP2 materialisation in the distributed path) — used to validate the
Ascend pipeline before real weights are available. ``enable_audio_output=False``
builds the thinker only (~16.764B params), mirroring the V100 trimmed run.
"""

from typing import Any

from transformers import AutoConfig

from hyper_parallel.models._transformers.auto_model import (
    HyperAutoModelForMultimodalLM,
)
from hyper_parallel.models.qwen3_omni.adapter.model_patches import (
    ensure_packed_audio_input,
    ensure_deepstack_2d_masks,
    ensure_patch_embed_linear,
    ensure_thinker_forward,
)

# ``from_config`` accepts these keyword arguments; forward only the ones the
# trainer build invoked with, ignoring the rest of the runtime build kwargs.
_FORWARD_KWARGS = (
    "distributed_setup",
    "device_mesh",
    "backend",
    "peft_config",
    "validate_placement",
    "qat_config",
    "fp8_config",
    "compile_config",
    "freeze_config",
    "activation_checkpoint",
    "swap_inputs",
    "activation_swap",
)


def build_qwen3_omni_randinit(
    pretrained_model_name_or_path: str,
    torch_dtype: Any = "float32",
    attn_implementation: str = "eager",
    enable_audio_output: bool = False,
    num_hidden_layers: int | None = None,
    **kwargs: Any,
) -> Any:
    """Build a random-initialised Qwen3-Omni model from its local config.json.

    Args:
        pretrained_model_name_or_path: Directory holding ``config.json`` (the
            stock Qwen3-Omni checkpoint dir works — tokenizer files are read
            from it by the processor, and no safetensors are required).
        torch_dtype: Model computation dtype.
        attn_implementation: ``eager`` (NPC-safe default) or ``sdpa``.
        enable_audio_output: If False, build the thinker only.
        num_hidden_layers: If given, trim the thinker decoder (MoE) to this many
            layers. The 48-layer full thinker (~16.7B params, ~33.5GB bf16)
            does not fit on 64GB NPUs without offload; a few layers keep the
            smoke run small, fast and memory-light.
        **kwargs: Absorbs the trainer build-time kwargs (``distributed_setup``,
            ``peft_config``, ``activation_checkpoint``, ...) plus any model
            section kwargs in the YAML; the ones ``from_config`` accepts are
            forwarded.
    """
    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path, trust_remote_code=True
    )
    config.enable_audio_output = bool(enable_audio_output)
    if num_hidden_layers is not None:
        # Qwen3-Omni nests the MoE decoder as thinker_config.text_config.
        tc = getattr(config, "thinker_config", None) or config
        for depth in range(3):
            tc = getattr(tc, "text_config", tc)
        tc.num_hidden_layers = int(num_hidden_layers)
        # Some sub-configs mirror num_hidden_layers; keep them consistent.
        thinker = getattr(config, "thinker_config", None)
        if thinker is not None and hasattr(thinker, "text_config"):
            thinker.text_config.num_hidden_layers = int(num_hidden_layers)

    # ``from_pretrained`` applies the family patch set (the generation-only
    # omni wrapper needs a training ``forward`` delegating to ``thinker``);
    # ``from_config`` does not, so apply it here too before the trainer calls
    # model(**batch).
    ensure_thinker_forward()

    # HF 4.57.1 stock-bug fixes, same set ``from_pretrained`` applies (details
    # in each ensure_* docstring): Conv3d-backward NPU segfault / packed-audio
    # dead branch / deepstack 3-D-mask joint-branch crash.
    ensure_patch_embed_linear()
    ensure_packed_audio_input()
    ensure_deepstack_2d_masks()

    fwd = {}
    for key in _FORWARD_KWARGS:
        if key in kwargs:
            fwd[key] = kwargs[key]

    return HyperAutoModelForMultimodalLM.from_config(
        config,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        **fwd,
    )


__all__ = ["build_qwen3_omni_randinit"]
