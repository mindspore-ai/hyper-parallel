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
"""Rebuild Qwen3-Omni derived positional/rotary buffers after materialization.

hyper-parallel materialises meta models with ``to_empty``, which erases every
buffer that a stock module derives inside its ``__init__`` (thinker/vision
rotary ``inv_freq``, audio Sinusoids ``positional_embedding``) and that is NOT
stored in the checkpoint. ``initialize_weights()`` only runs the modules'
``_init_weights`` — a no-op for these — so after loading they hold 0/NaN and
every forward is corrupted. This provider attaches declarative rebuild recipes
(the ``register_rebuildable_buffer`` contract) capturing the exact stock
``__init__`` values, computed on CPU: NPU pow/div is ~1 ulp off the CPU result
and stock transformers computes these buffers in a CPU/meta ``__init__``, so
device-native recomputation would break byte-parity with a plain
``from_pretrained`` model.

The captured source values stay float32 — stock transformers 4.57.1 computes
them with explicit ``.float()`` — and the model_init_dtype cast re-copies them
(the generic dtype path restores fp32-sourced rebuildable buffers after the
cast; see ``model_builder.apply_model_init_dtype``), so precision-parity with
the reference logits is preserved without any Trainer-side patching.
"""

import torch

from hyper_parallel.models.materialization import register_rebuildable_buffer

# Stock class names (transformers 4.57.x; 5.x renames are tolerated by the
# class-name guard — unknown classes are skipped).
_BUFFER_CLASSES = {
    "Qwen3OmniMoeThinkerTextRotaryEmbedding": ("inv_freq",),
    "Qwen3OmniMoeVisionRotaryEmbedding": ("inv_freq",),
    "SinusoidsPositionEmbedding": ("positional_embedding",),
}


def _stock_value(module: torch.nn.Module, name: str) -> torch.Tensor:
    """Recompute one buffer with the stock ``__init__`` CPU formula.

    The 4.57.1 ``__init__`` paths are replicated here because 5.x moved to
    methods some installs lack; building a reference module is fragile for
    the thinker rotary (config-dependent), so the formulas are inlined.
    """
    if name == "inv_freq":
        if getattr(module, "rope_init_fn", None) is not None and hasattr(module, "config"):
            inv_freq, attn_scaling = module.rope_init_fn(module.config, "cpu")
            module.attention_scaling = attn_scaling
            return inv_freq.float()
        dim = int(module.inv_freq.numel()) * 2
        return 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    # Sinusoids positional embedding (transformers 4.57.1 inline formula).
    import math  # pylint: disable=import-outside-toplevel
    length, channels = tuple(module.positional_embedding.shape)
    max_timescale = 10000.0
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(
        -log_timescale_increment * torch.arange(channels // 2, dtype=torch.float32)
    )
    scaled_time = torch.arange(length, dtype=torch.float32)[:, None] * inv_timescales[None, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


def register_omni_derived_buffer_materialization(model: torch.nn.Module) -> None:
    """Retain the stock CPU-derived positional buffers as rebuild recipes."""
    for module in model.modules():
        buffer_names = _BUFFER_CLASSES.get(type(module).__name__)
        if not buffer_names:
            continue
        for name in buffer_names:
            if getattr(module, name, None) is None:
                continue
            register_rebuildable_buffer(
                module, name, value=_stock_value(module, name).float(),
            )


__all__ = ["register_omni_derived_buffer_materialization"]
