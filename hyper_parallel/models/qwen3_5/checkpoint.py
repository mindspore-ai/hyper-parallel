# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""HuggingFace Qwen3.5-dense → hyper state_dict converter.

Reads the safetensors checkpoint of a dense Qwen3.5 variant
(e.g. ``Qwen/Qwen3.5-0.8B-Base`` / ``Qwen3.5-2B-Base`` / ``Qwen3.5-9B-Base``
/ ``Qwen3.5-27B``) and returns a state_dict keyed by hyper module paths
for :class:`Qwen3_5ForCausalLM` (text-only).

Key remapping (identical to qwen3_5_moe except NO expert unpacking)::

    model.language_model.embed_tokens.weight  → embed_tokens.weight
    model.language_model.layers.{i}.*         → layers.{i}.*
    model.language_model.norm.weight          → norm.weight
    lm_head.weight                            → lm_head.weight (if present)
    model.visual.*                            → (silently dropped)
    model.mtp.*                               → (silently dropped)

Linear-attention layers keep the upstream combined
``linear_attn.in_proj_qkv.weight`` of shape
``(key_dim * 2 + value_dim, hidden_size)`` (block layout ``[Q | K | V]``
along axis 0). Tensor parallelism slices that fused tensor by block at
parallelize/load time so single-card and sharded runs use the same source
checkpoint layout.

When ``tie_word_embeddings=True`` (default for 0.8B-Base), some checkpoints
omit ``lm_head.weight`` from the safetensors index; hyper handles this
transparently because :class:`Qwen3_5ForCausalLM.__init__` re-binds
``self.lm_head.weight = self.embed_tokens.weight`` at construction time.
"""
from __future__ import annotations

__all__ = ["load_hf_qwen3_5_state_dict"]

import json
import logging
import os
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)


def _remap_simple_key(hf_key: str, max_layer: int) -> Optional[str]:
    """Return hyper key for a dense-Qwen3.5 checkpoint key, or None to skip.

    ``max_layer`` is ``num_hidden_layers - 1``: any layer index beyond it
    is silently dropped (so hyper can use a smaller truncated model than
    the full checkpoint without erroring).
    """
    # Vision tower & MTP head — text-only model, drop silently.
    # 0.8B-Base stores MTP params at top-level ``mtp.*`` (not
    # ``model.mtp.*``), so we match both forms.
    if hf_key.startswith("model.visual."):
        return None
    if hf_key.startswith("model.mtp.") or hf_key.startswith("mtp."):
        return None

    # lm_head sits at root in the source checkpoint, same in hyper.
    if hf_key == "lm_head.weight":
        return "lm_head.weight"

    # model.language_model.{embed_tokens|norm|layers.X}.* → model.{...}
    prefix = "model.language_model."
    if hf_key.startswith(prefix):
        tail = hf_key[len(prefix):]
        if tail.startswith("layers."):
            try:
                layer_i = int(tail.split(".")[1])
            except (IndexError, ValueError):
                return None
            if layer_i > max_layer:
                return None
        return f"model.{tail}"

    logger.debug("Unmapped HF key dropped: %s", hf_key)
    return None

def _maybe_split_in_proj_qkv(
    hyper_sd: Dict[str, torch.Tensor],
    hyper_key: str,
    tensor: torch.Tensor,
    key_dim: int,
) -> None:
    """Insert ``tensor`` into ``hyper_sd``.

    The legacy name is kept for callers/tests that exercise the loader helper.
    Dense Qwen3.5 now keeps the checkpoint's fused ``in_proj_qkv`` parameter;
    TP blockwise slicing is handled later by :func:`qwen3_5_tp_load_transforms`.
    """
    del key_dim
    hyper_sd[hyper_key] = tensor


def load_hf_qwen3_5_state_dict(
    weights_path: str,
    num_hidden_layers: int,
    dtype: Optional[torch.dtype] = None,
    *,
    linear_key_dim: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Load a dense Qwen3.5 safetensors checkpoint into hyper state_dict.

    Args:
        weights_path: Directory with ``model.safetensors.index.json`` and
            one or more ``model-XXXXX-of-XXXXX.safetensors`` shards.
        num_hidden_layers: Truncate the checkpoint to the first
            ``num_hidden_layers`` text-decoder layers (later layers are
            silently dropped).
        dtype: Optional cast applied to every returned tensor.
        linear_key_dim: Kept for API compatibility with older split-QKV
            loaders. Dense Qwen3.5 keeps the combined checkpoint key unchanged.

    Returns:
        Dict keyed by hyper module paths, values on CPU.

    Raises:
        FileNotFoundError: If the safetensors index file is missing.
    """
    # pylint: disable=C0415
    from safetensors import safe_open

    idx_path = os.path.join(weights_path, "model.safetensors.index.json")
    if not os.path.isfile(idx_path):
        raise FileNotFoundError(
            f"Qwen3.5-dense loader needs {idx_path}; did you pass a "
            f"directory containing model.safetensors.index.json?"
        )
    with open(idx_path, "r", encoding="utf-8") as f:
        idx = json.load(f)
    weight_map: Dict[str, str] = idx["weight_map"]

    # Group keys by shard so we open each file only once.
    shard_to_keys: Dict[str, list] = {}
    for hf_key, shard in weight_map.items():
        shard_to_keys.setdefault(shard, []).append(hf_key)

    logger.info(
        "Loading Qwen3.5 dense safetensors from %s (%d keys across %d "
        "shards), num_hidden_layers=%d",
        weights_path, len(weight_map), len(shard_to_keys), num_hidden_layers,
    )

    hyper_sd: Dict[str, torch.Tensor] = {}
    max_layer = num_hidden_layers - 1

    def _cast(t: torch.Tensor) -> torch.Tensor:
        return t.to(dtype) if dtype is not None and t.dtype != dtype else t

    skipped = 0
    for shard in sorted(shard_to_keys.keys()):
        shard_path = os.path.join(weights_path, shard)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for hf_key in shard_to_keys[shard]:
                mapped = _remap_simple_key(hf_key, max_layer)
                if mapped is None:
                    skipped += 1
                    continue
                tensor = _cast(f.get_tensor(hf_key))
                if linear_key_dim is not None:
                    _maybe_split_in_proj_qkv(
                        hyper_sd, mapped, tensor, linear_key_dim,
                    )
                else:
                    hyper_sd[mapped] = tensor

    # ``tie_word_embeddings=True`` checkpoints drop ``lm_head.weight``;
    # ``fully_shard`` breaks the Python-level tie, so synthesize the tensor
    # explicitly from ``model.embed_tokens.weight``.
    if "lm_head.weight" not in hyper_sd and "model.embed_tokens.weight" in hyper_sd:
        hyper_sd["lm_head.weight"] = hyper_sd["model.embed_tokens.weight"].clone()
        logger.info(
            "Tie synthesis: derived lm_head.weight from model.embed_tokens.weight "
            "(tie_word_embeddings=True in HF checkpoint)"
        )

    logger.info(
        "HF → hyper state_dict ready: %d keys (%d skipped)",
        len(hyper_sd), skipped,
    )
    return hyper_sd
