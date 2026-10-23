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
"""FSDP execution boundaries shared by the Qwen visual model adapters."""

from typing import Any


def _visual_branch(model: Any) -> tuple[str, Any]:
    """Resolve the visual branch of a conditional-generation model."""
    for prefix in ("model.visual", "visual"):
        try:
            return prefix, model.get_submodule(prefix)
        except AttributeError:
            continue
    raise ValueError("Qwen visual FSDP policy requires a visual branch")


def get_fsdp_wrap_modules(model: Any) -> tuple[str, ...]:
    """Return visual FSDP units in their actual forward execution order."""
    prefix, visual = _visual_branch(model)
    indices = tuple(getattr(visual, "deepstack_visual_indexes", ()))
    mergers = getattr(visual, "deepstack_merger_list", ())
    if (
        len(indices) != len(mergers)
        or len(indices) != len(set(indices))
        or any(not isinstance(index, int) or index < 0 or index >= len(visual.blocks) for index in indices)
    ):
        raise ValueError("DeepStack indices must uniquely identify one visual block per merger")
    deepstack_by_block = {index: merger_index for merger_index, index in enumerate(indices)}
    # Interpolation reads pos_embed.weight.dtype outside the embedding forward.
    # Keep it root-owned so this observes the active root precision.
    order = [f"{prefix}.patch_embed"]
    for index in range(len(visual.blocks)):
        order.append(f"{prefix}.blocks.{index}")
        if index in deepstack_by_block:
            order.append(f"{prefix}.deepstack_merger_list.{deepstack_by_block[index]}")
    order.append(f"{prefix}.merger")
    return tuple(order)


def get_fsdp_excluded_subtrees(model: Any) -> tuple[str, ...]:
    """Keep visual containers out of automatic decoder-block discovery."""
    prefix, _ = _visual_branch(model)
    return (prefix,)


def get_fsdp_execution_order(model: Any, module_fqns: tuple[str, ...]) -> tuple[str, ...]:
    """Order selected visual units before the language execution branch."""
    selected = set(module_fqns)
    visual_order = tuple(fqn for fqn in get_fsdp_wrap_modules(model) if fqn in selected)
    visual_units = set(visual_order)
    return visual_order + tuple(fqn for fqn in module_fqns if fqn not in visual_units)


def get_fsdp_prefetch_groups(model: Any, module_fqns: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    """Keep visual backbone, feature heads and language prefetch independent.

    DeepStack heads feed different decoder layers, so their backward order is
    not the reverse of their visual forward order.
    """
    prefix, _ = _visual_branch(model)
    visual = tuple(fqn for fqn in module_fqns if fqn == prefix or fqn.startswith(f"{prefix}."))
    visual_units = set(visual)
    blocks = tuple(fqn for fqn in visual if fqn.startswith(f"{prefix}.blocks."))
    block_units = set(blocks)
    heads = tuple((fqn,) for fqn in visual if fqn not in block_units)
    language = tuple(fqn for fqn in module_fqns if fqn not in visual_units)
    return tuple(group for group in (blocks, *heads, language) if group)
