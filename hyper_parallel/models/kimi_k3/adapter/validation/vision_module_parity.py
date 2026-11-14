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

"""MoonViT block and projector parity against reviewed released definitions."""
# pylint: disable=forbidden-backend-import

import ast
from collections.abc import Sequence
from copy import deepcopy
from functools import partial
import hashlib
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import PretrainedConfig, PreTrainedModel

from hyper_parallel.models.kimi_k3.adapter.validation.native_module_parity import _final_model, _random
from hyper_parallel.models.validation_spec import ModuleParityCase

REVIEWED_VISION_SHA256 = "b9171c96726eda55234c92ac8dfae7e24c512fda68968ae8f2c3782b42665ea2"


def _released_vision(context: Any) -> dict:
    """Load reviewed mathematical definitions without importing the inference model."""
    key = "kimi_k3.released_vision"
    if key in context.options:
        return context.options[key]
    path = Path(context.manifest.reference_path) / "modeling_kimi_k3.py"
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != REVIEWED_VISION_SHA256:
        raise ValueError("K3 vision source differs from the reviewed revision")
    names = {"eager_attention", "_apply_rope_input_validation", "apply_rope", "MLP2",
             "MoonViTEncoderLayer", "PatchMergerMLPV2", "get_rope_shape",
             "get_1d_sincos_pos_embed_from_grid", "get_1d_sincos_pos_embed",
             "Learnable2DInterpPosEmbDivided_fixed", "MoonVision3dPatchEmbed",
             "Rope2DPosEmbRepeated", "MoonViT3dEncoder", "tpool_patch_merger", "MoonViT3dPretrainedModel"}
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if {node.name for node in nodes} != names:
        raise ValueError("K3 vision reference lacks required definitions")
    for node in nodes:
        if node.name == "get_rope_shape":
            # Remove compilation/warmup decorators, retaining the released math.
            node.decorator_list = []
    module_name = f"{__name__}._released"
    module = ModuleType(module_name)
    module.__file__ = str(path)
    namespace = module.__dict__
    namespace.update(torch=torch, nn=nn, F=F, math=math, Optional=Optional, np=np, Sequence=Sequence,
                     deepcopy=deepcopy, PreTrainedModel=PreTrainedModel,
                     PytorchGELUTanh=partial(nn.GELU, approximate="tanh"))
    # Transformers inspects the defining module during PreTrainedModel construction.
    sys.modules[module_name] = module
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)  # pylint: disable=exec-used
    namespace["VL_VISION_ATTENTION_FUNCTIONS"] = {"eager": namespace["eager_attention"]}
    context.options[key] = namespace
    context.evidence_store.write_json("module_parity/kimi_k3_vision_reference.json", {
        "file": str(path), "sha256": REVIEWED_VISION_SHA256, "symbols": sorted(names),
        "scope": "Encoder block, complete tower and projector; reference complex RoPE runs on CPU",
        "execution_only_changes": "get_rope_shape compilation and warmup decorators removed",
    })
    return namespace


class _VisionBlockCandidate(nn.Module):
    """Invoke the actual final model block on two independent media segments."""

    def __init__(self, block: nn.Module) -> None:
        """Retain the actual or independently constructed encoder block.

        Args:
            block: Encoder block used by this parity case.
        """
        super().__init__()
        self.block = block

    def forward(self, hidden: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """Apply lengths 4 and 8, including nontrivial rotary phases.

        Args:
            hidden: Patch activations from two independent media segments.
            angles: Spatial rotary phases matching the patch rows.

        Returns:
            Updated patch activations in the original row order.
        """
        return self.block(hidden, (4, 8), angles)


class _VisionBlockReference(_VisionBlockCandidate):
    """Retain the released complex-valued RoPE on a portable CPU oracle."""

    def forward(self, hidden: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """Differentiate the independent reference and return to the input device.

        Args:
            hidden: Patch activations retaining a gradient edge through the CPU copy.
            angles: Spatial rotary phases converted to the released complex format.

        Returns:
            Reference patch activations on the candidate input device.
        """
        self.block.to("cpu")
        angles = angles.float().cpu()
        frequencies = torch.polar(torch.ones_like(angles), angles)
        return self.block(hidden.cpu(), torch.tensor([0, 4, 12], dtype=torch.int32), 8,
                          frequencies).to(hidden.device)


def _vision_candidate(context: Any) -> nn.Module:
    model = _final_model(context)
    return _VisionBlockCandidate(model.vision_tower.encoder.blocks[0])


def _vision_reference(context: Any) -> nn.Module:
    config = _final_model(context).config.vision_config
    block = _released_vision(context)["MoonViTEncoderLayer"](
        num_heads=config.vt_num_attention_heads, hidden_dim=config.vt_hidden_size,
        mlp_dim=config.vt_intermediate_size, qkv_hidden_size=config.qkv_hidden_size,
        norm_type="rmsnorm", attn_implementation="eager", attn_bias=False, linear_bias=False,
        activation=partial(F.gelu, approximate="tanh"),
    )
    return _VisionBlockReference(block)


def _vision_inputs(context: Any) -> tuple[torch.Tensor, torch.Tensor]:
    config = _final_model(context).config.vision_config
    (hidden,) = _random(context, (12, config.vt_hidden_size))
    width = config.qkv_hidden_size // config.vt_num_attention_heads // 2
    angles = torch.arange(12 * width, device=context.device, dtype=torch.float32).reshape(12, width) * 0.013
    return hidden, angles


class _ProjectorCandidate(nn.Module):
    """Preserve the actual projector and its multi-media return contract."""

    def __init__(self, projector: nn.Module) -> None:
        """Retain the actual or independently constructed visual projector.

        Args:
            projector: Projector used by this parity case.
        """
        super().__init__()
        self.projector = projector

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Project two independent pooled spatial groups.

        Args:
            first: The first media's pooled spatial groups.
            second: The second media's pooled spatial groups.

        Returns:
            Projected feature tensors preserving the media boundary.
        """
        return self.projector([first, second])


class _ProjectorReference(_ProjectorCandidate):
    """Wrap the independently constructed released projector."""


def _projector_candidate(context: Any) -> nn.Module:
    return _ProjectorCandidate(_final_model(context).mm_projector)


def _projector_reference(context: Any) -> nn.Module:
    config = _final_model(context).config.vision_config
    options = SimpleNamespace(mm_hidden_size=config.mm_hidden_size, merge_kernel_size=(2, 2),
                              hidden_size=config.text_hidden_size, projector_ln_eps=config.projector_ln_eps)
    return _ProjectorReference(_released_vision(context)["PatchMergerMLPV2"](options))


def _projector_inputs(context: Any) -> tuple[torch.Tensor, ...]:
    width = _final_model(context).config.vision_config.vt_hidden_size
    return _random(context, (2, 4, width), (3, 4, width))



class _VisionTowerCandidate(nn.Module):
    """Keep the complete patch, position, encoder and temporal-pooling path."""

    def __init__(self, tower: nn.Module) -> None:
        """Retain the actual or independently constructed vision tower.

        Args:
            tower: Complete tower used by this parity case.
        """
        super().__init__()
        self.tower = tower

    def forward(self, pixels: torch.Tensor, grids: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Encode a mixed still-image and multi-frame record.

        Args:
            pixels: Flattened image/video patches with an input gradient.
            grids: Ordered temporal and spatial patch counts.

        Returns:
            Pooled spatial groups in original media order.
        """
        return tuple(self.tower(pixels, grids))


class _VisionTowerReference(_VisionTowerCandidate):
    """Execute released complex RoPE and the complete tower on the CPU oracle."""

    def forward(self, pixels: torch.Tensor, grids: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Preserve input gradients through device copies to the reference.

        Args:
            pixels: Patches from the same case as the candidate.
            grids: Geometry for a native-size image and an interpolated video group.

        Returns:
            Released tower outputs on the candidate input device.
        """
        self.tower.to("cpu")
        return tuple(value.to(pixels.device) for value in self.tower(pixels.cpu(), grids.cpu()))


def _tower_candidate(context: Any) -> nn.Module:
    return _VisionTowerCandidate(_final_model(context).vision_tower)


def _tower_reference(context: Any) -> nn.Module:
    config = _final_model(context).config.vision_config
    options = PretrainedConfig.from_dict({
        "hidden_size": config.vt_hidden_size, "intermediate_size": config.vt_intermediate_size,
        "num_hidden_layers": config.vt_num_hidden_layers, "num_attention_heads": config.vt_num_attention_heads,
        "qkv_hidden_size": config.qkv_hidden_size, "patch_size": config.patch_size,
        "init_pos_emb_height": config.init_pos_emb_height, "init_pos_emb_width": config.init_pos_emb_width,
        "init_pos_emb_time": config.init_pos_emb_time, "merge_kernel_size": config.merge_kernel_size,
        "merge_type": config.merge_type, "pos_emb_type": config.pos_emb_type,
        "pos_emb_interpolation_mode": config.pos_emb_interpolation_mode,
        "patch_embed_proj_bias": config.patch_embed_proj_bias, "norm_type": config.norm_type,
        "mlp_type": config.mlp_type, "attn_bias": config.attn_bias, "linear_bias": config.linear_bias,
        "attn_implementation": "eager",
    })
    return _VisionTowerReference(_released_vision(context)["MoonViT3dPretrainedModel"](options))


def _tower_inputs(context: Any) -> tuple[torch.Tensor, torch.Tensor]:
    config = _final_model(context).config.vision_config
    # Small mixed geometry exercises both position interpolation and t > 1 pooling.
    grids = torch.tensor([[1, 4, 4], [3, 2, 4]], device=context.device, dtype=torch.long)
    (pixels,) = _random(context, (40, 3, config.patch_size, config.patch_size))
    return pixels, grids


def _copy_weights(reference: nn.Module, candidate: nn.Module, context: Any) -> None:
    dtype = getattr(torch, context.dtype)
    candidate.to(device=context.device, dtype=dtype)
    reference.to(device=context.device, dtype=dtype)
    # Cases may select overlapping modules from the same final model instance.
    candidate.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)
    reference.load_state_dict(candidate.state_dict(), strict=True)


def get_vision_module_cases() -> tuple[ModuleParityCase, ...]:
    """Declare independent block, complete-tower and projector parity cases.

    Returns:
        Independent forward, input-gradient and parameter-gradient cases for the
        final builder's vision components in FP32 and BF16; scatter is separate.
    """
    return (
        ModuleParityCase(name="kimi_k3.vision_block", candidate_selector="<root>",
                         candidate_builder=_vision_candidate, reference_builder=_vision_reference,
                         input_builder=_vision_inputs, weight_adapter=_copy_weights,
                         required_dtypes=("float32", "bfloat16"),
                         required_parameter_patterns=("block.wqkv.weight", "block.wo.weight")),
        ModuleParityCase(name="kimi_k3.vision_tower", candidate_selector="<root>",
                         candidate_builder=_tower_candidate, reference_builder=_tower_reference,
                         input_builder=_tower_inputs, weight_adapter=_copy_weights,
                         required_dtypes=("float32", "bfloat16"),
                         required_parameter_patterns=("tower.patch_embed.proj.weight",
                                                      "tower.patch_embed.pos_emb.weight",
                                                      "tower.encoder.final_layernorm.weight")),
        ModuleParityCase(name="kimi_k3.projector", candidate_selector="<root>",
                         candidate_builder=_projector_candidate, reference_builder=_projector_reference,
                         input_builder=_projector_inputs, weight_adapter=_copy_weights,
                         required_dtypes=("float32", "bfloat16"),
                         required_parameter_patterns=("projector.proj.0.weight", "projector.proj.2.weight")),
    )
