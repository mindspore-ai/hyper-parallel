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
"""Instance-level Qwen3-MoE router adaptation for Actor replay."""
from functools import partial
from importlib.metadata import version
import re
from typing import Any

from packaging.version import Version
import torch

from rl.router_replay import RouterReplay


_GATE_PATH = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.gate$")
_ROUTER_RETURNS_LOGITS = Version(version("transformers")) >= Version("5.6.0")


def _replay_qwen3_router(
    controller: RouterReplay,
    layer_id: int,
    norm_topk_prob: bool,
    module: torch.nn.Module,
    inputs: tuple[Any, ...],
    output: Any,
) -> Any:
    """Replace choices while retaining current router scores and their gradients."""
    del module, inputs
    targets = controller.layer_targets(layer_id)
    if targets is None:
        return output
    if not isinstance(output, tuple) or len(output) != 3:
        raise ValueError("Qwen3-MoE replay requires gate (router values, scores, indices) output")
    router_values, native_scores, native_indices = output
    indices, valid = targets
    if (tuple(router_values.shape) != (indices.shape[0], controller.num_experts)
            or native_scores.shape != indices.shape or native_indices.shape != indices.shape):
        raise ValueError("Qwen3-MoE replay expects full B*S token rows on every TP sibling (SP/CP/PP disabled)")
    if router_values.device != indices.device:
        raise ValueError("Router replay targets and gate probabilities must use the same device")
    indices = torch.where(valid[:, None], indices.to(dtype=torch.long), native_indices)
    # Transformers 5.6 changed the first slot from FP32 probabilities to raw logits.
    probabilities = router_values.softmax(dim=-1, dtype=torch.float32) if _ROUTER_RETURNS_LOGITS else router_values
    scores = probabilities.gather(dim=-1, index=indices)
    if norm_topk_prob:
        scores = scores / scores.sum(dim=-1, keepdim=True)
    return (
        router_values,
        torch.where(valid[:, None], scores.to(dtype=native_scores.dtype), native_scores),
        indices.to(dtype=native_indices.dtype),
    )


def install_router_replay(model: torch.nn.Module) -> RouterReplay:
    """Validate and install removable hooks on one built Qwen3-MoE Actor instance.

    Args:
        model: Constructed Actor model with its final distributed/checkpoint modules.

    Returns:
        Controller owning only this model's gate hooks.
    """
    config = getattr(model, "config", None)
    if config is None or getattr(config, "model_type", None) != "qwen3_moe":
        raise ValueError("Router replay currently supports only Qwen3-MoE Actors")
    expected = tuple(
        index for index in range(config.num_hidden_layers)
        if index not in config.mlp_only_layers and (index + 1) % config.decoder_sparse_step == 0
    )
    gates = {}
    for name, module in model.named_modules():
        match = _GATE_PATH.search(name)
        if match is None:
            continue
        layer_id = int(match.group(1))
        if (type(module).__name__ != "Qwen3MoeTopKRouter"
                or getattr(module, "top_k", None) != config.num_experts_per_tok
                or getattr(module, "num_experts", None) != config.num_experts
                or getattr(module, "norm_topk_prob", None) != config.norm_topk_prob):
            raise ValueError(f"Unsupported Qwen3-MoE gate interface at {name}")
        if layer_id in gates:
            raise ValueError(f"Duplicate Qwen3-MoE decoder layer ID {layer_id}")
        gates[layer_id] = module
    if tuple(sorted(gates)) != expected:
        raise ValueError("Built Qwen3-MoE gates do not match configured sparse decoder layers")
    controller = RouterReplay(expected, config.num_experts, config.num_experts_per_tok)
    try:
        for layer_id, gate in gates.items():
            hook = partial(_replay_qwen3_router, controller, layer_id, bool(config.norm_topk_prob))
            controller.add_hook(gate.register_forward_hook(hook))
    except Exception:
        controller.remove()
        raise
    return controller
