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
"""SiTU/AttnRes parity against pinned released source, without inference imports.

Only the reviewed class and function are extracted from the local reference.
KDA, MLA, MoE/QB, complete-model and distributed acceptance remain separate gates.
"""
# Native Torch model adapters follow the same exception as other HF model families.
# pylint: disable=forbidden-backend-import

from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from hyper_parallel.models.kimi_k3.modeling_kimi_linear import SituAndMul
from hyper_parallel.components.functional.attention_residuals import block_attention_residuals
from hyper_parallel.models.validation_spec import ModuleParityCase


REVIEWED_SOURCE_SHA256 = "9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a"


def _released_symbols(context: Any) -> dict[str, Any]:
    """Reject unreviewed source before executing the two pure Torch definitions."""
    key = "kimi_k3.released_symbols"
    if key in context.options:
        return context.options[key]
    reference = context.manifest.reference_path
    if reference is None:
        raise ValueError("K3 module parity requires reference.source_path to a local released source checkout")
    path = Path(reference) / "modeling_kimi_linear.py"
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != REVIEWED_SOURCE_SHA256:
        raise ValueError("K3 reference source hash differs from the reviewed revision; review before parity")
    names = {"SituAndMul", "_apply_attn_res"}
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if {node.name for node in nodes} != names or len(nodes) != 2:
        raise ValueError("K3 reference must define exactly one reviewed SiTU and AttnRes symbol")
    namespace = {"torch": torch, "nn": nn, "__name__": "kimi_k3_released_reference"}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)  # pylint: disable=exec-used
    context.evidence_store.write_json("module_parity/kimi_k3_reference.json", {
        "file": str(path), "sha256": digest, "symbols": sorted(names),
        "scope": "SiTU and block AttnRes only; inference imports are not executed",
    })
    context.options[key] = namespace
    return namespace


def _final_model(context: Any) -> nn.Module:
    """Use exactly the builder declared in the manifest, including replacements."""
    key = "kimi_k3.final_model"
    if key not in context.options:
        builder = context.manifest.model.get("builder", {})
        target = context.manifest.import_callable(builder.get("_target_"))
        model = target(**{name: value for name, value in builder.items() if name != "_target_"})
        if not isinstance(model, nn.Module):
            raise ValueError("K3 parity model.builder must return the final nn.Module")
        context.options[key] = model
    return context.options[key]


def _situ_candidate(context: Any) -> nn.Module:
    model = _final_model(context)
    for fqn, module in model.named_modules():
        if isinstance(module, SituAndMul):
            context.evidence_store.write_json("module_parity/kimi_k3_situ_selection.json", {"fqn": fqn})
            return module
    raise ValueError("K3 SiTU parity requires an actual SituAndMul in the declared final model")


def _situ_reference(context: Any) -> nn.Module:
    candidate = _situ_candidate(context)
    return _released_symbols(context)["SituAndMul"](candidate.beta, candidate.linear_beta)


class _AttnResCandidate(nn.Module):
    """Expose the final model's output AttnRes boundary with its real parameters."""

    def __init__(self, model: nn.Module) -> None:
        """Retain the actual final model parameter objects and backend selection."""
        super().__init__()
        backbone = model.model
        self.proj = backbone.output_attn_res_proj
        self.norm = backbone.output_attn_res_norm
        self.backend = model.config.attn_res_backend

    def forward(self, prefix: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """Invoke the same operator boundary used by the production backbone.

        Args:
            prefix: Current residual prefix [tokens, hidden_size].
            history: Completed block residuals [tokens, blocks, hidden_size].

        Returns:
            Mixed residuals [tokens, hidden_size] using the final model's actual
            projection, normalization parameters and configured operator backend.
        """
        return block_attention_residuals(prefix, history, self.proj.weight, self.norm.weight,
                        norm_eps=self.norm.variance_epsilon, backend=self.backend)


class _AttnResReference(nn.Module):
    """Execute the released function with independent copies of all parameters."""

    def __init__(self, candidate: _AttnResCandidate, function: Callable) -> None:
        """Copy parameter storage and bind the independent released function."""
        super().__init__()
        self.proj = copy.deepcopy(candidate.proj)
        self.norm = copy.deepcopy(candidate.norm)
        self.function = function

    def forward(self, prefix: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """Preserve the released RMS epsilon and four gradient paths.

        Args:
            prefix: Current residual prefix [tokens, hidden_size].
            history: Completed block residuals [tokens, blocks, hidden_size].

        Returns:
            Independent reference residuals using copied projection and normalization
            parameters, allowing gradients for inputs and both parameter sets.
        """
        return self.function(prefix, history, self.proj, self.norm)


def _attn_res_candidate(context: Any) -> nn.Module:
    model = _final_model(context)
    if not getattr(model.model, "use_attn_residuals", False):
        raise ValueError("K3 AttnRes parity requires attn_res_block_size in the declared model")
    return _AttnResCandidate(model)


def _attn_res_reference(context: Any) -> nn.Module:
    return _AttnResReference(_attn_res_candidate(context), _released_symbols(context)["_apply_attn_res"])


def _copy_weights(reference: nn.Module, candidate: nn.Module, context: Any) -> None:
    del context
    reference.load_state_dict(candidate.state_dict(), strict=True)


def _random(context: Any, *shapes: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(20261024)
    dtype = getattr(torch, context.dtype)
    return tuple(torch.randn(shape, generator=generator).to(device=context.device, dtype=dtype).requires_grad_()
                 for shape in shapes)


def _situ_inputs(context: Any) -> tuple[torch.Tensor, ...]:
    return _random(context, (64, 2 * _final_model(context).config.hidden_size))


def _attn_res_inputs(context: Any) -> tuple[torch.Tensor, ...]:
    width = _final_model(context).config.hidden_size
    return _random(context, (64, width), (64, 2, width))


def get_module_cases() -> tuple[ModuleParityCase, ...]:
    """Return limited operator gates; never imply full model/oracle acceptance."""
    return (
        ModuleParityCase(
            name="kimi_k3.situ", candidate_selector="<root>", candidate_builder=_situ_candidate,
            reference_builder=_situ_reference, input_builder=_situ_inputs, weight_adapter=_copy_weights,
            required_dtypes=("float32", "bfloat16"),
        ),
        ModuleParityCase(
            name="kimi_k3.block_attn_res", candidate_selector="<root>", candidate_builder=_attn_res_candidate,
            reference_builder=_attn_res_reference, input_builder=_attn_res_inputs, weight_adapter=_copy_weights,
            required_dtypes=("float32", "bfloat16"), required_parameter_patterns=("proj.weight", "norm.weight"),
        ),
    )


__all__ = ["get_module_cases"]
