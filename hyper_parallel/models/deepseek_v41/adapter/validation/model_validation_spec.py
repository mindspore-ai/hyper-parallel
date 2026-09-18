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
"""Declare DeepSeek-V4.1 parity and self-consistency contracts."""
# pylint: disable=forbidden-backend-import

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

import torch

from hyper_parallel.components.modules.engram import EngramModule
from hyper_parallel.components.modules.shared_compressed_dsa_attention import (
    SharedCompressedDSAAttention,
)
from hyper_parallel.models.deepseek_v41.adapter.validation.shared_state_trace import (
    begin_shared_state_trace,
    finish_shared_state_trace,
)
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import (
    DeepseekV41Attention,
    DeepseekV41Engram,
)
from hyper_parallel.models.validation_spec import (
    DataValidationSpec,
    ModelValidationSpec,
    ModuleParityCase,
    ParameterProbeSpec,
    SharedStateValidationSpec,
    StateInvariantSpec,
)


def _validate_engram_scratch_state(model: Any, context: Any) -> dict[str, Any] | None:
    """Require official q/k initialization on the final replaced scratch model."""
    if getattr(getattr(model, "config", None), "v41_model_mode", "full") != "validation_crop":
        return None
    trainer_config = context.get("trainer_config", {}) if isinstance(context, dict) else {}
    model_target = trainer_config.get("model", {}) if isinstance(trainer_config, dict) else {}
    if model_target.get("load_base_model"):
        return None
    failures = []
    engram_count = 0
    for module_fqn, module in model.named_modules():
        if not isinstance(module, (DeepseekV41Engram, EngramModule)):
            continue
        engram_count += 1
        for parameter_name in ("q_weight", "k_weight"):
            parameter = getattr(module, parameter_name)
            local = parameter.to_local() if callable(getattr(parameter, "to_local", None)) else parameter
            if not torch.equal(local.detach(), torch.ones_like(local)):
                failures.append(f"{module_fqn}.{parameter_name}")
    if engram_count == 0:
        return {"final_engram_modules": 0}
    return {"non_official_initialization": failures} if failures else None


def _validate_canonical_module_tree(model: Any, _context: Any) -> dict[str, Any] | None:
    """Require executable V4.1 attention and Engram modules in the base model."""
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", ())
    invalid_attention = [
        f"model.layers.{layer_index}.self_attn"
        for layer_index, layer in enumerate(layers)
        if not isinstance(
            getattr(layer, "self_attn", None),
            (DeepseekV41Attention, SharedCompressedDSAAttention),
        )
    ]
    engram_layer_ids = tuple(
        int(layer_id)
        for layer_id in getattr(getattr(model, "config", None), "v41_engram_layer_ids", ())
    )
    invalid_engram = [
        f"model.layers.{layer_id}.engram"
        for layer_id in engram_layer_ids
        if layer_id >= len(layers)
        or not isinstance(
            getattr(layers[layer_id], "engram", None),
            (DeepseekV41Engram, EngramModule),
        )
    ]
    if invalid_attention or invalid_engram:
        return {
            "invalid_attention_modules": invalid_attention,
            "invalid_engram_modules": invalid_engram,
        }
    return None


def _run_native_module_parity(context: Any) -> dict[str, Any]:
    """Execute the model-owned native parity harness in a recorded process."""
    native_repo = context.manifest.reference_path
    if native_repo is None:
        return {
            "case": "deepseek_v41_native_modules",
            "status": "BLOCKED",
            "reason": "reference.source_path is required",
        }
    output = context.evidence_store.path(
        "module_parity/deepseek_v41_native_modules/native_report.json"
    )
    command = [
        sys.executable,
        "-m",
        "hyper_parallel.models.deepseek_v41.adapter.validation.native_module_parity",
        "--native-repo",
        str(native_repo),
        "--device",
        context.device.split(":", 1)[0],
        "--dtype",
        context.dtype,
        "--output",
        str(output),
    ]
    process = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if not output.is_file():
        return {
            "case": "deepseek_v41_native_modules",
            "status": "FAIL",
            "returncode": process.returncode,
            "stdout": process.stdout[-4000:],
            "stderr": process.stderr[-4000:],
        }
    report = json.loads(output.read_text(encoding="utf-8"))
    summary = report.get("summary", {})
    status = (
        "PASS"
        if process.returncode == 0 and not summary.get("fail") and not summary.get("error")
        else "FAIL"
    )
    result = {
        "case": "deepseek_v41_native_modules",
        "status": status,
        "returncode": process.returncode,
        "native_report": str(output),
        "summary": summary,
        "compatibility": report.get("compatibility", []),
        "observations": report.get("cases", []),
    }
    if process.returncode != 0:
        result.update(
            {
                "stdout": process.stdout[-4000:],
                "stderr": process.stderr[-4000:],
            }
        )
    return result


def get_validation_spec() -> ModelValidationSpec:
    """Return V4.1 parity, parameter-probe, state, and data contracts."""
    return ModelValidationSpec(
        module_cases=(
            ModuleParityCase(
                name="deepseek_v41_native_modules",
                candidate_selector="<root>",
                required_dtypes=("float32", "bfloat16"),
                execution="isolated_process",
                isolated_runner=_run_native_module_parity,
            ),
        ),
        parameter_probes=(
            ParameterProbeSpec("model.layers.*.engram.q_weight"),
            ParameterProbeSpec("model.layers.*.engram.k_weight"),
            ParameterProbeSpec("model.layers.*.engram.wkv.weight"),
            ParameterProbeSpec("model.layers.*.self_attn.sinks"),
            ParameterProbeSpec("model.layers.*.mlp.gate.weight"),
            ParameterProbeSpec("model.layers.*.mlp.gate.bias"),
            ParameterProbeSpec("model.layers.*.mlp.gate.bias_vl", required=False),
        ),
        state_invariants=(
            StateInvariantSpec(
                name="deepseek_v41.canonical_module_tree",
                checker=_validate_canonical_module_tree,
                phase="structure",
                error_code="HP-REPLACE-002",
            ),
            StateInvariantSpec(
                name="deepseek_v41.final_engram_qk_scratch_initialization",
                checker=_validate_engram_scratch_state,
                phase="materialization",
                error_code="HP-MAT-003",
            ),
        ),
        shared_state=SharedStateValidationSpec(
            producer_keys=("compressed_kv", "index_key", "topk_indices", "candidate_blocks"),
            consumer_keys=("compressed_kv", "index_key", "topk_indices", "candidate_blocks"),
            begin_trace=begin_shared_state_trace,
            finish_trace=finish_shared_state_trace,
        ),
        data=DataValidationSpec(
            required_forward_fields=("input_ids", "labels"),
            runtime_fields=("packed_seq_params",),
            cp_replicated_forward_fields=("packed_seq_params",),
            modality_fields=(
                "token_types",
                "pixel_values",
                "image_patch_offsets",
                "image_vit_grid_hw",
                "image_llm_grid_hw",
                "image_batch_indices",
                "image_token_starts",
            ),
            modality_parameter_patterns=("model.vision.*", "model.aligner.*"),
            labels_are_shifted=True,
            deterministic_replay_required=True,
        ),
        metadata={
            "required_roles": (
                "moe",
                "engram",
                "full_attention",
                "reuse_attention",
                "reindex_attention",
            ),
            "high_precision_only": True,
        },
    )


__all__ = ["get_validation_spec"]
