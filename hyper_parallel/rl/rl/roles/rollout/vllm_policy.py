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
"""Shared Qwen3.5 model-selection, naming, and fingerprint contracts."""

import json
from hashlib import sha256
from typing import Any, Mapping

HYPER_MODEL_IMPLEMENTATION = "hyper"
NATIVE_MODEL_IMPLEMENTATION = "native"
SUPPORTED_MODEL_IMPLEMENTATIONS = (
    HYPER_MODEL_IMPLEMENTATION,
    NATIVE_MODEL_IMPLEMENTATION,
)
HYPER_QWEN3_5_ARCHITECTURE = "HyperQwen3_5ForCausalLM"
NATIVE_QWEN3_5_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
POLICY_FINGERPRINT_ALGORITHM = "qwen3_5_norms_f32_v2"


def normalize_model_implementation(value: Any) -> str:
    """Validate and normalize one configured vLLM model implementation."""
    implementation = str(value or HYPER_MODEL_IMPLEMENTATION).strip().lower()
    if implementation not in SUPPORTED_MODEL_IMPLEMENTATIONS:
        raise ValueError(
            "rollout.vllm.model_implementation must be 'hyper' or 'native', "
            f"got {value!r}"
        )
    return implementation


def architecture_for_implementation(implementation: str) -> str:
    """Return the pinned vLLM architecture for one implementation."""
    normalized = normalize_model_implementation(implementation)
    if normalized == HYPER_MODEL_IMPLEMENTATION:
        return HYPER_QWEN3_5_ARCHITECTURE
    return NATIVE_QWEN3_5_ARCHITECTURE


def map_policy_state_dict(
    state_dict: Mapping[str, Any],
    implementation: str,
) -> dict[str, Any]:
    """Map Hyper Actor names to the selected vLLM checkpoint namespace.

    The native Qwen3.5 outer model consumes Hugging Face names under
    ``model.language_model`` and performs its own fusion and TP slicing.
    Tensor objects and insertion order are preserved.
    """
    normalized = normalize_model_implementation(implementation)
    mapped = {}
    for name, tensor in state_dict.items():
        mapped_name = name
        if (
            normalized == NATIVE_MODEL_IMPLEMENTATION
            and name.startswith("model.")
            and not name.startswith("model.language_model.")
        ):
            mapped_name = f"model.language_model.{name.removeprefix('model.')}"
        if mapped_name in mapped:
            raise ValueError(
                f"vLLM policy-name mapping collision: {name!r} maps to {mapped_name!r}"
            )
        mapped[mapped_name] = tensor
    return mapped


def canonical_policy_weight_name(name: str) -> str:
    """Normalize trainer, Hyper vLLM, and native vLLM parameter namespaces."""
    for prefix in (
        "model.language_model.",
        "language_model.model.",
    ):
        if name.startswith(prefix):
            return f"model.{name.removeprefix(prefix)}"
    if name.startswith("language_model.lm_head."):
        return name.removeprefix("language_model.")
    return name


def is_policy_fingerprint_weight(name: str) -> bool:
    """Return whether a representation-stable replicated norm participates."""
    canonical_name = canonical_policy_weight_name(name)
    return (
        canonical_name.startswith("model.")
        and canonical_name.endswith("norm.weight")
        and ".linear_attn." not in canonical_name
    )


def policy_fingerprint_header(name: str, shape: tuple[int, ...]) -> bytes:
    """Serialize one canonical parameter identity for deterministic hashing."""
    return json.dumps(
        [canonical_policy_weight_name(name), list(shape)],
        separators=(",", ":"),
    ).encode("utf-8")


def policy_tensor_fingerprint(name: str, shape: tuple[int, ...], values: bytes) -> tuple[str, str]:
    """Return one canonical name and content digest for a policy tensor."""
    canonical_name = canonical_policy_weight_name(name)
    digest = sha256()
    digest.update(policy_fingerprint_header(canonical_name, shape))
    digest.update(values)
    return canonical_name, digest.hexdigest()


def aggregate_policy_fingerprint(tensors: Mapping[str, str], value_count: int) -> dict[str, Any]:
    """Aggregate canonical per-tensor digests without namespace-order ambiguity."""
    digest = sha256()
    for name, tensor_digest in sorted(tensors.items()):
        digest.update(json.dumps([name, tensor_digest], separators=(",", ":")).encode("utf-8"))
    return {
        "algorithm": POLICY_FINGERPRINT_ALGORITHM,
        "tensor_count": len(tensors),
        "value_count": value_count,
        "digest": digest.hexdigest(),
        "tensors": dict(sorted(tensors.items())),
    }


__all__ = [
    "HYPER_MODEL_IMPLEMENTATION",
    "HYPER_QWEN3_5_ARCHITECTURE",
    "NATIVE_MODEL_IMPLEMENTATION",
    "NATIVE_QWEN3_5_ARCHITECTURE",
    "POLICY_FINGERPRINT_ALGORITHM",
    "SUPPORTED_MODEL_IMPLEMENTATIONS",
    "aggregate_policy_fingerprint",
    "architecture_for_implementation",
    "canonical_policy_weight_name",
    "is_policy_fingerprint_weight",
    "map_policy_state_dict",
    "normalize_model_implementation",
    "policy_fingerprint_header",
    "policy_tensor_fingerprint",
]
