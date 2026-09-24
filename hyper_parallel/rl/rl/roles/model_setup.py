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
"""Model identity and role construction shared by training and rollout."""

__all__ = [
    "HYPER_MODEL_IMPLEMENTATION",
    "HYPER_QWEN3_ARCHITECTURE",
    "ModelRegistration",
    "NATIVE_MODEL_IMPLEMENTATION",
    "NATIVE_QWEN3_ARCHITECTURE",
    "SUPPORTED_MODEL_IMPLEMENTATIONS",
    "VLLMModelRegistration",
    "architecture_for_implementation",
    "build_role_model",
    "build_role_optimizer",
    "iter_hsdp_roots",
    "normalize_model_implementation",
    "resolve_vllm_model",
]


import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

import torch

from hyper_parallel import HSDPModule

_TRANSFORMERS_BUILTIN_ATTENTION = "transformers_builtin"

HYPER_MODEL_IMPLEMENTATION = "hyper"
NATIVE_MODEL_IMPLEMENTATION = "native"
SUPPORTED_MODEL_IMPLEMENTATIONS = (
    HYPER_MODEL_IMPLEMENTATION,
    NATIVE_MODEL_IMPLEMENTATION,
)
HYPER_QWEN3_ARCHITECTURE = "HyperQwen3ForCausalLM"
NATIVE_QWEN3_ARCHITECTURE = "Qwen3ForCausalLM"


@dataclass(frozen=True)
class ModelRegistration:
    """Resolved logical model, HF identity, and local artifacts."""

    name: str
    hyper_model_name: str
    weights_path: str
    tokenizer_path: str
    hf_architecture: str
    model_type: str
    text_model_type: str
    tie_word_embeddings: bool

    @property
    def family(self) -> str:
        """Return the checkpoint-derived supported model family."""
        if (
            self.hf_architecture == "Qwen3ForCausalLM"
            and self.model_type == "qwen3"
        ):
            return "qwen3"
        raise ValueError(
            "Unsupported RL model identity: "
            f"architecture={self.hf_architecture!r}, model_type={self.model_type!r}, "
            f"text_model_type={self.text_model_type!r}"
        )


@dataclass(frozen=True)
class VLLMModelRegistration:
    """Resolved native or Hyper model contract shared by all vLLM paths."""

    model: ModelRegistration
    implementation: str
    architecture: str

    @property
    def family(self) -> str:
        """Return the checkpoint-derived model family."""
        return self.model.family

    @property
    def is_hyper(self) -> bool:
        """Return whether rollout uses the Hyper adapter."""
        return self.implementation == HYPER_MODEL_IMPLEMENTATION

    def actor_weight_name(self, name: str) -> Optional[str]:
        """Map one canonical Actor parameter name into the rollout namespace."""
        if name == "lm_head.weight" and self.model.tie_word_embeddings:
            return None
        return name


def normalize_model_implementation(value: Any) -> str:
    """Validate one rollout-side vLLM model implementation."""
    implementation = str(value or NATIVE_MODEL_IMPLEMENTATION).strip().lower()
    if implementation not in SUPPORTED_MODEL_IMPLEMENTATIONS:
        raise ValueError(
            "rollout.vllm.model_implementation must be 'hyper' or 'native', "
            f"got {value!r}"
        )
    return implementation


def architecture_for_implementation(
    implementation: str,
    model_family: str = "qwen3",
) -> str:
    """Return the available architecture for one rollout implementation."""
    normalized = normalize_model_implementation(implementation)
    architectures = {
        "qwen3": {
            HYPER_MODEL_IMPLEMENTATION: HYPER_QWEN3_ARCHITECTURE,
            NATIVE_MODEL_IMPLEMENTATION: NATIVE_QWEN3_ARCHITECTURE,
        },
    }
    try:
        return architectures[model_family][normalized]
    except KeyError as error:
        raise ValueError(f"Unsupported vLLM model family: {model_family!r}") from error


def resolve_vllm_model(
    model: ModelRegistration,
    implementation: Any,
) -> VLLMModelRegistration:
    """Resolve the single vLLM model contract used by engine and weight sync."""
    normalized = normalize_model_implementation(implementation)
    architecture = architecture_for_implementation(normalized, model.family)
    if (
        normalized == NATIVE_MODEL_IMPLEMENTATION
        and architecture != model.hf_architecture
    ):
        raise ValueError(
            "Native vLLM architecture does not match the checkpoint identity: "
            f"resolved={architecture!r}, checkpoint={model.hf_architecture!r}"
        )
    return VLLMModelRegistration(model, normalized, architecture)


def build_role_model(runtime_config: object, distributed_setup: object, *, frozen: bool) -> torch.nn.Module:
    """Build one finalized role model through the HyperAutoModel atomic loader."""
    activation_checkpoint = getattr(runtime_config.activation_checkpoint, "mode", "off")
    model = runtime_config.model.build(
        distributed_setup=distributed_setup,
        activation_checkpoint=activation_checkpoint,
        peft_config=runtime_config.peft,
    )
    if frozen:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
    return model


def build_role_optimizer(runtime_config: object, model: torch.nn.Module) -> tuple[object, object]:
    """Build an independent optimizer and scheduler for one trainable role."""
    optimizer = runtime_config.optimizer.target.build(model=model).get_optimizer()
    lr_scheduler = None
    if runtime_config.lr_scheduler is not None:
        lr_scheduler = runtime_config.lr_scheduler.build(
            optimizer=optimizer,
            train_iters=runtime_config.training.train_iters,
        ).get_lr_scheduler()
    return optimizer, lr_scheduler


def iter_hsdp_roots(model: torch.nn.Module) -> Iterator[HSDPModule]:
    """Yield every distinct HSDP root reachable from one role model."""
    seen = set()
    for _, candidate in model.named_modules():
        if isinstance(candidate, HSDPModule) and id(candidate) not in seen:
            seen.add(id(candidate))
            yield candidate


def _model_boolean(model: Mapping[str, Any], name: str, default: bool) -> bool:
    """Return one validated Boolean model option."""
    value = model.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"model.{name} must be a boolean")
    return value


def model_trust_remote_code(model: Mapping[str, Any]) -> bool:
    """Return whether the Trainer model may load remote implementation code."""
    return _model_boolean(model, "trust_remote_code", True)


def tokenizer_trust_remote_code(model: Mapping[str, Any]) -> bool:
    """Return the tokenizer remote-code choice, independent of the model choice."""
    return _model_boolean(model, "tokenizer_trust_remote_code", True)


def trainer_attention_implementation(model: Mapping[str, Any]) -> str:
    """Translate the explicit model-family attention selection for Transformers."""
    selection = model.get("attention_implementation")
    if selection is None:
        return str(model.get("attn_implementation", "sdpa"))
    if "attn_implementation" in model:
        raise ValueError(
            "model.attention_implementation and model.attn_implementation are mutually exclusive"
        )
    if selection != _TRANSFORMERS_BUILTIN_ATTENTION:
        raise ValueError(
            "Unsupported model.attention_implementation "
            f"{selection!r}; supported value is {_TRANSFORMERS_BUILTIN_ATTENTION!r}"
        )
    return "sdpa"


def resolve_model(model: Mapping[str, Any]) -> ModelRegistration:
    """Resolve the configured model shared by training and rollout."""
    name = model.get("registry_name")
    if not isinstance(name, str) or not name:
        raise ValueError("model.registry_name must be a non-empty string")
    config_path = Path(str(model["weights_path"])) / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Model config does not exist: {config_path}")
    with config_path.open(encoding="utf-8") as config_file:
        hf_config = json.load(config_file)
    architectures = hf_config.get("architectures")
    if not isinstance(architectures, list) or len(architectures) != 1:
        raise ValueError(
            f"Model config must define exactly one architecture, got {architectures!r}"
        )
    text_config = hf_config.get("text_config", hf_config)
    if not isinstance(text_config, Mapping):
        raise ValueError("Model text_config must be a mapping when present")
    registration = ModelRegistration(
        name=name,
        hyper_model_name=str(model["name"]),
        weights_path=str(model["weights_path"]),
        tokenizer_path=str(model["tokenizer_path"]),
        hf_architecture=str(architectures[0]),
        model_type=str(hf_config.get("model_type", "")),
        text_model_type=str(text_config.get("model_type", hf_config.get("model_type", ""))),
        tie_word_embeddings=bool(text_config.get("tie_word_embeddings", False)),
    )
    # Validate checkpoint identity and loader options before runtime construction.
    _ = registration.family
    model_trust_remote_code(model)
    tokenizer_trust_remote_code(model)
    trainer_attention_implementation(model)
    return registration
