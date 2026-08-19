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

from dataclasses import dataclass
from typing import Any, Iterator, Optional

from hyper_parallel import HSDPModule, get_platform

platform = get_platform()

HYPER_MODEL_IMPLEMENTATION = "hyper"
NATIVE_MODEL_IMPLEMENTATION = "native"
SUPPORTED_MODEL_IMPLEMENTATIONS = (
    HYPER_MODEL_IMPLEMENTATION,
    NATIVE_MODEL_IMPLEMENTATION,
)
HYPER_QWEN3_ARCHITECTURE = "HyperQwen3ForCausalLM"
HYPER_QWEN3_5_ARCHITECTURE = "HyperQwen3_5ForCausalLM"
NATIVE_QWEN3_ARCHITECTURE = "Qwen3ForCausalLM"
NATIVE_QWEN3_5_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
_HYPER_ARCHITECTURES = {
    "qwen3": HYPER_QWEN3_ARCHITECTURE,
    "qwen3_5": HYPER_QWEN3_5_ARCHITECTURE,
}


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
        """Return the supported Qwen model family."""
        if (
            self.hf_architecture == "Qwen3ForCausalLM"
            and self.model_type == "qwen3"
        ):
            return "qwen3"
        composite_qwen3_5 = (
            self.hf_architecture == "Qwen3_5ForConditionalGeneration"
            and self.model_type == "qwen3_5"
            and self.text_model_type == "qwen3_5_text"
        )
        text_qwen3_5 = (
            self.hf_architecture == "Qwen3_5ForCausalLM"
            and self.model_type == "qwen3_5_text"
            and self.text_model_type == "qwen3_5_text"
        )
        if composite_qwen3_5 or text_qwen3_5:
            return "qwen3_5"
        raise ValueError(
            "Unsupported RL model identity: "
            f"architecture={self.hf_architecture!r}, model_type={self.model_type!r}, "
            f"text_model_type={self.text_model_type!r}"
        )

    @property
    def native_uses_language_model_prefix(self) -> bool:
        """Return whether native vLLM wraps text weights under ``language_model``."""
        return self.hf_architecture == "Qwen3_5ForConditionalGeneration"


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
        if (
            not self.is_hyper
            and self.family == "qwen3_5"
            and self.model.native_uses_language_model_prefix
            and name.startswith("model.")
            and not name.startswith("model.language_model.")
        ):
            return f"model.language_model.{name.removeprefix('model.')}"
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
    model_family: str = "qwen3_5",
) -> str:
    """Return the default architecture for one implementation and family."""
    normalized = normalize_model_implementation(implementation)
    if normalized == HYPER_MODEL_IMPLEMENTATION:
        try:
            return _HYPER_ARCHITECTURES[model_family]
        except KeyError as error:
            raise ValueError(
                f"Unsupported Hyper-vLLM model family: {model_family!r}"
            ) from error
    if model_family == "qwen3":
        return NATIVE_QWEN3_ARCHITECTURE
    if model_family == "qwen3_5":
        return NATIVE_QWEN3_5_ARCHITECTURE
    raise ValueError(f"Unsupported native vLLM model family: {model_family!r}")


def resolve_vllm_model(
    model: ModelRegistration,
    implementation: Any,
) -> VLLMModelRegistration:
    """Resolve the single vLLM model contract used by engine and weight sync."""
    normalized = normalize_model_implementation(implementation)
    architecture = (
        architecture_for_implementation(normalized, model.family)
        if normalized == HYPER_MODEL_IMPLEMENTATION
        else model.hf_architecture
    )
    return VLLMModelRegistration(model, normalized, architecture)


def build_role_model(runtime_config: object, distributed_setup: object, *, frozen: bool) -> platform.Module:
    """Build one finalized role model through the HyperModels atomic loader."""
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


def build_role_optimizer(runtime_config: object, model: platform.Module) -> tuple[object, object]:
    """Build an independent optimizer and scheduler for one trainable role."""
    optimizer = runtime_config.optimizer.build(model=model).get_optimizer()
    lr_scheduler = None
    if runtime_config.lr_scheduler is not None:
        lr_scheduler = runtime_config.lr_scheduler.build(
            optimizer=optimizer,
            train_steps=runtime_config.training.train_iters,
        ).get_lr_scheduler()
    return optimizer, lr_scheduler


def iter_hsdp_roots(model: platform.Module) -> Iterator[HSDPModule]:
    """Yield every distinct HSDP root reachable from one role model."""
    seen = set()
    for _, candidate in platform.get_cells_and_names(model):
        if isinstance(candidate, HSDPModule) and id(candidate) not in seen:
            seen.add(id(candidate))
            yield candidate


__all__ = [
    "HYPER_MODEL_IMPLEMENTATION",
    "HYPER_QWEN3_ARCHITECTURE",
    "HYPER_QWEN3_5_ARCHITECTURE",
    "ModelRegistration",
    "NATIVE_MODEL_IMPLEMENTATION",
    "NATIVE_QWEN3_ARCHITECTURE",
    "NATIVE_QWEN3_5_ARCHITECTURE",
    "SUPPORTED_MODEL_IMPLEMENTATIONS",
    "VLLMModelRegistration",
    "architecture_for_implementation",
    "build_role_model",
    "build_role_optimizer",
    "iter_hsdp_roots",
    "normalize_model_implementation",
    "resolve_vllm_model",
]
