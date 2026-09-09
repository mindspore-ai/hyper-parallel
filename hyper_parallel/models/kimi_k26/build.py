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
"""Build Kimi-K2.6 multimodal pretraining through HyperAuto."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from transformers import DeepseekV3Config, PreTrainedModel

from hyper_parallel.distributed.mesh import DistributedSetup
from hyper_parallel.models._transformers import HyperAutoModelForImageTextToText
from hyper_parallel.models.build_options import CompileConfig
from hyper_parallel.models.kimi_k26.runtime import initialize_kimi_k26_parameters, prepare_kimi_k26_training
from hyper_parallel.models.kimi_k26.vision import register_kimi_k26_multimodal

logger = logging.getLogger(__name__)


def build_kimi_k26_for_pretraining(
    config_path: str,
    num_hidden_layers: int | None = None,
    num_vision_layers: int | None = None,
    torch_dtype: str = "bfloat16",
    attn_implementation: str = "sdpa",
    distributed_setup: DistributedSetup | None = None,
    peft_config: Any | None = None,
    compile_config: CompileConfig | dict[str, Any] | None = None,
    activation_checkpoint: str | None = None,
    swap_inputs: bool = False,
    activation_swap: str = "none",
    model_init_dtype: str | None = None,
    validate_placement: bool = False,
) -> PreTrainedModel:
    """Build Kimi multimodal pretraining through the existing HyperAuto pipeline.

    Args:
        config_path: Local official Kimi configuration assets.
        num_hidden_layers: Optional decoder depth; None retains the source architecture.
        num_vision_layers: Optional ViT depth; None retains the source architecture.
        torch_dtype: Parameter dtype for model construction.
        attn_implementation: Native Transformers attention implementation.
        distributed_setup: Trainer-provided FSDP and model-parallel topology.
        peft_config: PEFT configuration; unsupported for this pretraining builder.
        compile_config: Optional layer compilation configuration.
        activation_checkpoint: Activation recomputation mode.
        swap_inputs: Whether recomputation inputs are swapped.
        activation_swap: Attention activation swap mode.
        model_init_dtype: Optional final model parameter dtype.
        validate_placement: Whether to retain DTensor placement validation.

    Returns:
        A trainable native Transformers model, initialized and parallelized.
    """
    if peft_config is not None:
        raise ValueError("Kimi config-only pretraining does not implement PEFT")
    if distributed_setup is not None:
        mesh = distributed_setup.mesh_context
        if any(getattr(mesh, f"{axis}_size") != 1 for axis in ("tp", "cp", "pp")):
            raise ValueError("Kimi pretraining currently requires TP=CP=PP=1")
    path = Path(config_path).expanduser()
    if path.is_dir():
        path = path / "config.json"
    with path.open(encoding="utf-8") as stream:
        source = json.load(stream)
    values = source["text_config"]
    if num_hidden_layers is not None:
        if not 1 <= num_hidden_layers <= values["num_hidden_layers"]:
            raise ValueError("num_hidden_layers must be within the source decoder depth")
        values["num_hidden_layers"] = num_hidden_layers
    for key in ("model_type", "auto_map", "quantization_config", "_name_or_path"):
        values.pop(key, None)
    values.update(use_cache=False, architectures=["DeepseekV3ForCausalLM"])
    source["text_config"] = DeepseekV3Config(**values)
    if num_vision_layers is not None:
        if not 1 <= num_vision_layers <= source["vision_config"]["vt_num_hidden_layers"]:
            raise ValueError("num_vision_layers must be within the source ViT depth")
        source["vision_config"]["vt_num_hidden_layers"] = num_vision_layers
    source["vision_config"]["_attn_implementation"] = "sdpa"
    model_class = register_kimi_k26_multimodal(str(path.parent))
    for key in ("auto_map", "quantization_config"):
        source.pop(key, None)
    source["architectures"] = [model_class.__name__]
    config = model_class.config_class(**source)
    text_config = config.text_config
    text_config._attn_implementation = attn_implementation  # pylint: disable=protected-access
    logger.info(
        "Kimi pretraining: text_layers=%d, vision_layers=%s, hidden_size=%d, experts=%d, random initialization",
        text_config.num_hidden_layers,
        config.vision_config.vt_num_hidden_layers,
        text_config.hidden_size, text_config.n_routed_experts,
    )
    model = HyperAutoModelForImageTextToText.from_config(
        config,
        distributed_setup=distributed_setup,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        compile_config=compile_config,
        activation_checkpoint=activation_checkpoint,
        swap_inputs=swap_inputs,
        activation_swap=activation_swap,
        model_init_dtype=model_init_dtype,
        validate_placement=validate_placement,
        model_initializer=initialize_kimi_k26_parameters,
    )
    prepare_kimi_k26_training(model)
    return model
