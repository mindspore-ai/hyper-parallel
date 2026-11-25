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
"""Task-host adaptation for the independent Predictor."""

from typing import Any

import torch

from hyper_parallel.infer import GenerationRuntime
from hyper_parallel.models.registry import resolve_model_adapter
from hyper_parallel.predict import PredictionContext, PredictionSummary, Predictor
from hyper_parallel.trainer.runtime.run_mode import validate_run_capabilities


def run_prediction(trainer: Any) -> PredictionSummary:
    """Supply host-owned model/assets/provenance without exposing the host to Predictor.

    Args:
        trainer: Initialized prediction task host providing the model, tokenizer, task
            configuration, device, DP mesh, optional chat template, and restore outcome.

    Returns:
        Summary of the committed prediction output, shared by all prediction lanes.
    """
    config = trainer.config
    validate_run_capabilities(config, trainer.mesh)
    adapter = resolve_model_adapter(trainer.model)
    if adapter is None or not adapter.supports_generation:
        raise NotImplementedError("predict requires a model-owned adapter declaring supports_generation")
    if trainer.hsdp_model_parts:
        raise NotImplementedError("predict does not support FSDP/HSDP-wrapped models")
    template = trainer.chat_template
    runtime = GenerationRuntime(
        model_adapter=adapter, device=trainer.device,
        rng=torch.Generator(device=trainer.device).manual_seed(config.training.seed + trainer.mesh.dp_rank),
    )
    context = PredictionContext(
        rank=trainer.mesh.dp_rank, world_size=trainer.mesh.dp_size,
        group=trainer.mesh.dp_cp_mesh.get_group() if trainer.mesh.dp_cp_mesh is not None else None,
    )
    predictor = Predictor(
        trainer.model, trainer.tokenizer, config.prediction.generation, runtime, config.prediction,
        context=context,
        chat_template=template.get_jinja_template() if template is not None else None,
        chat_template_kwargs=getattr(template, "chat_template_kwargs", None),
        metadata={
            "seed": config.training.seed,
            "checkpoint": trainer.restore_outcome.path,
            "model_source": config.model.to_dict(),
        },
    )
    return predictor.run_file()
