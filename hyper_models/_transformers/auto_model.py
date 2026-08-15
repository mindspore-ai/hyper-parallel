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
"""HyperAutoModel classes — HF-compatible model interface.

Following design doc 01_hf_compatibility_layer.md §6.
Stub — provides from_pretrained/from_config as entry points.
"""

import logging
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSequenceClassification,
    PreTrainedModel,
)

from hyper_models._transformers.infrastructure import (
    apply_model_infrastructure,
    instantiate_infrastructure,
)
from hyper_models._transformers.model_init import _init_model
from hyper_models._transformers.registry import get_hf_config, get_is_hf_model
from hyper_models.components.distributed.infrastructure import DistributedSetup
from hyper_models.components.utils.device import get_device_id, get_device_type  # pylint: disable=syntax-error

logger = logging.getLogger(__name__)


def _current_device() -> torch.device:
    """Return the current accelerator device, or CPU when no accelerator exists."""
    device_type = get_device_type()
    if device_type == "cpu":
        return torch.device("cpu")
    return torch.device(device_type, get_device_id())


class _BaseHyperAutoModelClass:
    """Shared from_pretrained / from_config logic.

    Following design doc 01 §6.1-6.2.
    """

    @classmethod
    def _from_pretrained_parent_class(cls, *args, **kwargs):
        """Delegate to the parent HuggingFace AutoModel class.

        Used by the HF-native path in _init_model so that the model is loaded
        through the standard transformers checkpoint logic.
        """
        return super().from_pretrained(*args, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        distributed_setup: Optional[DistributedSetup] = None,
        backend=None,
        peft_config=None,
        torch_dtype="auto",
        attn_implementation="sdpa",
        force_hf: bool = False,
        validate_placement: bool = False,
        qat_config=None,
        fp8_config=None,
        compile_config=None,
        freeze_config=None,
        activation_checkpoint: Optional[str] = None,
        activation_swap: str = "none",
        **kwargs,
    ) -> PreTrainedModel:
        """HF-compatible from_pretrained entry point.

        Following design doc 01 §6.2:
        ① resolve distributed_setup → MeshContext
        ② instantiate_infrastructure → ShardingPlanner + FSDP2Manager
        ③ AutoConfig.from_pretrained → hf_config
        ④ get_is_hf_model → custom/HF path
        ⑤ _build_model → meta + shard + load
        """
        if distributed_setup is None:
            distributed_setup = DistributedSetup()
        mesh = distributed_setup.mesh_context

        # ② Instantiate infrastructure
        sharding_planner, fsdp2_manager, autopipeline = instantiate_infrastructure(
            distributed_setup=distributed_setup,
            device=_current_device(),
        )

        # ③ Get HF config
        hf_config = get_hf_config(
            pretrained_model_name_or_path, attn_implementation, torch_dtype, **kwargs
        )

        # ④ Determine model path
        is_hf_model = get_is_hf_model(hf_config, force_hf)

        # ⑤ Build model
        return cls._build_model(
            pretrained_model_name_or_path,
            *model_args,
            is_hf_model=is_hf_model,
            hf_config=hf_config,
            mesh=mesh,
            sharding_planner=sharding_planner,
            fsdp2_manager=fsdp2_manager,
            autopipeline=autopipeline,
            backend=backend,
            peft_config=peft_config,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            validate_placement=validate_placement,
            load_base_model=True,
            distributed_setup=distributed_setup,
            qat_config=qat_config,
            fp8_config=fp8_config,
            compile_config=compile_config,
            freeze_config=freeze_config,
            activation_checkpoint=activation_checkpoint,
            activation_swap=activation_swap,
            **kwargs,
        )

    @classmethod
    def from_config(  # pylint: disable=unused-argument
        cls,
        config,
        *model_args,
        distributed_setup=None,
        device_mesh=None,
        backend=None,
        torch_dtype="auto",
        attn_implementation="sdpa",
        activation_checkpoint: Optional[str] = None,
        activation_swap: str = "none",
        **kwargs,
    ) -> PreTrainedModel:
        """Build model from PretrainedConfig (no weight loading).

        Following design doc 01 §6.1.
        """
        if distributed_setup is None:
            distributed_setup = DistributedSetup()
        mesh = distributed_setup.mesh_context

        sharding_planner, fsdp2_manager, autopipeline = instantiate_infrastructure(
            distributed_setup=distributed_setup,
            device=_current_device(),
        )

        is_hf_model = get_is_hf_model(config, force_hf=False)

        return cls._build_model(
            None,
            *model_args,
            is_hf_model=is_hf_model,
            hf_config=config,
            mesh=mesh,
            sharding_planner=sharding_planner,
            fsdp2_manager=fsdp2_manager,
            autopipeline=autopipeline,
            backend=backend,
            peft_config=None,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            validate_placement=False,
            load_base_model=False,
            distributed_setup=distributed_setup,
            activation_checkpoint=activation_checkpoint,
            activation_swap=activation_swap,
            **kwargs,
        )

    @classmethod
    def _build_model(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        is_hf_model,
        hf_config,
        mesh,
        sharding_planner,
        fsdp2_manager,
        autopipeline,
        backend,
        peft_config,
        torch_dtype,
        attn_implementation,
        validate_placement,
        load_base_model,
        distributed_setup=None,
        qat_config=None,
        fp8_config=None,
        compile_config=None,
        freeze_config=None,
        activation_checkpoint: Optional[str] = None,
        activation_swap: str = "none",
        **kwargs,
    ) -> PreTrainedModel:
        """Core model building orchestration.

        Following design doc 01 §6.3:
        Step 1: Determine meta device
        Step 2: Build model (meta or real device)
        Step 3-12: apply_model_infrastructure (PP, PEFT, QAT, ShardingPlan,
        activation checkpoint, FSDP2, load, layer compile)
        """
        from contextlib import nullcontext
        from transformers.modeling_utils import ContextManagers
        try:
            from transformers.modeling_utils import no_init_weights
        except ImportError:
            from transformers.initialization import no_init_weights
        from hyper_models.components.utils.model_utils import init_empty_weights

        # Step 1: Determine meta device
        from hyper_models.components.distributed.init_utils import (  # pylint: disable=import-outside-toplevel
            get_world_size_safe,
        )
        is_meta_device = (
            get_world_size_safe() > 1 or not is_hf_model
        ) and kwargs.get("quantization_config") is None

        init_ctx = (
            ContextManagers([no_init_weights(), init_empty_weights()])
            if is_meta_device
            else nullcontext()
        )

        # Step 2: Build model
        with init_ctx:
            is_custom_model, model = _init_model(
                cls,
                pretrained_model_name_or_path,
                hf_config,
                attn_implementation,
                torch_dtype,
                is_hf_model,
                *model_args,
                backend=backend,
                **kwargs,
            )

        # Step 3-12: Apply infrastructure
        model = apply_model_infrastructure(
            model,
            mesh=mesh,
            sharding_planner=sharding_planner,
            fsdp2_manager=fsdp2_manager,
            autopipeline=autopipeline,
            peft_config=peft_config,
            qat_config=qat_config,
            fp8_config=fp8_config,
            freeze_config=freeze_config,
            compile_config=compile_config,
            is_meta_device=is_meta_device,
            is_hf_model=is_hf_model,
            device=_current_device(),
            load_base_model=load_base_model,
            pretrained_path=pretrained_model_name_or_path,
            validate_placement=validate_placement,
            activation_checkpoint=activation_checkpoint,
            activation_swap=activation_swap,
        )

        model.train()
        return model


class HyperAutoModelForCausalLM(_BaseHyperAutoModelClass, AutoModelForCausalLM):
    """Hyper-Parallel CausalLM — equivalent to AutoModelForCausalLM.

    Following design doc 01 §6.1.
    """


class HyperAutoModelForImageTextToText(_BaseHyperAutoModelClass, AutoModelForImageTextToText):
    """Hyper-Parallel VLM — equivalent to AutoModelForImageTextToText."""


class HyperAutoModelForSequenceClassification(_BaseHyperAutoModelClass, AutoModelForSequenceClassification):
    """Hyper-Parallel SequenceClassification."""
