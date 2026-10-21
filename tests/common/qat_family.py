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
"""Shared configuration fixtures for QAT Family integration tests."""

from torch import nn
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Experts

from hyper_parallel.distributed.mesh import DistributedSetup
from hyper_parallel.models.replacement import apply_module_replacements, compile_module_replacements
from hyper_parallel.trainer.config.parallelism import normalize_distributed_setup_overrides
from hyper_parallel.trainer.config.resolver import resolve_config
from hyper_parallel.trainer.config import TrainerConfig
from tests.common.grouped_experts_family import family_config


def qat_config(*, fake: bool = False, block_size: int = 32, weight_format: str = "mxfp4") -> TrainerConfig:
    """Resolve a named scheme through the actual training configuration parser.

    Args:
        fake: Select the existing fake W4A8 strategy.
        block_size: Quantization block size.
        weight_format: Weight quantization format.
    """
    raw = family_config(low_precision_dtype_scheme="experts").to_dict()
    raw["training"]["low_precision"] = {
        "dtype_schemes": {"experts": {
            "weight_format": weight_format, "act_format": "mxfp8",
            "is_fake_quantize": fake, "block_size": block_size}},
    }
    return resolve_config(raw)


def install_family(model: nn.Module, config: TrainerConfig) -> nn.Module:
    """Execute the existing resolver's policy injection and replacement engine.

    Args:
        model: Model containing the configured replacement targets.
        config: Resolved training configuration.
    """
    setup = DistributedSetup(low_precision_config=config.training.low_precision)
    normalize_distributed_setup_overrides(setup, config)
    rules = setup.module_replacements
    return apply_module_replacements(model, compile_module_replacements(model, rules), weights_mapping=[])[0]


def deepseek_experts_model() -> nn.Module:
    """Build the native packed experts used by V4.1, retaining its clamp semantics."""
    model = nn.Module()
    config = DeepseekV4Config(hidden_size=128, intermediate_size=256, num_local_experts=4,
                            num_experts_per_tok=2, swiglu_limit=10.0)
    model.experts = DeepseekV4Experts(config)
    for parameter in model.parameters():
        nn.init.normal_(parameter, std=0.02)
    return model


def deepseek_qat_config(*, fake: bool = False, weight_format: str = "mxfp8",
                       fused: bool = False) -> TrainerConfig:
    """Resolve the ordinary Family YAML fields for native V4.1 experts.

    Args:
        fake: Select the fake W4A8 strategy.
        weight_format: Weight format in the named scheme.
        fused: Request fusion, which remains unsupported for this activation.
    """
    raw = qat_config(fake=fake, weight_format=weight_format).to_dict()
    raw["model"] = {"_target_": "tests.common.qat_family.deepseek_experts_model"}
    entry = raw["plan_overrides"][0]
    entry["match"] = "experts"
    entry["module_type"] = "transformers.models.deepseek_v4.modeling_deepseek_v4.DeepseekV4Experts"
    entry["replace_module"]["fused"] = fused
    return resolve_config(raw)
