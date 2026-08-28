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
"""Qwen-Image DiT model registration."""

from hyper_parallel.models.spec.model_spec import ModelSpec
from hyper_parallel.models.spec.registry import register_spec
from .model import QwenImageDiT
from .parallelize import parallelize_qwen_image_dit


def build_model_fn(cfg):
    """Factory: build QwenImageDiT from YAML config."""
    init_device = getattr(cfg.train, "init_device", "npu")
    return QwenImageDiT(cfg.model, init_device=init_device)


def parallelize_fn(model, mesh, cfg):
    """Apply FSDP2 (+ optional AC) to Qwen-Image DiT blocks.

    Delegates to ``parallelize.py`` which builds proper FSDP kwargs
    (mesh, reshard_after_forward, comm_fusion, mp_policy, shard-placement
    overrides) following the framework convention.
    """
    return parallelize_qwen_image_dit(model, mesh, cfg)


register_spec("qwen_image_dit", ModelSpec(
    name="qwen_image_dit",
    build_model_fn=build_model_fn,
    parallelize_fn=parallelize_fn,
))
