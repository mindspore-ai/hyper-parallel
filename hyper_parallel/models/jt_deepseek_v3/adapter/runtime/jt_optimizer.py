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
"""Post-update QK clipping for JT MLA projections."""

import torch
import torch.distributed as dist

from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import JTDeepseekV3MLAAttention


@torch.no_grad()
def clip_qk(model: torch.nn.Module, threshold: float) -> dict[str, torch.Tensor]:
    """Clip coupled query/key projections and return detached global maxima.

    Args:
        model: JT model with MLA statistics.
        threshold: Positive clipping threshold from the optimizer adapter configuration.
    """
    metrics = {}
    for name, module in model.named_modules():
        if not isinstance(module, JTDeepseekV3MLAAttention):
            continue
        maximum = module.max_logits_val
        observed = maximum.detach().amax().clone()
        dist.all_reduce(observed, op=dist.ReduceOp.MAX, group=model.loss_group)
        metrics[f"optimizer/qkclip_maxlogits/{name}"] = observed
        scale = threshold / maximum.clamp_min(threshold)
        query = module.q_b_proj.weight.view(
            module.num_heads, module.qk_nope_head_dim + module.qk_rope_head_dim, -1)
        query[:, :module.qk_nope_head_dim].mul_(scale.sqrt()[:, None, None])
        query[:, module.qk_nope_head_dim:].mul_(scale[:, None, None])
        key_value = module.kv_b_proj.weight.view(module.num_heads, module.qk_nope_head_dim + module.v_head_dim, -1)
        key_value[:, :module.qk_nope_head_dim].mul_(scale.sqrt()[:, None, None])
        maximum.zero_()
    if metrics:
        metrics["optimizer/qkclip_maxlogits"] = torch.stack(list(metrics.values())).amax()
    return metrics
