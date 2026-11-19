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
"""Real-NPU parity for separate native mHC Pre/Post with delayed coefficients."""

import copy
import json

import torch
import torch_npu
from torch import nn

from hyper_parallel.models.deepseek_v41.adapter.distributed.fused_mhc import DeepseekV41FusedMhc
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41PipelinedHyperConnection


class _RmsNorm(nn.Module):
    """Independent coefficient RMS normalization."""

    variance_epsilon = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the flattened stream vector."""
        return x / torch.sqrt(x.square().mean(-1, keepdim=True) + self.variance_epsilon)


def _source(hidden: int) -> DeepseekV41PipelinedHyperConnection:
    """Construct the checkpoint-compatible source on CPU before device transfer."""
    module = nn.Module()
    module.input_norm = _RmsNorm()
    module.fn = nn.Parameter(torch.randn(24, 4 * hidden) * .01)
    module.base = nn.Parameter(torch.randn(24) * .01)
    module.scale = nn.Parameter(torch.tensor([.7, 1.2, .8]))
    module.hc_mult, module.hc_sinkhorn_iters, module.hc_eps = 4, 20, 1e-6
    return DeepseekV41PipelinedHyperConnection(module)


def test_pre_post_outputs_and_gradients():
    """Verify supported widths and the native 5120-wide training configuration."""
    torch_npu.npu.set_device(0)
    records = []
    for hidden, dtype in ((4096, torch.bfloat16), (5120, torch.bfloat16),
                          (7168, torch.bfloat16), (5120, torch.float16)):
        torch.manual_seed(42)
        reference = _source(hidden).to('npu')
        candidate = DeepseekV41FusedMhc(module=copy.deepcopy(reference))
        shapes = ((1, 128, hidden), (1, 128, 4, hidden), (1, 128, 4),
                  (1, 128, 4), (1, 128, 4, 4), (hidden,))
        inputs = [torch.randn(shape, device='npu', dtype=dtype if i < 2 or i == 5 else torch.float32)
                  .requires_grad_() for i, shape in enumerate(shapes)]
        oracle_inputs = [t.detach().clone().requires_grad_() for t in inputs]
        outputs = candidate.advance(*inputs)
        output, residual, old_pre, old_post, old_comb, weight = oracle_inputs
        mixed = (torch.einsum('bsij,bsih->bsjh', old_comb, residual.float())
                 + old_post.unsqueeze(-1) * output.float().unsqueeze(-2)).to(dtype)
        pre, post, combine = reference(mixed)
        collapsed = (old_pre.unsqueeze(-1) * mixed.float()).sum(2).to(dtype).float()
        norm = collapsed * torch.rsqrt(collapsed.square().mean(-1, keepdim=True) + candidate.norm_eps)
        expected = mixed, pre, post, combine, norm.to(dtype) * weight
        upstream = [torch.randn_like(t) for t in outputs]
        actual_grad = torch.autograd.grad(outputs, inputs + list(candidate.parameters()), upstream)
        oracle_grad = torch.autograd.grad(expected, oracle_inputs + list(reference.parameters()), upstream)
        errors = []
        for actual, oracle in zip(outputs + actual_grad, expected + oracle_grad):
            error = ((actual.float() - oracle.float()).norm() / oracle.float().norm().clamp_min(1e-12)).item()
            errors.append(error)
        torch.npu.synchronize()
        record = {'hidden': hidden, 'dtype': str(dtype), 'relative_l2': errors, 'max_relative_l2': max(errors)}
        print(json.dumps(record), flush=True)
        assert max(errors) < .005, f'Pre/Post parity mismatch: observed={errors}, allowed=0.005'
        records.append(record)
    print('FUSED_MHC_PRE_POST_PARITY_PASSED', json.dumps(records), flush=True)
