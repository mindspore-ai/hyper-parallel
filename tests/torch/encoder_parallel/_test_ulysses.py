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
"""Gloo worker comparing DeepSeek's padded Ulysses attention with native attention."""

import torch
import torch.distributed as dist
from torch.nn import functional

from hyper_parallel.distributed.encoder_parallel.attention import EncoderAttentionContext, activate_encoder_attention
from hyper_parallel.distributed.encoder_parallel.contracts import EncoderSegment, EncoderWorkItem
from hyper_parallel.models.deepseek_v41.components.vision_transformer import DeepseekV41EncoderParallelAttention
from hyper_parallel.models.deepseek_v41.vision import DeepseekV41VisionAttention


class Mesh:
    """Expose the three mesh operations used by the Ulysses executor."""

    def size(self) -> int:
        """Return the worker group size."""
        return dist.get_world_size()

    def get_local_rank(self) -> int:
        """Return this worker's group rank."""
        return dist.get_rank()

    def get_group(self) -> dist.ProcessGroup:
        """Return the distributed process group."""
        return dist.group.WORLD


def test_odd_patch_ulysses():
    """Feature: Ulysses with an odd image patch count.

    Description: Shard five valid patches with a sixth padded patch over two ranks.
    Expectation: Forward values and input gradients match full attention.
    """
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    torch.manual_seed(123)
    native = DeepseekV41VisionAttention(16, 4)
    replacement = DeepseekV41EncoderParallelAttention.from_source(native, module_fqn="vision.attn")
    full_hidden = torch.randn(6, 16, requires_grad=True)
    local_hidden = full_hidden.detach()[rank * 3:(rank + 1) * 3].clone().requires_grad_(True)
    cosine = torch.ones(3, 1, 2)
    sine = torch.zeros_like(cosine)
    item = EncoderWorkItem(0, 0, 5, 1, 0, 2, 0, 5, {})
    segment = EncoderSegment(item, rank * 3, 2 if rank else 3, 3, 0)
    context = EncoderAttentionContext((segment,), Mesh(),
                                      {name: 4 for name in ("query", "key", "value")},
                                      frozenset({"vision.attn"}))
    with activate_encoder_attention(context):
        out = replacement(local_hidden, cosine, sine)
    context.assert_exactly_once()
    out[:segment.local_valid_length].sum().backward()
    query, key, value = (tensor.view(6, 4, 4) for tensor in native.wqkv(full_hidden).chunk(3, dim=-1))
    mask = (torch.arange(6) < 5).view(1, 1, 6)
    attended = functional.scaled_dot_product_attention(  # pylint: disable=not-callable
        query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1),
        attn_mask=mask, is_causal=False,
    ).transpose(0, 1)
    ref = native.wo(attended.reshape(6, 16))
    ref[:5].sum().backward()
    assert torch.allclose(
        out[:segment.local_valid_length], ref[rank * 3:rank * 3 + segment.local_valid_length], atol=1e-5,
    ), f"output mismatch: rank={rank}"
    assert torch.allclose(local_hidden.grad, full_hidden.grad[rank * 3:(rank + 1) * 3], atol=1e-5), (
        f"gradient mismatch: rank={rank}; local={local_hidden.grad}; "
        f"reference={full_hidden.grad[rank * 3:(rank + 1) * 3]}"
    )
    if rank == 0:
        print("Ulysses odd-patch forward/backward passed", flush=True)
    dist.destroy_process_group()
