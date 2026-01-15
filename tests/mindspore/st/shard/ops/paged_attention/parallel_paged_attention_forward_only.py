# Copyright 2025 Huawei Technologies Co., Ltd
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
# ==========================================================================
"""parallel paged attention forward only"""

import math
import numpy as np
import mindspore as ms
import mindspore.communication.management as D
from mindspore import Tensor, nn, context, ops
from hyper_parallel import Layout, DTensor, shard, DistributedCustomOp
from tests.mindspore.st.shard.utils import local_to_global, global_to_local
import ms_custom_ops


def setup_module():
    """setup module"""
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="Ascend")


class PagedAttentionNet(nn.Cell):
    """Minimal network wrapper for paged_attention."""

    def __init__(self, q_head_num, kv_head_num, head_size, qk_scale):
        super().__init__()
        self.q_head_num = int(q_head_num)
        self.kv_head_num = int(kv_head_num)
        self.qk_scale = float(qk_scale)
        self.mask_type = 0
        self.batch_run_status_enable = False
        self.quant_type = 0
        self.out_data_type = -1
        self.has_quant_offset = False
        self.compress_type = 0
        self.calc_type = 0
        self.scale_type = 0
        self.input_layout = 0
        self.mla_v_dim = 0
        self.input_format = 0
        self._is_pynative = context.get_context("mode") == context.PYNATIVE_MODE
        # Wrap the custom op with DistributedCustomOp for automatic DTensor handling
        # Pass self as cell so DistributedCustomOp can read _current_input_layouts set by shard hook
        self.paged_attention = DistributedCustomOp(ms_custom_ops.paged_attention, cell=self)

    def construct(self, query, key_cache, value_cache, block_tables, context_lens, q_seq_lens):
        print(f"[PagedAttentionNet.construct] query type={type(query).__name__}, "
              f"is_dtensor={isinstance(query, DTensor)}")
        print(f"[PagedAttentionNet.construct] key_cache type={type(key_cache).__name__}, "
              f"is_dtensor={isinstance(key_cache, DTensor)}")
        print(f"[PagedAttentionNet.construct] value_cache type={type(value_cache).__name__}, "
              f"is_dtensor={isinstance(value_cache, DTensor)}")
        print(f"[PagedAttentionNet.construct] block_tables type={type(block_tables).__name__}, "
              f"is_dtensor={isinstance(block_tables, DTensor)}")

        if self._is_pynative:
            context_cpu = context_lens.move_to("CPU")
            q_seq_cpu = q_seq_lens.move_to("CPU")
        else:
            context_cpu = ops.move_to(context_lens, "CPU")
            q_seq_cpu = ops.move_to(q_seq_lens, "CPU")

        # DistributedCustomOp automatically detects DTensor and routes through infer_layout
        return self.paged_attention(
            query,
            key_cache,
            value_cache,
            block_tables,
            context_cpu,
            attn_mask=None,
            batch_run_status=None,
            k_descale=None,
            k_offset=None,
            v_descale=None,
            v_offset=None,
            razor_offset=None,
            p_scale=None,
            log_n=None,
            q_seq_lens=q_seq_cpu,
            q_head_num=self.q_head_num,
            qk_scale=self.qk_scale,
            kv_head_num=self.kv_head_num,
            mask_type=self.mask_type,
            batch_run_status_enable=self.batch_run_status_enable,
            quant_type=self.quant_type,
            out_data_type=self.out_data_type,
            has_quant_offset=self.has_quant_offset,
            compress_type=self.compress_type,
            calc_type=self.calc_type,
            scale_type=self.scale_type,
            input_layout=self.input_layout,
            mla_v_dim=self.mla_v_dim,
            input_format=self.input_format,
        )


def _build_inputs(batch_size, q_head_num, kv_head_num, head_size, block_size, num_blocks,
                  q_seq_lens_np, context_lens_np):
    num_tokens = int(np.sum(q_seq_lens_np))
    max_context_len = int(np.max(context_lens_np))
    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size

    rng = np.random.default_rng(0)
    query = Tensor(rng.standard_normal([num_tokens, q_head_num, head_size]).astype(np.float16))
    key_cache = Tensor(rng.standard_normal([num_blocks, block_size, kv_head_num, head_size]).astype(np.float16))
    value_cache = Tensor(rng.standard_normal([num_blocks, block_size, kv_head_num, head_size]).astype(np.float16))
    block_tables = Tensor(rng.integers(0, num_blocks, size=[batch_size, max_num_blocks_per_seq]).astype(np.int32))
    context_lens = Tensor(context_lens_np, dtype=ms.int32)
    q_seq_lens = Tensor(q_seq_lens_np, dtype=ms.int32)

    return query, key_cache, value_cache, block_tables, context_lens, q_seq_lens


def run_standalone_forward(query, key_cache, value_cache, block_tables, context_lens, q_seq_lens):
    """Run standalone forward."""
    head_size = query.shape[-1]
    q_head_num = query.shape[1]
    kv_head_num = key_cache.shape[2]
    qk_scale = 1.0 / math.sqrt(head_size)
    net = PagedAttentionNet(q_head_num, kv_head_num, head_size, qk_scale)
    return net(query, key_cache, value_cache, block_tables, context_lens, q_seq_lens)


def run_parallel_forward(local_query, local_key_cache, local_value_cache, local_block_tables,
                         local_context_lens, local_q_seq_lens, q_layout, kv_layout, block_layout,
                         context_layout, q_seq_layout):
    """Run parallel forward."""
    head_size = local_query.shape[-1]
    q_head_num = local_query.shape[1]
    kv_head_num = local_key_cache.shape[2]
    qk_scale = 1.0 / math.sqrt(head_size)

    model = PagedAttentionNet(q_head_num, kv_head_num, head_size, qk_scale)

    # Use shard to specify input layouts - this preserves DTensor through MindSpore hooks
    model_stra = {
        "forward": {
            "input": (q_layout, kv_layout, kv_layout, block_layout, context_layout, q_seq_layout)
        }
    }
    shard(model, model_stra)
    print(f"[run_parallel_forward] after shard, model.in_layout={getattr(model, 'in_layout', None)}")
    print(f"[run_parallel_forward] calling model...")

    output = model(local_query, local_key_cache, local_value_cache, local_block_tables,
                   local_context_lens, local_q_seq_lens)
    if not isinstance(output, DTensor):
        output = DTensor.from_local(output, q_layout)
    if isinstance(output, DTensor):
        output = local_to_global(output)
    return output


def base_case(dp, mp):
    """Base case for paged attention standalone + parallel forward."""
    D.init()

    batch_size = 4
    q_head_num = 8
    kv_head_num = 8
    head_size = 64
    block_size = 16
    num_blocks = 64
    q_seq_lens_np = np.array([1, 1, 1, 1], dtype=np.int32)
    context_lens_np = np.array([128, 96, 64, 32], dtype=np.int32)

    query, key_cache, value_cache, block_tables, context_lens, q_seq_lens = _build_inputs(
        batch_size,
        q_head_num,
        kv_head_num,
        head_size,
        block_size,
        num_blocks,
        q_seq_lens_np,
        context_lens_np,
    )

    standalone_out = run_standalone_forward(
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        q_seq_lens,
    )

    layout = Layout((dp, mp), ("dp", "mp"))
    q_layout = layout("dp", "mp", "None")
    kv_layout = layout("None", "None", "mp", "None")
    block_layout = layout("dp", "None")
    context_layout = layout("dp")
    q_seq_layout = layout("dp")

    local_query = global_to_local(query, q_layout).to_local()
    local_key_cache = global_to_local(key_cache, kv_layout).to_local()
    local_value_cache = global_to_local(value_cache, kv_layout).to_local()
    local_block_tables = global_to_local(block_tables, block_layout).to_local()
    local_context_lens = global_to_local(context_lens, context_layout).to_local()
    local_q_seq_lens = global_to_local(q_seq_lens, q_seq_layout).to_local()

    parallel_out = run_parallel_forward(
        local_query,
        local_key_cache,
        local_value_cache,
        local_block_tables,
        local_context_lens,
        local_q_seq_lens,
        q_layout,
        kv_layout,
        block_layout,
        context_layout,
        q_seq_layout,
    )

    print(f"[paged_attention] standalone_out.shape={standalone_out.shape}, parallel_out.shape={parallel_out.shape}")
    assert standalone_out.shape == parallel_out.shape
    standalone_np = standalone_out.asnumpy()
    parallel_np = parallel_out.asnumpy()
    print(f"[paged_attention] standalone_out sample={standalone_np.flatten()}")
    print(f"[paged_attention] parallel_out sample={parallel_np.flatten()}")
    print(f"[paged_attention] max_abs_diff={np.max(np.abs(standalone_np - parallel_np))}")
    assert np.allclose(standalone_np, parallel_np, 1e-3, 1e-3)


def test_parallel_paged_attention_forward_only():
    """
    Feature: PagedAttention forward.
    Description: Standalone + parallel forward for paged_attention in shard test.
    Expectation: Run success.
    """
    base_case(dp=4, mp=2)
