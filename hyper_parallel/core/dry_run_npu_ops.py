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
"""Meta kernels for Ascend vendor operators used by the DeepSeek-V4.1 stack.

The Omni Ascend custom operators ship Meta kernels for their forward passes but
not for every backward pass. Under ``FakeTensorMode`` the missing ones abort the
Dry-run with ``UnsupportedOperatorException``, which forces the model onto its
reference (pure-Torch) attention path. Registering shape-only Meta kernels lets
the Dry-run execute the same fused kernels the real training run uses, so the
activation lifecycle it measures is the production one.

Every kernel below only derives output *shapes*; gradients mirror the
differentiated forward inputs.
"""

import logging

import torch

logger = logging.getLogger(__name__)

_REGISTERED = False


def _register(op_name, kernel):
    """Register one Meta kernel if the vendor operator is present."""
    op = getattr(getattr(torch.ops, "custom", None), op_name, None)
    if op is None:
        return False
    try:
        torch.library.register_fake(f"custom::{op_name}")(kernel)
    except RuntimeError as error:  # already registered by the vendor
        logger.debug("Meta kernel for %s already present: %s", op_name, error)
        return False
    logger.info("Registered Dry-run Meta kernel for custom::%s", op_name)
    return True


def register_npu_dry_run_kernels() -> None:
    """Register Meta kernels for vendor forward/backward ops missing them."""
    global _REGISTERED  # pylint: disable=global-statement
    if _REGISTERED:
        return
    _REGISTERED = True
    if getattr(torch.ops, "custom", None) is None:
        return

    def sparse_attention_grad(
            query, key, value, sparse_indices, d_out, out, softmax_max, softmax_sum,
            scale_value, sparse_block_size, query_rope=None, key_rope=None,
            actual_seq_qlen=None, actual_seq_kvlen=None, layout="BSND",
            sparse_mode=3, pre_tokens=9223372036854775807,
            next_tokens=9223372036854775807, attention_mode=0, deterministic=False,
    ):
        """Gradients of query/key/value/rope mirror the forward inputs."""
        del (sparse_indices, d_out, out, softmax_max, softmax_sum, scale_value,
             sparse_block_size, actual_seq_qlen, actual_seq_kvlen, layout,
             sparse_mode, pre_tokens, next_tokens, attention_mode, deterministic)
        grad_query_rope = (
            torch.empty_like(query_rope) if query_rope is not None else torch.empty_like(query)
        )
        grad_key_rope = (
            torch.empty_like(key_rope) if key_rope is not None else torch.empty_like(key)
        )
        return (
            torch.empty_like(query),
            torch.empty_like(key),
            torch.empty_like(value),
            grad_query_rope,
            grad_key_rope,
        )

    def indexer_kl_loss_grad(
            query, key, query_index, key_index, weights, sparse_indices,
            softmax_max, softmax_sum, scale_value, query_rope=None, key_rope=None,
            actual_seq_qlen=None, actual_seq_klen=None, layout="BSND",
            sparse_mode=3, pre_tokens=2147483647, next_tokens=2147483647,
            deterministic=False, sparse_block_size=1,
    ):
        """Indexer gradients mirror their weights; the inner loss is scalar."""
        del (query, key, sparse_indices, softmax_max, softmax_sum, scale_value,
             query_rope, key_rope, actual_seq_qlen, actual_seq_klen, layout,
             sparse_mode, pre_tokens, next_tokens, deterministic, sparse_block_size)
        return (
            torch.empty_like(query_index),
            torch.empty_like(key_index),
            torch.empty_like(weights),
            torch.empty((), dtype=torch.float32, device=weights.device),
        )

    def aggregate_hidden_grad(grad_output, input, weight, mask=None):
        """Gradients mirror the differentiated forward inputs."""
        del grad_output, mask
        return torch.empty_like(input), torch.empty_like(weight)

    def mhc_pre_grad(
            x, phi, alpha, h_in_grad, h_post_grad, h_comb_before_grad, inv_rms,
            mm_res, h_pre, h_post, gamma=None, hc_eps=1.0e-6, grad_x_post=None,
    ):
        """Gradients follow (x, phi, alpha, beta, gamma) of the forward."""
        del (h_in_grad, h_post_grad, h_comb_before_grad, inv_rms, mm_res, h_pre,
             h_post, hc_eps, grad_x_post)
        grad_gamma = (
            torch.empty_like(gamma) if gamma is not None else torch.empty_like(alpha)
        )
        return (
            torch.empty_like(x),
            torch.empty_like(phi),
            torch.empty_like(alpha),
            torch.empty_like(alpha),
            grad_gamma,
        )

    def flash_attention_score_enhance(
            query, key, value, head_num, input_layout, pse=None,
            padding_mask=None, atten_mask=None, sink_tensor=None, query_rope=None,
            key_rope=None, scale=1.0, keep_prob=1.0, pre_tokens=2147483647,
            next_tokens=2147483647, inner_precise=0, prefix=None,
            actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0, sink_num=0,
            pse_type=1, softmaxOutLayout="", q_start_idx=None, kv_start_idx=None,
    ):
        """Attention output matches the query; LSE carries per-head scalars."""
        del (key, value, pse, padding_mask, atten_mask, sink_tensor, query_rope,
             key_rope, scale, keep_prob, pre_tokens, next_tokens, inner_precise,
             prefix, actual_seq_qlen, actual_seq_kvlen, sparse_mode, sink_num,
             pse_type, softmaxOutLayout, q_start_idx, kv_start_idx)
        tokens = query.shape[0] if input_layout == "TND" else query.shape[1]
        lse_shape = (head_num, int(tokens), 8)
        device = query.device
        return (
            torch.empty_like(query),
            torch.empty(lse_shape, dtype=torch.float32, device=device),
            torch.empty(lse_shape, dtype=torch.float32, device=device),
        )

    for name, kernel in (
            ("npu_sparse_flash_attention_grad_enhance", sparse_attention_grad),
            ("npu_sparse_lightning_indexer_grad_kl_loss_enhance", indexer_kl_loss_grad),
            ("npu_aggregate_hidden_grad", aggregate_hidden_grad),
            ("npu_manifold_constrained_hyper_connection_pre_grad", mhc_pre_grad),
            ("npu_flash_attention_score_enhance", flash_attention_score_enhance),
    ):
        _register(name, kernel)


__all__ = ["register_npu_dry_run_kernels"]
