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
"""Extensible FakeTensor operator support for dry-run.

The built-in kernels cover Ascend operators used by DeepSeek-V4.1. Users can
register additional shape-only kernels with :func:`register_fake_op`.
Registration takes effect immediately when the defining module is imported.

Every kernel below only derives output *shapes*; gradients mirror the
differentiated forward inputs.
"""

import logging
from collections.abc import Callable
from typing import Any

import torch  # pylint: disable=forbidden-backend-import

logger = logging.getLogger(__name__)


class DryRunOpRegistry:
    """Register and load shape-only implementations for custom operators."""

    def __init__(self) -> None:
        """Create an empty registry tied to the current process."""
        self._registered: set[str] = set()

    def register_fake(
            self,
            op_name: str,
            kernel: Callable[..., Any] | None = None,
            *,
            allow_missing: bool = False,
    ) -> Callable[..., Any]:
        """Register a FakeTensor kernel.

        Args:
            op_name: Operator name such as ``custom::my_forward``.
            kernel: Shape-only implementation; omit it when using this method
                as a decorator.
            allow_missing: Keep optional vendor kernels non-fatal when the
                operator is unavailable in the current backend build.

        Returns:
            The registered kernel or a decorator when ``kernel`` is omitted.
        """
        if not isinstance(op_name, str) or "::" not in op_name:
            raise ValueError("op_name must use the 'namespace::operator' form")

        def decorator(function: Callable[..., Any]) -> Callable[..., Any]:
            if not callable(function):
                raise TypeError("FakeTensor kernel must be callable")
            if op_name in self._registered:
                return function
            try:
                torch.library.register_fake(op_name)(function)
            except RuntimeError as error:
                if allow_missing and "does not exist" in str(error):
                    logger.debug("Optional operator %s is unavailable: %s", op_name, error)
                    return function
                raise RuntimeError(
                    f"Unable to register dry-run kernel for {op_name!r}: {error}"
                ) from error
            self._registered.add(op_name)
            logger.info("Registered Dry-run FakeTensor kernel for %s", op_name)
            return function

        return decorator if kernel is None else decorator(kernel)

    @property
    def registered_ops(self) -> tuple[str, ...]:
        """Return registered operator names for diagnostics."""
        return tuple(sorted(self._registered))


_REGISTRY = DryRunOpRegistry()
_BUILTINS_REGISTERED = False


def get_dryrun_op_registry() -> DryRunOpRegistry:
    """Return the process-wide dry-run operator registry."""
    return _REGISTRY


def register_fake_op(
        op_name: str,
        kernel: Callable[..., Any] | None = None,
        *,
        allow_missing: bool = False,
) -> Callable[..., Any]:
    """Register a custom FakeTensor kernel on the global dry-run registry."""
    return _REGISTRY.register_fake(op_name, kernel, allow_missing=allow_missing)


def _register(op_name, kernel):
    """Register one Meta kernel if the vendor operator is present."""
    op = getattr(getattr(torch.ops, "custom", None), op_name, None)
    if op is None:
        return False
    full_name = f"custom::{op_name}"
    try:
        _REGISTRY.register_fake(full_name, kernel)
    except RuntimeError as error:  # already registered by the vendor
        logger.debug("Meta kernel for %s already present: %s", full_name, error)
        return False
    return True


def register_builtin_dryrun_ops() -> None:
    """Register Meta kernels for vendor forward/backward ops missing them."""
    global _BUILTINS_REGISTERED  # pylint: disable=global-statement
    if _BUILTINS_REGISTERED:
        return
    _BUILTINS_REGISTERED = True
    if getattr(torch.ops, "custom", None) is None:
        return

    def _sparse_attention_grad(
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

    def _indexer_kl_loss_grad(
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

    def _aggregate_hidden_grad(grad_output, input, weight, mask=None):
        """Gradients mirror the differentiated forward inputs."""
        del grad_output, mask
        return torch.empty_like(input), torch.empty_like(weight)

    def _mhc_pre_grad(
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

    def _flash_attention_score_enhance(
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
            ("npu_sparse_flash_attention_grad_enhance", _sparse_attention_grad),
            ("npu_sparse_lightning_indexer_grad_kl_loss_enhance", _indexer_kl_loss_grad),
            ("npu_aggregate_hidden_grad", _aggregate_hidden_grad),
            ("npu_manifold_constrained_hyper_connection_pre_grad", _mhc_pre_grad),
            ("npu_flash_attention_score_enhance", _flash_attention_score_enhance),
    ):
        _register(name, kernel)


def unsupported_operator_hint(error: BaseException) -> str | None:
    """Return an actionable extension hint for an unsupported FakeTensor op."""
    message = str(error)
    if "UnsupportedOperatorException" not in message and "unsupported operator" not in message.lower():
        return None
    return (
        f"{message}\n"
        "Dry-run could not derive this custom operator under FakeTensor.\n"
        "Register a shape-only kernel before running dry-run, for example:\n"
        "  from hyper_parallel.tools.dryrun import register_fake_op\n"
        "  @register_fake_op(\"custom::my_operator\")\n"
        "  def my_operator_fake(...):\n"
        "      return torch.empty_like(input_tensor)\n"
        "Import the defining module from the model integration code so registration runs.\n"
        "Registered ops: "
        f"{', '.join(_REGISTRY.registered_ops) or '<none>'}."
    )


__all__ = [
    "DryRunOpRegistry",
    "get_dryrun_op_registry",
    "register_builtin_dryrun_ops",
    "register_fake_op",
    "unsupported_operator_hint",
]
