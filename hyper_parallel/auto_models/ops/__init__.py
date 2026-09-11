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
"""Public high-performance function interfaces."""

from importlib import import_module as _import_module  # pylint: disable=invalid-name


# Some operators require optional extension packages. Lazy public exports let
# independent torch_npu operators remain usable without importing every custom
# extension in this package.
_LAZY_EXPORTS = {
    "aggregate_hidden": ".aggregate_hidden",
    "apply_rotary_pos_emb": ".rotary_embedding",
    "apply_rotary_pos_emb_interleave": ".rotary_embedding",
    "attention_rescale": ".attention_rescale",
    "aux_loss_auto_scale": ".aux_loss",
    "dsa_indexer": ".dsa_indexer",
    "dsa_kl_loss": ".dsa_kl_loss",
    "dsa_sparse_attention": ".dsa_sparse_attention",
    "dsa_sparse_attention_rescale": ".dsa_sparse_attention_rescale",
    "grouped_matmul": ".grouped_matmul",
    "mhc_post": ".mhc_post",
    "mhc_pre": ".mhc_pre",
    "moe_token_permute": ".moe_token_permute",
    "moe_token_unpermute": ".moe_token_unpermute",
    "npu_fusion_attention_forward": ".npu_fusion_attention",
    "rms_norm": ".rms_norm",
    "set_aux_loss_scale": ".aux_loss",
    "sink_attention": ".sink_attention",
    "sinkhorn": ".sinkhorn",
    "swiglu": ".swiglu",
}


def __getattr__(name):  # pylint: disable=invalid-name
    """Lazily import a public high-performance function."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = _import_module(_LAZY_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():  # pylint: disable=invalid-name
    """Include lazy public exports in ``dir()``."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))

__all__ = [
    "aggregate_hidden",
    "apply_rotary_pos_emb",
    "apply_rotary_pos_emb_interleave",
    "attention_rescale",
    "aux_loss_auto_scale",
    "dsa_indexer",
    "dsa_kl_loss",
    "dsa_sparse_attention",
    "dsa_sparse_attention_rescale",
    "grouped_matmul",
    "mhc_post",
    "mhc_pre",
    "moe_token_permute",
    "moe_token_unpermute",
    "npu_fusion_attention_forward",
    "rms_norm",
    "set_aux_loss_scale",
    "sink_attention",
    "sinkhorn",
    "swiglu",
]
