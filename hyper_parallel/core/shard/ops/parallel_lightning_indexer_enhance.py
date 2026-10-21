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
"""Distributed implementation for the lightning_indexer_enhance operator.

The enhanced kernel keeps the topology of ``lightning_indexer`` bit for bit: the same three
inputs, the same output layout, and the same context-parallel treatment. It is therefore
registered as a rename-only subclass of :class:`LightningIndexerDistributedOp` -- inheriting
``preprocess`` / ``infer_layout`` / ``get_expand_impl`` on purpose.

Inheriting (instead of re-deriving) is what keeps the DSA CP fold intact: with
``dsa_cp_fold_enabled()`` the parent calls the kernel once per folded block, each on its own
causal key prefix. A wrapper that skipped this dispatch would call the kernel once with the
folded full-length key and silently select top-K over the wrong interval.
"""
from .parallel_lightning_indexer import LightningIndexerDistributedOp


class LightningIndexerEnhanceDistributedOp(LightningIndexerDistributedOp):
    """Distributed operator for ``npu_lightning_indexer_enhance``.

    Identical layout inference and CP folding to ``LightningIndexerDistributedOp``; only the
    registered op name differs (the YAML key), which is what routes DTensor calls here.
    """

    #: Attributes the enhanced kernel adds on top of the ``lightning_indexer`` signature.
    _EXTRA_KWARGS = ("sparse_block_size", "sparse_block_mode")

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """Peel the enhanced-kernel attributes before the shared normalizer runs.

        ``sparse_block_size`` / ``sparse_block_mode`` are absent from
        ``_normalize_lightning_indexer_args``, and ``fold_lightning_indexer`` forwards whatever
        comes back from here straight to the kernel, so they are re-attached afterwards.

        Args:
            args: Positional arguments of the call.
            kwargs: Keyword arguments of the call.

        Returns:
            tuple: (local_args, local_kwargs, cache_values) as the parent returns them.
        """
        kwargs = dict(kwargs)
        extra = {name: kwargs.pop(name) for name in self._EXTRA_KWARGS if name in kwargs}
        local_args, local_kwargs, cache_values = super().preprocess(args, kwargs)
        local_kwargs.update(extra)
        return local_args, local_kwargs, cache_values

