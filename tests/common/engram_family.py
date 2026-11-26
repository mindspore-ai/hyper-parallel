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
"""Small Engram assets shared by CPU layout tests and distributed workers."""

from types import SimpleNamespace


def small_config(rows: int = 16) -> SimpleNamespace:
    """Provide two hash columns and a padded table without external model assets."""
    return SimpleNamespace(
        assets={"layer_ids": [0], "max_ngram_size": 2, "num_heads": 2,
                "token_map": list(range(6)), "pad_token_id": 0,
                "primes": [[[5, 7]]], "multipliers": [[1, 3]]},
        layer_id=0, hidden_size=4, hc_mult=2, embedding_dim=2,
        eps=1e-6, clamp_value=1e-6, padded_num_embeddings=rows,
    )
