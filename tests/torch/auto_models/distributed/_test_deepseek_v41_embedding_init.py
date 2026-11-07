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
"""Gloo checks for global padding-row ownership after V4.1 meta initialization."""
# pylint: disable=wrong-import-position

from __future__ import annotations

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torch import nn

pytest.importorskip("transformers.models.deepseek_v4.configuration_deepseek_v4")
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard, StridedShard
from hyper_parallel.models._transformers.model_builder import _initialize_model_weights, _move_model_to_device
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import (
    DeepseekV41ForCausalLM,
    _initialize_embedding_shard_safe,
)


def _meta_model(vocab_size: int, padding_idx: int | None, assets_path: str) -> DeepseekV41ForCausalLM:
    """Build a tiny native V4.1 model with all parameters on meta."""
    config = DeepseekV4Config(  # pylint: disable=unexpected-keyword-arg
        vocab_size=vocab_size, hidden_size=16, moe_intermediate_size=24, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=1, head_dim=8, q_lora_rank=8,
        num_experts_per_tok=2, n_routed_experts=4, n_shared_experts=1,
        layer_types=["sliding_attention"], mlp_layer_types=["moe"], swiglu_limit=1.5,
        sliding_window=4, o_groups=2, o_lora_rank=4, index_n_heads=2,
        index_head_dim=4, index_topk=2, partial_rotary_factor=0.5,
        pad_token_id=padding_idx, use_cache=False,
    )
    config.v41_vision_enabled = False
    config.v41_compress_ratios = [0]
    config.v41_kv_source_layer_ids = []
    config.v41_index_source_layer_ids = []
    config.v41_engram_assets_path = assets_path
    with torch.device("meta"):
        return DeepseekV41ForCausalLM(config)


@pytest.mark.parametrize("padding_idx", [None, 0, -1])
def test_plain_embedding_padding_rows(padding_idx: int | None) -> None:
    """Preserve ordinary Tensor and normalized negative-padding behavior.

    Args:
        padding_idx: Optional padding row, including a normalized negative index.
    """
    torch.manual_seed(2026)
    embedding = nn.Embedding(8, 16, padding_idx=padding_idx)
    with torch.no_grad():
        _initialize_embedding_shard_safe(embedding, 0.02)
    zero_rows = embedding.weight.eq(0).all(dim=1)
    expected = torch.zeros(8, dtype=torch.bool)
    if embedding.padding_idx is not None:
        expected[embedding.padding_idx] = True
    torch.testing.assert_close(zero_rows, expected)


def test_meta_embedding_padding_rows(tmp_path) -> None:
    """Validate full model initialization on regular, uneven and strided shards.

    Args:
        tmp_path: Per-worker temporary directory holding the minimal Engram assets.
    """
    dist.init_process_group(
        backend="gloo", init_method=os.environ.get("HP_EMBEDDING_TEST_INIT_METHOD", "env://"),
        rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]),
        timeout=timedelta(seconds=60),
    )
    try:
        assert dist.get_world_size() == 4
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
        layouts = (
            (Shard(0), Replicate()),
            (Replicate(), Shard(0)),
            (Shard(0), Shard(1)),
            (Shard(0, uneven_shard=True), Replicate()),
            (StridedShard(0, 2), Shard(0)),
        )
        assets = tmp_path / "engram.json"
        assets.write_text('{"layer_ids": []}', encoding="utf-8")
        for vocab_size in (8, 9):
            for placements in layouts:
                for padding_idx in (None, 0, 2, 5, vocab_size - 1):
                    model = _meta_model(vocab_size, padding_idx, str(assets))
                    embedding = model.model.embed_tokens
                    embedding.weight = nn.Parameter(distribute_tensor(embedding.weight, mesh, placements))
                    model = _move_model_to_device(model, True, torch.device("cpu"))
                    assert isinstance(embedding.weight, DTensor)
                    assert not embedding.weight.to_local().is_meta
                    row_labels = torch.arange(vocab_size).unsqueeze(1).expand(vocab_size, 16).contiguous()
                    local_rows = distribute_tensor(row_labels, mesh, placements, src_data_rank=None).to_local()[:, 0]
                    _initialize_model_weights(model)
                    zero_rows = embedding.weight.to_local().eq(0).all(dim=1)
                    expected = local_rows == embedding.padding_idx if padding_idx is not None else local_rows < 0
                    torch.testing.assert_close(
                        zero_rows, expected,
                        msg=str((vocab_size, placements, padding_idx, local_rows.tolist(), zero_rows.tolist())),
                    )
        embedding = nn.Embedding(1, 16, padding_idx=0, device="meta")
        embedding.weight = nn.Parameter(distribute_tensor(embedding.weight, mesh, (Shard(0), Replicate())))
        embedding.to_empty(device="cpu")
        with torch.no_grad():
            _initialize_embedding_shard_safe(embedding, 0.02)
        assert embedding.weight.to_local().eq(0).all()
    finally:
        dist.destroy_process_group()
