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
"""Tests for the DSA architecture-specific TP template."""

from types import SimpleNamespace

import torch
from torch import nn

from hyper_parallel.auto_models.components.distributed import sharding_applier
from hyper_parallel.auto_models.components.distributed.sharding_config import (
    ModuleShardingSpec,
)
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard


class _LayoutAttention(nn.Module):
    def __init__(self, attention_type: str) -> None:
        super().__init__()
        self.attention_type = attention_type
        self.linear_proj = nn.Linear(8, 8, bias=False)


class _LayoutLayer(nn.Module):
    def __init__(self, attention_type: str) -> None:
        super().__init__()
        self.self_attention = _LayoutAttention(attention_type)


class _LayoutMtpBlock(nn.Module):
    def __init__(self, attention_type: str) -> None:
        super().__init__()
        self.self_attention = _LayoutAttention(attention_type)


class _LayoutMtpLayer(nn.Module):
    def __init__(self, attention_type: str) -> None:
        super().__init__()
        self.mtp_block = _LayoutMtpBlock(attention_type)


class _MixedAttentionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            _LayoutLayer("dsa"),
            _LayoutLayer("mla"),
            _LayoutLayer("gqa"),
            _LayoutMtpLayer("mla"),
        ])


class _TinyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param_sink_k_pe = nn.Parameter(torch.ones(8, 4))
        self.param_sink_compressed_kv = nn.Parameter(torch.ones(8, 4))


class _TinyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attention = _TinyAttention()


class _TinyVlModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(architectures=["V2VL"])
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([_TinyLayer()])


def test_linear_proj_sequence_axis_follows_attention_runtime_layout():
    """Mixed attention layers reduce-scatter on their actual sequence axis."""
    from hyper_parallel.auto_models.components.distributed.dsa_template import (
        build_dsa_specs,
    )

    specs = build_dsa_specs(_MixedAttentionModel())

    assert specs["layers.0.self_attention.linear_proj"].out_dst["output"]["tp"] == Shard(1)
    assert specs["layers.1.self_attention.linear_proj"].out_dst["output"]["tp"] == Shard(0)
    assert specs["layers.2.self_attention.linear_proj"].out_dst["output"]["tp"] == Shard(1)
    assert specs["layers.3.mtp_block.self_attention.linear_proj"].out_dst["output"]["tp"] == Shard(0)


def test_dsa_template_owns_only_sink_parameters():
    """The DSA template replicates sinks without absorbing MHC concerns."""
    from hyper_parallel.auto_models.components.distributed.dsa_template import (
        build_dsa_specs,
    )

    specs = build_dsa_specs(_TinyVlModel())
    fqn = "model.language_model.layers.0.self_attention"

    assert set(specs) == {fqn}
    assert specs[fqn].is_boundary is False
    assert set(specs[fqn].params) == {
        "param_sink_k_pe",
        "param_sink_compressed_kv",
    }
    assert all(
        isinstance(placements["tp"], Replicate)
        for placements in specs[fqn].params.values()
    )


def test_parameter_sharding_updates_explicit_head_count_owner(monkeypatch):
    """The extracted sharding phase resolves head owners on its current model."""
    model = nn.Module()
    model.attention = nn.Module()
    model.attention.projection = nn.Linear(4, 4, bias=False)
    spec = ModuleShardingSpec(params={})
    spec._head_count_owner = "attention"
    plan = SimpleNamespace(
        modules={"attention.projection": spec},
        mesh_dim_names=("tp",),
    )
    tp_mesh = SimpleNamespace(size=lambda: 2)
    calls = []

    monkeypatch.setattr(
        sharding_applier,
        "_shard_module_params",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        sharding_applier,
        "maybe_update_head_counts",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        sharding_applier,
        "update_module_head_counts",
        lambda owner, size, fqn: calls.append((owner, size, fqn)),
    )

    sharding_applier._shard_planned_parameters(
        [model], plan, object(), None, tp_mesh, validate_mode=False,
    )

    assert calls == [(model.attention, 2, "attention")]
