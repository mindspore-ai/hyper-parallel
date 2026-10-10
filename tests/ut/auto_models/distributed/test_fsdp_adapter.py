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
"""Unit tests for FSDP transformer-block discovery."""

from torch import nn

from hyper_parallel.distributed._builder.fsdp_adapter import FSDP2Manager


class _Block(nn.Module):
    """Minimal repeated transformer block."""


class _CheckpointingOwner(nn.Module):
    """HF-style owner with both repeated blocks and non-block children."""

    gradient_checkpointing = False

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleDict({"2": _Block(), "7": _Block()})
        self.patch_embed = nn.ModuleDict({"proj": nn.Linear(2, 2)})
        self.mixed = nn.ModuleDict({"linear": nn.Linear(2, 2), "norm": nn.LayerNorm(2)})


def test_transformer_block_discovery_preserves_names_and_ignores_non_blocks() -> None:
    """Only homogeneous repeated children are blocks, with their real FQNs."""
    model = nn.ModuleDict({"tower": _CheckpointingOwner()})

    blocks, wrapped_module_ids = FSDP2Manager._find_transformer_block_modules(model)

    assert [block.fqn for block in blocks] == ["tower.layers.2", "tower.layers.7"]
    assert [block.module for block in blocks] == [
        model["tower"].layers["2"],
        model["tower"].layers["7"],
    ]
    assert wrapped_module_ids == {id(block.module) for block in blocks}
