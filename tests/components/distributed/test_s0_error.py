# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S0.4: PlacementMismatchError message 内容。"""

import pytest

from hyper_models.components.distributed.sharding_config import (
    PlacementMismatchError,
)
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard


def test_message_contains_all_fields():
    err = PlacementMismatchError(
        "model.layers.0.self_attn", (Shard(0),), (Replicate(),), "out_src"
    )
    msg = str(err)
    assert "model.layers.0.self_attn" in msg
    assert "out_src" in msg
    assert "Shard" in msg and "Replicate" in msg
    assert err.module_name == "model.layers.0.self_attn"
    assert err.stage == "out_src"
    assert err.expected == (Shard(0),)
    assert err.actual == (Replicate(),)


def test_is_value_error():
    with pytest.raises(ValueError):
        raise PlacementMismatchError("m", 1, 2, "chain")
