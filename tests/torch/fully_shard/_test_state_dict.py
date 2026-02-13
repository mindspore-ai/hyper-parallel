# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""state_dict / load_state_dict tests for fully_shard.

Verified scenarios:
  T1 (2-card): state_dict returns DTensor with correct global shape
  T2 (2-card): load_state_dict with hyper DTensor (copy + assign)
  T3 (2-card): load_state_dict with torch Tensor (local shard + global)
  T4 (4-card): train -> save -> load(DTensor + Tensor) -> continue training
  T5 (8-card): full training round-trip (DTensor + local + global)
"""
import os
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

# pylint: disable=C0413
import torch
import torch_npu  # pylint: disable=W0611
import torch.distributed as dist

from hyper_parallel import init_device_mesh, SkipDTensorDispatch
from hyper_parallel.core.dtensor import DTensor
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.platform.torch.fully_shard.utils import MixedPrecisionPolicy
from tests.torch.common_net import FullyShardTestNet
from tests.torch.utils import init_dist

# --------------- constants ---------------
HIDDEN = 32
LAYERS = 2
BATCH = 4
MP = MixedPrecisionPolicy(
    param_dtype=torch.float32,
    reduce_dtype=torch.float32,
    output_dtype=torch.float32,
    cast_forward_inputs=True,
)


# --------------- helpers ---------------
def _rank():
    return dist.get_rank()


def _make_model(num_cards):
    """Create fully_shard wrapped FullyShardTestNet."""
    mesh = init_device_mesh(
        device_type="npu", mesh_shape=(num_cards,), mesh_dim_names=("dp",),
    )
    model = FullyShardTestNet(HIDDEN, LAYERS, has_bias=False)
    for dense_layer in model.dense_layers.layers:
        fully_shard(
            dense_layer, mesh=mesh, reshard_after_forward=True, mp_policy=MP,
        )
    fully_shard(model, mesh=mesh, reshard_after_forward=True, mp_policy=MP)
    return model


def _train_step(model, x, lr=0.01):
    """One SGD training step, returns loss value."""
    with SkipDTensorDispatch():
        loss = model(x)
        loss.backward(torch.ones_like(loss))
        for p in model.parameters():
            if p.grad is not None:
                p.data.add_(p.grad, alpha=-lr)
            p.grad = None
    return loss.item()


def _forward_val(model, x):
    """Forward pass, return scalar value."""
    with SkipDTensorDispatch():
        return model(x).item()


def _to_local_sd(sd):
    """Extract local shard tensors from state_dict."""
    return {
        k: v._local_tensor.clone() if isinstance(v, DTensor) else v.clone()
        for k, v in sd.items()
    }


def _to_full_sd(sd):
    """Extract global full tensors from state_dict."""
    return {
        k: v.full_tensor().clone() if isinstance(v, DTensor) else v.clone()
        for k, v in sd.items()
    }


def _assert_all_dtensor(model, tag=""):
    """Assert every parameter is a DTensor."""
    for name, p in model.named_parameters():
        assert isinstance(p, DTensor), f"{tag}: '{name}' is not DTensor"


def _assert_fwd_match(fwd_expected, fwd_actual, tag="", tol=1e-4):
    """Assert forward values match within tolerance."""
    diff = abs(fwd_expected - fwd_actual)
    assert diff < tol, (
        f"{tag}: expected {fwd_expected}, got {fwd_actual}, diff={diff}"
    )


# =====================================================================
# T1 (2-card): state_dict returns DTensor with global shape
# =====================================================================
def test_t1_state_dict_2cards():
    """After fully_shard, state_dict() returns DTensors with global shape."""
    init_dist()
    num_cards = 2
    model = _make_model(num_cards)
    sd = model.state_dict()

    assert len(sd) > 0, "state_dict is empty"
    for key, val in sd.items():
        assert isinstance(val, DTensor), f"sd[{key}] should be DTensor"
        assert tuple(val.shape) == tuple(val.size()), (
            f"sd[{key}]: .shape != .size()"
        )
        # Sharded dim: global[0] should be num_cards * local[0]
        local_dim0 = val._local_tensor.shape[0]
        global_dim0 = val.shape[0]
        assert global_dim0 == local_dim0 * num_cards, (
            f"sd[{key}]: global[0]={global_dim0} != local[0]*{num_cards}="
            f"{local_dim0 * num_cards}"
        )

    print(f"[rank{_rank()}] T1 PASS: state_dict has {len(sd)} DTensor entries")


# =====================================================================
# T2 (2-card): load_state_dict with hyper DTensor (copy + assign)
# =====================================================================
def test_t2_load_dtensor_2cards():
    """load_state_dict with hyper DTensor: both copy and assign paths."""
    init_dist()
    torch.manual_seed(42 + _rank())
    model_a = _make_model(2)
    x = torch.randn(BATCH, HIDDEN).npu()
    _train_step(model_a, x)
    fwd_a = _forward_val(model_a, x)
    sd = model_a.state_dict()

    # copy path (assign=False)
    model_b = _make_model(2)
    model_b.load_state_dict(sd)
    _assert_fwd_match(fwd_a, _forward_val(model_b, x), "DTensor copy")
    _assert_all_dtensor(model_b, "DTensor copy")

    # assign path (assign=True)
    model_c = _make_model(2)
    model_c.load_state_dict(sd, assign=True)
    _assert_fwd_match(fwd_a, _forward_val(model_c, x), "DTensor assign")
    _assert_all_dtensor(model_c, "DTensor assign")

    print(f"[rank{_rank()}] T2 PASS: DTensor load (copy + assign)")


# =====================================================================
# T3 (2-card): load_state_dict with torch Tensor (local + global)
# =====================================================================
def test_t3_load_tensor_2cards():
    """load_state_dict with plain torch.Tensor: local shard and global."""
    init_dist()
    torch.manual_seed(42 + _rank())
    model_a = _make_model(2)
    x = torch.randn(BATCH, HIDDEN).npu()
    _train_step(model_a, x)
    fwd_a = _forward_val(model_a, x)
    sd = model_a.state_dict()

    # --- local shard tensors ---
    local_sd = _to_local_sd(sd)

    model_b = _make_model(2)
    model_b.load_state_dict(local_sd)
    _assert_fwd_match(fwd_a, _forward_val(model_b, x), "local copy")

    model_c = _make_model(2)
    model_c.load_state_dict(local_sd, assign=True)
    _assert_fwd_match(fwd_a, _forward_val(model_c, x), "local assign")
    _assert_all_dtensor(model_c, "local assign")

    # --- global full tensors ---
    full_sd = _to_full_sd(sd)

    model_d = _make_model(2)
    model_d.load_state_dict(full_sd)
    _assert_fwd_match(fwd_a, _forward_val(model_d, x), "global copy")

    model_e = _make_model(2)
    model_e.load_state_dict(full_sd, assign=True)
    _assert_fwd_match(fwd_a, _forward_val(model_e, x), "global assign")
    _assert_all_dtensor(model_e, "global assign")

    print(f"[rank{_rank()}] T3 PASS: Tensor load (local + global, copy + assign)")


# =====================================================================
# T3b (2-card): load from single-NPU vanilla state_dict
# =====================================================================
def test_t3b_load_single_npu_checkpoint():
    """Simulate: single-NPU model saved with torch.save, loaded into fully_shard model.

    This is the real-world scenario:
        1. Train on 1 NPU, torch.save(model.state_dict(), "ckpt.pt")
        2. Load on N NPUs: model = fully_shard(Model()); model.load_state_dict(torch.load("ckpt.pt"))
    """
    init_dist()
    num_cards = 2

    # Phase 1: vanilla (non-sharded) model — params are plain Tensor on NPU
    torch.manual_seed(123)
    vanilla = FullyShardTestNet(HIDDEN, LAYERS, has_bias=False)
    vanilla_sd = {k: v.clone().detach() for k, v in vanilla.state_dict().items()}
    for k, v in vanilla_sd.items():
        assert not isinstance(v, DTensor), f"vanilla sd[{k}] should be plain Tensor"
        assert v.is_npu, f"vanilla sd[{k}] should be on NPU"

    x = torch.randn(BATCH, HIDDEN).npu()
    with torch.no_grad():
        ref_fwd = vanilla(x).item()

    # Phase 2: load global Tensor into fully_shard model
    model_a = _make_model(num_cards)
    model_a.load_state_dict(vanilla_sd)
    _assert_all_dtensor(model_a, "vanilla copy")
    _assert_fwd_match(ref_fwd, _forward_val(model_a, x), "vanilla copy")

    model_b = _make_model(num_cards)
    model_b.load_state_dict(vanilla_sd, assign=True)
    _assert_all_dtensor(model_b, "vanilla assign")
    _assert_fwd_match(ref_fwd, _forward_val(model_b, x), "vanilla assign")

    # Phase 3: continue training after load
    fwd_before = _forward_val(model_a, x)
    for _ in range(3):
        _train_step(model_a, x)
    fwd_after = _forward_val(model_a, x)
    assert fwd_before != fwd_after, "training had no effect after vanilla load"
    _assert_all_dtensor(model_a, "vanilla after train")

    print(f"[rank{_rank()}] T3b PASS: single-NPU checkpoint load + continue training")


# =====================================================================
# T4 (4-card): full training round-trip
# =====================================================================
def test_t4_roundtrip_4cards():
    """Train -> save -> load(DTensor + Tensor) -> continue training."""
    init_dist()
    assert dist.get_world_size() >= 4, "T4 requires 4+ cards"
    torch.manual_seed(2026 + _rank())

    model = _make_model(4)
    x = torch.randn(BATCH, HIDDEN).npu()

    # Phase 1: train 3 steps
    for _ in range(3):
        _train_step(model, x)
    fwd_trained = _forward_val(model, x)
    sd = model.state_dict()

    # Phase 2: load hyper DTensor, verify, continue training
    model2 = _make_model(4)
    model2.load_state_dict(sd, assign=True)
    _assert_fwd_match(fwd_trained, _forward_val(model2, x), "DTensor load")
    for _ in range(2):
        _train_step(model2, x)
    assert _forward_val(model2, x) != fwd_trained, "training had no effect after DTensor load"
    _assert_all_dtensor(model2, "DTensor round-trip")

    # Phase 3: load torch Tensor (local shard), verify, continue training
    local_sd = _to_local_sd(sd)
    model3 = _make_model(4)
    model3.load_state_dict(local_sd, assign=True)
    _assert_fwd_match(fwd_trained, _forward_val(model3, x), "Tensor load")
    for _ in range(2):
        _train_step(model3, x)
    assert _forward_val(model3, x) != fwd_trained, "training had no effect after Tensor load"
    _assert_all_dtensor(model3, "Tensor round-trip")

    print(f"[rank{_rank()}] T4 PASS: 4-card round-trip (DTensor + Tensor)")


# =====================================================================
# T5 (8-card): full training round-trip
# =====================================================================
def test_t5_roundtrip_8cards():
    """Full training round-trip on 8 cards: DTensor + local + global."""
    init_dist()
    assert dist.get_world_size() >= 8, "T5 requires 8 cards"
    torch.manual_seed(2026 + _rank())

    model = _make_model(8)
    x = torch.randn(BATCH, HIDDEN).npu()

    # Phase 1: train 3 steps
    for _ in range(3):
        _train_step(model, x)
    fwd_trained = _forward_val(model, x)
    sd = model.state_dict()

    # Phase 2: load hyper DTensor
    model2 = _make_model(8)
    model2.load_state_dict(sd, assign=True)
    _assert_fwd_match(fwd_trained, _forward_val(model2, x), "DTensor")

    # Phase 3: load torch Tensor (local shard)
    local_sd = _to_local_sd(sd)
    model3 = _make_model(8)
    model3.load_state_dict(local_sd)
    _assert_fwd_match(fwd_trained, _forward_val(model3, x), "local Tensor")

    # Phase 4: load torch Tensor (global full)
    full_sd = _to_full_sd(sd)
    model4 = _make_model(8)
    model4.load_state_dict(full_sd, assign=True)
    _assert_fwd_match(fwd_trained, _forward_val(model4, x), "global Tensor")

    # Continue training on loaded model
    for _ in range(2):
        _train_step(model4, x)
    assert _forward_val(model4, x) != fwd_trained, "training had no effect"
    _assert_all_dtensor(model4, "8-card round-trip")

    print(f"[rank{_rank()}] T5 PASS: 8-card round-trip (DTensor + local + global)")
