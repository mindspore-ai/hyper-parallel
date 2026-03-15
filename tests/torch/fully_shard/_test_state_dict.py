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
  T2  (allcards 2-card): state_dict shape + load DTensor (copy + assign)
  T3  (allcards 2-card): load Tensor (local shard + global + vanilla checkpoint)
  T5  (allcards 8-card): full training round-trip (DTensor + local + global)
  T6  (allcards 8-card): get_model_state_dict (sharded default + sharded cpu_offload)
  T7  (allcards 8-card): get_model_state_dict full + cpu_offload
  T8  (allcards 2-card): get_model_state_dict ignore_frozen_params
  T10 (onecard): _to_dtype_if_needed cast / no-op
  T11 (allcards 2-card): meta init -> load_state_dict -> backward
  T13 (allcards 2-card): _extra_state error paths (missing + unexpected)
  T15 (allcards 2-card): nested _extra_state round-trip (HSDP success-path integration)
  T16 (onecard): _extra_state with wrapper that strips prefix (Float16Module scenario)
  T17 (onecard): asymmetric get/set override — only get_extra_state overridden
  T18 (allcards 2-card): pre-hook injects missing _extra_state (HSDP Phase 0 integration)
  T19 (allcards 2-card): pre-hook + wrapper (HSDP integration, PrefixStrippingWrapper + injection)
"""
import os
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

# pylint: disable=C0413
import torch
import torch.distributed as dist
import torch_npu  # pylint: disable=W0611
from torch.distributed.checkpoint.state_dict import StateDictOptions

from hyper_parallel import init_device_mesh, SkipDTensorDispatch
from hyper_parallel.core.dtensor import DTensor
from hyper_parallel.core.fully_shard.api import (
    _build_prefix_to_module,
    _collect_missing_keys,
    _load_extra_states,
    _prepare_mutable_state_dict,
    _raise_if_incompatible,
    _run_load_pre_hooks,
    fully_shard,
    get_model_state_dict,
)
from hyper_parallel.platform.torch.fully_shard.state import _to_dtype_if_needed
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
        k: v.to_local().clone() if isinstance(v, DTensor) else v.clone()
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


def _run_extra_state_load_ut(module: torch.nn.Module, state_dict, strict: bool = True):
    """Replay the HSDP _extra_state load path on a plain nn.Module.

    This unit-test helper exercises the same prefix-discovery, pre-hook replay,
    _extra_state dispatch, and strict missing/unexpected classification used by
    HSDPModule.load_state_dict(), without requiring distributed init or DTensor.
    """
    expected_state = module.state_dict(keep_vars=True)
    mutable_sd = _prepare_mutable_state_dict(state_dict)
    prefix_to_module = _build_prefix_to_module(module, expected_state)

    module_to_prefix: dict[int, str] = {}
    for prefix, submodule in prefix_to_module.items():
        mid = id(submodule)
        if mid not in module_to_prefix:
            module_to_prefix[mid] = prefix

    _run_load_pre_hooks(module, mutable_sd, module_to_prefix, strict)

    missing_keys: list[str] = []
    unexpected_keys: list[str] = []
    _load_extra_states(
        module, mutable_sd, expected_state, strict,
        missing_keys, unexpected_keys, prefix_to_module,
    )

    if strict:
        missing_keys.extend(
            _collect_missing_keys(expected_state, set(mutable_sd.keys()))
        )
        if missing_keys or unexpected_keys:
            _raise_if_incompatible(
                module.__class__.__name__, missing_keys, unexpected_keys,
            )

    return mutable_sd, missing_keys, unexpected_keys


# =====================================================================
# T2 (allcards 2-card): state_dict shape + load DTensor
# =====================================================================
def test_t2_load_dtensor_2cards():
    """state_dict shape check + load_state_dict with hyper DTensor: copy and assign."""
    init_dist()
    num_cards = 2
    torch.manual_seed(42 + _rank())
    model_a = _make_model(num_cards)
    x = torch.randn(BATCH, HIDDEN).npu()
    _train_step(model_a, x)
    fwd_a = _forward_val(model_a, x)
    sd = model_a.state_dict()

    # ---- Shape validation (absorbed from T1) ----
    assert len(sd) > 0, "state_dict is empty"
    for key, val in sd.items():
        assert isinstance(val, DTensor), f"sd[{key}] should be DTensor"
        assert tuple(val.shape) == tuple(val.size()), (
            f"sd[{key}]: .shape != .size()"
        )
        local_dim0 = val.to_local().shape[0]
        global_dim0 = val.shape[0]
        assert global_dim0 == local_dim0 * num_cards, (
            f"sd[{key}]: global[0]={global_dim0} != local[0]*{num_cards}="
            f"{local_dim0 * num_cards}"
        )

    # ---- copy path (assign=False) ----
    model_b = _make_model(num_cards)
    model_b.load_state_dict(sd)
    _assert_fwd_match(fwd_a, _forward_val(model_b, x), "DTensor copy")
    _assert_all_dtensor(model_b, "DTensor copy")

    # ---- assign path (assign=True) ----
    model_c = _make_model(num_cards)
    model_c.load_state_dict(sd, assign=True)
    _assert_fwd_match(fwd_a, _forward_val(model_c, x), "DTensor assign")
    _assert_all_dtensor(model_c, "DTensor assign")

    print(f"[rank{_rank()}] T2 PASS: shape check + DTensor load (copy + assign)")


# =====================================================================
# T3 (allcards 2-card): load Tensor (local + global + vanilla)
# =====================================================================
def test_t3_load_tensor_2cards():
    """load_state_dict with plain torch.Tensor: local shard, global, and vanilla checkpoint."""
    init_dist()
    num_cards = 2
    torch.manual_seed(42 + _rank())
    model_a = _make_model(num_cards)
    x = torch.randn(BATCH, HIDDEN).npu()
    _train_step(model_a, x)
    fwd_a = _forward_val(model_a, x)
    sd = model_a.state_dict()

    # --- local shard tensors ---
    local_sd = _to_local_sd(sd)

    model_b = _make_model(num_cards)
    model_b.load_state_dict(local_sd)
    _assert_fwd_match(fwd_a, _forward_val(model_b, x), "local copy")

    model_c = _make_model(num_cards)
    model_c.load_state_dict(local_sd, assign=True)
    _assert_fwd_match(fwd_a, _forward_val(model_c, x), "local assign")
    _assert_all_dtensor(model_c, "local assign")

    # --- global full tensors ---
    full_sd = _to_full_sd(sd)

    model_d = _make_model(num_cards)
    model_d.load_state_dict(full_sd)
    _assert_fwd_match(fwd_a, _forward_val(model_d, x), "global copy")

    model_e = _make_model(num_cards)
    model_e.load_state_dict(full_sd, assign=True)
    _assert_fwd_match(fwd_a, _forward_val(model_e, x), "global assign")
    _assert_all_dtensor(model_e, "global assign")

    # --- vanilla single-NPU checkpoint (absorbed from T3b) ---
    # Real-world scenario: single-NPU model saved, loaded into fully_shard model.
    torch.manual_seed(123)
    vanilla = FullyShardTestNet(HIDDEN, LAYERS, has_bias=False)
    vanilla_sd = {k: v.clone().detach() for k, v in vanilla.state_dict().items()}
    for k, v in vanilla_sd.items():
        assert not isinstance(v, DTensor), f"vanilla sd[{k}] should be plain Tensor"
        assert v.is_npu, f"vanilla sd[{k}] should be on NPU"

    x_v = torch.randn(BATCH, HIDDEN).npu()
    with torch.no_grad():
        ref_fwd = vanilla(x_v).item()

    # copy path
    model_f = _make_model(num_cards)
    model_f.load_state_dict(vanilla_sd)
    _assert_all_dtensor(model_f, "vanilla copy")
    _assert_fwd_match(ref_fwd, _forward_val(model_f, x_v), "vanilla copy")

    # assign path
    model_g = _make_model(num_cards)
    model_g.load_state_dict(vanilla_sd, assign=True)
    _assert_all_dtensor(model_g, "vanilla assign")
    _assert_fwd_match(ref_fwd, _forward_val(model_g, x_v), "vanilla assign")

    # load then continue training — verify training has effect
    for _ in range(3):
        _train_step(model_g, x_v)
    fwd_after_train = _forward_val(model_g, x_v)
    assert fwd_after_train != ref_fwd, (
        "training after vanilla load had no effect"
    )
    _assert_all_dtensor(model_g, "vanilla train")

    print(f"[rank{_rank()}] T3 PASS: Tensor load (local + global + vanilla)")


# =====================================================================
# T5 (allcards 8-card): full training round-trip
# =====================================================================
def test_t5_roundtrip_8cards():
    """Full training round-trip on 8 cards: DTensor + local + global."""
    init_dist()
    assert dist.get_world_size() >= 8, "T5 requires 8 cards"
    num_cards = 8
    torch.manual_seed(2026 + _rank())

    model = _make_model(num_cards)
    x = torch.randn(BATCH, HIDDEN).npu()

    # Phase 1: train 3 steps
    for _ in range(3):
        _train_step(model, x)
    fwd_trained = _forward_val(model, x)
    sd = model.state_dict()

    # Phase 2: load hyper DTensor
    model2 = _make_model(num_cards)
    model2.load_state_dict(sd, assign=True)
    _assert_fwd_match(fwd_trained, _forward_val(model2, x), "DTensor")

    # Phase 3: load torch Tensor (local shard)
    local_sd = _to_local_sd(sd)
    model3 = _make_model(num_cards)
    model3.load_state_dict(local_sd)
    _assert_fwd_match(fwd_trained, _forward_val(model3, x), "local Tensor")

    # Phase 4: load torch Tensor (global full)
    full_sd = _to_full_sd(sd)
    model4 = _make_model(num_cards)
    model4.load_state_dict(full_sd, assign=True)
    _assert_fwd_match(fwd_trained, _forward_val(model4, x), "global Tensor")

    # Continue training on loaded model
    for _ in range(2):
        _train_step(model4, x)
    assert _forward_val(model4, x) != fwd_trained, "training had no effect"
    _assert_all_dtensor(model4, "round-trip")

    print(f"[rank{_rank()}] T5 PASS: 8-card round-trip (DTensor + local + global)")


# =====================================================================
# T6 (allcards 8-card): get_model_state_dict (sharded + cpu_offload)
# =====================================================================
def test_t6_get_model_sd_sharded():
    """get_model_state_dict: default sharded options AND sharded + cpu_offload."""
    init_dist()
    num_cards = 8
    torch.manual_seed(42 + _rank())
    model = _make_model(num_cards)
    x = torch.randn(BATCH, HIDDEN).npu()
    _train_step(model, x)

    # --- default options: sharded DTensor ---
    sd = get_model_state_dict(model)
    ref_sd = model.state_dict()

    assert set(sd.keys()) == set(ref_sd.keys()), (
        f"key mismatch: {set(sd.keys()) ^ set(ref_sd.keys())}"
    )
    for key, val in sd.items():
        assert isinstance(val, DTensor), f"sd[{key}] should be DTensor"
        ref_val = ref_sd[key]
        assert tuple(val.shape) == tuple(ref_val.shape), (
            f"sd[{key}]: shape mismatch {val.shape} vs {ref_val.shape}"
        )

    # --- sharded + cpu_offload (absorbed from T9) ---
    opts_cpu = StateDictOptions(cpu_offload=True)
    sd_cpu = get_model_state_dict(model, options=opts_cpu)

    assert len(sd_cpu) > 0, "state_dict should be non-empty"
    for key, val in sd_cpu.items():
        assert isinstance(val, DTensor), f"sd_cpu[{key}] should still be DTensor"
        local_tensor = val.to_local()
        assert local_tensor.device == torch.device("cpu"), (
            f"sd_cpu[{key}] local shard should be on CPU, got {local_tensor.device}"
        )

    print(f"[rank{_rank()}] T6 PASS: get_model_state_dict (sharded + sharded cpu)")


# =====================================================================
# T7 (allcards 8-card): get_model_state_dict full + cpu_offload
# =====================================================================
def test_t7_get_model_sd_full_cpu():
    """get_model_state_dict with full_state_dict=True, cpu_offload=True."""
    init_dist()
    num_cards = 8
    torch.manual_seed(42 + _rank())
    model = _make_model(num_cards)
    x = torch.randn(BATCH, HIDDEN).npu()
    _train_step(model, x)

    opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
    sd = get_model_state_dict(model, options=opts)

    if _rank() == 0:
        assert len(sd) > 0, "rank0 should get non-empty state_dict"
        for key, val in sd.items():
            assert isinstance(val, torch.Tensor), f"sd[{key}] should be Tensor"
            assert not isinstance(val, DTensor), f"sd[{key}] should not be DTensor"
            assert val.device == torch.device("cpu"), (
                f"sd[{key}] should be on CPU, got {val.device}"
            )
            # Full tensor: dim0 should equal global shape
            expected_dim0 = HIDDEN  # all params are (HIDDEN, HIDDEN)
            assert val.shape[0] == expected_dim0, (
                f"sd[{key}]: expected full dim0={expected_dim0}, got {val.shape[0]}"
            )
    else:
        assert len(sd) == 0, f"non-rank0 should get empty dict, got {len(sd)} keys"

    print(f"[rank{_rank()}] T7 PASS: get_model_state_dict full + cpu_offload")


# =====================================================================
# T8 (allcards 2-card): get_model_state_dict ignore_frozen_params
# =====================================================================
def test_t8_get_model_sd_ignore_frozen():
    """get_model_state_dict with ignore_frozen_params=True excludes frozen params."""
    init_dist()
    num_cards = 2
    torch.manual_seed(42 + _rank())
    model = _make_model(num_cards)

    # Freeze the first dense layer's weight
    all_params = list(model.named_parameters())
    assert len(all_params) > 1, "model should have >1 param to freeze one"
    frozen_name, frozen_param = all_params[0]
    frozen_param.requires_grad = False

    opts = StateDictOptions(ignore_frozen_params=True)
    sd = get_model_state_dict(model, options=opts)

    assert frozen_name not in sd, (
        f"frozen param '{frozen_name}' should not be in state_dict"
    )

    # All non-frozen params should still be present
    non_frozen_names = {name for name, p in all_params if p.requires_grad}
    for name in non_frozen_names:
        assert name in sd, f"non-frozen param '{name}' should be in state_dict"

    print(f"[rank{_rank()}] T8 PASS: get_model_state_dict ignore_frozen_params")


# =====================================================================
# T10 (dryrun): _to_dtype_if_needed
# =====================================================================
def test_t10_to_dtype_if_needed():
    """_to_dtype_if_needed: cast when dtype differs, no-op when same."""
    # Same dtype -> returns same object
    t_fp32 = torch.randn(4, 4)
    result = _to_dtype_if_needed(t_fp32, torch.float32)
    assert result is t_fp32, "same dtype should return same object"

    # None dtype -> returns same object
    result_none = _to_dtype_if_needed(t_fp32, None)
    assert result_none is t_fp32, "None dtype should return same object"

    # Different dtype -> returns cast tensor
    result_fp16 = _to_dtype_if_needed(t_fp32, torch.float16)
    assert result_fp16 is not t_fp32, "different dtype should return new tensor"
    assert result_fp16.dtype == torch.float16, (
        f"expected float16, got {result_fp16.dtype}"
    )
    assert torch.allclose(t_fp32, result_fp16.float(), atol=1e-3), (
        "cast values should be approximately equal"
    )

    print("T10 PASS: _to_dtype_if_needed")


# =====================================================================
# T11 (allcards 2-card): meta init -> load_state_dict -> backward
# =====================================================================
def test_t11_meta_load_backward():
    """Regression: load_state_dict into meta-init model must preserve
    requires_grad so that backward does not crash with
    'element 0 of tensors does not require grad'.
    """
    init_dist()
    num_cards = 2
    torch.manual_seed(42 + _rank())

    # Phase 1: reference model -- train and extract a global checkpoint
    model_ref = _make_model(num_cards)
    x = torch.randn(BATCH, HIDDEN).npu()
    _train_step(model_ref, x)
    fwd_ref = _forward_val(model_ref, x)
    checkpoint = _to_full_sd(model_ref.state_dict())

    # Phase 2: target model -- convert DTensor params to meta (simulate
    # lazy / meta init, e.g. HuggingFace init_empty_weights)
    model = _make_model(num_cards)
    grad_flags: dict[str, bool] = {}
    for name, p in model.named_parameters():
        grad_flags[name] = p.requires_grad
        if isinstance(p, DTensor):
            meta_local = torch.empty_like(p.to_local(), device="meta")
            meta_local.requires_grad_(p.requires_grad)
            p._local_tensor = meta_local  # pylint: disable=protected-access

    # Sanity: params are now meta
    for name, p in model.named_parameters():
        if isinstance(p, DTensor):
            assert p.to_local().is_meta, f"'{name}' should be meta"

    # Phase 3: load checkpoint (hits meta materialisation path)
    model.load_state_dict(checkpoint)

    # Phase 4: requires_grad must be preserved
    for name, p in model.named_parameters():
        assert p.requires_grad == grad_flags[name], (
            f"'{name}': requires_grad changed from {grad_flags[name]} "
            f"to {p.requires_grad} after load_state_dict"
        )
        assert not p.to_local().is_meta, f"'{name}' should be materialised"

    # Phase 5: forward values should match reference
    _assert_fwd_match(fwd_ref, _forward_val(model, x), "meta load fwd")

    # Phase 6: backward must succeed (the actual reported crash)
    _train_step(model, x)

    print(f"[rank{_rank()}] T11 PASS: meta init -> load -> backward")


# =====================================================================
# _extra_state helper classes
# =====================================================================
class _ExtraStateLinear(torch.nn.Linear):
    """Linear layer that persists extra non-tensor state."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._my_extra = {"scale": 2.5, "version": 3}

    def get_extra_state(self) -> dict:
        return self._my_extra.copy()

    def set_extra_state(self, state: dict) -> None:
        self._my_extra = state


class _ExtraStateNet(torch.nn.Module):
    """Simple model with a submodule that uses _extra_state."""

    def __init__(self, hidden: int):
        super().__init__()
        self.linear = _ExtraStateLinear(hidden, hidden, bias=False)

    def forward(self, x):
        return self.linear(x).sum()


# =====================================================================
# T13 (allcards 2-card): _extra_state error paths (missing + unexpected)
# =====================================================================
def test_t13_extra_state_error_paths():
    """_extra_state strict-mode error paths: missing AND unexpected keys.

    Part A (missing): checkpoint lacks _extra_state but model has
    get/set overrides. strict=True should raise, strict=False OK.

    Part B (unexpected, absorbed from T14): state_dict has _extra_state
    but target model has no set_extra_state override.
    strict=True should raise, strict=False OK.

    Runs through real HSDPModule.load_state_dict() to ensure error paths
    exercise the full HSDP dispatch, not just the helper functions.
    """
    init_dist()
    num_cards = 2
    mesh = init_device_mesh(
        device_type="npu", mesh_shape=(num_cards,), mesh_dim_names=("dp",),
    )

    # ---- Part A: missing _extra_state ----
    model_src = torch.nn.Module()
    model_src.linear = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
    vanilla_sd = {k: v.clone().detach() for k, v in model_src.state_dict().items()}
    assert "linear._extra_state" not in vanilla_sd

    # strict=True should raise because _extra_state is missing
    model_dst = _ExtraStateNet(HIDDEN)
    fully_shard(model_dst.linear, mesh=mesh, reshard_after_forward=True, mp_policy=MP)
    fully_shard(model_dst, mesh=mesh, reshard_after_forward=True, mp_policy=MP)

    raised = False
    try:
        model_dst.load_state_dict(vanilla_sd, strict=True)
    except RuntimeError as exc:
        raised = True
        assert "Missing key(s)" in str(exc), f"unexpected error: {exc}"
        assert "linear._extra_state" in str(exc), f"wrong missing key: {exc}"
    assert raised, "strict=True should raise on missing _extra_state"

    # strict=False should succeed, extra state keeps default values
    model_dst2 = _ExtraStateNet(HIDDEN)
    fully_shard(model_dst2.linear, mesh=mesh, reshard_after_forward=True, mp_policy=MP)
    fully_shard(model_dst2, mesh=mesh, reshard_after_forward=True, mp_policy=MP)
    model_dst2.load_state_dict(vanilla_sd, strict=False)
    assert model_dst2.linear.get_extra_state() == {"scale": 2.5, "version": 3}

    # ---- Part B: unexpected _extra_state (absorbed from T14) ----
    model_src_b = _ExtraStateNet(HIDDEN)
    fully_shard(model_src_b.linear, mesh=mesh, reshard_after_forward=True, mp_policy=MP)
    fully_shard(model_src_b, mesh=mesh, reshard_after_forward=True, mp_policy=MP)
    sd_with_extra = model_src_b.state_dict()
    assert "linear._extra_state" in sd_with_extra

    class _PlainNet(torch.nn.Module):  # pylint: disable=C0115
        def __init__(self, hidden: int):
            super().__init__()
            self.linear = torch.nn.Linear(hidden, hidden, bias=False)

        def forward(self, x):
            return self.linear(x).sum()

    model_dst_b = _PlainNet(HIDDEN)
    fully_shard(model_dst_b.linear, mesh=mesh, reshard_after_forward=True, mp_policy=MP)
    fully_shard(model_dst_b, mesh=mesh, reshard_after_forward=True, mp_policy=MP)

    # strict=True should raise with unexpected _extra_state key
    raised = False
    try:
        model_dst_b.load_state_dict(sd_with_extra, strict=True)
    except RuntimeError as exc:
        raised = True
        assert "Unexpected key(s)" in str(exc), f"unexpected error: {exc}"
        assert "linear._extra_state" in str(exc), f"wrong unexpected key: {exc}"
    assert raised, "strict=True should raise on unexpected _extra_state"

    # strict=False should succeed (ignore _extra_state silently)
    model_dst_b2 = _PlainNet(HIDDEN)
    fully_shard(model_dst_b2.linear, mesh=mesh, reshard_after_forward=True, mp_policy=MP)
    fully_shard(model_dst_b2, mesh=mesh, reshard_after_forward=True, mp_policy=MP)
    model_dst_b2.load_state_dict(sd_with_extra, strict=False)

    print(f"[rank{_rank()}] T13 PASS: _extra_state error paths (missing + unexpected)")


# =====================================================================
# T15 (allcards 2-card): nested _extra_state round-trip
# =====================================================================
class _NestedExtraStateNet(torch.nn.Module):
    """Model with _extra_state at multiple nesting levels."""

    def __init__(self, hidden: int):
        super().__init__()
        self.encoder = torch.nn.Sequential()
        self.encoder.add_module("layer0", _ExtraStateLinear(hidden, hidden, bias=False))
        self.encoder.add_module("layer1", _ExtraStateLinear(hidden, hidden, bias=False))
        self.head = torch.nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        return self.head(self.encoder(x)).sum()


def test_t15_nested_extra_state_roundtrip():
    """_extra_state round-trip with multi-level nested modules.

    HSDP integration smoke for _extra_state success path: verifies
    HSDPModule.load_state_dict() correctly dispatches set_extra_state
    for multiple nested modules via the real Phase 0 / Phase 2 pipeline.
    """
    init_dist()
    num_cards = 2
    mesh = init_device_mesh(
        device_type="npu", mesh_shape=(num_cards,), mesh_dim_names=("dp",),
    )

    # Source model with mutated extra state at two levels
    model_src = _NestedExtraStateNet(HIDDEN)
    for module in model_src.modules():
        if isinstance(module, _ExtraStateLinear):
            fully_shard(module, mesh=mesh, reshard_after_forward=True, mp_policy=MP)
    fully_shard(model_src, mesh=mesh, reshard_after_forward=True, mp_policy=MP)

    model_src.encoder.layer0.set_extra_state({"scale": 1.0, "version": 10})
    model_src.encoder.layer1.set_extra_state({"scale": 2.0, "version": 20})

    sd = model_src.state_dict()

    assert "encoder.layer0._extra_state" in sd
    assert "encoder.layer1._extra_state" in sd
    assert sd["encoder.layer0._extra_state"] == {"scale": 1.0, "version": 10}
    assert sd["encoder.layer1._extra_state"] == {"scale": 2.0, "version": 20}

    # Target model -- load and verify
    model_dst = _NestedExtraStateNet(HIDDEN)
    for module in model_dst.modules():
        if isinstance(module, _ExtraStateLinear):
            fully_shard(module, mesh=mesh, reshard_after_forward=True, mp_policy=MP)
    fully_shard(model_dst, mesh=mesh, reshard_after_forward=True, mp_policy=MP)

    assert model_dst.encoder.layer0.get_extra_state() == {"scale": 2.5, "version": 3}
    assert model_dst.encoder.layer1.get_extra_state() == {"scale": 2.5, "version": 3}

    model_dst.load_state_dict(sd, strict=True)

    assert model_dst.encoder.layer0.get_extra_state() == {"scale": 1.0, "version": 10}, (
        f"layer0 extra_state not restored: {model_dst.encoder.layer0.get_extra_state()}"
    )
    assert model_dst.encoder.layer1.get_extra_state() == {"scale": 2.0, "version": 20}, (
        f"layer1 extra_state not restored: {model_dst.encoder.layer1.get_extra_state()}"
    )

    print(f"[rank{_rank()}] T15 PASS: nested _extra_state round-trip")


# =====================================================================
# T16 (dryrun UT): _extra_state with wrapper that strips prefix
# =====================================================================
class _PrefixStrippingWrapper(torch.nn.Module):
    """Simulates Float16Module: wraps an inner model as self.module and
    overrides state_dict / load_state_dict to strip the 'module.' prefix.

    This causes state_dict keys ('linear.weight', 'linear._extra_state')
    to diverge from the raw module tree paths ('module.linear.weight',
    'module.linear._extra_state').
    """

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.module = inner

    def forward(self, x):
        return self.module(x)

    def state_dict(self, *args, **kwargs):
        # Delegate to inner module, stripping 'module.' prefix
        return self.module.state_dict(*args, **kwargs)


def test_t16_extra_state_prefix_stripped_wrapper():
    """_extra_state round-trip through a wrapper that strips key prefix.

    This is the Float16Module scenario: the wrapper stores the actual model
    as self.module, and state_dict() returns keys without the 'module.'
    prefix.  The old recursive tree walk used raw module paths (including
    'module.') and could never match, causing all _extra_state keys to be
    falsely reported as unexpected.
    """
    inner_src = _ExtraStateNet(HIDDEN)
    inner_src.linear.set_extra_state({"scale": 5.0, "version": 42})

    wrapper_src = _PrefixStrippingWrapper(inner_src)
    sd = wrapper_src.state_dict()

    assert "linear._extra_state" in sd, (
        f"expected 'linear._extra_state', got keys: {list(sd.keys())}"
    )
    assert "module.linear._extra_state" not in sd

    inner_dst = _ExtraStateNet(HIDDEN)
    wrapper_dst = _PrefixStrippingWrapper(inner_dst)
    assert wrapper_dst.module.linear.get_extra_state() == {"scale": 2.5, "version": 3}

    expected_state = wrapper_dst.state_dict(keep_vars=True)
    prefix_to_module = _build_prefix_to_module(wrapper_dst, expected_state)
    assert prefix_to_module["linear."] is wrapper_dst.module.linear

    _run_extra_state_load_ut(wrapper_dst, sd, strict=True)

    loaded = wrapper_dst.module.linear.get_extra_state()
    assert loaded == {"scale": 5.0, "version": 42}, (
        f"extra_state not restored through wrapper: {loaded}"
    )

    print("T16 PASS: _extra_state with prefix-stripping wrapper")


# =====================================================================
# T17 (dryrun UT): asymmetric get/set override
# =====================================================================
class _GetOnlyExtraStateLinear(torch.nn.Linear):
    """Linear that overrides get_extra_state but NOT set_extra_state.

    This is a pathological but valid scenario: state_dict() will include
    the _extra_state key, but load_state_dict should treat it as unexpected
    (matching PyTorch's _load_from_state_dict semantics).
    """

    def get_extra_state(self) -> dict:
        return {"read_only": True}

    # Intentionally does NOT override set_extra_state.


class _AsymmetricNet(torch.nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.linear = _GetOnlyExtraStateLinear(hidden, hidden, bias=False)

    def forward(self, x):
        return self.linear(x).sum()


def test_t17_asymmetric_extra_state_override():
    """Module overrides get_extra_state but not set_extra_state.

    state_dict() includes the _extra_state key (via get_extra_state), but
    load_state_dict should treat it as unexpected because set_extra_state
    is not overridden -- matching PyTorch's _load_from_state_dict check.
    """
    model_src = _AsymmetricNet(HIDDEN)
    sd = model_src.state_dict()
    assert "linear._extra_state" in sd, f"expected _extra_state key, got: {list(sd.keys())}"

    model_dst = _AsymmetricNet(HIDDEN)
    raised = False
    try:
        _run_extra_state_load_ut(model_dst, sd, strict=True)
    except RuntimeError as exc:
        raised = True
        assert "Unexpected key(s)" in str(exc), f"unexpected error: {exc}"
        assert "linear._extra_state" in str(exc), f"wrong key in error: {exc}"
    assert raised, "strict=True should raise for asymmetric get/set override"

    model_dst2 = _AsymmetricNet(HIDDEN)
    _run_extra_state_load_ut(model_dst2, sd, strict=False)

    sd_no_extra = {k: v for k, v in sd.items() if not k.endswith("_extra_state")}
    assert "linear._extra_state" not in sd_no_extra

    model_dst3 = _AsymmetricNet(HIDDEN)
    _run_extra_state_load_ut(model_dst3, sd_no_extra, strict=True)

    print("T17 PASS: asymmetric get/set _extra_state override")


# =====================================================================
# T18 (allcards 2-card): pre-hook injects missing _extra_state
# =====================================================================
class _PreHookExtraStateLinear(torch.nn.Linear):
    """Linear with get/set_extra_state AND a pre-hook that injects a
    default _extra_state key when the checkpoint lacks it -- mirroring the
    real-world pattern used by ColumnParallelLinear / output_layer.
    """

    _DEFAULT_EXTRA = {"injected": True, "value": 0}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._my_extra = self._DEFAULT_EXTRA.copy()
        self.register_load_state_dict_pre_hook(self._inject_extra_state)

    def get_extra_state(self) -> dict:
        return self._my_extra.copy()

    def set_extra_state(self, state: dict) -> None:
        self._my_extra = state

    @staticmethod
    def _inject_extra_state(
        module, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        """Pre-hook: if _extra_state is absent, inject a default."""
        extra_key = prefix + "_extra_state"
        if extra_key not in state_dict:
            state_dict[extra_key] = _PreHookExtraStateLinear._DEFAULT_EXTRA.copy()


class _PreHookNet(torch.nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.linear = _PreHookExtraStateLinear(hidden, hidden, bias=False)

    def forward(self, x):
        return self.linear(x).sum()


def test_t18_pre_hook_injects_extra_state():
    """Pre-hook injects _extra_state when checkpoint lacks it.

    HSDP integration test for the Phase 0 bug fix: without pre-hook
    replay, HSDPModule.load_state_dict() would report
    Missing key(s): 'linear._extra_state'.  With Phase 0 the hook
    fires, injects the key, and strict=True succeeds.
    """
    init_dist()
    num_cards = 2
    mesh = init_device_mesh(
        device_type="npu", mesh_shape=(num_cards,), mesh_dim_names=("dp",),
    )

    # Build model
    model = _PreHookNet(HIDDEN)
    fully_shard(model.linear, mesh=mesh, reshard_after_forward=True, mp_policy=MP)
    fully_shard(model, mesh=mesh, reshard_after_forward=True, mp_policy=MP)

    # Checkpoint WITHOUT _extra_state (simulating older / vanilla save)
    sd = model.state_dict()
    sd_no_extra = {k: v for k, v in sd.items() if not k.endswith("_extra_state")}
    assert "linear._extra_state" not in sd_no_extra

    # Load with strict=True — pre-hook should inject _extra_state
    model2 = _PreHookNet(HIDDEN)
    fully_shard(model2.linear, mesh=mesh, reshard_after_forward=True, mp_policy=MP)
    fully_shard(model2, mesh=mesh, reshard_after_forward=True, mp_policy=MP)
    model2.load_state_dict(sd_no_extra, strict=True)  # must NOT raise

    # Verify pre-hook's default was applied via set_extra_state
    loaded = model2.linear.get_extra_state()
    assert loaded == {"injected": True, "value": 0}, (
        f"pre-hook default not applied: {loaded}"
    )

    print(f"[rank{_rank()}] T18 PASS: pre-hook injects _extra_state")


# =====================================================================
# T19 (allcards 2-card): pre-hook + wrapper (PrefixStrippingWrapper + injection)
# =====================================================================
def test_t19_pre_hook_with_wrapper_prefix():
    """PrefixStrippingWrapper + inner module pre-hook that injects _extra_state.

    HSDP integration test combining wrapper prefix stripping (T16) with
    pre-hook injection (T18) through real HSDPModule.load_state_dict().
    Checkpoint lacks _extra_state; wrapper strips prefix; Phase 0 runs
    inner module's pre-hook which injects the key under the correct
    (stripped) prefix; Phase 2 loads it via set_extra_state.
    """
    init_dist()
    num_cards = 2
    mesh = init_device_mesh(
        device_type="npu", mesh_shape=(num_cards,), mesh_dim_names=("dp",),
    )

    # Source: build wrapped model, get state_dict with stripped keys
    inner_src = _PreHookNet(HIDDEN)
    fully_shard(inner_src.linear, mesh=mesh, reshard_after_forward=True, mp_policy=MP)
    fully_shard(inner_src, mesh=mesh, reshard_after_forward=True, mp_policy=MP)
    inner_src.linear.set_extra_state({"injected": False, "value": 42})

    wrapper_src = _PrefixStrippingWrapper(inner_src)
    fully_shard(wrapper_src, mesh=mesh, reshard_after_forward=True, mp_policy=MP)

    sd = wrapper_src.state_dict()
    assert "linear._extra_state" in sd, (
        f"expected 'linear._extra_state', got keys: {list(sd.keys())}"
    )
    assert "module.linear._extra_state" not in sd

    # Simulate checkpoint WITHOUT _extra_state
    sd_no_extra = {k: v for k, v in sd.items() if not k.endswith("_extra_state")}
    assert "linear._extra_state" not in sd_no_extra

    # Target: same structure, load through real HSDP path
    inner_dst = _PreHookNet(HIDDEN)
    fully_shard(inner_dst.linear, mesh=mesh, reshard_after_forward=True, mp_policy=MP)
    fully_shard(inner_dst, mesh=mesh, reshard_after_forward=True, mp_policy=MP)
    wrapper_dst = _PrefixStrippingWrapper(inner_dst)
    fully_shard(wrapper_dst, mesh=mesh, reshard_after_forward=True, mp_policy=MP)

    # Phase 0 pre-hook injects _extra_state, Phase 2 loads it
    wrapper_dst.load_state_dict(sd_no_extra, strict=True)  # must NOT raise

    loaded = wrapper_dst.module.linear.get_extra_state()
    assert loaded == {"injected": True, "value": 0}, (
        f"pre-hook default not applied through wrapper: {loaded}"
    )

    print(f"[rank{_rank()}] T19 PASS: pre-hook + wrapper prefix rewrite")
