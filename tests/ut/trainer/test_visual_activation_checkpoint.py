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
"""``_apply_visual_ac`` — recompute-wrap a trainable Qwen3-VL visual tower.

The text-only ``_apply_ac`` skips the visual, so a trainable tower would hold all
its block activations for the backward. ``_apply_visual_ac`` checkpoints the
visual blocks under the ``activation_checkpoint`` flag, gated on a trainable
(non-frozen) tower, idempotently. These cover the gating, idempotency, and the
bit-exact recompute contract on a minimal tower (no NPU / distributed).
"""
import importlib.util
import os
import sys
import types

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import pytest
import torch
from torch import nn

# ``_apply_visual_ac`` logs via ``logger.info_rank0``, a method patched onto
# ``logging.Logger`` when the trainer logging module is imported (as real runs do).
import hyper_parallel.trainer.utils.logging as hp_logging

# The Qwen3-VL stack reaches transformers' generation utils, which import
# scikit-learn; on some CI images that fails to load libgomp. Skip rather than
# breaking collection for the whole suite.
try:
    import hyper_parallel.models.qwen3_vl_moe.parallelize as _parallelize_module
    from hyper_parallel.models.qwen3_vl_moe.parallelize import (
        _apply_ac,
        _apply_visual_ac,
        _apply_vl_visual_tower,
        _visual_stage_entry,
    )
except ImportError as exc:  # pragma: no cover
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

# Importing this module must not change global state for the rest of the suite.
# The model import above pulls in ``transformers``, which has two side effects:
#
# 1. it rebinds ``logging.Logger.warning_once`` to its own helper process-wide,
#    so clear the install latch and re-assert hyper_parallel's rank-aware ones;
# 2. it registers lazy vision submodules whose ``__getattr__`` raises
#    ModuleNotFoundError (torchvision is not a declared dependency) rather than
#    AttributeError. ``unittest.assertWarns`` probes ``__warningregistry__`` on
#    every entry of ``sys.modules``, so those raise and fail any later
#    ``assertWarns`` test in the same process. Give them the attribute.
hp_logging._INSTALLED = False  # pylint: disable=protected-access
hp_logging._install_logger_methods()  # pylint: disable=protected-access

for _name, _module in list(sys.modules.items()):
    if not _name.startswith("transformers"):
        continue
    try:
        getattr(_module, "__warningregistry__", None)
    except Exception:  # pylint: disable=broad-except  # lazy import of an absent extra
        try:
            _module.__warningregistry__ = {}
        except Exception:  # pylint: disable=broad-except
            pass

if _IMPORT_ERROR is not None:  # pragma: no cover
    pytest.skip(f"Qwen3-VL model stack unavailable: {_IMPORT_ERROR}",
                allow_module_level=True)


# The pinned torch wrapper makes every WRAP correct, but the checkpoint
# REPLAY (recompute inside backward) cannot run everywhere: it needs a
# torch-resolved process, and on a torch_npu install the scheduled-recompute
# replay builds a device autocast context whose dtype probe lazily aclInits
# the NPU (error 500000 / a hard crash on device-less runners; gate runs
# 7974 and 7984). Run the recompute-backward tests only where neither bites.
from hyper_parallel.platform import get_platform  # noqa: E402  pylint: disable=wrong-import-position
from hyper_parallel.platform.platform import PlatformType  # noqa: E402  pylint: disable=wrong-import-position

_TORCH_RESOLVED = get_platform().platform_type == PlatformType.PYTORCH
_RECOMPUTE_BACKWARD_SKIP = pytest.mark.skipif(
    not _TORCH_RESOLVED or importlib.util.find_spec("torch_npu") is not None,
    reason="checkpoint recompute-backward needs a torch-resolved process "
           "without torch_npu: the replay's device autocast dtype probe "
           "lazily aclInits the NPU even for pure-CPU tensors",
)


@pytest.fixture(autouse=True)
def _pin_torch_checkpoint_wrapper(monkeypatch):
    """Pin the torch AC wrapper for every test in this torch-only module.

    ``core.activation_checkpoint`` binds ``checkpoint_wrapper`` to whichever
    platform resolved first in the process. In a shared pytest run another
    module can freeze it as MindSpore's, whose wrapper is not a
    ``torch.nn.Module`` and cannot be assigned into the tower's ModuleList.
    The env var set above cannot repair an already-frozen alias, so re-point
    the ``parallelize`` module's global at the torch implementation.
    """
    from hyper_parallel.platform.torch.activation_checkpoint.checkpoint_wrapper import (  # pylint: disable=import-outside-toplevel
        ckpt_wrapper,
    )
    monkeypatch.setattr(_parallelize_module, "checkpoint_wrapper", ckpt_wrapper)


class _Block(nn.Module):
    def __init__(self, dim: int) -> None:
        """Build a single linear block."""
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block."""
        return torch.relu(self.lin(x))


class _Visual(nn.Module):
    """Minimal stand-in for the vision tower: just a ``.blocks`` ModuleList."""

    def __init__(self, n: int = 3, dim: int = 8) -> None:
        """Build a tower of ``n`` blocks."""
        super().__init__()
        self.blocks = nn.ModuleList([_Block(dim) for _ in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run every block in sequence."""
        for blk in self.blocks:
            x = blk(x)
        return x


def _cfg(activation_checkpoint: str = "full", freeze=None, visual_num_layers=None,
         num_layers=None):
    """A minimal ``cfg`` exposing the fields the AC helpers read."""
    return types.SimpleNamespace(
        train=types.SimpleNamespace(
            gradient_checkpointing=types.SimpleNamespace(
                activation_checkpoint=activation_checkpoint,
                num_layers=num_layers,
                visual_num_layers=visual_num_layers,
            ),
        ),
        model=types.SimpleNamespace(freeze_modules=list(freeze or [])),
    )


class _Stage(nn.Module):
    """Minimal stand-in for a PP stage module: just a ``.layers`` ModuleList."""

    def __init__(self, n: int = 3, dim: int = 8) -> None:
        """Build ``n`` stacked blocks."""
        super().__init__()
        self.layers = nn.ModuleList([_Block(dim) for _ in range(n)])


def _wrapped(block) -> bool:
    # checkpoint_wrapper returns a CheckpointWrapper holding the original block,
    # so a wrapped element is no longer the bare _Block.
    return not isinstance(block, _Block)


def test_wraps_trainable_visual_when_ac_on():
    vis = _Visual()
    _apply_visual_ac(vis, _cfg("full"))
    assert all(_wrapped(b) for b in vis.blocks)
    assert getattr(vis, "_visual_ac_applied", False)


def test_noop_when_ac_off():
    vis = _Visual()
    _apply_visual_ac(vis, _cfg("off"))
    assert all(isinstance(b, _Block) for b in vis.blocks)
    assert not getattr(vis, "_visual_ac_applied", False)


def test_skips_frozen_visual():
    vis = _Visual()
    vis.requires_grad_(False)  # the trainer freeze runs before any parallelize path
    _apply_visual_ac(vis, _cfg("full", freeze=["model.visual"]))
    assert all(isinstance(b, _Block) for b in vis.blocks)


def test_partial_freeze_still_recomputes_trainable_blocks():
    # freeze_modules like ["visual.merger"] freezes a submodule only; the
    # still-trainable blocks must keep their recompute wrap.
    vis = _Visual()
    vis.blocks[2].requires_grad_(False)
    _apply_visual_ac(vis, _cfg("full", freeze=["model.visual.merger"]))
    assert all(_wrapped(b) for b in vis.blocks)


def test_idempotent_no_double_wrap():
    vis = _Visual()
    _apply_visual_ac(vis, _cfg("full"))
    once = list(vis.blocks)
    _apply_visual_ac(vis, _cfg("full"))
    assert list(vis.blocks) == once  # same objects, not re-wrapped


def test_no_blocks_is_noop():
    # A module without ``.blocks`` (e.g. the identity dataload preprocess) is skipped.
    _apply_visual_ac(nn.Linear(4, 4), _cfg("full"))


def test_none_cfg_is_noop():
    # The mpipe schedule builder treats cfg=None as legal; the AC hook must too.
    vis = _Visual()
    _apply_visual_ac(vis, None)
    assert all(isinstance(b, _Block) for b in vis.blocks)
    assert not getattr(vis, "_visual_ac_applied", False)


def test_visual_stage_entry_reads_pp_local_rank():
    mesh = {"pp": types.SimpleNamespace(get_local_rank=lambda: 2)}
    assert _visual_stage_entry(mesh) == 2
    assert _visual_stage_entry(None) == 0
    assert _visual_stage_entry({}) == 0


def test_depth_wraps_first_n_blocks_only():
    vis = _Visual(n=4)
    _apply_visual_ac(vis, _cfg("full", visual_num_layers=2))
    assert all(_wrapped(b) for b in list(vis.blocks)[:2])
    assert all(isinstance(b, _Block) for b in list(vis.blocks)[2:])
    assert getattr(vis, "_visual_ac_applied", False)


def test_depth_zero_disables_visual_recompute():
    vis = _Visual()
    _apply_visual_ac(vis, _cfg("full", visual_num_layers=0))
    assert all(isinstance(b, _Block) for b in vis.blocks)
    assert not getattr(vis, "_visual_ac_applied", False)


def test_depth_beyond_block_count_clamps():
    vis = _Visual(n=3)
    _apply_visual_ac(vis, _cfg("full", visual_num_layers=99))
    assert all(_wrapped(b) for b in vis.blocks)


def test_depth_negative_rejected():
    vis = _Visual()
    with pytest.raises(ValueError, match="visual_num_layers"):
        _apply_visual_ac(vis, _cfg("full", visual_num_layers=-1))
    assert all(isinstance(b, _Block) for b in vis.blocks)


def test_visual_depth_list_indexes_this_stage_entry():
    # [full, 3, 2, 0] with this rank's replica at global stage 1 -> wrap 3 of 4.
    vis = _Visual(n=4)
    _apply_visual_ac(vis, _cfg("full", visual_num_layers=[27, 3, 2, 0]),
                     stage_idx=1, num_stages=4)
    assert all(_wrapped(b) for b in list(vis.blocks)[:3])
    assert isinstance(vis.blocks[3], _Block)


def test_visual_depth_list_zero_entry_disables():
    vis = _Visual()
    _apply_visual_ac(vis, _cfg("full", visual_num_layers=[27, 3, 2, 0]),
                     stage_idx=3, num_stages=4)
    assert all(isinstance(b, _Block) for b in vis.blocks)
    assert not getattr(vis, "_visual_ac_applied", False)


def test_visual_depth_list_wrong_length_raises():
    vis = _Visual()
    with pytest.raises(ValueError, match="4 entries.*2 global stages"):
        _apply_visual_ac(vis, _cfg("full", visual_num_layers=[27, 3, 2, 0]),
                         stage_idx=0, num_stages=2)


def test_text_depth_default_wraps_all_layers():
    stage = _Stage(n=3)
    _apply_ac(stage, _cfg("full"))
    assert all(_wrapped(layer) for layer in stage.layers)


def test_text_depth_list_per_stage():
    # num_layers=[2, 1]: stage 0 wraps its first 2 of 3, stage 1 its first 1 of 3.
    cfg = _cfg("full", num_layers=[2, 1])
    s0, s1 = _Stage(n=3), _Stage(n=3)
    _apply_ac(s0, cfg, stage_idx=0, num_stages=2)
    _apply_ac(s1, cfg, stage_idx=1, num_stages=2)
    assert [_wrapped(layer) for layer in s0.layers] == [True, True, False]
    assert [_wrapped(layer) for layer in s1.layers] == [True, False, False]


def test_text_depth_uniform_int_applies_per_stage():
    cfg = _cfg("full", num_layers=1)
    s = _Stage(n=3)
    _apply_ac(s, cfg, stage_idx=2, num_stages=4)
    assert [_wrapped(layer) for layer in s.layers] == [True, False, False]


def test_text_depth_zero_leaves_stage_unwrapped():
    s = _Stage(n=3)
    _apply_ac(s, _cfg("full", num_layers=[0, 3]), stage_idx=0, num_stages=2)
    assert all(isinstance(layer, _Block) for layer in s.layers)


def test_text_depth_list_wrong_length_raises():
    with pytest.raises(ValueError, match="2 entries.*4 global stages"):
        _apply_ac(_Stage(n=3), _cfg("full", num_layers=[2, 1]),
                  stage_idx=0, num_stages=4)


@_RECOMPUTE_BACKWARD_SKIP
def test_partial_depth_recompute_matches_no_recompute():
    """Grads stay bit-exact when only a prefix of the blocks recomputes."""
    torch.manual_seed(0)
    ref = _Visual(n=4)
    ckpt = _Visual(n=4)
    ckpt.load_state_dict(ref.state_dict())
    _apply_visual_ac(ckpt, _cfg("full", visual_num_layers=2))

    x = torch.randn(4, 8, requires_grad=True)
    xr = x.detach().clone().requires_grad_(True)

    ref(x).sum().backward()
    ckpt(xr).sum().backward()

    assert torch.allclose(x.grad, xr.grad, atol=1e-6)
    for pr, pc in zip(ref.parameters(), ckpt.parameters()):
        assert pc.grad is not None
        assert torch.allclose(pr.grad, pc.grad, atol=1e-6)


def test_call_site_wraps_via_visual_tower_single_card(monkeypatch):
    """``_apply_vl_visual_tower`` applies visual AC before its no-mesh return.

    Drives the real call site (not the helper in isolation): with no resolvable
    dp mesh (single card, no parallel axes) the function returns before the FSDP
    wrap, so the wrap must already have happened. The world-size probe needs a
    process group in real runs, so stub it to 1 here.
    """
    from hyper_parallel.models.qwen3_vl_moe import parallelize as pz  # pylint: disable=import-outside-toplevel
    monkeypatch.setattr(pz, "_world_size", lambda: 1)
    model = types.SimpleNamespace(model=types.SimpleNamespace(visual=_Visual()))
    cfg = _cfg("full")
    cfg.model.config_overrides = {"vl": True}
    cfg.train.debug = types.SimpleNamespace(deterministic=False)
    cfg.train.accelerator = types.SimpleNamespace(
        dp_replicate=1, dp_shard=1, tp=1, cp=1, pp=1, ep=1, etp=1, pp_vpp=1,
        pp_mpipe_owner_backward=False,
    )
    _apply_vl_visual_tower(model, mesh=None, cfg=cfg)
    assert all(_wrapped(b) for b in model.model.visual.blocks)


def test_single_card_gate_does_not_skip_visual_ac(monkeypatch):
    """``parallelize_qwen3_vl_moe`` wraps a trainable visual even when the
    single-rank gate (``_should_skip_single_rank_fsdp``) skips the FSDP tower
    helper entirely: the visual AC call is hoisted outside that gate."""
    from hyper_parallel.models.qwen3_vl_moe import parallelize as pz  # pylint: disable=import-outside-toplevel
    monkeypatch.setattr(pz, "_world_size", lambda: 1)
    for name in ("_resolve_qwen3_vl_moe_fsdp_mesh", "_apply_deterministic_moe_sort",
                 "_apply_tp_cp_ep", "_apply_fsdp"):
        monkeypatch.setattr(pz, name, lambda *a, **k: None)
    model = types.SimpleNamespace(model=types.SimpleNamespace(visual=_Visual()))
    cfg = _cfg("full")
    cfg.model.config_overrides = {"vl": True}
    cfg.train.debug = types.SimpleNamespace(deterministic=False)
    cfg.train.accelerator = types.SimpleNamespace(
        dp_replicate=1, dp_shard=1, tp=1, cp=1, pp=1, ep=1, etp=1, pp_vpp=1,
        pp_mpipe_owner_backward=False,
    )
    cfg.train.mixed_precision = types.SimpleNamespace(
        enabled=False, param_dtype="float32", output_dtype=None)
    cfg.train.optimizer = types.SimpleNamespace(loss_aggregation="mean")
    assert pz._should_skip_single_rank_fsdp(cfg)  # pylint: disable=protected-access
    pz.parallelize_qwen3_vl_moe(model, mesh=None, cfg=cfg)
    assert all(_wrapped(b) for b in model.model.visual.blocks)


def test_wrapped_visual_under_no_grad_matches_unwrapped():
    """The MPipe transposed ship path runs the (wrapped) tower under ``no_grad``;
    that must be a clean no-op whose output matches the unwrapped forward."""
    torch.manual_seed(0)
    ref = _Visual()
    ckpt = _Visual()
    ckpt.load_state_dict(ref.state_dict())
    _apply_visual_ac(ckpt, _cfg("full"))
    x = torch.randn(4, 8)
    with torch.no_grad():
        assert torch.allclose(ref(x), ckpt(x), atol=1e-6)


@_RECOMPUTE_BACKWARD_SKIP
def test_recompute_matches_no_recompute():
    """The recompute-wrapped tower gives bit-exact grads vs the plain tower."""
    torch.manual_seed(0)
    ref = _Visual()
    ckpt = _Visual()
    ckpt.load_state_dict(ref.state_dict())
    _apply_visual_ac(ckpt, _cfg("full"))

    x = torch.randn(4, 8, requires_grad=True)
    xr = x.detach().clone().requires_grad_(True)

    ref(x).sum().backward()
    ckpt(xr).sum().backward()

    assert torch.allclose(x.grad, xr.grad, atol=1e-6)
    for pr, pc in zip(ref.parameters(), ckpt.parameters()):
        assert pc.grad is not None
        assert torch.allclose(pr.grad, pc.grad, atol=1e-6)


class _CountingBlock(_Block):
    """A block that counts forward invocations (recompute detection)."""

    def __init__(self, dim: int) -> None:
        """Build a block that counts its forward calls."""
        super().__init__(dim)
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Count the call and apply the block."""
        self.calls += 1
        return super().forward(x)


@_RECOMPUTE_BACKWARD_SKIP
def test_owner_backward_connected_forward_recomputes_wrapped_tower():
    """Owner-backward x visual AC: the emergent composition (!7 + this fix).

    An owner rank retains the tower graph via the executor's
    ``_connected_forward``. With the blocks checkpoint-wrapped, that retained
    graph must recompute each block during the owner backward (2 forward calls
    = 1 forward + 1 recompute; the unwrapped reference stays at 1) while the
    grads stay bit-exact -- the owner's activation hold shrinks to the
    checkpoint boundaries with no numerics change and no owner-specific code.
    """
    from hyper_parallel.platform.torch.pipeline_parallel.mpipe_transpose import (  # pylint: disable=import-outside-toplevel
        MPipeTransposeExecutor,
    )
    torch.manual_seed(0)
    ref = _Visual()
    ref.blocks = nn.ModuleList([_CountingBlock(8) for _ in range(3)])
    ckpt = _Visual()
    ckpt.blocks = nn.ModuleList([_CountingBlock(8) for _ in range(3)])
    ckpt.load_state_dict(ref.state_dict())
    _apply_visual_ac(ckpt, _cfg("full"))

    executor = object.__new__(MPipeTransposeExecutor)
    executor._preprocess = ckpt  # pylint: disable=protected-access

    x = torch.randn(4, 8)
    out = executor._connected_forward((x,), {})  # pylint: disable=protected-access
    assert out.grad_fn is not None  # the owner really retains a graph
    # The repo's CheckpointWrapper delegates unknown attrs to the inner module,
    # so ``calls`` reads through the wrapper.
    assert [b.calls for b in ckpt.blocks] == [1, 1, 1]

    out.sum().backward()  # the owner backward (grad_recv_with_backward's core)
    assert [b.calls for b in ckpt.blocks] == [2, 2, 2]  # forward + recompute

    ref(x).sum().backward()
    assert [b.calls for b in ref.blocks] == [1, 1, 1]  # plain path: no recompute
    for pr, pc in zip(ref.parameters(), ckpt.parameters()):
        assert pc.grad is not None
        assert torch.allclose(pr.grad, pc.grad, atol=1e-6)
