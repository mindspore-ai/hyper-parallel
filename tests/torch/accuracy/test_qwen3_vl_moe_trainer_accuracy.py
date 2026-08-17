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
"""Trainer-driven real-weight accuracy STs for Qwen3-VL-MoE PP / MPipe.

Drives the real ``scripts/train_vl.py`` on the on-disk ``Qwen3-VL-30B-A3B``
checkpoint reduced to a few text layers, and pins each PP schedule against an
FSDP reference on identical deterministic ``vl_dummy`` data (``atol=2e-3``
over 20 steps). FSDP stands in for single-card (a 4-layer VL single-card
reference does not fit one die) and was itself cross-validated against Hugging
Face ``transformers`` on the same reduced model.

Two MPipe tiers: FROZEN-tower tests (the production shape, incl. the
``data.load: single`` MB>PP variant) compare against the FSDP reference;
trainable-tower tests un-freeze the visual tower and pin MPipe (stage-0
backward or owner-does-backward) against plain 1F1B training the same tower.

Skips (exit 0) when the checkpoint is missing or too few NPUs are visible.
"""
import os

from tests.common.mark_utils import arg_mark
from tests.torch.accuracy.trainer_align_utils import (
    assert_trajectories_match,
    base_config,
    checkpoint_present,
    run_trainer_losses,
)

_MODEL = "qwen3_vl_moe"
# Point HP_QWEN3_VL_MOE_CKPT at a local Qwen3-VL-30B-A3B checkpoint to run these
# STs; without it ``checkpoint_present`` skips them.
_CKPT = os.environ.get("HP_QWEN3_VL_MOE_CKPT", "")
# PP needs > 3 layers (DeepStack pinned to stage 0); a 4-layer VL single-card
# reference OOMs, so PP is asserted against the FSDP-2 reference at 4 layers.
_PP_LAYERS = 4


def _accel_config(accel: dict, layers: int) -> dict:
    cfg = base_config(_MODEL, _CKPT, layers, vl=True)
    cfg["train"]["accelerator"].update(accel)
    return cfg


def _trainable_tower_config(accel: dict, layers: int) -> dict:
    """Like ``_accel_config`` but with a TRAINABLE visual tower.

    The frozen-tower configs above have no tower backward, so they can't exercise
    the MPipe stage-0 backward / owner-does-backward paths. Un-freezing the tower makes
    the transposed visual block a trainable module -- a plain pp-replica at
    ``dp_shard==1``, or FSDP-sharded at ``dp_shard>1`` (owner-backward composes
    with both: FSDP reduces the tower grad over the dp axes, MPipe over pp).
    """
    cfg = base_config(_MODEL, _CKPT, layers, vl=True)
    cfg["model"]["freeze_modules"] = []  # train the visual tower
    cfg["train"]["accelerator"].update(accel)
    return cfg


# PP=2 / MB=2 configs over a trainable tower: plain 1F1B (inline visual) is the
# reference; MPipe transposes the visual tower and trains it via the stage-0
# backward or, opt-in, owner-does-backward.
_PP_1F1B = {"pp": 2, "pp_micro_batch_num": 2, "pp_schedule": "1f1b"}
_PP_MPIPE = {"pp": 2, "pp_micro_batch_num": 2, "pp_schedule": "mpipe",
             "pp_mpipe_transpose_layers": "visual"}


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="unessential")
def test_qwen3_vl_moe_trainer_pp():
    """PP=2 (1F1B, DeepStack on stage 0) vs the FSDP-2 reference at 4 layers."""
    if not checkpoint_present(_CKPT):
        print("Skip vlmoe_trainer_pp: checkpoint missing.")
        return
    # Must match the PP micros' per-forward batch shape: the model is not
    # bit-batch-invariant and the MoE router amplifies that into loss drift.
    ref_cfg = _accel_config({"dp_shard": 2}, _PP_LAYERS)
    ref_cfg["train"]["micro_batch_size"] = 2
    ref = run_trainer_losses(ref_cfg, nproc=2, master_port=24220, vl=True)
    pp = run_trainer_losses(
        _accel_config({"pp": 2, "pp_micro_batch_num": 2, "pp_schedule": "1f1b"}, _PP_LAYERS),
        nproc=2, master_port=24222, vl=True)
    assert_trajectories_match("vlmoe_trainer_pp_vs_fsdp", ref, pp)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="unessential")
def test_qwen3_vl_moe_trainer_pp_mpipe_frozen():
    """PP=2 MPipe (transposed FROZEN visual tower — the production shape) vs the
    FSDP-2 reference at 4 layers.

    The frozen-tower transpose ships only the visual payload (no broadcast, no
    tower backward); its loss must track the FSDP reference exactly like plain
    1F1B does — the end-to-end precision check for the default mpipe configs.
    """
    if not checkpoint_present(_CKPT):
        print("Skip vlmoe_trainer_pp_mpipe_frozen: checkpoint missing.")
        return
    # 2-sample micros: match the reference's per-forward batch shape (see
    # test_qwen3_vl_moe_trainer_pp).
    ref_cfg = _accel_config({"dp_shard": 2}, _PP_LAYERS)
    ref_cfg["train"]["micro_batch_size"] = 2
    ref = run_trainer_losses(ref_cfg, nproc=2, master_port=24244, vl=True)
    mpipe = run_trainer_losses(_accel_config(_PP_MPIPE, _PP_LAYERS),
                               nproc=2, master_port=24246, vl=True)
    assert_trajectories_match("vlmoe_trainer_pp_mpipe_frozen_vs_fsdp", ref, mpipe)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="unessential")
def test_qwen3_vl_moe_trainer_pp_mpipe_frozen_single_dataload():
    """PP=2 MPipe (frozen tower) with ``data.load: single`` and MB=4 > PP=2 vs
    the FSDP-2 reference at 4 layers.

    MB > PP leaves micros >= NT untransposed: they load and encode inline on
    stage 0 via the kept body DATA_LOAD steps (the frozen-tower in-schedule
    dataload path), while micros < NT arrive via the prefix transpose. Both
    kinds must contribute to the same loss trajectory as the FSDP reference —
    the end-to-end check that per-rank-owned loading feeds every micro exactly
    once.
    """
    if not checkpoint_present(_CKPT):
        print("Skip vlmoe_trainer_pp_mpipe_frozen_single_dataload: checkpoint missing.")
        return
    ref = run_trainer_losses(_accel_config({"dp_shard": 2}, _PP_LAYERS),
                             nproc=2, master_port=24248, vl=True)
    single_cfg = _accel_config({**_PP_MPIPE, "pp_micro_batch_num": 4}, _PP_LAYERS)
    # gbs=4 / micro=1 / dp=1: single-load consumption (mb * micro * dp) == gbs.
    single_cfg["data"]["load"] = "single"
    single = run_trainer_losses(single_cfg, nproc=2, master_port=24250, vl=True)
    assert_trajectories_match("vlmoe_trainer_pp_mpipe_frozen_single_vs_fsdp", ref, single)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="unessential")
def test_qwen3_vl_moe_trainer_pp_mpipe_stage0_backward():
    """PP=2 MPipe (transposed TRAINABLE visual tower, stage-0 backward) vs PP=2
    1F1B (inline visual, same trainable tower).

    Validates the stage-0 backward actually trains the transposed tower (the
    feature gradient is read from the placed ``mpipe_visual`` tensors, not
    ``arg_mbs``) -- its loss must track 1F1B within ``atol`` on identical data.
    """
    if not checkpoint_present(_CKPT):
        print("Skip vlmoe_trainer_pp_mpipe_stage0_backward: checkpoint missing.")
        return
    ref = run_trainer_losses(_trainable_tower_config(_PP_1F1B, _PP_LAYERS),
                             nproc=2, master_port=24228, vl=True)
    mpipe = run_trainer_losses(_trainable_tower_config(_PP_MPIPE, _PP_LAYERS),
                               nproc=2, master_port=24230, vl=True)
    assert_trajectories_match("vlmoe_trainer_pp_mpipe_stage0_backward_vs_1f1b", ref, mpipe)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="unessential")
def test_qwen3_vl_moe_trainer_pp_mpipe_owner_backward():
    """PP=2 MPipe owner-does-backward (transposed TRAINABLE visual tower) vs PP=2
    1F1B (inline visual, same trainable tower).

    owner-backward distributes the tower backward to the owner ranks and
    SUM-reduces the tower grads to stage 0; the result must match the inline-tower
    1F1B run within ``atol`` on identical data (the end-to-end precision check).
    """
    if not checkpoint_present(_CKPT):
        print("Skip vlmoe_trainer_pp_mpipe_owner_backward: checkpoint missing.")
        return
    ref = run_trainer_losses(_trainable_tower_config(_PP_1F1B, _PP_LAYERS),
                             nproc=2, master_port=24232, vl=True)
    owner = run_trainer_losses(
        _trainable_tower_config({**_PP_MPIPE, "pp_mpipe_owner_backward": True}, _PP_LAYERS),
        nproc=2, master_port=24234, vl=True)
    assert_trajectories_match("vlmoe_trainer_pp_mpipe_owner_backward_vs_1f1b", ref, owner)
