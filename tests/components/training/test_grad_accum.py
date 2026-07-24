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
"""Tests for grad_accum helpers — FSDPModule mocked (design doc 03 §7.1)."""

import os
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

import hyper_models.components.training.grad_accum as grad_accum
from hyper_models.components.training.grad_accum import (
    AutoMFU,
    _dp_cp_all_reduce_sum,
    _infer_peak_tflops,
    _is_rank_0,
    _update_latest_symlink,
    calculate_mfu,
    filter_forward_kwargs,
    get_sync_ctx,
    prepare_after_first_microbatch,
    prepare_for_final_backward,
    prepare_for_grad_accumulation,
    scale_grads_and_clip_grad_norm,
    set_requires_gradient_sync,
)


class FakeFSDP:
    """Fake FSDPModule recording sync toggles."""

    def __init__(self):
        self.sync_calls = []
        self.lazy_init_calls = 0

    def set_requires_gradient_sync(self, flag):
        self.sync_calls.append(flag)

    def reset_lazy_init(self):
        self.lazy_init_calls += 1


@pytest.fixture
def fake_fsdp_cls(monkeypatch):
    monkeypatch.setattr(grad_accum, "FSDPModule", FakeFSDP)
    return FakeFSDP


# ── get_sync_ctx ──

def test_get_sync_ctx_not_optim_step(fake_fsdp_cls):
    mp = fake_fsdp_cls()
    ctx = get_sync_ctx([mp], is_optim_step=False)
    assert isinstance(ctx, nullcontext)
    assert mp.sync_calls == []


def test_get_sync_ctx_defer(fake_fsdp_cls):
    mp = fake_fsdp_cls()
    ctx = get_sync_ctx([mp], is_optim_step=True, defer_fsdp_grad_sync=True)
    assert isinstance(ctx, nullcontext)
    assert mp.sync_calls == [False]  # 关闭同步


def test_get_sync_ctx_final(fake_fsdp_cls):
    mp = fake_fsdp_cls()
    ctx = get_sync_ctx([mp], is_optim_step=True, defer_fsdp_grad_sync=False)
    assert isinstance(ctx, nullcontext)
    assert mp.sync_calls == []  # 保持同步（由 prepare_for_final_backward 开启）


# ── prepare_* ──

def test_prepare_for_grad_accumulation(fake_fsdp_cls):
    mp = fake_fsdp_cls()
    prepare_for_grad_accumulation([mp])
    assert mp.sync_calls == [False]
    assert mp._grad_accum_state == "deferred"


def test_prepare_for_final_backward(fake_fsdp_cls):
    mp = fake_fsdp_cls()
    prepare_for_final_backward([mp])
    assert mp.sync_calls == [True]
    assert mp._grad_accum_state == "final"


def test_prepare_for_final_backward_pp_not_implemented(fake_fsdp_cls):
    with pytest.raises(NotImplementedError):
        prepare_for_final_backward([fake_fsdp_cls(), fake_fsdp_cls()])


def test_prepare_after_first_microbatch(fake_fsdp_cls):
    mp = fake_fsdp_cls()
    prepare_after_first_microbatch([mp])
    assert mp.lazy_init_calls == 1
    assert mp._first_microbatch_done is True


def test_set_requires_gradient_sync(fake_fsdp_cls):
    mp1, mp2 = fake_fsdp_cls(), fake_fsdp_cls()
    set_requires_gradient_sync([mp1, mp2], True)
    assert mp1.sync_calls == [True]
    assert mp2.sync_calls == [True]
    set_requires_gradient_sync([mp1, mp2], False)
    assert mp1.sync_calls == [True, False]


# ── scale_grads_and_clip_grad_norm ──

def _make_model_with_grads(grad_value=4.0):
    model = nn.Linear(2, 2, bias=False)
    for p in model.parameters():
        p.grad = torch.full_like(p, grad_value)
    return model


def test_scale_grads_and_clip_grad_norm():
    model = _make_model_with_grads(grad_value=4.0)
    grad_norm = scale_grads_and_clip_grad_norm([model], 1e6, num_label_tokens=2)
    # 4 / 2 = 2 → 每个元素为 2；norm = sqrt(4 * 2^2) = 4
    for p in model.parameters():
        assert torch.allclose(p.grad, torch.full_like(p, 2.0))
    assert grad_norm == pytest.approx(4.0)


def test_scale_grads_no_num_tokens():
    model = _make_model_with_grads(grad_value=4.0)
    scale_grads_and_clip_grad_norm([model], 1e6, num_label_tokens=None)
    for p in model.parameters():
        assert torch.allclose(p.grad, torch.full_like(p, 4.0))  # 未除


def test_scale_grads_zero_guard():
    model = _make_model_with_grads(grad_value=4.0)
    scale_grads_and_clip_grad_norm([model], 1e6, num_label_tokens=0)
    for p in model.parameters():
        assert torch.allclose(p.grad, torch.full_like(p, 4.0))  # 零值保护：跳过除法


# ── filter_forward_kwargs ──

def test_filter_forward_kwargs():
    class Model(nn.Module):
        def forward(self, input_ids, labels=None):
            return input_ids

    batch = {"input_ids": 1, "labels": 2, "extra_key": 3}
    assert filter_forward_kwargs(Model(), batch) == {"input_ids": 1, "labels": 2}


def test_filter_forward_kwargs_uninspectable():
    model = MagicMock(spec=nn.Module)
    model.forward = None  # inspect.signature(None) → TypeError → 兜底
    batch = {"a": 1, "b": 2}
    assert filter_forward_kwargs(model, batch) == batch


# ── calculate_mfu / _infer_peak_tflops ──

def test_calculate_mfu():
    mfu = calculate_mfu(tps=1000, flops_per_token=1e8, peak_tflops=100, world_size=8)
    # (1000 * 1e8) / (100 * 8 * 1e12) = 1.25e-4
    assert mfu == pytest.approx(1.25e-4)


def test_calculate_mfu_zero_peak():
    assert calculate_mfu(1000, 1e8, 0.0, 8) == 0.0
    assert calculate_mfu(1000, 1e8, -1.0, 8) == 0.0


def test_calculate_mfu_clamp():
    assert calculate_mfu(1e12, 1e12, 100, 8) == 1.0


@pytest.mark.parametrize("name,expected", [
    ("NVIDIA H100 80GB HBM3", 989.0),
    ("NVIDIA A100-SXM4-80GB", 312.0),
    ("NVIDIA H20", 148.0),
    ("Tesla V100-SXM2-32GB", 125.0),
    ("NVIDIA GeForce RTX 4090", 330.0),
])
def test_infer_peak_tflops(name, expected):
    assert _infer_peak_tflops(name) == expected


def test_infer_peak_tflops_unknown():
    assert _infer_peak_tflops("Some Unknown Accelerator") == 200.0


# ── AutoMFU ──

def test_auto_mfu_from_config():
    model = nn.Linear(10, 10)  # 110 params
    mfu = AutoMFU.from_config(model)
    assert mfu.flops_per_token == pytest.approx(6.0 * 110)
    assert mfu.peak_tflops > 0  # 无 CUDA → 默认 200.0


# ── _is_rank_0 ──

def test_is_rank_0_dist_not_init():
    assert not torch.distributed.is_initialized()
    assert _is_rank_0() is True


# ── _update_latest_symlink ──

def test_update_latest_symlink(tmp_path):
    ckpt_dir = str(tmp_path)
    step_dir = os.path.join(ckpt_dir, "epoch_0_step_1")
    os.makedirs(step_dir)
    _update_latest_symlink(ckpt_dir, step_dir)
    latest = os.path.join(ckpt_dir, "LATEST")
    assert os.path.islink(latest)
    assert os.readlink(latest) == "epoch_0_step_1"  # 相对路径

    # 原子更新：指向新目录
    step_dir2 = os.path.join(ckpt_dir, "epoch_0_step_2")
    os.makedirs(step_dir2)
    _update_latest_symlink(ckpt_dir, step_dir2)
    assert os.readlink(latest) == "epoch_0_step_2"


# ── _dp_cp_all_reduce_sum ──

def test_dp_cp_all_reduce_sum_mock():
    mesh = MagicMock()
    mesh.get_group.return_value = "fake_group"
    with patch.object(grad_accum, "dist") as mock_dist:
        mock_dist.ReduceOp.SUM = torch.distributed.ReduceOp.SUM
        result = _dp_cp_all_reduce_sum(5, mesh)
        mock_dist.all_reduce.assert_called_once()
        args, kwargs = mock_dist.all_reduce.call_args
        assert kwargs["group"] == "fake_group"
        assert torch.is_tensor(result)
        assert result.item() == 5  # mock 不改变数值


def test_dp_cp_all_reduce_sum_no_dist():
    # dist 未初始化且 mesh 为 None → 原样返回（wrap 为 tensor）
    result = _dp_cp_all_reduce_sum(7, None)
    assert result.item() == 7
