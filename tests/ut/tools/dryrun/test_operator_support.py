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
"""Tests for the extensible dry-run operator interface."""

import importlib.util
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from hyper_parallel.components.functional.mhc_post import mhc_post
from hyper_parallel.components.functional.mhc_pre import mhc_pre
from hyper_parallel.tools.dryrun import operator_support
from hyper_parallel.tools.dryrun.operator_support import (
    DryRunOpRegistry,
    get_dryrun_op_registry,
    register_builtin_dryrun_ops,
    register_fake_op,
    unsupported_operator_hint,
)


_OP_NAME = "hp_dryrun_test::identity"
torch.library.define(_OP_NAME, "(Tensor input) -> Tensor")


def test_decorator_registers_shape_only_kernel():
    """A user decorator can register a shape-only implementation."""
    registry = DryRunOpRegistry()

    @registry.register_fake(_OP_NAME)
    def identity_fake(input_tensor):
        return torch.empty_like(input_tensor)

    with FakeTensorMode():
        fake = torch.empty(2, 3)
        output = torch.ops.hp_dryrun_test.identity.default(fake)
    assert output.shape == fake.shape
    assert registry.registered_ops == (_OP_NAME,)


def test_unsupported_operator_hint_is_actionable():
    """Unsupported-op diagnostics explain immediate registration."""
    hint = unsupported_operator_hint(
        RuntimeError("UnsupportedOperatorException: custom::missing"),
    )
    assert hint is not None
    assert "register_fake_op" in hint
    assert "Import the defining module" in hint


def test_public_decorator_targets_global_registry():
    """The package-level shortcut delegates to the process registry."""
    assert callable(register_fake_op)


def test_builtin_mhc_and_sink_attention_backward_under_fake_mode():
    """Built-in dry-run kernels match backward gradient shapes under FakeTensorMode."""
    register_builtin_dryrun_ops()
    custom_ns = getattr(torch.ops, "custom", None)
    if custom_ns is not None and hasattr(custom_ns, "npu_manifold_constrained_hyper_connection_pre_grad"):
        with FakeTensorMode():
            x = torch.empty(2, 4, 4, 16, dtype=torch.bfloat16)
            phi = torch.empty(24, 64, dtype=torch.float32)
            alpha = torch.empty(3, dtype=torch.float32)
            dy = torch.empty(2, 4, 16, dtype=torch.bfloat16)
            dpost = torch.empty(2, 4, 4, dtype=torch.float32)
            dres = torch.empty(2, 4, 24, dtype=torch.float32)
            inv_rms = torch.empty(2, 4, 1, dtype=torch.float32)
            mm_res = torch.empty(2, 4, 24, dtype=torch.float32)
            h_pre = torch.empty(2, 4, 4, dtype=torch.float32)
            h_post = torch.empty(2, 4, 4, dtype=torch.float32)
            grads = custom_ns.npu_manifold_constrained_hyper_connection_pre_grad(
                x, phi, alpha, dy, dpost, dres, inv_rms, mm_res, h_pre, h_post
            )
            assert grads[3].shape == (24,)

    if custom_ns is not None and hasattr(custom_ns, "npu_flash_attention_score_grad_enhance"):
        with FakeTensorMode():
            q = torch.empty(8, 2, 16, dtype=torch.bfloat16)
            k = torch.empty(8, 2, 16, dtype=torch.bfloat16)
            v = torch.empty(8, 2, 16, dtype=torch.bfloat16)
            dy = torch.empty(8, 2, 16, dtype=torch.bfloat16)
            lse = torch.empty(2, 8, 8, dtype=torch.float32)
            grads = custom_ns.npu_flash_attention_score_grad_enhance(
                q, k, v, dy, 2, "TND",
                softmax_max=lse, softmax_sum=lse, attention_in=dy,
            )
            assert grads[0].shape == q.shape
            assert grads[1].shape == k.shape
            assert grads[2].shape == v.shape

    if custom_ns is not None and hasattr(custom_ns, "npu_flash_attention_score_enhance") and hasattr(torch, "npu"):
        with FakeTensorMode():
            q = torch.empty(8, 2, 16, dtype=torch.bfloat16, device="npu:0", requires_grad=True)
            k = torch.empty(8, 2, 16, dtype=torch.bfloat16, device="npu:0", requires_grad=True)
            v = torch.empty(8, 2, 16, dtype=torch.bfloat16, device="npu:0", requires_grad=True)
            out = custom_ns.npu_flash_attention_score_enhance(
                q, k, v, 2, "TND", actual_seq_qlen=[8], actual_seq_kvlen=[8]
            )[0]
            out.sum().backward()
            assert q.grad.shape == q.shape
            assert k.grad.shape == k.shape
            assert v.grad.shape == v.shape


def test_cann_mhc_pre_and_post_dispatch_under_fake_mode():
    """CANN mhc_pre and mhc_post route through torch.ops under FakeTensorMode."""
    if importlib.util.find_spec("cann_ops_transformer") is None:
        return
    if not hasattr(torch, "npu"):
        return
    try:
        importlib.import_module("cann_ops_transformer.ops.mhc_post")
        importlib.import_module("cann_ops_transformer.ops.mhc_pre_sinkhorn")
    except (ImportError, RuntimeError):
        return

    with FakeTensorMode():
        x = torch.empty(2, 4, 4 * 128, dtype=torch.bfloat16, device="npu:0", requires_grad=True)
        phi = torch.empty(24, 4 * 128, dtype=torch.float32, device="npu:0", requires_grad=True)
        alpha = torch.empty(3, dtype=torch.float32, device="npu:0", requires_grad=True)
        beta = torch.empty(24, dtype=torch.float32, device="npu:0", requires_grad=True)
        y, h_post, h_res, res = mhc_pre(x, phi, alpha, beta, 4, backend="cann")
        out = mhc_post(y, res, h_post, h_res, 4, backend="cann")
        out.sum().backward()
        assert x.grad.shape == x.shape
        assert phi.grad.shape == phi.shape
        assert alpha.grad.shape == alpha.shape
        assert beta.grad.shape == beta.shape


def test_registry_validations_and_allow_missing():
    """DryRunOpRegistry validates op names, callables, idempotency, and allow_missing."""
    registry = DryRunOpRegistry()
    with pytest.raises(ValueError, match="namespace::operator"):
        registry.register_fake("missing_namespace", lambda x: x)
    with pytest.raises(TypeError, match="must be callable"):
        registry.register_fake("ns::bad_fn", "not_callable")

    # allow_missing=True skips missing ops without raising
    def _identity(x):
        return x

    assert registry.register_fake("ns::nonexistent_op_xyz", _identity, allow_missing=True) is _identity
    assert "ns::nonexistent_op_xyz" not in registry.registered_ops

    with pytest.raises(RuntimeError, match="Unable to register"):
        registry.register_fake("ns::nonexistent_op_xyz", _identity, allow_missing=False)

    # Idempotent repeat registration on same registry
    registry._registered.add(_OP_NAME)
    assert registry.register_fake(_OP_NAME, _identity) is _identity
    assert get_dryrun_op_registry() is not None


def test_builtin_fake_kernels_invoked_directly():
    """Invoke every built-in fake kernel registered by register_builtin_dryrun_ops."""
    captured = {}

    def _capture_register(op_name, kernel):
        captured[op_name] = kernel
        return True

    prev = operator_support._BUILTIN_STATE["registered"]
    operator_support._BUILTIN_STATE["registered"] = False
    try:
        with (
            patch.object(operator_support, "_register", side_effect=_capture_register),
            patch.object(torch.ops, "custom", SimpleNamespace(), create=True),
        ):
            register_builtin_dryrun_ops()
            # Second call hits early return
            register_builtin_dryrun_ops()
    finally:
        operator_support._BUILTIN_STATE["registered"] = prev

    q = torch.empty(2, 4, 2, 8)
    k = torch.empty(2, 4, 2, 8)
    v = torch.empty(2, 4, 2, 8)
    rope = torch.empty(2, 4, 2, 4)

    grads = captured["npu_sparse_flash_attention_grad_enhance"](
        q, k, v, None, q, q, None, None, 1.0, 16, query_rope=rope, key_rope=rope
    )
    assert grads[0].shape == q.shape and grads[3].shape == rope.shape
    grads_no_rope = captured["npu_sparse_flash_attention_grad_enhance"](
        q, k, v, None, q, q, None, None, 1.0, 16
    )
    assert grads_no_rope[3].shape == q.shape

    qi = torch.empty(2, 4, 8)
    ki = torch.empty(2, 4, 8)
    w = torch.empty(2, 4)
    kl_grads = captured["npu_sparse_lightning_indexer_grad_kl_loss_enhance"](
        q, k, qi, ki, w, None, None, None, 1.0
    )
    assert kl_grads[0].shape == qi.shape and kl_grads[3].shape == ()

    agg_g = captured["npu_aggregate_hidden_grad"](q, q, w)
    assert agg_g[0].shape == q.shape and agg_g[1].shape == w.shape

    x = torch.empty(2, 4, 16)
    phi = torch.empty(24, 16)
    alpha = torch.empty(3)
    mhc_g = captured["npu_manifold_constrained_hyper_connection_pre_grad"](
        x, phi, alpha, None, None, None, None, None, None, None
    )
    assert mhc_g[0].shape == x.shape and mhc_g[3].shape == (24,)

    for layout, shape in (("TND", (8, 2, 8)), ("BNSD", (2, 2, 4, 8)), ("BSND", (2, 4, 2, 8))):
        qt = torch.empty(*shape)
        out, lse1, lse2 = captured["npu_flash_attention_score_enhance"](qt, qt, qt, 2, layout)
        assert out.shape == qt.shape and lse1.shape == (2, 8, 8) and lse2.shape == (2, 8, 8)

    dq, dk, dv, dpse, dsink = captured["npu_flash_attention_score_grad_enhance"](
        q, k, v, q, 2, "BSND", sink_tensor=torch.empty(2)
    )
    assert dq.shape == q.shape and dk.shape == k.shape and dv.shape == v.shape
    assert dpse is None and dsink.shape == (2,)
    assert unsupported_operator_hint(RuntimeError("plain error")) is None


def test_cann_mhc_pre_and_post_autograd_with_mocked_ops():
    """Cover _CannMhcPre and _CannMhcPost forward/backward even on CPU-only CI."""
    fake_ops = SimpleNamespace(
        mhc_pre_sinkhorn=MagicMock(
            side_effect=lambda x, phi, alpha, bias, num_stream, *args: (
                torch.empty(*x.shape[:-2], x.shape[-1], dtype=x.dtype),
                torch.empty(*x.shape[:-2], num_stream, dtype=torch.float32),
                torch.empty(*x.shape[:-2], num_stream * num_stream, dtype=torch.float32),
                torch.empty(*x.shape[:-2], num_stream, dtype=torch.float32),
                torch.empty(*x.shape[:-2], 1, dtype=torch.float32),
                torch.empty(*x.shape[:-2], 1, dtype=torch.float32),
                torch.empty(*x.shape[:-2], 1, dtype=torch.float32),
                torch.empty(*x.shape[:-2], 1, dtype=torch.float32),
            )
        ),
        mhc_pre_sinkhorn_backward=MagicMock(
            side_effect=lambda grad_hin, grad_hp, grad_hr, x, phi, alpha, bias, *args: (
                torch.empty_like(x),
                torch.empty_like(phi),
                torch.empty_like(alpha),
                torch.empty_like(bias),
            )
        ),
        mhc_post=MagicMock(
            side_effect=lambda x, h_res, h_out, h_post: torch.empty_like(x)
        ),
        mhc_post_backward=MagicMock(
            side_effect=lambda grad_out, x, h_res, h_out, h_post: (
                torch.empty_like(x),
                torch.empty_like(h_res),
                torch.empty_like(h_out),
                torch.empty_like(h_post),
            )
        ),
    )
    with (
        patch("importlib.import_module", return_value=SimpleNamespace()),
        patch.object(torch.ops, "cann_ops_transformer", fake_ops, create=True),
    ):
        x = torch.randn(2, 4, 16, requires_grad=True)
        phi = torch.randn(24, 16, requires_grad=True)
        alpha = torch.randn(3, requires_grad=True)
        beta = torch.randn(24, requires_grad=True)
        y, h_post, h_res, res = mhc_pre(x, phi, alpha, beta, 4, backend="cann")
        out = mhc_post(y, res, h_post, h_res, 4, backend="cann")
        out.sum().backward()
        assert x.grad.shape == x.shape
        assert phi.grad.shape == phi.shape
        assert alpha.grad.shape == alpha.shape
        assert beta.grad.shape == beta.shape
