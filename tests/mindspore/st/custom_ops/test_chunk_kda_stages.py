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
"""NPU numerical coverage for explicit KDA stages, optional when FLA is absent.

Run with HYPER_PARALLEL_PLATFORM=mindspore after sourcing CANN and FLA OPP.
Tests skip missing stage APIs, but never swallow kernel or accuracy failures.
CPU reference uses PyTorch only for independent FP32 recurrence/autograd.
"""
# Optional dependency probes intentionally precede package imports.
# pylint: disable=wrong-import-position
import os

import numpy as np
import pytest

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

ms = pytest.importorskip("mindspore")
torch = pytest.importorskip("torch")

from hyper_parallel.custom_ops.experimental import (  # noqa: E402
    npu_chunk_kda,
    npu_chunk_kda_fwd_prepare,
    npu_chunk_kda_pre_process_fwd,
    npu_chunk_kda_fwd_h,
    npu_chunk_kda_fwd_finalize,
    npu_chunk_kda_bwd_prepare,
    npu_chunk_kda_pre_process_bwd,
    npu_chunk_kda_bwd_dhu,
    npu_chunk_kda_bwd_finalize,
    npu_chunk_kda_merge_fwd_bwd,
)
from hyper_parallel.platform.mindspore.custom_ops import custom_op_impl  # noqa: E402
from tests.common.mark_utils import arg_mark  # noqa: E402

_APIS = (
    "ChunkKdaFwdPrepare", "ChunkFwdH", "ChunkKdaFwdFinalize", "ChunkKdaBwdPrepare",
    "ChunkGatedDeltaRuleBwdDhu", "ChunkKdaBwdFinalize",
)


@pytest.fixture(scope="module", autouse=True)
def require_fla_stages() -> None:
    """Probe both native symbols before any tensor/kernel call."""
    extension = custom_op_impl._custom_ops  # pylint: disable=protected-access
    probe = getattr(extension, "has_chunk_kda_stage", None)
    if probe is None:
        pytest.skip("Loaded HP extension has no Chunk KDA stage bridges")
    missing = ["aclnn" + name for name in _APIS if not probe("aclnn" + name)]
    if missing:
        pytest.skip("Missing optional FLA-NPU APIs: " + ", ".join(missing))
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")


def _require_apis(*names):
    extension = custom_op_impl._custom_ops  # pylint: disable=protected-access
    missing = ["aclnn" + name for name in names if not extension.has_chunk_kda_stage("aclnn" + name)]
    if missing:
        pytest.skip("Missing optional FLA-NPU APIs: " + ", ".join(missing))


def _np(tensor):
    return tensor.astype(ms.float32).asnumpy()


def _check(actual, expected, label, relative=.04, absolute=3e-8):
    """Require finite values, relative RMS error and cosine on nonzero outputs."""
    actual = _np(actual) if isinstance(actual, ms.Tensor) else np.asarray(actual)
    expected = expected.detach().numpy() if isinstance(expected, torch.Tensor) else np.asarray(expected)
    assert actual.shape == expected.shape, (label, actual.shape, expected.shape)
    assert np.isfinite(actual).all(), label
    error = np.linalg.norm((actual - expected).reshape(-1))
    norm = np.linalg.norm(expected.reshape(-1))
    rms_error = error / np.sqrt(actual.size)
    assert error / max(norm, 1e-12) < relative or rms_error < absolute, (label, error, norm, rms_error)
    if norm > 1e-12:
        cosine = np.dot(actual.reshape(-1), expected.reshape(-1)) / max(
            np.linalg.norm(actual.reshape(-1)) * norm, 1e-30)
        assert cosine > .995, (label, cosine)


def _inputs(tokens, heads=1):
    rng = np.random.default_rng(42 + tokens)
    shape = (1, heads, tokens, 128)
    q, k, v = [ms.Tensor(rng.normal(0, .05, shape).astype(np.float32), ms.bfloat16) for _ in range(3)]
    g = ms.Tensor(rng.normal(-1., .2, shape).astype(np.float32))
    beta = ms.Tensor(rng.uniform(.2, .8, shape[:-1]).astype(np.float32))
    a_log = ms.Tensor(np.full(heads, -2., np.float32))
    bias = ms.Tensor(np.full(heads * 128, -.1, np.float32))
    d_o = ms.Tensor(rng.normal(0, .1, shape).astype(np.float32), ms.bfloat16)
    return (q, k, v, g, beta, a_log, bias), d_o


def _reference(inputs, d_o, cu=None, h0=None, dht=None, norm=False, sigmoid=False, negative=False):
    """Independent token recurrence; state decay precedes the delta update."""
    leaves = [torch.tensor(_np(tensor), requires_grad=True) for tensor in inputs]
    q, k, v, g, beta, a_log, bias = leaves
    if norm:
        q = q * (q.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
        k = k * (k.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
    if sigmoid:
        beta = beta.sigmoid() * (2 if negative else 1)
    h0_leaf = None if h0 is None else torch.tensor(_np(h0), requires_grad=True)
    gate = -5. * torch.sigmoid(a_log.exp().reshape(1, -1, 1, 1) *
                               (g + bias.reshape(1, g.shape[1], 1, 128)))
    boundaries = cu or [0, q.shape[2]]
    outputs, states = [], []
    for sequence, (begin, end) in enumerate(zip(boundaries, boundaries[1:])):
        state = torch.zeros((1, v.shape[1], 128, 128)) if h0 is None else h0_leaf[sequence:sequence + 1]
        for token in range(begin, end):
            state = state * gate[:, :, token].exp().unsqueeze(-1)
            kt = k[:, :, token]
            residual = (v[:, :, token] - (kt.unsqueeze(-1) * state).sum(-2)) * beta[:, :, token, None]
            state = state + kt.unsqueeze(-1) * residual.unsqueeze(-2)
            outputs.append((q[:, :, token].unsqueeze(-1) * state).sum(-2) * 128**-.5)
        states.append(state[0])
    out = torch.stack(outputs, dim=2)
    final = torch.stack(states)
    loss = (out * torch.tensor(_np(d_o))).sum()
    if dht is not None:
        loss = loss + (final * torch.tensor(_np(dht))).sum()
    targets = leaves + ([] if h0 is None else [h0_leaf])
    gradients = torch.autograd.grad(loss, targets)
    return out, final, gradients


def _stage_chain(inputs, d_o, cu=None, h0=None, dht=None, state_v_first=False,
                 norm=False, sigmoid=False, negative=False):
    q, k, v, g, beta, a_log, bias = inputs
    cache = npu_chunk_kda_fwd_prepare(
        q, k, v, g, beta, 128**-.5, a_log=a_log, dt_bias=bias,
        safe_gate=True, use_gate_in_kernel=True, cu_seqlens=cu,
        use_qk_l2norm_in_kernel=norm, use_beta_sigmoid_in_kernel=sigmoid,
        allow_neg_eigval=negative,
    )
    gk, aqk, akk, w, _, qg, kg, qs = cache[:8]
    h, vn, final = npu_chunk_kda_fwd_h(
        kg, w, cache[4], gk=gk, initial_state=h0, output_final_state=True,
        cu_seqlens=cu, state_v_first=state_v_first,
    )
    out = npu_chunk_kda_fwd_finalize(qs, aqk, vn, h, output_layout="BNSD",
                                      state_v_first=state_v_first, cu_seqlens=cu)
    da, dv_local, dq_raw = npu_chunk_kda_bwd_prepare(
        aqk, vn, d_o, h, scale=128**-.5, cu_seqlens=cu, state_v_first=state_v_first,
    )
    # Dhu's boundary states require the relay dtype, not FwdH's FP32 h0.
    h0_relay = None if h0 is None else h0.astype(q.dtype)
    dh, dh0, dv_scan = npu_chunk_kda_bwd_dhu(
        qg, kg, w, d_o, dv_local, 128**-.5, gk=gk, h0=h0_relay, dht=dht,
        cu_seqlens=cu, state_v_first=state_v_first,
    )
    grads = npu_chunk_kda_bwd_finalize(
        cache[8] if norm else q, cache[9] if norm else k, v, gk, g,
        cache[12] if sigmoid else beta, a_log, bias, akk, vn, h, dh, dv_scan, da, dq_raw,
        scale=128**-.5, safe_gate=True, cu_seqlens=cu, state_v_first=state_v_first,
        q_rstd=cache[10] if norm else None, k_rstd=cache[11] if norm else None,
        use_beta_sigmoid_in_kernel=sigmoid, allow_neg_eigval=negative,
    )
    return out, final, grads, dh0, cache


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard",
          essential_mark="essential")
@pytest.mark.parametrize("tokens,cu", [(64, None), (65, None), (66, [0, 1, 66])])
def test_stage_chain_reference_and_total_interface(tokens, cu):
    """CP=1: output and all seven gradients vs CPU; forward vs total API."""
    inputs, d_o = _inputs(tokens)
    out, final, grads, dh0, _ = _stage_chain(inputs, d_o, cu)
    ref_out, ref_final, ref_grads = _reference(inputs, d_o, cu)
    _check(out, ref_out, "out")
    _check(final, ref_final, "final_state")
    assert dh0 is None
    # Explicit finalize returns beta before gate; public input order is gate then beta.
    for name, actual, expected in zip(("dq", "dk", "dv", "dbeta", "dg", "dA", "dbias"),
                                      grads, [ref_grads[i] for i in (0, 1, 2, 4, 3, 5, 6)]):
        _check(actual, expected, name)
    q, k, v, g, beta, a_log, bias = inputs
    total, _ = npu_chunk_kda(q, k, v, g, beta, 128**-.5, safe_gate=True,
                             a_log=a_log, dt_bias=bias, cu_seqlens=cu)
    # Existing total API returns token-major output even for BNSD inputs.
    _check(out, total.transpose(0, 2, 1, 3), "stage_vs_total", relative=.01)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard",
          essential_mark="essential")
def test_stage_initial_and_terminal_state_gradients():
    """Explicit stages propagate supplied h0/dht; total autograd remains unchanged."""
    inputs, d_o = _inputs(65)
    rng = np.random.default_rng(19)
    h0 = ms.Tensor(rng.normal(0, .01, (1, 1, 128, 128)).astype(np.float32), ms.bfloat16)
    dht = ms.Tensor(rng.normal(0, .01, h0.shape).astype(np.float32), ms.bfloat16)
    out, final, grads, dh0, _ = _stage_chain(inputs, d_o, h0=h0, dht=dht)
    ref_out, ref_final, ref_grads = _reference(inputs, d_o, h0=h0, dht=dht)
    _check(out, ref_out, "out_h0")
    _check(final, ref_final, "final_h0")
    _check(dh0, ref_grads[-1], "dh0")
    for name, actual, expected in zip(("dq", "dk", "dv", "dbeta", "dg", "dA", "dbias"),
                                      grads, [ref_grads[i] for i in (0, 1, 2, 4, 3, 5, 6)]):
        _check(actual, expected, name)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard",
          essential_mark="essential")
@pytest.mark.parametrize("norm,sigmoid,negative", [(True, False, False), (False, True, False), (True, True, True)])
def test_stage_fused_norm_and_beta_gradients(norm, sigmoid, negative):
    """Saved q_hat/rstd and beta_eff yield gradients of the original inputs."""
    inputs, d_o = _inputs(65)
    out, final, grads, _, _ = _stage_chain(inputs, d_o, norm=norm, sigmoid=sigmoid, negative=negative)
    expected, expected_state, expected_grads = _reference(
        inputs, d_o, norm=norm, sigmoid=sigmoid, negative=negative,
    )
    _check(out, expected, "fused_out")
    _check(final, expected_state, "fused_state")
    for name, actual, reference in zip(("dq", "dk", "dv", "dbeta", "dg", "dA", "dbias"),
                                       grads, [expected_grads[i] for i in (0, 1, 2, 4, 3, 5, 6)]):
        _check(actual, reference, "fused_" + name)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard",
          essential_mark="essential")
@pytest.mark.parametrize("layout", ["BNSD", "BSND", "NTD", "TND"])
@pytest.mark.parametrize("mode", ["none", "forward", "recompute", "save"])
def test_prepare_layout_and_output_modes(layout, mode):
    """Four physical input layouts produce identical head-major caches/output."""
    inputs, _ = _inputs(65, heads=2)
    q, k, v, g, beta, a_log, bias = inputs
    baseline = npu_chunk_kda_fwd_prepare(
        q, k, v, g, beta, 128**-.5, safe_gate=True, use_gate_in_kernel=True,
        a_log=a_log, dt_bias=bias,
    )
    rank3 = layout in ("NTD", "TND")
    sequence_major = layout in ("BSND", "TND")

    def physical(tensor: ms.Tensor) -> ms.Tensor:
        """Construct the selected physical layout without a host synchronization."""
        if rank3:
            tensor = tensor.squeeze(0)
        if sequence_major:
            axes = list(range(tensor.ndim))
            base = 0 if rank3 else 1
            axes[base], axes[base + 1] = axes[base + 1], axes[base]
            tensor = tensor.transpose(tuple(axes)).contiguous()
        return tensor

    caches = npu_chunk_kda_fwd_prepare(
        *(physical(tensor) for tensor in (q, k, v, g, beta)), 128**-.5,
        layout=layout, safe_gate=True, use_gate_in_kernel=True, a_log=a_log, dt_bias=bias,
        cu_seqlens=[0, 65] if rank3 else None, backward_mode=mode,
    )
    required = {0, 1, 3, 4, 6, 7}
    if mode != "none":
        required.add(2)
    if mode in ("recompute", "save"):
        required.update(range(8, 13))
    if mode == "save":
        required.add(5)
    assert len(caches) == 13
    for index, tensor in enumerate(caches):
        if index not in required:
            assert tensor is None, (mode, index)
        else:
            expected = baseline[index].squeeze(0) if rank3 else baseline[index]
            _check(tensor, expected, f"cache_{layout}_{mode}_{index}", relative=.01)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard",
          essential_mark="essential")
@pytest.mark.parametrize("v_first", [True, False])
def test_fwd_h_state_layout(v_first):
    """FwdH supports V-first boundary/h layout independently of BwdFinalize."""
    inputs, d_o = _inputs(65)
    q, k, v, g, beta, a_log, bias = inputs
    cache = npu_chunk_kda_fwd_prepare(q, k, v, g, beta, 128**-.5, a_log=a_log,
                                      dt_bias=bias, safe_gate=True, use_gate_in_kernel=True)
    h0_kv = ms.Tensor(np.random.default_rng(5).normal(0, .01, (1, 1, 128, 128)), ms.bfloat16)
    h0 = h0_kv.transpose(0, 1, 3, 2).contiguous() if v_first else h0_kv
    h, vn, final = npu_chunk_kda_fwd_h(cache[6], cache[3], cache[4], gk=cache[0],
                                      initial_state=h0, output_final_state=True, state_v_first=v_first)
    out = npu_chunk_kda_fwd_finalize(cache[7], cache[1], vn, h,
                                     output_layout="BNSD", state_v_first=v_first)
    expected, expected_state, _ = _reference(inputs, d_o, h0=h0_kv)
    _check(out, expected, "v_first_output")
    _check(final.transpose(0, 1, 3, 2) if v_first else final, expected_state, "v_first_final")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard",
          essential_mark="essential")
@pytest.mark.parametrize("forward", [True, False])
@pytest.mark.parametrize("v_first", [True, False])
def test_summary_merge_against_cpu_compose(forward, v_first):
    """A local merge has observable aliasing and obeys ordered affine composition."""
    _require_apis("MergeFwdBwdKernel")
    rng = np.random.default_rng(23)
    summaries = rng.normal(0, .02, (3, 1, 128, 256)).astype(np.float32)
    summaries[:, :, :, 128:] += np.eye(128, dtype=np.float32)[None, None]
    # For forward rank=2,N=2 merge segments 0 then 1. Reverse rank=0,N=2
    # merges segments 2 then 1. Neither path reads the original state buffer.
    order = [0, 1] if forward else [2, 1]
    expected = summaries[order[0], :, :, :128]
    for index in order[1:]:
        expected = summaries[index, :, :, 128:] @ expected + summaries[index, :, :, :128]
    state = ms.Tensor(np.zeros((1, 128, 128), np.float32))
    result = npu_chunk_kda_merge_fwd_bwd(state, ms.Tensor(summaries), 2, 2 if forward else 0,
                                        forward=forward, state_v_first=v_first)
    target = expected.swapaxes(-1, -2) if v_first else expected
    _check(result, target, "merge", relative=.02)
    _check(state, target, "merge_alias", relative=.02)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard",
          essential_mark="essential")
def test_preprocess_summaries_and_boundary_scan():
    """One segment's affine E equals its zero-state result in both directions."""
    _require_apis("PreProcessFwdKernelMerged", "PreProcessBwdKernelMerged")
    inputs, d_o = _inputs(65)
    _, final, _, _, cache = _stage_chain(inputs, d_o, [0, 65])
    gk, aqk, _, w, u, qg, kg = cache[:7]
    (forward_summary,) = npu_chunk_kda_pre_process_fwd(kg, w, u, gk=gk, cu_seqlens=[0, 65])
    _check(forward_summary[0, :, :, :128], final[0], "forward_summary_E")
    h0 = ms.Tensor(np.zeros((1, 1, 128, 128), np.float32), ms.bfloat16)
    h, vn, _ = npu_chunk_kda_fwd_h(kg, w, u, gk=gk, initial_state=h0, cu_seqlens=[0, 65])
    _, dv_local, _ = npu_chunk_kda_bwd_prepare(aqk, vn, d_o, h, scale=128**-.5, cu_seqlens=[0, 65])
    backward_summary = npu_chunk_kda_pre_process_bwd(qg, kg, w, d_o, dv_local, 128**-.5,
                                                    gk=gk.astype(ms.bfloat16), cu_seqlens=[0, 65])
    _, dh0, _ = npu_chunk_kda_bwd_dhu(qg, kg, w, d_o, dv_local, 128**-.5,
                                     gk=gk, h0=h0, cu_seqlens=[0, 65])
    _check(backward_summary[:, :, :128], dh0[0], "backward_summary_E")
