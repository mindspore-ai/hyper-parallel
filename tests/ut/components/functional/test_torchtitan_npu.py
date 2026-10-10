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
"""Unit tests for lazy TorchTitan-NPU operator adapters."""

from types import SimpleNamespace

import pytest

from hyper_parallel.components import functional
from hyper_parallel.components.functional import torchtitan_npu

_ADAPTER_CASES = [
    (
        "torchtitan_inplace_partial_rotary_mul",
        "ascendc.inplace_partial_rotary_mul",
        "torchtitan_npu.ops.ascendc.inplace_partial_rotary_mul",
        "inplace_partial_rotary_mul",
    ),
    (
        "torchtitan_moe_re_routing",
        "ascendc.moe_re_routing",
        "torchtitan_npu.ops.ascendc.moe_re_routing",
        "npu_moe_re_routing",
    ),
    (
        "torchtitan_moe_token_permute",
        "ascendc.moe_token_permute",
        "torchtitan_npu.ops.ascendc.moe_token_permute",
        "npu_moe_token_permute",
    ),
    (
        "torchtitan_moe_token_unpermute",
        "ascendc.moe_token_unpermute",
        "torchtitan_npu.ops.ascendc.moe_token_unpermute",
        "npu_moe_token_unpermute",
    ),
    (
        "torchtitan_mhc_head_compute_mix",
        "tilelang.mhc_head_compute_mix",
        "torchtitan_npu.ops.tilelang",
        "mhc_head_compute_mix_tilelang",
    ),
    (
        "torchtitan_mhc_head_compute_mix_a5",
        "tilelang.mhc_head_compute_mix_a5",
        "torchtitan_npu.ops.tilelang",
        "tilelang_mhc_head_compute_mix_a5",
    ),
    (
        "torchtitan_mhc_post",
        "tilelang.mhc_post",
        "torchtitan_npu.ops.tilelang",
        "tilelang_mhc_post",
    ),
    (
        "torchtitan_mhc_pre",
        "tilelang.mhc_pre",
        "torchtitan_npu.ops.tilelang",
        "tilelang_mhc_pre",
    ),
    (
        "torchtitan_mhc_pre_v41",
        "tilelang.mhc_pre_v41",
        "torchtitan_npu.ops.tilelang",
        "tilelang_mhc_pre_v41",
    ),
    (
        "torchtitan_swiglu",
        "tilelang.swiglu",
        "torchtitan_npu.ops.tilelang",
        "tilelang_swiglu",
    ),
    (
        "torchtitan_topk_gate",
        "tilelang.topk_gate",
        "torchtitan_npu.ops.tilelang",
        "tilelang_topk_gate",
    ),
    (
        "torchtitan_gated_delta_rule",
        "triton.gated_delta_rule",
        "torchtitan_npu.ops.triton.gdn",
        "gated_delta_rule",
    ),
    (
        "torchtitan_mhc_post_bmm1",
        "triton.mhc_post_bmm1",
        "torchtitan_npu.ops.triton.mhc",
        "mhc_post_bmm1_op",
    ),
    (
        "torchtitan_mhc_post_bmm2",
        "triton.mhc_post_bmm2",
        "torchtitan_npu.ops.triton.mhc",
        "mhc_post_bmm2_op",
    ),
    (
        "torchtitan_mhc_pre_bmm",
        "triton.mhc_pre_bmm",
        "torchtitan_npu.ops.triton.mhc",
        "mhc_pre_bmm_op",
    ),
    (
        "torchtitan_mhc_pre_only_sinkhorn",
        "triton.mhc_pre_only_sinkhorn",
        "torchtitan_npu.ops.triton.mhc",
        "mhc_pre_only_sinkhorn_op",
    ),
    (
        "torchtitan_mhc_pre_sinkhorn",
        "triton.mhc_pre_sinkhorn",
        "torchtitan_npu.ops.triton.mhc",
        "mhc_pre_sinkhorn_op",
    ),
]


def _assert_equal(actual, expected, context):
    assert actual == expected, f"{context}: expected={expected!r}, actual={actual!r}"


def _assert_is(actual, expected, context):
    assert actual is expected, (
        f"{context}: expected identity={expected!r}, actual={actual!r}"
    )


@pytest.fixture(autouse=True)
def _clear_operator_cache():
    """Keep monkeypatched backend callables isolated between tests."""
    torchtitan_npu._resolve_operator.cache_clear()
    yield
    torchtitan_npu._resolve_operator.cache_clear()


def test_public_exports_match_lazy_functional_namespace():
    """Every adapter export is reachable through the lazy public namespace."""
    missing_exports = set(torchtitan_npu.__all__) - set(functional.__all__)
    _assert_equal(
        missing_exports,
        set(),
        "TorchTitan-NPU exports missing from functional namespace",
    )
    for name in torchtitan_npu.__all__:
        _assert_is(
            getattr(functional, name),
            getattr(torchtitan_npu, name),
            f"Lazy export {name}",
        )


def test_backend_probe_does_not_import_package(monkeypatch):
    """Availability discovery checks the package without loading its NPU dependencies."""
    calls = []

    def _find_spec(name):
        calls.append(name)
        return object()

    monkeypatch.setattr(torchtitan_npu.importlib.util, "find_spec", _find_spec)
    _assert_is(
        torchtitan_npu.is_torchtitan_npu_available(),
        True,
        "TorchTitan-NPU discovery result",
    )
    _assert_equal(calls, ["torchtitan_npu"], "Package discovery calls")


def test_generic_adapter_preserves_arguments_and_result(monkeypatch):
    """The generic adapter forwards the upstream contract without translation."""
    calls = []

    def _operator(*args, **kwargs):
        calls.append((args, kwargs))
        return "result"

    package = SimpleNamespace(tilelang_topk_gate=_operator)
    monkeypatch.setattr(
        torchtitan_npu.importlib, "import_module", lambda _name: package
    )

    result = torchtitan_npu.torchtitan_npu_op(
        "tilelang.topk_gate", "scores", 8, stable=True
    )

    _assert_equal(result, "result", "Generic adapter result")
    _assert_equal(
        calls, [(("scores", 8), {"stable": True})], "Generic adapter forwarding"
    )


@pytest.mark.parametrize(
    ("adapter_name", "operator", "_module_name", "_attribute"),
    _ADAPTER_CASES,
)
def test_semantic_adapters_select_expected_upstream_operator(
    monkeypatch, adapter_name, operator, _module_name, _attribute
):
    """Each named adapter selects its matching public TorchTitan-NPU operator."""
    calls = []

    def _dispatch(selected, *args, **kwargs):
        calls.append((selected, args, kwargs))
        return selected

    monkeypatch.setattr(torchtitan_npu, "torchtitan_npu_op", _dispatch)
    result = getattr(torchtitan_npu, adapter_name)("input", flag=True)

    _assert_equal(result, operator, f"Semantic adapter {adapter_name} result")
    _assert_equal(
        calls,
        [(operator, ("input",), {"flag": True})],
        f"Semantic adapter {adapter_name} forwarding",
    )


@pytest.mark.parametrize(
    ("_adapter_name", "operator", "module_name", "attribute"),
    _ADAPTER_CASES,
)
def test_resolver_selects_expected_upstream_module_and_callable(
    monkeypatch, _adapter_name, operator, module_name, attribute
):
    """Each allowlisted name resolves the public callable from its upstream module."""
    calls = []

    def _operator(*args, **kwargs):
        calls.append((args, kwargs))
        return "result"

    def _import_module(selected_module):
        _assert_equal(selected_module, module_name, f"Resolver module for {operator}")
        return SimpleNamespace(**{attribute: _operator})

    monkeypatch.setattr(torchtitan_npu.importlib, "import_module", _import_module)

    result = torchtitan_npu.torchtitan_npu_op(operator, "input", flag=True)

    _assert_equal(result, "result", f"Resolver result for {operator}")
    _assert_equal(
        calls, [(("input",), {"flag": True})], f"Resolver forwarding for {operator}"
    )


def test_invalid_operator_and_missing_backend_have_actionable_errors(monkeypatch):
    """Invalid names and native-loader failures are reported at the adapter boundary."""
    with pytest.raises(TypeError, match="operator must be str"):
        torchtitan_npu.torchtitan_npu_op(1)
    with pytest.raises(ValueError, match="Unsupported TorchTitan-NPU operator"):
        torchtitan_npu.torchtitan_npu_op("testing.private")

    def _fail_import(_name):
        raise ImportError("missing runtime")

    monkeypatch.setattr(torchtitan_npu.importlib, "import_module", _fail_import)
    with pytest.raises(RuntimeError, match="Install torchtitan-npu"):
        torchtitan_npu.torchtitan_topk_gate("scores", 8)


def test_revision_mismatch_rejects_missing_callable(monkeypatch):
    """A package with a mismatched public surface fails before execution."""
    monkeypatch.setattr(
        torchtitan_npu.importlib, "import_module", lambda _name: SimpleNamespace()
    )
    with pytest.raises(TypeError, match="does not expose callable"):
        torchtitan_npu.torchtitan_topk_gate("scores", 8)
