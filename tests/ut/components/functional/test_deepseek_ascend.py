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
"""Unit tests for lazy DeepSeek Ascend operator adapters."""

from types import SimpleNamespace

import pytest

from hyper_parallel.components import functional
from hyper_parallel.components.functional import deepseek_ascend


@pytest.fixture(autouse=True)
def _clear_operator_caches():
    """Keep monkeypatched backend callables isolated between tests."""
    deepseek_ascend._resolve_operator.cache_clear()
    deepseek_ascend._resolve_tile_operator.cache_clear()
    yield
    deepseek_ascend._resolve_operator.cache_clear()
    deepseek_ascend._resolve_tile_operator.cache_clear()


def _assert_equal(actual, expected, context):
    assert actual == expected, f"{context}: expected={expected!r}, actual={actual!r}"


def _assert_true(actual, context):
    assert actual is True, f"{context}: expected=True, actual={actual!r}"


def _assert_is(actual, expected, context):
    assert actual is expected, f"{context}: expected identity={expected!r}, actual={actual!r}"


def test_public_exports_match_lazy_functional_namespace():
    """Every adapter export is reachable through the lazy public namespace."""
    missing_exports = set(deepseek_ascend.__all__) - set(functional.__all__)
    _assert_equal(missing_exports, set(), "DeepSeek exports missing from functional namespace")
    matching_exports = all(
        getattr(functional, name) is getattr(deepseek_ascend, name) for name in deepseek_ascend.__all__
    )
    _assert_true(matching_exports, "Lazy functional exports resolve to their adapter implementations")


def test_backend_probe_validates_name(monkeypatch):
    """The discovery probe remains non-importing and rejects unknown names."""
    def _find_spec(name):
        return object() if name == "flash_mla" else None

    monkeypatch.setattr(deepseek_ascend.importlib.util, "find_spec", _find_spec)
    _assert_true(deepseek_ascend.is_deepseek_backend_available("flash_mla"), "FlashMLA discovery")
    _assert_true(not deepseek_ascend.is_deepseek_backend_available("deep_gemm"), "DeepGEMM absence discovery")
    with pytest.raises(ValueError, match="Unsupported DeepSeek backend"):
        deepseek_ascend.is_deepseek_backend_available("unknown")


@pytest.mark.parametrize("error_type", (ImportError, OSError, RuntimeError))
def test_missing_backend_has_actionable_error(monkeypatch, error_type):
    """Native-loader failures identify the package that needs to be built."""
    def _fail_import(_name):
        raise error_type("missing native extension")

    monkeypatch.setattr(deepseek_ascend.importlib, "import_module", _fail_import)
    with pytest.raises(RuntimeError, match="Install and build DeepGEMM-Ascend"):
        deepseek_ascend.deepseek_gemm("bf16_gemm_nt", object())


def test_gemm_and_select_forward_native_arguments(monkeypatch):
    """Dense and selection adapters preserve the native calling contract."""
    calls = []

    def _gemm(*args, **kwargs):
        calls.append(("gemm", args, kwargs))
        return "gemm-output"

    def _topk(*args, **kwargs):
        calls.append(("topk", args, kwargs))
        return "values", "indices"

    packages = {
        "deep_gemm": SimpleNamespace(bf16_gemm_nt=_gemm),
        "deep_select": SimpleNamespace(topk=_topk, get_stride_requirement=lambda: (32, 32)),
    }
    monkeypatch.setattr(deepseek_ascend.importlib, "import_module", packages.__getitem__)

    _assert_equal(
        deepseek_ascend.deepseek_gemm("bf16_gemm_nt", "x", "weight", out="output"),
        "gemm-output",
        "Generic GEMM result",
    )
    _assert_equal(
        deepseek_ascend.deepseek_bf16_gemm("nt", "x", "weight", out="output"),
        "gemm-output",
        "Semantic GEMM result",
    )
    _assert_equal(deepseek_ascend.deepseek_select_stride_requirement(), (32, 32), "DeepSelect stride requirement")
    _assert_equal(
        deepseek_ascend.deepseek_select_topk("scores", 8, sorted_index=True),
        ("values", "indices"),
        "DeepSelect result",
    )
    _assert_equal(calls[0], ("gemm", ("x", "weight"), {"out": "output"}), "Generic GEMM forwarding")
    _assert_equal(calls[1], ("gemm", ("x", "weight"), {"out": "output"}), "Semantic GEMM forwarding")
    _assert_equal(calls[2][0:2], ("topk", ("scores", 8)), "DeepSelect positional forwarding")
    _assert_is(calls[2][2]["indices_type"], deepseek_ascend.torch.int32, "DeepSelect default index dtype")
    _assert_true(calls[2][2]["sorted_index"], "DeepSelect sorted-index forwarding")
    with pytest.raises(ValueError, match="public callable"):
        deepseek_ascend.deepseek_gemm("__class__")
    with pytest.raises(ValueError, match="public callable"):
        deepseek_ascend.deepseek_gemm(1)


def test_deep_gemm_semantic_variants(monkeypatch):
    """Semantic GEMM helpers select only supported native variants."""
    seen = []

    def _operator(name):
        def _call(*args, **kwargs):
            seen.append((name, args, kwargs))
            return name

        return _call

    names = (
        "bf16_gemm_tt",
        "fp8_gemm_nn",
        "fp8_fp4_gemm_tn",
        "m_grouped_bf16_gemm_nt_contiguous",
        "k_grouped_fp8_gemm_tn_contiguous",
        "m_grouped_fp8_fp4_gemm_nn_contiguous",
        "fp8_einsum",
        "fp8_fp4_paged_mqa_logits",
        "get_paged_mqa_logits_metadata",
        "transform_k_grouped_sf_into_required_layout",
        "fp8_fp4_mega_moe",
        "transform_weights_for_mega_moe",
        "tf32_hc_prenorm_gemm",
    )
    package = SimpleNamespace(**{name: _operator(name) for name in names})
    monkeypatch.setattr(deepseek_ascend.importlib, "import_module", lambda _name: package)

    results = (
        deepseek_ascend.deepseek_bf16_gemm("tt", "x"),
        deepseek_ascend.deepseek_fp8_gemm("nn", "x"),
        deepseek_ascend.deepseek_fp8_fp4_gemm("tn", "x"),
        deepseek_ascend.deepseek_grouped_bf16_gemm("m_nt", "x"),
        deepseek_ascend.deepseek_grouped_fp8_gemm("k_tn", "x"),
        deepseek_ascend.deepseek_grouped_fp8_fp4_gemm("m_nn", "x"),
        deepseek_ascend.deepseek_einsum("equation", quantized=True),
        deepseek_ascend.deepseek_mqa_logits("q", paged=True),
        deepseek_ascend.deepseek_paged_mqa_logits_metadata("lengths", 64, "indices"),
        deepseek_ascend.deepseek_transform_scaling_factors("sf", grouped=True),
        deepseek_ascend.deepseek_mega_moe("x"),
        deepseek_ascend.deepseek_transform_mega_moe_weights("weight"),
        deepseek_ascend.deepseek_hc_prenorm_gemm("x"),
    )
    _assert_equal(results, names, "Semantic DeepGEMM operator selection")
    _assert_equal([item[0] for item in seen], list(names), "DeepGEMM invocation order")
    with pytest.raises(ValueError, match="Unsupported grouped BF16 GEMM variant"):
        deepseek_ascend.deepseek_grouped_bf16_gemm("bad", "x")
    with pytest.raises(ValueError, match="Unsupported bf16 GEMM layout"):
        deepseek_ascend.deepseek_bf16_gemm("bad", "x")


def test_flash_mla_adapters_map_hyper_names(monkeypatch):
    """FlashMLA adapters translate stable Hyper names to upstream names."""
    calls = []

    def _prefill(*args, **kwargs):
        calls.append(("prefill", args, kwargs))
        return "prefill-output"

    def _decode(*args, **kwargs):
        calls.append(("decode", args, kwargs))
        return "decode-output"

    package = SimpleNamespace(
        flash_mla_sparse_fwd=_prefill,
        flash_mla_with_kvcache=_decode,
        get_mla_metadata=lambda: ("metadata", None),
    )
    monkeypatch.setattr(deepseek_ascend.importlib, "import_module", lambda _name: package)

    _assert_equal(deepseek_ascend.deepseek_flash_mla_metadata(), ("metadata", None), "FlashMLA metadata")
    _assert_equal(
        deepseek_ascend.deepseek_flash_mla_sparse_prefill(
            "q", "kv", "indices", 0.125, attention_sink="sink"
        ),
        "prefill-output",
        "FlashMLA prefill result",
    )
    _assert_equal(
        deepseek_ascend.deepseek_flash_mla_sparse_decode(
            "q", "cache", "indices", "metadata", topk_length="lengths"
        ),
        "decode-output",
        "FlashMLA decode result",
    )

    _assert_equal(calls[0], (
        "prefill",
        ("q", "kv", "indices", 0.125),
        {"d_v": 512, "attn_sink": "sink", "topk_length": None},
    ), "FlashMLA prefill forwarding")
    _assert_equal(calls[1][1], ("q", "cache", None, None, 512, "metadata"), "FlashMLA decode positional forwarding")
    _assert_equal(calls[1][2]["indices"], "indices", "FlashMLA decode indices forwarding")
    _assert_equal(calls[1][2]["topk_length"], "lengths", "FlashMLA decode Top-K length forwarding")
    with pytest.raises(ValueError, match="value_head_dim=512"):
        deepseek_ascend.deepseek_flash_mla_sparse_prefill("q", "kv", "indices", 0.125, value_head_dim=128)
    with pytest.raises(ValueError, match="value_head_dim=512"):
        deepseek_ascend.deepseek_flash_mla_sparse_decode(
            "q", "cache", "indices", "metadata", value_head_dim=128
        )


def test_tile_kernel_restricts_namespace_and_forwards(monkeypatch):
    """TileKernels dispatch is limited to documented public namespaces."""
    imports = []

    def _operator(value, scale=1):
        return value * scale

    def _import_module(name):
        imports.append(name)
        if name == "tile_kernels":
            return SimpleNamespace()
        if name == "tile_kernels.quant":
            return SimpleNamespace(per_token_cast=_operator)
        raise ImportError(name)

    monkeypatch.setattr(deepseek_ascend.importlib, "import_module", _import_module)
    _assert_equal(
        deepseek_ascend.deepseek_tile_kernel("quant", "per_token_cast", 3, scale=2), 6, "Tile kernel result"
    )
    _assert_equal(
        deepseek_ascend.deepseek_tile_kernel("quant", "per_token_cast", 4, scale=2),
        8,
        "Cached tile kernel result",
    )
    _assert_equal(imports, ["tile_kernels", "tile_kernels.quant"], "Tile kernel resolution cache")
    with pytest.raises(ValueError, match="Unsupported TileKernels namespace"):
        deepseek_ascend.deepseek_tile_kernel("testing", "bench")
    with pytest.raises(AttributeError, match="no callable operator"):
        deepseek_ascend.deepseek_tile_kernel("quant", "missing")
    with pytest.raises(ValueError, match="public callable"):
        deepseek_ascend.deepseek_tile_kernel("quant", "__class__")


def test_explicit_tile_kernel_adapters(monkeypatch):
    """Common TileKernels families have discoverable explicit adapters."""
    calls = []

    def _operator(name):
        def _call(*args, **kwargs):
            calls.append((name, args, kwargs))
            return name

        return _call

    modules = {
        "tile_kernels": SimpleNamespace(),
        "tile_kernels.quant": SimpleNamespace(
            per_token_cast=_operator("per_token_cast"),
            per_block_cast=_operator("per_block_cast"),
            cast_back=_operator("cast_back"),
            swiglu_forward=_operator("swiglu_forward"),
            swiglu_backward=_operator("swiglu_backward"),
        ),
        "tile_kernels.moe": SimpleNamespace(
            topk_gate=_operator("topk_gate"),
            moe_topk_gate=_operator("moe_topk_gate"),
            moe_topk_gate_backward=_operator("moe_topk_gate_backward"),
            normalize_weight=_operator("normalize_weight"),
        ),
        "tile_kernels.modeling.engram": SimpleNamespace(engram_gate=_operator("engram_gate")),
        "tile_kernels.transform": SimpleNamespace(apply_rotary=_operator("apply_rotary")),
        "tile_kernels.engram": SimpleNamespace(engram_hash=_operator("engram_hash")),
    }
    monkeypatch.setattr(deepseek_ascend.importlib, "import_module", modules.__getitem__)

    results = (
        deepseek_ascend.deepseek_per_token_cast("x", "e4m3", 32),
        deepseek_ascend.deepseek_per_block_cast("x", "e4m3", (32, 32)),
        deepseek_ascend.deepseek_cast_back("x", "bf16", (1, 32)),
        deepseek_ascend.deepseek_swiglu_forward("x", "bf16"),
        deepseek_ascend.deepseek_swiglu_backward("x", "grad", "bf16"),
        deepseek_ascend.deepseek_topk_gate("scores", 8),
        deepseek_ascend.deepseek_moe_topk_gate("logits", 8, False, 0, 1.0, 0),
        deepseek_ascend.deepseek_moe_topk_gate_backward("scores", "indices", "weights", "gradient", 1.0),
        deepseek_ascend.deepseek_normalize_routing_weights("weights"),
        deepseek_ascend.deepseek_apply_rotary("q", "cache"),
        deepseek_ascend.deepseek_engram_hash("ids", "mul", "sizes", "offsets"),
        deepseek_ascend.deepseek_engram_gate("hidden", "kv", "wh", "we", 8.0, 1e-6),
    )
    expected = (
        "per_token_cast",
        "per_block_cast",
        "cast_back",
        "swiglu_forward",
        "swiglu_backward",
        "topk_gate",
        "moe_topk_gate",
        "moe_topk_gate_backward",
        "normalize_weight",
        None,
        "engram_hash",
        "engram_gate",
    )
    _assert_equal(results, expected, "Explicit TileKernels adapter results")
    _assert_equal(calls[6][1], ("logits", 8, False, 0, 1.0, 0), "Fused MoE forward positional arguments")
    _assert_equal(calls[6][2]["scoring_func"], "sqrtsoftplus", "Fused MoE forward scoring function")
    _assert_equal(
        calls[7][1],
        ("scores", "indices", "weights", "gradient", 1.0),
        "Fused MoE backward positional arguments",
    )
    _assert_equal(calls[7][2]["grad_scores_sum"], None, "Fused MoE backward auxiliary gradient")
    _assert_equal([item[0] for item in calls], [
        "per_token_cast",
        "per_block_cast",
        "cast_back",
        "swiglu_forward",
        "swiglu_backward",
        "topk_gate",
        "moe_topk_gate",
        "moe_topk_gate_backward",
        "normalize_weight",
        "apply_rotary",
        "engram_hash",
        "engram_gate",
    ], "TileKernels invocation order")
