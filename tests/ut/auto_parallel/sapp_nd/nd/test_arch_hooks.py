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
"""Tests for the arch hooks reading their op counts from data.

How to run this:
    pytest tests/ut/auto_parallel/sapp_nd/nd/test_arch_hooks.py -v
"""
import copy
import math
import os
import unittest
from types import SimpleNamespace
from typing import Any

from hyper_parallel.auto_parallel._layer_stack import LayerStack, StackGroup, derive_layers, resolve_layers
from hyper_parallel.auto_parallel._model_spec import OpCounts
from hyper_parallel.auto_parallel._op_profiles import LayerKind, known_archs, load_op_profile, resolve_ops
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2 import EvaluatorV2
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.layer_block import EvalAttn
from hyper_parallel.auto_parallel.sapp_nd.nd.common.arch_hooks import (
    CWrap,
    apply_layer_kind,
    check_and_apply_custom_hook,
    layer_groups,
)
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import CostModelConfig
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.estimate import estimate_comp, op_table
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.utils_classes import CustomConfig

_OPS = ("attMM", "attBMM", "ffMM", "softmax", "dropout", "normOp", "gather")

# The literal op counts each arch hook assigned before the counts became
# profiles, in _OPS order, except that mixtral's three expert projections
# were counted as batched matmuls and now count as the matmuls they are, and
# its softmaxes and norms are those its Transformers layers run (F51).
# headCast and ffAct were pinned to 1 by the estimator for every family.
_LEGACY_OPS = {
    "default": {"decoder": (4, 2, 3, 1, 0, 2, 4)},
    "llama2": {"decoder": (4, 2, 3, 1, 0, 2, 4)},
    "qwen": {"decoder": (4, 2, 3, 1, 0, 2, 4)},
    # Qwen3.5 was priced with the qwen hook; its full-attention kind keeps those counts.
    "qwen3_5": {"full_attention": (4, 2, 3, 1, 0, 2, 4)},
    "deepseek": {"decoder": (4, 2, 3, 1, 0, 2, 4)},
    "cm": {"decoder": (4, 2, 3, 1, 0, 2, 4)},
    "mixtral": {"decoder": (4, 2, 3, 1, 0, 2, 4)},
    "pangualpha": {"decoder": (4, 1, 2, 2, 5, 4, 4)},
    "t5": {"encoder": (4, 1, 2, 2, 5, 2, 4), "decoder": (8, 2, 2, 4, 7, 3, 6)},
    "vision": {"encoder": (4, 2, 2, 1, 0, 2, 4)},
}

# The byte widths the same hooks set: (bytes_grad at p > 1, at p == 1, bytes_dropout).
_LEGACY_BYTES = {
    "default": (4, 0, 0), "llama2": (2, 2, 0), "qwen": (4, 0, 0), "qwen3_5": (4, 0, 0),
    "deepseek": (4, 0, 0), "cm": (4, 0, 0), "mixtral": (2, 0, 0),
    "pangualpha": (4, 0, 1), "t5": (4, 0, 1), "vision": (4, 0, 0),
}


def _counts(**overrides) -> dict:
    """The default decoder's op counts, with *overrides* applied."""
    counts = dict(zip(_OPS, _LEGACY_OPS["default"]["decoder"]), headCast=1, ffAct=1, linrec=0)
    counts.update(overrides)
    return counts


def _hyper_config(name: str = "unit", **model) -> dict:
    """A small dense AutoModels config the hyper_v2 parser reads offline."""
    overrides = {
        "hidden_size": 1024,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "intermediate_size": 2816,
        "vocab_size": 32000,
        "max_position_embeddings": 2048,
    }
    overrides.update(model)
    return {
        "model": {"name": name, "config_overrides": overrides},
        "training": {"global_batch_size": 8, "micro_batch_size": 1},
        "accelerator": {"tp_size": 2, "pp_size": 1},
        "fsdp_config": {"dp_shard_size": 4},
        "dataset": {"data_transform": {"max_seq_len": 2048}},
        "context": {"max_device_memory": "64GB", "device_num": 8},
    }


def _ccfg(name: str = "unit", **model) -> CostModelConfig:
    """Parse :func:`_hyper_config` into a cost-model config."""
    return CostModelConfig(_hyper_config(name, **model), framework="hyper_v2")


def _bare(arch: str, op_counts=None, **fields) -> SimpleNamespace:
    """A config with just what the hooks read, as a test double would build."""
    cfg = SimpleNamespace(
        model_name="unit", arch=arch, op_counts=op_counts, has_op=False, p=2,
        t=2, h=16, dh=4, hff=32, hff_exp=64, n_lay=4, n_mtp=1, n_chosen_exp=2,
        n_exp=4, n_shared_exp=1, ep=2, k_1st_dense=2, config_format="yaml",
        shard_p_os_exp_partial=2, shard_p_os_non_exp=2, shard_embed=1,
        overwrite_eval_functions={},
    )
    for key, value in fields.items():
        setattr(cfg, key, value)
    return cfg


def _state(cfg) -> dict:
    """The op counts and byte widths a hook leaves on *cfg*."""
    names = [f"n_{op}" for op in _OPS] + [
        "n_headCast", "n_ffAct", "n_attParamCast", "n_ffParamCast",
        "bytes_grad", "bytes_os", "bytes_dropout", "bytes_norm",
    ]
    return {name: getattr(cfg, name) for name in names}


def _legacy_state(arch: str, kind: str, has_op: bool, p: int) -> dict:
    """What the literal callback of *arch* left for one layer kind."""
    counts = dict(zip(_OPS, _LEGACY_OPS[arch][kind]))
    grad_pp, grad_no_pp, dropout = _LEGACY_BYTES[arch]
    state = {f"n_{op}": count for op, count in counts.items()}
    state.update(
        n_headCast=1, n_ffAct=1,
        n_attParamCast=0 if has_op else counts["attMM"],
        n_ffParamCast=0 if has_op else counts["ffMM"],
        bytes_grad=grad_pp if p > 1 else grad_no_pp,
        bytes_os=4, bytes_dropout=dropout, bytes_norm=4,
    )
    return state


class TestParamCasts(unittest.TestCase):
    """A layer keeps a cast beside each matmul as its run says, else where the optimizer does not shard."""

    def test_the_run_decides(self):
        """
        Feature: the cast counts applying a family sets.
        Description: A config whose optimizer does not shard, derive not
            run; then derive saying its layers keep no cast; then keep them.
        Expectation: Unsaid, a cast per matmul; said, as said.
        """
        got = []
        for casts in (None, False, True):
            cfg = _bare("default", has_op=False, keeps_param_casts=casts)
            check_and_apply_custom_hook(CWrap(cfg))
            got.append((cfg.n_attParamCast, cfg.n_ffParamCast))
        self.assertEqual(got, [(4, 3), (0, 0), (4, 3)])


class TestHooksMatchTheirLiterals(unittest.TestCase):
    """Reading counts from profiles leaves each config as the literals did."""

    def _check(self, arch: str) -> None:
        """Assert *arch*'s hook leaves every layer kind as its literals did."""
        for has_op in (False, True):
            for p in (1, 2):
                with self.subTest(arch=arch, has_op=has_op, p=p):
                    table = resolve_ops(arch)
                    cfg = _bare(arch, table, has_op=has_op, p=p)
                    cfg.layer_stack = resolve_layers(arch, derive_layers(load_op_profile(arch), cfg.n_lay), table)
                    check_and_apply_custom_hook(CWrap(cfg))
                    if arch != "t5":
                        kind = next(iter(_LEGACY_OPS[arch]))
                        self.assertEqual(_state(cfg), _legacy_state(arch, kind, has_op, p))
                        continue
                    # t5 sets its counts per layer, from its stack's kinds, and its
                    # byte widths on the model, which every layer keeps.
                    bytes_names = ("bytes_grad", "bytes_os", "bytes_dropout", "bytes_norm")
                    legacy = _legacy_state(arch, "encoder", has_op, p)
                    self.assertEqual({name: getattr(cfg, name) for name in bytes_names},
                                     {name: legacy[name] for name in bytes_names})
                    for name, (kind, count) in zip(("encoder", "decoder"), layer_groups(cfg)):
                        layer = copy.deepcopy(cfg)
                        apply_layer_kind(CWrap(layer), kind)
                        self.assertEqual((kind.name, count), (name, cfg.n_lay // 2))
                        self.assertEqual(_state(layer), _legacy_state(arch, name, has_op, p))

    def test_counts_from_the_parser(self):
        """
        Feature: hooks read the counts the parser recorded.
        Description: Apply every family's hook to a config carrying its
            resolved profile, for each optimizer-sharding and pipeline case.
        Expectation: Counts and byte widths equal the old literals.
        """
        for arch in known_archs():
            self._check(arch)


class TestTwoKindStack(unittest.TestCase):
    """Each layer of a mixed stack is priced from its own kind's vector."""

    _LINEAR = {"attBMM": 0, "softmax": 0, "headCast": 0}

    def _stack(self, linear_layers: int = 2) -> CostModelConfig:
        """Four layers: full attention, then *linear_layers* without scores."""
        ccfg = _ccfg(arch="default")
        groups = (
            StackGroup(LayerKind("decoder", OpCounts.from_dict(_counts())), 4 - linear_layers),
            StackGroup(LayerKind("scoreless", OpCounts.from_dict(_counts(**self._LINEAR))), linear_layers),
        )
        ccfg.layer_stack = LayerStack("default", tuple(group for group in groups if group.count))
        return ccfg

    def test_each_layer_is_priced_from_its_kind_on_both_paths(self):
        """
        Feature: per-kind op counts on the performance and memory paths.
        Description: Price the stack, and the same stack with every layer
            full, on one pipeline stage; then record the softmax count the
            attention score formula sees at every layer, as the memory
            backbone applies each group's kind to the layers it covers.
        Expectation: The FLOP scores differ by two layers' worth of the
            three ops the linear kind does not run, and nothing else; the
            memory path sees two layers with the full kind's 1, then two
            with 0.
        """
        layers = [LayerType.NOT_REC_LAYER] * 4
        stages = [[[LayerType.EMBEDDING_LAYER] + layers + [LayerType.OUTPUT_LAYER]]]
        mixed = self._stack(linear_layers=2)
        full = self._stack(linear_layers=0)
        check_and_apply_custom_hook(mixed)
        check_and_apply_custom_hook(full)
        table = op_table(full)
        missing = 2 * (2 * table["n_attBMM"] + table["n_softmax"] + table["n_headCast"])
        saved = (estimate_comp(full, CustomConfig(), stages)[0]
                 - estimate_comp(mixed, CustomConfig(), stages)[0])
        self.assertAlmostEqual(saved / missing, 1.0, places=9)
        seen = {}

        def spy(ccfg: CostModelConfig, ctx: Any) -> float:
            """The real score formula, recording the counts it was given."""
            seen.setdefault(ctx.current_lay_id, set()).add(ccfg.n_softmax)
            return EvalAttn.attn_score_activations(ccfg, ctx)

        evaluator = EvaluatorV2(None, ccfg=self._stack())
        evaluator.set_attn_eval_fun(score=spy)
        evaluator.estimate_peak()
        self.assertEqual([seen[lay_id] for lay_id in sorted(seen)], [{1}, {1}, {0}, {0}])


_LINEAR_DIMS = {
    "linear_num_key_heads": 8, "linear_key_head_dim": 64, "linear_num_value_heads": 16,
    "linear_value_head_dim": 64, "linear_conv_kernel_dim": 4,
}
_L, _F = "linear_attention", "full_attention"


def _qwen35_model(layer_types, **model) -> dict:
    """The config_overrides of a small Qwen3.5-shaped model with *layer_types*."""
    fields = {
        "num_hidden_layers": len(layer_types), "layer_types": list(layer_types), "head_dim": 128,
        "num_key_value_heads": 2, "attn_output_gate": True, **_LINEAR_DIMS,
    }
    fields.update(model)
    return fields


class TestStackAsData(unittest.TestCase):
    """A stack the parser settled as data prices each layer by its kind."""

    def test_a_linear_layer_then_a_full_one_each_take_their_own_attention(self):
        """
        Feature: linear attention, and complete assignments.
        Description: Apply the linear kind of a Qwen3.5-shaped model with 16
            value heads of 64, then its full kind, in place, as the memory
            backbone does layer after layer.
        Expectation: The linear layer has the value heads on the q side, the
            key heads on the kv side, no score ops, one state update, no
            QK-norm, its kernel normalizing without weights, and the
            convolution, the gates' projections, each head's decay and
            time-step bias and the output norm's weight as extra parameters;
            the full layer has exactly the attention the family hook left,
            its one QK-norm included.
        """
        ccfg = _ccfg("qwen3_5_moe", **_qwen35_model([_L, _F]))
        check_and_apply_custom_hook(ccfg)
        kinds = {kind.name: kind for kind in ccfg.layer_stack.distinct_kinds()}
        names = ("attn_kind", "a", "dh", "n_kv", "attn_output_gate", "attn_extra_p", "n_qknorm",
                 "n_attBMM", "n_softmax", "n_headCast", "n_linrec", "lin_n_v")
        before = {name: getattr(ccfg, name) for name in names}
        layer = copy.deepcopy(ccfg)
        apply_layer_kind(layer, kinds[_L])
        self.assertEqual(
            (layer.attn_kind, layer.a, layer.dh, layer.n_kv, layer.attn_output_gate,
             layer.n_softmax, layer.n_linrec, layer.n_qknorm, layer.attn_extra_p),
            ("linear", 16, 64, 8.0, True, 0, 1, 0, 4 * (2 * 8 * 64 + 16 * 64) + 2 * 1024 * 16 + 2 * 16 + 64),
        )
        apply_layer_kind(layer, kinds[_F])
        self.assertEqual(({name: getattr(layer, name) for name in names}, before["n_qknorm"]), (before, 1))


_DEEPSEEK = {
    "num_hidden_layers": 4, "first_k_dense_replace": 1, "mtp_depth": 1, "num_experts": 8,
    "num_experts_per_tok": 2, "num_shared_experts": 1, "moe_intermediate_size": 512,
}


def _deepseek(name: str = "deepseek_v3") -> CostModelConfig:
    """A small DeepSeek-shaped model at ep 2: one dense layer, three MoE and one MTP."""
    config = _hyper_config(name, **_DEEPSEEK)
    config["accelerator"]["ep_size"] = 2
    return CostModelConfig(config, framework="hyper_v2")


class TestDenseThenMoE(unittest.TestCase):
    """DeepSeek's dense and MoE layers, and cm's, are kinds of the stack."""

    _FFN = ("hff", "n_chosen_exp", "n_exp", "n_shared_exp", "ep")

    def test_each_kind_assigns_the_whole_feed_forward(self):
        """
        Feature: complete assignments, and the strategy a kind sets.
        Description: Apply the dense kind, a MoE one, then the dense one
            again, in place through an evaluator, as the memory backbone
            does, whose guard refuses a strategy write inside a hook.
        Expectation: A dense layer runs one expert at the dense width, no
            shared one and no expert parallelism; a MoE layer runs the
            model's experts at their width and ep, written past the guard;
            and the dense layers match.
        """
        evaluator = EvaluatorV2(None, ccfg=_deepseek())
        dense, moe = (group.kind for group in evaluator.ccfg.layer_stack.groups[:2])
        states = []
        for kind in (dense, moe, dense):
            apply_layer_kind(evaluator, kind)
            states.append({name: getattr(evaluator.ccfg, name) for name in self._FFN})
        self.assertEqual(states[0], {"hff": 2816, "n_chosen_exp": 1, "n_exp": 1, "n_shared_exp": 0, "ep": 1})
        self.assertEqual(states[1], {"hff": 512, "n_chosen_exp": 2, "n_exp": 8, "n_shared_exp": 1, "ep": 2})
        self.assertEqual(states[2], states[0])

    def test_cm_shards_every_layer_as_its_hook_says(self):
        """
        Feature: fields the family gives every layer.
        Description: cm runs DeepSeek's stack with its own optimizer and
            embedding sharding on each layer.
        Expectation: Both kinds assign the sharding cm's hook derived.
        """
        ccfg = _deepseek("cm_llama_moe")
        self.assertEqual(ccfg.arch, "cm")
        check_and_apply_custom_hook(ccfg)
        expected = (ccfg.shard_p_os_exp_partial, math.gcd(ccfg.n_exp, ccfg.shard_p_os_non_exp), ccfg.t)
        for kind in ccfg.layer_stack.distinct_kinds():
            layer = copy.deepcopy(ccfg)
            apply_layer_kind(layer, kind)
            self.assertEqual((layer.shard_p_os_exp, layer.shard_p_os_non_exp_partial, layer.shard_embed),
                             expected, kind.name)


class TestParsersRecordTheArch(unittest.TestCase):
    """Every parser settles the arch before any hook runs."""

    def test_each_parser_records_the_arch(self):
        """
        Feature: the MindFormers and hyper_v2 parsers.
        Description: The MindFormers DeepSeek yaml, which names its model in
            free text only; a hyper_v2 spec that states no arch, and one that
            declares the default arch and a decoder's counts.
        Expectation: deepseekV3 resolves to the deepseek profile; the spec's
            arch, inferred from its name or declared, reaches the config,
            with the declared counts.
        """
        ccfg = CostModelConfig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepseek.yaml"))
        self.assertEqual((ccfg.arch, ccfg.op_counts), ("deepseek", resolve_ops("deepseek")))
        self.assertEqual(_ccfg("qwen3_moe").arch, "qwen")
        declared = _ccfg("qwen3_moe", arch="default", ops={"decoder": _counts(gather=6)})
        self.assertEqual((declared.arch, declared.op_counts["decoder"].gather), ("default", 6))


if __name__ == "__main__":
    unittest.main()
