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
"""Tests for the layer census, the activations it states in the model spec, and their pricing."""
import contextlib
import functools
import importlib
import importlib.util
import os
import sys
import tempfile
import textwrap
import types
import unittest
from typing import Any, Callable, Optional
from unittest.mock import patch

import torch
import yaml
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.utils.checkpoint import DefaultDeviceType

import hyper_parallel
from hyper_parallel.auto_parallel._hf_model_spec import resolve_hf_model_spec
from hyper_parallel.auto_parallel._layer_census import (  # pylint: disable=protected-access
    CensusUnavailable,
    census_activations,
    census_final_norm,
    census_flops,
    census_layer,
    census_output_activations,
    census_parameters,
    census_recomputed,
    census_saved_ops,
    census_stack,
    census_traffic,
    replacement_specs,
    _classes,
    _delta_rule_held,
    _matmul_flops,
    _measure,
    _modeling,
    _positions,
    _RecomputedMatmuls,
    _ForwardContext,
    _census_policy,
    _checkpointing,
    _kernel_op,
    _selective_contexts,
    tp_config,
    _ATTENTION_KERNELS,
    delta_rule_module,
)
from hyper_parallel.auto_parallel._model_spec import KindActivations, ModelSpec, ModelSpecError
from hyper_parallel.auto_parallel._npu_contracts import npu_contracts
from hyper_parallel.models.replacement import module_replacement
from hyper_parallel.core.activation_memory import api as activation_memory
from hyper_parallel.core.activation_memory import sac
from hyper_parallel.core.activation_memory.policy import CheckpointPolicy
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2 import EvaluatorV2
from hyper_parallel.auto_parallel.sapp_nd.nd.common.arch_hooks import bind_layer_stack

_HF_CONFIG = "hyper_parallel.auto_parallel._hf_model_spec._get_hf_config"
_CAUSAL_LM = "hyper_parallel.models._transformers.HyperAutoModelForCausalLM.from_pretrained"
_IMAGE_TEXT = "hyper_parallel.models._transformers.HyperAutoModelForImageTextToText.from_pretrained"


def _qwen35_text():
    """A two-layer Qwen3.5-MoE text config, one layer of each attention kind."""
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (  # pylint: disable=C0415
        Qwen3_5MoeTextConfig,
    )
    return Qwen3_5MoeTextConfig.from_dict({
        "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 16, "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
        "shared_expert_intermediate_size": 32, "linear_num_key_heads": 2, "linear_key_head_dim": 16,
        "linear_num_value_heads": 4, "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4,
        "vocab_size": 128, "max_position_embeddings": 256, "layer_types": ["linear_attention", "full_attention"],
    })


_STACK = [{"kind": "linear_attention", "count": 1}, {"kind": "full_attention", "count": 1}]

_BY_NAME_MODELING = textwrap.dedent('''
    import torch


    def torch_chunk_gated_delta_rule(query, key, value, g=None, beta=None, **kwargs):
        return value * 3, None


    class GatedDeltaNet(torch.nn.Module):
        def forward(self, hidden):
            return torch_chunk_gated_delta_rule(hidden, hidden, hidden, g=hidden, beta=hidden)[0]
''')


class _Layer(torch.nn.Module):
    """A layer of an attention projection and a feed-forward of two, 64 wide within 256."""

    def __init__(self) -> None:
        """Its three projections."""
        super().__init__()
        self.self_attn = torch.nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(64, 256, bias=False, dtype=torch.bfloat16), torch.nn.ReLU(),
                                       torch.nn.Linear(256, 64, bias=False, dtype=torch.bfloat16))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """The attention's projection, then the feed-forward."""
        return self.mlp(self.self_attn(hidden))


def _qwen35(text):
    """The vision-language config around *text*, its vision tower Transformers' default."""
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (  # pylint: disable=C0415
        Qwen3_5MoeConfig,
    )
    return Qwen3_5MoeConfig.from_dict({"text_config": text.to_dict()})


def _train_yaml(folder: str, target: str = _CAUSAL_LM, mode: str = "off", pp: int = 1,
                plan_overrides: Optional[list] = None, **context) -> str:
    """A train yaml of the hybrid model on two ranks, at DP shard 2 over PP 1 or on two stages."""
    config = {
        "model": {"_target_": target, "pretrained_model_name_or_path": "local/qwen3_5_moe",
                  "torch_dtype": "bfloat16"},
        "training": {"global_batch_size": 4, "micro_batch_size": 1},
        "accelerator": {"tp_size": 1, "pp_size": pp, "ep_size": 1, "cp_size": 1},
        "fsdp_config": {"dp_shard_size": 2 // pp},
        "activation_checkpoint": {"mode": mode},
        "dataset": {"data_transform": {"max_seq_len": 4096}},
        "context": dict(context, device_num=2),
    }
    if plan_overrides:
        config["plan_overrides"] = plan_overrides
    path = os.path.join(folder, "train.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)
    return path


def _evaluator(model_config, **train) -> EvaluatorV2:
    """The evaluator of :func:`_train_yaml`'s run of *model_config*."""
    with patch(_HF_CONFIG, return_value=model_config):
        with tempfile.TemporaryDirectory() as folder:
            return EvaluatorV2(_train_yaml(folder, **train), framework="hyper_v2", log_level=0)


def _spec(**activations) -> dict:
    """A two-kind hybrid spec stating *activations*."""
    return {
        "name": "qwen3_5_moe", "arch": "qwen3_5", "hidden_size": 64, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "vocab_size": 128,
        "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
        "layers": _STACK, "activations": activations,
    }


class TestLayerCensus(unittest.TestCase):
    """The census counts what a fake layer keeps and holds."""

    def test_a_tensor_parallel_rank_holds_a_share(self):
        """
        Feature: tp_config and census_layer.
        Description: Each kind's layer, and the same layer as one of two
            tensor-parallel ranks holds it.
        Expectation: The rank's layer has half the heads and widths, the same
            head width, and keeps and holds less, but more than half.
        """
        config = _qwen35_text()
        rank = tp_config(config, 2)
        self.assertEqual((rank.num_attention_heads, rank.num_key_value_heads, rank.head_dim), (2, 1, 16))
        self.assertEqual((rank.moe_intermediate_size, rank.linear_num_value_heads), (16, 2))
        for index in (0, 1):
            whole, share = census_layer(config, index, 64), census_layer(rank, index, 64)
            for full, half in zip(whole, share):
                self.assertGreater(full, half)
                self.assertGreater(2 * half, full)

    def test_a_gradient_the_size_of_a_parameter_is_an_activations(self):
        """
        Feature: _measure, which tells gradients apart.
        Description: A projection of a 64-wide input of 64 tokens, whose
            input's gradient has as many elements as its weight.
        Expectation: The forward keeps the input; when the backward peaks,
            the input, the output's gradient and the input's are the
            activations held, and the weight's gradient is not.
        """
        with FakeTensorMode():
            weight = torch.nn.Parameter(torch.empty(64, 64, dtype=torch.bfloat16))
            hidden = torch.empty(64, 64, dtype=torch.bfloat16, requires_grad=True)
            size = hidden.untyped_storage().nbytes()
            saved, working = _measure([weight], (hidden,), lambda: hidden @ weight.t(),
                                      lambda out: out.backward(torch.ones_like(out)))
        self.assertEqual((saved, working), (size, 3 * size))

    def test_a_graph_the_forward_drops_is_not_kept(self):
        """
        Feature: _measure, which counts what a forward still holds (D16).
        Description: A projection of a 64-wide input of 64 tokens whose
            output is gathered at indices a selection picks, the selection
            scoring the input through a second weight in float32 and keeping
            only the top indices, as V4.1's Indexer does; then the same
            forward with the selection under no_grad.
        Expectation: No backward reaches the scores, so the forward drops
            their graph once the indices are taken and both forwards keep
            and hold the same bytes, where the census counted that graph as
            kept, its own pack hook holding it alive.
        """
        def forward(graphed: bool) -> Callable[[], torch.Tensor]:
            """The forward, its selection recording a graph where *graphed*."""
            def run() -> torch.Tensor:
                """The projection gathered at the indices the selection picks."""
                with contextlib.nullcontext() if graphed else torch.no_grad():
                    chosen = (hidden @ score.t()).float().relu().topk(8, dim=-1).indices
                return (hidden @ weight.t()).gather(-1, chosen)
            return run

        got = []
        for graphed in (True, False):
            with FakeTensorMode():
                weight, score = (torch.nn.Parameter(torch.empty(64, 64, dtype=torch.bfloat16)) for _ in range(2))
                hidden = torch.empty(64, 64, dtype=torch.bfloat16, requires_grad=True)
                got.append(_measure([weight, score], (hidden,), forward(graphed),
                                    lambda out: out.backward(torch.ones_like(out))))
        self.assertEqual(got[0], got[1])

    def test_selective_checkpointing_keeps_every_other_matmul(self):
        """
        Feature: _measure of a forward under HyperParallel's selective
            activation checkpointing.
        Description: Two projections of a 64-wide input of 64 tokens with
            an activation between them, checkpointed as the trainer
            checkpoints a layer.
        Expectation: The forward keeps its input and the first projection's
            output, which the policy saves; it recomputes the activation
            and the second projection.
        """
        with FakeTensorMode():
            first, second = (torch.nn.Parameter(torch.empty(64, 64, dtype=torch.bfloat16)) for _ in range(2))
            hidden = torch.empty(64, 64, dtype=torch.bfloat16, requires_grad=True)
            size = hidden.untyped_storage().nbytes()
            kept, _ = _measure(
                [first, second], (hidden,),
                lambda: activation_memory.checkpoint(lambda states: torch.relu(states @ first.t()) @ second.t(),
                                                     hidden, swap_inputs=False, context_fn=_selective_contexts),
                lambda out: out.backward(torch.ones_like(out)), checkpointed=True)
        self.assertEqual(kept, 2 * size)

    def test_the_policy_runs_every_other_matmul_again(self):
        """
        Feature: the census's ledger of HyperParallel's selective policy.
        Description: A layer of an attention projection and a feed-forward
            of two, of 64 tokens, run under selective checkpointing.
        Expectation: The policy saves the first and the third projections'
            outputs and runs the second again: none of the attention's
            FLOPs, and half the feed-forward's.
        """
        with FakeTensorMode():
            layer = _Layer()
            hidden = torch.empty(64, 64, dtype=torch.bfloat16, requires_grad=True)
            ledger = _RecomputedMatmuls(layer)
            activation_memory.checkpoint(layer, hidden, swap_inputs=False,
                                         context_fn=functools.partial(_selective_contexts, ledger))
        self.assertEqual((ledger.share(attention=True), ledger.share(attention=False)), (0.0, 0.5))

    def test_a_selective_layer_keeps_what_the_policy_saves(self):
        """
        Feature: census_layer under HyperParallel's selective checkpointing.
        Description: Each kind's layer checkpointed selectively; the
            full-attention one also with SAC ignoring the ops that build
            empty tensors, as the trainer's selective setup has it.
        Expectation: A layer keeps less than without recompute and more
            than its input; the full-attention one keeps the attention's
            output and two statistics, 24 KiB at 64 tokens, whatever SAC
            ignores.
        """
        config = _qwen35_text()
        size = 64 * 64 * 2
        for index in (0, 1):
            (saved, _), (kept, _) = (census_layer(config, index, 64, selective=each) for each in (False, True))
            self.assertLess(kept, saved)
            self.assertGreater(kept, size)
        self.assertGreaterEqual(kept, size + 24 * 1024)
        ignored = set(sac.SAC_IGNORED_OPS) | {torch.ops.aten.empty.memory_format, torch.ops.aten.empty_like.default}
        with patch.object(sac, "SAC_IGNORED_OPS", ignored):
            self.assertEqual(census_layer(config, 1, 64, selective=True)[0], kept)

    def test_a_census_runs_on_transformers_grouped_mm_fallback(self):
        """
        Feature: census_layer where torch has no CPU grouped_mm kernel.
        Description: Each kind's MoE layer censused through Transformers'
            grouped_mm fallback, which it takes on CPU with torch 2.8 or
            older, the 910C's 2.7.1 among them; its backward reads the group
            offsets off a fake tensor, which holds no data.
        Expectation: The census runs, and the layer keeps what it keeps on
            torch's own grouped_mm, but for the offsets, which the fallback
            holds on its context, out of reach of autograd's saved-tensor hooks.
        """
        config = _qwen35_text()
        for index in (0, 1):
            native = census_layer(config, index, 64)
            with patch("transformers.integrations.moe._can_use_grouped_mm", return_value=False):
                fallback = census_layer(config, index, 64)
            self.assertLessEqual(abs(native[0] - fallback[0]), 8 * config.num_experts)

    def test_the_flop_count_needs_no_grouped_mm(self):
        """
        Feature: _matmul_flops on a torch without aten._grouped_mm.
        Description: A matmul and an addition counted where aten has no
            _grouped_mm, which first shipped in torch 2.8.
        Expectation: The matmul's FLOPs are counted and the addition's are 0.
        """
        aten = torch.ops.aten

        class _AtenWithoutGroupedMM:
            """aten as torch 2.7 has it."""

            def __getattr__(self, name):
                if name == "_grouped_mm":
                    raise AttributeError(name)
                return getattr(aten, name)

        left, right = torch.ones(4, 8), torch.ones(8, 2)
        with patch.object(torch.ops, "aten", _AtenWithoutGroupedMM()):
            self.assertEqual(_matmul_flops(aten.mm.default, (left, right)), 2 * 4 * 8 * 2)
            self.assertEqual(_matmul_flops(aten.add.Tensor, (left, left)), 0)

    def test_the_census_takes_the_delta_rule_a_run_takes(self):
        """
        Feature: census_layer's gated delta rule.
        Description: A linear-attention layer censused as the runtime's
            dispatcher runs it, which defaults to Transformers' chunked
            implementation, and censused with HyperParallel's kernel, whose
            contract saves the rule's inputs, its cumulated gates, beta and
            one chunk matrix and nothing more.
        Expectation: The default is the eager rule, the kernel's contract
            keeps strictly less, and a backend the runtime does not have is
            refused.  A full-attention layer, which runs no delta rule, is
            the same either way.
        """
        config = _qwen35_text()
        eager, _ = census_layer(config, 0, 64)
        kernel, _ = census_layer(config, 0, 64, gdn_backend="triton")
        self.assertEqual(census_layer(config, 0, 64, gdn_backend="eager")[0], eager)
        self.assertLess(kernel, eager)
        self.assertEqual(census_layer(config, 1, 64, gdn_backend="triton")[0],
                         census_layer(config, 1, 64)[0])
        with self.assertRaises(ValueError):
            census_layer(config, 0, 64, gdn_backend="fla")

    def test_a_rotary_on_three_grids_gets_a_row_of_ids_for_each(self):
        """
        Feature: _positions, the position ids a layer's rotary embedding is handed.
        Description: Both kinds of a Qwen3.5 layer censused with a rotary that
            refuses a single row of ids, as Transformers 5.17 and 5.18 build
            it, then a rotary on one grid.
        Expectation: Each kind keeps what it keeps where the rotary expands
            the row itself, so the census runs on either; the rotary on one
            grid is handed the one row.
        """
        config = _qwen35_text()
        expanding = [census_layer(config, index, 64) for index in (0, 1)]
        rotary_cls = importlib.import_module(
            "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe").Qwen3_5MoeTextRotaryEmbedding
        forward = rotary_cls.forward

        def refusing(rotary: Any, hidden: torch.Tensor, position_ids: torch.Tensor) -> Any:
            """The rotary as Transformers 5.17 builds it, which takes three rows of ids and no fewer."""
            if position_ids.ndim != 3:
                raise IndexError("too many indices for tensor of dimension 2")
            return forward(rotary, hidden, position_ids)

        with patch.object(rotary_cls, "forward", refusing):
            self.assertEqual([census_layer(config, index, 64) for index in (0, 1)], expanding)
        handed = []
        _positions(lambda hidden, ids: handed.append(tuple(ids.shape)), torch.zeros(1, 8, 4), 8)
        self.assertEqual(handed, [(1, 8)])

    def test_the_delta_rule_is_reached_however_transformers_calls_it(self):
        """
        Feature: delta_rule_module and _delta_rule_held.
        Description: A gated delta rule attention calling its modeling
            module's torch_chunk_gated_delta_rule by name and holding no
            rule, as Transformers 5.15 and later build it, one holding it, as
            earlier versions did, and a plain projection.
        Expectation: The first two run a delta rule and the projection none.
            While the census holds the layer, the one calling by name runs
            the rule the census gives it, through the attribute the census
            follows; afterwards the name is its modeling module's own again.
        """
        modeling = types.ModuleType("_by_name_modeling")
        exec(_BY_NAME_MODELING, modeling.__dict__)  # pylint: disable=exec-used
        by_name = modeling.__dict__["GatedDeltaNet"]()
        holding = torch.nn.Linear(2, 2)
        holding.chunk_gated_delta_rule = modeling.torch_chunk_gated_delta_rule
        self.assertEqual([delta_rule_module(module) for module in (by_name, holding, torch.nn.Linear(2, 2))],
                         [True, True, False])
        original = modeling.torch_chunk_gated_delta_rule
        layer = torch.nn.Sequential(by_name)
        hidden = torch.ones(2)
        followed = []
        with _delta_rule_held(modeling, layer, lambda query, key, value, **kwargs: (value + 1, None)):
            rule = by_name.chunk_gated_delta_rule
            by_name.chunk_gated_delta_rule = lambda *args, **kwargs: followed.append(True) or rule(*args, **kwargs)
            self.assertEqual(layer(hidden).tolist(), [2.0, 2.0])
        self.assertEqual(followed, [True])
        self.assertIs(modeling.torch_chunk_gated_delta_rule, original)
        self.assertEqual(layer(hidden).tolist(), [3.0, 3.0])

    def test_each_kind_gets_its_record(self):
        """
        Feature: census_activations.
        Description: The census of a stack of one linear-attention layer and
            one full-attention layer.
        Expectation: A record per kind, at the census's length, splitting
            each layer's bytes between what TP splits and what it does not,
            with and without selective checkpointing.
        """
        got = census_activations(_qwen35_text(), _STACK, 64)
        self.assertEqual(sorted(got), ["full_attention", "linear_attention"])
        for kind, index in (("linear_attention", 0), ("full_attention", 1)):
            record = got[kind]
            saved, working = census_layer(_qwen35_text(), index, 64)
            kept, _ = census_layer(_qwen35_text(), index, 64, selective=True)
            self.assertEqual(record.seq_length, 64)
            self.assertAlmostEqual((record.saved + record.saved_tp) * 64, saved)
            self.assertAlmostEqual((record.working + record.working_tp) * 64, working)
            self.assertAlmostEqual((record.selective + record.selective_tp) * 64, kept)
            self.assertEqual((record.selective_attention_mm, record.selective_ffn_mm),
                             census_recomputed(_qwen35_text(), index, 64))
            self.assertGreater(record.saved_tp, 0)
            self.assertGreater(record.selective_tp, 0)
            for share in (record.selective_attention_mm, record.selective_ffn_mm):
                self.assertTrue(0 < share < 1, share)
            self.assertAlmostEqual(sum(record.ops.values()), record.saved)
            self.assertAlmostEqual(sum(record.ops_tp.values()), record.saved_tp)
        self.assertGreater(got["linear_attention"].ops_tp["linrec"], 0)
        self.assertGreater(got["full_attention"].ops_tp["attBMM"], 0)

    def test_a_census_checkpoints_as_a_cpu_whatever_the_default_device(self):
        """
        Feature: census_layer and census_recomputed under selective
            checkpointing.
        Description: A selective full-attention layer measured with
            activation checkpointing's default device type left as it is,
            then with the default set to a type this host has no module for,
            as ``torch_npu`` sets its own on a host whose NPU cannot start.
        Expectation: Both measure the same, since a census replays its
            checkpoints as a CPU's and never reaches the default's module;
            the default is the one set once the census is done.
        """
        expected = (census_layer(_qwen35_text(), 1, 64, selective=True), census_recomputed(_qwen35_text(), 1, 64))
        before = DefaultDeviceType.get_device_type()
        DefaultDeviceType.set_device_type("nd_no_such_device")
        try:
            got = (census_layer(_qwen35_text(), 1, 64, selective=True), census_recomputed(_qwen35_text(), 1, 64))
            after = DefaultDeviceType.get_device_type()
        finally:
            DefaultDeviceType.set_device_type(before)
        self.assertEqual((got, after), (expected, "nd_no_such_device"))

    def test_what_a_layer_saves_for_each_op(self):
        """
        Feature: census_saved_ops, the op records the census fills.
        Description: A Llama layer of width 64, 4 query heads and 2 key
            heads 16 wide, a gated feed-forward 128 wide, in bf16.
        Expectation: Per token: the projections' input, the rotary tables
            and the output projection's input for the attention's
            projections; the kernel's queries, keys, values and output for
            its batched matmuls, and its fp32 statistics for the softmax;
            each RMSNorm's fp32 input, its scale and its bf16 output; the
            feed-forward's input, its gate and up outputs as its product
            takes them and that product for its projections, and the gate's
            output for its activation function.  They sum to what the layer
            keeps.
        """
        from transformers import LlamaConfig  # pylint: disable=C0415
        config = LlamaConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             intermediate_size=128, vocab_size=128)
        ops = census_saved_ops(config, 0, 64)
        self.assertEqual(ops, {
            "attMM": 2 * (64 + 2 * 16 + 64), "attBMM": 2 * (4 * 16 + 2 * 2 * 16 + 4 * 16), "softmax": 4 * 2 * 4 * 8,
            "normOp": 2 * (4 * 64 + 4 + 2 * 64), "ffMM": 2 * (64 + 3 * 128), "ffAct": 2 * 128})
        self.assertEqual(sum(ops.values()) * 64, census_layer(config, 0, 64)[0])

    def test_a_layers_parameters_by_part(self):
        """
        Feature: census_parameters and census_final_norm.
        Description: Each kind's layer of the model of width 64, with 4
            routed experts and a shared expert, each 32 wide.
        Expectation: The router holds a weight per expert; the routed
            experts three projections each; the shared expert its three and
            its gate; the norms two RMSNorms of the width, and in the
            full-attention layer the per-head query and key norms beside
            them.  The final norm is one RMSNorm.
        """
        config = _qwen35_text()
        for index, norms in ((0, 2 * 64), (1, 2 * 64 + 2 * 16)):
            parts = census_parameters(config, index)
            self.assertEqual((parts["router"], parts["routed"], parts["shared"], parts["norm"]),
                             (64 * 4, 4 * 3 * 64 * 32, 3 * 64 * 32 + 64, norms))
            self.assertGreater(parts["attention"], 0)
        self.assertEqual(census_final_norm(config), 64)

    def test_a_layers_forward_flops_by_part(self):
        """
        Feature: census_flops.
        Description: Each kind's layer of the model of width 64 on 32
            tokens: 4 query heads of 16 behind an output gate and 2 key
            heads; 4 routed experts 32 wide, 2 chosen, a shared expert 32
            wide and its gate.
        Expectation: The full-attention layer's projections; its scores and
            values at every pair of tokens; each token's two experts' three
            projections, the shared expert's three and its gate, and the
            router's.  The linear-attention layer's recurrence is counted
            apart from its projections, and it has no scores.
        """
        config, seq = _qwen35_text(), 32
        full = census_flops(config, 1, seq)
        self.assertEqual(full["attention"], 2 * seq * 64 * (2 * 4 * 16 + 2 * 2 * 16 + 4 * 16))
        self.assertEqual(full["scores"], 2 * 4 * seq * seq * (16 + 16))
        self.assertEqual((full["routed"], full["shared"], full["router"]),
                         (2 * seq * 2 * 3 * 64 * 32, 2 * seq * (3 * 64 * 32 + 64), 2 * seq * 64 * 4))
        linear = census_flops(config, 0, seq)
        self.assertGreater(linear["linrec"], 0)
        self.assertNotIn("scores", linear)

    def test_a_layers_forward_traffic_by_part(self):
        """
        Feature: census_traffic.
        Description: Each kind's layer of the model of width 64 on 32
            tokens, under Transformers' own chunked gated delta rule and
            under the kernel's contract.
        Expectation: Each part reads its own parameters once in a forward,
            so their bytes are its parameter count at two bytes apiece, and
            a part never moves fewer bytes than its parameters hold.  A
            linear-attention layer's recurrence moves the most bytes and
            dispatches the most ops of it, and moves far more for each FLOP
            it runs than the projections do, which is what a model priced by
            FLOPs alone leaves out.  A full-attention layer has a scores
            part, which the attention kernel runs in one op, and no
            recurrence.  The kernel's contract collapses the recurrence,
            where Transformers' rule runs matmuls a chunk at a time, and
            moves nothing else of the layer differently.
        """
        config, seq = _qwen35_text(), 32
        flops = census_flops(config, 0, seq)
        linear, full = (census_traffic(config, index, seq) for index in (0, 1))
        for index, traffic in ((0, linear), (1, full)):
            parameters = census_parameters(config, index)
            self.assertEqual(traffic.parameter,
                             {part: 2.0 * parameters.get(part, 0) for part in traffic.parameter})
            for part, moved in traffic.moved.items():
                self.assertGreaterEqual(moved, traffic.parameter[part], part)
        self.assertEqual(max(linear.moved, key=linear.moved.get), "linrec")
        self.assertEqual(max(linear.launches, key=linear.launches.get), "linrec")
        self.assertGreater(linear.moved["linrec"] / flops["linrec"],
                           linear.moved["attention"] / flops["attention"])
        self.assertEqual(full.launches["scores"], 1)
        self.assertNotIn("linrec", full.moved)
        kernel = census_traffic(config, 0, seq, gdn_backend="triton")
        self.assertLess(kernel.launches["linrec"], linear.launches["linrec"])
        self.assertLess(kernel.moved["linrec"], linear.moved["linrec"])
        self.assertEqual({part: size for part, size in kernel.moved.items() if part != "linrec"},
                         {part: size for part, size in linear.moved.items() if part != "linrec"})

    def test_mla_values_keep_their_width(self):
        """
        Feature: census_flops and census_layer on MLA.
        Description: A DeepSeek-V3 layer of width 64 on 32 tokens: 4 heads
            whose queries and keys are 16 wide, 12 and a rotary 4, and whose
            values are 8.
        Expectation: The scores run at the queries' and keys' width and the
            values at their own, unpadded, as under HyperParallel's sdpa; the
            layer runs its backward.
        """
        from transformers import DeepseekV3Config  # pylint: disable=C0415
        config = DeepseekV3Config(
            hidden_size=64, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4, q_lora_rank=32,
            kv_lora_rank=16, qk_nope_head_dim=12, qk_rope_head_dim=4, v_head_dim=8, intermediate_size=32,
            first_k_dense_replace=1, vocab_size=128, n_group=1, topk_group=1)
        self.assertEqual(census_flops(config, 0, 32)["scores"], 2 * 4 * 32 * 32 * (16 + 8))
        self.assertGreater(census_layer(config, 0, 32)[0], 0)

    def test_the_output_layer_keeps_its_fp32_log_probabilities(self):
        """
        Feature: census_output_activations.
        Description: The output layer of the model with a vocabulary of
            4096, which its logits outweigh.
        Expectation: Per token and vocabulary entry, the layer keeps the
            loss's fp32 log-probabilities and its backward holds three such
            tensors: the part a vocabulary-parallel loss splits.
        """
        config = _qwen35_text()
        config.vocab_size = 4096
        record = census_output_activations(config, 48)
        self.assertEqual((record.saved_tp, record.working_tp, record.seq_length), (4 * 4096, 12 * 4096, 48))
        self.assertGreater(record.saved, 0)


class TestKindActivations(unittest.TestCase):
    """The model spec states a census's records."""

    def test_round_trip_through_yaml(self):
        """
        Feature: ModelSpec.activations.
        Description: A spec stating each kind's record, dumped to YAML and
            read back.
        Expectation: The same spec.
        """
        record = {"saved": 2036.25, "saved_tp": 3812.5, "working": 2162.375, "working_tp": 3812.5,
                  "seq_length": 64}
        selective = dict(record, selective=512.0, selective_tp=1024.5, selective_attention_mm=0.25,
                         selective_ffn_mm=0.625, ops={"attMM": 1000.25, "other": 1036.0},
                         ops_tp={"attBMM": 3812.5})
        spec = ModelSpec.from_dict(dict(_spec(linear_attention=selective, full_attention=record),
                                         output_activations=record))
        self.assertEqual(spec.activations["linear_attention"].to_dict(), selective)
        self.assertEqual(spec.activations["full_attention"].to_dict(), record)
        self.assertIsInstance(spec.activations["linear_attention"], KindActivations)
        self.assertIsInstance(spec.output_activations, KindActivations)
        self.assertEqual(ModelSpec.from_dict(yaml.safe_load(yaml.safe_dump(spec.to_dict()))), spec)

    def test_a_record_is_checked(self):
        """
        Feature: KindActivations.from_dict and ModelSpec.validate.
        Description: A negative size, a missing field, what a layer keeps
            for each op without the part TP splits, an op the op vector
            lacks, a negative op's size, and a kind no layer of the stack
            is.
        Expectation: Each raises, naming what is wrong.
        """
        record = {"saved": 1.0, "saved_tp": 1.0, "working": 1.0, "working_tp": 1.0, "seq_length": 64}
        with self.assertRaisesRegex(ModelSpecError, "negative"):
            ModelSpec.from_dict(_spec(linear_attention=dict(record, saved=-1.0)))
        with self.assertRaisesRegex(ModelSpecError, "lacks"):
            ModelSpec.from_dict(_spec(linear_attention={"saved": 1.0}))
        with self.assertRaisesRegex(ModelSpecError, "without the other"):
            ModelSpec.from_dict(_spec(linear_attention=dict(record, selective=1.0)))
        with self.assertRaisesRegex(ModelSpecError, "exceed 1"):
            ModelSpec.from_dict(_spec(linear_attention=dict(record, selective_attention_mm=0.5,
                                                            selective_ffn_mm=1.5)))
        with self.assertRaisesRegex(ModelSpecError, "without the other"):
            ModelSpec.from_dict(_spec(linear_attention=dict(record, ops={"attMM": 1.0})))
        with self.assertRaisesRegex(ModelSpecError, "must map ops"):
            ModelSpec.from_dict(_spec(linear_attention=dict(record, ops={"matmul": 1.0}, ops_tp={})))
        with self.assertRaisesRegex(ModelSpecError, "negative"):
            ModelSpec.from_dict(_spec(linear_attention=dict(record, ops={"attMM": -1.0}, ops_tp={})))
        with self.assertRaisesRegex(ModelSpecError, "decoder"):
            ModelSpec.from_dict(_spec(decoder=record))
        with self.assertRaisesRegex(ModelSpecError, "output_activations must map"):
            ModelSpec.from_dict(dict(_spec(), output_activations=[1.0]))


# The recipe whose plan_overrides install HyperParallel's fused modules on
# the Qwen3-MoE Transformers builds, where the package that runs keeps it,
# which need not be beside these tests: a gate runs a copy of them against
# the installed package.
_RECIPE = os.path.join(
    os.path.dirname(os.path.abspath(hyper_parallel.__file__)),
    "models", "qwen3_moe", "recipes", "train.yaml",
)


def _qwen3_moe():
    """A two-layer Qwen3-MoE config of width 64, 4 experts 32 wide, as the recipe's rules match it."""
    from transformers import Qwen3MoeConfig  # pylint: disable=C0415
    return Qwen3MoeConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                          head_dim=16, moe_intermediate_size=32, num_experts=4, num_experts_per_tok=2,
                          intermediate_size=128, vocab_size=128)


def _recipe_specs():
    """The module replacements the shipped Qwen3-MoE recipe installs."""
    with open(_RECIPE, encoding="utf-8") as handle:
        return replacement_specs(yaml.safe_load(handle).get("plan_overrides"))


# The operator HyperParallel's fused attention dispatches to on the NPU.
_NPU_ATTENTION = "npu.npu_fusion_attention"


@contextlib.contextmanager
def _policy_naming(checkpointing: Any, names: tuple):
    """The trainer's selective policy, its list of the compute operators it saves replaced by *names*."""
    # pylint: disable=protected-access
    with patch.object(checkpointing, "_SELECTIVE_AC_COMPUTE_OP_NAMES", names):
        with patch.object(checkpointing, "_SELECTIVE_AC_MUST_SAVE_OPS",
                          checkpointing._build_selective_ac_must_save_ops()):
            yield


@module_replacement
def _needs_a_library(*, module, module_fqn, context):
    """A factory of a module this host cannot build."""
    del module, module_fqn, context
    raise ImportError("No module named 'a_native_extension'")


class TestModuleReplacements(unittest.TestCase):
    """The census runs the modules a train.yaml's plan_overrides install."""

    def test_the_entries_that_replace_a_module(self):
        """
        Feature: replacement_specs.
        Description: The shipped Qwen3-MoE recipe, whose plan_overrides state
            three replacements and two entries that shard alone, under
            ``when: cp`` and ``when: ep``.
        Expectation: One rule per replacement, in the order stated, each with
            its patterns, its factory and the source type it replaces; the
            sharding entries are left out.
        """
        specs = _recipe_specs()
        self.assertEqual([spec.factory.__name__ for spec in specs],
                         ["replace_qwen3_moe_rms_norm", "replace_qwen3_moe_flash_attention",
                          "replace_qwen3_moe_grouped_experts"])
        self.assertEqual(specs[0].match, ("*.input_layernorm", "*.post_attention_layernorm", "model.norm"))
        self.assertEqual([spec.module_type.__name__ for spec in specs],
                         ["Qwen3MoeRMSNorm", "Qwen3MoeAttention", "Qwen3MoeExperts"])

    def test_an_entry_is_checked(self):
        """
        Feature: replacement_specs' checks.
        Description: An entry that replaces a module without stating its
            type, one naming a module that cannot be imported, and one
            naming no factory of that name.
        Expectation: Each raises, naming the entry and what is wrong.
        """
        entry = {"match": "*.self_attn", "module_type": "torch.nn.Linear",
                 "replace_module": {"_target_": "hyper_parallel.models.replacement.module_replacement"}}
        cases = (
            ({**entry, "module_type": None}, "without module_type"),
            ({**entry, "module_type": "no.such.module.Type"}, "cannot import"),
            ({**entry, "replace_module": {"_target_": "torch.nn.no_such_factory"}}, "cannot import"),
        )
        for raw, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                replacement_specs([raw])

    def test_a_fused_layer_keeps_less_than_an_eager_one(self):
        """
        Feature: census_layer with replacements.
        Description: A Qwen3-MoE layer as Transformers builds it, and with
            the recipe's fused RMSNorm, grouped-query attention and grouped
            experts installed, at 256 tokens, plain and under HyperParallel's
            selective policy.
        Expectation: The fused layer keeps less between its passes and less
            under the policy, and its backward holds less; it holds the
            parameters its source held, so the parts count the same.
        """
        config, specs = _qwen3_moe(), _recipe_specs()
        eager = census_layer(config, 0, 256)
        fused = census_layer(config, 0, 256, replacements=specs)
        self.assertLess(fused[0], eager[0])
        self.assertLess(fused[1], eager[1])
        self.assertLess(census_layer(config, 0, 256, selective=True, replacements=specs)[0],
                        census_layer(config, 0, 256, selective=True)[0])
        self.assertEqual(census_parameters(config, 0, replacements=specs), census_parameters(config, 0))

    def test_what_the_fused_kernels_save_is_named_by_op(self):
        """
        Feature: census_saved_ops with replacements.
        Description: The same layer's bytes a token by op, eager and fused.
        Expectation: The fused layer's norms keep less, since its kernel
            keeps one reciprocal root mean square a row where the eager norm
            keeps its upcast input; its attention kernel's saves count toward
            the batched matmuls and its statistics toward the softmax, and
            they sum to what the layer keeps.
        """
        config, specs = _qwen3_moe(), _recipe_specs()
        eager = census_saved_ops(config, 0, 256)
        fused = census_saved_ops(config, 0, 256, replacements=specs)
        self.assertLess(fused["normOp"], eager["normOp"])
        self.assertGreater(fused["softmax"], 0)
        self.assertGreater(fused["attBMM"], 0)
        self.assertAlmostEqual(sum(fused.values()) * 256, census_layer(config, 0, 256, replacements=specs)[0])

    def test_a_census_leaves_the_runtimes_caches_as_it_found_them(self):
        """
        Feature: the caches HyperParallel's fused modules fill, after a census.
        Description: The Qwen3-MoE attention caches the compressed causal mask
            its sparse mode takes, one per device, the first time it runs.
        Expectation: A census of the fused layer leaves no fake tensor in that
            cache, so a real call in the same process builds its own mask
            (F55).
        """
        from hyper_parallel.models.qwen3_moe.adapter.runtime import attention  # pylint: disable=C0415
        census_layer(_qwen3_moe(), 0, 256, replacements=_recipe_specs())
        cached = attention._COMPRESSED_CAUSAL_MASKS  # pylint: disable=protected-access
        self.assertEqual([mask for mask in cached.values() if isinstance(mask, FakeTensor)], [])

    def test_a_replacement_this_host_cannot_build_is_refused(self):
        """
        Feature: census_layer with a replacement whose factory needs a
            library this host lacks, as DeepSeek-V3.2's attention needs a
            native extension.
        Expectation: The census refuses rather than measure Transformers' own
            modules, which the run does not train, and names the factory and
            the import that failed (F53).
        """
        from hyper_parallel.models.replacement import ModuleReplacementSpec  # pylint: disable=C0415
        from transformers.models.qwen3_moe import modeling_qwen3_moe  # pylint: disable=C0415
        spec = ModuleReplacementSpec(match=("*.input_layernorm",), factory=_needs_a_library,
                                     module_type=modeling_qwen3_moe.Qwen3MoeRMSNorm)
        with self.assertRaises(CensusUnavailable) as refusal:
            census_layer(_qwen3_moe(), 0, 256, replacements=(spec,))
        self.assertIn("a_native_extension", str(refusal.exception))
        self.assertIn("_needs_a_library", str(refusal.exception))

    def test_a_run_whose_modules_do_not_build_is_priced_by_the_formulas(self):
        """
        Feature: context.census with a plan_overrides entry this host cannot
            build, through the parser.
        Expectation: The run is priced without a census, as a run that asks
            for none, and a warning says why (F53).
        """
        plan = [{"match": "*.input_layernorm",
                 "module_type": "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeRMSNorm",
                 "replace_module": {"_target_": f"{__name__}._needs_a_library"}}]
        with self.assertLogs("hyper_parallel.auto_parallel._hf_model_spec", "WARNING") as logs:
            priced = _evaluator(_qwen35_text(), census=True, plan_overrides=plan)
        self.assertIsNone(priced.ccfg.census)
        self.assertEqual(priced.estimate_peak(), _evaluator(_qwen35_text()).estimate_peak())
        self.assertIn("a_native_extension", "".join(logs.output))

    def test_the_census_saves_what_the_installed_policy_saves(self):
        """
        Feature: the census's selective policy, beside the policy the
            installed trainer runs.
        Description: Each of the census's attention kernels under the
            census's policy, and the runtime operator it stands in for under
            the trainer's: the trainer's list of the operators it saves as
            installed, then without those two operators, then with both.
        Expectation: The census saves a kernel's outputs exactly where the
            trainer's policy saves its operator's, whatever that list names:
            each kernel is saved under one list and recomputed under another
            (F52).
        """
        checkpointing = _checkpointing()
        installed = checkpointing._SELECTIVE_AC_COMPUTE_OP_NAMES  # pylint: disable=protected-access
        operators = tuple(_ATTENTION_KERNELS.values())
        lists = {"installed": installed,
                 "without": tuple(name for name in installed if name not in operators),
                 "with": tuple(dict.fromkeys((*installed, *operators)))}
        decided = {kernel: set() for kernel in _ATTENTION_KERNELS}
        for case, names in lists.items():
            with self.subTest(case=case), _policy_naming(checkpointing, names):
                census = _census_policy()
                trainer = checkpointing._make_selective_checkpoint_policy_fn()  # pylint: disable=protected-access
                for kernel, name in _ATTENTION_KERNELS.items():
                    operator = checkpointing._resolve_torch_op(name)  # pylint: disable=protected-access
                    expected = (name in names if operator is None
                                else trainer(_ForwardContext(), operator) == CheckpointPolicy.MUST_SAVE)
                    saves = census(_ForwardContext(), _kernel_op(kernel)) == CheckpointPolicy.MUST_SAVE
                    self.assertEqual(saves, expected, kernel)
                    decided[kernel].add(saves)
        self.assertEqual(decided, {kernel: {False, True} for kernel in _ATTENTION_KERNELS})

    def test_a_selective_fused_layer_keeps_the_attention_its_policy_saves(self):
        """
        Feature: census_layer, selective, with the Qwen3-MoE recipe's fused
            modules.
        Description: The trainer's policy without the fused attention's
            operator, ``npu.npu_fusion_attention``, in its list of the
            operators it saves, then with it; then the policy as installed.
        Expectation: Where the policy saves that operator, the layer keeps
            the kernel's output and its two fp32 statistics of every head
            and token besides, and nothing else; as installed, it keeps what
            the installed list says (F52).
        """
        config, specs = _qwen3_moe(), _recipe_specs()
        checkpointing = _checkpointing()
        installed = checkpointing._SELECTIVE_AC_COMPUTE_OP_NAMES  # pylint: disable=protected-access
        kept = {}
        for case, names in (("without", tuple(name for name in installed if name != _NPU_ATTENTION)),
                            ("with", tuple(dict.fromkeys((*installed, _NPU_ATTENTION))))):
            with _policy_naming(checkpointing, names):
                kept[case] = census_layer(config, 0, 256, selective=True, replacements=specs)[0]
        output = config.head_dim * 2
        statistics = 2 * 8 * 4
        self.assertEqual(kept["with"] - kept["without"], 256 * config.num_attention_heads * (output + statistics))
        self.assertEqual(census_layer(config, 0, 256, selective=True, replacements=specs)[0],
                         kept["with" if _NPU_ATTENTION in installed else "without"])

    def test_the_contracts_stand_in_for_the_kernels(self):
        """
        Feature: npu_contracts.
        Description: The stand-in entered and left, whether or not the host
            has ``torch_npu`` and the suite has imported it; then a kernel it
            has no contract for.
        Expectation: While it stands in, ``torch_npu`` is the contracts and
            its fused RMSNorm kernel returns the normalized rows and one
            statistic a row; after, what was bound before is bound again,
            nothing where nothing was; a kernel with no contract raises,
            naming it.
        """
        before = sys.modules.get("torch_npu")
        with npu_contracts() as stand_in:
            self.assertIs(sys.modules["torch_npu"], stand_in)
            rows = torch.randn(2, 8, dtype=torch.bfloat16)
            # The stand-in's kernels are set on it as it is built, so they
            # are looked up by name, as the fused modules' calls find them.
            out, rstd = getattr(stand_in, "npu_rms_norm")(rows, torch.ones(8, dtype=torch.bfloat16), 1e-6)
            self.assertEqual((tuple(out.shape), tuple(rstd.shape), rstd.dtype), ((2, 8), (2, 1), torch.float32))
            with self.assertRaisesRegex(NotImplementedError, "npu_no_such_kernel"):
                getattr(stand_in, "npu_no_such_kernel")(rows)
        self.assertIs(sys.modules.get("torch_npu"), before)

    def test_a_runs_replacements_reach_its_census(self):
        """
        Feature: context.census with plan_overrides, through the parser.
        Description: A train.yaml of the Qwen3-MoE asking for a census, with
            the recipe's plan_overrides and without them.
        Expectation: ND prices the run with what the census measured of the
            modules the run builds, which keep less than Transformers' own.
        """
        with open(_RECIPE, encoding="utf-8") as handle:
            entries = yaml.safe_load(handle)["plan_overrides"]
        peaks = {}
        for name, plan in (("eager", None), ("fused", entries)):
            with patch(_HF_CONFIG, return_value=_qwen3_moe()), tempfile.TemporaryDirectory() as folder:
                path = _train_yaml(folder, census=True, plan_overrides=plan)
                ccfg = EvaluatorV2(path, framework="hyper_v2", log_level=0).ccfg
                peaks[name] = EvaluatorV2(path, framework="hyper_v2", log_level=0).estimate_peak()
                self.assertIsNotNone(ccfg.census)
        self.assertLess(peaks["fused"], peaks["eager"])


class TestCensusPricing(unittest.TestCase):
    """``context.census`` has the memory model price a layer with its kind's census."""

    def test_the_resolver_states_each_kinds_census(self):
        """
        Feature: resolve_hf_model_spec's census_seq_len.
        Description: The hybrid model resolved twice with a census at 48
            tokens, and a spec of config_overrides alone.
        Expectation: The spec states each kind's record at that length, the
            census runs once; with no checkpoint config there is none.
        """
        model = {"pretrained_model_name_or_path": "local/qwen3_5_moe"}
        with patch(_HF_CONFIG, return_value=_qwen35_text()), patch(
                "hyper_parallel.auto_parallel._hf_model_spec.census_activations", wraps=census_activations) as run:
            spec = resolve_hf_model_spec(model, census_seq_len=48)
            self.assertEqual(resolve_hf_model_spec(model, census_seq_len=48), spec)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(sorted(spec["activations"]), ["full_attention", "linear_attention"])
        self.assertEqual(spec["activations"]["linear_attention"]["seq_length"], 48)
        self.assertEqual(spec["output_activations"]["seq_length"], 48)
        overrides = {"name": "llama", "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
                     "intermediate_size": 128, "vocab_size": 128}
        with self.assertLogs("hyper_parallel.auto_parallel._hf_model_spec", "WARNING"):
            plain = resolve_hf_model_spec({"config_overrides": overrides}, census_seq_len=48)
        self.assertNotIn("activations", plain)
        self.assertNotIn("output_activations", plain)

    def test_each_kind_binds_its_record(self):
        """
        Feature: the Hyper parser's context.census.
        Description: The hybrid model parsed with a census and without, and
            its vision-language checkpoint trained with its tower.
        Expectation: With it, the config holds each kind's record and the
            output layer's at the dataset's length, and each kind binds its
            own; the tower holds none; without it, there is none.
        """
        ccfg = _evaluator(_qwen35_text(), census=True).ccfg
        self.assertEqual({record.seq_length for record in ccfg.census.values()}, {4096})
        self.assertEqual(ccfg.output_census.seq_length, 4096)
        self.assertIsNone(ccfg.kind_activations)
        bind_layer_stack(ccfg)
        for kind, fields in ccfg.layer_binding.items():
            self.assertIs(fields["kind_activations"], ccfg.census[kind])
        plain = _evaluator(_qwen35_text()).ccfg
        self.assertIsNone(plain.census)
        self.assertIsNone(plain.output_census)
        towers = _evaluator(_qwen35(_qwen35_text()), target=_IMAGE_TEXT, census=True).ccfg.mm_ccfgs
        self.assertEqual(towers["text"].census, ccfg.census)
        self.assertEqual(towers["text"].output_census, ccfg.output_census)
        for name in ("census", "kind_activations", "output_census"):
            self.assertIsNone(getattr(towers["vision"], name))

    def test_a_layer_is_priced_with_its_kinds_census(self):
        """
        Feature: the census on the memory path.
        Description: The hybrid model without recompute on one stage, and
            fully recomputed on two, each with a census.
        Expectation: Each layer keeps its kind's census bytes for the
            micro-batch's 4096 tokens, and the output layer its own; the
            output layer's backward holds what its census states beyond
            that; the first stage's last layer, which recomputes, holds its
            kind's working set in its backward.
        """
        evaluator = _evaluator(_qwen35_text(), census=True)
        census, output = evaluator.ccfg.census, evaluator.ccfg.output_census
        log = evaluator.estimate_peak_insight()[0]["Node Log"]
        for index, kind in enumerate(("linear_attention", "full_attention")):
            kept = 4096 * (census[kind].saved + census[kind].saved_tp) / 2 ** 20
            self.assertAlmostEqual(log[(0, 0, index, "N")]["_activ"], kept, delta=1)
        kept = output.saved + output.saved_tp
        self.assertAlmostEqual(log[(0, 0, "", "O")]["_activ"], 4096 * kept / 2 ** 20, delta=1)
        held = output.working + output.working_tp - kept
        self.assertAlmostEqual(log[(0, 0, "G_", "O")]["_activ"], 4096 * held / 2 ** 20, delta=1)
        log = _evaluator(_qwen35_text(), mode="full", pp=2, census=True).estimate_peak_insight()[0]["Node Log"]
        working = [value["_activ"] for key, value in log.items() if str(key[2]).startswith("rec_")]
        linear = census["linear_attention"]
        self.assertAlmostEqual(working[0], 4096 * (linear.working + linear.working_tp) / 2 ** 20, delta=1)

    def test_a_selective_layer_is_priced_with_its_kinds_census(self):
        """
        Feature: the census on the memory path, under HyperParallel's
            selective activation checkpointing.
        Description: The hybrid model checkpointed selectively on one stage,
            with a census.
        Expectation: Each layer keeps what its kind keeps under the
            trainer's selective policy for the micro-batch's 4096 tokens.
        """
        evaluator = _evaluator(_qwen35_text(), mode="selective", census=True)
        census = evaluator.ccfg.census
        log = evaluator.estimate_peak_insight()[0]["Node Log"]
        for index, kind in enumerate(("linear_attention", "full_attention")):
            kept = 4096 * (census[kind].selective + census[kind].selective_tp) / 2 ** 20
            self.assertAlmostEqual(log[(0, 0, index, "S")]["_activ"], kept, delta=1)


def _deepseek_v4():
    """A four-layer DeepSeek-V4 config of width 256, the family the V4.1 recipe's factory builds."""
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import (  # pylint: disable=C0415
        DeepseekV4Config,
    )
    return DeepseekV4Config(vocab_size=1024, hidden_size=256, num_hidden_layers=4,
                            num_attention_heads=4, num_key_value_heads=1, head_dim=64,
                            intermediate_size=128, moe_intermediate_size=128, n_routed_experts=4,
                            num_experts_per_tok=2, q_lora_rank=32)


def _saved_bytes(run) -> int:
    """The bytes *run*'s forward saves for its backward, on fake tensors."""
    seen, total = set(), 0

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        """Count *tensor*'s storage once, the first time the forward saves it."""
        nonlocal total
        address = tensor.untyped_storage()._cdata  # pylint: disable=protected-access
        if address not in seen:
            seen.add(address)
            total += tensor.untyped_storage().nbytes()
        return tensor

    with FakeTensorMode(allow_non_fake_inputs=True), \
            torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        run()
    return total


class TestV41SparseAttentionContract(unittest.TestCase):
    """The contract of DeepSeek-V4.1's sparse attention keeps what its kernel's caller keeps."""

    @staticmethod
    def _call(seq: int, heads: int = 8, head_dim: int = 64, selected: int = 32, rope: int = 8):
        """Run the contract once at *seq* tokens, with the caller's layouts."""
        from hyper_parallel.auto_parallel._npu_contracts import (  # pylint: disable=C0415
            npu_sparse_attention_with_scalar_sink,
        )
        query = torch.randn(1, heads, seq, head_dim, dtype=torch.bfloat16, requires_grad=True)
        key_value = torch.randn(1, 1, seq, head_dim, dtype=torch.bfloat16, requires_grad=True)
        indices = torch.zeros(1, seq, selected, dtype=torch.int32)
        sinks = torch.randn(heads, dtype=torch.bfloat16, requires_grad=True)
        out = npu_sparse_attention_with_scalar_sink(query, key_value, indices, sinks, rope, 0.125)
        return out

    def test_the_output_has_the_shape_its_caller_reshapes_to(self):
        """
        Feature: the contract's output.
        Expectation: A row per token and head, (batch, tokens, heads, head
            dim), which is the shape the kernel's caller returns after it
            rescales the sink's share out of the softmax sum.
        """
        with FakeTensorMode(allow_non_fake_inputs=True):
            out = self._call(seq=128)
        self.assertEqual(tuple(out.shape), (1, 128, 8, 64))

    def test_it_keeps_the_nine_tensors_its_caller_keeps(self):
        """
        Feature: what the contract saves for its backward.
        Expectation: Exactly the nine tensors
            _NpuSparseAttentionWithScalarSink hands save_for_backward, in
            its shapes: the queries and padded keys token-major, the two
            auxiliary rotary coordinates, the selected indices with the
            sink's column, two fp32 statistics of one scalar a token and
            head, the output, and the sink's share of each row.
        """
        seq, heads, head_dim, selected, rope = 128, 8, 64, 32, 8
        padded = max(seq + 1, selected + 2)
        want = (
            seq * heads * head_dim * 2          # queries, token-major
            + padded * head_dim * 2             # padded keys
            + seq * heads * rope * 2            # query rope coordinates
            + padded * rope * 2                 # key rope coordinates
            + seq * (selected + 1) * 4          # selected indices plus the sink's column
            + seq * heads * 4 * 2               # the two fp32 statistics
            + seq * heads * head_dim * 2        # the output
            + seq * heads * 4                   # the sink's share of each row
        )
        got = _saved_bytes(lambda: self._call(seq, heads, head_dim, selected, rope))
        self.assertEqual(got, want)

    def test_its_backward_hands_each_input_a_gradient_of_its_own_shape(self):
        """
        Feature: a backward through the contract.
        Expectation: The queries, the keys and the sinks each get a
            gradient shaped as they were passed, which is not the
            token-major layout the kernel is handed: the caller gives
            (batch, heads, tokens, head dim) and the kernel takes
            (tokens, heads, head dim). Returning the saved shapes instead
            makes autograd refuse the whole backward.
        """
        with FakeTensorMode(allow_non_fake_inputs=True):
            out = self._call(seq=64)
            out.sum().backward()
        # The backward ran, which is the assertion: autograd checks every
        # gradient against its input's shape and raises otherwise.
        self.assertEqual(tuple(out.shape), (1, 64, 8, 64))

    def test_the_installer_binds_both_paths_and_restores_them(self):
        """
        Feature: v41_attention_contract, which the census installs around a
            V4.1 layer.
        Expectation: The legacy Omni and reference bindings are replaced
            and restored together. An adapter with explicit fused/eager
            entry points is left unchanged: its eager path must not be
            priced as Omni's kernel.
        """
        from hyper_parallel.auto_parallel._npu_contracts import (  # pylint: disable=C0415
            npu_sparse_attention_with_scalar_sink,
            v41_attention_contract,
        )
        try:
            module = importlib.import_module(
                "hyper_parallel.components.modules.shared_compressed_dsa_attention")
        except ImportError:
            module = None
        names = ("npu_sparse_attention_with_scalar_sink", "_reference_sparse_attention")
        has_omni = module is not None and all(hasattr(module, name) for name in names)
        before = dict(vars(module)) if module is not None else {}
        with v41_attention_contract(64) as patched:
            self.assertEqual(patched, has_omni)
            if has_omni:
                self.assertIs(module.npu_sparse_attention_with_scalar_sink,
                              npu_sparse_attention_with_scalar_sink)
                self.assertIsNot(module._reference_sparse_attention, None)  # pylint: disable=W0212
            elif module is not None:
                self.assertEqual(vars(module).keys(), before.keys())
                for name, value in before.items():
                    self.assertIs(getattr(module, name), value)
        if module is not None:
            self.assertEqual(vars(module).keys(), before.keys())
            for name, value in before.items():
                self.assertIs(getattr(module, name), value)

    def test_the_indexer_contract_wraps_every_selection_and_restores_them(self):
        """
        Feature: v41_indexer_contract, which states that the fused Lightning
            Indexer keeps nothing for a backward.
        Expectation: Where the V4.1 attention is in the tree, all four
            selection functions are bound to a wrapper inside the context
            and restored on the way out. The wrapper computes the same
            indices, since no_grad changes whether a graph is recorded and
            never what is computed, so the contract needs no shape declared.
            Where the module is absent it is a no-op that says so.
        """
        from hyper_parallel.auto_parallel._npu_contracts import (  # pylint: disable=C0415
            _V41_SELECTION,
            v41_indexer_contract,
        )
        try:
            module = importlib.import_module(
                "hyper_parallel.components.modules.shared_compressed_dsa_attention")
        except ImportError:
            module = None
        before = ({name: getattr(module, name) for name in _V41_SELECTION}
                  if module is not None else {})
        with v41_indexer_contract() as wrapped:
            self.assertEqual(wrapped, module is not None)
            for name, original in before.items():
                self.assertIsNot(getattr(module, name), original)
        for name, original in before.items():
            self.assertIs(getattr(module, name), original)

    def test_what_it_keeps_follows_the_sequence_and_not_its_square(self):
        """
        Feature: the contract against the reference path beside it.
        Expectation: Doubling the tokens roughly doubles what is kept. This
            is the whole reason the contract is needed: the reference keeps
            an fp32 score of one sequence by another, which quadruples, and
            on the V4.1 crop that read 77.9 GiB a rank at 4096 tokens
            against a 38.5 GiB measured peak.
        """
        small = _saved_bytes(lambda: self._call(seq=256))
        large = _saved_bytes(lambda: self._call(seq=512))
        self.assertAlmostEqual(large / small, 2.0, delta=0.05)


class TestRecipeConfigFactory(unittest.TestCase):
    """A recipe that builds its model through a factory is asked, not its checkpoint path."""

    def test_a_section_naming_no_factory_is_left_alone(self):
        """
        Feature: _factory_config on a model section with no _target_, or one
            naming a factory it does not know.
        Expectation: None, so every model that resolves through AutoConfig
            keeps resolving through it and this path costs them nothing.
        """
        from hyper_parallel.auto_parallel._hf_model_spec import (  # pylint: disable=C0415
            _factory_config,
        )
        self.assertIsNone(_factory_config({}))
        self.assertIsNone(_factory_config({"_target_": "some.other.builder"}))
        self.assertIsNone(_factory_config({"pretrained_model_name_or_path": "Qwen/Qwen3-8B"}))

    def test_a_known_factory_that_cannot_build_is_refused_not_ignored(self):
        """
        Feature: _factory_config where _target_ names a factory it knows and
            that factory cannot build its config, as on a tree carrying no
            V4.1 model.
        Expectation: ValueError naming the factory. Falling back quietly to
            the formulas would price a different model from the one the
            recipe builds, which is how a recipe stating an eightfold crop
            came to be priced from the released config.
        """
        from hyper_parallel.auto_parallel._hf_model_spec import (  # pylint: disable=C0415
            _CONFIG_FACTORIES,
            _factory_config,
        )
        target = next(iter(_CONFIG_FACTORIES))
        with self.assertRaises(ValueError) as refusal:
            _factory_config({"_target_": target, "config_path": "no/such/dir",
                             "engram_assets_path": "no/such/file"})
        self.assertIn(target, str(refusal.exception))

    def test_every_factory_pair_names_the_same_module(self):
        """
        Feature: the _CONFIG_FACTORIES table.
        Expectation: Each model factory and its config factory come from one
            module, which is what makes them a declared pair rather than two
            names that happen to look alike.
        """
        from hyper_parallel.auto_parallel._hf_model_spec import (  # pylint: disable=C0415
            _CONFIG_FACTORIES,
        )
        self.assertTrue(_CONFIG_FACTORIES)
        for model_factory, config_factory in _CONFIG_FACTORIES.items():
            self.assertEqual(model_factory.rpartition(".")[0], config_factory.rpartition(".")[0])


class TestTensorParallelConfig(unittest.TestCase):
    """A rank's share of a config divides each width once, whatever it is called."""

    def test_one_width_under_two_names_is_divided_once(self):
        """
        Feature: tp_config on a DeepSeek-V4 config, whose attribute_map maps
            intermediate_size onto moe_intermediate_size, so a model whose
            feed-forward is all experts states one width under two names.
        Expectation: A rank of 2 holds half that width. Dividing each name in
            turn gave a quarter, so the layer the census built as a rank's
            share was half the width it should be, the split derived from the
            pair came out too large, and ND over-rewarded tensor parallelism.
        """
        config = _deepseek_v4()
        self.assertEqual(config.intermediate_size, config.moe_intermediate_size)
        whole = config.moe_intermediate_size
        half = tp_config(config, 2)
        self.assertEqual(half.moe_intermediate_size, whole // 2)
        self.assertEqual(half.intermediate_size, whole // 2)
        self.assertEqual(half.num_attention_heads, config.num_attention_heads // 2)

    def test_two_widths_under_two_names_are_each_divided(self):
        """
        Feature: tp_config on a Qwen3-MoE config, which keeps the dense and
            the expert width apart.
        Expectation: Both are halved, so de-duplicating the aliased pair does
            not stop a model that really has two widths from splitting both.
        """
        config = _qwen3_moe()
        self.assertNotEqual(config.intermediate_size, config.moe_intermediate_size)
        half = tp_config(config, 2)
        self.assertEqual(half.intermediate_size, config.intermediate_size // 2)
        self.assertEqual(half.moe_intermediate_size, config.moe_intermediate_size // 2)


class TestKernelContracts(unittest.TestCase):
    """A census of a model whose kernels no contract states is refused, not answered."""

    def test_a_model_whose_kernels_no_contract_states_is_refused(self):
        """
        Feature: census_layer on a DeepSeek-V4 config, the family the
            DeepSeek-V4.1 recipe's factory builds.
        Expectation: The census of one layer refuses, and for the reason
            that is true of it: the family's layers cannot be priced one at
            a time, because most of them consume what a source layer wrote
            earlier in the same forward, and the whole stack's census is
            the route.  Before this it raised AttributeError out of the
            rotary embedding, which no caller catches, so the whole cost
            model stopped instead of falling back to the formulas.
        """
        with self.assertRaises(CensusUnavailable) as refusal:
            census_layer(_deepseek_v4(), 0, 128)
        said = str(refusal.exception)
        self.assertIn("deepseek_v4", said)
        self.assertIn("one at a time", said)
        self.assertIn("whole stack", said)
        self.assertIn("census_stack", said)

    def test_the_whole_stack_is_measured_as_one(self):
        """
        Feature: census_activations on the same config, the entry the model
            spec calls (D18).
        Expectation: It runs the model's own forward over the whole stack
            and states the kind's record, where it refused: what a layer
            keeps and holds, a share tensor parallelism splits, and no
            selective part.  With one record for the four layers, the record
            gives back the stack's total at TP 1 to the byte.
        """
        config = _deepseek_v4()
        record = census_activations(config, [{"kind": "decoder", "count": 4}], seq_length=128)["decoder"]
        self.assertGreater(record.saved, 0)
        self.assertGreater(record.saved_tp, 0)
        self.assertGreater(record.working, record.saved)
        self.assertIsNone(record.selective)
        stack = census_stack(config, 128)
        self.assertEqual(len(stack), 4)
        self.assertEqual((record.saved + record.saved_tp) * 128 * 4, sum(kept for kept, _ in stack))

    def test_the_parameters_of_such_a_layer_are_still_counted(self):
        """
        Feature: census_parameters on the same config.
        Expectation: It answers, because which kernel runs a layer does not
            change how many parameters it has, and verify mode sets these
            beside what ND prices.  Only what a layer keeps for its
            backward depends on the kernel.
        """
        parts = census_parameters(_deepseek_v4(), 0)
        self.assertIn("routed", parts)
        self.assertGreater(parts["routed"], 0)
        self.assertGreater(parts["attention"], 0)

    def test_a_run_told_it_has_no_census_is_told_its_memory_is_not_a_bound(self):
        """
        Feature: resolve_hf_model_spec asked for a census with the model
            stated by hand, as a model Transformers cannot build must be.
        Expectation: The warning says the memory is an estimate and not a
            bound, because the formulas alone have measured 37 to 55% below
            a real peak and a search filters feasibility on that number.
        """
        model = {"config_overrides": {"hidden_size": 64, "num_hidden_layers": 2,
                                      "num_attention_heads": 4, "vocab_size": 128}}
        with self.assertLogs("hyper_parallel.auto_parallel._hf_model_spec", "WARNING") as logs:
            resolve_hf_model_spec(model, census_seq_len=128)
        said = "\n".join(logs.output)
        self.assertIn("NOT a bound", said)
        self.assertIn("may still run out of memory", said)

    def test_stated_records_are_taken_without_that_warning(self):
        """
        Feature: the same spec, with the census records its reader measured
            stated in the overrides.
        Expectation: The records reach the spec and nothing warns that the
            memory is not a bound, since it then rests on a measurement;
            this is the route a model Transformers cannot build has to its
            records.
        """
        record = {"saved": 1024.0, "saved_tp": 0.0, "working": 2048.0,
                  "working_tp": 0.0, "seq_length": 128}
        model = {"config_overrides": {"hidden_size": 64, "num_hidden_layers": 2,
                                      "num_attention_heads": 4, "vocab_size": 128,
                                      "activations": {"decoder": record}}}
        spec = resolve_hf_model_spec(model, census_seq_len=128)
        self.assertEqual(spec["activations"]["decoder"], record)


def _qwen3():
    """A two-layer Qwen3 config of width 64, whose layers run alone."""
    from transformers import Qwen3Config  # pylint: disable=C0415
    return Qwen3Config(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                       head_dim=16, intermediate_size=128, vocab_size=128)


class TestStackCensus(unittest.TestCase):
    """A census of the whole stack, for a family whose layers cannot run alone (D18)."""

    def test_a_layer_of_the_stack_keeps_what_the_layer_census_measures(self):
        """
        Feature: census_stack on a model whose layers do run alone.
        Description: A two-layer Qwen3 config of width 64 at 64 tokens, over
            the whole stack and one layer at a time.
        Expectation: The first layer keeps what the layer census measures of
            it, to the byte.  The second keeps that less the positions'
            embeddings, which the stack computes once for every layer and
            counts toward the first that saves them, where the layer census
            builds them for each layer it runs.
        """
        config = _qwen3()
        (first, _), (second, _) = census_stack(config, 64)
        alone = [census_layer(config, index, 64)[0] for index in (0, 1)]
        self.assertEqual(first, alone[0])
        with FakeTensorMode(allow_non_fake_inputs=True):
            _, rotary = _classes(_modeling(config))
            hidden = torch.empty(1, 64, config.hidden_size, dtype=torch.bfloat16)
            embedded = _positions(rotary(config=config), hidden, 64)[1]
            positions = sum(tensor.untyped_storage().nbytes() for tensor in embedded)
        self.assertEqual(alone[1] - second, positions)

    @unittest.skipIf(importlib.util.find_spec("hyper_parallel.models.deepseek_v41") is not None,
                     "this tree carries DeepSeek-V4.1")
    def test_a_model_this_tree_does_not_carry_is_refused(self):
        """
        Feature: census_activations on a DeepSeek-V4 config naming
            DeepSeek-V4.1's architecture, on a tree without that model.
        Expectation: CensusUnavailable naming the architecture, which the
            spec catches, so the run is priced by the formulas and told so.
            Building Transformers' DeepSeek-V4 in its place would price a
            different model.
        """
        config = _deepseek_v4()
        config.architectures = ["DeepseekV41ForCausalLM"]
        with self.assertRaises(CensusUnavailable) as refusal:
            census_activations(config, [{"kind": "decoder", "count": 4}], seq_length=128)
        self.assertIn("DeepseekV41ForCausalLM", str(refusal.exception))


if __name__ == "__main__":
    unittest.main()
