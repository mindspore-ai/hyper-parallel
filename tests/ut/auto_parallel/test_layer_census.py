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
import os
import sys
import tempfile
import unittest
from typing import Any, Callable, Optional
from unittest.mock import patch

import torch
import yaml
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.utils.checkpoint import DefaultDeviceType

import hyper_parallel
from hyper_parallel.auto_parallel._layer_census import (  # pylint: disable=protected-access
    CensusUnavailable,
    census_activations,
    census_layer,
    census_output_activations,
    census_recomputed,
    census_stack,
    replacement_specs,
    _classes,
    _measure,
    _modeling,
    _positions,
    _ForwardContext,
    _census_policy,
    _checkpointing,
    _kernel_op,
    _ATTENTION_KERNELS,
)
from hyper_parallel.auto_parallel._model_spec import ModelSpec, ModelSpecError
from hyper_parallel.auto_parallel._npu_contracts import npu_contracts
from hyper_parallel.models.replacement import module_replacement
from hyper_parallel.core.activation_memory.policy import CheckpointPolicy
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2 import EvaluatorV2

_HF_CONFIG = "hyper_parallel.auto_parallel._hf_model_spec._get_hf_config"
_CAUSAL_LM = "hyper_parallel.models._transformers.HyperAutoModelForCausalLM.from_pretrained"


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


def _saved_ops(config, layer_index, seq_length, replacements=()):
    """What layer *layer_index* keeps for its backward for each op, per token: the records a census fills."""
    ops = {}
    census_layer(config, layer_index, seq_length, ops=ops, replacements=replacements)
    return {op: size / seq_length for op, size in ops.items()}


class TestLayerCensus(unittest.TestCase):
    """The census counts what a fake layer keeps and holds."""

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
        Feature: census_layer's records per op, which the census fills.
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
        ops = _saved_ops(config, 0, 64)
        self.assertEqual(ops, {
            "attMM": 2 * (64 + 2 * 16 + 64), "attBMM": 2 * (4 * 16 + 2 * 2 * 16 + 4 * 16), "softmax": 4 * 2 * 4 * 8,
            "normOp": 2 * (4 * 64 + 4 + 2 * 64), "ffMM": 2 * (64 + 3 * 128), "ffAct": 2 * 128})
        self.assertEqual(sum(ops.values()) * 64, census_layer(config, 0, 64)[0])

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

    def test_a_record_is_checked(self):
        """
        Feature: ModelSpec.validate of the census records a spec states.
        Description: A record of a kind no layer of the stack is.
        Expectation: It raises, naming that kind.
        """
        record = {"saved": 1.0, "saved_tp": 1.0, "working": 1.0, "working_tp": 1.0, "seq_length": 64}
        with self.assertRaisesRegex(ModelSpecError, "decoder"):
            ModelSpec.from_dict(_spec(decoder=record))


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

    def test_an_entry_is_checked(self):
        """
        Feature: replacement_specs' checks.
        Description: An entry that replaces a module without stating its type.
        Expectation: It raises, naming what is wrong.
        """
        entry = {"match": "*.self_attn", "module_type": None,
                 "replace_module": {"_target_": "hyper_parallel.models.replacement.module_replacement"}}
        with self.assertRaisesRegex(ValueError, "without module_type"):
            replacement_specs([entry])

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


class TestCensusPricing(unittest.TestCase):
    """``context.census`` has the memory model price a layer with its kind's census."""

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


class TestKernelContracts(unittest.TestCase):
    """The DeepSeek-V4 config, whose layers a census refuses one at a time, is measured over its whole stack."""

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


if __name__ == "__main__":
    unittest.main()
