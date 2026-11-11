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
"""CPU contracts and independent arithmetic checks for PIK."""

import json
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch

import torch
import yaml
from torch._subclasses.fake_tensor import FakeTensorMode

import rl
from rl.config.resolve import build_runtime_config
from rl.unified_execution.components.row_parallel import PikLinear, pik_row_compute_fn, replace_pik_linear
from rl.unified_execution.contract import build_unified_payload, parse_unified_payload
from rl.unified_execution.execution_plan import PIK_MODULE_TARGET, pik_row_matches, pik_row_options
from rl.unified_execution.ops.pik import PikContext, PikPlan, linear
from rl.unified_execution.ops.pik.gemm import batched_leaves
from rl.unified_execution.ops.pik.reduction import fixed_tree
from rl.workers.actor import Actor


class TestPik(unittest.TestCase):
    """Validate contracts without initializing an NPU or a process group."""

    def test_plan_and_rank_alignment(self):
        """Each supported TP owns a power-of-two complete subtree."""
        plan = PikPlan(8)
        for world in (1, 2, 4, 8):
            plan.validate(4096, world)
        for leaves in (0, -1, 3, True):
            with self.assertRaises(ValueError):
                PikPlan(leaves)
        for k, world in ((4097, 2), (4096, 3), (4096, 16)):
            with self.assertRaises(ValueError):
                plan.validate(k, world)
        with self.assertRaises(ValueError):
            PikPlan(8, "auto")
        self.assertEqual(plan.contract()["tree_dtype"], "float32")

    def test_leaf_layouts_preserve_logical_k_intervals(self):
        """Strided and packed BMM inputs describe exactly the same consecutive leaves."""
        x = torch.arange(3 * 32, dtype=torch.float32).reshape(3, 32)[:, ::2]
        weight = torch.arange(5 * 32, dtype=torch.float32).reshape(5, 32)[:, ::2]
        expected = torch.stack([x[:, i * 4:(i + 1) * 4] @ weight[:, i * 4:(i + 1) * 4].T
                                for i in range(4)])
        for layout in ("strided", "packed"):
            result = batched_leaves(x, weight, plan=PikPlan(8, layout), local_leaves=4)
            torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_tree_fake_metadata_and_single_leaf(self):
        """The compiler sees FP32 internal nodes without loading Triton."""
        with FakeTensorMode():
            output = fixed_tree(torch.empty(8, 7, 13, dtype=torch.bfloat16))
            self.assertEqual(output.shape, (7, 13))
            self.assertEqual(output.dtype, torch.float32)
        value = torch.randn(1, 3, 5).bfloat16()
        torch.testing.assert_close(fixed_tree(value), value[0].float(), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            fixed_tree(torch.empty(3, 2, 4))

    def test_input_contract_rejects_shape_and_axis_errors(self):
        """Invalid local geometry fails before any data collective."""
        device = SimpleNamespace(type="npu", index=0)
        context = PikContext(PikPlan(), 64, 16, None, 2, 0, 0, True, 1, "all_to_all")
        value = SimpleNamespace(shape=(2, 6, 32), ndim=3, dtype=torch.bfloat16, device=device)
        weight = SimpleNamespace(shape=(16, 32), ndim=2, dtype=torch.bfloat16, device=device)
        self.assertEqual(linear._validate(value, weight, None, context), 1)
        value.shape = (2, 5, 32)
        with self.assertRaisesRegex(ValueError, "SP sequence"):
            linear._validate(value, weight, None, context)
        value.shape = (2, 6, 31)
        with self.assertRaisesRegex(ValueError, "local shapes"):
            linear._validate(value, weight, None, context)

    def test_forward_root_cast_precedes_bias(self):
        """Rank roots stay FP32 until the global root, then BF16 rounding precedes bias."""
        context = PikContext(PikPlan(), 8, 2, None, 1, 0, 0)
        value, weight = torch.ones(3, 8), torch.ones(2, 8)
        root = torch.tensor([[1.003, -1.003]]).expand(3, 2)
        bias = torch.tensor([0.001, -0.001])
        with patch.object(linear, "local_gemm", return_value=root):
            actual = linear._forward(value, weight, bias, context, 0)
        expected = root.bfloat16() + bias.bfloat16()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(expected, (root + bias).bfloat16()))


class TestPikModel(unittest.TestCase):
    """Test contract failure paths without initializing NPU or process groups."""

    def setUp(self) -> None:
        """Load the explicit shared recipe independently for each case."""
        recipe = Path(__file__).parent / "fixtures/qwen3_dense_pik.yaml"
        self.config = yaml.safe_load(recipe.read_text(encoding="utf-8"))

    def test_cross_topology_contract_roundtrip(self) -> None:
        """Preserve arithmetic while transporting distinct training and rollout topologies."""
        self.config["train"]["accelerator"].update(tp=4, sequence_parallel=True)
        self.config["rollout"]["vllm"].update(tensor_parallel_size=2, sequence_parallel=False)
        with TemporaryDirectory() as checkpoint:
            (Path(checkpoint) / "config.json").write_text(json.dumps({
                "model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
            }), encoding="utf-8")
            self.config["model"].update(weights_path=checkpoint, tokenizer_path=checkpoint)
            runtime = build_runtime_config(self.config)
        self.assertEqual(runtime.accelerator.tp_size, 4)
        payload = build_unified_payload(self.config)
        self.assertEqual(parse_unified_payload(payload), payload)
        self.assertEqual(pik_row_options(payload["plan_overrides"])["num_leaves"], 8)

    def test_reject_incomplete_or_different_arithmetic(self) -> None:
        """Reject partial, conditional or inconsistent PIK replacement plans."""
        plan = self.config["train"]["plan_overrides"]
        rows = [entry for entry in plan if entry.get("replace", {}).get("implementation", {}).get(
            "_target_") == PIK_MODULE_TARGET]
        for change in ("missing", "different", "conditional"):
            modified = deepcopy(plan)
            row = next(entry for entry in modified if entry["match"] == rows[0]["match"])
            if change == "missing":
                modified.remove(row)
            elif change == "different":
                row["replace"]["implementation"]["num_leaves"] = 3
            else:
                row["when"] = "sequence_parallel"
            with self.subTest(change=change), self.assertRaises(ValueError):
                pik_row_options(modified)

    def test_reject_double_reduction_and_policy_mismatch(self) -> None:
        self.config["train"]["plan_overrides"][-1]["out_src"]["output"]["tp"] = "partial"
        with self.assertRaisesRegex(ValueError, "parents"):
            build_unified_payload(self.config)
        self.setUp()
        self.config["consistency"]["batch_invariant"]["fill_uninitialized_memory"] = True
        with self.assertRaisesRegex(ValueError, "same explicit Level 3"):
            build_unified_payload(self.config)

    def test_factory_reads_current_parameters_and_keeps_gradients(self) -> None:
        """TP1 replacement preserves identity and reads later replacement weights."""
        original = torch.nn.Linear(8, 4)
        parameter = original.weight
        context = SimpleNamespace(world_size=1, sequence_parallel=False, group=None)
        with patch("rl.unified_execution.components.row_parallel.prepare_pik", return_value=context) as prepare:
            module = replace_pik_linear(module=original, module_fqn="mlp.down_proj", context={})
        self.assertIs(prepare.call_args.kwargs["group"], None)
        self.assertEqual(prepare.call_args.kwargs["policy"], {"fill_uninitialized_memory": False})
        x = torch.randn(2, 3, 8, requires_grad=True)
        with patch("rl.unified_execution.components.row_parallel.pik_linear",
                   side_effect=lambda x, w, b, **_: x @ w.T + b):
            module(x).sum().backward()
            self.assertIs(module.weight, parameter)
            torch.testing.assert_close(parameter.grad, x.detach().flatten(0, 1).sum(0).expand(4, 8))
            module.weight = torch.nn.Parameter(torch.zeros_like(parameter))
            torch.testing.assert_close(module(x), module.bias.expand(2, 3, 4))

    def test_parallel_requires_explicit_binding(self) -> None:
        """Never fall back to TP1 when HP failed to apply the parallel plan."""
        module = PikLinear(torch.nn.Linear(8, 4), PikPlan(), parallel=True, fill_uninitialized_memory=True)
        with self.assertRaisesRegex(RuntimeError, "bind"):
            module(torch.randn(1, 2, 8))
        group = object()
        mesh = SimpleNamespace(size=lambda: 2, get_group=lambda: group)
        with patch("rl.unified_execution.components.row_parallel.prepare_pik",
                   return_value=SimpleNamespace()) as prepare:
            pik_row_compute_fn(module=module, mesh=None, tp_mesh=mesh, cp_mesh=None, ep_mesh=None,
                               sequence_parallel=True)
        self.assertIs(prepare.call_args.kwargs["group"], group)
        self.assertEqual(prepare.call_args.kwargs["sequence_dim"], 1)
        self.assertTrue(prepare.call_args.kwargs["sequence_parallel"])
        with self.assertRaisesRegex(ValueError, "already bound"):
            module.bind_pik(group, sequence_parallel=True)

    def test_moe_installs_only_ordinary_attention_rows(self) -> None:
        """MoE shares the row adapter without accepting expert-internal PIK."""
        directory = Path(__file__).parent / "fixtures"
        recipes = (Path(rl.__file__).resolve().parents[1] / "examples/consistency/configs/qwen3_moe_4l_rl.yaml",
                   directory / "qwen3_moe_pik_replicated_experts.yaml")
        for recipe in recipes:
            with self.subTest(recipe=recipe.name):
                config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
                payload = build_unified_payload(config)
                self.assertEqual(parse_unified_payload(payload), payload)
                self.assertEqual(pik_row_matches("qwen3_moe"), ("*.self_attn.o_proj",))
                self.assertEqual(pik_row_options(payload["plan_overrides"])["num_leaves"], 8)
                added = deepcopy(next(entry for entry in self.config["train"]["plan_overrides"]
                                      if entry.get("match") == "*.mlp.down_proj"))
                config["train"]["plan_overrides"].append(added)
                with self.assertRaisesRegex(ValueError, "expert PIK"):
                    build_unified_payload(config)

    def test_moe_rejects_dense_attention_source_and_allows_ep_context(self) -> None:
        """Family checks reject a mismatched factory contract; ordinary rows ignore EP ownership."""
        recipe = Path(rl.__file__).resolve().parents[1] / "examples/consistency/configs/qwen3_moe_4l_rl.yaml"
        config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
        attention = next(entry for entry in config["train"]["plan_overrides"]
                         if entry.get("replace", {}).get("source_type", "").endswith("Qwen3MoeAttention"))
        attention["replace"]["source_type"] = "transformers.models.qwen3.modeling_qwen3.Qwen3Attention"
        with self.assertRaisesRegex(ValueError, "unconditional"):
            build_unified_payload(config)
        original = torch.nn.Linear(8, 4)
        with patch("rl.unified_execution.components.row_parallel.prepare_pik", return_value=SimpleNamespace()):
            row = replace_pik_linear(module=original, module_fqn="self_attn.o_proj", context={"ep": True})
        self.assertIs(row.weight, original.weight)


class _TokenModel(torch.nn.Module):
    """Expose input padding while retaining a real differentiable token projection."""

    def __init__(self) -> None:
        """Create an ordinary trainable projection and declare the SP input multiple."""
        super().__init__()
        self.projection = torch.nn.Embedding(11, 11)
        self.unified_sequence_multiple = 4
        self.observed = None

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, use_cache: bool = False,
                position_ids: Optional[torch.Tensor] = None) -> dict:
        """Record prepared model inputs without mutating caller token storage."""
        del use_cache
        if position_ids is None:
            # Model the HF default after SP embeddings, before attention gathers tokens.
            local_length = input_ids.shape[1] // self.unified_sequence_multiple
            position_ids = torch.arange(local_length, device=input_ids.device).unsqueeze(0)
        self.observed = (input_ids.detach().clone(), attention_mask.detach().clone(), position_ids.detach().clone())
        positional_bias = position_ids.unsqueeze(-1) * torch.arange(11, device=input_ids.device) / 13
        return {"logits": self.projection(input_ids) + positional_bias}


class TestPikActorPadding(unittest.TestCase):
    """Verify SP padding is invisible to original probabilities and gradients."""

    def test_odd_and_aligned_tokens(self) -> None:
        """Right padding must neither add loss positions nor change valid gradients."""
        for length in (7, 8, 9):
            with self.subTest(length=length):
                model = _TokenModel()
                actor = Actor(model, algorithm=SimpleNamespace(), micro_batch_size=1,
                              optimizer=torch.optim.SGD(model.parameters(), lr=0.1), device=torch.device("cpu"))
                tokens = torch.arange(length).unsqueeze(0) % 11
                mask = torch.ones_like(tokens)
                positions = torch.arange(length).view(1, length, 1)
                logits = model.projection(tokens) + positions * torch.arange(11) / 13
                expected = logits[:, :-1].log_softmax(-1).gather(
                    -1, tokens[:, 1:, None]).squeeze(-1)
                expected.sum().backward()
                gradient = model.projection.weight.grad.clone()
                model.zero_grad(set_to_none=True)
                actual = actor.sequence_log_probs(tokens, mask)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                actual.sum().backward()
                torch.testing.assert_close(model.projection.weight.grad, gradient, rtol=0, atol=0)
                seen_tokens, seen_mask, seen_positions = model.observed
                self.assertEqual(seen_tokens.shape[1] % 4, 0)
                self.assertEqual(seen_mask[:, length:].count_nonzero().item(), 0)
                torch.testing.assert_close(seen_positions, torch.arange(seen_tokens.shape[1]).unsqueeze(0))
                torch.testing.assert_close(seen_tokens[:, :length], tokens)
                self.assertEqual(mask.sum().item(), length)
