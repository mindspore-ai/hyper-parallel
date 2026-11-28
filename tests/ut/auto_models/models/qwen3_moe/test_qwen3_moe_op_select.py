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
"""Offline Qwen3-MoE and Transformers Trainer regression for basic/expert OpSelect."""

from copy import deepcopy
from tempfile import TemporaryDirectory
import unittest

import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM, Trainer, TrainingArguments
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts, Qwen3MoeRMSNorm

from hyper_parallel.components.modules import GroupedExperts, RMSNorm
from hyper_parallel.components.ops import ops
from tests.common.op_precision import assert_precision


class TestQwen3MoeOpSelect(unittest.TestCase):
    """Replace real HF norms/experts while retaining source Parameters and checkpoint keys."""

    def setUp(self) -> None:
        """Build fixed weights and inputs without downloading assets or initializing collectives."""
        self.previous = ops.get_backends()
        self.addCleanup(ops.set_backends, self.previous)
        ops.set_backends({name: "native" for name in (
            "rms_norm", "swiglu", "grouped_matmul", "moe_token_permute", "moe_token_unpermute",
        )})
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7213)
            config = Qwen3MoeConfig.from_dict({
                "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2,
                "num_key_value_heads": 1, "head_dim": 8, "vocab_size": 32, "moe_intermediate_size": 16,
                "num_experts": 4, "num_experts_per_tok": 2, "attention_dropout": 0.0, "use_cache": False,
            })
            config._attn_implementation = "eager"  # pylint: disable=protected-access
            self.reference = Qwen3MoeForCausalLM(config)
        self.candidate = deepcopy(self.reference)
        for name, module in list(self.candidate.named_modules()):
            if isinstance(module, Qwen3MoeRMSNorm):
                replacement = RMSNorm(module=module)
                self.assertIs(replacement.weight, module.weight)
                self.candidate.set_submodule(name, replacement)
            elif isinstance(module, Qwen3MoeExperts):
                replacement = GroupedExperts(module=module, context={"num_layers": 1})
                self.assertIs(replacement.gate_up_proj, module.gate_up_proj)
                self.assertIs(replacement.down_proj, module.down_proj)
                self.candidate.set_submodule(name, replacement)
        self.inputs = torch.tensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]])
        self.labels = self.inputs.roll(-1, dims=1)
        self.assertEqual(tuple(self.candidate.state_dict()), tuple(self.reference.state_dict()))

    def _assert_used(self) -> None:
        expected = {("rms_norm", "standard", "native"), ("swiglu", "standard", "native"),
                    ("grouped_matmul", "standard", "native"), ("moe_token_permute", "standard", "native"),
                    ("moe_token_unpermute", "standard", "native")}
        self.assertTrue(expected.issubset(set(ops.used_backends())),
                        f"actual={ops.used_backends()}, expected={expected}")

    def test_loss_gradients_and_update(self) -> None:
        """A standalone PyTorch step agrees with the original HF network for every parameter."""
        for model in (self.reference, self.candidate):
            model.train()
        expected = self.reference(self.inputs, labels=self.labels)
        actual = self.candidate(self.inputs, labels=self.labels)
        assert_precision(actual.logits, expected.logits, dtype=torch.float32, label="qwen3_moe/logits")
        assert_precision(actual.loss, expected.loss, dtype=torch.float32, label="qwen3_moe/loss")
        actual.loss.backward()
        expected.loss.backward()
        for name, parameter in self.candidate.named_parameters():
            reference = self.reference.get_parameter(name)
            self.assertEqual(parameter.grad is None, reference.grad is None, f"parameter={name}")
            if parameter.grad is not None:
                assert_precision(parameter.grad, reference.grad, dtype=torch.float32, backward=True, label=name)
        for model in (self.reference, self.candidate):
            torch.optim.SGD(model.parameters(), lr=1e-3).step()
        for name, parameter in self.candidate.state_dict().items():
            assert_precision(parameter, self.reference.state_dict()[name], dtype=torch.float32, label=f"update/{name}")
        self.candidate.load_state_dict(self.reference.state_dict(), strict=True)
        self._assert_used()

    def test_transformers_trainer_step(self) -> None:
        """The unwrapped HF Trainer executes migrated modules and matches a standalone SGD step."""
        self.reference.train()
        expected = self.reference(self.inputs, labels=self.labels).loss
        expected.backward()
        torch.optim.SGD(self.reference.parameters(), lr=1e-3).step()
        dataset = [{"input_ids": ids, "labels": labels} for ids, labels in zip(self.inputs, self.labels)]
        with TemporaryDirectory() as destination:
            args = TrainingArguments(
                output_dir=destination, use_cpu=True, report_to="none", max_steps=1,
                per_device_train_batch_size=2, learning_rate=1e-3, optim="sgd", max_grad_norm=0.0,
                lr_scheduler_type="constant", save_strategy="no", logging_strategy="no", disable_tqdm=True,
            )
            trainer = Trainer(model=self.candidate, args=args, train_dataset=dataset)
            result = trainer.train()
        self.assertEqual(result.global_step, 1)
        assert_precision(torch.tensor(result.training_loss), expected.detach(),
                         dtype=torch.float32, label="trainer/loss")
        for name, parameter in self.candidate.state_dict().items():
            assert_precision(parameter, self.reference.state_dict()[name], dtype=torch.float32, label=f"trainer/{name}")
        self._assert_used()
