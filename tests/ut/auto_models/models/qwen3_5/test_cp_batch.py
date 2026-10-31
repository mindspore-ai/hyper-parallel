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
"""Batch and causal-mask contracts for mixed GDN P2P / KV all-gather CP."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml
from torch.nn import functional as F

from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.data.parallel.dataloader_parallel import DataLoaderParallelContext
from hyper_parallel.distributed.context_parallel import wrappers
from hyper_parallel.models.qwen3_5.cp_batch import GDNCPDataCollator, GDNCPParallelBatch
from hyper_parallel.trainer.config.resolver import resolve_config


class TestGDNCPBatch(unittest.TestCase):
    """Exercise actual collation, CP slicing, labels, positions and mask semantics."""

    @staticmethod
    def sample(length):
        """Build distinct token IDs and pre-shifted targets."""
        return {"input_ids": torch.arange(length), "labels": torch.arange(1, length + 1)}

    def test_padding_and_contiguous_shards(self):
        """No real token is dropped, and every CP shard is chunk-aligned."""
        for cp_size in (1, 2, 4, 8):
            with self.subTest(cp_size=cp_size):
                collator = GDNCPDataCollator(SimpleNamespace(cp_size=cp_size, tp_size=1))
                batch = collator([self.sample(131)])
                length = batch["input_ids"].shape[1]
                self.assertEqual(length % (cp_size * 64), 0)
                self.assertEqual(batch["cu_seq_lens"].tolist(), [0, length])
                torch.testing.assert_close(batch["input_ids"][0, :131], torch.arange(131))
                self.assertTrue(torch.all(batch["labels"][0, 131:] == IGNORE_INDEX))
                shards = []
                for rank in range(cp_size):
                    context = DataLoaderParallelContext(cp_rank=rank, cp_world_size=cp_size)
                    with patch(
                        "hyper_parallel.data.batching.get_batch.create_dataloader_parallel_context",
                        return_value=context,
                    ):
                        runtime = GDNCPParallelBatch(
                            None, "cpu", None, {"labels_are_shifted": True}, False, source_type="online",
                        )
                    model, loss = runtime(iter([batch]))
                    shards.append(model["input_ids"])
                    local_length = length // cp_size
                    torch.testing.assert_close(
                        model["position_ids"], torch.arange(rank * local_length, (rank + 1) * local_length)[None],
                    )
                    self.assertEqual(model["attention_mask"], {"full_attention": None, "linear_attention": None})
                    # Trainer passes use_cache explicitly when expanding this mapping.
                    self.assertNotIn("use_cache", model)
                    torch.testing.assert_close(loss["loss_mask"], (model["labels"] >= 0).long())
                torch.testing.assert_close(torch.cat(shards, dim=1), batch["input_ids"])

    def test_reject_packing_and_tp(self):
        """Reject configurations that would silently mix independent recurrent states."""
        with self.assertRaisesRegex(ValueError, "micro_batch_size=1"):
            GDNCPDataCollator()([self.sample(64), self.sample(64)])
        with self.assertRaisesRegex(NotImplementedError, "tp_size=1"):
            GDNCPDataCollator(SimpleNamespace(tp_size=2))
        with self.assertRaisesRegex(ValueError, "nonempty"):
            GDNCPDataCollator()([self.sample(0)])

    def test_kv_allgather_causal_offsets_and_gradients(self):
        """Local Q with global KV matches dense causal SDPA, including the last rank."""
        torch.manual_seed(42)
        inputs = [torch.randn(1, 2, 32, 8, requires_grad=True) for _ in range(3)]
        expected = F.scaled_dot_product_attention(*inputs, is_causal=True)
        expected_grads = torch.autograd.grad(expected.square().sum(), inputs)
        split_inputs = [tensor.detach().clone().requires_grad_() for tensor in inputs]
        outputs = []
        for rank in range(4):
            mesh = SimpleNamespace(size=lambda: 4, get_local_rank=lambda rank=rank: rank)
            with patch.object(wrappers, "flex_cp_allgather", return_value=tuple(split_inputs[1:])):
                outputs.append(wrappers._cp_sdpa_call(
                    F.scaled_dot_product_attention, mesh,
                    *(tensor[:, :, rank * 8:(rank + 1) * 8] for tensor in split_inputs),
                    {"is_causal": True},
                ))
        actual = torch.cat(outputs, dim=2)
        torch.testing.assert_close(actual, expected)
        actual_grads = torch.autograd.grad(actual.square().sum(), split_inputs)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=2e-5)

    def test_training_yaml_selects_mixed_cp(self):
        """Parse the real training YAML so config drift cannot silently restore Ulysses."""
        root = Path(__file__).resolve().parents[5]
        config_path = root / "examples/qwen3_8/train_online_p2p_kv_allgather.yaml"
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        # Resolve the actual CP/batch entries without loading dataset C++ helpers
        # or model assets, neither of which belongs to this CPU contract test.
        config = resolve_config({
            "model": {"_target_": "torch.nn.Identity"},
            "optimizer": {"_target_": "torch.optim.AdamW", "lr": 1e-3},
            **{name: raw[name] for name in ("training", "accelerator", "dataloader", "plan_overrides")},
        })
        self.assertEqual(config.training.micro_batch_size, 1)
        self.assertEqual(config.accelerator.tp_size, 1)
        gdn, attention = config.plan_overrides[1:3]
        self.assertEqual(gdn.inner_wrapper.backend, "triton")
        self.assertEqual(gdn.inner_wrapper.chunk_size, 64)
        self.assertEqual(attention.inner_wrapper, "sdpa_hf")
        collator = config.dataloader.collate_fn.build(mesh_context=SimpleNamespace(cp_size=2, tp_size=1))
        self.assertIsInstance(collator, GDNCPDataCollator)
        runtime = config.dataloader.get_batch.build(
            mesh_context=None, device="cpu", tokenizer=None, data_config={}, pp_shared_data=False,
        )
        self.assertIsInstance(runtime, GDNCPParallelBatch)


if __name__ == "__main__":
    unittest.main()
