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
"""Collective VLM Trainer epochs, checkpoint callbacks and accumulation tails."""

import unittest
from contextlib import ExitStack
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.data.batching import build_dataloader
from hyper_parallel.data.vlm.collator import VLMCollator
from hyper_parallel.data.vlm.metadata import vlm_sample_metadata
from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.vlm_trainer import VLMTrainer
from tests.common.vlm_fixtures import build_image_corpus, vlm_loader_target


def _loader(dataset: object) -> object:
    mesh = SimpleNamespace(mesh_shape=(1,), mesh_dim_names=("dp",), rank_list=(0,))
    loaders, _ = build_dataloader(
        vlm_loader_target(), datasets=(dataset, None, None), collate_fn=VLMCollator(),
        mesh_context=SimpleNamespace(dp_rank=0, dp_size=1, device_mesh=mesh),
        training_config=SimpleNamespace(micro_batch_size=2, global_batch_size=4, seed=17),
        data_config={"load_balance": "native_batch_sampler", "distributed_dataloader": {"double_buffer": True}},
        max_seq_len=64, metadata_fn=vlm_sample_metadata,
    )
    return loaders[0]


class TestVLMTrainer(unittest.TestCase):
    """Keep the native model path while adapting collective DataLoader lifecycle."""

    def test_constructor_supplies_vlm_callback(self) -> None:
        """VLM Trainer initialization explicitly selects VLM rather than GPT metadata."""
        base = Mock()
        stages = ("_build_model_assets", "_build_data_transform", "_build_collate_fn", "_build_get_batch")
        with ExitStack() as stack:
            stack.enter_context(patch.object(BaseTrainer, "__new__", return_value=base))
            for stage in stages:
                stack.enter_context(patch.object(VLMTrainer, stage))
            VLMTrainer(SimpleNamespace())
        base._build_dataloader.assert_called_once_with(metadata_fn=vlm_sample_metadata)

    def test_resume_crosses_epochs_and_drains_prefetch(self) -> None:
        """A restored collective cursor survives first-epoch setup and configured stopping."""
        with TemporaryDirectory() as directory:
            dataset = build_image_corpus(directory, size=4)
            original = _loader(dataset)
            next(original)
            checkpoint = original.state_dict()
            original.wait_for_prefetch()
            loader = _loader(dataset)
            loader.load_state_dict(checkpoint)
            trainer = VLMTrainer.__new__(VLMTrainer)
            base = SimpleNamespace(
                config=SimpleNamespace(dataloader=SimpleNamespace(drop_last=True)), local_rank=0,
                state=SimpleNamespace(global_step=1, epoch=0), train_iters=5, train_epochs=1, train_steps=5,
                train_dataloader=loader, destroy_distributed=Mock(), on_train_begin=Mock(), on_train_end=Mock(),
                on_epoch_begin=Mock(), on_epoch_end=Mock(),
            )
            trainer.base = base
            delivered = []

            def train_step(iterator: object) -> None:
                """Consume a native round as a stand-in for an optimizer update."""
                delivered.append(sorted((next(iterator)["input_ids"][:, 0] - 100).tolist()))
                base.state.global_step += 1

            with patch.object(trainer, "train_step", side_effect=train_step), \
                    patch("hyper_parallel.trainer.vlm_trainer.synchronize"), \
                    patch("hyper_parallel.trainer.vlm_trainer.print_device_mem_info"), \
                    patch.object(loader, "wait_for_prefetch", wraps=loader.wait_for_prefetch) as drain:
                trainer.train()
                drain.assert_called_once()
            self.assertEqual(delivered, [[2, 3], [0, 1], [2, 3], [0, 1]])
            self.assertEqual(base.state.global_step, 5)
            self.assertEqual(base.state.epoch, 2)
            base.destroy_distributed.assert_called_once()
            loader.wait_for_prefetch()

    def test_step_checkpoint_sees_completed_update(self) -> None:
        """Callbacks see the same update count as the consumed native batches."""
        trainer = VLMTrainer.__new__(VLMTrainer)
        trainer.base = SimpleNamespace(
            config=SimpleNamespace(training=SimpleNamespace(max_grad_norm=1.0)),
            num_micro_batches=2, get_batch=next, model=Mock(), model_reshard=Mock(),
            configure_fsdp_gradient_sync=Mock(), state=SimpleNamespace(global_step=3),
            optimizer=Mock(), lr_scheduler=Mock(), on_step_begin=Mock(), on_step_end=Mock(),
            forward_backward_step=Mock(return_value=(torch.tensor(1.0), {"foundation_loss": torch.tensor(1.0)})),
        )
        observed = []
        trainer.base.on_step_end.side_effect = lambda **kwargs: observed.append(trainer.base.state.global_step)
        loss_inputs = {"labels": torch.tensor([[-100, 1, 2]])}
        batches = [({"input_ids": torch.ones(1, 3)}, loss_inputs)] * 2
        with patch("hyper_parallel.trainer.vlm_trainer.synchronize"), \
                patch("hyper_parallel.trainer.vlm_trainer.clip_grad_norm_", return_value=0.5):
            trainer.train_step(iter(batches))
        self.assertEqual(observed, [4])
        trainer.base.optimizer.step.assert_called_once()
        trainer.base.optimizer.reset_mock()
        trainer.base.forward_backward_step.reset_mock()
        with self.assertRaises(StopIteration):
            trainer.train_step(iter(batches[:1]))
        trainer.base.optimizer.step.assert_not_called()
        trainer.base.forward_backward_step.assert_not_called()
        self.assertEqual(trainer.base.state.global_step, 4)

    def test_native_training_loop_keeps_fixed_epoch_policy(self) -> None:
        """Non-collective DataLoaders still use the existing bounded epoch schedule."""
        trainer = VLMTrainer.__new__(VLMTrainer)
        trainer.base = SimpleNamespace(
            config=SimpleNamespace(dataloader=SimpleNamespace(drop_last=True)), local_rank=0,
            state=SimpleNamespace(global_step=0, epoch=0), train_iters=3, train_epochs=2, train_steps=2,
            train_dataloader=[0, 1], destroy_distributed=Mock(), on_train_begin=Mock(), on_train_end=Mock(),
            on_epoch_begin=Mock(), on_epoch_end=Mock(),
        )
        delivered = []

        def train_step(iterator: object) -> None:
            """Record fixed epoch iteration without model execution."""
            delivered.append(next(iterator))
            trainer.base.state.global_step += 1

        with patch.object(trainer, "train_step", side_effect=train_step), \
                patch("hyper_parallel.trainer.vlm_trainer.synchronize"), \
                patch("hyper_parallel.trainer.vlm_trainer.print_device_mem_info"):
            trainer.train()
        self.assertEqual(delivered, [0, 1, 0])
        self.assertEqual(trainer.base.state.global_step, 3)
        self.assertEqual(trainer.base.state.epoch, 2)
