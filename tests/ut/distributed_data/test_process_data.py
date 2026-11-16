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
"""Process isolation, shared-storage ownership and consumed-cursor recovery."""

import os
import unittest
from contextlib import closing
from pathlib import Path

import torch

from hyper_parallel.distributed_data import DistributedDatasetConfig, SampleMetadata, build_distributed_dataloader
from hyper_parallel.distributed_data.shared_batch import SharedBatchReader, SharedBatchWriter
from tests.ut.distributed_data.test_external_step import _CountingSource, _Mesh


def _pack(samples, seq_len):
    del seq_len
    return {
        "values": torch.cat([sample["values"] for sample in samples]),
        "pid": os.getpid(),
        "npu_initialized": bool(hasattr(torch, "npu") and torch.npu.is_initialized()),
    }


def _loader(source, max_steps=None):
    return build_distributed_dataloader(
        None, _Mesh(), DistributedDatasetConfig(seq_len=32, local_batch_size=1),
        external_step_source=source, metadata_fn=lambda sample: SampleMetadata(len(sample["values"])),
        pack_fn=_pack, collate_fn=list, cost_model=lambda metadata: metadata.cost,
        device="cpu", max_steps=max_steps,
    )


class TestSharedBatch(unittest.TestCase):
    """Ensure subsequent publications cannot mutate previously received tensors."""

    def test_growth_reuse_empty_and_noncontiguous_tensors(self):
        """Retain mixed dtype batches while reusing and growing the sender's buffers."""
        with closing(SharedBatchWriter()) as writer, closing(SharedBatchReader()) as reader:
            sample = {"x": torch.arange(12).view(3, 4).t(), "y": (torch.tensor(True), torch.empty(0)), "z": None}
            first = reader.read(writer.write(sample))
            for size in (1, 2048, 2):
                current = reader.read(writer.write({"x": torch.full((size,), -1, dtype=torch.int64)}))
                self.assertEqual(current["x"].tolist(), [-1] * size)
                torch.testing.assert_close(first["x"], sample["x"])
            self.assertEqual(first["y"][0].item(), True)
            self.assertEqual(first["y"][1].numel(), 0)
            self.assertIsNone(first["z"])


class TestProcessProducer(unittest.TestCase):
    """Use real CPU subprocesses without initializing distributed communication."""

    def test_process_isolation_prefetch_restore_and_epoch(self):
        """Replay a pending batch exactly once and retain old batch storage across steps."""
        steps = [[[{"values": torch.arange(size) + index * 100}]] for index, size in enumerate((1, 2, 32, 3))]
        with closing(_loader(_CountingSource(steps))) as loader:
            first = next(loader)[0]
            loader.prefetch_plan()
            loader.prefetch()
            loader.wait_for_prefetch()
            self.assertNotEqual(first["pid"], os.getpid())
            self.assertFalse(first["npu_initialized"])
            self.assertEqual(loader.step, 1)
            checkpoint = loader.state_dict()
            self.assertEqual(checkpoint["producer"]["source"]["position"], 2)
            expected = [batch[0]["values"].clone() for batch in loader]
            torch.testing.assert_close(first["values"], torch.tensor([0]))
            loader.load_state_dict(checkpoint)
            loader.set_epoch(0)
            restored = [batch[0]["values"] for batch in loader]
            self.assertEqual(len(restored), 3)
            for actual, reference in zip(restored, expected):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            loader.set_epoch(1)
            torch.testing.assert_close(next(loader)[0]["values"], torch.tensor([0]))
        self.assertIsNotNone(loader._producer._process.poll())

    def test_live_generator_is_rejected_before_process_start(self):
        """Require a serializable iterable instead of silently falling back to a thread."""
        source = (value for value in [])
        with closing(_loader(source)) as loader:
            with self.assertRaisesRegex(TypeError, "generator"):
                next(loader)
            self.assertIsNone(loader._producer)

    def test_killed_producer_reclaims_shared_storage(self):
        """Report a disconnected producer and unlink even its child-owned files."""
        steps = [[[{"values": torch.arange(4)}]]] * 4
        loader = _loader(_CountingSource(steps))
        try:
            next(loader)
            loader.wait_for_prefetch()
            shared_directory = Path(loader._producer._shared_directory)
            self.assertTrue(list(shared_directory.rglob("*")))
            loader._producer._process.kill()
            loader._producer._process.wait(timeout=10)
            # The already received speculative result remains valid.
            next(loader)
            with self.assertRaises((RuntimeError, BrokenPipeError, ConnectionResetError)):
                next(loader)
        finally:
            loader.close()
        self.assertFalse(shared_directory.exists())

    def test_missing_source_checkpoint_contract(self):
        """Plain iterables remain usable but cannot claim resumable source progress."""
        with closing(_loader([])) as loader:
            with self.assertRaisesRegex(NotImplementedError, "state_dict"):
                loader.state_dict()
            with self.assertRaisesRegex(NotImplementedError, "load_state_dict"):
                loader.load_state_dict({})
