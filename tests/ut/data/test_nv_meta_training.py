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
"""Nv-meta training paths and input contracts through real HP CPU pipelines."""

import importlib
import io
import json
import tempfile
import unittest
from contextlib import contextmanager
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import Mock, patch

import numpy as np
import torch
from PIL import Image
from torch.nn import functional

from hyper_parallel.data.batching.build_collate_fn import OmniCollator, TextPackingCollator
from hyper_parallel.data.batching.build_dataloader import OmniPackingLoader, TokenBatchLoader
from hyper_parallel.data.batching.get_batch import OmniParallelBatch, SynchronizedBatchReader, TextParallelBatch
from hyper_parallel.data.constants import IGNORE_INDEX, ONLINE_SOURCE_PATH_KEY
from hyper_parallel.data.nv_meta.sample_adapter import NvMetaSampleAdapter
from hyper_parallel.data.omni.omni_transform import OmniDataTransform, _PreprocessedOmniTransform
from hyper_parallel.data.online.provider import SOURCE_INFO_KEY, SourceInfo
from hyper_parallel.data.parallel.batch_sampler import build_dataset_batch_sampler
from hyper_parallel.data.text.text_transform import PlaintextTransform, PretokenizedTextTransform
from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.config.data import DatasetConfig
from hyper_parallel.trainer.config.resolver import resolve_component
from hyper_parallel.trainer.text_trainer import TextTrainer
from hyper_parallel.trainer.vlm_trainer import VLMTrainer
from tests.fixtures.data.nv_meta import make_nv_meta
from tests.ut.data.conftest import FakeTokenizer


class _ImageTransform(OmniDataTransform):
    """A deterministic image encoder with an observable model-owned batch hook."""

    def __init__(self, prepared: bool = False) -> None:
        """Use CPU inputs without downloading a pretrained processor."""
        super().__init__(max_seq_len=8, processor=object())
        self.prepared = prepared
        self.batch_calls = 0

    def encode_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Decode pixels only on the raw path; prepared samples must bypass this."""
        if self.prepared:
            raise AssertionError(f"Prepared sample was re-encoded: {tuple(sample)}")
        image = self.prepare_messages(sample)[0]["content"][0]["image"]
        pixels = torch.from_numpy(np.array(image)).float().mean(dim=(0, 1)).unsqueeze(0) / 255
        return {"input_ids": torch.tensor([1, 2]), "labels": torch.tensor([IGNORE_INDEX, 3]),
                "pixel_values": pixels}

    def encode_batch(self, batch: Any) -> Any:
        """Make the post-packing model hook observable in the training input."""
        self.batch_calls += 1
        return {**batch, "pixel_values": batch["pixel_values"] * 2}


def _record(omni: bool, prepared: bool) -> tuple[dict[str, bytes], dict[str, Any], Any]:
    if not omni:
        if prepared:
            return ({"json": b'{"input_ids":[1,2],"labels":[2,3],"loss_mask":[0,1]}'},
                    {}, PretokenizedTextTransform(8))
        return {"txt": b"abc"}, {"record_part": "txt"}, PlaintextTransform(
            tokenizer=FakeTokenizer(), max_seq_len=8)
    stream = io.BytesIO()
    if prepared:
        np.savez(stream, input_ids=np.array([1, 2], dtype=np.int32), labels=[IGNORE_INDEX, 3],
                 pixel_values=np.ones((1, 3), dtype=np.float16))
        return {"npz": stream.getvalue()}, {"record_part": "npz"}, _ImageTransform(True)
    Image.new("RGB", (2, 3), (25, 50, 75)).save(stream, format="PNG")
    record = {"messages": [{"role": "user", "content": [{"type": "image", "image": "part:.png"}]},
                           {"role": "assistant", "content": "answer"}]}
    return {"json": json.dumps(record).encode(), "png": stream.getvalue()}, {}, _ImageTransform()


class TestNvMetaTraining(unittest.TestCase):
    """Exercise each supported ingestion path once, then its critical boundaries."""

    @contextmanager
    def _training_case(self, omni: bool, prepared: bool, access: str, scenario: str) -> Iterator[Any]:
        with tempfile.TemporaryDirectory() as directory:
            parts, source_options, transform = _record(omni, prepared)
            root = make_nv_meta(Path(directory), [parts] * 4)
            builder = "omni.build_dataset.build_online_omni_mapping_dataset" if omni else {
                "mapping": "text.build_dataset.build_online_text_mapping_dataset",
                "iterable": "text.build_dataset.build_online_iterable_dataset",
            }[access]
            node = {"_target_": f"hyper_parallel.data.{builder}", "data_config": {},
                    "source": {"_target_": "hyper_parallel.data.nv_meta.NvMetaSource", "data_path": str(root),
                               "data_config": {"cache_dir": str(root / "cache")}, **source_options}}
            if scenario == "legacy":
                node["source"].pop("record_part", None)
                node["sample_adapter"] = {
                    "_target_": "hyper_parallel.data.nv_meta.NvMetaSampleAdapter", **source_options}
            if omni:
                node["preprocessed"] = prepared
            elif prepared:
                node["data_transform"] = {
                    "_target_": "hyper_parallel.data.text.text_transform.PretokenizedTextTransform", "max_seq_len": 8}
            config = resolve_component(node, annotation=DatasetConfig, path="$.dataset")
            restored = resolve_component(config.to_dict(), annotation=DatasetConfig, path="$.dataset")
            self.assertEqual(restored.to_dict(), config.to_dict())
            if restored.data_transform is not None:
                transform = restored.data_transform.build()
            dataset = restored.build(transform=transform, mesh_context=SimpleNamespace())
            micro_batches = 2 if scenario == "empty" else 1
            loader_options = {}
            if access == "mapping":
                loader_options["batch_sampler"] = build_dataset_batch_sampler(
                    total_samples=len(dataset), micro_batch_size=1, global_batch_size=micro_batches,
                    dp_rank=0, dp_world_size=1)
            loader_type, collator = (OmniPackingLoader, OmniCollator()) if omni else (
                TokenBatchLoader, TextPackingCollator())
            loader = loader_type(dataset, collate_fn=collator, batch_size=1, dp_world_size=1,
                                 max_seq_len=8, min_buffered_samples=1, **loader_options)
            runtime = OmniParallelBatch(mesh_context=SimpleNamespace(), device=torch.device("cpu")) if omni else (
                TextParallelBatch(SimpleNamespace(), torch.device("cpu"), None, {}, False,
                                  source_type="online", attention_mode="compressed"))
            if scenario != "legacy":
                runtime = SynchronizedBatchReader(runtime, num_micro_batches=micro_batches, device="cpu")
            trainer, observed = self._cpu_trainer(omni, loader, runtime, micro_batches)
            observed["transform"] = transform
            try:
                yield trainer, observed
            finally:
                dataset.source_dataset.source.source.close()

    @staticmethod
    def _cpu_trainer(omni: bool, loader: Any, runtime: Any, micro_batches: int) -> tuple[Any, dict[str, Any]]:
        model = torch.nn.Sequential(torch.nn.Embedding(128, 4), torch.nn.Linear(4, 128))
        vision = torch.nn.Linear(3, 128)
        optimizer = torch.optim.SGD([*model.parameters(), *vision.parameters()], lr=0.01)
        observed = {"steps": 0, "epochs": 0, "initial": model[0].weight.detach().clone(), "model": model}

        def forward_backward(model_inputs: dict[str, Any], loss_inputs: dict[str, Any]) -> Any:
            """Use real packed labels, media inputs and autograd, including ignored tokens."""
            logits = model(model_inputs["input_ids"])
            if omni:
                logits = logits + vision(model_inputs["pixel_values"].float()).mean(dim=0)
            logits.retain_grad()
            loss = functional.cross_entropy(logits.reshape(-1, 128), loss_inputs["labels"].reshape(-1))
            loss.backward()
            observed.update(inputs=model_inputs, labels=loss_inputs["labels"], gradient=logits.grad, loss=loss.detach())
            return loss.detach(), {"loss": loss.detach()}

        def optimizer_step() -> None:
            """Count actual SGD updates and clear accumulated gradients."""
            observed["steps"] += 1
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        def epoch_begin() -> None:
            """Record the epochs actually entered by the existing Trainer loop."""
            observed["epochs"] += 1

        base = SimpleNamespace(
            config=SimpleNamespace(training=SimpleNamespace(train_iters=4, train_samples=None,
                                                            global_batch_size=micro_batches),
                                   dataloader=SimpleNamespace(drop_last=True)),
            state=SimpleNamespace(global_step=0, epoch=0), local_rank=0, train_dataloader=loader,
            num_micro_batches=micro_batches, get_batch=runtime, model_integration=SimpleNamespace(begin_step=Mock()),
            model_reshard=Mock(), configure_fsdp_gradient_sync=Mock(), begin_fsdp_runtime_diagnostics=Mock(),
            forward_backward_step=forward_backward, prepare_optimizer_step=Mock(return_value=0.0),
            step_optimizers_and_schedulers=optimizer_step, _end_model_integration_step=Mock(),
            on_train_begin=Mock(), on_train_end=Mock(), on_epoch_begin=epoch_begin, on_epoch_end=Mock(),
            on_step_begin=Mock(), on_step_end=Mock(), on_micro_step_begin=Mock(), destroy_distributed=Mock())
        BaseTrainer._compute_train_iters(base)  # No public standalone train-budget calculation API.
        trainer_type = VLMTrainer if omni else TextTrainer
        trainer = trainer_type.__new__(trainer_type)
        trainer.base = base
        return trainer, observed

    def test_training_paths_complete_budget_and_preserve_legacy_loop(self) -> None:
        """Train all six input paths; check resume, insufficient GA and legacy limits."""
        paths = ((False, "mapping"), (False, "iterable"), (True, "mapping"))
        cases = [(omni, prepared, access, "fresh") for (omni, access), prepared in product(paths, (False, True))]
        cases += [(omni, True, "mapping", scenario) for omni in (False, True)
                  for scenario in ("resume", "legacy", "empty")]
        for omni, prepared, access, scenario in cases:
            with self.subTest(omni=omni, prepared=prepared, access=access, scenario=scenario), self._training_case(
                omni, prepared, access, scenario,
            ) as (trainer, observed):
                if scenario == "resume":
                    loader = trainer.base.train_dataloader
                    iterator = iter(loader)
                    next(iterator)
                    self.assertEqual(list(iterator), [])
                    loader.load_state_dict(loader.state_dict())
                    trainer.base.state.global_step = 1
                module = importlib.import_module(type(trainer).__module__)
                # Only device telemetry and unconfigured distributed/model hooks are mocked.
                with patch.object(module, "print_device_mem_info"), patch.object(module, "synchronize"):
                    if scenario == "empty":
                        with self.assertRaisesRegex(ValueError, "no complete optimizer step"):
                            trainer.train()
                        self.assertEqual(observed["steps"], 0)
                        continue
                    trainer.train()
                self.assertEqual(trainer.base.state.global_step, 1 if scenario == "legacy" else 4)
                self.assertEqual(observed["steps"], {"legacy": 1, "resume": 3}.get(scenario, 4))
                self.assertEqual(observed["epochs"], 1 if scenario == "legacy" else 4)
                self.assertTrue(torch.isfinite(observed["loss"]))
                self.assertFalse(torch.equal(observed["initial"], observed["model"][0].weight))
                expected_ids = [1, 2] if omni or prepared else [97, 98]
                expected_labels = [IGNORE_INDEX, 3] if omni or prepared else [98, 99]
                torch.testing.assert_close(observed["inputs"]["input_ids"], torch.tensor([expected_ids * 4]))
                torch.testing.assert_close(observed["labels"], torch.tensor([expected_labels * 4]))
                self.assertEqual(torch.count_nonzero(observed["gradient"][observed["labels"] == IGNORE_INDEX]), 0)
                self.assertGreater(torch.count_nonzero(observed["gradient"]).item(), 0)
                if omni:
                    self.assertGreater(observed["transform"].batch_calls, 0)
                    pixels = observed["inputs"]["pixel_values"]
                    expected = torch.full((4, 3), 2, dtype=torch.float16) if prepared else (
                        torch.tensor([[25, 50, 75]] * 4) * (2 / 255))
                    torch.testing.assert_close(pixels, expected)

    def test_prepared_contracts_and_safe_media_decoding(self) -> None:
        """Check semantic loss prevention, shared tensor storage and decode boundaries."""
        text = PretokenizedTextTransform(3)
        source = {"input_ids": torch.tensor([1, 2]), "labels": torch.tensor([2, 3])}
        self.assertIs(text(source)[0]["labels"], source["labels"])
        masked = text({**source, "loss_mask": [0, 1]})[0]
        self.assertIs(masked["input_ids"], source["input_ids"])
        torch.testing.assert_close(masked["labels"], torch.tensor([IGNORE_INDEX, 3]))
        torch.testing.assert_close(source["labels"], torch.tensor([2, 3]))
        self.assertEqual(text({**source, "loss_mask": [0, 0]}), [])
        chunks = text({"input_ids": torch.arange(8)})
        torch.testing.assert_close(torch.cat([chunk["labels"] for chunk in chunks]), torch.arange(1, 8))
        omni = _PreprocessedOmniTransform(_ImageTransform(True))
        prepared = {**source, "position_ids": torch.tensor([[0, 1], [0, 0], [0, 0]]),
                    "pixel_values": torch.ones(1, 3, dtype=torch.bfloat16), "metadata": {"ignored": True}}
        encoded = omni.encode_sample(prepared)
        self.assertIs(encoded["pixel_values"], prepared["pixel_values"])
        self.assertIs(encoded["position_ids"], prepared["position_ids"])
        self.assertNotIn("metadata", encoded)
        invalid = [
            (text, {"loss_mask": [1, 0.5]}), (text, {"loss_mask": [1, float("nan")]}),
            (text, {"loss_mask": [1]}), (text, {"attention_mask": [1, 1]}),
            (text, {"position_ids": [0, 1]}), (text, {"input_ids": [1.0, 2.0]}),
            (text, {"input_ids": [-1, 2]}), (text, {"labels": [1]}), (text, {"labels": [1, -1]}),
            (omni.encode_sample, {"attention_mask": [[1, 0], [1, 1]]}),
            (omni.encode_sample, {"position_ids": [[0, 1, 2]]}), (omni.encode_sample, {"labels": [1]}),
            (omni.encode_sample, {"input_ids": [[1, 2]]}), (omni.encode_sample, {"loss_mask": [1, -1]}),
            (omni.encode_sample, {"pixel_values": torch.ones(1, 3, requires_grad=True)}),
        ]
        for transform, fields in invalid:
            with self.subTest(transform=type(transform).__name__, fields=fields), self.assertRaises(ValueError):
                transform({**source, **fields})
        stream = io.BytesIO()
        torch.save({"pixel_values": prepared["pixel_values"]}, stream)
        restored = NvMetaSampleAdapter(record_part="pth")({"parts": {"pth": stream.getvalue()}})
        torch.testing.assert_close(restored["pixel_values"], prepared["pixel_values"])
        token_part = io.BytesIO()
        np.save(token_part, np.array([4, 5], dtype=np.int32))
        parts, _, _ = _record(True, True)
        adapter = NvMetaSampleAdapter(record_part="npz", field_map={"input_ids": "tokens.npy"})
        decoded = adapter({"parts": {**parts, "tokens.npy": token_part.getvalue()}})
        np.testing.assert_array_equal(decoded["input_ids"], [4, 5])
        self.assertEqual(adapter.required_parts, ("npz", "tokens.npy"))
        parts, _, _ = _record(True, False)
        with tempfile.TemporaryDirectory() as directory:
            record = json.loads(parts["json"])
            record["messages"][0]["content"].append({"type": "image", "image": "images/external.png"})
            parts["json"] = json.dumps(record).encode()
            sample = NvMetaSampleAdapter()({"parts": parts,
                                           SOURCE_INFO_KEY: SourceInfo("nv-meta", source_path=directory)})
            self.assertIn(ONLINE_SOURCE_PATH_KEY, sample)
            messages = _ImageTransform().prepare_messages(sample)
            self.assertEqual(Path(messages[0]["content"][1]["image"]), Path(directory) / "images/external.png")
            decoded = NvMetaSampleAdapter(image_mode="bytes")({"parts": parts})
            self.assertIs(decoded["messages"][0]["content"][0]["data"], parts["png"])
        unsafe = io.BytesIO()
        np.save(unsafe, np.array([{"value": 1}], dtype=object))
        video = {"messages": [{"content": [{"type": "video", "video": "part:mp4"}]}]}
        errors = [
            (NvMetaSampleAdapter(), {}, "missing part"),
            (NvMetaSampleAdapter(record_part=None, field_map={"input_ids": "npy"}),
             {"npy": unsafe.getvalue()}, "allow_pickle=False"),
            (NvMetaSampleAdapter(), {"json": json.dumps(video).encode(), "mp4": b"video"}, "processor-specific"),
        ]
        for adapter, payload, message in errors:
            with self.subTest(error=message), self.assertRaisesRegex(ValueError, message):
                adapter({"parts": payload})
