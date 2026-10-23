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
"""CPU regression tests for the opt-in multimodal data path."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import torch
import yaml

import hyper_parallel.data as data_package
from hyper_parallel.data.omni.build_dataset import OmniMappingDataset
from hyper_parallel.data.omni.collator import OmniCollator
from hyper_parallel.data.omni.get_batch import OmniParallelBatch
from hyper_parallel.data.omni.loader import OmniPackingLoader
from hyper_parallel.data.omni.omni_transform import (
    AutoProcessorTransform, OmniDataTransform, _OmniTransformStrategy, build_auto_processor,
)
from hyper_parallel.data.omni.packing import SamplePacker
from hyper_parallel.data.omni.parallel import OmniCPBatchSharder, OmniTPBatchBroadcaster
from hyper_parallel.data.batching.build_dataloader import build_dataloader
from hyper_parallel.data.parallel.dataloader_parallel import DataLoaderParallelContext
from hyper_parallel.trainer.config.resolver import resolve_component
from hyper_parallel.trainer.config.data import DatasetConfig, DataLoaderConfig
from hyper_parallel.trainer.vlm_trainer import VLMTrainer


class FakeProcessor:
    """Return stable processor-shaped data without loading any model."""

    def apply_chat_template(self, messages: list, *, add_generation_prompt: bool, **kwargs: Any) -> dict:
        """Expose a masked media prefix and a supervised assistant suffix."""
        del messages, kwargs
        tokens = [1, 2, 3] if add_generation_prompt else [1, 2, 3, 4, 5]
        return {
            "input_ids": torch.tensor([tokens]),
            "attention_mask": torch.ones(1, len(tokens), dtype=torch.long),
            "mm_token_type_ids": torch.tensor([[0, 1, 0] + ([] if add_generation_prompt else [0, 0])]),
            "pixel_values": torch.ones(2, 3),
            "image_grid_thw": torch.tensor([[1, 1, 2]]),
        }


class BoundaryMergeProcessor(FakeProcessor):
    """Model expanded image tokens and an empty-thinking BPE boundary merge."""

    def __init__(self) -> None:
        """Initialize exact prompt/full token sequences for a boundary merge."""
        self.full_tokens = [1, 2, 2, 3, 271, 4, 271, 5, 6]
        self.prompt_tokens = [1, 2, 2, 3, 198]
        self.full_text = "image<think>\n\n</think>\n\nanswer"
        self.prompt_text = "image<think>\n"
        self.tokenizer = self

    def decode(self, tokens: list[int], **kwargs: Any) -> str:
        """Keep special tokens and whitespace visible during boundary checks."""
        del kwargs
        vocabulary = {1: "user", 2: "<image>", 3: "<think>", 198: "\n", 271: "\n\n",
                      4: "</think>", 5: "answer", 6: "<eos>", 7: "different", 8: "\ufffd"}
        return "".join(vocabulary[token] for token in tokens)

    def apply_chat_template(self, messages: list, *, add_generation_prompt: bool, **kwargs: Any) -> Any:
        """Return native sequence-aligned fields or rendered template text."""
        del messages
        if not kwargs.get("tokenize", True):
            return self.prompt_text if add_generation_prompt else self.full_text
        tokens = self.prompt_tokens if add_generation_prompt else self.full_tokens
        return {
            "input_ids": torch.tensor([tokens]),
            "attention_mask": torch.ones(1, len(tokens), dtype=torch.long),
            "mm_token_type_ids": torch.tensor([[0, 1, 1] + [0] * (len(tokens) - 3)]),
            "pixel_values": torch.ones(2, 3),
            "image_grid_thw": torch.tensor([[1, 1, 2]]),
        }


class DeferredTransform(OmniDataTransform):
    """Record encoding order for packing and resume tests."""

    def __init__(self) -> None:
        """Initialize metadata-only selection and record encoded sample IDs."""
        super().__init__(max_seq_len=16, processor=object())
        self.encoded = []

    def preencode_sample(self, sample: dict) -> dict:
        """Produce only selection metadata."""
        return {"sample_id": sample["sample_id"], "packing_length": sample["length"]}

    def postencode_sample(self, sample: dict) -> dict:
        """Allocate tensors after candidate selection."""
        self.encoded.append(sample["sample_id"])
        ids = torch.full((sample["packing_length"],), sample["sample_id"], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}


class TestOmniData(unittest.TestCase):
    """Verify lifecycle, packing, transport, and trainer integration."""

    def test_data_root_keeps_multimodal_exports_isolated(self):
        """The data root does not expose either old or new multimodal builders."""
        for name in ("build_vlm_get_batch", "build_vlm_dataset", "build_vlm_collator",
                     "OmniParallelBatch", "OmniPackingLoader", "AutoProcessorTransform"):
            self.assertNotIn(name, getattr(data_package, "__all__", ()))
            self.assertFalse(hasattr(data_package, name))

    def test_auto_processor_retains_tokenizer_template_fallback(self):
        """The replacement preserves the old processor's tokenizer-template fallback."""
        processor = SimpleNamespace(tokenizer=SimpleNamespace(chat_template="template"), chat_template=None)
        with patch("hyper_parallel.data.omni.omni_transform.AutoProcessor.from_pretrained", return_value=processor):
            result = build_auto_processor(pretrained_model_name_or_path="/model")
        self.assertIs(result, processor)
        self.assertEqual(result.chat_template, "template")

    def test_vlm_trainer_default_omni_batch_and_assets(self):
        """Build the real default Omni batch path without importing removed VLM modules."""
        trainer = VLMTrainer.__new__(VLMTrainer)
        trainer.base = SimpleNamespace(
            config=SimpleNamespace(
                dataset=object(), model=SimpleNamespace(tokenizer_path="/model"),
                dataloader=SimpleNamespace(get_batch=None, pp_shared_data=False),
            ),
            mesh=SimpleNamespace(tp_size=1, cp_size=1, pp_size=1), device="cpu", model_config=object(),
        )
        with patch("hyper_parallel.trainer.vlm_trainer.build_auto_processor") as processor_builder:
            trainer._build_model_assets()
            processor_builder.assert_called_once_with(pretrained_model_name_or_path="/model")
            self.assertIs(trainer.base.processor, processor_builder.return_value)
            self.assertIs(trainer.base.tokenizer, processor_builder.return_value.tokenizer)
        trainer._build_get_batch()
        self.assertIsInstance(trainer.base.get_batch, OmniParallelBatch)
        batch = {"input_ids": torch.tensor([[1, 2]]), "labels": torch.tensor([[-100, 2]])}
        model_inputs, loss_inputs = trainer.base.get_batch(None, external_batch=batch)
        self.assertEqual(model_inputs["input_ids"].tolist(), [[1, 2]])
        self.assertEqual(loss_inputs["loss_mask"].tolist(), [[False, True]])

    def test_processor_labels_and_length(self):
        """Final assistant is supervised; overlong media samples are not sliced."""
        transform = AutoProcessorTransform(max_seq_len=5, processor=FakeProcessor())
        sample = {"messages": [{"role": "user", "content": "image"}, {"role": "assistant", "content": "answer"}]}
        result = transform.encode_sample(sample)
        self.assertEqual(result["labels"].tolist(), [-100, -100, -100, 4, 5])
        self.assertEqual(tuple(result["pixel_values"].shape), (2, 3))
        with self.assertRaisesRegex(ValueError, "exceeds max_seq_len"):
            AutoProcessorTransform(max_seq_len=4, processor=FakeProcessor()).encode_sample(sample)
        with self.assertRaisesRegex(ValueError, "final assistant"):
            transform.encode_sample({"messages": sample["messages"][:-1]})

    def test_processor_masks_merged_boundary_without_changing_inputs(self):
        """Empty thinking preserves image expansion and masks the shared newline token."""
        processor = BoundaryMergeProcessor()
        transform = AutoProcessorTransform(max_seq_len=16, processor=processor)
        messages = [{"role": "user", "content": "image"}, {"role": "assistant", "content": "answer"}]
        original = processor.apply_chat_template(messages, add_generation_prompt=False)
        result = transform.encode_sample({"messages": messages})
        self.assertEqual(result["labels"].tolist(), [-100] * 5 + [4, 271, 5, 6])
        for field in ("input_ids", "attention_mask", "mm_token_type_ids"):
            self.assertTrue(torch.equal(result[field], original[field].squeeze(0)))
        for field in ("pixel_values", "image_grid_thw"):
            self.assertTrue(torch.equal(result[field], original[field]))

    def test_processor_rejects_real_rewrites_and_lossy_decoding(self):
        """Text rewrites and changed expanded tokens must not be hidden by the fallback."""
        sample = {"messages": [{"role": "user"}, {"role": "assistant"}]}
        cases = (("prompt_text", "changed", "text prefix"),
                 ("prompt_tokens", [1, 7, 2, 3, 198], "lossless text"),
                 ("prompt_tokens", [1, 2, 2, 3, 8], "lossless text"))
        for field, value, error in cases:
            with self.subTest(field=field, value=value):
                processor = BoundaryMergeProcessor()
                setattr(processor, field, value)
                with self.assertRaisesRegex(ValueError, error):
                    AutoProcessorTransform(max_seq_len=16, processor=processor).encode_sample(sample)

    def test_processor_rejects_no_targets_after_boundary_masking(self):
        """A merge consuming all remaining tokens does not create a supervised answer."""
        processor = BoundaryMergeProcessor()
        processor.full_tokens = [1, 2, 2, 3, 271]
        sample = {"messages": [{"role": "user"}, {"role": "assistant"}]}
        with self.assertRaisesRegex(ValueError, "no supervised"):
            AutoProcessorTransform(max_seq_len=16, processor=processor).encode_sample(sample)

    def test_invalid_lifecycle_is_rejected(self):
        """Eager and deferred hooks cannot be mixed accidentally."""
        class MixedTransform(DeferredTransform):
            def encode_sample(self, sample: dict) -> dict:
                """Provide an intentionally conflicting lifecycle hook."""
                return sample
        with self.assertRaisesRegex(TypeError, "cannot combine"):
            _OmniTransformStrategy.from_transform(MixedTransform())

    def test_relative_media_paths_do_not_mutate_records(self):
        """Resolve relative images against the source file, not the process CWD."""
        transform = AutoProcessorTransform(max_seq_len=8, processor=FakeProcessor())
        sample = {
            "__online_source_path__": "/data/records/samples.jsonl",
            "messages": [{"role": "user", "content": [
                {"type": "image", "image": "images/a.png"},
                {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
            ]}],
        }
        before = copy.deepcopy(sample)
        result = transform.prepare_messages(sample)
        self.assertEqual(result[0]["content"][0]["image"], "/data/records/images/a.png")
        self.assertEqual(result[0]["content"][1]["image_url"]["url"], "https://example.com/a.png")
        self.assertEqual(sample, before)

    def test_packing_preserves_media_offsets(self):
        """Token spans shift while pixels retain their native dtype and values."""
        samples = [
            {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([-100, 2, 3]),
             "pixel_values": torch.ones(2, 3, dtype=torch.bfloat16), "image_token_starts": torch.tensor([1])},
            {"input_ids": torch.tensor([4, 5]), "labels": torch.tensor([-100, 5]),
             "pixel_values": torch.zeros(1, 3, dtype=torch.bfloat16), "image_token_starts": torch.tensor([0])},
        ]
        packed = SamplePacker().pack_selected_samples(samples)
        self.assertEqual(packed["cu_seq_lens"].tolist(), [0, 3, 5])
        self.assertEqual(packed["image_token_starts"].tolist(), [1, 3])
        batch = OmniCollator()([packed])
        self.assertEqual(tuple(batch["input_ids"].shape), (1, 5))
        self.assertEqual(tuple(batch["pixel_values"].shape), (3, 3))
        self.assertEqual(batch["pixel_values"].dtype, torch.bfloat16)

    def test_deferred_loader_resume_and_tail(self):
        """Restore unconsumed candidates and drain the final incomplete budget."""
        for save_by_idx in (True, False):
            with self.subTest(save_by_idx=save_by_idx), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "samples.jsonl"
                records = [
                    {"sample_id": index, "length": length, "messages": [{"role": "user", "content": "x"}]}
                    for index, length in enumerate([3, 2, 4, 1, 2], start=1)
                ]
                path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
                transform = DeferredTransform()
                dataset = OmniMappingDataset(str(path), transform)
                self.assertEqual(dataset[0]["packing_length"], 3)
                self.assertEqual(transform.encoded, [])

                def make_loader(source: OmniMappingDataset) -> OmniPackingLoader:
                    """Use identical budgets and checkpoint options for both runs."""
                    return OmniPackingLoader(
                        dataset=source, collate_fn=OmniCollator(), batch_size=1, max_seq_len=16,
                        token_budget=5, min_buffered_samples=3, save_by_idx=save_by_idx,
                    )

                loader = make_loader(dataset)
                iterator = iter(loader)
                first = next(iterator)
                self.assertEqual(first["input_ids"].tolist(), [[1, 1, 1, 2, 2]])
                self.assertEqual(transform.encoded, [1, 2])
                state = loader.state_dict()
                expected = list(iterator)
                replay = make_loader(OmniMappingDataset(str(path), DeferredTransform()))
                replay.load_state_dict(state)
                actual = list(replay)
                self.assertEqual(len(actual), len(expected))
                for left, right in zip(actual, expected):
                    for field in left:
                        torch.testing.assert_close(left[field], right[field], rtol=0, atol=0)
                self.assertEqual(actual[-1]["input_ids"].tolist(), [[5, 5]])
                state["dp_world_size"] = 2
                with self.assertRaisesRegex(ValueError, "DP world-size"):
                    replay.load_state_dict(state)

    def test_cp_slices_only_tokens(self):
        """Keep full images and global spans while padding/slicing token fields."""
        pixels = torch.randn(7, 3)
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]), "labels": torch.tensor([[-100, 2, 3, 4, 5]]),
            "pixel_values": pixels, "image_token_starts": torch.tensor([[1]]),
            "cu_seq_lens": torch.tensor([0, 3, 5], dtype=torch.int32),
        }
        for rank in range(2):
            context = DataLoaderParallelContext(cp_world_size=2, cp_rank=rank)
            output = OmniCPBatchSharder(context, {"input_ids": 0, "labels": -100}).shard(batch)
            self.assertIs(output["pixel_values"], pixels)
            self.assertIs(output["image_token_starts"], batch["image_token_starts"])
            self.assertEqual(output["cu_seq_lens"].tolist(), [0, 3, 5, 8])
            self.assertEqual(output["input_ids"].tolist(), [[1, 2, 3, 4]] if rank == 0 else [[5, 0, 0, 0]])
            if rank == 1:
                self.assertEqual(output["labels"].tolist(), [[5, -100, -100, -100]])

    def test_tp_metadata_broadcast_preserves_dtype(self):
        """Non-source ranks reconstruct mixed-shape and mixed-dtype fields."""
        source = {"input_ids": torch.tensor([[1, 2]]),
                  "pixel_values": torch.ones(3, 4, dtype=torch.bfloat16), "flag": True}
        context = DataLoaderParallelContext(tp_world_size=2, tp_rank=1, tp_group=object())
        metadata = [(name, tuple(value.shape), value.dtype, None) if torch.is_tensor(value)
                    else (name, None, None, value) for name, value in source.items()]
        payloads = iter(value for value in source.values() if torch.is_tensor(value))

        def broadcast_meta(items: list, **kwargs: Any) -> None:
            """Supply source metadata to the simulated receiving rank."""
            self.assertEqual(kwargs["src"], 4)
            items[0] = metadata

        def broadcast_tensor(tensor: torch.Tensor, **kwargs: Any) -> None:
            """Fill receiver storage with the corresponding source payload."""
            self.assertEqual(kwargs["src"], 4)
            tensor.copy_(next(payloads))

        with patch("torch.distributed.get_process_group_ranks", return_value=[4, 7]), \
                patch("torch.distributed.broadcast_object_list", side_effect=broadcast_meta), \
                patch("torch.distributed.broadcast", side_effect=broadcast_tensor):
            result = OmniTPBatchBroadcaster(context, "cpu").broadcast(None)
        for name, expected in source.items():
            if torch.is_tensor(expected):
                torch.testing.assert_close(result[name], expected, rtol=0, atol=0)
            else:
                self.assertEqual(result[name], expected)

    def test_runtime_adapter_is_required_for_packed_attention(self):
        """Do not silently train cross-sample attention without model integration."""
        context = DataLoaderParallelContext()
        with patch("hyper_parallel.data.omni.get_batch.create_dataloader_parallel_context", return_value=context):
            get_batch = OmniParallelBatch(mesh_context=SimpleNamespace(), device="cpu")
        batch = {"input_ids": torch.tensor([[1, 2, 3]]), "labels": torch.tensor([[-100, 2, 3]]),
                 "cu_seq_lens": torch.tensor([0, 1, 3])}
        with self.assertRaisesRegex(ValueError, "attention isolation"):
            get_batch(None, external_batch=batch)
        batch["cu_seq_lens"] = torch.tensor([0, 3])
        model_inputs, loss_inputs = get_batch(None, external_batch=batch)
        self.assertNotIn("cu_seq_lens", model_inputs)
        self.assertEqual(loss_inputs["loss_mask"].tolist(), [[False, True, True]])
        with self.assertRaisesRegex(NotImplementedError, "pipeline"):
            OmniParallelBatch(mesh_context=SimpleNamespace(pp_size=2), device="cpu")

    def test_example_config_resolves_without_text_changes(self):
        """Resolve only new opt-in targets through the existing config resolver."""
        path = Path(__file__).resolve().parents[4] / "examples/data/omni_data.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        dataset = resolve_component(config["dataset"], expected_type=DatasetConfig, path="dataset")
        loader = resolve_component(config["dataloader"], expected_type=DataLoaderConfig, path="dataloader")
        self.assertIs(dataset.data_transform.callable, AutoProcessorTransform)
        self.assertIs(loader.target.callable, OmniPackingLoader)
        self.assertEqual(loader.dataloader_type, "single")

    def test_example_dataset_loader_and_get_batch_integration(self):
        """Exercise actual config targets and the existing distributed sampler builder."""
        path = Path(__file__).resolve().parents[4] / "examples/data/omni_data.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "samples.jsonl"
            record = {"messages": [{"role": "user", "content": "image"},
                                   {"role": "assistant", "content": "answer"}]}
            source.write_text((json.dumps(record) + "\n") * 4, encoding="utf-8")
            config["dataset"]["data_path"] = str(source)
            dataset_config = resolve_component(config["dataset"], expected_type=DatasetConfig, path="dataset")
            loader_config = resolve_component(config["dataloader"], expected_type=DataLoaderConfig, path="dataloader")
            transform = dataset_config.data_transform.build(processor=FakeProcessor())
            dataset = dataset_config.build(transform=transform)
            mesh = SimpleNamespace(dp_size=1, dp_rank=0, tp_size=1, cp_size=1, pp_size=1)
            loaders, _ = build_dataloader(
                loader_config, datasets=(dataset, None, None), collate_fn=loader_config.collate_fn.build(),
                mesh_context=mesh, training_config=SimpleNamespace(micro_batch_size=1, global_batch_size=1, seed=1234),
                max_seq_len=transform.max_seq_len,
            )
            get_batch = loader_config.get_batch.build(mesh_context=mesh, device="cpu")
            batches = list(loaders[0])
            self.assertEqual(len(batches), 4)
            model_inputs, loss_inputs = get_batch(None, external_batch=batches[0])
            self.assertEqual(model_inputs["input_ids"].tolist(), [[1, 2, 3, 4, 5]])
            self.assertEqual(loss_inputs["labels"].tolist(), [[-100, -100, -100, 4, 5]])

    def test_vlm_trainer_default_and_opt_in_adapter_binding(self):
        """The default batch path is Omni; explicit runtime adapters are built."""
        trainer = VLMTrainer.__new__(VLMTrainer)
        mesh = SimpleNamespace()
        trainer.base = SimpleNamespace(
            config=SimpleNamespace(dataloader=SimpleNamespace(get_batch=None, pp_shared_data=False)),
            mesh=mesh, device="cpu",
        )
        with patch("hyper_parallel.trainer.vlm_trainer.OmniParallelBatch") as default_builder:
            trainer._build_get_batch()
            default_builder.assert_called_once_with(mesh_context=mesh, device="cpu", pp_shared_data=False)
            self.assertIs(trainer.base.get_batch, default_builder.return_value)

        builder = Mock(runtime_input_adapter={"_target_": "tests.runtime.Adapter"})
        trainer.base.config.dataloader.get_batch = builder
        with patch("hyper_parallel.trainer.vlm_trainer.resolve_component") as resolve:
            trainer._build_get_batch()
            resolve.assert_called_once()
            builder.build.assert_called_once_with(
                mesh_context=mesh, device="cpu", pp_shared_data=False,
                runtime_input_adapter=resolve.return_value.build.return_value,
            )

    def test_cp_adapter_contract_and_collision_validation(self):
        """CP requires an adapter; it receives global media metadata and local tokens."""
        context = DataLoaderParallelContext(cp_world_size=2, cp_rank=1, distributed_enabled=True)
        adapter = Mock()
        adapter.build.return_value = {"packed_metadata": "model-owned"}
        with patch("hyper_parallel.data.omni.get_batch.create_dataloader_parallel_context", return_value=context):
            with self.assertRaisesRegex(ValueError, "runtime adapter"):
                OmniParallelBatch(mesh_context=SimpleNamespace(cp_size=2), device="cpu")
            get_batch = OmniParallelBatch(
                mesh_context=SimpleNamespace(cp_size=2), device="cpu", runtime_input_adapter=adapter,
            )
        batch = {"input_ids": torch.tensor([[1, 2, 3, 4]]), "labels": torch.tensor([[-100, 2, 3, 4]]),
                 "pixel_values": torch.ones(3, 2), "cu_seq_lens": torch.tensor([0, 4])}
        model_inputs, _ = get_batch(None, external_batch=batch)
        self.assertEqual(model_inputs["input_ids"].tolist(), [[3, 4]])
        self.assertEqual(model_inputs["packed_metadata"], "model-owned")
        received = adapter.build.call_args.kwargs["batch"]
        self.assertEqual(received["cu_seq_lens"].tolist(), [0, 4])
        torch.testing.assert_close(received["pixel_values"], batch["pixel_values"], rtol=0, atol=0)
        adapter.build.return_value = {"input_ids": torch.tensor([[99]])}
        with self.assertRaisesRegex(ValueError, "overwrite"):
            get_batch(None, external_batch=batch)
        adapter.build.return_value = None
        with self.assertRaisesRegex(TypeError, "mapping"):
            get_batch(None, external_batch=batch)
