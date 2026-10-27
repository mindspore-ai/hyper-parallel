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
"""Configuration, partitioning and forward contracts for Transformer PP/VPP."""

import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from hyper_parallel.core.pipeline_parallel import Schedule1F1B, ScheduleGPipe, ScheduleInterleaved1F1B
from hyper_parallel.models._transformers.pipeline import (
    AutoPipeline,
    PipelineConfig,
    _prune_model,
    _stage_assignments,
)


class _FakePPMesh:
    """One-dimensional mesh fake; no process groups are initialized."""

    mesh_dim_names = ("pp",)

    def __init__(self, rank: int, size: int = 2):
        self._rank = rank
        self.rank_list = tuple(range(size))

    def size(self):
        """Return the physical PP degree."""
        return len(self.rank_list)

    def get_local_rank(self):
        """Return the fake physical rank."""
        return self._rank

    def get_group(self):
        """No distributed group is needed during ordinary schedule construction."""
        return None


def _tiny_qwen3(layers: int = 4):
    """Build a CPU model with the original HF module names."""
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=layers,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=16,
        use_cache=False,
        _attn_implementation="eager",
    )
    return Qwen3ForCausalLM(config)


class TestPipelineConfig(unittest.TestCase):
    """Validate user-facing schedule and layer-count options."""

    def test_defaults_and_independent_split_lists(self):
        """PP is opt-in and P2P overlap is enabled by default."""
        config = PipelineConfig()
        self.assertFalse(config.enabled)
        self.assertTrue(config.p2p_overlap)
        self.assertFalse(config.p2p_multi_stream)
        config.split.append(1)
        self.assertEqual(PipelineConfig().split, [])

    def test_invalid_configurations(self):
        """Reject malformed counts and unsupported option combinations."""
        cases = [
            ({"pp_microbatch_size": 0}, "positive integer"),
            ({"pp_microbatch_size": True}, "positive integer"),
            ({"schedule": "unknown"}, "schedule"),
            ({"split": "2,2"}, "must be a list"),
            ({"split": [[], []]}, "non-empty"),
            ({"split": [[1, 1], 2]}, "non-empty"),
            ({"split": [1, [1]]}, "positive integers"),
            ({"split": [0, 2]}, "positive integers"),
            ({"split": [-1, 2]}, "positive integers"),
            ({"split": [1.5, 2]}, "positive integers"),
            ({"split": [True, 2]}, "positive integers"),
            ({"split": [[1, 1], [1, 1]], "schedule": "1f1b"}, "requires schedule"),
            ({"split": [[1, 1], [1, 1]], "schedule": "gpipe"}, "requires schedule"),
            ({"schedule": "interleaved1f1b"}, "requires a nested"),
            ({"split": [[1, 1]], "schedule": "interleaved1f1b"}, "at least two"),
            ({"split": [1, 1], "schedule": "interleaved1f1b"}, "requires a nested"),
            ({"p2p_multi_stream": True, "p2p_overlap": False}, "requires pipeline.p2p_overlap"),
        ]
        for values, message in cases:
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, message):
                PipelineConfig(**values)


class TestAutoPipeline(unittest.TestCase):
    """Exercise real model pruning and schedule construction on CPU."""

    def test_split_preserves_original_layer_fqns(self):
        """Only local layers remain and their original numeric keys survive."""
        model = _tiny_qwen3()
        assignments = _stage_assignments(model, 2, PipelineConfig(split=[2, 2]))
        self.assertEqual(assignments[0][0], "model.layers.0")
        self.assertEqual(assignments[1][0], "model.layers.2")
        self.assertIn("model.embed_tokens", assignments[0])
        self.assertIn("model.norm", assignments[1])
        self.assertIn("lm_head", assignments[1])
        stage = _prune_model(model, assignments[1])
        self.assertEqual(list(stage.model.layers), ["2", "3"])
        self.assertIsNone(stage.model.embed_tokens)
        self.assertIsNotNone(stage.model.norm)
        self.assertIsNotNone(stage.lm_head)
        self.assertIn("model.layers.2.self_attn.q_proj.weight", dict(stage.named_parameters()))

    def test_ten_layer_pp_and_vpp_assignments(self):
        """The agreed split examples assign layers in chunk-major order."""
        model = _tiny_qwen3(10)
        cases = [
            ([2, 3, 3, 2], [[0, 1], [2, 3, 4], [5, 6, 7], [8, 9]]),
            ([[1, 1, 1, 1], [1, 2, 2, 1]], [[0], [1], [2], [3], [4], [5, 6], [7, 8], [9]]),
            ([], [[0, 1, 2], [3, 4, 5], [6, 7], [8, 9]]),
        ]
        for split, expected in cases:
            with self.subTest(split=split):
                schedule = "interleaved1f1b" if split and isinstance(split[0], list) else "1f1b"
                assignments = _stage_assignments(
                    model, 4, PipelineConfig(schedule=schedule, split=split)
                )
                actual = [
                    [int(name.rsplit(".", 1)[1]) for name in part if name.startswith("model.layers.")]
                    for part in assignments
                ]
                self.assertEqual(actual, expected, f"split={split}: expected={expected}, actual={actual}")

    def test_partition_validation(self):
        """Reject the wrong PP width, layer total, and empty automatic stages."""
        model = _tiny_qwen3()
        for split, pp_size in [([4], 2), ([[1, 1], [2]], 2), ([1, 1], 2), ([], 5)]:
            with self.subTest(split=split, pp_size=pp_size), self.assertRaisesRegex(ValueError, "pipeline.split"):
                schedule = "interleaved1f1b" if split and isinstance(split[0], list) else "1f1b"
                _stage_assignments(model, pp_size, PipelineConfig(schedule=schedule, split=split))

    def test_schedule_selection_and_p2p_options(self):
        """All schedules receive overlap and transport flags, without collectives."""
        model = _tiny_qwen3()
        for name, split, expected in [
            ("gpipe", [2, 2], ScheduleGPipe),
            ("1f1b", [2, 2], Schedule1F1B),
            ("interleaved1f1b", [[1, 1], [1, 1]], ScheduleInterleaved1F1B),
        ]:
            for overlap, multi_stream in [(False, False), (True, False), (True, True)]:
                with self.subTest(name=name, split=split, overlap=overlap, multi_stream=multi_stream):
                    config = PipelineConfig(
                        enabled=True, schedule=name, split=split,
                        p2p_overlap=overlap, p2p_multi_stream=multi_stream,
                    )
                    with patch(
                        "hyper_parallel.core.pipeline_parallel.scheduler.create_p2p_multi_stream_groups",
                        return_value={},
                    ):
                        pipeline = AutoPipeline(_FakePPMesh(0), config).build(model, batch_size=4)
                    schedule = pipeline.info.schedule
                    self.assertIsInstance(schedule, expected)
                    self.assertEqual(schedule.micro_batch_num, 4)
                    self.assertEqual(schedule._overlap_p2p, overlap)  # pylint: disable=protected-access
                    self.assertEqual(
                        schedule._p2p_mode,  # pylint: disable=protected-access
                        "multi_stream" if multi_stream else "plain",
                    )

    def test_disabled_and_invalid_batch_sizes(self):
        """Batch validation runs before distributed stage construction."""
        model = _tiny_qwen3()
        with self.assertRaisesRegex(ValueError, "enabled=true"):
            AutoPipeline(_FakePPMesh(0), PipelineConfig()).build(model)
        config = PipelineConfig(enabled=True, pp_microbatch_size=2)
        for batch_size in (0, -2, 3):
            with self.subTest(batch_size=batch_size), self.assertRaisesRegex(ValueError, "DP-local batch"):
                AutoPipeline(_FakePPMesh(0), config).build(model, batch_size=batch_size)

    def test_meta_build_uses_current_accelerator_before_materialization(self):
        """PP communication never derives its device from meta stage parameters."""
        with torch.device("meta"):
            model = _tiny_qwen3()
        for schedule, split in [("1f1b", [2, 2]),
                                ("interleaved1f1b", [[1, 1], [1, 1]])]:
            with self.subTest(schedule=schedule):
                config = PipelineConfig(enabled=True, schedule=schedule, split=split)
                pipeline = AutoPipeline(_FakePPMesh(0), config)
                seen_meta = []

                def parallelize(part):
                    seen_meta.append(all(parameter.is_meta for parameter in part.parameters()))
                    return part

                with patch("torch.accelerator.current_accelerator", return_value=torch.device("cpu")):
                    pipeline.build(model, parallelize_fn=parallelize, batch_size=4)
                self.assertTrue(all(seen_meta))
                self.assertTrue(all(parameter.is_meta for parameter in pipeline.parameters()))
                for stage in pipeline.info.stages:
                    self.assertEqual(stage.device, torch.device("cpu"))
                pipeline.to_empty(device="cpu")
                self.assertTrue(all(not parameter.is_meta for parameter in pipeline.parameters()))
                for part, stage in zip(pipeline.parts, pipeline.info.stages):
                    self.assertIs(stage.submodule, part)

    def test_vpp_local_chunks_parallelization_and_checkpoint_aliases(self):
        """Each local chunk is transformed independently and retains checkpoint aliases."""
        model = _tiny_qwen3(10)
        config = PipelineConfig(
            enabled=True,
            schedule="interleaved1f1b",
            split=[[1, 1, 1, 1], [1, 2, 2, 1]],
        )
        for rank, expected_layers in [(0, [["0"], ["4"]]), (1, [["1"], ["5", "6"]]),
                                      (2, [["2"], ["7", "8"]]), (3, [["3"], ["9"]])]:
            with self.subTest(rank=rank):
                calls = []

                def parallelize(stage):
                    calls.append(stage)
                    return stage

                pipeline = AutoPipeline(_FakePPMesh(rank, 4), config).build(
                    model, parallelize_fn=parallelize, batch_size=4,
                )
                self.assertEqual(len(calls), 2)
                self.assertEqual([list(part.model.layers) for part in pipeline.parts], expected_layers)
                self.assertEqual([stage.stage_index for stage in pipeline.info.stages], [rank, rank + 4])
                self.assertEqual(pipeline.info.has_first_stage, rank == 0)
                self.assertEqual(pipeline.info.has_last_stage, rank == 3)
                for index, part in enumerate(pipeline.parts):
                    self.assertIs(calls[index], part)
                    original = f"model.layers.{expected_layers[index][0]}.self_attn.q_proj.weight"
                    self.assertEqual(pipeline.checkpoint_fqn_aliases[original], f"parts.{index}.{original}")

    def test_pp_and_vpp_forward_and_gradients_match_full_model(self):
        """Patched stages preserve full-model outputs, local kwargs and parameter gradients."""
        torch.manual_seed(42)
        model = _tiny_qwen3().eval()
        input_ids = torch.randint(0, 32, (2, 6))
        attention_mask = torch.ones_like(input_ids)
        attention_mask[0, -1] = 0
        logits = model(input_ids, attention_mask=attention_mask).logits
        reference_loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 32), input_ids.reshape(-1))
        reference_loss.backward()
        for split in ([2, 2], [[1, 1], [1, 1]]):
            with self.subTest(split=split):
                schedule = "interleaved1f1b" if isinstance(split[0], list) else "1f1b"
                config = PipelineConfig(enabled=True, schedule=schedule, split=split)
                pipelines = [AutoPipeline(_FakePPMesh(rank), config).build(model) for rank in range(2)]
                stages = sorted((stage for pipe in pipelines for stage in pipe.info.stages),
                                key=lambda stage: stage.stage_index)
                hidden = input_ids
                for stage in stages:
                    hidden = stage.submodule(hidden, attention_mask=attention_mask, labels=input_ids)
                torch.testing.assert_close(hidden, reference_loss, rtol=1e-5, atol=1e-6)
                hidden.backward()
                reference_params = dict(model.named_parameters())
                for stage in stages:
                    for name, param in stage.submodule.named_parameters():
                        torch.testing.assert_close(param.grad, reference_params[name].grad, rtol=1e-4, atol=1e-6)
