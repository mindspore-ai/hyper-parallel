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
"""Tests for DeepSeek-V4.1 multimodal sequence alignment."""

import unittest

from hyper_parallel.models.deepseek_v41.dataset_preprocessing.processor import (
    DeepseekV41Processor,
)
from hyper_parallel.models.deepseek_v41.dataset_preprocessing.transform_fn import (
    DeepseekV41PretrainTransform,
    build_deepseek_v41_omni_transform,
)
from hyper_parallel.models.deepseek_v41.deepseek_v41_spec import (
    build_parameter_sharding_rules,
)
from hyper_parallel.models.deepseek_v41.validation.model_validation_spec import (
    get_validation_spec,
)


class TestDeepseekV41OmniTransform(unittest.TestCase):
    """Validate topology-neutral packed sequence alignment."""

    @staticmethod

    def _processor() -> DeepseekV41Processor:
        """Build the smallest processor contract needed by the transform."""
        return DeepseekV41Processor(
            tokenizer=object(),
            vision_n_layers=0,
            vision_patch_size=14,
            vision_max_wh_ratio=None,
            vision_min_pixels=1,
            vision_downsample_ratio=1,
            vision_max_n_token=1,
            image_token_id=3,
        )


    def test_builder_retains_cross_topology_alignment(self) -> None:
        """Keep the configured packing alignment in the resolved transform."""
        transform = build_deepseek_v41_omni_transform(
            processor=self._processor(),
            training_mode="pretrain",
            sample_alignment_size=4,
        )

        self.assertEqual(transform.sample_alignment_size, 4)

    def test_constructor_rejects_non_positive_alignment(self) -> None:
        """Reject values that violate the positive packing alignment contract."""
        for sample_alignment_size in (0, -2):
            with self.subTest(sample_alignment_size=sample_alignment_size):
                with self.assertRaisesRegex(ValueError, "sample_alignment_size must be a positive integer"):
                    DeepseekV41PretrainTransform(
                        self._processor(),
                        sample_alignment_size=sample_alignment_size,
                    )

    def test_validation_replicates_complete_image_fields_across_cp(self) -> None:
        """Deduplicate complete image tensors while retaining sharded token types."""
        replicated_fields = set(
            get_validation_spec().data.cp_replicated_forward_fields
        )
        expected_fields = {
            "pixel_values",
            "image_patch_offsets",
            "image_vit_grid_hw",
            "image_llm_grid_hw",
            "image_batch_indices",
            "image_token_starts",
        }

        self.assertTrue(expected_fields.issubset(replicated_fields))
        self.assertNotIn("token_types", replicated_fields)

    def test_sharding_policy_keeps_aligner_replicated(self) -> None:
        """Keep both visual aligner linears out of CP attention inference."""
        rules = build_parameter_sharding_rules()
        matching_roles = [
            role
            for patterns, role in rules
            if isinstance(patterns, list)
            and {"aligner.w1", "aligner.w2"}.issubset(patterns)
        ]

        self.assertEqual(len(matching_roles), 1)
        self.assertEqual(matching_roles[0].name, "REPLICATED")


if __name__ == "__main__":
    unittest.main()
