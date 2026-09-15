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
"""Tests for the Transformers-compatible AutoModel entry points."""

import os
from types import SimpleNamespace
import unittest
from unittest import mock

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

from hyper_parallel.models._transformers import auto_model  # pylint: disable=wrong-import-position


class TestHyperAutoModelFromConfig(unittest.TestCase):
    """Tests for ``HyperAutoModel.from_config`` input normalization."""

    def test_string_config_is_resolved_before_build(self):
        """A model identifier is resolved to a config before model construction."""
        resolved_config = SimpleNamespace(architectures=["FakeForCausalLM"])
        distributed_setup = SimpleNamespace(mesh_context=object())
        expected_model = object()

        with (
            mock.patch.object(auto_model, "DistributedSetup", return_value=distributed_setup),
            mock.patch.object(auto_model, "_current_device", return_value="cpu"),
            mock.patch.object(
                auto_model,
                "instantiate_infrastructure",
                return_value=("planner", "fsdp2_manager"),
            ),
            mock.patch.object(auto_model, "get_hf_config", return_value=resolved_config) as get_hf_config,
            mock.patch.object(auto_model, "get_is_hf_model", return_value=False) as get_is_hf_model,
            mock.patch.object(
                auto_model.HyperAutoModelForCausalLM,
                "_build_model",
                return_value=expected_model,
            ) as build_model,
        ):
            model = auto_model.HyperAutoModelForCausalLM.from_config(
                "org/model",
                torch_dtype="float16",
                attn_implementation="flash_attention_2",
                revision="main",
            )

        self.assertIs(model, expected_model)
        get_hf_config.assert_called_once_with(
            "org/model", "flash_attention_2", "float16", revision="main"
        )
        get_is_hf_model.assert_called_once_with(resolved_config, force_hf=False)
        _, build_kwargs = build_model.call_args
        self.assertIs(build_kwargs["hf_config"], resolved_config)
        self.assertFalse(build_kwargs["load_base_model"])
        self.assertEqual(build_kwargs["revision"], "main")


if __name__ == "__main__":
    unittest.main()
