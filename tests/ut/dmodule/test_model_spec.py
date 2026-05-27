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
"""Unit tests for :mod:`hyper_parallel.dmodule.model_spec`."""

import unittest
from dataclasses import dataclass

from tests.ut.dmodule._ensure_torch_dmodule import ensure_torch_platform_for_dmodule

ensure_torch_platform_for_dmodule()

import torch
from torch import nn

from hyper_parallel.dmodule.model import BaseModel
from hyper_parallel.dmodule.model_spec import ModelSpec
from hyper_parallel.dmodule.module import Module


class FakeModel(BaseModel):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        hidden: int = 8

        def update_from_config(self, *, trainer_config, **kwargs) -> None:
            del trainer_config, kwargs

        def get_nparams_and_flops(self, model: Module, seq_len: int) -> tuple[int, int]:
            del model, seq_len
            return 0, 0

    def __init__(self, config: "FakeModel.Config"):
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.empty(config.hidden, config.hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T


class TestModelSpec(unittest.TestCase):
    """ModelSpec v2 validation and construction."""

    def test_model_spec_with_model_config(self):
        model_cfg = FakeModel.Config(hidden=16)
        spec = ModelSpec(
            name="fake",
            model=model_cfg,
            parallelize_fn=lambda *args, **kwargs: None,
        )
        self.assertEqual(spec.name, "fake")
        self.assertIs(spec.model, model_cfg)
        self.assertIsNone(spec.build_model_fn)

    def test_model_spec_with_build_model_fn(self):
        def build_fn(cfg):
            del cfg
            return FakeModel(FakeModel.Config())

        spec = ModelSpec(name="legacy", build_model_fn=build_fn)
        self.assertIsNotNone(spec.build_model_fn)
        self.assertIsNone(spec.model)

    def test_model_spec_requires_one_constructor(self):
        with self.assertRaises(ValueError) as ctx:
            ModelSpec(name="bad")
        self.assertIn("requires build_model_fn or model", str(ctx.exception))

    def test_model_spec_rejects_both_constructors(self):
        with self.assertRaises(ValueError) as ctx:
            ModelSpec(
                name="bad",
                build_model_fn=lambda c: None,
                model=FakeModel.Config(),
            )
        self.assertIn("must not set both", str(ctx.exception))

    def test_model_config_build(self):
        model = FakeModel.Config(hidden=4).build()
        self.assertIsInstance(model, FakeModel)
        self.assertEqual(model.config.hidden, 4)


if __name__ == "__main__":
    unittest.main()
