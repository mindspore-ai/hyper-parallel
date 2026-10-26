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
"""Fallback specs must never re-declare override-owned parameters.

DeepSeek-V4.1's official YAML declares ``engram.wkv.weight`` /
``engram.embed.weight`` on the OUTER ``model.layers.*.engram`` boundary.
The all-Replicate leftover fallback (qwen-round F3, 0cfe9542) builds specs
for the scattered ``engram.wkv`` / ``engram.embed`` groups and re-declared
those same parameters — HP-PLAN-003 (D-14 param uniqueness) then rejects
the whole plan, a regression of the official configuration.  The fallback
must yield: parameters an override already declares stay out of the
fallback spec, and a fallback left with no parameters is dropped.
"""

import unittest

from torch import nn

from hyper_parallel.distributed._builder.planner import (
    ShardingPlanner,
    _override_declared_params,
)
from hyper_parallel.distributed.plan import ModuleShardingSpec


def _engram_style_model() -> nn.Module:
    """Minimal model with the V4.1 engram scattering shape.

    Layers carry an ``engram`` whose params (q_weight/k_weight directly,
    wkv.weight / embed.weight on nested submodules) are all REPLICATED by
    family rules — the exact shape that scatters to root and back.
    """

    class Engram(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_weight = nn.Parameter(torch_zeros(4))
            self.k_weight = nn.Parameter(torch_zeros(4))
            self.wkv = nn.Linear(4, 4, bias=False)
            self.embed = nn.Linear(4, 4, bias=False)

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.engram = Engram()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer(), Layer()])

    class Wrapper(nn.Module):
        # official HF-style root: parameters are addressed as model.layers.*
        # (named_parameters of Model() alone would yield bare layers.* FQNs,
        #  which no `model.layers.*` override key can ever match)
        def __init__(self):
            super().__init__()
            self.model = Model()

    model = Wrapper()
    return model


def torch_zeros(n):
    import torch

    return torch.zeros(n)


class TestFallbackYieldsToOverrides(unittest.TestCase):
    """The pure-Replicate fallback never collides with user overrides."""

    def _override(self):
        # The official V4.1 entry: outer boundary declares nested params.
        return {
            "model.layers.*.engram": ModuleShardingSpec(
                params={
                    "q_weight": {},
                    "k_weight": {},
                    "wkv.weight": {},
                    "embed.weight": {},
                },
            ),
        }

    def test_override_declared_params_resolves_glob(self):
        model = _engram_style_model()
        declared = _override_declared_params(self._override(), model)
        all_declared = [fqn for fqns in declared.values() for fqn in fqns]
        self.assertIn("model.layers.0.engram.wkv.weight", all_declared)
        self.assertIn("model.layers.1.engram.embed.weight", all_declared)
        self.assertIn("model.layers.0.engram.q_weight", all_declared)

    def test_fallback_skips_override_owned_params(self):
        from hyper_parallel.distributed.tensor_parallel.param_role import (
            ParamRole,
        )
        model = _engram_style_model()
        planner = ShardingPlanner(plan_overrides=self._override())
        parameters_by_name = dict(model.named_parameters())
        declared = _override_declared_params(planner._plan_overrides, model)

        group = [("model.layers.0.engram.wkv.weight", ParamRole.REPLICATED)]
        spec = planner._build_replicated_fallback_spec(
            "model.layers.0.engram.wkv",
            group,
            model,
            parameters_by_name,
            override_declared=declared,
        )
        # wkv.weight is owned by the outer override -> no fallback at all
        self.assertIsNone(spec)

    def test_fallback_keeps_unowned_params(self):
        from hyper_parallel.distributed.tensor_parallel.param_role import (
            ParamRole,
        )
        model = _engram_style_model()
        # a replicated leftover the override does NOT declare (added BEFORE
        # the declared-map is computed, mirroring the real plan flow)
        model.model.layers[0].some_leftover = nn.Linear(4, 4, bias=False)
        planner = ShardingPlanner(plan_overrides=self._override())
        parameters_by_name = dict(model.named_parameters())
        declared = _override_declared_params(planner._plan_overrides, model)
        spec = planner._build_replicated_fallback_spec(
            "model.layers.0.some_leftover",
            [("model.layers.0.some_leftover.weight", ParamRole.REPLICATED)],
            model,
            parameters_by_name,
            override_declared=declared,
        )
        self.assertIsNotNone(spec)
        self.assertIn("weight", spec.params)


if __name__ == "__main__":
    unittest.main()
