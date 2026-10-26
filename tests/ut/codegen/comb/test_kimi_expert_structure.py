# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may not obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""EP structure detection must recognize the w1/w3/w2 expert spelling.

Kimi's ``KimiBlockSparseMLP`` experts (``KimiSparseMoeBlock`` in the kimi_k3
/ kimi_linear families) name their SwiGLU projections ``w1/w3/w2`` — the
same three-matrix layout the runtime weight resolver
(``resolve_swiglu_weights``) already accepts.  The plan-time structure
detector previously only recognized ``gate_proj/up_proj/down_proj`` and
rejected genuinely SwiGLU expert lists with a factually wrong message.
"""

import unittest

from torch import nn

from hyper_parallel.distributed.expert_parallel.structure import (
    UnsupportedModuleStructure,
    _expert_projection_names,
    detect_moe_structure,
)


def _kimi_style_expert() -> nn.Module:
    """A minimal expert with the DeepSeek/Kimi w1/w3/w2 spelling."""

    class Expert(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = nn.Linear(8, 16, bias=False)  # gate
            self.w3 = nn.Linear(8, 16, bias=False)  # up
            self.w2 = nn.Linear(16, 8, bias=False)  # down

        def forward(self, x):
            return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))

    return Expert()


class _KimiStyleMoe(nn.Module):
    """ModuleList experts + a gate returning the kimi 2-tuple contract."""

    def __init__(self, num_experts: int = 4):
        super().__init__()
        self.experts = nn.ModuleList([_kimi_style_expert() for _ in range(num_experts)])
        self.gate = nn.Linear(8, num_experts, bias=False)
        self.top_k = 2


class TestKimiExpertListStructure(unittest.TestCase):
    """w1/w3/w2 experts must be detected as SwiGLU, not misdiagnosed."""

    def test_projection_names_recognize_both_spellings(self):
        self.assertEqual(
            _expert_projection_names(_kimi_style_expert()),
            ("w1", "w3", "w2"),
        )
        hf_style = type("HfExpert", (nn.Module,), {})()
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(hf_style, name, nn.Linear(8, 8, bias=False))
        self.assertEqual(
            _expert_projection_names(hf_style),
            ("gate_proj", "up_proj", "down_proj"),
        )

    def test_dense_module_list_experts_detected(self):
        """A w1/w3/w2 ModuleList with a plain linear top-k gate freezes to
        module_list storage (previously rejected as "not a SwiGLU list")."""
        moe = _KimiStyleMoe()
        moe.gate = nn.Linear(8, 4, bias=False)
        moe.top_k = 2
        structure = detect_moe_structure(moe)
        self.assertEqual(structure.expert_storage, "module_list")
        self.assertEqual(structure.router, "softmax_topk")

    def test_non_swiglu_expert_list_message_is_accurate(self):
        """A genuinely unsupported expert still fails, with both spellings named."""

        class NotSwiglu(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(8, 8)
                self.fc2 = nn.Linear(8, 8)

        moe = _KimiStyleMoe()
        moe.experts = nn.ModuleList([NotSwiglu() for _ in range(2)])
        with self.assertRaisesRegex(
            UnsupportedModuleStructure,
            r"w1/w3/w2",
        ):
            detect_moe_structure(moe)


if __name__ == "__main__":
    unittest.main()
