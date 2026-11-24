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
"""Lightweight launchers for CPU/Gloo Encoder Parallel system tests."""

import os
from pathlib import Path
from unittest.mock import patch

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark

_HERE = Path(__file__).resolve().parent


@arg_mark(plat_marks=["cpu_linux"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_odd_patch_ulysses():
    """Feature: two-rank padded Ulysses.

    Description: Launch the worker with five valid image patches.
    Expectation: Forward and backward match full attention.
    """
    torchrun_case(str(_HERE / "_test_ulysses.py"), "test_odd_patch_ulysses", num_proc=2)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_e0_runtime():
    """Feature: eight-rank E0 visual execution.

    Description: Launch packed ViT, aligner, dummy bucket, and feature exchange.
    Expectation: Exact image owners and present visual gradients.
    """
    torchrun_case(str(_HERE / "_test_encoder_runtime.py"), "test_e0_runtime", num_proc=8)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_tp_gather():
    """Feature: CP1/NOSP TP variable gather.

    Description: Launch four producers and consumers on Gloo.
    Expectation: Each rank receives full records and gradients reach producers.
    """
    torchrun_case(str(_HERE / "_test_exchange.py"), "test_tp_gather", num_proc=4)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_tp_a2a():
    """Feature: CP1/SP TP variable A2A.

    Description: Route E0 image spans across four TP token intervals.
    Expectation: Only intersecting owners receive records and gradients return.
    """
    torchrun_case(str(_HERE / "_test_exchange.py"), "test_tp_a2a", num_proc=4)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parent_nosp():
    """Feature: CP2/NOSP parent variable A2A.

    Description: Route E0 images across the eight-rank decoder parent.
    Expectation: Each CP owner receives a full record on every TP rank.
    """
    torchrun_case(str(_HERE / "_test_exchange.py"), "test_parent_nosp", num_proc=8)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_cross_decoder_domain_precision():
    """Feature: Cross-parent encoder precision.

    Description: Train a real visual tower/projector for ten Muon updates with Encoder-CP4 and decoder CP2.
    Expectation: Initial tensor parity and ten-step loss/norm pass; tensor update probes remain visible.
    """
    with patch.dict(os.environ, {"HP_ENCODER_BACKEND": "gloo", "HP_ENCODER_LLM_CP": "2",
                                 "HP_ENCODER_LLM_TP": "1", "HP_ENCODER_CP": "4",
                                 "HP_ENCODER_DOMAIN": "auto", "HP_ENCODER_FAMILY": "deepseek"}):
        torchrun_case(str(_HERE / "_test_encoder_domains.py"), "test_encoder_domain_precision", num_proc=4)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_global_encoder_domain_precision():
    """Feature: Explicit pooling despite a locally divisible encoder degree.

    Description: Train the HF axial visual source with global pooling, decoder CP1/TP2 and Encoder-CP2.
    Expectation: Initial parity and ten-step loss/norm pass; strict update probes are reported separately.
    """
    with patch.dict(os.environ, {"HP_ENCODER_BACKEND": "gloo", "HP_ENCODER_LLM_CP": "1",
                                 "HP_ENCODER_LLM_TP": "2", "HP_ENCODER_CP": "2",
                                 "HP_ENCODER_DOMAIN": "global", "HP_ENCODER_FAMILY": "qwen"}):
        torchrun_case(str(_HERE / "_test_encoder_domains.py"), "test_encoder_domain_precision", num_proc=4)
