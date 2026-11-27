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
"""Launch distributed CPU checks for the eager text Trainer scheduler."""
from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_deepseek_trainer_pipeline():
    """Feature: Eager Trainer pipeline parallelism.
    Description: Run both schedules across two Gloo ranks with different valid-token counts.
    Expectation: Every owned gradient matches the single-stage reference before and after clipping.
    """
    worker = Path(__file__).with_name("trainer_pipeline_worker.py")
    torchrun_case(str(worker), "test_pipeline_gradients", num_proc=2)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_deepseek_trainer_pipeline_hsdp():
    """Feature: Eager Trainer pipeline with real HSDP stage roots.
    Description: Wrap each partitioned stage with fully_shard and run 1F1B over two Gloo ranks.
    Expectation: HSDP stage initialization succeeds and losses and gradients match the unsplit model.
    """
    worker = Path(__file__).with_name("trainer_pipeline_worker.py")
    torchrun_case(str(worker), "test_pipeline_hsdp_gradients", num_proc=2)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_deepseek_trainer_mpipe_vision():
    """Feature: MPipe visual transpose with eager Trainer PP.
    Description: Run the MPipe schedule across two Gloo ranks with frozen and trainable vision weights.
    Expectation: Visual injection, decoder losses and all gradients match the unsplit VLM.
    """
    worker = Path(__file__).with_name("trainer_pipeline_worker.py")
    torchrun_case(str(worker), "test_mpipe_vision_gradients", num_proc=2)
