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
"""VL training entry point."""
from hyper_parallel.trainer.utils import init_logger

init_logger()

# pylint: disable=wrong-import-position  # init_logger() must run first
from hyper_parallel.trainer.config import HyperTrainerConfig, parse_args
from hyper_parallel.trainer.utils.discovery import discover_model_spec
from hyper_parallel.trainer.vl_trainer import VLTrainer

if __name__ == "__main__":
    args = parse_args(HyperTrainerConfig)
    discover_model_spec(args.model.name)
    trainer = VLTrainer(args)
    trainer.train()
