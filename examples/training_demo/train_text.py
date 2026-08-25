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
"""Run text training from a resolved Trainer YAML configuration."""

from hyper_parallel.auto_models.config.manager import parse_training_args
from hyper_parallel.auto_models.trainer.config import TrainerConfig
from hyper_parallel.auto_models.trainer.text_trainer import TextTrainer


def main() -> None:
    """Build Trainer components and execute the configured training loop."""
    config: TrainerConfig = parse_training_args()
    trainer = TextTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
