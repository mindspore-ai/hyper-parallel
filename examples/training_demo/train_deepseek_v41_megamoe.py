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
"""Run DSV4.1 MegaMoe text training with ordered native-resource shutdown."""

from hyper_parallel.models.deepseek_v41.adapter.megamoe_training import DeepseekV41TrainingExperts
from hyper_parallel.trainer.config import TrainerConfig
from hyper_parallel.trainer.config.manager import parse_training_args
from hyper_parallel.trainer.text_trainer import TextTrainer


class MegaMoeTextTrainer(TextTrainer):
    """Release SHMEM while the Trainer's EP process groups are still alive."""

    def __init__(self, config: TrainerConfig, *, share_workspace: bool = True) -> None:
        """Build the Trainer and bind compatible serial layers to one workspace.

        Args:
            config: Resolved text-training recipe.
            share_workspace: Share transient native storage before the first forward.
        """
        super().__init__(config)
        if share_workspace:
            experts = [module for module in self.base.model.modules()
                       if isinstance(module, DeepseekV41TrainingExperts)]
            DeepseekV41TrainingExperts.share_execution_resources(experts)

    def on_train_end(self) -> None:
        """Finish callbacks, then close all expert executors in model order."""
        super().on_train_end()
        for module in tuple(self.base.model.modules()):
            if isinstance(module, DeepseekV41TrainingExperts):
                module.close()


def main() -> None:
    """Run the explicit MegaMoe recipe with the standard text training loop."""
    trainer = MegaMoeTextTrainer(parse_training_args())
    trainer.train()


if __name__ == "__main__":
    main()
