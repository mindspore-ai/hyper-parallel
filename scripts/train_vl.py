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
"""Vision-language training entry point for the AutoModel workflow."""

from hyper_parallel.codegen.train_sidecar import run_training_with_codegen
from hyper_parallel.trainer.config.parser import parse_training_args
from hyper_parallel.trainer.vlm_trainer import VLMTrainer


def main() -> None:
    """Resolve configured components and execute vision-language training."""
    run_training_with_codegen(parse_training_args, VLMTrainer)


if __name__ == "__main__":
    main()
