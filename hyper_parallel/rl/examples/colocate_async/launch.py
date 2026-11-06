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
"""Launch colocated asynchronous GRPO without changing existing trainer entries."""
import argparse
import logging

from rl.config.loader import load_config
from rl.trainer.colocate_async import ColocateAsyncTrainer


def main() -> None:
    """Load one authoritative recipe and run its colocated trainer under torchrun."""
    parser = argparse.ArgumentParser(description="Colocated asynchronous Hyper-RL training")
    parser.add_argument("config")
    arguments, overrides = parser.parse_known_args()
    logging.basicConfig(level=logging.INFO)
    ColocateAsyncTrainer(load_config(arguments.config, overrides)).train()


if __name__ == "__main__":
    main()
