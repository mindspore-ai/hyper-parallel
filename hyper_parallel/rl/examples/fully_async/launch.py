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
"""Launch fixed-topology fully asynchronous GRPO under torchrun."""
import argparse
import logging

from rl.trainer.fully_async import FullyAsyncTrainer
from rl.config.loader import load_config


def main() -> None:
    """Parse one authoritative configuration and run the asynchronous trainer."""
    parser = argparse.ArgumentParser(description="Fully asynchronous Hyper-RL")
    parser.add_argument("config")
    arguments, overrides = parser.parse_known_args()
    logging.basicConfig(level=logging.INFO)
    FullyAsyncTrainer(load_config(arguments.config, overrides)).train()


if __name__ == "__main__":
    main()
