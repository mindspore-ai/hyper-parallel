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
"""Torchrun entry point; configuration selects the concrete RL mode."""
import argparse
import logging

from rl.config.loader import load_config


def main() -> None:
    """Load one RL configuration and run its declared training mode."""
    parser = argparse.ArgumentParser(description="Hyper-Parallel RL trainer")
    parser.add_argument("config", help="Path to the RL YAML configuration")
    args, overrides = parser.parse_known_args()
    # Keep configuration loading and --help independent of training dependencies.
    from rl.trainer.build import build_trainer

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    trainer = build_trainer(load_config(args.config, overrides))
    trainer.train()


if __name__ == "__main__":
    main()
