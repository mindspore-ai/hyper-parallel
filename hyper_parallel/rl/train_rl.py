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
"""``torchrun`` entry point for synchronous Hyper-RL training."""

import argparse
import logging

# Preserve the historical CLI helper imports for existing callers.
from rl.cli import load_config, _apply_override, _parse_override_value
from rl.trainer import SyncTrainer

__all__ = ["load_config", "_apply_override", "_parse_override_value", "main"]


def main() -> None:
    """Parse configuration and execute the Hyper-RL trainer."""
    parser = argparse.ArgumentParser(description="Hyper-Parallel synchronous RL trainer")
    parser.add_argument(
        "config",
        help="Path to Hyper-RL YAML configuration",
    )
    args, overrides = parser.parse_known_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = load_config(args.config, overrides)
    trainer = SyncTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
