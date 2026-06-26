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
"""LM pretraining / SFT entry point.

Usage:
    torchrun --nproc_per_node=8 scripts/train_lm.py config.yaml

Adding a new model
------------------
No edit to this file is required. Create
``hyper_parallel/models/<name>/__init__.py`` that calls
``register_spec("<name>", ModelSpec(...))`` at import time, then set
``model.name: <name>`` in the YAML. See ``trainer/utils/discovery.py``
for the resolution rules (built-in vs. fully-qualified external package).
"""
from hyper_parallel.trainer.utils import init_logger
from hyper_parallel.trainer.config import parse_args, HyperTrainerConfig
from hyper_parallel.trainer.utils.discovery import discover_model_spec
from hyper_parallel.trainer.llm_trainer import LLMTrainer

# Configure the root logger before training. Module-level loggers in the
# imports above resolve their handler lazily (at first emit, which happens
# during training), so doing this after the imports keeps the same format.
init_logger()

if __name__ == "__main__":
    args = parse_args(HyperTrainerConfig)
    # Resolve ``model.name`` to a Python package and import it. The package's
    # ``__init__.py`` calls ``register_spec`` as a side effect, populating the
    # global registry that ``BaseTrainer.__init__`` then queries via
    # ``get_spec(args.model.name)``.
    discover_model_spec(args.model.name)
    trainer = LLMTrainer(args)
    trainer.train()
