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
"""Top-level training public API.

Rebuilt in stage 7 (05 §10.1): the trainer classes moved here from
``auto_models/trainer/`` (S7d). Exports are lazy so that importing the
package (e.g. for ``trainer.config`` or ``trainer.callbacks``) stays cheap
and does not pull the model/data stack until a trainer class is touched.
"""

__all__ = [
    "BaseTrainer",
    "TextTrainer",
    "VLMTrainer",
    "TrainerState",
]

_LAZY_EXPORTS = {
    "BaseTrainer": "hyper_parallel.trainer.base",
    "TextTrainer": "hyper_parallel.trainer.text_trainer",
    "VLMTrainer": "hyper_parallel.trainer.vlm_trainer",
    "TrainerState": "hyper_parallel.trainer.state",
}


def __getattr__(name):
    """Resolve trainer classes lazily on first attribute access."""
    if name in _LAZY_EXPORTS:
        import importlib  # pylint: disable=import-outside-toplevel

        value = getattr(importlib.import_module(_LAZY_EXPORTS[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """List lazy exports for interactive discovery."""
    return sorted(__all__)
