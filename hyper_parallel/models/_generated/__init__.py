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
"""Internal support for directly executable generated model sources."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def run_generated_training(
    namespace: Mapping[str, Any],
    architecture: str,
    argv: Sequence[str] | None = None,
    *,
    model_class_name: str | None = None,
) -> None:
    """Load the training-only entry lazily and run a generated model."""
    # Generated sources import this module for inspection too; defer the
    # Trainer dependency until the file is actually executed as a script.
    from hyper_parallel.models._generated.entry import (  # pylint: disable=import-outside-toplevel
        run_generated_training as run,
    )

    run(namespace, architecture, argv, model_class_name=model_class_name)

__all__ = ["run_generated_training"]
