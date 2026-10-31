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
"""Device-free orchestration for snapshot-driven code generation."""

from typing import Any, Optional

from hyper_parallel.codegen.artifact import ArtifactLayout
from hyper_parallel.codegen.model_build_overlay import use_codegen_model_build_overlay
from hyper_parallel.codegen.offline_build import create_offline_distributed_setup
from hyper_parallel.codegen.planner_overlay import use_codegen_planner_overlay
from hyper_parallel.codegen.session import CodegenBuildSession
from hyper_parallel.trainer.config import normalize_distributed_setup_overrides


def generate_from_native_build(
    config: Any,
    *,
    artifact_dir: Optional[str] = None,
) -> ArtifactLayout:
    """Observe the configured model on meta and publish its frozen snapshot.

    Args:
        config: Resolved TrainerConfig. Only model and planning settings are
            consumed; data, optimizer, scheduler, and training loops are never built.
        artifact_dir: Optional explicit bundle output directory.

    Returns:
        The generated or reused artifact layout.

    Raises:
        RuntimeError: If the native model does not publish a complete artifact.
    """
    return _build_and_publish(config, artifact_dir)


def _build_and_publish(config: Any, artifact_dir: Optional[str]) -> ArtifactLayout:
    """Build and publish from a process-group-free planning topology."""
    distributed_setup = create_offline_distributed_setup(config)
    normalize_distributed_setup_overrides(distributed_setup, config)

    session = CodegenBuildSession()
    with use_codegen_model_build_overlay(session, artifact_dir) as result, use_codegen_planner_overlay():
        config.model.build(
            distributed_setup=distributed_setup,
            peft_config=config.peft,
            activation_checkpoint=config.activation_checkpoint.mode,
            activation_checkpoint_selection=config.activation_checkpoint.selection,
            swap_inputs=getattr(config.activation_checkpoint, "swap_inputs", False),
            allow_cache_entry_mutation=getattr(
                config.activation_checkpoint,
                "allow_cache_entry_mutation",
                False,
            ),
            activation_swap=config.activation_swap,
            compile_config=config.compile,
            model_init_dtype=config.model_init_dtype,
        )

    if result.artifact_layout is None:
        raise RuntimeError("native model build completed without publishing a codegen artifact")
    return result.artifact_layout


__all__ = ["generate_from_native_build"]
