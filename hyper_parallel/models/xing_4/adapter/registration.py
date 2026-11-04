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
"""Register the Xing4 model and its adapter.

Expert parallelism and the optimizer are selected by the recipe through
``plan_overrides`` and the ``optimizer`` target.
"""

from typing import Any

from hyper_parallel.models.adapter_spec import ModelAdapterSpec, RecomputePolicy
from hyper_parallel.models.registry import register_custom_model, register_model_adapter


def build_recompute_policy() -> RecomputePolicy:
    """Expose whole decoder and MTP blocks; layer selection remains a recipe choice."""
    return RecomputePolicy(region_patterns=("model.layers.*", "mtp.layers.*"))


def get_fsdp_excluded_subtrees(model: Any) -> tuple[str, ...]:
    """Use the declared MTP depths instead of inferring nested decoder units."""
    return () if model.mtp is None else ("mtp",)


def get_fsdp_wrap_modules(model: Any) -> tuple[str, ...]:
    """Keep each MTP depth's parameters under the module that calls its decoder."""
    return (() if model.mtp is None else
            tuple(f"mtp.layers.{name}" for name, _ in model.mtp.layers.named_children()))


def build_parameter_sharding_rules() -> list[tuple[list[str], Any]]:
    """Declare Xing4 MLA roles before the generic naming rules.

    The mHC, stream-collapse head and MTP fusion parameters are declared
    replicated by the recipe's ``plan_overrides``.

    Returns:
        Name patterns and their parameter roles.
    """
    # Discovery imports this module without requiring the distributed runtime.
    from hyper_parallel.distributed.tensor_parallel.param_role import (  # pylint: disable=import-outside-toplevel
        ParamRole,
    )

    return [
        (["linear_qkv"], ParamRole.REPLICATED),
        (["q_b_proj", "kv_b_proj"], ParamRole.COLWISE),
    ]


def _load_replacements():
    """Return the family's replacement-factory module (lazy provider)."""
    from hyper_parallel.models.xing_4.adapter.conversion import (  # pylint: disable=import-outside-toplevel
        module_replacement,
    )

    return module_replacement


def _load_context_parallel():
    """Load the recipe-selected CP wrappers without importing the runtime at discovery."""
    from hyper_parallel.models.xing_4.adapter.distributed import (  # pylint: disable=C0415
        context_parallel,
    )
    return context_parallel


def _load_expert_parallel():
    """Load the recipe-selected shared EP executor adapter lazily."""
    from hyper_parallel.models.xing_4.adapter.distributed import (  # pylint: disable=C0415
        expert_parallel,
    )
    return expert_parallel


register_custom_model(
    "Xing4ForCausalLM",
    "hyper_parallel.models.xing_4.adapter.training",
    "Xing4ForCausalLM",
)

XING4_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="Xing4ForCausalLM",
    model_type="xing4_0",
    replacements=_load_replacements,
    context_parallel=_load_context_parallel,
    expert_parallel=_load_expert_parallel,
    recompute=build_recompute_policy,
    sharding_rules=build_parameter_sharding_rules,
    fsdp_wrap_modules=get_fsdp_wrap_modules,
    fsdp_excluded_subtrees=get_fsdp_excluded_subtrees,
)
register_model_adapter(XING4_ADAPTER_SPEC)

__all__ = ["XING4_ADAPTER_SPEC"]
