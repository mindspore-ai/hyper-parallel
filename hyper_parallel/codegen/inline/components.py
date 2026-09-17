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
"""Framework-generic component replacements for the inline pipeline.

Every model family that replaces a norm / expert / attention module with the
framework's high-performance generic component shares the *same* wiring: the
component import, the constructor shape (plain swap vs ``wrap_source``), the
keyword-only ctor arguments, and whether the source class is kept as the
wrapper input. That wiring is framework knowledge and lives here, keyed by
component kind, so an adapter declares only which generic component a YAML
target maps to plus the source class it matches -- never a copy of the
argument list.

Only the *source* class name (``Qwen3MoeRMSNorm``) and the attention kernel
interface symbol (``run_qwen3_moe_flash_attention``) stay per-family: the
former is the HF class actually present in the modeling file, the latter is
the family's mask/cache contract handed to the generic attention component.
"""

from __future__ import annotations

from hyper_parallel.codegen.inline.ir import ImportPatch
from hyper_parallel.codegen.inline.spec_bundle import ReplacementSpec

_COMPONENT_MODULES = "hyper_parallel.components.modules"


def rms_norm_replacement(old_ctor: str) -> ReplacementSpec:
    """Declare a same-name-same-structure RMSNorm swap.

    Model RMS norms share the generic ``RMSNorm`` contract, so the wiring is
    fully framework-owned; only the matched source class differs per family.
    """
    return ReplacementSpec(
        old_ctor=old_ctor,
        new_ctor="RMSNorm",
        imports=(ImportPatch(_COMPONENT_MODULES, ("RMSNorm",)),),
        replacement_note=f"RMSNorm replaces {old_ctor}.",
    )


def grouped_experts_replacement(old_ctor: str) -> ReplacementSpec:
    """Declare a batched-experts wrap onto the generic ``GroupedExperts``.

    The source class is kept as the wrapper input, so the generated call is
    ``GroupedExperts(module=<source class instance>, module_fqn='', context=None)``
    -- the keyword shape every generic component wrapper shares.
    """
    return ReplacementSpec(
        old_ctor=old_ctor,
        new_ctor="GroupedExperts",
        mode="wrap_source",
        keyword_args=("module_fqn=''", "context=None"),
        imports=(ImportPatch(_COMPONENT_MODULES, ("GroupedExperts",)),),
        remove_class=False,
        replacement_note=(
            f"GroupedExperts wraps {old_ctor}; the original class is kept "
            "as the wrapper input."
        ),
    )


def gqa_attention_replacement(
    *,
    old_ctor: str,
    attention_interface: str,
    generated_imports: tuple[ImportPatch, ...],
    snippets: tuple,
) -> ReplacementSpec:
    """Declare a fused-QKV attention wrap onto the generated ``GQAAttention``.

    The source attention class is kept (``remove_class=False``) so the
    decoder-layer call can wrap it, exactly like the runtime replacement. The
    generated class copies the real component's keyword-only ctor
    ``(*, module=..., attention_interface=...)``, so the wrap passes the
    family's mask/cache kernel entry as ``attention_interface``.
    """
    return ReplacementSpec(
        old_ctor=old_ctor,
        new_ctor="GQAAttention",
        mode="wrap_source",
        keyword_args=(
            "module_fqn=''",
            "context=None",
            f"attention_interface={attention_interface}",
        ),
        remove_class=False,
        imports=(
            ImportPatch("hyper_parallel.codegen.runtime", ("get_inline_parallel_state",)),
            ImportPatch(_COMPONENT_MODULES, ("RMSNorm",)),
            ImportPatch("hyper_parallel.platform", ("get_platform",)),
            *generated_imports,
        ),
        snippets=snippets,
        replacement_note=f"GQAAttention replaces {old_ctor}.",
    )


__all__ = [
    "gqa_attention_replacement",
    "grouped_experts_replacement",
    "rms_norm_replacement",
]
