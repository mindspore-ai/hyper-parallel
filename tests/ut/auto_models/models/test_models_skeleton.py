# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""M1 skeleton gates: models import boundary + lazy registry discovery.

The M0 characterization (empty MODEL_ARCH_MAPPING → HF fallback) lives in
tests/ut/auto_models/_transformers/test_config_resolver.py; these tests
pin the M1 additions (adjust doc §8 M1 门禁):
- ``import hyper_parallel.models`` pulls in neither
  Trainer/Data nor the legacy components/models god files;
- the registry discovers family adapter specs lazily, without touching
  the legacy Qwen implementation files;
- the S5f split (05 stage-5 item 4) removed the ``_transformers/registry.py``
  compat facade: HF config helpers live in
  ``_transformers/config_resolver.py`` and consume the family registry
  directly.
"""

import subprocess
import sys

import pytest

from tests.common.mark_utils import arg_mark

from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.registry import (
    MODEL_ADAPTER_REGISTRY,
    get_model_adapter,
    register_model_adapter,
)


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
         card_mark="onecard", essential_mark="essential")
def test_models_import_boundary_subprocess():
    """Importing models must not pull Trainer/Data/components.models."""
    code = (
        "import sys\n"
        "import hyper_parallel.models\n"
        "banned = [n for n in sys.modules if n.startswith((\n"
        "    'hyper_parallel.models.trainer',\n"
        "    'hyper_parallel.data',\n"
        "    'hyper_parallel.models.components.models',\n"
        "))]\n"
        "assert not banned, banned\n"
        "import hyper_parallel.models.registry\n"
        "delta = [n for n in sys.modules if n.startswith((\n"
        "    'hyper_parallel.models.trainer',\n"
        "    'hyper_parallel.data',\n"
        "    'hyper_parallel.models.components.models',\n"
        "))]\n"
        "assert not delta, delta\n"
        # lazy discovery self-registers without touching legacy code; M5
        # deleted the god files outright
        "from hyper_parallel.models.registry import get_model_adapter\n"
        "spec = get_model_adapter('qwen3_moe')\n"
        "assert spec.architecture == 'Qwen3MoeForCausalLM', spec\n"
        "import importlib.util\n"
        "assert importlib.util.find_spec(\n"
        "    'hyper_parallel.models.components'\n"
        ") is None\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
         card_mark="onecard", essential_mark="essential")
def test_lazy_family_discovery():
    """qwen3_moe self-registers on first lookup with the declared identity."""
    spec = get_model_adapter("qwen3_moe")
    assert isinstance(spec, ModelAdapterSpec), "case: spec_type"
    assert spec.architecture == "Qwen3MoeForCausalLM", "case: architecture"
    assert spec.model_type == "qwen3_moe", "case: model_type"


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
         card_mark="onecard", essential_mark="essential")
def test_unknown_family_resolves_to_none():
    """No class-name guessing: an unknown model_type finds no adapter."""
    assert get_model_adapter("no_such_family") is None, "case: unknown"


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
         card_mark="onecard", essential_mark="essential")
def test_register_model_adapter_idempotent_and_conflict():
    """Re-registering the same spec is fine; a conflicting one fails."""
    spec = ModelAdapterSpec(architecture="_M1Arch", model_type="_m1_probe")
    register_model_adapter(spec)
    register_model_adapter(spec)  # idempotent
    try:
        with pytest.raises(ValueError, match="conflicting adapter spec"):
            register_model_adapter(
                ModelAdapterSpec(architecture="_M1Other",
                                 model_type="_m1_probe"))
    finally:
        MODEL_ADAPTER_REGISTRY.pop("_m1_probe", None)


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
         card_mark="onecard", essential_mark="essential")
def test_transformers_registry_facade_is_removed():
    """S5f split: the _transformers.registry compat facade no longer exists.

    ``_transformers/config_resolver.py`` consumes the family registry
    directly — HF config resolution and family discovery no longer share
    one registry module (05 stage-5 item 4).
    """
    import importlib.util
    assert importlib.util.find_spec(
        "hyper_parallel.models._transformers.registry"
    ) is None, "case: facade_removed"
    from hyper_parallel.models._transformers import config_resolver
    from hyper_parallel.models import registry as new
    assert config_resolver._resolve_custom_model_cls is new._resolve_custom_model_cls, \
        "case: resolver"
