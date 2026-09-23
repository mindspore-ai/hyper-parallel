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
"""Shared fixtures for ``tests/components/distributed``.

The toy models (``TinyConfig`` / ``TinyLlamaAttention`` /
``TinyLlamaForCausalLM``) and the distributed harness moved to
``tests/ut/dual_mode_dtensor/conftest.py`` in the auto_models reorg.  This
module re-exports the fixtures so the suites under this package keep using
them without a second copy drifting from the canonical one.
"""

# Re-export the *fixture* providers (a bare module-level import is what makes
# pytest register them for this directory); ``TinyConfig`` and friends are
# imported directly by the test modules and need no re-export here.
from tests.ut.dual_mode_dtensor.conftest import (  # noqa: F401  pylint: disable=unused-import
    make_mesh,
    tiny_hf_batched_moe,
    tiny_hf_llama,
    tiny_hf_native_moe,
    tiny_llama,
    tiny_moe,
)
