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
"""Register configuration resolvers used by the VeRL examples."""

from omegaconf import OmegaConf


def register_cp_token_budget() -> None:
    """Register the per-rank token budget derived from prompt, response and CP sizes."""
    if not OmegaConf.has_resolver("cp_token_budget"):
        OmegaConf.register_new_resolver(
            "cp_token_budget",
            lambda prompt, response, cp: (int(prompt) + int(response) + int(cp) - 1) // int(cp),
        )
