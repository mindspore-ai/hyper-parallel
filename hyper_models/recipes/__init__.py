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
"""Recipes package — RECIPE_REGISTRY（03 §13）。"""

from hyper_models.recipes.llm.train_ft import FinetuneRecipe

# 03 §13：main() 通过 cfg.recipe 取字符串 → RECIPE_REGISTRY[name] 查表；
# 未命中时由 main() 尝试 importlib.import_module 动态导入。
# YAML 未设置 recipe 时默认使用 "FinetuneRecipe"。
RECIPE_REGISTRY = {
    "FinetuneRecipe": FinetuneRecipe,
}

__all__ = ["RECIPE_REGISTRY", "FinetuneRecipe"]
