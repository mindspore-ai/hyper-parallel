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
"""Mixin exposing a model.generate style API."""

from hyper_parallel.infer.generation import generate


class GenerateMixin:
    """Mixin that forwards to :func:`hyper_parallel.infer.generate`."""

    def generate(self, input_ids, generation_config=None, attention_mask=None, **kwargs):
        return generate(
            self,
            input_ids=input_ids,
            generation_config=generation_config,
            attention_mask=attention_mask,
            **kwargs,
        )
