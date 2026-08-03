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
"""Tests for basic data-transform components."""

from hyper_models.components.data import IdentityDataTransform


def test_identity_data_transform_retains_tokenizer_and_example() -> None:
    """Retain the upstream tokenizer while leaving examples unchanged."""
    tokenizer = object()
    transform = IdentityDataTransform(tokenizer=tokenizer)
    example = {"text": "hello"}

    assert transform.tokenizer is tokenizer
    assert transform(example) is example
