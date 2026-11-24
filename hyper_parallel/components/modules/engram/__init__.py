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
"""Public Engram contracts and legacy class exports."""
from .engram import Engram, EngramModule, EPEngramModule
from .ngram_hash_mapping import NgramHashMapping
from .parallel_engram import local_token_mask

__all__ = ["Engram", "EngramModule", "EPEngramModule", "NgramHashMapping", "local_token_mask"]
