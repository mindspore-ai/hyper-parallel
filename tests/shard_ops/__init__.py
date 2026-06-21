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
"""Cross-platform shared utilities exclusively for shard ops ST.

Scope discipline: this package serves only ``tests/torch/shard/ops/`` and
``tests/mindspore/st/shard/ops/``. It must not import ``torch`` or
``mindspore`` directly — platform-specific code lives under
``tests/<plat>/shard/ops/framework/`` and registers itself with the shared
backend registry at import time.
"""
