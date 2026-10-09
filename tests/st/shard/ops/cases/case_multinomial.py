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
"""Shard ops cases for ``torch.multinomial``.

.. note::
    A-class issue: the old tests manually call ``torch.manual_seed()`` before
    each standalone and distributed call to synchronise random state.  The new
    framework does not support per-case seed resets, so standalone and
    distributed executions use different random states and produce mismatched
    samples.  Re-enable once the framework supports a ``seed_reset`` hook.
"""
