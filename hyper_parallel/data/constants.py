# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared dataset constants.

Split from the former ``components/utils/constants.py`` in stage 6
(05 §15.10 step 3): the cross-modal token indexes live in
``hyper_parallel.data.vlm.constants``; only the loss-mask sentinel used
by the text pipeline stays here.
"""


IGNORE_INDEX = -100
