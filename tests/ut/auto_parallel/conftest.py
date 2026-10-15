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
"""The environment the ND cost model's unit tests run in.

Pytest imports this file before any test module below it, so these settings
hold before the cost model is imported, and for the whole session: no other
part of HyperParallel reads them.
"""
import os

# With no backend named, matplotlib picks an interactive one where a display
# exists (TkAgg on Windows), and a Python whose Tcl/Tk files are missing then
# fails whichever drawing test first opens a figure, so the red one depends on
# test order. No test shows a window: draw to files.
os.environ.setdefault("MPLBACKEND", "Agg")

# A layer census is kept on disk between runs (``ND_CENSUS_CACHE``). A test
# that read a census an earlier run kept, or kept one in the user's cache,
# would test the folder rather than the census: none is kept, and the cache's
# own tests name a folder of their own.
os.environ["ND_CENSUS_CACHE"] = "off"
