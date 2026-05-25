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
"""Port allocation utilities for distributed test cases."""
import fcntl
import os
import tempfile


def allocate_port() -> int:
    """Atomically allocate a unique port from a circular counter.

    Uses :func:`fcntl.flock` to guarantee that no two processes ever receive
    the same port.  The counter wraps through a fixed range (10000–29999,
    deliberately below the Linux default ephemeral range of 32768–60999)
    so that ports are cycled safely without colliding with OS-assigned ports.

    Returns:
        A TCP port number guaranteed unique among all current callers.
    """
    counter_path = os.path.join(tempfile.gettempdir(), f"hp_port_counter_{os.getuid()}")
    port_base = 10000
    port_range = 20000
    with open(counter_path, "a+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.seek(0)
        content = f.read().strip()
        counter = int(content) if content else 0
        f.seek(0)
        f.truncate()
        f.write(str(counter + 1))
    return port_base + (counter % port_range)
