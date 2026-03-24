# Copyright 2025 Huawei Technologies Co., Ltd
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

"""MS acl shmem Ops Package

This module provides the public API for one-sided communication Operations for MindSpore framework
based on aclshmem backend.
It includes functions for registering one-sided communication operators and managing
their lifecycle within the MindSpore framework.

Example usage:
    >>> import aclshmem_ms
    >>> aclshmem_ms.register_op("my_custom_op", ...)
"""

import os
import glob

def _find_plugin():
    """Find plugin .so file path automatically."""
    current_path = os.path.dirname(os.path.abspath(__file__))
    pattern = os.path.join(current_path, "aclshmem_ms.so")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"Plugin .so not found in {current_path}")
    return matches[0]


_find_plugin()

# pylint: disable=wrong-import-position
from .aclshmem_ms import *

# Import generated ops interfaces
try:
    from .gen_ops_def import *
except ImportError:
    pass  # Generated files may not exist during development

try:
    from .gen_ops_prim import *
except ImportError:
    pass  # Generated files may not exist during development

__all__ = []
# Add ops from gen_ops_def if available
try:
    import aclshmem_ms.gen_ops_def as gen_ops_def  # pylint: disable=consider-using-from-import
    if hasattr(gen_ops_def, '__all__'):
        __all__.extend(gen_ops_def.__all__)
    else:
        # If no __all__ defined, add all public functions
        __all__.extend([name for name in dir(gen_ops_def) if not name.startswith('_')])
except ImportError:
    pass

# Add ops from gen_ops_prim if available
try:
    import aclshmem_ms.gen_ops_prim as gen_ops_prim  # pylint: disable=consider-using-from-import
    if hasattr(gen_ops_prim, '__all__'):
        __all__.extend(gen_ops_prim.__all__)
    else:
        # If no __all__ defined, add all public functions
        __all__.extend([name for name in dir(gen_ops_prim) if not name.startswith('_')])
except ImportError:
    pass
