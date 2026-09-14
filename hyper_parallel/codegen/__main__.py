# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""``python -m hyper_parallel.codegen`` entry point."""
from __future__ import annotations

import sys

from hyper_parallel.codegen.cli import main

if __name__ == "__main__":
    sys.exit(main())
