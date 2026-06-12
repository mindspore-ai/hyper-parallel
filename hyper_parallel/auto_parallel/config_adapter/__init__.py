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
"""Auto parallel strategy search configuration interfaces.

Provides a pipeline for reading, validating, searching, and writing
parallel strategy search configurations::

    from hyper_parallel.auto_parallel.config_adapter import (
        read_search_config, validate_strict, search_strategies,
    )

    config = read_search_config("search.yaml")
    validate_strict(config)
    result = search_strategies(config)

    config.resolved_strategy = result
    write_strategy_config(config, "output/strategy.json")
"""

from hyper_parallel.auto_parallel.config_adapter._normalized_config import (  # noqa: E402, F401
    NormalizedConfig,
    ValidationError,
    ValidationSeverity,
)
from hyper_parallel.auto_parallel.config_adapter._config_loader import (  # noqa: E402
    read_hp_yaml_config,
    read_search_config,
)
from hyper_parallel.auto_parallel.config_adapter._search_runner import (  # noqa: E402
    search_strategies,
)
from hyper_parallel.auto_parallel.config_adapter._constraint_checker import (  # noqa: E402
    validate,
    validate_strict,
)
from hyper_parallel.auto_parallel.config_adapter._strategy_output import (  # noqa: E402
    normalized_to_summary,
    write_ppb_config,
    write_resolved_strategy,
    write_resolved_yaml,
    write_strategy_config,
)

__all__ = [
    "NormalizedConfig",
    "ValidationError",
    "ValidationSeverity",
    "normalized_to_summary",
    "read_hp_yaml_config",
    "read_search_config",
    "search_strategies",
    "validate",
    "validate_strict",
    "write_ppb_config",
    "write_resolved_strategy",
    "write_resolved_yaml",
    "write_strategy_config",
]
