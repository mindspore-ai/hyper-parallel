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
"""YAML configuration parser for PP optimization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import yaml


def _validate_int(raw: Any, field_name: str) -> int:
    """Convert *raw* to int, rejecting floats and bools.

    ``int(1.9)`` silently truncates to ``1``.  Even ``int(3.0)`` accepts
    a float where an integer was intended.  ``int(True)`` yields ``1``,
    which is almost never the intent for a pipeline-parallel integer
    config field.  This helper raises ``ValueError`` when the input is
    a float or bool.

    Args:
        raw: Value from parsed YAML.
        field_name: Human-readable field name for error messages.

    Returns:
        Integer value.

    Raises:
        ValueError: If *raw* is a float or bool.
    """
    if isinstance(raw, bool):
        raise ValueError(
            f"{field_name} must be an integer, got bool {raw}"
        )
    if isinstance(raw, float):
        raise ValueError(
            f"{field_name} must be an integer, got {raw}"
        )
    return int(raw)


def _to_bool(value: Any, field_name: str = "field") -> bool:
    """Convert a YAML-parsed value to bool with strict string handling.

    ``bool("false")`` is ``True`` in Python because any non-empty string
    is truthy.  This helper uses a whitelist of recognized boolean
    representations and raises ``ValueError`` for ambiguous inputs.

    Accepted values:

    * ``bool`` — ``True`` / ``False`` directly.
    * ``int`` — only ``0`` (False) or ``1`` (True); other integers
      are rejected.
    * ``float`` — only ``0.0`` (False) or ``1.0`` (True); other
      floats are rejected.
    * ``str`` — ``"true"``/``"yes"``/``"1"`` → True,
      ``"false"``/``"no"``/``"0"`` → False; anything else is
      rejected.

    Args:
        value: Value from ``yaml.safe_load`` — typically ``bool``,
            ``int``, ``float``, or ``str``.
        field_name: Human-readable field name for error messages.

    Returns:
        Boolean interpretation of *value*.

    Raises:
        ValueError: If *value* is not a recognized boolean
            representation (e.g. int other than 0/1, ambiguous
            string, etc.).
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        if value == 0:
            return False
        if value == 1:
            return True
        raise ValueError(
            f"{field_name}: cannot interpret {value} as boolean; "
            f"expected 0 or 1"
        )
    if isinstance(value, float) and value.is_integer():
        int_val = int(value)
        if int_val == 0:
            return False
        if int_val == 1:
            return True
        raise ValueError(
            f"{field_name}: cannot interpret {value} as boolean; "
            f"expected 0.0 or 1.0"
        )
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "yes", "1"):
            return True
        if low in ("false", "no", "0"):
            return False
        raise ValueError(
            f"{field_name}: cannot interpret '{value}' as boolean; "
            f"expected true/false/yes/no/1/0"
        )
    raise ValueError(
        f"{field_name}: cannot interpret {value!r} as boolean"
    )


@dataclass
class YamlOptimizationConfig:
    """Parsed YAML configuration for PP optimization.

    Attributes:
        pp_degree: Number of pipeline stages.  Parsed from the YAML key
            ``pipeline_num``; the field name uses ILP domain terminology.
        micro_batch_num: Number of micro batches.
        num_layer: Number of homogeneous body layers.  Optional; when
            ``None``, actual layer counts come from the JSON profile
            via :func:`generate_layers_list`.
        num_of_interleave: VPP interleaving factor.
        vpp_less_memory: Use the less-memory VPP schedule (``vpp2``).
        optimization_level: ILP optimization level (0-2).
        memory_limit: Per-stage memory limit in MB.  Must be positive
            for ILP load balancing.
        constant_memory: Constant memory per stage in MB.
        enable_simulation: Whether to run the pipeline simulator after
            ILP solving.
        sim_comm_time: P2P communication time between adjacent stages
            in ms.  Used only by the pipeline simulator; does NOT
            affect ILP optimization.
        use_backward_time: Whether to use real backward times from the
            JSON profile for simulation.  When ``False`` (default), the
            simulator derives backward time from forward time using
            ``backward_ratio`` (original behaviour).  When ``True``,
            the simulator uses actual backward times from profiling.
    """

    pp_degree: int
    micro_batch_num: int
    num_layer: Optional[int] = None
    num_of_interleave: int = 1
    vpp_less_memory: bool = False
    optimization_level: int = 1
    memory_limit: int = 0
    constant_memory: int = 0
    enable_simulation: bool = True
    sim_comm_time: float = 0.0
    use_backward_time: bool = False

    def _validate_field_types(self) -> None:
        """Check that every config field has the expected Python type.

        Integer fields reject ``bool`` and ``float``; boolean fields
        must be genuine ``bool`` instances; ``sim_comm_time`` must be
        a finite ``int`` or ``float`` (not ``bool``).

        Raises:
            ValueError: If any field has an unexpected type.
        """
        int_fields = (
            "pp_degree", "micro_batch_num", "num_of_interleave",
            "optimization_level", "memory_limit", "constant_memory",
        )
        for name in int_fields:
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, int):
                raise ValueError(
                    f"{name} must be an integer, "
                    f"got {type(val).__name__} {val!r}"
                )
        if self.num_layer is not None:
            if (isinstance(self.num_layer, bool)
                    or not isinstance(self.num_layer, int)):
                raise ValueError(
                    f"num_layer must be an integer or None, "
                    f"got {type(self.num_layer).__name__} "
                    f"{self.num_layer!r}"
                )
        for name in ("vpp_less_memory", "enable_simulation", "use_backward_time"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(
                    f"{name} must be a boolean, "
                    f"got {type(getattr(self, name)).__name__}"
                )
        if (isinstance(self.sim_comm_time, bool)
                or not isinstance(self.sim_comm_time, (int, float))
                or not math.isfinite(self.sim_comm_time)):
            raise ValueError(
                f"sim_comm_time must be a finite number, "
                f"got {self.sim_comm_time}"
            )

    def validate(self) -> None:
        """Validate all pipeline configuration fields.

        Checks both types (via :meth:`_validate_field_types`) and
        value ranges.

        Raises:
            ValueError: If any field has an invalid type or value.
        """
        self._validate_field_types()
        if self.pp_degree <= 0:
            raise ValueError(
                f"pp_degree must be positive, got {self.pp_degree}"
            )
        if self.micro_batch_num <= 0:
            raise ValueError(
                f"micro_batch_num must be positive, "
                f"got {self.micro_batch_num}"
            )
        if self.num_layer is not None and self.num_layer <= 0:
            raise ValueError(
                f"num_layer must be positive when provided, "
                f"got {self.num_layer}"
            )
        if self.num_of_interleave <= 0:
            raise ValueError(
                f"num_of_interleave must be positive, "
                f"got {self.num_of_interleave}"
            )
        if self.optimization_level not in (0, 1, 2):
            raise ValueError(
                f"optimization_level must be 0, 1, or 2, "
                f"got {self.optimization_level}"
            )
        if self.memory_limit < 0:
            raise ValueError(
                f"memory_limit must be non-negative, "
                f"got {self.memory_limit}"
            )
        if self.constant_memory < 0:
            raise ValueError(
                f"constant_memory must be non-negative, "
                f"got {self.constant_memory}"
            )
        if self.sim_comm_time < 0.0:
            raise ValueError(
                f"sim_comm_time must be non-negative, "
                f"got {self.sim_comm_time}"
            )


def _extract_required_fields(
        pipeline_cfg: dict,
        yaml_path: str,
) -> tuple[int, Optional[int], int]:
    """Extract and validate required pipeline topology fields.

    Args:
        pipeline_cfg: The ``pipeline_config`` mapping from the YAML file.
        yaml_path: Path to the YAML file (for error messages).

    Returns:
        ``(pp_degree, num_layer, micro_batch_num)`` tuple.

    Raises:
        ValueError: If a required field is missing or invalid.
    """
    pipeline_num = pipeline_cfg.get("pipeline_num")
    if pipeline_num is None:
        raise ValueError(
            f"{yaml_path}: pipeline_config.pipeline_num is required"
        )
    pp_degree = _validate_int(pipeline_num, "pipeline_config.pipeline_num")
    if pp_degree <= 0:
        raise ValueError(
            f"{yaml_path}: pipeline_config.pipeline_num must be "
            f"positive, got {pp_degree}"
        )

    num_layer_raw = pipeline_cfg.get("num_layer")
    num_layer: Optional[int] = None
    if num_layer_raw is not None:
        num_layer = _validate_int(num_layer_raw, "pipeline_config.num_layer")
        if num_layer <= 0:
            raise ValueError(
                f"pipeline_config.num_layer must be positive, got {num_layer}"
            )

    micro_batch_num = pipeline_cfg.get("micro_batch_num")
    if micro_batch_num is None:
        raise ValueError("YAML pipeline_config.micro_batch_num is required")
    micro_batch_num = _validate_int(
        micro_batch_num, "pipeline_config.micro_batch_num",
    )
    if micro_batch_num <= 0:
        raise ValueError(
            f"pipeline_config.micro_batch_num must be positive, "
            f"got {micro_batch_num}"
        )

    return pp_degree, num_layer, micro_batch_num


def _extract_optional_fields(pipeline_cfg: dict) -> tuple[int, bool, int, int, int, bool, float, bool]:
    """Extract and type-convert optional pipeline configuration fields.

    Range validation is delegated to :meth:`YamlOptimizationConfig.validate`
    so that direct construction also benefits from the same checks.

    Args:
        pipeline_cfg: The ``pipeline_config`` mapping from the YAML file.

    Returns:
        ``(num_of_interleave, vpp_less_memory, optimization_level,
        memory_limit, constant_memory, enable_simulation,
        sim_comm_time, use_backward_time)`` tuple.

    Raises:
        ValueError: If a field cannot be converted to the expected type
            (e.g. float where int is required, ambiguous boolean string).
    """
    num_of_interleave = _validate_int(
        pipeline_cfg.get("num_of_interleave", 1),
        "pipeline_config.num_of_interleave",
    )

    vpp_less_memory = _to_bool(
        pipeline_cfg.get("vpp_less_memory", False),
        "pipeline_config.vpp_less_memory",
    )

    optimization_level = _validate_int(
        pipeline_cfg.get("optimization_level", 1),
        "pipeline_config.optimization_level",
    )

    memory_limit = _validate_int(
        pipeline_cfg.get("memory_limit", 0),
        "pipeline_config.memory_limit",
    )

    constant_memory = _validate_int(
        pipeline_cfg.get("constant_memory", 0),
        "pipeline_config.constant_memory",
    )

    enable_simulation = _to_bool(
        pipeline_cfg.get("enable_simulation", True),
        "pipeline_config.enable_simulation",
    )

    sim_comm_time = float(pipeline_cfg.get("sim_comm_time", 0.0))

    use_backward_time = _to_bool(
        pipeline_cfg.get("use_backward_time", False),
        "pipeline_config.use_backward_time",
    )

    return (
        num_of_interleave, vpp_less_memory, optimization_level,
        memory_limit, constant_memory, enable_simulation, sim_comm_time,
        use_backward_time,
    )


def parse_yaml_for_optimization(yaml_path: str) -> YamlOptimizationConfig:
    """Parse a YAML configuration file for PP optimization.

    The YAML must contain a ``pipeline_config`` section with
    ``pipeline_num`` and ``micro_batch_num``.
    ``num_layer`` is optional; when omitted it defaults to ``None``
    and the layer count is derived from the JSON profile.
    ``num_of_interleave`` is optional and defaults to 1.
    ``memory_limit``, ``constant_memory``, ``enable_simulation``,
    ``sim_comm_time`` are optional.

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        :class:`YamlOptimizationConfig` with all required fields.

    Raises:
        ValueError: If required fields are missing or values are invalid.
        FileNotFoundError: If the YAML file does not exist.

    Example:
        >>> config = parse_yaml_for_optimization("pp_config.yaml")
        >>> config.pp_degree
        4
        >>> config.num_layer
        32
    """
    with open(yaml_path, encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)

    if not isinstance(cfg, dict):
        raise ValueError(
            f"YAML file {yaml_path} must contain a top-level mapping, "
            f"got {type(cfg).__name__}"
        )

    pipeline_cfg: dict = cfg.get("pipeline_config", {})
    if not isinstance(pipeline_cfg, dict):
        raise ValueError(
            f"YAML file {yaml_path} must contain a 'pipeline_config' section"
        )

    pp_degree, num_layer, micro_batch_num = _extract_required_fields(
        pipeline_cfg, yaml_path,
    )
    (
        num_of_interleave, vpp_less_memory, optimization_level,
        memory_limit, constant_memory, enable_simulation, sim_comm_time,
        use_backward_time,
    ) = _extract_optional_fields(pipeline_cfg)

    config = YamlOptimizationConfig(
        pp_degree=pp_degree,
        num_layer=num_layer,
        micro_batch_num=micro_batch_num,
        num_of_interleave=num_of_interleave,
        vpp_less_memory=vpp_less_memory,
        optimization_level=optimization_level,
        memory_limit=memory_limit,
        constant_memory=constant_memory,
        enable_simulation=enable_simulation,
        sim_comm_time=sim_comm_time,
        use_backward_time=use_backward_time,
    )
    config.validate()
    return config
