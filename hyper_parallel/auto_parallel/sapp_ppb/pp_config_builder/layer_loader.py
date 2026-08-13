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
"""Layer loader — load sapp-ppb Layer objects from native JSON and post-process."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.yaml_parser import (
    YamlOptimizationConfig,
)

try:
    from hyper_parallel.auto_parallel.sapp_ppb.sapp.sapp_pipeline import SappPipeline
    from hyper_parallel.auto_parallel.sapp_ppb.utils import recompute as Recompute
    from hyper_parallel.auto_parallel.sapp_ppb.utils.layer import generate_layers_list
    SAPP_PPB_AVAILABLE = True
except ImportError:
    SAPP_PPB_AVAILABLE = False
    SappPipeline = None
    Recompute = None
    generate_layers_list = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _get_pipeline_layer_class() -> Any:
    """Return the ``Layer`` class that ``SappPipeline`` actually uses.

    ``SappPipeline`` imports ``Layer`` from ``sapp_ppb.utils.layer``.
    Depending on import order, the top-level ``sapp_ppb`` package may
    resolve to the ``hyper_parallel.auto_parallel.sapp_ppb`` namespace,
    creating a *different* ``Layer`` class (with a different
    ``type_enum``) than what ``SappPipeline`` uses internally.  To
    avoid ``filter_layer_type`` returning empty lists due to enum
    identity mismatches, we always retrieve ``Layer`` from
    ``SappPipeline``'s own module globals.
    """
    if SappPipeline is None:
        return None
    return SappPipeline.__init__.__globals__.get("Layer")


def _apply_recompute_considered(
    layer: Any,
    recompute_considered: Dict[Any, bool],
) -> None:
    """Override ``recompute_considered_`` on a sapp-ppb Layer object.

    The ``Layer`` constructor ignores the ``None`` semantics in
    ``backward_time_rec`` and ``memory_activation_rec`` — it treats
    ``None`` as "auto-fill with DEFAULT_COEF" rather than "disable".  This
    method forces the Layer's ``recompute_considered_`` to match the
    desired inference so that the ILP solver only creates decision
    variables for genuinely configured recompute types.

    Args:
        layer: A sapp-ppb Layer object with a ``recompute_considered_``
            dict attribute.
        recompute_considered: Desired mapping from
            :class:`Recompute.TYPE` to bool.
    """
    for rec_type in Recompute.TYPE:
        layer.recompute_considered_[rec_type] = recompute_considered.get(
            rec_type, False,
        )
    layer.compute_internal_time()


def _head_tail_recompute_considered() -> Dict[Any, bool]:
    """Return ``recompute_considered`` for HEAD/TAIL layers.

    HEAD and TAIL layers never participate in recompute decisions; only
    ``NONE`` is considered.

    Returns:
        Dict mapping each :class:`Recompute.TYPE` to ``False`` except
        ``NONE`` which is ``True``.
    """
    return {r: (r == Recompute.TYPE.NONE) for r in Recompute.TYPE}


class LayerBuilder:
    """Build sapp-ppb Layer objects from YAML config + native JSON.

    Uses :func:`generate_layers_list` from the native sapp-ppb parser
    to read the ``layers_description`` from the JSON file, then applies
    recompute-considered overrides and consistency validation to produce
    a list of sapp-ppb ``Layer`` objects that can be consumed by
    :class:`PPBalancer`.

    Args:
        yaml_config: YAML configuration with pipeline topology and
            ILP constraints.
        json_path: Path to the native sapp-ppb JSON file (containing
            ``layers_description``).

    Example:
        >>> builder = LayerBuilder(yaml_config, json_path)
        >>> layers = builder.layers_sapp_ppb
    """

    def __init__(
        self,
        yaml_config: YamlOptimizationConfig,
        json_path: str,
    ) -> None:
        """Initialize LayerBuilder.

        Args:
            yaml_config: YAML configuration with pipeline topology.
            json_path: Path to the native sapp-ppb JSON file.

        Raises:
            ImportError: If sapp-ppb module is not available.
            ValueError: If ``json_path`` is empty or layers cannot be
                parsed.
        """
        if not SAPP_PPB_AVAILABLE:
            raise ImportError(
                "sapp-ppb module is not available. "
                "Please ensure sapp-ppb is installed and accessible."
            )

        if not json_path:
            raise ValueError(
                "LayerBuilder requires a json_path. "
                "Please provide a valid JSON profile path."
            )

        yaml_config.validate()

        self.yaml_config = yaml_config
        self._memory_limit = yaml_config.memory_limit
        self._constant_memory = yaml_config.constant_memory
        self._enable_simulation = yaml_config.enable_simulation
        self._use_backward_time = yaml_config.use_backward_time

        layer_folder = os.path.dirname(json_path)
        model_name = os.path.splitext(os.path.basename(json_path))[0]

        layers = generate_layers_list(layer_folder, model_name)
        if not layers:
            raise ValueError(
                f"No layers parsed from '{json_path}'. "
                f"Ensure the JSON file contains a 'layers_description' section."
            )

        self._post_process_layers(layers)
        self._validate_recompute_consistency(layers)
        self._validate_group_names(layers)
        self._validate_num_layer_consistency(layers)

        self.layers_sapp_ppb = layers

    @property
    def memory_limit(self) -> Optional[int]:
        """Maximum memory per stage in MB (from YAML config)."""
        return self._memory_limit

    @property
    def constant_memory(self) -> int:
        """Constant memory overhead per stage in MB (from YAML config)."""
        return self._constant_memory

    @property
    def use_backward_time(self) -> bool:
        """Whether to use backward time in ILP optimization."""
        return self._use_backward_time

    def _post_process_layers(self, layers: List[Any]) -> None:
        """Override recompute_considered on HEAD/TAIL layers.

        For HEAD/TAIL layers, only NONE recompute is considered.
        BODY layers retain the recompute_considered mask inferred by
        the native ``Layer.find_recompute_considered()`` result.

        Args:
            layers: List of sapp-ppb Layer objects from
                :func:`generate_layers_list`.
        """
        pipeline_layer = _get_pipeline_layer_class()
        if pipeline_layer is None:
            return

        for layer in layers:
            if layer.type_ in (pipeline_layer.type_enum.HEAD, pipeline_layer.type_enum.TAIL):
                _apply_recompute_considered(
                    layer, _head_tail_recompute_considered(),
                )

    @staticmethod
    def _validate_recompute_consistency(layers: List[Any]) -> None:
        """Validate that all BODY groups share the same recompute_considered mask.

        In a multi-body-group configuration, all BODY groups must enable
        the same set of recompute types so that the ILP solver can
        construct a coherent global mask.  If different groups enable
        different recompute types, the solver would produce incorrect
        decision variable assignments.

        Args:
            layers: List of sapp-ppb Layer objects with
                ``.recompute_considered_`` and ``.type_`` attributes.

        Raises:
            ValueError: If two or more BODY groups have different
                ``recompute_considered_`` masks.
        """
        pipeline_layer = _get_pipeline_layer_class()
        if pipeline_layer is None:
            return

        body_groups: Dict[str, Any] = {}
        for lay in layers:
            if lay.type_ == pipeline_layer.type_enum.BODY:
                body_groups[lay.name_] = lay

        if len(body_groups) <= 1:
            return

        ref_name = next(iter(body_groups))
        ref_mask = body_groups[ref_name].recompute_considered_

        inconsistent = []
        for name, lay in body_groups.items():
            if lay.recompute_considered_ != ref_mask:
                inconsistent.append(name)

        if inconsistent:
            raise ValueError(
                f"All BODY groups must share the same recompute_considered mask, "
                f"but groups {inconsistent} differ from '{ref_name}'. "
                f"Please ensure all BODY groups enable the same set of recompute types."
            )

    @staticmethod
    def _validate_group_names(layers: List[Any]) -> None:
        """Validate that body group names do not conflict with ILP solver internal variables.

        ``SappSolver._create_variables_to_solve_`` stores internal variables
        (e.g. ``max_stage_time``) and body-layer variables in the same
        ``variables_`` dict keyed by name.  A body group whose name matches
        a solver internal variable would overwrite the internal entry,
        causing ``'list' object has no attribute 'varValue'`` errors during
        result extraction.

        Args:
            layers: List of sapp-ppb Layer objects with ``.name_`` attribute.

        Raises:
            ValueError: If any group name conflicts with a solver reserved name.
        """
        from hyper_parallel.auto_parallel.sapp_ppb.sapp.sapp_solver import SappSolver  # pylint: disable=C0415
        pipeline_layer = _get_pipeline_layer_class()
        if pipeline_layer is None:
            return

        reserved = {
            v for k, v in vars(SappSolver).items()
            if isinstance(v, str) and k.isupper()
        }
        body_names = {
            lay.name_ for lay in layers
            if lay.type_ == pipeline_layer.type_enum.BODY
        }
        conflicts = body_names & reserved
        if conflicts:
            raise ValueError(
                f"body_groups names {conflicts} conflict with ILP solver "
                f"internal variables; please use different names"
            )

    def _validate_num_layer_consistency(self, layers: List[Any]) -> None:
        """Validate that YAML ``num_layer`` matches the actual body layer count from JSON.

        When ``num_layer`` is specified in the YAML config, it must agree with
        the total ``nb_layer_`` of all BODY layers in the JSON profile.  A
        mismatch would silently produce incorrect ILP results because the
        solver uses the JSON-derived count while the user expects the YAML
        value to be authoritative.

        Args:
            layers: List of sapp-ppb Layer objects with ``.type_`` and
                ``.nb_layer_`` attributes.

        Raises:
            ValueError: If ``num_layer`` is provided and does not match the
                total body layers from JSON.
        """
        if self.yaml_config.num_layer is None:
            return

        pipeline_layer = _get_pipeline_layer_class()
        if pipeline_layer is None:
            return

        actual_body = sum(
            lay.nb_layer_ for lay in layers
            if lay.type_ == pipeline_layer.type_enum.BODY
        )
        if actual_body != self.yaml_config.num_layer:
            raise ValueError(
                f"num_layer in YAML ({self.yaml_config.num_layer}) does not match "
                f"the total body layers from JSON ({actual_body}). "
                f"Please ensure both sources agree or omit num_layer in YAML."
            )
