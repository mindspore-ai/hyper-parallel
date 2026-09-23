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
"""Declarative data classes for shard op cases — no platform deps."""
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np

# A placement tuple, one entry per tensor dimension.
# Entries are Placement objects (Shard/Replicate) from hyper_parallel.
PlacementSpec = Tuple[Any, ...]


@dataclass(frozen=True)
class InputSpec:
    """Declarative spec for a single tensor input.

    The framework constructs a numpy array using ``init`` and ``seed``, then
    asks the backend to materialize it on device. ``range`` is used only by
    ``init="uniform"``. When ``data`` is provided it takes precedence over
    all other fields — the given array is used as-is.
    """
    shape: Tuple[int, ...]
    dtype: str = "float32"        # platform-neutral name; backend maps to its dtype
    init: str = "randn"           # "randn" | "uniform" | "ones" | "zeros" | "arange"
    seed: Optional[int] = None
    range: Optional[Tuple[float, float]] = None
    data: Optional[np.ndarray] = None   # exact values; overrides init/seed


@dataclass(frozen=True)
class DerivedSpec:
    """A derived input computed once from the full primary tensors.

    Some ops consume tensors that are *derived* from the primary inputs via a
    device op needing global information — e.g. attention ``sm_max/sm_sum`` from
    ``FlashAttentionScore`` over the full K dimension. These cannot be faked as
    static data, nor recomputed inside ``fn`` on sharded inputs (the derive op
    would see inconsistently-sharded tensors). Instead the framework computes
    ``fn(*full_primary_tensors)`` once on the full tensors, then distributes the
    result with ``placement`` for the parallel path — matching the old tests
    that pre-compute stats on full tensors and ``distribute_tensor`` them.

    The derived tensors are appended after the primary tensors (and before
    ``extra_inputs``) in the call: ``fn(*primary, *derived, *extra, **kwargs)``.
    """
    fn: Callable                  # (*full_primary_tensors) -> derived tensor
    placement: PlacementSpec      # how to shard the derived tensor for the parallel path


@dataclass(frozen=True)
class CompareSpec:
    """How to compare standalone vs distributed output."""
    kind: str = "allclose"        # "equal" | "allclose" | "shape"
    rtol: float = 1e-5
    atol: float = 1e-5

    @classmethod
    def equal(cls) -> "CompareSpec":
        """Exact equality (``torch.equal`` / ``np.array_equal``)."""
        return cls(kind="equal")

    @classmethod
    def allclose(cls, rtol: float = 1e-5, atol: float = 1e-5) -> "CompareSpec":
        """Floating-point closeness with given tolerances."""
        return cls(kind="allclose", rtol=rtol, atol=atol)

    @classmethod
    def shape(cls) -> "CompareSpec":
        """Shape (and dtype) only — values are not compared.

        For ops whose values are inherently non-deterministic or undefined
        (random dropout, uninitialized ``empty_like``); the gathered output's
        shape/dtype is still verified against the single-card reference.
        """
        return cls(kind="shape")


@dataclass
class OpShardCase:
    """A single op case — the unit of execution and selection.

    The framework runs the case as:
        derived  = [d.fn(*full_tensors) for d in derived_inputs]
        ref = fn(*full_tensors, *derived,      *extra_inputs, **kwargs)
        out = fn(*dist_tensors,  *dist_derived, *extra_inputs, **kwargs)
    then compares ``ref`` and ``gathered(out)`` via ``compare``. ``derived`` is
    computed once on the full tensors; ``dist_derived`` is the same data sliced
    per each ``DerivedSpec.placement`` (never recomputed on shards).

    When ``needs_mesh`` is True the signature becomes:
        fn(mesh, *tensors, *derived, *extra_inputs, **kwargs)

    ``tags`` carries platform-level routing entries like ``("cpu_level1",
    "npu_level1")``. ``build_suite_groups(tag_include=...)`` selects cases
    whose tags intersect the include set — no separate ``level`` field is
    needed.
    """
    name: str
    fn: Callable
    inputs: Sequence[InputSpec]
    placements: Sequence[PlacementSpec]
    kwargs: dict = field(default_factory=dict)
    extra_inputs: list = field(default_factory=list)
    derived_inputs: Sequence[DerivedSpec] = ()
    compare: CompareSpec = field(default_factory=CompareSpec.allclose)
    needs_mesh: bool = False
    tags: Tuple[str, ...] = ()
    mesh_shape: Optional[Tuple[int, ...]] = None
    """Device-mesh shape, one entry per mesh axis (``(2,)`` / ``(2,2)`` /
    ``(2,2,2)`` → 2 / 4 / 8 cards). **Always honored as written**: it sets
    ``num_proc`` (``math.prod``) and the topology the mesh is built with.
    Canonicalization (see ``mesh_dim_names``) never touches the shape. Defaults
    to ``build_suite_groups``'s ``mesh_shape`` arg when left ``None``."""
    mesh_dim_names: Optional[Tuple[str, ...]] = None
    """Names for the mesh axes.

    For ordinary cases (``needs_mesh=False`` — the vast majority) these are
    **pure labels**: ``distribute`` applies placements by axis *index* and the
    ``fn`` never reads names. So ``_bucket_key`` canonicalizes them by ndim
    (1D→``("dp",)``, 2D→``("dp","tp")``, 3D→``("dp","cp","tp")``) and the mesh is
    built with the canonical names — **whatever you write here is overridden at
    run time** and only serves as in-code documentation. You may write
    descriptive names (``("sp",)``) for readability, or omit it entirely.

    For MC2 ops (``needs_mesh=True``) the names ARE live: the ``fn`` resolves its
    communication group via ``mesh.get_group(group_dim)``, so they are kept
    as-written (not canonicalized) and **must match the ``group_dim`` in
    ``kwargs``**."""
    num_proc: Optional[int] = None
    """Process/card count for this case. Defaults to ``math.prod(mesh_shape)``
    when ``None``; only set it explicitly if it must differ from the mesh size."""
    source_module: str = field(default="", repr=False)
    compare_outputs: Optional[Tuple[int, ...]] = None
    solo_launcher: bool = False
    """Run this case in its own launcher process (never packed with others).

    Escape hatch for a CANN MC2 limit: two or more *4-card-or-larger* MC2 fusion
    ops (``all_gather_matmul`` / ``matmul_reduce_scatter``) in one process fail to
    set up their CANN sub-comm groups (4-card repro: ``NnopbaseExecutorGetMc2Num
    failed``; 8-card 3-D: ``HcclAllocComResourceByTiling`` ret=15 / E39999).
    2-card MC2 ops may share a process freely. **Currently no case sets this** —
    4-card MC2 cases are kept ≤1 per group by the deterministic
    ``(source_module, name)`` sort, and the redundant 3-D MC2 cases were removed.
    Set it only if a new MC2 case cannot otherwise be kept alone in its group."""


@dataclass
class OpSpec:
    """Op-level declaration for ``register_op_family``."""
    name: str
    fn: Callable
    default_compare: CompareSpec = field(default_factory=CompareSpec.allclose)
    default_input: dict = field(default_factory=lambda: {"init": "randn"})
    tags: Tuple[str, ...] = ()


@dataclass
class CaseSpec:
    """Configuration-level declaration — op-agnostic.

    Cross-multiplied with a list of ``OpSpec`` to produce ``OpShardCase``
    instances named ``"{op.name}[{case.name}]"``.
    """
    name: str
    shapes: Sequence[Tuple[int, ...]]
    placements: Sequence[PlacementSpec]
    extra_inputs: list = field(default_factory=list)
    kwargs: dict = field(default_factory=dict)
    seed: int = 1
    init_override: Optional[str] = None
    compare_override: Optional[CompareSpec] = None
    only_for: Tuple[str, ...] = ()
    skip_for: Tuple[str, ...] = ()
