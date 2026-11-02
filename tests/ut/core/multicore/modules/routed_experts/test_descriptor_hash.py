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
"""CPU checks for immutable route value hashing and transient cache state."""

from copy import copy, deepcopy
from dataclasses import FrozenInstanceError, asdict, fields, replace
import pickle
import unittest
from unittest.mock import patch

from hyper_parallel.core.multicore.modules.routed_experts.descriptor import RouteDescriptor, RouteEdge
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestDescriptorHash(unittest.TestCase):
    """Hash caches must preserve value identity, serialization and immutability."""

    @staticmethod
    def _route() -> RouteDescriptor:
        """Create one valid local route with a shared incoming/outgoing edge."""
        edge = RouteEdge(0, 0, 0, 0, 0, 3)
        return RouteDescriptor(0, 2, 3, 3, (0,), (3,), (edge,), (edge,))

    @staticmethod
    def _structural_hash(route: RouteDescriptor) -> int:
        """Reproduce the original generated dataclass hash from declared fields."""
        return hash(tuple(getattr(route, field.name) for field in fields(route)))

    def test_hash_matches_declared_values_and_equal_dictionary_key(self) -> None:
        """Distinct equal objects must resolve the same dictionary entry."""
        route, equivalent = self._route(), self._route()
        expected = self._structural_hash(route)
        self.assertEqual(hash(route), expected)
        self.assertEqual(hash(equivalent), expected)
        self.assertIsNot(route, equivalent)
        self.assertEqual({route: "metadata"}[equivalent], "metadata")
        self.assertEqual(route, equivalent)

    def test_nested_edges_are_hashed_only_on_first_lookup(self) -> None:
        """Repeated direction keys reuse one structural hash for the route."""
        route = self._route()
        original, calls = RouteEdge.__hash__, []

        def counted(edge: RouteEdge) -> int:
            """Track edge visits without changing the structural hash value."""
            calls.append(edge)
            return original(edge)

        with patch.object(RouteEdge, "__hash__", counted):
            hash(route)
            initial = len(calls)
            for phase in (False, True, "readiness") * 6:
                hash((route, phase))
        self.assertEqual(initial, 2)
        self.assertEqual(len(calls), initial)

    def test_cache_is_absent_from_fields_repr_and_asdict(self) -> None:
        """Caching must not add a public constructor or equality field."""
        route = self._route()
        names = tuple(field.name for field in fields(route))
        state, representation = asdict(route), repr(route)
        hash(route)
        self.assertEqual(names, ("rank", "ranks", "source_rows", "received_rows", "active_experts",
                                 "boundaries", "outgoing", "incoming"))
        self.assertEqual(asdict(route), state)
        self.assertEqual(repr(route), representation)
        self.assertEqual(route, self._route())

    def test_pickle_omits_cache_and_preserves_value_identity(self) -> None:
        """A cached object serializes identically to its original uncached state."""
        route = self._route()
        before = pickle.dumps(route)
        expected = hash(route)
        after = pickle.dumps(route)
        self.assertEqual(before, after)
        restored = pickle.loads(after)
        self.assertNotIn("_value_hash", restored.__dict__)
        self.assertEqual(restored, route)
        self.assertEqual(hash(restored), expected)

    def test_copies_and_replacement_recompute_from_their_own_fields(self) -> None:
        """Copies discard derived state and changed fields cannot inherit a stale hash."""
        route = self._route()
        expected = hash(route)
        for duplicate in (copy(route), deepcopy(route)):
            self.assertNotIn("_value_hash", duplicate.__dict__)
            self.assertEqual(duplicate, route)
            self.assertEqual(hash(duplicate), expected)
        changed = replace(route, ranks=3)
        self.assertNotIn("_value_hash", changed.__dict__)
        self.assertNotEqual(changed, route)
        self.assertEqual(hash(changed), self._structural_hash(changed))

    def test_empty_route_and_frozen_fields_keep_original_contract(self) -> None:
        """An empty rank remains hashable and public fields stay frozen."""
        route = RouteDescriptor(1, 2, 0, 0, (), (), (), ())
        route.validate()
        self.assertEqual(hash(route), self._structural_hash(route))
        with self.assertRaises(FrozenInstanceError):
            route.rank = 0

    def test_hash_collision_still_uses_value_equality(self) -> None:
        """A cached integer remains a hash, never an identity key."""
        first, second = self._route(), replace(self._route(), ranks=3)
        with patch.object(RouteDescriptor, "_value_hash", 0):
            values = {first: "first", second: "second"}
            self.assertEqual(len(values), 2)
            self.assertEqual(values[self._route()], "first")
            self.assertEqual(values[replace(self._route(), ranks=3)], "second")
