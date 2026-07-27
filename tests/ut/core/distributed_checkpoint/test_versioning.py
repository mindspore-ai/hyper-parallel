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
"""UT for :mod:`hyper_parallel.core.distributed_checkpoint.versioning`."""
import unittest

from hyper_parallel.core.distributed_checkpoint.metadata import (
    BytesStorageMetadata,
    Metadata,
    TensorProperties,
    TensorStorageMetadata,
)
from hyper_parallel.core.distributed_checkpoint.versioning import (
    CURRENT_CHECKPOINT_VERSION,
    migrate_metadata,
)


class TestVersioning(unittest.TestCase):
    """Tests for checkpoint metadata version migration."""

    def test_current_version_passthrough(self):
        """
        Feature: migrate_metadata passthrough.
        Description: Metadata already at current version.
        Expectation: Returned as-is without modification.
        """
        md = Metadata(
            state_dict_metadata={"w": BytesStorageMetadata()},
            version=CURRENT_CHECKPOINT_VERSION,
        )
        result = migrate_metadata(md)
        self.assertIs(result, md)
        self.assertEqual(result.version, CURRENT_CHECKPOINT_VERSION)

    def test_missing_version_treated_as_1_0(self):
        """
        Feature: migrate_metadata handles missing instance-level version.
        Description: Old pickled Metadata without version field.
        Expectation: Treated as 1.0 and migrated to current version.
        """
        md = Metadata(state_dict_metadata={"w": BytesStorageMetadata()})
        del md.__dict__["version"]
        result = migrate_metadata(md)
        self.assertEqual(result.version, CURRENT_CHECKPOINT_VERSION)

    def test_1_0_to_2_0_migration(self):
        """
        Feature: migrate_metadata 1.0 to 2.0.
        Description: Metadata with explicit version "1.0".
        Expectation: Version upgraded to "2.0" and data preserved.
        """
        props = TensorProperties(dtype="float32")
        tensor_md = TensorStorageMetadata(properties=props, size=(4,))
        md = Metadata(
            state_dict_metadata={"w": tensor_md},
            version="1.0",
        )
        result = migrate_metadata(md)
        self.assertEqual(result.version, CURRENT_CHECKPOINT_VERSION)
        self.assertIn("w", result.state_dict_metadata)
        self.assertEqual(result.state_dict_metadata["w"].size, (4,))

    def test_unknown_version_raises(self):
        """
        Feature: migrate_metadata rejects unknown versions.
        Description: Metadata with version "0.5".
        Expectation: ValueError with source and target version info.
        """
        md = Metadata(
            state_dict_metadata={},
            version="0.5",
        )
        with self.assertRaises(ValueError) as ctx:
            migrate_metadata(md)
        msg = str(ctx.exception)
        self.assertIn("0.5", msg)
        self.assertIn(CURRENT_CHECKPOINT_VERSION, msg)

    def test_future_version_raises(self):
        """
        Feature: migrate_metadata rejects future versions.
        Description: Metadata with version "3.0" (beyond current).
        Expectation: ValueError mentioning the unknown version.
        """
        md = Metadata(
            state_dict_metadata={},
            version="3.0",
        )
        with self.assertRaises(ValueError) as ctx:
            migrate_metadata(md)
        self.assertIn("3.0", str(ctx.exception))

    def test_migration_no_progress_raises(self):
        """
        Feature: migrate_metadata detects no-progress rule.
        Description: Inject a rule that does not advance the version.
        Expectation: ValueError mentioning no progress.
        """
        from hyper_parallel.core.distributed_checkpoint import versioning as vmod

        def _bad_rule(metadata: Metadata) -> Metadata:
            metadata.version = "1.0"
            return metadata

        old_rules = vmod._MIGRATION_RULES.copy()
        vmod._MIGRATION_RULES["1.0"] = _bad_rule
        try:
            md = Metadata(state_dict_metadata={}, version="1.0")
            with self.assertRaises(ValueError) as ctx:
                migrate_metadata(md)
            self.assertIn("did not advance", str(ctx.exception))
        finally:
            vmod._MIGRATION_RULES = old_rules

    def test_migration_result_type_error(self):
        """
        Feature: migrate_metadata validates migration result type.
        Description: A rule returns a non-Metadata object.
        Expectation: ValueError mentioning type mismatch.
        """
        from hyper_parallel.core.distributed_checkpoint import versioning as vmod

        def _bad_type_rule(metadata: Metadata) -> dict:
            return {"version": "2.0"}

        old_rules = vmod._MIGRATION_RULES.copy()
        vmod._MIGRATION_RULES["1.0"] = _bad_type_rule
        try:
            md = Metadata(state_dict_metadata={}, version="1.0")
            with self.assertRaises(ValueError) as ctx:
                migrate_metadata(md)
            self.assertIn("dict", str(ctx.exception))
        finally:
            vmod._MIGRATION_RULES = old_rules


if __name__ == "__main__":
    unittest.main()
