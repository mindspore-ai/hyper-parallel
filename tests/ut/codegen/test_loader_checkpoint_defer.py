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
"""Regression tests for generated-checkpoint loading (org-issues#43571).

A meta-constructed generated model must not be materialized on CPU nor loaded
from the checkpoint inside ``_load_generated_checkpoint``: doing so allocates
the full parameter set on every rank (host OOM for large models) and the
weights are discarded anyway by the infrastructure stage's ``to_empty``.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from torch import nn

from hyper_parallel.codegen import loader
from hyper_parallel.models._transformers import checkpoint_loader
from tests.common.mark_utils import arg_mark


class _TinyGeneratedModel(nn.Module):
    """Stand-in for a generated modeling class."""

    def __init__(self) -> None:
        """Build a single linear layer to stand in for generated weights."""
        super().__init__()
        self.proj = nn.Linear(4, 4)


class TestGeneratedCheckpointLoading(unittest.TestCase):
    """Generated-model loading defers to the infrastructure stage on meta."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_meta_model_is_not_materialized_or_loaded(self) -> None:
        """A meta model returns immediately: no to_empty, no checkpoint load."""
        model_cls = mock.Mock(spec_set=["_from_config"])
        model_cls._from_config.return_value = _TinyGeneratedModel().to_empty(
            device="meta"
        )
        manager = mock.Mock()
        with mock.patch.object(
            checkpoint_loader, "CheckpointManager", return_value=manager
        ) as manager_cls, mock.patch.object(
            loader, "_source_model_shapes", return_value={}
        ):
            result = loader._load_generated_checkpoint(  # pylint: disable=protected-access
                model_cls,
                "/tmp/pretrained",
                SimpleNamespace(),
                (),
                "auto",
                {},
            )

        self.assertTrue(all(p.is_meta for p in result.parameters()))
        # The deferred load means neither the manager nor its loader ran.
        manager_cls.assert_not_called()
        manager.load_checkpoint.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_real_model_still_loads_checkpoint(self) -> None:
        """A non-meta (single-process) model keeps the eager load."""
        model_cls = mock.Mock(spec_set=["_from_config"])
        model_cls._from_config.return_value = _TinyGeneratedModel()
        manager = mock.Mock()
        with mock.patch.object(
            checkpoint_loader, "CheckpointManager", return_value=manager
        ) as manager_cls, mock.patch.object(
            loader, "_source_model_shapes", return_value={}
        ):
            loader._load_generated_checkpoint(  # pylint: disable=protected-access
                model_cls,
                "/tmp/pretrained",
                SimpleNamespace(),
                (),
                "auto",
                {},
            )

        manager_cls.assert_called_once()
        manager.load_checkpoint.assert_called_once_with(
            "/tmp/pretrained", strict=False
        )


if __name__ == "__main__":
    unittest.main()
