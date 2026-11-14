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
"""UT for :mod:`hyper_parallel.core.distributed_checkpoint.checkpoint_io`."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from hyper_parallel.core.distributed_checkpoint.checkpoint_io import (
    load_checkpoint_file,
    save_checkpoint_file,
    unshare_storage,
)
from tests.common.mark_utils import arg_mark


class TestUnshareStorage(unittest.TestCase):
    """Aliasing entries are copied apart; everything else is left alone."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_tied_entries_stop_sharing_and_keep_their_values(self):
        """The tied-embedding case: both names survive with equal values."""
        embedding = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        state = {"embed.weight": embedding, "lm_head.weight": embedding}

        unshared = unshare_storage(state)

        self.assertEqual(set(unshared), set(state))
        pointers = {name: tensor.untyped_storage().data_ptr()
                    for name, tensor in unshared.items()}
        self.assertNotEqual(pointers["embed.weight"], pointers["lm_head.weight"])
        for name in state:
            self.assertTrue(torch.equal(unshared[name], state[name]), name)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_independent_entries_are_returned_untouched(self):
        """No aliasing means no copies: the same objects come back."""
        state = {"a": torch.ones(2), "b": torch.zeros(3)}
        unshared = unshare_storage(state)
        for name, tensor in state.items():
            self.assertIs(unshared[name], tensor)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_a_view_of_another_entry_is_copied_with_its_own_values(self):
        """A slice aliases its base; the copy keeps the slice's contents."""
        base = torch.arange(6, dtype=torch.float32)
        state = {"base": base, "tail": base[3:]}
        unshared = unshare_storage(state)
        self.assertNotEqual(unshared["base"].untyped_storage().data_ptr(),
                            unshared["tail"].untyped_storage().data_ptr())
        self.assertTrue(torch.equal(unshared["tail"], base[3:]))
        self.assertTrue(unshared["tail"].is_contiguous())

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_empty_tensors_are_not_treated_as_aliases(self):
        """Storage-less entries coincide trivially and need no copy."""
        state = {"x": torch.empty(0), "y": torch.empty(0)}
        unshared = unshare_storage(state)
        self.assertIs(unshared["x"], state["x"])
        self.assertIs(unshared["y"], state["y"])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_non_tensor_values_pass_through(self):
        state = {"step": 7, "w": torch.ones(2)}
        self.assertEqual(unshare_storage(state)["step"], 7)


class TestSaveTiedCheckpoint(unittest.TestCase):
    """The round trip safetensors used to refuse.

    Positive control for the fix: the same state_dict handed straight to
    ``safetensors.torch.save_file`` raises, which is the failure a bring-up
    met only at the end of a thousand-step run.
    """

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_tied_state_dict_saves_and_loads_back_equal(self):
        embedding = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        state = {"embed.weight": embedding, "lm_head.weight": embedding,
                 "norm.weight": torch.ones(4)}
        with TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "shard.safetensors")

            from safetensors.torch import save_file
            with self.assertRaises(RuntimeError):
                save_file(tensors=state, filename=str(Path(tmp) / "raw.safetensors"))

            save_checkpoint_file(state, path)
            # Detach from the loader's mapping before the directory goes away.
            restored = {name: tensor.clone()
                        for name, tensor in load_checkpoint_file(path).items()}

        self.assertEqual(set(restored), set(state))
        for name in state:
            self.assertTrue(torch.equal(restored[name], state[name]), name)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_the_pickle_format_is_unaffected(self):
        embedding = torch.ones(2, 2)
        state = {"a": embedding, "b": embedding}
        with TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "shard.pt")
            save_checkpoint_file(state, path, ckpt_format="pickle")
            restored = load_checkpoint_file(path, ckpt_format="pickle")
        for name in state:
            self.assertTrue(torch.equal(restored[name], state[name]), name)
