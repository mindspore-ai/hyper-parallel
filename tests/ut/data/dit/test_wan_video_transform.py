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
"""UT for the Wan online video transform's raw-sample filter."""
import unittest

from hyper_parallel.data.dit.wan_video import WanVideoTransform


class TestWanVideoTransformIsValidSample(unittest.TestCase):
    """Filter behavior of :meth:`WanVideoTransform.is_valid_sample`."""

    def _transform(self, **kwargs):
        return WanVideoTransform(task=kwargs.pop("task", "t2v"), **kwargs)

    def _check(self, transform, sample):
        valid = transform.is_valid_sample(sample)
        # A record that passes the filter must also survive _preprocess, which is what the
        # filter promises the source loader.
        if valid:
            prompt, _, videos = transform._preprocess(sample)  # pylint: disable=protected-access
            self.assertTrue(bool(prompt) and bool(videos))
        return valid

    def test_record_with_prompt_and_video_is_valid(self):
        """A prompt plus a video passes, whatever key each is stored under."""
        transform = self._transform()
        self.assertTrue(self._check(transform, {"prompt": "a cat", "video_bytes": b"..." }))
        self.assertTrue(self._check(transform, {"caption": "a cat", "video_path": "cat.mp4"}))

    def test_record_missing_prompt_is_invalid(self):
        """Without any prompt key holding a value, the record is filtered out."""
        transform = self._transform()
        self.assertFalse(self._check(transform, {"video_bytes": b"..."}))
        self.assertFalse(self._check(transform, {"prompt": "", "video_bytes": b"..."}))

    def test_record_missing_video_is_invalid(self):
        """Without any video key holding a value, the record is filtered out."""
        transform = self._transform()
        self.assertFalse(self._check(transform, {"prompt": "a cat"}))
        self.assertFalse(self._check(transform, {"prompt": "a cat", "video": None}))

    def test_image_is_optional_for_both_tasks(self):
        """The filter demands no image: i2v falls back to the video's first frame."""
        for task in ("t2v", "i2v"):
            transform = self._transform(task=task)
            self.assertTrue(
                self._check(transform, {"prompt": "a cat", "video_bytes": b"..."})
            )

    def test_tom_and_jerry_fast_path_record_is_valid(self):
        """The dataset's prompt/video_bytes fast path satisfies the filter."""
        transform = self._transform(source_name="Tom-and-Jerry-VideoGeneration-Dataset")
        self.assertTrue(
            self._check(transform, {"prompt": "a cat", "video_bytes": b"..."})
        )
