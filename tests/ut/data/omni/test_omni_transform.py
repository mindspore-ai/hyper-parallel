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
"""CPU tests for external image binding before processor encoding."""

import copy
import io
import unittest

from PIL import Image

from hyper_parallel.data.omni.omni_transform import AutoProcessorTransform
from tests.common.mark_utils import arg_mark


def image_bytes(color: str, mode: str = "RGB") -> bytes:
    """Build a tiny real PNG without accessing external files."""
    with io.BytesIO() as buffer:
        Image.new("RGB", (2, 3), color).convert(mode).save(buffer, format="PNG")
        return buffer.getvalue()


class TestOmniImageBinding(unittest.TestCase):
    """Verify image ordering, error handling, and legacy input preservation."""

    def setUp(self) -> None:
        """Build a transform for message preparation independently of tokenization."""
        self.transform = AutoProcessorTransform(max_seq_len=4096, processor=object())

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_bytes_bind_across_messages_and_text_blocks(self):
        """Verify image order across a conversation.

        Feature: External image binding.
        Description: Bind three colored images across strings and text blocks.
        Expectation: Pixels follow placeholder order without mutating source records.
        """
        sample = {
            "messages": [
                {"role": "user", "content": "before<image>\nafter"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": [{"type": "text", "text": "<image><image>compare"}]},
            ],
            "images": [{"bytes": image_bytes(color), "path": "missing.png"}
                       for color in ("red", "green", "blue")],
        }
        before = copy.deepcopy(sample)
        messages = self.transform.prepare_messages(sample)
        self.assertEqual(sample, before)
        self.assertEqual(messages[0]["content"][0], {"type": "text", "text": "before"})
        self.assertEqual(messages[0]["content"][2], {"type": "text", "text": "\nafter"})
        self.assertEqual(messages[1]["content"], [{"type": "text", "text": "answer"}])
        self.assertEqual(messages[2]["content"][-1], {"type": "text", "text": "compare"})
        images = [block["image"] for message in (messages[0], messages[2])
                  for block in message["content"] if block["type"] == "image"]
        self.assertEqual([image.getpixel((0, 0)) for image in images], [(255, 0, 0), (0, 128, 0), (0, 0, 255)])
        self.assertTrue(all(image.mode == "RGB" and image.size == (2, 3) for image in images))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_path_fallback_and_bare_image_block(self):
        """Verify path-only external images.

        Feature: External image path fallback.
        Description: Mix explicit media and external relative/remote paths.
        Expectation: Only placeholders consume external images; relative paths resolve.
        """
        sample = {
            "__online_source_path__": "/data/shard.parquet",
            "messages": [{"role": "user", "content": [
                {"type": "image", "image": "explicit.png"},
                {"type": "image"}, {"type": "text", "text": "<image>"},
            ]}],
            "images": [{"bytes": None, "path": "relative.png"}, "https://example.com/img.png"],
        }
        content = self.transform.prepare_messages(sample)[0]["content"]
        self.assertEqual([block["image"] for block in content],
                         ["/data/explicit.png", "/data/relative.png", "https://example.com/img.png"])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_image_count_mismatch_is_not_silent(self):
        """Verify image-count validation.

        Feature: External image validation.
        Description: Supply missing or unused images.
        Expectation: Every mismatch raises before processor encoding.
        """
        for text, images in (("<image>", []), ("<image><image>", ["a.png"]), ("no placeholder", ["a.png"])):
            with self.subTest(text=text, images=images):
                with self.assertRaisesRegex(ValueError, "image.*(count|placeholder)|placeholder.*image"):
                    self.transform.prepare_messages({"messages": [{"content": text}], "images": images})

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_bad_images_fail_with_index(self):
        """Verify malformed-image diagnostics.

        Feature: External image validation.
        Description: Supply corrupt bytes or missing image sources.
        Expectation: Errors identify the image index without path fallback.
        """
        for item in ({"bytes": b"not an image", "path": "a.png"}, {"bytes": b""},
                     {"bytes": "base64"}, {"bytes": None, "path": None}, None):
            with self.subTest(item=item), self.assertRaisesRegex((ValueError, TypeError), r"images\[0\]"):
                self.transform.prepare_messages({"messages": [{"content": "<image>"}], "images": [item]})

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_decode_grayscale_to_rgb(self):
        """Verify decoded image ownership.

        Feature: Image bytes decoding.
        Description: Decode a grayscale PNG and inspect pixels after buffer closure.
        Expectation: The returned RGB image owns usable pixel data.
        """
        sample = {"messages": [{"content": "<image>"}], "images": [{"bytes": image_bytes("white", "L")}]}
        image = self.transform.prepare_messages(sample)[0]["content"][0]["image"]
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.getpixel((1, 2)), (255, 255, 255))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_arrow_null_image_fields_are_not_extra_sources(self):
        """Verify nullable Arrow image structs.

        Feature: Parquet image normalization.
        Description: Prepare image blocks containing absent fields represented as nulls.
        Expectation: Null fields do not become extra processor image sources.
        """
        sample = {
            "__online_source_path__": "/data/shard.parquet",
            "messages": [{"content": [
                {"type": "image", "path": "a.png", "url": None, "image": None},
                {"type": "image", "path": None, "url": None, "image": None},
            ]}],
            "images": [{"bytes": image_bytes("red")}],
        }
        content = self.transform.prepare_messages(sample)[0]["content"]
        self.assertEqual(content[0], {"type": "image", "path": "/data/a.png"})
        self.assertEqual(set(content[1]), {"type", "image"})

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_no_images_preserves_legacy_messages(self):
        """Verify legacy message compatibility.

        Feature: Omni format compatibility.
        Description: Prepare messages without an external image list.
        Expectation: Ordinary strings and explicit image paths remain unchanged.
        """
        messages = [{"role": "user", "content": "question"},
                    {"role": "user", "content": [{"type": "image", "image": "a.png"}]},
                    {"role": "assistant", "content": "answer"}]
        for images in (None, []):
            with self.subTest(images=images):
                self.assertEqual(self.transform.prepare_messages({"messages": messages, "images": images}), messages)
