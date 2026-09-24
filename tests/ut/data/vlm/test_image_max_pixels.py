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
"""``image_max_pixels`` area-budget resizing for the Kimi VLM transform.

Left alone, the transform hands images to the processor at the processor's own
patch-unit budget: a 1024x1024 image is accepted unresized as a 74x74 patch grid
= 1369 media tokens, so ten of them overflow a 12288-slot window. The
``image_max_pixels`` option applies MindSpeed-MM's ``_preprocess_image`` contract
instead -- an *area* (pixel-count) budget, both sides scaled by
``sqrt(budget / area)`` and truncated with ``int()``, aspect ratio preserved,
never upscaled. No patch-grid rounding happens here: the processor rounds each
side up to ``patch_size * merge_size`` itself, so a 512x512 result is rendered as
a 38x38 patch grid (361 media tokens). These tests pin the resize step on a stub
processor, so no checkpoint, tokenizer or Hub access is involved.
"""
# pylint: disable=wrong-import-position

import importlib.util
import os
import pathlib
import tempfile
import unittest

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

import torch

from tests.common.mark_utils import arg_mark

# The VLM facade pulls transformers.AutoProcessor, whose lazy import chain ends
# in torchvision; some CI pythons lack liblzma and cannot import it.
_HAS_LZMA = importlib.util.find_spec("_lzma") is not None
# The transform imports PIL lazily, so probe it before relying on it too.
_HAS_PIL = importlib.util.find_spec("PIL") is not None


class _StubImageProcessor:
    """Image-processor stub carrying the fields the real processor exposes."""

    patch_size = 14
    merge_size = 2


class _StubProcessor:
    """Processor stand-in: no checkpoint, no tokenizer, no Hub access.

    It mirrors the real ``kimi_k25`` shape (an ``image_processor`` exposing
    ``patch_size``/``merge_size``) so that a regression back to patch-grid
    rounding would shift the asserted sizes instead of passing silently.
    """

    image_processor = _StubImageProcessor()


def _image(size, color=(16, 32, 64)):
    """Build a plain RGB PIL image of ``size``."""
    from PIL import Image  # pylint: disable=C0415

    return Image.new("RGB", size, color=color)


def _transform(image_max_pixels):
    """Build the transform over the stub processor for one pixel budget."""
    from hyper_parallel.data.omni.kimi_transform import KimiVLMChatTransform

    return KimiVLMChatTransform(processor=_StubProcessor(),
                                image_max_pixels=image_max_pixels)


@unittest.skipIf(not (_HAS_LZMA and _HAS_PIL), "needs liblzma (_lzma) and PIL")
class TestResizeMediaAreaBudget(unittest.TestCase):
    """``_resize_media`` applies MindSpeed's area budget to a single image."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_square_image_matches_mindspeed_area_math(self):
        """A 1024x1024 image at 262144 px comes back 512x512 (sqrt(1/4) scale)."""
        out = _transform(262144)._resize_media(_image((1024, 1024)))

        self.assertEqual(out.size, (512, 512))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_non_square_image_keeps_aspect_and_truncates_both_sides(self):
        """Both sides are scaled by the same factor and truncated with ``int()``."""
        # sqrt(262144 / 524288) = 0.70710678: 1024 -> 724, 512 -> 362.
        out = _transform(262144)._resize_media(_image((1024, 512)))

        self.assertEqual(out.size, (724, 362))
        self.assertLessEqual(out.size[0] * out.size[1], 262144)
        self.assertAlmostEqual(out.size[0] / out.size[1], 1024 / 512, places=2)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_half_scale_budget_yields_exact_halves(self):
        """At half the area budget the scale factor is exactly 0.5."""
        out = _transform(131072)._resize_media(_image((1024, 512)))

        self.assertEqual(out.size, (512, 256))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_image_within_budget_is_never_upscaled(self):
        """An image at or under the budget keeps its size and its pixels."""
        transform = _transform(262144)
        for size in ((512, 512), (200, 100)):
            with self.subTest(size=size):
                image = _image(size)

                out = transform._resize_media(image)

                self.assertEqual(out.size, size)
                self.assertEqual(out.getpixel((0, 0)), image.getpixel((0, 0)))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_str_path_is_opened_and_resized(self):
        """A path string is opened, converted to RGB and resized in memory."""
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "big.png"
            _image((1024, 1024)).save(path)

            out = _transform(262144)._resize_media(str(path))

        self.assertEqual(out.size, (512, 512))
        self.assertEqual(out.mode, "RGB")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_non_image_media_is_returned_as_is(self):
        """Tensors, arrays and path-like objects are left for the processor."""
        transform = _transform(262144)
        tensor = torch.zeros(3, 1024, 1024)
        for media in (tensor, 7, pathlib.Path("clip.png")):
            with self.subTest(media=type(media).__name__):
                self.assertIs(transform._resize_media(media), media)


@unittest.skipIf(not (_HAS_LZMA and _HAS_PIL), "needs liblzma (_lzma) and PIL")
class TestResizeMediaImagesRewrite(unittest.TestCase):
    """``_resize_media_images`` rewrites only ``image`` content items."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_rewrites_only_image_items(self):
        """Text and video items, string content and content-less turns survive."""
        image = _image((1024, 1024))
        text_item = {"type": "text", "text": "look"}
        image_item = {"type": "image", "url": image}
        video_item = {"type": "video", "url": "clip.mp4"}
        plain = {"role": "assistant", "content": "plain string"}
        empty = {"role": "system"}
        messages = [
            {"role": "user", "content": [text_item, image_item, video_item]},
            plain,
            empty,
        ]

        out = _transform(262144)._resize_media_images(messages)

        self.assertIsNot(out, messages)
        self.assertIs(out[1], plain)
        self.assertIs(out[2], empty)
        self.assertIs(out[0]["content"][0], text_item)
        self.assertIs(out[0]["content"][2], video_item)
        self.assertEqual(out[0]["content"][1]["url"].size, (512, 512))
        self.assertEqual(out[0]["content"][1]["type"], "image")
        # The caller's input is not mutated in place.
        self.assertEqual(image_item["url"].size, (1024, 1024))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_unset_budget_is_an_exact_no_op(self):
        """``image_max_pixels=None`` returns the very same messages object."""
        messages = [
            {"role": "user", "content": [{"type": "image", "url": _image((1024, 1024))}]},
        ]
        transform = _transform(None)

        self.assertIs(transform._resize_media_images(messages), messages)
        self.assertEqual(
            transform._resize_media(messages[0]["content"][0]["url"]).size, (1024, 1024)
        )


@unittest.skipIf(not (_HAS_LZMA and _HAS_PIL), "needs liblzma (_lzma) and PIL")
class TestImageMaxPixelsOption(unittest.TestCase):
    """The option is validated on the class and forwarded by the builder."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_rejects_non_positive_or_non_integer_budget(self):
        """``0``, negatives, bools, floats and strings all fail fast."""
        for bad in (0, -1, True, 2.5, "512"):
            with self.subTest(image_max_pixels=bad):
                with self.assertRaisesRegex(ValueError, "must be a positive integer"):
                    _transform(bad)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_accepts_none_and_positive_integer(self):
        """``None`` disables resizing; a positive int is stored as given."""
        self.assertIsNone(_transform(None).image_max_pixels)
        self.assertEqual(_transform(262144).image_max_pixels, 262144)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_builder_forwards_the_option_and_patch_rounding_is_gone(self):
        """The builder wires ``image_max_pixels`` and no patch grid is needed."""
        from hyper_parallel.data.omni.kimi_transform import (
            KimiVLMChatTransform,
            build_kimi_vlm_data_transform,
        )

        transform = build_kimi_vlm_data_transform(
            processor=_StubProcessor(), image_max_pixels=262144
        )

        self.assertEqual(transform.image_max_pixels, 262144)
        self.assertFalse(hasattr(transform, "_resize_factor"))
        # The processor's patch/merge fields are no longer read, so a bare object
        # is enough to configure the resize.
        bare = KimiVLMChatTransform(processor=object(), image_max_pixels=262144)
        self.assertEqual(bare._resize_media(_image((1024, 1024))).size, (512, 512))


if __name__ == "__main__":
    unittest.main()
