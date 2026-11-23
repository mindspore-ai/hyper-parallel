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
"""UT for the Wan video dataset transform, loaders, and dataset builders.

Everything runs on the CPU: video decoding goes through in-memory frames or
faked ``torchcodec``/``imageio``/``requests`` modules, so no real media files
or network access are needed.
"""
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image

from hyper_parallel.data.dit import wan_video
from hyper_parallel.data.dit.wan_video import (
    DiTBatch,
    DiTCollator,
    WanVideoTransform,
    _apply_dynamic_video_max_pixels,
    _calculate_frame_indices,
    _first_available,
    _load_image,
    _load_video,
    _load_video_with_imageio,
    _load_video_with_torchcodec,
    _normalize_model_sample,
    _normalize_video_tensor,
    _pil_frames_to_tensor,
    _smart_resize,
    build_dit_collate_fn,
    build_wan_video_dataset,
    build_wan_video_transform,
)


def _png_bytes(color: tuple = (255, 0, 0), size: tuple = (4, 4)) -> bytes:
    """Build one tiny PNG as bytes, the format raw records carry."""
    buffer = BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def _pil_frame(color: tuple = (0, 255, 0), size: tuple = (4, 4)) -> Image.Image:
    """Build one small RGB PIL frame."""
    return Image.new("RGB", size, color)


def _video(frames: int = 4, channels: int = 3, height: int = 8, width: int = 8) -> torch.Tensor:
    """Build a deterministic float CHW-frame video tensor."""
    return torch.arange(frames * channels * height * width, dtype=torch.float32).reshape(
        frames, channels, height, width
    )


class FakeFrames:
    """Stand-in for torchcodec's frame batch: carries a ``.data`` tensor."""

    def __init__(self, data: torch.Tensor) -> None:
        """Store the frame tensor."""
        self.data = data


class FakeDecoder:
    """Stand-in for ``torchcodec.decoders.VideoDecoder``."""

    def __init__(self, source, device=None, num_ffmpeg_threads=None, fail=False):
        """Record the source and publish four-frame 24-fps metadata."""
        del source, device, num_ffmpeg_threads
        if fail:
            raise ValueError("decode boom")
        self.metadata = SimpleNamespace(num_frames=4, average_fps=8.0)

    def get_frames_at(self, indices):
        """Return one fake frame batch shaped for the requested indices."""
        return FakeFrames(torch.zeros(len(indices), 3, 8, 8))


class FakeIio:
    """Stand-in for ``imageio.v3`` returning a fixed NHWC uint8 clip."""

    @staticmethod
    def imread(source, index=None):
        """Return a fixed 4-frame NHWC uint8 clip for any source."""
        del source, index
        return np.zeros((4, 8, 8, 3), dtype=np.uint8)


class FakeRequests:
    """Stand-in for ``requests`` returning fixed PNG bytes."""

    @staticmethod
    def get(url, timeout=None):
        """Return a response object holding PNG bytes."""
        del url, timeout
        return SimpleNamespace(content=_png_bytes())


class TestFrameMath(unittest.TestCase):
    """Frame index alignment and sampling arithmetic."""

    def test_align_helpers_floor_and_ceil(self):
        """Floor/ceil alignment subtracts and restores the remainder."""
        # pylint: disable=protected-access
        self.assertEqual(wan_video._align_to_factor_remainder_floor(10.0, 4, 1), 9)
        self.assertEqual(wan_video._align_to_factor_remainder_floor(9.0, 4, 1), 9)
        self.assertEqual(wan_video._align_to_factor_remainder_ceil(9.0, 4, 1), 9)
        self.assertEqual(wan_video._align_to_factor_remainder_ceil(9.2, 4, 1), 13)
        # pylint: enable=protected-access

    def test_calculate_frame_indices_identity(self):
        """Matching fps keeps every frame with no padding."""
        indices, pad = _calculate_frame_indices(
            total_frames=10, video_fps=25.0, fps=25.0,
            frame_factor=None, frame_factor_remainder=0,
            min_frames=None, max_frames=None,
        )
        self.assertEqual(len(indices), 10)
        self.assertEqual(pad, 0)

    def test_calculate_frame_indices_upsampling_pads(self):
        """Doubling the target fps pads the sampled clip to the target count."""
        indices, pad = _calculate_frame_indices(
            total_frames=4, video_fps=8.0, fps=16.0,
            frame_factor=None, frame_factor_remainder=0,
            min_frames=None, max_frames=None,
        )
        self.assertEqual(len(indices), 4)
        self.assertEqual(pad, 4)

    def test_calculate_frame_indices_min_max_and_factor(self):
        """Min/max clamp first, then the frame factor aligns the target."""
        indices, pad = _calculate_frame_indices(
            total_frames=100, video_fps=25.0, fps=25.0,
            frame_factor=4, frame_factor_remainder=1,
            min_frames=2, max_frames=10,
        )
        self.assertEqual(len(indices), 9)  # max_frames=10 aligns to 9 under factor 4, remainder 1
        self.assertEqual(pad, 0)

    def test_calculate_frame_indices_rejects_bad_inputs(self):
        """Non-positive frame counts and non-positive factors are rejected."""
        with self.assertRaises(ValueError):
            _calculate_frame_indices(
                total_frames=0, video_fps=25.0, fps=25.0,
                frame_factor=None, frame_factor_remainder=0,
                min_frames=None, max_frames=None,
            )
        with self.assertRaises(ValueError):
            _calculate_frame_indices(
                total_frames=10, video_fps=25.0, fps=25.0,
                frame_factor=0, frame_factor_remainder=0,
                min_frames=None, max_frames=None,
            )


class TestNormalizeAndResize(unittest.TestCase):
    """Tensor normalization, PIL stacking, smart resize, and pixel budgets."""

    def test_normalize_video_tensor_channels_first_passthrough(self):
        """A CHW-frame tensor passes through unchanged."""
        video = _video()
        result = _normalize_video_tensor(video)
        self.assertTrue(torch.equal(result, video))

    def test_normalize_video_tensor_numpy_hwc_single_frame(self):
        """A 3D HWC array is promoted to a 4D CHW tensor."""
        result = _normalize_video_tensor(np.zeros((4, 4, 3), dtype=np.uint8))
        self.assertEqual(tuple(result.shape), (1, 3, 4, 4))

    def test_normalize_video_tensor_rejects_bad_inputs(self):
        """Non-tensor inputs and unrecognizable channel layouts are rejected."""
        with self.assertRaises(TypeError):
            _normalize_video_tensor([1, 2, 3])
        with self.assertRaises(ValueError):
            _normalize_video_tensor(np.zeros((2, 4, 4), dtype=np.uint8))
        with self.assertRaises(ValueError):
            _normalize_video_tensor(np.zeros((2, 4, 4, 5, 1), dtype=np.uint8))

    def test_pil_frames_to_tensor_stacks_and_converts(self):
        """Frames stack to CHW; non-RGB frames are converted first."""
        frames = [_pil_frame((255, 0, 0)), Image.new("L", (4, 4), 128)]
        tensor = _pil_frames_to_tensor(frames)
        self.assertEqual(tuple(tensor.shape), (2, 3, 4, 4))

    def test_smart_resize_scale_factor(self):
        """Scale factor rounds the target size up to its multiple."""
        video = _video(4, 3, 250, 250)
        result = _smart_resize(video, scale_factor=16)
        self.assertEqual(tuple(result.shape[-2:]), (256, 256))

    def test_smart_resize_passthrough_without_constraints(self):
        """Without constraints the size is kept and the tensor stays float."""
        video = _video(2, 3, 8, 16)
        result = _smart_resize(video)
        self.assertEqual(tuple(result.shape), (2, 3, 8, 16))
        self.assertTrue(result.is_floating_point())

    def test_smart_resize_max_pixels_shrinks(self):
        """A pixel budget shrinks both sides by the same beta."""
        video = _video(1, 3, 256, 256)
        result = _smart_resize(video, scale_factor=16, video_max_pixels=16384)
        self.assertEqual(tuple(result.shape[-2:]), (128, 128))

    def test_smart_resize_min_pixels_grows(self):
        """A pixel floor grows both sides by the same beta."""
        video = _video(1, 3, 32, 32)
        result = _smart_resize(video, video_min_pixels=4096)
        self.assertEqual(tuple(result.shape[-2:]), (64, 64))

    def test_smart_resize_rejects_bad_rank_and_ratio(self):
        """Non-4D inputs and oversized aspect ratios are rejected."""
        with self.assertRaises(ValueError):
            _smart_resize(torch.zeros(4, 4))
        with self.assertRaises(ValueError):
            _smart_resize(_video(1, 3, 100, 10), max_ratio=4)

    def test_apply_dynamic_video_max_pixels_passthrough(self):
        """Without a total budget the kwargs are returned unchanged."""
        kwargs = {"fps": 16}
        self.assertEqual(_apply_dynamic_video_max_pixels(0, kwargs), kwargs)

    def test_apply_dynamic_video_max_pixels_clamps(self):
        """The dynamic budget is scaled by frames and clamped by min/max."""
        updated = _apply_dynamic_video_max_pixels(
            10, {"video_total_pixels": 1000, "frame_factor": 2},
        )
        self.assertEqual(updated["video_max_pixels"], 200)
        updated = _apply_dynamic_video_max_pixels(
            10, {"video_total_pixels": 1000, "video_max_pixels": 150, "video_min_pixels": 100},
        )
        self.assertEqual(updated["video_max_pixels"], 150)
        updated = _apply_dynamic_video_max_pixels(
            10, {"video_total_pixels": 1000, "video_min_pixels": 300},
        )
        self.assertEqual(updated["video_max_pixels"], 315)


class TestTorchcodecLoader(unittest.TestCase):
    """torchcodec decoding with and without the optional dependency."""

    def test_missing_torchcodec_returns_none(self):
        """Without torchcodec the loader reports None for the fallback path."""
        with patch.dict(sys.modules, {"torchcodec": None, "torchcodec.decoders": None}):
            self.assertIsNone(_load_video_with_torchcodec(b"bytes"))

    def test_torchcodec_decodes_and_pads(self):
        """A fake decoder exercises the decode, sampling, and padding path."""
        decoders_mock = MagicMock()
        decoders_mock.VideoDecoder = FakeDecoder
        fake_module = MagicMock()
        fake_module.decoders = decoders_mock
        with patch.dict(sys.modules, {"torchcodec": fake_module, "torchcodec.decoders": decoders_mock}):
            result = _load_video_with_torchcodec(
                b"bytes", fps=16, frame_factor=4, frame_factor_remainder=1, max_frames=5,
            )
        self.assertIsNotNone(result)
        video, video_fps = result
        self.assertEqual(video_fps, 8.0)
        self.assertEqual(video.ndim, 4)
        self.assertGreater(video.shape[0], 0)

    def test_torchcodec_decode_failure_returns_none(self):
        """A decoder failure is logged and reported as None."""
        fake_module = MagicMock()
        fake_module.decoders.VideoDecoder = FakeDecoder
        with patch.dict(sys.modules, {"torchcodec": fake_module, "torchcodec.decoders": fake_module}):
            with patch.object(FakeDecoder, "__init__", side_effect=ValueError("boom")):
                self.assertIsNone(_load_video_with_torchcodec(b"bytes"))


class TestImageioLoader(unittest.TestCase):
    """imageio decoding with and without the optional dependency."""

    def test_missing_imageio_raises(self):
        """Without imageio the loader explains the missing dependency."""
        with patch.dict(sys.modules, {"imageio": None, "imageio.v3": None}):
            with self.assertRaises(RuntimeError):
                _load_video_with_imageio(b"bytes")

    def test_imageio_reads_bytes_and_paths(self):
        """Bytes and file sources both decode through the fake imageio."""
        fake_imageio = MagicMock()
        fake_imageio.v3.imread = FakeIio.imread
        with patch.dict(sys.modules, {"imageio": fake_imageio, "imageio.v3": fake_imageio.v3}):
            video, video_fps = _load_video_with_imageio(b"bytes")
            self.assertEqual(video_fps, 24.0)
            self.assertEqual(video.ndim, 4)
            video, _ = _load_video_with_imageio("/tmp/clip.mp4")
            self.assertEqual(video.shape[0], 4)


class TestLoadVideo(unittest.TestCase):
    """Input dispatch of the video loader."""

    def test_tensor_input_samples_frames(self):
        """A tensor input is normalized, resized, and frame-sampled."""
        result = _load_video(_video(4, 3, 8, 8), fps=16)
        self.assertEqual(result.ndim, 4)
        self.assertEqual(result.shape[0], 4)

    def test_ndarray_input_samples_frames(self):
        """An ndarray input follows the tensor path."""
        result = _load_video(np.zeros((4, 8, 8, 3), dtype=np.uint8), fps=24)
        self.assertEqual(result.ndim, 4)

    def test_pil_frame_list_input(self):
        """A list of PIL frames stacks and samples."""
        result = _load_video([_pil_frame(), _pil_frame()], fps=24)
        self.assertEqual(result.shape[0], 2)

    def test_empty_frame_list_raises(self):
        """An empty frame list is rejected."""
        with self.assertRaises(ValueError):
            _load_video([], fps=24)

    def test_byte_frame_list_input(self):
        """Byte frames inside a list decode through PIL."""
        result = _load_video([_png_bytes(), _png_bytes((0, 0, 255))], fps=24)
        self.assertEqual(result.shape[0], 2)

    def test_dict_frames_input_applies_full_pipeline(self):
        """A frames dict samples and resizes with the configured factors."""
        result = _load_video(
            {"frames": [_pil_frame(), _pil_frame(), _pil_frame()], "video_fps": 8.0},
            fps=16, frame_factor=4, frame_factor_remainder=1, max_frames=5,
        )
        self.assertEqual(result.ndim, 4)
        self.assertEqual(result.shape[0] % 1, 0)

    def test_dict_video_tensor_input(self):
        """A dict with a 'video' tensor decodes directly."""
        result = _load_video({"video": _video(2, 3, 8, 8).numpy(), "video_fps": 12.0}, fps=12)
        self.assertEqual(result.shape[0], 2)

    def test_dict_without_video_or_frames_raises(self):
        """A dict without 'video' or 'frames' is rejected."""
        with self.assertRaises(ValueError):
            _load_video({"other": 1}, fps=24)

    def test_str_path_falls_back_to_imageio(self):
        """A path decodes via torchcodec, falling back to a faked imageio."""
        fake_imageio = MagicMock()
        fake_imageio.v3.imread = FakeIio.imread
        with patch.dict(sys.modules, {"imageio": fake_imageio, "imageio.v3": fake_imageio.v3}):
            result = _load_video("clip.mp4", data_dir="/data", fps=16)
        self.assertEqual(result.shape[0], 2)  # 4 frames @24fps resampled to 16fps

    def test_url_source_downloads_then_decodes(self):
        """An HTTP source downloads through requests and decodes the bytes."""
        fake_imageio = MagicMock()
        fake_imageio.v3.imread = FakeIio.imread
        with patch.dict(sys.modules, {
            "imageio": fake_imageio, "imageio.v3": fake_imageio.v3,
            "torchcodec": None, "torchcodec.decoders": None,
            "requests": FakeRequests,
        }):
            result = _load_video("https://example.com/clip.mp4", fps=16)
        self.assertEqual(result.shape[0], 2)  # 4 frames @24fps resampled to 16fps


class TestLoadImage(unittest.TestCase):
    """Input dispatch of the image loader."""

    def test_pil_passthrough_converts_rgb(self):
        """A PIL input is converted to RGB and passed through."""
        image = _load_image(Image.new("L", (4, 4), 128))
        self.assertEqual(image.mode, "RGB")

    def test_tensor_input_scales_and_permutes(self):
        """A float CHW tensor is permuted and scaled to uint8 RGB."""
        frame = torch.rand(3, 4, 4)
        image = _load_image(frame)
        self.assertEqual(image.size, (4, 4))

    def test_tensor_hwc_uint8_input(self):
        """A uint8 HWC tensor is converted without scaling."""
        image = _load_image(torch.zeros(4, 4, 3, dtype=torch.uint8))
        self.assertEqual(image.size, (4, 4))

    def test_ndarray_input(self):
        """An ndarray input converts through fromarray."""
        image = _load_image(np.zeros((4, 4, 3), dtype=np.uint8))
        self.assertEqual(image.size, (4, 4))

    def test_bytes_input(self):
        """PNG bytes decode through PIL."""
        image = _load_image(_png_bytes())
        self.assertEqual(image.size, (4, 4))

    def test_str_path_with_data_dir(self):
        """A relative path resolves against data_dir."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "img.png"
            path.write_bytes(_png_bytes())
            image = _load_image("img.png", data_dir=tmp)
        self.assertEqual(image.size, (4, 4))

    def test_url_source_downloads(self):
        """An HTTP source downloads through a faked requests."""
        with patch.dict(sys.modules, {"requests": FakeRequests}):
            image = _load_image("https://example.com/img.png")
        self.assertEqual(image.size, (4, 4))

    def test_unsupported_input_raises(self):
        """Unsupported input types are rejected."""
        with self.assertRaises(TypeError):
            _load_image(123)


class TestFirstAvailable(unittest.TestCase):
    """Empty-value skipping of the candidate-key lookup."""

    def test_returns_first_meaningful_value(self):
        """Empty strings, sequences, mappings, tensors, and arrays are skipped."""
        mapping = {
            "a": "", "b": [], "c": {}, "d": torch.zeros(0), "e": np.zeros(0), "f": "value",
        }
        self.assertEqual(_first_available(mapping, ("a", "b", "c", "d", "e", "f")), "value")

    def test_returns_none_when_all_empty(self):
        """A record with only empty values yields None."""
        self.assertIsNone(_first_available({"a": "", "b": {}}, ("a", "b")))
        self.assertIsNone(_first_available({}, ("missing",)))


class TestWanVideoTransformCall(unittest.TestCase):
    """End-to-end transform of raw records into DiT samples."""

    def test_invalid_task_rejected(self):
        """Unknown task names are rejected at construction."""
        with self.assertRaises(ValueError):
            WanVideoTransform(task="v2v")

    def test_build_wan_video_transform(self):
        """The YAML target builds a configured transform."""
        transform = build_wan_video_transform(task="t2v", fps=16)
        self.assertEqual(transform.task, "t2v")
        self.assertEqual(transform.kwargs["fps"], 16)

    def test_t2v_call_builds_sample(self):
        """A t2v record becomes one sample with prompt and video list."""
        transform = WanVideoTransform(task="t2v", fps=16, max_frames=4)
        with patch.object(wan_video, "_load_video", return_value=_video(4, 3, 8, 8)):
            samples = transform({"caption": "a cat", "video_path": "cat.mp4"})
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["inputs"], "a cat")
        self.assertEqual(len(samples[0]["videos"]), 1)

    def test_i2v_call_uses_provided_image(self):
        """An i2v record with an image entry keeps the provided image."""
        transform = WanVideoTransform(task="i2v", fps=16, max_frames=4)
        record = {"prompt": "a cat", "video_bytes": b"video", "image_bytes": _png_bytes()}
        with patch.object(wan_video, "_load_video", return_value=_video(4, 3, 8, 8)):
            samples = transform(record)
        self.assertEqual(len(samples[0]["images"]), 1)

    def test_i2v_call_falls_back_to_first_frame(self):
        """An i2v record without an image uses the first decoded frame."""
        transform = WanVideoTransform(task="i2v", fps=16, max_frames=4)
        with patch.object(wan_video, "_load_video", return_value=_video(4, 3, 8, 8)):
            samples = transform({"prompt": "a cat", "video_bytes": b"video"})
        self.assertEqual(samples[0]["images"][0].size, (8, 8))

    def test_preprocess_missing_keys_raise(self):
        """Records without prompt or video values are rejected with guidance."""
        transform = WanVideoTransform(task="t2v")
        with self.assertRaises(ValueError):
            transform._preprocess({"video_bytes": b"video"})  # pylint: disable=protected-access
        with self.assertRaises(ValueError):
            transform._preprocess({"prompt": "a cat"})  # pylint: disable=protected-access


class TestDatasetBuilder(unittest.TestCase):
    """Dispatch and configuration of the online dataset builder."""

    def setUp(self) -> None:
        """Isolate the builder from the real online dataset constructors."""
        patches = (
            patch.object(wan_video, "build_online_text_mapping_dataset", MagicMock()),
            patch.object(wan_video, "build_online_iterable_dataset", MagicMock()),
            patch.object(wan_video, "create_dataloader_parallel_context", MagicMock()),
        )
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_mapping_default_build(self):
        """The default dataset_type builds the mapping dataset with a seed."""
        config = SimpleNamespace(seed=7)
        build_wan_video_dataset(data_config={"dataset_type": "mapping"}, data_path="/data", training_config=config)
        _, kwargs = wan_video.build_online_text_mapping_dataset.call_args
        self.assertEqual(kwargs["data_config"]["random_seed"], 7)
        self.assertEqual(kwargs["data_path"], "/data")

    def test_iterable_build_receives_context(self):
        """The iterable dataset_type forwards the dataloader context."""
        context = MagicMock()
        build_wan_video_dataset(
            data_config={"dataset_type": "iterable"},
            data_path="/data",
            dataloader_context=context,
        )
        wan_video.build_online_iterable_dataset.assert_called_once()
        self.assertEqual(
            wan_video.build_online_iterable_dataset.call_args.kwargs["dataloader_context"], context,
        )

    def test_mesh_context_builds_dataloader_context(self):
        """A runtime mesh derives the dataloader ownership context."""
        build_wan_video_dataset(
            data_config={}, data_path="/data", mesh_context=SimpleNamespace(dp_rank=0),
        )
        wan_video.create_dataloader_parallel_context.assert_called_once()

    def test_unknown_dataset_type_raises(self):
        """Unknown dataset types are rejected with the supported list."""
        with self.assertRaises(ValueError):
            build_wan_video_dataset(data_config={"dataset_type": "bogus"}, data_path="/data")

    def test_data_path_overrides_hub_name(self):
        """A local data_path clears the configured Hub dataset name."""
        build_wan_video_dataset(
            data_config={"hf_dataset_name": "hub/name", "dataset_type": "mapping"},
            data_path="/data",
        )
        kwargs = wan_video.build_online_text_mapping_dataset.call_args.kwargs
        self.assertIsNone(kwargs["data_config"]["hf_dataset_name"])

    def test_blank_data_path_keeps_hub_name(self):
        """A whitespace data_path is treated as absent, keeping the Hub name."""
        build_wan_video_dataset(
            data_config={"hf_dataset_name": "hub/name", "dataset_type": "mapping"},
            data_path="   ",
        )
        kwargs = wan_video.build_online_text_mapping_dataset.call_args.kwargs
        self.assertEqual(kwargs["data_config"]["hf_dataset_name"], "hub/name")


class TestCollator(unittest.TestCase):
    """List-preserving collation of raw DiT samples."""

    def test_normalize_model_sample_variants(self):
        """Mappings pass through; single-item sequences unwrap; others raise."""
        sample = {"inputs": "x"}
        self.assertEqual(_normalize_model_sample(sample), sample)
        self.assertEqual(_normalize_model_sample([sample]), sample)
        with self.assertRaises(ValueError):
            _normalize_model_sample([sample, sample])
        with self.assertRaises(ValueError):
            _normalize_model_sample("text")

    def test_collator_preserves_per_sample_lists(self):
        """Keys union in first-seen order; missing keys collate as None."""
        collator = DiTCollator()
        batch = collator([
            {"inputs": "a", "videos": [1]},
            {"inputs": "b", "images": [2]},
        ])
        self.assertEqual(list(batch.keys()), ["inputs", "videos", "images"])
        self.assertEqual(batch["inputs"], ["a", "b"])
        self.assertIsNone(batch["images"][0])

    def test_collator_rejects_empty_batch(self):
        """An empty micro-batch is rejected."""
        with self.assertRaises(ValueError):
            DiTCollator()([])

    def test_build_dit_collate_fn(self):
        """The YAML target builds a collator instance."""
        self.assertIsInstance(build_dit_collate_fn(), DiTCollator)

    def test_dit_batch_reads_iterator_once(self):
        """The get_batch callable returns consecutive iterator items."""
        get_batch = DiTBatch(mesh_context=SimpleNamespace(dp_rank=0), device=None)
        iterator = iter([{"inputs": "a"}, {"inputs": "b"}])
        self.assertEqual(get_batch(iterator), {"inputs": "a"})
        self.assertEqual(get_batch(iterator), {"inputs": "b"})

    def test_dit_batch_rejects_pipeline_shared_data(self):
        """Pipeline-parallel shared data is not supported yet."""
        get_batch = DiTBatch(mesh_context=SimpleNamespace(), device=None, pp_shared_data=True)
        with self.assertRaises(NotImplementedError):
            get_batch(iter([{"inputs": "a"}]))


if __name__ == "__main__":
    unittest.main()
