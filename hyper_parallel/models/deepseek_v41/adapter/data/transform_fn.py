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
"""DeepSeek-V4.1 transform backed by its native processor."""

from __future__ import annotations

import logging
import os
import shutil
import ssl
import stat
from collections.abc import Mapping, Sequence
from http.client import HTTPException
from tempfile import TemporaryDirectory
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import torch  # pylint: disable=forbidden-backend-import
from PIL import Image

from hyper_parallel.data.constants import IGNORE_INDEX, ONLINE_SOURCE_PATH_KEY
from hyper_parallel.data.dataset_logging import get_dataset_logger
from hyper_parallel.data.omni.omni_transform import OmniDataTransform, build_labels_from_supervised_start
from hyper_parallel.data.online.format import (
    convert_interleaved_to_content,
    convert_sharegpt_to_messages,
    normalize_chat_media_fields,
    resolve_media_path,
)
from hyper_parallel.models.deepseek_v41.adapter.data.encoding import process_image_messages
from hyper_parallel.models.deepseek_v41.adapter.data.image_processor import (
    IMAGE,
    IMAGE_NEW_LINE,
    TEXT,
    ImageInput,
    prepare_image_input,
)
from hyper_parallel.models.deepseek_v41.adapter.data.processor import (
    DeepseekV41Processor,
)

logger = get_dataset_logger("hyper_parallel.data.deepseek_v41")


class _DeepseekV41OmniTransformBase(OmniDataTransform):
    """Share native image metadata and sample assembly across training modes."""

    def __init__(
            self,
            processor: DeepseekV41Processor,
            *,
            max_seq_len: int = 4096,
            training_mode: str = "sft",
            sample_alignment_size: int = 1,
    ) -> None:
        """Bind the native processor to the generic Omni lifecycle."""
        if not isinstance(processor, DeepseekV41Processor):
            raise TypeError("DeepSeek-V4.1 transforms require DeepseekV41Processor")

        if (isinstance(sample_alignment_size, bool) or not isinstance(sample_alignment_size, int)
                or sample_alignment_size <= 0):
            raise ValueError("sample_alignment_size must be a positive integer")

        self.sample_alignment_size = sample_alignment_size
        super().__init__(
            max_seq_len=max_seq_len,
            processor=processor,
            training_mode=training_mode,
        )

    def _align_sample(self, sample: dict[str, Any]) -> dict[str, Any] | None:
        """Align samples before selection so packing costs and image offsets include padding."""
        sequence_length = sample["input_ids"].numel()
        pad_length = (-sequence_length) % self.sample_alignment_size
        if not pad_length:
            return sample

        if sequence_length + pad_length > self.max_seq_len:
            logger.warning(
                "Skipping DeepSeek-V4.1 sample: aligned_length=%d exceeds max_seq_len=%d",
                sequence_length + pad_length, self.max_seq_len,
            )
            return None

        pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = self.processor.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("Sample alignment requires a tokenizer pad_token_id or eos_token_id")

        for field, pad_value in (("input_ids", pad_token_id), ("labels", IGNORE_INDEX), ("token_types", TEXT)):
            sample[field] = torch.nn.functional.pad(sample[field], (0, pad_length), value=pad_value)
        return sample

    @staticmethod
    def _validate_local_image(
            sample: Mapping[str, Any],
            image_record: Mapping[str, Any],
    ) -> None:
        """Verify and decode one local image; report data errors without choosing a mode.

        Example:
            source=/data/train.jsonl + url=images/a.jpg -> /data/images/a.jpg.
            Valid image -> no exception; missing/empty/corrupt image -> ValueError.
        """
        if image_record.get("data") is not None:
            raise ValueError("Image validation requires local file paths, not embedded image data")

        image_path = image_record.get("url")
        image_source = image_record.get("source")
        if isinstance(image_source, Mapping):
            if image_source.get("data") is not None:
                raise ValueError("Image validation requires local file paths, not embedded image data")

            image_path = image_source.get("url") or image_path
        if not isinstance(image_path, str) or not image_path.strip():
            raise ValueError("Image reference must be a nonempty local file path")

        if image_path.lower().startswith(("http://", "https://", "data:")):
            raise ValueError(f"Image must be downloaded before source indexing: {image_path}")

        source_path = sample.get(ONLINE_SOURCE_PATH_KEY)
        source_directory = os.path.dirname(os.path.abspath(source_path)) if source_path else os.getcwd()
        image_path = resolve_media_path(image_path, source_directory)
        try:
            file_stat = os.stat(image_path)
            if not stat.S_ISREG(file_stat.st_mode):
                raise ValueError(f"Image path is not a regular file: {image_path}")

            if file_stat.st_size == 0:
                raise ValueError(f"Image file is empty: {image_path}")

            with open(image_path, "rb") as image_file:
                with Image.open(image_file) as image:
                    image.verify()
                # Verification does not decode pixels; reopen the same file to check the payload.
                image_file.seek(0)
                with Image.open(image_file) as image:
                    image.load()
        except (FileNotFoundError, NotADirectoryError) as error:
            raise ValueError(f"Image file not found: {image_path}") from error

        except OSError as error:
            # Pillow reports invalid streams without errno; storage errors must abort the build.
            if error.errno is not None:
                raise

            raise ValueError(f"Image cannot be decoded: {image_path}: {error}") from error

        except (SyntaxError, ValueError, Image.DecompressionBombError) as error:
            raise ValueError(f"Image verification failed: {image_path}: {error}") from error

    def encode_batch(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Convert packing metadata into the V4.1 model batch contract.

        Args:
            batch: Packed token fields and optional image metadata.
        """
        encoded_batch = dict(batch)
        if "pixel_values" not in encoded_batch:
            return encoded_batch

        required_fields = (
            "image_patch_lengths",
            "image_vit_grid_hw",
            "image_llm_grid_hw",
            "image_token_starts",
        )
        missing_fields = []
        for field in required_fields:
            if field not in encoded_batch:
                missing_fields.append(field)
        if missing_fields:
            raise ValueError(
                "DeepSeek-V4.1 image batch is missing fields: "
                + ", ".join(missing_fields)
            )

        patch_lengths = encoded_batch.pop("image_patch_lengths")
        # [images] belongs to one packed row; [batch, images] retains its row ownership.
        if patch_lengths.ndim == 1:
            image_batch_indices = patch_lengths.new_zeros(patch_lengths.numel())
        elif patch_lengths.ndim == 2:
            batch_size, images_per_batch = patch_lengths.shape
            image_batch_indices = torch.arange(
                batch_size, dtype=patch_lengths.dtype, device=patch_lengths.device,
            ).repeat_interleave(images_per_batch)
        else:
            raise ValueError("image_patch_lengths must have shape [images] or [batch, images]")

        flat_patch_lengths = patch_lengths.reshape(-1)
        zero = flat_patch_lengths.new_zeros(1)
        encoded_batch["image_patch_offsets"] = torch.cat(
            (zero, flat_patch_lengths.cumsum(dim=0))
        )
        encoded_batch["image_batch_indices"] = image_batch_indices
        encoded_batch["image_vit_grid_hw"] = encoded_batch["image_vit_grid_hw"].reshape(-1, 2)
        encoded_batch["image_llm_grid_hw"] = encoded_batch["image_llm_grid_hw"].reshape(-1, 2)
        encoded_batch["image_token_starts"] = encoded_batch["image_token_starts"].reshape(-1)
        if logger.isEnabledFor(logging.DEBUG) and encoded_batch["input_ids"].device.type == "cpu":
            starts = encoded_batch["image_token_starts"].tolist()
            grids = encoded_batch["image_llm_grid_hw"].tolist()
            # Native image spans contain start/end tokens and one newline per grid row.
            ends = [start + height * (width + 1) + 2 for start, (height, width) in zip(starts, grids)]
            rows = image_batch_indices.tolist()
            sequence_length = encoded_batch["input_ids"].shape[-1]
            offsets = encoded_batch["image_patch_offsets"].tolist()
            valid_spans = (len(starts) == len(grids) == len(rows)
                           and all(0 <= start < end <= sequence_length for start, end in zip(starts, ends))
                           and all(0 <= row < encoded_batch["input_ids"].shape[0] for row in rows))
            valid_patches = (all(length > 0 for length in flat_patch_lengths.tolist())
                             and offsets[-1] == encoded_batch["pixel_values"].shape[0])
            logger.debug(
                "DeepSeek image batch: rows=%s token_starts=%s token_ends=%s llm_grids=%s "
                "patch_offsets=%s pixel_shape=%s valid_spans=%s valid_patches=%s",
                rows, starts, ends, grids, offsets, tuple(encoded_batch["pixel_values"].shape),
                valid_spans, valid_patches, enabled=True,
            )
            if not valid_spans or not valid_patches:
                logger.warning("DeepSeek image batch has inconsistent token spans or patch metadata")
        return encoded_batch

    @staticmethod
    def _build_image_fields(
            image_inputs: Sequence[ImageInput],
    ) -> dict[str, torch.Tensor]:
        """Flatten official ImageInput objects into packing-ready tensors."""
        if not image_inputs:
            image_fields = {}
            return image_fields

        patches = []
        patch_lengths = []
        vit_grid_hw = []
        llm_grid_hw = []
        token_starts = []
        for image_input in image_inputs:
            if not isinstance(image_input, ImageInput):
                raise TypeError("DeepSeek-V4.1 image_processor must return ImageInput objects")

            patches.append(image_input.patches)
            patch_lengths.append(image_input.patches.shape[0])
            vit_grid_hw.append((image_input.n_vit_h, image_input.n_vit_w))
            grid_height = int((image_input.types == IMAGE_NEW_LINE).sum())
            image_token_count = int((image_input.types == IMAGE).sum())
            if grid_height <= 0 or image_token_count % grid_height != 0:
                raise ValueError("DeepSeek-V4.1 image token types do not describe a rectangular grid")

            grid_width = image_token_count // grid_height
            llm_grid_hw.append((grid_height, grid_width))
            token_starts.append(image_input.start)

        image_fields = {
            "pixel_values": torch.cat(patches, dim=0),
            "image_patch_lengths": torch.tensor(patch_lengths, dtype=torch.long),
            "image_vit_grid_hw": torch.tensor(vit_grid_hw, dtype=torch.long),
            "image_llm_grid_hw": torch.tensor(llm_grid_hw, dtype=torch.long),
            "image_token_starts": torch.tensor(token_starts, dtype=torch.long),
        }
        return image_fields


class DeepseekV41SFTTransform(_DeepseekV41OmniTransformBase):
    """Encode OpenAI-style image conversations for DeepSeek-V4.1 SFT."""

    def __init__(
            self,
            processor: DeepseekV41Processor,
            *,
            max_seq_len: int = 4096,
            thinking_mode: str = "chat",
            drop_thinking: bool = True,
            sample_format: str = "messages",
            sample_alignment_size: int = 1,
    ) -> None:
        """Bind the existing SFT template and supervision policy."""
        super().__init__(processor, max_seq_len=max_seq_len, sample_alignment_size=sample_alignment_size)
        self.thinking_mode = thinking_mode
        self.drop_thinking = drop_thinking
        if sample_format not in ("messages", "openai", "hf", "sharegpt"):
            raise ValueError(f"Unsupported DeepSeek-V4.1 SFT sample format: {sample_format!r}")

        self.sample_format = sample_format

    @property
    def unique_identifiers(self) -> Mapping[str, Any]:
        """Identify the conversation schema and local-image validation rules."""
        filter_identifiers = {
            "class": f"{type(self).__module__}.{type(self).__qualname__}",
            "version": 1,
            "sample_format": self.sample_format,
        }
        return filter_identifiers

    def is_valid_sample(self, sample: Mapping[str, Any]) -> bool:
        """Validate the configured conversation and every image without rendering a prompt."""
        try:
            if self.sample_format == "sharegpt":
                messages = convert_sharegpt_to_messages(sample)
            else:
                messages = normalize_chat_media_fields(sample)
            if len(messages) < 2 or messages[-1].get("role") != "assistant":
                raise ValueError("DeepSeek-V4.1 SFT sample must end with a final assistant message")

            _, image_records = process_image_messages(messages)
        except ValueError as error:
            raise ValueError(f"Invalid {self.sample_format} sample: {error}") from error

        for image_record in image_records:
            self._validate_local_image(sample, image_record)

        return True

    def encode_sample(self, sample: dict[str, Any]) -> dict[str, Any] | None:
        """Encode native image conversations and supervise the final assistant reply.

        Args:
            sample: Raw conversation record, optionally containing images.

        Example:
            Input: {"messages": [{"role": "user", "content": "Hi"},
                                 {"role": "assistant", "content": "Hello"}]}
            Output: native input_ids, token_types, labels, and optional image fields.
            Labels mask the prompt/media and supervise the final reply and ending.
        """
        if self.sample_format == "sharegpt":
            messages = convert_sharegpt_to_messages(sample)
        else:
            messages = normalize_chat_media_fields(sample)
        if len(messages) < 2 or messages[-1].get("role") != "assistant":
            raise ValueError("DeepSeek-V4.1 SFT sample must end with a final assistant message")

        prompt, media = self.processor.chat_template(
            messages, thinking_mode=self.thinking_mode, drop_thinking=self.drop_thinking,
            return_multi_modal_data=True,
        )
        input_ids, token_types, image_inputs = self.processor.image_processor(
            prompt, media["images"], self.processor.tokenizer, self.processor,
        )
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        token_types = torch.tensor(token_types, dtype=torch.long)
        if input_ids.numel() > self.max_seq_len:
            logger.warning(
                "Skipping overlong DeepSeek-V4.1 SFT sample: expanded_length=%d exceeds max_seq_len=%d",
                input_ids.numel(), self.max_seq_len,
            )
            return None

        image_fields = self._build_image_fields([] if image_inputs is None else image_inputs)
        prompt, _ = self.processor.chat_template(
            list(messages[:-1]),
            thinking_mode=self.thinking_mode,
            drop_thinking=self.drop_thinking,
            return_multi_modal_data=True,
        )
        prompt_tokens = self.processor.tokenizer.encode(prompt)
        placeholder_count = sum(token == self.processor.image_token_id for token in prompt_tokens)
        image_grids = image_fields.get("image_llm_grid_hw", ())
        if placeholder_count > len(image_grids):
            raise ValueError("Assistant prefix contains more images than the encoded sample")

        assistant_start = len(prompt_tokens)
        # One image placeholder expands to BOS + h * (w patches + newline) + EOS.
        # A 2x2 grid occupies 8 tokens, so the reply starts 7 positions later.
        for height, width in image_grids[:placeholder_count]:
            assistant_start += int(height) * (int(width) + 1) + 1
        if not 0 <= assistant_start < input_ids.numel():
            raise ValueError("DeepSeek-V4.1 assistant response must contain at least one token")

        labels = build_labels_from_supervised_start(
            input_ids, assistant_start, media_mask=token_types != TEXT,
        )
        encoded_sample = {
            "input_ids": input_ids,
            "token_types": token_types,
            "labels": labels,
            **image_fields,
        }
        aligned_sample = self._align_sample(encoded_sample)
        return aligned_sample


class DeepseekV41PretrainTransform(_DeepseekV41OmniTransformBase):
    """Encode image-caption pairs or ordered interleaved text/images with native image spans."""

    def __init__(
            self, processor: DeepseekV41Processor, *, max_seq_len: int = 4096,
            sample_format: str = "image_caption",
            sample_alignment_size: int = 1,
            download_verify_ssl: bool = True,
    ) -> None:
        """Select native image-caption or interleaved pretraining explicitly."""
        super().__init__(
            processor, max_seq_len=max_seq_len, training_mode="pretrain", sample_alignment_size=sample_alignment_size,
        )
        if sample_format not in ("image_caption", "interleaved"):
            raise ValueError(f"Unsupported DeepSeek-V4.1 pretrain sample format: {sample_format!r}")

        if not isinstance(download_verify_ssl, bool):
            raise ValueError("download_verify_ssl must be a boolean")

        self.sample_format = sample_format
        self.download_verify_ssl = download_verify_ssl

    @property
    def unique_identifiers(self) -> Mapping[str, Any]:
        """Identify the pretraining schema and download validation settings."""
        filter_identifiers = {
            "class": f"{type(self).__module__}.{type(self).__qualname__}",
            "version": 1,
            "sample_format": self.sample_format,
            "download_verify_ssl": self.download_verify_ssl,
        }
        return filter_identifiers

    def _download_image_caption(self, sample: Mapping[str, Any]) -> None:
        """Recover an invalid local caption image from its top-level HTTP(S) URL.

        Example:
            image=images/a.jpg + url=https://host/a.jpg -> download temporary file
            -> verify/decode -> atomically replace images/a.jpg.
        """
        image_path = sample.get("image")
        if (not isinstance(image_path, str) or not image_path.strip()
                or image_path.lower().startswith(("http://", "https://", "data:"))):
            raise ValueError("Image-caption download requires a nonempty local image path")

        remote_url = sample.get("url")
        if not isinstance(remote_url, str) or not remote_url.lower().startswith(("http://", "https://")):
            raise ValueError("Image-caption download requires an HTTP(S) url")

        source_path = sample.get(ONLINE_SOURCE_PATH_KEY)
        source_directory = os.path.dirname(os.path.abspath(source_path)) if source_path else os.getcwd()
        image_path = resolve_media_path(image_path, source_directory)
        if os.path.exists(image_path) and not stat.S_ISREG(os.stat(image_path).st_mode):
            raise ValueError(f"Image path is not a regular file: {image_path}")

        image_directory = os.path.dirname(image_path)
        os.makedirs(image_directory, exist_ok=True)
        context = ssl.create_default_context()
        if not self.download_verify_ssl:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE

        with TemporaryDirectory(dir=image_directory, prefix=".image-") as temporary_directory:
            temporary_path = os.path.join(temporary_directory, "download")
            try:
                with open(temporary_path, "wb") as file, urlopen(remote_url, timeout=30, context=context) as response:
                    shutil.copyfileobj(response, file)
            except (URLError, TimeoutError, ConnectionError, HTTPException, ssl.SSLError) as error:
                raise ValueError(f"Image download failed: {remote_url}: {error}") from error

            self._validate_local_image(sample, {"url": temporary_path})
            os.replace(temporary_path, image_path)

    def is_valid_sample(self, sample: Mapping[str, Any]) -> bool:
        """Validate the selected pretraining schema and all required local images."""
        if self.sample_format == "interleaved":
            try:
                blocks = convert_interleaved_to_content(sample)
            except ValueError as error:
                raise ValueError(f"Invalid interleaved sample: {error}") from error

            if not any(block["type"] == "text" and block["text"].strip() for block in blocks):
                raise ValueError("Interleaved pretraining requires nonempty text")

        else:
            image, caption = sample.get("image"), sample.get("caption")
            if not isinstance(caption, str) or not caption.strip():
                raise ValueError("Image-caption sample requires a nonempty caption string")

            blocks = [{"type": "image", "url": image}]

        for block in blocks:
            if block["type"] == "image":
                try:
                    self._validate_local_image(sample, block)
                except ValueError as validation_error:
                    if self.sample_format != "image_caption":
                        raise ValueError(
                            f"URL recovery requires image_caption; got {self.sample_format!r}. "
                            f"Image validation failed: {validation_error}"
                        ) from validation_error

                    remote_url = sample.get("url")
                    if not isinstance(remote_url, str) or not remote_url.strip():
                        raise ValueError(
                            "Cannot recover local image: sample.url must be a nonempty string. "
                            f"Image validation failed: {validation_error}"
                        ) from validation_error

                    if not remote_url.lower().startswith(("http://", "https://")):
                        raise ValueError(
                            "Cannot recover local image: sample.url must use HTTP or HTTPS. "
                            f"Image validation failed: {validation_error}"
                        ) from validation_error

                    self._download_image_caption(sample)

        return True

    def encode_sample(self, sample: dict[str, Any]) -> dict[str, Any] | None:
        """Encode image-caption or interleaved pretraining without adding chat roles.

        Args:
            sample: Raw image-caption or interleaved record.

        Example:
            image-caption input: {"image": "cat.jpg", "caption": "A cat."}
            Interleaved input: {"texts": ["Look: ", None, "A cat."],
                                "images": [None, "cat.jpg", None]}
            Output: input_ids = [BOS, text/image spans in source order, EOS].
            Labels mask BOS and images, and supervise text plus EOS.
            Source validation may recover a local image from the top-level url.
            Encoding reads that local image and does not retry downloads.
        """
        if self.sample_format == "interleaved":
            blocks = convert_interleaved_to_content(sample)
        else:
            image, caption = sample.get("image"), sample.get("caption")
            if not isinstance(image, str) or not isinstance(caption, str):
                raise ValueError("Image-caption sample requires image and caption strings")

            if not image.strip() or not caption.strip():
                raise ValueError("Image-caption image and caption must be nonempty")

            source_path = sample.get(ONLINE_SOURCE_PATH_KEY)
            source_directory = os.path.dirname(os.path.abspath(source_path)) if source_path else os.getcwd()
            image_path = resolve_media_path(image, source_directory)
            blocks = [{"type": "image", "url": image_path}, {"type": "text", "text": caption}]
        tokenizer = self.processor.tokenizer
        bos_id, eos_id = tokenizer.bos_token_id, tokenizer.eos_token_id
        if bos_id is None or eos_id is None:
            raise ValueError("DeepSeek-V4.1 pretrain tokenizer requires BOS and EOS token IDs")

        # Validate text before image IO, including captions that tokenize to nothing.
        text_tokens = [tokenizer.encode(block["text"], add_special_tokens=False)
                       if block["type"] == "text" else [] for block in blocks]
        if not any(text_tokens):
            raise ValueError("DeepSeek-V4.1 pretrain caption/text must encode to at least one token")

        token_ids, token_type_ids = [bos_id], [TEXT]
        image_inputs = []
        for block, text_ids in zip(blocks, text_tokens):
            if block["type"] == "text":
                token_ids.extend(text_ids)
                token_type_ids.extend([TEXT] * len(text_ids))
                continue
            image_reference = block["url"]
            image_input = prepare_image_input({"url": image_reference}, self.processor, start=len(token_ids))
            image_inputs.append(image_input)
            token_ids.extend([self.processor.image_token_id] * image_input.types.numel())
            token_type_ids.extend(image_input.types.tolist())
        token_ids.append(eos_id)
        token_type_ids.append(TEXT)
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        token_types = torch.tensor(token_type_ids, dtype=torch.long)
        if input_ids.numel() > self.max_seq_len:
            logger.warning(
                "Skipping overlong DeepSeek-V4.1 pretrain sample: expanded_length=%d exceeds max_seq_len=%d",
                input_ids.numel(), self.max_seq_len,
            )
            return None

        labels = build_labels_from_supervised_start(input_ids, 1, media_mask=token_types != TEXT)
        encoded_sample = {
            "input_ids": input_ids,
            "token_types": token_types,
            "labels": labels,
            **self._build_image_fields(image_inputs),
        }
        aligned_sample = self._align_sample(encoded_sample)
        return aligned_sample


def build_deepseek_v41_omni_transform(
        *,
        processor: DeepseekV41Processor,
        training_mode: str,
        max_seq_len: int = 4096,
        thinking_mode: str = "chat",
        drop_thinking: bool = True,
        sample_format: str | None = None,
        sample_alignment_size: int = 1,
        download_verify_ssl: bool = True,
) -> OmniDataTransform:
    """Select the native transform using the explicitly configured dataset.training_mode.

    Args:
        processor: Model-owned tokenizer and native image processor settings.
        training_mode: Required ``sft`` for conversations or ``pretrain`` for raw text/images.
        max_seq_len: Maximum expanded sample length. Longer samples emit a warning
            and are skipped without truncating image spans.
        thinking_mode: SFT chat-template reasoning mode.
        drop_thinking: Whether the SFT template drops reasoning content.
        sample_format: messages/openai/hf/sharegpt for SFT, image_caption/interleaved for pretrain.
            None retains the existing messages (SFT) or image_caption (pretrain) default.
        sample_alignment_size: Pad each sample before packing to this multiple, with ignored labels.
            Use the least common multiple of positive CSA compression ratios (2 for CSA2).
            Defaults to 1 to preserve unpadded samples. Samples exceeding max_seq_len after padding are skipped.
        download_verify_ssl: Verify HTTPS certificates when recovering an image-caption image from its url.
            False disables certificate and hostname checks for that download request only.

    Returns:
        SFTTransform for final-reply supervision or PretrainTransform for image-caption/interleaved
        text supervision. Both produce unshifted labels for the existing Omni model loss.

    Raises:
        ValueError: If training_mode is neither ``sft`` nor ``pretrain``.
    """
    if training_mode == "pretrain":
        data_transform = DeepseekV41PretrainTransform(
            processor,
            max_seq_len=max_seq_len,
            sample_format="image_caption" if sample_format is None else sample_format,
            sample_alignment_size=sample_alignment_size,
            download_verify_ssl=download_verify_ssl,
        )
    elif training_mode == "sft":
        data_transform = DeepseekV41SFTTransform(
            processor,
            max_seq_len=max_seq_len,
            thinking_mode=thinking_mode,
            drop_thinking=drop_thinking,
            sample_format="messages" if sample_format is None else sample_format,
            sample_alignment_size=sample_alignment_size,
        )
    else:
        raise ValueError(f"Unsupported DeepSeek-V4.1 dataset training mode: {training_mode!r}")

    return data_transform


__all__ = [
    "DeepseekV41SFTTransform", "DeepseekV41PretrainTransform",
    "build_deepseek_v41_omni_transform",
]
