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
"""Image preprocessing.
An image becomes a `n_vit_h x n_vit_w` patch grid for the ViT and a `n_llm_h x n_llm_w` token grid
after the 3x3 aligner downsample, which the LLM sees as
    [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]
Every one of those positions carries `image_token_id` in `input_ids`; only the token type tells them
apart. The IMAGE slots are filled with aligner rows in reading order.
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from urllib.request import urlopen

import numpy as np
import torch  # pylint: disable=forbidden-backend-import
from PIL import Image, ImageOps

from hyper_parallel.models.deepseek_v41.adapter.data.encoding import IMAGE_PLACEHOLDER

TEXT = -1
IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)


@dataclass
class ImageInput:
    start: int
    patches: torch.Tensor
    n_vit_h: int
    n_vit_w: int
    types: torch.Tensor


def num_image_tokens(n_llm_h: int, n_llm_w: int) -> int:
    """Return the LLM token count for an image grid.

    Args:
        n_llm_h: Image grid height after aligner downsampling.
        n_llm_w: Image grid width after aligner downsampling.

    Returns:
        Number of image-span tokens consumed by the LLM.
    """
    image_token_count = n_llm_h * (n_llm_w + 1) + 2
    return image_token_count


def llm_grid(
        best_height: int,
        best_width: int,
        patch_size: int,
        downsample_ratio: int,
) -> Tuple[int, int]:
    """Calculate the token grid produced by the aligner.

    Args:
        best_height: Resized image height in pixels.
        best_width: Resized image width in pixels.
        patch_size: ViT patch size in pixels.
        downsample_ratio: Aligner downsampling ratio.

    Returns:
        Token-grid height and width.
    """
    grid_height = math.ceil((best_height // patch_size) / downsample_ratio)
    grid_width = math.ceil(
        (best_width // patch_size) / downsample_ratio
    )
    grid_shape = (grid_height, grid_width)
    return grid_shape


def solve_resize_ratio(
        height: Union[int, float],
        width: Union[int, float],
        patch_size: int,
        downsample_ratio: int,
        max_n_token: int,
) -> Tuple[int, int]:
    """Find the largest aspect-preserving size within the token limit.

    Args:
        height: Original image height.
        width: Original image width.
        patch_size: ViT patch size in pixels.
        downsample_ratio: Aligner downsampling ratio.
        max_n_token: Maximum number of image-span tokens.

    Returns:
        Resized height and width in pixels.
    """
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    cell = patch_size * downsample_ratio
    if max_w_float < 1.0:  # very tall: collapse to a single column
        resized_shape = ((max_n_token - 2) // 2 * cell, cell)
        return resized_shape
    if max_h_float < 1.0:  # very wide: collapse to a single row
        resized_shape = (cell, (max_n_token - 3) * cell)
        return resized_shape
    beta = min(math.floor(max_w_float) * cell / width, math.floor(max_h_float) * cell / height)
    resized_height = math.floor(height * beta / patch_size) * patch_size
    resized_width = math.floor(width * beta / patch_size) * patch_size
    resized_shape = (resized_height, resized_width)
    return resized_shape


def safe_resize(
        height: Union[int, float],
        width: Union[int, float],
        best_height: int,
        best_width: int,
        patch_size: int,
        downsample_ratio: int,
        max_n_token: int,
) -> Tuple[int, int, int, int]:
    """Shrink an image until it fits within the LLM token limit.

    Args:
        height: Original image height.
        width: Original image width.
        best_height: Candidate resized height.
        best_width: Candidate resized width.
        patch_size: ViT patch size in pixels.
        downsample_ratio: Aligner downsampling ratio.
        max_n_token: Maximum number of image-span tokens.

    Returns:
        LLM grid height, grid width, resized height, and resized width.
    """
    n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
    if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
        best_height, best_width = solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token)
        n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
        if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
            raise ValueError("image resize failed to satisfy the configured token limit")
    resize_plan = (n_llm_h, n_llm_w, best_height, best_width)
    return resize_plan


def load_image_bytes(record: Dict[str, Any]) -> bytes:
    """Load image bytes from supported record formats.

    Args:
        record: Image record containing bytes, base64 data, a URL, or a path.

    Returns:
        Encoded image bytes.
    """
    data = record.get("data")
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        image_bytes = base64.b64decode(data)
        return image_bytes

    source = record.get("source")
    if isinstance(source, dict):
        if source.get("data") is not None:
            image_bytes = base64.b64decode(source["data"])
            return image_bytes
        if source.get("url"):
            image_bytes = load_image_bytes({"url": source["url"]})
            return image_bytes

    url = record.get("url")
    if isinstance(url, str) and url:
        if url.startswith("data:"):
            header, _, payload = url.partition(",")
            if ";base64" not in header:
                raise ValueError(f"Unsupported data URL encoding: {header}")
            image_bytes = base64.b64decode(payload)
            return image_bytes
        if url.startswith(("http://", "https://")):
            with urlopen(url, timeout=30) as response:
                image_bytes = response.read()
            return image_bytes
        with open(url, "rb") as file:
            image_bytes = file.read()
        return image_bytes

    raise ValueError(f"Cannot load image from record: {list(record.keys())}")


def plan_image_grid(width: int, height: int, args: Any) -> Tuple[int, int, int, int]:
    """Build a resize plan for an image.

    Args:
        width: Original image width.
        height: Original image height.
        args: DeepSeek vision preprocessing configuration.

    Returns:
        LLM grid height, grid width, resized height, and resized width.
    """
    p = args.vision_patch_size
    if args.vision_max_wh_ratio is not None and width > height * args.vision_max_wh_ratio:
        width = height * args.vision_max_wh_ratio
    if 0 < width * height < args.vision_min_pixels:
        ratio = (args.vision_min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    resize_plan = safe_resize(
        height,
        width,
        best_height,
        best_width,
        p,
        args.vision_downsample_ratio,
        args.vision_max_n_token,
    )
    return resize_plan


def load_image(record: Dict[str, Any], args: Any) -> Tuple[torch.Tensor, int, int, int, int]:
    """Load and transform one image record into ViT patches.

    Args:
        record: Image record accepted by ``load_image_bytes``.
        args: DeepSeek vision preprocessing configuration.

    Returns:
        Patch tensor and ViT/LLM grid dimensions.
    """
    p = args.vision_patch_size
    with Image.open(io.BytesIO(load_image_bytes(record))) as source:
        image = source.convert("RGB")
    n_llm_h, n_llm_w, best_height, best_width = plan_image_grid(image.width, image.height, args)
    n_vit_h, n_vit_w = best_height // p, best_width // p
    if args.vision_max_wh_ratio is not None and image.width >= args.vision_max_wh_ratio * image.height:
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    x = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255
    x = ((x - 0.5) / 0.5).to(torch.bfloat16)
    patches = x.reshape(3, n_vit_h, p, n_vit_w, p).permute(1, 3, 0, 2, 4).reshape(n_vit_h * n_vit_w, 3, p, p)
    processed_image = (patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w)
    return processed_image


def image_token_types(n_llm_h: int, n_llm_w: int) -> torch.Tensor:
    """Default layout: the aligner grid in reading order, one IMAGE_NEW_LINE per row."""
    types = [IMAGE_START]
    types += ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
    types.append(IMAGE_END)
    token_types = torch.tensor(types, dtype=torch.int64)
    return token_types


def prepare_vl_inputs(
        prompt: str,
        images: Sequence[Dict[str, Any]],
        tokenizer: Any,
        args: Any,
) -> Tuple[List[int], List[int], Optional[List[ImageInput]]]:
    """Tokenize a prompt and expand each image placeholder into an image span.

    Args:
        prompt: Encoded conversation prompt.
        images: Image records ordered by their prompt placeholders.
        tokenizer: Tokenizer used by the language model.
        args: DeepSeek vision preprocessing configuration.

    Returns:
        Token IDs, token-type IDs, and optional prepared image inputs. Image-span
        positions carry ``args.image_token_id`` and token types distinguish their
        semantic roles.
    """
    # The placeholder is spelled differently across tokenizer revisions, so the id comes from the
    # config; only cross-check it when this tokenizer does know the training-time spelling.
    image_token_id = args.image_token_id
    placeholder_id = tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
    if placeholder_id is not None and placeholder_id != tokenizer.unk_token_id:
        if placeholder_id != image_token_id:
            raise ValueError(
                "tokenizer image placeholder does not match the configured image token id: "
                f"{placeholder_id} != {image_token_id}"
            )
    prompt_tokens = tokenizer.encode(prompt)
    num_placeholders = sum(token == image_token_id for token in prompt_tokens)
    if num_placeholders != len(images):
        raise ValueError(f"Found {num_placeholders} image tokens but got {len(images)} images")
    if num_placeholders and not args.vision_enabled:
        raise ValueError("The model config has no vision tower (vision_n_layers == 0) but the prompt contains images")

    tokens, token_types, image_inputs = [], [], []
    image_iter = iter(images)
    for tok in prompt_tokens:
        if tok != image_token_id:
            tokens.append(tok)
            token_types.append(TEXT)
            continue
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = load_image(next(image_iter), args)
        types = image_token_types(n_llm_h, n_llm_w)
        image_inputs.append(ImageInput(len(tokens), patches, n_vit_h, n_vit_w, types))
        tokens += [image_token_id] * types.numel()
        token_types += types.tolist()
    prepared_images = image_inputs or None
    prepared_inputs = (tokens, token_types, prepared_images)
    return prepared_inputs
