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
"""Small real-image corpus and deterministic processor for native VLM tests."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from hyper_parallel.data.vlm.build_data_transform import VLMChatTransform
from hyper_parallel.data.vlm.dataset import build_vlm_dataset


class ImageTestProcessor:
    """Decode local images while avoiding remote tokenizer/model dependencies."""

    chat_template = "test"

    def apply_chat_template(self, messages: list, *, add_generation_prompt: bool, **kwargs: object) -> dict:
        """Emit processor-shaped fields with distinct text/image ownership markers."""
        del kwargs
        index = int(messages[0]["content"][0]["text"].split()[-1])
        tokens, modality = [100 + index], [0]
        grids, pixels = [], []
        for message in messages:
            for part in message["content"]:
                if part["type"] != "image":
                    continue
                with Image.open(part["url"]) as image:
                    rgb = torch.tensor(np.asarray(image.convert("RGB")), dtype=torch.float32)
                    height, width = image.height // 2, image.width // 2
                    patches = rgb.reshape(height, 2, width, 2, 3).permute(0, 2, 1, 3, 4).reshape(-1, 12)
                grids.append([1, height, width])
                pixels.append(patches)
                tokens.extend([99] * (height * width // 4) + [50])
                modality.extend([1] * (height * width // 4) + [0])
        tokens.append(51)
        modality.append(0)
        if not add_generation_prompt:
            tokens.extend([200 + index] * (index % 3 + 1))
            modality.extend([0] * (index % 3 + 1))
        return {
            "input_ids": [tokens], "attention_mask": [[1] * len(tokens)], "mm_token_type_ids": [modality],
            "pixel_values": torch.cat(pixels) if pixels else torch.empty((0, 12)),
            "image_grid_thw": torch.tensor(grids, dtype=torch.long).reshape(-1, 3),
        }


def build_image_corpus(directory: str, size: int = 18) -> object:
    """Build real JSON/image input through the unmodified HP Dataset and transform."""
    root = Path(directory)
    records = []
    shapes = (((16, 16), (8, 8)), ((16, 16),), ((4, 4),), ())
    for index in range(size):
        images = []
        for image_index, shape in enumerate(shapes[index % 4]):
            name = f"sample_{index}_{image_index}.png"
            Image.new("RGB", shape, color=(index, image_index, 7)).save(root / name)
            images.append(name)
        records.append({
            "images": images,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": f"sample {index}"}] +
                 [{"type": "image", "url": name} for name in images]},
                {"role": "assistant", "content": "answer"},
            ],
        })
    path = root / "conversations.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return build_vlm_dataset(
        data_path=str(path), data_config={"source_type": "online"},
        transform=VLMChatTransform(ImageTestProcessor(), max_seq_len=64),
    )


def vlm_loader_target(**kwargs: object) -> SimpleNamespace:
    """Return Trainer DataLoader knobs for lightweight CPU integration tests."""
    options = {"dataloader_type": "single", "drop_last": True, "num_workers": 0, "pin_memory": False}
    options.update(kwargs)
    return SimpleNamespace(**options)
