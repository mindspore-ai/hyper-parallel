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
"""Field rules isolated from the text data path."""

from hyper_parallel.data.constants import IGNORE_INDEX

ONLINE_SOURCE_PATH_KEY = "__online_source_path__"

DEFAULT_FIELD_PACK_DIMS = {
    "input_ids": -1,
    "labels": -1,
    "attention_mask": -1,
    "loss_mask": -1,
    "position_ids": -1,
    "text_position_ids": -1,
    "router_attention_mask": -1,
    "mm_token_type_ids": -1,
    "token_types": -1,
    "pixel_values": 0,
    "pixel_values_videos": 0,
    "image_grid_thw": 0,
    "video_grid_thw": 0,
    "audio_values": 0,
}
PACKING_METADATA_FIELDS = {"packing_length", "cu_seq_lens"}
OMNI_CONCAT_FIELDS = {
    "audio_values",
    "image_grid_thw",
    "pixel_values",
    "pixel_values_videos",
    "video_grid_thw",
}
OMNI_LOSS_INPUT_FIELDS = {"labels", "loss_mask", "stream_loss_mask"}
OMNI_INTERNAL_FIELDS = {"loss_mask", "stream_loss_mask", "cu_seq_lens", "packing_length"}
OMNI_CP_TOKEN_FIELDS = frozenset({
    "input_ids", "labels", "loss_mask", "stream_loss_mask", "attention_mask", "position_ids",
    "text_position_ids", "router_attention_mask", "mm_token_type_ids", "token_types",
})
OMNI_CP_PAD_VALUES = {"labels": IGNORE_INDEX, "token_types": -1, "mm_token_type_ids": -1}
