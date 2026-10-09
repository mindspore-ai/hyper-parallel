# Copyright 2025-2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Static constants used by text, Omni, Online, batching, and indexed datasets."""


# Text and Omni token semantics.
IGNORE_INDEX = -100
ROLE_SUPPORTED = ["system", "user", "assistant", "tool"]
TEXT_LAZY_EXPORTS = {
    "AutoTokenizer": "hyper_parallel.data.text.build_tokenizer",
    "ChatTemplate": "hyper_parallel.data.text.chat_template",
    "IdentityDataTransform": "hyper_parallel.data.text.text_transform",
    "PlaintextTransform": "hyper_parallel.data.text.text_transform",
    "TextConversationTransform": "hyper_parallel.data.text.text_transform",
    "build_chat_template": "hyper_parallel.data.text.chat_template",
    "build_indexed_text_dataset": "hyper_parallel.data.text.build_dataset",
    "build_text_transform": "hyper_parallel.data.text.text_transform",
    "build_online_iterable_dataset": "hyper_parallel.data.text.build_dataset",
    "build_online_text_mapping_dataset": "hyper_parallel.data.text.build_dataset",
    "build_tokenizer": "hyper_parallel.data.text.build_tokenizer",
}
IMAGE_INPUT_INDEX = -200
VIDEO_INPUT_INDEX = -300
AUDIO_INPUT_INDEX = -400
IMAGE_OUTPUT_INDEX = -201
VIDEO_OUTPUT_INDEX = -301
AUDIO_OUTPUT_INDEX = -401
TYPE2INDEX = {
    "input": {
        "image": IMAGE_INPUT_INDEX,
        "video": VIDEO_INPUT_INDEX,
        "audio": AUDIO_INPUT_INDEX,
    },
    "output": {
        "image": IMAGE_OUTPUT_INDEX,
        "video": VIDEO_OUTPUT_INDEX,
        "audio": AUDIO_OUTPUT_INDEX,
    },
}
MODALITY = TYPE2INDEX["input"].keys() | TYPE2INDEX["output"].keys()

# Online source and split policy.
ONLINE_SPLIT_NAMES = ("train", "valid", "test")
ONLINE_SPLIT_COUNT = len(ONLINE_SPLIT_NAMES)
ONLINE_SOURCE_PATH_KEY = "__online_source_path__"
ONLINE_STOPPING_STRATEGIES = (
    "first_exhausted",
    "all_exhausted",
    "all_exhausted_without_replacement",
)
ONLINE_BLEND_STRATEGIES = ("local_weighted", "global_interleave")
ONLINE_FILE_FORMATS = {
    ".arrow": "arrow",
    ".csv": "csv",
    ".json": "json",
    ".jsonl": "json",
    ".parquet": "parquet",
}

# Packing and Omni batch field rules.
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
OMNI_CP_PAD_VALUES = {"labels": -100, "token_types": -1, "mm_token_type_ids": -1}

# Indexed dataset storage and object-store paths.
INDEX_HEADER = b"MMIDIDX\x00\x00"
INDEX_VERSION = 1
INDEX_DTYPE_CODES = {
    1: "uint8",
    2: "int8",
    3: "int16",
    4: "int32",
    5: "int64",
    6: "float64",
    7: "float32",
    8: "uint16",
}
PAD_TOKEN_ID = -1
S3_PREFIX = "s3://"
MSC_PREFIX = "msc://"

# Dataset logger presentation.
DATASET_LOGGER_NAME = "hyper_parallel.data"
DATASET_LOG_FORMAT = (
    "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] "
    "[rank:%(rank_id)d] \t> %(message)s"
)
DATASET_DATE_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
