# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Load model parameters from checkpoint file."""
import mindspore as ms


def load_checkpoint(file_path: str):
    """
    Load model parameters from checkpoint file.

    Args:
        file_path (str): Path to the checkpoint file.

    Returns:
        dict: A dictionary containing the loaded parameters.
"""
    param_dict = ms.load_checkpoint(ckpt_file_name=file_path, format="safetensors")
    return param_dict
