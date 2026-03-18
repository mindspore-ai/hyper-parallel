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
"""Standalone baseline training for precision comparison"""
import os
from typing import Optional
import numpy as np
import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn
from tests.mindspore.st.common_net import SlimLeNet16

# Use a fixed temp directory that won't be deleted when subprocess exits
TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp_baseline")

ms.set_seed(1)
ms.set_deterministic(True)


def create_dataset(local_batch_size: int, num_shards: Optional[int] = None, shard_id: Optional[int] = None):
    """create mnist dataset"""
    dataset_path = "/home/workspace/mindspore_dataset/mnist/train"
    if (num_shards is None) or (shard_id is None):
        dataset = ds.MnistDataset(dataset_path, shuffle=False)
    else:
        dataset = ds.MnistDataset(
            dataset_path, num_shards=num_shards, shard_id=shard_id, shuffle=False)
    image_transforms = [
        ds.vision.Rescale(1.0 / 255.0, 0),
        ds.vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        ds.vision.HWC2CHW()
    ]
    label_transform = ds.transforms.TypeCast(ms.int32)
    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    dataset = dataset.batch(local_batch_size)
    return dataset


loss_fn = nn.CrossEntropyLoss()


def get_forward_fn(net):
    def forward_fn(data, label):
        logits = net(data)
        loss = loss_fn(logits, label)
        return loss, logits
    return forward_fn


# Global hyper parameters:
local_bs = 32
learning_rate = 1e-3
max_step = 20


def generate_checkpoint():
    """Generate initial checkpoint"""
    os.makedirs(TEMP_DIR, exist_ok=True)
    net = SlimLeNet16()
    ckpt_path = os.path.join(TEMP_DIR, "init_baseline.ckpt")
    ms.save_checkpoint(net, ckpt_path)
    print(f"Generated checkpoint at: {ckpt_path}")
    return ckpt_path


def run_baseline_standalone(ckpt_path: str):
    """Run standalone single-card training"""
    data_set = create_dataset(local_batch_size=local_bs * 8)
    net = SlimLeNet16()
    param_dict = ms.load_checkpoint(ckpt_path)
    param_not_load, _ = ms.load_param_into_net(net, param_dict)
    assert not param_not_load, f"For baseline test case, not completely load ckpt from {ckpt_path}"

    optimizer = nn.Adam(net.trainable_params(), learning_rate=learning_rate)
    grad_fn = ms.value_and_grad(get_forward_fn(net), None, net.trainable_params(), has_aux=True)

    losses = []
    i = 0
    for data, label in data_set:
        (loss, _), grads = grad_fn(data, label)
        optimizer(grads)
        losses.append(float(loss.asnumpy()))
        print(f"step: {i}, loss: {loss}")
        i += 1
        if i >= max_step:
            break

    return losses


def test_generate_checkpoint_and_baseline():
    """
    Feature: generate standalone baseline artifacts for fully shard precision test.
    Description: Generate initial checkpoint, run baseline training, and dump baseline losses.
    Expectation: Run success and save baseline losses to a local npy file.
    """
    ckpt_path = generate_checkpoint()
    losses = run_baseline_standalone(ckpt_path)

    losses_file = os.path.join(TEMP_DIR, "baseline_losses.npy")
    np.save(losses_file, np.array(losses))
    print(f"Saved baseline losses to: {losses_file}")
    print(f"Baseline losses: {losses[:5]}... (total {len(losses)} steps)")


if __name__ == "__main__":
    test_generate_checkpoint_and_baseline()
