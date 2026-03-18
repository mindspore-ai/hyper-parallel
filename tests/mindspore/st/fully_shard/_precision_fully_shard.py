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
"""Fully_shard multi-card training for precision comparison"""
import os
import numpy as np
import mindspore as ms
import mindspore.dataset as ds

from mindspore._c_expression import NoFallbackGuard
from mindspore.communication import get_rank, get_group_size
from mindspore import nn, ops
from mindspore.communication import init
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel import init_device_mesh
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.dtensor.dtensor import DTensor
from tests.mindspore.st.common_net import SlimLeNet16

# Use to same temp directory as precision_baseline.py
TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp_baseline")


ms.set_seed(1)
ms.set_deterministic(True)


def create_dataset(local_batch_size: int, num_shards: int, shard_id: int):
    """create mnist dataset"""
    dataset_path = "/home/workspace/mindspore_dataset/mnist/train"
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


def compare_losses(baseline_losses: list, fully_shard_losses: list, rtol: float = 1e-5, atol: float = 1e-5):
    """Compare losses between baseline and fully_shard"""
    assert len(baseline_losses) == len(fully_shard_losses), \
        f"Loss count mismatch: baseline={len(baseline_losses)}, fully_shard={len(fully_shard_losses)}"

    for i, (baseline_loss, fs_loss) in enumerate(zip(baseline_losses, fully_shard_losses)):
        rel_error = abs(fs_loss - baseline_loss) / (abs(baseline_loss) + 1e-10)
        abs_error = abs(fs_loss - baseline_loss)

        print(f"Step {i}: baseline={baseline_loss:.6f}, fully_shard={fs_loss:.6f}, "
              f"rel_error={rel_error:.2e}, abs_error={abs_error:.2e}")

        assert np.isclose(fs_loss, baseline_loss, rtol=rtol, atol=atol), \
            f"Step {i}: Loss mismatch - baseline={baseline_loss:.6f}, fully_shard={fs_loss:.6f}, " \
            f"rel_error={rel_error:.2e}, abs_error={abs_error:.2e}"


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


def run_fully_shard_multi_card(ckpt_path, mesh):
    """Run fully_shard multi-card training"""
    dp_size = get_group_size()
    rank_id = get_rank()

    data_set = create_dataset(
        local_batch_size=local_bs, num_shards=dp_size, shard_id=rank_id)
    net = SlimLeNet16()
    param_dict = ms.load_checkpoint(ckpt_path)
    param_not_load, _ = ms.load_param_into_net(net, param_dict)
    assert not param_not_load, f"For fully_shard test case, not completely load ckpt from {ckpt_path}"

    mp_policy = MixedPrecisionPolicy(
        param_dtype=ms.float32,
        reduce_dtype=ms.float32,
        output_dtype=ms.float32,
        cast_forward_inputs=False
    )

    origin_shapes = [p.shape for p in net.trainable_params()]
    shard_dim_size = mesh.shape[-1]
    print("shard dim size is ", shard_dim_size)

    fully_shard(net.dense_relu_sequential[0], mesh=mesh, mp_policy=mp_policy)
    fully_shard(net.dense_relu_sequential[2], mesh=mesh, mp_policy=mp_policy)
    fully_shard(net.dense_relu_sequential[4], mesh=mesh, mp_policy=mp_policy)
    fully_shard(net, mesh=mesh, mp_policy=mp_policy)

    for idx, param in enumerate(net.trainable_params()):
        assert isinstance(param, DTensor), f"Parameter {idx} is not a DTensor"

        local_shape = param.to_local().shape
        original_shape = origin_shapes[idx]
        expected_local_shape = (original_shape[0] // shard_dim_size,) + original_shape[1:]
        assert local_shape == expected_local_shape, (
            f"Shape mismatch at index {idx}: "
            f"Expected {expected_local_shape}, got {local_shape}. "
            f"Original shape was {original_shape}, shard_size={shard_dim_size}"
        )

    grad_fn = ms.value_and_grad(get_forward_fn(net), None, net.trainable_params(), has_aux=True)
    optimizer = nn.Adam(net.trainable_params(), learning_rate=learning_rate)
    loss_sync_allreduce = ops.AllReduce(ops.ReduceOp.SUM)

    losses = []
    i = 0
    for data, label in data_set:
        net.zero_grad()
        (loss, _), grads = grad_fn(data, label)
        with NoFallbackGuard():
            optimizer(grads)
        reduced_loss = loss_sync_allreduce(loss)
        final_loss = reduced_loss / dp_size
        if rank_id == 0:
            losses.append(float(final_loss.asnumpy()))
            print(f"step: {i}, loss: {final_loss}")
        i += 1
        if i >= max_step:
            break

    return losses


def run_fully_shard_multi_card_ignored(ckpt_path, mesh):
    """Run fully_shard multi-card training with replicate_params."""
    dp_size = get_group_size()
    rank_id = get_rank()

    data_set = create_dataset(
        local_batch_size=local_bs, num_shards=dp_size, shard_id=rank_id
    )
    net = SlimLeNet16()
    param_dict = ms.load_checkpoint(ckpt_path)
    param_not_load, _ = ms.load_param_into_net(net, param_dict)
    assert not param_not_load, f"For fully_shard replicate_params test, not completely load ckpt from {ckpt_path}"

    mp_policy = MixedPrecisionPolicy(
        param_dtype=ms.float32,
        reduce_dtype=ms.float32,
        output_dtype=ms.float32,
        cast_forward_inputs=False,
    )

    replicate_params = {p for p in net.trainable_params() if "bias" in p.name}

    fully_shard(net, mesh=mesh, mp_policy=mp_policy, replicate_params=replicate_params)

    grad_fn = ms.value_and_grad(get_forward_fn(net), None, net.trainable_params(), has_aux=True)
    optimizer = nn.Adam(net.trainable_params(), learning_rate=learning_rate)
    loss_sync_allreduce = ops.AllReduce(ops.ReduceOp.SUM)

    losses = []
    i = 0
    for data, label in data_set:
        net.zero_grad()
        (loss, _), grads = grad_fn(data, label)
        with NoFallbackGuard():
            optimizer(grads)
        reduced_loss = loss_sync_allreduce(loss)
        final_loss = reduced_loss / dp_size
        if rank_id == 0:
            losses.append(float(final_loss.asnumpy()))
            print(f"[replicate_params] step: {i}, loss: {final_loss}")
        i += 1
        if i >= max_step:
            break

    return losses


def run_fully_shard(mesh):
    """Run fully_shard with different mesh"""
    init()

    rank_id = get_rank()

    ckpt_path = os.path.join(TEMP_DIR, "init_baseline.ckpt")
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}, please run precision_baseline.py first"

    losses = run_fully_shard_multi_card(ckpt_path, mesh)

    if rank_id == 0:
        losses_file = os.path.join(TEMP_DIR, "fully_shard_losses.npy")
        np.save(losses_file, np.array(losses))
        print(f"Saved fully_shard losses to: {losses_file}")
        print(f"Fully_shard losses: {losses[:5]}... (total {len(losses)} steps)")

        baseline_losses_file = os.path.join(TEMP_DIR, "baseline_losses.npy")
        assert os.path.exists(baseline_losses_file), f"Baseline losses not found: {baseline_losses_file}"

        baseline_losses = list(np.load(baseline_losses_file))
        print(f"Loaded baseline losses: {baseline_losses[:5]}... (total {len(baseline_losses)} steps)")

        compare_losses(baseline_losses, losses, rtol=1e-5, atol=1e-5)
        print("Precision comparison passed!")


def run_fully_shard_ignored(mesh):
    """Run fully_shard with replicate_params and compare with baseline."""
    init()

    rank_id = get_rank()

    ckpt_path = os.path.join(TEMP_DIR, "init_baseline.ckpt")
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}, please run precision_baseline.py first"

    losses = run_fully_shard_multi_card_ignored(ckpt_path, mesh)

    if rank_id == 0:
        losses_file = os.path.join(TEMP_DIR, "fully_shard_ignored_losses.npy")
        np.save(losses_file, np.array(losses))
        print(f"Saved fully_shard(replicate_params) losses to: {losses_file}")
        print(f"Fully_shard(ignored) losses: {losses[:5]}... (total {len(losses)} steps)")

        baseline_losses_file = os.path.join(TEMP_DIR, "baseline_losses.npy")
        assert os.path.exists(baseline_losses_file), f"Baseline losses not found: {baseline_losses_file}"

        baseline_losses = list(np.load(baseline_losses_file))
        print(f"Loaded baseline losses: {baseline_losses[:5]}... (total {len(baseline_losses)} steps)")

        compare_losses(baseline_losses, losses, rtol=1e-5, atol=1e-5)
        print("Precision comparison with replicate_params passed!")


def test_ms_zero3_fully_shard():
    """
    Feature: Compare fully_shard precision with standalone baseline
    Description: Run standalone baseline and fully_shard multi-card training, then compare losses on rank 0
    Expectation: Losses should match within tolerance
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    run_fully_shard(mesh)


def test_ms_zero3_partial_shard():
    """
    Feature: Compare partial_shard precision with standalone baseline
    Description: Run standalone baseline and partial_shard multi-card training, then compare losses on rank 0
    Expectation: Losses should match within tolerance
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "op"))
    run_fully_shard(mesh)


def test_ms_zero3_fully_shard_replicate_params():
    """
    Feature: Compare fully_shard(replicate_params) precision with standalone baseline
    Description: Run standalone baseline and fully_shard multi-card training with replicate_params,
                 then compare losses on rank 0
    Expectation: Losses should match within tolerance
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    run_fully_shard_ignored(mesh)
