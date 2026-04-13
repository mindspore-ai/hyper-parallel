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
import argparse
import os

import mindspore as ms
import mindspore.dataset as ds
import numpy as np
from hyper_parallel import SkipDTensorDispatch, init_device_mesh
from hyper_parallel.core.activation_checkpoint import checkpoint_wrapper, swap_wrapper, CheckpointPolicy, SwapManager
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.init_weights import init_empty_weights
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.platform.mindspore.autograd_compat import enable_mindspore_backward_compat
from mindspore import nn, ops
from mindspore.communication import get_group_size, get_rank, init
from tests.mindspore.st.common_net import SlimLeNet16

# Shared temp directory for generated baseline artifacts
TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp_baseline")

ms.set_seed(1)
ms.set_deterministic(True)
enable_mindspore_backward_compat()


def create_dataset(local_batch_size: int, num_shards=None, shard_id=None):
    """create mnist dataset"""
    dataset_path = "/home/workspace/mindspore_dataset/mnist/train"
    if (num_shards is None) or (shard_id is None):
        dataset = ds.MnistDataset(dataset_path, shuffle=False)
    else:
        dataset = ds.MnistDataset(
            dataset_path, num_shards=num_shards, shard_id=shard_id, shuffle=False
        )
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


def setup_prefetch(net):
    """Configure one-hop forward/backward prefetch for wrapped child modules."""
    wrapped_modules = [
        net.dense_relu_sequential[0],
        net.dense_relu_sequential[2],
        net.dense_relu_sequential[4],
    ]

    for i, layer in enumerate(wrapped_modules[:-1]):
        layer.set_modules_to_forward_prefetch([wrapped_modules[i + 1]])

    wrapped_modules.reverse()
    for i, layer in enumerate(wrapped_modules[:-1]):
        layer.set_modules_to_backward_prefetch([wrapped_modules[i + 1]])


def apply_recompute(net):
    """Wrap fully_shard-target child modules with activation recompute."""
    def recomp_policy_fn(ctx, op, *args, **kwargs):  # pylint: disable=W0613,W0621
        return CheckpointPolicy.MUST_RECOMPUTE
    def swap_policy_fn(ctx, op, *args, **kwargs):  # pylint: disable=W0613,W0621
        return CheckpointPolicy.MUST_SWAP
    net.dense_relu_sequential[0] = swap_wrapper(net.dense_relu_sequential[0])
    net.dense_relu_sequential[1] = checkpoint_wrapper(net.dense_relu_sequential[1], policy_fn=recomp_policy_fn)
    net.dense_relu_sequential[2] = checkpoint_wrapper(net.dense_relu_sequential[2], policy_fn=swap_policy_fn)
    net.dense_relu_sequential[3] = checkpoint_wrapper(net.dense_relu_sequential[3])

    for i in range(len(net.dense_relu_sequential) - 1):
        SwapManager().set_forward_prefetch_layer(net.dense_relu_sequential[i], net.dense_relu_sequential[i+1])



def get_backward_grads(net, expected_dtype=None):
    """Collect DTensor gradients from fully_shard-managed DTensor params."""
    grads = []
    for idx, param in enumerate(net.trainable_params()):
        grad = param.grad
        assert isinstance(grad, DTensor), f"Parameter grad {idx} is not a DTensor"
        assert grad.shape == param.shape, (
            f"Gradient global shape mismatch at index {idx}: "
            f"Expected {param.shape}, got {grad.shape}"
        )
        assert grad.local_shape == param.local_shape, (
            f"Gradient local shape mismatch at index {idx}: "
            f"Expected {param.local_shape}, got {grad.local_shape}"
        )
        if expected_dtype is not None:
            assert grad.dtype == expected_dtype, (
                f"Returned grad {idx} dtype mismatch: expected {expected_dtype}, got {grad.dtype}"
            )
        grads.append(grad)
    return tuple(grads)


def assert_sharded_param_layout(
    net,
    origin_shapes=None,
    shard_dim_size=None,
    expected_dtype=None,
):
    """Validate that trainable params stay as sharded DTensor parameters."""
    param_type_names = []
    for idx, param in enumerate(net.trainable_params()):
        param_type_names.append(type(param).__name__)
        assert isinstance(param, DTensor), f"Parameter {idx} is not a DTensor"
        if expected_dtype is not None:
            assert param.dtype == expected_dtype, (
                f"Parameter {idx} dtype mismatch: expected {expected_dtype}, got {param.dtype}"
            )
        if origin_shapes is None or shard_dim_size is None:
            continue
        local_shape = param.to_local().shape
        original_shape = origin_shapes[idx]
        expected_local_shape = (original_shape[0] // shard_dim_size,) + original_shape[1:]
        assert local_shape == expected_local_shape, (
            f"Shape mismatch at index {idx}: "
            f"Expected {expected_local_shape}, got {local_shape}. "
            f"Original shape was {original_shape}, shard_size={shard_dim_size}"
        )
    return param_type_names


def run_accumulated_step(data, label, forward_fn, net, sync_grad_on_last_micro_step=False):
    """Run one fully_shard step with per-micro-step gradient accumulation."""
    micro_step = 4
    micro_size = data.shape[0] // micro_step
    data_list = ops.split(data, micro_size)
    label_list = ops.split(label, micro_size)
    assert len(data_list) >= micro_step, (
        f"Expected at least {micro_step} micro batches, got {len(data_list)}"
    )
    total_loss = 0
    for micro_idx in range(micro_step):
        if sync_grad_on_last_micro_step:
            net.set_requires_gradient_sync(micro_idx == micro_step - 1)
        loss, _ = forward_fn(data_list[micro_idx], label_list[micro_idx])
        loss.backward()
        total_loss = total_loss + loss
    if sync_grad_on_last_micro_step:
        net.set_requires_gradient_sync(True)
    grads = get_backward_grads(net)
    return total_loss, grads, micro_step


# Global hyper parameters:
local_bs = 32
learning_rate = 1e-2
max_step = 20


def generate_checkpoint():
    """Generate the shared initial checkpoint for baseline and fully_shard runs."""
    os.makedirs(TEMP_DIR, exist_ok=True)
    net = SlimLeNet16()
    ckpt_path = os.path.join(TEMP_DIR, "init_baseline.ckpt")
    ms.save_checkpoint(net, ckpt_path)
    print(f"Generated checkpoint at: {ckpt_path}")
    return ckpt_path


def run_baseline_standalone(ckpt_path: str):
    """Run standalone single-card training with the same backward path as distributed."""
    data_set = create_dataset(local_batch_size=local_bs * 8)
    net = SlimLeNet16()
    param_dict = ms.load_checkpoint(ckpt_path)
    param_not_load, _ = ms.load_param_into_net(net, param_dict)
    assert not param_not_load, f"For baseline test case, not completely load ckpt from {ckpt_path}"

    params = tuple(net.trainable_params())
    optimizer = nn.Adam(params, learning_rate=learning_rate)

    losses = []
    i = 0
    for data, label in data_set:
        for param in params:
            param.grad = None
        loss, _ = get_forward_fn(net)(data, label)
        loss.backward()
        grads = tuple(param.grad for param in params)
        optimizer(grads)
        losses.append(float(loss.asnumpy()))
        print(f"step: {i}, loss: {loss}")
        i += 1
        if i >= max_step:
            break

    return losses


def generate_baseline_artifacts():
    """Generate initial checkpoint and standalone baseline losses."""
    ckpt_path = generate_checkpoint()
    losses = run_baseline_standalone(ckpt_path)

    os.makedirs(TEMP_DIR, exist_ok=True)
    losses_file = os.path.join(TEMP_DIR, "baseline_losses.npy")
    np.save(losses_file, np.array(losses))
    print(f"Saved baseline losses to: {losses_file}")
    print(f"Baseline losses: {losses[:5]}... (total {len(losses)} steps)")


def run_fully_shard_multi_card(
    ckpt_path,
    mesh,
    accumulate_grad=False,
    sync_grad_on_last_micro_step=False,
    replicate_all_params=False,
    enable_prefetch=False,
    enable_recompute=False,
):
    """Run fully_shard multi-card training."""
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
    replicate_params = set(net.trainable_params()) if replicate_all_params else None

    if enable_recompute:
        apply_recompute(net)

    fully_shard(
        net.dense_relu_sequential[0], mesh=mesh, mp_policy=mp_policy, replicate_params=replicate_params
    )
    fully_shard(
        net.dense_relu_sequential[2], mesh=mesh, mp_policy=mp_policy, replicate_params=replicate_params
    )
    fully_shard(
        net.dense_relu_sequential[4], mesh=mesh, mp_policy=mp_policy, replicate_params=replicate_params
    )
    fully_shard(net, mesh=mesh, mp_policy=mp_policy, replicate_params=replicate_params)
    if enable_prefetch:
        setup_prefetch(net)

    if not replicate_all_params:
        assert_sharded_param_layout(net, origin_shapes, shard_dim_size)

    forward_fn = get_forward_fn(net)
    optimizer = nn.Adam(net.trainable_params(), learning_rate=learning_rate)
    loss_sync_allreduce = ops.AllReduce(ops.ReduceOp.SUM)

    losses = []
    final_param_type_names = []
    i = 0
    for data, label in data_set:
        net.zero_grad()
        if accumulate_grad:
            total_loss, grads, micro_step = run_accumulated_step(
                data, label, forward_fn, net,
                sync_grad_on_last_micro_step=sync_grad_on_last_micro_step
            )
            with SkipDTensorDispatch():
                for grad in grads:
                    grad /= micro_step
                optimizer(grads)
            reduced_loss = loss_sync_allreduce(total_loss)
            final_loss = reduced_loss / (micro_step * dp_size)
        else:
            loss, _ = forward_fn(data, label)
            loss.backward()
            grads = get_backward_grads(net)
            with SkipDTensorDispatch():
                optimizer(grads)
            reduced_loss = loss_sync_allreduce(loss)
            final_loss = reduced_loss / dp_size
        if not replicate_all_params:
            final_param_type_names = assert_sharded_param_layout(net)
        if rank_id == 0:
            losses.append(float(final_loss.asnumpy()))
            print(f"rank: {rank_id} step: {i}, loss: {final_loss}")
        i += 1
        if i >= max_step:
            break

    return losses, final_param_type_names


def run_fully_shard_multi_card_with_empty_init(
    ckpt_path,
    mesh,
    accumulate_grad=False,
):
    """Run fully_shard with empty initializer multi-card training."""
    dp_size = get_group_size()
    rank_id = get_rank()
    data_set = create_dataset(
        local_batch_size=local_bs, num_shards=dp_size, shard_id=rank_id)
    mp_policy = MixedPrecisionPolicy(
        param_dtype=ms.float32,
        reduce_dtype=ms.float32,
        output_dtype=ms.float32,
        cast_forward_inputs=False
    )

    with init_empty_weights():
        net = SlimLeNet16()
        fully_shard(net.dense_relu_sequential[0], mesh=mesh, mp_policy=mp_policy)
        fully_shard(net.dense_relu_sequential[2], mesh=mesh, mp_policy=mp_policy)
        fully_shard(net.dense_relu_sequential[4], mesh=mesh, mp_policy=mp_policy)
        fully_shard(net, mesh=mesh, mp_policy=mp_policy)

    origin_shapes = [p.shape for p in net.trainable_params()]
    shard_dim_size = mesh.shape[-1]
    assert_sharded_param_layout(net, origin_shapes, shard_dim_size)

    param_dict = ms.load_checkpoint(ckpt_path)
    net.load_state_dict(param_dict, strict=True)
    forward_fn = get_forward_fn(net)
    optimizer = nn.Adam(net.trainable_params(), learning_rate=learning_rate)
    loss_sync_allreduce = ops.AllReduce(ops.ReduceOp.SUM)

    losses = []
    final_param_type_names = []
    i = 0
    for data, label in data_set:
        net.zero_grad()
        if accumulate_grad:
            total_loss, grads, micro_step = run_accumulated_step(
                data, label, forward_fn, net
            )
            with SkipDTensorDispatch():
                for grad in grads:
                    grad /= micro_step
                optimizer(grads)
            reduced_loss = loss_sync_allreduce(total_loss)
            final_loss = reduced_loss / (micro_step * dp_size)
        else:
            loss, _ = forward_fn(data, label)
            loss.backward()
            grads = get_backward_grads(net)
            with SkipDTensorDispatch():
                optimizer(grads)
            reduced_loss = loss_sync_allreduce(loss)
            final_loss = reduced_loss / dp_size
        final_param_type_names = assert_sharded_param_layout(net)
        if rank_id == 0:
            losses.append(float(final_loss.asnumpy()))
            print(f"rank: {rank_id} step: {i}, loss: {final_loss}")
        i += 1
        if i >= max_step:
            break

    return losses, final_param_type_names


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

    replicate_params = {p for p in net.trainable_params() if "weight" in p.name}

    fully_shard(net, mesh=mesh, mp_policy=mp_policy, replicate_params=replicate_params)

    forward_fn = get_forward_fn(net)
    optimizer = nn.Adam(net.trainable_params(), learning_rate=learning_rate)
    loss_sync_allreduce = ops.AllReduce(ops.ReduceOp.SUM)

    losses = []
    i = 0
    for data, label in data_set:
        net.zero_grad()
        loss, _ = forward_fn(data, label)
        loss.backward()
        grads = get_backward_grads(net)
        with SkipDTensorDispatch():
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


def run_fully_shard(mesh, use_empty_weight=False, enable_prefetch=False, enable_recompute=False):
    """Run fully_shard with different mesh"""
    init()

    rank_id = get_rank()

    ckpt_path = os.path.join(TEMP_DIR, "init_baseline.ckpt")
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}, please generate baseline artifacts first"

    if use_empty_weight:
        losses, _ = run_fully_shard_multi_card_with_empty_init(ckpt_path, mesh)
    else:
        losses, _ = run_fully_shard_multi_card(
            ckpt_path, mesh, enable_prefetch=enable_prefetch, enable_recompute=enable_recompute
        )

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
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}, please generate baseline artifacts first"

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


def run_fully_shard_with_grad_accum(mesh, enable_prefetch=False, enable_recompute=False):
    """Run fully_shard gradient accumulation and compare with large-batch baseline."""
    init()

    rank_id = get_rank()

    ckpt_path = os.path.join(TEMP_DIR, "init_baseline.ckpt")
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}, please generate baseline artifacts first"

    losses, _ = run_fully_shard_multi_card(
        ckpt_path,
        mesh,
        accumulate_grad=True,
        enable_prefetch=enable_prefetch,
        enable_recompute=enable_recompute,
    )

    if rank_id == 0:
        baseline_losses_file = os.path.join(TEMP_DIR, "baseline_losses.npy")
        assert os.path.exists(baseline_losses_file), f"Baseline losses not found: {baseline_losses_file}"

        baseline_losses = list(np.load(baseline_losses_file))
        compare_losses(baseline_losses, losses, rtol=1e-5, atol=1e-5)
        print("Precision comparison with gradient accumulation passed!")


def run_fully_shard_zero1_with_manual_grad_sync(mesh):
    """Run zero1-style gradient accumulation with manual final-step sync."""
    init()

    rank_id = get_rank()

    ckpt_path = os.path.join(TEMP_DIR, "init_baseline.ckpt")
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}, please generate baseline artifacts first"

    losses, _ = run_fully_shard_multi_card(
        ckpt_path,
        mesh,
        accumulate_grad=True,
        sync_grad_on_last_micro_step=True,
        replicate_all_params=True,
    )

    if rank_id == 0:
        baseline_losses_file = os.path.join(TEMP_DIR, "baseline_losses.npy")
        assert os.path.exists(baseline_losses_file), f"Baseline losses not found: {baseline_losses_file}"

        baseline_losses = list(np.load(baseline_losses_file))
        compare_losses(baseline_losses, losses, rtol=1e-5, atol=1e-5)
        print("Precision comparison with zero1 manual grad sync passed!")


def test_ms_zero3_fully_shard():
    """
    Feature: Compare fully_shard precision with standalone baseline
    Description: Run standalone baseline and fully_shard multi-card training,
                 then compare reduced losses on rank 0 only
    Expectation: Losses should match within tolerance
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    run_fully_shard(mesh)


def test_ms_zero3_fully_shard_prefetch():
    """
    Feature: Compare fully_shard prefetch precision with standalone baseline
    Description: Run standalone baseline and fully_shard multi-card training with child-module prefetch enabled,
                 then compare reduced losses on rank 0 only
    Expectation: Losses should match within tolerance
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    run_fully_shard(mesh, enable_prefetch=True)


def test_ms_zero3_fully_shard_prefetch_recompute():
    """
    Feature: Compare fully_shard prefetch + recompute precision with standalone baseline
    Description: Run standalone baseline and fully_shard multi-card training with child-module prefetch
                 and activation recompute enabled, then compare reduced losses on rank 0 only
    Expectation: Losses should match within tolerance
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    run_fully_shard(mesh, enable_prefetch=True, enable_recompute=True)


def test_ms_zero3_fully_shard_empty_weight():
    """
    Feature: Compare empty init fully_shard precision with standalone baseline
    Description: Run standalone baseline and fully_shard multi-card training with init_empty_weights True,
                 then compare reduced losses on rank 0 only
    Expectation: Losses should match within tolerance
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    run_fully_shard(mesh, use_empty_weight=True)


def test_ms_zero3_partial_shard():
    """
    Feature: Compare partial_shard precision with standalone baseline
    Description: Run standalone baseline and partial_shard multi-card training,
                 then compare reduced losses on rank 0 only
    Expectation: Losses should match within tolerance
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "op"))
    run_fully_shard(mesh)


def test_ms_zero3_partial_shard_prefetch_recompute():
    """
    Feature: Compare partial_shard prefetch + recompute precision with standalone baseline
    Description: Run standalone baseline and partial_shard multi-card training with child-module
                 prefetch and activation recompute enabled, then compare reduced losses on rank 0 only
    Expectation: Losses should match within tolerance
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "op"))
    run_fully_shard(mesh, enable_prefetch=True, enable_recompute=True)


def test_ms_zero3_fully_shard_replicate_params():
    """
    Feature: Compare fully_shard(replicate_params) precision with standalone baseline
    Description: Run standalone baseline and fully_shard multi-card training with replicate_params,
                 then compare reduced losses on rank 0 only
    Expectation: Losses should match within tolerance
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    run_fully_shard_ignored(mesh)


def test_ms_zero3_fully_shard_grad_accum():
    """
    Feature: Compare fully_shard gradient accumulation precision with large-batch standalone baseline
    Description: Run fully_shard multi-card training with per-micro-step gradient accumulation,
                 then compare reduced losses against the large-batch baseline on rank 0 only
    Expectation: Losses should match the large-batch baseline within tolerance
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    run_fully_shard_with_grad_accum(mesh)


def test_ms_zero3_fully_shard_prefetch_recompute_grad_accum():
    """
    Feature: Compare fully_shard prefetch + recompute + gradient accumulation precision
             with large-batch standalone baseline
    Description: Run fully_shard multi-card training with child-module prefetch, activation
                 recompute, and per-micro-step gradient accumulation, then compare reduced
                 losses against the large-batch baseline on rank 0 only
    Expectation: Losses should match the large-batch baseline within tolerance
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    run_fully_shard_with_grad_accum(mesh, enable_prefetch=True, enable_recompute=True)


def test_ms_zero1_fully_shard_grad_accum():
    """
    Feature: Compare zero1-style fully_shard gradient accumulation precision with large-batch standalone baseline
    Description: Run fully_shard on replicated parameters (zero1-style) and disable gradient synchronization for
                 early micro steps via set_requires_gradient_sync(False), then synchronize on the final micro step
    Expectation: Losses should match the large-batch baseline within tolerance
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    run_fully_shard_zero1_with_manual_grad_sync(mesh)


def _parse_args():
    parser = argparse.ArgumentParser(description="Precision baseline and fully_shard test helper.")
    parser.add_argument(
        "--generate-baseline",
        action="store_true",
        help="Generate standalone checkpoint and baseline loss artifacts.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.generate_baseline:
        generate_baseline_artifacts()
