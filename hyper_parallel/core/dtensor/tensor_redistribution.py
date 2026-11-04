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
"""tensor_redistribution"""
# pylint: disable=C9006,C9007
import logging

import torch
import torch.distributed as dist

from hyper_parallel.core.dtensor import _utils
from hyper_parallel.core.dtensor._ragged_utils import (
    _compute_ragged_all_to_all_splits,
    _compute_ragged_slice,
    _compute_ragged_splits,
)
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.layout import Layout, RaggedShardInfo, infer_slice_area_by_rank
from hyper_parallel.core.dtensor.placement_types import Replicate
from hyper_parallel.core.dtensor.redistribute_infer import RedistributionOperatorInfer

logger = logging.getLogger(__name__)


def _construct_layout_tuple_for_transform_operator_list(from_layout, to_layout, from_full_shape):
    """_construct_layout_tuple_for_transform_operator_list"""
    from_layout_dict = from_layout.to_dict()
    to_layout_dict = to_layout.to_dict()
    from_layout_tuple = (
        from_layout_dict["mesh_shape"], from_layout_dict["tensor_map"], list(from_full_shape)
    )
    # NOTE: consider reshape scenario when to_full_shape differs from from_full_shape
    to_layout_tuple = (
        to_layout_dict["mesh_shape"], to_layout_dict["tensor_map"], list(from_full_shape)
    )
    return from_layout_tuple, to_layout_tuple


class TensorRedistribution:
    """
    TensorRedistribution.
    """
    def __init__(self):
        self.is_init = False
        self.rank_id = None # current rank_id (global)
        self._transform_cache = {}
        self._construct_op_operator = {
            "Reshape": self._construct_reshape,
            "AllConcat": self._construct_all_concat,
            "StridedSlice": self._construct_strided_slice,
            "all_concat": TensorRedistribution._construct_all_concat_new,
            "all_split": self._construct_all_split,
            "all_to_all": self._construct_all_to_all
        }

    @staticmethod
    def _construct_reshape(x, *args):
        """args: (*shape)"""
        return x.view(args)

    @staticmethod
    def _construct_all_concat(x, *args):
        """args: (*rank_list, concat_dim)"""
        rank_list = args[0:-1]
        concat_dim = args[-1]
        group = _utils.create_group(rank_list)
        concat_size = len(rank_list)
        logger.debug(
            "differentiable_all_gather_concat: input_shape=%s, concat_dim=%d, "
            "concat_size=%d, rank_list=%s",
            tuple(x.shape), concat_dim, concat_size, rank_list,
        )
        return _utils.differentiable_all_gather_concat(x, group, concat_size, concat_dim, rank_list)


    @staticmethod
    def _construct_strided_slice(x, *args):
        """args: (begin, end, strides)"""
        dims = len(args) // 3
        return _utils.construct_strided_slice(x, args[0: dims], args[dims: 2 * dims], args[2 * dims:])

    @staticmethod
    def _construct_all_concat_new(x, *args):
        """args: (concat_dim, concat_size, group)"""
        rank_list = args[2]
        concat_dim = args[0]
        concat_size = args[1]
        group = _utils.create_group(rank_list)
        logger.debug(
            "differentiable_all_gather_concat: input_shape=%s, concat_dim=%d, "
            "concat_size=%d, rank_list=%s",
            tuple(x.shape), concat_dim, concat_size, rank_list,
        )
        return _utils.differentiable_all_gather_concat(x, group, concat_size, concat_dim, rank_list)

    def _construct_all_split(self, x, *args):
        """args: (split_dim, split_size, group)"""
        rank_list = list(args[2])
        split_dim = args[0]
        split_size = args[1]
        idx = rank_list.index(self.rank_id)
        return torch.chunk(x, split_size, dim=split_dim)[idx]

    @staticmethod
    def _construct_all_to_all(x, *args):
        """args: (split_dim, concat_dim, permute_size, group)"""
        split_dim, concat_dim, split_count, rank_list = args
        group = _utils.create_group(rank_list)
        logger.debug(
            "differentiable_all_to_all: input_shape=%s, split_dim=%d, "
            "concat_dim=%d, split_count=%d, rank_list=%s",
            tuple(x.shape), split_dim, concat_dim, split_count, rank_list,
        )
        original_shape = x.shape

        dim_size = original_shape[split_dim]
        if dim_size % split_count != 0:
            raise ValueError(f"Dimension {split_dim} with size {dim_size} "
                             f"cannot be evenly split into {split_count} parts")

        split_size = dim_size // split_count
        final_shape = list(original_shape)
        if split_dim != concat_dim:
            final_shape[split_dim] = split_size
            final_shape[concat_dim] = final_shape[concat_dim] * split_count
        final_shape = tuple(final_shape)

        pre_special_handle = all(original_shape[i] == 1 for i in range(split_dim))
        if pre_special_handle:
            reshape_shape = (split_count * split_size,) + original_shape[split_dim + 1:]
            x_reshaped = x.view(reshape_shape)
        else:
            reshape_dims = list(original_shape)
            reshape_dims[split_dim] = split_count
            reshape_dims.insert(split_dim + 1, split_size)

            trans_dims = list(range(len(reshape_dims)))
            trans_dims.remove(split_dim)
            trans_dims.insert(0, split_dim)

            x_reshaped = x.reshape(reshape_dims).permute(trans_dims).contiguous()

            reshape_shape = list(x_reshaped.shape)
            reshape_shape[0] = reshape_shape[0] * reshape_shape[1]
            reshape_shape.pop(1)
            reshape_shape = tuple(reshape_shape)
            x_reshaped = x_reshaped.reshape(reshape_shape)
        x_reshaped = x_reshaped.contiguous()
        output_tensor = _utils.differentiable_all_to_all(
            input_data=x_reshaped,
            output_shape=reshape_shape,
            group=group
        )

        post_special_handle = all(final_shape[i] == 1 for i in range(concat_dim))
        if post_special_handle:
            return output_tensor.view(final_shape)

        # When pre_special_handle collapsed leading size-1 dims, the A2A was executed
        # in a reduced-rank space where the effective concat axis is shifted left by
        # split_dim positions.  Use recon_concat_dim for all post-A2A reshaping so
        # that split_count is merged into the correct dimension.
        recon_concat_dim = (concat_dim - split_dim) if pre_special_handle else concat_dim

        output_reshape = list(output_tensor.shape)
        output_reshape[0] = split_count
        output_reshape.insert(1, output_tensor.shape[0] // split_count)

        out_trans_dims = list(range(len(output_reshape)))
        first_dim = out_trans_dims.pop(0)
        if recon_concat_dim >= len(out_trans_dims):
            out_trans_dims.append(first_dim)
        else:
            out_trans_dims.insert(recon_concat_dim, first_dim)

        final_output = output_tensor.reshape(output_reshape).permute(out_trans_dims).contiguous()

        final_reshape = list(final_output.shape)
        if recon_concat_dim < len(final_reshape) - 1:
            final_reshape[recon_concat_dim] = (
                final_reshape[recon_concat_dim] * final_reshape[recon_concat_dim + 1]
            )
            final_reshape.pop(recon_concat_dim + 1)

        result = final_output.reshape(final_reshape)
        if pre_special_handle:
            result = result.view(final_shape)
        return result

    @staticmethod
    def _apply_eazy_redistribute(src_layout, dst_layout):
        """_apply_eazy_redistribute"""
        if (src_layout.mesh_shape != dst_layout.mesh_shape or
                src_layout.rank_list != dst_layout.rank_list):
            return False

        tensor_map_size = len(src_layout.tensor_map)
        if len(dst_layout.tensor_map) != tensor_map_size:
            return False
        return True

    def _redistribution_without_shape(self, local_x, src_layout, dst_layout, key, rank_list):
        """_redistribution_without_shape"""
        inferrer = RedistributionOperatorInfer(
            dev_mat=src_layout.mesh_shape,
            in_tensor_map=list(src_layout.tensor_map),
            out_tensor_map=list(dst_layout.tensor_map)
        )
        op_list = inferrer.infer_ops_list(self.rank_id, rank_list)
        self._transform_cache[key] = op_list
        for op in op_list:
            local_x = self._construct_op_operator[op[0]](local_x, *op[1])
        return local_x

    @staticmethod
    def _to_normal_layout(layout: Layout, tensor_dim: int) -> Layout:
        """Build the normal view consumed by the legacy redistribution path."""
        normal_layout = Layout.from_device_mesh(layout.mesh)
        normal_layout.set_placements(layout.normal_placements)
        normal_layout.placement_to_tensor_map(tensor_dim)
        return normal_layout

    def _redistribute_ragged(self, input_x: DTensor, to_layout: Layout) -> DTensor:
        """Adapt RaggedShard layouts to the supported redistribution primitives."""
        from_layout = input_x.layout
        source_info = from_layout.ragged_shard
        target_info = to_layout.ragged_shard
        source_is_ragged = isinstance(source_info, RaggedShardInfo)
        target_is_ragged = isinstance(target_info, RaggedShardInfo)
        tensor_dim = len(input_x.shape)

        if source_is_ragged and target_is_ragged:
            source_normal_layout = self._to_normal_layout(from_layout, tensor_dim)
            target_normal_layout = self._to_normal_layout(to_layout, tensor_dim)
            if (
                source_info.mesh_dim == target_info.mesh_dim
                and source_info.placement.dims == target_info.placement.dims
                and source_normal_layout == target_normal_layout
            ):
                return self.ragged_to_ragged(input_x, to_layout)
            return self.ragged_to_ragged_via_replicate(
                input_x,
                source_normal_layout,
                to_layout,
            )

        if source_is_ragged:
            source_normal_layout = self._to_normal_layout(from_layout, tensor_dim)
            normal = self.ragged_to_normal(input_x, source_normal_layout)
            if normal.layout == to_layout:
                return normal
            return self._redistribution_normal(normal, to_layout)

        target_normal_layout = self._to_normal_layout(to_layout, tensor_dim)
        normal = input_x
        if from_layout != target_normal_layout:
            normal = self._redistribution_normal(normal, target_normal_layout)
        return self.normal_to_ragged(normal, to_layout)

    def redistribution(self, input_x: DTensor, to_layout: Layout) -> DTensor:
        """Redistribute a tensor to the requested layout.

        Args:
            input_x: Distributed input tensor.
            to_layout: Requested output layout on the same rank list.
        """
        x_layout = input_x.layout
        if (x_layout.has_uneven_shard
                and not isinstance(x_layout.ragged_shard, RaggedShardInfo)
                and not isinstance(to_layout.ragged_shard, RaggedShardInfo)):
            # Validate the export-only path before any collective or layout rewrite.
            if x_layout.rank_list != to_layout.rank_list:
                raise ValueError("Uneven redistribution requires the same rank list")
            return self._gather_uneven_to_replicate(input_x, to_layout)
        x = input_x
        if input_x.layout.is_partial():
            # Solve partial status first
            if input_x.layout.mesh_shape == to_layout.mesh_shape:
                x = self.reduce_partial(input_x, to_layout)
            else:
                x = self.reduce_partial(input_x, x_layout)

        from_layout = x.layout
        if from_layout.rank_list != to_layout.rank_list:
            raise ValueError(f"The from_layout rank list: {from_layout.rank_list} is not equal to "
                             f"to_layout rank list: {to_layout.rank_list}")
        if isinstance(from_layout.ragged_shard, RaggedShardInfo) or isinstance(
            to_layout.ragged_shard, RaggedShardInfo
        ):
            return self._redistribute_ragged(x, to_layout)
        if from_layout.has_uneven_shard or to_layout.has_uneven_shard:
            return self._gather_uneven_to_replicate(x, to_layout)
        return self._redistribution_normal(x, to_layout)

    @staticmethod
    @torch.no_grad()
    def _padded_all_gather(local_tensor, sizes, tensor_dim, rank_list):
        """Gather export buffers; return independent storage without padding."""
        if len(rank_list) == 1:
            return local_tensor.detach().clone()
        max_size = max(sizes)
        if tensor_dim != 0 and any(size != max_size for size in sizes):
            raise NotImplementedError("Uneven export communication only supports dim=0")
        padded = local_tensor.detach().contiguous()
        if tensor_dim == 0 and local_tensor.shape[0] != max_size:
            padded_shape = (max_size, *local_tensor.shape[1:])
            padded = local_tensor.new_zeros(padded_shape)
            padded[:local_tensor.shape[0]].copy_(local_tensor)
        if padded.numel() == 0:
            output_shape = list(padded.shape)
            output_shape[tensor_dim] = sum(sizes)
            return padded.new_empty(output_shape)
        group = _utils.create_group(rank_list)
        group_ranks = dist.get_process_group_ranks(group)
        gathered = [torch.empty_like(padded) for _ in group_ranks]
        dist.all_gather(gathered, padded, group=group)
        group_indices = {rank: index for index, rank in enumerate(group_ranks)}
        # cat copies every trimmed shard into owned storage. Returning a view
        # (or contiguous(), which may be a no-op) would retain padded buffers.
        return torch.cat([
            gathered[group_indices[rank]].narrow(tensor_dim, 0, size)
            for rank, size in zip(rank_list, sizes)
        ], dim=tensor_dim)

    @staticmethod
    def _validate_uneven_local_shape(input_x: DTensor) -> None:
        """Reject padded storage masquerading as a true local shard."""
        source = input_x.layout
        area = infer_slice_area_by_rank(
            source.mesh_shape, source.tensor_map, source.rank_list.index(source.mesh.rank),
            tuple(input_x.shape), source.uneven_shard_mesh_dims,
        )
        expected_shape = tuple(end - start for start, end in area)
        local_shape = tuple(input_x.to_local().shape)
        if local_shape != expected_shape:
            raise ValueError(f"Uneven shard local shape {local_shape} does not match logical slice {expected_shape}")

    @staticmethod
    def _gather_uneven_to_replicate(input_x: DTensor, to_layout: Layout) -> DTensor:
        """Reconstruct detached FSDP weights, including empty dim-zero shards.

        Undo each tensor dimension's sharding from innermost to outermost.
        This respects both ceil-chunk FSDP shards and balanced TP shards,
        including the effective ordering encoded for StridedShard layouts.
        This is an export-only path, not a differentiable redistribution.
        Uneven nonzero axes and Partial placements remain unsupported.
        """
        source = input_x.layout
        if (source.mesh_shape != to_layout.mesh_shape
                or not all(isinstance(placement, Replicate) for placement in to_layout.placements)):
            raise NotImplementedError(
                "Uneven chunk-sharded redistribution only supports a fully replicated target on the same mesh."
            )
        full_shape = tuple(input_x.shape)
        rank = source.mesh.rank
        tensor_map = list(source.tensor_map)
        uneven_dims = source.uneven_shard_mesh_dims
        local_tensor = input_x.to_local()
        if source.is_partial():
            raise NotImplementedError("Uneven FSDP weight export does not support Partial placements")
        if any(source.placements[axis].dim != 0 for axis in uneven_dims):
            raise NotImplementedError("Uneven FSDP weight export only supports dim=0")
        if torch.is_grad_enabled() and local_tensor.requires_grad:
            raise RuntimeError("Uneven FSDP weight export is not differentiable; use torch.no_grad() or detach()")
        TensorRedistribution._validate_uneven_local_shape(input_x)
        for tensor_dim, mapping in enumerate(source.tensor_map):
            axes = list(mapping if isinstance(mapping, tuple) else (mapping,))
            axes = [axis for axis in axes if axis != -1]
            while axes:
                mesh_dim = len(source.mesh_shape) - 1 - axes[-1]
                ranks = source.mesh.get_devices_for_axis(mesh_dim, rank)
                areas = [
                    infer_slice_area_by_rank(
                        source.mesh_shape, tensor_map, source.rank_list.index(peer), full_shape, uneven_dims,
                    ) for peer in ranks
                ]
                sizes = [area[tensor_dim][1] - area[tensor_dim][0] for area in areas]
                local_tensor = TensorRedistribution._padded_all_gather(local_tensor, sizes, tensor_dim, ranks)
                axes.pop()
                tensor_map[tensor_dim] = tuple(axes) if axes else -1
        return DTensor.from_local_with_layout(local_tensor, to_layout, shape=full_shape)

    def _redistribution_normal(self, input_x: DTensor, to_layout: Layout) -> DTensor:
        """Run the existing redistribution path for normal layouts."""
        from_layout = input_x.layout
        x = input_x
        if not self.is_init:
            self.rank_id = dist.get_rank()
            self.is_init = True
        key = from_layout.compact_str + to_layout.compact_str + str(self.rank_id)
        if key in self._transform_cache:
            x = x.to_local()
            transform_operator_list = self._transform_cache[key]
            for transform_operator in transform_operator_list:
                x = self._construct_op_operator[transform_operator[0]](x, *transform_operator[1])
            return DTensor.from_local(x, to_layout.mesh, to_layout.alias_placements)

        full_shape = x.shape
        key_and_shape = key + str(full_shape)
        x = x.to_local()
        if key_and_shape in self._transform_cache:
            transform_operator_list = self._transform_cache[key_and_shape]
            for transform_operator in transform_operator_list:
                x = self._construct_op_operator[transform_operator[0]](x, *transform_operator[1])
            return DTensor.from_local(x, to_layout.mesh, to_layout.alias_placements)

        rank_list = from_layout.rank_list
        if self._apply_eazy_redistribute(from_layout, to_layout):
            if from_layout.is_partial():
                from_layout.reset_partial()
            x = self._redistribution_without_shape(x, from_layout, to_layout, key, rank_list)
        else:
            transform_operator_list = self._infer_transform_operator_list(from_layout, to_layout,
                                                                          full_shape, key_and_shape, rank_list)
            for transform_operator in transform_operator_list:
                x = self._construct_op_operator[transform_operator[0]](x, *transform_operator[1])
        return DTensor.from_local(x, to_layout.mesh, to_layout.alias_placements)

    @staticmethod
    def ragged_to_normal(input_x: DTensor, to_layout: Layout) -> DTensor:
        """Gather one flat RaggedShard into its Replicate normal view."""
        from_layout = input_x.layout
        info = from_layout.ragged_shard
        if not isinstance(info, RaggedShardInfo):
            raise ValueError("ragged_to_normal requires a RaggedShard source layout")
        global_shape = tuple(input_x.shape)
        output_splits = _compute_ragged_splits(global_shape, from_layout)
        local_tensor = input_x.to_local()
        if len(output_splits) == 1:
            gathered = local_tensor
        else:
            group = from_layout.mesh.get_group(info.mesh_dim)
            gathered = _utils.differentiable_variable_all_gather(
                local_tensor,
                output_splits,
                group,
            )
        return DTensor.from_local_with_layout(
            gathered.reshape(global_shape),
            to_layout,
            shape=global_shape,
        )

    @staticmethod
    def normal_to_ragged(input_x: DTensor, to_layout: Layout) -> DTensor:
        """Slice a normal-view tensor into the target flat RaggedShard."""
        global_shape = tuple(input_x.shape)
        local_slice = _compute_ragged_slice(global_shape, to_layout)
        flat_tensor = input_x.to_local().reshape((-1,))
        local_tensor = flat_tensor[
            local_slice.flat_start:local_slice.flat_end
        ].clone()
        return DTensor.from_local_with_layout(
            local_tensor,
            to_layout,
            shape=global_shape,
        )

    @staticmethod
    def ragged_to_ragged_via_replicate(
        input_x: DTensor,
        replicate_layout: Layout,
        to_layout: Layout,
    ) -> DTensor:
        """Redistribute different RaggedShard dims through a Replicate layout."""
        from_layout = input_x.layout
        if (
            from_layout.mesh_shape != to_layout.mesh_shape
            or from_layout.rank_list != to_layout.rank_list
        ):
            raise ValueError("ragged_to_ragged only supports changes on the same device mesh")
        replicated = TensorRedistribution.ragged_to_normal(
            input_x,
            replicate_layout,
        )
        return TensorRedistribution.normal_to_ragged(replicated, to_layout)

    @staticmethod
    def ragged_to_ragged(input_x: DTensor, to_layout: Layout) -> DTensor:
        """Redistribute a local-units-only RaggedShard change with variable all-to-all."""
        from_layout = input_x.layout
        source_info = from_layout.ragged_shard
        if (
            from_layout.mesh_shape != to_layout.mesh_shape
            or from_layout.rank_list != to_layout.rank_list
        ):
            raise ValueError("ragged_to_ragged only supports local_units changes on the same device mesh")

        global_shape = tuple(input_x.shape)
        input_splits, output_splits = _compute_ragged_all_to_all_splits(
            global_shape,
            from_layout,
            to_layout,
        )
        flat_input = input_x.to_local().reshape((-1,))

        if flat_input.shape[0] != sum(input_splits):
            raise ValueError(
                "RaggedShard source storage does not match all-to-all splits, "
                f"got local_numel={flat_input.shape[0]}, input_splits={input_splits!r}"
            )
        if len(input_splits) == 1:
            flat_output = flat_input
        else:
            group = from_layout.mesh.get_group(source_info.mesh_dim)
            flat_output = _utils.differentiable_all_to_all_single(
                flat_input,
                input_splits,
                output_splits,
                group,
            )
        if flat_output.shape[0] != sum(output_splits):
            raise ValueError(
                "RaggedShard target storage does not match all-to-all splits, "
                f"got local_numel={flat_output.shape[0]}, output_splits={output_splits!r}"
            )
        return DTensor.from_local_with_layout(
            flat_output,
            to_layout,
            shape=global_shape,
        )

    def _infer_transform_operator_list(self, from_layout, to_layout, from_full_shape, key, rank_list):
        """infer transform operator list"""
        from_layout_tuple, to_layout_tuple = \
            _construct_layout_tuple_for_transform_operator_list(from_layout, to_layout, from_full_shape)
        self._transform_cache[key] = \
            _utils.get_tensor_transform().transform_tensor_sharding(from_layout_tuple, to_layout_tuple,
                                                                      rank_list, False, self.rank_id)
        return self._transform_cache[key]

    @staticmethod
    def _allreduce_along_dev_dim(x, op, layout, dev_dim):
        """Do allreduce at specified axis along dev_dim."""
        logger.debug(
            "differentiable_all_reduce: input_shape=%s, op=%s, dev_dim=%s",
            tuple(x.shape), op, dev_dim,
        )
        group = layout.get_comm_group_by_axis(dev_dim)
        zero_dim = x.dim() == 0
        if zero_dim:
            x = x.unsqueeze(0)
        if op == 'avg':
            dev_num = layout.mesh_shape[layout.alias_name.index(dev_dim)]
            x = _utils.differentiable_all_reduce(x, 'sum', group)
            x = x / dev_num
        elif op == 'all':
            x_int32 = _utils.tensor_type_cast(x.bool(), 'int32')  # True→1, False→0
            x = _utils.differentiable_all_reduce(x_int32, 'all', group)
            x = x.bool()
        else:
            x = _utils.differentiable_all_reduce(x, op, group)
        if zero_dim:
            x = x.squeeze(0)
        return x

    @staticmethod
    def _reduce_scatter_along_dev_dim_with_axis(x, axis, op, layout, dev_dim):
        """Do reduce_scatter at specified axis along dev_dim."""
        dev_num = layout.mesh_shape[layout.alias_name.index(dev_dim)]
        logger.debug(
            "differentiable_reduce_scatter: input_shape=%s, axis=%d, "
            "op=%s, dev_dim=%s, dev_num=%d",
            tuple(x.shape), axis, op, dev_dim, dev_num,
        )
        group = layout.get_comm_group_by_axis(dev_dim)
        output_tensor = _utils.differentiable_reduce_scatter(x, dev_num, axis, op, group)
        return output_tensor

    def reduce_partial(self, input_x, to_layout):
        """Reduce partial status."""
        from_layout = input_x.layout
        x = input_x
        if from_layout is None or not from_layout.is_partial():
            return x

        x = x.to_local()
        if from_layout.mesh_shape != to_layout.mesh_shape:
            raise ValueError(f"For reduce partial, mesh_shape between from_layout and to_layout must be the same, "
                             f"but got {from_layout.mesh_shape} and {to_layout.mesh_shape}")
        if to_layout.is_partial():
            raise ValueError(f"For reduce partial, to_layout must be non-partial status, but got to_layout.partial: "
                             f"{to_layout.partial}")

        dev_map_order = {}
        for dev_axis in to_layout.alias_tensor_map:
            if isinstance(dev_axis, tuple):
                for i, sub_dev_axis in enumerate(dev_axis):
                    dev_map_order[sub_dev_axis] = i
            else:
                dev_map_order[dev_axis] = 0

        pending_reduce_op_list = []  # List[Tuple[comm_op, op, dev_dim, reduce_dim]]
        for dev_axis_index, op in enumerate(from_layout.partial):
            if op is None:
                continue
            dev_axis = from_layout.alias_name[dev_axis_index]
            apply_shard_dim = to_layout.get_dev_axis_apply_shard_axis(dev_axis)
            comm_op = "ReduceScatter" if apply_shard_dim is not None else "AllReduce"
            pending_reduce_op_list.append((comm_op, op, dev_axis, apply_shard_dim))

        # sort reduce op
        # 1. ReduceScatter is executed before AllReduce
        # 2. If multiple split, the dev axis split outer will be execute first.
        #    e.g. ("cp", "tp"), will execute reduce_scatter along "cp" before "tp"
        # 3. Lower dev_id execute before higher dev_id
        def _reduce_pair_sort_key(reduce_pair):
            return (reduce_pair[0] != "ReduceScatter",
                    dev_map_order.get(reduce_pair[2], 0),
                    to_layout.mesh.axis_id(reduce_pair[2]))

        sorted_pending_reduce_op_list = sorted(pending_reduce_op_list, key=_reduce_pair_sort_key)

        output_alias_tensor_map = list(from_layout.alias_tensor_map)
        for reduce_op_pair in sorted_pending_reduce_op_list:
            comm_op = reduce_op_pair[0]
            op = reduce_op_pair[1]
            dev_axis = reduce_op_pair[2]
            if comm_op == "AllReduce":
                x = TensorRedistribution._allreduce_along_dev_dim(x, op, from_layout, dev_axis)
            elif comm_op == "ReduceScatter":
                reduce_axis = reduce_op_pair[3]
                x = self._reduce_scatter_along_dev_dim_with_axis(x, reduce_axis, op, from_layout, dev_axis)
                if output_alias_tensor_map[reduce_axis] == "None":
                    output_alias_tensor_map[reduce_axis] = dev_axis
                elif isinstance(output_alias_tensor_map[reduce_axis], tuple):
                    output_alias_tensor_map[reduce_axis] += (dev_axis,)
                else:
                    output_alias_tensor_map[reduce_axis] = (output_alias_tensor_map[reduce_axis], dev_axis)

        output_layout = from_layout(*output_alias_tensor_map)
        output_layout.reset_partial()
        return DTensor.from_local(x, output_layout.mesh, output_layout.alias_placements)


_tensor_redistribution = TensorRedistribution()
