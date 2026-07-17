# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
import logging

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.redistribute_infer import RedistributionOperatorInfer
from hyper_parallel.platform import get_platform
platform = get_platform()

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
        group = platform.create_group(rank_list)
        concat_size = len(rank_list)
        logger.debug(
            "differentiable_all_gather_concat: input_shape=%s, concat_dim=%d, "
            "concat_size=%d, rank_list=%s",
            tuple(x.shape), concat_dim, concat_size, rank_list,
        )
        return platform.differentiable_all_gather_concat(x, group, concat_size, concat_dim, rank_list)


    @staticmethod
    def _construct_strided_slice(x, *args):
        """args: (begin, end, strides)"""
        dims = len(args) // 3
        return platform.construct_strided_slice(x, args[0: dims], args[dims: 2 * dims], args[2 * dims:])

    @staticmethod
    def _construct_all_concat_new(x, *args):
        """args: (concat_dim, concat_size, group)"""
        rank_list = args[2]
        concat_dim = args[0]
        concat_size = args[1]
        group = platform.create_group(rank_list)
        logger.debug(
            "differentiable_all_gather_concat: input_shape=%s, concat_dim=%d, "
            "concat_size=%d, rank_list=%s",
            tuple(x.shape), concat_dim, concat_size, rank_list,
        )
        return platform.differentiable_all_gather_concat(x, group, concat_size, concat_dim, rank_list)

    def _construct_all_split(self, x, *args):
        """args: (split_dim, split_size, group)"""
        rank_list = list(args[2])
        split_dim = args[0]
        split_size = args[1]
        idx = rank_list.index(self.rank_id)
        return platform.chunk(x, split_dim, split_size, idx)

    @staticmethod
    def _construct_all_to_all(x, *args):
        """args: (split_dim, concat_dim, permute_size, group)"""
        split_dim, concat_dim, split_count, rank_list = args
        group = platform.create_group(rank_list)
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
        output_tensor = platform.differentiable_all_to_all(
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

    def redistribution(self, input_x, to_layout):
        """tensor redistribution"""
        x_layout = input_x.layout
        x = input_x
        if input_x.layout.is_partial():
            # Solve partial status first
            if input_x.layout.mesh_shape == to_layout.mesh_shape:
                x = self.reduce_partial(input_x, to_layout)
            else:
                x = self.reduce_partial(input_x, x_layout)

        from_layout = x.layout
        if not self.is_init:
            self.rank_id = platform.get_rank()
            self.is_init = True
        if from_layout.rank_list != to_layout.rank_list:
            raise ValueError(f"The from_layout rank list: {from_layout.rank_list} is not equal to "
                             f"to_layout rank list: {to_layout.rank_list}")
        key = from_layout.compact_str + to_layout.compact_str +  str(self.rank_id)
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

    def _infer_transform_operator_list(self, from_layout, to_layout, from_full_shape, key, rank_list):
        """infer transform operator list"""
        from_layout_tuple, to_layout_tuple = \
            _construct_layout_tuple_for_transform_operator_list(from_layout, to_layout, from_full_shape)
        self._transform_cache[key] = \
            platform.get_tensor_transform().transform_tensor_sharding(from_layout_tuple, to_layout_tuple,
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
            x = platform.differentiable_all_reduce(x, 'sum', group)
            x = x / dev_num
        elif op == 'all':
            x_int32 = platform.tensor_type_cast(x.bool(), 'int32')  # True→1, False→0
            x = platform.differentiable_all_reduce(x_int32, 'all', group)
            x = x.bool()
        else:
            x = platform.differentiable_all_reduce(x, op, group)
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
        output_tensor = platform.differentiable_reduce_scatter(x, dev_num, axis, op, group)
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
