"""
Utility functions for distributed tensor operations.

This module provides helper functions for computing local shapes, global offsets,
and other layout-related calculations in distributed settings.
"""
from hyper_parallel.core.layout import Layout

def compute_local_shape_and_global_offset(global_shape, device_mesh, placement):
    """
        Compute local shard shape and its global offset.

        Args:
            global_shape: Shape of the global tensor.
            mesh: Device mesh for distributed execution.
            placements: Sharding placements for each dimension.

        Returns:
            tuple: (local_shape, global_offset)
    """
    total_layout = Layout.from_device_mesh(device_mesh)
    layout = total_layout(placement)
    layout.placement_to_tensor_map(len(global_shape))
    slice_shape = list(global_shape)
    alias_tensor_map = layout.alias_tensor_map
    for i, axis_name in enumerate(alias_tensor_map):
        if isinstance(axis_name, str):
            axis_name = (axis_name,)
        for sub_axis_name in axis_name:
            if sub_axis_name != "None":
                num_devices = layout.mesh.get_device_num_along_axis(sub_axis_name)
                local_rank = layout.mesh.get_local_rank(sub_axis_name)
                global_size = slice_shape[i]
                remainder = global_size % num_devices
                # Consistent with torch.chunk: first `remainder` ranks get one extra element
                if remainder != 0 and local_rank < remainder:
                    slice_shape[i] = global_size // num_devices + 1
                else:
                    slice_shape[i] = global_size // num_devices
    return slice_shape
