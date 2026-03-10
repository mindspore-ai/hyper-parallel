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
                slice_shape[i] = slice_shape[i] // layout.mesh.get_device_num_along_axis(sub_axis_name)
    return slice_shape
