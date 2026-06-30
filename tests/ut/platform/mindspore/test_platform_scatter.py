# Copyright 2026 Huawei Technologies Co., Ltd
"""Unit tests for MindSporePlatform scatter adapter."""
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("mindspore")

from hyper_parallel.platform.mindspore.platform import MindSporePlatform


def test_scatter_none_list_uses_output_placeholders():
    """Non-source ranks pass scatter_list=None (PyTorch parity); MS needs a sized list."""
    output = MagicMock(name="output")
    group = "test_group"

    with patch(
        "hyper_parallel.platform.mindspore.platform.get_group_size",
        return_value=2,
    ) as mock_group_size, patch(
        "hyper_parallel.platform.mindspore.platform.dist.scatter",
        return_value=None,
    ) as mock_scatter:
        result = MindSporePlatform.scatter(output, None, src=1, group=group)

    mock_group_size.assert_called_once_with(group)
    mock_scatter.assert_called_once_with(output, [output, output], 1, group, async_op=False)
    assert result is output


def test_scatter_forwards_source_chunk_list():
    """Source rank still forwards the real scatter list."""
    output = MagicMock(name="output")
    chunk0 = MagicMock(name="chunk0")
    chunk1 = MagicMock(name="chunk1")
    chunk0.is_contiguous.return_value = True
    chunk1.is_contiguous.return_value = True
    scatter_list = [chunk0, chunk1]

    with patch(
        "hyper_parallel.platform.mindspore.platform.MindSporePlatform.get_process_group_ranks",
        return_value=[10, 11],
    ), patch(
        "hyper_parallel.platform.mindspore.platform.dist.scatter",
        return_value=None,
    ) as mock_scatter:
        result = MindSporePlatform.scatter(
            output,
            scatter_list,
            group_src=0,
            group="mesh_group",
        )

    mock_scatter.assert_called_once_with(output, scatter_list, 10, "mesh_group", async_op=False)
    assert result is output
