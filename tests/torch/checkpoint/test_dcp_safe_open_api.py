"""test DCP safe_open torch API"""
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

DCP_SAFE_OPEN_API = "tests/torch/checkpoint/dcp_safe_open_api.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="unessential")
def test_dcp_safe_open_api_group1():
    """
    Feature: parallel run case in DCP safe_open API.
    Description:
        1.test_dcp_safe_open_lazy_tensor_lookup
        2.test_dcp_safe_open_slice_lookup
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DCP_SAFE_OPEN_API, "test_dcp_safe_open_lazy_tensor_lookup", 12259, 1),
        TorchCase(DCP_SAFE_OPEN_API, "test_dcp_safe_open_slice_lookup", 12260, 1),
    ])
