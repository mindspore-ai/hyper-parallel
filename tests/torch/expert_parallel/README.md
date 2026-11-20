# Expert-parallel system tests

This directory exercises model recipe execution, routed expert computation, grouped matmul, and router loss on Ascend. Individual launchers state their card requirements.

For declarative placement and the local compute boundary, see the [EP guide](../../../docs/guide/expert_parallel.md). CPU coverage for the recipe entry lives in [test_ep_moe_local_region.py](../../ut/dual_mode_dtensor/test_ep_moe_local_region.py).
