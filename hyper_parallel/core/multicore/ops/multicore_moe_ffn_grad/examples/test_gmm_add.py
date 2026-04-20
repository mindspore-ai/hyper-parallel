import torch
import torch_npu
import numpy as np

import mindspore

batch_size = 2048
hidden_size = 4096

# mat1 = torch.randn(3072, 128, dtype=torch.float16).npu()
# mat2 = torch.randn(128, 7168, dtype=torch.float16).npu()

# mat1 = torch.randn(1024, 2048, dtype=torch.float16).npu()
# mat2 = torch.randn(2048, 7168, dtype=torch.float16).npu()

# add1 = torch.randn(4096, 7168, dtype=torch.float16).npu()
# add2 = torch.randn(4096, 7168, dtype=torch.float16).npu()

# mattemp1 = torch.randn(8192, 16384, dtype=torch.float16).npu()
# mattemp2 = torch.randn(16384, 8192, dtype=torch.float16).npu()

# dtype = np.float16
# x1 = torch.randn(3072, 7168, device='npu', dtype=torch.float16)
# x = [x1]
# weight1 = torch.randn(2, 7168, 4096, device='npu', dtype=torch.float16)
# weight = [weight1]
# group_list = torch.tensor([1536, 3072], device='npu', dtype=torch.int64)
# split_item = 3
# group_type = 0

dtype = np.float16
x1 = torch.randn(32768, 7168, device='npu', dtype=torch.float16)
x = [x1]
weight1 = torch.randn(8, 7168, 4096, device='npu', dtype=torch.float16)
weight = [weight1]
group_list = torch.tensor([4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768], device='npu', dtype=torch.int64)
split_item = 3
group_type = 0

weight2 = torch.randn(8, 2048, 7168, device='npu', dtype=torch.float16)

swiglu_x = torch.randn(32768, 4096, device='npu', dtype=torch.float16)

stream2 = torch.npu.Stream() #0x678fa860
event1 = torch.npu.Event()

experimental_config = torch_npu.profiler._ExperimentalConfig(
    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    l2_cache=True)

with torch_npu.profiler.profile(
    activities=[
        torch_npu.profiler.ProfilerActivity.NPU,
        torch_npu.profiler.ProfilerActivity.CPU
    ],
    with_stack=False,
    record_shapes=False,
    profile_memory=False,
    experimental_config=experimental_config,
    schedule=torch_npu.profiler.schedule(
        wait=0, warmup=0, active=0, repeat=1, skip_first=0
    ),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./torch_gmm_add/")
) as prof:
    # mat_out = torch.matmul(mat1, mat2)
    gmm_out = torch_npu.npu_grouped_matmul(x, weight, group_list=group_list, split_item=split_item, group_type=group_type)
    swiglu_out = torch_npu.npu_swiglu(gmm_out[0], dim=-1)
    gmm_out_2 = torch_npu.npu_grouped_matmul([swiglu_out], [weight2], group_list=group_list, split_item=split_item, group_type=group_type)

    # add_out = torch.add(add1, add2)

    # mat_out.reshape((4, 28, 1024, 256)).transpose(1, 2).reshape(4096, 7168)

    # res = torch.matmul(mattemp1, mattemp2)
    # res = torch.matmul(mattemp1, mattemp2)
    # res = torch.matmul(mattemp1, mattemp2)
    # event1.record()
    # mat_out = torch.matmul(mat1, mat2)
    # with torch.npu.stream(stream2):
    #     event1.wait()
    #     add_out = torch.add(add1, add2)


