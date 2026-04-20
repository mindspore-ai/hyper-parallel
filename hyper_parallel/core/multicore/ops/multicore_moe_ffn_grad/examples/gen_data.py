# #!/usr/bin/python3
# # -*- coding:utf-8 -*-
# # Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# # This file is a part of the CANN Open Software.
# # Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# # Please refer to the License for details. You may not use this file except in compliance with the License.
# # THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# # INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# # See LICENSE in the root of the software repository for the full text of the License.
# # ======================================================================================================================

# import os
# import torch
# import numpy as np

# def gen_golden_data_simple():
#     dtype = np.float16
#     self_shape = [32, 7168] # 4096
#     mat2_shape = [7168, 2048]
#     output_shape = [32, 2048]

#     # self = np.random.uniform(-1, 1, self_shape).astype(dtype)
#     # mat2 = np.random.uniform(-1, 1, mat2_shape).astype(dtype)

#     self = torch.randn(self_shape[0], self_shape[1], dtype=torch.float16).npu()
#     mat2 = torch.randn(mat2_shape[0], mat2_shape[1], dtype=torch.float16).npu()

#     # golden = self @ mat2
#     golden = torch.matmul(self, mat2)

#     os.system("mkdir -p input")
#     os.system("mkdir -p output")
#     self.cpu().numpy().astype(dtype).tofile("./input/input_self.bin")
#     mat2.cpu().numpy().astype(dtype).tofile("./input/input_mat2.bin")
#     golden.cpu().numpy().tofile("./output/golden.bin")

# if __name__ == "__main__":
#     gen_golden_data_simple()



#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import os
import torch
import torch_npu
import torch.nn.functional as F
import numpy as np

def SiLU(x):
    y = x*(1/(1+np.exp(-x)))
    return y

def gen_golden_data_simple():
    # self = torch.randn(2048, 4096).to(torch.float16)

    dim = -1

    # os.system("mkdir -p input")
    # os.system("mkdir -p output")

    # self.numpy().astype(np.float16).tofile("./input/input_x.bin")
    # x = torch.chunk(self, 2, dim=dim)
    # x0 = x[0].type(torch.float16)
    # x1 = x[1].type(torch.float16)
    # result = F.silu(x0) * x1

    # # result.numpy().tofile("./output/output_golden_out.bin")

    input_x_shape = [8192, 4096]
    input_shape = [8192, 2048]
    output_shape = [8192, 2048]

    dtype = np.float16
    x = np.random.uniform(-1, 1, input_x_shape).astype(dtype)
    y = np.random.uniform(-1, 1, input_shape).astype(dtype)

    result = SiLU(x[:, :2048]) * x[:, 2048:]

    golden = np.add(result, y)

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    x.astype(dtype).tofile("./input/input_x.bin")
    y.astype(dtype).tofile("./input/input_y.bin")
    # y.astype(dtype).tofile("./input/input_y.bin")
    golden.tofile("./output/output_swiglu_addcustom.bin")


    input_x_shape = [8192, 4096]
    input_shape = [8192, 2048]
    output_shape = [8192, 2048]

    dtype = np.float16
    x = np.random.uniform(-1, 1, input_x_shape).astype(dtype)
    y = np.random.uniform(-1, 1, input_shape).astype(dtype)

    result = SiLU(x[:, :2048]) * x[:, 2048:]

    golden = np.add(result, y)

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    x.astype(dtype).tofile("./input/input_x_1.bin")
    y.astype(dtype).tofile("./input/input_y_1.bin")
    # y.astype(dtype).tofile("./input/input_y.bin")
    golden.tofile("./output/output_swiglu_addcustom_1.bin")


def gen_golden_data_matmul_add():
    dtype = np.float16
    x1 = torch.randn(32768, 7168, device='npu', dtype=torch.float16)
    x = [x1]
    weight1 = torch.randn(8, 7168, 4096, device='npu', dtype=torch.float16)
    weight = [weight1]
    group_list = torch.tensor([4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768], device='npu', dtype=torch.int64)
    split_item = 3
    group_type = 0
    npu_out = torch_npu.npu_grouped_matmul(x, weight, group_list=group_list, split_item=split_item, group_type=group_type)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    x1.cpu().numpy().astype(dtype).tofile("./input/input_gmm_x_512.bin")
    weight1.cpu().numpy().astype(dtype).tofile("./input/input_gmm_weight_512.bin")
    npu_out[0].cpu().numpy().tofile("./output/gmm_golden.bin")
    print("npu_out.shape:", npu_out[0].shape)

    add_x_shape = [4096, 7168]
    add_y_shape = [4096, 7168]
    add_output_shape = [4096, 7168]
    dtype = np.float16
    x = np.random.uniform(-1, 1, add_x_shape).astype(dtype)
    y = np.random.uniform(-1, 1, add_y_shape).astype(dtype)
    golden = np.add(x, y)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    x.astype(dtype).tofile("./input/input_add_x.bin")
    y.astype(dtype).tofile("./input/input_add_y.bin")
    golden.tofile("./output/add_golden.bin")


def gen_golden_data_matmul_swiglu():
    dtype = np.float16
    x1 = torch.randn(32768, 7168, device='npu', dtype=torch.float16)
    x = [x1]
    weight1 = torch.randn(8, 7168, 4096, device='npu', dtype=torch.float16)
    weight = [weight1]
    group_list = torch.tensor([4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768], device='npu', dtype=torch.int64)
    split_item = 3
    group_type = 0

    weight2 = torch.randn(8, 2048, 7168, device='npu', dtype=torch.float16)

    swiglu_in = torch.randn(32768, 4096, device='npu', dtype=torch.float16)

    gmm_out = torch_npu.npu_grouped_matmul(x, weight, group_list=group_list, split_item=split_item, group_type=group_type)
    # in_data = gmm_out[0].cpu().numpy()
    # print("in_data.shape:", in_data.shape)
    # swiglu_out = SiLU(in_data[:, :2048]) * in_data[:, 2048:]
    # swiglu_out = torch.tensor(swiglu_out, device='npu', dtype=torch.float16)
    swiglu_out = torch_npu.npu_swiglu(swiglu_in, dim = -1)
    gmm_out_2 = torch_npu.npu_grouped_matmul([swiglu_out], [weight2], group_list=group_list, split_item=split_item, group_type=group_type)

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    x1.cpu().numpy().astype(dtype).tofile("./input/input_gmm_x_512.bin")
    weight1.cpu().numpy().astype(dtype).tofile("./input/input_gmm_weight_512.bin")
    weight2.cpu().numpy().astype(dtype).tofile("./input/input_gmm_weight2_512.bin")
    gmm_out[0].cpu().numpy().tofile("./output/gmm_golden.bin")
    swiglu_in.cpu().numpy().astype(dtype).tofile("./input/swiglu_in.bin")
    swiglu_out.cpu().numpy().astype(dtype).tofile("./output/swiglu_out.bin")
    gmm_out_2[0].cpu().numpy().tofile("./output/gmm2_golden.bin")

if __name__ == "__main__":
    gen_golden_data_matmul_swiglu()


