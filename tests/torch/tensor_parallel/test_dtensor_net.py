import os
import torch
import torch_npu  # 昇腾NPU核心适配
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from hyper_parallel import DTensor, Layout

# ===================== 1. 昇腾分布式初始化（仅保留核心） =====================
def init_dist():
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    torch.npu.set_device(rank)
    device_id = rank % 8  # 8卡对应device_id 0-7
    torch.npu.set_device(device_id)
    return rank, device_id

# ===================== 2. 定义极简模型（单Linear层） =====================
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(10, 2))  # 直接注册到model._parameters

    def forward(self, x):
        """用matmul实现线性变换，等价于原nn.Linear的forward"""
        return torch.matmul(x, self.weight)

def mse_loss_sum(y_pred, y_true):
    """
    小算子拼接实现MSE（求和模式）：(y_pred - y_true)² 的所有元素求和
    等价于 nn.MSELoss(reduction='sum')
    """
    error = torch.sub(y_pred, y_true)
    square_error = torch.square(error)
    mse = torch.sum(square_error)  # 基础算子：求和
    return mse

# ===================== 3. 核心训练逻辑（模型+AllReduce梯度同步） =====================
def main():
    rank, device_id = init_dist()
    is_master = (rank == 0)
    world_size = dist.get_world_size()
    # 1. 初始化模型+优化器（移到NPU）
    model = SimpleModel().npu()
    

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    #loss_fn = nn.MSELoss().npu()

    # 2. 构造输入/标签Tensor（直接造，无Dataset）
    # 每张卡构造不同的输入（rank区分，方便验证同步效果）
    layout = Layout((4, 2), ("dp", "tp"))
    x_layout = layout("dp", "None")
    y_layout = layout("dp", "tp")
    w_layout = layout("None", "tp")
    x = DTensor.from_local(torch.randn(32, 10).npu() * (rank + 1), x_layout)  # 输入：32个样本，10维特征
    y = DTensor.from_local(torch.randn(32, 2).npu(), y_layout)             # 标签：32个样本，1维输出
    for key, param in model._parameters.items():
        if param is not None and not isinstance(param, DTensor):
            model.register_parameter(
                key,
                nn.Parameter(DTensor.from_local(param.data, w_layout)),
            )
    # 3. 前向传播
    optimizer.zero_grad()  # 清空梯度
    y_pred = model(x)
    loss = mse_loss_sum(y_pred, y)
    loss.reduce_partial()
    # 4. 反向传播（计算本地梯度）
    loss.backward()
    # 5. AllReduce梯度同步（核心：8卡梯度求和后同步）
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= 4  # 梯度求和后取均值（分布式标准操作）

    # 6. 更新参数
    optimizer.step()

    # 7. 主卡打印结果（验证运行成功）
    if rank == 0:
        print(f"单卡损失 rank:{rank}: {loss.item():.4f}, loss.layout:{loss.layout}")
        w = model.weight
        w_data = w.data
        w_data_slice = w_data[:5]
        print(f"rank:{rank}, 模型Linear层权重（前5个）: {w_data_slice.cpu().numpy()}")


if __name__ == "__main__":
    main()
