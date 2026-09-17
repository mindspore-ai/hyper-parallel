# Push / pull 整合计划与验收

基底：`origin/fix/megamoe-permute-grad-aclnn` (`947b4ed5`)。
整合分支：`feat/megamoe-push-pull`，独立 worktree。
原 push 分支 `98e3c833`，原 pull HEAD `706895ba` 及未提交显存改动已另存备份快照。

## 共同改动

- 保留基底的 metadata-only ACLNN permutation gradient、新 SHMEM API 和 profiler。
- 接收区与反向 dX 的安全复用、提前释放 scratch、2 MiB heap 取整。
- 混合精度优化器直接回写，保留 push 显存分支中的通用优化。
- 原缓存分配顺序实验没有进入产品，不随整合引入。

## 可切换的通信路径

`MegaMoeExperts(..., dispatch_mode="push" | "pull")`，默认 push 保持基底行为。
模式在构造时确定；资源共享 key 包含模式，禁止共享不同模式的同一 workspace。
两种模式的 combine 均为 PUT。

- push：SHMEM 接收区按最大接收容量配置，保留原 PUT dispatch。
- pull：SHMEM source 按本地 routed rows 配置；普通 HBM 按实际接收量配置。
  保留 GET 双缓冲、4 KiB DMA 粒度、完成握手、轮询间隔和负载分块选择。
- pull 的 unpermute-grad 直接输出 SHMEM source，保留独立 Router 梯度保存状态。
- 显式容量上限的协调溢出检查在两条路径均生效。

## 必须处理的接口交汇

- runtime header 同时容纳基底 profiler 和 pull completion/protocol 字段，保持 64 字节。
- 同时支持 PUT / GET 描述符，scheduler、native reader 和 profiler 使用同一布局。
- CANN 细节封装在新版 SHMEM 边界内，不能恢复已删除的旧 SHMEM Python 生命周期。
- 两条 autograd 路径共同使用 ACLNN permutation gradient，不能重新保存 hidden 原值。
- 保留基底 profiling ABI，并验证 instrumented GET 路径。

## 验收

1. CPU：路由元数据、heap、共享资源隔离、调度序列化、autograd 生命周期、native ABI。
2. 完整 Release 构建；新 worktree 专用 editable 环境，校验实际导入和 payload。
3. 设备 4–7：两种模式的输出、输入/Router/专家梯度，空专家与热点，延迟及重复 backward。
4. 同进程双模式资源共存、模式间切换使用，checkpoint 和 stream 生命周期。
5. profiler 开启/关闭的正确性与记录；小 heap 的 pull 热点验证。
6. 本地提交和基底 ancestry 核查；不发布远端。

本文件将在实现及验证后补充最终接口和结果。

## 改动归属

| 改动 | push | pull | 基底处理 |
| --- | --- | --- | --- |
| metadata-only ACLNN permute grad | 保留 | 保留 | 直接继承 `947b4ed5` |
| backward 接收区/dX 复用，事件依赖证明与回退 | 共享 | 共享 | 从 push 显存分支带入 |
| 提前释放 scratch、按实际接收量保存激活 | 共享 | 共享 | 统一 autograd 生命周期 |
| heap 按 2 MiB 取整、optimizer 直接回写 | 共享 | 共享 | 从 push 显存分支带入 |
| source 对称分配、GET 到普通 HBM | 不启用 | 保留 | 迁移至新 SHMEM API |
| 4 KiB ping-pong GET、快速轮询与自适应分块 | 不启用 | 保留 | 封装到新 SHMEM CANN 边界 |
| unpermute-grad 直接写 SHMEM source | 不启用 | 保留 | 独立 native schema，保留 Router 梯度 |
| profiler 与完成握手 | profiler | 两者并存 | 重排 header，保持 64 字节及 profiling ABI |

旧分支中的历史性能报告、Dirichlet 扫描脚本与分配顺序实验保留在原分支及备份快照中。
整合分支集中保留产品实现、可切换 Qwen 示例和回归测试；未把已有历史结果标成新基底实测。
Qwen 示例使用 `--dispatch-mode push|pull`；模式在模块构造时确定，不支持原地修改已初始化模块。

## 验证记录

证据目录：`build/validation/integration_20260917`（本地构建产物，不入 Git）。
专用 editable 环境指向当前整合 worktree，Release payload 同样在当前目录构建。
源分支和原 worktree 均保留；旧 pull 未提交实现快照为
`backup/megamoe-pull-integration-20260917` (`5e6b6e10`)。

- CPU：129 项测试、491 个子用例通过，含 C++ header round-trip、双模式非对称分块、
  路由偏移、模式资源隔离、heap、跨 rank 布局拒绝、优化器和 native 版本拒绝。
- 首轮三项设备回归通过：双模式共存、pull 小 heap 热点、push 显存复用。
  共存测试覆盖输出/dX/Router/W1/W2 梯度、两个未 backward 的 forward、重复 backward、
  non-reentrant checkpoint、交替 stream、热点与空 rank，以及 profiler 前反向记录。
- pull 小 heap：EP4、T4096/H1024/I128/E8/K2，34 MiB heap 包含32 MiB对称数据；
  push 同 shape 仅对称数据即80 MiB。热点在rank0/rank3切换，全部梯度与 common MoE 一致。
- 真实 shape：H5120/I1792/E48/K8/T4096，EP4 的 push/pull 在1×、1.25×、2.5×负载下，
  输出和所有梯度均与 common MoE 一致。覆盖 pull 的1024/1024、128/512、128/128调度。
- 基底的 ACLNN permutation-gradient ST 通过，覆盖 BF16/FP16/FP32、奇数尺寸、
  非连续梯度和输出存储独立性。
- pre-commit 的 pylint、lizard、Markdown 及 clang-format 检查通过。
  autogit 聚合检查的额外 cpplint 默认80列与本仓库120列规范不一致，
  codespell 将 CANN/cann 误报为 CAN/can；这些工具报告保存在证据目录，不改动厂商名称或基底风格。
- 附加的三层 shared-workspace 生命周期 ST 曾因 Jenkins 占用卡5而中止；审计停止本轮子进程，
  等待空闲后单独补测通过（`st_lifecycle.log`）。总计6项设备回归通过。
- 本轮未重跑 EP8、整网端到端性能或 HBM 峰值扫描；这里的真实 shape 结果是 EP4 正确性验收。
