# 内存快照内容与字段含义

基于公开的 NPU 内存快照接口（`torch_npu` memory snapshot，与
`torch.cuda.memory._snapshot()` 同形）。快照记录某时刻 / 某段时间内的显存
分配事件与块状态，供 `breakdown.py` 解析。字段随 torch/torch_npu 版本略有
差异，解析防御式处理，缺字段标 `[待解析]`。

## 快照形态

快照是一组**段（segment）**与**块（block）**，外加一串**分配事件**：

| 概念 | 含义 |
|---|---|
| segment | 一次向设备申请的大块保留内存（reserved）|
| block | segment 内的一段，`state` 为 active（已分配）/ inactive（空洞）|
| allocation event | alloc / free / segment_alloc / segment_free / oom 等事件流 |

## 关键字段

| 字段 | 含义 | 用于 |
|---|---|---|
| `address` / `size` | 块地址与大小 | 峰值构成、Tensor 粒度 |
| `state` | active / inactive | 碎片分析（inactive=空洞）|
| `frames` / 调用栈 | 分配点的 Python/算子调用栈 | 定位分配来源（阶段 / module）|
| `stream` | 所属 stream | 阶段归类辅助 |
| 事件 `action` | alloc/free/oom 等 | 泄露（跨 step 不 free）与峰值时刻 |

## 采集

- 默认采前 1-3 step；采集期开启 record（含调用栈），结束后 dump 出快照
  文件（pickle / json 视接口）。
- 调用栈采集有开销；只在分析期开，不留在生产训练里。
- OOM 现场可在 OOM 时自动 dump 快照（若接口支持），用于峰值构成复盘。

## 碎片 / 泄露 / 峰值 的信号

- **碎片**：reserved 远大于 active 峰值，且 inactive 块多而碎 → 碎片严重。
- **泄露**：跨 step 的 active 总量单调增长、free 不对称 → 疑似泄露，看增长
  块的调用栈。
- **峰值**：active 总量的最大时刻 → 记录该时刻常驻块集合（按 size 排序 +
  调用栈归因）。
