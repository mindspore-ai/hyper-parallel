# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""FinetuneRecipe — LLM 微调 Recipe 主骨架。

Following design doc 03_training_loop.md §5（setup）、§6（训练主循环）、
§6.1（验证流程）、§7（单步优化器步进）、§8（前向+反向传播）。
"""

import logging
import time
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from hyper_models.components.distributed.cp_utils import shard_batch_for_cp
from hyper_models.components.distributed.infrastructure import (
    apply_cache_compatibility_patches,
    create_distributed_setup_from_config,
    destroy_process_group,
    initialize_distributed,
    setup_logging,
)
from hyper_models.components.loss import calculate_loss, calculate_mtp_loss
from hyper_models.components.models.common.model_utils import build_model
from hyper_models.components.training import (
    AutoMFU,
    StatefulRNG,
    StepState,
    _dp_cp_all_reduce_sum,
    build_callback_manager,
    calculate_mfu,
    filter_forward_kwargs,
    get_sync_ctx,
    prepare_after_first_microbatch,
    prepare_for_final_backward,
    prepare_for_grad_accumulation,
    scale_grads_and_clip_grad_norm,
    setup_magi,
)
from hyper_models.data import build_dataloader, build_validation_dataloader
from hyper_models.trainer.config import TrainerConfig
from hyper_models.recipes.base_recipe import BaseRecipe

# FSDP2（torch >= 2.4）
try:
    from torch.distributed.fsdp import FSDPModule
except ImportError:  # pragma: no cover
    FSDPModule = None  # type: ignore

logger = logging.getLogger(__name__)


class FinetuneRecipe(BaseRecipe):
    """LLM 微调 Recipe（03 §5-§8）。"""

    # ── §5.3 setup() ──

    def setup(self, cfg: TrainerConfig) -> None:
        """按依赖顺序构建训练组件。

        两类构建方式：
        - cfg.<typed>.build(**runtime_deps) → optimizer, lr_scheduler, step_scheduler, loss, checkpoint
        - 独立构建函数 或 from_pretrained → model, peft, dataloader, tokenizer

        03 步骤编号（①–⑲）→ §2 主时序树编号映射（③.x 与 01 §4.1 层级对齐）：
          ①=③.1（initialize_distributed）, ②=日志/补丁,
          ③=③.2（RNG）, ④=③.3（distributed_setup）,
          ⑤=MagiAttention, ⑥=③.3a（callback_manager）,
          ⑦=③.5（Loss）, ⑧=PP 配置, ⑨=③.6（PEFT）,
          ⑩=③.7（Checkpointer）, ⑪=③.8（Model）, ⑫=③.9（Optimizer）,
          ⑬=③.10（DataLoader）, ⑭=③.11（Val DataLoader）,
          ⑮=③.12（StepScheduler）, ⑯=③.13（LR Scheduler）,
          ⑰=注册追踪状态, ⑱=③.14（load_checkpoint）, ⑲=③.15（MFU）
        """
        self.cfg = cfg

        # ① 分布式初始化
        self.dist_env = initialize_distributed("nccl")
        # dist_env 为 torch.distributed 模块（infrastructure stub 返回）；
        # device / world_size 在此派生并缓存，供数据搬运与 MFU 计算使用。
        self._device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1

        # ② 日志 + 兼容性补丁
        setup_logging()
        apply_cache_compatibility_patches()

        # ③ RNG
        self.rng = StatefulRNG(seed=cfg.training.seed, ranked=True)

        # ④ 分布式策略
        self.distributed_setup = create_distributed_setup_from_config(cfg)
        self.mesh = self.distributed_setup.mesh_context
        # DP+CP 联合子 mesh：MeshContext 已根据 device_mesh 推导好，直接复用。
        self.dp_cp_mesh = self.mesh.dp_cp_mesh

        # ⑤ MagiAttention（可选）
        self.magi = setup_magi(cfg, self.mesh.device_mesh) if getattr(cfg, "magi", None) else None

        # ⑥ Callback 管理器 —— 注册所有内置 callback
        self.callback_manager = build_callback_manager(
            self, cfg,
            pbar_total=cfg.step_scheduler.max_steps if cfg.step_scheduler.max_steps > 0 else None,
        )

        # ⑦ Loss —— typed: .build()
        self.loss = self.cfg.loss.build()

        # ⑧ PP 配置
        self.pp_enabled = self.mesh.pp_size > 1
        self._configure_pp(cfg)

        # ⑨ PEFT —— 由 build_model 内部处理，不在 setup 中单独构建
        self.peft_config = cfg.peft  # 传入 build_model，用于判断 is_peft

        # ⑩ Checkpoint —— typed: .build(dp_rank=..., ...)
        self.checkpoint_config = self.cfg.checkpoint  # CheckpointingConfig 实例
        self.checkpointer = self.checkpoint_config.build(
            dp_rank=self._get_dp_rank(),
            tp_rank=self._get_tp_rank(),
            pp_rank=self._get_pp_rank(),
            # 06 D-10 口径：MeshContext 无 moe_mesh 字段，getattr 恒为 None
            moe_mesh=getattr(self.mesh, "moe_mesh", None),
        )

        # ⑪ Model —— build_model 内部走 HyperAutoModelForCausalLM.from_pretrained()
        self.model, self.optimizer_init = build_model(
            cfg.model, self.peft_config,
            distributed_setup=self.distributed_setup,
        )
        self.model_parts = self.model.parts if hasattr(self.model, "parts") else [self.model]

        # ⑫ Optimizer —— typed: .build(model, device_mesh=...)
        #     返回 list[Optimizer]（canonical）
        self.optimizer = self.cfg.optimizer.build(
            self.model, device_mesh=self.mesh.device_mesh,
            optimizer_init=self.optimizer_init,  # 传入 build_model 导出的 param 分组
            is_peft=self.peft_config is not None,
        )

        # ⑬ DataLoader —— 调用 02_data_pipeline.md::build_dataloader()
        #     global_batch_size 与 StepSchedulerConfig.global_batch_size 同源
        self.dataloader, self.tokenizer = build_dataloader(
            cfg.dataset, cfg.dataloader, cfg.model, cfg.packed_sequence,
            seed=cfg.training.seed,
            local_batch_size=cfg.step_scheduler.local_batch_size,
            global_batch_size=cfg.step_scheduler.global_batch_size,
            max_steps=cfg.step_scheduler.max_steps,
            val_check_interval=cfg.step_scheduler.val_every_steps,
            dp_rank=self._get_dp_rank(),
            dp_world_size=self._get_dp_group_size(),
            pp_enabled=self.pp_enabled,
            cp_size=self.mesh.cp_size,
            model=self.model_parts[0],
        )

        # ⑭ Validation DataLoader（02_data_pipeline.md §3.3；返回 dict[str, DataLoader]）
        self.val_dataloaders = build_validation_dataloader(
            cfg.dataset, cfg.dataloader, cfg.model, cfg.packed_sequence,
            cfg.training.seed,
            local_batch_size=cfg.step_scheduler.local_batch_size,
            global_batch_size=cfg.step_scheduler.global_batch_size,
            dp_rank=self._get_dp_rank(),
            dp_world_size=self._get_dp_group_size(),
            pp_enabled=self.pp_enabled,
            cp_size=self.mesh.cp_size,
            model=self.model_parts[0],
        )

        # ⑮ StepScheduler —— typed: .build(dataloader, dp_size, local_batch_size)
        self.step_scheduler = self.cfg.step_scheduler.build(
            self.dataloader,
            self._get_dp_group_size(),
            cfg.step_scheduler.local_batch_size,
        )

        # ⑯ LR Scheduler —— typed: .build(optimizer, step_scheduler)
        self.lr_scheduler = (
            self.cfg.lr_scheduler.build(self.optimizer, self.step_scheduler)
            if self.cfg.lr_scheduler is not None
            else None
        )

        # ⑰ 注册 checkpoint 追踪状态（kind 与 04 `_state_path` 对齐）
        self.register_state("model", "model")
        self.register_state("optimizer", "optimizer")
        self.register_state("lr_scheduler", "lr_scheduler")
        self.register_state("rng", "rng")
        self.register_state("dataloader", "dataloader")
        # step_scheduler 以 "train_state" kind 注册：save 侧写 extra_state.json，
        # load 侧读回恢复 self.epoch/self.step，确保 save/load 对称。
        self.register_state("step_scheduler", "train_state")

        # ⑱ 断点续训（04 §8 canonical load_checkpoint，1 参 restore_from）
        self.load_checkpoint(self.checkpoint_config.restore_from)

        # ⑲ MFU 计算器 + 模型信息打印
        self.mfu_calc = AutoMFU.from_config(self.model_parts[0])
        self._log_model_and_optimizer_details()

    def _configure_pp(self, cfg: TrainerConfig) -> None:
        """PP 配置（03 §5.3 ⑧）。

        Stub — PP runtime 未落地前仅记录开关；pp_enabled=True 时
        prepare_for_final_backward 的 PP 钩子会显式报错（grad_accum）。
        """
        if self.pp_enabled:
            logger.warning(
                "Pipeline parallelism (pp_size=%d) requested but PP runtime "
                "is not yet implemented; backward hooks will raise.",
                self.mesh.pp_size,
            )

    def _log_model_and_optimizer_details(self) -> None:
        """打印模型参数量与优化器信息（03 §5.3 ⑲）。"""
        num_params = sum(p.numel() for p in self.model_parts[0].parameters())
        num_trainable = sum(
            p.numel() for p in self.model_parts[0].parameters() if p.requires_grad
        )
        num_optimizers = (
            len(self.optimizer) if isinstance(self.optimizer, list) else 1
        )
        logger.info(
            "model params=%s trainable=%s, optimizers=%d, flops_per_token=%.3e",
            f"{num_params:,}", f"{num_trainable:,}", num_optimizers,
            self.mfu_calc.flops_per_token,
        )

    # ── §6 训练主循环 ──

    def run_train_validation_loop(self) -> None:
        """训练主循环 —— 核心显式 + 外围 Callback 混合方案。

        核心流程（forward/backward/optimizer step）显式在 Recipe 中编排；
        外围关注点（checkpoint、验证、日志、GC、WandB、tqdm）通过
        callback_manager.on_step_end(StepState) 驱动。
        """
        for mp in self.model_parts:
            mp.train()

        # ── Callback: 训练开始 ──
        self.callback_manager.on_train_begin()

        # 预绑 None：零迭代场景下 final save 需守卫
        train_metrics: Optional[dict] = None
        self._last_val_losses: Optional[dict] = None

        try:
            for epoch in self.step_scheduler.epochs:
                self.step_scheduler.set_epoch(epoch)

                for batches in self.step_scheduler:
                    # ── 核心训练：显式可见 ──
                    train_metrics = self._run_train_optim_step(
                        batches,
                        max_grad_norm=getattr(self.cfg.optimizer, "max_grad_norm", 1.0),
                    )

                    # sigterm_received 内部是 all_gather 集合通信，每步只查询
                    # 一次并复用结果（避免 StepState 构造期间多次 all_gather）
                    sigterm = self.step_scheduler.sigterm_received

                    # ── 外围关注点：Callback 统一驱动 ──
                    state = StepState(
                        step=self.step_scheduler.step,
                        epoch=epoch,
                        is_final_step=(
                            self.step_scheduler._max_steps_reached or sigterm
                        ),
                        is_ckpt_step=self.step_scheduler.is_ckpt_step,
                        is_val_step=self.step_scheduler.is_val_step,
                        is_log_step=self.step_scheduler.is_log_step,
                        is_gc_step=self.step_scheduler.is_gc_step,
                        sigterm_received=sigterm,
                        loss=train_metrics.get("loss", 0.0),
                        grad_norm=train_metrics.get("grad_norm"),
                        lr=train_metrics.get("lr", 0.0),
                        tps=train_metrics.get("tps", 0.0),
                        mfu=train_metrics.get("mfu", 0.0),
                        num_tokens=train_metrics.get("num_tokens", 0),
                    )
                    self.callback_manager.on_step_end(state)

            # ── 正常结束：最终 checkpoint ──
            # 顺序约束（与 04 对齐）：final save 必须先于 checkpointer.close()。
            # 最终步/SIGTERM 的保存统一由本处完成：CheckpointCallback 对
            # is_final_step 跳过（§4.2.4），避免同一步重复保存。
            self.save_checkpoint(
                self.cfg.checkpoint.checkpoint_dir,
                self.step_scheduler.epoch,
                self.step_scheduler.global_step,
                (train_metrics or {}).get("loss", 0.0),
                self._last_val_losses if (self.val_dataloaders and self._last_val_losses) else None,
                is_final_checkpoint=True,
            )
        finally:
            # ── Callback: 训练结束 + checkpointer 资源清理 ──
            self.callback_manager.on_train_end()
            self.checkpointer.close()

        destroy_process_group()

    # ── §6.1 验证流程 ──

    def _run_validation_epoch(self, val_dl) -> dict:
        """单次 validation epoch：torch.no_grad + DP all-reduce mean 聚合 val loss。

        返回 {"loss": float, "num_tokens": int}（num_tokens 用于加权聚合）。
        """
        # 切换 eval/validate 模式（关闭 dropout 等）
        for mp in self.model_parts:
            mp.eval()

        total_loss_sum = 0.0      # 跨 microbatch 累加 CE sum
        total_label_tokens = 0    # 本 rank 累计 label token 数

        try:
            with torch.no_grad():
                for batch in val_dl:
                    # 数据 → GPU
                    batch = {
                        k: v.to(self._device, non_blocking=True)
                        if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }

                    # CP batch 准备（与训练 §8 Step 2 一致）
                    if self.mesh.cp_size > 1:
                        if hasattr(self.model_parts[0], "prepare_model_inputs_for_cp"):
                            batch = self.model_parts[0].prepare_model_inputs_for_cp(**batch)
                        else:
                            batch = shard_batch_for_cp(batch, self.mesh.cp_mesh)

                    labels = batch.pop("labels", None)
                    filtered_batch = filter_forward_kwargs(self.model_parts[0], batch)

                    output = self.model_parts[0](**filtered_batch)
                    logits = output.logits if hasattr(output, "logits") else output

                    # 统计本 microbatch 的 label token 数，DP+CP 联合 all-reduce
                    num_tok = int((labels != -100).sum().item())
                    num_tok_global = _dp_cp_all_reduce_sum(num_tok, self.dp_cp_mesh).item()

                    # calculate_loss 返回 raw ce_sum（不除 N）；累加后末尾跨
                    # DP+CP all-reduce SUM 再除以全局 token 数，还原 token-mean
                    local_ce_sum = calculate_loss(
                        self.loss,
                        logits=logits,
                        labels=labels,
                        model=self.model_parts[0],
                        num_label_tokens=num_tok_global,
                    )

                    total_loss_sum += local_ce_sum.detach().item()
                    total_label_tokens += num_tok
        finally:
            # 恢复 train 模式
            for mp in self.model_parts:
                mp.train()

        # token-mean = Σ_{dp,cp} ce_sum_local / Σ_{dp,cp} num_tok_local
        global_ce_sum = _dp_cp_all_reduce_sum(total_loss_sum, self.dp_cp_mesh).item()
        global_tokens = _dp_cp_all_reduce_sum(total_label_tokens, self.dp_cp_mesh).item()
        global_val_loss = global_ce_sum / max(global_tokens, 1)

        return {"loss": global_val_loss, "num_tokens": global_tokens}

    # ── §7 单步优化器步进 ──

    def _run_train_optim_step(self, batches: list, max_grad_norm: float) -> dict:
        """执行一个完整的 optimizer step。

        三阶段：
        Phase 1: 统计全局 token 数
        Phase 2: 梯度累积（前向+反向 × grad_acc_steps）
        Phase 3: 梯度裁剪 + optimizer.step + lr_scheduler.step
        """
        num_batches = len(batches)

        self._step_start_time = time.time()  # Track step timing for throughput

        # ── Phase 1: 统计全局 token 数 ──
        num_label_tokens = 0
        for batch in batches:
            labels = batch.get("labels")
            if labels is not None:
                num_label_tokens += (labels != -100).sum().item()

        # DP+CP joint all-reduce (CP also shards the sequence, tokens need full count)
        num_label_tokens = _dp_cp_all_reduce_sum(num_label_tokens, self.dp_cp_mesh).item()

        # ── Phase 2: 梯度累积 ──
        loss_buffer = []
        prepare_for_grad_accumulation(self.model_parts)

        for i, batch in enumerate(batches):
            is_last = (i == num_batches - 1)

            # FSDP2: 最后一个 microbatch 才同步梯度（遍历所有 PP stage）
            for mp in self.model_parts:
                if FSDPModule is not None and isinstance(mp, FSDPModule):
                    mp.set_requires_gradient_sync(is_last)
            if is_last:
                prepare_for_final_backward(self.model_parts)

            self._forward_backward_step(
                i, batch,
                loss_buffer=loss_buffer,
                num_label_tokens=num_label_tokens,
                num_batches=num_batches,
            )

            if i == 0:
                prepare_after_first_microbatch(self.model_parts)

        # ── Phase 3: 梯度裁剪 + optimizer step ──
        # num_label_tokens 仅在 token_weighted 且非 PP 时传入：rank_average
        # 等 mean 尺度 loss（§10）不能再除 N，PP 场景由 PP runtime 平衡（§10.1）
        _token_weighted = (
            getattr(self.cfg.loss, "loss_aggregation", "token_weighted") == "token_weighted"
        )
        grad_norm = scale_grads_and_clip_grad_norm(
            self.model_parts, max_grad_norm,
            num_label_tokens=(
                num_label_tokens if (_token_weighted and not self.pp_enabled) else None
            ),
        )

        self.checkpointer.maybe_wait_for_staging()

        for opt in (self.optimizer if isinstance(self.optimizer, list) else [self.optimizer]):
            opt.step()
            opt.zero_grad()

        # lr_scheduler 可能为 None（setup 中条件赋值）——加守卫避免 AttributeError
        schedulers = (
            self.lr_scheduler
            if isinstance(self.lr_scheduler, list)
            else ([self.lr_scheduler] if self.lr_scheduler is not None else [])
        )
        for sch in schedulers:
            sch.step()

        # ── Loss 聚合（logged loss = token-mean） ──
        # local_loss 为 raw ce_sum（未除 N，见 §8 Step 5）。
        # 日志损失 = Σ_{microbatches, dp_ranks} ce_sum_local / N_global
        total_ce_sum = sum(loss_buffer)
        global_ce_sum = _dp_cp_all_reduce_sum(total_ce_sum, self.dp_cp_mesh).item()
        global_loss = global_ce_sum / max(num_label_tokens, 1)

        # ── 计算吞吐 ──
        step_time = time.time() - self._step_start_time
        tps = num_label_tokens / max(step_time, 1e-8)
        mfu = calculate_mfu(
            tps, self.mfu_calc.flops_per_token, self.mfu_calc.peak_tflops,
            self._world_size,
        )

        if schedulers:
            lr = schedulers[-1].get_last_lr()[0]
        else:
            first_opt = (
                self.optimizer[0] if isinstance(self.optimizer, list) else self.optimizer
            )
            lr = first_opt.param_groups[0]["lr"]

        return {
            "loss": global_loss,
            "grad_norm": grad_norm,
            "lr": lr,
            "step_time": step_time,
            "tps": tps,
            "mfu": mfu,
            "num_tokens": num_label_tokens,
        }

    # ── §8 前向+反向传播 ──

    def _forward_backward_step(
        self, idx: int, batch: dict, *,
        loss_buffer: list,
        num_label_tokens: int,
        num_batches: int,
    ) -> None:
        """单次 microbatch 的前向 + 反向传播。"""
        model = self.model_parts[0]

        # ── Step 1: 数据 → GPU ──
        batch = {
            k: v.to(self._device, non_blocking=True)
            if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # ── Step 2: CP batch 准备 ──
        # CP 包装由 apply_sharding_plan 编译期完成（_wrap_cp_inner_attention），
        # 训练循环无需任何 CP context manager / hook。
        if self.mesh.cp_size > 1:
            if hasattr(model, "prepare_model_inputs_for_cp"):
                batch = model.prepare_model_inputs_for_cp(**batch)
            else:
                batch = shard_batch_for_cp(batch, self.mesh.cp_mesh)

        # ── Step 3: 分离 labels ──
        labels = batch.pop("labels", None)

        # ── Step 4: 前向传播 ──
        # FSDP2 梯度同步策略：只有非最后一个 microbatch 才 defer 梯度 all-reduce
        sync_ctx = get_sync_ctx(
            self.model_parts,
            is_optim_step=True,
            defer_fsdp_grad_sync=(idx != num_batches - 1),
        )

        with sync_ctx:
            # 过滤 forward 不接受的 kwargs
            filtered_batch = filter_forward_kwargs(model, batch)

            output = model(**filtered_batch)

            # ── Step 5: Loss 计算 ──
            # local_loss = ce_sum_local（raw，不除 N）；token-mean 归一化推迟到
            # §7.1 scale_grads 统一完成，避免 calculate_loss 与 scale_grads 双除
            logits = output.logits if hasattr(output, "logits") else output
            local_loss = calculate_loss(
                self.loss,
                logits=logits,
                labels=labels,
                model=model,
                num_label_tokens=num_label_tokens,
                loss_aggregation=getattr(self.cfg.loss, "loss_aggregation", "token_weighted"),
                hidden_states=getattr(output, "hidden_states", None),
                lm_weight=(
                    model.lm_head.weight
                    if hasattr(model, "lm_head") and model.lm_head is not None
                    else None
                ),
            )

            # MTP loss（Qwen3.5 等）
            if hasattr(output, "mtp_per_depth_logits"):
                local_loss = local_loss + calculate_mtp_loss(
                    output.mtp_per_depth_logits,
                    output.mtp_per_depth_h,
                    labels,
                    self.loss,
                )

            loss_buffer.append(local_loss.detach())

            # ── Step 6: 反向传播 ──
            # loss 乘 dp_size 抵消 FSDP2 的 DP-mean 除法；cp 维度不需要额外
            # 乘法（每个 cp rank 处理不同的序列段，不是冗余计算）
            dp_group_size = self.mesh.dp_size
            (local_loss * dp_group_size).backward()
