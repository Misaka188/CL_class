import torch
import torch.nn as nn
import torch.nn.functional as F
from avalanche.training import Naive, Replay
from avalanche.training.templates import SupervisedTemplate
from typing import Dict, Any, Optional
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from avalanche.training.plugins import ReplayPlugin, SupervisedPlugin
from copy import deepcopy


def get_strategy(strategy_name: str, model: nn.Module, config: Dict[str, Any]) -> SupervisedTemplate:
    """
    根据名称返回相应的持续学习策略

    Args:
        strategy_name: 策略名称 ('Naive' 或 'Replay')
        model: 要训练的PyTorch模型
        config: 包含超参数的配置字典

    Returns:
        指定策略的实例
    """
    # 创建Adam优化器，使用配置中指定的学习率
    optimizer = Adam(model.parameters(), lr=config['lr'])
    use_scheduler = bool(config.get("use_scheduler", True))
    scheduler = MultiStepLR(optimizer, milestones=[7], gamma=0.1) if use_scheduler else None

    def _scheduler_plugin():
        if scheduler is None:
            return None
        return StepLRSchedulerPlugin(scheduler)

    if strategy_name.lower() == 'naive':
        # 基础持续学习策略（增量微调）
        # 这是最简单的持续学习方法，不使用任何正则化或回放机制
        plugins = []
        if use_scheduler:
            plugins.append(_scheduler_plugin())

        strategy = Naive(
            model=model,                           # 要训练的模型
            optimizer=optimizer,                   # 优化器
            criterion=torch.nn.CrossEntropyLoss(), # 损失函数（交叉熵）
            train_mb_size=config['batch_size'],    # 训练批次大小
            train_epochs=config['n_epochs'],       # 每个任务的训练轮数
            eval_mb_size=128,                      # 评估批次大小
            device=torch.device(config['device']), # 训练设备
            plugins=[p for p in plugins if p is not None]  # 学习率调度（可选）
        )
    elif strategy_name.lower() == 'replay':
        # 经验回放策略
        # 通过在缓冲区中存储旧任务样本并在新任务训练时重放来减轻灾难性遗忘
        storage_plugin = ReplayPlugin(
            mem_size=config['buffer_size']         # 设置缓冲区大小
        )

        plugins = [storage_plugin]
        if use_scheduler:
            plugins.append(_scheduler_plugin())

        strategy = Replay(
            model=model,                           # 要训练的模型
            optimizer=optimizer,                   # 优化器
            criterion=torch.nn.CrossEntropyLoss(), # 损失函数（交叉熵）
            mem_size=config['buffer_size'],        # 回放缓冲区大小
            train_mb_size=config['batch_size'],    # 训练批次大小
            train_epochs=config['n_epochs'],       # 每个任务的训练轮数
            eval_mb_size=128,                      # 评估批次大小
            device=torch.device(config['device']), # 训练设备
            plugins=[p for p in plugins if p is not None]  # 回放 + 学习率调度（可选）
        )
    elif strategy_name.lower() == 'custom':
        # 自定义策略：Replay + LwF（知识蒸馏）
        strategy = CustomStrategy(
            model=model,                           # 要训练的模型
            optimizer=optimizer,                   # 优化器
            criterion=torch.nn.CrossEntropyLoss(), # 损失函数（交叉熵）
            config=config,                         # 配置字典
            plugins=[p for p in [_scheduler_plugin()] if p is not None]  # 学习率调度（可选，ReplayPlugin 在内部追加）
        )
    else:
        # 如果提供了未知的策略名称，则抛出错误
        raise ValueError(f"Unknown strategy: {strategy_name}. Supported: 'Naive', 'Replay', 'Custom'")

    return strategy


class CustomStrategy(SupervisedTemplate):
    """
    自定义策略：Replay + LwF（Learning without Forgetting）

    - Replay：使用 Avalanche 的 ReplayPlugin 进行样本回放
    - LwF：维护一个 teacher（上一任务训练完成后的模型快照），对当前 batch 做 logits 蒸馏，缓解遗忘
    """
    def __init__(self, *, model, optimizer, criterion, config, **kwargs):
        """
        初始化自定义策略

        Args:
            model: 要训练的PyTorch模型
            optimizer: 优化器
            criterion: 损失函数
            config: 配置字典
        """
        # 调用父类构造函数初始化基本参数
        plugins = list(kwargs.pop("plugins", []) or [])
        plugins.append(ReplayPlugin(mem_size=int(config.get("buffer_size", 2000))))

        super().__init__(
            model=model,                           # 要训练的模型
            optimizer=optimizer,                   # 优化器
            criterion=criterion,                   # 损失函数
            train_mb_size=config['batch_size'],    # 训练批次大小
            train_epochs=config['n_epochs'],       # 每个任务的训练轮数
            eval_mb_size=128,                      # 评估批次大小
            device=torch.device(config['device']), # 训练设备
            plugins=plugins,                       # 回放插件 + 可能的其他插件
            **kwargs                               # 其他参数
        )
        # 保存配置信息供后续使用
        self.config = config

        # 蒸馏超参
        self.distill_lambda: float = float(config.get("distill_lambda", 0.5))
        self.distill_temp: float = float(config.get("distill_temp", 2.0))

        # teacher（上一任务模型快照）
        self.old_model: Optional[nn.Module] = None
        self._old_logits: Optional[torch.Tensor] = None

    def forward(self):
        """
        前向传播函数

        Args:
            mb_x: 小批次输入数据（由 Avalanche 在 self.mb_x 中提供）

        Returns:
            模型输出
        """
        mb_x = self.mb_x
        # current logits
        out = self.model(mb_x)

        # teacher logits（用于蒸馏）
        if self.old_model is not None and self.distill_lambda > 0:
            with torch.no_grad():
                self.old_model.eval()
                self._old_logits = self.old_model(mb_x)
        else:
            self._old_logits = None

        return out

    def criterion(self):
        """
        定义自定义损失计算
        这里是您实现自定义损失函数的主要位置
        """
        # 标准监督损失（交叉熵）
        ce_loss = super().criterion()

        # LwF 蒸馏损失：KL(current || teacher)
        if self._old_logits is None or self.distill_lambda <= 0:
            return ce_loss

        T = max(float(self.distill_temp), 1e-6)
        kl = F.kl_div(
            F.log_softmax(self.mb_output / T, dim=1),
            F.softmax(self._old_logits / T, dim=1),
            reduction="batchmean",
        ) * (T ** 2)

        lam = min(max(float(self.distill_lambda), 0.0), 1.0)
        return (1.0 - lam) * ce_loss + lam * kl

    def after_training_exp(self, **kwargs):
        """
        在每次经验训练后执行的操作
        这是另一个可自定义的重要位置
        """
        # 在任务结束后，更新 teacher 为当前模型快照
        try:
            self.old_model = deepcopy(self.model)
            self.old_model.to(self.device)
            self.old_model.eval()
            for p in self.old_model.parameters():
                p.requires_grad_(False)
        except Exception:
            # teacher 更新失败时不影响主训练流程
            self.old_model = None

        super().after_training_exp(**kwargs)

    def before_forward(self, **kwargs):
        """
        在前向传播之前执行的操作
        用于预处理输入或准备中间状态
        """
        # TODO: 如果需要，可以在这里添加预处理步骤
        super().before_forward(**kwargs)

    def after_forward(self, **kwargs):
        """
        在前向传播之后执行的操作
        用于处理模型输出或计算额外的度量
        """
        # TODO: 如果需要，可以在这里添加后处理步骤
        super().after_forward(**kwargs)


class StepLRSchedulerPlugin(SupervisedPlugin):
    """简单的 MultiStepLR 调度插件：每个训练 epoch 结束时 step 一次。"""

    def __init__(self, scheduler: torch.optim.lr_scheduler._LRScheduler):
        super().__init__()
        self.scheduler = scheduler

    def after_training_epoch(self, strategy, **kwargs):
        try:
            self.scheduler.step()
        except Exception:
            pass