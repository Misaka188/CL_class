import torch
import torch.nn as nn
from avalanche.training import Naive, Replay
from avalanche.training.templates.supervised import SupervisedTemplate
from avalanche.models import SimpleMLP
from typing import Union, Dict, Any
from torch.optim import Adam
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.storage_policy import ReservoirSamplingBuffer


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

    if strategy_name.lower() == 'naive':
        # 基础持续学习策略（增量微调）
        # 这是最简单的持续学习方法，不使用任何正则化或回放机制
        strategy = Naive(
            model=model,                           # 要训练的模型
            optimizer=optimizer,                   # 优化器
            criterion=torch.nn.CrossEntropyLoss(), # 损失函数（交叉熵）
            train_mb_size=config['batch_size'],    # 训练批次大小
            train_epochs=config['n_epochs'],       # 每个任务的训练轮数
            eval_mb_size=128,                      # 评估批次大小
            device=torch.device(config['device'])  # 训练设备
        )
    elif strategy_name.lower() == 'replay':
        # 经验回放策略
        # 通过在缓冲区中存储旧任务样本并在新任务训练时重放来减轻灾难性遗忘
        storage_plugin = ReplayPlugin(
            mem_size=config['buffer_size']         # 设置缓冲区大小
        )

        strategy = Replay(
            model=model,                           # 要训练的模型
            optimizer=optimizer,                   # 优化器
            criterion=torch.nn.CrossEntropyLoss(), # 损失函数（交叉熵）
            mem_size=config['buffer_size'],        # 回放缓冲区大小
            train_mb_size=config['batch_size'],    # 训练批次大小
            train_epochs=config['n_epochs'],       # 每个任务的训练轮数
            eval_mb_size=128,                      # 评估批次大小
            device=torch.device(config['device']), # 训练设备
            plugins=[storage_plugin]               # 添加回放插件
        )
    elif strategy_name.lower() == 'custom':
        # 自定义策略占位符 - 用于实现您的自定义持续学习方法
        strategy = CustomStrategy(
            model=model,                           # 要训练的模型
            optimizer=optimizer,                   # 优化器
            criterion=torch.nn.CrossEntropyLoss(), # 损失函数（交叉熵）
            config=config                          # 配置字典
        )
    else:
        # 如果提供了未知的策略名称，则抛出错误
        raise ValueError(f"Unknown strategy: {strategy_name}. Supported: 'Naive', 'Replay', 'Custom'")

    return strategy


class CustomStrategy(SupervisedTemplate):
    """
    自定义策略类，继承自SupervisedTemplate
    这是占位符，您可以在此处使用自定义损失函数实现自己的持续学习方法
    """
    def __init__(self, model, optimizer, criterion, config, **kwargs):
        """
        初始化自定义策略

        Args:
            model: 要训练的PyTorch模型
            optimizer: 优化器
            criterion: 损失函数
            config: 配置字典
        """
        # 调用父类构造函数初始化基本参数
        super().__init__(
            model=model,                           # 要训练的模型
            optimizer=optimizer,                   # 优化器
            criterion=criterion,                   # 损失函数
            train_mb_size=config['batch_size'],    # 训练批次大小
            train_epochs=config['n_epochs'],       # 每个任务的训练轮数
            eval_mb_size=128,                      # 评估批次大小
            device=torch.device(config['device']), # 训练设备
            **kwargs                               # 其他参数
        )
        # 保存配置信息供后续使用
        self.config = config

        # TODO: 在此处添加自定义一致性损失
        # 示例：self.custom_loss = YourCustomLossFunction()
        # TODO: 实现任何自定义回放机制或正则化技术
        # 示例：self.replay_buffer = YourCustomReplayBuffer()

    def forward(self, mb_x):
        """
        前向传播函数

        Args:
            mb_x: 小批次输入数据

        Returns:
            模型输出
        """
        return self.model(mb_x)

    def criterion(self):
        """
        定义自定义损失计算
        这里是您实现自定义损失函数的主要位置
        """
        # TODO: 实现结合标准交叉熵和其他项的自定义损失
        # 示例实现：
        # base_loss = torch.nn.CrossEntropyLoss()(self.mb_output, self.mb_y)
        # consistency_loss = self.calculate_consistency_loss()
        # total_loss = base_loss + self.config.get('consistency_weight', 0.1) * consistency_loss
        # return total_loss
        return super().criterion()

    def after_training_exp(self, **kwargs):
        """
        在每次经验训练后执行的操作
        这是另一个可自定义的重要位置
        """
        # TODO: 可在此处添加特定于您的自定义策略的逻辑
        # 例如：更新自定义缓冲区、调整学习率、记录特定指标等
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