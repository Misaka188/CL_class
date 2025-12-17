import torch
import torchvision.models as models
from torch import nn
from typing import Optional


def get_resnet18(pretrained: bool = True) -> nn.Module:
    """
    返回一个ResNet-18模型，可选择加载ImageNet预训练权重

    Args:
        pretrained: 是否加载ImageNet预训练权重

    Returns:
        ResNet-18模型
    """
    # 加载预训练的ResNet-18模型，如果pretrained为True则使用ImageNet权重
    model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)

    # 修改最终的全连接层以适配CORe50数据集的50个类别
    # 获取原始全连接层的输入特征数
    num_features = model.fc.in_features
    # 替换为新的全连接层，输出维度为50（CORe50数据集的类别数量）
    model.fc = nn.Linear(num_features, 50)  # CORe50有50个类别

    return model