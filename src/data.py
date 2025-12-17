import torch
from avalanche.benchmarks.classic import CORe50
from avalanche.benchmarks.utils import classification_dataset
from torch.utils.data import DataLoader
from typing import Tuple


def get_core50_benchmark(scenario: str = 'ni', root: str = './data') -> Tuple:
    """
    加载指定场景下的CORe50基准数据集

    Args:
        scenario: 场景类型 - 'ni' (New Instances) 用于域增量学习
        root: 数据集根目录

    Returns:
        包含训练和测试流的基准实例
    """
    # 加载CORe50基准数据集，针对'ni'(新实例)场景
    # 在'ni'场景中，每个任务包含来自所有类别的样本，适合域增量学习设置
    benchmark = CORe50(
        scenario=scenario,    # 使用'ni'场景，即新实例场景
        dataset_root=root     # 指定数据集下载/存储路径
    )

    return benchmark