import random
import numpy as np
import torch
import logging
from typing import Optional
import os


def set_seed(seed: int = 42) -> None:
    """
    设置随机种子以确保结果可重现

    Args:
        seed: 随机种子值
    """
    # 设置Python内置random模块的种子
    random.seed(seed)
    # 设置NumPy的随机种子
    np.random.seed(seed)
    # 设置PyTorch的CPU随机种子
    torch.manual_seed(seed)
    # 如果CUDA可用，设置GPU随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)        # 为当前GPU设置种子
        torch.cuda.manual_seed_all(seed)    # 为所有GPU设置种子
    # 设置PyTorch的cudnn后端为确定性模式，确保结果可重现
    torch.backends.cudnn.deterministic = True
    # 禁用cuDNN的自动调优功能，确保每次运行结果一致
    torch.backends.cudnn.benchmark = False


def setup_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    设置具有指定名称和可选文件输出的日志记录器

    Args:
        name: 日志记录器名称
        log_file: 日志文件路径（可选）
        level: 日志级别

    Returns:
        配置好的日志记录器实例
    """
    # 创建日志格式器，包含时间戳、日志级别和消息
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    # 创建控制台处理器，用于在终端输出日志
    handler = logging.StreamHandler()
    # 设置处理器的格式
    handler.setFormatter(formatter)

    # 获取或创建指定名称的日志记录器
    logger = logging.getLogger(name)
    # 设置日志记录器的级别
    logger.setLevel(level)
    # 添加控制台处理器
    logger.addHandler(handler)

    # 如果提供了日志文件路径，则添加文件处理器
    if log_file:
        # 创建文件处理器，将日志写入文件
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        # 设置文件处理器的格式
        file_handler.setFormatter(formatter)
        # 添加文件处理器到日志记录器
        logger.addHandler(file_handler)

    return logger