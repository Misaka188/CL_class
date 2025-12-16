import argparse
import yaml
import torch
from torch import nn
from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin
from src.models import get_resnet18
from src.data import get_core50_benchmark
from src.strategies import get_strategy
from src.utils import set_seed, setup_logger
import os


def main():
    """
    主函数：执行域增量学习实验
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Domain Incremental Learning with CORe50')
    # 添加配置文件路径参数，默认为configs/config.yaml
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    # 从配置文件加载超参数
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 设置随机种子以确保实验结果可重现
    set_seed(config['seed'])

    # 设置计算设备（GPU或CPU）
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    config['device'] = device

    # 设置日志记录器，用于记录实验过程
    logger = setup_logger('DIL-Experiment', config.get('log_dir', './logs') + '/experiment.log')

    # 加载CORe50基准数据集，使用'ni'(新实例)场景
    benchmark = get_core50_benchmark(scenario='ni', root=config['dataset_root'])

    # 初始化模型：加载预训练的ResNet-18，并移动到指定设备
    model = get_resnet18(pretrained=True)
    model.to(device)

    # 设置评估插件，定义要计算的指标
    eval_plugin = EvaluationPlugin(
        # 准确率指标：在小批次、轮次、经验(任务)和流级别计算
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        # 遗忘指标：在经验(任务)和流级别计算
        forgetting_metrics(experience=True, stream=True),
        # 日志记录器：交互式日志和文本日志
        loggers=[InteractiveLogger(), TextLogger(open(os.path.join(config['log_dir'], 'results.txt'), 'w'))]
    )

    # 要比较的策略列表
    strategies = ['Naive', 'Replay']

    # 存储实验结果的字典
    results = {}

    # 遍历每种策略进行训练和评估
    for strategy_name in strategies:
        print(f"\nTraining with {strategy_name} strategy...")
        logger.info(f"Starting training with {strategy_name} strategy")

        # 初始化指定策略
        strategy = get_strategy(strategy_name, model, config)
        # 设置策略的评估器
        strategy.evaluator = eval_plugin

        # 训练循环：遍历每个任务
        for task_idx, experience in enumerate(benchmark.train_stream):
            print(f"Training on task {task_idx}")
            logger.info(f"Training on task {task_idx}")

            # 在当前经验(任务)上训练
            strategy.train(experience, num_workers=4)

            # 在测试流上评估模型性能
            res = strategy.eval(benchmark.test_stream)
            print(f"Results after task {task_idx}: {res}")
            logger.info(f"Results after task {task_idx}: {res}")

        # 保存该策略的结果
        results[strategy_name] = strategy
        print(f"Completed training with {strategy_name} strategy\n")
        logger.info(f"Completed training with {strategy_name} strategy")

    print("All experiments completed!")
    logger.info("All experiments completed!")


if __name__ == "__main__":
    main()