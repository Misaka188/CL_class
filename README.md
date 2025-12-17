## 介绍
这是一个基于 Avalanche 的 Domain Incremental Learning (DIL) 小型项目，使用 PyTorch 在 CORe50（NI 场景）上验证持续学习策略，包括基线 Naive、Replay，以及自定义的 Replay + LwF 方案。

## 环境与依赖
- Python 3.10（建议在独立虚拟环境/conda 环境中运行）
- 主要依赖（见 `requirements.txt`）：
  - torch==2.4.1
  - torchvision>=0.10.0
  - avalanche-lib>=0.4.0
  - pyyaml>=5.4.0
  - tqdm>=4.60.0
  - numpy>=1.21.0

安装：
```bash
pip install -r requirements.txt
```

## 项目结构
- `main.py`：主入口，运行完整 DIL 训练与评估。
- `smoke_test_one_epoch.py`：快速冒烟测试，只跑 1 个 task / 1 个 epoch，验证日志与指标保存。
- `configs/config.yaml`：超参数配置（epoch、lr、buffer、蒸馏权重等）。
- `src/`：模型、数据加载、策略定义、工具函数。
- `logs/`：运行时生成的各类日志（终端原始/清洗版、CSV、JSON、txt）。
- `clean_terminal_log.py`：独立的终端日志清洗脚本。

## 配置要点（`configs/config.yaml`）
- `n_epochs`: 10  
- `lr`: 0.001（配合 MultiStepLR：第 7 个 epoch 后 *0.1）  
- `buffer_size`: 2000（Replay 缓冲）  
- `distill_lambda`: 0.5（LwF 蒸馏权重）  
- `distill_temp`: 2.0（蒸馏温度）  
- `batch_size`: 128  
- `device`: "cuda"

## 训练与评估
完整实验：
```bash
cd ~/project/CL
python main.py --config configs/config.yaml
```

冒烟测试（更快，仅 1 task / 1 epoch）：
```bash
cd ~/project/CL
python smoke_test_one_epoch.py
```

数据集：CORe50（NI），默认根目录 `./data`，可在 `config.yaml` 修改 `dataset_root`。

## 策略简介（`src/strategies.py`）
- **Naive**：纯增量微调。
- **Replay**：经验回放（`mem_size=buffer_size`）。
- **Custom (Replay + LwF)**：
  - 回放：`ReplayPlugin`。
  - 蒸馏：上一任务模型为 teacher，KL 蒸馏，权重 `distill_lambda`，温度 `distill_temp`。
  - 学习率调度：`MultiStepLR(milestones=[7], gamma=0.1)`，通过 `StepLRSchedulerPlugin` 在每个 epoch 结束 `step()`。

## 日志与输出
运行后 `logs/` 下会出现：
- `terminal_<ts>.log`：完整 stdout/stderr。
- `terminal_<ts>.log.clean.log`：自动清洗版（去掉 tqdm 进度条）。
- `training_results.csv` / `eval_results.csv`：Avalanche CSVLogger 输出的结构化指标。
- `run_<run_id>_*.json`：包含元数据、任务耗时、评估指标等。
- `results_<run_id>.txt`：简要结果文本。

## 模型权重
训练时会在每个任务结束自动保存模型权重，命名格式：
`<checkpoint_dir>/<strategy_name>_<run_id>_task<task_idx>.pt`
默认目录为 `./checkpoints`，可在 `configs/config.yaml` 里修改 `checkpoint_dir`。


