import argparse
import yaml
import torch
from torch import nn
from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.logging.csv_logger import CSVLogger
from avalanche.training.plugins import EvaluationPlugin
from src.models import get_resnet18
from src.data import get_core50_benchmark
from src.strategies import get_strategy
from src.utils import set_seed, setup_logger
import os
import json
import time
import platform
import getpass
import socket
import re
from datetime import datetime, timezone
from avalanche.training.plugins import SupervisedPlugin
from src.json_metrics_logger import JsonMetricsLogger


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(obj):
    """
    将 Avalanche/PyTorch 返回的结果结构尽可能转成可 JSON 序列化的对象。
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    # torch scalar / numpy scalar
    try:
        import numpy as np  # noqa: F401
        import numpy
        if isinstance(obj, numpy.generic):
            return obj.item()
    except Exception:
        pass
    try:
        if isinstance(obj, torch.Tensor):
            if obj.numel() == 1:
                return obj.item()
            return obj.detach().cpu().tolist()
    except Exception:
        pass
    # fallback: string repr
    return str(obj)


def _write_json_atomic(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=False)
    os.replace(tmp_path, path)


def _extract_final_schema_metrics(eval_metrics: dict) -> dict:
    """
    将 Avalanche 的 metrics dict 规整为固定 schema，便于后续画图/做表。
    重点抽取：
    - acc_task0..acc_task7：来自 Top1_Acc_Stream/eval_phase/test_stream/TaskXYZ
    - forgetting_stream：来自 StreamForgetting/eval_phase/test_stream（若存在）
    """
    out = {
        "acc_tasks": {},
        "forgetting_stream": None,
    }

    # acc_task{idx}
    for k, v in (eval_metrics or {}).items():
        ks = str(k)
        if "Top1_Acc_Stream/eval_phase/test_stream/Task" in ks:
            # 例：Top1_Acc_Stream/eval_phase/test_stream/Task000
            tail = ks.split("/Task")[-1]
            task_str = tail[:3] if len(tail) >= 3 else tail
            try:
                idx = int(task_str)
                out["acc_tasks"][f"acc_task{idx}"] = v
            except Exception:
                continue

    # forgetting_stream（常见键：StreamForgetting/eval_phase/test_stream）
    for k, v in (eval_metrics or {}).items():
        ks = str(k)
        if ks == "StreamForgetting/eval_phase/test_stream" or "StreamForgetting/eval_phase/test_stream" in ks:
            out["forgetting_stream"] = v
            break

    return out


# ---------------- terminal log cleaning（去掉 tqdm 进度条行） ----------------
_PROGRESS_PATTERNS = [
    re.compile(r"^\s*\d+%?\|.+it/s\]"),      # 例："  1%| | 1/118 [..it/s]"
    re.compile(r"^\s*\d+it\s*\[.+it/s\]"),   # 例："0it [00:00, ?it/s]"
    re.compile(r"^\s*\r"),                   # 残留的 CR 行
]


def _is_progress_line(line: str) -> bool:
    s = line.rstrip("\n")
    for pat in _PROGRESS_PATTERNS:
        if pat.search(s):
            return True
    return False


def _clean_terminal_log(inp: str, out: str | None = None) -> str:
    """读取 inp，去掉 tqdm 进度条行，写到 out（默认 inp.clean.log），返回 out 路径。"""
    out = out or (inp + ".clean.log")
    try:
        with open(inp, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return ""

    kept = []
    for ln in lines:
        if _is_progress_line(ln):
            continue
        kept.append(ln)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.writelines(kept)
    return out


class EpochMetricsCapturePlugin(SupervisedPlugin):
    """
    捕获每个 epoch 的指标快照（训练/评估阶段），追加到外部 list。
    注意：Avalanche 的指标由 EvaluationPlugin 计算；这里从 evaluator.get_last_metrics() 抓取快照。
    """

    def __init__(self, sink_list: list, phase: str):
        super().__init__()
        self._sink = sink_list
        self._phase = phase

    def after_training_epoch(self, strategy, **kwargs):
        if self._phase != "train":
            return
        m = {}
        try:
            m = strategy.evaluator.get_last_metrics() or {}
        except Exception as e:
            m = {"_error_get_last_metrics": str(e)}
        # fallback：某些版本可能把 last_metrics 存在私有属性中
        if not m:
            m = getattr(strategy.evaluator, "_last_metrics", {}) or {}
        rec = {
            "ts_utc": _utc_now_iso(),
            "phase": "train",
            "task_idx": getattr(strategy.experience, "current_experience", None),
            "epoch_idx": getattr(getattr(strategy, "clock", None), "train_exp_epochs", None),
            "metrics": _jsonable(m),
        }
        self._sink.append(rec)

    def after_eval_epoch(self, strategy, **kwargs):
        if self._phase != "eval":
            return
        m = {}
        try:
            m = strategy.evaluator.get_last_metrics() or {}
        except Exception as e:
            m = {"_error_get_last_metrics": str(e)}
        if not m:
            m = getattr(strategy.evaluator, "_last_metrics", {}) or {}
        rec = {
            "ts_utc": _utc_now_iso(),
            "phase": "eval",
            "task_idx": getattr(strategy.experience, "current_experience", None),
            "epoch_idx": getattr(getattr(strategy, "clock", None), "eval_exp_epochs", None),
            "metrics": _jsonable(m),
        }
        self._sink.append(rec)


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

    # 基线脚本固定：10 epoch，关闭学习率调度
    config["n_epochs"] = 10
    config["use_scheduler"] = False

    # 设置随机种子以确保实验结果可重现
    set_seed(config['seed'])

    # 设置计算设备（GPU或CPU）
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    config['device'] = device

    # 设置日志记录器，用于记录实验过程
    log_dir = config.get('log_dir', './logs')
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger('DIL-Experiment', os.path.join(log_dir, 'experiment.log'))

    # 结构化实验日志：每次运行生成一个 run_id，且每个策略输出一个 json
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_txt_path = os.path.join(log_dir, f"results_{run_id}.txt")
    run_meta = {
        "run_id": run_id,
        "started_at_utc": _utc_now_iso(),
        "cwd": os.getcwd(),
        "user": getpass.getuser(),
        "host": socket.gethostname(),
        "platform": {
            "python": platform.python_version(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "versions": {
            "torch": getattr(torch, "__version__", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_runtime": getattr(torch.version, "cuda", None),
        },
        "config_path": args.config,
        "config": _jsonable(config),
        "paths": {
            "log_dir": os.path.abspath(log_dir),
            "experiment_log": os.path.abspath(os.path.join(log_dir, "experiment.log")),
            "results_txt": os.path.abspath(results_txt_path),
        },
    }
    if torch.cuda.is_available():
        try:
            run_meta["gpu"] = {
                "device_count": torch.cuda.device_count(),
                "devices": [
                    {
                        "idx": i,
                        "name": torch.cuda.get_device_name(i),
                        "capability": list(torch.cuda.get_device_capability(i)),
                    }
                    for i in range(torch.cuda.device_count())
                ],
            }
        except Exception as e:
            run_meta["gpu"] = {"error": str(e)}

    # 加载CORe50基准数据集，使用'ni'(新实例)场景
    benchmark = get_core50_benchmark(scenario='ni', root=config['dataset_root'])

    # 初始化模型：加载预训练的ResNet-18，并移动到指定设备
    # 注意：每个策略都应使用“独立初始化的模型”，避免策略之间互相污染实验结果

    # 设置评估插件，定义要计算的指标
    # 额外 logger：用于把“终端能打印的指标”收集成 dict，供 JSON 落盘
    json_logger = JsonMetricsLogger()
    results_f = open(results_txt_path, "w", buffering=1)
    csv_logger = CSVLogger(log_folder=log_dir)
    eval_plugin = EvaluationPlugin(
        # 准确率指标：在小批次、轮次、经验(任务)和流级别计算
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        # 遗忘指标：在经验(任务)和流级别计算
        forgetting_metrics(experience=True, stream=True),
        # 日志记录器：交互式日志和文本日志
        loggers=[
            InteractiveLogger(),   # 终端实时输出
            TextLogger(results_f), # 追加到 results_*.txt
            csv_logger,            # 写入 logs/training_results.csv & logs/eval_results.csv
            json_logger,           # 自定义 JSON 收集器（后续可选解析）
        ]
    )

    # 要比较的策略列表（默认启用 Custom：Replay + LwF）
    # 仅运行基线：Naive + Replay
    strategies = ['Naive', 'Replay']

    # 存储实验结果的字典
    results = {}

    # 遍历每种策略进行训练和评估
    for strategy_name in strategies:
        print(f"\nTraining with {strategy_name} strategy...")
        logger.info(f"Starting training with {strategy_name} strategy")

        # 每个策略独立模型，保证对比公平
        model = get_resnet18(pretrained=True)
        model.to(device)

        # 初始化指定策略
        strategy = get_strategy(strategy_name, model, config)
        # 设置策略的评估器
        strategy.evaluator = eval_plugin

        # 关键：把 JsonMetricsLogger 注册为 strategy plugin，才能在 after_training_epoch/after_eval_exp 中结算缓存
        try:
            strategy.plugins.append(json_logger)
        except Exception:
            pass

        # 记录每个 epoch 的指标（训练/评估），写入 JSON
        epoch_records: list = []
        try:
            strategy.plugins.append(EpochMetricsCapturePlugin(epoch_records, phase="train"))
        except Exception:
            pass

        # 为该策略准备 json 日志结构
        strategy_json_path = os.path.join(log_dir, f"run_{run_id}_{strategy_name.lower()}.json")
        strategy_log = {
            **run_meta,
            "strategy": strategy_name,
            "strategy_json": strategy_json_path,
            "events": [],
            "tasks": [],
            "summary": {},
            "epochs": epoch_records,
            "epoch_metrics": [],
            "eval_exp_metrics": [],
            "finished_at_utc": None,
            "error": None,
        }
        _write_json_atomic(strategy_json_path, strategy_log)

        # 训练循环：遍历每个任务
        for task_idx, experience in enumerate(benchmark.train_stream):
            print(f"Training on task {task_idx}")
            logger.info(f"Training on task {task_idx}")

            # 在当前经验(任务)上训练
            t0 = time.time()
            strategy_log["events"].append({
                "ts_utc": _utc_now_iso(),
                "type": "train_start",
                "task_idx": task_idx,
            })
            _write_json_atomic(strategy_json_path, strategy_log)

            try:
                train_res = strategy.train(experience, num_workers=4)
            except Exception as e:
                strategy_log["error"] = {
                    "ts_utc": _utc_now_iso(),
                    "type": "exception",
                    "where": "train",
                    "task_idx": task_idx,
                    "message": str(e),
                }
                strategy_log["finished_at_utc"] = _utc_now_iso()
                _write_json_atomic(strategy_json_path, strategy_log)
                raise
            train_s = time.time() - t0

            # 从 JsonMetricsLogger 收集该 task 的 epoch 级指标（可能包含多个 epoch）
            try:
                if json_logger.train_epoch_records:
                    for r in json_logger.train_epoch_records:
                        strategy_log["epoch_metrics"].append(_jsonable(r))
                    json_logger.train_epoch_records = []
            except Exception:
                pass

            # 在测试流上评估模型性能
            t1 = time.time()
            strategy_log["events"].append({
                "ts_utc": _utc_now_iso(),
                "type": "eval_start",
                "task_idx": task_idx,
            })
            _write_json_atomic(strategy_json_path, strategy_log)

            try:
                res = strategy.eval(benchmark.test_stream)
            except Exception as e:
                strategy_log["error"] = {
                    "ts_utc": _utc_now_iso(),
                    "type": "exception",
                    "where": "eval",
                    "task_idx": task_idx,
                    "message": str(e),
                }
                strategy_log["finished_at_utc"] = _utc_now_iso()
                _write_json_atomic(strategy_json_path, strategy_log)
                raise
            eval_s = time.time() - t1

            # 以 JsonMetricsLogger 收集到的“eval_exp”指标为准（和终端输出一致）
            last_metrics = json_logger.last_eval_exp_metrics or {}

            res_json = _jsonable(res)
            last_metrics_json = _jsonable(last_metrics)

            print(f"Results after task {task_idx}: {last_metrics}")
            logger.info(f"Results after task {task_idx}: {last_metrics}")

            strategy_log["tasks"].append({
                "task_idx": task_idx,
                "train_seconds": train_s,
                "eval_seconds": eval_s,
                "train_result": _jsonable(train_res),
                "eval_result": res_json,
                "eval_metrics": last_metrics_json,
                "ts_utc": _utc_now_iso(),
            })

            # 保存本次 eval_exp 的完整指标快照序列（便于复查）
            try:
                if json_logger.eval_exp_records:
                    for r in json_logger.eval_exp_records:
                        strategy_log["eval_exp_metrics"].append(_jsonable(r))
                    json_logger.eval_exp_records = []
            except Exception:
                pass
            strategy_log["events"].append({
                "ts_utc": _utc_now_iso(),
                "type": "task_done",
                "task_idx": task_idx,
                "train_seconds": train_s,
                "eval_seconds": eval_s,
            })

            # ---- 自动保存模型权重：策略名 + run_id + task ----
            try:
                ckpt_dir = os.path.abspath(config.get("checkpoint_dir", "./checkpoints"))
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(
                    ckpt_dir,
                    f"{strategy_name}_{run_id}_task{task_idx}.pt",
                )
                torch.save(model.state_dict(), ckpt_path)
                strategy_log.setdefault("checkpoints", []).append({
                    "task_idx": task_idx,
                    "path": ckpt_path,
                    "ts_utc": _utc_now_iso(),
                })
            except Exception as e:
                strategy_log.setdefault("checkpoints_errors", []).append({
                    "task_idx": task_idx,
                    "error": str(e),
                    "ts_utc": _utc_now_iso(),
                })
            _write_json_atomic(strategy_json_path, strategy_log)

        # 保存该策略的结果
        results[strategy_name] = strategy
        print(f"Completed training with {strategy_name} strategy\n")
        logger.info(f"Completed training with {strategy_name} strategy")

        # 写一个简要 summary（便于后续画图/对比）
        if strategy_log["tasks"]:
            final_metrics = strategy_log["tasks"][-1].get("eval_metrics") or {}
            # 常用指标：最终 stream acc / stream forgetting（如果存在）
            summary = {
                "final_task_idx": strategy_log["tasks"][-1].get("task_idx"),
                "final_metrics_schema": _extract_final_schema_metrics(final_metrics),
            }
            strategy_log["summary"] = _jsonable(summary)

        strategy_log["finished_at_utc"] = _utc_now_iso()
        _write_json_atomic(strategy_json_path, strategy_log)

    try:
        results_f.flush()
        results_f.close()
    except Exception:
        pass

    print("All experiments completed!")
    logger.info("All experiments completed!")


if __name__ == "__main__":
    # 将所有终端输出（stdout + stderr）同时写入日志文件，方便事后完整回溯
    import sys
    import io
    import contextlib

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 日志目录：优先用 ./logs，相对当前文件所在目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    terminal_log_path = os.path.join(log_dir, f"terminal_{ts}.log")

    class _Tee(io.TextIOBase):
        def __init__(self, *streams):
            super().__init__()
            self._streams = streams

        def write(self, s):
            for t in self._streams:
                try:
                    t.write(s)
                except Exception:
                    pass
            return len(s)

        def flush(self):
            for t in self._streams:
                try:
                    t.flush()
                except Exception:
                    pass

    with open(terminal_log_path, "w", buffering=1, encoding="utf-8") as f:
        tee = _Tee(sys.__stdout__, f)
        # 同时重定向 stdout/stderr，这样你在终端看到的所有东西都会落到文件里
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            main()
    # 运行结束后，生成去除进度条的精简版日志
    cleaned = _clean_terminal_log(terminal_log_path)
    if cleaned:
        print(f"[log cleaned] {cleaned}")