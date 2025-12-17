"""
Smoke test: 只跑 1 个 task + 1 个 epoch，用于快速验证实验数据保存逻辑是否正确。

会在 logs/ 下生成：
- results_<run_id>_smoke.txt
- run_<run_id>_smoke_custom.json
"""

import os
import time
from datetime import datetime
import io
import re
import contextlib

import yaml
import torch

from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin

from src.data import get_core50_benchmark
from src.models import get_resnet18
from src.strategies import get_strategy
from src.utils import set_seed, setup_logger
from src.json_metrics_logger import JsonMetricsLogger

# 复用 main.py 里已经实现好的 json/metrics 逻辑
import main as dil_main


_METRIC_LINE_RE = re.compile(r"^\\s*([A-Za-z0-9_./\\-]+)\\s*=\\s*([-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?)\\s*$")


def _parse_printed_metrics(text: str) -> dict:
    """
    解析 Avalanche/Logger 打印出来的指标行（形如 `Top1_Acc_Stream/... = 0.1234`）。
    """
    out = {}
    for line in (text or "").splitlines():
        m = _METRIC_LINE_RE.match(line)
        if not m:
            continue
        k, v = m.group(1), m.group(2)
        try:
            out[k] = float(v)
        except Exception:
            pass
    return out


def smoke_test(config_path: str = "configs/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 强制 1 epoch，避免长跑
    config["n_epochs"] = 1
    # 更快一点（可选）
    config["batch_size"] = int(config.get("batch_size", 128))

    set_seed(config["seed"])

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    config["device"] = device

    log_dir = config.get("log_dir", "./logs")
    os.makedirs(log_dir, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger("DIL-SmokeTest", os.path.join(log_dir, f"experiment_{run_id}_smoke.log"))

    results_txt_path = os.path.join(log_dir, f"results_{run_id}_smoke.txt")
    strategy_json_path = os.path.join(log_dir, f"run_{run_id}_smoke_custom.json")

    # 加载数据，只取第 0 个 task（train）和第 0 个 task（test）做验证
    benchmark = get_core50_benchmark(scenario="ni", root=config["dataset_root"])
    train_stream = list(benchmark.train_stream)[:1]
    test_stream = list(benchmark.test_stream)[:1]

    json_logger = JsonMetricsLogger()
    results_f = open(results_txt_path, "w", buffering=1)
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        loggers=[InteractiveLogger(), TextLogger(results_f), json_logger],
    )

    model = get_resnet18(pretrained=True).to(device)
    strategy = get_strategy("Custom", model, config)
    strategy.evaluator = eval_plugin

    # 关键：把 JsonMetricsLogger 也注册为 strategy plugin，才能触发 after_training_epoch/after_eval_exp 结算缓存
    try:
        strategy.plugins.append(json_logger)
    except Exception:
        pass

    # 让 epoch 记录也生效（写到 strategy_log["epochs"]）
    epoch_records: list = []
    try:
        strategy.plugins.append(dil_main.EpochMetricsCapturePlugin(epoch_records, phase="train"))
    except Exception:
        pass

    run_meta = {
        "run_id": run_id,
        "started_at_utc": dil_main._utc_now_iso(),
        "config_path": config_path,
        "config": dil_main._jsonable(config),
        "strategy": "Custom",
        "paths": {
            "log_dir": os.path.abspath(log_dir),
            "results_txt": os.path.abspath(results_txt_path),
            "strategy_json": os.path.abspath(strategy_json_path),
        },
    }

    strategy_log = {
        **run_meta,
        "events": [],
        "tasks": [],
        "epochs": epoch_records,
        "summary": {},
        "finished_at_utc": None,
        "error": None,
    }
    dil_main._write_json_atomic(strategy_json_path, strategy_log)

    # 只跑 1 个 task
    exp = train_stream[0]
    task_idx = 0

    logger.info("Smoke test: train_start")
    strategy_log["events"].append({"ts_utc": dil_main._utc_now_iso(), "type": "train_start", "task_idx": task_idx})
    dil_main._write_json_atomic(strategy_json_path, strategy_log)

    # 捕获“终端打印的指标”（最稳：与用户看到完全一致）
    train_buf = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(train_buf), contextlib.redirect_stderr(train_buf):
        strategy.train(exp, num_workers=0)
    train_s = time.time() - t0
    train_text = train_buf.getvalue()
    train_metrics = _parse_printed_metrics(train_text)

    logger.info("Smoke test: eval_start")
    strategy_log["events"].append({"ts_utc": dil_main._utc_now_iso(), "type": "eval_start", "task_idx": task_idx})
    dil_main._write_json_atomic(strategy_json_path, strategy_log)

    eval_buf = io.StringIO()
    t1 = time.time()
    with contextlib.redirect_stdout(eval_buf), contextlib.redirect_stderr(eval_buf):
        res = strategy.eval(test_stream)
    eval_s = time.time() - t1
    eval_text = eval_buf.getvalue()
    eval_metrics = _parse_printed_metrics(eval_text)

    # 规整 schema
    schema = dil_main._extract_final_schema_metrics(eval_metrics)

    strategy_log["tasks"].append(
        {
            "task_idx": task_idx,
            "train_seconds": train_s,
            "eval_seconds": eval_s,
            "eval_result": dil_main._jsonable(res),
            "train_metrics": dil_main._jsonable(train_metrics),
            "eval_metrics": dil_main._jsonable(eval_metrics),
            "captured_stdout": {
                "train": train_text[-20000:],  # 限制大小，避免 JSON 过大
                "eval": eval_text[-20000:],
            },
            "ts_utc": dil_main._utc_now_iso(),
        }
    )
    strategy_log["summary"] = {
        "final_task_idx": task_idx,
        "final_metrics_schema": dil_main._jsonable(schema),
    }

    strategy_log["events"].append(
        {
            "ts_utc": dil_main._utc_now_iso(),
            "type": "task_done",
            "task_idx": task_idx,
            "train_seconds": train_s,
            "eval_seconds": eval_s,
        }
    )
    strategy_log["finished_at_utc"] = dil_main._utc_now_iso()
    dil_main._write_json_atomic(strategy_json_path, strategy_log)

    print(f"[OK] Smoke test finished. JSON: {strategy_json_path}")
    print(f"[OK] results txt: {results_txt_path}")
    try:
        results_f.flush()
        results_f.close()
    except Exception:
        pass


if __name__ == "__main__":
    smoke_test()


