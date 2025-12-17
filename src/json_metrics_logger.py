from __future__ import annotations

from typing import Dict, Any, List, Optional

from avalanche.core import SupervisedPlugin
from avalanche.evaluation.metric_results import MetricValue
from avalanche.logging import BaseLogger


class JsonMetricsLogger(BaseLogger, SupervisedPlugin):
    """
    把 EvaluationPlugin 产生的 MetricValue 通过 BaseLogger 回调收集起来。

    说明：
    - Avalanche 的指标并不是通过 `get_last_metrics()` 统一返回的；
      InteractiveLogger/TextLogger 之所以能打印，是因为它们实现了 BaseLogger，
      在 `log_single_metric` 中接收到了指标。
    - 本 logger 复用相同机制，把指标收集成 dict，供主程序写入 JSON。
    """

    def __init__(self):
        super().__init__()
        # EvaluationPlugin 会通过 BaseLogger.log_metrics -> log_single_metric
        # 把 MetricValue 推送到这里。我们先缓存，等 strategy 回调触发时再“结算”成记录。
        self._metric_buf: Dict[str, Any] = {}
        self.last_metrics: Dict[str, Any] = {}

        # 方便主程序按粒度写入 JSON
        self.last_train_epoch_metrics: Optional[Dict[str, Any]] = None
        self.last_eval_exp_metrics: Optional[Dict[str, Any]] = None

        self.train_epoch_records: List[Dict[str, Any]] = []
        self.eval_exp_records: List[Dict[str, Any]] = []

        # 原始 MetricValue 快照（可选）
        self.raw_records: List[Dict[str, Any]] = []

    # BaseLogger API
    def log_single_metric(self, name, value, x_plot):
        # BaseLogger.log_metrics 会把 dict 值拆成 name/key 形式再调用本方法
        self._metric_buf[str(name)] = value
        self.last_metrics[str(name)] = value

    # SupervisedPlugin callbacks: 需要把本对象 append 到 strategy.plugins 才会触发
    def after_training_epoch(self, strategy, **kwargs):
        if not self._metric_buf:
            return
        metrics = dict(self._metric_buf)
        # 只保留 epoch 粒度训练指标
        if any("/train_phase/" in k and "_Epoch/" in k for k in metrics.keys()):
            self.last_train_epoch_metrics = metrics
            self.train_epoch_records.append(
                {
                    "phase": "train_epoch",
                    "task_idx": getattr(strategy.experience, "current_experience", None),
                    "epoch_idx": getattr(strategy.clock, "train_exp_epochs", None),
                    "metrics": metrics,
                }
            )
        self._metric_buf = {}

    def after_eval_exp(self, strategy, **kwargs):
        if not self._metric_buf:
            return
        metrics = dict(self._metric_buf)
        if any("/eval_phase/" in k and "_Exp/" in k for k in metrics.keys()) or any(
            "/eval_phase/" in k and "_Stream/" in k for k in metrics.keys()
        ):
            self.last_eval_exp_metrics = metrics
            self.eval_exp_records.append(
                {
                    "phase": "eval_exp",
                    "task_idx": getattr(strategy.experience, "current_experience", None),
                    "metrics": metrics,
                }
            )
        self._metric_buf = {}

    def reset(self):
        self._metric_buf = {}
        self.last_metrics = {}
        self.last_train_epoch_metrics = None
        self.last_eval_exp_metrics = None
        self.train_epoch_records = []
        self.eval_exp_records = []
        self.raw_records = []


