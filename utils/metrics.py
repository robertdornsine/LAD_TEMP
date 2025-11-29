"""
评估指标 - 流量预测
"""

import numpy as np
import torch
import logging
from typing import Dict, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error


def compute_metrics(
        predictions: np.ndarray,
        targets: np.ndarray,
        mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    计算流量预测的各种指标

    Args:
        predictions: [N, 1, H, W] or [N, H, W]
        targets: [N, 1, H, W] or [N, H, W]
        mask: [N, 1, H, W] or [N, H, W] - 可选的mask（用于忽略某些区域）

    Returns:
        metrics: dict of metric_name -> value
    """
    # 确保shape一致
    if predictions.ndim == 4:
        predictions = predictions.squeeze(1)
    if targets.ndim == 4:
        targets = targets.squeeze(1)

    if mask is not None and mask.ndim == 4:
        mask = mask.squeeze(1)

    # Flatten
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()

    if mask is not None:
        mask_flat = mask.flatten().astype(bool)
        pred_flat = pred_flat[mask_flat]
        target_flat = target_flat[mask_flat]

    metrics = {}

    # ===== 基础回归指标 =====

    # MSE (Mean Squared Error)
    metrics['mse'] = mean_squared_error(target_flat, pred_flat)

    # RMSE (Root Mean Squared Error)
    metrics['rmse'] = np.sqrt(metrics['mse'])

    # MAE (Mean Absolute Error)
    metrics['mae'] = mean_absolute_error(target_flat, pred_flat)

    # MAPE (Mean Absolute Percentage Error)
    # 避免除零
    epsilon = 1e-8
    mape = np.abs((target_flat - pred_flat) / (target_flat + epsilon))
    metrics['mape'] = np.mean(mape) * 100  # 百分比

    # ===== R² Score =====
    ss_res = np.sum((target_flat - pred_flat) ** 2)
    ss_tot = np.sum((target_flat - np.mean(target_flat)) ** 2)
    metrics['r2'] = 1 - (ss_res / (ss_tot + epsilon))

    # ===== 相关系数 =====
    correlation = np.corrcoef(target_flat, pred_flat)[0, 1]
    metrics['correlation'] = correlation if not np.isnan(correlation) else 0.0

    # ===== Peak Signal-to-Noise Ratio (PSNR) =====
    # 假设数据范围是[0, 1]
    max_val = 1.0
    metrics['psnr'] = 10 * np.log10((max_val ** 2) / (metrics['mse'] + epsilon))

    # ===== 结构相似性指标 (SSIM) =====
    # 对每个样本计算SSIM，然后平均
    from skimage.metrics import structural_similarity as ssim

    ssim_values = []
    for i in range(len(predictions)):
        try:
            ssim_val = ssim(
                targets[i],
                predictions[i],
                data_range=1.0,
                multichannel=False
            )
            ssim_values.append(ssim_val)
        except:
            pass

    metrics['ssim'] = np.mean(ssim_values) if ssim_values else 0.0

    # ===== 峰值误差 =====
    abs_errors = np.abs(predictions - targets)
    metrics['max_error'] = np.max(abs_errors)
    metrics['p95_error'] = np.percentile(abs_errors, 95)
    metrics['p99_error'] = np.percentile(abs_errors, 99)

    # ===== 稀疏性指标（流量矩阵特定）=====
    # 零值占比
    pred_zero_ratio = np.sum(pred_flat < 0.01) / len(pred_flat)
    target_zero_ratio = np.sum(target_flat < 0.01) / len(target_flat)

    metrics['pred_sparsity'] = pred_zero_ratio
    metrics['target_sparsity'] = target_zero_ratio
    metrics['sparsity_diff'] = abs(pred_zero_ratio - target_zero_ratio)

    # ===== 分位数误差 =====
    # 低流量、中流量、高流量的误差
    q25, q75 = np.percentile(target_flat, [25, 75])

    low_traffic = target_flat < q25
    high_traffic = target_flat > q75
    mid_traffic = ~(low_traffic | high_traffic)

    if np.sum(low_traffic) > 0:
        metrics['mae_low_traffic'] = np.mean(np.abs(pred_flat[low_traffic] - target_flat[low_traffic]))
    else:
        metrics['mae_low_traffic'] = 0.0

    if np.sum(mid_traffic) > 0:
        metrics['mae_mid_traffic'] = np.mean(np.abs(pred_flat[mid_traffic] - target_flat[mid_traffic]))
    else:
        metrics['mae_mid_traffic'] = 0.0

    if np.sum(high_traffic) > 0:
        metrics['mae_high_traffic'] = np.mean(np.abs(pred_flat[high_traffic] - target_flat[high_traffic]))
    else:
        metrics['mae_high_traffic'] = 0.0

    return metrics


def compute_temporal_metrics(
        predictions: np.ndarray,
        targets: np.ndarray,
        timestamps: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    计算时间维度的指标（用于序列预测）

    Args:
        predictions: [N, T, H, W] - 预测序列
        targets: [N, T, H, W] - 目标序列
        timestamps: [N, T] - 时间戳（可选）
    """
    N, T = predictions.shape[:2]

    metrics = {}

    # 逐时间步的误差
    for t in range(T):
        step_metrics = compute_metrics(predictions[:, t], targets[:, t])
        for key, value in step_metrics.items():
            metrics[f'step_{t}_{key}'] = value

    # 平均误差
    avg_metrics = compute_metrics(
        predictions.reshape(-1, *predictions.shape[2:]),
        targets.reshape(-1, *targets.shape[2:])
    )

    for key, value in avg_metrics.items():
        metrics[f'avg_{key}'] = value

    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """格式化打印指标"""
    print(f"\n{'=' * 80}")
    print(f"{title:^80}")
    print(f"{'=' * 80}")

    # 分组
    groups = {
        'Error Metrics': ['mse', 'rmse', 'mae', 'mape', 'max_error', 'p95_error', 'p99_error'],
        'Quality Metrics': ['r2', 'correlation', 'psnr', 'ssim'],
        'Traffic-specific': ['pred_sparsity', 'target_sparsity', 'sparsity_diff',
                             'mae_low_traffic', 'mae_mid_traffic', 'mae_high_traffic']
    }

    for group_name, keys in groups.items():
        print(f"\n{group_name}:")
        print("-" * 80)
        for key in keys:
            if key in metrics:
                value = metrics[key]
                print(f"  {key:25s}: {value:12.6f}")

    print("=" * 80 + "\n")


def setup_logging(log_file: Optional[str] = None):
    """设置日志"""
    handlers = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


# ===== PyTorch版本的指标（用于训练中）=====

class MetricsTracker:
    """训练过程中的指标跟踪器"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.values = {}
        self.counts = {}

    def update(self, metrics: Dict[str, float], n: int = 1):
        """
        更新指标

        Args:
            metrics: 指标字典
            n: batch size（用于加权平均）
        """
        for key, value in metrics.items():
            if key not in self.values:
                self.values[key] = 0.0
                self.counts[key] = 0

            self.values[key] += value * n
            self.counts[key] += n

    def compute(self) -> Dict[str, float]:
        """计算平均指标"""
        return {
            key: self.values[key] / self.counts[key]
            for key in self.values.keys()
        }

    def summary(self) -> str:
        """生成摘要字符串"""
        metrics = self.compute()

        summary_keys = ['mse', 'rmse', 'mae', 'ssim', 'psnr']
        parts = []

        for key in summary_keys:
            if key in metrics:
                parts.append(f"{key}={metrics[key]:.4f}")

        return ", ".join(parts)


@torch.no_grad()
def compute_batch_metrics(
        predictions: torch.Tensor,
        targets: torch.Tensor
) -> Dict[str, float]:
    """
    计算一个batch的指标（快速版本，用于训练）

    Args:
        predictions: [B, 1, H, W]
        targets: [B, 1, H, W]
    """
    # 转为numpy
    pred_np = predictions.cpu().numpy()
    target_np = targets.cpu().numpy()

    # 计算基础指标
    mse = np.mean((pred_np - target_np) ** 2)
    mae = np.mean(np.abs(pred_np - target_np))

    return {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(np.sqrt(mse))
    }