import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import json


# ==========================================
# ✅ 改进1: MixUp数据增强
# ==========================================
class MixUpAugmentation:
    """
    MixUp数据增强

    原理：
    混合两个样本：
        x_mixed = λ * x1 + (1-λ) * x2
        y_mixed = λ * y1 + (1-λ) * y2
    其中 λ ~ Beta(α, α)

    优势：
    - 正则化效果，减少过拟合
    - 增加样本多样性
    - 平滑决策边界

    适用场景：
    - 小数据集（如Abilene）
    - 容易过拟合的模型

    参考：
    Zhang et al. "mixup: Beyond Empirical Risk Minimization" ICLR 2018
    """
    def __init__(self, alpha=0.2, prob=0.5):
        """
        Args:
            alpha: Beta分布参数，越大混合越均匀
                   α=0.2: 轻度混合（推荐）
                   α=1.0: 均匀混合
            prob: 应用MixUp的概率
        """
        self.alpha = alpha
        self.prob = prob

    def __call__(self, images, targets):
        """
        Args:
            images: [B, T, C, H, W] - 输入历史帧
            targets: [B, 1, H, W] - 目标帧

        Returns:
            mixed_images, mixed_targets, lambda_value
        """
        if not self.training or np.random.rand() > self.prob:
            return images, targets, 1.0

        batch_size = images.size(0)

        # 采样lambda
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        # 随机配对
        index = torch.randperm(batch_size, device=images.device)

        # 混合
        mixed_images = lam * images + (1 - lam) * images[index]
        mixed_targets = lam * targets + (1 - lam) * targets[index]

        return mixed_images, mixed_targets, lam

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


# ==========================================
# ✅ 改进2: 感知损失
# ==========================================
class PerceptualLoss(nn.Module):
    """
    感知损失 - 使用预训练CNN的中间特征

    原理：
    不直接比较像素，而是比较特征空间的距离：
        L_perceptual = ||φ(pred) - φ(target)||²
    其中 φ 是预训练CNN的特征提取器

    优势：
    - 捕捉语义相似性
    - 对小的像素偏移不敏感
    - 生成更自然的图像

    实现：
    使用训练好的VisionEncoder提取特征
    """
    def __init__(self, feature_extractor, layers=[2, 3], weights=[0.5, 0.5]):
        """
        Args:
            feature_extractor: 预训练的CNN（如VisionEncoder的model部分）
            layers: 使用哪些层的特征（列表索引）
            weights: 各层特征的权重
        """
        super().__init__()
        self.feature_extractor = feature_extractor
        self.layers = layers
        self.weights = weights

        # 冻结特征提取器
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.feature_extractor.eval()

    def extract_features(self, x):
        """
        提取中间层特征

        Args:
            x: [B, C, H, W]
        Returns:
            list of features
        """
        features = []

        # 假设feature_extractor是ImprovedCNNEncoder
        # 逐层提取
        h = x

        # Stem
        if hasattr(self.feature_extractor, 'stem'):
            h = self.feature_extractor.stem(h)

        # Blocks
        if hasattr(self.feature_extractor, 'blocks'):
            for i, block in enumerate(self.feature_extractor.blocks):
                h = block(h)
                if i in self.layers:
                    features.append(h)

        return features

    def forward(self, pred, target):
        """
        Args:
            pred: [B, 1, H, W] - 预测图像
            target: [B, 1, H, W] - 目标图像
        Returns:
            loss: 标量
        """
        # 确保在评估模式
        self.feature_extractor.eval()

        with torch.no_grad():
            target_features = self.extract_features(target)

        pred_features = self.extract_features(pred)

        # 计算各层损失
        loss = 0.0
        for i, (pred_feat, target_feat, weight) in enumerate(
            zip(pred_features, target_features, self.weights)
        ):
            loss += weight * F.mse_loss(pred_feat, target_feat)

        return loss


# ==========================================
# ✅ 改进3: 带重启的余弦退火学习率
# ==========================================
class CosineAnnealingWarmRestarts:
    """
    带重启的余弦退火

    原理：
    周期性地重启学习率，避免陷入局部最优

    学习率曲线：
        |
        |  ╱╲      ╱╲      ╱╲
        | ╱  ╲    ╱  ╲    ╱  ╲
        |╱    ╲  ╱    ╲  ╱    ╲
        +--------------------→ epoch
             T0    2T0    3T0

    参数：
        T_0: 第一次重启的周期
        T_mult: 周期倍增因子（通常为1或2）
        eta_min: 最小学习率

    优势：
    - 逃离局部最优
    - 多次收敛机会
    - 适合长训练
    """
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=1e-7, last_epoch=-1):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.last_epoch = last_epoch

        self.T_cur = last_epoch
        self.T_i = T_0
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.T_cur = self.T_cur + 1

        if self.T_cur >= self.T_i:
            self.T_cur = 0
            self.T_i = self.T_i * self.T_mult

        # Cosine annealing
        lr = self.eta_min + (self.base_lr - self.eta_min) * \
             (1 + np.cos(np.pi * self.T_cur / self.T_i)) / 2

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


# ==========================================
# ✅ 改进4: 早停机制
# ==========================================
class EarlyStopping:
    """
    早停机制

    原理：
    监控验证指标，如果连续N个epoch没有改善则停止训练

    参数：
        patience: 容忍的epoch数
        min_delta: 最小改善幅度
        mode: 'min'（越小越好）或'max'（越大越好）
    """
    def __init__(self, patience=30, min_delta=0.001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


# ==========================================
# ✅ 改进5: EMA模型
# ==========================================
class EMAModel:
    """
    指数移动平均模型

    原理：
    维护参数的移动平均：
        θ_ema = decay * θ_ema + (1 - decay) * θ

    优势：
    - 平滑参数更新
    - 提升泛化性能
    - 推理时用EMA模型

    典型decay：
        0.999: 标准（约1000步的平均）
        0.9999: 长期平均（约10000步）
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # 初始化shadow参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """更新EMA参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + \
                             self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """应用EMA参数（推理前调用）"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """恢复原始参数（推理后调用）"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# ==========================================
# ✅ 改进6: 损失计算器（集成所有损失）
# ==========================================
class CompositeLoss(nn.Module):
    """
    复合损失函数

    整合：
    1. 基础MSE/加权MSE
    2. 多尺度损失
    3. 梯度损失
    4. 感知损失
    """
    def __init__(self, config, feature_extractor=None):
        super().__init__()
        self.config = config

        # 加权MSE
        self.use_weighted_mse = config['model'].get('use_weighted_mse', True)
        if self.use_weighted_mse:
            self.low_threshold = config['model'].get('low_traffic_threshold', 0.15)
            self.mid_threshold = config['model'].get('mid_traffic_threshold', 0.5)
            self.low_weight = config['model'].get('low_traffic_weight', 4.0)
            self.mid_weight = config['model'].get('mid_traffic_weight', 2.5)
            self.high_weight = config['model'].get('high_traffic_weight', 2.0)

        # 多尺度损失
        self.use_multiscale = config['model'].get('use_multiscale_loss', True)
        if self.use_multiscale:
            self.multiscale_scales = config['model'].get('multiscale_scales', [1, 2])
            self.multiscale_weights = config['model'].get('multiscale_weights', [1.0, 0.3])

        # 梯度损失
        self.use_gradient = config['model'].get('use_gradient_loss', True)
        self.gradient_weight = config['model'].get('gradient_loss_weight', 0.15)

        # 感知损失
        self.use_perceptual = config['model'].get('use_perceptual_loss', True)
        self.perceptual_weight = config['model'].get('perceptual_loss_weight', 0.1)

        if self.use_perceptual and feature_extractor is not None:
            perceptual_layers = config['model'].get('perceptual_feature_layers', [2, 3])
            self.perceptual_loss = PerceptualLoss(
                feature_extractor,
                layers=perceptual_layers
            )
        else:
            self.perceptual_loss = None

    def weighted_mse_loss(self, pred, target):
        """加权MSE损失"""
        mse = (pred - target) ** 2

        # 根据流量大小分配权重
        weights = torch.ones_like(target)
        weights[target < self.low_threshold] = self.low_weight
        weights[(target >= self.low_threshold) & (target < self.mid_threshold)] = self.mid_weight
        weights[target >= self.mid_threshold] = self.high_weight

        return (mse * weights).mean()

    def multiscale_loss(self, pred, target):
        """多尺度损失"""
        loss = 0.0

        for scale, weight in zip(self.multiscale_scales, self.multiscale_weights):
            if scale == 1:
                loss += weight * F.mse_loss(pred, target)
            else:
                # 下采样
                pred_down = F.avg_pool2d(pred, kernel_size=scale, stride=scale)
                target_down = F.avg_pool2d(target, kernel_size=scale, stride=scale)
                loss += weight * F.mse_loss(pred_down, target_down)

        return loss

    def gradient_loss(self, pred, target):
        """梯度损失（保持边缘清晰）"""

        # ✅ 添加这段维度检查代码
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        if pred.size(1) != 1:
            pred = pred.mean(dim=1, keepdim=True)
        if target.size(1) != 1:
            target = target.mean(dim=1, keepdim=True)

        # Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3)

        # 计算梯度
        pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, sobel_y, padding=1)

        target_grad_x = F.conv2d(target, sobel_x, padding=1)
        target_grad_y = F.conv2d(target, sobel_y, padding=1)

        # 梯度损失
        loss = F.mse_loss(pred_grad_x, target_grad_x) + \
               F.mse_loss(pred_grad_y, target_grad_y)

        return loss

    def forward(self, pred, target):
        """
        Args:
            pred: [B, 1, H, W] - 预测
            target: [B, 1, H, W] - 真实值
        Returns:
            loss: 标量
            loss_dict: 各项损失的字典（用于日志）
        """

        # ✅ 添加这段维度检查代码
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        if pred.size(1) != 1:
            pred = pred.mean(dim=1, keepdim=True)
        if target.size(1) != 1:
            target = target.mean(dim=1, keepdim=True)

        assert pred.shape == target.shape, \
            f"Shape mismatch: pred {pred.shape} vs target {target.shape}"

        loss_dict = {}
        total_loss = 0.0

        # 计算MSE/MAE/MAPE用于监控
        pure_mse = F.mse_loss(pred, target)
        loss_dict['pure_mse'] = pure_mse.item()

        mae = F.l1_loss(pred, target)
        loss_dict['mae'] = mae.item()

        mape = torch.mean(torch.abs((pred - target) / target)) * 100
        loss_dict['mape'] = mape.item()

        # 1. 基础MSE/加权MSE
        if self.use_weighted_mse:
            mse_loss = self.weighted_mse_loss(pred, target)
            loss_dict['weighted_mse'] = mse_loss.item()
        else:
            mse_loss = F.mse_loss(pred, target)
            loss_dict['mse'] = mse_loss.item()

        total_loss += mse_loss

        # 2. 多尺度损失
        if self.use_multiscale:
            ms_loss = self.multiscale_loss(pred, target)
            loss_dict['multiscale'] = ms_loss.item()
            total_loss += ms_loss

        # 3. 梯度损失
        if self.use_gradient:
            grad_loss = self.gradient_loss(pred, target)
            loss_dict['gradient'] = grad_loss.item()
            total_loss += self.gradient_weight * grad_loss

        # 4. 感知损失
        if self.use_perceptual and self.perceptual_loss is not None:
            perc_loss = self.perceptual_loss(pred, target)
            loss_dict['perceptual'] = perc_loss.item()
            total_loss += self.perceptual_weight * perc_loss

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


# ==========================================
# ✅ 改进7: 完整的训练器
# ==========================================
class ImprovedTrainer:
    """
    改进的训练器 - 整合所有改进
    """
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config,
        device='cuda',
        feature_extractor=None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # 优化器
        self.optimizer = self._build_optimizer()

        # 学习率调度器
        self.scheduler = self._build_scheduler()

        # 混合精度
        self.use_amp = config['training'].get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None

        # 梯度累积
        self.grad_accum_steps = config['training'].get('gradient_accumulation_steps', 1)

        # MixUp
        mixup_config = config['data'].get('augmentation', {})
        self.mixup = MixUpAugmentation(
            alpha=mixup_config.get('mixup_alpha', 0.2),
            prob=mixup_config.get('mixup_prob', 0.5)
        )

        # 损失函数
        self.criterion = CompositeLoss(config, feature_extractor)

        # EMA
        self.use_ema = config['training'].get('use_ema', True)
        if self.use_ema:
            ema_decay = config['training'].get('ema_decay', 0.999)
            self.ema = EMAModel(self.model, decay=ema_decay)
        else:
            self.ema = None

        # 早停
        early_stop_config = config.get('early_stopping', {})
        if early_stop_config.get('enabled', False):
            self.early_stopping = EarlyStopping(
                patience=early_stop_config.get('patience', 30),
                min_delta=early_stop_config.get('min_delta', 0.001),
                mode=early_stop_config.get('mode', 'min')
            )
        else:
            self.early_stopping = None

        # 路径
        self.output_dir = Path(config['paths']['output_dir'])
        self.checkpoint_dir = Path(config['paths']['checkpoint_dir'])
        self.log_dir = Path(config['paths']['log_dir'])

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 日志
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

        # 最佳指标
        self.best_val_loss = float('inf')
        self.best_epoch = 0

    def _build_optimizer(self):
        """构建优化器"""
        opt_config = self.config['training']
        opt_type = opt_config.get('optimizer', 'adamw').lower()

        params = self.model.parameters()
        lr = opt_config['learning_rate']
        weight_decay = opt_config.get('weight_decay', 0.05)

        if opt_type == 'adamw':
            optimizer = torch.optim.AdamW(
                params,
                lr=lr,
                betas=(opt_config.get('beta1', 0.9), opt_config.get('beta2', 0.999)),
                weight_decay=weight_decay
            )
        elif opt_type == 'adam':
            optimizer = torch.optim.Adam(
                params,
                lr=lr,
                betas=(opt_config.get('beta1', 0.9), opt_config.get('beta2', 0.999)),
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")

        return optimizer

    def _build_scheduler(self):
        """构建学习率调度器"""
        sch_config = self.config['training']
        sch_type = sch_config.get('lr_scheduler', 'cosine_with_restarts').lower()

        if sch_type == 'cosine_with_restarts':
            restart_period = sch_config.get('restart_period', 100)
            min_lr = sch_config.get('min_lr', 1e-7)
            scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=restart_period,
                eta_min=min_lr
            )
        elif sch_type == 'cosine':
            num_epochs = sch_config['num_epochs']
            min_lr = sch_config.get('min_lr', 1e-7)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs,
                eta_min=min_lr
            )
        elif sch_type == 'step':
            step_size = sch_config.get('lr_step_size', 50)
            gamma = sch_config.get('lr_gamma', 0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
        else:
            scheduler = None

        return scheduler

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        self.mixup.train()

        epoch_loss = 0.0
        epoch_loss_dict = {}

        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            # 数据加载
            images = batch['history'].to(self.device)  # [B, T, C, H, W]
            target = batch['target'].to(self.device)  # [B, 1, H, W]

            # MixUp增强
            images, target, lam = self.mixup(images, target)

            # 前向传播
            with autocast(enabled=self.use_amp):
                # 这里需要完整的前向传播（包括vision encoder, qwen, condition encoder, unet）
                # 简化版本，假设model已经封装好
                output = self.model(images, target)  # 返回预测的噪声

                # 计算损失
                loss, loss_dict = self.criterion(output['pred'], target)

                # 梯度累积
                loss = loss / self.grad_accum_steps

            # 反向传播
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # 梯度更新（每grad_accum_steps步）
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                # 梯度裁剪
                if self.config['training'].get('clip_grad_norm'):
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['clip_grad_norm']
                    )

                # 优化器步进
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

                # EMA更新
                if self.ema is not None:
                    self.ema.update()

            # 统计
            epoch_loss += loss.item() * self.grad_accum_steps

            for key, value in loss_dict.items():
                epoch_loss_dict[key] = epoch_loss_dict.get(key, 0) + value

            # 更新进度条
            pbar.set_postfix({
                'loss': loss.item() * self.grad_accum_steps,
                'mse': loss_dict.get('pure_mse', 0.0),  # 显示纯 MSE
                'mae': loss_dict.get('mae', 0.0),  # 显示纯 MAE
                'mape': loss_dict.get('mape', 0.0),
                'lr': self.optimizer.param_groups[0]['lr']
            })

        # 平均损失
        epoch_loss /= num_batches
        for key in epoch_loss_dict:
            epoch_loss_dict[key] /= num_batches

        return epoch_loss, epoch_loss_dict

    def validate(self, epoch):
        """验证"""
        self.model.eval()
        self.mixup.eval()

        # 使用EMA模型（如果有）
        if self.ema is not None:
            self.ema.apply_shadow()

        val_loss = 0.0
        val_loss_dict = {}

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                images = batch['history'].to(self.device)
                target = batch['target'].to(self.device)

                # 前向传播
                output = self.model(images, target)

                # 计算损失
                loss, loss_dict = self.criterion(output['pred'], target)

                val_loss += loss.item()
                for key, value in loss_dict.items():
                    val_loss_dict[key] = val_loss_dict.get(key, 0) + value

        # 恢复原始模型
        if self.ema is not None:
            self.ema.restore()

        # 平均损失
        val_loss /= len(self.val_loader)
        for key in val_loss_dict:
            val_loss_dict[key] /= len(self.val_loader)

        return val_loss, val_loss_dict

    def save_checkpoint(self, epoch, is_best=False):
        """保存checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        if self.ema is not None:
            checkpoint['ema_shadow'] = self.ema.shadow

        # 保存最新
        path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)

        # 保存最佳
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"✅ Saved best model at epoch {epoch}")

        # 清理旧checkpoint（保留最近5个）
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        if len(checkpoints) > 5:
            for old_ckpt in checkpoints[:-5]:
                old_ckpt.unlink()

    def plot_curves(self, epoch):
        """绘制训练曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss曲线
        axes[0].plot(self.train_losses, label='Train Loss', linewidth=2)
        axes[0].plot(self.val_losses, label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 学习率曲线
        axes[1].plot(self.learning_rates, linewidth=2, color='orange')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')

        plt.tight_layout()
        plt.savefig(self.log_dir / f'training_curves_epoch_{epoch}.png', dpi=150)
        plt.close()

    def train(self):
        """完整训练流程"""
        num_epochs = self.config['training']['num_epochs']
        warmup_epochs = self.config['training'].get('warmup_epochs', 0)
        log_interval = self.config['training'].get('log_interval', 50)
        val_interval = self.config['training'].get('val_interval', 5)
        save_interval = self.config['training'].get('save_interval', 25)

        print("="*60)
        print("🚀 Starting Training")
        print("="*60)
        print(f"Total epochs: {num_epochs}")
        print(f"Batch size: {self.config['training']['batch_size']}")
        print(f"Gradient accumulation: {self.grad_accum_steps}")
        print(f"Effective batch size: {self.config['training']['batch_size'] * self.grad_accum_steps}")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"EMA: {self.use_ema}")
        print(f"MixUp: alpha={self.mixup.alpha}, prob={self.mixup.prob}")
        print("="*60)

        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()

            # Warmup阶段
            if epoch <= warmup_epochs:
                lr_scale = epoch / warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.config['training']['learning_rate'] * lr_scale

            # 训练
            train_loss, train_loss_dict = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            # 学习率调度
            if self.scheduler is not None and epoch > warmup_epochs:
                self.scheduler.step(epoch)

            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)

            # 验证
            if epoch % val_interval == 0:
                val_loss, val_loss_dict = self.validate(epoch)
                self.val_losses.append(val_loss)

                # 打印信息
                epoch_time = time.time() - epoch_start_time
                print(f"\nEpoch {epoch}/{num_epochs} - Time: {epoch_time:.2f}s")
                print(f"Train Loss: {train_loss:.6f}")
                print(f"Val Loss:   {val_loss:.6f}")
                print(f"LR:         {current_lr:.2e}")

                # 检查是否最佳
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.best_mse = val_loss_dict['pure_mse']
                    self.best_mae = val_loss_dict['mae']
                    self.best_mape = val_loss_dict['mape']
                    self.best_epoch = epoch

                print(f"Best Val Loss: {self.best_val_loss:.6f} (Epoch {self.best_epoch})")
                print(f"Best MSE: {self.best_mse:.6f}")
                print(f"Best MAE: {self.best_mae:.6f}")
                print(f"Best MAPE: {self.best_mape:.6f}")

                # 早停检查
                if self.early_stopping is not None:
                    if self.early_stopping(val_loss, epoch):
                        print(f"\n⚠️ Early stopping triggered at epoch {epoch}")
                        print(f"Best epoch was {self.best_epoch} with loss {self.best_val_loss:.6f}")
                        break

            # 保存checkpoint
            if epoch % save_interval == 0 or epoch == num_epochs:
                self.save_checkpoint(epoch, is_best=(epoch == self.best_epoch))

            # 绘制曲线
            if epoch % (val_interval * 2) == 0:
                self.plot_curves(epoch)

        print("\n" + "="*60)
        print("✅ Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.6f} at epoch {self.best_epoch}")
        print("="*60)

        # 最终保存
        self.save_checkpoint(num_epochs, is_best=False)
        self.plot_curves(num_epochs)

        # 保存训练历史
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch
        }

        with open(self.log_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
