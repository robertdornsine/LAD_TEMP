#!/usr/bin/env python3
"""
Qwen-Adapter Traffic Diffusion - 图像版本训练脚本
支持单卡/多卡训练、混合精度、EMA等
"""

import os
import sys
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import yaml
import logging
from easydict import EasyDict
import numpy as np
import random
import time
from datetime import datetime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# ===== 导入改进的模块 =====
from models.qwen_adapter_model_image_enhanced_1117 import (
    ImprovedQwenTrafficDiffusion,
    create_model_from_config
)
from utils.training_1117 import (
    ImprovedTrainer,
    MixUpAugmentation,
    EarlyStopping,
    EMAModel
)
from scripts.traffic_image_dataset import TrafficImageDataset


# ==========================================
# 命令行参数解析
# ==========================================
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Train Improved Qwen-Adapter Traffic Diffusion Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ===== 配置文件 =====
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config YAML file (e.g., abilene_config_improved_v2.yaml)'
    )

    # ===== 数据路径 =====
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='Override data directory'
    )

    # ===== 训练参数 =====
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Override batch size'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=None,
        help='Override number of epochs'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='Override learning rate'
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=None,
        help='Override gradient accumulation steps'
    )

    # ===== 混合精度 =====
    parser.add_argument(
        '--use_amp',
        action='store_true',
        help='Enable automatic mixed precision'
    )
    parser.add_argument(
        '--no_amp',
        action='store_true',
        help='Disable automatic mixed precision (override config)'
    )

    # ===== 梯度裁剪 =====
    parser.add_argument(
        '--clip_grad_norm',
        type=float,
        default=None,
        help='Override gradient clipping threshold'
    )

    # ===== 设备 =====
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use'
    )
    parser.add_argument(
        '--gpu_ids',
        type=str,
        default=None,
        help='GPU IDs to use (e.g., "0,1,2,3")'
    )

    # ===== 分布式训练 =====
    parser.add_argument(
        '--distributed',
        action='store_true',
        help='Use distributed training'
    )
    parser.add_argument(
        '--local_rank',
        type=int,
        default=0,
        help='Local rank for distributed training'
    )

    # ===== 恢复训练 =====
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume from checkpoint path'
    )
    parser.add_argument(
        '--load_pretrained',
        type=str,
        default=None,
        help='Load pretrained weights (not resume, just weights)'
    )

    # ===== 新增：改进功能开关 =====
    parser.add_argument(
        '--use_mixup',
        action='store_true',
        help='Enable MixUp data augmentation'
    )
    parser.add_argument(
        '--use_ema',
        action='store_true',
        help='Enable EMA model'
    )
    parser.add_argument(
        '--use_early_stopping',
        action='store_true',
        help='Enable early stopping'
    )
    parser.add_argument(
        '--early_stop_patience',
        type=int,
        default=30,
        help='Early stopping patience (epochs)'
    )

    # ===== 调试 =====
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Debug mode (small dataset, no saving)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--val_interval',
        type=int,
        default=None,
        help='Override validation interval'
    )
    parser.add_argument(
        '--save_interval',
        type=int,
        default=None,
        help='Override checkpoint save interval'
    )

    # ===== 其他 =====
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Override output directory'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default=None,
        help='Experiment name (used for output dir naming)'
    )

    return parser.parse_args()


# ==========================================
# 工具函数
# ==========================================
def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 提示：确定性训练会降低性能
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """加载配置文件"""
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return EasyDict(config)


def override_config(config, args):
    """用命令行参数覆盖配置"""

    # 数据路径
    if args.data_dir:
        config.data.data_dir = args.data_dir

    # 训练参数
    if args.batch_size:
        config.training.batch_size = args.batch_size

    if args.num_epochs:
        config.training.num_epochs = args.num_epochs

    if args.learning_rate:
        config.training.learning_rate = args.learning_rate

    if args.gradient_accumulation_steps:
        config.training.gradient_accumulation_steps = args.gradient_accumulation_steps

    if args.clip_grad_norm:
        config.training.clip_grad_norm = args.clip_grad_norm

    # 混合精度
    if args.use_amp:
        config.training.use_amp = True
    if args.no_amp:
        config.training.use_amp = False

    # 输出目录
    if args.output_dir:
        config.paths.output_dir = args.output_dir
    elif args.experiment_name:
        # 根据实验名生成输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.paths.output_dir = f"outputs/{args.experiment_name}_{timestamp}"

    # 验证和保存间隔
    if args.val_interval:
        config.training.val_interval = args.val_interval

    if args.save_interval:
        config.training.save_interval = args.save_interval

    # 改进功能
    if args.use_mixup:
        config.data.augmentation = config.data.get('augmentation', {})
        config.data.augmentation['mixup_prob'] = 0.5

    if args.use_ema:
        config.training.use_ema = True

    if args.use_early_stopping:
        config.early_stopping = config.get('early_stopping', {})
        config.early_stopping['enabled'] = True
        config.early_stopping['patience'] = args.early_stop_patience

    # 调试模式
    if args.debug:
        config.training.num_epochs = 2
        config.training.val_interval = 1
        config.training.save_interval = 1
        config.debug = True

    return config


def setup_distributed(args):
    """设置分布式训练"""
    if args.distributed:
        # 初始化进程组
        dist.init_process_group(backend='nccl')

        # 获取rank和world_size
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # 设置设备
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)

        return rank, world_size, device

    else:
        # 单卡训练
        if args.gpu_ids:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

        if torch.cuda.is_available() and args.device == 'cuda':
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        return 0, 1, device


def setup_logging(log_file, rank):
    """设置日志"""
    if rank == 0:
        log_file.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        # 非主进程只输出到文件
        logging.basicConfig(
            level=logging.WARNING,
            format=f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file.parent / f'train_rank{rank}.log')
            ]
        )


def setup_paths(config, rank):
    """设置输出路径"""
    if rank == 0:
        output_dir = Path(config.paths.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        subdirs = ['checkpoints', 'logs', 'results', 'visualizations']
        for subdir in subdirs:
            (output_dir / subdir).mkdir(exist_ok=True)

        # 更新配置中的路径
        config.paths.checkpoint_dir = str(output_dir / 'checkpoints')
        config.paths.log_dir = str(output_dir / 'logs')
        config.paths.result_dir = str(output_dir / 'results')
        config.paths.vis_dir = str(output_dir / 'visualizations')

        # 保存配置
        with open(output_dir / 'config.yaml', 'w') as f:
            yaml.dump(dict(config), f, default_flow_style=False)

        print(f"\n✅ Output directory: {output_dir}")
        print(f"   - Checkpoints: {config.paths.checkpoint_dir}")
        print(f"   - Logs: {config.paths.log_dir}")
        print(f"   - Results: {config.paths.result_dir}")
        print(f"   - Visualizations: {config.paths.vis_dir}")


# ==========================================
# 数据加载器创建
# ==========================================
def create_dataloaders(config, is_distributed, rank, world_size, num_workers, debug=False):
    """创建数据加载器"""

    print(f"\n[Rank {rank}] Creating datasets...")

    # 训练集
    train_dataset = TrafficImageDataset(
        data_dir=config.data.data_dir,
        split='train',
        history_len=config.data.history_len,
        image_size=config.data.image_size,
        normalize=config.data.normalize,
        augmentation=config.data.get('augmentation', None)
    )

    # 验证集
    val_dataset = TrafficImageDataset(
        data_dir=config.data.data_dir,
        split='val',
        history_len=config.data.history_len,
        image_size=config.data.image_size,
        normalize=config.data.normalize,
        augmentation=None  # 验证集不使用数据增强
    )

    # 测试集（可选）
    try:
        test_dataset = TrafficImageDataset(
            data_dir=config.data.data_dir,
            split='test',
            history_len=config.data.history_len,
            image_size=config.data.image_size,
            normalize=config.data.normalize,
            augmentation=None
        )
    except Exception:
        test_dataset = None
        print(f"[Rank {rank}] No test dataset found, skipping...")

    # 调试模式：限制数据量
    if debug:
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, range(min(100, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(50, len(val_dataset))))
        print(f"[Rank {rank}] DEBUG MODE: Limited dataset size")

    print(f"[Rank {rank}] Dataset sizes:")
    print(f"   - Train: {len(train_dataset)}")
    print(f"   - Val:   {len(val_dataset)}")
    if test_dataset:
        print(f"   - Test:  {len(test_dataset)}")

    # 分布式采样器
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        )
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        sampler=train_sampler,
        shuffle=shuffle if train_sampler is None else False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0
    )

    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    return train_loader, val_loader, test_loader


# ==========================================
# 模型创建
# ==========================================
def create_model(config, device, is_distributed, rank):
    """创建改进的模型"""

    print(f"\n[Rank {rank}] Creating ImprovedQwenTrafficDiffusion model...")

    # 使用改进的模型
    model = ImprovedQwenTrafficDiffusion(config.model)

    # 移到设备
    model = model.to(device)

    # 分布式包装
    if is_distributed:
        # 找出哪些参数需要梯度
        find_unused = config.training.get('find_unused_parameters', False)

        model = DDP(
            model,
            device_ids=[device],
            output_device=device,
            find_unused_parameters=find_unused
        )
        print(f"[Rank {rank}] Model wrapped with DDP")

    return model


# ==========================================
# 主函数
# ==========================================
def main():
    """主函数"""

    print("=" * 80)
    print("Qwen-Adapter Traffic Diffusion - Improved Training")
    print("=" * 80)

    # ===== 1. 解析参数 =====
    args = parse_args()

    # ===== 2. 设置随机种子 =====
    set_seed(args.seed)
    print(f"✅ Random seed set to {args.seed}")

    # ===== 3. 设置分布式 =====
    rank, world_size, device = setup_distributed(args)
    is_distributed = args.distributed
    is_main_process = (rank == 0)

    if is_main_process:
        print(f"\n✅ Device: {device}")
        if is_distributed:
            print(f"   Distributed training with {world_size} GPUs")
        else:
            print(f"   Single GPU/CPU training")

    # ===== 4. 加载配置 =====
    config = load_config(args.config)
    config = override_config(config, args)

    # ===== 5. 设置路径 =====
    setup_paths(config, rank)

    # ===== 6. 设置日志 =====
    if is_main_process:
        log_file = Path(config.paths.log_dir) / 'train.log'
        setup_logging(log_file, rank)

        logging.info("=" * 80)
        logging.info("Training Configuration")
        logging.info("=" * 80)
        logging.info(f"Config file: {args.config}")
        logging.info(f"Device: {device}")
        logging.info(f"Distributed: {is_distributed}")

        if is_distributed:
            logging.info(f"World size: {world_size}")

        logging.info(f"\nTraining Parameters:")
        logging.info(f"  Batch size: {config.training.batch_size}")
        logging.info(f"  Gradient accumulation: {config.training.get('gradient_accumulation_steps', 1)}")
        logging.info(
            f"  Effective batch size: {config.training.batch_size * config.training.get('gradient_accumulation_steps', 1)}")
        logging.info(f"  Num epochs: {config.training.num_epochs}")
        logging.info(f"  Learning rate: {config.training.learning_rate}")
        logging.info(f"  Optimizer: {config.training.get('optimizer', 'adamw')}")
        logging.info(f"  LR scheduler: {config.training.get('lr_scheduler', 'cosine_with_restarts')}")

        logging.info(f"\nAdvanced Features:")
        logging.info(f"  Mixed precision (AMP): {config.training.get('use_amp', False)}")
        logging.info(f"  Gradient clipping: {config.training.get('clip_grad_norm', None)}")
        logging.info(f"  EMA: {config.training.get('use_ema', False)}")
        logging.info(f"  MixUp: {config.data.get('augmentation', {}).get('mixup_prob', 0) > 0}")
        logging.info(f"  Early stopping: {config.get('early_stopping', {}).get('enabled', False)}")

        logging.info(f"\nModel Configuration:")
        logging.info(f"  Vision encoder: {config.model.get('vision_encoder_type', 'improved_cnn')}")
        logging.info(f"  Condition type: {config.model.get('condition_type', 'dual')}")
        logging.info(f"  UNet base channels: {config.model.get('unet_base_channels', 128)}")
        logging.info(f"  Diffusion steps (train): {config.model.get('ddpm_train_steps', 20)}")
        logging.info(f"  Prediction type: {config.model.get('prediction_type', 'v_prediction')}")

        logging.info("=" * 80)

    # ===== 7. 创建数据加载器 =====
    train_loader, val_loader, test_loader = create_dataloaders(
        config,
        is_distributed,
        rank,
        world_size,
        args.num_workers,
        debug=args.debug
    )

    # ===== 8. 创建模型 =====
    model = create_model(config, device, is_distributed, rank)

    # ===== 9. 加载预训练权重（如果指定） =====
    if args.load_pretrained and is_main_process:
        logging.info(f"Loading pretrained weights from: {args.load_pretrained}")
        try:
            # 如果是DDP模型，需要访问.module
            base_model = model.module if hasattr(model, 'module') else model
            base_model.load_pretrained(args.load_pretrained, strict=False)
            logging.info("✅ Pretrained weights loaded successfully")
        except Exception as e:
            logging.warning(f"Failed to load pretrained weights: {e}")

    # ===== 10. 获取feature_extractor（用于感知损失） =====
    base_model = model.module if hasattr(model, 'module') else model
    feature_extractor = base_model.vision_encoder.model  # ImprovedCNNEncoder

    # ===== 11. 创建训练器 =====
    if is_main_process:
        logging.info("\nCreating ImprovedTrainer...")

    trainer = ImprovedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        feature_extractor=feature_extractor
    )

    # ===== 12. 恢复训练（如果指定） =====
    if args.resume:
        if is_main_process:
            logging.info(f"Resuming from checkpoint: {args.resume}")

        checkpoint = torch.load(args.resume, map_location=device)

        # 加载模型权重
        base_model = model.module if hasattr(model, 'module') else model
        base_model.load_state_dict(checkpoint['model_state_dict'])

        # 加载优化器状态
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 恢复训练历史
        if 'train_losses' in checkpoint:
            trainer.train_losses = checkpoint['train_losses']
            trainer.val_losses = checkpoint['val_losses']
            trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        # 恢复EMA
        if trainer.ema and 'ema_shadow' in checkpoint:
            trainer.ema.shadow = checkpoint['ema_shadow']

        start_epoch = checkpoint.get('epoch', 0) + 1

        if is_main_process:
            logging.info(f"✅ Resumed from epoch {start_epoch}")

    # ===== 13. 开始训练 =====
    if is_main_process:
        logging.info("\n" + "=" * 80)
        logging.info("Starting Training")
        logging.info("=" * 80)

    try:
        start_time = time.time()

        # 调用训练器的train方法
        trainer.train()

        total_time = time.time() - start_time

        if is_main_process:
            logging.info("\n" + "=" * 80)
            logging.info("Training Completed Successfully!")
            logging.info("=" * 80)
            logging.info(f"Total training time: {total_time / 3600:.2f} hours")
            logging.info(f"Best validation loss: {trainer.best_val_loss:.6f} (Epoch {trainer.best_epoch})")
            logging.info(f"Final model saved to: {config.paths.checkpoint_dir}/best_model.pt")

    except KeyboardInterrupt:
        if is_main_process:
            logging.info("\n" + "=" * 80)
            logging.info("Training interrupted by user")
            logging.info("=" * 80)

            # 保存中断时的checkpoint
            trainer.save_checkpoint(
                epoch=len(trainer.train_losses),
                is_best=False
            )
            logging.info("Checkpoint saved before exit")

    except Exception as e:
        if is_main_process:
            logging.error(f"Training failed with error: {e}")
            import traceback
            traceback.print_exc()
        raise

    finally:
        # ===== 14. 清理 =====
        if is_distributed:
            dist.destroy_process_group()

        if is_main_process:
            logging.info("\n✅ Training script finished")


# ==========================================
# 入口
# ==========================================
if __name__ == '__main__':
    main()