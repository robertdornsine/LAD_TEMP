import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from PIL import Image
import random
from datetime import datetime


class TrafficImageDataset(Dataset):
    """
    流量图像数据集 - PNG时间序列版本

    数据格式：
    - PNG图像文件
    - 时间戳命名：YYYY-MM-DD-HH-MM.png
    - 单一文件夹，按时间顺序排列
    """

    def __init__(
            self,
            data_dir,
            split='train',
            history_len=16,
            image_size=12,
            normalize=True,
            augmentation=None,
            train_ratio=0.7,  # ✅ 新增：训练集比例
            val_ratio=0.15,  # ✅ 新增：验证集比例
            # test_ratio = 1 - train_ratio - val_ratio
    ):
        """
        Args:
            data_dir: PNG图像文件夹路径
            split: 'train', 'val', 'test'
            history_len: 历史帧数（16帧）
            image_size: 目标尺寸（12x12）
            normalize: 是否归一化到[-1, 1]
            augmentation: 数据增强配置
            train_ratio: 训练集比例（0.7 = 70%）
            val_ratio: 验证集比例（0.15 = 15%）
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.history_len = history_len
        self.image_size = image_size
        self.normalize = normalize
        self.augmentation = augmentation or {}

        # ===== 加载所有图像文件 =====
        self.image_files = self._load_image_files()

        # ===== 划分数据集 =====
        self._split_dataset(train_ratio, val_ratio)

        # ===== 检测图像尺寸 =====
        self._detect_image_size()

        print(f"✅ [{split.upper()}] Dataset initialized")
        print(f"   - Data dir: {self.data_dir}")
        print(f"   - Total images: {len(self.image_files)}")
        print(f"   - Samples (sequences): {len(self)}")
        print(f"   - History length: {self.history_len}")
        print(f"   - Image size: {self.img_height}x{self.img_width} → {self.image_size}x{self.image_size}")
        print(f"   - Normalize: {self.normalize}")
        print(f"   - Augmentation: {bool(self.augmentation)}")

    def _load_image_files(self):
        """
        加载所有PNG图像文件并按时间排序

        Returns:
            sorted list of Path objects
        """
        # 查找所有PNG文件
        image_files = list(self.data_dir.glob('*.png'))

        if len(image_files) == 0:
            raise ValueError(f"No PNG images found in {self.data_dir}")

        # 按文件名排序（时间戳格式保证了字典序 = 时间序）
        image_files = sorted(image_files)

        print(f"   Found {len(image_files)} PNG images")
        print(f"   First: {image_files[0].name}")
        print(f"   Last: {image_files[-1].name}")

        return image_files

    def _split_dataset(self, train_ratio, val_ratio):
        """
        将图像序列划分为train/val/test

        策略：按时间顺序划分（保证时间连续性）
        - train: 前70%
        - val: 中间15%
        - test: 最后15%
        """
        total_images = len(self.image_files)

        # 计算可用的样本数（需要history_len+1张图像构成一个样本）
        max_samples = total_images - self.history_len

        if max_samples <= 0:
            raise ValueError(
                f"Not enough images! Need at least {self.history_len + 1}, "
                f"but only have {total_images}"
            )

        # 按时间顺序划分
        train_end = int(total_images * train_ratio)
        val_end = int(total_images * (train_ratio + val_ratio))

        if self.split == 'train':
            # 训练集：索引 0 到 train_end
            self.start_idx = 0
            self.end_idx = train_end

        elif self.split == 'val':
            # 验证集：索引 train_end 到 val_end
            self.start_idx = train_end
            self.end_idx = val_end

        elif self.split == 'test':
            # 测试集：索引 val_end 到 最后
            self.start_idx = val_end
            self.end_idx = total_images

        else:
            raise ValueError(f"Invalid split: {self.split}")

        # 计算该split的样本数
        self.num_samples = max(0, self.end_idx - self.start_idx - self.history_len)

        if self.num_samples == 0:
            raise ValueError(
                f"No samples available for {self.split} split!\n"
                f"  Start idx: {self.start_idx}\n"
                f"  End idx: {self.end_idx}\n"
                f"  History len: {self.history_len}\n"
                f"  Try adjusting train_ratio and val_ratio."
            )

        print(f"   [{self.split.upper()}] Image range: [{self.start_idx}, {self.end_idx})")
        print(f"   [{self.split.upper()}] Samples: {self.num_samples}")

    def _detect_image_size(self):
        """检测图像尺寸（从第一张图像）"""
        first_image = self._load_single_image(self.image_files[self.start_idx])
        self.img_height, self.img_width = first_image.shape

    def _load_single_image(self, path):
        """
        加载单张PNG图像

        Args:
            path: Path object

        Returns:
            numpy array [H, W], dtype=float32, range=[0, 1]
        """
        # 使用PIL加载图像
        img = Image.open(path).convert('L')  # 转为灰度图
        img = np.array(img, dtype=np.float32)

        # 归一化到[0, 1]（如果原始是0-255）
        if img.max() > 1.0:
            img = img / 255.0

        return img

    def _apply_augmentation(self, image):
        """
        应用数据增强（只对训练集）

        Args:
            image: [H, W] numpy array, range=[0, 1]

        Returns:
            augmented image: [H, W] numpy array
        """
        if not self.augmentation or self.split != 'train':
            return image

        img = torch.FloatTensor(image)

        # 1. 随机噪声
        if 'random_noise' in self.augmentation and random.random() < 0.5:
            noise_std = self.augmentation['random_noise']
            noise = torch.randn_like(img) * noise_std
            img = img + noise
            img = torch.clamp(img, 0, 1)

        # 2. 随机缩放
        if 'random_scale' in self.augmentation and random.random() < 0.5:
            scale_range = self.augmentation['random_scale']
            scale = 1 + random.uniform(-scale_range, scale_range)

            h, w = img.shape
            new_h, new_w = int(h * scale), int(w * scale)

            img = img.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            img = torch.nn.functional.interpolate(
                img, size=(new_h, new_w), mode='bilinear', align_corners=False
            )

            # 裁剪或填充
            if scale > 1:
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                img = img[:, :, start_h:start_h + h, start_w:start_w + w]
            else:
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                img = torch.nn.functional.pad(
                    img, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0
                )
                img = img[:, :, :h, :w]

            img = img.squeeze(0).squeeze(0)

        # 3. 随机平移
        if 'random_shift' in self.augmentation and random.random() < 0.5:
            max_shift = self.augmentation['random_shift']
            shift_h = random.randint(-max_shift, max_shift)
            shift_w = random.randint(-max_shift, max_shift)

            img = img.unsqueeze(0).unsqueeze(0)
            img = torch.roll(img, shifts=(shift_h, shift_w), dims=(2, 3))
            img = img.squeeze(0).squeeze(0)

        return img.numpy()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        获取一个样本

        Args:
            idx: 样本索引（相对于当前split）

        Returns:
            {
                'history': [T, 1, H, W] 历史序列
                'target': [1, H, W] 目标图像
                'idx': int 样本索引
                'timestamps': list 时间戳列表
            }
        """
        # 转换为全局图像索引
        global_idx = self.start_idx + idx

        # 读取连续的 history_len + 1 张图像
        images = []
        timestamps = []

        for i in range(self.history_len + 1):
            img_path = self.image_files[global_idx + i]
            img = self._load_single_image(img_path)
            images.append(img)
            timestamps.append(img_path.stem)  # 文件名（不含扩展名）

        images = np.stack(images, axis=0)  # [T+1, H, W]

        # 分离历史和目标
        history = images[:self.history_len]  # [T, H, W]
        target = images[self.history_len]  # [H, W]

        # 数据增强（只对训练集）
        if self.augmentation and self.split == 'train':
            # 对历史序列的每一帧应用增强
            history_aug = []
            for t in range(self.history_len):
                history_aug.append(self._apply_augmentation(history[t]))
            history = np.stack(history_aug, axis=0)

            # 目标也应用增强
            target = self._apply_augmentation(target)

        # Resize（如果需要）
        if self.img_height != self.image_size or self.img_width != self.image_size:
            # History
            history = torch.FloatTensor(history).unsqueeze(1)  # [T, 1, H, W]
            history = torch.nn.functional.interpolate(
                history,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(1).numpy()  # [T, H, W]

            # Target
            target = torch.FloatTensor(target).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            target = torch.nn.functional.interpolate(
                target,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            ).squeeze().numpy()  # [H, W]

        # 转为Tensor
        history = torch.FloatTensor(history).unsqueeze(1)  # [T, 1, H, W]
        target = torch.FloatTensor(target).unsqueeze(0)  # [1, H, W]

        # 归一化到[-1, 1]（扩散模型需要）
        if self.normalize:
            history = history * 2 - 1
            target = target * 2 - 1

        # 返回结果
        result = {
            'history': history,
            'target': target,
            'idx': idx,
            'timestamps': timestamps  # 包含时间戳信息
        }

        return result


def create_dataloaders(config, is_distributed=False, rank=0, world_size=1, num_workers=None):
    """
    创建数据加载器

    Args:
        config: 配置对象
        is_distributed: 是否分布式训练
        rank: 当前进程rank
        world_size: 总进程数
        num_workers: DataLoader worker数量

    Returns:
        {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    # 提取配置
    data_dir = config.data.data_dir
    history_len = config.data.history_len
    image_size = config.data.image_size
    normalize = config.data.normalize
    augmentation = config.data.get('augmentation', None)
    batch_size = config.training.batch_size

    # 数据集划分比例
    train_ratio = config.data.get('train_ratio', 0.7)
    val_ratio = config.data.get('val_ratio', 0.15)

    if num_workers is None:
        num_workers = config.hardware.get('num_workers', 4)

    print(f"\n{'=' * 80}")
    print(f"[Rank {rank}] Creating datasets from PNG images")
    print(f"{'=' * 80}")
    print(f"  Data dir: {data_dir}")
    print(f"  History length: {history_len}")
    print(f"  Image size: {image_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Train/Val/Test ratio: {train_ratio:.1%}/{val_ratio:.1%}/{1 - train_ratio - val_ratio:.1%}")

    loaders = {}

    # ===== 训练集 =====
    try:
        train_dataset = TrafficImageDataset(
            data_dir=data_dir,
            split='train',
            history_len=history_len,
            image_size=image_size,
            normalize=normalize,
            augmentation=augmentation,
            train_ratio=train_ratio,
            val_ratio=val_ratio
        )

        # 分布式采样器
        if is_distributed:
            from torch.utils.data.distributed import DistributedSampler
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True
            )
            shuffle = False
        else:
            train_sampler = None
            shuffle = True

        loaders['train'] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if num_workers > 0 else False
        )

        print(f"\n✅ [Rank {rank}] Train loader created: {len(train_dataset)} samples")

    except Exception as e:
        print(f"\n❌ [Rank {rank}] Failed to create train dataset: {e}")
        raise

    # ===== 验证集 =====
    try:
        val_dataset = TrafficImageDataset(
            data_dir=data_dir,
            split='val',
            history_len=history_len,
            image_size=image_size,
            normalize=normalize,
            augmentation=None,  # 验证集不使用数据增强
            train_ratio=train_ratio,
            val_ratio=val_ratio
        )

        if is_distributed:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False
            )
        else:
            val_sampler = None

        loaders['val'] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )

        print(f"✅ [Rank {rank}] Val loader created: {len(val_dataset)} samples")

    except Exception as e:
        print(f"⚠️ [Rank {rank}] Failed to create val dataset: {e}")

    # ===== 测试集 =====
    try:
        test_dataset = TrafficImageDataset(
            data_dir=data_dir,
            split='test',
            history_len=history_len,
            image_size=image_size,
            normalize=normalize,
            augmentation=None,
            train_ratio=train_ratio,
            val_ratio=val_ratio
        )

        if is_distributed:
            test_sampler = DistributedSampler(
                test_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False
            )
        else:
            test_sampler = None

        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=test_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )

        print(f"✅ [Rank {rank}] Test loader created: {len(test_dataset)} samples")

    except Exception as e:
        print(f"⚠️ [Rank {rank}] Failed to create test dataset: {e}")

    print(f"{'=' * 80}\n")

    return loaders
