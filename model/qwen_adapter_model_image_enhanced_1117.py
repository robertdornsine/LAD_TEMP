"""
Architecture:
    Grayscale Images [B, 16, 1, H, W]
    ↓ Pretrained Vision Encoder (CLIP/Qwen-VL)
    ↓ Visual Features [B, 16, num_patches, visual_dim]
    ↓ Temporal Aggregator (Q-Former)
    ↓ Aggregated Features [B, 32, visual_dim]
    ↓ Projection → Qwen Space
    ↓ Qwen Tokens [B, 32, 4096]
    ↓ Qwen (Frozen) + Enhanced Adapter
    ↓ LLM Features [B, 32, 4096]
    ↓ Multi-Scale Condition Encoder
    ↓ Condition (Single or Multi-Scale)
    ↓ Enhanced Diffusion UNet
    ↓ Predicted Traffic Image [B, 1, H, W]
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from diffusers import DDPMScheduler, DDIMScheduler
from typing import Dict, Optional, Union
import logging
import torch.nn.functional as F
import pathlib as Path
import yaml

from .vision_encoder_enhanced_1117 import VisionEncoder
from .adapter_enhanced_1117 import (
    MultiLayerAdapterV2,
    create_adapter_from_config,
    load_legacy_adapter_weights
)
from .diffusion_unet_enhanced_1117 import ImprovedDiffusionUNet
from .condition_encoder_enhanced_1117 import ConditionEncoder, ConditionalDropout



# 主模型
class ImprovedQwenTrafficDiffusion(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config  # 保存config
        self.image_size = config.image_size
        self.history_len = config.history_len

        # 条件配置
        condition_type = config.get('condition_type', 'dual')
        self.condition_type = condition_type

        if condition_type == 'dual':
            self.global_condition_dim = config.get('global_condition_dim', 512)
            self.sequence_condition_dim = config.get('sequence_condition_dim', 512)
            logging.info("Using Dual Condition Mode (global + sequence)")
        else:
            # 兼容旧的multiscale模式
            self.global_condition_dim = config.get('global_condition_dim', 256)
            self.local_condition_dim = config.get('local_condition_dim', 512)
            self.temporal_condition_dim = config.get('temporal_condition_dim', 256)
            logging.info("Using MultiScale Condition Mode (legacy)")

        # 扩散配置
        self.use_full_training = config.get('use_full_training', True)
        self.ddpm_train_steps = config.get('ddpm_train_steps', 20)
        self.ddpm_val_steps = config.get('ddpm_val_steps', 50)
        self.num_diffusion_steps = config.get('num_diffusion_steps', 1000)
        self.num_inference_steps = config.get('num_inference_steps', 50)

        # CFG配置
        self.use_cfg = config.get('use_cfg', True)
        self.cfg_dropout_prob = config.get('cfg_dropout_prob', 0.15)
        self.cfg_scale = config.get('cfg_scale', 2.5)

        # Prediction Type
        self.prediction_type = config.get('prediction_type', 'v_prediction')
        logging.info(f"✅ Prediction type: {self.prediction_type}")

        # 损失配置
        self.use_weighted_mse = config.get('use_weighted_mse', True)
        self.low_traffic_threshold = config.get('low_traffic_threshold', 0.15)
        self.mid_traffic_threshold = config.get('mid_traffic_threshold', 0.5)
        self.low_traffic_weight = config.get('low_traffic_weight', 4.0)
        self.mid_traffic_weight = config.get('mid_traffic_weight', 2.5)
        self.high_traffic_weight = config.get('high_traffic_weight', 2.0)

        self.use_multiscale_loss = config.get('use_multiscale_loss', True)
        self.multiscale_scales = config.get('multiscale_scales', [1, 2])
        self.multiscale_weights = config.get('multiscale_weights', [1.0, 0.3])

        self.use_gradient_loss = config.get('use_gradient_loss', True)
        self.gradient_loss_weight = config.get('gradient_loss_weight', 0.15)

        # 1. 视觉编码器
        print("\n[1/7] Initializing Improved Vision Encoder...")
        self.vision_encoder = VisionEncoder(config)
        self.visual_dim = self.vision_encoder.visual_dim
        print(f"[OK] Vision Encoder: output_dim={self.visual_dim}")

        # 2. Qwen模型
        print("\n[2/7] Loading Qwen Model...")
        qwen_model_name = config.get('qwen_model_name')
        print(f"   Model: {qwen_model_name}")

        try:
            self.qwen = AutoModelForCausalLM.from_pretrained(
                qwen_model_name,
                torch_dtype=torch.bfloat16,
                device_map=None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        except Exception as e:
            logging.warning(f"Failed to load Qwen: {e}")
            logging.warning("Creating from config...")
            qwen_config = AutoConfig.from_pretrained(
                qwen_model_name,
                trust_remote_code=True
            )
            self.qwen = AutoModelForCausalLM.from_config(
                qwen_config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )

        # 冻结Qwen
        freeze_layers = config.get('qwen_freeze_layers', None)
        if freeze_layers is not None:
            # 部分冻结
            total_layers = self.qwen.config.num_hidden_layers
            for i, layer in enumerate(self.qwen.model.layers):
                if i in freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
            logging.info(f"✅ Qwen partially frozen: layers {freeze_layers}")
        else:
            # 全部冻结
            for param in self.qwen.parameters():
                param.requires_grad = False
            logging.info("✅ Qwen fully frozen")

        self.qwen_hidden_size = self.qwen.config.hidden_size
        self.qwen_num_layers = self.qwen.config.num_hidden_layers

        print(f"   Hidden size: {self.qwen_hidden_size}")
        print(f"   Num layers: {self.qwen_num_layers}")

        # 3. 视觉→Qwen投影
        print("\n[3/7] Initializing Visual-to-Qwen Projection...")
        self.visual_to_qwen = nn.Sequential(
            nn.LayerNorm(self.visual_dim),
            nn.Linear(self.visual_dim, self.qwen_hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(self.qwen_hidden_size // 2, self.qwen_hidden_size),
            nn.LayerNorm(self.qwen_hidden_size)
        )

        # Xavier初始化
        for module in self.visual_to_qwen.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        print(f"[OK] Projection: {self.visual_dim} → {self.qwen_hidden_size}")

        # 4. Adapter
        print("\n[4/7] Initializing Enhanced Adapter V2...")

        # adapter配置
        adapter_config = {
            'num_layers': self.qwen_num_layers,
            'hidden_size': self.qwen_hidden_size,
            'adapter_size': config.get('adapter_size', 128),  # 默认128
            'adapter_layers': config.get('adapter_layers', 'all'),
            'dropout': config.get('dropout', 0.1),
            'use_conv': config.get('adapter_use_conv', True),
            'use_gate': config.get('adapter_use_gate', True),
            'use_attention': config.get('adapter_use_attention', False),  # 默认关闭
            'conv_kernel_sizes': config.get('adapter_conv_kernels', [3, 5]),
            'use_layernorm': config.get('adapter_use_layernorm', True),
            'use_gradient_checkpointing': config.get('adapter_gradient_checkpointing', False)
        }

        self.adapter = create_adapter_from_config(adapter_config)

        print(f"[OK] Adapter V2: {adapter_config['adapter_layers']} layers")
        print(f"     Features: conv={adapter_config['use_conv']}, "
              f"gate={adapter_config['use_gate']}, "
              f"attn={adapter_config['use_attention']}")

        # 条件编码器
        print("\n[5/7] Initializing Improved Condition Encoder...")
        self.condition_encoder = ConditionEncoder(config)

        if condition_type == 'dual':
            print(f"[OK] Dual Condition Encoder:")
            print(f"     Global:   {self.global_condition_dim}")
            print(f"     Sequence: {self.sequence_condition_dim}")
        else:
            print(f"[OK] MultiScale Condition Encoder (legacy)")

        # 6. 条件Dropout（用于CFG)
        self.conditional_dropout = ConditionalDropout(
            dropout_prob=self.cfg_dropout_prob
        )

        # 7. 改进的UNet
        print("\n[6/7] Initializing Improved Diffusion UNet...")

        # 传递条件维度到UNet
        unet_config = dict(config)
        unet_config['condition_type'] = condition_type

        if condition_type == 'dual':
            unet_config['global_condition_dim'] = self.global_condition_dim
            unet_config['sequence_condition_dim'] = self.sequence_condition_dim

        self.diffusion_unet = ImprovedDiffusionUNet(unet_config)
        print(f"[OK] UNet initialized for {self.image_size}×{self.image_size}")

        # 8. 噪声调度器
        print("\n[7/7] Initializing Noise Scheduler...")
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_steps,
            beta_schedule=config.get('beta_schedule', 'squaredcos_cap_v2'),
            prediction_type=self.prediction_type,
            beta_start=0.0001,
            beta_end=0.02,
            clip_sample=False
        )
        print(f"[OK] Scheduler: {self.num_diffusion_steps} steps, {config.get('beta_schedule', 'cosine')} schedule")

        # 9. Null条件向量（CFG用）
        if condition_type == 'dual':
            self.register_buffer(
                'null_global_condition',
                torch.zeros(1, self.global_condition_dim)
            )
            self.register_buffer(
                'null_sequence_condition',
                torch.zeros(1, config.get('num_output_tokens', 64), self.sequence_condition_dim)
            )
        else:
            # 兼容
            self.register_buffer(
                'null_condition',
                torch.zeros(1, 512)
            )

        # 统计信息
        self._print_model_summary()

    def _print_model_summary(self):
        print("\n" + "=" * 80)
        print(" Model Summary")
        print("=" * 80)

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        print(f"Total Parameters:     {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,} ({trainable_params / total_params * 100:.2f}%)")
        print(f"Frozen Parameters:    {frozen_params:,} ({frozen_params / total_params * 100:.2f}%)")

        print("\nComponent Breakdown:")
        print("-" * 80)

        components = {
            'Vision Encoder': self.vision_encoder,
            'Visual→Qwen Projection': self.visual_to_qwen,
            'Qwen (Frozen)': self.qwen,
            'Adapter': self.adapter,
            'Condition Encoder': self.condition_encoder,
            'Diffusion UNet': self.diffusion_unet
        }

        for name, module in components.items():
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)

            if total > 0:
                trainable_pct = trainable / total * 100
                print(f"{name:30s}: {total:12,} params ({trainable:12,} trainable, {trainable_pct:5.1f}%)")

        print("=" * 80 + "\n")

    # 损失函数
    def compute_weighted_mse(self, pred, target):
        weights = torch.ones_like(target)

        weights[target < self.low_traffic_threshold] = self.low_traffic_weight
        weights[(target >= self.low_traffic_threshold) &
                (target < self.mid_traffic_threshold)] = self.mid_traffic_weight
        weights[target >= self.mid_traffic_threshold] = self.high_traffic_weight

        squared_error = (pred - target) ** 2
        return (weights * squared_error).mean()

    def compute_multiscale_loss(self, pred, target):
        total_loss = 0.0

        for scale, weight in zip(self.multiscale_scales, self.multiscale_weights):
            if scale == 1:
                if self.use_weighted_mse:
                    loss = self.compute_weighted_mse(pred, target)
                else:
                    loss = F.mse_loss(pred, target)
            else:
                pred_down = F.avg_pool2d(pred, kernel_size=scale, stride=scale)
                target_down = F.avg_pool2d(target, kernel_size=scale, stride=scale)

                if self.use_weighted_mse:
                    loss = self.compute_weighted_mse(pred_down, target_down)
                else:
                    loss = F.mse_loss(pred_down, target_down)

            total_loss += weight * loss

        return total_loss

    def compute_gradient_loss(self, pred, target):
        # Sobel算子
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=pred.dtype,
            device=pred.device
        ).view(1, 1, 3, 3)

        sobel_y = sobel_x.transpose(2, 3)

        pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, sobel_y, padding=1)

        target_grad_x = F.conv2d(target, sobel_x, padding=1)
        target_grad_y = F.conv2d(target, sobel_y, padding=1)

        loss = F.mse_loss(pred_grad_x, target_grad_x) + \
               F.mse_loss(pred_grad_y, target_grad_y)

        return loss

    # 前向传播（核心）
    def forward(
            self,
            images: torch.Tensor,
            target: Optional[torch.Tensor] = None,
            mode: str = 'train'
    ) -> Dict[str, torch.Tensor]:

        B, T, C, H, W = images.shape

        # 视觉编码
        aggregated_features = self.vision_encoder(images)  # [B, 64, 512]

        # 投影到Qwen空间
        qwen_tokens = self.visual_to_qwen(aggregated_features)  # [B, 64, 896]

        # Qwen处理
        with torch.no_grad():
            qwen_tokens_bf16 = qwen_tokens.to(self.qwen.dtype)
            qwen_outputs = self.qwen(
                inputs_embeds=qwen_tokens_bf16,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False
            )
            all_hidden_states = qwen_outputs.hidden_states

        # 转回float32
        all_hidden_states = [h.float() for h in all_hidden_states]

        # Adapter处理
        adapted_states = []
        for layer_idx, hidden_state in enumerate(all_hidden_states):
            adapted_state = self.adapter(hidden_state, layer_idx=layer_idx)
            adapted_states.append(adapted_state)

        llm_features = adapted_states[-1]  # [B, 64, 896]

        # 条件编码
        condition_dict = self.condition_encoder(llm_features)

        # Step 6: CFG Dropout
        if self.training and self.use_cfg:
            condition_dict = self.conditional_dropout(condition_dict)

        # Step 7: 扩散模型
        if mode == 'train' or mode == 'val':
            if self.use_full_training:
                # 完整扩散训练模式
                return self._full_diffusion_forward(
                    condition_dict, target, images, mode
                )
            else:
                # 简化训练模式
                return self._simplified_forward(
                    condition_dict, target, images, mode
                )
        else:
            # 推理模式
            return self._inference_forward(condition_dict, H, W)

    def _simplified_forward(
            self,
            condition_dict: Dict[str, torch.Tensor],
            target: torch.Tensor,
            images: torch.Tensor,
            mode: str
    ) -> Dict[str, torch.Tensor]:

        B = images.size(0)
        device = images.device

        if target.dim() == 5:
            target = target.squeeze(1)

        if mode == 'train':
            # 随机采样时间步
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (B,),
                device=device,
                dtype=torch.long
            )

            # 添加噪声
            noise = torch.randn_like(target)
            noisy_target = self.noise_scheduler.add_noise(target, noise, timesteps)

            # UNet预测
            pred_output = self.diffusion_unet(
                noisy_target,
                timesteps,
                condition_dict
            )

            # 计算目标
            if self.prediction_type == 'epsilon':
                target_output = noise
            elif self.prediction_type == 'v_prediction':
                alpha_prod_t = self.noise_scheduler.alphas_cumprod[timesteps]
                alpha_prod_t = alpha_prod_t.view(-1, 1, 1, 1)

                sqrt_alpha_prod = alpha_prod_t ** 0.5
                sqrt_one_minus_alpha_prod = (1 - alpha_prod_t) ** 0.5

                target_output = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * target
            else:
                target_output = noise

            # 损失
            loss = F.mse_loss(pred_output, target_output)

            return {
                'loss': loss,
                'pred': target,  # 训练时返回原始target（用于监控）
                'condition': condition_dict
            }

        else:
            # 验证模式：使用DDPM采样
            prediction = self.ddpm_sample_with_cfg(
                condition_dict,
                num_steps=self.ddpm_val_steps,
                height=target.size(2),
                width=target.size(3),
                cfg_scale=self.cfg_scale
            )

            loss = F.mse_loss(prediction, target)

            return {
                'loss': loss,
                'pred': prediction.squeeze(1),
                'condition': condition_dict
            }

    def _full_diffusion_forward(
            self,
            condition_dict: Dict[str, torch.Tensor],
            target: torch.Tensor,
            images: torch.Tensor,
            mode: str
    ) -> Dict[str, torch.Tensor]:

        B = images.size(0)
        H, W = target.size(-2), target.size(-1)

        if target.dim() == 5:
            target = target.squeeze(1)

        # DDPM采样
        num_steps = self.ddpm_train_steps if mode == 'train' else self.ddpm_val_steps
        cfg_scale = 1.0 if mode == 'train' else self.cfg_scale

        prediction = self.ddpm_sample_with_cfg(
            condition_dict,
            num_steps=num_steps,
            height=H,
            width=W,
            cfg_scale=cfg_scale
        )

        # 计算损失
        if mode == 'train':
            losses = {}

            # 主损失
            if self.use_multiscale_loss:
                main_loss = self.compute_multiscale_loss(prediction, target)
                losses['multiscale'] = main_loss.item()
            elif self.use_weighted_mse:
                main_loss = self.compute_weighted_mse(prediction, target)
                losses['weighted_mse'] = main_loss.item()
            else:
                main_loss = F.mse_loss(prediction, target)
                losses['mse'] = main_loss.item()

            # 梯度损失
            if self.use_gradient_loss:
                grad_loss = self.compute_gradient_loss(prediction, target)
                losses['gradient'] = grad_loss.item()
                main_loss = main_loss + self.gradient_loss_weight * grad_loss

            losses['total'] = main_loss.item()

            return {
                'loss': main_loss,
                'pred': prediction.squeeze(1),
                'condition': condition_dict,
                **losses
            }
        else:
            # 验证模式
            loss = F.mse_loss(prediction, target)

            return {
                'loss': loss,
                'pred': prediction,
                'condition': condition_dict,
                'val_mse': loss.item()
            }

    def _inference_forward(
            self,
            condition_dict: Dict[str, torch.Tensor],
            height: int,
            width: int
    ) -> Dict[str, torch.Tensor]:

        prediction = self.ddpm_sample_with_cfg(
            condition_dict,
            num_steps=self.num_inference_steps,
            height=height,
            width=width,
            cfg_scale=self.cfg_scale
        )

        return {
            'prediction': prediction,
            'pred': prediction.squeeze(1),
            'condition': condition_dict
        }


    # DDPM采样（支持CFG）
    def ddpm_sample_with_cfg(
            self,
            condition_dict: Dict[str, torch.Tensor],
            num_steps: int,
            height: int,
            width: int,
            cfg_scale: float = 1.0,
            generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:

        B = condition_dict['global'].size(0)
        device = condition_dict['global'].device

        # 初始噪声
        x = torch.randn(
            (B, 1, height, width),
            device=device,
            dtype=torch.float32,
            generator=generator
        )

        # 设置采样时间步
        self.noise_scheduler.set_timesteps(num_steps, device=device)

        # 准备null条件（用于CFG）
        if cfg_scale > 1.0:
            null_condition_dict = {
                'global': self.null_global_condition.expand(B, -1),
                'sequence': self.null_sequence_condition.expand(B, -1, -1)
            }

        # 逐步去噪
        for t in self.noise_scheduler.timesteps:
            t_batch = t.unsqueeze(0).repeat(B).to(device)

            if cfg_scale > 1.0:
                # 预测条件和无条件
                noise_uncond = self.diffusion_unet(x, t_batch, null_condition_dict)
                noise_cond = self.diffusion_unet(x, t_batch, condition_dict)

                # 线性组合
                pred_noise = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
            else:
                # 无CFG
                pred_noise = self.diffusion_unet(x, t_batch, condition_dict)

            # 去噪一步
            x = self.noise_scheduler.step(
                pred_noise,
                t,
                x,
                generator=generator
            ).prev_sample

        return x


    # DDIM采样（更快）
    @torch.no_grad()
    def ddim_sample(
            self,
            condition_dict: Dict[str, torch.Tensor],
            num_steps: int,
            height: int,
            width: int,
            eta: float = 0.0,
            generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:

        B = condition_dict['global'].size(0)
        device = condition_dict['global'].device

        # 创建DDIM调度器
        ddim_scheduler = DDIMScheduler(
            num_train_timesteps=self.noise_scheduler.config.num_train_timesteps,
            beta_schedule=self.noise_scheduler.config.beta_schedule,
            prediction_type="epsilon"
        )
        ddim_scheduler.set_timesteps(num_steps, device=device)

        # 初始噪声
        x = torch.randn(
            (B, 1, height, width),
            device=device,
            dtype=torch.float32,
            generator=generator
        )

        # 逐步去噪
        for t in ddim_scheduler.timesteps:
            t_batch = t.unsqueeze(0).repeat(B).to(device)

            pred_noise = self.diffusion_unet(x, t_batch, condition_dict)

            x = ddim_scheduler.step(
                pred_noise,
                t,
                x,
                eta=eta,
                generator=generator
            ).prev_sample

        return x


    # 高级生成接口
    @torch.no_grad()
    def generate(
            self,
            images: torch.Tensor,
            num_samples: int = 1,
            num_steps: int = 50,
            sampler: str = 'ddpm',
            cfg_scale: Optional[float] = None,
            eta: float = 0.0,
            generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:

        self.eval()

        B, T, C, H, W = images.shape

        # 提取条件
        outputs = self.forward(images, mode='inference')
        condition_dict = outputs['condition']

        # 指定cfg_scale
        if cfg_scale is None:
            cfg_scale = self.cfg_scale

        all_predictions = []

        for _ in range(num_samples):
            if sampler == 'ddpm':
                pred = self.ddpm_sample_with_cfg(
                    condition_dict,
                    num_steps=num_steps,
                    height=H,
                    width=W,
                    cfg_scale=cfg_scale,
                    generator=generator
                )
            elif sampler == 'ddim':
                pred = self.ddim_sample(
                    condition_dict,
                    num_steps=num_steps,
                    height=H,
                    width=W,
                    eta=eta,
                    generator=generator
                )
            else:
                raise ValueError(f"Unknown sampler: {sampler}")

            all_predictions.append(pred.squeeze(1))

        # [B, num_samples, H, W]
        predictions = torch.stack(all_predictions, dim=1)

        return predictions

    # 模型保存和加载
    def save_pretrained(self, save_path: str):
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # 保存可训练权重
        trainable_state = {
            'vision_encoder': self.vision_encoder.state_dict(),
            'visual_to_qwen': self.visual_to_qwen.state_dict(),
            'adapter': self.adapter.state_dict(),
            'condition_encoder': self.condition_encoder.state_dict(),
            'diffusion_unet': self.diffusion_unet.state_dict(),
            'config': self.config
        }

        torch.save(trainable_state, save_path / 'trainable_weights.pt')

        # 保存配置
        import yaml
        with open(save_path / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)

        print(f"Model saved to {save_path}")

    def load_pretrained(self, load_path: str, strict: bool = True):
        load_path = Path(load_path)

        checkpoint = torch.load(
            load_path / 'trainable_weights.pt',
            map_location='cpu'
        )

        # 加载各组件
        self.vision_encoder.load_state_dict(checkpoint['vision_encoder'], strict=strict)
        self.visual_to_qwen.load_state_dict(checkpoint['visual_to_qwen'], strict=strict)
        self.adapter.load_state_dict(checkpoint['adapter'], strict=strict)
        self.condition_encoder.load_state_dict(checkpoint['condition_encoder'], strict=strict)
        self.diffusion_unet.load_state_dict(checkpoint['diffusion_unet'], strict=strict)

        print(f"Model loaded from {load_path}")

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# 辅助函数
def create_model_from_config(config_path: str, device: str = 'cuda') -> ImprovedQwenTrafficDiffusion:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model = ImprovedQwenTrafficDiffusion(config['model'])
    model = model.to(device)

    return model