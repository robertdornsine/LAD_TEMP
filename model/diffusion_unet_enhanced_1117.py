import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple

# 自适应层数配置
def get_channel_mult_for_image_size(image_size):

    max_levels = int(math.log2(image_size)) - 1

    if image_size <= 12:
        # 2次下采样
        channel_mult = [1, 2, 4]
        attention_resolutions = [3]  # 只在3×3用attention

    elif image_size <= 16:
        # 2次下采样
        channel_mult = [1, 2, 4]
        attention_resolutions = [4]

    elif image_size <= 32:
        # 3次下采样
        channel_mult = [1, 2, 4, 4]
        attention_resolutions = [8, 4]

    elif image_size <= 64:
        # 4次下采样
        channel_mult = [1, 2, 4, 8, 8]
        attention_resolutions = [16, 8]

    else:
        # 更大的图像
        channel_mult = [1, 2, 4, 8, 16]
        attention_resolutions = [32, 16, 8]

    return channel_mult, attention_resolutions


# 时间步Embedding增强
class TimestepEmbedding(nn.Module):

    def __init__(self, channels, time_embed_dim, max_period=10000):
        super().__init__()
        self.channels = channels
        self.max_period = max_period

        # 投影层
        self.time_embed = nn.Sequential(
            nn.Linear(channels, time_embed_dim),
            nn.SiLU(),  # Swish激活
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        # Sinusoidal编码
        half = self.channels // 2
        freqs = torch.exp(
            -math.log(self.max_period) *
            torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
        )

        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if self.channels % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        # 投影
        embedding = self.time_embed(embedding)

        return embedding


# 自适应组归一化（AdaGN）
class AdaptiveGroupNorm(nn.Module):

    def __init__(self, num_channels, num_groups=32, condition_dim=512):
        super().__init__()
        self.num_channels = num_channels
        self.num_groups = min(num_groups, num_channels)  # 确保可以整除

        # GroupNorm
        self.norm = nn.GroupNorm(self.num_groups, num_channels, affine=False)

        # 条件投影
        self.condition_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, num_channels * 2)  # *2 for scale and shift
        )

    def forward(self, x, condition):
        # GroupNorm
        x_norm = self.norm(x)

        # 生成scale和shift
        condition_params = self.condition_proj(condition)  # [B, 2C]
        scale, shift = condition_params.chunk(2, dim=1)  # [B, C], [B, C]

        # 调制
        scale = scale[:, :, None, None]  # [B, C, 1, 1]
        shift = shift[:, :, None, None]

        return x_norm * (1 + scale) + shift


# 残差块增强
class ResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        time_embed_dim,
        global_condition_dim=512,
        dropout=0.1,
        use_scale_shift_norm=True
    ):
        super().__init__()
        self.use_scale_shift_norm = use_scale_shift_norm

        # 第一个卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # 时间步投影
        self.time_proj = nn.Linear(time_embed_dim, out_channels * 2 if use_scale_shift_norm else out_channels)

        # 归一化
        if use_scale_shift_norm:
            self.norm1 = AdaptiveGroupNorm(in_channels, condition_dim=global_condition_dim)
            self.norm2 = nn.GroupNorm(32, out_channels)
        else:
            self.norm1 = nn.GroupNorm(32, in_channels)
            self.norm2 = nn.GroupNorm(32, out_channels)

        # 第二个卷积
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, time_emb, global_cond=None):
        h = x

        # 第一个卷积块
        if self.use_scale_shift_norm and global_cond is not None:
            # 使用AdaGN
            h = self.norm1(h, global_cond)
        else:
            h = self.norm1(h)

        h = F.silu(h)
        h = self.conv1(h)

        # 添加时间步
        time_out = self.time_proj(F.silu(time_emb))

        if self.use_scale_shift_norm:
            scale, shift = time_out.chunk(2, dim=1)
            h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        else:
            h = h + time_out[:, :, None, None]

        # 第二个卷积块
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # 残差连接
        return h + self.shortcut(x)


# Cross-Attention增强
class CrossAttentionBlock(nn.Module):

    def __init__(
        self,
        channels,
        sequence_dim=512,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()
        self.channels = channels

        # Query投影
        self.norm_feat = nn.GroupNorm(32, channels)
        self.to_q = nn.Linear(channels, channels)

        # Key/Value投影（从sequence条件）
        self.norm_seq = nn.LayerNorm(sequence_dim)
        self.to_k = nn.Linear(sequence_dim, channels)
        self.to_v = nn.Linear(sequence_dim, channels)

        # Multi-head attention
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        # 输出投影
        self.to_out = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Dropout(dropout)
        )

    def forward(self, x, sequence_cond):
        B, C, H, W = x.shape

        # 转换为序列格式
        h = self.norm_feat(x)
        h = h.view(B, C, H * W).transpose(1, 2)  # [B, HW, C]

        # 生成Q/K/V
        q = self.to_q(h)  # [B, HW, C]

        sequence_cond = self.norm_seq(sequence_cond)  # [B, L, D]
        k = self.to_k(sequence_cond)  # [B, L, C]
        v = self.to_v(sequence_cond)  # [B, L, C]

        # Reshape for multi-head
        q = q.view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)  # [B, nh, HW, hd]
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)      # [B, nh, L, hd]
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)      # [B, nh, L, hd]

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, nh, HW, L]
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # [B, nh, HW, hd]

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, H * W, C)  # [B, HW, C]
        out = self.to_out(out)

        # 恢复空间维度
        out = out.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]

        # 残差连接
        return x + out


# Self-Attention块（Self-Attention）
class AttentionBlock(nn.Module):

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj_out = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.view(B, C, H * W)

        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        # Multi-head
        q = q.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)
        k = k.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)
        v = v.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(2, 3).contiguous().view(B, C, H * W)

        out = self.proj_out(out)
        out = out.view(B, C, H, W)

        return x + out



# 下采样/上采样块
class Downsample(nn.Module):

    def __init__(self, channels, use_conv=True):
        super().__init__()
        if use_conv:
            self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(2)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):

    def __init__(self, channels, use_conv=True):
        super().__init__()
        if use_conv:
            self.op = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(channels, channels, 3, padding=1)
            )
        else:
            self.op = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        return self.op(x)



# 改进的UNet

class ImprovedDiffusionUNet(nn.Module):

    def __init__(self, config):
        super().__init__()

        # 配置参数
        self.image_size = config.get('image_size', 12)
        in_channels = 1  # 灰度图
        out_channels = 1

        # 自动确定层数
        channel_mult, attention_resolutions = get_channel_mult_for_image_size(self.image_size)

        # 允许配置覆盖
        channel_mult = config.get('unet_channel_mult', channel_mult)
        attention_resolutions = config.get('unet_attention_resolutions', attention_resolutions)

        print(f"UNet for {self.image_size}×{self.image_size}: channel_mult={channel_mult}, attention_res={attention_resolutions}")

        base_channels = config.get('unet_base_channels', 128)
        num_res_blocks = config.get('unet_num_res_blocks', 3)
        dropout = config.get('dropout', 0.1)
        use_scale_shift_norm = config.get('unet_use_scale_shift_norm', True)

        # 条件维度
        condition_type = config.get('condition_type', 'dual')
        if condition_type == 'dual':
            global_cond_dim = config.get('global_condition_dim', 512)
            sequence_cond_dim = config.get('sequence_condition_dim', 512)
        else:
            global_cond_dim = config.get('global_condition_dim', 256)
            sequence_cond_dim = config.get('local_condition_dim', 512)

        self.condition_type = condition_type

        # 时间步embedding
        time_embed_dim = base_channels * 4
        self.time_embed = TimestepEmbedding(base_channels, time_embed_dim)

        # 输入层
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        channels_list = [base_channels]  # 用于记录每个层的通道数

        now_channels = base_channels
        current_res = self.image_size

        for level, mult in enumerate(channel_mult):
            out_channels_level = base_channels * mult

            # 该层的残差块
            for i in range(num_res_blocks):
                block = ResidualBlock(
                    now_channels,
                    out_channels_level,
                    time_embed_dim,
                    global_cond_dim,
                    dropout,
                    use_scale_shift_norm
                )
                now_channels = out_channels_level

                # Self-Attention
                if current_res in attention_resolutions:
                    attn = AttentionBlock(now_channels)
                else:
                    attn = None

                # Cross-Attention（用于sequence条件）
                cross_attn = CrossAttentionBlock(now_channels, sequence_cond_dim)

                self.down_blocks.append(nn.ModuleDict({
                    'resblock': block,
                    'attn': attn,
                    'cross_attn': cross_attn
                }))

                channels_list.append(now_channels)

            # 下采样（除了最后一层）
            if level != len(channel_mult) - 1:
                self.down_blocks.append(nn.ModuleDict({
                    'downsample': Downsample(now_channels)
                }))
                channels_list.append(now_channels)
                current_res = current_res // 2

        # Middle
        self.middle_block1 = ResidualBlock(
            now_channels, now_channels, time_embed_dim,
            global_cond_dim, dropout, use_scale_shift_norm
        )
        self.middle_attn = AttentionBlock(now_channels)
        self.middle_cross_attn = CrossAttentionBlock(now_channels, sequence_cond_dim)
        self.middle_block2 = ResidualBlock(
            now_channels, now_channels, time_embed_dim,
            global_cond_dim, dropout, use_scale_shift_norm
        )

        # Decoder
        self.up_blocks = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_mult))):
            out_channels_level = base_channels * mult

            for i in range(num_res_blocks + 1):  # skip connection + 1
                # Skip connection
                skip_channels = channels_list.pop()

                block = ResidualBlock(
                    now_channels + skip_channels,
                    out_channels_level,
                    time_embed_dim,
                    global_cond_dim,
                    dropout,
                    use_scale_shift_norm
                )
                now_channels = out_channels_level

                # Attention
                if current_res in attention_resolutions:
                    attn = AttentionBlock(now_channels)
                else:
                    attn = None

                cross_attn = CrossAttentionBlock(now_channels, sequence_cond_dim)

                self.up_blocks.append(nn.ModuleDict({
                    'resblock': block,
                    'attn': attn,
                    'cross_attn': cross_attn
                }))

            # 上采样（除了最后一层）
            if level != 0:
                self.up_blocks.append(nn.ModuleDict({
                    'upsample': Upsample(now_channels)
                }))
                current_res = current_res * 2

        # ===== 输出层 =====
        self.norm_out = nn.GroupNorm(32, base_channels)
        self.conv_out = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def forward(self, x, timesteps, condition_dict):
        # 提取条件
        if self.condition_type == 'dual':
            global_cond = condition_dict['global']
            sequence_cond = condition_dict['sequence']
        else:
            # 兼容multiscale
            global_cond = condition_dict['global']
            sequence_cond = condition_dict.get('unified', condition_dict.get('local'))

        # 时间步embedding
        time_emb = self.time_embed(timesteps)  # [B, time_dim]

        # 输入
        h = self.conv_in(x)

        # Encoder
        skip_connections = [h]

        for module_dict in self.down_blocks:
            if 'resblock' in module_dict:
                h = module_dict['resblock'](h,time_emb,global_cond)

                if module_dict['attn'] is not None:
                    h = module_dict['attn'](h)

                h = module_dict['cross_attn'](h,sequence_cond)

                skip_connections.append(h)

            elif 'downsample' in module_dict:
                h = module_dict['downsample'](h)
                skip_connections.append(h)

        # Middle
        h = self.middle_block1(h, time_emb, global_cond)
        h = self.middle_attn(h)
        h = self.middle_cross_attn(h, sequence_cond)
        h = self.middle_block2(h, time_emb, global_cond)

        # Decoder
        for module_dict in self.up_blocks:
            if 'resblock' in module_dict:
                # Skip connection
                skip = skip_connections.pop()
                h = torch.cat([h, skip], dim=1)

                h = module_dict['resblock'](h,time_emb,global_cond)

                if module_dict['attn'] is not None:
                    h = module_dict['attn'](h)

                h = module_dict['cross_attn'](h,sequence_cond)

            elif 'upsample' in module_dict:
                h = module_dict['upsample'](h)

        # 输出
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        return h