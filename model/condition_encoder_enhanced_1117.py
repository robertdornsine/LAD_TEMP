import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np



# 注意力加权池化
class AttentionPooling(nn.Module):

    def __init__(self, input_dim, hidden_dim=None):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = input_dim // 4  # 默认4倍压缩

        # 注意力计算网络
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),  # 平滑激活
            nn.Linear(hidden_dim, 1)  # 输出标量分数
        )

    def forward(self, x, mask=None):

        # 计算注意力分数
        attn_scores = self.attention(x).squeeze(-1)  # [B, L]

        # 若有padding,应用mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax归一化
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, L]

        # 加权求和
        pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # [B, D]

        return pooled, attn_weights



class MultiScaleConditionEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        input_dim = config.get('qwen_hidden_size', 896)
        global_dim = config.get('global_condition_dim', 256)
        local_dim = config.get('local_condition_dim', 512)
        temporal_dim = config.get('temporal_condition_dim', 256)

        # Global encoder
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, global_dim)
        )

        # Local encoder
        local_encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=8,
            dim_feedforward=input_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.local_encoder = nn.TransformerEncoder(local_encoder_layer, num_layers=4)
        self.local_proj = nn.Linear(input_dim, local_dim)

        # Temporal encoder
        temporal_encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=8,
            dim_feedforward=input_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_encoder_layer, num_layers=2)
        self.temporal_proj = nn.Linear(input_dim, temporal_dim)

        # Unified encoder
        unified_dim = global_dim + local_dim + temporal_dim
        self.unified_encoder = nn.Sequential(
            nn.Linear(unified_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512)
        )

    def forward(self, qwen_output, attention_mask=None):
        B, L, D = qwen_output.shape

        # Global
        global_pool = self.global_pool(qwen_output.transpose(1, 2)).squeeze(-1)
        global_cond = self.global_encoder(global_pool)

        # Local
        local_feat = self.local_encoder(qwen_output)
        local_cond = self.local_proj(local_feat)

        # Temporal
        temporal_feat = self.temporal_encoder(qwen_output)
        temporal_cond = self.temporal_proj(temporal_feat)

        # Unified
        global_expanded = global_cond.unsqueeze(1).expand(-1, L, -1)
        unified_input = torch.cat([
            global_expanded,
            local_cond,
            temporal_cond
        ], dim=-1)
        unified_cond = self.unified_encoder(unified_input)

        return {
            'global': global_cond,
            'local': local_cond,
            'temporal': temporal_cond,
            'unified': unified_cond
        }



# 条件编码器
class ConditionEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        # 确定条件类型
        use_multiscale = config.get('use_multiscale_condition', False)
        condition_type = config.get('condition_type', 'dual')

        if condition_type == 'dual' or not use_multiscale:
            print("Using DualConditionEncoder (simplified, recommended)")
            self.encoder = DualConditionEncoder(config)
            self.condition_type = 'dual'
        else:
            print("Using MultiScaleConditionEncoder (legacy, compatible)")
            self.encoder = MultiScaleConditionEncoder(config)
            self.condition_type = 'multiscale'

        # 记录输出维度（用于UNet）
        if self.condition_type == 'dual':
            self.global_dim = config.get('condition_dim', 512)
            self.sequence_dim = config.get('condition_dim', 512)
        else:
            self.global_dim = config.get('global_condition_dim', 256)
            self.local_dim = config.get('local_condition_dim', 512)
            self.temporal_dim = config.get('temporal_condition_dim', 256)
            self.unified_dim = 512

    def forward(self, qwen_output, attention_mask=None):
        return self.encoder(qwen_output, attention_mask)



# 条件可视化

class ConditionVisualizer:

    @staticmethod
    def visualize_attention_weights(attention_weights, save_path=None):
        weights = attention_weights.detach().cpu().numpy()
        B, L = weights.shape

        fig, axes = plt.subplots(min(B, 4), 1, figsize=(12, 3 * min(B, 4)))
        if B == 1:
            axes = [axes]

        for i, ax in enumerate(axes[:B]):
            ax.bar(range(L), weights[i])
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Attention Weight')
            ax.set_title(f'Sample {i} - Attention Distribution')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Attention weights saved to {save_path}")
        else:
            plt.show()

        plt.close()

    @staticmethod
    def analyze_condition_stats(condition_dict):

        stats = {}

        for key, value in condition_dict.items():
            if isinstance(value, torch.Tensor):
                stats[key] = {
                    'shape': tuple(value.shape),
                    'mean': value.mean().item(),
                    'std': value.std().item(),
                    'min': value.min().item(),
                    'max': value.max().item(),
                    'norm': value.norm().item()
                }

        return stats



# 条件dropout（用于CFG训练）
class ConditionalDropout(nn.Module):

    def __init__(self, dropout_prob=0.1):
        super().__init__()
        self.dropout_prob = dropout_prob

    def forward(self, condition_dict, force_drop=False):
        if not self.training and not force_drop:
            return condition_dict

        # 决定是否dropout
        if force_drop or torch.rand(1).item() < self.dropout_prob:
            # 全部置零
            dropped_dict = {}
            for key, value in condition_dict.items():
                if isinstance(value, torch.Tensor):
                    dropped_dict[key] = torch.zeros_like(value)
                else:
                    dropped_dict[key] = value
            return dropped_dict
        else:
            return condition_dict
