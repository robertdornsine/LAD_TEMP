import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math



#  增强的时序卷积模块
class EnhancedTemporalConv(nn.Module):

    def __init__(
            self,
            channels: int,
            kernel_sizes: list = [3, 5],
            dropout: float = 0.1
    ):
        super().__init__()
        self.channels = channels
        self.kernel_sizes = kernel_sizes

        # 多尺度卷积
        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            self.convs.append(
                nn.Sequential(
                    # Depthwise卷积
                    nn.Conv1d(
                        channels, channels,
                        kernel_size=k,
                        padding=k // 2,
                        groups=channels
                    ),
                    nn.GELU(),
                    # Pointwise卷积
                    nn.Conv1d(channels, channels, kernel_size=1),
                    nn.Dropout(dropout)
                )
            )

        # 融合投影
        self.fusion_proj = nn.Linear(channels * len(kernel_sizes), channels)

        # LayerNorm
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # 转换为Conv1d格式
        h = x.transpose(1, 2)  # [B, C, L]

        # 多尺度卷积
        conv_outputs = []
        for conv in self.convs:
            conv_outputs.append(conv(h))

        # 拼接
        h = torch.cat(conv_outputs, dim=1)  # [B, C*K, L]
        h = h.transpose(1, 2)  # [B, L, C*K]

        # 融合
        h = self.fusion_proj(h)  # [B, L, C]

        # 残差 + Norm
        h = self.norm(x + h)

        return h


# 轻量级注意力
class LightweightAttention(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int = 4,
            dropout: float = 0.1
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # QKV投影
        self.qkv = nn.Linear(dim, dim * 3)

        # 输出投影
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # LayerNorm
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, L, C = x.shape

        # QKV
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, nh, L, hd]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, nh, L, L]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Output
        out = (attn @ v).transpose(1, 2).reshape(B, L, C)  # [B, L, C]
        out = self.proj(out)
        out = self.dropout(out)

        # 残差 + Norm
        out = self.norm(x + out)

        return out


# 增强的TemporalAdapter
class TemporalAdapterV2(nn.Module):

    def __init__(
            self,
            hidden_size: int = 896,  # Qwen2-0.5B的hidden_size
            adapter_size: int = 128,
            dropout: float = 0.1,
            use_conv: bool = True,
            use_gate: bool = True,
            use_attention: bool = False,
            conv_kernel_sizes: list = None,  # 卷积核尺寸
            use_layernorm: bool = True,  # 使用LayerNorm
            init_scale: float = 0.1  # 可配置的初始化缩放
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.adapter_size = adapter_size
        self.use_conv = use_conv
        self.use_gate = use_gate
        self.use_attention = use_attention
        self.use_layernorm = use_layernorm

        # 下投影
        self.down_proj = nn.Linear(hidden_size, adapter_size)

        if use_layernorm:
            self.down_norm = nn.LayerNorm(adapter_size)


        # 上投影
        if use_layernorm:
            self.up_norm = nn.LayerNorm(adapter_size)

        self.up_proj = nn.Linear(adapter_size, hidden_size)

        # Dropout和激活
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

        # 可学习的残差权重
        self.scale = nn.Parameter(torch.ones(1) * init_scale)

        # 初始化
        self._reset_parameters()

    def _reset_parameters(self):
        # 下投影
        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)

        # 上投影
        nn.init.xavier_uniform_(self.up_proj.weight, gain=0.01)
        nn.init.zeros_(self.up_proj.bias)

        # 门控
        if self.use_gate:
            nn.init.xavier_uniform_(self.gate_proj[0].weight, gain=0.1)
            nn.init.constant_(self.gate_proj[0].bias, 0.0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states

        # 1. 下投影
        h = self.down_proj(hidden_states)  # [B, L, adapter_size]

        if self.use_layernorm:
            h = self.down_norm(h)

        # 2. 时序卷积
        if self.use_conv:
            h_conv = self.temporal_conv(h)

            # 门控融合
            if self.use_gate:
                gate = self.gate_proj(h)
                h = gate * h_conv + (1 - gate) * h
            else:
                h = h_conv

        # 3. 注意力
        if self.use_attention:
            h = self.attention(h)

        # 4. 激活和dropout
        h = self.act(h)
        h = self.dropout(h)

        # 5. 上投影前的Norm
        if self.use_layernorm:
            h = self.up_norm(h)

        # 6. 上投影
        h = self.up_proj(h)  # [B, L, hidden_size]

        # 7. 带缩放的残差连接
        output = residual + self.scale * h

        return output



# MultiLayerAdapter
class MultiLayerAdapterV2(nn.Module):

    def __init__(
            self,
            num_layers: int,
            hidden_size: int = 896,
            adapter_size: int = 128,
            adapter_layers: str = 'all',
            dropout: float = 0.1,
            use_conv: bool = True,
            use_gate: bool = True,
            use_attention: bool = False,
            conv_kernel_sizes: list = None,
            use_layernorm: bool = True,
            use_gradient_checkpointing: bool = False
    ):
        super().__init__()

        self.num_layers = num_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # 使用adapter的层
        if adapter_layers == 'all':
            self.adapter_layer_indices = list(range(num_layers))
        elif adapter_layers == 'last':
            self.adapter_layer_indices = [num_layers - 1]
        elif adapter_layers == 'middle':
            # 中间层
            mid = num_layers // 2
            self.adapter_layer_indices = list(range(mid - 2, mid + 2))
        elif adapter_layers == 'every_other':
            # 每隔一层
            self.adapter_layer_indices = list(range(0, num_layers, 2))
        else:
            # 解析字符串
            self.adapter_layer_indices = [int(x) for x in adapter_layers.split(',')]

        print(f"Adapter enabled at layers: {self.adapter_layer_indices}")

        # 创建TemporalAdapterV2
        self.adapters = nn.ModuleList([
            TemporalAdapterV2(
                hidden_size=hidden_size,
                adapter_size=adapter_size,
                dropout=dropout,
                use_conv=use_conv,
                use_gate=use_gate,
                use_attention=use_attention,
                conv_kernel_sizes=conv_kernel_sizes,
                use_layernorm=use_layernorm
            ) if i in self.adapter_layer_indices else nn.Identity()
            for i in range(num_layers)
        ])

    def forward(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if layer_idx >= len(self.adapters):
            return hidden_states

        # 梯度检查点
        if self.use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self.adapters[layer_idx],
                hidden_states,
                use_reentrant=False
            )
        else:
            return self.adapters[layer_idx](hidden_states)

    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# 兼容
TemporalAdapter = TemporalAdapterV2
MultiLayerAdapter = MultiLayerAdapterV2

# 工具函数
def create_adapter_from_config(config: dict) -> MultiLayerAdapterV2:
    return MultiLayerAdapterV2(
        num_layers=config.get('num_layers', 24),
        hidden_size=config.get('hidden_size', 896),
        adapter_size=config.get('adapter_size', 128),
        adapter_layers=config.get('adapter_layers', 'all'),
        dropout=config.get('dropout', 0.1),
        use_conv=config.get('use_conv', True),
        use_gate=config.get('use_gate', True),
        use_attention=config.get('use_attention', False),
        conv_kernel_sizes=config.get('conv_kernel_sizes', [3, 5]),
        use_layernorm=config.get('use_layernorm', True),
        use_gradient_checkpointing=config.get('use_gradient_checkpointing', False)
    )


def load_legacy_adapter_weights(adapter_v2, legacy_checkpoint):
    try:
        adapter_v2.load_state_dict(legacy_checkpoint, strict=False)
        print("Legacy adapter weights loaded (partial match)")
    except Exception as e:
        print(f"Failed to load legacy weights: {e}")
        print("Will initialize from scratch")