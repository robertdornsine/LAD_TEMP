import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPImageProcessor
import math


# 残差块
class ResidualBlock(nn.Module):
    """
    x → Conv → BN → GELU → Conv → BN → (+shortcut) → GELU → out
    """
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        
        # 第一个卷积块
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 第二个卷积块
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Dropout
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
        
        # Shortcut连接
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = self.shortcut(x)  # Shortcut路径
        
        # 主路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.gelu(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 残差连接
        out = out + identity
        out = F.gelu(out)
        
        return out

# 空间注意力
class SpatialAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        
        # 通道压缩 → 空间注意力
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, 1, kernel_size=1),
            nn.Sigmoid()  # 输出[0,1]的权重
        )
    
    def forward(self, x):
        attn_map = self.attention(x)  # [B, 1, H, W]
        return x * attn_map  # 广播相乘


# 改进的CNN编码器
class ImprovedCNNEncoder(nn.Module):
    """
    → Stem(Conv+BN+GELU) → [12×12×32]
    → ResBlock1 → [12×12×64]
    → ResBlock2 → [12×12×128]
    → ResBlock3 → [12×12×256]
    → ResBlock4 → [12×12×256]
    → SpatialAttention → [12×12×256]
    → Dual Pooling → [512]
    → Projection → [512]
    """
    def __init__(self, config):
        super().__init__()
        
        # 配置参数
        in_channels = 1  # 灰度图
        channels = config.get('cnn_channels', [32, 64, 128, 256])
        self.visual_dim = config.get('visual_dim', 512)
        use_residual = config.get('cnn_use_residual', True)
        use_attention = config.get('cnn_use_attention', True)
        pool_type = config.get('cnn_pool_type', 'dual')  # 'dual', 'avg', 'max'
        dropout = config.get('dropout', 0.1)
        
        # 1.Stem（入口层）
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.GELU()
        )
        
        # 2.残差块序列
        self.blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            if use_residual:
                self.blocks.append(
                    ResidualBlock(channels[i], channels[i+1], dropout=dropout)
                )
            else:
                # 兼容
                self.blocks.append(
                    nn.Sequential(
                        nn.Conv2d(channels[i], channels[i+1], 3, padding=1),
                        nn.BatchNorm2d(channels[i+1]),
                        nn.GELU()
                    )
                )
        
        # 3.空间注意力
        if use_attention:
            self.spatial_attn = SpatialAttention(channels[-1])
        else:
            self.spatial_attn = None
        
        # 4.全局池化
        self.pool_type = pool_type
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 计算池化后的特征维度
        if pool_type == 'dual':
            pool_dim = channels[-1] * 2  # Avg + Max
        else:
            pool_dim = channels[-1]
        
        # 5.投影层（MLP）
        self.projection = nn.Sequential(
            nn.Linear(pool_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, self.visual_dim)
        )
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, images):
        B, T, C, H, W = images.shape
        
        # 合并batch和时间维度
        x = images.view(B * T, C, H, W)  # [B*T, 1, 12, 12]
        
        # Stem
        x = self.stem(x)  # [B*T, 32, 12, 12]
        
        # 残差块序列
        for block in self.blocks:
            x = block(x)  # [B*T, 256, 12, 12]
        
        # 空间注意力
        if self.spatial_attn is not None:
            x = self.spatial_attn(x)  # [B*T, 256, 12, 12]
        
        # 全局池化
        if self.pool_type == 'dual':
            x_avg = self.global_avg_pool(x).view(B * T, -1)  # [B*T, 256]
            x_max = self.global_max_pool(x).view(B * T, -1)  # [B*T, 256]
            x = torch.cat([x_avg, x_max], dim=1)  # [B*T, 512]
        elif self.pool_type == 'avg':
            x = self.global_avg_pool(x).view(B * T, -1)
        else:  # 'max'
            x = self.global_max_pool(x).view(B * T, -1)
        
        # 投影到目标维度
        x = self.projection(x)  # [B*T, 512]
        
        # 恢复时间维度
        x = x.view(B, T, 1, self.visual_dim)  # [B, T, 1, 512]
        
        return x


# 原SimpleCNN（兼容）
class SimpleCNNEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_channels = 1
        channels = config.get('cnn_channels', [16, 32, 64])
        self.visual_dim = config.get('visual_dim', 256)
        
        layers = []
        for out_channels in channels:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])
            in_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[-1], self.visual_dim)
    
    def forward(self, images):
        B, T, C, H, W = images.shape
        x = images.view(B * T, C, H, W)
        x = self.features(x)
        x = self.pool(x).view(B * T, -1)
        x = self.fc(x)
        x = x.view(B, T, 1, self.visual_dim)
        return x


# CLIP编码器（不使用）
"""
class CLIPEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        model_name = config.get('clip_model_name', 'openai/clip-vit-base-patch32')
        self.model = CLIPVisionModel.from_pretrained(model_name)
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.visual_dim = self.model.config.hidden_size
        
        if config.get('freeze_vision_encoder', False):
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, images):
        B, T, C, H, W = images.shape
        images_flat = images.view(B * T, C, H, W)
        
        if C == 1:
            images_flat = images_flat.repeat(1, 3, 1, 1)
        
        outputs = self.model(pixel_values=images_flat)
        features = outputs.last_hidden_state
        features = features.view(B, T, features.size(1), features.size(2))
        
        return features
"""


# 增强的时序聚合器
class EnhancedTemporalAggregator(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        visual_dim = config.get('visual_dim', 512)
        num_layers = config.get('num_temporal_layers', 4)
        self.num_output_tokens = config.get('num_output_tokens', 64)
        use_causal = config.get('temporal_use_causal_mask', True)
        use_time_embed = config.get('temporal_use_time_embed', True)
        
        self.use_causal = use_causal
        self.use_time_embed = use_time_embed
        
        # 1.时间位置编码（可学习）
        if use_time_embed:
            history_len = config.get('history_len', 16)
            self.time_pos_embed = nn.Parameter(
                torch.randn(1, history_len, visual_dim) * 0.02
            )
        else:
            self.time_pos_embed = None
        
        # 2.可学习的Query向量
        self.queries = nn.Parameter(
            torch.randn(1, self.num_output_tokens, visual_dim) * 0.02
        )
        
        # 3.Temporal Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=visual_dim,
            nhead=8,
            dim_feedforward=visual_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 4.Cross-Attention层（2层）
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                visual_dim, 
                num_heads=8, 
                dropout=0.1,
                batch_first=True
            )
            for _ in range(2)
        ])
        
        self.cross_attn_norms = nn.ModuleList([
            nn.LayerNorm(visual_dim) for _ in range(2)
        ])
        
        self.cross_attn_ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(visual_dim, visual_dim * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(visual_dim * 4, visual_dim),
                nn.Dropout(0.1)
            )
            for _ in range(2)
        ])
        
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(visual_dim) for _ in range(2)
        ])
    
    def forward(self, visual_features):
        B, T, N, D = visual_features.shape
        
        # 展平空间维度
        features = visual_features.view(B, T * N, D)  # [B, T*N, D]
        
        # 1.添加时间位置编码
        if self.use_time_embed and self.time_pos_embed is not None:
            if N == 1:  # SimpleCNN情况
                time_emb = self.time_pos_embed.expand(B, -1, -1)  # [B, T, D]
                features = features + time_emb
            else:  # CLIP情况：每个spatial token都加同样的时间编码（实际上没有使用）
                time_emb = self.time_pos_embed.unsqueeze(2).expand(B, T, N, D)
                time_emb = time_emb.reshape(B, T * N, D)
                features = features + time_emb
        
        # 2.因果Transformer编码
        if self.use_causal:
            # 生成因果mask
            seq_len = features.size(1)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=features.device) * float('-inf'),
                diagonal=1
            )
        else:
            causal_mask = None
        
        encoded = self.temporal_encoder(
            features,
            mask=causal_mask
        )  # [B, T*N, D]
        
        # 3.Query Cross-Attention
        queries = self.queries.expand(B, -1, -1)  # [B, 64, D]
        
        for cross_attn, norm1, ffn, norm2 in zip(
            self.cross_attn_layers,
            self.cross_attn_norms,
            self.cross_attn_ffns,
            self.ffn_norms
        ):
            # Cross-Attention
            attn_out, _ = cross_attn(
                queries,   # Q
                encoded,   # K
                encoded    # V
            )
            queries = norm1(queries + attn_out)
            
            # FFN
            ffn_out = ffn(queries)
            queries = norm2(queries + ffn_out)
        
        return queries  # [B, 64, D]


# 原TemporalAggregator（兼容）
class TemporalAggregator(nn.Module):

    def __init__(self, config):
        super().__init__()
        visual_dim = config.get('visual_dim', 256)
        num_layers = config.get('num_temporal_layers', 6)
        self.num_output_tokens = config.get('num_output_tokens', 32)
        
        self.queries = nn.Parameter(
            torch.randn(1, self.num_output_tokens, visual_dim)
        )
        nn.init.trunc_normal_(self.queries, std=0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=visual_dim,
            nhead=8,
            dim_feedforward=visual_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=visual_dim,
            nhead=8,
            dim_feedforward=visual_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.query_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
    
    def forward(self, visual_features):
        B, T, N, D = visual_features.shape
        features = visual_features.view(B, T * N, D)
        
        encoded = self.temporal_encoder(features)
        queries = self.queries.expand(B, -1, -1)
        output = self.query_decoder(queries, encoded)
        
        return output


# VisionEncoder
class VisionEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        encoder_type = config.get('vision_encoder_type', 'simple_cnn')
        
        if encoder_type == 'Improved_cnn':
            print("Using ImprovedCNNEncoder with residual blocks and attention")
            self.model = ImprovedCNNEncoder(config)
        elif encoder_type == 'simple_cnn':
            print("Using SimpleCNNEncoder (legacy)")
            self.model = SimpleCNNEncoder(config)
        #elif encoder_type == 'clip':
         #   print("Using CLIPEncoder")
          #  self.model = CLIPEncoder(config)
        else:
            raise ValueError(f"Unknown vision_encoder_type: {encoder_type}")
        
        self.visual_dim = self.model.visual_dim
        
        # 时序聚合器
        aggregation_method = config.get('temporal_aggregation_method', 'standard')
        
        if aggregation_method == 'enhanced':
            print("Using EnhancedTemporalAggregator with causal mask and time embedding")
            self.temporal_aggregator = EnhancedTemporalAggregator(config)
        else:
            print("Using TemporalAggregator (legacy)")
            self.temporal_aggregator = TemporalAggregator(config)
    
    def forward(self, images):
        visual_features = self.model(images)  # [B, T, N, D]
        aggregated_features = self.temporal_aggregator(visual_features)  # [B, tokens, D]
        return aggregated_features