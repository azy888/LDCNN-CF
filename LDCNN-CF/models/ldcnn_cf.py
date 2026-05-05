"""
LDCNN-CF model definition
Lightweight Dual-Branch CNN with Cross-Attention Fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from typing import Tuple, Optional


class ConvBlock(nn.Module):
    """Convolutional block: 2 conv layers + BN + ReLU + optional downsampling"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class SpectralBranch(nn.Module):
    """Mel-spectrogram branch"""
    
    def __init__(self, in_channels: int = 1, filters: list = [32, 64, 128, 256]):
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        
        for i, out_ch in enumerate(filters):
            stride = 2 if i == 0 else 1
            self.conv_blocks.append(ConvBlock(in_channels, out_ch, stride))
            in_channels = out_ch
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, None))
    
    def forward(self, x):
        # x shape: [B, 1, n_mels, T]
        for conv in self.conv_blocks:
            x = conv(x)
        # x shape: [B, C, H, T]
        x = self.global_pool(x)  # [B, C, 1, T]
        x = x.squeeze(2)  # [B, C, T]
        return x


class PerceptualBranch(nn.Module):
    """Psychoacoustic features branch"""
    
    def __init__(self, in_channels: int = 1, filters: list = [32, 64, 128, 256]):
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        
        for i, out_ch in enumerate(filters):
            stride = 2 if i == 0 else 1
            self.conv_blocks.append(ConvBlock(in_channels, out_ch, stride))
            in_channels = out_ch
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, None))
    
    def forward(self, x):
        # x shape: [B, 1, 4, T]
        for conv in self.conv_blocks:
            x = conv(x)
        x = self.global_pool(x)
        x = x.squeeze(2)
        return x


class CrossAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention fusion module
    MHA1: Mel → Psycho
    MHA2: Psycho → Mel
    """
    
    def __init__(self, d_model: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        # MHA1: Mel as Query, Psycho as Key/Value
        self.W_q1 = nn.Linear(d_model, d_model)
        self.W_k1 = nn.Linear(d_model, d_model)
        self.W_v1 = nn.Linear(d_model, d_model)
        
        # MHA2: Psycho as Query, Mel as Key/Value
        self.W_q2 = nn.Linear(d_model, d_model)
        self.W_k2 = nn.Linear(d_model, d_model)
        self.W_v2 = nn.Linear(d_model, d_model)
        
        self.W_o = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, F_m: torch.Tensor, F_p: torch.Tensor, 
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            F_m: Mel features [B, T, d_model]
            F_p: Psychoacoustic features [B, T, d_model]
            return_attention: Whether to return attention weights
        
        Returns:
            fused: Fused features [B, T, d_model]
            attention: Attention weights (optional) [B, num_heads, T, T]
        """
        B, T, _ = F_m.shape
        
        # ===== MHA1: Mel → Psycho =====
        Q1 = self.W_q1(F_m).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K1 = self.W_k1(F_p).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V1 = self.W_v1(F_p).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn1 = torch.matmul(Q1, K1.transpose(-2, -1)) * self.scale
        attn1 = F.softmax(attn1, dim=-1)
        attn1 = self.dropout(attn1)
        
        F_m_prime = torch.matmul(attn1, V1).transpose(1, 2).reshape(B, T, self.d_model)
        
        # ===== MHA2: Psycho → Mel =====
        Q2 = self.W_q2(F_p).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K2 = self.W_k2(F_m).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V2 = self.W_v2(F_m).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn2 = torch.matmul(Q2, K2.transpose(-2, -1)) * self.scale
        attn2 = F.softmax(attn2, dim=-1)
        attn2 = self.dropout(attn2)
        
        F_p_prime = torch.matmul(attn2, V2).transpose(1, 2).reshape(B, T, self.d_model)
        
        # Concatenate and fuse
        concat = torch.cat([F_m_prime, F_p_prime], dim=-1)
        fused = self.W_o(concat)
        
        if return_attention:
            attn_avg = (attn1 + attn2).mean(dim=1)
            return fused, attn_avg
        return fused, None


class LDCNN_CF(nn.Module):
    """LDCNN-CF model"""
    
    def __init__(self, config: dict):
        super().__init__()
        self.d_model = config['model']['head_dim'] * config['model']['num_heads']
        num_heads = config['model']['num_heads']
        dropout = config['model']['dropout']
        filters = config['model']['conv_filters']
        
        # Dual branches
        self.spectral_branch = SpectralBranch(in_channels=1, filters=filters)
        self.perceptual_branch = PerceptualBranch(in_channels=1, filters=filters)
        
        # Projection layers (project conv output to d_model)
        self.proj_m = nn.Linear(filters[-1], self.d_model)
        self.proj_p = nn.Linear(filters[-1], self.d_model)
        
        # Cross-attention fusion
        self.cross_attention = CrossAttentionFusion(
            d_model=self.d_model, num_heads=num_heads, dropout=dropout
        )
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output comfort score 1-10
        )
    
    def forward(self, mel_spec: torch.Tensor, psycho_features: torch.Tensor,
                return_attention: bool = False) -> torch.Tensor:
        """
        Args:
            mel_spec: Mel-spectrogram [B, 1, n_mels, T]
            psycho_features: Psychoacoustic features [B, 1, 4, T]
            return_attention: Whether to return attention weights
        
        Returns:
            score: Comfort score [B, 1]
            attention: Attention weights (optional)
        """
        # Dual branch feature extraction
        F_m = self.spectral_branch(mel_spec)  # [B, C, T]
        F_p = self.perceptual_branch(psycho_features)  # [B, C, T]
        
        # Transpose to [B, T, C]
        F_m = F_m.transpose(1, 2)
        F_p = F_p.transpose(1, 2)
        
        # Project to d_model
        F_m = self.proj_m(F_m)
        F_p = self.proj_p(F_p)
        
        # Cross-attention fusion
        fused, attention = self.cross_attention(F_m, F_p, return_attention)
        
        # Global average pooling (time dimension)
        fused = fused.mean(dim=1)  # [B, d_model]
        
        # Regression
        score = self.regressor(fused)  # [B, 1]
        score = torch.clamp(score, 1.0, 10.0)  # Clamp to 1-10 range
        
        if return_attention:
            return score, attention
        return score


# Example usage
if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model = LDCNN_CF(config)
    
    # Test forward pass
    B = 4
    T = 480
    mel_spec = torch.randn(B, 1, 64, T)
    psycho = torch.randn(B, 1, 4, T)
    
    score = model(mel_spec, psycho)
    print(f"Output shape: {score.shape}, range: [{score.min():.2f}, {score.max():.2f}]")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")