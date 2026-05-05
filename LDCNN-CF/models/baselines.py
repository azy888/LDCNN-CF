"""
Baseline models: CNN, LSTM, DCNN-CaF, AST (simplified), Conformer (simplified)
For comparison experiments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import math


class CNNBaseline(nn.Module):
    """Standard CNN baseline model"""
    
    def __init__(self, input_channels: int = 1, conv_filters: list = [32, 64, 128, 256]):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        
        in_ch = input_channels
        for out_ch in conv_filters:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2)
                )
            )
            in_ch = out_ch
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(conv_filters[-1], 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
    
    def forward(self, mel_spec: torch.Tensor, psycho_features: torch.Tensor = None):
        """Forward pass
        
        Args:
            mel_spec: Mel-spectrogram [B, 1, n_mels, T]
            psycho_features: Psychoacoustic features (optional) [B, 1, 4, T]
        """
        if psycho_features is not None:
            x = torch.cat([mel_spec, psycho_features], dim=1)
        else:
            x = mel_spec
        
        for conv in self.conv_layers:
            x = conv(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        score = self.fc(x)
        return torch.clamp(score, 1.0, 10.0)


class LSTMBaseline(nn.Module):
    """LSTM baseline model"""
    
    def __init__(self, input_dim: int = 64 + 4, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                            batch_first=True, bidirectional=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim * 2, 1)
    
    def forward(self, mel_spec: torch.Tensor, psycho_features: torch.Tensor):
        """Forward pass"""
        B, C, H, T = mel_spec.shape
        mel_flat = mel_spec.reshape(B, C, -1).transpose(1, 2)  # [B, T, C]
        
        B2, C2, H2, T2 = psycho_features.shape
        psycho_flat = psycho_features.reshape(B2, C2, -1).transpose(1, 2)  # [B, T, C2]
        
        # Concatenate features
        x = torch.cat([mel_flat, psycho_flat], dim=-1)  # [B, T, C+C2]
        
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        score = self.fc(last_out)
        return torch.clamp(score, 1.0, 10.0)


class DCNN_CaF(nn.Module):
    """
    DCNN-CaF baseline model
    Dual-branch CNN with channel attention fusion (larger parameter count)
    """
    
    def __init__(self, config: dict):
        super().__init__()
        filters = [64, 128, 256, 512]  # Larger parameter count
        d_model = 512
        
        # Dual branches
        self.spectral_branch = self._make_branch(1, filters)
        self.perceptual_branch = self._make_branch(1, filters)
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(filters[-1] * 2, filters[-1]),
            nn.ReLU(),
            nn.Linear(filters[-1], filters[-1] * 2),
            nn.Sigmoid()
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filters[-1] * 2, 1)
    
    def _make_branch(self, in_channels, filters):
        layers = []
        for out_ch in filters:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2)
                )
            )
            in_channels = out_ch
        return nn.Sequential(*layers)
    
    def forward(self, mel_spec: torch.Tensor, psycho_features: torch.Tensor):
        """Forward pass"""
        F_m = self.spectral_branch(mel_spec)
        F_p = self.perceptual_branch(psycho_features)
        
        # Ensure same spatial dimensions
        if F_m.shape[-2:] != F_p.shape[-2:]:
            F_p = F.interpolate(F_p, size=F_m.shape[-2:], mode='bilinear', align_corners=False)
        
        # Channel attention
        combined = torch.cat([F_m, F_p], dim=1)
        attn = self.channel_attention(combined).unsqueeze(-1).unsqueeze(-1)
        
        fused = combined * attn
        x = self.global_pool(fused)
        x = x.view(x.size(0), -1)
        score = self.fc(x)
        return torch.clamp(score, 1.0, 10.0)


class SimpleAST(nn.Module):
    """
    Simplified Audio Spectrogram Transformer (AST)
    For small dataset comparison
    """
    
    def __init__(self, input_dim: int = 64, d_model: int = 128, nhead: int = 4, 
                 num_layers: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
    
    def forward(self, mel_spec: torch.Tensor, psycho_features: torch.Tensor = None):
        """Forward pass"""
        B, C, H, T = mel_spec.shape
        x = mel_spec.reshape(B, H, T).transpose(1, 2)  # [B, T, H]
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        score = self.fc(x)
        return torch.clamp(score, 1.0, 10.0)


class SimpleConformer(nn.Module):
    """
    Simplified Conformer (CNN + Transformer)
    For small dataset comparison
    """
    
    def __init__(self, input_dim: int = 64, d_model: int = 128, 
                 kernel_size: int = 5, num_layers: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.conformer_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.conformer_layers.append(ConformerBlock(d_model, kernel_size))
        
        self.fc = nn.Linear(d_model, 1)
    
    def forward(self, mel_spec: torch.Tensor, psycho_features: torch.Tensor = None):
        """Forward pass"""
        B, C, H, T = mel_spec.shape
        x = mel_spec.reshape(B, H, T).transpose(1, 2)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        
        for layer in self.conformer_layers:
            x = layer(x)
        
        x = x.mean(dim=1)
        score = self.fc(x)
        return torch.clamp(score, 1.0, 10.0)


class ConformerBlock(nn.Module):
    """Conformer block"""
    
    def __init__(self, d_model: int, kernel_size: int = 5):
        super().__init__()
        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model)
        )
        self.mhsa = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True, dropout=0.1)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(0.1)
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # FFN1
        x = x + 0.5 * self.ffn1(self.norm1(x))
        # MHSA
        attn_out, _ = self.mhsa(self.norm2(x), self.norm2(x), self.norm2(x))
        x = x + attn_out
        # Conv
        x_conv = x.transpose(1, 2)
        x_conv = self.conv(x_conv)
        x = x + self.norm3(x_conv.transpose(1, 2))
        # FFN2
        x = x + 0.5 * self.ffn2(self.norm4(x))
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


def get_baseline_model(name: str, config: dict):
    """Factory function to get baseline model"""
    if name == 'cnn':
        return CNNBaseline()
    elif name == 'lstm':
        return LSTMBaseline()
    elif name == 'dcnn_caf':
        return DCNN_CaF(config)
    elif name == 'ast':
        return SimpleAST()
    elif name == 'conformer':
        return SimpleConformer()
    else:
        raise ValueError(f"Unknown model: {name}")


if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    B = 4
    T = 480
    mel = torch.randn(B, 1, 64, T)
    psycho = torch.randn(B, 1, 4, T)
    
    for name in ['cnn', 'lstm', 'dcnn_caf', 'ast', 'conformer']:
        model = get_baseline_model(name, config)
        with torch.no_grad():
            out = model(mel, psycho)
        print(f"{name.upper()}: Output shape {out.shape}")