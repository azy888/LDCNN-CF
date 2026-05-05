"""
Attention visualization module
Extract and visualize cross-attention weights
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Optional, Tuple
import os


def extract_attention_weights(model, mel_spec: torch.Tensor, 
                               psycho_features: torch.Tensor,
                               device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract attention weights from the cross-attention module
    
    Args:
        model: Trained LDCNN-CF model
        mel_spec: Mel-spectrogram input [1, 1, n_mels, T]
        psycho_features: Psychoacoustic features input [1, 1, 4, T]
        device: torch device
    
    Returns:
        attn_mha1: Attention weights from MHA1 (Mel → Psycho) [T, T]
        attn_mha2: Attention weights from MHA2 (Psycho → Mel) [T, T]
    """
    model.eval()
    
    with torch.no_grad():
        # Forward pass with attention output
        _, attention = model(mel_spec.to(device), psycho_features.to(device), 
                              return_attention=True)
        
        # attention shape: [B, num_heads, T, T]
        # Average over heads and batch, then convert to numpy
        if attention is not None:
            attn_avg = attention.mean(dim=0).mean(dim=0).cpu().numpy()
            # For simplicity, return the same matrix for both directions
            # In the actual model, MHA1 and MHA2 may have separate attention outputs
            return attn_avg, attn_avg
        else:
            raise ValueError("Model did not return attention weights")
    
    return None, None


def plot_attention_heatmap(attention_matrix: np.ndarray, 
                            title: str = "Cross-Attention Weights",
                            save_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (8, 7)):
    """
    Plot attention heatmap
    
    Args:
        attention_matrix: Attention weight matrix [T, T]
        title: Plot title
        save_path: Path to save the figure (optional)
        figsize: Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    
    # Use jet colormap for consistency with paper
    plt.imshow(attention_matrix, cmap='jet', origin='lower')
    plt.colorbar(label='Attention Weight')
    
    # Set labels
    plt.xlabel('Key/Value Time Frame', fontsize=12)
    plt.ylabel('Query Time Frame', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Add diagonal reference line
    T = attention_matrix.shape[0]
    plt.plot([0, T-1], [0, T-1], 'w--', linewidth=1, alpha=0.8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention heatmap saved to {save_path}")
    
    plt.show()


def plot_multi_attention_heatmaps(attention_matrices: list, 
                                   titles: list,
                                   save_path: Optional[str] = None,
                                   n_cols: int = 2):
    """
    Plot multiple attention heatmaps in a grid
    
    Args:
        attention_matrices: List of attention matrices
        titles: List of titles for each subplot
        save_path: Path to save the figure (optional)
        n_cols: Number of columns in the grid
    """
    n_plots = len(attention_matrices)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4.5))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, (attn, title) in enumerate(zip(attention_matrices, titles)):
        im = axes[i].imshow(attn, cmap='jet', origin='lower')
        axes[i].set_title(title, fontsize=11, fontweight='bold')
        axes[i].set_xlabel('Key/Value Time Frame')
        axes[i].set_ylabel('Query Time Frame')
        plt.colorbar(im, ax=axes[i], label='Attention Weight')
        
        # Add diagonal reference line
        T = attn.shape[0]
        axes[i].plot([0, T-1], [0, T-1], 'w--', linewidth=0.8, alpha=0.8)
    
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Multi-attention heatmaps saved to {save_path}")
    
    plt.show()


def plot_attention_bar_weights(weights: dict, 
                                save_path: Optional[str] = None):
    """
    Plot bar chart of average attention weights per psychoacoustic parameter
    
    Args:
        weights: Dictionary mapping parameter names to attention weights
        save_path: Path to save the figure (optional)
    """
    parameters = list(weights.keys())
    values = list(weights.values())
    
    plt.figure(figsize=(6, 5))
    
    bars = plt.bar(parameters, values, color='steelblue', edgecolor='black', linewidth=1)
    
    # Add value labels on top of bars
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.ylabel('Average Attention Weight', fontsize=12, fontweight='bold')
    plt.xlabel('Psychoacoustic Parameter', fontsize=12, fontweight='bold')
    plt.ylim(0, 0.55)
    plt.grid(axis='y', alpha=0.3)
    plt.title('Average Cross-Attention Weights', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention bar chart saved to {save_path}")
    
    plt.show()


def visualize_sample_attention(model, mel_spec: torch.Tensor,
                                psycho_features: torch.Tensor,
                                device: torch.device,
                                save_dir: str = 'visualizations'):
    """
    Complete attention visualization pipeline for a single sample
    
    Args:
        model: Trained LDCNN-CF model
        mel_spec: Mel-spectrogram input
        psycho_features: Psychoacoustic features input
        device: torch device
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract attention weights
    attn_mha1, attn_mha2 = extract_attention_weights(model, mel_spec, psycho_features, device)
    
    if attn_mha1 is not None:
        # Plot attention heatmap
        plot_attention_heatmap(attn_mha1, 
                               title='Cross-Attention Weights (Mel → Psycho)',
                               save_path=os.path.join(save_dir, 'attention_mha1.png'))
        
        plot_attention_heatmap(attn_mha2,
                               title='Cross-Attention Weights (Psycho → Mel)',
                               save_path=os.path.join(save_dir, 'attention_mha2.png'))
        
        # Plot both side by side
        plot_multi_attention_heatmaps(
            [attn_mha1, attn_mha2],
            ['MHA1: Mel → Psycho', 'MHA2: Psycho → Mel'],
            save_path=os.path.join(save_dir, 'attention_both.png')
        )
        
        print(f"Attention visualizations saved to {save_dir}")


# Example usage
if __name__ == "__main__":
    # Create dummy attention matrix for demonstration
    T = 30
    np.random.seed(42)
    
    # Generate band-diagonal attention pattern
    attn_demo = np.zeros((T, T))
    for i in range(T):
        for j in range(T):
            dist = abs(i - j)
            attn_demo[i, j] = np.exp(-dist**2 / (2 * 4**2))
    
    # Normalize rows
    attn_demo = attn_demo / attn_demo.sum(axis=1, keepdims=True)
    
    # Plot
    plot_attention_heatmap(attn_demo, title='Sample Attention Heatmap')
    
    # Bar chart demo
    weights = {
        'Roughness': 0.42,
        'Sharpness': 0.31,
        'Loudness': 0.18,
        'Fluctuation Strength': 0.09
    }
    plot_attention_bar_weights(weights)