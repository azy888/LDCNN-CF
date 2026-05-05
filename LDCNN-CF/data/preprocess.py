"""
Audio preprocessing module
Functions: load audio, trim to 15 seconds, resample, denoise
"""

import os
import numpy as np
import soundfile as sf
import librosa
import yaml
import random
import torch


def set_seed(seed: int):
    """Fix random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_audio(filepath: str, target_sr: int = 22050, target_duration: float = 15.0):
    """
    Load audio file
    
    Args:
        filepath: Path to audio file
        target_sr: Target sample rate (Hz)
        target_duration: Target duration (seconds)
    
    Returns:
        audio: Audio array (shape: [samples])
        sr: Sample rate
    """
    # Load with librosa (supports multiple formats)
    audio, sr = librosa.load(filepath, sr=None, mono=True)
    
    # Resample to target sample rate
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    
    # Trim to middle target_duration seconds
    target_samples = int(target_duration * sr)
    if len(audio) > target_samples:
        start = (len(audio) - target_samples) // 2
        audio = audio[start:start + target_samples]
    elif len(audio) < target_samples:
        # Zero-pad if shorter
        audio = np.pad(audio, (0, target_samples - len(audio)))
    
    return audio, sr


def reduce_noise_spectral_subtraction(audio: np.ndarray, sr: int, noise_duration: float = 0.3):
    """
    Noise reduction using spectral subtraction
    
    Args:
        audio: Audio array
        sr: Sample rate
        noise_duration: Duration for noise estimation (seconds)
    
    Returns:
        denoised: Denoised audio
    """
    noise_samples = int(noise_duration * sr)
    if len(audio) > noise_samples:
        noise_profile = audio[:noise_samples]
    else:
        noise_profile = audio
    
    # Compute noise spectrum
    noise_fft = np.fft.rfft(noise_profile)
    noise_mag = np.abs(noise_fft)
    
    # Compute audio spectrum
    audio_fft = np.fft.rfft(audio)
    audio_mag = np.abs(audio_fft)
    audio_phase = np.angle(audio_fft)
    
    # Spectral subtraction
    mag_reduced = audio_mag - noise_mag
    mag_reduced = np.maximum(mag_reduced, audio_mag * 0.1)  # Keep at least 10%
    
    # Reconstruct
    fft_reduced = mag_reduced * np.exp(1j * audio_phase)
    denoised = np.fft.irfft(fft_reduced)
    
    # Trim to original length
    denoised = denoised[:len(audio)]
    
    # Normalize
    max_val = np.max(np.abs(denoised))
    if max_val > 0:
        denoised = denoised / max_val * 0.95
    
    return denoised


def save_audio(filepath: str, audio: np.ndarray, sr: int):
    """Save audio file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    sf.write(filepath, audio, sr)


# Example usage
if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    set_seed(config['seed'])
    
    # Test
    audio, sr = load_audio('test.m4a', target_sr=22050, target_duration=15)
    print(f"Loaded: {len(audio)} samples, {sr} Hz")
    
    audio_denoised = reduce_noise_spectral_subtraction(audio, sr)
    print(f"Denoising complete")
    
    save_audio('output.wav', audio_denoised, sr)
    print("Saved")