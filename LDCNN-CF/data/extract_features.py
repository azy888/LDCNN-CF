"""
Feature extraction module
Extract Mel-spectrogram and psychoacoustic parameters
"""

import numpy as np
import librosa
import yaml
from scipy.signal import hilbert
from typing import Tuple


class FeatureExtractor:
    """Feature extractor for audio signals"""
    
    def __init__(self, config: dict):
        """
        Initialize feature extractor
        
        Args:
            config: Configuration dictionary
        """
        self.sr = config['data']['sample_rate']
        self.n_mels = config['data']['n_mels']
        self.n_fft = config['data']['n_fft']
        self.hop_length = config['data']['hop_length']
        self.duration = config['data']['duration']
        
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract Mel-spectrogram
        
        Args:
            audio: Audio array (shape: [samples])
        
        Returns:
            mel_spec: Mel-spectrogram (shape: [n_mels, time_frames])
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        # Convert to log scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec.astype(np.float32)
    
    def extract_psychoacoustic_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract psychoacoustic parameters
        
        Args:
            audio: Audio array (shape: [samples])
        
        Returns:
            psycho_features: Psychoacoustic features (shape: [4, time_frames])
                            Order: [loudness, sharpness, roughness, fluctuation_strength]
        """
        n_frames = 1 + (len(audio) - self.n_fft) // self.hop_length
        
        loudness = np.zeros(n_frames)
        sharpness = np.zeros(n_frames)
        roughness = np.zeros(n_frames)
        fluctuation = np.zeros(n_frames)
        
        for i in range(n_frames):
            start = i * self.hop_length
            end = start + self.n_fft
            frame = audio[start:end]
            
            if len(frame) < self.n_fft:
                frame = np.pad(frame, (0, self.n_fft - len(frame)))
            
            # Compute spectrum
            fft = np.fft.rfft(frame * np.hamming(self.n_fft))
            mag = np.abs(fft)
            freqs = np.fft.rfftfreq(self.n_fft, 1/self.sr)
            
            # 1. Loudness (sone) - simplified calculation
            loudness[i] = np.sum(mag ** 2) / self.n_fft
            
            # 2. Sharpness (acum) - high-frequency weighting
            high_freq_mask = freqs > 2000
            low_freq_mask = freqs <= 2000
            high_energy = np.sum(mag[high_freq_mask] ** 2)
            total_energy = np.sum(mag ** 2) + 1e-8
            sharpness[i] = high_energy / total_energy * 10
            
            # 3. Roughness (asper) - 15-300 Hz modulation
            envelope = np.abs(hilbert(frame))
            envelope_fft = np.fft.rfft(envelope)
            envelope_freqs = np.fft.rfftfreq(len(envelope), 1/self.sr)
            mod_mask = (envelope_freqs >= 15) & (envelope_freqs <= 300)
            roughness[i] = np.sum(np.abs(envelope_fft[mod_mask]) ** 2) / len(envelope)
            
            # 4. Fluctuation strength (vacil) - below 15 Hz modulation
            low_mod_mask = envelope_freqs <= 15
            fluctuation[i] = np.sum(np.abs(envelope_fft[low_mod_mask]) ** 2) / len(envelope)
        
        # Normalize
        loudness = (loudness - loudness.min()) / (loudness.max() - loudness.min() + 1e-8)
        sharpness = np.clip(sharpness / 10, 0, 1)
        roughness = np.clip(roughness / 0.1, 0, 1)
        fluctuation = np.clip(fluctuation / 0.05, 0, 1)
        
        features = np.stack([loudness, sharpness, roughness, fluctuation], axis=0)
        return features.astype(np.float32)
    
    def extract(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract all features
        
        Args:
            audio: Audio array
        
        Returns:
            mel_spec: Mel-spectrogram (n_mels, time_frames)
            psycho_features: Psychoacoustic features (4, time_frames)
        """
        mel_spec = self.extract_mel_spectrogram(audio)
        psycho_features = self.extract_psychoacoustic_features(audio)
        
        # Align time dimensions
        min_frames = min(mel_spec.shape[1], psycho_features.shape[1])
        mel_spec = mel_spec[:, :min_frames]
        psycho_features = psycho_features[:, :min_frames]
        
        return mel_spec, psycho_features


# Example usage
if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    extractor = FeatureExtractor(config)
    
    from preprocess import load_audio
    audio, sr = load_audio('test.wav')
    print(f"Audio loaded: {len(audio)} samples")
    
    mel, psycho = extractor.extract(audio)
    print(f"Mel-spectrogram shape: {mel.shape}")
    print(f"Psychoacoustic features shape: {psycho.shape}")