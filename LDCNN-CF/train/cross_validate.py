"""
8-fold leave-one-brand-out cross-validation
"""

import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import yaml
import os
from tqdm import tqdm

from models.ldcnn_cf import LDCNN_CF
from models.baselines import get_baseline_model
from train.trainer import Trainer
from data.extract_features import FeatureExtractor
from data.preprocess import load_audio, reduce_noise_spectral_subtraction


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute evaluation metrics
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
    
    Returns:
        metrics: Dictionary with MAE, RMSE, MAPE, R²
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2
    }


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, 
                 n_iterations: int = 1000, ci: float = 0.95) -> dict:
    """
    Bootstrap confidence intervals
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        n_iterations: Number of bootstrap iterations
        ci: Confidence level
    
    Returns:
        confidence_intervals: Confidence intervals for each metric
    """
    n = len(y_true)
    metrics_names = ['mae', 'rmse', 'mape', 'r2']
    bootstrap_results = {name: [] for name in metrics_names}
    
    for _ in range(n_iterations):
        indices = np.random.choice(n, n, replace=True)
        y_true_bs = y_true[indices]
        y_pred_bs = y_pred[indices]
        
        metrics = compute_metrics(y_true_bs, y_pred_bs)
        for name in metrics_names:
            bootstrap_results[name].append(metrics[name])
    
    # Compute confidence intervals
    ci_lower = (1 - ci) / 2
    ci_upper = 1 - ci_lower
    
    intervals = {}
    for name in metrics_names:
        intervals[name] = {
            'lower': np.percentile(bootstrap_results[name], ci_lower * 100),
            'upper': np.percentile(bootstrap_results[name], ci_upper * 100)
        }
    
    return intervals


def paired_t_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray) -> dict:
    """
    Paired t-test for model comparison
    
    Args:
        y_true: Ground truth values
        y_pred1: Predictions from model 1
        y_pred2: Predictions from model 2
    
    Returns:
        result: Dictionary with t-statistic, p-value, and significance flag
    """
    error1 = np.abs(y_true - y_pred1)
    error2 = np.abs(y_true - y_pred2)
    
    t_stat, p_value = stats.ttest_rel(error1, error2)
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


class CrossValidator:
    """8-fold leave-one-brand-out cross-validator"""
    
    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        self.brands = config['dataset']['brands']
        self.n_folds = len(self.brands)
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor(config)
    
    def load_and_extract_features(self, audio_paths: list, labels: list) -> tuple:
        """
        Load audio and extract features
        
        Returns:
            mel_features: List of Mel-spectrograms
            psycho_features: List of psychoacoustic features
            labels: List of labels
        """
        mel_features = []
        psycho_features = []
        
        for path in tqdm(audio_paths, desc="Extracting features"):
            audio, sr = load_audio(path, 
                                   target_sr=self.config['data']['sample_rate'],
                                   target_duration=self.config['data']['duration'])
            audio = reduce_noise_spectral_subtraction(audio, sr)
            mel, psycho = self.feature_extractor.extract(audio)
            mel_features.append(mel)
            psycho_features.append(psycho)
        
        return mel_features, psycho_features, labels
    
    def run_cross_validation(self, audio_paths: list, labels: list, 
                             brand_labels: list, model_name: str = 'ldcnn_cf'):
        """
        Run cross-validation
        
        Args:
            audio_paths: List of audio file paths
            labels: List of comfort scores
            brand_labels: List of brand labels (B1-B8)
            model_name: Name of the model to evaluate
        
        Returns:
            results: Dictionary with per-fold results and summary statistics
        """
        fold_results = []
        all_y_true = []
        all_y_pred = []
        
        for fold, test_brand in enumerate(self.brands):
            print(f"\n{'='*50}")
            print(f"Fold {fold+1}/{self.n_folds}: Test Brand = {test_brand}")
            print(f"{'='*50}")
            
            # Split data
            train_indices = []
            val_indices = []
            test_indices = []
            
            for idx, brand in enumerate(brand_labels):
                if brand == test_brand:
                    test_indices.append(idx)
                else:
                    # Use one brand as validation
                    if len(train_indices) % 7 == 6:
                        val_indices.append(idx)
                    else:
                        train_indices.append(idx)
            
            print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}, Test samples: {len(test_indices)}")
            
            # Create model
            if model_name == 'ldcnn_cf':
                model = LDCNN_CF(self.config)
            else:
                model = get_baseline_model(model_name, self.config)
            
            trainer = Trainer(model, self.config, self.device)
            
            # Note: In practice, you need to implement data loaders here
            # trainer.fit(train_loader, val_loader, epochs=self.config['training']['epochs'])
            
            # Predict
            # y_pred = trainer.predict(test_loader)
            
            # Evaluate
            # metrics = compute_metrics(y_true, y_pred)
            # fold_results.append(metrics)
            # all_y_true.extend(y_true)
            # all_y_pred.extend(y_pred)
        
        # Summarize results
        summary = self._summarize_results(fold_results, all_y_true, all_y_pred)
        
        return summary
    
    def _summarize_results(self, fold_results: list, y_true: list, y_pred: list) -> dict:
        """Summarize cross-validation results"""
        if not fold_results:
            return {}
        
        metrics_names = ['mae', 'rmse', 'mape', 'r2']
        fold_metrics = {name: [] for name in metrics_names}
        
        for result in fold_results:
            for name in metrics_names:
                fold_metrics[name].append(result[name])
        
        # Compute mean and std
        summary = {}
        for name in metrics_names:
            summary[f'{name}_mean'] = np.mean(fold_metrics[name])
            summary[f'{name}_std'] = np.std(fold_metrics[name])
        
        # Overall metrics
        if y_true and y_pred:
            overall_metrics = compute_metrics(np.array(y_true), np.array(y_pred))
            summary['overall'] = overall_metrics
            
            # Bootstrap confidence intervals
            cis = bootstrap_ci(np.array(y_true), np.array(y_pred))
            summary['confidence_intervals'] = cis
        
        return summary


def compare_models(config: dict, audio_paths: list, labels: list, 
                   brand_labels: list, device: torch.device):
    """
    Compare multiple models
    """
    models = ['ldcnn_cf', 'cnn', 'lstm', 'dcnn_caf', 'ast', 'conformer']
    results = {}
    
    cv = CrossValidator(config, device)
    
    for model_name in models:
        print(f"\n\n{'#'*60}")
        print(f"Evaluating model: {model_name.upper()}")
        print(f"{'#'*60}")
        
        result = cv.run_cross_validation(audio_paths, labels, brand_labels, model_name)
        results[model_name] = result
    
    # Print comparison results
    print("\n\n" + "="*80)
    print("Model Comparison Results")
    print("="*80)
    
    for model_name, result in results.items():
        print(f"\n{model_name.upper()}:")
        if 'mae_mean' in result:
            print(f"  MAE: {result['mae_mean']:.4f} ± {result['mae_std']:.4f}")
            print(f"  R²: {result['r2_mean']:.4f} ± {result['r2_std']:.4f}")
    
    return results


if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Provide real data paths in practice
    print("Cross-validation framework ready. Please provide real data paths.")