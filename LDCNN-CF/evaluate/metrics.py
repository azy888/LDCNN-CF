"""
Evaluation metrics module
Compute MAE, RMSE, MAPE, R², and statistical tests
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from typing import Dict, List, Tuple


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute regression metrics
    
    Args:
        y_true: Ground truth values (shape: [N])
        y_pred: Predicted values (shape: [N])
    
    Returns:
        metrics: Dictionary containing MAE, RMSE, MAPE, R²
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAPE (Mean Absolute Percentage Error)
    # Add small epsilon to avoid division by zero
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2
    }


def compute_metrics_with_std(y_true_list: List[np.ndarray], 
                              y_pred_list: List[np.ndarray]) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics with mean and std across multiple folds
    
    Args:
        y_true_list: List of ground truth arrays for each fold
        y_pred_list: List of prediction arrays for each fold
    
    Returns:
        metrics_stats: Dictionary with mean and std for each metric
    """
    all_metrics = {name: [] for name in ['mae', 'rmse', 'mape', 'r2']}
    
    for y_true, y_pred in zip(y_true_list, y_pred_list):
        metrics = compute_metrics(y_true, y_pred)
        for name in all_metrics:
            all_metrics[name].append(metrics[name])
    
    results = {}
    for name in all_metrics:
        results[name] = {
            'mean': np.mean(all_metrics[name]),
            'std': np.std(all_metrics[name])
        }
    
    return results


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, 
                 n_iterations: int = 1000, ci_level: float = 0.95) -> Dict[str, Dict[str, float]]:
    """
    Bootstrap confidence intervals for metrics
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        n_iterations: Number of bootstrap iterations
        ci_level: Confidence level (e.g., 0.95 for 95% CI)
    
    Returns:
        cis: Confidence intervals for each metric
    """
    n = len(y_true)
    metrics_names = ['mae', 'rmse', 'mape', 'r2']
    bootstrap_metrics = {name: [] for name in metrics_names}
    
    for _ in range(n_iterations):
        indices = np.random.choice(n, n, replace=True)
        y_true_bs = y_true[indices]
        y_pred_bs = y_pred[indices]
        
        metrics = compute_metrics(y_true_bs, y_pred_bs)
        for name in metrics_names:
            bootstrap_metrics[name].append(metrics[name])
    
    alpha = 1 - ci_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    cis = {}
    for name in metrics_names:
        cis[name] = {
            'lower': np.percentile(bootstrap_metrics[name], lower_percentile),
            'upper': np.percentile(bootstrap_metrics[name], upper_percentile)
        }
    
    return cis


def paired_t_test(y_true: np.ndarray, 
                   y_pred1: np.ndarray, 
                   y_pred2: np.ndarray) -> Dict[str, float]:
    """
    Paired t-test for comparing two models
    
    Args:
        y_true: Ground truth values
        y_pred1: Predictions from model 1
        y_pred2: Predictions from model 2
    
    Returns:
        result: Dictionary with t-statistic and p-value
    """
    error1 = np.abs(y_true - y_pred1)
    error2 = np.abs(y_true - y_pred2)
    
    t_stat, p_value = stats.ttest_rel(error1, error2)
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def wilcoxon_test(y_true: np.ndarray, 
                   y_pred1: np.ndarray, 
                   y_pred2: np.ndarray) -> Dict[str, float]:
    """
    Wilcoxon signed-rank test (non-parametric alternative to paired t-test)
    
    Args:
        y_true: Ground truth values
        y_pred1: Predictions from model 1
        y_pred2: Predictions from model 2
    
    Returns:
        result: Dictionary with statistic and p-value
    """
    error1 = np.abs(y_true - y_pred1)
    error2 = np.abs(y_true - y_pred2)
    
    statistic, p_value = stats.wilcoxon(error1, error2)
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def compare_all_models(y_true: np.ndarray, 
                       predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Compare multiple models and return results as DataFrame
    
    Args:
        y_true: Ground truth values
        predictions: Dictionary mapping model names to predictions
    
    Returns:
        df: DataFrame with comparison results
    """
    import pandas as pd
    
    results = []
    
    for model_name, y_pred in predictions.items():
        metrics = compute_metrics(y_true, y_pred)
        metrics['model'] = model_name
        results.append(metrics)
    
    df = pd.DataFrame(results)
    df = df.set_index('model')
    
    return df


# Example usage
if __name__ == "__main__":
    # Generate dummy data
    np.random.seed(42)
    n = 100
    y_true = np.random.uniform(1, 10, n)
    y_pred1 = y_true + np.random.normal(0, 0.8, n)
    y_pred2 = y_true + np.random.normal(0, 1.2, n)
    
    # Compute metrics
    metrics1 = compute_metrics(y_true, y_pred1)
    metrics2 = compute_metrics(y_true, y_pred2)
    
    print("Model 1 Metrics:")
    for k, v in metrics1.items():
        print(f"  {k.upper()}: {v:.4f}")
    
    print("\nModel 2 Metrics:")
    for k, v in metrics2.items():
        print(f"  {k.upper()}: {v:.4f}")
    
    # Paired t-test
    ttest = paired_t_test(y_true, y_pred1, y_pred2)
    print(f"\nPaired t-test: t = {ttest['t_statistic']:.4f}, p = {ttest['p_value']:.4f}")
    print(f"Significant: {ttest['significant']}")
    
    # Bootstrap CI
    ci = bootstrap_ci(y_true, y_pred1)
    print("\nBootstrap 95% CI for Model 1:")
    for metric, interval in ci.items():
        print(f"  {metric.upper()}: [{interval['lower']:.4f}, {interval['upper']:.4f}]")