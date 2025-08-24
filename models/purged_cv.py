"""
Purged Cross-Validation for Time Series Data
Implements leak-free evaluation with purging and embargo periods.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from typing import Generator, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PurgedKFold(TimeSeriesSplit):
    """
    Purged K-Fold CV with embargo for time-series data (López de Prado style).
    
    Args:
        n_splits: Number of folds (e.g., 5).
        purge_pct: Fraction of data to purge before/after test folds (e.g., 0.01).
        embargo_pct: Fraction to embargo after test (e.g., 0.01).
    """
    
    def __init__(self, n_splits: int = 5, purge_pct: float = 0.01, embargo_pct: float = 0.01):
        super().__init__(n_splits=n_splits)
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct
        
    def split(self, X, y=None, groups=None, t1: Optional[pd.Series] = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate purged train/test splits.
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
            groups: Group labels (optional)
            t1: Series of timestamps for purging/embargo
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        if t1 is None:
            logger.warning("No timestamps provided for purging. Using standard TimeSeriesSplit.")
            yield from super().split(X, y, groups)
            return
            
        indices = np.arange(X.shape[0])
        
        for train_idx, test_idx in super().split(indices):
            try:
                # Calculate purge and embargo sizes
                purge_before = int(len(X) * self.purge_pct)
                purge_after = int(len(X) * self.purge_pct)
                embargo_size = int(len(X) * self.embargo_pct)
                
                # Purge: Remove train samples near test period
                purged_train_idx = train_idx[
                    (train_idx < test_idx[0] - purge_before) | 
                    (train_idx > test_idx[-1] + purge_after)
                ]
                
                # Embargo: Skip post-test buffer
                embargo_end_idx = test_idx[-1] + embargo_size
                if embargo_end_idx < len(X):
                    purged_train_idx = purged_train_idx[purged_train_idx < embargo_end_idx]
                
                # Ensure we have enough training data
                if len(purged_train_idx) < len(X) * 0.1:  # At least 10% for training
                    logger.warning(f"Purging resulted in very small training set: {len(purged_train_idx)} samples")
                
                yield purged_train_idx, test_idx
                
            except Exception as e:
                logger.error(f"Error in purged split: {e}")
                # Fallback to standard split
                yield train_idx, test_idx


def block_bootstrap_pnl(
    pnl_series: pd.Series, 
    block_size: int = 60, 
    n_blocks: int = 100, 
    n_bootstraps: int = 1000, 
    metric: str = 'sharpe'
) -> np.ndarray:
    """
    Block bootstrap for P&L confidence intervals (preserves autocorrelation).
    
    Args:
        pnl_series: Series of returns/P&L
        block_size: Block size (e.g., 60 for 1-hour blocks with 1-min data)
        n_blocks: Number of blocks to sample per bootstrap
        n_bootstraps: Number of bootstrap resamples
        metric: 'sharpe', 'return', or 'drawdown'
        
    Returns:
        Array of [2.5%, 50%, 97.5%] percentiles
    """
    if len(pnl_series) < block_size:
        logger.warning(f"Series length {len(pnl_series)} < block_size {block_size}")
        return np.array([0.0, 0.0, 0.0])
    
    bootstrapped_metrics = []
    
    try:
        for _ in range(n_bootstraps):
            # Sample block starts with replacement
            max_start = len(pnl_series) - block_size + 1
            if max_start <= 0:
                continue
                
            block_starts = np.random.choice(max_start, min(n_blocks, max_start), replace=True)
            
            # Concatenate blocks
            bootstrapped_pnl = np.concatenate([
                pnl_series.iloc[start:start + block_size].values 
                for start in block_starts
            ])
            
            # Calculate metric
            if metric == 'sharpe':
                mean_ret = np.mean(bootstrapped_pnl)
                std_ret = np.std(bootstrapped_pnl)
                if std_ret > 0:
                    # Annualize for intraday (390 min/day, 252 trading days)
                    value = mean_ret / std_ret * np.sqrt(252 * 390)
                else:
                    value = 0.0
                    
            elif metric == 'return':
                value = np.sum(bootstrapped_pnl)
                
            elif metric == 'drawdown':
                cum_pnl = np.cumsum(bootstrapped_pnl)
                if len(cum_pnl) > 0:
                    running_max = np.maximum.accumulate(cum_pnl)
                    drawdown = (running_max - cum_pnl) / np.maximum(running_max, 1e-8)
                    value = np.max(drawdown)
                else:
                    value = 0.0
            else:
                raise ValueError(f"Unknown metric: {metric}")
                
            bootstrapped_metrics.append(value)
        
        if not bootstrapped_metrics:
            return np.array([0.0, 0.0, 0.0])
            
        return np.percentile(bootstrapped_metrics, [2.5, 50, 97.5])
        
    except Exception as e:
        logger.error(f"Error in block bootstrap: {e}")
        return np.array([0.0, 0.0, 0.0])


class PurgedCVEvaluator:
    """
    Utility class for running purged cross-validation evaluation.
    """
    
    def __init__(self, n_splits: int = 5, purge_pct: float = 0.01, embargo_pct: float = 0.01):
        self.cv = PurgedKFold(n_splits=n_splits, purge_pct=purge_pct, embargo_pct=embargo_pct)
        
    def evaluate_strategy(self, X: np.ndarray, y: np.ndarray, timestamps: pd.Series, 
                         strategy_func, **kwargs) -> dict:
        """
        Evaluate a trading strategy using purged cross-validation.
        
        Args:
            X: Feature matrix
            y: Target returns
            timestamps: Timestamp series for purging
            strategy_func: Function that takes (X_train, y_train, X_test) and returns predictions
            **kwargs: Additional arguments for strategy_func
            
        Returns:
            Dictionary with evaluation metrics
        """
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(self.cv.split(X, y, t1=timestamps)):
            try:
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Run strategy
                predictions = strategy_func(X_train, y_train, X_test, **kwargs)
                
                # Calculate returns
                strategy_returns = predictions * y_test
                
                # Calculate metrics
                fold_metrics = {
                    'fold': fold,
                    'total_return': np.sum(strategy_returns),
                    'sharpe': self._calculate_sharpe(strategy_returns),
                    'max_drawdown': self._calculate_max_drawdown(strategy_returns),
                    'win_rate': np.mean(strategy_returns > 0),
                    'n_trades': len(strategy_returns),
                    'n_train': len(train_idx),
                    'n_test': len(test_idx)
                }
                
                fold_results.append(fold_metrics)
                
            except Exception as e:
                logger.error(f"Error in fold {fold}: {e}")
                continue
        
        # Aggregate results
        if not fold_results:
            return {'error': 'No successful folds'}
            
        return self._aggregate_results(fold_results)
    
    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns) * np.sqrt(252 * 390)
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0
        cum_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (running_max - cum_returns) / np.maximum(running_max, 1e-8)
        return np.max(drawdown)
    
    def _aggregate_results(self, fold_results: list) -> dict:
        """Aggregate results across folds."""
        metrics = ['total_return', 'sharpe', 'max_drawdown', 'win_rate']
        
        aggregated = {
            'n_folds': len(fold_results),
            'fold_results': fold_results
        }
        
        for metric in metrics:
            values = [fold[metric] for fold in fold_results if metric in fold]
            if values:
                aggregated[f'{metric}_mean'] = np.mean(values)
                aggregated[f'{metric}_std'] = np.std(values)
                aggregated[f'{metric}_min'] = np.min(values)
                aggregated[f'{metric}_max'] = np.max(values)
        
        return aggregated
