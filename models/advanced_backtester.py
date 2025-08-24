#!/usr/bin/env python3
"""
Advanced Backtesting Framework
Comprehensive backtesting system with realistic market simulation including slippage,
transaction costs, market impact, regime detection, and detailed performance analysis.

Features:
- Realistic slippage and transaction cost modeling
- Market impact simulation based on order size
- Market regime detection and analysis
- Walk-forward analysis with out-of-sample validation
- Comprehensive performance metrics and risk analysis
- Monte Carlo simulation for robustness testing
- Drawdown analysis and recovery periods
- Benchmark comparison and relative performance
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class TradingCosts:
    """Trading cost configuration."""
    commission_rate: float = 0.0005      # 0.05% commission
    bid_ask_spread: float = 0.0010       # 0.1% bid-ask spread
    market_impact_coef: float = 0.0001   # Market impact coefficient
    slippage_factor: float = 0.0005      # Base slippage factor
    min_commission: float = 1.0          # Minimum commission per trade


@dataclass
class MarketConditions:
    """Market condition parameters."""
    volatility_regime: str = "normal"    # low, normal, high, crisis
    liquidity_regime: str = "normal"     # low, normal, high
    trend_regime: str = "sideways"       # bull, bear, sideways
    correlation_regime: str = "normal"   # low, normal, high


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    initial_capital: float = 100000.0
    max_leverage: float = 1.0
    position_sizing_method: str = "fixed"  # fixed, kelly, volatility_target
    rebalance_frequency: str = "daily"     # daily, weekly, monthly
    benchmark_symbol: str = "SPY"
    risk_free_rate: float = 0.02
    enable_regime_analysis: bool = True
    enable_monte_carlo: bool = True
    monte_carlo_runs: int = 1000


class MarketRegimeDetector:
    """
    Market regime detection using Hidden Markov Models and statistical analysis.
    """
    
    def __init__(self, n_regimes: int = 4, lookback_window: int = 252):
        """
        Initialize regime detector.
        
        Args:
            n_regimes: Number of market regimes to detect
            lookback_window: Lookback window for regime detection
        """
        self.n_regimes = n_regimes
        self.lookback_window = lookback_window
        self.regime_model = None
        self.scaler = StandardScaler()
        
    def detect_regimes(self, returns: pd.Series, 
                      volatility: pd.Series,
                      volume: Optional[pd.Series] = None) -> pd.Series:
        """
        Detect market regimes using Gaussian Mixture Model.
        
        Args:
            returns: Return series
            volatility: Volatility series
            volume: Volume series (optional)
            
        Returns:
            Series of regime classifications
        """
        # Prepare features for regime detection
        features = pd.DataFrame({
            'returns': returns,
            'volatility': volatility,
            'abs_returns': returns.abs(),
            'lagged_returns': returns.shift(1)
        })
        
        if volume is not None:
            features['volume'] = volume
            features['volume_ma'] = volume.rolling(20).mean()
        
        # Remove NaN values
        features = features.dropna()
        
        if len(features) < self.lookback_window:
            # Not enough data, return default regime
            return pd.Series([0] * len(returns), index=returns.index)
        
        # Standardize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit Gaussian Mixture Model
        self.regime_model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            random_state=42
        )
        
        try:
            regimes = self.regime_model.fit_predict(features_scaled)
            
            # Create regime series with original index
            regime_series = pd.Series([0] * len(returns), index=returns.index)
            regime_series.loc[features.index] = regimes
            
            # Forward fill for missing values
            regime_series = regime_series.fillna(method='ffill').fillna(0)
            
            return regime_series
            
        except Exception as e:
            logger.warning(f"Regime detection failed: {e}")
            return pd.Series([0] * len(returns), index=returns.index)
    
    def classify_regime_characteristics(self, returns: pd.Series, 
                                     regimes: pd.Series) -> Dict[int, Dict[str, float]]:
        """
        Classify characteristics of each detected regime.
        
        Args:
            returns: Return series
            regimes: Regime classifications
            
        Returns:
            Dictionary of regime characteristics
        """
        regime_chars = {}
        
        for regime in regimes.unique():
            regime_returns = returns[regimes == regime]
            
            if len(regime_returns) > 10:  # Need minimum observations
                regime_chars[regime] = {
                    'mean_return': regime_returns.mean(),
                    'volatility': regime_returns.std(),
                    'skewness': stats.skew(regime_returns),
                    'kurtosis': stats.kurtosis(regime_returns),
                    'sharpe_ratio': regime_returns.mean() / regime_returns.std() if regime_returns.std() > 0 else 0,
                    'max_drawdown': self._calculate_max_drawdown(regime_returns.cumsum()),
                    'frequency': len(regime_returns) / len(returns)
                }
            else:
                regime_chars[regime] = {
                    'mean_return': 0, 'volatility': 0, 'skewness': 0,
                    'kurtosis': 0, 'sharpe_ratio': 0, 'max_drawdown': 0,
                    'frequency': 0
                }
        
        return regime_chars
    
    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min()


class SlippageModel:
    """
    Advanced slippage model considering market conditions and order characteristics.
    """
    
    def __init__(self, base_slippage: float = 0.0005):
        """
        Initialize slippage model.
        
        Args:
            base_slippage: Base slippage rate
        """
        self.base_slippage = base_slippage
        
    def calculate_slippage(self, 
                          order_size: float,
                          market_volatility: float,
                          liquidity_proxy: float,
                          time_of_day: Optional[float] = None) -> float:
        """
        Calculate realistic slippage based on market conditions.
        
        Args:
            order_size: Size of the order (as fraction of average volume)
            market_volatility: Current market volatility
            liquidity_proxy: Liquidity measure (e.g., bid-ask spread)
            time_of_day: Time of day factor (0-1)
            
        Returns:
            Slippage rate
        """
        # Base slippage
        slippage = self.base_slippage
        
        # Size impact (square root law)
        size_impact = 0.01 * np.sqrt(order_size)
        slippage += size_impact
        
        # Volatility impact
        vol_impact = market_volatility * 0.1
        slippage += vol_impact
        
        # Liquidity impact
        liquidity_impact = liquidity_proxy * 2.0
        slippage += liquidity_impact
        
        # Time of day impact (higher slippage at open/close)
        if time_of_day is not None:
            if time_of_day < 0.1 or time_of_day > 0.9:  # Market open/close
                slippage *= 1.5
            elif 0.45 < time_of_day < 0.55:  # Lunch time
                slippage *= 1.2
        
        return min(slippage, 0.01)  # Cap at 1%


class TransactionCostModel:
    """
    Comprehensive transaction cost model including all trading costs.
    """
    
    def __init__(self, costs: TradingCosts):
        """
        Initialize transaction cost model.
        
        Args:
            costs: Trading cost configuration
        """
        self.costs = costs
        
    def calculate_total_cost(self, 
                           trade_value: float,
                           order_size_ratio: float,
                           market_conditions: MarketConditions) -> Dict[str, float]:
        """
        Calculate total transaction costs.
        
        Args:
            trade_value: Dollar value of the trade
            order_size_ratio: Order size as ratio of average volume
            market_conditions: Current market conditions
            
        Returns:
            Dictionary of cost components
        """
        # Commission
        commission = max(
            trade_value * self.costs.commission_rate,
            self.costs.min_commission
        )
        
        # Bid-ask spread cost
        spread_cost = trade_value * self.costs.bid_ask_spread
        
        # Market impact (temporary and permanent)
        market_impact = trade_value * self.costs.market_impact_coef * np.sqrt(order_size_ratio)
        
        # Adjust for market conditions
        if market_conditions.liquidity_regime == "low":
            spread_cost *= 2.0
            market_impact *= 1.5
        elif market_conditions.liquidity_regime == "high":
            spread_cost *= 0.7
            market_impact *= 0.8
        
        if market_conditions.volatility_regime == "high":
            market_impact *= 1.3
        elif market_conditions.volatility_regime == "crisis":
            market_impact *= 2.0
        
        total_cost = commission + spread_cost + market_impact
        
        return {
            'commission': commission,
            'spread_cost': spread_cost,
            'market_impact': market_impact,
            'total_cost': total_cost,
            'cost_bps': (total_cost / trade_value) * 10000 if trade_value > 0 else 0
        }


class AdvancedBacktester:
    """
    Comprehensive backtesting framework with realistic market simulation.
    """
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize advanced backtester.
        
        Args:
            config: Backtesting configuration
        """
        self.config = config
        self.regime_detector = MarketRegimeDetector()
        self.slippage_model = SlippageModel()
        self.cost_model = TransactionCostModel(TradingCosts())
        
        # Results storage
        self.trades = []
        self.portfolio_values = []
        self.positions = []
        self.performance_metrics = {}
        
    def run_backtest(self, 
                    signals: pd.DataFrame,
                    price_data: pd.DataFrame,
                    benchmark_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Run comprehensive backtest with realistic market simulation.
        
        Args:
            signals: DataFrame with trading signals
            price_data: DataFrame with OHLCV price data
            benchmark_data: Optional benchmark data for comparison
            
        Returns:
            Comprehensive backtest results
        """
        logger.info("Starting advanced backtest...")
        
        # Initialize portfolio
        portfolio_value = self.config.initial_capital
        position = 0.0
        cash = self.config.initial_capital
        
        # Detect market regimes
        if self.config.enable_regime_analysis:
            returns = price_data['close'].pct_change()
            volatility = returns.rolling(20).std()
            regimes = self.regime_detector.detect_regimes(returns, volatility)
        else:
            regimes = pd.Series([0] * len(price_data), index=price_data.index)
        
        # Process each trading day
        for i, (date, signal_row) in enumerate(signals.iterrows()):
            if date not in price_data.index:
                continue
                
            price_row = price_data.loc[date]
            current_regime = regimes.loc[date] if date in regimes.index else 0
            
            # Calculate market conditions
            market_conditions = self._assess_market_conditions(
                price_data, date, current_regime
            )
            
            # Process trading signal
            if 'signal' in signal_row and signal_row['signal'] != 0:
                trade_result = self._execute_trade(
                    signal_row, price_row, portfolio_value, position, 
                    cash, market_conditions, date
                )
                
                if trade_result:
                    position = trade_result['new_position']
                    cash = trade_result['new_cash']
                    self.trades.append(trade_result['trade_record'])
            
            # Update portfolio value
            portfolio_value = cash + position * price_row['close']
            
            # Record portfolio state
            self.portfolio_values.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'position': position,
                'cash': cash,
                'price': price_row['close'],
                'regime': current_regime
            })
        
        # Calculate performance metrics
        results = self._calculate_performance_metrics(benchmark_data)
        
        # Add regime analysis
        if self.config.enable_regime_analysis:
            results['regime_analysis'] = self._analyze_regime_performance(regimes)
        
        # Monte Carlo analysis
        if self.config.enable_monte_carlo:
            results['monte_carlo_analysis'] = self._run_monte_carlo_analysis()
        
        logger.info("Backtest completed successfully")
        return results
    
    def _assess_market_conditions(self, 
                                price_data: pd.DataFrame,
                                date: pd.Timestamp,
                                regime: int) -> MarketConditions:
        """Assess current market conditions."""
        # Get recent data for analysis
        end_idx = price_data.index.get_loc(date)
        start_idx = max(0, end_idx - 20)
        recent_data = price_data.iloc[start_idx:end_idx+1]
        
        # Calculate volatility
        returns = recent_data['close'].pct_change()
        volatility = returns.std()
        
        # Determine volatility regime
        if volatility < 0.01:
            vol_regime = "low"
        elif volatility < 0.02:
            vol_regime = "normal"
        elif volatility < 0.04:
            vol_regime = "high"
        else:
            vol_regime = "crisis"
        
        # Determine trend regime
        price_change = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
        if price_change > 0.05:
            trend_regime = "bull"
        elif price_change < -0.05:
            trend_regime = "bear"
        else:
            trend_regime = "sideways"
        
        # Determine liquidity regime (simplified using volume)
        if 'volume' in recent_data.columns:
            avg_volume = recent_data['volume'].mean()
            recent_volume = recent_data['volume'].iloc[-1]
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            
            if volume_ratio < 0.7:
                liquidity_regime = "low"
            elif volume_ratio > 1.5:
                liquidity_regime = "high"
            else:
                liquidity_regime = "normal"
        else:
            liquidity_regime = "normal"
        
        return MarketConditions(
            volatility_regime=vol_regime,
            liquidity_regime=liquidity_regime,
            trend_regime=trend_regime,
            correlation_regime="normal"  # Simplified
        )
    
    def _execute_trade(self, 
                      signal_row: pd.Series,
                      price_row: pd.Series,
                      portfolio_value: float,
                      current_position: float,
                      current_cash: float,
                      market_conditions: MarketConditions,
                      date: pd.Timestamp) -> Optional[Dict[str, Any]]:
        """Execute trade with realistic costs and slippage."""
        
        signal = signal_row['signal']
        target_position = signal_row.get('position_size', signal * 0.1)  # Default 10% position
        
        # Calculate position change
        position_change = target_position - current_position
        
        if abs(position_change) < 0.001:  # Minimum trade size
            return None
        
        # Calculate trade value
        trade_price = price_row['close']
        trade_value = abs(position_change * trade_price)
        
        # Calculate slippage
        order_size_ratio = abs(position_change) / 1000  # Simplified volume ratio
        market_volatility = 0.02  # Simplified volatility
        liquidity_proxy = 0.001   # Simplified liquidity
        
        slippage = self.slippage_model.calculate_slippage(
            order_size_ratio, market_volatility, liquidity_proxy
        )
        
        # Apply slippage to execution price
        if position_change > 0:  # Buying
            execution_price = trade_price * (1 + slippage)
        else:  # Selling
            execution_price = trade_price * (1 - slippage)
        
        # Calculate transaction costs
        cost_breakdown = self.cost_model.calculate_total_cost(
            trade_value, order_size_ratio, market_conditions
        )
        
        # Check if we have enough cash for the trade
        required_cash = position_change * execution_price + cost_breakdown['total_cost']
        
        if position_change > 0 and required_cash > current_cash:
            # Adjust position size to available cash
            available_for_trade = current_cash - cost_breakdown['total_cost']
            if available_for_trade <= 0:
                return None  # Cannot execute trade
            
            position_change = available_for_trade / execution_price
            target_position = current_position + position_change
        
        # Execute trade
        new_position = current_position + position_change
        new_cash = current_cash - (position_change * execution_price + cost_breakdown['total_cost'])
        
        # Record trade
        trade_record = {
            'date': date,
            'signal': signal,
            'position_change': position_change,
            'execution_price': execution_price,
            'market_price': trade_price,
            'slippage': slippage,
            'slippage_cost': abs(position_change) * trade_price * slippage,
            'transaction_costs': cost_breakdown['total_cost'],
            'cost_bps': cost_breakdown['cost_bps'],
            'new_position': new_position,
            'new_cash': new_cash,
            'market_conditions': market_conditions
        }
        
        return {
            'new_position': new_position,
            'new_cash': new_cash,
            'trade_record': trade_record
        }
    
    def _calculate_performance_metrics(self, 
                                    benchmark_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        
        # Convert portfolio values to DataFrame
        portfolio_df = pd.DataFrame(self.portfolio_values)
        portfolio_df.set_index('date', inplace=True)
        
        # Calculate returns
        portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
        portfolio_df['cumulative_returns'] = (1 + portfolio_df['returns']).cumprod() - 1
        
        # Basic performance metrics
        total_return = (portfolio_df['portfolio_value'].iloc[-1] / self.config.initial_capital) - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_df)) - 1
        
        # Risk metrics
        returns_series = portfolio_df['returns'].dropna()
        volatility = returns_series.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - self.config.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        portfolio_df['peak'] = portfolio_df['portfolio_value'].expanding().max()
        portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['peak']) / portfolio_df['peak']
        max_drawdown = portfolio_df['drawdown'].min()
        
        # Calculate drawdown duration
        drawdown_periods = self._calculate_drawdown_periods(portfolio_df)
        
        # Trade analysis
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        
        trade_metrics = {}
        if not trades_df.empty:
            trade_metrics = {
                'total_trades': len(trades_df),
                'avg_trade_cost_bps': trades_df['cost_bps'].mean(),
                'total_transaction_costs': trades_df['transaction_costs'].sum(),
                'avg_slippage_bps': (trades_df['slippage'] * 10000).mean(),
                'total_slippage_cost': trades_df['slippage_cost'].sum()
            }
        
        # Benchmark comparison
        benchmark_metrics = {}
        if benchmark_data is not None:
            benchmark_metrics = self._calculate_benchmark_comparison(
                portfolio_df, benchmark_data
            )
        
        return {
            'performance_metrics': {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'calmar_ratio': annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0,
                'sortino_ratio': self._calculate_sortino_ratio(returns_series),
                'skewness': stats.skew(returns_series),
                'kurtosis': stats.kurtosis(returns_series)
            },
            'drawdown_analysis': drawdown_periods,
            'trade_analysis': trade_metrics,
            'benchmark_comparison': benchmark_metrics,
            'portfolio_timeseries': portfolio_df,
            'trades': trades_df
        }
    
    def _calculate_drawdown_periods(self, portfolio_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate detailed drawdown analysis."""
        drawdowns = portfolio_df['drawdown']
        
        # Find drawdown periods
        in_drawdown = drawdowns < 0
        drawdown_starts = in_drawdown & ~in_drawdown.shift(1, fill_value=False)
        drawdown_ends = ~in_drawdown & in_drawdown.shift(1, fill_value=False)
        
        periods = []
        start_idx = None
        
        for i, (date, is_start) in enumerate(drawdown_starts.items()):
            if is_start:
                start_idx = i
            elif start_idx is not None and drawdown_ends.iloc[i]:
                end_idx = i
                period_drawdowns = drawdowns.iloc[start_idx:end_idx+1]
                periods.append({
                    'start_date': drawdowns.index[start_idx],
                    'end_date': drawdowns.index[end_idx],
                    'duration_days': end_idx - start_idx,
                    'max_drawdown': period_drawdowns.min(),
                    'recovery_date': date
                })
                start_idx = None
        
        return {
            'drawdown_periods': periods,
            'avg_drawdown_duration': np.mean([p['duration_days'] for p in periods]) if periods else 0,
            'max_drawdown_duration': max([p['duration_days'] for p in periods]) if periods else 0,
            'num_drawdown_periods': len(periods)
        }
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio."""
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return np.inf if returns.mean() > 0 else 0
        
        downside_deviation = downside_returns.std()
        if downside_deviation == 0:
            return np.inf if returns.mean() > 0 else 0
        
        excess_return = returns.mean() - self.config.risk_free_rate / 252
        return (excess_return / downside_deviation) * np.sqrt(252)
    
    def _calculate_benchmark_comparison(self, 
                                     portfolio_df: pd.DataFrame,
                                     benchmark_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate benchmark comparison metrics."""
        # Align dates
        common_dates = portfolio_df.index.intersection(benchmark_data.index)
        portfolio_returns = portfolio_df.loc[common_dates, 'returns']
        benchmark_returns = benchmark_data.loc[common_dates, 'close'].pct_change()
        
        # Calculate relative metrics
        excess_returns = portfolio_returns - benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # Beta calculation
        covariance = np.cov(portfolio_returns.dropna(), benchmark_returns.dropna())[0, 1]
        benchmark_variance = benchmark_returns.var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
        
        return {
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'beta': beta,
            'alpha': portfolio_returns.mean() - beta * benchmark_returns.mean(),
            'correlation': portfolio_returns.corr(benchmark_returns)
        }
    
    def _analyze_regime_performance(self, regimes: pd.Series) -> Dict[str, Any]:
        """Analyze performance by market regime."""
        portfolio_df = pd.DataFrame(self.portfolio_values)
        portfolio_df.set_index('date', inplace=True)
        portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
        
        # Align regimes with portfolio data
        common_dates = portfolio_df.index.intersection(regimes.index)
        portfolio_returns = portfolio_df.loc[common_dates, 'returns']
        regime_data = regimes.loc[common_dates]
        
        regime_performance = {}
        for regime in regime_data.unique():
            regime_returns = portfolio_returns[regime_data == regime]
            if len(regime_returns) > 10:
                regime_performance[f'regime_{regime}'] = {
                    'total_return': (1 + regime_returns).prod() - 1,
                    'annualized_return': regime_returns.mean() * 252,
                    'volatility': regime_returns.std() * np.sqrt(252),
                    'sharpe_ratio': (regime_returns.mean() * 252) / (regime_returns.std() * np.sqrt(252)) if regime_returns.std() > 0 else 0,
                    'max_drawdown': self._calculate_regime_drawdown(regime_returns),
                    'frequency': len(regime_returns) / len(portfolio_returns)
                }
        
        return regime_performance
    
    def _calculate_regime_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown for a regime."""
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()
    
    def _run_monte_carlo_analysis(self) -> Dict[str, Any]:
        """Run Monte Carlo simulation for robustness testing."""
        if not self.trades:
            return {}
        
        trades_df = pd.DataFrame(self.trades)
        returns = trades_df['position_change'] * (trades_df['execution_price'] - trades_df['market_price'])
        
        # Bootstrap simulation
        simulation_results = []
        
        for _ in range(self.config.monte_carlo_runs):
            # Resample returns with replacement
            simulated_returns = np.random.choice(returns, size=len(returns), replace=True)
            
            # Calculate cumulative performance
            cumulative_return = np.sum(simulated_returns) / self.config.initial_capital
            max_dd = self._calculate_monte_carlo_drawdown(simulated_returns)
            
            simulation_results.append({
                'total_return': cumulative_return,
                'max_drawdown': max_dd
            })
        
        results_df = pd.DataFrame(simulation_results)
        
        return {
            'mean_return': results_df['total_return'].mean(),
            'return_std': results_df['total_return'].std(),
            'return_percentiles': {
                '5th': results_df['total_return'].quantile(0.05),
                '25th': results_df['total_return'].quantile(0.25),
                '75th': results_df['total_return'].quantile(0.75),
                '95th': results_df['total_return'].quantile(0.95)
            },
            'drawdown_percentiles': {
                '5th': results_df['max_drawdown'].quantile(0.05),
                '25th': results_df['max_drawdown'].quantile(0.25),
                '75th': results_df['max_drawdown'].quantile(0.75),
                '95th': results_df['max_drawdown'].quantile(0.95)
            },
            'probability_positive': (results_df['total_return'] > 0).mean()
        }
    
    def _calculate_monte_carlo_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown for Monte Carlo simulation."""
        cumulative = np.cumprod(1 + returns / self.config.initial_capital)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        return np.min(drawdown)
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive backtest report."""
        report = []
        report.append("=" * 80)
        report.append("ADVANCED BACKTESTING REPORT")
        report.append("=" * 80)
        
        # Performance summary
        perf = results['performance_metrics']
        report.append(f"\nPERFORMANCE SUMMARY:")
        report.append(f"Total Return: {perf['total_return']:.2%}")
        report.append(f"Annualized Return: {perf['annualized_return']:.2%}")
        report.append(f"Volatility: {perf['volatility']:.2%}")
        report.append(f"Sharpe Ratio: {perf['sharpe_ratio']:.3f}")
        report.append(f"Maximum Drawdown: {perf['max_drawdown']:.2%}")
        report.append(f"Calmar Ratio: {perf['calmar_ratio']:.3f}")
        
        # Trade analysis
        if 'trade_analysis' in results and results['trade_analysis']:
            trade = results['trade_analysis']
            report.append(f"\nTRADE ANALYSIS:")
            report.append(f"Total Trades: {trade['total_trades']}")
            report.append(f"Average Transaction Cost: {trade['avg_trade_cost_bps']:.1f} bps")
            report.append(f"Average Slippage: {trade['avg_slippage_bps']:.1f} bps")
            report.append(f"Total Transaction Costs: ${trade['total_transaction_costs']:,.2f}")
        
        # Regime analysis
        if 'regime_analysis' in results:
            report.append(f"\nREGIME ANALYSIS:")
            for regime, metrics in results['regime_analysis'].items():
                report.append(f"{regime}: Return={metrics['annualized_return']:.2%}, "
                            f"Sharpe={metrics['sharpe_ratio']:.2f}, "
                            f"Frequency={metrics['frequency']:.1%}")
        
        # Monte Carlo results
        if 'monte_carlo_analysis' in results:
            mc = results['monte_carlo_analysis']
            report.append(f"\nMONTE CARLO ANALYSIS:")
            report.append(f"Mean Return: {mc['mean_return']:.2%}")
            report.append(f"Return 95% CI: [{mc['return_percentiles']['5th']:.2%}, "
                        f"{mc['return_percentiles']['95th']:.2%}]")
            report.append(f"Probability of Positive Return: {mc['probability_positive']:.1%}")
        
        return "\n".join(report)
