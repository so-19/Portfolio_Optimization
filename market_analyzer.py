import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import yfinance as yf
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
import xlwings as xw
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

class MarketRegimeAnalyzer:
    def __init__(self, market_ticker='SPY', start_date=None, end_date=None, lookback_window=252, regime_count=3):
        self.market_ticker = market_ticker
        self.start_date = start_date
        self.end_date = end_date
        self.lookback_window = lookback_window
        self.regime_count = regime_count
        self.market_data = None
        self.returns = None
        self.volatility = None
        self.regimes = None
        
    def fetch_data(self):
        self.market_data = yf.download(self.market_ticker, start=self.start_date, end=self.end_date)['Close']
        return self.market_data
    
    def calculate_returns_and_volatility(self, returns_window=1, vol_window=21):
        self.returns = np.log(self.market_data / self.market_data.shift(returns_window)).dropna()
        self.volatility = self.returns.rolling(window=vol_window).std() * np.sqrt(252)
        return self.returns, self.volatility
    
    def identify_regimes_gmm(self):
        # First ensure we have returns and volatility calculated
        if self.returns is None or self.volatility is None:
            self.calculate_returns_and_volatility()
        
        # Ensure data is 1-dimensional
        returns_1d = self.returns.values.flatten() if isinstance(self.returns, pd.DataFrame) else self.returns
        volatility_1d = self.volatility.values.flatten() if isinstance(self.volatility, pd.DataFrame) else self.volatility
        
        # Create DataFrame with proper index and 1D data
        df = pd.DataFrame({
            'returns': returns_1d,
            'volatility': volatility_1d
        }, index=self.returns.index).dropna()
        
        X = np.column_stack([df['returns'].values, df['volatility'].values])
        
        gmm = GaussianMixture(n_components=self.regime_count, covariance_type='full', random_state=42)
        gmm.fit(X)
        
        df['regime'] = gmm.predict(X)
        
        regime_stats = {}
        for regime in range(self.regime_count):
            regime_data = df[df['regime'] == regime]
            regime_stats[regime] = {
                'mean_return': regime_data['returns'].mean() * 252,
                'mean_volatility': regime_data['volatility'].mean(),
                'count': len(regime_data),
                'pct': len(regime_data) / len(df),
                'sharpe': (regime_data['returns'].mean() * 252) / regime_data['volatility'].mean() if regime_data['volatility'].mean() > 0 else 0
            }
        
        self.regimes = df['regime']
        
        return df, regime_stats
    
    def identify_regimes_kmeans(self):
        df = pd.DataFrame({
            'returns': self.returns,
            'volatility': self.volatility
        }).dropna()
        
        X = np.column_stack([df['returns'].values, df['volatility'].values])
        
        kmeans = KMeans(n_clusters=self.regime_count, random_state=42)
        kmeans.fit(X)
        
        df['regime'] = kmeans.predict(X)
        
        regime_stats = {}
        for regime in range(self.regime_count):
            regime_data = df[df['regime'] == regime]
            regime_stats[regime] = {
                'mean_return': regime_data['returns'].mean() * 252,
                'mean_volatility': regime_data['volatility'].mean(),
                'count': len(regime_data),
                'pct': len(regime_data) / len(df),
                'sharpe': (regime_data['returns'].mean() * 252) / regime_data['volatility'].mean() if regime_data['volatility'].mean() > 0 else 0
            }
        
        self.regimes = df['regime']
        
        return df, regime_stats
    
    def forecast_regime_transition(self, horizon=21):
        if self.regimes is None:
            raise ValueError("Must identify regimes first")
        
        regime_series = pd.Series(self.regimes)
        
        transition_matrix = np.zeros((self.regime_count, self.regime_count))
        
        for i in range(len(regime_series)-1):
            from_regime = regime_series.iloc[i]
            to_regime = regime_series.iloc[i+1]
            transition_matrix[from_regime, to_regime] += 1
        
        for i in range(self.regime_count):
            if transition_matrix[i].sum() > 0:
                transition_matrix[i] = transition_matrix[i] / transition_matrix[i].sum()
        
        current_regime = regime_series.iloc[-1]
        current_distribution = np.zeros(self.regime_count)
        current_distribution[current_regime] = 1
        
        forecast = []
        for _ in range(horizon):
            current_distribution = np.dot(current_distribution, transition_matrix)
            forecast.append(current_distribution.copy())
        
        return pd.DataFrame(forecast), transition_matrix
    
    def plot_regimes(self):
        if self.regimes is None:
            raise ValueError("Must identify regimes first")
            
        df = pd.DataFrame({
            'price': self.market_data,
            'returns': self.returns,
            'volatility': self.volatility,
            'regime': self.regimes
        }).dropna()
        
        fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        colors = ['green', 'yellow', 'red']
        if self.regime_count > 3:
            colors = plt.cm.tab10(np.linspace(0, 1, self.regime_count))
        
        for i, regime in enumerate(df['regime'].unique()):
            regime_data = df[df['regime'] == regime]
            axs[0].scatter(regime_data.index, regime_data['returns'], 
                          color=colors[i], alpha=0.5, label=f'Regime {regime}')
            axs[1].scatter(regime_data.index, regime_data['volatility'], 
                          color=colors[i], alpha=0.5, label=f'Regime {regime}')
        
        for regime in df['regime'].unique():
            mask = df['regime'] == regime
            axs[2].plot(df.index, df['price'], 'lightgray', linewidth=1)
            axs[2].scatter(df.index[mask], df['price'][mask], color=colors[regime], s=20, alpha=0.7, label=f'Regime {regime}')
        
        axs[0].set_title('Returns by Market Regime')
        axs[0].set_ylabel('Daily Returns')
        axs[0].legend()
        axs[0].grid(True)
        
        axs[1].set_title('Volatility by Market Regime')
        axs[1].set_ylabel('Annualized Volatility')
        axs[1].legend()
        axs[1].grid(True)
        
        axs[2].set_title('Price with Market Regimes')
        axs[2].set_ylabel('Price')
        axs[2].legend()
        axs[2].grid(True)
        
        plt.tight_layout()
        return fig

    def export_regimes_to_excel(self, file_path):
        if self.regimes is None:
            raise ValueError("Must identify regimes first")
            
        df = pd.DataFrame({
            'price': self.market_data,
            'returns': self.returns,
            'volatility': self.volatility,
            'regime': self.regimes
        }).dropna()
        
        wb = xw.Book()
        
        sht = wb.sheets.add('Market Regimes')
        
        sht.range('A1').value = 'Market Regime Analysis'
        sht.range('A1').font.bold = True
        sht.range('A1').font.size = 16
        
        regime_stats = {}
        for regime in df['regime'].unique():
            regime_data = df[df['regime'] == regime]
            regime_stats[regime] = {
                'mean_return': regime_data['returns'].mean() * 252,
                'mean_volatility': regime_data['volatility'].mean(),
                'count': len(regime_data),
                'pct': len(regime_data) / len(df),
                'sharpe': (regime_data['returns'].mean() * 252) / regime_data['volatility'].mean() if regime_data['volatility'].mean() > 0 else 0
            }
        
        sht.range('A3').value = 'Regime Statistics'
        headers = ['Regime', 'Mean Annual Return', 'Mean Volatility', 'Count', 'Percentage', 'Sharpe Ratio']
        sht.range('A4').value = [headers]
        
        rows = []
        for regime, stats in regime_stats.items():
            rows.append([
                f'Regime {regime}',
                stats['mean_return'],
                stats['mean_volatility'],
                stats['count'],
                stats['pct'],
                stats['sharpe']
            ])
        
        sht.range('A5').value = rows
        sht.range('A4:F4').font.bold = True
        sht.range('B5:C' + str(4 + len(regime_stats))).number_format = '0.00%'
        sht.range('E5:E' + str(4 + len(regime_stats))).number_format = '0.00%'
        sht.range('F5:F' + str(4 + len(regime_stats))).number_format = '0.00'
        
        forecast, transition = self.forecast_regime_transition()
        
        sht.range('A' + str(8 + len(regime_stats))).value = 'Regime Transition Matrix'
        
        headers = ['From/To'] + [f'Regime {i}' for i in range(self.regime_count)]
        rows = []
        for i in range(self.regime_count):
            row = [f'Regime {i}'] + list(transition[i])
            rows.append(row)
        
        sht.range('A' + str(9 + len(regime_stats))).value = [headers]
        sht.range('A' + str(10 + len(regime_stats))).value = rows
        
        range_end = chr(65 + self.regime_count)
        sht.range(f'A{9 + len(regime_stats)}:{range_end}{9 + len(regime_stats)}').font.bold = True
        sht.range(f'B{10 + len(regime_stats)}:{range_end}{9 + len(regime_stats) + self.regime_count}').number_format = '0.00%'
        
        sht.range('A' + str(13 + len(regime_stats) + self.regime_count)).value = 'Regime Forecast (21 days)'
        
        headers = ['Day'] + [f'Prob Regime {i}' for i in range(self.regime_count)]
        rows = []
        for i in range(len(forecast)):
            row = [i+1] + list(forecast.iloc[i].values)
            rows.append(row)
        
        sht.range('A' + str(14 + len(regime_stats) + self.regime_count)).value = [headers]
        sht.range('A' + str(15 + len(regime_stats) + self.regime_count)).value = rows
        
        range_end = chr(65 + self.regime_count)
        sht.range(f'A{14 + len(regime_stats) + self.regime_count}:{range_end}{14 + len(regime_stats) + self.regime_count}').font.bold = True
        sht.range(f'B{15 + len(regime_stats) + self.regime_count}:{range_end}{14 + len(regime_stats) + self.regime_count + 21}').number_format = '0.00%'
        
        data_sheet = wb.sheets.add('Regime Data')
        data_sheet.range('A1').value = df.reset_index()
        
        wb.save(file_path)
        return file_path

class PortfolioBacktester:
    def __init__(self, portfolio_optimizer, start_date=None, end_date=None, rebalance_frequency='quarterly'):
        self.optimizer = portfolio_optimizer
        self.start_date = start_date
        self.end_date = end_date
        self.rebalance_frequency = rebalance_frequency
        self.backtest_data = None
        self.backtest_returns = None
        self.rebalancing_dates = None
        self.optimal_weights_history = []
        self.performance_history = None
        self.benchmark_data = None
        self.market_regimes = None
    
    def fetch_extended_data(self):
        tickers = self.optimizer.tickers
        extended_data = yf.download(tickers, start=self.start_date, end=self.end_date)['Close']
        self.backtest_data = extended_data
        return extended_data
    
    def calculate_returns(self):
        self.backtest_returns = np.log(self.backtest_data / self.backtest_data.shift(1)).dropna()
        return self.backtest_returns
    
    def set_rebalancing_dates(self):
        if self.rebalance_frequency == 'monthly':
            self.rebalancing_dates = pd.date_range(start=self.backtest_returns.index[0], 
                                                 end=self.backtest_returns.index[-1], 
                                                 freq='M')
        elif self.rebalance_frequency == 'quarterly':
            self.rebalancing_dates = pd.date_range(start=self.backtest_returns.index[0], 
                                                 end=self.backtest_returns.index[-1], 
                                                 freq='Q')
        elif self.rebalance_frequency == 'annual':
            self.rebalancing_dates = pd.date_range(start=self.backtest_returns.index[0], 
                                                 end=self.backtest_returns.index[-1], 
                                                 freq='A')
        else:
            raise ValueError("Rebalance frequency must be 'monthly', 'quarterly', or 'annual'")
        
        self.rebalancing_dates = [date for date in self.rebalancing_dates 
                                if date >= self.backtest_returns.index[0] and date <= self.backtest_returns.index[-1]]
        
        valid_rebalancing_dates = []
        for date in self.rebalancing_dates:
            nearest_dates = self.backtest_returns.index[self.backtest_returns.index >= date]
            if len(nearest_dates) > 0:
                valid_rebalancing_dates.append(nearest_dates[0])
        
        self.rebalancing_dates = valid_rebalancing_dates
        return self.rebalancing_dates
    
    def run_backtest(self, initial_investment=10000):
        if self.rebalancing_dates is None:
            self.set_rebalancing_dates()
        
        portfolio_value = pd.Series(index=self.backtest_returns.index)
        portfolio_value.iloc[0] = initial_investment
        
        current_weights = np.ones(len(self.optimizer.tickers)) / len(self.optimizer.tickers)
        
        lookback_window = 252
        
        self.optimal_weights_history = []
        
        for i, date in enumerate(self.backtest_returns.index):
            if i == 0:
                continue
                
            daily_return = np.sum(self.backtest_returns.iloc[i] * current_weights)
            portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 + daily_return)
            
            if date in self.rebalancing_dates:
                lookback_start_idx = max(0, i - lookback_window)
                
                lookback_returns = self.backtest_returns.iloc[lookback_start_idx:i]
                
                mean_returns = lookback_returns.mean() * 252
                cov_matrix = lookback_returns.cov() * 252
                
                self.optimizer.mean_returns = mean_returns
                self.optimizer.cov_matrix = cov_matrix
                
                optimal_portfolio = self.optimizer.optimize_sharpe_ratio()
                current_weights = optimal_portfolio['weights']
                
                self.optimal_weights_history.append({
                    'date': date,
                    'weights': current_weights,
                    'return': optimal_portfolio['return'],
                    'volatility': optimal_portfolio['volatility'],
                    'sharpe_ratio': optimal_portfolio['sharpe_ratio']
                })
        
        self.performance_history = portfolio_value
        
        return portfolio_value, self.optimal_weights_history
    
    def add_benchmark(self, benchmark_ticker='SPY'):
        benchmark_data = yf.download(benchmark_ticker, start=self.start_date, end=self.end_date)['Close']
        
        benchmark_returns = np.log(benchmark_data / benchmark_data.shift(1)).dropna()
        
        benchmark_value = pd.Series(index=benchmark_returns.index)
        benchmark_value.iloc[0] = self.performance_history.iloc[0]
        
        for i in range(1, len(benchmark_returns)):
            benchmark_value.iloc[i] = benchmark_value.iloc[i-1] * (1 + benchmark_returns.iloc[i])
        
        self.benchmark_data = benchmark_value
        
        return benchmark_value
    
    def add_market_regimes(self, regime_analyzer):
        if regime_analyzer.regimes is None:
            raise ValueError("Market regimes must be identified first")
        
        regime_df = pd.DataFrame({'regime': regime_analyzer.regimes})
        self.market_regimes = regime_df
        
        return regime_df
    
    def calculate_performance_metrics(self):
        portfolio_returns = self.performance_history.pct_change().dropna()
        
        metrics = {
            'total_return': (self.performance_history.iloc[-1] / self.performance_history.iloc[0]) - 1,
            'annualized_return': (1 + ((self.performance_history.iloc[-1] / self.performance_history.iloc[0]) - 1)) ** (252 / len(portfolio_returns)) - 1,
            'volatility': portfolio_returns.std() * np.sqrt(252),
            'sharpe_ratio': (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252)),
            'max_drawdown': self._calculate_max_drawdown(portfolio_returns),
            'var_95': self._calculate_var(portfolio_returns, 0.05),
            'cvar_95': self._calculate_cvar(portfolio_returns, 0.05),
            'kurtosis': portfolio_returns.kurtosis(),
            'skewness': portfolio_returns.skew()
        }
        
        if self.benchmark_data is not None:
            benchmark_returns = self.benchmark_data.pct_change().dropna()
            metrics['benchmark_total_return'] = (self.benchmark_data.iloc[-1] / self.benchmark_data.iloc[0]) - 1
            metrics['benchmark_annualized_return'] = (1 + ((self.benchmark_data.iloc[-1] / self.benchmark_data.iloc[0]) - 1)) ** (252 / len(benchmark_returns)) - 1
            metrics['benchmark_volatility'] = benchmark_returns.std() * np.sqrt(252)
            metrics['benchmark_sharpe_ratio'] = (benchmark_returns.mean() * 252) / (benchmark_returns.std() * np.sqrt(252))
            metrics['benchmark_max_drawdown'] = self._calculate_max_drawdown(benchmark_returns)
            
            metrics['alpha'] = self._calculate_alpha(portfolio_returns, benchmark_returns)
            metrics['beta'] = self._calculate_beta(portfolio_returns, benchmark_returns)
            metrics['tracking_error'] = self._calculate_tracking_error(portfolio_returns, benchmark_returns)
            metrics['information_ratio'] = metrics['alpha'] / metrics['tracking_error'] if metrics['tracking_error'] > 0 else 0
            metrics['up_capture'] = self._calculate_up_capture(portfolio_returns, benchmark_returns)
            metrics['down_capture'] = self._calculate_down_capture(portfolio_returns, benchmark_returns)
        
        if self.market_regimes is not None:
            regime_metrics = self._calculate_regime_performance(portfolio_returns)
            metrics['regime_performance'] = regime_metrics
        
        return metrics
    
    def _calculate_max_drawdown(self, returns):
        wealth_index = (1 + returns).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        return drawdowns.min()
    
    def _calculate_var(self, returns, alpha=0.05):
        return np.percentile(returns, alpha * 100)
    
    def _calculate_cvar(self, returns, alpha=0.05):
        var = self._calculate_var(returns, alpha)
        return returns[returns <= var].mean()
    
    def _calculate_alpha(self, portfolio_returns, benchmark_returns):
        X = benchmark_returns.values.reshape(-1, 1)
        X = np.concatenate([np.ones_like(X), X], axis=1)
        beta, alpha = np.linalg.lstsq(X, portfolio_returns.values, rcond=None)[0]
        return alpha * 252
    
    def _calculate_beta(self, portfolio_returns, benchmark_returns):
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        return covariance / benchmark_variance
    
    def _calculate_tracking_error(self, portfolio_returns, benchmark_returns):
        tracking_diff = portfolio_returns - benchmark_returns
        return tracking_diff.std() * np.sqrt(252)
    
    def _calculate_up_capture(self, portfolio_returns, benchmark_returns):
        up_market = benchmark_returns[benchmark_returns > 0]
        
        if len(up_market) == 0:
            return 0
            
        portfolio_up_market = portfolio_returns[benchmark_returns > 0]
        
        up_capture = (1 + portfolio_up_market).prod() ** (1 / len(up_market)) - 1
        benchmark_up = (1 + up_market).prod() ** (1 / len(up_market)) - 1
        
        return up_capture / benchmark_up if benchmark_up > 0 else 0
    
    def _calculate_down_capture(self, portfolio_returns, benchmark_returns):
        down_market = benchmark_returns[benchmark_returns < 0]
        
        if len(down_market) == 0:
            return 0
            
        portfolio_down_market = portfolio_returns[benchmark_returns < 0]
        
        down_capture = (1 + portfolio_down_market).prod() ** (1 / len(down_market)) - 1
        benchmark_down = (1 + down_market).prod() ** (1 / len(down_market)) - 1
        
        return down_capture / benchmark_down if benchmark_down < 0 else 0
    
    def _calculate_regime_performance(self, portfolio_returns):
        merged_data = pd.DataFrame({
            'returns': portfolio_returns,
            'regime': self.market_regimes['regime']
        }).dropna()
        
        regime_stats = {}
        for regime in merged_data['regime'].unique():
            regime_data = merged_data[merged_data['regime'] == regime]
            regime_returns = regime_data['returns']
            
            regime_stats[int(regime)] = {
                'mean_return': regime_returns.mean() * 252,
                'volatility': regime_returns.std() * np.sqrt(252),
                'count': len(regime_returns),
                'sharpe': (regime_returns.mean() * 252) / (regime_returns.std() * np.sqrt(252)) if regime_returns.std() > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(regime_returns)
            }
        
        return regime_stats
    
    def plot_performance(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.plot(self.performance_history, label='Portfolio')
        
        if self.benchmark_data is not None:
            ax.plot(self.benchmark_data, label='Benchmark')
        
        if self.market_regimes is not None:
            for i, regime in enumerate(sorted(self.market_regimes['regime'].unique())):
                regime_start_dates = self.market_regimes[self.market_regimes['regime'] == regime].index
                for start_date in regime_start_dates:
                    if start_date == self.market_regimes.index[-1]:
                        continue
                        
                    end_dates = self.market_regimes.index[self.market_regimes.index > start_date]
                    if len(end_dates) == 0:
                        end_date = self.market_regimes.index[-1]
                    else:
                        next_regime_start = end_dates[0]
                        prev_dates = self.market_regimes.index[self.market_regimes.index < next_regime_start]
                        end_date = prev_dates[-1] if len(prev_dates) > 0 else next_regime_start
                    
                    ax.axvspan(start_date, end_date, alpha=0.2, color=plt.cm.tab10(i % 10))
        
        ax.set_title('Portfolio Performance')
        ax.set_ylabel('Value ($)')
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def plot_weights_over_time(self):
        weights_df = pd.DataFrame(
            [weights['weights'] for weights in self.optimal_weights_history],
            index=[weights['date'] for weights in self.optimal_weights_history],
            columns=self.optimizer.tickers
        )
        
        fig, ax = plt.subplots(figsize=(12, 8))
        weights_df.plot(kind='area', stacked=True, ax=ax)
        
        ax.set_title('Portfolio Allocation Over Time')
        ax.set_ylabel('Weight')
        ax.set_xlabel('Date')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(True)
        
        return fig
    
    def export_results_to_excel(self, file_path):
        wb = xw.Book()
        
        # Performance summary sheet
        summary_sheet = wb.sheets.add('Performance Summary')
        summary_sheet.range('A1').value = 'Portfolio Backtest Results'
        summary_sheet.range('A1').font.bold = True
        summary_sheet.range('A1').font.size = 16
        
        metrics = self.calculate_performance_metrics()
        
        summary_sheet.range('A3').value = 'Portfolio Metrics'
        summary_sheet.range('A3').font.bold = True
        
        perf_metrics = [
            ['Total Return', metrics['total_return']],
            ['Annualized Return', metrics['annualized_return']],
            ['Volatility', metrics['volatility']],
            ['Sharpe Ratio', metrics['sharpe_ratio']],
            ['Max Drawdown', metrics['max_drawdown']],
            ['VaR (95%)', metrics['var_95']],
            ['CVaR (95%)', metrics['cvar_95']],
            ['Kurtosis', metrics['kurtosis']],
            ['Skewness', metrics['skewness']]
        ]
        
        summary_sheet.range('A4').value = perf_metrics
        summary_sheet.range('B4:B8').number_format = '0.00%'
        summary_sheet.range('B9:B10').number_format = '0.00'
        summary_sheet.range('B11:B12').number_format = '0.00'
        
        if 'benchmark_total_return' in metrics:
            summary_sheet.range('D3').value = 'Benchmark Comparison'
            summary_sheet.range('D3').font.bold = True
            
            benchmark_metrics = [
                ['Portfolio Return', metrics['annualized_return']],
                ['Benchmark Return', metrics['benchmark_annualized_return']],
                ['Alpha', metrics['alpha']],
                ['Beta', metrics['beta']],
                ['Tracking Error', metrics['tracking_error']],
                ['Information Ratio', metrics['information_ratio']],
                ['Up Capture', metrics['up_capture']],
                ['Down Capture', metrics['down_capture']]
            ]
            
            summary_sheet.range('D4').value = benchmark_metrics
            summary_sheet.range('E4:E5').number_format = '0.00%'
            summary_sheet.range('E6:E8').number_format = '0.00%'
            summary_sheet.range('E9').number_format = '0.00'
            summary_sheet.range('E10:E11').number_format = '0.00'
        
        if 'regime_performance' in metrics:
            summary_sheet.range('A15').value = 'Performance by Market Regime'
            summary_sheet.range('A15').font.bold = True
            
            headers = ['Regime', 'Mean Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Count']
            summary_sheet.range('A16').value = [headers]
            summary_sheet.range('A16:F16').font.bold = True
            
            rows = []
            for regime, stats in metrics['regime_performance'].items():
                rows.append([
                    f'Regime {regime}',
                    stats['mean_return'],
                    stats['volatility'],
                    stats['sharpe'],
                    stats['max_drawdown'],
                    stats['count']
                ])
            
            summary_sheet.range('A17').value = rows
            summary_sheet.range('B17:B' + str(16 + len(rows))).number_format = '0.00%'
            summary_sheet.range('C17:C' + str(16 + len(rows))).number_format = '0.00%'
            summary_sheet.range('D17:E' + str(16 + len(rows))).number_format = '0.00'
        
        # Portfolio weights sheet
        weights_sheet = wb.sheets.add('Portfolio Weights')
        
        weights_df = pd.DataFrame(
            [weights['weights'] for weights in self.optimal_weights_history],
            index=[weights['date'] for weights in self.optimal_weights_history],
            columns=self.optimizer.tickers
        )
        
        weights_sheet.range('A1').value = 'Portfolio Weights Over Time'
        weights_sheet.range('A1').font.bold = True
        weights_sheet.range('A1').font.size = 16
        
        weights_sheet.range('A3').value = weights_df.reset_index()
        
        # Performance data sheet
        perf_sheet = wb.sheets.add('Performance Data')
        
        perf_data = pd.DataFrame({
            'Portfolio': self.performance_history
        })
        
        if self.benchmark_data is not None:
            perf_data['Benchmark'] = self.benchmark_data
        
        if self.market_regimes is not None:
            perf_data['Regime'] = self.market_regimes
        
        perf_sheet.range('A1').value = 'Daily Performance Data'
        perf_sheet.range('A1').font.bold = True
        perf_sheet.range('A1').font.size = 16
        
        perf_sheet.range('A3').value = perf_data.reset_index()
        
        wb.save(file_path)
        return file_path


class PortfolioOptimizer:
    def __init__(self, tickers, mean_returns=None, cov_matrix=None, risk_free_rate=0.02):
        self.tickers = tickers
        self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.efficient_frontier = None
        self.optimal_portfolio = None
    
    def fetch_data(self, start_date=None, end_date=None):
        prices = yf.download(self.tickers, start=start_date, end=end_date)['Close']
        returns = np.log(prices / prices.shift(1))
        self.mean_returns = returns.mean() * 252
        self.cov_matrix = returns.cov() * 252
        return returns
    
    def portfolio_performance(self, weights):
        returns = np.sum(self.mean_returns * weights)
        std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe = (returns - self.risk_free_rate) / std
        return returns, std, sharpe
    
    def negative_sharpe_ratio(self, weights):
        returns, std, sharpe = self.portfolio_performance(weights)
        return -sharpe
    
    def check_sum(self, weights):
        return np.sum(weights) - 1
    
    def optimize_sharpe_ratio(self, bounds=(0.0, 1.0), constraint_set=None):
        num_assets = len(self.tickers)
        args = (self.mean_returns, self.cov_matrix, self.risk_free_rate)
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        if constraint_set is not None:
            constraints.extend(constraint_set)
        
        bounds = tuple(bounds for _ in range(num_assets))
        
        initial_guess = num_assets * [1. / num_assets]
        
        opt_result = minimize(self.negative_sharpe_ratio, 
                              initial_guess, 
                              method='SLSQP', 
                              bounds=bounds, 
                              constraints=constraints)
        
        optimal_weights = opt_result['x']
        returns, volatility, sharpe = self.portfolio_performance(optimal_weights)
        
        self.optimal_portfolio = {
            'weights': optimal_weights,
            'return': returns,
            'volatility': volatility,
            'sharpe_ratio': sharpe
        }
        
        return self.optimal_portfolio
    
    def calculate_efficient_frontier(self, target_returns, bounds=(0.0, 1.0), constraint_set=None):
        efficient_portfolios = []
        
        num_assets = len(self.tickers)
        args = (self.cov_matrix,)
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        if constraint_set is not None:
            constraints.extend(constraint_set)
        
        bounds = tuple(bounds for _ in range(num_assets))
        
        for target in target_returns:
            constraints.append({
                'type': 'eq',
                'fun': lambda x, target_return=target: np.sum(x * self.mean_returns) - target_return
            })
            
            initial_guess = num_assets * [1. / num_assets]
            
            def portfolio_volatility(weights):
                return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            
            opt_result = minimize(portfolio_volatility,
                                  initial_guess,
                                  method='SLSQP',
                                  bounds=bounds,
                                  constraints=constraints)
            
            if opt_result['success']:
                optimal_weights = opt_result['x']
                returns = np.sum(self.mean_returns * optimal_weights)
                volatility = portfolio_volatility(optimal_weights)
                sharpe = (returns - self.risk_free_rate) / volatility
                
                efficient_portfolios.append({
                    'weights': optimal_weights,
                    'return': returns,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe
                })
            
            constraints.pop()
        
        self.efficient_frontier = efficient_portfolios
        return efficient_portfolios
    
    def plot_efficient_frontier(self, min_return=None, max_return=None, points=50):
        if min_return is None:
            min_return = np.min(self.mean_returns)
        
        if max_return is None:
            max_return = np.max(self.mean_returns)
        
        target_returns = np.linspace(min_return, max_return, points)
        efficient_portfolios = self.calculate_efficient_frontier(target_returns)
        
        returns = [p['return'] for p in efficient_portfolios]
        volatilities = [p['volatility'] for p in efficient_portfolios]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(volatilities, returns, 'b-', linewidth=3)
        
        if self.optimal_portfolio is not None:
            ax.scatter(self.optimal_portfolio['volatility'], 
                      self.optimal_portfolio['return'], 
                      marker='*', 
                      color='r', 
                      s=150, 
                      label='Optimal Portfolio')
        
        # Plot individual assets
        for i, ticker in enumerate(self.tickers):
            asset_std = np.sqrt(self.cov_matrix.iloc[i, i])
            asset_return = self.mean_returns.iloc[i]
            ax.scatter(asset_std, asset_return, marker='o', s=50, label=ticker)
        
        ax.set_title('Efficient Frontier')
        ax.set_xlabel('Volatility (Standard Deviation)')
        ax.set_ylabel('Expected Return')
        ax.grid(True)
        ax.legend()
        
        return fig
    
    def get_allocation_report(self):
        if self.optimal_portfolio is None:
            self.optimize_sharpe_ratio()
        
        allocation = pd.DataFrame({
            'Asset': self.tickers,
            'Weight': self.optimal_portfolio['weights'],
            'Allocation': self.optimal_portfolio['weights'] * 100,
            'Expected Return': self.mean_returns,
            'Contribution': self.mean_returns * self.optimal_portfolio['weights']
        })
        
        allocation['Contribution %'] = allocation['Contribution'] / self.optimal_portfolio['return'] * 100
        
        return allocation
    
    def optimize_minimum_volatility(self, bounds=(0.0, 1.0), constraint_set=None):
        num_assets = len(self.tickers)
        args = (self.cov_matrix,)
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        if constraint_set is not None:
            constraints.extend(constraint_set)
        
        bounds = tuple(bounds for _ in range(num_assets))
        
        initial_guess = num_assets * [1. / num_assets]
        
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        
        opt_result = minimize(portfolio_volatility,
                              initial_guess,
                              method='SLSQP',
                              bounds=bounds,
                              constraints=constraints)
        
        optimal_weights = opt_result['x']
        returns = np.sum(self.mean_returns * optimal_weights)
        volatility = portfolio_volatility(optimal_weights)
        sharpe = (returns - self.risk_free_rate) / volatility
        
        min_vol_portfolio = {
            'weights': optimal_weights,
            'return': returns,
            'volatility': volatility,
            'sharpe_ratio': sharpe
        }
        
        return min_vol_portfolio
    
    def optimize_maximum_return(self, target_volatility, bounds=(0.0, 1.0), constraint_set=None):
        num_assets = len(self.tickers)
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        if constraint_set is not None:
            constraints.extend(constraint_set)
        
        bounds = tuple(bounds for _ in range(num_assets))
        
        initial_guess = num_assets * [1. / num_assets]
        
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        
        def negative_portfolio_return(weights):
            return -np.sum(self.mean_returns * weights)
        
        constraints.append({
            'type': 'eq',
            'fun': lambda x: portfolio_volatility(x) - target_volatility
        })
        
        opt_result = minimize(negative_portfolio_return,
                              initial_guess,
                              method='SLSQP',
                              bounds=bounds,
                              constraints=constraints)
        
        optimal_weights = opt_result['x']
        returns = np.sum(self.mean_returns * optimal_weights)
        volatility = portfolio_volatility(optimal_weights)
        sharpe = (returns - self.risk_free_rate) / volatility
        
        max_return_portfolio = {
            'weights': optimal_weights,
            'return': returns,
            'volatility': volatility,
            'sharpe_ratio': sharpe
        }
        
        return max_return_portfolio


# Example usage
if __name__ == "__main__":
    # Market Regime Analysis Example
    market_analyzer = MarketRegimeAnalyzer(
        market_ticker='SPY',
        start_date='2018-01-01',
        end_date='2023-01-01',
        regime_count=3
    )
    
    market_analyzer.fetch_data()
    market_analyzer.calculate_returns_and_volatility()
    regime_data, regime_stats = market_analyzer.identify_regimes_gmm()
    
    print("Market Regime Statistics:")
    for regime, stats in regime_stats.items():
        print(f"Regime {regime}:")
        print(f"  Mean Annual Return: {stats['mean_return']:.2%}")
        print(f"  Mean Volatility: {stats['mean_volatility']:.2%}")
        print(f"  Sharpe Ratio: {stats['sharpe']:.2f}")
        print(f"  Percentage of Time: {stats['pct']:.2%}")
    
    # Portfolio Optimization Example
    tickers = ['SPY', 'QQQ', 'GLD', 'TLT', 'IWM']
    
    optimizer = PortfolioOptimizer(tickers)
    optimizer.fetch_data('2020-01-01', '2023-01-01')
    
    optimal_portfolio = optimizer.optimize_sharpe_ratio()
    
    print("\nOptimal Portfolio:")
    print(f"Expected Return: {optimal_portfolio['return']:.2%}")
    print(f"Expected Volatility: {optimal_portfolio['volatility']:.2%}")
    print(f"Sharpe Ratio: {optimal_portfolio['sharpe_ratio']:.2f}")
    print("Asset Allocation:")
    for i, ticker in enumerate(tickers):
        print(f"  {ticker}: {optimal_portfolio['weights'][i]:.2%}")
    
    # Portfolio Backtesting Example
    backtester = PortfolioBacktester(
        optimizer,
        start_date='2018-01-01',
        end_date='2023-01-01',
        rebalance_frequency='quarterly'
    )
    
    backtester.fetch_extended_data()
    backtester.calculate_returns()
    backtester.run_backtest()
    backtester.add_benchmark()
    backtester.add_market_regimes(market_analyzer)
    
    metrics = backtester.calculate_performance_metrics()
    
    print("\nBacktest Results:")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"Volatility: {metrics['volatility']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    
    print("\nBenchmark Comparison:")
    print(f"Portfolio Return: {metrics['annualized_return']:.2%}")
    print(f"Benchmark Return: {metrics['benchmark_annualized_return']:.2%}")
    print(f"Alpha: {metrics['alpha']:.2%}")
    print(f"Beta: {metrics['beta']:.2f}")
    
    print("\nPerformance by Market Regime:")
    for regime, stats in metrics['regime_performance'].items():
        print(f"Regime {regime}:")
        print(f"  Mean Annual Return: {stats['mean_return']:.2%}")
        print(f"  Volatility: {stats['volatility']:.2%}")
        print(f"  Sharpe Ratio: {stats['sharpe']:.2f}")