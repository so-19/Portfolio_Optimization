import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sco
import yfinance as yf
from datetime import datetime, timedelta
import xlwings as xw
import seaborn as sns
from sklearn.covariance import LedoitWolf

class PortfolioOptimizer:
    def __init__(self, tickers_stocks, tickers_bonds, start_date, end_date, risk_free_rate=0.02):
        self.tickers_stocks = tickers_stocks
        self.tickers_bonds = tickers_bonds
        self.tickers = tickers_stocks + tickers_bonds
        self.start_date = start_date
        self.end_date = end_date
        self.risk_free_rate = risk_free_rate
        self.data = None
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None
        self.stock_weights = None
        self.bond_weights = None
        self.efficient_frontier = None
        self.optimal_portfolio = None
        self.monte_carlo_results = None
    
    def fetch_data(self):
        self.data = yf.download(self.tickers, start=self.start_date, end=self.end_date)['Close']
        if isinstance(self.data, pd.Series):
            self.data = pd.DataFrame(self.data)
            self.data.columns = self.tickers
        return self.data
    
    def calculate_returns(self, log_returns=True):
        if log_returns:
            self.returns = np.log(self.data / self.data.shift(1)).dropna()
        else:
            self.returns = (self.data / self.data.shift(1) - 1).dropna()
        self.mean_returns = self.returns.mean() * 252
        return self.returns
    
    def calculate_covariance(self, method='standard'):
        if method == 'standard':
            self.cov_matrix = self.returns.cov() * 252
        elif method == 'ledoit_wolf':
            lw = LedoitWolf().fit(self.returns)
            self.cov_matrix = pd.DataFrame(lw.covariance_ * 252, 
                                          index=self.returns.columns, 
                                          columns=self.returns.columns)
        else:
            raise ValueError("Method must be either 'standard' or 'ledoit_wolf'")
        return self.cov_matrix
    
    def _portfolio_performance(self, weights):
        weights = np.array(weights)
        portfolio_return = np.sum(self.mean_returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    def _negative_sharpe(self, weights):
        return -self._portfolio_performance(weights)[2]
    
    def _portfolio_volatility(self, weights):
        return self._portfolio_performance(weights)[1]
    
    def _portfolio_return(self, weights):
        return self._portfolio_performance(weights)[0]
    
    def optimize_sharpe_ratio(self, constraint_set='mixed'):
        num_assets = len(self.tickers)
        args = (self.mean_returns, self.cov_matrix, self.risk_free_rate)
        constraints = []
        
        if constraint_set == 'mixed':
            num_stocks = len(self.tickers_stocks)
            num_bonds = len(self.tickers_bonds)
            
            bounds = tuple((0, 1) for asset in range(num_assets))
            
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'ineq', 'fun': lambda x: np.sum(x[:num_stocks]) - 0.3},  
                {'type': 'ineq', 'fun': lambda x: 0.8 - np.sum(x[:num_stocks])},  
                {'type': 'ineq', 'fun': lambda x: np.sum(x[num_stocks:]) - 0.2},  
                {'type': 'ineq', 'fun': lambda x: 0.7 - np.sum(x[num_stocks:])}   
            ]
        else:
            bounds = tuple((0, 1) for asset in range(num_assets))
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        result = sco.minimize(self._negative_sharpe, num_assets * [1./num_assets], 
                             method='SLSQP', bounds=bounds, constraints=constraints)
        
        self.optimal_portfolio = {
            'weights': result['x'],
            'return': self._portfolio_return(result['x']),
            'volatility': self._portfolio_volatility(result['x']),
            'sharpe_ratio': -result['fun']
        }
        
        self.stock_weights = result['x'][:len(self.tickers_stocks)]
        self.bond_weights = result['x'][len(self.tickers_stocks):]
        
        return self.optimal_portfolio
    
    def efficient_frontier_optimization(self, target_returns):
        self.efficient_frontier = []
        
        num_assets = len(self.tickers)
        bounds = tuple((0, 1) for asset in range(num_assets))
        
        for target in target_returns:
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: self._portfolio_return(x) - target}
            ]
            
            result = sco.minimize(self._portfolio_volatility, num_assets * [1./num_assets],
                                 method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result['success']:
                self.efficient_frontier.append({
                    'target_return': target,
                    'volatility': result['fun'],
                    'weights': result['x']
                })
        
        return self.efficient_frontier
    
    def run_monte_carlo_simulation(self, num_portfolios=10000):
        results = []
        weights_record = []
        
        num_assets = len(self.tickers)
        
        for i in range(num_portfolios):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            weights_record.append(weights)
            
            portfolio_return, portfolio_volatility, sharpe_ratio = self._portfolio_performance(weights)
            
            results.append({
                'return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'weights': weights
            })
        
        self.monte_carlo_results = pd.DataFrame(results)
        
        return self.monte_carlo_results
    
    def plot_efficient_frontier_with_mc(self):
        plt.figure(figsize=(12, 8))
        
        plt.scatter(self.monte_carlo_results['volatility'], 
                   self.monte_carlo_results['return'],
                   c=self.monte_carlo_results['sharpe_ratio'],
                   cmap='viridis', alpha=0.3)
        
        plt.colorbar(label='Sharpe Ratio')
        
        ef_volatility = [ef['volatility'] for ef in self.efficient_frontier]
        ef_return = [ef['target_return'] for ef in self.efficient_frontier]
        
        plt.plot(ef_volatility, ef_return, 'r-', linewidth=3, label='Efficient Frontier')
        
        plt.scatter(self.optimal_portfolio['volatility'], 
                   self.optimal_portfolio['return'],
                   marker='*', color='r', s=300, label='Optimal Portfolio')
        
        plt.title('Portfolio Optimization - Efficient Frontier with Monte Carlo Simulation')
        plt.xlabel('Volatility (Standard Deviation)')
        plt.ylabel('Expected Return')
        plt.legend()
        
        return plt
    
    def plot_optimal_allocation(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        colors = sns.color_palette('viridis', len(self.tickers))
        
        labels = self.tickers
        sizes = self.optimal_portfolio['weights'] * 100
        
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Optimal Portfolio Allocation')
        ax1.axis('equal')
        
        stock_weight_sum = np.sum(self.stock_weights)
        bond_weight_sum = np.sum(self.bond_weights)
        
        asset_class_weights = [stock_weight_sum, bond_weight_sum]
        asset_class_labels = ['Stocks', 'Bonds']
        asset_class_colors = ['#FF9999', '#66B2FF']
        
        ax2.pie(asset_class_weights, labels=asset_class_labels, autopct='%1.1f%%', 
               colors=asset_class_colors, startangle=90)
        ax2.set_title('Asset Class Allocation')
        ax2.axis('equal')
        
        plt.tight_layout()
        
        return fig
    
    def run_portfolio_stress_test(self, scenario_adjustments):
        stress_results = {}
        
        base_mean_returns = self.mean_returns.copy()
        base_cov_matrix = self.cov_matrix.copy()
        
        for scenario_name, adjustment in scenario_adjustments.items():
            return_adjustment = adjustment.get('returns', 0)
            vol_adjustment = adjustment.get('volatility', 1)
            
            self.mean_returns = base_mean_returns + return_adjustment
            self.cov_matrix = base_cov_matrix * vol_adjustment
            
            stress_portfolio = self.optimize_sharpe_ratio()
            
            stress_results[scenario_name] = {
                'return': stress_portfolio['return'],
                'volatility': stress_portfolio['volatility'],
                'sharpe_ratio': stress_portfolio['sharpe_ratio'],
                'weights': stress_portfolio['weights']
            }
        
        self.mean_returns = base_mean_returns
        self.cov_matrix = base_cov_matrix
        
        return stress_results
    
    def historical_performance_simulation(self, weights, window_size=252):
        total_data_points = len(self.returns)
        num_windows = total_data_points - window_size + 1
        
        historical_performance = []
        
        for i in range(num_windows):
            window_returns = self.returns.iloc[i:i+window_size]
            
            daily_portfolio_returns = window_returns.dot(weights)
            
            cumulative_return = (1 + daily_portfolio_returns).prod() - 1
            annualized_return = (1 + cumulative_return) ** (252 / window_size) - 1
            annualized_volatility = daily_portfolio_returns.std() * np.sqrt(252)
            sharpe = (annualized_return - self.risk_free_rate) / annualized_volatility
            max_drawdown = self._calculate_max_drawdown(daily_portfolio_returns)
            
            historical_performance.append({
                'start_date': self.returns.index[i],
                'end_date': self.returns.index[i+window_size-1],
                'cumulative_return': cumulative_return,
                'annualized_return': annualized_return,
                'annualized_volatility': annualized_volatility,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown
            })
        
        return pd.DataFrame(historical_performance)
    
    def _calculate_max_drawdown(self, returns):
        wealth_index = (1 + returns).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        return drawdowns.min()
    
    def export_to_excel(self, file_path):
        wb = xw.Book()
        
        self._create_overview_sheet(wb)
        self._create_optimization_sheet(wb)
        self._create_monte_carlo_sheet(wb)
        self._create_efficient_frontier_sheet(wb)
        self._create_stress_test_sheet(wb)
        
        wb.save(file_path)
        return file_path
    
    def _create_overview_sheet(self, wb):
        sht = wb.sheets.add('Overview')
        
        sht.range('A1').value = 'Portfolio Optimization Results'
        sht.range('A1').font.bold = True
        sht.range('A1').font.size = 16
        
        sht.range('A3').value = 'Optimization Parameters'
        sht.range('A4').value = [['Start Date', self.start_date], 
                                 ['End Date', self.end_date],
                                 ['Risk-Free Rate', self.risk_free_rate],
                                 ['Total Assets', len(self.tickers)],
                                 ['Stock Assets', len(self.tickers_stocks)],
                                 ['Bond Assets', len(self.tickers_bonds)]]
        
        sht.range('A11').value = 'Optimal Portfolio Performance'
        sht.range('A12').value = [['Expected Annual Return', self.optimal_portfolio['return']],
                                 ['Expected Annual Volatility', self.optimal_portfolio['volatility']],
                                 ['Sharpe Ratio', self.optimal_portfolio['sharpe_ratio']]]
        
        sht.range('A17').value = 'Asset Allocation'
        asset_data = []
        for i, ticker in enumerate(self.tickers):
            asset_data.append([ticker, self.optimal_portfolio['weights'][i]])
        
        sht.range('A18').value = asset_data
        
        sht.range('A18:B' + str(17 + len(self.tickers))).number_format = '0.00%'
        
        asset_class_data = [['Stocks', sum(self.stock_weights)], 
                           ['Bonds', sum(self.bond_weights)]]
        sht.range('D17').value = 'Asset Class Allocation'
        sht.range('D18').value = asset_class_data
        sht.range('D18:E19').number_format = '0.00%'
        
        chart = sht.charts.add(50, 300, 300, 300)
        chart.chart_type = 'pie'
        chart.set_source_data(sht.range('D18:E19'))
        chart.chart_title = 'Asset Class Allocation'
        
        sht.autofit()
    
    def _create_optimization_sheet(self, wb):
        sht = wb.sheets.add('Optimization')
        
        sht.range('A1').value = 'Portfolio Optimization Details'
        sht.range('A1').font.bold = True
        sht.range('A1').font.size = 16
        
        sht.range('A3').value = 'Mean Annual Returns'
        returns_data = [[ticker, self.mean_returns[ticker]] for ticker in self.tickers]
        sht.range('A4').value = returns_data
        sht.range('A4:B' + str(3 + len(self.tickers))).number_format = '0.00%'
        
        sht.range('D3').value = 'Covariance Matrix (Annual)'
        sht.range('D4').value = self.cov_matrix
        
        sht.range('A' + str(7 + len(self.tickers))).value = 'Optimal Weights'
        weights_data = [[ticker, self.optimal_portfolio['weights'][i]] for i, ticker in enumerate(self.tickers)]
        sht.range('A' + str(8 + len(self.tickers))).value = weights_data
        sht.range('A' + str(8 + len(self.tickers)) + ':B' + str(7 + 2*len(self.tickers))).number_format = '0.00%'
        
        sht.autofit()
    
    def _create_monte_carlo_sheet(self, wb):
        sht = wb.sheets.add('Monte Carlo')
        
        sht.range('A1').value = 'Monte Carlo Simulation Results'
        sht.range('A1').font.bold = True
        sht.range('A1').font.size = 16
        
        sht.range('A3').value = 'Top 10 Portfolios by Sharpe Ratio'
        top_portfolios = self.monte_carlo_results.sort_values(by='sharpe_ratio', ascending=False).head(10)
        
        headers = ['Return', 'Volatility', 'Sharpe Ratio'] + self.tickers
        data = []
        
        for i, row in top_portfolios.iterrows():
            data_row = [row['return'], row['volatility'], row['sharpe_ratio']]
            data_row.extend(row['weights'])
            data.append(data_row)
        
        sht.range('A4').value = [headers]
        sht.range('A5').value = data
        
        cols = len(headers)
        sht.range('A4:' + chr(64 + cols) + '4').font.bold = True
        sht.range('A5:C14').number_format = '0.00%'
        sht.range('D5:' + chr(64 + cols) + '14').number_format = '0.00%'
        
        sht.range('A16').value = 'Monte Carlo Statistics'
        sht.range('A17').value = [['Statistic', 'Return', 'Volatility', 'Sharpe Ratio'],
                                  ['Min', self.monte_carlo_results['return'].min(), 
                                   self.monte_carlo_results['volatility'].min(),
                                   self.monte_carlo_results['sharpe_ratio'].min()],
                                  ['Max', self.monte_carlo_results['return'].max(), 
                                   self.monte_carlo_results['volatility'].max(),
                                   self.monte_carlo_results['sharpe_ratio'].max()],
                                  ['Mean', self.monte_carlo_results['return'].mean(), 
                                   self.monte_carlo_results['volatility'].mean(),
                                   self.monte_carlo_results['sharpe_ratio'].mean()],
                                  ['Median', self.monte_carlo_results['return'].median(), 
                                   self.monte_carlo_results['volatility'].median(),
                                   self.monte_carlo_results['sharpe_ratio'].median()],
                                  ['Std Dev', self.monte_carlo_results['return'].std(), 
                                   self.monte_carlo_results['volatility'].std(),
                                   self.monte_carlo_results['sharpe_ratio'].std()]]
        
        sht.range('A17:A22').font.bold = True
        sht.range('B18:D22').number_format = '0.00%'
        
        sht.autofit()
    
    def _create_efficient_frontier_sheet(self, wb):
        sht = wb.sheets.add('Efficient Frontier')
        
        sht.range('A1').value = 'Efficient Frontier Data'
        sht.range('A1').font.bold = True
        sht.range('A1').font.size = 16
        
        sht.range('A3').value = ['Target Return', 'Volatility']
        
        ef_data = []
        for point in self.efficient_frontier:
            ef_data.append([point['target_return'], point['volatility']])
        
        sht.range('A4').value = ef_data
        sht.range('A4:B' + str(3 + len(self.efficient_frontier))).number_format = '0.00%'
        
        sht.range('D3').value = 'Optimal Portfolio'
        sht.range('D4').value = [['Return', self.optimal_portfolio['return']],
                                ['Volatility', self.optimal_portfolio['volatility']],
                                ['Sharpe Ratio', self.optimal_portfolio['sharpe_ratio']]]
        sht.range('D4:E6').number_format = '0.00%'
        
        sht.autofit()
    
    def _create_stress_test_sheet(self, wb):
        scenario_adjustments = {
            'Market Crash': {'returns': -0.10, 'volatility': 2.0},
            'Economic Boom': {'returns': 0.05, 'volatility': 0.8},
            'Recession': {'returns': -0.05, 'volatility': 1.5},
            'Rising Interest Rates': {'returns': -0.02, 'volatility': 1.2},
            'Falling Interest Rates': {'returns': 0.02, 'volatility': 0.9}
        }
        
        stress_results = self.run_portfolio_stress_test(scenario_adjustments)
        
        sht = wb.sheets.add('Stress Tests')
        
        sht.range('A1').value = 'Portfolio Stress Test Results'
        sht.range('A1').font.bold = True
        sht.range('A1').font.size = 16
        
        sht.range('A3').value = 'Base Case'
        sht.range('A4').value = [['Return', self.optimal_portfolio['return']],
                                ['Volatility', self.optimal_portfolio['volatility']],
                                ['Sharpe Ratio', self.optimal_portfolio['sharpe_ratio']]]
        
        row = 8
        for scenario, results in stress_results.items():
            sht.range('A' + str(row)).value = scenario
            sht.range('A' + str(row+1)).value = [['Return', results['return']],
                                               ['Volatility', results['volatility']],
                                               ['Sharpe Ratio', results['sharpe_ratio']]]
            
            sht.range('D' + str(row)).value = 'Difference from Base'
            sht.range('D' + str(row+1)).value = [['Return Diff', results['return'] - self.optimal_portfolio['return']],
                                               ['Volatility Diff', results['volatility'] - self.optimal_portfolio['volatility']],
                                               ['Sharpe Ratio Diff', results['sharpe_ratio'] - self.optimal_portfolio['sharpe_ratio']]]
            
            row += 5
        
        sht.range('A4:B6').number_format = '0.00%'
        for r in range(9, row, 5):
            sht.range('A' + str(r) + ':B' + str(r+2)).number_format = '0.00%'
            sht.range('D' + str(r) + ':E' + str(r+2)).number_format = '0.00%'
        
        sht.autofit()

class PortfolioAnalyzer:
    def __init__(self, portfolio_optimizer, initial_investment=100000):
        self.optimizer = portfolio_optimizer
        self.initial_investment = initial_investment
        self.before_optimization = None
        self.after_optimization = None
    
    def set_suboptimal_portfolio(self, weights):
        self.before_optimization = {
            'weights': weights,
            'return': self.optimizer._portfolio_return(weights),
            'volatility': self.optimizer._portfolio_volatility(weights),
            'sharpe_ratio': self.optimizer._portfolio_performance(weights)[2]
        }
        return self.before_optimization
    
    def generate_equal_weight_portfolio(self):
        equal_weights = np.ones(len(self.optimizer.tickers)) / len(self.optimizer.tickers)
        return self.set_suboptimal_portfolio(equal_weights)
    
    def generate_stock_heavy_portfolio(self, stock_allocation=0.8):
        num_stocks = len(self.optimizer.tickers_stocks)
        num_bonds = len(self.optimizer.tickers_bonds)
        
        if num_stocks + num_bonds != len(self.optimizer.tickers):
            raise ValueError("Ticker lists inconsistent")
        
        stock_weight = stock_allocation / num_stocks
        bond_weight = (1 - stock_allocation) / num_bonds
        
        weights = np.array([stock_weight] * num_stocks + [bond_weight] * num_bonds)
        
        return self.set_suboptimal_portfolio(weights)
    
    def generate_bond_heavy_portfolio(self, bond_allocation=0.8):
        return self.generate_stock_heavy_portfolio(stock_allocation=(1-bond_allocation))
    
    def compare_performance(self):
        self.after_optimization = self.optimizer.optimal_portfolio
        
        if self.before_optimization is None:
            self.generate_equal_weight_portfolio()
        
        comparison = {
            'before': self.before_optimization,
            'after': self.after_optimization,
            'return_improvement': self.after_optimization['return'] - self.before_optimization['return'],
            'volatility_reduction': self.before_optimization['volatility'] - self.after_optimization['volatility'],
            'sharpe_improvement': self.after_optimization['sharpe_ratio'] - self.before_optimization['sharpe_ratio']
        }
        
        return comparison
    
    def simulate_investment_growth(self, years=10, num_simulations=1000):
        before_portfolio = self.before_optimization
        after_portfolio = self.after_optimization
        
        time_periods = years * 252
        
        before_results = self._run_investment_simulation(before_portfolio, time_periods, num_simulations)
        after_results = self._run_investment_simulation(after_portfolio, time_periods, num_simulations)
        
        return {
            'before': before_results,
            'after': after_results
        }
    
    def _run_investment_simulation(self, portfolio, time_periods, num_simulations):
        mu = portfolio['return'] / 252
        sigma = portfolio['volatility'] / np.sqrt(252)
        
        simulation_df = pd.DataFrame()
        
        for i in range(num_simulations):
            daily_returns = np.random.normal(mu, sigma, time_periods)
            price_list = [self.initial_investment]
            
            for r in daily_returns:
                price_list.append(price_list[-1] * (1 + r))
                
            simulation_df[i] = price_list
            
        return {
            'simulations': simulation_df,
            'mean_final': simulation_df.iloc[-1].mean(),
            'median_final': simulation_df.iloc[-1].median(),
            'min_final': simulation_df.iloc[-1].min(),
            'max_final': simulation_df.iloc[-1].max(),
            'std_final': simulation_df.iloc[-1].std()
        }
    
    def plot_investment_comparison(self, simulation_results, percentiles=[5, 50, 95]):
        before_sims = simulation_results['before']['simulations']
        after_sims = simulation_results['after']['simulations']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        for p in percentiles:
            before_percentile = before_sims.apply(lambda x: np.percentile(x, p), axis=1)
            ax1.plot(before_percentile, label=f"{p}th Percentile")
            
            after_percentile = after_sims.apply(lambda x: np.percentile(x, p), axis=1)
            ax2.plot(after_percentile, label=f"{p}th Percentile")
        
        ax1.set_title('Before Optimization')
        ax1.set_xlabel('Trading Days')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True)
        
        ax2.set_title('After Optimization')
        ax2.set_xlabel('Trading Days')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        fig_diff, ax_diff = plt.subplots(figsize=(10, 6))
        
        for p in percentiles:
            before_percentile = before_sims.apply(lambda x: np.percentile(x, p), axis=1)
            after_percentile = after_sims.apply(lambda x: np.percentile(x, p), axis=1)
            
            diff_percentile = after_percentile - before_percentile
            ax_diff.plot(diff_percentile, label=f"{p}th Percentile")
        
        ax_diff.set_title('Value Difference (After - Before Optimization)')
        ax_diff.set_xlabel('Trading Days')
        ax_diff.set_ylabel('Value Difference ($)')
        ax_diff.legend()
        ax_diff.grid(True)
        ax_diff.axhline(y=0, color='r', linestyle='-')
        
        return fig, fig_diff
    
    def export_comparison_to_excel(self, file_path, simulation_results=None):
        wb = xw.Book(file_path)
        
        sht = wb.sheets.add('Performance Comparison')
        
        sht.range('A1').value = 'Portfolio Performance Comparison'
        sht.range('A1').font.bold = True
        sht.range('A1').font.size = 16
        
        comparison = self.compare_performance()
        
        sht.range('A3').value = 'Metric'
        sht.range('B3').value = 'Before Optimization'
        sht.range('C3').value = 'After Optimization'
        sht.range('D3').value = 'Improvement'
        
        sht.range('A4').value = [
            ['Expected Return', comparison['before']['return'], comparison['after']['return'], comparison['return_improvement']],
            ['Volatility', comparison['before']['volatility'], comparison['after']['volatility'], comparison['volatility_reduction'] * -1],
            ['Sharpe Ratio', comparison['before']['sharpe_ratio'], comparison['after']['sharpe_ratio'], comparison['sharpe_improvement']]
        ]
        
        sht.range('A3:D3').font.bold = True
        sht.range('B4:D6').number_format = '0'
        sht.range('A3:D3').font.bold = True
        sht.range('B4:D6').number_format = '0.00%'
        
        sht.range('A8').value = 'Before Optimization Weights'
        before_weights = [[ticker, weight] for ticker, weight in zip(self.optimizer.tickers, comparison['before']['weights'])]
        sht.range('A9').value = before_weights
        sht.range('A9:B' + str(8 + len(self.optimizer.tickers))).number_format = '0.00%'
        
        sht.range('D8').value = 'After Optimization Weights'
        after_weights = [[ticker, weight] for ticker, weight in zip(self.optimizer.tickers, comparison['after']['weights'])]
        sht.range('D9').value = after_weights
        sht.range('D9:E' + str(8 + len(self.optimizer.tickers))).number_format = '0.00%'
        
        if simulation_results:
            sht.range('A' + str(11 + len(self.optimizer.tickers))).value = 'Investment Growth Simulation (10 Year)'
            
            metrics = ['Mean Final Value', 'Median Final Value', 'Min Final Value', 'Max Final Value', 'Std Dev Final Value']
            sim_data = [
                [metrics[0], simulation_results['before']['mean_final'], simulation_results['after']['mean_final'], 
                 simulation_results['after']['mean_final'] - simulation_results['before']['mean_final']],
                [metrics[1], simulation_results['before']['median_final'], simulation_results['after']['median_final'], 
                 simulation_results['after']['median_final'] - simulation_results['before']['median_final']],
                [metrics[2], simulation_results['before']['min_final'], simulation_results['after']['min_final'], 
                 simulation_results['after']['min_final'] - simulation_results['before']['min_final']],
                [metrics[3], simulation_results['before']['max_final'], simulation_results['after']['max_final'], 
                 simulation_results['after']['max_final'] - simulation_results['before']['max_final']],
                [metrics[4], simulation_results['before']['std_final'], simulation_results['after']['std_final'], 
                 simulation_results['after']['std_final'] - simulation_results['before']['std_final']]
            ]
            
            sht.range('A' + str(12 + len(self.optimizer.tickers))).value = sim_data
            sht.range('B' + str(12 + len(self.optimizer.tickers)) + ':D' + str(16 + len(self.optimizer.tickers))).number_format = '$#,##0.00'
        
        charts_sheet = wb.sheets.add('Comparison Charts')
        
        comparison_chart = charts_sheet.charts.add(10, 10, 400, 300)
        comparison_chart.chart_type = 'column'
        comparison_chart.set_source_data(sht.range('A3:C6'))
        comparison_chart.chart_title = 'Before vs After Optimization'
        
        sht.autofit()
        wb.save()
        
        return wb

def main():
    tickers_stocks = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'JNJ', 'JPM', 'V', 'PG']
    tickers_bonds = ['AGG', 'BND', 'TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'MUB', 'VCSH', 'VCIT']
    
    start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    optimizer = PortfolioOptimizer(tickers_stocks, tickers_bonds, start_date, end_date)
    
    print("Fetching historical data...")
    optimizer.fetch_data()
    
    print("Calculating returns and covariance...")
    optimizer.calculate_returns()
    optimizer.calculate_covariance(method='ledoit_wolf')
    
    print("Running optimization...")
    optimizer.optimize_sharpe_ratio(constraint_set='mixed')
    
    print("Running Monte Carlo simulation...")
    optimizer.run_monte_carlo_simulation(num_portfolios=5000)
    
    print("Calculating efficient frontier...")
    target_returns = np.linspace(0.05, 0.25, 20)
    optimizer.efficient_frontier_optimization(target_returns)
    
    print("Creating plots...")
    ef_plot = optimizer.plot_efficient_frontier_with_mc()
    ef_plot.savefig('efficient_frontier.png', dpi=300)
    
    allocation_plot = optimizer.plot_optimal_allocation()
    allocation_plot.savefig('allocation.png', dpi=300)
    
    print("Creating Excel report...")
    excel_file = optimizer.export_to_excel('portfolio_optimization.xlsx')
    
    print("Running comparison analysis...")
    analyzer = PortfolioAnalyzer(optimizer)
    analyzer.generate_stock_heavy_portfolio(stock_allocation=0.7)
    
    comparison = analyzer.compare_performance()
    print(f"Return improvement: {comparison['return_improvement']*100:.2f}%")
    print(f"Volatility reduction: {comparison['volatility_reduction']*100:.2f}%")
    print(f"Sharpe ratio improvement: {comparison['sharpe_improvement']:.2f}")
    
    print("Running investment simulation...")
    simulation_results = analyzer.simulate_investment_growth(years=10)
    performance_plots = analyzer.plot_investment_comparison(simulation_results)
    performance_plots[0].savefig('investment_comparison.png', dpi=300)
    performance_plots[1].savefig('investment_difference.png', dpi=300)
    
    print("Exporting final comparison to Excel...")
    analyzer.export_comparison_to_excel('portfolio_optimization.xlsx', simulation_results)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()