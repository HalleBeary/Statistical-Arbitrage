import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
import matplotlib.pyplot as plt


class PairTrader: 
    def __init__(self, price_data):
        self.data = price_data
        self.stock1 = price_data.columns[0]
        self.stock2 = price_data.columns[1] 

    def check_cointegration(self): # check if two stocks are cointegrated, which means two time series mantain a stable long-term relationship. If cointegreated, spread will move back to mean-value.
        _, pvalue, _ = coint(self.data[self.stock1], self.data[self.stock2])
        return pvalue < 0.05
    
    def validate_trading_pair(self):
        _, pvalue, _ = coint(self.data[self.stock1], self.data[self.stock2])
        print(f"\nCointegration Analysis:")
        print(f"Pair: {self.stock1}-{self.stock2}")
        print(f"p-value: {pvalue:.4f}")
        print(f"Suitable for pairs trading: {'Yes' if pvalue < 0.05 else 'No'}")
        
        if pvalue < 0.05:
            return True
        else:
            return False
        
    def calculate_fixed_beta(self):
        x = self.data[self.stock2]
        y = self.data[self.stock1]
        beta = np.cov(x, y)[0,1] / np.var(x) # np.cov returns 2x2 matrix [var(x) cov(x,y), cov(x,y) var(y)]
        return beta
        
    
    def calculate_spread_fixed(self):
        beta = self.calculate_fixed_beta()
        # Normalize prices before calculating spread
        norm1 = self.data[self.stock1] / self.data[self.stock1].iloc[0]
        norm2 = self.data[self.stock2] / self.data[self.stock2].iloc[0]
        spread = norm1 - beta * norm2
        return spread, beta
    
    def kalman_filter(self): # Central function computing the Kalman gain, and adjusting beta. It now uses a dynamic approach in determining R and Q 
        # Initialize state space model
        beta = np.zeros(len(self.data))
        P = np.zeros(len(self.data))
        
        # Initialize with fixed beta
        beta[0] = self.calculate_fixed_beta()
        P[0] = 1
        
        window = 20  # Window for volatility calculation
        
        # R is the measurement noise, and is calculated based on measurement (price) volatility. Measurement noise can occur via bid-ask spreads, microstructure noise (trades)
        returns1 = np.log(self.data[self.stock1] / self.data[self.stock1].shift(1))
        returns2 = np.log(self.data[self.stock2] / self.data[self.stock2].shift(1))

        vol1 = returns1.rolling(window=window, min_periods=1).std()
        vol2 = returns2.rolling(window=window, min_periods=1).std()
        
        # Setting R proportional to volatility of stock1 squared +  stock2 squared. 
        R = (vol1 * vol1 + vol2 * vol2).fillna(0.0001)  # nan's can occur at start of series when not enough data, or if price data is missing.

        
        # Q is the state noise, and is calculated based on rolling beta. Continously calculate the covariance between stock 1 and 2. 
        # Q is different for different market conditions (different beta), volatile conditions imply more larger Q.
        rolling_betas = pd.Series(index=self.data.index)
        for t in range(window, len(self.data)):
            x = self.data[self.stock2].iloc[t-window:t]
            y = self.data[self.stock1].iloc[t-window:t]
            rolling_betas.iloc[t] = np.cov(x, y)[0,1] / np.var(x)
        
        # Set Q
        Q = rolling_betas.rolling(window=window, min_periods=1).std() ** 2
        Q = Q.fillna(0.000001) 
        
        # Scale factors for R and Q (can be tuned)
        R_scale = 1
        Q_scale = 0.5 # be a bit conservative with changing beta regimes
        
        for t in range(1, len(self.data)):
            #### Prediction STEP
            beta_pred = beta[t-1]
            P_pred = P[t-1] + Q.iloc[t] * Q_scale # Uncertainty is predicted considering Q, state noise.
            
            ### Updating STEP
            x = self.data[self.stock2].iloc[t]
            y = self.data[self.stock1].iloc[t]
            
            innovation = y - beta_pred * x
            S = x * P_pred * x + R.iloc[t] * R_scale
            K = P_pred * x / S
            
            beta[t] = beta_pred + K * innovation
            P[t] = P_pred * (1 - K * x)
            
            # Some Safeguards
            if P[t] < 0:  # Ensure positive variance
                P[t] = 0.0001
            if abs(beta[t]) > 5:  # Limit extreme beta values
                beta[t] = np.sign(beta[t]) * 5
        
        return beta, P

    def calculate_spread_kalman(self):
        betas, P = self.kalman_filter()
        norm1 = self.data[self.stock1] / self.data[self.stock1].iloc[0]
        norm2 = self.data[self.stock2] / self.data[self.stock2].iloc[0]
        spreads = pd.Series(index=self.data.index)
        
        for t in range(len(self.data)):
            spreads.iloc[t] = norm1.iloc[t] - betas[t] * norm2.iloc[t]
                
        return spreads, betas, P
    
    def generate_signals(self, spread, rolling_std, n_std=2):

        signals = pd.Series(index=spread.index, data=0)
        
        # Long spread when it's too negative (buy stock1, sell stock2)
        signals[spread < -n_std * rolling_std] = 1
        
        # Short spread when it's too positive (sell stock1, buy stock2)
        signals[spread > n_std * rolling_std] = -1
        
        return signals

    def generate_kalman_signals(self, spread, betas, P, n_std=2):

        signals = pd.Series(index=spread.index, data=0)
        
        # Calculate dynamic volatility-based threshold
        rolling_vol = spread.rolling(window=20).std()
        uncertainty = np.sqrt(P)
        
        # Combine spread volatility and Kalman uncertainty
        dynamic_threshold = n_std * (rolling_vol + uncertainty)
        
        # Generate base signals
        signals[spread < -dynamic_threshold] = 1
        signals[spread > dynamic_threshold] = -1
        
        # Add market regime filter
        beta_trend = pd.Series(betas).diff(5).rolling(10).mean()
        vol_regime = rolling_vol.rolling(20).mean()
        vol_thresh = vol_regime.rolling(100).mean() * 1.5
        
        # Don't trade in high volatility or unstable beta regimes
        signals[vol_regime > vol_thresh] = 0
        signals[abs(beta_trend) > 0.01] = 0
         
        return signals
    
    def calculate_returns(self, signals, spread):
        signals = signals.shift(1).fillna(0)  # Avoid look-ahead bias
        
        # Calculate returns based on spread changes
        spread_returns = spread.diff().fillna(0)
        
        # Calculate strategy returns
        strategy_returns = signals * spread_returns
        
        # Calculate metrics
        cumulative_returns = (1 + strategy_returns).cumprod()
        sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
        
        return strategy_returns, cumulative_returns, sharpe

    def compare_strategies(self):
        # Fixed beta approach
        spread_fixed, beta_fixed = self.calculate_spread_fixed()
        rolling_std_fixed = spread_fixed.rolling(window=20).std()
        signals_fixed = self.generate_signals(spread_fixed, rolling_std_fixed)
        returns_fixed, cum_returns_fixed, sharpe_fixed = self.calculate_returns(signals_fixed, spread_fixed)
        
        # Kalman filter approach
        spread_kalman, betas_kalman, P = self.calculate_spread_kalman()
        rolling_std_kalman = pd.Series(spread_kalman).rolling(window=20).std()
        signals_kalman = self.generate_kalman_signals(pd.Series(spread_kalman), rolling_std_kalman, P)
        returns_kalman, cum_returns_kalman, sharpe_kalman = self.calculate_returns(signals_kalman, pd.Series(spread_kalman))
        
        # Plot results
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot stock prices
        normalized_data = self.data / self.data.iloc[0]
        self.data.plot(ax=ax1)
        ax1.set_title('Stock Prices')
        ax1.set_ylabel('Price')
        
        # Plot spreads and signals
        ax2.plot(spread_fixed.index, spread_fixed, label='Fixed Beta Spread', alpha=0.7)
        ax2.plot(spread_fixed.index, spread_kalman, label='Kalman Spread', alpha=0.7)
        ax2.plot(signals_fixed[signals_fixed != 0].index, spread_fixed[signals_fixed != 0], '^', markersize=10, label='Fixed Beta Signals')
        ax2.plot(signals_kalman[signals_kalman != 0].index, 
                pd.Series(spread_kalman)[signals_kalman != 0], 
                'v', markersize=10, label='Kalman Signals')
        ax2.set_title('Spreads and Trading Signals')
        ax2.legend()
        
        # Plot betas
        ax3.axhline(y=beta_fixed, color='r', label=f'Fixed Beta: {beta_fixed:.2f}')
        ax3.plot(spread_fixed.index, betas_kalman, label='Kalman Beta')
        ax3.set_title('Beta Comparison')
        ax3.legend()
        
        # Plot cumulative returns
        ax4.plot(cum_returns_fixed, label=f'Fixed Beta (Sharpe: {sharpe_fixed:.2f})')
        ax4.plot(cum_returns_kalman, label=f'Kalman (Sharpe: {sharpe_kalman:.2f})')
        ax4.set_title('Cumulative Returns')
        ax4.legend()
        
        plt.tight_layout()
        plt.show()
        