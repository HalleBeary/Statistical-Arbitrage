import yfinance as yf
import pandas as pd
import pair_trader

'''
Simple statistical arbitrage pair trading program using state space models (Kalman Filter) 

=================================================================================================

Spread between pair of cointegrated (correlated long term) stocks is expected to return to equilibrium. 
If spread is out of eq, this present a trading opportunity. Go long on 1 stock, short on other or vice versa.

stock1 = beta * stock2 + error [Measurement Equation]

Kalman filter approach to improve the relationship for beta from fixed to dynamic:

beta_t+1 = beta_t + noise [State equation]

Then, Kalman filter also adjusts the error for noise: "How much should we trust the new data point considering uncertainty and noise distributions?"

beta_t = beta_pred + K * innovation (=uncorrected error)

====> 

Comparison between fixed and dynamic (Kalman) approach is made. Improvement in Sharpe ratio and returns.

'''


def get_price_data(ticker1="DNA", ticker2="ZM", start="2022-01-01", end="2023-01-01"): # Pulls price data for a pair of stocks 

    stock1 = yf.download(ticker1, start=start, end=end) # Using yahoo finance for now
    stock2 = yf.download(ticker2, start=start, end=end)
    
    combined_df = pd.DataFrame()
    combined_df[ticker1] = stock1['Adj Close'] # Using closing price.
    combined_df[ticker2] = stock2['Adj Close']
    
    return combined_df.dropna()


pair_data = get_price_data() # pull price data for a pair of stocks
trader = pair_trader.PairTrader(pair_data)
if trader.validate_trading_pair():
    trader.compare_strategies()