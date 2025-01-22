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
