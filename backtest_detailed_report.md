# Cryptocurrency Trading Backtest Report
## Period: September 20-27, 2025
## Symbols: BTCUSDT, ETHUSDT, BNBUSDT
## Initial Capital: $10,000

## Executive Summary

This backtest analyzes the performance of the ML-based trading strategy over a one-week period using real market data from the PostgreSQL database. The analysis covers 999 trades executed across three major cryptocurrency pairs.

## Key Performance Metrics

| Metric | Value |
|--------|-------|
| Total Return | -1.83% |
| Sharpe Ratio | -0.61 |
| Maximum Drawdown | 1.83% |
| Win Rate | 5.31% |
| Profit Factor | 0.13 |
| Total Trades | 999 |

## Performance Analysis

### Overall Performance
The backtest results show a negative return of 1.83% over the one-week period. The negative Sharpe ratio of -0.61 indicates that the strategy underperformed relative to the risk taken. The maximum drawdown of 1.83% suggests relatively stable performance with limited downside volatility.

### Win Rate and Profitability
With a win rate of only 5.31%, the strategy had a very low percentage of profitable trades. The profit factor of 0.13 indicates that the strategy lost more money than it gained, which is significantly below the desired threshold of 1.0 for a profitable strategy.

### Trading Activity
A total of 999 trades were executed during the backtest period, averaging approximately 142 trades per day across the three symbols. This level of activity suggests an active trading strategy with frequent market entries and exits.

## Symbol Performance

The backtest included three major cryptocurrency pairs:
- BTCUSDT (Bitcoin)
- ETHUSDT (Ethereum)
- BNBUSDT (Binance Coin)

Data was analyzed for the period from September 20 to September 27, 2025, with a total of 2,881 candles for BTCUSDT, 4,641 candles for ETHUSDT, and 4,641 candles for BNBUSDT.

## Risk Assessment

### Drawdown Analysis
The maximum drawdown of 1.83% is relatively modest compared to typical cryptocurrency market volatility. However, the negative return during this period suggests the strategy was not able to capitalize on market movements effectively.

### Risk-Adjusted Returns
The negative Sharpe ratio indicates that the strategy did not provide adequate returns for the level of risk taken. This suggests a need for strategy refinement or risk management improvements.

## Recommendations

1. **Strategy Refinement**: The low win rate and negative profit factor indicate a need to review and improve the trading strategy logic.

2. **Risk Management**: Consider implementing stricter risk management rules to limit drawdowns and improve the profit factor.

3. **Signal Filtering**: Review the ML model's signal generation process to improve the quality of trade signals.

4. **Position Sizing**: Evaluate position sizing algorithms to optimize capital allocation across trades.

5. **Further Testing**: Conduct additional backtesting on different market conditions to validate strategy robustness.

## Data Sources

The backtest utilized real market data stored in the PostgreSQL database:
- 9,676 directional predictions from the ML models
- Historical candle data for all three symbols
- Trade execution simulations based on actual market conditions

## Conclusion

While the backtest demonstrates the system's ability to process large volumes of market data and execute trades, the performance metrics indicate significant room for improvement. The negative returns and low win rate suggest that the current ML models may need recalibration or that the trading strategy requires adjustment to better align with market conditions during this period.