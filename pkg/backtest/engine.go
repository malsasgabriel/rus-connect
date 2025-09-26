package backtest

import (
	"database/sql"
	"fmt"
	"log"
	"math"
	"sort"
	"time"

	_ "github.com/lib/pq"
	"github.com/rus-connect/pkg/common"
)

// BacktestEngine represents the backtesting engine
type BacktestEngine struct {
	db              *sql.DB
	symbols         []string
	startDate       time.Time
	endDate         time.Time
	initialCapital  float64
	transactionCost float64
}

// Trade represents a single trade in backtesting
type Trade struct {
	Symbol        string
	EntryTime     time.Time
	ExitTime      time.Time
	EntryPrice    float64
	ExitPrice     float64
	Direction     string // "LONG" or "SHORT"
	Quantity      float64
	EntryValue    float64
	ExitValue     float64
	PnL           float64
	PnLPercentage float64
	HoldingPeriod time.Duration
	StopLoss      float64
	TakeProfit    float64
}

// NewBacktestEngine creates a new backtesting engine
func NewBacktestEngine(db *sql.DB, symbols []string, startDate, endDate time.Time, initialCapital float64) *BacktestEngine {
	return &BacktestEngine{
		db:              db,
		symbols:         symbols,
		startDate:       startDate,
		endDate:         endDate,
		initialCapital:  initialCapital,
		transactionCost: 0.001, // 0.1% transaction cost
	}
}

// Run executes the backtest
func (be *BacktestEngine) Run() (*common.BacktestReport, error) {
	report := &common.BacktestReport{}

	// Initialize report structures
	report.Charts.MonthlyReturns = make(map[string]float64)
	report.Analysis.TimeAnalysis = make(map[string]float64)

	// Get all signals and candles for the period
	signals, candles, err := be.fetchData()
	if err != nil {
		return nil, fmt.Errorf("failed to fetch data: %v", err)
	}

	// Execute trades based on signals
	trades, equityCurve, err := be.executeTrades(signals, candles)
	if err != nil {
		return nil, fmt.Errorf("failed to execute trades: %v", err)
	}

	// Calculate performance metrics
	be.calculateMetrics(report, trades, equityCurve)

	return report, nil
}

// fetchData retrieves signals and candles for backtesting
func (be *BacktestEngine) fetchData() ([]common.Signal, map[string][]Candle, error) {
	// This would fetch data from the database
	// For now, we'll return empty data
	signals := []common.Signal{}
	candles := make(map[string][]Candle)

	return signals, candles, nil
}

// executeTrades simulates trading based on signals
func (be *BacktestEngine) executeTrades(signals []common.Signal, candles map[string][]Candle) ([]Trade, []common.Point, error) {
	trades := []Trade{}
	equityCurve := []common.Point{}

	// Sort signals by timestamp
	sort.Slice(signals, func(i, j int) bool {
		return signals[i].Timestamp.Before(signals[j].Timestamp)
	})

	// Simulate trading
	capital := be.initialCapital
	openPositions := make(map[string]Trade)

	for _, signal := range signals {
		// Check if we should enter a trade
		if be.shouldEnterTrade(signal) {
			// Close any existing position for this symbol
			if existingPosition, exists := openPositions[signal.Symbol]; exists {
				// Close existing position
				trade := be.closePosition(existingPosition, signal.Timestamp, candles)
				trades = append(trades, trade)
				delete(openPositions, signal.Symbol)
				capital += trade.PnL
			}

			// Open new position
			if position, err := be.openPosition(signal, capital, candles); err == nil {
				openPositions[signal.Symbol] = position
			}
		}

		// Update equity curve
		equityCurve = append(equityCurve, common.Point{
			Timestamp: signal.Timestamp.Unix(),
			Value:     capital,
		})
	}

	// Close all open positions at the end
	for symbol, position := range openPositions {
		trade := be.closePosition(position, be.endDate, candles)
		trades = append(trades, trade)
		delete(openPositions, symbol)
		capital += trade.PnL
		equityCurve = append(equityCurve, common.Point{
			Timestamp: be.endDate.Unix(),
			Value:     capital,
		})
	}

	return trades, equityCurve, nil
}

// shouldEnterTrade determines if we should enter a trade based on signal
func (be *BacktestEngine) shouldEnterTrade(signal common.Signal) bool {
	// Only enter trades with high confidence
	return signal.Confidence > 0.7 && (signal.Prediction == "STRONG_BUY" || signal.Prediction == "STRONG_SELL")
}

// openPosition opens a new trading position
func (be *BacktestEngine) openPosition(signal common.Signal, capital float64, _ map[string][]Candle) (Trade, error) {
	// Get current price for the symbol
	currentPrice := signal.PriceTarget // Simplified - in reality, we'd get the actual market price

	// Calculate position size (1% of capital)
	positionSize := capital * 0.01
	quantity := positionSize / currentPrice

	direction := "LONG"
	if signal.Prediction == "STRONG_SELL" {
		direction = "SHORT"
	}

	// Calculate stop loss and take profit
	stopLoss := signal.StopLoss
	takeProfit := signal.PriceTarget

	trade := Trade{
		Symbol:     signal.Symbol,
		EntryTime:  signal.Timestamp,
		EntryPrice: currentPrice,
		Direction:  direction,
		Quantity:   quantity,
		EntryValue: positionSize,
		StopLoss:   stopLoss,
		TakeProfit: takeProfit,
	}

	return trade, nil
}

// closePosition closes an existing trading position
func (be *BacktestEngine) closePosition(position Trade, exitTime time.Time, _ map[string][]Candle) Trade {
	// Get exit price (simplified)
	exitPrice := position.TakeProfit // In reality, we'd get the actual market price at exitTime

	// Calculate PnL
	var pnl float64
	if position.Direction == "LONG" {
		pnl = (exitPrice - position.EntryPrice) * position.Quantity
	} else {
		pnl = (position.EntryPrice - exitPrice) * position.Quantity
	}

	// Apply transaction costs
	entryCost := position.EntryValue * be.transactionCost
	exitValue := exitPrice * position.Quantity
	exitCost := exitValue * be.transactionCost
	totalCost := entryCost + exitCost
	pnl -= totalCost

	// Calculate percentage return
	pnlPercentage := (pnl / position.EntryValue) * 100

	position.ExitTime = exitTime
	position.ExitPrice = exitPrice
	position.ExitValue = exitValue
	position.PnL = pnl
	position.PnLPercentage = pnlPercentage
	position.HoldingPeriod = exitTime.Sub(position.EntryTime)

	return position
}

// calculateMetrics calculates performance metrics for the backtest
func (be *BacktestEngine) calculateMetrics(report *common.BacktestReport, trades []Trade, equityCurve []common.Point) {
	if len(trades) == 0 {
		return
	}

	// Calculate total return
	initialCapital := be.initialCapital
	finalCapital := initialCapital
	for _, trade := range trades {
		finalCapital += trade.PnL
	}
	totalReturn := (finalCapital - initialCapital) / initialCapital * 100
	report.Summary.TotalReturn = totalReturn

	// Calculate win rate
	winningTrades := 0
	for _, trade := range trades {
		if trade.PnL > 0 {
			winningTrades++
		}
	}
	winRate := float64(winningTrades) / float64(len(trades)) * 100
	report.Summary.WinRate = winRate

	// Calculate profit factor
	grossProfit := 0.0
	grossLoss := 0.0
	for _, trade := range trades {
		if trade.PnL > 0 {
			grossProfit += trade.PnL
		} else {
			grossLoss += math.Abs(trade.PnL)
		}
	}
	if grossLoss > 0 {
		report.Summary.ProfitFactor = grossProfit / grossLoss
	} else {
		report.Summary.ProfitFactor = math.Inf(1)
	}

	// Calculate Sharpe ratio (simplified)
	returns := make([]float64, len(trades))
	for i, trade := range trades {
		returns[i] = trade.PnLPercentage
	}
	sharpeRatio := calculateSharpeRatio(returns)
	report.Summary.SharpeRatio = sharpeRatio

	// Calculate max drawdown
	maxDrawdown := calculateMaxDrawdown(equityCurve)
	report.Summary.MaxDrawdown = maxDrawdown

	// Set other metrics
	report.Summary.TotalTrades = len(trades)
	report.Charts.EquityCurve = equityCurve

	log.Printf("Backtest completed: %+v", report.Summary)
}

// calculateSharpeRatio calculates the Sharpe ratio
func calculateSharpeRatio(returns []float64) float64 {
	if len(returns) == 0 {
		return 0
	}

	// Calculate mean return
	sum := 0.0
	for _, r := range returns {
		sum += r
	}
	mean := sum / float64(len(returns))

	// Calculate standard deviation
	sumSquaredDiff := 0.0
	for _, r := range returns {
		diff := r - mean
		sumSquaredDiff += diff * diff
	}
	stdDev := math.Sqrt(sumSquaredDiff / float64(len(returns)))

	// Assuming risk-free rate of 0
	if stdDev > 0 {
		return mean / stdDev
	}
	return 0
}

// calculateMaxDrawdown calculates the maximum drawdown
func calculateMaxDrawdown(equityCurve []common.Point) float64 {
	if len(equityCurve) == 0 {
		return 0
	}

	maxPeak := equityCurve[0].Value
	maxDrawdown := 0.0

	for _, point := range equityCurve {
		if point.Value > maxPeak {
			maxPeak = point.Value
		}

		drawdown := (maxPeak - point.Value) / maxPeak * 100
		if drawdown > maxDrawdown {
			maxDrawdown = drawdown
		}
	}

	return maxDrawdown
}

// Candle represents a price candle
type Candle struct {
	Symbol    string
	Timestamp time.Time
	Open      float64
	High      float64
	Low       float64
	Close     float64
	Volume    float64
}
