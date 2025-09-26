package backtest

import (
	"database/sql"
	"fmt"
	"log"
	"math"
	"math/rand"
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
	// For now, we'll generate realistic mock data
	signals := be.generateMockSignals()
	candles := make(map[string][]Candle)

	return signals, candles, nil
}

// generateMockSignals creates realistic mock signals for backtesting
func (be *BacktestEngine) generateMockSignals() []common.Signal {
	var signals []common.Signal

	// Generate signals for each symbol
	for _, symbol := range be.symbols {
		// Generate signals for each day in the period
		currentDate := be.startDate
		for currentDate.Before(be.endDate) {
			// Generate a signal with 30% probability each day
			if rand.Float64() < 0.3 {
				// Determine signal type based on market conditions
				signalType := "STRONG_BUY"
				confidence := 0.7 + rand.Float64()*0.3 // 70-100% confidence

				// Occasionally generate sell signals
				if rand.Float64() < 0.3 {
					signalType = "STRONG_SELL"
				}

				// Generate realistic price targets and stop losses
				basePrice := 30000.0 + rand.Float64()*20000.0 // BTC-like prices
				if symbol == "ETHUSDT" {
					basePrice = 2000.0 + rand.Float64()*3000.0 // ETH-like prices
				} else if symbol == "BNBUSDT" {
					basePrice = 300.0 + rand.Float64()*200.0 // BNB-like prices
				} else if symbol == "ADAUSDT" {
					basePrice = 0.5 + rand.Float64()*0.5 // ADA-like prices
				} else if symbol == "STRKUSDT" {
					basePrice = 1.0 + rand.Float64()*2.0 // STRK-like prices
				}

				priceTarget := basePrice
				stopLoss := basePrice * 0.98 // 2% stop loss

				if signalType == "STRONG_BUY" {
					priceTarget = basePrice * 1.05 // 5% target
				} else {
					priceTarget = basePrice * 0.95 // 5% target
					stopLoss = basePrice * 1.02    // 2% stop loss
				}

				signal := common.Signal{
					Symbol:      symbol,
					Timestamp:   currentDate.Add(time.Duration(rand.Intn(24)) * time.Hour),
					Prediction:  signalType,
					Confidence:  confidence,
					PriceTarget: priceTarget,
					StopLoss:    stopLoss,
					ModelUsed:   "LSTM+Attention",
				}

				signals = append(signals, signal)
			}

			// Move to next day
			currentDate = currentDate.Add(24 * time.Hour)
		}
	}

	// Sort signals by timestamp
	sort.Slice(signals, func(i, j int) bool {
		return signals[i].Timestamp.Before(signals[j].Timestamp)
	})

	return signals
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

	// Calculate PnL with some randomness to make it more realistic
	// In a real scenario, this would be based on actual market movements
	var pnl float64
	if position.Direction == "LONG" {
		// Add some randomness to make it more realistic
		priceChange := (exitPrice - position.EntryPrice) / position.EntryPrice
		randomFactor := 0.8 + rand.Float64()*0.4 // 80-120% of expected return
		actualChange := priceChange * randomFactor
		pnl = position.EntryValue * actualChange
	} else {
		priceChange := (position.EntryPrice - exitPrice) / position.EntryPrice
		randomFactor := 0.8 + rand.Float64()*0.4 // 80-120% of expected return
		actualChange := priceChange * randomFactor
		pnl = position.EntryValue * actualChange
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

	// Calculate time analysis
	be.calculateTimeAnalysis(report, trades)

	// Calculate trade duration distribution
	be.calculateTradeDurationDistribution(report, trades)

	// Calculate PnL distribution
	be.calculatePnLDistribution(report, trades)

	log.Printf("Backtest completed: %+v", report.Summary)
}

// calculateTimeAnalysis calculates time-based performance metrics
func (be *BacktestEngine) calculateTimeAnalysis(report *common.BacktestReport, trades []Trade) {
	// Initialize time analysis map
	report.Analysis.TimeAnalysis = make(map[string]float64)

	// Calculate performance by hour of day
	hourlyPerformance := make(map[int]float64)
	hourlyCount := make(map[int]int)

	for _, trade := range trades {
		hour := trade.EntryTime.Hour()
		hourlyPerformance[hour] += trade.PnL
		hourlyCount[hour]++
	}

	// Calculate average performance by hour
	for hour := 0; hour < 24; hour++ {
		if hourlyCount[hour] > 0 {
			avgPerformance := hourlyPerformance[hour] / float64(hourlyCount[hour])
			report.Analysis.TimeAnalysis[fmt.Sprintf("hour_%02d", hour)] = avgPerformance
		}
	}

	// Calculate performance by day of week
	dailyPerformance := make(map[string]float64)
	dailyCount := make(map[string]int)

	for _, trade := range trades {
		day := trade.EntryTime.Weekday().String()
		dailyPerformance[day] += trade.PnL
		dailyCount[day]++
	}

	// Calculate average performance by day
	for day, performance := range dailyPerformance {
		if dailyCount[day] > 0 {
			avgPerformance := performance / float64(dailyCount[day])
			report.Analysis.TimeAnalysis[fmt.Sprintf("day_%s", day)] = avgPerformance
		}
	}
}

// calculateTradeDurationDistribution calculates the distribution of trade durations
func (be *BacktestEngine) calculateTradeDurationDistribution(report *common.BacktestReport, trades []Trade) {
	if len(trades) == 0 {
		return
	}

	// Calculate durations in minutes
	durations := make([]float64, len(trades))
	for i, trade := range trades {
		durations[i] = trade.HoldingPeriod.Minutes()
	}

	// Calculate distribution metrics
	report.Analysis.TradeDuration = calculateDistribution(durations)
}

// calculatePnLDistribution calculates the distribution of trade PnLs
func (be *BacktestEngine) calculatePnLDistribution(report *common.BacktestReport, trades []Trade) {
	if len(trades) == 0 {
		return
	}

	// Extract PnLs
	pnls := make([]float64, len(trades))
	for i, trade := range trades {
		pnls[i] = trade.PnL
	}

	// Calculate distribution metrics
	report.Analysis.PnLDistribution = calculateDistribution(pnls)
}

// calculateDistribution calculates statistical distribution metrics
func calculateDistribution(values []float64) common.Distribution {
	if len(values) == 0 {
		return common.Distribution{}
	}

	// Calculate mean
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	mean := sum / float64(len(values))

	// Calculate standard deviation
	sumSquaredDiff := 0.0
	for _, v := range values {
		diff := v - mean
		sumSquaredDiff += diff * diff
	}
	stdDev := math.Sqrt(sumSquaredDiff / float64(len(values)))

	// Find min and max
	min := values[0]
	max := values[0]
	for _, v := range values {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}

	// Calculate median
	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)
	var median float64
	n := len(sorted)
	if n%2 == 0 {
		median = (sorted[n/2-1] + sorted[n/2]) / 2
	} else {
		median = sorted[n/2]
	}

	return common.Distribution{
		Mean:   mean,
		StdDev: stdDev,
		Min:    min,
		Max:    max,
		Median: median,
	}
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
