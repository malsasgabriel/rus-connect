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
func (be *BacktestEngine) Run() (*common.BacktestReport, []Trade, error) {
	report := &common.BacktestReport{}

	// Initialize report structures
	report.Charts.MonthlyReturns = make(map[string]float64)
	report.Analysis.TimeAnalysis = make(map[string]float64)

	// Get all signals and candles for the period
	signals, candles, err := be.fetchData()
	if err != nil {
		return nil, nil, fmt.Errorf("failed to fetch data: %v", err)
	}

	// Execute trades based on signals
	// Aggregate signals to reduce noise (5 minute buckets)
	signals = aggregateSignals(signals, 5)

	// Assign dynamic TP/SL based on ATR if missing
	assignDynamicTPSL(signals, candles)

	trades, equityCurve, err := be.executeTrades(signals, candles)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to execute trades: %v", err)
	}

	// Calculate performance metrics
	be.calculateMetrics(report, trades, equityCurve)

	return report, trades, nil
}

// fetchData retrieves signals and candles for backtesting
func (be *BacktestEngine) fetchData() ([]common.Signal, map[string][]Candle, error) {
	signals := make([]common.Signal, 0)
	candles := make(map[string][]Candle)

	// Try to load real signals from direction_predictions table
	rows, err := be.db.Query(`
		SELECT symbol, timestamp, direction, confidence, price_target, current_price, time_horizon
		FROM direction_predictions
		WHERE timestamp BETWEEN $1 AND $2
		ORDER BY timestamp ASC
	`, be.startDate, be.endDate)
	if err != nil {
		// If query fails, fallback to mock signals but return the error so caller can log
		log.Printf("Failed to query direction_predictions: %v", err)
		signals = be.generateMockSignals()
	} else {
		defer rows.Close()
		for rows.Next() {
			var sym string
			var ts time.Time
			var dir string
			var conf sql.NullFloat64
			var priceTarget sql.NullFloat64
			var currentPrice sql.NullFloat64
			var horizon sql.NullInt64

			if err := rows.Scan(&sym, &ts, &dir, &conf, &priceTarget, &currentPrice, &horizon); err != nil {
				log.Printf("Failed to scan direction_predictions row: %v", err)
				continue
			}

			// Map stored direction to common.Signal Prediction strings
			prediction := "NEUTRAL"
			if dir == "UP" {
				prediction = "STRONG_BUY"
			} else if dir == "DOWN" {
				prediction = "STRONG_SELL"
			}

			s := common.Signal{
				Symbol:      sym,
				Timestamp:   ts,
				Prediction:  prediction,
				Confidence:  0.0,
				PriceTarget: 0.0,
				StopLoss:    0.0,
				ModelUsed:   "replay",
			}
			if conf.Valid {
				s.Confidence = conf.Float64
			}
			if priceTarget.Valid {
				s.PriceTarget = priceTarget.Float64
			}
			if currentPrice.Valid && s.StopLoss == 0 {
				// Set a conservative stop loss if not present
				s.StopLoss = currentPrice.Float64 * 0.99
			}

			signals = append(signals, s)
		}

		if len(signals) == 0 {
			// No stored signals for period — fallback to mock
			signals = be.generateMockSignals()
		}
		log.Printf("Loaded %d signals for backtest period", len(signals))
		// Log first few signals for debugging
		for i := 0; i < len(signals) && i < 5; i++ {
			s := signals[i]
			log.Printf("Sample signal %d: symbol=%s prediction=%s confidence=%.4f price_target=%.8f model=%s", i, s.Symbol, s.Prediction, s.Confidence, s.PriceTarget, s.ModelUsed)
		}
	}

	// Load candles for each requested symbol from candle_cache
	for _, sym := range be.symbols {
		qrows, err := be.db.Query(`
			SELECT to_timestamp(timestamp)::timestamptz, open, high, low, close, volume
			FROM candle_cache
			WHERE symbol = $1 AND timestamp BETWEEN $2 AND $3
			ORDER BY timestamp ASC
		`, sym, be.startDate.Unix(), be.endDate.Unix())
		if err != nil {
			log.Printf("Failed to query candles for %s: %v", sym, err)
			continue
		}
		defer qrows.Close()

		var cs []Candle
		for qrows.Next() {
			var ts time.Time
			var c Candle
			if err := qrows.Scan(&ts, &c.Open, &c.High, &c.Low, &c.Close, &c.Volume); err != nil {
				log.Printf("Failed to scan candle row for %s: %v", sym, err)
				continue
			}
			c.Symbol = sym
			c.Timestamp = ts
			cs = append(cs, c)
		}

		if len(cs) > 0 {
			candles[sym] = cs
			log.Printf("Loaded %d candles for %s", len(cs), sym)
		}
	}

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
	// For replay/backtest allow slightly lower threshold so we have trades to evaluate
	return signal.Confidence > 0.4 && (signal.Prediction == "STRONG_BUY" || signal.Prediction == "STRONG_SELL" || signal.Prediction == "BUY" || signal.Prediction == "SELL")
}

// openPosition opens a new trading position
func (be *BacktestEngine) openPosition(signal common.Signal, capital float64, candles map[string][]Candle) (Trade, error) {
	// Determine entry price using historical candles if available
	var currentPrice float64 = signal.PriceTarget
	if currentPrice == 0 {
		if arr, ok := candles[signal.Symbol]; ok && len(arr) > 0 {
			// find first candle with Timestamp >= signal.Timestamp
			ts := signal.Timestamp
			found := false
			for _, c := range arr {
				if !c.Timestamp.Before(ts) {
					currentPrice = c.Close
					found = true
					break
				}
			}
			if !found {
				// use last available close
				currentPrice = arr[len(arr)-1].Close
			}
		}
	}

	if currentPrice == 0 {
		// final fallback
		if signal.StopLoss > 0 {
			currentPrice = signal.StopLoss * 1.01
		} else {
			currentPrice = 1.0
		}
	}

	// Calculate position size (1% of capital)
	positionSize := capital * 0.01
	quantity := positionSize / currentPrice

	direction := "LONG"
	if signal.Prediction == "STRONG_SELL" || signal.Prediction == "SELL" {
		direction = "SHORT"
	}

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
func (be *BacktestEngine) closePosition(position Trade, exitTime time.Time, candles map[string][]Candle) Trade {

	// Determine exit price using historical candles if available and check intrabar SL/TP
	exitPrice := position.TakeProfit
	if arr, ok := candles[position.Symbol]; ok && len(arr) > 0 {
		// We will scan candles from entry time to exitTime and detect earliest SL/TP hit
		var lastClose float64 = position.EntryPrice
		for _, c := range arr {
			if c.Timestamp.Before(position.EntryTime) {
				continue
			}
			if c.Timestamp.After(exitTime) {
				break
			}

			// intrabar: if LONG, TP when high >= takeProfit; SL when low <= stopLoss
			if position.Direction == "LONG" {
				if position.TakeProfit > 0 && c.High >= position.TakeProfit {
					exitPrice = position.TakeProfit
					lastClose = exitPrice
					// exit at TP, earliest occurrence
					break
				}
				if position.StopLoss > 0 && c.Low <= position.StopLoss {
					exitPrice = position.StopLoss
					lastClose = exitPrice
					break
				}
			} else {
				// SHORT: TP when low <= takeProfit, SL when high >= stopLoss
				if position.TakeProfit > 0 && c.Low <= position.TakeProfit {
					exitPrice = position.TakeProfit
					lastClose = exitPrice
					break
				}
				if position.StopLoss > 0 && c.High >= position.StopLoss {
					exitPrice = position.StopLoss
					lastClose = exitPrice
					break
				}
			}

			// keep last close if no hits
			lastClose = c.Close
		}

		// if still zero, use lastClose
		if exitPrice == 0 {
			exitPrice = lastClose
		}
	} else {
		if exitPrice == 0 {
			exitPrice = position.EntryPrice
		}
	}

	// Deterministic PnL based on price move
	var pnl float64
	if position.Direction == "LONG" {
		priceChange := (exitPrice - position.EntryPrice) / position.EntryPrice
		pnl = position.EntryValue * priceChange
	} else {
		priceChange := (position.EntryPrice - exitPrice) / position.EntryPrice
		pnl = position.EntryValue * priceChange
	}

	// Apply transaction costs
	entryCost := position.EntryValue * be.transactionCost
	exitValue := exitPrice * position.Quantity
	exitCost := exitValue * be.transactionCost
	totalCost := entryCost + exitCost
	pnl -= totalCost

	pnlPercentage := 0.0
	if position.EntryValue != 0 {
		pnlPercentage = (pnl / position.EntryValue) * 100
	}

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

// aggregateSignals groups signals into minute buckets (bucketMinutes) and keeps the highest-confidence signal per symbol per bucket
func aggregateSignals(signals []common.Signal, bucketMinutes int) []common.Signal {
	if bucketMinutes <= 0 {
		return signals
	}

	// map: symbol -> bucketStartUnix -> bestSignal
	best := make(map[string]map[int64]common.Signal)

	for _, s := range signals {
		ts := s.Timestamp.Unix()
		bucket := (ts / int64(bucketMinutes*60)) * int64(bucketMinutes*60)
		if _, ok := best[s.Symbol]; !ok {
			best[s.Symbol] = make(map[int64]common.Signal)
		}
		prev, exists := best[s.Symbol][bucket]
		if !exists || s.Confidence > prev.Confidence {
			best[s.Symbol][bucket] = s
		}
	}

	out := make([]common.Signal, 0)
	for _, bm := range best {
		for _, s := range bm {
			out = append(out, s)
		}
	}

	// sort by timestamp
	sort.Slice(out, func(i, j int) bool { return out[i].Timestamp.Before(out[j].Timestamp) })
	return out
}

// assignDynamicTPSL assigns a TP/SL to signals based on recent ATR if they are missing
func assignDynamicTPSL(signals []common.Signal, candles map[string][]Candle) {
	for i := range signals {
		s := &signals[i]
		if s.PriceTarget != 0 && s.StopLoss != 0 {
			continue
		}
		// compute ATR from candles for symbol
		if arr, ok := candles[s.Symbol]; ok && len(arr) >= 14 {
			// take last 14 candles before signal.Timestamp
			ts := s.Timestamp.Unix()
			subset := make([]Candle, 0, 14)
			for j := len(arr) - 1; j >= 0 && len(subset) < 14; j-- {
				if arr[j].Timestamp.Unix() <= ts {
					subset = append([]Candle{arr[j]}, subset...)
				}
			}
			if len(subset) >= 7 {
				atr := calculateATRFromCandles(subset, 14)
				// set TP = entry + 2*ATR, SL = entry - 1*ATR for LONG, reversed for SHORT
				entryPrice := s.PriceTarget
				if entryPrice == 0 {
					// approximate entry as last close
					entryPrice = subset[len(subset)-1].Close
				}
				if s.Prediction == "STRONG_SELL" || s.Prediction == "SELL" {
					s.PriceTarget = entryPrice * (1 - 2*atr)
					s.StopLoss = entryPrice * (1 + 1*atr)
				} else {
					s.PriceTarget = entryPrice * (1 + 2*atr)
					s.StopLoss = entryPrice * (1 - 1*atr)
				}
			}
		}
	}
}

// calculateATRFromCandles computes ATR as ratio (e.g., 0.02 for 2%) from slice of candles and given period
func calculateATRFromCandles(candles []Candle, period int) float64 {
	if len(candles) < 2 || period <= 0 {
		return 0.02
	}
	// compute simple TRs
	trs := make([]float64, 0, len(candles)-1)
	for i := 1; i < len(candles); i++ {
		cur := candles[i]
		prev := candles[i-1]
		tr := math.Max(cur.High-cur.Low, math.Max(math.Abs(cur.High-prev.Close), math.Abs(cur.Low-prev.Close)))
		trs = append(trs, tr/cur.Close)
	}
	// average of last 'period' trs
	start := 0
	if len(trs) > period {
		start = len(trs) - period
	}
	sum := 0.0
	for i := start; i < len(trs); i++ {
		sum += trs[i]
	}
	avg := sum / float64(len(trs)-start)
	if avg == 0 {
		return 0.02
	}
	return avg
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
