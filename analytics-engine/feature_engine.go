package main

import (
	"math"
	"sync"
	"time"
)

// Define Candle struct here to avoid import cycle with feature_engine
type Candle struct {
	Symbol    string
	Timestamp int64
	Open      float64
	High      float64
	Low       float64
	Close     float64
	Volume    float64
}

// FeatureSet represents the calculated features for ML model input.
type FeatureSet struct {
	Symbol             string  `json:"symbol"`
	Timestamp          int64   `json:"timestamp"`
	Price              float64 `json:"price"`
	Volume             float64 `json:"volume"`
	SMA5               float64 `json:"sma5"`
	SMA15              float64 `json:"sma15"`
	SMA30              float64 `json:"sma30"`
	RSI                float64 `json:"rsi"`
	VolumeSMA30        float64 `json:"volume_sma30"`
	VolumeSpikeRatio   float64 `json:"volume_spike_ratio"`
	PriceChange5m      float64 `json:"price_change_5m"`
	PriceChange15m     float64 `json:"price_change_15m"`
	Volatility         float64 `json:"volatility"` // Renamed from Volatility5m
	Volatility5m       float64 `json:"volatility_5m"`
	OrderBookImbalance float64 `json:"order_book_imbalance"` // To be filled from OrderBook data
	IsVolumeSpike      bool    `json:"is_volume_spike"`
	IsPriceBreakout    bool    `json:"is_price_breakout"`
	// Enhanced features for better ML detection
	VolumeMomentum    float64 `json:"volume_momentum"`
	PriceMomentum     float64 `json:"price_momentum"`
	BollingerPosition float64 `json:"bollinger_position"`
	VWAP              float64 `json:"vwap"`
	MACD              float64 `json:"macd"`
	TrendStrength     float64 `json:"trend_strength"`
}

// FeatureEngine manages historical candle data and calculates features.
type FeatureEngine struct {
	history map[string][]Candle
	maxSize int        // Max number of 1-minute candles to keep (e.g., 1440 for 24 hours)
	mu      sync.Mutex // Protects history map
}

func NewFeatureEngine() *FeatureEngine {
	return &FeatureEngine{
		history: make(map[string][]Candle),
		maxSize: 1440, // 24 hours of 1-minute candles
	}
}

func (fe *FeatureEngine) AddCandle(candle Candle) FeatureSet {
	fe.mu.Lock()
	defer fe.mu.Unlock()

	symbol := candle.Symbol
	if _, exists := fe.history[symbol]; !exists {
		fe.history[symbol] = make([]Candle, 0, fe.maxSize)
	}

	fe.history[symbol] = append(fe.history[symbol], candle)

	// Maintain window size
	if len(fe.history[symbol]) > fe.maxSize {
		fe.history[symbol] = fe.history[symbol][1:]
	}

	return fe.CalculateFeatures(symbol)
}

// GetCandleHistory retrieves the candle history for a given symbol.
func (fe *FeatureEngine) GetCandleHistory(symbol string) []Candle {
	fe.mu.Lock()
	defer fe.mu.Unlock()
	return fe.history[symbol]
}

func (fe *FeatureEngine) CalculateFeatures(symbol string) FeatureSet {
	fe.mu.Lock()
	defer fe.mu.Unlock()

	candles := fe.history[symbol]
	if len(candles) < 30 { // Need at least 30 candles for SMA30, VolumeSMA30 etc.
		return FeatureSet{Symbol: symbol, Timestamp: time.Now().Unix()} // Return minimal if not enough history
	}

	current := candles[len(candles)-1]

	features := FeatureSet{
		Symbol:    symbol,
		Timestamp: current.Timestamp,
		Price:     current.Close,
		Volume:    current.Volume,
	}

	// Calculate SMAs
	features.SMA5 = fe.calculateSMA(candles, 5)
	features.SMA15 = fe.calculateSMA(candles, 15)
	features.SMA30 = fe.calculateSMA(candles, 30)
	features.VolumeSMA30 = fe.calculateVolumeSMA(candles, 30)

	// Volume spike ratio
	if features.VolumeSMA30 > 0 {
		features.VolumeSpikeRatio = current.Volume / features.VolumeSMA30
		features.IsVolumeSpike = features.VolumeSpikeRatio > 3.0 // Threshold 3x average volume
	}

	// Price changes
	if len(candles) >= 6 { // 5 minutes ago = current index - 5 (0-indexed)
		price5mAgo := candles[len(candles)-6].Close
		if price5mAgo != 0 {
			features.PriceChange5m = (current.Close - price5mAgo) / price5mAgo
		}
	}

	if len(candles) >= 16 { // 15 minutes ago
		price15mAgo := candles[len(candles)-16].Close
		if price15mAgo != 0 {
			features.PriceChange15m = (current.Close - price15mAgo) / price15mAgo
		}
	}

	// Volatility (standard deviation of returns over last 5 minutes)
	features.Volatility5m = fe.calculateVolatility(candles, 5)
	features.Volatility = features.Volatility5m // For compatibility

	// RSI approximation (requires more history, e.g., 14 periods for standard RSI)
	features.RSI = fe.calculateRSI(candles, 14)

	// Enhanced features for better detection
	features.VolumeMomentum = fe.calculateVolumeMomentum(candles, 5)
	features.PriceMomentum = fe.calculatePriceMomentum(candles, 5)
	features.BollingerPosition = fe.calculateBollingerPosition(candles, 20)
	features.VWAP = fe.calculateVWAP(candles, 20)
	features.MACD = fe.calculateMACD(candles)
	features.TrendStrength = fe.calculateTrendStrength(candles, 10)

	// Placeholder for OrderBookImbalance (requires real-time order book data)
	features.OrderBookImbalance = 0.0

	// Price Breakout (price > high_30m, simplified)
	if len(candles) >= 30 {
		highest30m := 0.0
		for i := len(candles) - 30; i < len(candles)-1; i++ {
			if candles[i].High > highest30m {
				highest30m = candles[i].High
			}
		}
		features.IsPriceBreakout = current.Close > highest30m && highest30m != 0
	}

	return features
}

// Helper function to calculate Simple Moving Average (SMA)
func (fe *FeatureEngine) calculateSMA(candles []Candle, period int) float64 {
	if len(candles) < period {
		return 0
	}
	sum := 0.0
	for i := len(candles) - period; i < len(candles); i++ {
		sum += candles[i].Close
	}
	return sum / float64(period)
}

// Helper function to calculate Volume Simple Moving Average (Volume SMA)
func (fe *FeatureEngine) calculateVolumeSMA(candles []Candle, period int) float64 {
	if len(candles) < period {
		return 0
	}
	sum := 0.0
	for i := len(candles) - period; i < len(candles); i++ {
		sum += candles[i].Volume
	}
	return sum / float64(period)
}

// Helper function to calculate Volatility (Standard Deviation of log returns)
func (fe *FeatureEngine) calculateVolatility(candles []Candle, period int) float64 {
	if len(candles) < period+1 { // Need at least period+1 candles to calculate period returns
		return 0
	}
	returns := make([]float64, 0, period)
	for i := len(candles) - period; i < len(candles); i++ {
		if candles[i-1].Close != 0 {
			ret := math.Log(candles[i].Close / candles[i-1].Close)
			returns = append(returns, ret)
		}
	}
	if len(returns) == 0 {
		return 0
	}
	mean := 0.0
	for _, r := range returns {
		mean += r
	}
	mean /= float64(len(returns))

	variance := 0.0
	for _, r := range returns {
		variance += math.Pow(r-mean, 2)
	}
	variance /= float64(len(returns))
	return math.Sqrt(variance)
}

// Helper function to calculate RSI (Relative Strength Index)
func (fe *FeatureEngine) calculateRSI(candles []Candle, period int) float64 {
	if len(candles) < period+1 {
		return 0
	}

	var avgGain, avgLoss float64
	gains := make([]float64, 0, period)
	losses := make([]float64, 0, period)

	// Calculate initial gains and losses for the first 'period' candles
	for i := len(candles) - period; i < len(candles); i++ {
		change := candles[i].Close - candles[i-1].Close
		if change > 0 {
			gains = append(gains, change)
			losses = append(losses, 0)
		} else {
			gains = append(gains, 0)
			losses = append(losses, -change)
		}
	}

	// Simple average for initial calculation (more robust if using Wilders' smoothing for subsequent)
	for _, g := range gains {
		avgGain += g
	}
	avgGain /= float64(period)

	for _, l := range losses {
		avgLoss += l
	}
	avgLoss /= float64(period)

	if avgLoss == 0 {
		return 100 // Avoid division by zero, RSI is 100 if no losses
	}

	rs := avgGain / avgLoss
	rsi := 100 - (100 / (1 + rs))
	return rsi
}

// calculateVolumeMomentum calculates volume momentum over specified period
func (fe *FeatureEngine) calculateVolumeMomentum(candles []Candle, period int) float64 {
	if len(candles) < period*2 {
		return 0
	}

	// Compare recent average volume to previous average volume
	recentSum := 0.0
	previousSum := 0.0

	for i := len(candles) - period; i < len(candles); i++ {
		recentSum += candles[i].Volume
	}

	for i := len(candles) - period*2; i < len(candles)-period; i++ {
		previousSum += candles[i].Volume
	}

	recentAvg := recentSum / float64(period)
	previousAvg := previousSum / float64(period)

	if previousAvg == 0 {
		return 0
	}

	return (recentAvg - previousAvg) / previousAvg
}

// calculatePriceMomentum calculates price momentum over specified period
func (fe *FeatureEngine) calculatePriceMomentum(candles []Candle, period int) float64 {
	if len(candles) < period+1 {
		return 0
	}

	current := candles[len(candles)-1].Close
	previous := candles[len(candles)-period-1].Close

	if previous == 0 {
		return 0
	}

	return (current - previous) / previous
}

// calculateBollingerPosition calculates position relative to Bollinger Bands
func (fe *FeatureEngine) calculateBollingerPosition(candles []Candle, period int) float64 {
	if len(candles) < period {
		return 0.5 // Middle position if not enough data
	}

	// Calculate SMA and standard deviation
	sma := fe.calculateSMA(candles, period)
	current := candles[len(candles)-1].Close

	// Calculate standard deviation
	sum := 0.0
	for i := len(candles) - period; i < len(candles); i++ {
		diff := candles[i].Close - sma
		sum += diff * diff
	}
	stdDev := math.Sqrt(sum / float64(period))

	if stdDev == 0 {
		return 0.5
	}

	// Position between lower band (sma - 2*stdDev) and upper band (sma + 2*stdDev)
	lowerBand := sma - 2*stdDev
	upperBand := sma + 2*stdDev

	if upperBand == lowerBand {
		return 0.5
	}

	position := (current - lowerBand) / (upperBand - lowerBand)

	// Clamp between 0 and 1
	if position < 0 {
		return 0
	}
	if position > 1 {
		return 1
	}

	return position
}

// calculateVWAP calculates Volume Weighted Average Price
func (fe *FeatureEngine) calculateVWAP(candles []Candle, period int) float64 {
	if len(candles) < period {
		return 0
	}

	totalVolumePrice := 0.0
	totalVolume := 0.0

	for i := len(candles) - period; i < len(candles); i++ {
		typicalPrice := (candles[i].High + candles[i].Low + candles[i].Close) / 3
		volumePrice := typicalPrice * candles[i].Volume
		totalVolumePrice += volumePrice
		totalVolume += candles[i].Volume
	}

	if totalVolume == 0 {
		return 0
	}

	return totalVolumePrice / totalVolume
}

// calculateMACD calculates Moving Average Convergence Divergence
func (fe *FeatureEngine) calculateMACD(candles []Candle) float64 {
	if len(candles) < 26 {
		return 0
	}

	// Calculate EMA12 and EMA26
	ema12 := fe.calculateEMA(candles, 12)
	ema26 := fe.calculateEMA(candles, 26)

	return ema12 - ema26
}

// calculateEMA calculates Exponential Moving Average
func (fe *FeatureEngine) calculateEMA(candles []Candle, period int) float64 {
	if len(candles) < period {
		return 0
	}

	alpha := 2.0 / (float64(period) + 1.0)
	ema := candles[len(candles)-period].Close

	for i := len(candles) - period + 1; i < len(candles); i++ {
		ema = alpha*candles[i].Close + (1-alpha)*ema
	}

	return ema
}

// calculateTrendStrength calculates trend strength using linear regression
func (fe *FeatureEngine) calculateTrendStrength(candles []Candle, period int) float64 {
	if len(candles) < period {
		return 0
	}

	// Linear regression to find trend slope
	n := float64(period)
	sumX := 0.0
	sumY := 0.0
	sumXY := 0.0
	sumX2 := 0.0

	for i := 0; i < period; i++ {
		x := float64(i)
		y := candles[len(candles)-period+i].Close
		sumX += x
		sumY += y
		sumXY += x * y
		sumX2 += x * x
	}

	// Calculate slope (trend strength)
	numerator := n*sumXY - sumX*sumY
	denominator := n*sumX2 - sumX*sumX

	if denominator == 0 {
		return 0
	}

	slope := numerator / denominator

	// Normalize slope relative to current price to get percentage trend strength
	currentPrice := candles[len(candles)-1].Close
	if currentPrice == 0 {
		return 0
	}

	return slope / currentPrice * float64(period) // Scale by period for meaningful values
}
