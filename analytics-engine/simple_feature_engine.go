package main

import (
	"math"
	"sync"
)

// SimpleFeatureEngine extracts features from candles
type SimpleFeatureEngine struct {
	history map[string][]Candle
	maxSize int
	mu      sync.Mutex
}

// NewSimpleFeatureEngine creates a new feature engine
func NewSimpleFeatureEngine() *SimpleFeatureEngine {
	return &SimpleFeatureEngine{
		history: make(map[string][]Candle),
		maxSize: 60, // Reduced from 100 to 60 - only need 1 hour for feature calculation
	}
}

// AddCandle adds a candle and returns features
func (fe *SimpleFeatureEngine) AddCandle(candle Candle) []float64 {
	fe.mu.Lock()
	defer fe.mu.Unlock()

	symbol := candle.Symbol
	if _, exists := fe.history[symbol]; !exists {
		fe.history[symbol] = make([]Candle, 0, fe.maxSize)
	}

	fe.history[symbol] = append(fe.history[symbol], candle)

	// Keep only last maxSize candles
	// FIX: Properly reclaim memory by creating new slice
	if len(fe.history[symbol]) > fe.maxSize {
		oldSlice := fe.history[symbol]
		newSlice := make([]Candle, fe.maxSize)
		copy(newSlice, oldSlice[len(oldSlice)-fe.maxSize:])
		fe.history[symbol] = newSlice
	}

	return fe.extractFeaturesFromCandles(fe.history[symbol])
}

// ExtractFeatures extracts 16 features from candle history
func (fe *SimpleFeatureEngine) ExtractFeatures(symbol string) []float64 {
	fe.mu.Lock()
	defer fe.mu.Unlock()

	return fe.extractFeaturesFromCandles(fe.history[symbol])
}

func (fe *SimpleFeatureEngine) extractFeaturesFromCandles(candles []Candle) []float64 {
	if len(candles) < 3 {
		// Return nil if not enough data (need at least 3 for basic momentum)
		return nil
	}

	features := make([]float64, 16)
	current := candles[len(candles)-1]

	// Feature 1: Price change (5 minutes)
	if len(candles) >= 6 {
		price5mAgo := candles[len(candles)-6].Close
		if price5mAgo != 0 {
			features[0] = (current.Close - price5mAgo) / price5mAgo
		}
	}

	// Feature 2: Price change (15 minutes)
	if len(candles) >= 16 {
		price15mAgo := candles[len(candles)-16].Close
		if price15mAgo != 0 {
			features[1] = (current.Close - price15mAgo) / price15mAgo
		}
	}

	// Feature 3: Volume change
	if len(candles) >= 6 {
		volume5mAgo := candles[len(candles)-6].Volume
		if volume5mAgo != 0 {
			features[2] = (current.Volume - volume5mAgo) / volume5mAgo
		}
	}

	// Feature 4: RSI (14 periods)
	features[3] = fe.calculateRSI(candles, 14)

	// Feature 5: SMA 5 (Relative to Close)
	sma5 := fe.calculateSMA(candles, 5)
	if sma5 != 0 {
		features[4] = (current.Close - sma5) / sma5
	}

	// Feature 6: SMA 15 (Relative to Close)
	sma15 := fe.calculateSMA(candles, 15)
	if sma15 != 0 {
		features[5] = (current.Close - sma15) / sma15
	}

	// Feature 7: Volatility (5 periods)
	features[6] = fe.CalculateVolatility(candles, 5)

	// Feature 8: High-Low ratio
	if current.Low != 0 {
		features[7] = (current.High - current.Low) / current.Low
	}

	// Feature 9: Open-Close ratio
	if current.Open != 0 {
		features[8] = (current.Close - current.Open) / current.Open
	}

	// Feature 10: Volume SMA ratio
	volumeSMA := fe.calculateVolumeSMA(candles, 10)
	if volumeSMA != 0 {
		features[9] = current.Volume / volumeSMA
	}

	// Feature 11: Price position in recent range
	features[10] = fe.calculatePricePosition(candles, 20)

	// Feature 12: Trend strength
	features[11] = fe.calculateTrendStrength(candles, 10)

	// Feature 13: MACD (Relative to Close)
	macd := fe.calculateMACD(candles)
	if current.Close != 0 {
		features[12] = macd / current.Close
	}

	// Feature 14: Bollinger position
	features[13] = fe.calculateBollingerPosition(candles, 20)

	// Feature 15: Momentum
	features[14] = fe.calculateMomentum(candles, 5)

	// Feature 16: Time of day (normalized)
	features[15] = float64(current.Timestamp%86400) / 86400.0 // 0-1 based on seconds in day

	return features
}

// GetHistory returns the candle history for a symbol (for label generation)
func (fe *SimpleFeatureEngine) GetHistory(symbol string) []Candle {
	fe.mu.Lock()
	defer fe.mu.Unlock()

	if candles, exists := fe.history[symbol]; exists {
		// Return a copy to avoid race conditions
		result := make([]Candle, len(candles))
		copy(result, candles)
		return result
	}
	return nil
}

// Helper functions for technical indicators

func (fe *SimpleFeatureEngine) calculateSMA(candles []Candle, period int) float64 {
	if len(candles) < period {
		return 0
	}

	sum := 0.0
	for i := len(candles) - period; i < len(candles); i++ {
		sum += candles[i].Close
	}
	return sum / float64(period)
}

func (fe *SimpleFeatureEngine) calculateVolumeSMA(candles []Candle, period int) float64 {
	if len(candles) < period {
		return 0
	}

	sum := 0.0
	for i := len(candles) - period; i < len(candles); i++ {
		sum += candles[i].Volume
	}
	return sum / float64(period)
}

func (fe *SimpleFeatureEngine) calculateRSI(candles []Candle, period int) float64 {
	if len(candles) < period+1 {
		return 50 // Neutral RSI
	}

	var avgGain, avgLoss float64

	for i := len(candles) - period; i < len(candles); i++ {
		change := candles[i].Close - candles[i-1].Close
		if change > 0 {
			avgGain += change
		} else {
			avgLoss += -change
		}
	}

	avgGain /= float64(period)
	avgLoss /= float64(period)

	if avgLoss == 0 {
		return 100
	}

	rs := avgGain / avgLoss
	return 100 - (100 / (1 + rs))
}

func (fe *SimpleFeatureEngine) CalculateVolatility(candles []Candle, period int) float64 {
	if len(candles) < period+1 {
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

	// Calculate standard deviation
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

func (fe *SimpleFeatureEngine) calculatePricePosition(candles []Candle, period int) float64 {
	if len(candles) < period {
		return 0.5
	}

	recent := candles[len(candles)-period:]
	highest := recent[0].High
	lowest := recent[0].Low

	for _, candle := range recent {
		if candle.High > highest {
			highest = candle.High
		}
		if candle.Low < lowest {
			lowest = candle.Low
		}
	}

	current := candles[len(candles)-1].Close
	if highest == lowest {
		return 0.5
	}

	return (current - lowest) / (highest - lowest)
}

func (fe *SimpleFeatureEngine) calculateTrendStrength(candles []Candle, period int) float64 {
	if len(candles) < period {
		return 0
	}

	start := candles[len(candles)-period].Close
	end := candles[len(candles)-1].Close

	if start == 0 {
		return 0
	}

	return (end - start) / start
}

func (fe *SimpleFeatureEngine) calculateMACD(candles []Candle) float64 {
	if len(candles) < 26 {
		return 0
	}

	ema12 := fe.calculateEMA(candles, 12)
	ema26 := fe.calculateEMA(candles, 26)

	return ema12 - ema26
}

func (fe *SimpleFeatureEngine) calculateEMA(candles []Candle, period int) float64 {
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

func (fe *SimpleFeatureEngine) calculateBollingerPosition(candles []Candle, period int) float64 {
	if len(candles) < period {
		return 0.5
	}

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

	lowerBand := sma - 2*stdDev
	upperBand := sma + 2*stdDev

	if upperBand == lowerBand {
		return 0.5
	}

	position := (current - lowerBand) / (upperBand - lowerBand)

	// Clamp to [0, 1]
	if position < 0 {
		return 0
	}
	if position > 1 {
		return 1
	}

	return position
}

func (fe *SimpleFeatureEngine) calculateMomentum(candles []Candle, period int) float64 {
	if len(candles) < period+1 {
		return 0
	}

	current := candles[len(candles)-1].Close
	past := candles[len(candles)-1-period].Close

	if past == 0 {
		return 0
	}

	return (current - past) / past
}
