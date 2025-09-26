package main

import (
	"math"
)

// Technical indicator calculations for direction prediction

func calculateSMA(history []Candle, period int) float64 {
	if len(history) < period {
		return 0
	}

	sum := 0.0
	for i := len(history) - period; i < len(history); i++ {
		sum += history[i].Close
	}
	return sum / float64(period)
}

func calculateVolumeSMA(history []Candle, period int) float64 {
	if len(history) < period {
		return 0
	}

	sum := 0.0
	for i := len(history) - period; i < len(history); i++ {
		sum += history[i].Volume
	}
	return sum / float64(period)
}

func calculateRSI(history []Candle, period int) float64 {
	if len(history) < period+1 {
		return 50 // Neutral
	}

	gains := 0.0
	losses := 0.0

	for i := len(history) - period; i < len(history); i++ {
		change := history[i].Close - history[i-1].Close
		if change > 0 {
			gains += change
		} else {
			losses -= change
		}
	}

	avgGain := gains / float64(period)
	avgLoss := losses / float64(period)

	if avgLoss == 0 {
		return 100
	}

	rs := avgGain / avgLoss
	return 100 - (100 / (1 + rs))
}

func calculateMACD(history []Candle) (float64, float64, float64) {
	if len(history) < 26 {
		return 0, 0, 0
	}

	ema12 := calculateEMA(history, 12)
	ema26 := calculateEMA(history, 26)
	macd := ema12 - ema26

	// Simple signal line (9-period EMA of MACD)
	signal := macd * 0.8 // Simplified
	hist := macd - signal

	return macd, signal, hist
}

func calculateEMA(history []Candle, period int) float64 {
	if len(history) < period {
		return 0
	}

	multiplier := 2.0 / (float64(period) + 1.0)
	ema := history[len(history)-period].Close

	for i := len(history) - period + 1; i < len(history); i++ {
		ema = (history[i].Close * multiplier) + (ema * (1 - multiplier))
	}

	return ema
}

func calculateATR(history []Candle, period int) float64 {
	if len(history) < period+1 {
		return 0
	}

	trSum := 0.0
	for i := len(history) - period; i < len(history); i++ {
		high := history[i].High
		low := history[i].Low
		prevClose := history[i-1].Close

		tr1 := high - low
		tr2 := math.Abs(high - prevClose)
		tr3 := math.Abs(low - prevClose)

		tr := math.Max(tr1, math.Max(tr2, tr3))
		trSum += tr
	}

	return trSum / float64(period)
}

func calculateBollingerBands(history []Candle, period int, stdDevs float64) (float64, float64) {
	if len(history) < period {
		return 0, 0
	}

	sma := calculateSMA(history, period)

	// Calculate standard deviation
	sum := 0.0
	for i := len(history) - period; i < len(history); i++ {
		diff := history[i].Close - sma
		sum += diff * diff
	}

	variance := sum / float64(period)
	stdDev := math.Sqrt(variance)

	upper := sma + (stdDevs * stdDev)
	lower := sma - (stdDevs * stdDev)

	return upper, lower
}

func calculateSupportResistance(history []Candle) (float64, float64) {
	if len(history) < 20 {
		return 0, 0
	}

	// Find recent highs and lows
	recentLows := make([]float64, 0)
	recentHighs := make([]float64, 0)

	lookback := 100
	if len(history) < lookback {
		lookback = len(history)
	}

	start := len(history) - lookback

	for i := start + 2; i < len(history)-2; i++ {
		// Local minimum
		if history[i].Low < history[i-1].Low && history[i].Low < history[i+1].Low &&
			history[i].Low < history[i-2].Low && history[i].Low < history[i+2].Low {
			recentLows = append(recentLows, history[i].Low)
		}

		// Local maximum
		if history[i].High > history[i-1].High && history[i].High > history[i+1].High &&
			history[i].High > history[i-2].High && history[i].High > history[i+2].High {
			recentHighs = append(recentHighs, history[i].High)
		}
	}

	var support, resistance float64

	if len(recentLows) > 0 {
		// Find most significant support (highest of recent lows)
		support = recentLows[0]
		for _, low := range recentLows {
			if low > support {
				support = low
			}
		}
	}

	if len(recentHighs) > 0 {
		// Find most significant resistance (lowest of recent highs)
		resistance = recentHighs[0]
		for _, high := range recentHighs {
			if high < resistance {
				resistance = high
			}
		}
	}

	return support, resistance
}

func calculateSRStrength(history []Candle, support, resistance float64) float64 {
	if support == 0 || resistance == 0 {
		return 0
	}

	// Count touches near support/resistance levels
	tolerance := 0.002 // 0.2% tolerance

	supportTouches := 0
	resistanceTouches := 0

	lookback := 50
	if len(history) < lookback {
		lookback = len(history)
	}

	start := len(history) - lookback

	for i := start; i < len(history); i++ {
		// Check support touches
		if math.Abs(history[i].Low-support)/support < tolerance {
			supportTouches++
		}

		// Check resistance touches
		if math.Abs(history[i].High-resistance)/resistance < tolerance {
			resistanceTouches++
		}
	}

	// Strength based on number of touches
	totalTouches := supportTouches + resistanceTouches
	return math.Min(float64(totalTouches)/10.0, 1.0) // Normalize to 0-1
}

func calculateOrderFlow(history []Candle) (float64, float64) {
	if len(history) < 10 {
		return 0.5, 0.5
	}

	buyPressure := 0.0
	sellPressure := 0.0

	// Analyze last 20 candles for order flow
	lookback := 20
	if len(history) < lookback {
		lookback = len(history)
	}

	start := len(history) - lookback

	for i := start; i < len(history); i++ {
		candle := history[i]

		// Calculate where close is relative to high-low range
		if candle.High != candle.Low {
			closePosition := (candle.Close - candle.Low) / (candle.High - candle.Low)

			// Weight by volume
			volumeWeight := candle.Volume

			buyPressure += closePosition * volumeWeight
			sellPressure += (1 - closePosition) * volumeWeight
		}
	}

	total := buyPressure + sellPressure
	if total > 0 {
		buyPressure /= total
		sellPressure /= total
	}

	return buyPressure, sellPressure
}

func calculateTrend(history []Candle) (float64, float64) {
	if len(history) < 20 {
		return 0, 0
	}

	// Linear regression on closing prices
	n := 50
	if len(history) < n {
		n = len(history)
	}

	start := len(history) - n

	sumX := 0.0
	sumY := 0.0
	sumXY := 0.0
	sumX2 := 0.0

	for i := 0; i < n; i++ {
		x := float64(i)
		y := history[start+i].Close

		sumX += x
		sumY += y
		sumXY += x * y
		sumX2 += x * x
	}

	nFloat := float64(n)
	slope := (nFloat*sumXY - sumX*sumY) / (nFloat*sumX2 - sumX*sumX)

	// Normalize slope
	avgPrice := sumY / nFloat
	trendDirection := slope / avgPrice * float64(n) // Scale by period

	// Calculate R-squared for trend strength
	yMean := avgPrice
	ssRes := 0.0
	ssTot := 0.0

	for i := 0; i < n; i++ {
		x := float64(i)
		y := history[start+i].Close
		predicted := (sumY-slope*sumX)/nFloat + slope*x

		ssRes += (y - predicted) * (y - predicted)
		ssTot += (y - yMean) * (y - yMean)
	}

	rSquared := 1.0 - ssRes/ssTot
	if rSquared < 0 {
		rSquared = 0
	}

	trendStrength := rSquared

	return trendStrength, math.Tanh(trendDirection) // Bound between -1 and 1
}

func calculateVolumeProfile(history []Candle) float64 {
	if len(history) < 10 {
		return 0.5
	}

	// Calculate volume-weighted average position
	totalVolume := 0.0
	weightedPosition := 0.0

	lookback := 20
	if len(history) < lookback {
		lookback = len(history)
	}

	start := len(history) - lookback

	for i := start; i < len(history); i++ {
		candle := history[i]
		if candle.High != candle.Low {
			position := (candle.Close - candle.Low) / (candle.High - candle.Low)
			weightedPosition += position * candle.Volume
			totalVolume += candle.Volume
		}
	}

	if totalVolume > 0 {
		return weightedPosition / totalVolume
	}

	return 0.5
}

func identifyMarketPhase(history []Candle) string {
	if len(history) < 50 {
		return "UNKNOWN"
	}

	// Calculate multiple timeframe moving averages
	ma20 := calculateSMA(history, 20)
	ma50 := calculateSMA(history, 50)
	currentPrice := history[len(history)-1].Close

	// Volume analysis
	recentVolume := calculateVolumeSMA(history, 10)
	longerVolume := calculateVolumeSMA(history, 30)

	trendStrength, trendDirection := calculateTrend(history)

	// Determine phase based on multiple factors
	if trendStrength > 0.7 && math.Abs(trendDirection) > 0.3 {
		return "TRENDING"
	} else if recentVolume > longerVolume*1.2 && currentPrice > ma20 && ma20 > ma50 {
		return "ACCUMULATION"
	} else if recentVolume > longerVolume*1.2 && currentPrice < ma20 && ma20 < ma50 {
		return "DISTRIBUTION"
	} else {
		return "SIDEWAYS"
	}
}
