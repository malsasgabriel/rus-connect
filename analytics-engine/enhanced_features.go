// Self-Learning Trading AI System with LSTM + Ensemble Models
// Transforms existing technical indicators into ML-ready features for crypto prediction

package main

import (
	"math"
	"time"
)

// 🧠 ML-Ready Feature Set for Self-Learning Trading AI
type MLReadyFeatureSet struct {
	EnhancedFeatureSet // Inherit all 25+ existing indicators

	// 🎯 LSTM Sequential Features (1440 data points for neural network)
	SequentialFeatures [][]float64        `json:"sequential_features"`  // [1440][30+] matrix
	TimeBasedPatterns  TimeSeriesPatterns `json:"time_patterns"`        // Temporal patterns
	CrossAssetFeatures CrossAssetSignals  `json:"cross_asset_features"` // BTC, ETH correlations

	// 🎯 Prediction Targets for Training
	FuturePrice1H       float64 `json:"future_price_1h"`      // Actual price 1 hour later
	PriceDirection1H    string  `json:"price_direction_1h"`   // "UP", "DOWN", "SIDEWAYS"
	PercentChange1H     float64 `json:"percent_change_1h"`    // % change for regression
	VolatilityPredicted float64 `json:"volatility_predicted"` // Expected volatility

	// 🎯 ML Quality Labels
	ConfidenceLabel float64 `json:"confidence_label"` // Model confidence 0-1
	SignalStrength  float64 `json:"signal_strength"`  // Strong/Weak signal 0-1
	RiskScore       float64 `json:"risk_score"`       // Risk assessment 0-1
	MarketCondition string  `json:"market_condition"` // "BULLISH", "BEARISH", "NEUTRAL"

	// 🎯 Feature Engineering for ML
	NormalizedFeatures []float64          `json:"normalized_features"` // Scaled 0-1
	FeatureImportance  map[string]float64 `json:"feature_importance"`  // Dynamic importance
	PatternSignals     []PatternDetection `json:"pattern_signals"`     // Chart patterns
}

// 🕒 Time Series Patterns for LSTM
type TimeSeriesPatterns struct {
	HourlyTrend     TrendPattern `json:"hourly_trend"`
	DailyPattern    TrendPattern `json:"daily_pattern"`
	WeeklyPattern   TrendPattern `json:"weekly_pattern"`
	Seasonality     float64      `json:"seasonality"`    // 0-1 seasonal strength
	CyclicPattern   float64      `json:"cyclic_pattern"` // Market cycles
	VolatilityCycle float64      `json:"volatility_cycle"`
}

type TrendPattern struct {
	Direction   string  `json:"direction"`   // "UP", "DOWN", "SIDEWAYS"
	Strength    float64 `json:"strength"`    // 0-1
	Duration    int     `json:"duration"`    // periods
	Reliability float64 `json:"reliability"` // historical accuracy
}

// 🔗 Cross-Asset Analysis
type CrossAssetSignals struct {
	BTCCorrelation  float64 `json:"btc_correlation"`  // -1 to 1
	ETHCorrelation  float64 `json:"eth_correlation"`  // -1 to 1
	MarketSentiment float64 `json:"market_sentiment"` // Overall crypto sentiment
	DominanceShift  float64 `json:"dominance_shift"`  // BTC dominance change
	SectorRotation  string  `json:"sector_rotation"`  // Which coins leading
}

// 📊 Pattern Recognition
type PatternDetection struct {
	PatternType  string  `json:"pattern_type"`  // "HEAD_SHOULDERS", "TRIANGLE", etc.
	Confidence   float64 `json:"confidence"`    // Pattern confidence 0-1
	BreakoutProb float64 `json:"breakout_prob"` // Breakout probability
	TargetPrice  float64 `json:"target_price"`  // Pattern target
	Timeframe    string  `json:"timeframe"`     // "5M", "1H", "4H"
}

// EnhancedFeatureSet contains all advanced trading indicators
type EnhancedFeatureSet struct {
	Symbol    string  `json:"symbol"`
	Timestamp int64   `json:"timestamp"`
	Price     float64 `json:"price"`
	Volume    float64 `json:"volume"`

	// Moving Averages & MACD
	EMA_12         float64 `json:"ema_12"`
	EMA_26         float64 `json:"ema_26"`
	MACD           float64 `json:"macd"`
	MACD_Signal    float64 `json:"macd_signal"`
	MACD_Histogram float64 `json:"macd_histogram"`

	// Bollinger Bands
	BB_Upper    float64 `json:"bb_upper"`
	BB_Lower    float64 `json:"bb_lower"`
	BB_Position float64 `json:"bb_position"`

	// Oscillators
	RSI_14       float64 `json:"rsi_14"`
	Stochastic_K float64 `json:"stochastic_k"`
	Williams_R   float64 `json:"williams_r"`
	CCI          float64 `json:"cci"`

	// Volatility & Momentum
	ATR      float64 `json:"atr"`
	ADX      float64 `json:"adx"`
	Momentum float64 `json:"momentum"`
	ROC      float64 `json:"roc"`

	// Volume Analysis
	VWAP         float64 `json:"vwap"`
	Volume_Delta float64 `json:"volume_delta"`

	// Support/Resistance & Patterns
	Support_Level    float64 `json:"support_level"`
	Resistance_Level float64 `json:"resistance_level"`
	Doji_Pattern     float64 `json:"doji_pattern"`
	Hammer_Pattern   float64 `json:"hammer_pattern"`

	// Market Regime & ML Features
	Market_Regime     float64 `json:"market_regime"`
	Trend_Strength    float64 `json:"trend_strength"`
	Price_Momentum_5m float64 `json:"price_momentum_5m"`
	Volume_Momentum   float64 `json:"volume_momentum"`
	Liquidity_Score   float64 `json:"liquidity_score"`
}

// 🧠 Advanced ML Feature Engine for Self-Learning Trading AI
type EnhancedFeatureEngine struct {
	history           map[string][]Candle
	mlReadyHistory    map[string][]MLReadyFeatureSet
	crossAssetData    map[string]*CrossAssetAnalyzer
	patternDetector   *PatternRecognitionEngine
	featureNormalizer *FeatureNormalizer
	maxSize           int
}

// 🔄 Cross-Asset Analysis Engine
type CrossAssetAnalyzer struct {
	correlationMatrix map[string]map[string]float64
	sentimentBuffer   []float64
	dominanceHistory  []float64
	lastUpdated       time.Time
}

// 🗖️ Pattern Recognition Engine
type PatternRecognitionEngine struct {
	patternDatabase map[string]PatternTemplate
	detectionBuffer map[string][]PatternDetection
	accuracyTracker map[string]float64
}

type PatternTemplate struct {
	Name          string    `json:"name"`
	Sequence      []float64 `json:"sequence"`       // Normalized price sequence
	MinConfidence float64   `json:"min_confidence"` // Minimum confidence to emit
	SuccessRate   float64   `json:"success_rate"`   // Historical success rate
	TimeToTarget  int       `json:"time_to_target"` // Average periods to target
}

// 📊 Feature Normalization Engine
type FeatureNormalizer struct {
	featureStats map[string]*FeatureStatistics
	lookbackDays int
}

type FeatureStatistics struct {
	Mean   float64 `json:"mean"`
	StdDev float64 `json:"std_dev"`
	Min    float64 `json:"min"`
	Max    float64 `json:"max"`
}

func NewEnhancedFeatureEngine() *EnhancedFeatureEngine {
	efe := &EnhancedFeatureEngine{
		history:           make(map[string][]Candle),
		mlReadyHistory:    make(map[string][]MLReadyFeatureSet),
		crossAssetData:    make(map[string]*CrossAssetAnalyzer),
		patternDetector:   NewPatternRecognitionEngine(),
		featureNormalizer: NewFeatureNormalizer(),
		maxSize:           1440, // 24 hours of 1-minute data
	}

	// Initialize cross-asset analyzers for major crypto pairs
	majorPairs := []string{"BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "STRKUSDT"}
	for _, pair := range majorPairs {
		efe.crossAssetData[pair] = &CrossAssetAnalyzer{
			correlationMatrix: make(map[string]map[string]float64),
			sentimentBuffer:   make([]float64, 0, 100),
			dominanceHistory:  make([]float64, 0, 100),
			lastUpdated:       time.Now(),
		}
	}

	return efe
}

func NewPatternRecognitionEngine() *PatternRecognitionEngine {
	pre := &PatternRecognitionEngine{
		patternDatabase: make(map[string]PatternTemplate),
		detectionBuffer: make(map[string][]PatternDetection),
		accuracyTracker: make(map[string]float64),
	}

	// Initialize common patterns
	pre.initializeCommonPatterns()
	return pre
}

// Initialize common chart patterns for detection
func (pre *PatternRecognitionEngine) initializeCommonPatterns() {
	// Head and Shoulders pattern
	pre.patternDatabase["HEAD_SHOULDERS"] = PatternTemplate{
		Name:          "HEAD_SHOULDERS",
		Sequence:      []float64{0.8, 1.0, 0.7, 1.2, 0.6, 0.9, 0.5}, // Normalized pattern
		MinConfidence: 0.75,
		SuccessRate:   0.68,
		TimeToTarget:  15,
	}

	// Triangle pattern
	pre.patternDatabase["TRIANGLE"] = PatternTemplate{
		Name:          "TRIANGLE",
		Sequence:      []float64{1.0, 0.9, 0.95, 0.85, 0.9, 0.8, 0.85},
		MinConfidence: 0.65,
		SuccessRate:   0.72,
		TimeToTarget:  10,
	}

	// Double bottom pattern
	pre.patternDatabase["DOUBLE_BOTTOM"] = PatternTemplate{
		Name:          "DOUBLE_BOTTOM",
		Sequence:      []float64{1.0, 0.5, 0.8, 0.5, 1.0},
		MinConfidence: 0.70,
		SuccessRate:   0.75,
		TimeToTarget:  12,
	}
}

func NewFeatureNormalizer() *FeatureNormalizer {
	return &FeatureNormalizer{
		featureStats: make(map[string]*FeatureStatistics),
		lookbackDays: 30,
	}
}

func (efe *EnhancedFeatureEngine) AddCandle(candle Candle) EnhancedFeatureSet {
	symbol := candle.Symbol
	if _, exists := efe.history[symbol]; !exists {
		efe.history[symbol] = make([]Candle, 0, efe.maxSize)
	}

	efe.history[symbol] = append(efe.history[symbol], candle)
	if len(efe.history[symbol]) > efe.maxSize {
		efe.history[symbol] = efe.history[symbol][1:]
	}

	return efe.CalculateEnhancedFeatures(symbol)
}

func (efe *EnhancedFeatureEngine) CalculateEnhancedFeatures(symbol string) EnhancedFeatureSet {
	candles := efe.history[symbol]
	if len(candles) < 20 {
		return EnhancedFeatureSet{Symbol: symbol}
	}

	current := candles[len(candles)-1]
	features := EnhancedFeatureSet{
		Symbol:    symbol,
		Timestamp: current.Timestamp,
		Price:     current.Close,
		Volume:    current.Volume,
	}

	// Calculate all enhanced indicators
	if len(candles) >= 26 {
		features.EMA_12 = calculateEMA(candles, 12)
		features.EMA_26 = calculateEMA(candles, 26)
		macd, signal, histogram := calculateMACD(candles)
		features.MACD = macd
		features.MACD_Signal = signal
		features.MACD_Histogram = histogram
	}

	if len(candles) >= 20 {
		upper, lower := calculateBollingerBands(candles, 20, 2.0)
		features.BB_Upper = upper
		features.BB_Lower = lower
		if upper > lower {
			features.BB_Position = (features.Price - lower) / (upper - lower)
		}
	}

	if len(candles) >= 14 {
		features.RSI_14 = calculateRSI(candles, 14)
		features.Williams_R = calculateWilliamsR(candles, 14)
		features.CCI = calculateCCI(candles, 14)
		features.ATR = calculateATR(candles, 14)
		features.ADX = calculateADX(candles, 14)
		k, _ := calculateStochastic(candles, 14)
		features.Stochastic_K = k
	}

	if len(candles) >= 10 {
		features.Momentum = calculateMomentum(candles, 10)
		features.ROC = calculateROC(candles, 10)
		features.VWAP = calculateVWAP(candles, 20)
		features.Volume_Delta = calculateVolumeDelta(candles, 10)
	}

	if len(candles) >= 50 {
		support, resistance := calculateSupportResistance(candles)
		features.Support_Level = support
		features.Resistance_Level = resistance
		features.Market_Regime = calculateMarketRegime(candles)
		features.Trend_Strength = calculateTrendStrength(candles, 20)
	}

	// Candlestick patterns
	if isDoji(current) {
		features.Doji_Pattern = 1.0
	}
	if isHammer(current) {
		features.Hammer_Pattern = 1.0
	}

	// ML-ready features
	if len(candles) >= 5 {
		price5ago := candles[len(candles)-6].Close
		features.Price_Momentum_5m = (features.Price - price5ago) / price5ago
	}
	features.Volume_Momentum = calculateVolumeMomentum(candles, 10)
	features.Liquidity_Score = calculateLiquidityScore(candles, 20)

	return features
}

// Helper functions for technical indicators (using functions from technical_indicators.go)

func calculateWilliamsR(candles []Candle, period int) float64 {
	if len(candles) < period {
		return 0
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
		return -50
	}

	return ((highest - current) / (highest - lowest)) * -100
}

func calculateCCI(candles []Candle, period int) float64 {
	if len(candles) < period {
		return 0
	}

	// Calculate typical prices
	typicalPrices := make([]float64, period)
	for i := 0; i < period; i++ {
		idx := len(candles) - period + i
		typicalPrices[i] = (candles[idx].High + candles[idx].Low + candles[idx].Close) / 3.0
	}

	// Calculate SMA of typical prices
	sma := 0.0
	for _, tp := range typicalPrices {
		sma += tp
	}
	sma /= float64(period)

	// Calculate mean deviation
	meanDev := 0.0
	for _, tp := range typicalPrices {
		meanDev += math.Abs(tp - sma)
	}
	meanDev /= float64(period)

	if meanDev == 0 {
		return 0
	}

	currentTP := typicalPrices[len(typicalPrices)-1]
	return (currentTP - sma) / (0.015 * meanDev)
}

func calculateADX(candles []Candle, period int) float64 {
	if len(candles) < period+1 {
		return 0
	}

	// Simplified ADX calculation
	plusDM := 0.0
	minusDM := 0.0
	trueRange := 0.0

	for i := len(candles) - period; i < len(candles); i++ {
		if i == 0 {
			continue
		}

		current := candles[i]
		previous := candles[i-1]

		highDiff := current.High - previous.High
		lowDiff := previous.Low - current.Low

		if highDiff > lowDiff && highDiff > 0 {
			plusDM += highDiff
		}
		if lowDiff > highDiff && lowDiff > 0 {
			minusDM += lowDiff
		}

		tr := math.Max(current.High-current.Low,
			math.Max(math.Abs(current.High-previous.Close),
				math.Abs(current.Low-previous.Close)))
		trueRange += tr
	}

	if trueRange == 0 {
		return 0
	}

	plusDI := (plusDM / trueRange) * 100
	minusDI := (minusDM / trueRange) * 100

	if plusDI+minusDI == 0 {
		return 0
	}

	return math.Abs(plusDI-minusDI) / (plusDI + minusDI) * 100
}

func calculateStochastic(candles []Candle, period int) (float64, float64) {
	if len(candles) < period {
		return 50, 50
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
		return 50, 50
	}

	k := ((current - lowest) / (highest - lowest)) * 100
	d := k * 0.8 // Simplified %D calculation

	return k, d
}

func calculateMomentum(candles []Candle, period int) float64 {
	if len(candles) < period+1 {
		return 0
	}

	current := candles[len(candles)-1].Close
	past := candles[len(candles)-1-period].Close

	return (current - past) / past
}

func calculateROC(candles []Candle, period int) float64 {
	if len(candles) < period+1 {
		return 0
	}

	current := candles[len(candles)-1].Close
	past := candles[len(candles)-1-period].Close

	return ((current - past) / past) * 100
}

func calculateVWAP(candles []Candle, period int) float64 {
	if len(candles) < period {
		return 0
	}

	totalVolumePrice := 0.0
	totalVolume := 0.0

	for i := len(candles) - period; i < len(candles); i++ {
		typical := (candles[i].High + candles[i].Low + candles[i].Close) / 3.0
		totalVolumePrice += typical * candles[i].Volume
		totalVolume += candles[i].Volume
	}

	if totalVolume == 0 {
		return 0
	}

	return totalVolumePrice / totalVolume
}

func calculateVolumeDelta(candles []Candle, period int) float64 {
	if len(candles) < period {
		return 0
	}

	// Simplified volume delta - compare recent to average
	recentVolume := 0.0
	totalVolume := 0.0

	for i := len(candles) - period; i < len(candles); i++ {
		totalVolume += candles[i].Volume
		if i >= len(candles)-period/2 {
			recentVolume += candles[i].Volume
		}
	}

	avgVolume := totalVolume / float64(period)
	recentAvg := recentVolume / float64(period/2)

	if avgVolume == 0 {
		return 0
	}

	return (recentAvg - avgVolume) / avgVolume
}

func calculateMarketRegime(candles []Candle) float64 {
	if len(candles) < 50 {
		return 0
	}

	// Simple trending vs ranging market detection
	recent := candles[len(candles)-20:]
	prices := make([]float64, len(recent))
	for i, candle := range recent {
		prices[i] = candle.Close
	}

	// Calculate linear regression slope
	n := float64(len(prices))
	sumX := 0.0
	sumY := 0.0
	sumXY := 0.0
	sumX2 := 0.0

	for i, price := range prices {
		x := float64(i)
		sumX += x
		sumY += price
		sumXY += x * price
		sumX2 += x * x
	}

	slope := (n*sumXY - sumX*sumY) / (n*sumX2 - sumX*sumX)

	// Normalize slope to 0-1 range
	return math.Tanh(math.Abs(slope) * 1000)
}

func calculateTrendStrength(candles []Candle, period int) float64 {
	if len(candles) < period {
		return 0
	}

	start := candles[len(candles)-period].Close
	end := candles[len(candles)-1].Close

	change := math.Abs(end-start) / start
	return math.Min(change*10, 1.0) // Normalize to 0-1
}

func isDoji(candle Candle) bool {
	bodySize := math.Abs(candle.Close - candle.Open)
	totalRange := candle.High - candle.Low

	if totalRange == 0 {
		return false
	}

	return bodySize/totalRange < 0.1 // Body is less than 10% of total range
}

func isHammer(candle Candle) bool {
	bodySize := math.Abs(candle.Close - candle.Open)
	lowerShadow := math.Min(candle.Open, candle.Close) - candle.Low
	upperShadow := candle.High - math.Max(candle.Open, candle.Close)

	return lowerShadow > bodySize*2 && upperShadow < bodySize*0.5
}

func calculateVolumeMomentum(candles []Candle, period int) float64 {
	if len(candles) < period*2 {
		return 0
	}

	recentVolume := 0.0
	pastVolume := 0.0

	for i := len(candles) - period; i < len(candles); i++ {
		recentVolume += candles[i].Volume
	}

	for i := len(candles) - period*2; i < len(candles)-period; i++ {
		pastVolume += candles[i].Volume
	}

	if pastVolume == 0 {
		return 0
	}

	return (recentVolume - pastVolume) / pastVolume
}

func calculateLiquidityScore(candles []Candle, period int) float64 {
	if len(candles) < period {
		return 0
	}

	totalVolume := 0.0
	avgSpread := 0.0

	for i := len(candles) - period; i < len(candles); i++ {
		totalVolume += candles[i].Volume
		spread := (candles[i].High - candles[i].Low) / candles[i].Close
		avgSpread += spread
	}

	avgVolume := totalVolume / float64(period)
	avgSpread /= float64(period)

	// Higher volume and lower spread = higher liquidity
	if avgSpread == 0 {
		return math.Log(1 + avgVolume)
	}

	return math.Log(1+avgVolume) / (1 + avgSpread*100)
}
