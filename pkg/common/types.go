package common

import (
	"time"
)

// Point represents a point in a chart
type Point struct {
	Timestamp int64
	Value     float64
}

// Distribution represents statistical distribution metrics
type Distribution struct {
	Mean   float64
	StdDev float64
	Min    float64
	Max    float64
	Median float64
}

// BacktestReport represents the complete backtesting report
type BacktestReport struct {
	Summary struct {
		TotalReturn  float64
		SharpeRatio  float64
		MaxDrawdown  float64
		WinRate      float64
		ProfitFactor float64
		TotalTrades  int
	}
	Charts struct {
		EquityCurve    []Point
		DrawdownChart  []Point
		MonthlyReturns map[string]float64
	}
	Analysis struct {
		TradeDuration   Distribution
		PnLDistribution Distribution
		TimeAnalysis    map[string]float64 // по часам, дням недели
	}
}

// RiskConfig represents risk management configuration
type RiskConfig struct {
	MaxDrawdown     float64
	MaxPositionSize float64
	StopLossPercent float64
	TakeProfitRatio float64
	DailyLossLimit  float64
}

// Position represents an open trading position
type Position struct {
	Symbol     string
	EntryPrice float64
	Quantity   float64
	Direction  string // "LONG" or "SHORT"
	EntryTime  time.Time
	StopLoss   float64
	TakeProfit float64
}

// Signal represents a trading signal
type Signal struct {
	Symbol      string
	Timestamp   time.Time
	Prediction  string // "STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL"
	Confidence  float64
	PriceTarget float64
	StopLoss    float64
	ModelUsed   string
}

// PerformanceMetrics represents trading performance metrics
type PerformanceMetrics struct {
	TotalReturn    float64
	SharpeRatio    float64
	MaxDrawdown    float64
	WinRate        float64
	ProfitFactor   float64
	TotalTrades    int
	AvgTradeReturn float64
}

// RiskMetrics represents risk-related metrics
type RiskMetrics struct {
	ValueAtRisk         float64
	ExpectedShortfall   float64
	StabilityScore      float64
	CorrelationExposure float64
}

// PredictionAccuracy represents prediction accuracy metrics
type PredictionAccuracy struct {
	OverallAccuracy float64
	SymbolAccuracy  map[string]float64
	BestTrade       float64
	WorstTrade      float64
}
