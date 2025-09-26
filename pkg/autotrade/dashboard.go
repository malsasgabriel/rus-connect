package autotrade

import (
	"fmt"
	"time"

	"github.com/rus-connect/pkg/common"
)

// Dashboard represents the real-time monitoring dashboard
type Dashboard struct {
	RealTime struct {
		OpenPositions []common.Position
		TodayPnL      float64
		Signals       []common.Signal
		MarketStatus  map[string]string
	}
	Analytics struct {
		Performance common.PerformanceMetrics
		Risk        common.RiskMetrics
		Predictions common.PredictionAccuracy
	}
}

// NewDashboard creates a new dashboard
func NewDashboard() *Dashboard {
	dashboard := &Dashboard{}

	// Initialize structures
	dashboard.RealTime.OpenPositions = []common.Position{}
	dashboard.RealTime.Signals = []common.Signal{}
	dashboard.RealTime.MarketStatus = make(map[string]string)

	return dashboard
}

// UpdateRealTimeData updates the real-time data section
func (d *Dashboard) UpdateRealTimeData(positions []common.Position, pnl float64, signals []common.Signal, marketStatus map[string]string) {
	d.RealTime.OpenPositions = positions
	d.RealTime.TodayPnL = pnl
	d.RealTime.Signals = signals
	d.RealTime.MarketStatus = marketStatus
}

// UpdateAnalytics updates the analytics section
func (d *Dashboard) UpdateAnalytics(performance common.PerformanceMetrics, risk common.RiskMetrics, predictions common.PredictionAccuracy) {
	d.Analytics.Performance = performance
	d.Analytics.Risk = risk
	d.Analytics.Predictions = predictions
}

// PrintDashboard prints the current dashboard status
func (d *Dashboard) PrintDashboard() {
	fmt.Println("📈 REAL-TIME TRADING DASHBOARD")
	fmt.Println("================================")

	// Real-time section
	fmt.Println("⏱️  REAL-TIME DATA")
	fmt.Printf("├── Today's PnL: +%.2f%% 📈\n", d.RealTime.TodayPnL)
	fmt.Printf("├── Open Positions: %d\n", len(d.RealTime.OpenPositions))
	fmt.Printf("└── New Signals: %d\n", len(d.RealTime.Signals))
	fmt.Println()

	// Open positions
	if len(d.RealTime.OpenPositions) > 0 {
		fmt.Println("📊 OPEN POSITIONS")
		for i, pos := range d.RealTime.OpenPositions {
			duration := time.Since(pos.EntryTime)
			prefix := "├──"
			if i == len(d.RealTime.OpenPositions)-1 {
				prefix = "└──"
			}
			fmt.Printf("%s %s %s: Qty=%.4f, Entry=$%.2f, SL=$%.2f, TP=$%.2f (%.0fmin)\n",
				prefix, pos.Direction, pos.Symbol, pos.Quantity, pos.EntryPrice, pos.StopLoss, pos.TakeProfit, duration.Minutes())
		}
		fmt.Println()
	}

	// Market status
	fmt.Println("🌍 MARKET STATUS")
	i := 0
	for symbol, status := range d.RealTime.MarketStatus {
		prefix := "├──"
		if i == len(d.RealTime.MarketStatus)-1 {
			prefix = "└──"
		}
		fmt.Printf("%s %s: %s\n", prefix, symbol, status)
		i++
	}
	fmt.Println()

	// Analytics section
	fmt.Println("📊 PERFORMANCE ANALYTICS")
	fmt.Printf("├── Total Return: +%.1f%% 📈\n", d.Analytics.Performance.TotalReturn)
	fmt.Printf("├── Sharpe Ratio: %.2f\n", d.Analytics.Performance.SharpeRatio)
	fmt.Printf("├── Max Drawdown: -%.1f%%\n", d.Analytics.Performance.MaxDrawdown)
	fmt.Printf("├── Win Rate: %.1f%% ✅\n", d.Analytics.Performance.WinRate)
	fmt.Printf("└── Profit Factor: %.2f\n", d.Analytics.Performance.ProfitFactor)
	fmt.Println()

	// Risk metrics
	fmt.Println("🛡️  RISK METRICS")
	fmt.Printf("├── VaR 95%%: -%.1f%%\n", d.Analytics.Risk.ValueAtRisk)
	fmt.Printf("├── Expected Shortfall: -%.1f%%\n", d.Analytics.Risk.ExpectedShortfall)
	fmt.Printf("├── Stability Score: %.0f/100\n", d.Analytics.Risk.StabilityScore)
	fmt.Printf("└── Correlation Exposure: %.2f\n", d.Analytics.Risk.CorrelationExposure)
	fmt.Println()

	// Prediction accuracy
	fmt.Println("🎯 PREDICTION ACCURACY")
	fmt.Printf("├── Overall Accuracy: %.1f%%\n", d.Analytics.Predictions.OverallAccuracy)
	fmt.Printf("├── Best Trade: +%.1f%%\n", d.Analytics.Predictions.BestTrade)
	fmt.Printf("└── Worst Trade: %.1f%%\n", d.Analytics.Predictions.WorstTrade)

	// Symbol accuracy
	if len(d.Analytics.Predictions.SymbolAccuracy) > 0 {
		fmt.Println("   Symbol Accuracy:")
		i := 0
		for symbol, accuracy := range d.Analytics.Predictions.SymbolAccuracy {
			prefix := "   ├──"
			if i == len(d.Analytics.Predictions.SymbolAccuracy)-1 {
				prefix = "   └──"
			}
			fmt.Printf("%s %s: %.1f%%\n", prefix, symbol, accuracy)
			i++
		}
	}
	fmt.Println()
}
