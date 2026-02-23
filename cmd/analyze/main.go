package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"
)

type BacktestResult struct {
	Summary struct {
		TotalReturn  float64 `json:"TotalReturn"`
		SharpeRatio  float64 `json:"SharpeRatio"`
		MaxDrawdown  float64 `json:"MaxDrawdown"`
		WinRate      float64 `json:"WinRate"`
		ProfitFactor float64 `json:"ProfitFactor"`
		TotalTrades  int     `json:"TotalTrades"`
	} `json:"Summary"`
	Charts struct {
		EquityCurve []struct {
			Timestamp int64   `json:"Timestamp"`
			Value     float64 `json:"Value"`
		} `json:"EquityCurve"`
	} `json:"Charts"`
}

func main() {
	// Parse command line flags
	backtestFile := flag.String("backtest-file", "", "Path to backtest results file (required)")
	generateCharts := flag.Bool("generate-charts", false, "Generate ASCII charts from backtest results")

	flag.Parse()

	if *backtestFile == "" {
		fmt.Println("❌ Error: --backtest-file is required")
		flag.Usage()
		os.Exit(1)
	}

	fmt.Println("🔍 Analyzing backtest results...")
	fmt.Printf("├── Backtest File: %s\n", *backtestFile)
	fmt.Println()

	// Load backtest results
	data, err := os.ReadFile(*backtestFile)
	if err != nil {
		fmt.Printf("❌ Error reading file: %v\n", err)
		os.Exit(1)
	}

	var result BacktestResult
	if err := json.Unmarshal(data, &result); err != nil {
		fmt.Printf("❌ Error parsing JSON: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("📈 BACKTEST ANALYSIS RESULTS")
	fmt.Println("├── Strategy Performance:")
	fmt.Printf("│   ├── Total Return: %.2f%%\n", result.Summary.TotalReturn*100)
	fmt.Printf("│   ├── Sharpe Ratio: %.2f\n", result.Summary.SharpeRatio)
	fmt.Printf("│   ├── Max Drawdown: %.2f%%\n", result.Summary.MaxDrawdown*100)
	fmt.Printf("│   ├── Win Rate: %.2f%%\n", result.Summary.WinRate*100)
	fmt.Printf("│   └── Profit Factor: %.2f\n", result.Summary.ProfitFactor)
	fmt.Printf("│   └── Total Trades: %d\n", result.Summary.TotalTrades)
	fmt.Println("└── Market Conditions:")
	fmt.Println("    └── (Market condition analysis requires full OHLCV data)")
	fmt.Println()

	if *generateCharts {
		generateEquityCurve(result.Charts.EquityCurve)
	}
}

func generateEquityCurve(curve []struct {
	Timestamp int64   `json:"Timestamp"`
	Value     float64 `json:"Value"`
}) {
	if len(curve) == 0 {
		fmt.Println("⚠️ No equity curve data available")
		return
	}

	fmt.Println("📊 EQUITY CURVE (ASCII Approximation)")
	
	// Simple sampling if too many points
	maxPoints := 40
	step := 1
	if len(curve) > maxPoints {
		step = len(curve) / maxPoints
	}

	minVal := curve[0].Value
	maxVal := curve[0].Value
	for _, p := range curve {
		if p.Value < minVal {
			minVal = p.Value
		}
		if p.Value > maxVal {
			maxVal = p.Value
		}
	}
	
	rangeVal := maxVal - minVal
	if rangeVal == 0 {
		rangeVal = 1
	}

	fmt.Printf("Max: %.2f\n", maxVal)
	for i := 0; i < len(curve); i += step {
		val := curve[i].Value
		normalized := int((val - minVal) / rangeVal * 20)
		
		bar := ""
		for j := 0; j < normalized; j++ {
			bar += "█"
		}
		
		ts := time.Unix(curve[i].Timestamp, 0).Format("01-02 15:04")
		fmt.Printf("%s | %s %.2f\n", ts, bar, val)
	}
	fmt.Printf("Min: %.2f\n", minVal)
}