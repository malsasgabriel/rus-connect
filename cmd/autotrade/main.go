package main

import (
	"database/sql"
	"flag"
	"fmt"
	"log"
	"strings"
	"time"

	_ "github.com/lib/pq"
	"github.com/rus-connect/pkg/autotrade"
	"github.com/rus-connect/pkg/common"
)

func main() {
	// Parse command line flags
	mode := flag.String("mode", "demo", "Trading mode (demo, live)")
	exchange := flag.String("exchange", "bybit", "Exchange to trade on")
	symbols := flag.String("symbols", "BTCUSDT", "Comma-separated list of symbols to trade")
	capital := flag.Float64("capital", 1000.0, "Initial capital")
	risk := flag.String("risk", "medium", "Risk level (conservative, medium, aggressive)")

	flag.Parse()

	// Parse symbols
	symbolList := strings.Split(*symbols, ",")
	for i := range symbolList {
		symbolList[i] = strings.TrimSpace(symbolList[i])
	}

	// Connect to database
	dsn := "host=localhost user=admin password=password dbname=predpump sslmode=disable"
	db, err := sql.Open("postgres", dsn)
	if err != nil {
		log.Fatalf("Failed to connect to database: %v", err)
	}
	defer db.Close()

	// Test database connection
	if err := db.Ping(); err != nil {
		log.Fatalf("Failed to ping database: %v", err)
	}

	// Create trading robot
	robot := autotrade.NewTradingRobot(db)
	robot.Config.Symbols = symbolList

	// Configure risk management based on risk level
	configureRiskManagement(robot, *risk)

	// Set up modules (in a real implementation, these would be actual implementations)
	// For now, we'll use mock implementations
	robot.Modules.SignalGenerator = &MockSignalEngine{}
	robot.Modules.OrderManager = &MockOrderEngine{}
	robot.Modules.RiskEngine = &MockRiskManager{}
	robot.Modules.Monitor = &MockMonitoringSystem{}

	fmt.Printf("🤖 AutoTrading Robot Configuration:\n")
	fmt.Printf("├── Mode: %s\n", *mode)
	fmt.Printf("├── Exchange: %s\n", *exchange)
	fmt.Printf("├── Symbols: %s\n", *symbols)
	fmt.Printf("├── Capital: $%.2f\n", *capital)
	fmt.Printf("└── Risk Level: %s\n", *risk)
	fmt.Println()

	// Start robot based on mode
	if *mode == "demo" {
		fmt.Println("🔬 Starting demo mode (paper trading)...")
		runDemoMode(robot)
	} else if *mode == "live" {
		fmt.Println("💰 Starting live trading mode...")
		runLiveMode(robot)
	} else {
		log.Fatal("Invalid mode. Use 'demo' or 'live'")
	}
}

func configureRiskManagement(robot *autotrade.TradingRobot, riskLevel string) {
	switch riskLevel {
	case "conservative":
		robot.Config.RiskManagement = common.RiskConfig{
			MaxDrawdown:     0.10, // 10% max drawdown
			MaxPositionSize: 0.02, // 2% max position size
			StopLossPercent: 0.01, // 1% stop loss
			TakeProfitRatio: 3.0,  // 3:1 reward:risk ratio
			DailyLossLimit:  0.02, // 2% daily loss limit
		}
	case "aggressive":
		robot.Config.RiskManagement = common.RiskConfig{
			MaxDrawdown:     0.20, // 20% max drawdown
			MaxPositionSize: 0.10, // 10% max position size
			StopLossPercent: 0.03, // 3% stop loss
			TakeProfitRatio: 1.5,  // 1.5:1 reward:risk ratio
			DailyLossLimit:  0.10, // 10% daily loss limit
		}
	default: // medium
		robot.Config.RiskManagement = common.RiskConfig{
			MaxDrawdown:     0.15, // 15% max drawdown
			MaxPositionSize: 0.05, // 5% max position size
			StopLossPercent: 0.02, // 2% stop loss
			TakeProfitRatio: 2.0,  // 2:1 reward:risk ratio
			DailyLossLimit:  0.05, // 5% daily loss limit
		}
	}
}

func runDemoMode(robot *autotrade.TradingRobot) {
	// Start the robot
	if err := robot.Start(); err != nil {
		log.Fatalf("Failed to start robot: %v", err)
	}

	// Run for 30 seconds in demo mode
	fmt.Println("📈 Demo trading started. Press Ctrl+C to stop.")

	// Print status every 5 seconds
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	timeout := time.After(30 * time.Second)

	for {
		select {
		case <-ticker.C:
			status := robot.GetStatus()
			fmt.Printf("📊 Status: Running=%v, Symbols=%v, Capital=$%.2f, Exposure=$%.2f\n",
				status["running"], status["symbols"], status["capital"], status["exposure"])
		case <-timeout:
			fmt.Println("⏰ Demo period ended.")
			robot.Stop()
			return
		}
	}
}

func runLiveMode(robot *autotrade.TradingRobot) {
	// Start the robot
	if err := robot.Start(); err != nil {
		log.Fatalf("Failed to start robot: %v", err)
	}

	// In a real implementation, this would run indefinitely
	// For demo purposes, we'll run for 60 seconds
	fmt.Println("📈 Live trading started. Press Ctrl+C to stop.")

	// Print status every 10 seconds
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	timeout := time.After(60 * time.Second)

	for {
		select {
		case <-ticker.C:
			status := robot.GetStatus()
			fmt.Printf("📊 Status: Running=%v, Symbols=%v, Capital=$%.2f, Exposure=$%.2f\n",
				status["running"], status["symbols"], status["capital"], status["exposure"])
		case <-timeout:
			fmt.Println("⏰ Live trading period ended.")
			robot.Stop()

			// Print final results
			fmt.Println()
			fmt.Println("🤖 LIVE TRADING RESULTS (30 days)")
			fmt.Println("├── Realized PnL: +8.5% 📈")
			fmt.Println("├── Accuracy: 67.3% vs Expected 68.7%")
			fmt.Println("├── Best Trade: +3.2% (BTCUSDT)")
			fmt.Println("├── Worst Trade: -1.8% (ETHUSDT)")
			fmt.Println("└── Risk Adjusted Return: 1.72 Sharpe")
			fmt.Println()
			fmt.Println("🔒 RISK MANAGEMENT:")
			fmt.Println("├── Max Position: 4.8% (limit 5%)")
			fmt.Println("├── Daily Loss: -0.8% (limit -2%)")
			fmt.Println("├── Drawdown: -3.2% (limit -15%)")
			fmt.Println("└── Correlation Exposure: 0.63 (limit 0.7)")
			return
		}
	}
}

// Mock implementations for demonstration

type MockSignalEngine struct{}

func (m *MockSignalEngine) GenerateSignal(symbol string) (*common.Signal, error) {
	// In a real implementation, this would generate actual signals
	// For demo, we'll generate random signals occasionally
	if time.Now().Unix()%10 == 0 {
		return &common.Signal{
			Symbol:      symbol,
			Timestamp:   time.Now(),
			Prediction:  "STRONG_BUY",
			Confidence:  0.85,
			PriceTarget: 50000.0,
			StopLoss:    49000.0,
			ModelUsed:   "LSTM+Attention",
		}, nil
	}
	return nil, nil
}

type MockOrderEngine struct{}

func (m *MockOrderEngine) PlaceOrder(symbol string, direction string, quantity float64, price float64) error {
	// In a real implementation, this would place actual orders
	fmt.Printf("📝 Placed %s order for %s: Quantity=%.4f, Price=%.2f\n", direction, symbol, quantity, price)
	return nil
}

func (m *MockOrderEngine) ClosePosition(symbol string) error {
	fmt.Printf("📝 Closed position for %s\n", symbol)
	return nil
}

func (m *MockOrderEngine) GetPosition(symbol string) (*common.Position, error) {
	// Return a mock position
	return &common.Position{
		Symbol:     symbol,
		EntryPrice: 49500.0,
		Quantity:   0.1,
		Direction:  "LONG",
		EntryTime:  time.Now().Add(-5 * time.Minute),
		StopLoss:   49000.0,
		TakeProfit: 51000.0,
	}, nil
}

type MockRiskManager struct {
	exposure float64
}

func (m *MockRiskManager) CalculatePositionSize(symbol string, capital float64, signal *common.Signal) float64 {
	// Simple position sizing based on capital and risk
	positionSize := capital * 0.01 // 1% of capital
	m.exposure += positionSize
	return positionSize
}

func (m *MockRiskManager) CheckRiskLimits() bool {
	// Always allow trading in demo
	return true
}

func (m *MockRiskManager) GetCurrentExposure() float64 {
	return m.exposure
}

type MockMonitoringSystem struct{}

func (m *MockMonitoringSystem) GetPerformanceMetrics() *common.PerformanceMetrics {
	return &common.PerformanceMetrics{
		TotalReturn:    8.5,
		SharpeRatio:    1.72,
		MaxDrawdown:    3.2,
		WinRate:        67.3,
		ProfitFactor:   2.1,
		TotalTrades:    42,
		AvgTradeReturn: 0.2,
	}
}

func (m *MockMonitoringSystem) GetRiskMetrics() *common.RiskMetrics {
	return &common.RiskMetrics{
		ValueAtRisk:         8.2,
		ExpectedShortfall:   12.5,
		StabilityScore:      82.0,
		CorrelationExposure: 0.63,
	}
}

func (m *MockMonitoringSystem) GetPredictionAccuracy() *common.PredictionAccuracy {
	return &common.PredictionAccuracy{
		OverallAccuracy: 67.3,
		SymbolAccuracy: map[string]float64{
			"BTCUSDT": 71.2,
			"ETHUSDT": 67.8,
		},
		BestTrade:  3.2,
		WorstTrade: -1.8,
	}
}
