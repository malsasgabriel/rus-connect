package autotrade

import (
	"database/sql"
	"log"
	"time"

	"github.com/rus-connect/pkg/common"
)

// TradingRobot represents the auto-trading robot
type TradingRobot struct {
	Config struct {
		Symbols        []string
		PositionSizing string // "kelly", "fixed", "volatility"
		RiskManagement common.RiskConfig
		TradingHours   []string // "24/7" or specific hours
	}
	Modules struct {
		SignalGenerator SignalEngine
		OrderManager    OrderEngine
		RiskEngine      RiskManager
		Monitor         MonitoringSystem
	}
	db      *sql.DB
	running bool
}

// SignalEngine interface for signal generation
type SignalEngine interface {
	GenerateSignal(symbol string) (*common.Signal, error)
}

// OrderEngine interface for order management
type OrderEngine interface {
	PlaceOrder(symbol string, direction string, quantity float64, price float64) error
	ClosePosition(symbol string) error
	GetPosition(symbol string) (*common.Position, error)
}

// RiskManager interface for risk management
type RiskManager interface {
	CalculatePositionSize(symbol string, capital float64, signal *common.Signal) float64
	CheckRiskLimits() bool
	GetCurrentExposure() float64
}

// MonitoringSystem interface for monitoring
type MonitoringSystem interface {
	GetPerformanceMetrics() *common.PerformanceMetrics
	GetRiskMetrics() *common.RiskMetrics
	GetPredictionAccuracy() *common.PredictionAccuracy
}

// NewTradingRobot creates a new trading robot
func NewTradingRobot(db *sql.DB) *TradingRobot {
	robot := &TradingRobot{
		db:      db,
		running: false,
	}

	// Initialize default config
	robot.Config.RiskManagement = common.RiskConfig{
		MaxDrawdown:     0.15, // 15% max drawdown
		MaxPositionSize: 0.05, // 5% max position size
		StopLossPercent: 0.02, // 2% stop loss
		TakeProfitRatio: 2.0,  // 2:1 reward:risk ratio
		DailyLossLimit:  0.05, // 5% daily loss limit
	}

	robot.Config.TradingHours = []string{"24/7"}

	return robot
}

// Start starts the trading robot
func (tr *TradingRobot) Start() error {
	if tr.running {
		return nil // Already running
	}

	tr.running = true
	log.Println("🤖 Trading robot started")

	// Start trading loop
	go tr.tradingLoop()

	return nil
}

// Stop stops the trading robot
func (tr *TradingRobot) Stop() error {
	tr.running = false
	log.Println("🤖 Trading robot stopped")
	return nil
}

// IsRunning checks if the robot is running
func (tr *TradingRobot) IsRunning() bool {
	return tr.running
}

// tradingLoop is the main trading loop
func (tr *TradingRobot) tradingLoop() {
	ticker := time.NewTicker(1 * time.Minute) // Check for signals every minute
	defer ticker.Stop()

	for tr.running {
		<-ticker.C
		tr.checkAndTrade()
	}
}

// checkAndTrade checks for signals and executes trades
func (tr *TradingRobot) checkAndTrade() {
	capital := tr.getCurrentCapital()

	for _, symbol := range tr.Config.Symbols {
		// Generate signal
		signal, err := tr.Modules.SignalGenerator.GenerateSignal(symbol)
		if err != nil {
			log.Printf("Error generating signal for %s: %v", symbol, err)
			continue
		}

		// Validate signal
		if !tr.isValidSignal(signal) {
			continue
		}

		// Check risk limits
		if !tr.Modules.RiskEngine.CheckRiskLimits() {
			log.Printf("Risk limits exceeded, skipping trade for %s", symbol)
			continue
		}

		// Calculate position size
		positionSize := tr.Modules.RiskEngine.CalculatePositionSize(symbol, capital, signal)
		if positionSize <= 0 {
			continue
		}

		// Determine direction
		direction := "LONG"
		if signal.Prediction == "STRONG_SELL" {
			direction = "SHORT"
		}

		// Place order
		err = tr.Modules.OrderManager.PlaceOrder(symbol, direction, positionSize, signal.PriceTarget)
		if err != nil {
			log.Printf("Error placing order for %s: %v", symbol, err)
			continue
		}

		log.Printf("🤖 Placed %s order for %s: Quantity=%.4f, Price=%.2f",
			direction, symbol, positionSize, signal.PriceTarget)
	}
}

// isValidSignal validates a trading signal
func (tr *TradingRobot) isValidSignal(signal *common.Signal) bool {
	// Check confidence level
	if signal.Confidence < 0.7 {
		return false
	}

	// Check if it's a strong signal
	if signal.Prediction != "STRONG_BUY" && signal.Prediction != "STRONG_SELL" {
		return false
	}

	// Check if signal is recent (within last 5 minutes)
	if time.Since(signal.Timestamp) > 5*time.Minute {
		return false
	}

	return true
}

// getCurrentCapital gets the current available capital
func (tr *TradingRobot) getCurrentCapital() float64 {
	// In a real implementation, this would query the exchange or database
	// For now, we'll return a fixed amount
	return 10000.0
}

// UpdateConfig updates the robot configuration
func (tr *TradingRobot) UpdateConfig(config common.RiskConfig) {
	tr.Config.RiskManagement = config
	log.Println("🤖 Robot configuration updated")
}

// GetStatus returns the current status of the robot
func (tr *TradingRobot) GetStatus() map[string]interface{} {
	return map[string]interface{}{
		"running":  tr.running,
		"symbols":  tr.Config.Symbols,
		"capital":  tr.getCurrentCapital(),
		"exposure": tr.Modules.RiskEngine.GetCurrentExposure(),
	}
}
