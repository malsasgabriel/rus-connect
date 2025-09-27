// 🔗 ML System Integration - Connects LSTM AI with Analytics Engine
package main

import (
	"fmt"
	"log"
	"time"
)

// 🧠 ML Analytics Integration Engine
type MLAnalyticsIntegration struct {
	lstmAI          *LSTMTradingAI
	analyticsEngine *AnalyticsEngine
	featureEngine   *EnhancedFeatureEngine
	signalProcessor *MLSignalProcessor
	isActive        bool
}

// 📊 ML Signal Processor for Real-time Processing
type MLSignalProcessor struct {
	mlSignals      chan MLSignal
	tradingSignals chan TradingSignal
	isProcessing   bool
}

// 🎯 ML Signal Structure
type MLSignal struct {
	Symbol      string    `json:"symbol"`
	Timestamp   int64     `json:"timestamp"`
	Prediction  string    `json:"prediction"`
	Confidence  float64   `json:"confidence"`
	PriceTarget float64   `json:"price_target"`
	Features    []float64 `json:"features"`
}

// 🚀 Initialize ML Integration
func NewMLAnalyticsIntegration(analyticsEngine *AnalyticsEngine) *MLAnalyticsIntegration {
	integration := &MLAnalyticsIntegration{
		lstmAI:          NewLSTMTradingAI(),
		analyticsEngine: analyticsEngine,
		featureEngine:   NewEnhancedFeatureEngine(),
		signalProcessor: &MLSignalProcessor{
			mlSignals:      make(chan MLSignal, 1000),
			tradingSignals: make(chan TradingSignal, 1000),
			isProcessing:   false,
		},
		isActive: true,
	}

	// Start ML signal processing
	go integration.startMLProcessing()

	log.Printf("🧠 ML Analytics Integration initialized and active")
	return integration
}

// 🔄 Start ML Processing Pipeline
func (mli *MLAnalyticsIntegration) startMLProcessing() {
	mli.signalProcessor.isProcessing = true

	for mli.isActive {
		select {
		case candle := <-mli.analyticsEngine.candleStream:
			// Process new candle through ML pipeline
			mli.processNewCandle(candle)

		case <-time.After(60 * time.Second):
			// Generate predictions every minute
			mli.generateMLPredictions()
		}
	}
}

// 📊 Process New Candle through ML Pipeline
func (mli *MLAnalyticsIntegration) processNewCandle(candle Candle) {
	// Add candle to feature engine
	features := mli.featureEngine.AddCandle(candle)

	// Check if we have enough data for ML prediction
	if mli.hasEnoughDataForPrediction(candle.Symbol) {
		// Generate ML prediction
		signal, err := mli.lstmAI.Predict1Hour(candle.Symbol)
		if err != nil {
			log.Printf("❌ ML prediction failed for %s: %v", candle.Symbol, err)
			return
		}

		// Send signal to processing queue
		mli.signalProcessor.tradingSignals <- *signal

		// Emit to analytics engine
		mli.emitMLSignal(candle.Symbol, signal, features)
	}
}

// 🎯 Generate ML Predictions for All Symbols
func (mli *MLAnalyticsIntegration) generateMLPredictions() {
	majorPairs := []string{"BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "STRKUSDT"}

	for _, symbol := range majorPairs {
		if mli.hasEnoughDataForPrediction(symbol) {
			signal, err := mli.lstmAI.Predict1Hour(symbol)
			if err != nil {
				continue
			}

			// Create ML signal for Kafka
			mlSignal := MLSignal{
				Symbol:      symbol,
				Timestamp:   time.Now().Unix(),
				Prediction:  signal.Prediction,
				Confidence:  signal.Confidence,
				PriceTarget: signal.PriceTarget,
				Features:    []float64{0.5}, // Default features since FeatureImportance doesn't exist
			}

			mli.signalProcessor.mlSignals <- mlSignal
		}
	}
}

// ✅ Check if enough data for prediction
func (mli *MLAnalyticsIntegration) hasEnoughDataForPrediction(symbol string) bool {
	candles := mli.featureEngine.history[symbol]
	return len(candles) >= 1440 // Need 24 hours of data
}

// 📡 Emit ML Signal to Analytics Engine
func (mli *MLAnalyticsIntegration) emitMLSignal(symbol string, signal *TradingSignal, features EnhancedFeatureSet) {
	// Convert to DirectionSignal format for compatibility
	directionSignal := DirectionSignal{
		Symbol:       symbol,
		Timestamp:    signal.Timestamp,
		Direction:    mli.convertPredictionToDirection(signal.Prediction),
		Confidence:   signal.Confidence,
		CurrentPrice: features.Price,
		PriceTarget:  signal.PriceTarget,
		StopLoss:     signal.StopLoss,
		TimeHorizon:  1,                  // Convert string to int (1 hour)
		Factors:      signal.KeyFeatures, // Use KeyFeatures field
		RiskLevel:    signal.RiskLevel,
		ModelType:    "LSTM-AI",
		Version:      signal.ModelUsed, // Use ModelUsed field
	}

	// Emit through analytics engine
	mli.analyticsEngine.emitDirectionSignal(directionSignal)

	log.Printf("🎯 ML Signal emitted: %s %s (%.1f%% confidence)",
		symbol, signal.Prediction, signal.Confidence*100)

	// also publish per-model analysis
	payload := map[string]interface{}{
		"symbol":       symbol,
		"model_name":   "LSTM",
		"prediction":   signal.Prediction,
		"confidence":   signal.Confidence,
		"price_target": signal.PriceTarget,
		"timestamp":    time.Now().UTC().Format(time.RFC3339),
	}
	go PublishModelAnalysisDBAndKafka(mli.analyticsEngine.db, mli.analyticsEngine.kafkaBrokers, payload)
}

// 🔄 Convert ML Prediction to Direction
func (mli *MLAnalyticsIntegration) convertPredictionToDirection(prediction string) string {
	switch prediction {
	case "STRONG_BUY":
		return "STRONG_UP"
	case "BUY":
		return "UP"
	case "STRONG_SELL":
		return "STRONG_DOWN"
	case "SELL":
		return "DOWN"
	default:
		return "SIDEWAYS"
	}
}

// 📊 Get ML Performance Metrics
func (mli *MLAnalyticsIntegration) GetMLPerformanceMetrics() map[string]interface{} {
	metrics := make(map[string]interface{})

	for symbol, stats := range mli.lstmAI.performanceStats {
		metrics[symbol] = map[string]interface{}{
			"accuracy":            stats.Accuracy,
			"total_predictions":   stats.TotalPredictions,
			"correct_predictions": stats.CorrectPredictions,
			"sharpe_ratio":        stats.SharpeRatio,
			"max_drawdown":        stats.MaxDrawdown,
			"last_updated":        stats.LastUpdated,
		}
	}

	return metrics
}

// 🎛️ Configure ML Parameters
func (mli *MLAnalyticsIntegration) ConfigureMLParameters(config map[string]interface{}) error {
	if minAccuracy, ok := config["min_accuracy"].(float64); ok {
		mli.lstmAI.config.MinAccuracy = minAccuracy
	}

	if confidenceThreshold, ok := config["confidence_threshold"].(float64); ok {
		mli.lstmAI.config.ConfidenceThreshold = confidenceThreshold
	}

	if retrainThreshold, ok := config["retrain_threshold"].(float64); ok {
		mli.lstmAI.config.RetrainThreshold = retrainThreshold
	}

	log.Printf("🎛️ ML parameters updated: accuracy=%.2f, confidence=%.2f, retrain=%.2f",
		mli.lstmAI.config.MinAccuracy,
		mli.lstmAI.config.ConfidenceThreshold,
		mli.lstmAI.config.RetrainThreshold)

	return nil
}

// 🔄 Online Learning Integration
func (mli *MLAnalyticsIntegration) OnlineLearn(feedback map[string]interface{}) error {
	symbol, ok := feedback["symbol"].(string)
	if !ok {
		return fmt.Errorf("invalid symbol in feedback")
	}

	actualDirection, ok := feedback["actual_direction"].(string)
	if !ok {
		return fmt.Errorf("invalid actual_direction in feedback")
	}

	// Update model performance based on feedback (simplified implementation)
	// TODO: Implement actual feedback processing in LSTMTradingAI
	log.Printf("📝 Processing feedback for %s: %s", symbol, actualDirection)
	return nil
}

// 🚀 Start ML System
func (mli *MLAnalyticsIntegration) Start() {
	mli.isActive = true
	log.Printf("🚀 ML Analytics Integration started")
}

// 🛑 Stop ML System
func (mli *MLAnalyticsIntegration) Stop() {
	mli.isActive = false
	mli.signalProcessor.isProcessing = false
	log.Printf("🛑 ML Analytics Integration stopped")
}
