package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"time"
)

// ModelPerformanceMonitor tracks and manages model performance across all trading pairs
type ModelPerformanceMonitor struct {
	selfLearningEngine *SelfLearningEngine
	metricsHistory     map[string][]ModelMetricsSnapshot
	alertThresholds    *AlertThresholds
	lastModelUpdate    map[string]time.Time
	performanceTarget  *PerformanceTarget
	retrainingSchedule map[string]time.Time
}

// ModelMetricsSnapshot represents a point-in-time performance snapshot
type ModelMetricsSnapshot struct {
	Timestamp          time.Time          `json:"timestamp"`
	Symbol             string             `json:"symbol"`
	Accuracy           float64            `json:"accuracy"`
	Precision          float64            `json:"precision"`
	Recall             float64            `json:"recall"`
	F1Score            float64            `json:"f1_score"`
	ProfitFactor       float64            `json:"profit_factor"`
	SharpeRatio        float64            `json:"sharpe_ratio"`
	MaxDrawdown        float64            `json:"max_drawdown"`
	TotalPredictions   int                `json:"total_predictions"`
	CorrectPredictions int                `json:"correct_predictions"`
	AvgConfidence      float64            `json:"avg_confidence"`
	VolatilityAdjusted float64            `json:"volatility_adjusted_return"`
	ModelVersion       string             `json:"model_version"`
	FeatureImportance  map[string]float64 `json:"feature_importance"`
	TrainingDataSize   int                `json:"training_data_size"`
}

// AlertThresholds defines when to trigger alerts and retraining
type AlertThresholds struct {
	MinAccuracy    float64 `json:"min_accuracy"`     // 0.65 = 65%
	MaxDrawdown    float64 `json:"max_drawdown"`     // 0.15 = 15%
	MinSharpeRatio float64 `json:"min_sharpe_ratio"` // 1.0
	AccuracyDrop   float64 `json:"accuracy_drop"`    // 0.1 = 10% drop
	ConfidenceDrop float64 `json:"confidence_drop"`  // 0.15 = 15% drop
	StaleDataHours int     `json:"stale_data_hours"` // 24 hours
	MinPredictions int     `json:"min_predictions"`  // 10 predictions for valid metrics
}

// PerformanceTarget defines target performance metrics
type PerformanceTarget struct {
	TargetAccuracy     float64 `json:"target_accuracy"`      // 0.70 = 70%
	TargetSharpeRatio  float64 `json:"target_sharpe_ratio"`  // 1.5
	TargetProfitFactor float64 `json:"target_profit_factor"` // 1.3
	MaxDrawdownLimit   float64 `json:"max_drawdown_limit"`   // 0.10 = 10%
	MinConfidence      float64 `json:"min_confidence"`       // 0.60 = 60%
}

// ModelAlert represents a performance alert
type ModelAlert struct {
	Timestamp      time.Time `json:"timestamp"`
	Symbol         string    `json:"symbol"`
	AlertType      string    `json:"alert_type"` // "ACCURACY_DROP", "HIGH_DRAWDOWN", "STALE_DATA"
	Severity       string    `json:"severity"`   // "LOW", "MEDIUM", "HIGH", "CRITICAL"
	Message        string    `json:"message"`
	CurrentValue   float64   `json:"current_value"`
	ThresholdValue float64   `json:"threshold_value"`
	Recommendation string    `json:"recommendation"`
	AutoAction     string    `json:"auto_action"` // "RETRAIN", "ADJUST_WEIGHTS", "PAUSE_MODEL"
}

// NewModelPerformanceMonitor creates a new performance monitor
func NewModelPerformanceMonitor(selfLearningEngine *SelfLearningEngine) *ModelPerformanceMonitor {
	return &ModelPerformanceMonitor{
		selfLearningEngine: selfLearningEngine,
		metricsHistory:     make(map[string][]ModelMetricsSnapshot),
		lastModelUpdate:    make(map[string]time.Time),
		retrainingSchedule: make(map[string]time.Time),

		alertThresholds: &AlertThresholds{
			MinAccuracy:    0.65,
			MaxDrawdown:    0.15,
			MinSharpeRatio: 1.0,
			AccuracyDrop:   0.1,
			ConfidenceDrop: 0.15,
			StaleDataHours: 24,
			MinPredictions: 10,
		},

		performanceTarget: &PerformanceTarget{
			TargetAccuracy:     0.70,
			TargetSharpeRatio:  1.5,
			TargetProfitFactor: 1.3,
			MaxDrawdownLimit:   0.10,
			MinConfidence:      0.60,
		},
	}
}

// MonitorPerformance continuously monitors model performance
func (mpm *ModelPerformanceMonitor) MonitorPerformance() {
	log.Println("🔍 Starting Model Performance Monitor...")

	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			mpm.collectMetrics()
			mpm.analyzePerformance()
			mpm.checkRetrainingSchedule()

		case <-time.After(6 * time.Minute):
			continue
		}
	}
}

// collectMetrics collects current performance metrics for all models
func (mpm *ModelPerformanceMonitor) collectMetrics() {
	for symbol, model := range mpm.selfLearningEngine.Models {
		snapshot := mpm.createPerformanceSnapshot(symbol, model)
		mpm.storeMetricsSnapshot(snapshot)

		log.Printf("📊 Performance snapshot for %s: Accuracy=%.2f%%, Confidence=%.2f%%, Predictions=%d",
			symbol, snapshot.Accuracy*100, snapshot.AvgConfidence*100, snapshot.TotalPredictions)
	}
}

// createPerformanceSnapshot creates a performance snapshot for a model
func (mpm *ModelPerformanceMonitor) createPerformanceSnapshot(symbol string, model *MLTradingModel) ModelMetricsSnapshot {
	// Get prediction history for analysis
	predictions := mpm.selfLearningEngine.PredictionLog[symbol]

	// Calculate current metrics
	totalPredictions := len(predictions)
	correctPredictions := 0
	totalConfidence := 0.0
	profits := make([]float64, 0)

	for _, pred := range predictions {
		if pred.IsCorrect {
			correctPredictions++
		}
		totalConfidence += pred.Confidence
		profits = append(profits, pred.ProfitLoss)
	}

	accuracy := 0.0
	avgConfidence := 0.0
	if totalPredictions > 0 {
		accuracy = float64(correctPredictions) / float64(totalPredictions)
		avgConfidence = totalConfidence / float64(totalPredictions)
	}

	// Calculate financial metrics
	profitFactor := mpm.calculateProfitFactor(profits)
	sharpeRatio := mpm.calculateSharpeRatio(profits)
	maxDrawdown := mpm.calculateMaxDrawdown(profits)
	volatilityAdjusted := mpm.calculateVolatilityAdjustedReturn(profits)

	// Calculate feature importance (simplified)
	featureImportance := make(map[string]float64)
	featureImportance["RSI"] = 0.15
	featureImportance["MACD"] = 0.12
	featureImportance["BollingerBands"] = 0.10
	featureImportance["Volume"] = 0.08
	featureImportance["PriceAction"] = 0.20
	featureImportance["TechnicalIndicators"] = 0.25
	featureImportance["MarketRegime"] = 0.10

	return ModelMetricsSnapshot{
		Timestamp:          time.Now(),
		Symbol:             symbol,
		Accuracy:           accuracy,
		Precision:          accuracy, // Simplified
		Recall:             accuracy, // Simplified
		F1Score:            accuracy, // Simplified
		ProfitFactor:       profitFactor,
		SharpeRatio:        sharpeRatio,
		MaxDrawdown:        maxDrawdown,
		TotalPredictions:   totalPredictions,
		CorrectPredictions: correctPredictions,
		AvgConfidence:      avgConfidence,
		VolatilityAdjusted: volatilityAdjusted,
		ModelVersion:       "LSTM+Attention_v1.0",
		FeatureImportance:  featureImportance,
		TrainingDataSize:   len(model.TrainingBuffer),
	}
}

// storeMetricsSnapshot stores a metrics snapshot in history
func (mpm *ModelPerformanceMonitor) storeMetricsSnapshot(snapshot ModelMetricsSnapshot) {
	symbol := snapshot.Symbol

	if mpm.metricsHistory[symbol] == nil {
		mpm.metricsHistory[symbol] = make([]ModelMetricsSnapshot, 0)
	}

	mpm.metricsHistory[symbol] = append(mpm.metricsHistory[symbol], snapshot)

	// Keep only last 100 snapshots
	if len(mpm.metricsHistory[symbol]) > 100 {
		mpm.metricsHistory[symbol] = mpm.metricsHistory[symbol][1:]
	}
}

// analyzePerformance analyzes performance and generates alerts
func (mpm *ModelPerformanceMonitor) analyzePerformance() {
	for symbol, history := range mpm.metricsHistory {
		if len(history) < 2 {
			continue
		}

		current := history[len(history)-1]
		previous := history[len(history)-2]

		// Check for various alert conditions
		alerts := mpm.checkAlertConditions(symbol, current, previous)

		for _, alert := range alerts {
			mpm.handleAlert(alert)
		}
	}
}

// checkAlertConditions checks for various alert conditions
func (mpm *ModelPerformanceMonitor) checkAlertConditions(symbol string, current, previous ModelMetricsSnapshot) []ModelAlert {
	alerts := make([]ModelAlert, 0)

	// Check accuracy drop
	if current.TotalPredictions >= mpm.alertThresholds.MinPredictions {
		if current.Accuracy < mpm.alertThresholds.MinAccuracy {
			alerts = append(alerts, ModelAlert{
				Timestamp:      time.Now(),
				Symbol:         symbol,
				AlertType:      "LOW_ACCURACY",
				Severity:       "HIGH",
				Message:        fmt.Sprintf("Model accuracy %.2f%% below threshold %.2f%%", current.Accuracy*100, mpm.alertThresholds.MinAccuracy*100),
				CurrentValue:   current.Accuracy,
				ThresholdValue: mpm.alertThresholds.MinAccuracy,
				Recommendation: "Consider retraining model with fresh data",
				AutoAction:     "RETRAIN",
			})
		}

		// Check accuracy drop compared to previous
		if previous.TotalPredictions >= mpm.alertThresholds.MinPredictions {
			accuracyDrop := previous.Accuracy - current.Accuracy
			if accuracyDrop > mpm.alertThresholds.AccuracyDrop {
				alerts = append(alerts, ModelAlert{
					Timestamp:      time.Now(),
					Symbol:         symbol,
					AlertType:      "ACCURACY_DROP",
					Severity:       "MEDIUM",
					Message:        fmt.Sprintf("Model accuracy dropped by %.2f%% from %.2f%% to %.2f%%", accuracyDrop*100, previous.Accuracy*100, current.Accuracy*100),
					CurrentValue:   accuracyDrop,
					ThresholdValue: mpm.alertThresholds.AccuracyDrop,
					Recommendation: "Monitor closely, retrain if continues",
					AutoAction:     "ADJUST_WEIGHTS",
				})
			}
		}
	}

	// Check drawdown
	if current.MaxDrawdown > mpm.alertThresholds.MaxDrawdown {
		alerts = append(alerts, ModelAlert{
			Timestamp:      time.Now(),
			Symbol:         symbol,
			AlertType:      "HIGH_DRAWDOWN",
			Severity:       "CRITICAL",
			Message:        fmt.Sprintf("Maximum drawdown %.2f%% exceeds threshold %.2f%%", current.MaxDrawdown*100, mpm.alertThresholds.MaxDrawdown*100),
			CurrentValue:   current.MaxDrawdown,
			ThresholdValue: mpm.alertThresholds.MaxDrawdown,
			Recommendation: "Immediate model review and risk reduction",
			AutoAction:     "PAUSE_MODEL",
		})
	}

	// Check Sharpe ratio
	if current.SharpeRatio < mpm.alertThresholds.MinSharpeRatio && current.TotalPredictions >= mpm.alertThresholds.MinPredictions {
		alerts = append(alerts, ModelAlert{
			Timestamp:      time.Now(),
			Symbol:         symbol,
			AlertType:      "LOW_SHARPE_RATIO",
			Severity:       "MEDIUM",
			Message:        fmt.Sprintf("Sharpe ratio %.2f below threshold %.2f", current.SharpeRatio, mpm.alertThresholds.MinSharpeRatio),
			CurrentValue:   current.SharpeRatio,
			ThresholdValue: mpm.alertThresholds.MinSharpeRatio,
			Recommendation: "Review risk-adjusted returns",
			AutoAction:     "ADJUST_WEIGHTS",
		})
	}

	// Check stale data
	if time.Since(current.Timestamp) > time.Duration(mpm.alertThresholds.StaleDataHours)*time.Hour {
		alerts = append(alerts, ModelAlert{
			Timestamp:      time.Now(),
			Symbol:         symbol,
			AlertType:      "STALE_DATA",
			Severity:       "HIGH",
			Message:        fmt.Sprintf("Model data is %v old, exceeds %d hour threshold", time.Since(current.Timestamp), mpm.alertThresholds.StaleDataHours),
			CurrentValue:   time.Since(current.Timestamp).Hours(),
			ThresholdValue: float64(mpm.alertThresholds.StaleDataHours),
			Recommendation: "Update model with fresh market data",
			AutoAction:     "RETRAIN",
		})
	}

	return alerts
}

// handleAlert processes and acts on performance alerts
func (mpm *ModelPerformanceMonitor) handleAlert(alert ModelAlert) {
	// Log the alert
	log.Printf("⚠️ PERFORMANCE ALERT [%s]: %s - %s", alert.Severity, alert.Symbol, alert.Message)

	// Take automatic action based on alert type and severity
	switch alert.AutoAction {
	case "RETRAIN":
		if alert.Severity == "HIGH" || alert.Severity == "CRITICAL" {
			mpm.scheduleRetraining(alert.Symbol, "immediate")
		} else {
			mpm.scheduleRetraining(alert.Symbol, "next_window")
		}

	case "ADJUST_WEIGHTS":
		mpm.adjustModelWeights(alert.Symbol, alert.AlertType)

	case "PAUSE_MODEL":
		if alert.Severity == "CRITICAL" {
			mpm.pauseModel(alert.Symbol, alert.Message)
		}
	}

	// Send alert to monitoring systems (placeholder)
	mpm.sendAlertNotification(alert)
}

// scheduleRetraining schedules model retraining
func (mpm *ModelPerformanceMonitor) scheduleRetraining(symbol string, priority string) {
	var scheduleTime time.Time

	switch priority {
	case "immediate":
		scheduleTime = time.Now().Add(5 * time.Minute)
	case "next_window":
		scheduleTime = time.Now().Add(30 * time.Minute)
	default:
		scheduleTime = time.Now().Add(2 * time.Hour)
	}

	mpm.retrainingSchedule[symbol] = scheduleTime
	log.Printf("📅 Scheduled retraining for %s at %s (priority: %s)", symbol, scheduleTime.Format("15:04:05"), priority)
}

// adjustModelWeights adjusts ensemble weights based on performance
func (mpm *ModelPerformanceMonitor) adjustModelWeights(symbol string, alertType string) {
	currentWeight := mpm.selfLearningEngine.EnsembleWeights[symbol]

	switch alertType {
	case "ACCURACY_DROP":
		// Reduce weight by 10%
		newWeight := currentWeight * 0.9
		mpm.selfLearningEngine.EnsembleWeights[symbol] = math.Max(0.1, newWeight)

	case "LOW_SHARPE_RATIO":
		// Reduce weight by 15%
		newWeight := currentWeight * 0.85
		mpm.selfLearningEngine.EnsembleWeights[symbol] = math.Max(0.1, newWeight)
	}

	log.Printf("⚖️ Adjusted ensemble weight for %s: %.3f -> %.3f",
		symbol, currentWeight, mpm.selfLearningEngine.EnsembleWeights[symbol])
}

// pauseModel temporarily pauses model predictions
func (mpm *ModelPerformanceMonitor) pauseModel(symbol string, reason string) {
	// Set weight to minimum
	mpm.selfLearningEngine.EnsembleWeights[symbol] = 0.1
	log.Printf("⏸️ Paused model for %s due to: %s", symbol, reason)
}

// checkRetrainingSchedule checks and executes scheduled retraining
func (mpm *ModelPerformanceMonitor) checkRetrainingSchedule() {
	for symbol, scheduleTime := range mpm.retrainingSchedule {
		if time.Now().After(scheduleTime) {
			mpm.executeRetraining(symbol)
			delete(mpm.retrainingSchedule, symbol)
		}
	}
}

// executeRetraining executes model retraining
func (mpm *ModelPerformanceMonitor) executeRetraining(symbol string) {
	log.Printf("🔄 Executing scheduled retraining for %s", symbol)

	// Get fresh training data
	model := mpm.selfLearningEngine.Models[symbol]
	if model == nil {
		log.Printf("❌ No model found for %s", symbol)
		return
	}

	// Trigger retraining through self-learning engine
	if len(model.TrainingBuffer) > 50 {
		trainingData := model.TrainingBuffer[len(model.TrainingBuffer)-50:] // Use recent data
		mpm.selfLearningEngine.OnlineTrain(symbol, trainingData)

		// Reset ensemble weight to full after retraining
		mpm.selfLearningEngine.EnsembleWeights[symbol] = 1.0
		mpm.lastModelUpdate[symbol] = time.Now()

		log.Printf("✅ Retraining completed for %s with %d samples", symbol, len(trainingData))
	} else {
		log.Printf("⚠️ Insufficient training data for %s (%d samples)", symbol, len(model.TrainingBuffer))
	}
}

// sendAlertNotification sends alerts to monitoring systems
func (mpm *ModelPerformanceMonitor) sendAlertNotification(alert ModelAlert) {
	// Convert alert to JSON for logging/sending
	alertJSON, _ := json.Marshal(alert)
	log.Printf("📤 Alert notification: %s", string(alertJSON))

	// In a production system, this would send to:
	// - Slack/Discord webhooks
	// - Email notifications
	// - PagerDuty/OpsGenie
	// - Database logging
	// - Grafana/monitoring dashboards
}

// Financial metrics calculation methods

func (mpm *ModelPerformanceMonitor) calculateProfitFactor(profits []float64) float64 {
	if len(profits) == 0 {
		return 0.0
	}

	totalWins := 0.0
	totalLosses := 0.0

	for _, profit := range profits {
		if profit > 0 {
			totalWins += profit
		} else {
			totalLosses += math.Abs(profit)
		}
	}

	if totalLosses == 0 {
		return 2.0 // Perfect performance
	}

	return totalWins / totalLosses
}

func (mpm *ModelPerformanceMonitor) calculateSharpeRatio(profits []float64) float64 {
	if len(profits) < 2 {
		return 0.0
	}

	// Calculate mean return
	mean := 0.0
	for _, profit := range profits {
		mean += profit
	}
	mean /= float64(len(profits))

	// Calculate standard deviation
	variance := 0.0
	for _, profit := range profits {
		variance += math.Pow(profit-mean, 2)
	}
	variance /= float64(len(profits))
	stdDev := math.Sqrt(variance)

	if stdDev == 0 {
		return 0.0
	}

	// Assume risk-free rate of 2% annually (simplified)
	riskFreeRate := 0.02 / 365.0 // Daily rate
	return (mean - riskFreeRate) / stdDev
}

func (mpm *ModelPerformanceMonitor) calculateMaxDrawdown(profits []float64) float64 {
	if len(profits) == 0 {
		return 0.0
	}

	// Calculate cumulative returns
	cumulative := make([]float64, len(profits))
	cumulative[0] = profits[0]

	for i := 1; i < len(profits); i++ {
		cumulative[i] = cumulative[i-1] + profits[i]
	}

	// Find maximum drawdown
	maxDrawdown := 0.0
	peak := cumulative[0]

	for _, value := range cumulative {
		if value > peak {
			peak = value
		}

		drawdown := (peak - value) / math.Max(math.Abs(peak), 1.0)
		if drawdown > maxDrawdown {
			maxDrawdown = drawdown
		}
	}

	return maxDrawdown
}

func (mpm *ModelPerformanceMonitor) calculateVolatilityAdjustedReturn(profits []float64) float64 {
	if len(profits) < 2 {
		return 0.0
	}

	mean := 0.0
	for _, profit := range profits {
		mean += profit
	}
	mean /= float64(len(profits))

	variance := 0.0
	for _, profit := range profits {
		variance += math.Pow(profit-mean, 2)
	}
	variance /= float64(len(profits))
	volatility := math.Sqrt(variance)

	if volatility == 0 {
		return mean
	}

	return mean / volatility
}

// GetPerformanceReport generates a comprehensive performance report
func (mpm *ModelPerformanceMonitor) GetPerformanceReport() map[string]interface{} {
	report := make(map[string]interface{})

	for symbol, history := range mpm.metricsHistory {
		if len(history) == 0 {
			continue
		}

		latest := history[len(history)-1]

		symbolReport := map[string]interface{}{
			"symbol":             symbol,
			"accuracy":           latest.Accuracy,
			"total_predictions":  latest.TotalPredictions,
			"avg_confidence":     latest.AvgConfidence,
			"profit_factor":      latest.ProfitFactor,
			"sharpe_ratio":       latest.SharpeRatio,
			"max_drawdown":       latest.MaxDrawdown,
			"model_version":      latest.ModelVersion,
			"last_update":        latest.Timestamp,
			"training_data_size": latest.TrainingDataSize,
			"feature_importance": latest.FeatureImportance,
		}

		report[symbol] = symbolReport
	}

	return report
}
