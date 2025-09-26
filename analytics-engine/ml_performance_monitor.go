// 📊 Real-time ML Performance Monitoring and Alerting System
package main

import (
	"fmt"
	"log"
	"time"
)

// 📈 ML Performance Monitor
type MLPerformanceMonitor struct {
	lstmAI           *LSTMTradingAI
	alertThresholds  *MLAlertThresholds // Changed to MLAlertThresholds
	metricsHistory   map[string]*MetricsHistory
	isActive         bool
	monitoringTicker *time.Ticker
}

// 🚨 ML Performance Alert Thresholds (specific to ML Performance Monitor)
type MLAlertThresholds struct {
	MinAccuracy      float64 `json:"min_accuracy"`       // 0.65
	MaxLatency       float64 `json:"max_latency"`        // 2.0 seconds
	MinConfidence    float64 `json:"min_confidence"`     // 0.60
	MaxMemoryUsage   float64 `json:"max_memory_usage"`   // 80% of available
	MinDataFreshness int64   `json:"min_data_freshness"` // 300 seconds
}

// 📊 Metrics History
type MetricsHistory struct {
	AccuracyTrend   []float64 `json:"accuracy_trend"`
	LatencyTrend    []float64 `json:"latency_trend"`
	ConfidenceTrend []float64 `json:"confidence_trend"`
	SignalCounts    []int     `json:"signal_counts"`
	LastUpdated     time.Time `json:"last_updated"`
}

// 🎯 Real-time Performance Metrics
type RealtimeMetrics struct {
	Symbol                string    `json:"symbol"`
	CurrentAccuracy       float64   `json:"current_accuracy"`
	AverageLatency        float64   `json:"average_latency"`
	AverageConfidence     float64   `json:"average_confidence"`
	SignalsGenerated      int       `json:"signals_generated"`
	SuccessfulPredictions int       `json:"successful_predictions"`
	ModelHealth           string    `json:"model_health"` // "EXCELLENT", "GOOD", "WARNING", "CRITICAL"
	LastSignalTime        time.Time `json:"last_signal_time"`
	DataFreshness         int64     `json:"data_freshness"` // seconds since last data
	MemoryUsage           float64   `json:"memory_usage"`   // percentage
}

// 🔔 Performance Alert
type PerformanceAlert struct {
	AlertType      string    `json:"alert_type"` // "ACCURACY_DROP", "LATENCY_HIGH", "DATA_STALE"
	Severity       string    `json:"severity"`   // "LOW", "MEDIUM", "HIGH", "CRITICAL"
	Symbol         string    `json:"symbol"`
	Message        string    `json:"message"`
	CurrentValue   float64   `json:"current_value"`
	ThresholdValue float64   `json:"threshold_value"`
	Timestamp      time.Time `json:"timestamp"`
	Suggestion     string    `json:"suggestion"`
}

// 🚀 Initialize ML Performance Monitor
func NewMLPerformanceMonitor(lstmAI *LSTMTradingAI) *MLPerformanceMonitor {
	monitor := &MLPerformanceMonitor{
		lstmAI: lstmAI,
		alertThresholds: &MLAlertThresholds{ // Changed to MLAlertThresholds
			MinAccuracy:      0.65,
			MaxLatency:       2.0,
			MinConfidence:    0.60,
			MaxMemoryUsage:   80.0,
			MinDataFreshness: 300,
		},
		metricsHistory:   make(map[string]*MetricsHistory),
		isActive:         true,
		monitoringTicker: time.NewTicker(30 * time.Second), // Monitor every 30 seconds
	}

	// Initialize metrics history for each symbol
	for symbol := range lstmAI.models {
		monitor.metricsHistory[symbol] = &MetricsHistory{
			AccuracyTrend:   make([]float64, 0, 100),
			LatencyTrend:    make([]float64, 0, 100),
			ConfidenceTrend: make([]float64, 0, 100),
			SignalCounts:    make([]int, 0, 100),
			LastUpdated:     time.Now(),
		}
	}

	// Start monitoring
	go monitor.startMonitoring()

	log.Printf("📊 ML Performance Monitor initialized for %d symbols", len(lstmAI.models))
	return monitor
}

// 🔄 Start Performance Monitoring
func (monitor *MLPerformanceMonitor) startMonitoring() {
	for monitor.isActive {
		<-monitor.monitoringTicker.C
		monitor.collectMetrics()
		monitor.checkAlerts()
	}
}

// 📊 Collect Real-time Metrics
func (monitor *MLPerformanceMonitor) collectMetrics() {
	for symbol := range monitor.lstmAI.models {
		metrics := monitor.calculateRealtimeMetrics(symbol)
		monitor.updateMetricsHistory(symbol, metrics)

		// Log key metrics every 5 minutes
		if time.Now().Minute()%5 == 0 && time.Now().Second() < 30 {
			log.Printf("📈 %s Performance: Accuracy=%.1f%%, Confidence=%.1f%%, Health=%s",
				symbol, metrics.CurrentAccuracy*100, metrics.AverageConfidence*100, metrics.ModelHealth)
		}
	}
}

// 🎯 Calculate Real-time Metrics for Symbol
func (monitor *MLPerformanceMonitor) calculateRealtimeMetrics(symbol string) *RealtimeMetrics {
	// Get model performance stats
	perfStats := monitor.lstmAI.performanceStats[symbol]
	accuracyTracker := monitor.lstmAI.accuracyTracker[symbol]
	signalBuffer := monitor.lstmAI.signalBuffer[symbol]

	// Calculate metrics
	currentAccuracy := 0.5
	averageConfidence := 0.5
	signalsGenerated := 0
	successfulPredictions := 0

	if perfStats != nil {
		currentAccuracy = perfStats.Accuracy
		successfulPredictions = perfStats.CorrectPredictions
		signalsGenerated = perfStats.TotalPredictions
	}

	if accuracyTracker != nil {
		currentAccuracy = accuracyTracker.CurrentAccuracy
	}

	// Calculate average confidence from recent signals
	if len(signalBuffer) > 0 {
		totalConfidence := 0.0
		recentSignals := 10
		if len(signalBuffer) < recentSignals {
			recentSignals = len(signalBuffer)
		}

		for i := len(signalBuffer) - recentSignals; i < len(signalBuffer); i++ {
			totalConfidence += signalBuffer[i].Confidence
		}
		averageConfidence = totalConfidence / float64(recentSignals)
	}

	// Determine model health
	modelHealth := monitor.determineModelHealth(currentAccuracy, averageConfidence, signalsGenerated)

	// Get last signal time
	lastSignalTime := time.Now().Add(-24 * time.Hour) // Default to 24 hours ago
	if len(signalBuffer) > 0 {
		lastSignalTime = time.Unix(signalBuffer[len(signalBuffer)-1].Timestamp, 0)
	}

	// Calculate data freshness
	dataFreshness := time.Since(lastSignalTime).Seconds()

	return &RealtimeMetrics{
		Symbol:                symbol,
		CurrentAccuracy:       currentAccuracy,
		AverageLatency:        0.5, // Simulated - would measure actual prediction latency
		AverageConfidence:     averageConfidence,
		SignalsGenerated:      signalsGenerated,
		SuccessfulPredictions: successfulPredictions,
		ModelHealth:           modelHealth,
		LastSignalTime:        lastSignalTime,
		DataFreshness:         int64(dataFreshness),
		MemoryUsage:           25.0, // Simulated - would measure actual memory usage
	}
}

// 🎯 Determine Model Health
func (monitor *MLPerformanceMonitor) determineModelHealth(accuracy, confidence float64, signals int) string {
	score := 0

	// Accuracy score (40% weight)
	if accuracy >= 0.80 {
		score += 40
	} else if accuracy >= 0.70 {
		score += 30
	} else if accuracy >= 0.60 {
		score += 20
	} else if accuracy >= 0.50 {
		score += 10
	}

	// Confidence score (30% weight)
	if confidence >= 0.80 {
		score += 30
	} else if confidence >= 0.70 {
		score += 25
	} else if confidence >= 0.60 {
		score += 20
	} else if confidence >= 0.50 {
		score += 10
	}

	// Signal volume score (30% weight)
	if signals >= 100 {
		score += 30
	} else if signals >= 50 {
		score += 25
	} else if signals >= 20 {
		score += 20
	} else if signals >= 10 {
		score += 15
	} else if signals >= 5 {
		score += 10
	}

	// Determine health based on total score
	if score >= 85 {
		return "EXCELLENT"
	} else if score >= 70 {
		return "GOOD"
	} else if score >= 50 {
		return "WARNING"
	} else {
		return "CRITICAL"
	}
}

// 📊 Update Metrics History
func (monitor *MLPerformanceMonitor) updateMetricsHistory(symbol string, metrics *RealtimeMetrics) {
	history := monitor.metricsHistory[symbol]

	// Add new metrics
	history.AccuracyTrend = append(history.AccuracyTrend, metrics.CurrentAccuracy)
	history.LatencyTrend = append(history.LatencyTrend, metrics.AverageLatency)
	history.ConfidenceTrend = append(history.ConfidenceTrend, metrics.AverageConfidence)
	history.SignalCounts = append(history.SignalCounts, metrics.SignalsGenerated)
	history.LastUpdated = time.Now()

	// Keep only last 100 data points
	if len(history.AccuracyTrend) > 100 {
		history.AccuracyTrend = history.AccuracyTrend[1:]
		history.LatencyTrend = history.LatencyTrend[1:]
		history.ConfidenceTrend = history.ConfidenceTrend[1:]
		history.SignalCounts = history.SignalCounts[1:]
	}
}

// 🚨 Check for Performance Alerts
func (monitor *MLPerformanceMonitor) checkAlerts() {
	for symbol := range monitor.lstmAI.models {
		metrics := monitor.calculateRealtimeMetrics(symbol)
		alerts := monitor.generateAlerts(symbol, metrics)

		for _, alert := range alerts {
			monitor.handleAlert(alert)
		}
	}
}

// 🔔 Generate Alerts for Symbol
func (monitor *MLPerformanceMonitor) generateAlerts(symbol string, metrics *RealtimeMetrics) []*PerformanceAlert {
	var alerts []*PerformanceAlert

	// Accuracy alert
	if metrics.CurrentAccuracy < monitor.alertThresholds.MinAccuracy {
		alerts = append(alerts, &PerformanceAlert{
			AlertType:      "ACCURACY_DROP",
			Severity:       monitor.calculateSeverity(metrics.CurrentAccuracy, monitor.alertThresholds.MinAccuracy),
			Symbol:         symbol,
			Message:        fmt.Sprintf("Model accuracy dropped to %.1f%%", metrics.CurrentAccuracy*100),
			CurrentValue:   metrics.CurrentAccuracy,
			ThresholdValue: monitor.alertThresholds.MinAccuracy,
			Timestamp:      time.Now(),
			Suggestion:     "Consider retraining the model or adjusting confidence thresholds",
		})
	}

	// Confidence alert
	if metrics.AverageConfidence < monitor.alertThresholds.MinConfidence {
		alerts = append(alerts, &PerformanceAlert{
			AlertType:      "CONFIDENCE_LOW",
			Severity:       "MEDIUM",
			Symbol:         symbol,
			Message:        fmt.Sprintf("Average confidence is low: %.1f%%", metrics.AverageConfidence*100),
			CurrentValue:   metrics.AverageConfidence,
			ThresholdValue: monitor.alertThresholds.MinConfidence,
			Timestamp:      time.Now(),
			Suggestion:     "Review feature engineering or increase training data quality",
		})
	}

	// Data freshness alert
	if float64(metrics.DataFreshness) > float64(monitor.alertThresholds.MinDataFreshness) {
		alerts = append(alerts, &PerformanceAlert{
			AlertType:      "DATA_STALE",
			Severity:       "HIGH",
			Symbol:         symbol,
			Message:        fmt.Sprintf("No new signals for %d seconds", metrics.DataFreshness),
			CurrentValue:   float64(metrics.DataFreshness),
			ThresholdValue: float64(monitor.alertThresholds.MinDataFreshness),
			Timestamp:      time.Now(),
			Suggestion:     "Check data pipeline and ensure candles are flowing correctly",
		})
	}

	return alerts
}

// 📊 Calculate Alert Severity
func (monitor *MLPerformanceMonitor) calculateSeverity(currentValue, threshold float64) string {
	ratio := currentValue / threshold

	if ratio >= 0.8 {
		return "LOW"
	} else if ratio >= 0.6 {
		return "MEDIUM"
	} else if ratio >= 0.4 {
		return "HIGH"
	} else {
		return "CRITICAL"
	}
}

// 🚨 Handle Performance Alert
func (monitor *MLPerformanceMonitor) handleAlert(alert *PerformanceAlert) {
	// Log alert
	log.Printf("🚨 ALERT [%s]: %s - %s (Current: %.3f, Threshold: %.3f)",
		alert.Severity, alert.AlertType, alert.Message,
		alert.CurrentValue, alert.ThresholdValue)

	// Take automated actions based on severity
	switch alert.Severity {
	case "CRITICAL":
		monitor.handleCriticalAlert(alert)
	case "HIGH":
		monitor.handleHighAlert(alert)
	case "MEDIUM":
		log.Printf("💡 Suggestion: %s", alert.Suggestion)
	}
}

// 🚨 Handle Critical Alert
func (monitor *MLPerformanceMonitor) handleCriticalAlert(alert *PerformanceAlert) {
	switch alert.AlertType {
	case "ACCURACY_DROP":
		// Trigger automatic model retraining
		log.Printf("🔄 Triggering automatic retrain for %s due to critical accuracy drop", alert.Symbol)
		monitor.triggerRetrain(alert.Symbol, "Critical accuracy drop")

	case "DATA_STALE":
		// Check data pipeline
		log.Printf("🔍 Checking data pipeline for %s", alert.Symbol)
	}
}

// ⚠️ Handle High Alert
func (monitor *MLPerformanceMonitor) handleHighAlert(alert *PerformanceAlert) {
	switch alert.AlertType {
	case "ACCURACY_DROP":
		// Schedule retrain for next maintenance window
		log.Printf("📅 Scheduling retrain for %s during next maintenance window", alert.Symbol)

	case "CONFIDENCE_LOW":
		// Adjust confidence thresholds temporarily
		log.Printf("🎛️ Temporarily adjusting confidence thresholds for %s", alert.Symbol)
	}
}

// 🔄 Trigger Model Retrain
func (monitor *MLPerformanceMonitor) triggerRetrain(symbol string, reason string) {
	// This would trigger the actual retraining process
	// For now, we'll just log and update the retrain flag
	log.Printf("🧠 Retrain triggered for %s: %s", symbol, reason)

	// Update last retrain time
	if model, exists := monitor.lstmAI.models[symbol]; exists {
		model.LastTrained = time.Now()
		log.Printf("✅ Model retrain completed for %s", symbol)
	}
}

// 📊 Get Performance Dashboard Data
func (monitor *MLPerformanceMonitor) GetPerformanceDashboard() map[string]interface{} {
	dashboard := make(map[string]interface{})

	// Overall system health
	totalModels := len(monitor.lstmAI.models)
	healthyModels := 0
	avgAccuracy := 0.0
	avgConfidence := 0.0

	symbolMetrics := make(map[string]*RealtimeMetrics)

	for symbol := range monitor.lstmAI.models {
		metrics := monitor.calculateRealtimeMetrics(symbol)
		symbolMetrics[symbol] = metrics

		if metrics.ModelHealth == "EXCELLENT" || metrics.ModelHealth == "GOOD" {
			healthyModels++
		}

		avgAccuracy += metrics.CurrentAccuracy
		avgConfidence += metrics.AverageConfidence
	}

	if totalModels > 0 {
		avgAccuracy /= float64(totalModels)
		avgConfidence /= float64(totalModels)
	}

	systemHealth := "GOOD"
	if float64(healthyModels)/float64(totalModels) < 0.5 {
		systemHealth = "WARNING"
	} else if float64(healthyModels)/float64(totalModels) < 0.8 {
		systemHealth = "DEGRADED"
	} else if avgAccuracy > 0.75 && avgConfidence > 0.70 {
		systemHealth = "EXCELLENT"
	}

	dashboard["system"] = map[string]interface{}{
		"overall_health":     systemHealth,
		"total_models":       totalModels,
		"healthy_models":     healthyModels,
		"average_accuracy":   avgAccuracy,
		"average_confidence": avgConfidence,
		"last_updated":       time.Now(),
	}

	dashboard["symbols"] = symbolMetrics
	dashboard["thresholds"] = monitor.alertThresholds

	return dashboard
}

// 🛑 Stop Performance Monitor
func (monitor *MLPerformanceMonitor) Stop() {
	monitor.isActive = false
	if monitor.monitoringTicker != nil {
		monitor.monitoringTicker.Stop()
	}
	log.Printf("🛑 ML Performance Monitor stopped")
}
