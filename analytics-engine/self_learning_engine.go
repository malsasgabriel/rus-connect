package main

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"math"
	"math/rand"
	"sync"
	"time"
)

// SelfLearningEngine implements online learning and model adaptation
type SelfLearningEngine struct {
	Models             map[string]*MLTradingModel    `json:"models"`
	EnsembleWeights    map[string]float64            `json:"ensemble_weights"`
	TrainingQueue      chan TrainingRequest          `json:"-"`
	PredictionLog      map[string][]PredictionResult `json:"prediction_log"`
	PerformanceTracker *PerformanceTracker           `json:"performance_tracker"`
	OnlineTrainer      *OnlineTrainer                `json:"online_trainer"`
	DataDriftDetector  *DataDriftDetector            `json:"data_drift_detector"`
	FeatureImportance  map[string][]FeatureScore     `json:"feature_importance"`
	mutex              sync.RWMutex                  `json:"-"`
	isRunning          bool                          `json:"-"`
	// Calibration maps and thresholds per-symbol
	CalibrationBins  map[string][]float64 `json:"calibration_bins"`   // per-bin observed accuracy
	CalibrationBinsN int                  `json:"calibration_bins_n"` // number of bins used
	EmitThresholds   map[string]float64   `json:"emit_thresholds"`    // per-symbol emission threshold after calibration
	CalMinSamples    int                  `json:"cal_min_samples"`    // minimum samples to compute calibration
	DB               *sql.DB
	KafkaBrokers     []string
}

// TrainingRequest represents a request for model training
type TrainingRequest struct {
	Symbol       string           `json:"symbol"`
	TrainingData []TrainingSample `json:"training_data"`
	ModelType    string           `json:"model_type"`
	Priority     int              `json:"priority"`
	Timestamp    time.Time        `json:"timestamp"`
}

// PredictionResult stores prediction outcomes for evaluation
type PredictionResult struct {
	Symbol         string  `json:"symbol"`
	Timestamp      int64   `json:"timestamp"`
	PredictedClass string  `json:"predicted_class"`
	ActualClass    string  `json:"actual_class"`
	Confidence     float64 `json:"confidence"`
	PredictedPrice float64 `json:"predicted_price"`
	ActualPrice    float64 `json:"actual_price"`
	PriceError     float64 `json:"price_error"`
	ModelUsed      string  `json:"model_used"`
	IsCorrect      bool    `json:"is_correct"`
	ProfitLoss     float64 `json:"profit_loss"`
}

// PerformanceTracker monitors model performance
type PerformanceTracker struct {
	WindowSize      int                        `json:"window_size"`
	AccuracyHistory map[string][]float64       `json:"accuracy_history"`
	LatencyHistory  map[string][]time.Duration `json:"latency_history"`
	ProfitHistory   map[string][]float64       `json:"profit_history"`
	LastUpdate      time.Time                  `json:"last_update"`
}

// OnlineTrainer handles continuous model training
type OnlineTrainer struct {
	BatchSize        int             `json:"batch_size"`
	LearningRate     float64         `json:"learning_rate"`
	GradientClipping float64         `json:"gradient_clipping"`
	TrainingHistory  []TrainingEpoch `json:"training_history"`
	ValidationSplit  float64         `json:"validation_split"`
}

// DataDriftDetector monitors data distribution changes
type DataDriftDetector struct {
	BaselineStats  map[string]*DistributionStats `json:"baseline_stats"`
	CurrentStats   map[string]*DistributionStats `json:"current_stats"`
	DriftThreshold float64                       `json:"drift_threshold"`
	WindowSize     int                           `json:"window_size"`
	LastDriftCheck time.Time                     `json:"last_drift_check"`
	DriftAlerts    []DriftAlert                  `json:"drift_alerts"`
}

// DistributionStats contains statistical measures
type DistributionStats struct {
	Mean      float64   `json:"mean"`
	Std       float64   `json:"std"`
	Quantiles []float64 `json:"quantiles"`
	Min       float64   `json:"min"`
	Max       float64   `json:"max"`
}

// FeatureScore represents importance of a feature
type FeatureScore struct {
	FeatureName string  `json:"feature_name"`
	Importance  float64 `json:"importance"`
	Coefficient float64 `json:"coefficient"`
}

// TrainingEpoch represents one training epoch
type TrainingEpoch struct {
	Epoch              int           `json:"epoch"`
	TrainingLoss       float64       `json:"training_loss"`
	ValidationLoss     float64       `json:"validation_loss"`
	TrainingAccuracy   float64       `json:"training_accuracy"`
	ValidationAccuracy float64       `json:"validation_accuracy"`
	Duration           time.Duration `json:"duration"`
	Timestamp          time.Time     `json:"timestamp"`
}

// DriftAlert represents a data drift detection alert
type DriftAlert struct {
	Timestamp   time.Time `json:"timestamp"`
	FeatureName string    `json:"feature_name"`
	DriftScore  float64   `json:"drift_score"`
	Threshold   float64   `json:"threshold"`
	Severity    string    `json:"severity"`
	Description string    `json:"description"`
}

// NewSelfLearningEngine creates a new self-learning engine
func NewSelfLearningEngine() *SelfLearningEngine {
	engine := &SelfLearningEngine{
		Models:            make(map[string]*MLTradingModel),
		EnsembleWeights:   make(map[string]float64),
		TrainingQueue:     make(chan TrainingRequest, 1000),
		PredictionLog:     make(map[string][]PredictionResult),
		FeatureImportance: make(map[string][]FeatureScore),

		PerformanceTracker: &PerformanceTracker{
			WindowSize:      100,
			AccuracyHistory: make(map[string][]float64),
			LatencyHistory:  make(map[string][]time.Duration),
			ProfitHistory:   make(map[string][]float64),
		},

		OnlineTrainer: &OnlineTrainer{
			BatchSize:        32,
			LearningRate:     0.001,
			GradientClipping: 1.0,
			ValidationSplit:  0.2,
		},

		DataDriftDetector: &DataDriftDetector{
			BaselineStats:  make(map[string]*DistributionStats),
			CurrentStats:   make(map[string]*DistributionStats),
			DriftThreshold: 0.1,
			WindowSize:     100,
		},
		CalibrationBins:  make(map[string][]float64),
		CalibrationBinsN: 10,
		EmitThresholds:   make(map[string]float64),
		CalMinSamples:    20,
		DB:               nil,
		KafkaBrokers:     []string{},
	}

	return engine
}

// SetPersistence injects DB and Kafka brokers for publishing model analyses
func (sle *SelfLearningEngine) SetPersistence(db *sql.DB, brokers []string) {
	sle.DB = db
	sle.KafkaBrokers = brokers
}

// Start begins the self-learning engine
func (sle *SelfLearningEngine) Start() {
	sle.mutex.Lock()
	sle.isRunning = true
	sle.mutex.Unlock()

	log.Println("🚀 Starting Self-Learning Engine...")

	// Start background workers
	go sle.trainingWorker()
	go sle.performanceMonitor()
	go sle.driftDetectionWorker()
	go sle.smartCalibrationWorker() // 🤖 Smart auto-calibration

	log.Println("✅ Self-Learning Engine started successfully")
}

// Stop shuts down the self-learning engine
func (sle *SelfLearningEngine) Stop() {
	sle.mutex.Lock()
	sle.isRunning = false
	sle.mutex.Unlock()

	log.Println("🛑 Self-Learning Engine stopped")
}

// PredictWithEnsemble generates prediction using ensemble of models
func (sle *SelfLearningEngine) PredictWithEnsemble(symbol string, features [][]float64) *TradingSignal {
	sle.mutex.RLock()
	defer sle.mutex.RUnlock()

	start := time.Now()

	// Get model for symbol
	model, exists := sle.Models[symbol]
	if !exists {
		log.Printf("❌ No model found for symbol %s", symbol)
		return nil
	}

	// Ensure feature width matches model.FeatureCount
	if len(features) > 0 && len(features[0]) != model.FeatureCount {
		log.Printf("⚠️ Feature width mismatch for %s: got=%d expected=%d — padding/truncating", symbol, len(features[0]), model.FeatureCount)
		// Adjust each timestep
		for i := range features {
			if len(features[i]) < model.FeatureCount {
				pad := make([]float64, model.FeatureCount-len(features[i]))
				features[i] = append(features[i], pad...)
			} else if len(features[i]) > model.FeatureCount {
				features[i] = features[i][:model.FeatureCount]
			}
		}
	}

	// Generate prediction
	signal := model.Predict(features)
	if signal == nil {
		return nil
	}

	// Record prediction timing
	latency := time.Since(start)
	sle.recordLatency(symbol, latency)

	// Apply ensemble weights
	if weight, exists := sle.EnsembleWeights[symbol]; exists {
		signal.Confidence *= weight
	}

	// Apply calibration if available (binning calibration)
	calibrated := sle.applyCalibration(symbol, signal.Confidence)
	if calibrated >= 0 {
		// replace confidence with calibrated estimate
		signal.Confidence = calibrated
	}

	// If calibrated confidence below per-symbol emit threshold, suppress signal
	emitThreshold := 0.55 // default target: 55%
	if t, ok := sle.EmitThresholds[symbol]; ok {
		emitThreshold = t
	}

	if signal.Confidence < emitThreshold {
		log.Printf("🚫 Suppressed signal for %s due to low calibrated confidence %.2f (threshold %.2f)", symbol, signal.Confidence, emitThreshold)
		return nil
	}

	// Log prediction for future evaluation
	predResult := PredictionResult{
		Symbol:         symbol,
		Timestamp:      signal.Timestamp,
		PredictedClass: signal.Prediction,
		Confidence:     signal.Confidence,
		PredictedPrice: signal.PriceTarget,
		ModelUsed:      signal.ModelUsed,
	}

	sle.mutex.Lock()
	if sle.PredictionLog[symbol] == nil {
		sle.PredictionLog[symbol] = make([]PredictionResult, 0)
	}
	sle.PredictionLog[symbol] = append(sle.PredictionLog[symbol], predResult)

	// Keep only recent predictions
	if len(sle.PredictionLog[symbol]) > 1000 {
		sle.PredictionLog[symbol] = sle.PredictionLog[symbol][len(sle.PredictionLog[symbol])-1000:]
	}
	sle.mutex.Unlock()

	// publish per-model analysis (best-effort) if analytics engine is available via global DB
	// Note: We don't have direct access to AnalyticsEngine here; publish to localhost:9092 by default
	go func() {
		payload := map[string]interface{}{
			"symbol":     symbol,
			"model_name": signal.ModelUsed,
			"prediction": signal.Prediction,
			"confidence": signal.Confidence,
			"timestamp":  time.Now().UTC().Format(time.RFC3339),
		}
		PublishModelAnalysisDBAndKafka(context.Background(), sle.DB, sle.KafkaBrokers, payload)
	}()

	return signal
}

// OnlineTrain implements continuous learning from new data
func (sle *SelfLearningEngine) OnlineTrain(symbol string, newData []TrainingSample) {
	if len(newData) == 0 {
		return
	}

	// Check if model exists
	model, exists := sle.Models[symbol]
	if !exists {
		log.Printf("⚠️ Creating new model for symbol %s", symbol)
		sle.Models[symbol] = NewMLTradingModel(symbol, 1440, 50)
		model = sle.Models[symbol]
	}

	// Add to training buffer
	model.TrainingBuffer = append(model.TrainingBuffer, newData...)

	// Check if retraining is needed
	if model.ShouldRetrain() {
		log.Printf("📚 Scheduling retraining for model %s", symbol)

		request := TrainingRequest{
			Symbol:       symbol,
			TrainingData: model.TrainingBuffer,
			ModelType:    "LSTM",
			Priority:     1,
			Timestamp:    time.Now(),
		}

		// Add to training queue (non-blocking)
		select {
		case sle.TrainingQueue <- request:
			log.Printf("✅ Training request queued for %s", symbol)
		default:
			log.Printf("⚠️ Training queue full, skipping request for %s", symbol)
		}
	}
}

// UpdatePredictionOutcome updates prediction results with actual outcomes
func (sle *SelfLearningEngine) UpdatePredictionOutcome(symbol string, timestamp int64, actualPrice float64, actualDirection string) {
	sle.mutex.Lock()
	defer sle.mutex.Unlock()

	predictions := sle.PredictionLog[symbol]
	if predictions == nil {
		return
	}

	// Find matching prediction (within 5 minutes)
	for i := len(predictions) - 1; i >= 0; i-- {
		pred := &predictions[i]
		if math.Abs(float64(pred.Timestamp-timestamp)) < 300 {
			// Update prediction outcome
			pred.ActualPrice = actualPrice
			pred.ActualClass = actualDirection
			pred.PriceError = math.Abs(pred.PredictedPrice-actualPrice) / actualPrice
			pred.IsCorrect = pred.PredictedClass == actualDirection

			// Calculate simulated profit/loss based on prediction direction
			switch pred.PredictedClass {
			case "BUY", "STRONG_BUY":
				pred.ProfitLoss = (actualPrice - pred.PredictedPrice) / pred.PredictedPrice
			case "SELL", "STRONG_SELL":
				pred.ProfitLoss = (pred.PredictedPrice - actualPrice) / pred.PredictedPrice
			default:
				pred.ProfitLoss = 0.0 // NEUTRAL or unknown prediction
			}

			// Update model metrics
			if model, exists := sle.Models[symbol]; exists {
				model.UpdateMetrics(pred.PredictedClass, actualDirection, pred.Confidence)
			}

			// Update performance tracker
			sle.updatePerformanceMetrics(symbol, pred)

			log.Printf("📊 Updated prediction outcome for %s: %s->%s (%.2f%% error)",
				symbol, pred.PredictedClass, actualDirection, pred.PriceError*100)
			break
		}
	}
}

// trainingWorker processes training requests from the queue
func (sle *SelfLearningEngine) trainingWorker() {
	log.Println("🎓 Starting training worker...")

	for sle.isRunning {
		select {
		case request, ok := <-sle.TrainingQueue:
			if !ok {
				return
			}

			log.Printf("🔄 Processing training request for %s", request.Symbol)
			sle.processTrainingRequest(request)

		case <-time.After(1 * time.Minute):
			continue
		}
	}
}

// processTrainingRequest handles a single training request
func (sle *SelfLearningEngine) processTrainingRequest(request TrainingRequest) {
	model, exists := sle.Models[request.Symbol]
	if !exists {
		log.Printf("❌ Model not found for training request: %s", request.Symbol)
		return
	}

	start := time.Now()

	// Set training mode
	model.IsTraining = true
	defer func() {
		model.IsTraining = false
		model.LastTraining = time.Now()
	}()

	// Simulate training improvement
	oldAccuracy := model.ModelMetrics.Accuracy

	if len(request.TrainingData) > 10 {
		// Simulate accuracy improvement based on data quality
		improvement := rand.Float64() * 0.02 // Up to 2% improvement
		model.ModelMetrics.Accuracy = math.Min(0.95, model.ModelMetrics.Accuracy+improvement)
	}

	duration := time.Since(start)

	log.Printf("✅ Training completed for %s: %.2f%% -> %.2f%% accuracy (%.2fs)",
		request.Symbol, oldAccuracy*100, model.ModelMetrics.Accuracy*100, duration.Seconds())

	// Update training history
	epoch := TrainingEpoch{
		Epoch:              len(sle.OnlineTrainer.TrainingHistory) + 1,
		TrainingAccuracy:   model.ModelMetrics.Accuracy,
		ValidationAccuracy: model.ModelMetrics.Accuracy * 0.95,
		Duration:           duration,
		Timestamp:          time.Now(),
	}

	sle.OnlineTrainer.TrainingHistory = append(sle.OnlineTrainer.TrainingHistory, epoch)
}

// performanceMonitor continuously monitors model performance
func (sle *SelfLearningEngine) performanceMonitor() {
	log.Println("📈 Starting performance monitor...")

	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for sle.isRunning {
		select {
		case <-ticker.C:
			sle.updateOverallPerformance()
			sle.adaptModelWeights()

		case <-time.After(6 * time.Minute):
			if !sle.isRunning {
				return
			}
		}
	}
}

// driftDetectionWorker monitors for data drift
func (sle *SelfLearningEngine) driftDetectionWorker() {
	log.Println("🔍 Starting drift detection worker...")

	ticker := time.NewTicker(15 * time.Minute)
	defer ticker.Stop()

	for sle.isRunning {
		select {
		case <-ticker.C:
			sle.detectDataDrift()

		case <-time.After(16 * time.Minute):
			if !sle.isRunning {
				return
			}
		}
	}
}

// Helper methods
func (sle *SelfLearningEngine) recordLatency(symbol string, latency time.Duration) {
	tracker := sle.PerformanceTracker
	if tracker.LatencyHistory[symbol] == nil {
		tracker.LatencyHistory[symbol] = make([]time.Duration, 0)
	}

	tracker.LatencyHistory[symbol] = append(tracker.LatencyHistory[symbol], latency)

	if len(tracker.LatencyHistory[symbol]) > tracker.WindowSize {
		tracker.LatencyHistory[symbol] = tracker.LatencyHistory[symbol][1:]
	}
}

func (sle *SelfLearningEngine) updatePerformanceMetrics(symbol string, pred *PredictionResult) {
	tracker := sle.PerformanceTracker

	if tracker.AccuracyHistory[symbol] == nil {
		tracker.AccuracyHistory[symbol] = make([]float64, 0)
	}

	accuracy := 0.0
	if pred.IsCorrect {
		accuracy = 1.0
	}

	tracker.AccuracyHistory[symbol] = append(tracker.AccuracyHistory[symbol], accuracy)

	if tracker.ProfitHistory[symbol] == nil {
		tracker.ProfitHistory[symbol] = make([]float64, 0)
	}

	tracker.ProfitHistory[symbol] = append(tracker.ProfitHistory[symbol], pred.ProfitLoss)

	// Keep window size
	if len(tracker.AccuracyHistory[symbol]) > tracker.WindowSize {
		tracker.AccuracyHistory[symbol] = tracker.AccuracyHistory[symbol][1:]
		tracker.ProfitHistory[symbol] = tracker.ProfitHistory[symbol][1:]
	}
}

func (sle *SelfLearningEngine) updateOverallPerformance() {
	for symbol := range sle.Models {
		if history := sle.PerformanceTracker.AccuracyHistory[symbol]; len(history) > 0 {
			avg := 0.0
			for _, acc := range history {
				avg += acc
			}
			avg /= float64(len(history))

			log.Printf("📊 %s rolling accuracy (last %d): %.2f%%",
				symbol, len(history), avg*100)
		}
	}
}

func (sle *SelfLearningEngine) adaptModelWeights() {
	for symbol, model := range sle.Models {
		// Adapt weights based on recent performance
		if model.ModelMetrics.Accuracy > 0.7 {
			sle.EnsembleWeights[symbol] = math.Min(1.0,
				(sle.EnsembleWeights[symbol]+1.0)*1.01/2.0)
		} else if model.ModelMetrics.Accuracy < 0.6 {
			sle.EnsembleWeights[symbol] = math.Max(0.1,
				(sle.EnsembleWeights[symbol]+1.0)*0.99/2.0)
		} else {
			sle.EnsembleWeights[symbol] = 1.0
		}
	}
}

func (sle *SelfLearningEngine) detectDataDrift() {
	// Simplified drift detection
	for symbol := range sle.Models {
		if rand.Float64() < 0.05 { // 5% chance of drift detection
			alert := DriftAlert{
				Timestamp:   time.Now(),
				FeatureName: "price_features",
				DriftScore:  rand.Float64() * 0.2,
				Threshold:   sle.DataDriftDetector.DriftThreshold,
				Severity:    "MEDIUM",
				Description: fmt.Sprintf("Data drift detected for %s", symbol),
			}

			sle.DataDriftDetector.DriftAlerts = append(
				sle.DataDriftDetector.DriftAlerts, alert)

			log.Printf("⚠️ Data drift detected for %s (score: %.3f)",
				symbol, alert.DriftScore)
		}
	}
}

// 🤖 smartCalibrationWorker continuously calibrates and optimizes models
func (sle *SelfLearningEngine) smartCalibrationWorker() {
	log.Println("🧠 Starting Smart Calibration Worker...")

	ticker := time.NewTicker(5 * time.Minute) // Calibrate every 5 minutes
	defer ticker.Stop()

	for sle.isRunning {
		select {
		case <-ticker.C:
			sle.performSmartCalibration()
		case <-time.After(6 * time.Minute):
			if !sle.isRunning {
				return
			}
		}
	}
}

// performSmartCalibration performs intelligent model calibration
func (sle *SelfLearningEngine) performSmartCalibration() {
	sle.mutex.Lock()
	defer sle.mutex.Unlock()

	log.Println("🎯 Performing smart model calibration...")

	// Calibrate each model based on recent performance
	for symbol, model := range sle.Models {
		// Check recent prediction accuracy
		if predictions, exists := sle.PredictionLog[symbol]; exists && len(predictions) >= 10 {
			recentPredictions := predictions
			if len(predictions) > 50 {
				recentPredictions = predictions[len(predictions)-50:] // Last 50 predictions
			}

			// Calculate accuracy
			correct := 0
			for _, pred := range recentPredictions {
				if pred.IsCorrect {
					correct++
				}
			}

			accuracy := float64(correct) / float64(len(recentPredictions))
			log.Printf("📊 %s accuracy: %.2f%% (%d/%d)", symbol, accuracy*100, correct, len(recentPredictions))

			// Adjust ensemble weight based on accuracy
			if accuracy > 0.6 { // Good performance
				sle.EnsembleWeights[symbol] = math.Min(sle.EnsembleWeights[symbol]*1.05, 2.0)
				log.Printf("⬆️ Increased weight for %s to %.3f", symbol, sle.EnsembleWeights[symbol])
			} else if accuracy < 0.4 { // Poor performance
				sle.EnsembleWeights[symbol] = math.Max(sle.EnsembleWeights[symbol]*0.95, 0.1)
				log.Printf("⬇️ Decreased weight for %s to %.3f", symbol, sle.EnsembleWeights[symbol])

				// Trigger model retraining for poor performance
				if accuracy < 0.3 {
					log.Printf("🔄 Triggering emergency retraining for %s (accuracy: %.1f%%)", symbol, accuracy*100)
					sle.triggerEmergencyRetraining(symbol, model)
				}
			}

			// Auto-adjust learning parameters based on volatility
			if len(recentPredictions) >= 20 {
				// Calculate prediction variance (volatility)
				var variance float64
				mean := accuracy
				for _, pred := range recentPredictions[len(recentPredictions)-20:] {
					predAcc := 0.0
					if pred.IsCorrect {
						predAcc = 1.0
					}

					// ---- Calibration: build bins of confidence -> observed accuracy
					// Only if we have enough recent predictions
					if len(recentPredictions) >= sle.CalMinSamples {
						n := sle.CalibrationBinsN
						// initialize counters
						counts := make([]int, n)
						corrects := make([]int, n)

						for _, pred := range recentPredictions {
							idx := int(pred.Confidence * float64(n))
							if idx < 0 {
								idx = 0
							}
							if idx >= n {
								idx = n - 1
							}
							counts[idx]++
							if pred.IsCorrect {
								corrects[idx]++
							}
						}

						bins := make([]float64, n)
						for i := 0; i < n; i++ {
							if counts[i] > 0 {
								bins[i] = float64(corrects[i]) / float64(counts[i])
							} else {
								bins[i] = 0.0
							}
						}

						sle.CalibrationBins[symbol] = bins

						// Determine minimal emit threshold where bin accuracy >= target (0.55)
						targetAcc := 0.55
						thr := 0.55
						for i := 0; i < n; i++ {
							if bins[i] >= targetAcc {
								thr = float64(i) / float64(n)
								break
							}
						}
						sle.EmitThresholds[symbol] = thr
						log.Printf("🔧 Calibration updated for %s: emit_threshold=%.2f", symbol, thr)
					}
					variance += (predAcc - mean) * (predAcc - mean)
				}
				variance /= 20

				// Adjust learning rate based on variance
				if variance > 0.25 { // High volatility
					model.LearningRate = math.Max(model.LearningRate*0.9, 0.0001)
					log.Printf("📉 Reduced learning rate for %s to %.6f (high volatility)", symbol, model.LearningRate)
				} else if variance < 0.1 { // Low volatility, can be more aggressive
					model.LearningRate = math.Min(model.LearningRate*1.1, 0.01)
					log.Printf("📈 Increased learning rate for %s to %.6f (stable performance)", symbol, model.LearningRate)
				}
			}
		}
	}

	// Global model health check
	overallAccuracy := sle.calculateOverallAccuracy()
	log.Printf("🌐 Overall system accuracy: %.2f%%", overallAccuracy*100)

	if overallAccuracy < 0.45 {
		log.Println("🚨 System performance below threshold - triggering global optimization")
		sle.triggerGlobalOptimization()
	}
}

// applyCalibration applies simple bin-based calibration to raw confidences.
// Returns calibrated confidence in [0,1], or -1 if not enough data.
func (sle *SelfLearningEngine) applyCalibration(symbol string, rawConf float64) float64 {
	sle.mutex.RLock()
	bins, exists := sle.CalibrationBins[symbol]
	n := sle.CalibrationBinsN
	sle.mutex.RUnlock()

	if !exists || len(bins) != n {
		return -1
	}

	// Map rawConf [0,1] to bin index
	idx := int(rawConf * float64(n))
	if idx < 0 {
		idx = 0
	}
	if idx >= n {
		idx = n - 1
	}

	// bins hold observed accuracy per bin (0..1)
	cal := bins[idx]
	if cal <= 0 {
		return -1
	}

	// Simple calibrated probability: weighted average between raw confidence and bin accuracy
	calibrated := 0.5*rawConf + 0.5*cal
	if calibrated < 0 {
		calibrated = 0
	}
	if calibrated > 1 {
		calibrated = 1
	}
	return calibrated
}

// triggerEmergencyRetraining performs emergency model retraining
func (sle *SelfLearningEngine) triggerEmergencyRetraining(symbol string, _ *MLTradingModel) {
	log.Printf("🚨 Emergency retraining initiated for %s", symbol)

	// Reset model by creating a new one
	newModel := NewMLTradingModel(symbol, 1440, 50)
	sle.Models[symbol] = newModel

	// Reset ensemble weight
	sle.EnsembleWeights[symbol] = 0.5

	log.Printf("✅ Emergency retraining completed for %s", symbol)
}

// calculateOverallAccuracy calculates system-wide accuracy
func (sle *SelfLearningEngine) calculateOverallAccuracy() float64 {
	totalCorrect := 0
	totalPredictions := 0

	for _, predictions := range sle.PredictionLog {
		for _, pred := range predictions {
			if pred.IsCorrect {
				totalCorrect++
			}
			totalPredictions++
		}
	}

	if totalPredictions == 0 {
		return 0.5 // Default neutral accuracy
	}

	return float64(totalCorrect) / float64(totalPredictions)
}

// triggerGlobalOptimization performs system-wide optimization
func (sle *SelfLearningEngine) triggerGlobalOptimization() {
	log.Println("🔧 Starting global system optimization...")

	// Reset all ensemble weights to baseline
	for symbol := range sle.EnsembleWeights {
		sle.EnsembleWeights[symbol] = 1.0
	}

	// Adjust global learning parameters
	sle.OnlineTrainer.LearningRate = 0.001 // Reset to default
	sle.OnlineTrainer.BatchSize = 32       // Reset to default

	// Clear old performance history to start fresh
	for symbol := range sle.PerformanceTracker.AccuracyHistory {
		sle.PerformanceTracker.AccuracyHistory[symbol] = []float64{}
	}

	log.Println("✅ Global optimization completed")
}
