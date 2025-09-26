package main

import (
	"database/sql"
	"encoding/json"
	"log"
	"math"
	"time"
)

// OnlineLearner implements continuous learning for pump detection
type OnlineLearner struct {
	db             *sql.DB
	learningRate   float64
	adaptationRate float64
	batchSize      int
	minSamples     int
	modelWeights   map[string]float64
	featureStats   map[string]FeatureStats
	performance    PerformanceMetrics
}

// FeedbackData represents manual feedback or actual pump confirmations
type FeedbackData struct {
	Symbol        string     `json:"symbol"`
	Timestamp     time.Time  `json:"timestamp"`
	PredictedProb float64    `json:"predicted_prob"`
	ActualPump    bool       `json:"actual_pump"`
	FeedbackType  string     `json:"feedback_type"` // "manual", "auto", "market_confirmed"
	Confidence    float64    `json:"confidence"`
	Features      FeatureSet `json:"features"`
}

// FeatureStats tracks statistical properties of features for normalization
type FeatureStats struct {
	Mean       float64   `json:"mean"`
	StdDev     float64   `json:"std_dev"`
	Min        float64   `json:"min"`
	Max        float64   `json:"max"`
	Count      int64     `json:"count"`
	LastUpdate time.Time `json:"last_update"`
}

// PerformanceMetrics tracks model performance for adaptive learning
type PerformanceMetrics struct {
	Accuracy       float64   `json:"accuracy"`
	Precision      float64   `json:"precision"`
	Recall         float64   `json:"recall"`
	F1Score        float64   `json:"f1_score"`
	FalsePositives int       `json:"false_positives"`
	TruePositives  int       `json:"true_positives"`
	FalseNegatives int       `json:"false_negatives"`
	TrueNegatives  int       `json:"true_negatives"`
	LastUpdate     time.Time `json:"last_update"`
	SampleCount    int64     `json:"sample_count"`
}

// NewOnlineLearner creates a new online learning system
func NewOnlineLearner(db *sql.DB) *OnlineLearner {
	learner := &OnlineLearner{
		db:             db,
		learningRate:   0.01,  // Start conservative
		adaptationRate: 0.001, // How fast to adapt to changing markets
		batchSize:      50,    // Process feedback in batches
		minSamples:     100,   // Minimum samples before adapting
		modelWeights:   make(map[string]float64),
		featureStats:   make(map[string]FeatureStats),
		performance: PerformanceMetrics{
			LastUpdate: time.Now(),
		},
	}

	// Initialize database tables for online learning
	learner.initializeTables()

	// Load existing model state
	learner.loadModelState()

	return learner
}

// initializeTables creates necessary tables for continuous learning
func (ol *OnlineLearner) initializeTables() {
	// Feedback data table
	_, err := ol.db.Exec(`
		CREATE TABLE IF NOT EXISTS feedback_data (
			id BIGSERIAL PRIMARY KEY,
			symbol VARCHAR(20) NOT NULL,
			timestamp TIMESTAMPTZ NOT NULL,
			predicted_prob DECIMAL(5,4) NOT NULL,
			actual_pump BOOLEAN NOT NULL,
			feedback_type VARCHAR(20) NOT NULL,
			confidence DECIMAL(5,4) DEFAULT 1.0,
			features JSONB,
			created_at TIMESTAMPTZ DEFAULT NOW(),
			processed BOOLEAN DEFAULT FALSE
		)
	`)
	if err != nil {
		log.Printf("Failed to create feedback_data table: %v", err)
	}

	// Model performance tracking
	_, err = ol.db.Exec(`
		CREATE TABLE IF NOT EXISTS model_performance (
			id BIGSERIAL PRIMARY KEY,
			model_version INTEGER NOT NULL,
			accuracy DECIMAL(5,4),
			precision_score DECIMAL(5,4),
			recall_score DECIMAL(5,4),
			f1_score DECIMAL(5,4),
			false_positives INTEGER DEFAULT 0,
			true_positives INTEGER DEFAULT 0,
			false_negatives INTEGER DEFAULT 0,
			true_negatives INTEGER DEFAULT 0,
			sample_count INTEGER DEFAULT 0,
			created_at TIMESTAMPTZ DEFAULT NOW()
		)
	`)
	if err != nil {
		log.Printf("Failed to create model_performance table: %v", err)
	}

	// Feature statistics for normalization
	_, err = ol.db.Exec(`
		CREATE TABLE IF NOT EXISTS feature_stats (
			id BIGSERIAL PRIMARY KEY,
			feature_name VARCHAR(50) UNIQUE NOT NULL,
			mean_value DECIMAL(15,8),
			std_dev DECIMAL(15,8),
			min_value DECIMAL(15,8),
			max_value DECIMAL(15,8),
			sample_count BIGINT DEFAULT 0,
			last_updated TIMESTAMPTZ DEFAULT NOW()
		)
	`)
	if err != nil {
		log.Printf("Failed to create feature_stats table: %v", err)
	}

	// Model weights storage
	_, err = ol.db.Exec(`
		CREATE TABLE IF NOT EXISTS model_weights (
			id BIGSERIAL PRIMARY KEY,
			feature_name VARCHAR(50) UNIQUE NOT NULL,
			weight_value DECIMAL(15,8),
			last_updated TIMESTAMPTZ DEFAULT NOW()
		)
	`)
	if err != nil {
		log.Printf("Failed to create model_weights table: %v", err)
	}

	// Create indexes for performance
	ol.db.Exec(`CREATE INDEX IF NOT EXISTS idx_feedback_symbol_time ON feedback_data (symbol, timestamp DESC);`)
	ol.db.Exec(`CREATE INDEX IF NOT EXISTS idx_feedback_processed ON feedback_data (processed, created_at);`)
	ol.db.Exec(`CREATE INDEX IF NOT EXISTS idx_performance_version ON model_performance (model_version DESC);`)
}

// loadModelState loads existing model weights and statistics
func (ol *OnlineLearner) loadModelState() {
	// Load model weights
	rows, err := ol.db.Query("SELECT feature_name, weight_value FROM model_weights")
	if err != nil {
		log.Printf("Failed to load model weights: %v", err)
		return
	}
	defer rows.Close()

	for rows.Next() {
		var featureName string
		var weight float64
		if err := rows.Scan(&featureName, &weight); err == nil {
			ol.modelWeights[featureName] = weight
		}
	}

	// Load feature statistics
	rows, err = ol.db.Query("SELECT feature_name, mean_value, std_dev, min_value, max_value, sample_count, last_updated FROM feature_stats")
	if err != nil {
		log.Printf("Failed to load feature stats: %v", err)
		return
	}
	defer rows.Close()

	for rows.Next() {
		var featureName string
		var stats FeatureStats
		if err := rows.Scan(&featureName, &stats.Mean, &stats.StdDev, &stats.Min, &stats.Max, &stats.Count, &stats.LastUpdate); err == nil {
			ol.featureStats[featureName] = stats
		}
	}

	log.Printf("🧠 Loaded %d model weights and %d feature statistics", len(ol.modelWeights), len(ol.featureStats))
}

// ProcessFeedback processes feedback data to improve the model
func (ol *OnlineLearner) ProcessFeedback(feedback FeedbackData) {
	// Store feedback in database
	ol.storeFeedback(feedback)

	// Update performance metrics
	ol.updatePerformanceMetrics(feedback)

	// Update feature statistics with new data
	ol.updateFeatureStats(feedback.Features)

	// Adaptive learning based on performance
	if ol.shouldTriggerLearning() {
		ol.performOnlineLearning()
	}

	log.Printf("📚 Processed feedback for %s: predicted=%.3f, actual=%t",
		feedback.Symbol, feedback.PredictedProb, feedback.ActualPump)
}

// storeFeedback saves feedback data to database
func (ol *OnlineLearner) storeFeedback(feedback FeedbackData) {
	featuresJSON, _ := json.Marshal(feedback.Features)

	_, err := ol.db.Exec(`
		INSERT INTO feedback_data 
		(symbol, timestamp, predicted_prob, actual_pump, feedback_type, confidence, features)
		VALUES ($1, $2, $3, $4, $5, $6, $7)
	`, feedback.Symbol, feedback.Timestamp, feedback.PredictedProb,
		feedback.ActualPump, feedback.FeedbackType, feedback.Confidence, featuresJSON)

	if err != nil {
		log.Printf("Failed to store feedback: %v", err)
	}
}

// updatePerformanceMetrics updates model performance tracking
func (ol *OnlineLearner) updatePerformanceMetrics(feedback FeedbackData) {
	// Determine prediction result
	predicted := feedback.PredictedProb > 0.7 // Using same threshold as main detector
	actual := feedback.ActualPump

	// Update confusion matrix
	if predicted && actual {
		ol.performance.TruePositives++
	} else if predicted && !actual {
		ol.performance.FalsePositives++
	} else if !predicted && actual {
		ol.performance.FalseNegatives++
	} else {
		ol.performance.TrueNegatives++
	}

	ol.performance.SampleCount++

	// Calculate metrics
	total := float64(ol.performance.TruePositives + ol.performance.FalsePositives +
		ol.performance.TrueNegatives + ol.performance.FalseNegatives)

	if total > 0 {
		ol.performance.Accuracy = float64(ol.performance.TruePositives+ol.performance.TrueNegatives) / total

		if ol.performance.TruePositives+ol.performance.FalsePositives > 0 {
			ol.performance.Precision = float64(ol.performance.TruePositives) /
				float64(ol.performance.TruePositives+ol.performance.FalsePositives)
		}

		if ol.performance.TruePositives+ol.performance.FalseNegatives > 0 {
			ol.performance.Recall = float64(ol.performance.TruePositives) /
				float64(ol.performance.TruePositives+ol.performance.FalseNegatives)
		}

		if ol.performance.Precision+ol.performance.Recall > 0 {
			ol.performance.F1Score = 2 * (ol.performance.Precision * ol.performance.Recall) /
				(ol.performance.Precision + ol.performance.Recall)
		}
	}

	ol.performance.LastUpdate = time.Now()
}

// updateFeatureStats updates feature statistics for better normalization
func (ol *OnlineLearner) updateFeatureStats(features FeatureSet) {
	featureMap := map[string]float64{
		"volume_spike_ratio": features.VolumeSpikeRatio,
		"price_change_5m":    features.PriceChange5m,
		"volatility":         features.Volatility,
		"rsi":                features.RSI,
		"volume_momentum":    features.VolumeMomentum,
		"price_momentum":     features.PriceMomentum,
	}

	for name, value := range featureMap {
		stats, exists := ol.featureStats[name]
		if !exists {
			stats = FeatureStats{
				Mean:       value,
				Min:        value,
				Max:        value,
				Count:      1,
				LastUpdate: time.Now(),
			}
		} else {
			// Online update of statistics
			oldMean := stats.Mean
			stats.Count++
			stats.Mean = oldMean + (value-oldMean)/float64(stats.Count)

			// Update variance (for standard deviation)
			if stats.Count > 1 {
				variance := stats.StdDev * stats.StdDev
				variance = ((float64(stats.Count-1) * variance) + (value-oldMean)*(value-stats.Mean)) / float64(stats.Count)
				stats.StdDev = math.Sqrt(variance)
			}

			if value < stats.Min {
				stats.Min = value
			}
			if value > stats.Max {
				stats.Max = value
			}
			stats.LastUpdate = time.Now()
		}

		ol.featureStats[name] = stats

		// Persist to database
		ol.saveFeatureStats(name, stats)
	}
}

// saveFeatureStats saves feature statistics to database
func (ol *OnlineLearner) saveFeatureStats(name string, stats FeatureStats) {
	_, err := ol.db.Exec(`
		INSERT INTO feature_stats (feature_name, mean_value, std_dev, min_value, max_value, sample_count, last_updated)
		VALUES ($1, $2, $3, $4, $5, $6, $7)
		ON CONFLICT (feature_name) DO UPDATE SET
			mean_value = EXCLUDED.mean_value,
			std_dev = EXCLUDED.std_dev,
			min_value = EXCLUDED.min_value,
			max_value = EXCLUDED.max_value,
			sample_count = EXCLUDED.sample_count,
			last_updated = EXCLUDED.last_updated
	`, name, stats.Mean, stats.StdDev, stats.Min, stats.Max, stats.Count, stats.LastUpdate)

	if err != nil {
		log.Printf("Failed to save feature stats for %s: %v", name, err)
	}
}

// shouldTriggerLearning determines if online learning should be triggered
func (ol *OnlineLearner) shouldTriggerLearning() bool {
	// Trigger learning if:
	// 1. We have enough samples
	// 2. Performance is declining
	// 3. It's been a while since last retrain

	if ol.performance.SampleCount < int64(ol.minSamples) {
		return false
	}

	// Check if performance is declining (trigger more frequent updates)
	if ol.performance.F1Score < 0.6 && ol.performance.SampleCount > 50 {
		return true
	}

	// Regular retraining every hour with sufficient data
	timeSinceLastTrain := time.Since(ol.performance.LastUpdate)
	if timeSinceLastTrain > time.Hour && ol.performance.SampleCount > 20 {
		return true
	}

	return false
}

// performOnlineLearning performs gradient descent update on model weights
func (ol *OnlineLearner) performOnlineLearning() {
	log.Println("🎯 Performing online learning update...")

	// Get recent feedback data for training
	rows, err := ol.db.Query(`
		SELECT predicted_prob, actual_pump, features 
		FROM feedback_data 
		WHERE processed = FALSE 
		ORDER BY created_at DESC 
		LIMIT $1
	`, ol.batchSize)

	if err != nil {
		log.Printf("Failed to get feedback data for learning: %v", err)
		return
	}
	defer rows.Close()

	var trainingData []struct {
		predicted float64
		actual    bool
		features  FeatureSet
	}

	for rows.Next() {
		var predicted float64
		var actual bool
		var featuresJSON []byte

		if err := rows.Scan(&predicted, &actual, &featuresJSON); err != nil {
			continue
		}

		var features FeatureSet
		if err := json.Unmarshal(featuresJSON, &features); err != nil {
			continue
		}

		trainingData = append(trainingData, struct {
			predicted float64
			actual    bool
			features  FeatureSet
		}{predicted, actual, features})
	}

	if len(trainingData) == 0 {
		return
	}

	// Perform gradient descent update
	ol.gradientDescentUpdate(trainingData)

	// Mark processed feedback as processed
	ol.db.Exec("UPDATE feedback_data SET processed = TRUE WHERE processed = FALSE")

	log.Printf("🧠 Online learning completed with %d samples. F1-Score: %.3f",
		len(trainingData), ol.performance.F1Score)
}

// gradientDescentUpdate performs gradient descent on the model weights
func (ol *OnlineLearner) gradientDescentUpdate(trainingData []struct {
	predicted float64
	actual    bool
	features  FeatureSet
}) {
	for _, sample := range trainingData {
		// Convert actual to float (1.0 for pump, 0.0 for no pump)
		actualFloat := 0.0
		if sample.actual {
			actualFloat = 1.0
		}

		// Calculate error
		error := sample.predicted - actualFloat

		// Update weights for each feature
		featureMap := map[string]float64{
			"volume_spike_ratio": sample.features.VolumeSpikeRatio,
			"price_change_5m":    sample.features.PriceChange5m,
			"volatility":         sample.features.Volatility,
			"rsi":                sample.features.RSI,
			"volume_momentum":    sample.features.VolumeMomentum,
			"price_momentum":     sample.features.PriceMomentum,
		}

		for featureName, value := range featureMap {
			// Normalize feature value
			normalizedValue := ol.normalizeFeature(featureName, value)

			// Gradient descent update
			currentWeight := ol.modelWeights[featureName]
			gradient := error * normalizedValue
			newWeight := currentWeight - ol.learningRate*gradient

			ol.modelWeights[featureName] = newWeight

			// Save updated weight
			ol.saveModelWeight(featureName, newWeight)
		}
	}

	// Adaptive learning rate based on performance
	if ol.performance.F1Score > 0.8 {
		ol.learningRate *= 0.95 // Slow down if performing well
	} else if ol.performance.F1Score < 0.5 {
		ol.learningRate *= 1.05 // Speed up if performing poorly
	}

	// Keep learning rate in reasonable bounds
	if ol.learningRate < 0.001 {
		ol.learningRate = 0.001
	}
	if ol.learningRate > 0.1 {
		ol.learningRate = 0.1
	}
}

// normalizeFeature normalizes a feature value using stored statistics
func (ol *OnlineLearner) normalizeFeature(name string, value float64) float64 {
	stats, exists := ol.featureStats[name]
	if !exists || stats.StdDev == 0 {
		return value
	}

	// Z-score normalization
	return (value - stats.Mean) / stats.StdDev
}

// saveModelWeight saves a model weight to database
func (ol *OnlineLearner) saveModelWeight(featureName string, weight float64) {
	_, err := ol.db.Exec(`
		INSERT INTO model_weights (feature_name, weight_value, last_updated)
		VALUES ($1, $2, NOW())
		ON CONFLICT (feature_name) DO UPDATE SET
			weight_value = EXCLUDED.weight_value,
			last_updated = EXCLUDED.last_updated
	`, featureName, weight)

	if err != nil {
		log.Printf("Failed to save model weight for %s: %v", featureName, err)
	}
}

// GetEnhancedPrediction returns an enhanced prediction using learned weights
func (ol *OnlineLearner) GetEnhancedPrediction(features FeatureSet) float64 {
	if len(ol.modelWeights) == 0 {
		// No learned weights yet, return original prediction
		return 0.0
	}

	featureMap := map[string]float64{
		"volume_spike_ratio": features.VolumeSpikeRatio,
		"price_change_5m":    features.PriceChange5m,
		"volatility":         features.Volatility,
		"rsi":                features.RSI,
		"volume_momentum":    features.VolumeMomentum,
		"price_momentum":     features.PriceMomentum,
	}

	var weightedSum float64
	var totalWeight float64

	for name, value := range featureMap {
		if weight, exists := ol.modelWeights[name]; exists {
			normalizedValue := ol.normalizeFeature(name, value)
			weightedSum += weight * normalizedValue
			totalWeight += math.Abs(weight)
		}
	}

	if totalWeight == 0 {
		return 0.0
	}

	// Apply sigmoid to get probability
	enhancedScore := 1.0 / (1.0 + math.Exp(-weightedSum))

	return enhancedScore
}

// GetPerformanceMetrics returns current performance metrics
func (ol *OnlineLearner) GetPerformanceMetrics() PerformanceMetrics {
	return ol.performance
}
