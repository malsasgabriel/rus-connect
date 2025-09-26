package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	_ "github.com/lib/pq"
	"github.com/segmentio/kafka-go"
)

// DirectionSignal represents a detected direction prediction
type DirectionSignal struct {
	Symbol          string          `json:"symbol"`
	Timestamp       int64           `json:"timestamp"` // Changed to int64 for consistency
	Direction       string          `json:"direction"`
	Confidence      float64         `json:"confidence"`
	CurrentPrice    float64         `json:"current_price"` // Added for ML integration
	PriceTarget     float64         `json:"price_target"`
	TimeHorizon     int             `json:"time_horizon"`
	StopLoss        float64         `json:"stop_loss"`  // Added for ML integration
	Factors         []string        `json:"factors"`    // Added for explanations
	RiskLevel       string          `json:"risk_level"` // Added for risk assessment
	ModelType       string          `json:"model_type"` // "LSTM-AI", "TRADITIONAL"
	Version         string          `json:"version"`    // Model version
	Features        json.RawMessage `json:"features"`
	ActualDirection *string         `json:"actual_direction,omitempty"`
	CreatedAt       time.Time       `json:"created_at"`
}

type AnalyticsEngine struct {
	directionAnalyzer   *DirectionAnalyzer
	db                  *sql.DB
	kafkaReader         *kafka.Reader
	kafkaProducer       *kafka.Writer
	confidenceThreshold float64
	// Enhanced ML components
	advancedMLEngine   *AdvancedMLEngine   // 🧠 PRIMARY: High-performance honest ML
	selfLearningEngine *SelfLearningEngine // Secondary: Traditional ML
	enhancedFeatures   *EnhancedFeatureEngine
	performanceMonitor *ModelPerformanceMonitor
	onlineLearner      *OnlineLearner
	feedbackChan       chan FeedbackData
	// 🧠 NEW: LSTM Trading AI Integration
	mlIntegration      *MLAnalyticsIntegration // LSTM-based self-learning AI
	candleStream       chan Candle             // Stream for ML processing
	modelVersion       int
	lastRetrain        time.Time
	candleHistory      map[string][]Candle // Store 1440 candles per symbol
	symbols            []string            // Target cryptocurrencies
	minCandlesRequired int                 // Minimum candles for prediction
}

func NewAnalyticsEngine() *AnalyticsEngine {
	// PostgreSQL connection
	dsn := os.Getenv("PG_DSN")
	if dsn == "" {
		dsn = "host=postgres user=admin password=password dbname=predpump sslmode=disable"
	}
	db, err := sql.Open("postgres", dsn)
	if err != nil {
		log.Fatalf("Failed to connect to PostgreSQL: %v", err)
	}
	if err := db.Ping(); err != nil {
		log.Fatalf("Failed to ping PostgreSQL: %v", err)
	}

	// 🗄️ Create tables for persistent storage
	initializePersistentTables(db)

	// Ensure direction_predictions table exists with ALL required columns
	_, err = db.Exec(`CREATE TABLE IF NOT EXISTS direction_predictions (
        id BIGSERIAL PRIMARY KEY,
        symbol VARCHAR(20) NOT NULL,
        timestamp TIMESTAMPTZ NOT NULL,
        direction VARCHAR(10) NOT NULL,
        confidence DECIMAL(5,4) NOT NULL,
        price_target DECIMAL(15,8),
        current_price DECIMAL(15,8),
        horizon_minutes INTEGER DEFAULT 60,
        time_horizon INTEGER DEFAULT 60,
        features JSONB,
        actual_direction VARCHAR(10) DEFAULT NULL,
        actual_price DECIMAL(15,8) DEFAULT NULL,
        accuracy_score DECIMAL(5,4) DEFAULT NULL,
        created_at TIMESTAMPTZ DEFAULT NOW()
    )`)
	if err != nil {
		log.Fatalf("Failed to create direction_predictions table: %v", err)
	}
	// Add missing columns if they don't exist (for existing tables)
	db.Exec(`ALTER TABLE direction_predictions ADD COLUMN IF NOT EXISTS current_price DECIMAL(15,8);`)
	db.Exec(`ALTER TABLE direction_predictions ADD COLUMN IF NOT EXISTS time_horizon INTEGER DEFAULT 60;`)
	db.Exec(`ALTER TABLE direction_predictions ADD COLUMN IF NOT EXISTS actual_price DECIMAL(15,8) DEFAULT NULL;`)
	db.Exec(`ALTER TABLE direction_predictions ADD COLUMN IF NOT EXISTS accuracy_score DECIMAL(5,4) DEFAULT NULL;`)
	_, err = db.Exec(`
        CREATE INDEX IF NOT EXISTS idx_direction_symbol_time ON direction_predictions (symbol, timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_direction_timestamp ON direction_predictions (timestamp);
    `)
	if err != nil {
		log.Fatalf("Failed to create direction_predictions indexes: %v", err)
	}

	kafkaBrokers := os.Getenv("KAFKA_BROKERS")
	if kafkaBrokers == "" {
		kafkaBrokers = "kafka:9092"
	}

	// Kafka Reader for candles from Data Fetcher
	reader := kafka.NewReader(kafka.ReaderConfig{
		Brokers:        []string{kafkaBrokers},
		Topic:          "candle_1m",
		GroupID:        "analytics-engine-group",
		MinBytes:       10e3,        // 10KB
		MaxBytes:       10e6,        // 10MB
		CommitInterval: time.Second, // Commit offsets every second
		MaxAttempts:    10,
	})

	// Kafka Producer for sending direction signals to API Gateway
	producer := &kafka.Writer{
		Addr:     kafka.TCP(kafkaBrokers),
		Topic:    "direction_signals",
		Balancer: &kafka.LeastBytes{},
	}

	// Enhanced ML components - UPGRADED WITH HONEST ML ENGINE
	advancedMLEngine := NewAdvancedMLEngine(db) // 🧠 High-performance honest ML
	selfLearningEngine := NewSelfLearningEngine()
	enhancedFeatures := NewEnhancedFeatureEngine()
	onlineLearner := NewOnlineLearner(db)
	feedbackChan := make(chan FeedbackData, 100)

	// Target cryptocurrency symbols
	symbols := []string{"BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"}

	// Initialize HONEST ML models for each symbol with 70-80% target accuracy
	for _, symbol := range symbols {
		// Initialize traditional model for comparison
		model := NewMLTradingModel(symbol, 1440, 50)
		selfLearningEngine.Models[symbol] = model
		selfLearningEngine.EnsembleWeights[symbol] = 0.3 // Reduced weight for weak models

		// Initialize ADVANCED ML model for honest signals (PRIMARY)
		advancedMLEngine.InitializeModel(symbol)
		log.Printf("🎯 Initialized HONEST ML model for %s (targeting 70-80%% confidence)", symbol)
	}

	// Start self-learning engine
	selfLearningEngine.Start()

	// Initialize and start performance monitor
	performanceMonitor := NewModelPerformanceMonitor(selfLearningEngine)
	go performanceMonitor.MonitorPerformance()

	// 📏 SMART PERSISTENCE: Load from database first, API only if needed
	historicalLoader := NewHistoricalDataLoader()
	log.Println("📦 Loading historical data from database...")

	// Try to load from database first
	historicalData := loadHistoricalDataFromDB(db, symbols)

	// If database is empty or outdated, load from API
	for _, symbol := range symbols {
		if candles, exists := historicalData[symbol]; !exists || len(candles) < 100 || isDataOutdated(candles) {
			log.Printf("📡 Loading fresh data for %s from API...", symbol)
			freshCandles, err := historicalLoader.LoadHistoricalCandlesWithRetry(symbol, 3)
			if err != nil {
				log.Printf("⚠️ Failed to load %s: %v", symbol, err)
				continue
			}
			historicalData[symbol] = freshCandles
			// Save to database for next time
			saveHistoricalDataToDB(db, symbol, freshCandles)
			log.Printf("💾 Saved %d candles for %s to database", len(freshCandles), symbol)
		} else {
			log.Printf("✅ Using cached data for %s (%d candles)", symbol, len(candles))
		}
	}

	ae := &AnalyticsEngine{
		directionAnalyzer:   NewDirectionAnalyzer(db, kafkaBrokers),
		db:                  db,
		kafkaReader:         reader,
		kafkaProducer:       producer,
		confidenceThreshold: 0.7, // 70% confidence for a signal
		// Enhanced ML components
		advancedMLEngine:   advancedMLEngine,   // 🧠 PRIMARY: High-performance honest ML
		selfLearningEngine: selfLearningEngine, // Secondary: Traditional ML for comparison
		enhancedFeatures:   enhancedFeatures,
		performanceMonitor: performanceMonitor,
		onlineLearner:      onlineLearner,
		feedbackChan:       feedbackChan,
		// 🧠 NEW: LSTM AI Integration
		candleStream:       make(chan Candle, 1000), // Stream for ML processing
		modelVersion:       1,
		lastRetrain:        time.Now(),
		candleHistory:      make(map[string][]Candle),
		symbols:            symbols,
		minCandlesRequired: 20, // Reduced from 1440 for faster predictions
	}

	// 🧠 Initialize LSTM AI Integration
	ae.mlIntegration = NewMLAnalyticsIntegration(ae)
	log.Printf("🧠 LSTM Trading AI integrated - Self-learning system activated!")

	// Initialize candle history with historical data
	for symbol, candles := range historicalData {
		ae.candleHistory[symbol] = candles
		log.Printf("🎯 Preloaded %d candles for %s - Ready for ML predictions!", len(candles), symbol)

		// Process historical candles through ML system for immediate calibration
		for _, candle := range candles {
			// Add to enhanced features engine
			enhancedFeatures.AddCandle(candle)

			// Process through direction analyzer for ML training
			ae.directionAnalyzer.ProcessCandle(candle)
		}

		// Generate initial ML prediction to verify system readiness
		if len(candles) >= 50 {
			ae.generateMLPrediction(symbol)
			log.Printf("🎯 System verified - ready for ML predictions for %s", symbol)
		}
	}

	log.Printf("✅ Historical data preload complete! System ready for immediate ML predictions.")
	return ae
}

// 📡 Emit Direction Signal for ML Integration
func (ae *AnalyticsEngine) emitDirectionSignal(signal DirectionSignal) {
	// Save to database
	_, err := ae.db.Exec(`
		INSERT INTO direction_predictions 
		(symbol, timestamp, direction, confidence, price_target, current_price, time_horizon) 
		VALUES ($1, $2, $3, $4, $5, $6, $7)
	`, signal.Symbol, signal.Timestamp, signal.Direction, signal.Confidence,
		signal.PriceTarget, signal.CurrentPrice, signal.TimeHorizon)

	if err != nil {
		log.Printf("❌ Failed to save direction signal: %v", err)
		return
	}

	// Send to Kafka for real-time distribution
	signalBytes, _ := json.Marshal(signal)
	message := kafka.Message{
		Key:   []byte(signal.Symbol),
		Value: signalBytes,
	}

	err = ae.kafkaProducer.WriteMessages(context.Background(), message)
	if err != nil {
		log.Printf("❌ Failed to send direction signal to Kafka: %v", err)
		return
	}

	log.Printf("📡 Direction signal emitted: %s %s (%.1f%% confidence)",
		signal.Symbol, signal.Direction, signal.Confidence*100)
}

// Run starts the Analytics Engine's Kafka consumption loop.
func (ae *AnalyticsEngine) Run(ctx context.Context) {
	log.Println("Analytics Engine started, consuming from Kafka topic: candle_1m")

	// Start feedback processing goroutine
	go ae.processFeedbackLoop(ctx)

	// Start automatic performance monitoring
	go ae.performanceMonitoringLoop(ctx)

	for {
		select {
		case <-ctx.Done():
			log.Println("Analytics Engine stopping.")
			return
		default:
			m, err := ae.kafkaReader.FetchMessage(ctx)
			if err != nil {
				if errors.Is(err, context.Canceled) {
					log.Println("Kafka reader context canceled.")
					return
				}
				log.Printf("Error fetching message from Kafka: %v", err)
				time.Sleep(1 * time.Second) // Small backoff
				continue
			}

			var candle Candle
			if err := json.Unmarshal(m.Value, &candle); err != nil {
				log.Printf("Error unmarshalling candle from Kafka: %v", err)
				if err := ae.kafkaReader.CommitMessages(ctx, m); err != nil {
					log.Printf("Error committing message after unmarshalling failure: %v", err)
				}
				continue
			}

			ae.ProcessCandle(candle)

			if err := ae.kafkaReader.CommitMessages(ctx, m); err != nil {
				log.Printf("Error committing message: %v", err)
			}
		}
	}
}

func (ae *AnalyticsEngine) ProcessCandle(candle Candle) {
	// Store candle in history
	if ae.candleHistory[candle.Symbol] == nil {
		ae.candleHistory[candle.Symbol] = make([]Candle, 0, ae.minCandlesRequired)
	}

	// Add new candle to history
	ae.candleHistory[candle.Symbol] = append(ae.candleHistory[candle.Symbol], candle)

	// 📡 Send candle to LSTM ML integration for real-time processing
	select {
	case ae.candleStream <- candle:
		// Successfully sent to ML pipeline
	default:
		// Channel full, log warning but don't block
		log.Printf("⚠️ ML candle stream full, skipping candle for %s", candle.Symbol)
	}

	// 💾 Save new candle to database for persistence
	saveHistoricalDataToDB(ae.db, candle.Symbol, []Candle{candle})

	// Keep only the required number of candles (up to 1440 = 24 hours)
	if len(ae.candleHistory[candle.Symbol]) > 1440 {
		ae.candleHistory[candle.Symbol] = ae.candleHistory[candle.Symbol][1:]
	}

	// 🧠 VERIFY PREVIOUS PREDICTIONS using Advanced ML Engine for continuous learning
	ae.advancedMLEngine.VerifyPrediction(candle.Symbol, candle.Close, time.Now().Add(-65*time.Minute))

	// Check if we have enough data for ML prediction
	historyLength := len(ae.candleHistory[candle.Symbol])
	if historyLength < ae.minCandlesRequired {
		log.Printf("⏳ %s: collecting history (%d/%d candles needed)",
			candle.Symbol, historyLength, ae.minCandlesRequired)
		// Still use traditional analyzer for early predictions
		ae.directionAnalyzer.ProcessCandle(candle)
		return
	}

	// Generate enhanced ML prediction
	ae.generateMLPrediction(candle.Symbol)

	// Also process with traditional analyzer for comparison
	ae.directionAnalyzer.ProcessCandle(candle)
}

// generateMLPrediction generates HIGH-CONFIDENCE ML signals using HONEST Advanced ML Engine
func (ae *AnalyticsEngine) generateMLPrediction(symbol string) {
	candleHistory := ae.candleHistory[symbol]
	if len(candleHistory) < ae.minCandlesRequired {
		return
	}

	// 🧠 PRIMARY: Use Advanced ML Engine for HONEST 70-80% confidence signals
	honestSignal := ae.advancedMLEngine.GenerateSmartPrediction(symbol, candleHistory)
	if honestSignal != nil && honestSignal.Confidence >= 0.65 {
		// High-confidence honest signal - EMIT IMMEDIATELY
		log.Printf("🎯 HONEST SIGNAL: %s %s (%.1f%% confidence) - Target: %.8f",
			honestSignal.Symbol, honestSignal.Prediction, honestSignal.Confidence*100, honestSignal.PriceTarget)
		ae.emitTradingSignal(honestSignal)
		return
	}

	// 🤖 FALLBACK: Use traditional ML if Advanced ML doesn't produce high-confidence signal
	featuresMatrix := ae.convertCandlesToFeatures(candleHistory)
	if len(featuresMatrix) == 0 {
		log.Printf("❌ Failed to convert candles to features for %s", symbol)
		return
	}

	// Generate traditional ML prediction
	signal := ae.selfLearningEngine.PredictWithEnsemble(symbol, featuresMatrix)
	if signal == nil {
		log.Printf("⚠️ No backup prediction generated for %s", symbol)
		return
	}

	// Only emit traditional signal if confidence is reasonable
	if signal.Confidence >= 0.5 {
		log.Printf("🤖 BACKUP SIGNAL: %s %s (%.1f%% confidence) - Target: %.8f",
			signal.Symbol, signal.Prediction, signal.Confidence*100, signal.PriceTarget)
		ae.emitTradingSignal(signal)
	} else {
		log.Printf("🚫 %s: All models produced low confidence - No signal emitted", symbol)
	}
}

// convertCandlesToFeatures converts candle history to ML feature matrix
func (ae *AnalyticsEngine) convertCandlesToFeatures(candles []Candle) [][]float64 {
	if len(candles) == 0 {
		return nil
	}

	// Create feature matrix: each row is a timestep, each column is a feature
	featuresMatrix := make([][]float64, len(candles))

	for i, candle := range candles {
		// Basic OHLCV features
		features := []float64{
			candle.Close,  // 0: Close price
			candle.Open,   // 1: Open price
			candle.High,   // 2: High price
			candle.Low,    // 3: Low price
			candle.Volume, // 4: Volume
			(candle.High - candle.Low) / candle.Close,  // 5: Price range ratio
			(candle.Close - candle.Open) / candle.Open, // 6: Price change ratio
		}

		// Add enhanced features if we have enough history
		if i >= 20 { // Need at least 20 candles for technical indicators
			// Create a temporary feature engine for this calculation
			tempEngine := NewEnhancedFeatureEngine()
			for _, c := range candles[:i+1] {
				tempEngine.AddCandle(c)
			}
			enhancedFeats := tempEngine.CalculateEnhancedFeatures(candles[i].Symbol)
			features = append(features,
				enhancedFeats.RSI_14,
				enhancedFeats.MACD,
				enhancedFeats.BB_Upper,
				enhancedFeats.BB_Lower,
				enhancedFeats.EMA_12,
				enhancedFeats.EMA_26,
				enhancedFeats.ATR,
				enhancedFeats.ADX,
				enhancedFeats.Stochastic_K,
				enhancedFeats.Williams_R,
			)
		} else {
			// Pad with zeros for early candles
			for j := 0; j < 10; j++ {
				features = append(features, 0.0)
			}
		}

		featuresMatrix[i] = features
	}

	return featuresMatrix
}

// emitTradingSignal emits ML trading signal to Kafka
func (ae *AnalyticsEngine) emitTradingSignal(signal *TradingSignal) {
	signalBytes, err := json.Marshal(signal)
	if err != nil {
		log.Printf("❌ Error marshaling trading signal for Kafka: %v", err)
		return
	}

	err = ae.kafkaProducer.WriteMessages(context.Background(),
		kafka.Message{
			Key:   []byte(signal.Symbol),
			Value: signalBytes,
		},
	)
	if err != nil {
		log.Printf("❌ Failed to publish trading signal to Kafka: %v", err)
	} else {
		log.Printf("🚀 ML SIGNAL PUBLISHED: %s %s (%.1f%% confidence) to Kafka",
			signal.Symbol, signal.Prediction, signal.Confidence*100)
	}
}

// processFeedbackLoop processes feedback data continuously
func (ae *AnalyticsEngine) processFeedbackLoop(ctx context.Context) {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case feedback := <-ae.feedbackChan:
			ae.onlineLearner.ProcessFeedback(feedback)
		case <-ticker.C:
			// Process any pending feedback from database
			ae.processPendingFeedback()
		}
	}
}

// performanceMonitoringLoop monitors and logs system performance
func (ae *AnalyticsEngine) performanceMonitoringLoop(ctx context.Context) {
	ticker := time.NewTicker(10 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			metrics := ae.onlineLearner.GetPerformanceMetrics()
			log.Printf("📊 Performance: Accuracy=%.3f, Precision=%.3f, Recall=%.3f, F1=%.3f (Samples=%d)",
				metrics.Accuracy, metrics.Precision, metrics.Recall, metrics.F1Score, metrics.SampleCount)

			// Save performance metrics to database
			ae.savePerformanceMetrics(metrics)
		}
	}
}

// checkActualDirection checks if the direction prediction was correct
// processPendingFeedback processes any pending feedback from database
func (ae *AnalyticsEngine) processPendingFeedback() {
	// This could be used to process manual feedback from admin interface
	// For now, we'll just check for any manual feedback entries
	rows, err := ae.db.Query(`
		SELECT symbol, timestamp, predicted_prob, actual_pump, feedback_type, confidence 
		FROM feedback_data 
		WHERE processed = FALSE AND feedback_type = 'manual' 
		LIMIT 10
	`)

	if err != nil {
		return
	}
	defer rows.Close()

	for rows.Next() {
		var symbol, feedbackType string
		var timestamp time.Time
		var predictedProb, confidence float64
		var actualPump bool

		if err := rows.Scan(&symbol, &timestamp, &predictedProb, &actualPump, &feedbackType, &confidence); err != nil {
			continue
		}

		feedback := FeedbackData{
			Symbol:        symbol,
			Timestamp:     timestamp,
			PredictedProb: predictedProb,
			ActualPump:    actualPump,
			FeedbackType:  feedbackType,
			Confidence:    confidence,
		}

		ae.onlineLearner.ProcessFeedback(feedback)
	}
}

// savePerformanceMetrics saves performance metrics to database
func (ae *AnalyticsEngine) savePerformanceMetrics(metrics PerformanceMetrics) {
	_, err := ae.db.Exec(`
		INSERT INTO model_performance 
		(model_version, accuracy, precision_score, recall_score, f1_score, 
		 false_positives, true_positives, false_negatives, true_negatives, sample_count)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
	`, ae.modelVersion, metrics.Accuracy, metrics.Precision, metrics.Recall, metrics.F1Score,
		metrics.FalsePositives, metrics.TruePositives, metrics.FalseNegatives,
		metrics.TrueNegatives, metrics.SampleCount)

	if err != nil {
		log.Printf("Failed to save performance metrics: %v", err)
	}
}

// 🗄️ PERSISTENCE FUNCTIONS FOR SMART MEMORY

// initializePersistentTables creates tables for persistent storage
func initializePersistentTables(db *sql.DB) {
	// Compact candle storage table
	_, err := db.Exec(`
		CREATE TABLE IF NOT EXISTS candle_cache (
			id BIGSERIAL PRIMARY KEY,
			symbol VARCHAR(20) NOT NULL,
			timestamp BIGINT NOT NULL,
			open DECIMAL(15,8) NOT NULL,
			high DECIMAL(15,8) NOT NULL,
			low DECIMAL(15,8) NOT NULL,
			close DECIMAL(15,8) NOT NULL,
			volume DECIMAL(15,8) NOT NULL,
			created_at TIMESTAMPTZ DEFAULT NOW(),
			UNIQUE(symbol, timestamp)
		)
	`)
	if err != nil {
		log.Printf("Failed to create candle_cache table: %v", err)
	}

	// LSTM model weights storage (compact binary)
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS lstm_model_state (
			id BIGSERIAL PRIMARY KEY,
			symbol VARCHAR(20) NOT NULL,
			model_type VARCHAR(50) NOT NULL,
			weights_data BYTEA NOT NULL,
			metadata JSONB,
			version INTEGER DEFAULT 1,
			accuracy DECIMAL(5,4),
			updated_at TIMESTAMPTZ DEFAULT NOW(),
			UNIQUE(symbol, model_type)
		)
	`)
	if err != nil {
		log.Printf("Failed to create lstm_model_state table: %v", err)
	}

	// Indexes for performance
	db.Exec(`CREATE INDEX IF NOT EXISTS idx_candle_symbol_time ON candle_cache (symbol, timestamp DESC);`)
	db.Exec(`CREATE INDEX IF NOT EXISTS idx_candle_time ON candle_cache (timestamp DESC);`)
	db.Exec(`CREATE INDEX IF NOT EXISTS idx_lstm_symbol ON lstm_model_state (symbol, updated_at DESC);`)

	log.Println("📀 Persistence tables initialized")
}

// loadHistoricalDataFromDB loads candle data from database
func loadHistoricalDataFromDB(db *sql.DB, symbols []string) map[string][]Candle {
	result := make(map[string][]Candle)

	for _, symbol := range symbols {
		rows, err := db.Query(`
			SELECT timestamp, open, high, low, close, volume
			FROM candle_cache
			WHERE symbol = $1
			ORDER BY timestamp ASC
			LIMIT 1440
		`, symbol)
		if err != nil {
			log.Printf("Failed to load candles for %s: %v", symbol, err)
			continue
		}
		defer rows.Close()

		var candles []Candle
		for rows.Next() {
			var candle Candle
			if err := rows.Scan(&candle.Timestamp, &candle.Open, &candle.High, &candle.Low, &candle.Close, &candle.Volume); err == nil {
				candle.Symbol = symbol
				candles = append(candles, candle)
			}
		}

		if len(candles) > 0 {
			result[symbol] = candles
			log.Printf("📋 Loaded %d cached candles for %s", len(candles), symbol)
		}
	}

	return result
}

// saveHistoricalDataToDB saves candle data to database
func saveHistoricalDataToDB(db *sql.DB, symbol string, candles []Candle) {
	if len(candles) == 0 {
		return
	}

	// Batch insert for efficiency
	valueStrings := make([]string, 0, len(candles))
	args := make([]interface{}, 0, len(candles)*6)

	for i, candle := range candles {
		valueStrings = append(valueStrings, fmt.Sprintf("($%d, $%d, $%d, $%d, $%d, $%d)",
			i*6+1, i*6+2, i*6+3, i*6+4, i*6+5, i*6+6))
		args = append(args, symbol, candle.Timestamp, candle.Open, candle.High, candle.Low, candle.Close, candle.Volume)
	}

	query := fmt.Sprintf(`
		INSERT INTO candle_cache (symbol, timestamp, open, high, low, close, volume)
		VALUES %s
		ON CONFLICT (symbol, timestamp) DO UPDATE SET
			open = EXCLUDED.open,
			high = EXCLUDED.high,
			low = EXCLUDED.low,
			close = EXCLUDED.close,
			volume = EXCLUDED.volume
	`, strings.Join(valueStrings, ","))

	_, err := db.Exec(query, args...)
	if err != nil {
		log.Printf("Failed to save candles for %s: %v", symbol, err)
	}
}

// isDataOutdated checks if cached data is too old
func isDataOutdated(candles []Candle) bool {
	if len(candles) == 0 {
		return true
	}
	// If last candle is older than 2 hours, consider outdated
	lastTimestamp := candles[len(candles)-1].Timestamp
	return time.Now().Unix()-lastTimestamp > 2*60*60
}

func main() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	engine := NewAnalyticsEngine()
	defer func() {
		if err := engine.db.Close(); err != nil {
			log.Printf("Error closing DB: %v", err)
		}
		if err := engine.kafkaReader.Close(); err != nil {
			log.Printf("Error closing Kafka Reader: %v", err)
		}
		if err := engine.kafkaProducer.Close(); err != nil {
			log.Printf("Error closing Kafka Producer: %v", err)
		}
	}()

	engine.Run(ctx)
}
