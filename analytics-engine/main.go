package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	_ "github.com/ClickHouse/clickhouse-go/v2"
	"github.com/gin-gonic/gin"
	"github.com/segmentio/kafka-go"
)

// Constants for AnalyticsEngine
const (
	MaxCandleHistory       = 1440
	HTTPReadTimeout        = 5 * time.Second
	HTTPWriteTimeout       = 10 * time.Second
	HTTPIdleTimeout        = 60 * time.Second
	CandleStreamBufferSize = 1000
	FeedbackChanBufferSize = 100
)

// DirectionSignal is defined in types.go - removed duplicate definition

type AnalyticsEngine struct {
	// Removed undefined types - using simplified structure
	// directionAnalyzer   *DirectionAnalyzer  // Not defined
	// advancedMLEngine   *AdvancedMLEngine   // Not defined
	// selfLearningEngine *SelfLearningEngine // Not defined
	// enhancedFeatures   *EnhancedFeatureEngine // Not defined
	// performanceMonitor *ModelPerformanceMonitor // Not defined
	// onlineLearner      *OnlineLearner // Not defined
	// feedbackChan       chan FeedbackData // Not defined
	// mlIntegration      *MLAnalyticsIntegration // Not defined
	// anomalyDetector    *VolumeAnomalyDetector // Not defined
	// ensemble           *EnsembleTradingAI // Not defined

	db                           *sql.DB
	kafkaReader                  *kafka.Reader
	kafkaProducer                *kafka.Writer
	confidenceThreshold          float64
	candleStream                 chan Candle // Stream for ML processing
	modelVersion                 int
	lastRetrain                  time.Time
	candleHistory                map[string][]Candle  // Store 1440 candles per symbol
	candleHistoryMu              sync.RWMutex         // Mutex to protect candleHistory map
	symbolAccessTime             map[string]time.Time // Track last access time for each symbol
	symbols                      []string             // Target cryptocurrencies
	minCandlesRequired           int32                // Minimum candles for prediction (atomic)
	featureEngine                *SimpleFeatureEngine
	normalizerManager            *FeatureNormalizerManager
	models                       map[string]*SimpleNeuralNetwork
	pendingExamples              map[string][]PendingExample
	trainingData                 map[string][]TrainingExample
	lastTrainedAt                map[string]time.Time
	mlMu                         sync.RWMutex
	predictionHorizonMinutes     int
	neutralThreshold             float64
	neutralThresholdFloor        float64
	neutralThresholdCeil         float64
	neutralAdjustStep            float64
	sidewaysCeilingPct           float64
	symbolNeutralThresholds      map[string]float64
	trainEpochs                  int
	retrainInterval              time.Duration
	maxTrainingExamples          int
	bootstrapMinCandles          int
	steadyMinCandles             int
	autopilotEnabled             bool
	autopilotInterval            time.Duration
	autopilotSilenceTimeout      time.Duration
	autopilotActionCooldown      time.Duration
	autopilotMinDirectional      float64
	autopilotMinSignalSamples    int
	autopilotResetMinAccuracy    float64
	autopilotResetMinPredictions int
	autopilotLastAction          map[string]time.Time
	historyReadyAt               map[string]time.Time
	signalsEmittedBySymbol       map[string]int64
	lastZeroSignalWarnAt         map[string]time.Time
	signalsEmittedCount          int64
	httpServer                   *http.Server // HTTP server for API endpoints
	kafkaBrokers                 []string
	// Context for cancellation
	ctx       context.Context
	cancel    context.CancelFunc
	startTime time.Time // Track start time for uptime calculation
}

func NewAnalyticsEngine() *AnalyticsEngine {
	// ClickHouse connection
	dsn := os.Getenv("CH_DSN")
	if dsn == "" {
		dsn = "clickhouse://app:app_password@clickhouse:9000/default?dial_timeout=5s&max_execution_time=60"
	}

	db, err := sql.Open("clickhouse", dsn)
	if err != nil {
		log.Fatalf("Failed to connect to ClickHouse: %v", err)
	}
	if err := db.Ping(); err != nil {
		log.Fatalf("Failed to ping ClickHouse: %v", err)
	}

	// 🗄️ Create tables for persistent storage
	initializePersistentTables(db)

	// Ensure direction_predictions table exists with ALL required columns
	_, err = db.Exec(`CREATE TABLE IF NOT EXISTS direction_predictions (
        symbol String,
        timestamp DateTime,
        direction String,
        confidence Float64,
        trust_stage String,
        model_age_sec Int64,
        label_horizon_min Int32,
        class_probs String,
        price_target Float64,
        current_price Float64,
        stop_loss Float64,
        volatility Float64,
        model_used String,
        time_horizon Int32,
        features String,
        actual_direction String,
        actual_price Float64,
        accuracy_score Float64,
        created_at DateTime DEFAULT now()
    ) ENGINE = MergeTree
    PARTITION BY toYYYYMM(timestamp)
    ORDER BY (symbol, timestamp, created_at)`)
	if err != nil {
		log.Fatalf("Failed to create direction_predictions table: %v", err)
	}
	_, _ = db.Exec(`ALTER TABLE direction_predictions ADD COLUMN IF NOT EXISTS trust_stage String DEFAULT 'cold_start'`)
	_, _ = db.Exec(`ALTER TABLE direction_predictions ADD COLUMN IF NOT EXISTS model_age_sec Int64 DEFAULT 0`)
	_, _ = db.Exec(`ALTER TABLE direction_predictions ADD COLUMN IF NOT EXISTS label_horizon_min Int32 DEFAULT 0`)
	_, _ = db.Exec(`ALTER TABLE direction_predictions ADD COLUMN IF NOT EXISTS class_probs String DEFAULT '{}'`)
	_, _ = db.Exec(`ALTER TABLE direction_predictions ADD COLUMN IF NOT EXISTS stop_loss Float64 DEFAULT 0`)
	_, _ = db.Exec(`ALTER TABLE direction_predictions ADD COLUMN IF NOT EXISTS volatility Float64 DEFAULT 0`)
	_, _ = db.Exec(`ALTER TABLE direction_predictions ADD COLUMN IF NOT EXISTS model_used String DEFAULT 'SimpleNN'`)
	_, _ = db.Exec(`ALTER TABLE direction_predictions ADD COLUMN IF NOT EXISTS prediction_count Int32 DEFAULT 0`)

	kafkaBrokers := os.Getenv("KAFKA_BROKERS")
	if kafkaBrokers == "" {
		// Default to kafka service inside Docker Compose network
		kafkaBrokers = "kafka:9092"
	}
	// support comma-separated list of brokers
	brokerList := strings.Split(kafkaBrokers, ",")

	// Kafka Reader for candles from Data Fetcher
	reader := kafka.NewReader(kafka.ReaderConfig{
		Brokers:        brokerList,
		Topic:          "candle_1m",
		GroupID:        "analytics-engine-group",
		MinBytes:       10e3,        // 10KB
		MaxBytes:       10e6,        // 10MB
		CommitInterval: time.Second, // Commit offsets every second
		MaxAttempts:    10,
	})

	// Kafka Producer for sending direction signals to API Gateway
	producer := &kafka.Writer{
		Addr:     kafka.TCP(brokerList...),
		Topic:    "direction_signals",
		Balancer: &kafka.LeastBytes{},
	}

	// Ensure model_analyses table exists (if function exists)
	// EnsureModelAnalysesTable(db) // Commented out - function may not exist

	// Target cryptocurrency symbols
	symbols := []string{"BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"}

	// Simplified initialization - removed undefined types
	ae := &AnalyticsEngine{
		// directionAnalyzer:   NewDirectionAnalyzer(db, kafkaBrokers), // Commented out - not defined
		db:                     db,
		kafkaReader:            reader,
		kafkaProducer:          producer,
		confidenceThreshold:    getEnvFloat("CONFIDENCE_THRESHOLD", 0.55),
		candleStream:           make(chan Candle, 5000),
		modelVersion:           1,
		lastRetrain:            time.Now(),
		candleHistory:          make(map[string][]Candle),
		symbolAccessTime:       make(map[string]time.Time),
		symbols:                symbols,
		minCandlesRequired:     1440, // Overridden by online learning config
		historyReadyAt:         make(map[string]time.Time),
		signalsEmittedBySymbol: make(map[string]int64),
		lastZeroSignalWarnAt:   make(map[string]time.Time),
	}
	ae.kafkaBrokers = brokerList
	ae.initializeOnlineLearning()

	// Create context for cancellation
	ae.ctx, ae.cancel = context.WithCancel(context.Background())
	if ae.autopilotEnabled {
		go ae.runMLAutopilot(ae.ctx)
	}

	// Track start time for uptime calculation
	ae.startTime = time.Now()

	// 🚀 START HTTP SERVER EARLY so healthchecks can pass during long init
	go ae.startHTTPServer(ae.ctx)
	log.Println("🚀 HTTP server starting in background...")

	// Asynchronously bootstrap historical data to avoid blocking startup
	go ae.bootstrapHistoricalData()

	// Start candle stream monitoring
	go ae.monitorCandleStream()

	// Start memory usage monitoring (FIX: prevent memory leaks)
	go ae.monitorMemoryUsage()

	// Start ML prediction labeling job
	go ae.runLabelingJob(ae.ctx)

	log.Printf("✅ AnalyticsEngine initialized. Historical data loading in background.")
	return ae
}

// bootstrapHistoricalData loads and processes historical data in the background
// nolint:unused // Reserved for future use when historical data loading is implemented
func (ae *AnalyticsEngine) bootstrapHistoricalData() {
	select {
	case <-ae.ctx.Done():
		log.Println("Bootstrap cancelled due to context cancellation")
		return
	default:
	}

	log.Println("[BG] Loading historical data from database...")
	historicalData := loadHistoricalDataFromDB(ae.db, ae.symbols)

	minLoaded := 0
	for symbol, candles := range historicalData {
		select {
		case <-ae.ctx.Done():
			log.Println("Bootstrap cancelled during initialization")
			return
		default:
		}

		ae.candleHistoryMu.Lock()
		ae.candleHistory[symbol] = candles
		ae.symbolAccessTime[symbol] = time.Now()
		ae.candleHistoryMu.Unlock()
		if ae.featureEngine != nil {
			for _, c := range candles {
				ae.featureEngine.AddCandle(c)
			}
		}
		seeded := ae.seedTrainingDataFromHistory(symbol, candles)
		if seeded > 0 {
			log.Printf("[BG] Seeded %d historical training examples for %s", seeded, symbol)
			ae.maybeRetrain(symbol, true)
		}

		if minLoaded == 0 || len(candles) < minLoaded {
			minLoaded = len(candles)
		}
		log.Printf("[BG] Preloaded %d candles for %s", len(candles), symbol)
	}

	if minLoaded >= 5 && ae.getMinCandlesRequired() > minLoaded {
		ae.setMinCandlesRequired(minLoaded)
		log.Printf("inference bootstrap adjusted to available history: required=%d (steady target=%d)", ae.getMinCandlesRequired(), ae.steadyMinCandles)
	}
	log.Printf("[BG] Historical data preload complete")
}

// emitDirectionSignal emits ML trading signal to Kafka
// nolint:unused // Reserved for future use when direction signal emission is implemented
func (ae *AnalyticsEngine) emitDirectionSignal(signal DirectionSignal) {
	if signal.Timestamp == 0 {
		signal.Timestamp = time.Now().Unix()
	}
	if signal.TrustStage == "" {
		signal.TrustStage = "cold_start"
	}
	signal.CreatedAt = time.Now().UTC()
	signal.TimestampISO = signal.CreatedAt.Format(time.RFC3339)

	if ae.db != nil {
		classProbsJSON, _ := json.Marshal(signal.ClassProbs)
		_, err := ae.db.Exec(`
			INSERT INTO direction_predictions
			(symbol, timestamp, direction, confidence, trust_stage, model_age_sec, prediction_count, label_horizon_min, class_probs, price_target, current_price, stop_loss, volatility, model_used, time_horizon, features)
			VALUES (?, toDateTime(?), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		`, signal.Symbol, signal.Timestamp, signal.Direction, signal.Confidence,
			signal.TrustStage, signal.ModelAgeSec, signal.PredictionCount, signal.LabelHorizonMin, string(classProbsJSON),
			signal.PriceTarget, signal.CurrentPrice, signal.StopLoss, signal.Volatility, signal.ModelUsed, signal.TimeHorizon, "")
		if err != nil {
			log.Printf("Failed to save direction signal: %v", err)
			return
		}
	}

	ae.mlMu.Lock()
	ae.signalsEmittedCount++
	ae.signalsEmittedBySymbol[signal.Symbol]++
	ae.mlMu.Unlock()

	if ae.kafkaProducer != nil {
		if signal.ModelUsed == "" {
			if signal.ModelType != "" {
				signal.ModelUsed = signal.ModelType
			} else if signal.Version != "" {
				signal.ModelUsed = signal.Version
			} else {
				signal.ModelUsed = "unknown"
			}
		}

		signalBytes, _ := json.Marshal(signal)
		message := kafka.Message{
			Key:   []byte(signal.Symbol),
			Value: signalBytes,
		}

		if err := ae.kafkaProducer.WriteMessages(context.Background(), message); err != nil {
			log.Printf("Failed to send direction signal to Kafka: %v", err)
			return
		}
	}

	log.Printf("signals_emitted_count=%d symbol=%s direction=%s confidence=%.4f",
		ae.signalsEmittedCount, signal.Symbol, signal.Direction, signal.Confidence)
}

// Run starts the Analytics Engine's Kafka consumption loop.
func (ae *AnalyticsEngine) Run(ctx context.Context) {
	// Start background goroutines for feedback loop, performance monitoring, etc.
	// Commented out - selfLearningEngine not defined
	// if ae.selfLearningEngine != nil {
	// 	go ae.selfLearningEngine.Start()
	// }
	// Start feedback processing loop
	// if ae.selfLearningEngine != nil {
	// 	go ae.processFeedbackLoop(ctx)
	// }

	// Start performance monitoring loop
	// Commented out - performanceMonitor not defined
	// if ae.performanceMonitor != nil {
	// 	go ae.performanceMonitoringLoop(ctx)
	// }
	//
	// // Start performance monitoring
	// if ae.performanceMonitor != nil {
	// 	go ae.performanceMonitor.MonitorPerformance(ctx)
	// }

	// Main loop: consume from Kafka
	log.Println("🚀 Starting main event loop: consuming from Kafka...")
	for {
		select {
		case <-ctx.Done():
			log.Println("Analytics Engine stopping.")
			// Stop all components
			// Commented out - selfLearningEngine and performanceMonitor not defined
			// if ae.selfLearningEngine != nil {
			// 	ae.selfLearningEngine.Stop()
			// }
			// if ae.performanceMonitor != nil {
			// 	ae.performanceMonitor.Stop()
			// }
			// Shutdown HTTP server gracefully
			if ae.httpServer != nil {
				ae.httpServer.Shutdown(context.Background())
			}
			// Cancel our own context
			if ae.cancel != nil {
				ae.cancel()
			}
			return
		case <-ae.ctx.Done():
			log.Println("Analytics Engine stopping due to internal context cancellation.")
			// Stop all components
			// Commented out - selfLearningEngine and performanceMonitor not defined
			// if ae.selfLearningEngine != nil {
			// 	ae.selfLearningEngine.Stop()
			// }
			// if ae.performanceMonitor != nil {
			// 	ae.performanceMonitor.Stop()
			// }
			// Shutdown HTTP server gracefully
			if ae.httpServer != nil {
				ae.httpServer.Shutdown(context.Background())
			}
			return
		default:
			// Check if kafkaReader is nil before using it
			if ae.kafkaReader == nil {
				log.Println("Kafka reader is nil, skipping message fetch")
				time.Sleep(1 * time.Second) // Small backoff
				continue
			}

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
				// Check if kafkaReader is nil before committing messages
				if ae.kafkaReader != nil {
					if err := ae.kafkaReader.CommitMessages(ctx, m); err != nil {
						log.Printf("Error committing message after unmarshalling failure: %v", err)
					}
				}
				continue
			}

			ae.ProcessCandle(candle)

			// Check if kafkaReader is nil before committing messages
			if ae.kafkaReader != nil {
				if err := ae.kafkaReader.CommitMessages(ctx, m); err != nil {
					log.Printf("Error committing message: %v", err)
				}
			}
		}
	}
}

// startHTTPServer starts the HTTP server for ML metrics API
func (ae *AnalyticsEngine) startHTTPServer(ctx context.Context) {
	// Use explicit gin engine with logger/recovery so logs are not swallowed
	r := gin.New()
	r.Use(gin.Logger())
	r.Use(gin.Recovery())

	// Standardized health endpoints
	r.GET("/health", ae.handleHealthz)
	r.GET("/healthz", ae.handleHealthz)
	r.GET("/readyz", ae.handleReadyz)

	// Compatibility endpoints used by API gateway
	r.GET("/model/performance", ae.handleModelPerformance)
	r.POST("/api/v1/feedback", ae.handleFeedback)
	r.GET("/api/v1/ml/training-history", ae.handleTrainingHistory)
	r.GET("/api/v1/ml/signal-stats", ae.handleSignalStats)
	r.GET("/api/v1/ml/signals/recent", ae.handleRecentSignals)
	r.GET("/api/v1/ml/signals/history", ae.handleSignalsHistory)

	// ML Metrics endpoints
	r.GET("/api/v1/ml/metrics", ae.handleGetMLMetrics)
	r.GET("/api/v1/ml/calibration", ae.handleGetCalibrationStatus)
	r.POST("/api/v1/ml/calibration/start", ae.handleStartAutoCalibration)
	r.GET("/api/v1/ml/learning-progress", ae.handleGetMLLearningProgress)
	// Infrastructure monitoring endpoints
	r.GET("/api/v1/infrastructure/metrics", ae.handleGetInfrastructureMetrics)
	// Trader Mind endpoint: aggregated model analysis for a symbol
	r.GET("/api/v1/trader-mind/:symbol", ae.handleTraderMind)
	r.GET("/api/v1/trader-mind/full/:symbol", ae.handleTraderMindFull)
	r.GET("/api/v1/model-analyses/:symbol", ae.handleModelAnalyses)
	r.POST("/api/v1/model/retrain", ae.handleModelRetrain)

	// Get port from environment or default to 8081
	port := os.Getenv("ANALYTICS_ENGINE_PORT")
	if port == "" {
		port = "8081"
	}

	// Log important environment for diagnostics
	log.Printf("Starting Analytics Engine HTTP server (port=%s, kafka_brokers=%v)", port, ae.kafkaBrokers)

	// Get timeout values from environment variables or use defaults
	readTimeout := HTTPReadTimeout
	writeTimeout := HTTPWriteTimeout
	idleTimeout := HTTPIdleTimeout

	if readTimeoutStr := os.Getenv("HTTP_READ_TIMEOUT"); readTimeoutStr != "" {
		if d, err := time.ParseDuration(readTimeoutStr); err == nil {
			readTimeout = d
		}
	}

	if writeTimeoutStr := os.Getenv("HTTP_WRITE_TIMEOUT"); writeTimeoutStr != "" {
		if d, err := time.ParseDuration(writeTimeoutStr); err == nil {
			writeTimeout = d
		}
	}

	if idleTimeoutStr := os.Getenv("HTTP_IDLE_TIMEOUT"); idleTimeoutStr != "" {
		if d, err := time.ParseDuration(idleTimeoutStr); err == nil {
			idleTimeout = d
		}
	}

	ae.httpServer = &http.Server{
		Addr:         ":" + port, // bind all interfaces
		Handler:      r,
		ReadTimeout:  readTimeout,
		WriteTimeout: writeTimeout,
		IdleTimeout:  idleTimeout,
	}

	// Start server in a goroutine so we can listen for context cancellation
	go func() {
		<-ctx.Done()
		log.Println("Analytics Engine HTTP server stopping due to context cancellation")
		if ae.httpServer != nil {
			ae.httpServer.Shutdown(context.Background())
		}
	}()

	// Listen in a blocking call inside the goroutine and log clearly on success/error
	if err := ae.httpServer.ListenAndServe(); err != nil {
		if err == http.ErrServerClosed {
			log.Printf("Analytics Engine HTTP server closed")
		} else {
			log.Printf("Analytics Engine HTTP server error: %v", err)
		}
	}
}

func (ae *AnalyticsEngine) handleHealthz(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status": "ok",
		"time":   time.Now().UTC(),
	})
}

func (ae *AnalyticsEngine) handleReadyz(c *gin.Context) {
	dbReady := false
	kafkaReady := ae.kafkaReader != nil

	if ae.db != nil && ae.db.Ping() == nil {
		dbReady = true
	}

	if !dbReady || !kafkaReady {
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"status":      "not_ready",
			"db_ready":    dbReady,
			"kafka_ready": kafkaReady,
			"time":        time.Now().UTC(),
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"status":      "ready",
		"db_ready":    true,
		"kafka_ready": true,
		"time":        time.Now().UTC(),
	})
}

func (ae *AnalyticsEngine) handleModelPerformance(c *gin.Context) {
	c.JSON(http.StatusOK, ae.GetDetailedMLMetrics())
}

func (ae *AnalyticsEngine) handleFeedback(c *gin.Context) {
	var req struct {
		Symbol        string  `json:"symbol" binding:"required"`
		Timestamp     int64   `json:"timestamp" binding:"required"`
		PredictedProb float64 `json:"predicted_prob" binding:"required"`
		ActualPump    bool    `json:"actual_pump" binding:"required"`
		Confidence    float64 `json:"confidence"`
		Notes         string  `json:"notes"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if ae.db != nil {
		_, err := ae.db.Exec(`
			INSERT INTO feedback_data (symbol, timestamp, predicted_prob, actual_pump, feedback_type, confidence, notes, processed)
			VALUES (?, toDateTime(?), ?, ?, 'manual', ?, ?, 0)
		`, req.Symbol, req.Timestamp, req.PredictedProb, req.ActualPump, req.Confidence, req.Notes)
		if err != nil {
			log.Printf("Failed to store feedback: %v", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to store feedback"})
			return
		}
	}

	c.JSON(http.StatusOK, gin.H{
		"status":  "accepted",
		"symbol":  strings.ToUpper(req.Symbol),
		"created": time.Now().UTC(),
	})
}

func (ae *AnalyticsEngine) handleTrainingHistory(c *gin.Context) {
	symbol := strings.ToUpper(strings.TrimSpace(c.DefaultQuery("symbol", "BTCUSDT")))
	limit, err := strconv.Atoi(c.DefaultQuery("limit", "50"))
	if err != nil || limit <= 0 || limit > 500 {
		limit = 50
	}

	history := []map[string]interface{}{}
	if ae.db != nil {
		query := fmt.Sprintf(`
			SELECT created_at, symbol, sample_count, train_samples, val_samples, class_down, class_sideways, class_up,
			       val_accuracy, threshold, trust_stage, weight_down, weight_sideways, weight_up
			FROM training_events
			WHERE symbol = ?
			ORDER BY created_at DESC
			LIMIT %d
		`, limit)
		rows, qerr := ae.db.Query(query, symbol)
		if qerr == nil {
			defer rows.Close()
			for rows.Next() {
				var createdAt time.Time
				var rowSymbol string
				var sampleCount, trainSamples, valSamples, classDown, classSideways, classUp sql.NullInt64
				var valAccuracy, threshold, weightDown, weightSideways, weightUp sql.NullFloat64
				var trustStage sql.NullString
				if scanErr := rows.Scan(
					&createdAt, &rowSymbol, &sampleCount, &trainSamples, &valSamples, &classDown, &classSideways, &classUp,
					&valAccuracy, &threshold, &trustStage, &weightDown, &weightSideways, &weightUp,
				); scanErr != nil {
					continue
				}
				history = append(history, map[string]interface{}{
					"created_at":    createdAt.Unix(),
					"symbol":        rowSymbol,
					"sample_count":  nullInt(sampleCount),
					"train_samples": nullInt(trainSamples),
					"val_samples":   nullInt(valSamples),
					"class_counts": map[string]int64{
						"down":     nullInt(classDown),
						"sideways": nullInt(classSideways),
						"up":       nullInt(classUp),
					},
					"accuracy":  nullFloat(valAccuracy), // compatibility alias
					"f1_score":  nullFloat(valAccuracy), // compatibility alias
					"threshold": nullFloat(threshold),
					"trust_stage": func() string {
						if trustStage.Valid {
							return trustStage.String
						}
						return "cold_start"
					}(),
					"class_weights": map[string]float64{
						"down":     nullFloat(weightDown),
						"sideways": nullFloat(weightSideways),
						"up":       nullFloat(weightUp),
					},
				})
			}
		}
	}

	c.JSON(http.StatusOK, gin.H{
		"symbol":  symbol,
		"limit":   limit,
		"history": history,
	})
}

func (ae *AnalyticsEngine) handleSignalStats(c *gin.Context) {
	symbol := strings.ToUpper(strings.TrimSpace(c.DefaultQuery("symbol", "BTCUSDT")))
	hours, err := strconv.Atoi(c.DefaultQuery("hours", "24"))
	if err != nil || hours <= 0 || hours > 720 {
		hours = 24
	}

	result := gin.H{
		"symbol":           symbol,
		"hours":            hours,
		"total_signals":    0,
		"above_threshold":  0,
		"up":               0,
		"down":             0,
		"sideways":         0,
		"up_signals":       0,
		"down_signals":     0,
		"sideways_signals": 0,
		"cold_start":       0,
		"avg_confidence":   0.0,
		"up_rate":          0.0,
		"down_rate":        0.0,
		"sideways_rate":    0.0,
		"cold_start_rate":  0.0,
		"direction_distribution": gin.H{
			"up":       0.0,
			"down":     0.0,
			"sideways": 0.0,
		},
	}

	if ae.db != nil {
		query := fmt.Sprintf(`
			SELECT direction, confidence, trust_stage
			FROM direction_predictions
			WHERE symbol = ? AND created_at >= now() - INTERVAL %d HOUR
		`, hours)
		rows, qerr := ae.db.Query(query, symbol)
		if qerr == nil {
			defer rows.Close()
			total := 0
			up := 0
			down := 0
			sideways := 0
			coldStart := 0
			confSum := 0.0
			for rows.Next() {
				var dir string
				var conf float64
				var trustStage sql.NullString
				if scanErr := rows.Scan(&dir, &conf, &trustStage); scanErr != nil {
					continue
				}
				total++
				confSum += conf
				if trustStage.Valid && trustStage.String == "cold_start" {
					coldStart++
				}
				switch strings.ToUpper(dir) {
				case "UP":
					up++
				case "DOWN":
					down++
				default:
					sideways++
				}
			}
			avg := 0.0
			upRate := 0.0
			downRate := 0.0
			sidewaysRate := 0.0
			coldStartRate := 0.0
			if total > 0 {
				avg = confSum / float64(total)
				upRate = float64(up) / float64(total)
				downRate = float64(down) / float64(total)
				sidewaysRate = float64(sideways) / float64(total)
				coldStartRate = float64(coldStart) / float64(total)
			}
			result = gin.H{
				"symbol":          symbol,
				"hours":           hours,
				"total_signals":   total,
				"up":              up,
				"down":            down,
				"sideways":        sideways,
				"cold_start":      coldStart,
				"avg_confidence":  avg,
				"up_rate":         upRate,
				"down_rate":       downRate,
				"sideways_rate":   sidewaysRate,
				"cold_start_rate": coldStartRate,
				"direction_distribution": gin.H{
					"up":       upRate * 100.0,
					"down":     downRate * 100.0,
					"sideways": sidewaysRate * 100.0,
				},
				// Compatibility aliases for older frontend widgets.
				"up_signals":       up,
				"down_signals":     down,
				"sideways_signals": sideways,
				"above_threshold":  up + down,
			}
		}
	}

	c.JSON(http.StatusOK, result)
}

func (ae *AnalyticsEngine) handleRecentSignals(c *gin.Context) {
	limit, err := strconv.Atoi(c.DefaultQuery("limit", "50"))
	if err != nil || limit <= 0 || limit > 500 {
		limit = 50
	}

	hours, err := strconv.Atoi(c.DefaultQuery("hours", "24"))
	if err != nil || hours <= 0 || hours > 720 {
		hours = 24
	}

	symbolFilter := strings.ToUpper(strings.TrimSpace(c.Query("symbol")))
	result := []gin.H{}

	if ae.db == nil {
		c.JSON(http.StatusOK, gin.H{
			"signals": result,
			"count":   0,
		})
		return
	}

	baseQuery := fmt.Sprintf(`
		SELECT symbol, direction, confidence, price_target, current_price, time_horizon,
		       timestamp, stop_loss, volatility, trust_stage, model_age_sec, model_used,
		       class_probs, label_horizon_min, created_at
		FROM direction_predictions
		WHERE created_at >= now() - INTERVAL %d HOUR
	`, hours)

	var (
		rows *sql.Rows
		qerr error
	)

	if symbolFilter != "" {
		query := baseQuery + fmt.Sprintf(" AND symbol = ? ORDER BY created_at DESC LIMIT %d", limit)
		rows, qerr = ae.db.Query(query, symbolFilter)
	} else {
		query := baseQuery + fmt.Sprintf(" ORDER BY created_at DESC LIMIT %d", limit)
		rows, qerr = ae.db.Query(query)
	}
	if qerr != nil {
		log.Printf("handleRecentSignals query error: %v", qerr)
		c.JSON(http.StatusOK, gin.H{
			"signals": result,
			"count":   0,
			"errors":  []string{qerr.Error()},
		})
		return
	}
	defer rows.Close()

	for rows.Next() {
		var (
			symbol          string
			direction       string
			confidence      float64
			priceTarget     sql.NullFloat64
			currentPrice    sql.NullFloat64
			timeHorizon     sql.NullInt64
			timestamp       time.Time
			stopLoss        sql.NullFloat64
			volatility      sql.NullFloat64
			trustStage      sql.NullString
			modelAgeSec     sql.NullInt64
			modelUsed       sql.NullString
			classProbsRaw   sql.NullString
			labelHorizonMin sql.NullInt64
			createdAt       time.Time
		)

		if scanErr := rows.Scan(
			&symbol, &direction, &confidence, &priceTarget, &currentPrice, &timeHorizon,
			&timestamp, &stopLoss, &volatility, &trustStage, &modelAgeSec, &modelUsed,
			&classProbsRaw, &labelHorizonMin, &createdAt,
		); scanErr != nil {
			continue
		}

		classProbs := gin.H{"down": 0.0, "sideways": 0.0, "up": 0.0}
		if classProbsRaw.Valid && strings.TrimSpace(classProbsRaw.String) != "" {
			var parsed map[string]float64
			if err := json.Unmarshal([]byte(classProbsRaw.String), &parsed); err == nil {
				if v, ok := parsed["down"]; ok {
					classProbs["down"] = v
				}
				if v, ok := parsed["sideways"]; ok {
					classProbs["sideways"] = v
				}
				if v, ok := parsed["up"]; ok {
					classProbs["up"] = v
				}
			}
		}

		ts := timestamp.Unix()
		if ts == 0 {
			ts = createdAt.Unix()
		}

		result = append(result, gin.H{
			"symbol":            symbol,
			"direction":         strings.ToUpper(direction),
			"confidence":        confidence,
			"price_target":      nullFloat(priceTarget),
			"current_price":     nullFloat(currentPrice),
			"time_horizon":      int(nullInt(timeHorizon)),
			"label_horizon_min": int(nullInt(labelHorizonMin)),
			"timestamp":         ts,
			"stop_loss":         nullFloat(stopLoss),
			"volatility":        nullFloat(volatility),
			"trust_stage": func() string {
				if trustStage.Valid && trustStage.String != "" {
					return trustStage.String
				}
				return "cold_start"
			}(),
			"model_age_sec": int64(nullInt(modelAgeSec)),
			"model_used": func() string {
				if modelUsed.Valid && modelUsed.String != "" {
					return modelUsed.String
				}
				return "SimpleNN"
			}(),
			"class_probs": classProbs,
			"created_at":  createdAt.Unix(),
		})
	}

	c.JSON(http.StatusOK, gin.H{
		"signals": result,
		"count":   len(result),
	})
}

// handleSignalsHistory handles GET /api/v1/ml/signals/history
// Returns historical signals with pagination and filtering support
func (ae *AnalyticsEngine) handleSignalsHistory(c *gin.Context) {
	limit, err := strconv.Atoi(c.DefaultQuery("limit", "100"))
	if err != nil || limit <= 0 || limit > 500 {
		limit = 100
	}

	hours, err := strconv.Atoi(c.DefaultQuery("hours", "168"))
	if err != nil || hours <= 0 || hours > 720 {
		hours = 168 // 7 days default
	}

	symbolFilter := strings.ToUpper(strings.TrimSpace(c.Query("symbol")))
	directionFilter := strings.ToUpper(strings.TrimSpace(c.Query("direction")))

	result := []gin.H{}

	if ae.db == nil {
		c.JSON(http.StatusOK, gin.H{
			"signals": result,
			"count":   0,
		})
		return
	}

	var rows *sql.Rows
	var qerr error

	// Build WHERE clause with filters
	whereClauses := []string{fmt.Sprintf("created_at >= now() - INTERVAL %d HOUR", hours)}
	args := []interface{}{}

	if symbolFilter != "" {
		whereClauses = append(whereClauses, "symbol = ?")
		args = append(args, symbolFilter)
	}

	if directionFilter != "" && directionFilter != "ALL" {
		whereClauses = append(whereClauses, "direction = ?")
		args = append(args, directionFilter)
	}

	whereClause := strings.Join(whereClauses, " AND ")
	query := fmt.Sprintf(`
		SELECT symbol, direction, confidence, price_target, current_price, time_horizon,
		       timestamp, stop_loss, volatility, trust_stage, model_age_sec, model_used,
		       class_probs, label_horizon_min, created_at, actual_direction, actual_price, accuracy_score
		FROM direction_predictions
		WHERE %s
		ORDER BY created_at DESC
		LIMIT %d
	`, whereClause, limit)

	rows, qerr = ae.db.Query(query, args...)
	if qerr != nil {
		log.Printf("handleSignalsHistory query error: %v", qerr)
		c.JSON(http.StatusOK, gin.H{
			"signals": result,
			"count":   0,
			"errors":  []string{qerr.Error()},
		})
		return
	}
	defer rows.Close()

	for rows.Next() {
		var (
			symbol          string
			direction       string
			confidence      float64
			priceTarget     sql.NullFloat64
			currentPrice    sql.NullFloat64
			timeHorizon     sql.NullInt64
			timestamp       time.Time
			stopLoss        sql.NullFloat64
			volatility      sql.NullFloat64
			trustStage      sql.NullString
			modelAgeSec     sql.NullInt64
			modelUsed       sql.NullString
			classProbsRaw   sql.NullString
			labelHorizonMin sql.NullInt64
			createdAt       time.Time
			actualDir       sql.NullString
			actualPrice     sql.NullFloat64
			accuracyScore   sql.NullFloat64
		)

		if scanErr := rows.Scan(
			&symbol, &direction, &confidence, &priceTarget, &currentPrice, &timeHorizon,
			&timestamp, &stopLoss, &volatility, &trustStage, &modelAgeSec, &modelUsed,
			&classProbsRaw, &labelHorizonMin, &createdAt, &actualDir, &actualPrice, &accuracyScore,
		); scanErr != nil {
			log.Printf("handleSignalsHistory scan error: %v", scanErr)
			continue
		}

		classProbs := gin.H{"down": 0.0, "sideways": 0.0, "up": 0.0}
		if classProbsRaw.Valid && strings.TrimSpace(classProbsRaw.String) != "" {
			var parsed map[string]float64
			if err := json.Unmarshal([]byte(classProbsRaw.String), &parsed); err == nil {
				if v, ok := parsed["down"]; ok {
					classProbs["down"] = v
				}
				if v, ok := parsed["sideways"]; ok {
					classProbs["sideways"] = v
				}
				if v, ok := parsed["up"]; ok {
					classProbs["up"] = v
				}
			}
		}

		ts := timestamp.Unix()
		if ts == 0 {
			ts = createdAt.Unix()
		}

		signal := gin.H{
			"symbol":            symbol,
			"direction":         strings.ToUpper(direction),
			"confidence":        confidence,
			"price_target":      nullFloat(priceTarget),
			"current_price":     nullFloat(currentPrice),
			"time_horizon":      int(nullInt(timeHorizon)),
			"label_horizon_min": int(nullInt(labelHorizonMin)),
			"timestamp":         ts,
			"stop_loss":         nullFloat(stopLoss),
			"volatility":        nullFloat(volatility),
			"trust_stage": func() string {
				if trustStage.Valid && trustStage.String != "" {
					return trustStage.String
				}
				return "cold_start"
			}(),
			"model_age_sec": int64(nullInt(modelAgeSec)),
			"model_used": func() string {
				if modelUsed.Valid && modelUsed.String != "" {
					return modelUsed.String
				}
				return "SimpleNN"
			}(),
			"class_probs": classProbs,
			"created_at":  createdAt.Format(time.RFC3339),
		}

		// Add optional fields if they exist
		if actualDir.Valid && actualDir.String != "" {
			signal["actual_direction"] = actualDir.String
		}
		if actualPrice.Valid {
			signal["actual_price"] = actualPrice.Float64
		}
		if accuracyScore.Valid {
			signal["accuracy_score"] = accuracyScore.Float64
		}

		result = append(result, signal)
	}

	c.JSON(http.StatusOK, gin.H{
		"signals": result,
		"count":   len(result),
	})
}

func nullFloat(v sql.NullFloat64) float64 {
	if v.Valid {
		return v.Float64
	}
	return 0
}

func nullInt(v sql.NullInt64) int64 {
	if v.Valid {
		return v.Int64
	}
	return 0
}

// monitorCandleStream monitors the candleStream channel usage and logs statistics
func (ae *AnalyticsEngine) monitorCandleStream() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			if ae.candleStream != nil {
				usage := float64(len(ae.candleStream)) / float64(cap(ae.candleStream)) * 100
				log.Printf("📊 ML candle stream usage: %.1f%% (len=%d, cap=%d)",
					usage, len(ae.candleStream), cap(ae.candleStream))
			}
		case <-ae.ctx.Done():
			log.Println("Candle stream monitoring stopping due to context cancellation")
			return
		}
	}
}

// monitorMemoryUsage monitors memory usage and triggers cleanup when needed
func (ae *AnalyticsEngine) monitorMemoryUsage() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			var memStats runtime.MemStats
			runtime.ReadMemStats(&memStats)
			
			// Log memory stats
			allocMB := float64(memStats.Alloc) / 1024 / 1024
			sysMB := float64(memStats.Sys) / 1024 / 1024
			numGC := memStats.NumGC
			
			log.Printf("📊 Memory: Alloc=%.1fMB, Sys=%.1fMB, NumGC=%d, Objects=%d",
				allocMB, sysMB, numGC, memStats.HeapObjects)
			
			// Get map sizes
			ae.candleHistoryMu.RLock()
			candleHistorySize := len(ae.candleHistory)
			ae.candleHistoryMu.RUnlock()
			
			ae.mlMu.RLock()
			pendingSize := 0
			for _, v := range ae.pendingExamples {
				pendingSize += len(v)
			}
			trainingSize := 0
			for _, v := range ae.trainingData {
				trainingSize += len(v)
			}
			ae.mlMu.RUnlock()
			
			log.Printf("📊 Data structures: candleHistory=%d symbols, pending=%d examples, training=%d examples",
				candleHistorySize, pendingSize, trainingSize)
			
			// FIX: Trigger GC if memory is high
			if allocMB > 400 { // If using more than 400MB
				log.Printf("⚠️ High memory usage (%.1fMB), triggering GC", allocMB)
				runtime.GC()
				
				// Force cleanup of old pending examples
				ae.cleanupOldPendingExamples()
			}
			
		case <-ae.ctx.Done():
			log.Println("Memory monitoring stopping due to context cancellation")
			return
		}
	}
}

// cleanupOldPendingExamples removes pending examples older than 2 hours
func (ae *AnalyticsEngine) cleanupOldPendingExamples() {
	ae.mlMu.Lock()
	defer ae.mlMu.Unlock()
	
	cutoff := time.Now().Add(-2 * time.Hour)
	cleaned := 0
	
	for symbol, pending := range ae.pendingExamples {
		var kept []PendingExample
		for _, p := range pending {
			if p.Timestamp.After(cutoff) {
				kept = append(kept, p)
			} else {
				cleaned++
			}
		}
		ae.pendingExamples[symbol] = kept
	}
	
	if cleaned > 0 {
		log.Printf("🗑️ Cleaned up %d old pending examples", cleaned)
	}
}

// handleModelAnalyses returns aggregated per-model stats for a symbol from model_analyses table
func (ae *AnalyticsEngine) handleModelAnalyses(c *gin.Context) {
	symbol := c.Param("symbol")
	if symbol == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "symbol required"})
		return
	}

	limitStr := c.Query("limit")
	limit := 200
	if limitStr != "" {
		if v, err := strconv.Atoi(limitStr); err == nil && v > 0 && v <= 1000 {
			limit = v
		}
	}

	// Query recent model_analyses rows with nil check
	if ae.db == nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "database not available"})
		return
	}

	query := fmt.Sprintf(`SELECT model_name, prediction, confidence, payload, created_at FROM model_analyses WHERE symbol=? ORDER BY created_at DESC LIMIT %d`, limit)
	rows, err := ae.db.Query(query, symbol)
	if err != nil {
		log.Printf("model analyses db error: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "db error"})
		return
	}
	defer rows.Close()

	type rec struct {
		ModelName  string
		Prediction sql.NullString
		Confidence sql.NullFloat64
		Payload    []byte
		CreatedAt  time.Time
	}

	var recs []rec
	for rows.Next() {
		var r rec
		if err := rows.Scan(&r.ModelName, &r.Prediction, &r.Confidence, &r.Payload, &r.CreatedAt); err == nil {
			recs = append(recs, r)
		}
	}

	// Aggregate per-model
	agg := map[string]map[string]interface{}{}
	for _, r := range recs {
		m := r.ModelName
		if agg[m] == nil {
			agg[m] = map[string]interface{}{"count": 0, "avg_confidence": 0.0, "last_prediction": "", "confidences": []float64{}}
		}
		a := agg[m]
		aCount := a["count"].(int)
		a["count"] = aCount + 1
		if r.Confidence.Valid {
			avg := a["avg_confidence"].(float64)
			avg = (avg*float64(aCount) + r.Confidence.Float64) / float64(aCount+1)
			a["avg_confidence"] = avg
			// append confidences
			a["confidences"] = append(a["confidences"].([]float64), r.Confidence.Float64)
		}
		if r.Prediction.Valid && a["last_prediction"] == "" {
			a["last_prediction"] = r.Prediction.String
		}
	}

	c.JSON(http.StatusOK, gin.H{"symbol": symbol, "limit": limit, "models": agg})
}

func (ae *AnalyticsEngine) handleModelRetrain(c *gin.Context) {
	symbol := strings.ToUpper(strings.TrimSpace(c.Query("symbol")))
	updated, err := ae.forceRetrain(symbol)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	scope := "all_symbols"
	if symbol != "" {
		scope = symbol
	}
	c.JSON(http.StatusOK, gin.H{
		"status":         "ok",
		"scope":          scope,
		"models_updated": updated,
		"updated_at":     time.Now().UTC(),
	})
}

// handleGetMLMetrics handles GET /api/v1/ml/metrics
func (ae *AnalyticsEngine) handleGetMLMetrics(c *gin.Context) {
	metrics := ae.GetDetailedMLMetrics()
	c.JSON(http.StatusOK, metrics)
}

// handleGetMLLearningProgress handles GET /api/v1/ml/learning-progress
func (ae *AnalyticsEngine) handleGetMLLearningProgress(c *gin.Context) {
	progress := ae.GetMLLearningProgress()
	c.JSON(http.StatusOK, progress)
}

// handleGetCalibrationStatus handles GET /api/v1/ml/calibration
func (ae *AnalyticsEngine) handleGetCalibrationStatus(c *gin.Context) {
	status := ae.GetCalibrationStatus()
	c.JSON(http.StatusOK, status)
}

// handleStartAutoCalibration handles POST /api/v1/ml/calibration/start
// Implements REAL calibration: calculates optimal emit thresholds based on historical accuracy
func (ae *AnalyticsEngine) handleStartAutoCalibration(c *gin.Context) {
	log.Printf("🔧 Starting auto-calibration process...")

	if ae.db == nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "database not available"})
		return
	}

	jobID := "cal_" + time.Now().Format("20060102150405")
	calibratedCount := 0
	errors := []string{}

	// Get all symbols with labeled predictions
	rows, err := ae.db.Query(`
		SELECT DISTINCT symbol FROM direction_predictions 
		WHERE actual_direction != '' AND actual_direction IS NOT NULL
	`)
	if err != nil {
		log.Printf("Calibration query error: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	defer rows.Close()

	var symbols []string
	for rows.Next() {
		var symbol string
		if err := rows.Scan(&symbol); err == nil {
			symbols = append(symbols, symbol)
		}
	}

	if len(symbols) == 0 {
		// No labeled data yet - use default thresholds
		symbols = []string{"BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"}
		log.Printf("⚠️  No labeled predictions found, using default thresholds for %d symbols", len(symbols))
	}

	// Calculate optimal threshold for each symbol
	for _, symbol := range symbols {
		// Query all predictions with accuracy for this symbol
		predRows, err := ae.db.Query(`
			SELECT confidence, accuracy_score 
			FROM direction_predictions 
			WHERE symbol = ? AND accuracy_score IS NOT NULL
			ORDER BY created_at DESC
			LIMIT 500
		`, symbol)
		if err != nil {
			errors = append(errors, fmt.Sprintf("%s: %v", symbol, err))
			continue
		}

		var confidences []float64
		var accuracies []float64
		for predRows.Next() {
			var conf, acc float64
			if err := predRows.Scan(&conf, &acc); err == nil {
				confidences = append(confidences, conf)
				accuracies = append(accuracies, acc)
			}
		}
		predRows.Close()

		// Find optimal threshold that maximizes accuracy while maintaining signal rate
		optimalThreshold := 0.50 // Default
		bestScore := 0.0

		for threshold := 0.30; threshold <= 0.80; threshold += 0.05 {
			// Calculate metrics at this threshold
			signalCount := 0
			correctCount := 0
			for i, conf := range confidences {
				if conf >= threshold {
					signalCount++
					if accuracies[i] > 0.5 {
						correctCount++
					}
				}
			}

			if signalCount < 10 {
				continue // Not enough signals at this threshold
			}

			accuracy := float64(correctCount) / float64(signalCount)
			signalRate := float64(signalCount) / float64(len(confidences))

			// Score = accuracy * sqrt(signalRate) to balance both
			score := accuracy * math.Sqrt(signalRate)

			if score > bestScore {
				bestScore = score
				optimalThreshold = threshold
			}
		}

		// If no labeled data, use conservative default
		if len(confidences) == 0 {
			optimalThreshold = 0.50
		}

		// Save calibration to database
		_, err = ae.db.Exec(`
			INSERT INTO model_calibration (symbol, emit_threshold, accuracy_at_threshold, signals_at_threshold, updated_at)
			VALUES (?, ?, ?, ?, now())
			ON DUPLICATE KEY UPDATE 
				emit_threshold = VALUES(emit_threshold),
				accuracy_at_threshold = VALUES(accuracy_at_threshold),
				signals_at_threshold = VALUES(signals_at_threshold),
				updated_at = now()
		`, symbol, optimalThreshold, bestScore, len(confidences))
		if err != nil {
			// Try without ON DUPLICATE KEY for ClickHouse
			_, err = ae.db.Exec(`
				INSERT INTO model_calibration (symbol, emit_threshold, updated_at)
				VALUES (?, ?, now())
			`, symbol, optimalThreshold)
			if err != nil {
				errors = append(errors, fmt.Sprintf("%s: save error: %v", symbol, err))
				continue
			}
		}

		// Update in-memory threshold
		ae.mlMu.Lock()
		if ae.symbolNeutralThresholds == nil {
			ae.symbolNeutralThresholds = make(map[string]float64)
		}
		ae.symbolNeutralThresholds[symbol] = optimalThreshold
		ae.mlMu.Unlock()

		calibratedCount++
		log.Printf("✅ Calibrated %s: threshold=%.2f (based on %d predictions)", 
			symbol, optimalThreshold, len(confidences))
	}

	response := gin.H{
		"status":          "success",
		"message":         fmt.Sprintf("Calibrated %d symbols", calibratedCount),
		"job_id":          jobID,
		"calibrated":      calibratedCount,
		"total":           len(symbols),
		"calibrated_at":   time.Now().Unix(),
	}
	if len(errors) > 0 {
		response["errors"] = errors
	}

	log.Printf("✅ Auto-calibration completed: %d/%d symbols", calibratedCount, len(symbols))
	c.JSON(http.StatusOK, response)
}

// handleTraderMind returns an aggregated trader mind analysis for a symbol
func (ae *AnalyticsEngine) handleTraderMind(c *gin.Context) {
	symbol := c.Param("symbol")
	if symbol == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "symbol required"})
		return
	}

	// Fetch latest N predictions for symbol with nil check
	if ae.db == nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "database not available"})
		return
	}

	rows, err := ae.db.Query(`SELECT direction, confidence, price_target, current_price, created_at FROM direction_predictions WHERE symbol=? ORDER BY created_at DESC LIMIT 50`, symbol)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "db error"})
		return
	}
	defer rows.Close()

	type rec struct {
		Direction    string
		Confidence   float64
		PriceTarget  sql.NullFloat64
		CurrentPrice sql.NullFloat64
		CreatedAt    time.Time
	}

	var recs []rec
	for rows.Next() {
		var r rec
		if err := rows.Scan(&r.Direction, &r.Confidence, &r.PriceTarget, &r.CurrentPrice, &r.CreatedAt); err == nil {
			recs = append(recs, r)
		}
	}

	// Simple ensemble: count votes and average confidence
	counts := map[string]int{"UP": 0, "DOWN": 0, "SIDEWAYS": 0}
	var sumConf float64
	for _, r := range recs {
		counts[r.Direction]++
		sumConf += r.Confidence
	}

	total := len(recs)
	avgConf := 0.0
	if total > 0 {
		avgConf = sumConf / float64(total)
	}

	// Decide final action
	action := "HOLD"
	if counts["UP"] > counts["DOWN"] && counts["UP"] >= counts["SIDEWAYS"] {
		action = "ENTER_LONG"
	} else if counts["DOWN"] > counts["UP"] && counts["DOWN"] >= counts["SIDEWAYS"] {
		action = "ENTER_SHORT"
	}

	resp := gin.H{
		"symbol":         symbol,
		"samples":        total,
		"vote_counts":    counts,
		"avg_confidence": avgConf,
		"final_decision": gin.H{
			"action":     action,
			"confidence": avgConf,
		},
		"updated_at": time.Now(),
	}

	c.JSON(http.StatusOK, resp)
}

// handleTraderMindFull returns a detailed Trader Mind payload: summary + per-model analyses + recent events + simple risk
func (ae *AnalyticsEngine) handleTraderMindFull(c *gin.Context) {
	symbol := c.Param("symbol")
	if symbol == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "symbol required"})
		return
	}

	// Summary (reuse existing logic: fetch latest N direction_predictions) with nil check
	if ae.db == nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "database not available"})
		return
	}

	rows, err := ae.db.Query(`SELECT direction, confidence, price_target, current_price, created_at FROM direction_predictions WHERE symbol=? ORDER BY created_at DESC LIMIT 100`, symbol)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "db error"})
		return
	}
	defer rows.Close()

	type rec struct {
		Direction    string
		Confidence   float64
		PriceTarget  sql.NullFloat64
		CurrentPrice sql.NullFloat64
		CreatedAt    time.Time
	}

	var recs []rec
	for rows.Next() {
		var r rec
		if err := rows.Scan(&r.Direction, &r.Confidence, &r.PriceTarget, &r.CurrentPrice, &r.CreatedAt); err == nil {
			recs = append(recs, r)
		}
	}

	counts := map[string]int{"UP": 0, "DOWN": 0, "SIDEWAYS": 0}
	var sumConf float64
	for _, r := range recs {
		counts[r.Direction]++
		sumConf += r.Confidence
	}
	total := len(recs)
	avgConf := 0.0
	if total > 0 {
		avgConf = sumConf / float64(total)
	}
	action := "HOLD"
	if counts["UP"] > counts["DOWN"] && counts["UP"] >= counts["SIDEWAYS"] {
		action = "ENTER_LONG"
	} else if counts["DOWN"] > counts["UP"] && counts["DOWN"] >= counts["SIDEWAYS"] {
		action = "ENTER_SHORT"
	}

	// Per-model aggregation from model_analyses with nil check
	var mrows *sql.Rows
	if ae.db != nil {
		mrows, err = ae.db.Query(`SELECT model_name, prediction, confidence, payload, created_at FROM model_analyses WHERE symbol=? ORDER BY created_at DESC LIMIT 200`, symbol)
		if err != nil {
			log.Printf("model analyses db error: %v", err)
		}
	}
	defer func() {
		if mrows != nil {
			mrows.Close()
		}
	}()

	type mrec struct {
		ModelName  string
		Prediction sql.NullString
		Confidence sql.NullFloat64
		Payload    []byte
		CreatedAt  time.Time
	}
	var mlist []mrec
	agg := map[string]map[string]interface{}{}
	if mrows != nil {
		for mrows.Next() {
			var m mrec
			if err := mrows.Scan(&m.ModelName, &m.Prediction, &m.Confidence, &m.Payload, &m.CreatedAt); err == nil {
				mlist = append(mlist, m)
				if agg[m.ModelName] == nil {
					agg[m.ModelName] = map[string]interface{}{"count": 0, "avg_confidence": 0.0, "last_prediction": "", "confidences": []float64{}}
				}
				a := agg[m.ModelName]
				ccount := a["count"].(int)
				a["count"] = ccount + 1
				if m.Confidence.Valid {
					avg := a["avg_confidence"].(float64)
					avg = (avg*float64(ccount) + m.Confidence.Float64) / float64(ccount+1)
					a["avg_confidence"] = avg
					a["confidences"] = append(a["confidences"].([]float64), m.Confidence.Float64)
				}
				if m.Prediction.Valid && a["last_prediction"] == "" {
					a["last_prediction"] = m.Prediction.String
				}
			}
		}
	}

	// Simple risk estimate: estimate volatility from recent price targets/current prices
	vol := 0.0
	samples := 0
	for _, r := range recs {
		if r.PriceTarget.Valid && r.CurrentPrice.Valid && r.CurrentPrice.Float64 > 0 {
			diff := math.Abs(r.PriceTarget.Float64-r.CurrentPrice.Float64) / r.CurrentPrice.Float64
			vol += diff
			samples++
		}
	}
	if samples > 0 {
		vol = vol / float64(samples)
	}

	resp := gin.H{
		"symbol":              symbol,
		"summary":             gin.H{"samples": total, "vote_counts": counts, "avg_confidence": avgConf, "final_decision": gin.H{"action": action, "confidence": avgConf}},
		"model_aggregations":  agg,
		"recent_model_events": mlist,
		"risk_estimate":       gin.H{"volatility": vol},
		"updated_at":          time.Now(),
	}

	c.JSON(http.StatusOK, resp)
}

func (ae *AnalyticsEngine) ProcessCandle(candle Candle) {
	// Store candle in history with mutex protection
	ae.candleHistoryMu.Lock()
	minCandlesRequired := ae.getMinCandlesRequired()
	if ae.candleHistory[candle.Symbol] == nil {
		// Pre-allocate with capacity to avoid repeated allocations
		ae.candleHistory[candle.Symbol] = make([]Candle, 0, 1440)
	}

	// Add new candle to history
	ae.candleHistory[candle.Symbol] = append(ae.candleHistory[candle.Symbol], candle)

	// Update symbol access time
	ae.symbolAccessTime[candle.Symbol] = time.Now()

	// Keep only the required number of candles (up to 1440 = 24 hours)
	// FIX: Use proper slice re-slice to allow GC to reclaim memory
	if len(ae.candleHistory[candle.Symbol]) > 1440 {
		oldSlice := ae.candleHistory[candle.Symbol]
		// Create new slice to allow GC to reclaim old memory
		newSlice := make([]Candle, 1440)
		copy(newSlice, oldSlice[len(oldSlice)-1440:])
		ae.candleHistory[candle.Symbol] = newSlice
	}

	// Limit the total number of symbols to prevent unbounded growth
	const maxSymbols = 20 // Reduced from 50 to save memory
	if len(ae.candleHistory) > maxSymbols {
		// Remove the least recently used symbols
		ae.removeLeastRecentlyUsedSymbols(maxSymbols)
	}
	ae.candleHistoryMu.Unlock()

	// Run volume anomaly detector (best-effort) with nil check
	// Commented out - anomalyDetector not defined
	// if ae.anomalyDetector != nil {
	// 	go ae.anomalyDetector.ProcessCandle(candle)
	// }

	// 📡 Send candle to LSTM ML integration for real-time processing with nil check
	if ae.candleStream != nil {
		// Monitor channel usage and log warnings when it's getting full
		// This helps in identifying performance issues before data loss occurs
		if len(ae.candleStream) > int(float64(cap(ae.candleStream))*0.8) {
			log.Printf("⚠️ ML candle stream is 80%% full for %s (len=%d, cap=%d)",
				candle.Symbol, len(ae.candleStream), cap(ae.candleStream))
		}

		// Use a non-blocking send with context cancellation to prevent data loss
		// while still allowing graceful shutdown
		select {
		case ae.candleStream <- candle:
			// Successfully sent to ML pipeline
		case <-ae.ctx.Done():
			// Context cancelled, don't block or drop data during shutdown
			log.Printf("⚠️ Context cancelled, skipping candle send for %s during shutdown", candle.Symbol)
		default:
			// Channel is full, log warning but don't block the main processing loop
			// This prevents data loss by not blocking the main processing thread
			log.Printf("⚠️ ML candle stream full, skipping candle for %s (len=%d, cap=%d)",
				candle.Symbol, len(ae.candleStream), cap(ae.candleStream))
		}
	}

	// 💾 Save new candle to database for persistence with nil check
	if ae.db != nil {
		saveHistoricalDataToDB(ae.db, candle.Symbol, []Candle{candle})
	}

	// 🧠 VERIFY PREVIOUS PREDICTIONS using Advanced ML Engine for continuous learning with nil check
	// Commented out - advancedMLEngine not defined
	// if ae.advancedMLEngine != nil {
	// 	ae.advancedMLEngine.VerifyPrediction(candle.Symbol, candle.Close, time.Now().Add(-65*time.Minute))
	// }
	// Check if we have enough data for ML prediction
	ae.candleHistoryMu.RLock()
	historyLength := len(ae.candleHistory[candle.Symbol])
	ae.candleHistoryMu.RUnlock()
	minCandlesRequired = ae.getMinCandlesRequired()
	if historyLength < minCandlesRequired {
		log.Printf("collecting history: symbol=%s count=%d required=%d", candle.Symbol, historyLength, minCandlesRequired)
		return
	}

	ae.mlMu.Lock()
	if _, exists := ae.historyReadyAt[candle.Symbol]; !exists {
		ae.historyReadyAt[candle.Symbol] = time.Now().UTC()
		log.Printf("history_ready: symbol=%s candles=%d threshold=%d", candle.Symbol, historyLength, minCandlesRequired)
	}
	ae.mlMu.Unlock()

	ae.processOnlineLearning(candle)

	const noSignalWarnInterval = 10 * time.Minute
	ae.mlMu.Lock()
	readyAt := ae.historyReadyAt[candle.Symbol]
	emitted := ae.signalsEmittedBySymbol[candle.Symbol]
	lastWarn := ae.lastZeroSignalWarnAt[candle.Symbol]
	if !readyAt.IsZero() && emitted == 0 && time.Since(readyAt) >= noSignalWarnInterval && time.Since(lastWarn) >= noSignalWarnInterval {
		ae.lastZeroSignalWarnAt[candle.Symbol] = time.Now().UTC()
		log.Printf("WARN no signals emitted yet: symbol=%s ready_since=%s reason=confidence_or_training_gate", candle.Symbol, readyAt.Format(time.RFC3339))
	}
	ae.mlMu.Unlock()

	// Generate enhanced ML prediction with nil check
	// Commented out - advancedMLEngine not defined
	// if ae.advancedMLEngine != nil {
	// 	ae.generateMLPrediction(candle.Symbol)
	// }

	// Also process with traditional analyzer for comparison with nil check
	// Commented out - directionAnalyzer not defined
	// if ae.directionAnalyzer != nil {
	// 	ae.directionAnalyzer.ProcessCandle(candle)
	// }
}

// removeLeastRecentlyUsedSymbols removes the least recently used symbols to keep candleHistory within limits
func (ae *AnalyticsEngine) removeLeastRecentlyUsedSymbols(maxSymbols int) {
	// Create a slice of symbol-time pairs for sorting
	type symbolTime struct {
		symbol string
		time   time.Time
	}

	symbolTimes := make([]symbolTime, 0, len(ae.symbolAccessTime))
	for symbol, accessTime := range ae.symbolAccessTime {
		// Only consider symbols that are in candleHistory
		if _, exists := ae.candleHistory[symbol]; exists {
			symbolTimes = append(symbolTimes, symbolTime{symbol: symbol, time: accessTime})
		}
	}

	// Sort by access time (oldest first)
	sort.Slice(symbolTimes, func(i, j int) bool {
		return symbolTimes[i].time.Before(symbolTimes[j].time)
	})

	// Remove the oldest symbols until we're within the limit
	symbolsToRemove := len(symbolTimes) - maxSymbols
	for i := 0; i < symbolsToRemove && i < len(symbolTimes); i++ {
		symbol := symbolTimes[i].symbol
		// Clear the slice to allow GC to reclaim memory
		ae.candleHistory[symbol] = nil
		delete(ae.candleHistory, symbol)
		delete(ae.symbolAccessTime, symbol)
		// Also clean up related maps to prevent leaks
		ae.mlMu.Lock()
		delete(ae.historyReadyAt, symbol)
		delete(ae.pendingExamples, symbol)
		delete(ae.trainingData, symbol)
		delete(ae.lastTrainedAt, symbol)
		delete(ae.symbolNeutralThresholds, symbol)
		ae.mlMu.Unlock()
		log.Printf("🗑️ Removed LRU symbol %s from all maps (accessed: %v)",
			symbol, symbolTimes[i].time)
	}
}

// generateMLPrediction generates HIGH-CONFIDENCE ML signals using HONEST Advanced ML Engine
func (ae *AnalyticsEngine) generateMLPrediction(symbol string) {
	// Use mutex to protect candleHistory access
	ae.candleHistoryMu.RLock()
	candleHistory := ae.candleHistory[symbol]
	// Update symbol access time
	ae.symbolAccessTime[symbol] = time.Now()
	// Create a copy to avoid holding the lock during processing
	candleHistoryCopy := make([]Candle, len(candleHistory))
	copy(candleHistoryCopy, candleHistory)
	ae.candleHistoryMu.RUnlock()

	if len(candleHistoryCopy) < ae.getMinCandlesRequired() {
		return
	}

	// 🧠 PRIMARY: Use Advanced ML Engine for HONEST 70-80% confidence signals with nil check
	// Commented out - advancedMLEngine not defined
	// if ae.advancedMLEngine != nil {
	// 	honestSignal := ae.advancedMLEngine.GenerateSmartPrediction(symbol, candleHistoryCopy)
	// 	if honestSignal != nil && honestSignal.Confidence >= 0.65 {
	// 		// High-confidence honest signal - EMIT IMMEDIATELY
	// 		log.Printf("🎯 HONEST SIGNAL: %s %s (%.1f%% confidence) - Target: %.8f",
	// 			honestSignal.Symbol, honestSignal.Prediction, honestSignal.Confidence*100, honestSignal.PriceTarget)
	// 		ae.emitTradingSignal(honestSignal)
	// 		return
	// 	}
	// }

	// 🤖 FALLBACK: Use traditional ML if Advanced ML doesn't produce high-confidence signal with nil checks
	// Commented out - selfLearningEngine and enhancedFeatures not defined
	// if ae.selfLearningEngine == nil || ae.enhancedFeatures == nil {
	// 	log.Printf("❌ Missing components for traditional ML prediction for %s", symbol)
	// 	return
	// }

	featuresMatrix := ae.convertCandlesToFeatures(candleHistoryCopy)
	if len(featuresMatrix) == 0 {
		log.Printf("❌ Failed to convert candles to features for %s", symbol)
		return
	}

	// Generate traditional ML prediction
	// signal := ae.selfLearningEngine.PredictWithEnsemble(symbol, featuresMatrix) // Commented out - not defined
	// Placeholder - traditional ML prediction not implemented
	// var signal *DirectionSignal = nil
	// if signal == nil {
	// 	log.Printf("⚠️ No backup prediction generated for %s", symbol)
	// 	return
	// }
	// Note: Traditional ML prediction code commented out - selfLearningEngine not implemented
	// When implemented, uncomment the following:
	// signal := ae.selfLearningEngine.PredictWithEnsemble(symbol, featuresMatrix)
	// if signal != nil && signal.Confidence >= 0.5 {
	// 	log.Printf("🤖 BACKUP SIGNAL: %s %s (%.1f%% confidence) - Target: %.8f",
	// 		signal.Symbol, signal.Direction, signal.Confidence*100, signal.PriceTarget)
	// 	ae.emitTradingSignal(signal)
	// } else {
	// 	log.Printf("🚫 %s: All models produced low confidence - No signal emitted", symbol)
	// }
	log.Printf("⚠️ Traditional ML prediction not implemented for %s", symbol)
}

// convertCandlesToFeatures converts candle history to ML feature matrix
func (ae *AnalyticsEngine) convertCandlesToFeatures(candles []Candle) [][]float64 {
	if len(candles) == 0 {
		return nil
	}

	// Create feature matrix: each row is a timestep, each column is a feature
	featuresMatrix := make([][]float64, len(candles))

	// Reuse a single enhanced feature engine and add candles incrementally
	// tempEngine := NewEnhancedFeatureEngine() // Commented out - function not defined
	for i, candle := range candles {
		// Basic OHLCV features
		features := []float64{
			candle.Close,  // 0: Close price
			candle.Open,   // 1: Open price
			candle.High,   // 2: High price
			candle.Low,    // 3: Low price
			candle.Volume, // 4: Volume
			0.0,           // 5: Price range ratio (filled below)
			0.0,           // 6: Price change ratio (filled below)
		}

		// safe math: avoid divide by zero
		if candle.Close != 0 {
			features[5] = (candle.High - candle.Low) / candle.Close
		}
		if candle.Open != 0 {
			features[6] = (candle.Close - candle.Open) / candle.Open
		}

		// Add the candle to the incremental engine (commented out - function not defined)
		// tempEngine.AddCandle(candle)

		// Add enhanced features when engine has enough history (commented out - function not defined)
		// enhancedFeats := tempEngine.CalculateEnhancedFeatures(candle.Symbol)
		// Dummy struct with all required fields set to 0.0 to prevent compilation errors
		// Append a broad set of enhanced features (aiming 50+ features)
		// All enhanced features set to 0.0 as placeholder since enhancedFeatures engine not implemented
		features = append(features,
			0.0, // EMA_12
			0.0, // EMA_26
			0.0, // MACD
			0.0, // MACD_Signal
			0.0, // MACD_Histogram
			0.0, // BB_Upper
			0.0, // BB_Lower
			0.0, // BB_Position
			0.0, // RSI_14
			0.0, // Stochastic_K
			0.0, // Williams_R
			0.0, // CCI
			0.0, // ATR
			0.0, // ADX
			0.0, // Momentum
			0.0, // ROC
			0.0, // VWAP
			0.0, // Volume_Delta
			0.0, // Support_Level
			0.0, // Resistance_Level
			0.0, // Doji_Pattern
			0.0, // Hammer_Pattern
			0.0, // Market_Regime
			0.0, // Trend_Strength
			0.0, // Price_Momentum_5m
			0.0, // Volume_Momentum
			0.0, // Liquidity_Score
		)

		// Ensure each timestep has same length: pad to targetWidth (50)
		targetWidth := 50
		if len(features) < targetWidth {
			pad := make([]float64, targetWidth-len(features))
			features = append(features, pad...)
		} else if len(features) > targetWidth {
			// keep but truncate to targetWidth to keep stable model input
			features = features[:targetWidth]
		}

		featuresMatrix[i] = features
	}

	// Log matrix dimensions for debugging
	if len(featuresMatrix) > 0 {
		log.Printf("🔢 Features matrix generated: timesteps=%d width=%d", len(featuresMatrix), len(featuresMatrix[0]))
	}

	return featuresMatrix
}

// emitTradingSignal emits ML trading signal to Kafka with nil checks
func (ae *AnalyticsEngine) emitTradingSignal(signal *DirectionSignal) {
	// Check if required components are nil
	if signal == nil || ae.kafkaProducer == nil {
		log.Println("Cannot emit trading signal: signal or kafkaProducer is nil")
		return
	}

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
			signal.Symbol, signal.Direction, signal.Confidence*100)
	}
}

// processFeedbackLoop processes feedback data continuously
// Commented out - feedbackChan and onlineLearner not defined
// func (ae *AnalyticsEngine) processFeedbackLoop(ctx context.Context) {
// 	ticker := time.NewTicker(5 * time.Second)
// 	defer ticker.Stop()
//
// 	for {
// 		select {
// 		case <-ctx.Done():
// 			return
// 		case feedback := <-ae.feedbackChan:
// 			// Add nil check for onlineLearner
// 			if ae.onlineLearner != nil {
// 				ae.onlineLearner.ProcessFeedback(feedback)
// 			}
// 		case <-ticker.C:
// 			// Process any pending feedback from database
// 			ae.processPendingFeedback()
// 		}
// 	}
// }

// performanceMonitoringLoop monitors and logs system performance
// nolint:unused // Reserved for future use when performance monitoring is implemented
func (ae *AnalyticsEngine) performanceMonitoringLoop(ctx context.Context) {
	ticker := time.NewTicker(10 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			// Commented out - onlineLearner not defined
			// if ae.onlineLearner != nil {
			// 	metrics := ae.onlineLearner.GetPerformanceMetrics()
			// 	log.Printf("📊 Performance: Accuracy=%.3f, Precision=%.3f, Recall=%.3f, F1=%.3f (Samples=%d)",
			// 		metrics.Accuracy, metrics.Precision, metrics.Recall, metrics.F1Score, metrics.SampleCount)
			//
			// 	// Save performance metrics to database
			// 	ae.savePerformanceMetrics(metrics)
			// }
		}
	}
}

// GetDetailedMLMetrics returns detailed metrics for all ML models
func (ae *AnalyticsEngine) GetDetailedMLMetrics() map[string]interface{} {
	metrics := make(map[string]interface{})
	symbolMetrics := make(map[string]interface{})
	errorsList := []string{}

	totalModels := 0
	healthyModels := 0
	avgConfidence := 0.0

	if ae.db != nil {
		rows, err := ae.db.Query(`
			SELECT symbol, count(), avg(confidence)
			FROM direction_predictions
			WHERE created_at >= now() - INTERVAL 24 HOUR
			GROUP BY symbol
		`)
		if err != nil {
			errorsList = append(errorsList, err.Error())
		} else {
			defer rows.Close()
			for rows.Next() {
				var symbol string
				var count int64
				var avgConf float64
				if scanErr := rows.Scan(&symbol, &count, &avgConf); scanErr != nil {
					continue
				}
				symbolMetrics[symbol] = map[string]interface{}{
					"signals_24h":    count,
					"avg_confidence": avgConf,
					"class_distribution_24h": map[string]int64{
						"up":       0,
						"down":     0,
						"sideways": 0,
						"total":    count,
					},
					"directional_rate_24h": 0.0,
				}
				totalModels++
				avgConfidence += avgConf
				if avgConf >= ae.confidenceThreshold {
					healthyModels++
				}
			}
		}

		distRows, distErr := ae.db.Query(`
			SELECT symbol, direction, count()
			FROM direction_predictions
			WHERE created_at >= now() - INTERVAL 24 HOUR
			GROUP BY symbol, direction
		`)
		if distErr != nil {
			errorsList = append(errorsList, distErr.Error())
		} else {
			defer distRows.Close()
			for distRows.Next() {
				var symbol, direction string
				var count int64
				if scanErr := distRows.Scan(&symbol, &direction, &count); scanErr != nil {
					continue
				}
				raw, ok := symbolMetrics[symbol]
				if !ok {
					continue
				}
				row, ok := raw.(map[string]interface{})
				if !ok {
					continue
				}
				distAny := row["class_distribution_24h"]
				dist, ok := distAny.(map[string]int64)
				if !ok {
					dist = map[string]int64{"up": 0, "down": 0, "sideways": 0, "total": 0}
				}
				switch strings.ToUpper(direction) {
				case "UP":
					dist["up"] += count
				case "DOWN":
					dist["down"] += count
				default:
					dist["sideways"] += count
				}
				if total, ok := row["signals_24h"].(int64); ok && total > 0 {
					row["directional_rate_24h"] = float64(dist["up"]+dist["down"]) / float64(total)
				}
				row["class_distribution_24h"] = dist
				symbolMetrics[symbol] = row
			}
		}

		// Add accuracy metrics for each symbol
		accRows, accErr := ae.db.Query(`
			SELECT symbol, 
			       avg(accuracy_score) as avg_accuracy,
			       count(*) as total_predictions,
			       sum(if(accuracy_score > 0.5, 1, 0)) as correct_predictions
			FROM direction_predictions
			WHERE accuracy_score IS NOT NULL AND created_at >= now() - INTERVAL 24 HOUR
			GROUP BY symbol
		`)
		if accErr != nil {
			errorsList = append(errorsList, accErr.Error())
		} else {
			defer accRows.Close()
			for accRows.Next() {
				var symbol string
				var avgAccuracy float64
				var total, correct int64
				if scanErr := accRows.Scan(&symbol, &avgAccuracy, &total, &correct); scanErr != nil {
					continue
				}
				if raw, ok := symbolMetrics[symbol]; ok {
					if row, ok := raw.(map[string]interface{}); ok {
						row["accuracy_24h"] = avgAccuracy
						row["total_predictions_24h"] = total
						row["correct_predictions_24h"] = correct
						row["accuracy_rate_24h"] = float64(correct) / float64(total)
						symbolMetrics[symbol] = row
					}
				}
			}
		}
	} else {
		errorsList = append(errorsList, "database not available")
	}

	avgAccuracy := 0.0
	if totalModels > 0 {
		avgConfidence /= float64(totalModels)
		avgAccuracy = avgConfidence
	}

	systemHealth := "EMPTY"
	if totalModels > 0 {
		healthRatio := float64(healthyModels) / float64(totalModels)
		systemHealth = "GOOD"
		if healthRatio < 0.5 {
			systemHealth = "WARNING"
		} else if healthRatio < 0.8 {
			systemHealth = "DEGRADED"
		} else if avgConfidence > 0.70 {
			systemHealth = "EXCELLENT"
		}
	}

	dataStatus := "empty"
	if totalModels > 0 {
		dataStatus = "real"
	}

	metrics["system"] = map[string]interface{}{
		"overall_health":     systemHealth,
		"total_models":       totalModels,
		"healthy_models":     healthyModels,
		"average_accuracy":   avgAccuracy,
		"average_confidence": avgConfidence,
		"last_updated":       time.Now().Unix(),
	}
	metrics["symbols"] = symbolMetrics
	metrics["temporal_analysis"] = ae.getTemporalPerformanceMetrics()
	metrics["risk_metrics"] = ae.getRiskMetrics()
	metrics["automation"] = ae.getAutomationStatus()
	metrics["data_status"] = dataStatus
	metrics["errors"] = errorsList

	return metrics
}

// GetMLLearningProgress returns detailed ML learning progress metrics
func (ae *AnalyticsEngine) GetMLLearningProgress() map[string]interface{} {
	progress := make(map[string]interface{})
	symbolProgress := make(map[string]interface{})
	errorsList := []string{}

	totalPredictions := 0
	totalLabeled := 0
	totalPending := 0

	if ae.db != nil {
		// Get overall prediction stats
		var totalPreds, totalLabeledPreds int64
		err := ae.db.QueryRow(`
			SELECT count(), countIf(actual_direction != '' AND actual_direction IS NOT NULL)
			FROM direction_predictions
		`).Scan(&totalPreds, &totalLabeledPreds)
		if err != nil {
			errorsList = append(errorsList, err.Error())
		} else {
			totalPredictions = int(totalPreds)
			totalLabeled = int(totalLabeledPreds)
			totalPending = totalPredictions - totalLabeled
		}

		// Get per-symbol stats
		rows, err := ae.db.Query(`
			SELECT 
				symbol,
				count() as total,
				countIf(actual_direction != '' AND actual_direction IS NOT NULL) as labeled,
				avgIf(accuracy_score, accuracy_score IS NOT NULL) as avg_accuracy,
				avg(confidence) as avg_confidence,
				min(created_at) as first_prediction,
				max(created_at) as last_prediction
			FROM direction_predictions
			GROUP BY symbol
			ORDER BY symbol
		`)
		if err != nil {
			errorsList = append(errorsList, err.Error())
		} else {
			defer rows.Close()
			for rows.Next() {
				var symbol string
				var total, labeled int64
				var avgAccuracy, avgConfidence sql.NullFloat64
				var firstPred, lastPred time.Time

				if err := rows.Scan(&symbol, &total, &labeled, &avgAccuracy, &avgConfidence, &firstPred, &lastPred); err != nil {
					continue
				}

				pending := total - labeled
				accuracyRate := 0.0
				if labeled > 0 && avgAccuracy.Valid {
					accuracyRate = avgAccuracy.Float64
				}
				confidence := 0.0
				if avgConfidence.Valid {
					confidence = avgConfidence.Float64
				}

				// Get model info
				ae.mlMu.RLock()
				model, modelExists := ae.models[symbol]
				var modelAccuracy float64
				var modelPredictions int
				var modelTrained bool
				var trustStage string
				if modelExists {
					modelAccuracy = model.GetAccuracy()
					modelPredictions = model.PredictionCount
					modelTrained = model.Trained
					trustStage = ae.getModelTrustStage(model)
				}
				ae.mlMu.RUnlock()

				// Get training data info
				ae.mlMu.RLock()
				trainingDataCount := 0
				pendingExamplesCount := 0
				if td, exists := ae.trainingData[symbol]; exists {
					trainingDataCount = len(td)
				}
				if pe, exists := ae.pendingExamples[symbol]; exists {
					pendingExamplesCount = len(pe)
				}
				ae.mlMu.RUnlock()

				// Calculate learning status
				status := "COLD_START"
				if modelTrained {
					status = "TRAINED"
				} else if labeled > 20 {
					status = "WARMING"
				} else if labeled > 0 {
					status = "LEARNING"
				}

				symbolProgress[symbol] = map[string]interface{}{
					"status":               status,
					"total_predictions":    total,
					"labeled_predictions":  labeled,
					"pending_predictions":  pending,
					"label_rate":           float64(labeled) / float64(maxInt64(total, 1)),
					"accuracy":             accuracyRate,
					"avg_confidence":       confidence,
					"model_accuracy":       modelAccuracy,
					"model_predictions":    modelPredictions,
					"model_trained":        modelTrained,
					"trust_stage":          trustStage,
					"training_examples":    trainingDataCount,
					"pending_examples":     pendingExamplesCount,
					"first_prediction":     firstPred.Unix(),
					"last_prediction":      lastPred.Unix(),
				}
			}
		}

		// Get labeling job stats
		var recentLabels int64
		err = ae.db.QueryRow(`
			SELECT count()
			FROM direction_predictions
			WHERE actual_direction != '' AND actual_direction IS NOT NULL
			  AND created_at >= now() - INTERVAL 1 HOUR
		`).Scan(&recentLabels)
		if err != nil {
			errorsList = append(errorsList, err.Error())
		}

		// Get training events stats
		var recentTrainings int64
		err = ae.db.QueryRow(`
			SELECT count()
			FROM training_events
			WHERE created_at >= now() - INTERVAL 24 HOUR
		`).Scan(&recentTrainings)
		if err != nil {
			errorsList = append(errorsList, err.Error())
		}

		progress["labeling_stats"] = map[string]interface{}{
			"recent_labels_1h":     recentLabels,
			"recent_trainings_24h": recentTrainings,
		}
	} else {
		errorsList = append(errorsList, "database not available")
	}

	// Get in-memory model stats
	ae.mlMu.RLock()
	totalModels := len(ae.models)
	totalTrainingData := 0
	totalPendingExamples := 0
	for _, td := range ae.trainingData {
		totalTrainingData += len(td)
	}
	for _, pe := range ae.pendingExamples {
		totalPendingExamples += len(pe)
	}
	ae.mlMu.RUnlock()

	progress["system"] = map[string]interface{}{
		"total_predictions":      totalPredictions,
		"total_labeled":          totalLabeled,
		"total_pending":          totalPending,
		"overall_label_rate":     float64(totalLabeled) / float64(maxInt(1, totalPredictions)),
		"total_models":           totalModels,
		"total_training_data":    totalTrainingData,
		"total_pending_examples": totalPendingExamples,
		"last_updated":           time.Now().Unix(),
	}

	progress["symbols"] = symbolProgress
	progress["errors"] = errorsList

	// Add learning timeline estimate
	progress["learning_timeline"] = map[string]interface{}{
		"cold_start_phase":  "0-15 min (waiting for first labels)",
		"warming_phase":     "15-60 min (collecting labeled data)",
		"trained_phase":     "60+ min (55%+ validation accuracy)",
		"calibration_phase": "100+ labeled predictions per symbol",
		"current_estimate":  getLearningPhaseEstimate(totalLabeled),
	}

	return progress
}

func getLearningPhaseEstimate(labeledCount int) string {
	if labeledCount == 0 {
		return "COLD_START - waiting for predictions to mature"
	} else if labeledCount < 20 {
		return "EARLY_LEARNING - collecting initial labels"
	} else if labeledCount < 100 {
		return "WARMING - building training data"
	} else {
		return "MATURE - ready for calibration"
	}
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func maxInt64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

func (ae *AnalyticsEngine) getAutomationStatus() map[string]interface{} {
	status := map[string]interface{}{
		"enabled":                  ae.autopilotEnabled,
		"interval_sec":             int64(ae.autopilotInterval.Seconds()),
		"silence_timeout_sec":      int64(ae.autopilotSilenceTimeout.Seconds()),
		"action_cooldown_sec":      int64(ae.autopilotActionCooldown.Seconds()),
		"min_directional_rate":     ae.autopilotMinDirectional,
		"min_signal_samples":       ae.autopilotMinSignalSamples,
		"reset_min_accuracy":       ae.autopilotResetMinAccuracy,
		"reset_min_predictions":    ae.autopilotResetMinPredictions,
		"last_actions_by_symbol":   map[string]int64{},
		"recent_actions_24h_count": 0,
	}

	ae.mlMu.Lock()
	lastActions := map[string]int64{}
	for symbol, ts := range ae.autopilotLastAction {
		lastActions[symbol] = ts.Unix()
	}
	ae.mlMu.Unlock()
	status["last_actions_by_symbol"] = lastActions

	if ae.db == nil {
		return status
	}

	var actions24h int64
	if err := ae.db.QueryRow(`
		SELECT count()
		FROM automation_events
		WHERE created_at >= now() - INTERVAL 24 HOUR
	`).Scan(&actions24h); err == nil {
		status["recent_actions_24h_count"] = actions24h
	}

	return status
}

// getTemporalPerformanceMetrics returns temporal performance analysis data
func (ae *AnalyticsEngine) getTemporalPerformanceMetrics() map[string]interface{} {
	metrics := make(map[string]interface{})
	hourlyPerformance := make(map[string]float64)
	dailyPerformance := make(map[string]float64)
	errorsList := []string{}

	if ae.db != nil {
		rows, err := ae.db.Query(`
			SELECT toHour(created_at) as hour, avg(accuracy) as avg_accuracy
			FROM model_performance
			WHERE created_at >= now() - INTERVAL 24 HOUR
			GROUP BY toHour(created_at)
			ORDER BY hour
		`)
		if err != nil {
			errorsList = append(errorsList, err.Error())
		} else {
			defer rows.Close()
			for rows.Next() {
				var hour uint8
				var avgAccuracy float64
				if scanErr := rows.Scan(&hour, &avgAccuracy); scanErr == nil {
					hourlyPerformance[fmt.Sprintf("%02d", hour)] = avgAccuracy
				}
			}
		}

		rows, err = ae.db.Query(`
			SELECT toDayOfWeek(created_at) as day_num, avg(accuracy) as avg_accuracy
			FROM model_performance
			WHERE created_at >= now() - INTERVAL 7 DAY
			GROUP BY toDayOfWeek(created_at)
			ORDER BY day_num
		`)
		if err != nil {
			errorsList = append(errorsList, err.Error())
		} else {
			defer rows.Close()
			for rows.Next() {
				var dayNum uint8
				var avgAccuracy float64
				if scanErr := rows.Scan(&dayNum, &avgAccuracy); scanErr == nil {
					dayName := map[uint8]string{
						1: "Monday",
						2: "Tuesday",
						3: "Wednesday",
						4: "Thursday",
						5: "Friday",
						6: "Saturday",
						7: "Sunday",
					}[dayNum]
					if dayName == "" {
						dayName = fmt.Sprintf("day_%d", dayNum)
					}
					dailyPerformance[dayName] = avgAccuracy
				}
			}
		}
	} else {
		errorsList = append(errorsList, "database not available")
	}

	dataStatus := "empty"
	if len(hourlyPerformance) > 0 || len(dailyPerformance) > 0 {
		dataStatus = "real"
	}

	metrics["hourly_performance"] = hourlyPerformance
	metrics["daily_performance"] = dailyPerformance
	metrics["data_status"] = dataStatus
	metrics["errors"] = errorsList
	return metrics
}

// GetInfrastructureMetrics returns infrastructure monitoring data
func (ae *AnalyticsEngine) GetInfrastructureMetrics() map[string]interface{} {
	metrics := make(map[string]interface{})

	// Get real memory stats from runtime
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	// Convert bytes to GB
	usedGB := float64(m.Alloc) / 1024 / 1024 / 1024
	totalGB := float64(m.Sys) / 1024 / 1024 / 1024
	memoryUsagePercent := 0.0
	if totalGB > 0 {
		memoryUsagePercent = (usedGB / totalGB) * 100
	}

	// Get CPU core count (real)
	cpuCores := runtime.NumCPU()

	// Estimate CPU usage based on goroutines and GC activity
	// This is a rough estimate since Go doesn't expose CPU usage directly
	numGoroutines := runtime.NumGoroutine()
	cpuUsage := math.Min(100.0, float64(numGoroutines)*1.5) // Rough estimate

	metrics["cpu"] = map[string]interface{}{
		"usage_percent": cpuUsage,
		"core_count":    cpuCores,
		"measurement":   "estimated",
	}

	metrics["memory"] = map[string]interface{}{
		"usage_percent": memoryUsagePercent,
		"used_gb":       usedGB,
		"total_gb":      totalGB,
		"measurement":   "measured",
	}

	// Get real Kafka metrics
	kafkaHealth := "healthy"
	kafkaLag := int64(0)
	kafkaMessagesPerSec := int64(0)

	if ae.kafkaReader != nil {
		// Check if reader is healthy by attempting to fetch stats
		// Note: kafka-go doesn't directly expose lag, so we estimate
		// In production, you'd query Kafka admin API
		kafkaHealth = "healthy"
		// Estimate lag based on message processing (would need tracking)
		kafkaLag = 0            // Would need to track this separately
		kafkaMessagesPerSec = 0 // Would need to track message rate
	} else {
		kafkaHealth = "unhealthy"
	}

	kafkaMetrics := map[string]interface{}{
		"lag":               kafkaLag,
		"messages_per_sec":  kafkaMessagesPerSec,
		"consumption_rate":  kafkaMessagesPerSec, // Same as messages_per_sec
		"connection_status": kafkaHealth,
		"measurement":       "estimated",
	}

	metrics["kafka"] = kafkaMetrics

	// Database metrics
	dbQueriesPerSec := int64(0)
	dbLatency := 0.0
	activeConnections := 0
	dbHealth := "unhealthy"
	var dbStats sql.DBStats
	if ae.db != nil {
		dbStats = ae.db.Stats()
		activeConnections = dbStats.OpenConnections
		dbHealth = "healthy"
		if err := ae.db.Ping(); err != nil {
			dbHealth = "unhealthy"
		}
		if dbStats.MaxOpenConnections > 0 {
			usageRatio := float64(activeConnections) / float64(dbStats.MaxOpenConnections)
			dbQueriesPerSec = int64(usageRatio * 100)
		}
	}

	dbMetrics := map[string]interface{}{
		"queries_per_sec":    dbQueriesPerSec,
		"response_latency":   dbLatency,
		"active_connections": activeConnections,
		"connection_status":  dbHealth,
		"measurement":        "estimated",
	}

	metrics["database"] = dbMetrics

	// Calculate real uptime
	uptimeSeconds := int64(time.Since(ae.startTime).Seconds())

	// Calculate average latency from DB stats (use DB latency as proxy)
	avgLatency := dbLatency
	if avgLatency == 0 && dbStats.MaxOpenConnections > 0 {
		// Estimate latency based on connection pool usage
		avgLatency = float64(activeConnections) * 2.0 // Rough estimate
	}

	metrics["uptime_seconds"] = uptimeSeconds
	metrics["avg_latency_ms"] = avgLatency

	// Last updated timestamp
	metrics["last_updated"] = time.Now().Unix()

	return metrics
}

// handleGetInfrastructureMetrics handles GET /api/v1/infrastructure/metrics
func (ae *AnalyticsEngine) handleGetInfrastructureMetrics(c *gin.Context) {
	metrics := ae.GetInfrastructureMetrics()
	c.JSON(http.StatusOK, metrics)
}

// getRiskMetrics returns risk analysis metrics
func (ae *AnalyticsEngine) getRiskMetrics() map[string]interface{} {
	metrics := make(map[string]interface{})
	errorsList := []string{}
	dataStatus := "empty"

	if ae.db != nil {
		var valueAtRisk, expectedShortfall, stabilityScore, correlationExposure sql.NullFloat64
		err := ae.db.QueryRow(`
			SELECT
				quantileExact(0.05)(accuracy) as value_at_risk,
				quantileExact(0.01)(accuracy) as expected_shortfall,
				avg(if(accuracy >= 0.7, 100.0, 0.0)) as stability_score,
				stddevPop(accuracy) as correlation_exposure
			FROM model_performance
			WHERE created_at >= now() - INTERVAL 24 HOUR
		`).Scan(&valueAtRisk, &expectedShortfall, &stabilityScore, &correlationExposure)
		if err != nil {
			errorsList = append(errorsList, err.Error())
		} else {
			if valueAtRisk.Valid {
				if !math.IsNaN(valueAtRisk.Float64) && !math.IsInf(valueAtRisk.Float64, 0) {
					metrics["value_at_risk"] = valueAtRisk.Float64
				}
			}
			if expectedShortfall.Valid {
				if !math.IsNaN(expectedShortfall.Float64) && !math.IsInf(expectedShortfall.Float64, 0) {
					metrics["expected_shortfall"] = expectedShortfall.Float64
				}
			}
			if stabilityScore.Valid {
				if !math.IsNaN(stabilityScore.Float64) && !math.IsInf(stabilityScore.Float64, 0) {
					metrics["stability_score"] = stabilityScore.Float64
				}
			}
			if correlationExposure.Valid {
				if !math.IsNaN(correlationExposure.Float64) && !math.IsInf(correlationExposure.Float64, 0) {
					metrics["correlation_exposure"] = correlationExposure.Float64
				}
			}
			if len(metrics) > 0 {
				dataStatus = "real"
			}
		}
	} else {
		errorsList = append(errorsList, "database not available")
	}

	metrics["data_status"] = dataStatus
	metrics["errors"] = errorsList
	return metrics
}

// GetCalibrationStatus returns the current calibration status of all models
func (ae *AnalyticsEngine) GetCalibrationStatus() map[string]interface{} {
	status := make(map[string]interface{})
	models := make(map[string]interface{})
	errorsList := []string{}

	if ae.db == nil {
		status["models"] = models
		status["system"] = map[string]interface{}{
			"overall_status": "EMPTY",
			"completed":      0,
			"total":          0,
		}
		status["data_status"] = "empty"
		status["errors"] = []string{"database not available"}
		return status
	}

	// Get calibration data with accuracy metrics
	rows, err := ae.db.Query(`
		SELECT symbol, emit_threshold, updated_at 
		FROM model_calibration
		ORDER BY symbol
	`)
	if err != nil {
		status["models"] = models
		status["system"] = map[string]interface{}{
			"overall_status": "EMPTY",
			"completed":      0,
			"total":          0,
		}
		status["data_status"] = "empty"
		status["errors"] = []string{err.Error()}
		return status
	}
	defer rows.Close()

	totalModels := 0
	calibratedModels := 0

	for rows.Next() {
		var symbol string
		var emitThreshold float64
		var updatedAt time.Time
		if scanErr := rows.Scan(&symbol, &emitThreshold, &updatedAt); scanErr != nil {
			errorsList = append(errorsList, scanErr.Error())
			continue
		}

		// Get accuracy stats for this symbol
		var avgAccuracy sql.NullFloat64
		var totalSignals sql.NullInt64
		var correctSignals sql.NullInt64
		err = ae.db.QueryRow(`
			SELECT 
				avg(accuracy_score),
				count(*),
				sum(if(accuracy_score > 0.5, 1, 0))
			FROM direction_predictions
			WHERE symbol = ? AND accuracy_score IS NOT NULL
		`, symbol).Scan(&avgAccuracy, &totalSignals, &correctSignals)

		accuracy := 0.0
		signalCount := 0
		if avgAccuracy.Valid {
			accuracy = avgAccuracy.Float64
		}
		if totalSignals.Valid {
			signalCount = int(totalSignals.Int64)
		}

		models[symbol] = map[string]interface{}{
			"emit_threshold": emitThreshold,
			"accuracy":       accuracy,
			"signals_count":  signalCount,
			"status":         "CALIBRATED",
			"last_calibrated": updatedAt.Unix(),
		}
		totalModels++
		calibratedModels++
	}

	// Check for symbols without calibration
	allSymbolsRows, err := ae.db.Query(`
		SELECT DISTINCT symbol FROM direction_predictions
		WHERE created_at >= now() - INTERVAL 7 DAY
	`)
	if err == nil {
		defer allSymbolsRows.Close()
		for allSymbolsRows.Next() {
			var symbol string
			if err := allSymbolsRows.Scan(&symbol); err == nil {
				if _, exists := models[symbol]; !exists {
					// Get basic stats for uncalibrated symbols
					var totalSignals sql.NullInt64
					var labeledSignals sql.NullInt64
					err = ae.db.QueryRow(`
						SELECT count(*), sum(if(accuracy_score IS NOT NULL, 1, 0))
						FROM direction_predictions
						WHERE symbol = ?
					`, symbol).Scan(&totalSignals, &labeledSignals)
					
					total := 0
					labeled := 0
					if totalSignals.Valid {
						total = int(totalSignals.Int64)
					}
					if labeledSignals.Valid {
						labeled = int(labeledSignals.Int64)
					}

					models[symbol] = map[string]interface{}{
						"emit_threshold": 0.50, // Default
						"accuracy":       0.0,
						"signals_count":  total,
						"labeled_count":  labeled,
						"status":         "PENDING_LABELS",
					}
					totalModels++
				}
			}
		}
	}

	overallStatus := "COMPLETE"
	if calibratedModels == 0 {
		overallStatus = "EMPTY"
	} else if calibratedModels < totalModels {
		overallStatus = "PARTIAL"
	}

	status["models"] = models
	status["system"] = map[string]interface{}{
		"overall_status": overallStatus,
		"calibrated":     calibratedModels,
		"total":          totalModels,
	}
	status["data_status"] = "real"
	if len(errorsList) > 0 {
		status["errors"] = errorsList
	}

	return status
}

// checkActualDirection checks if the direction prediction was correct
// nolint:unused // Reserved for future use in model evaluation
func (ae *AnalyticsEngine) checkActualDirection(prediction, actual string) bool {
	return prediction == actual
}

// processPendingFeedback processes any pending feedback from database
// nolint:unused // Reserved for future use when feedback processing is implemented
func (ae *AnalyticsEngine) processPendingFeedback() {
	// Add nil check for db
	if ae.db == nil {
		return
	}

	// This could be used to process manual feedback from admin interface
	// For now, we'll just check for any manual feedback entries
	rows, err := ae.db.Query(`
		SELECT symbol, timestamp, predicted_prob, actual_pump, feedback_type, confidence 
		FROM feedback_data 
		WHERE processed = 0 AND feedback_type = 'manual' 
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
		var actualPumpRaw uint8

		if err := rows.Scan(&symbol, &timestamp, &predictedProb, &actualPumpRaw, &feedbackType, &confidence); err != nil {
			continue
		}
		// Commented out - FeedbackData and onlineLearner not defined
		// feedback := FeedbackData{
		// 	Symbol:        symbol,
		// 	Timestamp:     timestamp,
		// 	PredictedProb: predictedProb,
		// 	ActualPump:    actualPump,
		// 	FeedbackType:  feedbackType,
		// 	Confidence:    confidence,
		// }
		//
		// // Add nil check for onlineLearner
		// if ae.onlineLearner != nil {
		// 	ae.onlineLearner.ProcessFeedback(feedback)
		// }
	}
}

// savePerformanceMetrics saves performance metrics to database
// nolint:unused // Reserved for future use when performance metrics saving is implemented
func (ae *AnalyticsEngine) savePerformanceMetrics(metrics map[string]interface{}) {
	// Add nil check for db
	if ae.db == nil {
		return
	}

	// Extract values from map[string]interface{} with type assertions
	accuracy, _ := metrics["accuracy"].(float64)
	precision, _ := metrics["precision"].(float64)
	recall, _ := metrics["recall"].(float64)
	f1Score, _ := metrics["f1_score"].(float64)
	falsePositives, _ := metrics["false_positives"].(int)
	truePositives, _ := metrics["true_positives"].(int)
	falseNegatives, _ := metrics["false_negatives"].(int)
	trueNegatives, _ := metrics["true_negatives"].(int)
	sampleCount, _ := metrics["sample_count"].(int)

	_, err := ae.db.Exec(`
		INSERT INTO model_performance 
		(model_version, accuracy, precision_score, recall_score, f1_score, 
		 false_positives, true_positives, false_negatives, true_negatives, sample_count)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
	`, ae.modelVersion, accuracy, precision, recall, f1Score,
		falsePositives, truePositives, falseNegatives,
		trueNegatives, sampleCount)

	if err != nil {
		log.Printf("Failed to save performance metrics: %v", err)
	}
}

func (ae *AnalyticsEngine) saveTrainingEvent(symbol string, report TrainingReport, classWeights [3]float64, threshold float64, trustStage string) error {
	if ae.db == nil {
		return nil
	}
	_, err := ae.db.Exec(`
		INSERT INTO training_events
		(symbol, sample_count, train_samples, val_samples, class_down, class_sideways, class_up, val_accuracy, threshold, trust_stage, weight_down, weight_sideways, weight_up)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
	`, symbol, report.SampleCount, report.TrainSamples, report.ValSamples,
		report.ClassDown, report.ClassSideways, report.ClassUp, report.BestValidationAccuracy,
		threshold, trustStage, classWeights[0], classWeights[1], classWeights[2])
	return err
}

func (ae *AnalyticsEngine) saveAutomationEvent(symbol, action, reason string, directionalRate, sidewaysRate, modelAccuracy, threshold float64, trainingExamples int) error {
	if ae.db == nil {
		return nil
	}
	_, err := ae.db.Exec(`
		INSERT INTO automation_events
		(symbol, action, reason, directional_rate, sideways_rate, model_accuracy, threshold, training_examples)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)
	`, symbol, action, reason, directionalRate, sidewaysRate, modelAccuracy, threshold, trainingExamples)
	return err
}

// 🗄️ PERSISTENCE FUNCTIONS FOR SMART MEMORY

// initializePersistentTables creates tables for persistent storage
func initializePersistentTables(db *sql.DB) {
	ddl := []string{
		`CREATE TABLE IF NOT EXISTS candle_cache (
			symbol String,
			timestamp Int64,
			open Float64,
			high Float64,
			low Float64,
			close Float64,
			volume Float64,
			created_at DateTime DEFAULT now()
		) ENGINE = ReplacingMergeTree(created_at)
		PARTITION BY toYYYYMM(toDateTime(timestamp))
		ORDER BY (symbol, timestamp)`,
		`CREATE TABLE IF NOT EXISTS model_weights (
			symbol String,
			weights String,
			biases String,
			hidden_weights String,
			hidden_biases String,
			learning_rate Float64,
			last_accuracy Float64,
			prediction_count Int64,
			correct_count Int64,
			updated_at DateTime DEFAULT now()
		) ENGINE = ReplacingMergeTree(updated_at)
		ORDER BY symbol`,
		`CREATE TABLE IF NOT EXISTS feature_normalizers (
			symbol String,
			params String,
			updated_at DateTime DEFAULT now()
		) ENGINE = ReplacingMergeTree(updated_at)
		ORDER BY symbol`,
		`CREATE TABLE IF NOT EXISTS feedback_data (
			symbol String,
			timestamp DateTime,
			predicted_prob Float64,
			actual_pump UInt8,
			feedback_type String,
			confidence Float64,
			notes String,
			processed UInt8,
			created_at DateTime DEFAULT now()
		) ENGINE = MergeTree
		PARTITION BY toYYYYMM(timestamp)
		ORDER BY (symbol, timestamp, created_at)`,
		`CREATE TABLE IF NOT EXISTS model_performance (
			model_version Int32,
			accuracy Float64,
			precision_score Float64,
			recall_score Float64,
			f1_score Float64,
			false_positives Int64,
			true_positives Int64,
			false_negatives Int64,
			true_negatives Int64,
			sample_count Int64,
			created_at DateTime DEFAULT now()
		) ENGINE = MergeTree
		PARTITION BY toYYYYMM(created_at)
		ORDER BY (created_at, model_version)`,
		`CREATE TABLE IF NOT EXISTS training_events (
			symbol String,
			sample_count Int64,
			train_samples Int64,
			val_samples Int64,
			class_down Int64,
			class_sideways Int64,
			class_up Int64,
			val_accuracy Float64,
			threshold Float64,
			trust_stage String,
			weight_down Float64,
			weight_sideways Float64,
			weight_up Float64,
			created_at DateTime DEFAULT now()
		) ENGINE = MergeTree
		PARTITION BY toYYYYMM(created_at)
		ORDER BY (symbol, created_at)`,
		`CREATE TABLE IF NOT EXISTS model_calibration (
			symbol String,
			emit_threshold Float64,
			updated_at DateTime DEFAULT now()
		) ENGINE = ReplacingMergeTree(updated_at)
		ORDER BY symbol`,
		`CREATE TABLE IF NOT EXISTS automation_events (
			symbol String,
			action String,
			reason String,
			directional_rate Float64,
			sideways_rate Float64,
			model_accuracy Float64,
			threshold Float64,
			training_examples Int64,
			created_at DateTime DEFAULT now()
		) ENGINE = MergeTree
		PARTITION BY toYYYYMM(created_at)
		ORDER BY (symbol, created_at, action)`,
		`CREATE TABLE IF NOT EXISTS model_analyses (
			symbol String,
			model_name String,
			prediction String,
			confidence Float64,
			payload String,
			created_at DateTime DEFAULT now()
		) ENGINE = MergeTree
		PARTITION BY toYYYYMM(created_at)
		ORDER BY (symbol, created_at, model_name)`,
	}

	for _, q := range ddl {
		if _, err := db.Exec(q); err != nil {
			log.Printf("Failed to initialize table: %v", err)
		}
	}
	log.Println("Persistence tables initialized (ClickHouse)")
}

// loadHistoricalDataFromDB loads candle data from database
func loadHistoricalDataFromDB(db *sql.DB, symbols []string) map[string][]Candle {
	if db == nil {
		return make(map[string][]Candle)
	}

	result := make(map[string][]Candle)
	for _, symbol := range symbols {
		rows, err := db.Query(`
			SELECT timestamp, open, high, low, close, volume
			FROM candle_cache
			WHERE symbol = ?
			ORDER BY timestamp ASC
			LIMIT 1440
		`, symbol)
		if err != nil {
			log.Printf("Failed to load candles for %s: %v", symbol, err)
			continue
		}

		var candles []Candle
		for rows.Next() {
			var candle Candle
			if scanErr := rows.Scan(&candle.Timestamp, &candle.Open, &candle.High, &candle.Low, &candle.Close, &candle.Volume); scanErr == nil {
				candle.Symbol = symbol
				candles = append(candles, candle)
			}
		}
		rows.Close()

		if len(candles) > 0 {
			result[symbol] = candles
			log.Printf("Loaded %d cached candles for %s", len(candles), symbol)
		}
	}
	return result
}

// saveHistoricalDataToDB saves candle data to database
func saveHistoricalDataToDB(db *sql.DB, symbol string, candles []Candle) {
	if db == nil || len(candles) == 0 {
		return
	}

	for _, c := range candles {
		_, err := db.Exec(`
			INSERT INTO candle_cache (symbol, timestamp, open, high, low, close, volume)
			VALUES (?, ?, ?, ?, ?, ?, ?)
		`, symbol, c.Timestamp, c.Open, c.High, c.Low, c.Close, c.Volume)
		if err != nil {
			log.Printf("Failed to save candle for %s: %v", symbol, err)
			return
		}
	}
}

// isDataOutdated checks if cached data is too old
// nolint:unused // Reserved for future use when data freshness checking is implemented
func isDataOutdated(candles []Candle) bool {
	if len(candles) == 0 {
		return true
	}
	// If last candle is older than 2 hours, consider outdated
	lastTimestamp := candles[len(candles)-1].Timestamp
	return time.Now().Unix()-lastTimestamp > 2*60*60
}

// labelMaturedPredictions scans direction_predictions and labels those that have matured
// A prediction is matured when label_horizon_min minutes have passed since the prediction timestamp
func (ae *AnalyticsEngine) labelMaturedPredictions() {
	if ae.db == nil {
		log.Printf("⚠️ labelMaturedPredictions: DB connection is nil")
		return
	}

	log.Printf("🔍 Starting labeling pass...")

	// Query predictions that need labeling (actual_direction is NULL/empty)
	// and have matured (current time > prediction timestamp + label_horizon_min)
	rows, err := ae.db.Query(`
		SELECT symbol, timestamp, label_horizon_min, direction, confidence
		FROM direction_predictions
		WHERE (actual_direction = '' OR actual_direction IS NULL)
		  AND toDateTime(timestamp) + INTERVAL toInt32(label_horizon_min) MINUTE <= now()
		ORDER BY timestamp ASC
		LIMIT 200
	`)
	if err != nil {
		log.Printf("❌ labelMaturedPredictions query error: %v", err)
		return
	}
	defer rows.Close()

	labeledCount := 0
	errorCount := 0
	skippedCount := 0

	for rows.Next() {
		var (
			symbol          string
			timestamp       time.Time
			labelHorizonMin int64
			direction       string
			confidence      float64
		)
		if err := rows.Scan(&symbol, &timestamp, &labelHorizonMin, &direction, &confidence); err != nil {
			log.Printf("❌ labelMaturedPredictions scan error: %v", err)
			errorCount++
			continue
		}

		// Calculate target timestamp
		targetTime := timestamp.Add(time.Duration(labelHorizonMin) * time.Minute)
		maturityTime := targetTime.Unix()
		nowTime := time.Now().Unix()
		ageMinutes := (nowTime - timestamp.Unix()) / 60

		log.Printf("🔍 Processing: %s %s @ %s (age=%dm, horizon=%dm, maturity=%ds)",
			symbol, direction, timestamp.Format(time.RFC3339), ageMinutes, labelHorizonMin, maturityTime)

		// Fetch price at prediction time (entry price)
		var startPrice, targetPrice sql.NullFloat64
		err = ae.db.QueryRow(`
			SELECT close FROM candle_cache
			WHERE symbol = ? AND timestamp <= ?
			ORDER BY timestamp DESC LIMIT 1
		`, symbol, timestamp.Unix()).Scan(&startPrice)
		if err != nil || !startPrice.Valid {
			log.Printf("⚠️ Skipping %s: cannot find entry price at %s (err=%v, valid=%v)",
				symbol, timestamp.Format(time.RFC3339), err, startPrice.Valid)
			skippedCount++
			continue
		}

		// Fetch price at target time (exit price) - use wider window
		err = ae.db.QueryRow(`
			SELECT close FROM candle_cache
			WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
			ORDER BY timestamp ASC LIMIT 1
		`, symbol, targetTime.Unix(), targetTime.Add(10*time.Minute).Unix()).Scan(&targetPrice)
		if err != nil || !targetPrice.Valid {
			// Fallback: use most recent price if target price not found
			log.Printf("⚠️ Target price not found for %s at %s, using latest price", symbol, targetTime.Format(time.RFC3339))
			err = ae.db.QueryRow(`
				SELECT close FROM candle_cache
				WHERE symbol = ?
				ORDER BY timestamp DESC LIMIT 1
			`, symbol).Scan(&targetPrice)
			if err != nil || !targetPrice.Valid {
				log.Printf("❌ Skipping %s: cannot find any price data (err=%v)", symbol, err)
				skippedCount++
				continue
			}
		}

		// Determine actual direction using same threshold as training
		priceChange := (targetPrice.Float64 - startPrice.Float64) / startPrice.Float64
		var actualDirection string
		threshold := ae.getNeutralThreshold(symbol) // Use adaptive threshold
		if threshold <= 0 {
			threshold = 0.002 // Fallback to 0.2%
		}

		if priceChange > threshold {
			actualDirection = "UP"
		} else if priceChange < -threshold {
			actualDirection = "DOWN"
		} else {
			actualDirection = "SIDEWAYS"
		}

		// Calculate accuracy (1 if correct, 0 if wrong)
		correct := (direction == "UP" && actualDirection == "UP") ||
			(direction == "DOWN" && actualDirection == "DOWN") ||
			(direction == "SIDEWAYS" && actualDirection == "SIDEWAYS")
		accuracyScore := 0.0
		if correct {
			accuracyScore = 1.0
		}

		// Update the prediction with actual results
		_, err = ae.db.Exec(`
			UPDATE direction_predictions
			SET actual_direction = ?, actual_price = ?, accuracy_score = ?
			WHERE symbol = ? AND timestamp = toDateTime(?)
		`, actualDirection, targetPrice.Float64, accuracyScore, symbol, timestamp.Unix())
		if err != nil {
			log.Printf("❌ labelMaturedPredictions update error for %s: %v", symbol, err)
			errorCount++
			continue
		}

		labeledCount++
		log.Printf("✅ LABELED: %s %s -> %s (entry: %.6f, exit: %.6f, change: %.4f%%, threshold: %.4f%%, correct: %v)",
			symbol, direction, actualDirection, startPrice.Float64, targetPrice.Float64,
			priceChange*100, threshold*100, correct)

		// Update in-memory model accuracy for online learning
		ae.updateModelAccuracyFromLabel(symbol, direction, actualDirection)
	}

	if labeledCount > 0 {
		log.Printf("📊 LABELING COMPLETE: labeled=%d, skipped=%d, errors=%d", labeledCount, skippedCount, errorCount)
	} else {
		log.Printf("📊 LABELING PASS: no mature predictions found (skipped=%d, errors=%d)", skippedCount, errorCount)
	}
}

// updateModelAccuracyFromLabel updates the in-memory model accuracy when a prediction is labeled
func (ae *AnalyticsEngine) updateModelAccuracyFromLabel(symbol, predicted, actual string) {
	ae.mlMu.Lock()
	defer ae.mlMu.Unlock()

	model, exists := ae.models[symbol]
	if !exists {
		return
	}

	isCorrect := (predicted == "UP" && actual == "UP") ||
		(predicted == "DOWN" && actual == "DOWN") ||
		(predicted == "SIDEWAYS" && actual == "SIDEWAYS")

	model.UpdateAccuracy(isCorrect)
	log.Printf("📈 Model %s updated: predictions=%d, correct=%d, accuracy=%.4f",
		symbol, model.PredictionCount, model.CorrectCount, model.GetAccuracy())
}

// runLabelingJob runs the labeling job periodically
func (ae *AnalyticsEngine) runLabelingJob(ctx context.Context) {
	// Run immediately on startup to label any mature predictions from previous session
	log.Println("🏷️  Starting ML prediction labeling job...")
	time.Sleep(2 * time.Second) // Wait for DB connection to stabilize
	ae.labelMaturedPredictions()
	
	// Then run every 1 minute for faster feedback (reduced from 5 minutes)
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			log.Println("Labeling job stopped")
			return
		case <-ticker.C:
			ae.labelMaturedPredictions()
		}
	}
}

func main() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	engine := NewAnalyticsEngine()
	defer func() {
		if engine != nil {
			// Close database connection with nil check
			if engine.db != nil {
				if err := engine.db.Close(); err != nil {
					log.Printf("Error closing DB: %v", err)
				}
			}

			// Close Kafka reader with nil check
			if engine.kafkaReader != nil {
				if err := engine.kafkaReader.Close(); err != nil {
					log.Printf("Error closing Kafka Reader: %v", err)
				}
			}

			// Close Kafka producer with nil check
			if engine.kafkaProducer != nil {
				if err := engine.kafkaProducer.Close(); err != nil {
					log.Printf("Error closing Kafka Producer: %v", err)
				}
			}

			// Cancel context if it exists
			if engine.cancel != nil {
				engine.cancel()
			}
		}
	}()

	// Run engine with nil check
	if engine != nil {
		engine.Run(ctx)
	}
}
