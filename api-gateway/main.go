package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"github.com/rus-connect/pkg/validator"
	"github.com/segmentio/kafka-go"
	"golang.org/x/time/rate"
)

// MarketPair represents a simplified market pair data structure for frontend.
type MarketPair struct {
	Symbol       string  `json:"symbol"`
	Price        float64 `json:"price"`
	Volume       float64 `json:"volume"`
	AnomalyScore float64 `json:"anomaly_score"`
	LastUpdate   int64   `json:"last_update"`
}

// PumpSignal (simplified for WebSocket broadcast)
type PumpSignal struct {
	Symbol      string  `json:"symbol"`
	Probability float64 `json:"probability"`
	Timestamp   int64   `json:"timestamp"`
	Alert       string  `json:"alert"`
}

// DirectionSignal represents ML direction predictions
type DirectionSignal struct {
	Symbol       string  `json:"symbol"`
	Direction    string  `json:"direction"`  // "UP", "DOWN", "SIDEWAYS"
	Confidence   float64 `json:"confidence"` // 0.0 - 1.0
	ClassProbs   struct {
		Down     float64 `json:"down"`
		Sideways float64 `json:"sideways"`
		Up       float64 `json:"up"`
	} `json:"class_probs"`
	PriceTarget  float64 `json:"price_target"`
	CurrentPrice float64 `json:"current_price"`
	TimeHorizon  int     `json:"time_horizon"` // Minutes
	LabelHorizonMin int   `json:"label_horizon_min"`
	Timestamp    int64   `json:"timestamp"`    // Unix timestamp (matches analytics engine)
	StopLoss     float64 `json:"stop_loss"`    // Stop loss price
	Volatility   float64 `json:"volatility"`   // Market volatility (0-1 scale)
	TrustStage   string  `json:"trust_stage"`
	ModelAgeSec  int64   `json:"model_age_sec"`
	ModelUsed    string  `json:"model_used"`
}

// TickerData mirrors data-fetcher's DataPoint for ticker topic.
type TickerData struct {
	Symbol string  `json:"symbol"`
	Price  float64 `json:"price"`
	Volume float64 `json:"volume"`
	Time   int64   `json:"timestamp"`
}

var (
	// In-memory store for market pairs (simulate Redis cache)
	marketData = make(map[string]MarketPair)
	dataMutex  sync.RWMutex

	// WebSocket clients
	upgrader = websocket.Upgrader{
		CheckOrigin: func(r *http.Request) bool {
			// Allow connections from localhost and the frontend domain
			origin := r.Header.Get("Origin")
			return origin == "http://localhost:3000" || origin == "http://frontend:80" || origin == ""
		},
	}
	wsClients = make(map[*websocket.Conn]bool)
	wsMutex   sync.Mutex

	// Kafka Consumers
	tickerReader    *kafka.Reader
	signalReader    *kafka.Reader
	directionReader *kafka.Reader // 🤖 For ML direction signals

	// HTTP client for communicating with analytics engine
	httpClient = &http.Client{
		Timeout: 30 * time.Second,
	}

	// Rate limiter for API requests
	limiter = rate.NewLimiter(100, 200) // 100 req/sec, burst 200
)

func main() {
	log.Println("🚀 API Gateway starting...")

	r := gin.Default()

	// Add security headers middleware
	r.Use(securityHeadersMiddleware())

	// Add rate limiting middleware
	r.Use(rateLimitMiddleware())

	// Initialize Kafka Consumers
	kafkaBrokers := os.Getenv("KAFKA_BROKERS")
	if kafkaBrokers == "" {
		kafkaBrokers = "kafka:9092"
	}

	tickerReader = kafka.NewReader(kafka.ReaderConfig{
		Brokers:       []string{kafkaBrokers},
		Topic:         "ticker",
		GroupID:       "api-gateway-ticker-group",
		MinBytes:      10e3, // 10KB
		MaxBytes:      1e6,  // Уменьшил с 10MB до 1MB
		MaxAttempts:   10,
		QueueCapacity: 10, // Ограничиваем размер внутренней очереди
	})

	signalReader = kafka.NewReader(kafka.ReaderConfig{
		Brokers:       []string{kafkaBrokers},
		Topic:         "pump_signals",
		GroupID:       "api-gateway-signals-group",
		MinBytes:      10e3,
		MaxBytes:      1e6, // Уменьшил с 10MB до 1MB
		MaxAttempts:   10,
		QueueCapacity: 10, // Ограничиваем размер внутренней очереди
	})

	// 🤖 Direction signals reader for ML predictions
	directionReader = kafka.NewReader(kafka.ReaderConfig{
		Brokers:       []string{kafkaBrokers},
		Topic:         "direction_signals",
		GroupID:       "api-gateway-direction-group",
		MinBytes:      10e3,
		MaxBytes:      1e6, // Уменьшил с 10MB до 1MB
		MaxAttempts:   10,
		QueueCapacity: 10, // Ограничиваем размер внутренней очереди
	})

	// REST API Endpoints
	r.GET("/health", handleHealthz)
	r.GET("/healthz", handleHealthz)
	r.GET("/readyz", handleReadyz)

	r.GET("/api/v1/market/pairs", getAllPairs)
	r.GET("/api/v1/market/pairs/:symbol", getPair)
	r.GET("/api/v1/market/scan", scanMarket)

	// Admin endpoints for continuous learning
	r.POST("/api/v1/admin/feedback", submitFeedback)
	r.GET("/api/v1/admin/performance", getPerformanceMetrics)
	r.GET("/api/v1/admin/model-stats", getModelStats)

	// ML Metrics and Calibration endpoints
	r.GET("/api/v1/ml/metrics", getMLMetrics)
	r.GET("/api/v1/ml/calibration", getCalibrationStatus)
	r.POST("/api/v1/ml/calibration/start", startAutoCalibration)
	r.GET("/api/v1/ml/training-history", getTrainingHistory)
	r.GET("/api/v1/ml/signal-stats", getSignalStats)
	r.GET("/api/v1/ml/signals/recent", getRecentSignals)
	r.GET("/api/v1/ml/signals/history", getSignalsHistory)
	// Infrastructure monitoring endpoint
	r.GET("/api/v1/infrastructure/metrics", getInfrastructureMetrics)

	// Proxy trader-mind endpoints to analytics-engine
	r.GET("/api/v1/trader-mind/:symbol", proxyTraderMind)
	r.GET("/api/v1/trader-mind/full/:symbol", proxyTraderMindFull)
	r.POST("/api/v1/model/retrain", proxyModelRetrain)

	// WebSocket Endpoint
	r.GET("/ws", wsHandler)

	// ✅ Setup context for background goroutines
	ctx, cancel := context.WithCancel(context.Background())

	// Start background goroutines
	go consumeTickers(ctx)
	go consumePumpSignals(ctx)
	go consumeDirectionSignals(ctx) // 🤖 ML direction signals
	go broadcastMarketData(ctx)     // Re-use existing broadcast with real data

	// ✅ Setup graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM, syscall.SIGINT)

	// Start HTTP server in goroutine
	apiGatewayPort := os.Getenv("API_GATEWAY_PORT")
	if apiGatewayPort == "" {
		apiGatewayPort = "8080"
	}
	srv := &http.Server{
		Addr:    ":" + apiGatewayPort,
		Handler: r,
	}

	go func() {
		log.Printf("🌐 API Gateway listening on :%s", apiGatewayPort)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("❌ Failed to run API Gateway: %v", err)
		}
	}()

	// Wait for shutdown signal
	sig := <-sigChan
	log.Printf("🛑 Received signal %v, initiating graceful shutdown...", sig)

	// Cancel context to stop background goroutines
	cancel()

	// Shutdown HTTP server with timeout
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer shutdownCancel()

	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Printf("❌ HTTP server forced to shutdown: %v", err)
	} else {
		log.Println("✅ HTTP server stopped gracefully")
	}

	// Close Kafka readers
	log.Println("🔒 Closing Kafka readers...")
	if err := tickerReader.Close(); err != nil {
		log.Printf("❌ Error closing ticker reader: %v", err)
	}
	if err := signalReader.Close(); err != nil {
		log.Printf("❌ Error closing signal reader: %v", err)
	}
	if err := directionReader.Close(); err != nil {
		log.Printf("❌ Error closing direction reader: %v", err)
	}
	log.Println("✅ Kafka readers closed")

	// Close all WebSocket connections
	wsMutex.Lock()
	for client := range wsClients {
		client.Close()
	}
	wsMutex.Unlock()
	log.Println("✅ WebSocket connections closed")

	log.Println("✅ API Gateway stopped gracefully")
}

func handleHealthz(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status": "ok",
		"time":   time.Now().UTC(),
	})
}

func handleReadyz(c *gin.Context) {
	kafkaReady := tickerReader != nil && signalReader != nil && directionReader != nil

	analyticsEngineURL := os.Getenv("ANALYTICS_ENGINE_URL")
	if analyticsEngineURL == "" {
		analyticsEngineURL = "http://analytics-engine:8081"
	}

	analyticsReady := false
	resp, err := httpClient.Get(analyticsEngineURL + "/readyz")
	if err == nil {
		analyticsReady = resp.StatusCode == http.StatusOK
		resp.Body.Close()
	}

	if !kafkaReady || !analyticsReady {
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"status":          "not_ready",
			"kafka_ready":     kafkaReady,
			"analytics_ready": analyticsReady,
			"time":            time.Now().UTC(),
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"status":          "ready",
		"kafka_ready":     true,
		"analytics_ready": true,
		"time":            time.Now().UTC(),
	})
}

// securityHeadersMiddleware adds security headers to responses
func securityHeadersMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Header("X-Frame-Options", "DENY")
		c.Header("X-Content-Type-Options", "nosniff")
		c.Header("X-XSS-Protection", "1; mode=block")
		c.Header("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
		c.Header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'self' ws: wss:; frame-ancestors 'none'; object-src 'none'")
		c.Next()
	}
}

// rateLimitMiddleware provides rate limiting for API requests
func rateLimitMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		if !limiter.Allow() {
			c.JSON(429, gin.H{"error": "Rate limit exceeded"})
			c.Abort()
			return
		}
		c.Next()
	}
}

// consumeTickers consumes ticker data from Kafka and updates in-memory cache.
func consumeTickers(ctx context.Context) {
	log.Println("Consuming tickers from Kafka...")
	for {
		m, err := tickerReader.FetchMessage(ctx)
		if err != nil {
			if errors.Is(err, context.Canceled) {
				log.Println("Ticker consumer context canceled.")
				return
			}
			log.Printf("Error fetching ticker message from Kafka: %v", err)
			time.Sleep(1 * time.Second)
			continue
		}

		var td TickerData
		if err := json.Unmarshal(m.Value, &td); err != nil {
			log.Printf("Error unmarshalling ticker data: %v", err)
			_ = tickerReader.CommitMessages(ctx, m)
			continue
		}

		dataMutex.Lock()
		existing := marketData[td.Symbol]
		existing.Symbol = td.Symbol
		existing.Price = td.Price
		existing.Volume = td.Volume
		existing.LastUpdate = time.Now().Unix()
		marketData[td.Symbol] = existing
		dataMutex.Unlock()

		if err := tickerReader.CommitMessages(ctx, m); err != nil {
			log.Printf("Error committing ticker message: %v", err)
		}
	}
}

// consumePumpSignals consumes pump signals from Kafka and updates anomaly scores or broadcasts directly.
func consumePumpSignals(ctx context.Context) {
	log.Println("Consuming pump signals from Kafka...")
	for {
		m, err := signalReader.FetchMessage(ctx)
		if err != nil {
			if errors.Is(err, context.Canceled) {
				log.Println("Signal consumer context canceled.")
				return
			}
			log.Printf("Error fetching signal message from Kafka: %v", err)
			time.Sleep(1 * time.Second)
			continue
		}

		var signal PumpSignal
		if err := json.Unmarshal(m.Value, &signal); err != nil {
			log.Printf("Error unmarshalling pump signal: %v", err)
			_ = signalReader.CommitMessages(ctx, m)
			continue
		}

		dataMutex.Lock()
		if existing, ok := marketData[signal.Symbol]; ok {
			// Update anomaly score and last update
			existing.AnomalyScore = signal.Probability * 100 // Convert to 0-100 scale
			existing.LastUpdate = time.Now().Unix()
			marketData[signal.Symbol] = existing
		} else {
			// If pair not seen yet, create a minimal entry
			marketData[signal.Symbol] = MarketPair{
				Symbol:       signal.Symbol,
				AnomalyScore: signal.Probability * 100,
				LastUpdate:   time.Now().Unix(),
				Price:        0, // Unknown without ticker data
				Volume:       0,
			}
		}
		dataMutex.Unlock()

		// Also broadcast the signal directly over WebSocket
		broadcastWebSocketMessage(map[string]interface{}{
			"type": "pump_signal_update",
			"data": signal,
		})

		if err := signalReader.CommitMessages(ctx, m); err != nil {
			log.Printf("Error committing signal message: %v", err)
		}
	}
}

// 🤖 consumeDirectionSignals consumes ML direction signals from Kafka
func consumeDirectionSignals(ctx context.Context) {
	log.Println("🤖 Consuming ML direction signals from Kafka...")
	for {
		m, err := directionReader.FetchMessage(ctx)
		if err != nil {
			if errors.Is(err, context.Canceled) {
				log.Println("Direction consumer context canceled.")
				return
			}
			log.Printf("Error fetching direction message from Kafka: %v", err)
			time.Sleep(1 * time.Second)
			continue
		}

		// Parse direction signal from analytics engine
		var dirSignal DirectionSignal
		if err := json.Unmarshal(m.Value, &dirSignal); err != nil {
			log.Printf("Error unmarshalling direction signal: %v", err)
			_ = directionReader.CommitMessages(ctx, m)
			continue
		}

		// Use timestamp directly from analytics engine (int64 Unix timestamp)
		timestampUnix := dirSignal.Timestamp
		if timestampUnix == 0 {
			// Fallback to current time if timestamp is missing
			timestampUnix = time.Now().Unix()
		}

		// Log the ML signal for monitoring
		log.Printf("🤖 ML SIGNAL: %s %s (%.1f%% confidence) - Target: %.8f",
			dirSignal.Symbol, dirSignal.Direction, dirSignal.Confidence*100, dirSignal.PriceTarget)

		// Use model_used field directly from analytics engine
		resolvedModel := dirSignal.ModelUsed
		if resolvedModel == "" {
			resolvedModel = "SimpleNN" // Default model name
		}

		// Convert to frontend-compatible format
		frontendSignal := map[string]interface{}{
			"symbol":        dirSignal.Symbol,
			"direction":     dirSignal.Direction,
			"confidence":    dirSignal.Confidence,
			"class_probs":   dirSignal.ClassProbs,
			"price_target":  dirSignal.PriceTarget,
			"current_price": dirSignal.CurrentPrice,
			"time_horizon":  dirSignal.TimeHorizon,
			"label_horizon_min": dirSignal.LabelHorizonMin,
			"timestamp":     timestampUnix,
			"stop_loss":     dirSignal.StopLoss,
			"volatility":    dirSignal.Volatility,
			"trust_stage":   dirSignal.TrustStage,
			"model_age_sec": dirSignal.ModelAgeSec,
			"model_used":    resolvedModel,
		}

		// Broadcast the direction signal to frontend
		broadcastWebSocketMessage(map[string]interface{}{
			"type": "direction_signal",
			"data": frontendSignal,
		})

		if err := directionReader.CommitMessages(ctx, m); err != nil {
			log.Printf("Error committing direction message: %v", err)
		}
	}
}

func getAllPairs(c *gin.Context) {
	dataMutex.RLock()
	defer dataMutex.RUnlock()
	list := make([]MarketPair, 0, len(marketData))
	for _, p := range marketData {
		list = append(list, p)
	}
	c.JSON(http.StatusOK, list)
}

func getPair(c *gin.Context) {
	symbol := c.Param("symbol")
	if err := validator.ValidateSymbol(symbol); err != nil {
		c.JSON(400, gin.H{"error": "Invalid symbol format"})
		return
	}
	dataMutex.RLock()
	defer dataMutex.RUnlock()
	if p, ok := marketData[symbol]; ok {
		c.JSON(http.StatusOK, p)
		return
	}
	c.JSON(http.StatusNotFound, gin.H{"error": "not found"})
}

func scanMarket(c *gin.Context) {
	// For now, scan is same as get all pairs. Can add filters later.
	getAllPairs(c)
}

func wsHandler(c *gin.Context) {
	conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		log.Printf("WebSocket upgrade error: %v", err)
		return
	}
	wsMutex.Lock()
	wsClients[conn] = true
	wsMutex.Unlock()
	defer func() {
		wsMutex.Lock()
		delete(wsClients, conn)
		wsMutex.Unlock()
		_ = conn.Close()
	}()

	// Send initial data to new client
	dataMutex.RLock()
	initialData := make([]MarketPair, 0, len(marketData))
	for _, p := range marketData {
		initialData = append(initialData, p)
	}
	dataMutex.RUnlock()

	if err := conn.WriteJSON(map[string]interface{}{"type": "initial_data", "data": initialData}); err != nil {
		log.Printf("Error sending initial data to WS client: %v", err)
		return
	}

	// Keep connection alive
	for {
		_, _, err := conn.ReadMessage() // Read messages to detect client disconnect
		if err != nil {
			break // Client disconnected
		}
		time.Sleep(1 * time.Second) // Prevent busy-looping
	}
}

// broadcastMarketData broadcasts current market data to all connected WebSocket clients.
func broadcastMarketData(ctx context.Context) {
	// Увеличил интервал с 2 до 5 секунд для уменьшения нагрузки
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			log.Println("Broadcast goroutine stopped")
			return
		case <-ticker.C:
			dataMutex.RLock()
			data := make([]MarketPair, 0, len(marketData))
			for _, p := range marketData {
				data = append(data, p)
			}
			dataMutex.RUnlock()

			broadcastWebSocketMessage(map[string]interface{}{
				"type": "market_update",
				"data": data,
			})
		}
	}
}

// broadcastWebSocketMessage sends a message to all connected WebSocket clients.
func broadcastWebSocketMessage(message interface{}) {
	wsMutex.Lock()
	defer wsMutex.Unlock()
	for client := range wsClients {
		if err := client.WriteJSON(message); err != nil {
			log.Printf("Error sending to WebSocket client: %v", err)
			_ = client.Close() // Close connection if sending fails
			delete(wsClients, client)
		}
	}
}

// submitFeedback handles manual feedback submission for continuous learning
func submitFeedback(c *gin.Context) {
	var feedback struct {
		Symbol        string  `json:"symbol" binding:"required"`
		Timestamp     int64   `json:"timestamp" binding:"required"`
		PredictedProb float64 `json:"predicted_prob" binding:"required"`
		ActualPump    bool    `json:"actual_pump" binding:"required"`
		Confidence    float64 `json:"confidence"`
		Notes         string  `json:"notes"`
	}

	if err := c.ShouldBindJSON(&feedback); err != nil {
		c.JSON(400, gin.H{"error": "Invalid feedback data: " + err.Error()})
		return
	}

	// Validate input fields
	if err := validator.ValidateSymbol(feedback.Symbol); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	if err := validator.ValidateTimestamp(feedback.Timestamp); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	if err := validator.ValidateConfidence(feedback.PredictedProb); err != nil {
		c.JSON(400, gin.H{"error": "predicted_prob: " + err.Error()})
		return
	}
	if feedback.Confidence != 0 { // Only validate confidence if it's provided
		if err := validator.ValidateConfidence(feedback.Confidence); err != nil {
			c.JSON(400, gin.H{"error": "confidence: " + err.Error()})
			return
		}
	}

	// Set default confidence if not provided
	if feedback.Confidence == 0 {
		feedback.Confidence = 1.0
	}

	// Forward to Analytics Engine
	analyticsEngineURL := os.Getenv("ANALYTICS_ENGINE_URL")
	if analyticsEngineURL == "" {
		analyticsEngineURL = "http://analytics-engine:8081"
	}

	jsonData, err := json.Marshal(feedback)
	if err != nil {
		c.JSON(500, gin.H{"error": "Failed to serialize feedback"})
		return
	}

	resp, err := httpClient.Post(analyticsEngineURL+"/api/v1/feedback", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		log.Printf("Error sending feedback to analytics engine: %v", err)
		c.JSON(500, gin.H{"error": "Failed to connect to analytics engine"})
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		log.Printf("Analytics engine returned error: %d", resp.StatusCode)
		c.JSON(resp.StatusCode, gin.H{"error": "Analytics engine rejected feedback"})
		return
	}

	c.JSON(200, gin.H{
		"status":      "success",
		"message":     "Feedback submitted successfully",
		"feedback_id": time.Now().Unix(),
	})
}

// getPerformanceMetrics returns current model performance metrics
func getPerformanceMetrics(c *gin.Context) {
	// Get analytics engine URL from environment or use default
	analyticsEngineURL := os.Getenv("ANALYTICS_ENGINE_URL")
	if analyticsEngineURL == "" {
		analyticsEngineURL = "http://analytics-engine:8081"
	}

	// Make HTTP request to analytics engine
	resp, err := httpClient.Get(analyticsEngineURL + "/model/performance")
	if err != nil {
		log.Printf("Error fetching performance metrics from analytics engine: %v", err)
		c.JSON(503, gin.H{
			"error":  "Analytics Engine unavailable",
			"status": "down",
		})
		return
	}
	defer resp.Body.Close()

	// Check if response status is OK
	if resp.StatusCode != http.StatusOK {
		log.Printf("Analytics engine returned non-OK status for performance metrics: %d", resp.StatusCode)
		c.JSON(resp.StatusCode, gin.H{
			"error":  "Analytics Engine returned error",
			"status": "error",
		})
		return
	}

	// Decode response
	var metrics map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&metrics); err != nil {
		log.Printf("Error decoding performance metrics response: %v", err)
		c.JSON(500, gin.H{
			"error":  "Failed to decode response",
			"status": "error",
		})
		return
	}

	c.JSON(resp.StatusCode, metrics)
}

// getMLMetrics returns detailed ML model performance metrics
func getMLMetrics(c *gin.Context) {
	// Get analytics engine URL from environment or use default
	analyticsEngineURL := os.Getenv("ANALYTICS_ENGINE_URL")
	if analyticsEngineURL == "" {
		analyticsEngineURL = "http://analytics-engine:8081"
	}

	// Make HTTP request to analytics engine
	resp, err := httpClient.Get(analyticsEngineURL + "/api/v1/ml/metrics")
	if err != nil {
		log.Printf("Error fetching ML metrics from analytics engine: %v", err)
		c.JSON(503, gin.H{
			"error":  "Analytics Engine unavailable",
			"system": gin.H{"overall_health": "UNKNOWN"},
		})
		return
	}
	defer resp.Body.Close()

	// Decode response
	var metrics map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&metrics); err != nil {
		log.Printf("Error decoding ML metrics response: %v", err)
		c.JSON(500, gin.H{
			"error":  "Failed to decode response",
			"system": gin.H{"overall_health": "UNKNOWN"},
		})
		return
	}

	c.JSON(resp.StatusCode, metrics)
}

// getCalibrationStatus returns the current calibration status of all models
func getCalibrationStatus(c *gin.Context) {
	// Get analytics engine URL from environment or use default
	analyticsEngineURL := os.Getenv("ANALYTICS_ENGINE_URL")
	if analyticsEngineURL == "" {
		analyticsEngineURL = "http://analytics-engine:8081"
	}

	// Make HTTP request to analytics engine
	resp, err := httpClient.Get(analyticsEngineURL + "/api/v1/ml/calibration")
	if err != nil {
		log.Printf("Error fetching calibration status from analytics engine: %v", err)
		c.JSON(503, gin.H{
			"error":  "Analytics Engine unavailable",
			"system": gin.H{"overall_status": "UNKNOWN"},
		})
		return
	}
	defer resp.Body.Close()

	// Decode response
	var status map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&status); err != nil {
		log.Printf("Error decoding calibration status response: %v", err)
		c.JSON(500, gin.H{
			"error":  "Failed to decode response",
			"system": gin.H{"overall_status": "UNKNOWN"},
		})
		return
	}

	c.JSON(resp.StatusCode, status)
}

// getInfrastructureMetrics returns infrastructure monitoring metrics
func getInfrastructureMetrics(c *gin.Context) {
	// Get analytics engine URL from environment or use default
	analyticsEngineURL := os.Getenv("ANALYTICS_ENGINE_URL")
	if analyticsEngineURL == "" {
		analyticsEngineURL = "http://analytics-engine:8081"
	}

	// Make HTTP request to analytics engine
	resp, err := httpClient.Get(analyticsEngineURL + "/api/v1/infrastructure/metrics")
	if err != nil {
		log.Printf("Error fetching infrastructure metrics from analytics engine: %v", err)
		c.JSON(503, gin.H{"error": "Analytics Engine unavailable"})
		return
	}
	defer resp.Body.Close()

	// Decode response
	var metrics map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&metrics); err != nil {
		log.Printf("Error decoding infrastructure metrics response: %v", err)
		c.JSON(500, gin.H{"error": "Failed to decode response"})
		return
	}

	c.JSON(resp.StatusCode, metrics)
}

// getTrainingHistory returns model training history from analytics engine
func getTrainingHistory(c *gin.Context) {
	// Get analytics engine URL from environment or use default
	analyticsEngineURL := os.Getenv("ANALYTICS_ENGINE_URL")
	if analyticsEngineURL == "" {
		analyticsEngineURL = "http://analytics-engine:8081"
	}

	// Get query parameters
	symbol := c.DefaultQuery("symbol", "BTCUSDT")
	limit := c.DefaultQuery("limit", "50")

	// Make HTTP request to analytics engine
	url := fmt.Sprintf("%s/api/v1/ml/training-history?symbol=%s&limit=%s", analyticsEngineURL, symbol, limit)
	resp, err := httpClient.Get(url)
	if err != nil {
		log.Printf("Error fetching training history from analytics engine: %v", err)
		c.JSON(503, gin.H{"error": "Analytics Engine unavailable"})
		return
	}
	defer resp.Body.Close()

	// Check if response status is OK
	if resp.StatusCode != http.StatusOK {
		log.Printf("Analytics engine returned non-OK status for training history: %d", resp.StatusCode)
		c.JSON(resp.StatusCode, gin.H{"error": "Analytics Engine returned error"})
		return
	}

	// Decode response
	var history map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&history); err != nil {
		log.Printf("Error decoding training history response: %v", err)
		c.JSON(500, gin.H{"error": "Failed to decode response"})
		return
	}

	c.JSON(resp.StatusCode, history)
}

// getSignalStats returns signal statistics from analytics engine
func getSignalStats(c *gin.Context) {
	// Get analytics engine URL from environment or use default
	analyticsEngineURL := os.Getenv("ANALYTICS_ENGINE_URL")
	if analyticsEngineURL == "" {
		analyticsEngineURL = "http://analytics-engine:8081"
	}

	// Get query parameters
	symbol := c.DefaultQuery("symbol", "BTCUSDT")
	hours := c.DefaultQuery("hours", "24")

	// Make HTTP request to analytics engine
	url := fmt.Sprintf("%s/api/v1/ml/signal-stats?symbol=%s&hours=%s", analyticsEngineURL, symbol, hours)
	resp, err := httpClient.Get(url)
	if err != nil {
		log.Printf("Error fetching signal statistics from analytics engine: %v", err)
		c.JSON(503, gin.H{"error": "Analytics Engine unavailable"})
		return
	}
	defer resp.Body.Close()

	// Check if response status is OK
	if resp.StatusCode != http.StatusOK {
		log.Printf("Analytics engine returned non-OK status for signal statistics: %d", resp.StatusCode)
		c.JSON(resp.StatusCode, gin.H{"error": "Analytics Engine returned error"})
		return
	}

	// Decode response
	var stats map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&stats); err != nil {
		log.Printf("Error decoding signal statistics response: %v", err)
		c.JSON(500, gin.H{"error": "Failed to decode response"})
		return
	}

	c.JSON(resp.StatusCode, stats)
}

// getRecentSignals returns recent persisted ML signals from analytics engine
func getRecentSignals(c *gin.Context) {
	analyticsEngineURL := os.Getenv("ANALYTICS_ENGINE_URL")
	if analyticsEngineURL == "" {
		analyticsEngineURL = "http://analytics-engine:8081"
	}

	limit := c.DefaultQuery("limit", "50")
	hours := c.DefaultQuery("hours", "24")
	symbol := c.Query("symbol")

	url := fmt.Sprintf("%s/api/v1/ml/signals/recent?limit=%s&hours=%s", analyticsEngineURL, limit, hours)
	if symbol != "" {
		url += "&symbol=" + symbol
	}

	resp, err := httpClient.Get(url)
	if err != nil {
		log.Printf("Error fetching recent signals from analytics engine: %v", err)
		c.JSON(503, gin.H{"error": "Analytics Engine unavailable"})
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		log.Printf("Analytics engine returned non-OK status for recent signals: %d", resp.StatusCode)
		c.JSON(resp.StatusCode, gin.H{"error": "Analytics Engine returned error"})
		return
	}

	var payload map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		log.Printf("Error decoding recent signals response: %v", err)
		c.JSON(500, gin.H{"error": "Failed to decode response"})
		return
	}

	c.JSON(resp.StatusCode, payload)
}

// getSignalsHistory returns historical ML signals from analytics engine
func getSignalsHistory(c *gin.Context) {
	analyticsEngineURL := os.Getenv("ANALYTICS_ENGINE_URL")
	if analyticsEngineURL == "" {
		analyticsEngineURL = "http://analytics-engine:8081"
	}

	limit := c.DefaultQuery("limit", "100")
	hours := c.DefaultQuery("hours", "168")
	symbol := c.Query("symbol")
	direction := c.Query("direction")

	url := fmt.Sprintf("%s/api/v1/ml/signals/history?limit=%s&hours=%s", analyticsEngineURL, limit, hours)
	if symbol != "" {
		url += "&symbol=" + symbol
	}
	if direction != "" {
		url += "&direction=" + direction
	}

	resp, err := httpClient.Get(url)
	if err != nil {
		log.Printf("Error fetching signals history from analytics engine: %v", err)
		c.JSON(503, gin.H{"error": "Analytics Engine unavailable"})
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		log.Printf("Analytics engine returned non-OK status for signals history: %d", resp.StatusCode)
		c.JSON(resp.StatusCode, gin.H{"error": "Analytics Engine returned error"})
		return
	}

	var payload map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		log.Printf("Error decoding signals history response: %v", err)
		c.JSON(500, gin.H{"error": "Failed to decode response"})
		return
	}

	c.JSON(resp.StatusCode, payload)
}

// startAutoCalibration triggers automatic calibration of all models
func startAutoCalibration(c *gin.Context) {
	log.Printf("🔧 API Gateway: Starting auto-calibration request...")

	// Get analytics engine URL from environment or use default
	analyticsEngineURL := os.Getenv("ANALYTICS_ENGINE_URL")
	if analyticsEngineURL == "" {
		analyticsEngineURL = "http://analytics-engine:8081"
	}

	log.Printf("🔧 API Gateway: Making request to analytics engine at %s", analyticsEngineURL)

	// Make HTTP request to analytics engine
	resp, err := httpClient.Post(analyticsEngineURL+"/api/v1/ml/calibration/start", "application/json", nil)
	if err != nil {
		log.Printf("❌ API Gateway: Error starting auto calibration: %v", err)
		c.JSON(503, gin.H{"error": "Analytics Engine unavailable"})
		return
	}
	defer resp.Body.Close()

	log.Printf("🔧 API Gateway: Received response from analytics engine with status %d", resp.StatusCode)

	// Decode response
	var response map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		log.Printf("❌ API Gateway: Error decoding auto calibration response: %v", err)
		c.JSON(500, gin.H{"error": "Failed to decode response"})
		return
	}

	log.Printf("✅ API Gateway: Successfully forwarded auto-calibration response")
	c.JSON(resp.StatusCode, response)
}

// proxyTraderMind forwards a simple summary request to analytics-engine
func proxyTraderMind(c *gin.Context) {
	symbol := c.Param("symbol")
	if symbol == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "symbol required"})
		return
	}

	analyticsEngineURL := os.Getenv("ANALYTICS_ENGINE_URL")
	if analyticsEngineURL == "" {
		analyticsEngineURL = "http://analytics-engine:8081"
	}

	resp, err := httpClient.Get(analyticsEngineURL + "/api/v1/trader-mind/" + symbol)
	if err != nil {
		log.Printf("Error contacting analytics engine for trader-mind: %v", err)
		c.JSON(502, gin.H{"error": "failed to contact analytics engine", "details": err.Error()})
		return
	}
	defer resp.Body.Close()

	// Check if response status is OK
	if resp.StatusCode != http.StatusOK {
		log.Printf("Analytics engine returned non-OK status for trader-mind: %d", resp.StatusCode)
		c.JSON(resp.StatusCode, gin.H{"error": "analytics engine error", "status": resp.StatusCode})
		return
	}

	var payload map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		log.Printf("Error decoding trader-mind response: %v", err)
		c.JSON(502, gin.H{"error": "invalid response from analytics engine", "details": err.Error()})
		return
	}
	c.JSON(resp.StatusCode, payload)
}

// proxyTraderMindFull forwards the full trader mind payload to frontend
func proxyTraderMindFull(c *gin.Context) {
	symbol := c.Param("symbol")
	if symbol == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "symbol required"})
		return
	}

	analyticsEngineURL := os.Getenv("ANALYTICS_ENGINE_URL")
	if analyticsEngineURL == "" {
		analyticsEngineURL = "http://analytics-engine:8081"
	}

	resp, err := httpClient.Get(analyticsEngineURL + "/api/v1/trader-mind/full/" + symbol)
	if err != nil {
		log.Printf("Error contacting analytics engine for trader-mind/full: %v", err)
		c.JSON(502, gin.H{"error": "failed to contact analytics engine", "details": err.Error()})
		return
	}
	defer resp.Body.Close()

	// Check if response status is OK
	if resp.StatusCode != http.StatusOK {
		log.Printf("Analytics engine returned non-OK status for trader-mind/full: %d", resp.StatusCode)
		c.JSON(resp.StatusCode, gin.H{"error": "analytics engine error", "status": resp.StatusCode})
		return
	}

	var payload map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		log.Printf("Error decoding trader-mind/full response: %v", err)
		c.JSON(502, gin.H{"error": "invalid response from analytics engine", "details": err.Error()})
		return
	}
	c.JSON(resp.StatusCode, payload)
}

func proxyModelRetrain(c *gin.Context) {
	analyticsEngineURL := os.Getenv("ANALYTICS_ENGINE_URL")
	if analyticsEngineURL == "" {
		analyticsEngineURL = "http://analytics-engine:8081"
	}

	targetURL := analyticsEngineURL + "/api/v1/model/retrain"
	if symbol := c.Query("symbol"); symbol != "" {
		targetURL += "?symbol=" + symbol
	}

	resp, err := httpClient.Post(targetURL, "application/json", nil)
	if err != nil {
		log.Printf("Error contacting analytics engine for model retrain: %v", err)
		c.JSON(http.StatusBadGateway, gin.H{"error": "failed to contact analytics engine"})
		return
	}
	defer resp.Body.Close()

	var payload map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": "invalid response from analytics engine"})
		return
	}

	c.JSON(resp.StatusCode, payload)
}

// getModelStats returns current model statistics and learning status
func getModelStats(c *gin.Context) {
	// Get analytics engine URL from environment or use default
	analyticsEngineURL := os.Getenv("ANALYTICS_ENGINE_URL")
	if analyticsEngineURL == "" {
		analyticsEngineURL = "http://analytics-engine:8081"
	}

	// Make HTTP request to analytics engine
	resp, err := httpClient.Get(analyticsEngineURL + "/model/performance")
	if err != nil {
		log.Printf("Error fetching model stats from analytics engine: %v", err)
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"error":       "Analytics Engine unavailable",
			"data_status": "empty",
		})
		return
	}
	defer resp.Body.Close()

	// Check if response status is OK
	if resp.StatusCode != http.StatusOK {
		log.Printf("Analytics engine returned non-OK status for model stats: %d", resp.StatusCode)
		c.JSON(resp.StatusCode, gin.H{
			"error":       "Analytics Engine returned error",
			"data_status": "empty",
		})
		return
	}

	// Decode response
	var stats map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&stats); err != nil {
		log.Printf("Error decoding model stats response: %v", err)
		c.JSON(http.StatusBadGateway, gin.H{
			"error":       "invalid response from analytics engine",
			"data_status": "empty",
		})
		return
	}

	c.JSON(resp.StatusCode, stats)
}
