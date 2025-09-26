package main

import (
	"context"
	"encoding/json"
	"errors"
	"log"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"github.com/segmentio/kafka-go"
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
	PriceTarget  float64 `json:"price_target"`
	CurrentPrice float64 `json:"current_price"`
	TimeHorizon  int     `json:"time_horizon"` // Minutes
	Timestamp    string  `json:"timestamp"`    // Received as string from time.Time
	CreatedAt    string  `json:"created_at"`   // Alternative timestamp field
	Prediction   string  `json:"prediction"`   // Alternative field name
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
	upgrader  = websocket.Upgrader{CheckOrigin: func(r *http.Request) bool { return true }}
	wsClients = make(map[*websocket.Conn]bool)
	wsMutex   sync.Mutex

	// Kafka Consumers
	tickerReader    *kafka.Reader
	signalReader    *kafka.Reader
	directionReader *kafka.Reader // 🤖 For ML direction signals
)

func main() {
	r := gin.Default()

	// Initialize Kafka Consumers
	kafkaBrokers := os.Getenv("KAFKA_BROKERS")
	if kafkaBrokers == "" {
		kafkaBrokers = "kafka:9092"
	}

	tickerReader = kafka.NewReader(kafka.ReaderConfig{
		Brokers:     []string{kafkaBrokers},
		Topic:       "ticker",
		GroupID:     "api-gateway-ticker-group",
		MinBytes:    10e3, // 10KB
		MaxBytes:    10e6, // 10MB
		MaxAttempts: 10,
	})
	defer tickerReader.Close()

	signalReader = kafka.NewReader(kafka.ReaderConfig{
		Brokers:     []string{kafkaBrokers},
		Topic:       "pump_signals",
		GroupID:     "api-gateway-signals-group",
		MinBytes:    10e3,
		MaxBytes:    10e6,
		MaxAttempts: 10,
	})
	defer signalReader.Close()

	// 🤖 Direction signals reader for ML predictions
	directionReader = kafka.NewReader(kafka.ReaderConfig{
		Brokers:     []string{kafkaBrokers},
		Topic:       "direction_signals",
		GroupID:     "api-gateway-direction-group",
		MinBytes:    10e3,
		MaxBytes:    10e6,
		MaxAttempts: 10,
	})
	defer directionReader.Close()

	// REST API Endpoints
	r.GET("/api/v1/market/pairs", getAllPairs)
	r.GET("/api/v1/market/pairs/:symbol", getPair)
	r.GET("/api/v1/market/scan", scanMarket)

	// Admin endpoints for continuous learning
	r.POST("/api/v1/admin/feedback", submitFeedback)
	r.GET("/api/v1/admin/performance", getPerformanceMetrics)
	r.GET("/api/v1/admin/model-stats", getModelStats)

	// WebSocket Endpoint
	r.GET("/ws", wsHandler)

	// Start background goroutines
	go consumeTickers(context.Background())
	go consumePumpSignals(context.Background())
	go consumeDirectionSignals(context.Background()) // 🤖 ML direction signals
	go broadcastMarketData()                         // Re-use existing broadcast with real data

	log.Println("API Gateway starting on :8080")
	if err := r.Run(":8080"); err != nil {
		log.Fatalf("Failed to run API Gateway: %v", err)
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

		var dirSignal DirectionSignal
		if err := json.Unmarshal(m.Value, &dirSignal); err != nil {
			log.Printf("Error unmarshalling direction signal: %v", err)
			_ = directionReader.CommitMessages(ctx, m)
			continue
		}

		// Parse timestamp for frontend (convert from RFC3339 string to unix timestamp)
		var timestampUnix int64
		if dirSignal.Timestamp != "" {
			if parsedTime, err := time.Parse(time.RFC3339, dirSignal.Timestamp); err == nil {
				timestampUnix = parsedTime.Unix()
			} else {
				// Fallback: try parsing as Unix timestamp string
				if ts, err := strconv.ParseInt(dirSignal.Timestamp, 10, 64); err == nil {
					timestampUnix = ts
				} else {
					timestampUnix = time.Now().Unix()
				}
			}
		} else if dirSignal.CreatedAt != "" {
			if parsedTime, err := time.Parse(time.RFC3339, dirSignal.CreatedAt); err == nil {
				timestampUnix = parsedTime.Unix()
			} else {
				timestampUnix = time.Now().Unix()
			}
		} else {
			timestampUnix = time.Now().Unix()
		}

		// Log the ML signal for monitoring
		log.Printf("🤖 ML SIGNAL: %s %s (%.1f%% confidence) - Target: %.8f",
			dirSignal.Symbol, dirSignal.Direction, dirSignal.Confidence*100, dirSignal.PriceTarget)

		// Convert to frontend-compatible format
		frontendSignal := map[string]interface{}{
			"symbol":        dirSignal.Symbol,
			"direction":     dirSignal.Direction,
			"confidence":    dirSignal.Confidence,
			"price_target":  dirSignal.PriceTarget,
			"current_price": dirSignal.CurrentPrice,
			"time_horizon":  dirSignal.TimeHorizon,
			"timestamp":     timestampUnix,
			"prediction":    dirSignal.Prediction,
			"model_used":    dirSignal.ModelUsed,
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
func broadcastMarketData() {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		dataMutex.RLock()
		snapshot := make([]MarketPair, 0, len(marketData))
		for _, p := range marketData {
			snapshot = append(snapshot, p)
		}
		dataMutex.RUnlock()

		broadcastWebSocketMessage(map[string]interface{}{
			"type": "market_update",
			"data": snapshot,
		})
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

	// Set default confidence if not provided
	if feedback.Confidence == 0 {
		feedback.Confidence = 1.0
	}

	// TODO: Store feedback in database or send to Analytics Engine
	// For now, just log and acknowledge
	log.Printf("📝 Manual feedback received: %s predicted=%.3f actual=%t confidence=%.2f",
		feedback.Symbol, feedback.PredictedProb, feedback.ActualPump, feedback.Confidence)

	c.JSON(200, gin.H{
		"status":      "success",
		"message":     "Feedback submitted successfully",
		"feedback_id": time.Now().Unix(),
	})
}

// getPerformanceMetrics returns current model performance metrics
func getPerformanceMetrics(c *gin.Context) {
	// TODO: Fetch real performance metrics from Analytics Engine
	// For now, return mock data
	metrics := gin.H{
		"accuracy":        0.75,
		"precision":       0.72,
		"recall":          0.68,
		"f1_score":        0.70,
		"sample_count":    250,
		"last_updated":    time.Now().Unix(),
		"model_version":   1,
		"false_positives": 45,
		"true_positives":  120,
		"false_negatives": 55,
		"true_negatives":  30,
	}

	c.JSON(200, metrics)
}

// getModelStats returns current model statistics and learning status
func getModelStats(c *gin.Context) {
	// TODO: Fetch real model stats from Analytics Engine
	// For now, return mock data
	stats := gin.H{
		"learning_rate":         0.01,
		"adaptation_rate":       0.001,
		"model_weights_count":   6,
		"feature_stats_count":   6,
		"last_retrain":          time.Now().Add(-2 * time.Hour).Unix(),
		"pending_feedback":      15,
		"auto_feedback_enabled": true,
		"pump_threshold":        0.7,
		"status":                "active",
		"features": []string{
			"volume_spike_ratio",
			"price_change_5m",
			"volatility",
			"rsi",
			"volume_momentum",
			"price_momentum",
		},
	}

	c.JSON(200, stats)
}
