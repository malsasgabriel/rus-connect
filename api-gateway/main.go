package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"github.com/rus-connect/pkg/validator"
	"github.com/segmentio/kafka-go"
	"golang.org/x/time/rate"
)

const (
	wsWriteWait      = 10 * time.Second
	wsPongWait       = 60 * time.Second
	wsPingPeriod     = (wsPongWait * 9) / 10
	wsMaxMessageSize = 4096
	wsSendBuffer     = 256
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
	Symbol     string  `json:"symbol"`
	Direction  string  `json:"direction"`  // "UP", "DOWN", "SIDEWAYS"
	Confidence float64 `json:"confidence"` // 0.0 - 1.0
	ClassProbs struct {
		Down     float64 `json:"down"`
		Sideways float64 `json:"sideways"`
		Up       float64 `json:"up"`
	} `json:"class_probs"`
	PriceTarget     float64 `json:"price_target"`
	CurrentPrice    float64 `json:"current_price"`
	TimeHorizon     int     `json:"time_horizon"` // Minutes
	LabelHorizonMin int     `json:"label_horizon_min"`
	Timestamp       int64   `json:"timestamp"`  // Unix timestamp (matches analytics engine)
	StopLoss        float64 `json:"stop_loss"`  // Stop loss price
	Volatility      float64 `json:"volatility"` // Market volatility (0-1 scale)
	TrustStage      string  `json:"trust_stage"`
	ModelAgeSec     int64   `json:"model_age_sec"`
	ModelUsed       string  `json:"model_used"`
}

// TickerData mirrors data-fetcher's DataPoint for ticker topic.
type TickerData struct {
	Symbol string  `json:"symbol"`
	Price  float64 `json:"price"`
	Volume float64 `json:"volume"`
	Time   int64   `json:"timestamp"`
}

// wsClient owns a single websocket connection. gorilla/websocket does not
// support concurrent writers, so every frame goes through `send` and is written
// by exactly one goroutine (writePump).
type wsClient struct {
	conn *websocket.Conn
	send chan []byte
}

var (
	// In-memory store for market pairs (simulate Redis cache)
	marketData = make(map[string]MarketPair)
	dataMutex  sync.RWMutex

	upgrader = websocket.Upgrader{
		ReadBufferSize:  1024,
		WriteBufferSize: 1024,
		CheckOrigin:     checkOrigin,
	}

	wsClients = make(map[*wsClient]bool)
	wsMutex   sync.Mutex

	// Kafka Consumers
	tickerReader    *kafka.Reader
	signalReader    *kafka.Reader
	directionReader *kafka.Reader

	// HTTP client for communicating with analytics engine
	httpClient = &http.Client{Timeout: 30 * time.Second}
	// Short-timeout client for readiness probes; the 30s client makes /readyz
	// outlive the container healthcheck interval.
	probeClient = &http.Client{Timeout: 2 * time.Second}

	// Global safety valve; per-IP limiting happens in rateLimitMiddleware.
	globalLimiter = rate.NewLimiter(500, 1000)

	ipLimiters = make(map[string]*ipLimiterEntry)
	ipMutex    sync.Mutex
)

type ipLimiterEntry struct {
	limiter  *rate.Limiter
	lastSeen time.Time
}

func checkOrigin(r *http.Request) bool {
	origin := r.Header.Get("Origin")
	if origin == "" {
		// Non-browser clients (curl, native app) do not send Origin.
		return true
	}

	allowed := os.Getenv("WS_ALLOWED_ORIGINS")
	if allowed == "" {
		allowed = "http://localhost:3000,http://127.0.0.1:3000"
	}
	for _, candidate := range strings.Split(allowed, ",") {
		if strings.EqualFold(strings.TrimSpace(candidate), origin) {
			return true
		}
	}
	log.Printf("Rejected websocket origin: %s", origin)
	return false
}

func analyticsBaseURL() string {
	if v := os.Getenv("ANALYTICS_ENGINE_URL"); v != "" {
		return strings.TrimRight(v, "/")
	}
	return "http://analytics-engine:8081"
}

func main() {
	if os.Getenv("GIN_MODE") == "" {
		gin.SetMode(gin.ReleaseMode)
	}

	log.Println("\U0001F680 API Gateway starting...")

	r := gin.New()
	r.Use(gin.Recovery())
	// Skip access logs for probes: they run every few seconds and drown the log.
	r.Use(gin.LoggerWithConfig(gin.LoggerConfig{SkipPaths: []string{"/health", "/healthz", "/readyz"}}))
	r.Use(securityHeadersMiddleware())
	r.Use(rateLimitMiddleware())

	kafkaBrokers := os.Getenv("KAFKA_BROKERS")
	if kafkaBrokers == "" {
		kafkaBrokers = "kafka:9092"
	}

	// MinBytes 10e3 + the default 10s MaxWait meant a ticker could sit in the
	// broker for seconds before the gateway saw it. Realtime data wants MinBytes 1.
	newReader := func(topic, group string) *kafka.Reader {
		return kafka.NewReader(kafka.ReaderConfig{
			Brokers:       strings.Split(kafkaBrokers, ","),
			Topic:         topic,
			GroupID:       group,
			MinBytes:      1,
			MaxBytes:      1e6,
			MaxWait:       500 * time.Millisecond,
			MaxAttempts:   10,
			QueueCapacity: 100,
		})
	}

	tickerReader = newReader("ticker", "api-gateway-ticker-group")
	signalReader = newReader("pump_signals", "api-gateway-signals-group")
	directionReader = newReader("direction_signals", "api-gateway-direction-group")

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
	r.GET("/api/v1/infrastructure/metrics", getInfrastructureMetrics)

	// Proxy trader-mind endpoints to analytics-engine
	r.GET("/api/v1/trader-mind/:symbol", proxyTraderMind)
	r.GET("/api/v1/trader-mind/full/:symbol", proxyTraderMindFull)
	r.POST("/api/v1/model/retrain", proxyModelRetrain)

	// WebSocket Endpoint
	r.GET("/ws", wsHandler)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go consumeTickers(ctx)
	go consumePumpSignals(ctx)
	go consumeDirectionSignals(ctx)
	go broadcastMarketData(ctx)
	go cleanupIPLimiters(ctx)

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	apiGatewayPort := os.Getenv("API_GATEWAY_PORT")
	if apiGatewayPort == "" {
		apiGatewayPort = "8080"
	}
	srv := &http.Server{
		Addr:    ":" + apiGatewayPort,
		Handler: r,
		// No WriteTimeout: it would also cap hijacked websocket connections.
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       120 * time.Second,
	}

	go func() {
		log.Printf("\U0001F310 API Gateway listening on :%s", apiGatewayPort)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("\u274C Failed to run API Gateway: %v", err)
		}
	}()

	sig := <-sigChan
	log.Printf("\U0001F6D1 Received signal %v, initiating graceful shutdown...", sig)

	cancel()

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer shutdownCancel()

	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Printf("\u274C HTTP server forced to shutdown: %v", err)
	} else {
		log.Println("\u2705 HTTP server stopped gracefully")
	}

	log.Println("\U0001F512 Closing Kafka readers...")
	for name, reader := range map[string]*kafka.Reader{
		"ticker":    tickerReader,
		"signals":   signalReader,
		"direction": directionReader,
	} {
		if reader == nil {
			continue
		}
		if err := reader.Close(); err != nil {
			log.Printf("\u274C Error closing %s reader: %v", name, err)
		}
	}
	log.Println("\u2705 Kafka readers closed")

	wsMutex.Lock()
	for client := range wsClients {
		delete(wsClients, client)
		close(client.send)
	}
	wsMutex.Unlock()
	log.Println("\u2705 WebSocket connections closed")

	log.Println("\u2705 API Gateway stopped gracefully")
}

func handleHealthz(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status": "ok",
		"time":   time.Now().UTC(),
	})
}

func handleReadyz(c *gin.Context) {
	kafkaReady := tickerReader != nil && signalReader != nil && directionReader != nil

	analyticsReady := false
	req, err := http.NewRequestWithContext(c.Request.Context(), http.MethodGet, analyticsBaseURL()+"/readyz", nil)
	if err == nil {
		if resp, err := probeClient.Do(req); err == nil {
			analyticsReady = resp.StatusCode == http.StatusOK
			_, _ = io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
		}
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

func limiterForIP(ip string) *rate.Limiter {
	ipMutex.Lock()
	defer ipMutex.Unlock()

	entry, ok := ipLimiters[ip]
	if !ok {
		entry = &ipLimiterEntry{limiter: rate.NewLimiter(50, 100)}
		ipLimiters[ip] = entry
	}
	entry.lastSeen = time.Now()
	return entry.limiter
}

// cleanupIPLimiters keeps the per-IP limiter map from growing without bound.
func cleanupIPLimiters(ctx context.Context) {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			cutoff := time.Now().Add(-15 * time.Minute)
			ipMutex.Lock()
			for ip, entry := range ipLimiters {
				if entry.lastSeen.Before(cutoff) {
					delete(ipLimiters, ip)
				}
			}
			ipMutex.Unlock()
		}
	}
}

// rateLimitMiddleware applies a per-client budget plus a global safety valve.
// A single global bucket let one noisy client starve everybody else.
func rateLimitMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		if !globalLimiter.Allow() || !limiterForIP(c.ClientIP()).Allow() {
			c.JSON(http.StatusTooManyRequests, gin.H{"error": "Rate limit exceeded"})
			c.Abort()
			return
		}
		c.Next()
	}
}

// consumeLoop is the shared body of the three Kafka consumers.
func consumeLoop(ctx context.Context, name string, reader *kafka.Reader, handle func(kafka.Message)) {
	log.Printf("Consuming %s from Kafka...", name)

	backoff := 100 * time.Millisecond
	const maxBackoff = 30 * time.Second

	for {
		m, err := reader.FetchMessage(ctx)
		if err != nil {
			if ctx.Err() != nil || errors.Is(err, context.Canceled) || errors.Is(err, io.EOF) {
				log.Printf("%s consumer stopped.", name)
				return
			}
			log.Printf("Error fetching %s message from Kafka: %v (retry in %v)", name, err, backoff)
			select {
			case <-ctx.Done():
				return
			case <-time.After(backoff):
			}
			if backoff < maxBackoff {
				backoff *= 2
			}
			continue
		}
		backoff = 100 * time.Millisecond

		handle(m)

		if err := reader.CommitMessages(ctx, m); err != nil && ctx.Err() == nil {
			log.Printf("Error committing %s message: %v", name, err)
		}
	}
}

func consumeTickers(ctx context.Context) {
	consumeLoop(ctx, "ticker", tickerReader, func(m kafka.Message) {
		var td TickerData
		if err := json.Unmarshal(m.Value, &td); err != nil {
			log.Printf("Error unmarshalling ticker data: %v", err)
			return
		}

		dataMutex.Lock()
		existing := marketData[td.Symbol]
		existing.Symbol = td.Symbol
		existing.Price = td.Price
		existing.Volume = td.Volume
		existing.LastUpdate = time.Now().Unix()
		marketData[td.Symbol] = existing
		dataMutex.Unlock()
	})
}

func consumePumpSignals(ctx context.Context) {
	consumeLoop(ctx, "pump signals", signalReader, func(m kafka.Message) {
		var sig PumpSignal
		if err := json.Unmarshal(m.Value, &sig); err != nil {
			log.Printf("Error unmarshalling pump signal: %v", err)
			return
		}

		dataMutex.Lock()
		existing, ok := marketData[sig.Symbol]
		if !ok {
			existing = MarketPair{Symbol: sig.Symbol}
		}
		existing.AnomalyScore = sig.Probability * 100 // 0-100 scale
		existing.LastUpdate = time.Now().Unix()
		marketData[sig.Symbol] = existing
		dataMutex.Unlock()

		broadcastWebSocketMessage(map[string]interface{}{
			"type": "pump_signal_update",
			"data": sig,
		})
	})
}

func consumeDirectionSignals(ctx context.Context) {
	consumeLoop(ctx, "ML direction signals", directionReader, func(m kafka.Message) {
		var dirSignal DirectionSignal
		if err := json.Unmarshal(m.Value, &dirSignal); err != nil {
			log.Printf("Error unmarshalling direction signal: %v", err)
			return
		}

		timestampUnix := dirSignal.Timestamp
		if timestampUnix == 0 {
			timestampUnix = time.Now().Unix()
		}

		resolvedModel := dirSignal.ModelUsed
		if resolvedModel == "" {
			resolvedModel = "SimpleNN"
		}

		log.Printf("\U0001F916 ML SIGNAL: %s %s (%.1f%% confidence) - Target: %.8f",
			dirSignal.Symbol, dirSignal.Direction, dirSignal.Confidence*100, dirSignal.PriceTarget)

		broadcastWebSocketMessage(map[string]interface{}{
			"type": "direction_signal",
			"data": map[string]interface{}{
				"symbol":            dirSignal.Symbol,
				"direction":         dirSignal.Direction,
				"confidence":        dirSignal.Confidence,
				"class_probs":       dirSignal.ClassProbs,
				"price_target":      dirSignal.PriceTarget,
				"current_price":     dirSignal.CurrentPrice,
				"time_horizon":      dirSignal.TimeHorizon,
				"label_horizon_min": dirSignal.LabelHorizonMin,
				"timestamp":         timestampUnix,
				"stop_loss":         dirSignal.StopLoss,
				"volatility":        dirSignal.Volatility,
				"trust_stage":       dirSignal.TrustStage,
				"model_age_sec":     dirSignal.ModelAgeSec,
				"model_used":        resolvedModel,
			},
		})
	})
}

func marketSnapshot() []MarketPair {
	dataMutex.RLock()
	defer dataMutex.RUnlock()
	list := make([]MarketPair, 0, len(marketData))
	for _, p := range marketData {
		list = append(list, p)
	}
	return list
}

func getAllPairs(c *gin.Context) {
	c.JSON(http.StatusOK, marketSnapshot())
}

func getPair(c *gin.Context) {
	symbol := c.Param("symbol")
	if err := validator.ValidateSymbol(symbol); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid symbol format"})
		return
	}
	dataMutex.RLock()
	p, ok := marketData[symbol]
	dataMutex.RUnlock()
	if !ok {
		c.JSON(http.StatusNotFound, gin.H{"error": "not found"})
		return
	}
	c.JSON(http.StatusOK, p)
}

func scanMarket(c *gin.Context) {
	getAllPairs(c)
}

func removeClient(client *wsClient) {
	wsMutex.Lock()
	if _, ok := wsClients[client]; ok {
		delete(wsClients, client)
		close(client.send)
	}
	wsMutex.Unlock()
}

func (client *wsClient) writePump() {
	ticker := time.NewTicker(wsPingPeriod)
	defer func() {
		ticker.Stop()
		_ = client.conn.Close()
	}()

	for {
		select {
		case payload, ok := <-client.send:
			_ = client.conn.SetWriteDeadline(time.Now().Add(wsWriteWait))
			if !ok {
				_ = client.conn.WriteMessage(websocket.CloseMessage, []byte{})
				return
			}
			if err := client.conn.WriteMessage(websocket.TextMessage, payload); err != nil {
				return
			}
		case <-ticker.C:
			_ = client.conn.SetWriteDeadline(time.Now().Add(wsWriteWait))
			if err := client.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}

func (client *wsClient) readPump() {
	defer func() {
		removeClient(client)
		_ = client.conn.Close()
	}()

	client.conn.SetReadLimit(wsMaxMessageSize)
	_ = client.conn.SetReadDeadline(time.Now().Add(wsPongWait))
	client.conn.SetPongHandler(func(string) error {
		return client.conn.SetReadDeadline(time.Now().Add(wsPongWait))
	})

	for {
		// ReadMessage blocks until data or error, so the old time.Sleep here only
		// delayed disconnect detection.
		if _, _, err := client.conn.ReadMessage(); err != nil {
			return
		}
	}
}

func wsHandler(c *gin.Context) {
	conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		log.Printf("WebSocket upgrade error: %v", err)
		return
	}

	client := &wsClient{conn: conn, send: make(chan []byte, wsSendBuffer)}

	initial, err := json.Marshal(map[string]interface{}{
		"type": "initial_data",
		"data": marketSnapshot(),
	})
	if err != nil {
		log.Printf("Error encoding initial data: %v", err)
		_ = conn.Close()
		return
	}
	client.send <- initial

	wsMutex.Lock()
	wsClients[client] = true
	wsMutex.Unlock()

	go client.writePump()
	client.readPump()
}

// broadcastMarketData broadcasts current market data to all connected clients.
func broadcastMarketData(ctx context.Context) {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			log.Println("Broadcast goroutine stopped")
			return
		case <-ticker.C:
			broadcastWebSocketMessage(map[string]interface{}{
				"type": "market_update",
				"data": marketSnapshot(),
			})
		}
	}
}

// broadcastWebSocketMessage queues a message for every connected client.
// It never writes to a socket directly, so a slow or dead client can no longer
// block the broadcaster (or race with another writer).
func broadcastWebSocketMessage(message interface{}) {
	payload, err := json.Marshal(message)
	if err != nil {
		log.Printf("Error encoding websocket message: %v", err)
		return
	}

	wsMutex.Lock()
	defer wsMutex.Unlock()
	for client := range wsClients {
		select {
		case client.send <- payload:
		default:
			// Client cannot keep up: drop it instead of stalling everyone.
			delete(wsClients, client)
			close(client.send)
		}
	}
}

// proxyJSON forwards a request to the analytics engine and relays the JSON body.
func proxyJSON(c *gin.Context, method, target string, body io.Reader, unavailable gin.H) {
	req, err := http.NewRequestWithContext(c.Request.Context(), method, target, body)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to build upstream request"})
		return
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		log.Printf("Error contacting analytics engine (%s): %v", target, err)
		payload := gin.H{"error": "Analytics Engine unavailable"}
		for k, v := range unavailable {
			payload[k] = v
		}
		c.JSON(http.StatusServiceUnavailable, payload)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		log.Printf("Analytics engine returned %d for %s", resp.StatusCode, target)
		payload := gin.H{"error": "Analytics Engine returned error", "status": resp.StatusCode}
		for k, v := range unavailable {
			payload[k] = v
		}
		c.JSON(resp.StatusCode, payload)
		return
	}

	var out map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		log.Printf("Error decoding analytics engine response (%s): %v", target, err)
		c.JSON(http.StatusBadGateway, gin.H{"error": "invalid response from analytics engine"})
		return
	}

	c.JSON(http.StatusOK, out)
}

// submitFeedback handles manual feedback submission for continuous learning
func submitFeedback(c *gin.Context) {
	// NOTE: no `binding:"required"` on the numeric/bool fields. Gin's `required`
	// rejects zero values, so `actual_pump: false` and `predicted_prob: 0` were
	// impossible to submit. They are validated explicitly below instead.
	var feedback struct {
		Symbol        string  `json:"symbol" binding:"required"`
		Timestamp     int64   `json:"timestamp"`
		PredictedProb float64 `json:"predicted_prob"`
		ActualPump    bool    `json:"actual_pump"`
		Confidence    float64 `json:"confidence"`
		Notes         string  `json:"notes"`
	}

	if err := c.ShouldBindJSON(&feedback); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid feedback data: " + err.Error()})
		return
	}

	if err := validator.ValidateSymbol(feedback.Symbol); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if feedback.Timestamp == 0 {
		feedback.Timestamp = time.Now().Unix()
	}
	if err := validator.ValidateTimestamp(feedback.Timestamp); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if err := validator.ValidateConfidence(feedback.PredictedProb); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "predicted_prob: " + err.Error()})
		return
	}
	if feedback.Confidence == 0 {
		feedback.Confidence = 1.0
	}
	if err := validator.ValidateConfidence(feedback.Confidence); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "confidence: " + err.Error()})
		return
	}

	jsonData, err := json.Marshal(feedback)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to serialize feedback"})
		return
	}

	proxyJSON(c, http.MethodPost, analyticsBaseURL()+"/api/v1/feedback", bytes.NewReader(jsonData), nil)
}

func getPerformanceMetrics(c *gin.Context) {
	proxyJSON(c, http.MethodGet, analyticsBaseURL()+"/model/performance", nil, gin.H{"status": "down"})
}

func getModelStats(c *gin.Context) {
	proxyJSON(c, http.MethodGet, analyticsBaseURL()+"/model/performance", nil, gin.H{"data_status": "empty"})
}

func getMLMetrics(c *gin.Context) {
	proxyJSON(c, http.MethodGet, analyticsBaseURL()+"/api/v1/ml/metrics", nil, gin.H{"system": gin.H{"overall_health": "UNKNOWN"}})
}

func getCalibrationStatus(c *gin.Context) {
	proxyJSON(c, http.MethodGet, analyticsBaseURL()+"/api/v1/ml/calibration", nil, gin.H{"system": gin.H{"overall_status": "UNKNOWN"}})
}

func startAutoCalibration(c *gin.Context) {
	proxyJSON(c, http.MethodPost, analyticsBaseURL()+"/api/v1/ml/calibration/start", nil, nil)
}

func getInfrastructureMetrics(c *gin.Context) {
	proxyJSON(c, http.MethodGet, analyticsBaseURL()+"/api/v1/infrastructure/metrics", nil, nil)
}

func getTrainingHistory(c *gin.Context) {
	symbol := c.DefaultQuery("symbol", "BTCUSDT")
	if err := validator.ValidateSymbol(symbol); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid symbol format"})
		return
	}
	target := fmt.Sprintf("%s/api/v1/ml/training-history?symbol=%s&limit=%s",
		analyticsBaseURL(), url.QueryEscape(symbol), url.QueryEscape(c.DefaultQuery("limit", "50")))
	proxyJSON(c, http.MethodGet, target, nil, nil)
}

func getSignalStats(c *gin.Context) {
	symbol := c.DefaultQuery("symbol", "BTCUSDT")
	if err := validator.ValidateSymbol(symbol); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid symbol format"})
		return
	}
	target := fmt.Sprintf("%s/api/v1/ml/signal-stats?symbol=%s&hours=%s",
		analyticsBaseURL(), url.QueryEscape(symbol), url.QueryEscape(c.DefaultQuery("hours", "24")))
	proxyJSON(c, http.MethodGet, target, nil, nil)
}

func getRecentSignals(c *gin.Context) {
	query := url.Values{}
	query.Set("limit", c.DefaultQuery("limit", "50"))
	query.Set("hours", c.DefaultQuery("hours", "24"))
	if symbol := c.Query("symbol"); symbol != "" {
		if err := validator.ValidateSymbol(symbol); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid symbol format"})
			return
		}
		query.Set("symbol", symbol)
	}
	proxyJSON(c, http.MethodGet, analyticsBaseURL()+"/api/v1/ml/signals/recent?"+query.Encode(), nil, nil)
}

func getSignalsHistory(c *gin.Context) {
	query := url.Values{}
	query.Set("limit", c.DefaultQuery("limit", "100"))
	query.Set("hours", c.DefaultQuery("hours", "168"))
	if symbol := c.Query("symbol"); symbol != "" {
		if err := validator.ValidateSymbol(symbol); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid symbol format"})
			return
		}
		query.Set("symbol", symbol)
	}
	if direction := c.Query("direction"); direction != "" {
		switch strings.ToUpper(direction) {
		case "UP", "DOWN", "SIDEWAYS":
			query.Set("direction", strings.ToUpper(direction))
		default:
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid direction"})
			return
		}
	}
	proxyJSON(c, http.MethodGet, analyticsBaseURL()+"/api/v1/ml/signals/history?"+query.Encode(), nil, nil)
}

func proxyTraderMind(c *gin.Context) {
	symbol := c.Param("symbol")
	if err := validator.ValidateSymbol(symbol); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid symbol format"})
		return
	}
	proxyJSON(c, http.MethodGet, analyticsBaseURL()+"/api/v1/trader-mind/"+url.PathEscape(symbol), nil, nil)
}

func proxyTraderMindFull(c *gin.Context) {
	symbol := c.Param("symbol")
	if err := validator.ValidateSymbol(symbol); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid symbol format"})
		return
	}
	proxyJSON(c, http.MethodGet, analyticsBaseURL()+"/api/v1/trader-mind/full/"+url.PathEscape(symbol), nil, nil)
}

func proxyModelRetrain(c *gin.Context) {
	target := analyticsBaseURL() + "/api/v1/model/retrain"
	if symbol := c.Query("symbol"); symbol != "" {
		if err := validator.ValidateSymbol(symbol); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid symbol format"})
			return
		}
		target += "?symbol=" + url.QueryEscape(symbol)
	}
	proxyJSON(c, http.MethodPost, target, nil, nil)
}
