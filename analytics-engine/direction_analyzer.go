package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"log"
	"math"
	"math/rand"
	"sync"
	"time"

	"strings"

	_ "github.com/lib/pq"
	"github.com/segmentio/kafka-go"
)

// DirectionPrediction represents prediction for next hour price direction
type DirectionPrediction struct {
	Symbol       string            `json:"symbol"`
	Timestamp    time.Time         `json:"timestamp"`
	Direction    string            `json:"direction"`    // "UP", "DOWN", "SIDEWAYS"
	Confidence   float64           `json:"confidence"`   // 0.0 - 1.0
	PriceTarget  float64           `json:"price_target"` // Expected price
	CurrentPrice float64           `json:"current_price"`
	TimeHorizon  int               `json:"time_horizon"` // Minutes (60 for 1 hour)
	Features     DirectionFeatures `json:"features"`
	CreatedAt    time.Time         `json:"created_at"`
}

// DirectionFeatures contains all features for direction prediction
type DirectionFeatures struct {
	// Price Action Features
	PriceMA5      float64 `json:"price_ma5"`
	PriceMA20     float64 `json:"price_ma20"`
	PriceMA50     float64 `json:"price_ma50"`
	PricePosition float64 `json:"price_position"` // Current price relative to MAs

	// Volume Features
	VolumeMA20    float64 `json:"volume_ma20"`
	VolumeRatio   float64 `json:"volume_ratio"`   // Current vs average
	VolumeProfile float64 `json:"volume_profile"` // Volume distribution

	// Momentum Features
	RSI14      float64 `json:"rsi14"`
	MACD       float64 `json:"macd"`
	MACDSignal float64 `json:"macd_signal"`
	MACDHist   float64 `json:"macd_hist"`

	// Volatility Features
	ATR14          float64 `json:"atr14"` // Average True Range
	BollingerUpper float64 `json:"bollinger_upper"`
	BollingerLower float64 `json:"bollinger_lower"`
	BollingerPos   float64 `json:"bollinger_pos"` // Position in bands

	// Support/Resistance Features
	SupportLevel    float64 `json:"support_level"`
	ResistanceLevel float64 `json:"resistance_level"`
	SRStrength      float64 `json:"sr_strength"` // S/R strength

	// Order Flow Features (derived from trades/orderbook)
	BuyPressure    float64 `json:"buy_pressure"`
	SellPressure   float64 `json:"sell_pressure"`
	OrderImbalance float64 `json:"order_imbalance"`

	// Market Structure Features
	TrendStrength  float64 `json:"trend_strength"`
	TrendDirection float64 `json:"trend_direction"` // -1 to 1
	MarketPhase    string  `json:"market_phase"`    // "ACCUMULATION", "DISTRIBUTION", "TRENDING"
}

// DirectionAnalyzer is the main prediction engine
type DirectionAnalyzer struct {
	db            *sql.DB
	candleHistory map[string][]Candle
	historyMutex  sync.RWMutex
	predictor     *DirectionML
	kafkaProducer *kafka.Writer
	kafkaBrokers  []string
}

// DirectionML contains the ML model for direction prediction
type DirectionML struct {
	weights            map[string]float64
	biases             map[string]float64
	featureStats       map[string]FeatureStat
	learningRate       float64
	momentum           float64
	historicalAccuracy map[string]float64 // per symbol accuracy
	// Enhanced ML features
	modelWeights    []float64 // Neural network weights
	neuronCount     int       // Number of neurons in hidden layer
	trainingData    []TrainingRecord
	maxTrainingData int
}

type TrainingRecord struct {
	Features     []float64
	ActualDir    string
	PredictedDir string
	Confidence   float64
	Timestamp    time.Time
	Symbol       string
}

type FeatureStat struct {
	Mean   float64
	StdDev float64
	Min    float64
	Max    float64
	Count  int64
}

func NewDirectionAnalyzer(db *sql.DB, kafkaBrokers string) *DirectionAnalyzer {
	// Initialize database tables
	initDirectionTables(db)

	// support comma-separated list of brokers
	brokerList := []string{kafkaBrokers}
	if strings.Contains(kafkaBrokers, ",") {
		brokerList = strings.Split(kafkaBrokers, ",")
	}

	// Kafka producer for predictions
	producer := &kafka.Writer{
		Addr:     kafka.TCP(brokerList...),
		Topic:    "direction_signals", // Use same topic as main analytics engine
		Balancer: &kafka.LeastBytes{},
	}

	return &DirectionAnalyzer{
		db:            db,
		candleHistory: make(map[string][]Candle),
		predictor:     NewDirectionML(),
		kafkaProducer: producer,
		kafkaBrokers:  brokerList,
	}
}

func NewDirectionML() *DirectionML {
	ml := &DirectionML{
		weights:            make(map[string]float64),
		biases:             make(map[string]float64),
		featureStats:       make(map[string]FeatureStat),
		learningRate:       0.001,
		momentum:           0.9,
		historicalAccuracy: make(map[string]float64),
		neuronCount:        50, // Hidden layer with 50 neurons
		trainingData:       make([]TrainingRecord, 0),
		maxTrainingData:    1000, // Keep last 1000 training records
	}
	ml.initializeWeights() // Use existing method
	return ml
}

func initDirectionTables(db *sql.DB) {
	// Direction predictions table
	_, err := db.Exec(`
		CREATE TABLE IF NOT EXISTS direction_predictions (
			id BIGSERIAL PRIMARY KEY,
			symbol VARCHAR(20) NOT NULL,
			timestamp TIMESTAMPTZ NOT NULL,
			direction VARCHAR(10) NOT NULL,
			confidence DECIMAL(5,4) NOT NULL,
			price_target DECIMAL(15,8),
			current_price DECIMAL(15,8),
			time_horizon INTEGER DEFAULT 60,
			features JSONB,
			created_at TIMESTAMPTZ DEFAULT NOW(),
			actual_direction VARCHAR(10) DEFAULT NULL,
			actual_price DECIMAL(15,8) DEFAULT NULL,
			accuracy_score DECIMAL(5,4) DEFAULT NULL
		)
	`)
	if err != nil {
		log.Printf("Failed to create direction_predictions table: %v", err)
	}

	// Create indexes
	db.Exec(`CREATE INDEX IF NOT EXISTS idx_direction_symbol_time ON direction_predictions (symbol, timestamp DESC);`)
	db.Exec(`CREATE INDEX IF NOT EXISTS idx_direction_created ON direction_predictions (created_at DESC);`)
}

// ProcessCandle processes new candle data and generates predictions
func (da *DirectionAnalyzer) ProcessCandle(candle Candle) {
	da.historyMutex.Lock()

	// Maintain 1440 candles (24 hours) for each symbol
	if _, exists := da.candleHistory[candle.Symbol]; !exists {
		da.candleHistory[candle.Symbol] = make([]Candle, 0, 1440)
	}

	history := da.candleHistory[candle.Symbol]
	history = append(history, candle)

	// Keep only last 1440 candles
	if len(history) > 1440 {
		history = history[len(history)-1440:]
	}

	da.candleHistory[candle.Symbol] = history
	da.historyMutex.Unlock()

	// Generate prediction if we have enough history (reduced from 50 to 20 for faster startup)
	if len(history) >= 20 {
		prediction := da.generatePrediction(candle.Symbol, history)
		if prediction != nil {
			da.savePrediction(*prediction)
			da.publishPrediction(*prediction)
			log.Printf("🎯 Direction prediction for %s: %s (%.2f%% confidence) - Target: %.8f",
				prediction.Symbol, prediction.Direction, prediction.Confidence*100, prediction.PriceTarget)
		}
	} else {
		log.Printf("⏳ %s: collecting history (%d/20 candles needed)", candle.Symbol, len(history))
	}
}

// generatePrediction creates a direction prediction based on historical data
func (da *DirectionAnalyzer) generatePrediction(symbol string, history []Candle) *DirectionPrediction {
	if len(history) < 20 { // Reduced from 50 to 20 for faster startup
		return nil
	}

	// Extract features from last 1440 candles
	features := da.extractFeatures(history)

	// Use ML model to predict direction
	direction, confidence, priceTarget := da.predictor.Predict(features, history[len(history)-1].Close)

	return &DirectionPrediction{
		Symbol:       symbol,
		Timestamp:    time.Unix(history[len(history)-1].Timestamp, 0),
		Direction:    direction,
		Confidence:   confidence,
		PriceTarget:  priceTarget,
		CurrentPrice: history[len(history)-1].Close,
		TimeHorizon:  60, // 1 hour
		Features:     features,
		CreatedAt:    time.Now(),
	}
}

// extractFeatures extracts all features for ML prediction
func (da *DirectionAnalyzer) extractFeatures(history []Candle) DirectionFeatures {
	if len(history) < 20 { // Reduced from 50 to 20 for faster startup
		return DirectionFeatures{}
	}

	current := history[len(history)-1]

	// Price Action Features
	priceMA5 := calculateSMA(history, 5)
	priceMA20 := calculateSMA(history, 20)
	priceMA50 := calculateSMA(history, 50)

	// Volume Features
	volumeMA20 := calculateVolumeSMA(history, 20)
	volumeRatio := current.Volume / volumeMA20

	// Technical Indicators
	rsi14 := calculateRSI(history, 14)
	macd, macdSignal, macdHist := calculateMACD(history)
	atr14 := calculateATR(history, 14)

	// Bollinger Bands
	bbUpper, bbLower := calculateBollingerBands(history, 20, 2.0)
	bbPos := (current.Close - bbLower) / (bbUpper - bbLower)

	// Support/Resistance
	support, resistance := calculateSupportResistance(history)

	// Order Flow (simplified from price action)
	buyPressure, sellPressure := calculateOrderFlow(history)

	// Market Structure
	trendStrength, trendDirection := calculateTrend(history)
	marketPhase := identifyMarketPhase(history)

	return DirectionFeatures{
		PriceMA5:        priceMA5,
		PriceMA20:       priceMA20,
		PriceMA50:       priceMA50,
		PricePosition:   (current.Close - priceMA20) / priceMA20,
		VolumeMA20:      volumeMA20,
		VolumeRatio:     volumeRatio,
		VolumeProfile:   calculateVolumeProfile(history),
		RSI14:           rsi14,
		MACD:            macd,
		MACDSignal:      macdSignal,
		MACDHist:        macdHist,
		ATR14:           atr14,
		BollingerUpper:  bbUpper,
		BollingerLower:  bbLower,
		BollingerPos:    bbPos,
		SupportLevel:    support,
		ResistanceLevel: resistance,
		SRStrength:      calculateSRStrength(history, support, resistance),
		BuyPressure:     buyPressure,
		SellPressure:    sellPressure,
		OrderImbalance:  buyPressure - sellPressure,
		TrendStrength:   trendStrength,
		TrendDirection:  trendDirection,
		MarketPhase:     marketPhase,
	}
}

// Predict uses enhanced ML to predict direction
func (ml *DirectionML) Predict(features DirectionFeatures, currentPrice float64) (string, float64, float64) {
	// Convert features to normalized array
	normalizedFeatures := ml.extractFeatureVector(features)

	// Apply neural network for better prediction
	scores := ml.neuralNetworkPredict(normalizedFeatures)

	// Apply softmax for probabilities
	probs := softmax(scores)

	// Determine prediction with enhanced logic
	maxProb := math.Max(probs[0], math.Max(probs[1], probs[2]))
	directions := []string{"UP", "DOWN", "SIDEWAYS"}

	var direction string
	var priceTarget float64

	// Find direction with highest probability
	for i, prob := range probs {
		if prob == maxProb {
			direction = directions[i]
			break
		}
	}

	// Calculate price target based on volatility and momentum
	atr := features.ATR14
	if atr == 0 {
		atr = currentPrice * 0.02 // 2% default volatility
	}

	momentumFactor := (features.RSI14 - 50) / 50 // -1 to 1
	volatilityMultiplier := 1.0 + math.Abs(features.MACDHist)*2

	switch direction {
	case "UP":
		priceTarget = currentPrice * (1.0 + (atr/currentPrice)*volatilityMultiplier*(1+momentumFactor))
	case "DOWN":
		priceTarget = currentPrice * (1.0 - (atr/currentPrice)*volatilityMultiplier*(1-momentumFactor))
	default: // SIDEWAYS
		priceTarget = currentPrice
	}

	return direction, maxProb, priceTarget
}

// extractFeatureVector converts DirectionFeatures to normalized float array
func (ml *DirectionML) extractFeatureVector(features DirectionFeatures) []float64 {
	vector := []float64{
		// Normalize RSI to 0-1
		features.RSI14 / 100.0,
		// Normalize MACD histogram with tanh
		math.Tanh(features.MACDHist),
		// Bollinger position (already 0-1)
		features.BollingerPos,
		// Price position relative to MA
		math.Tanh(features.PricePosition),
		// Volume ratio (capped at 5x)
		math.Min(features.VolumeRatio/5.0, 1.0),
		// Trend strength
		features.TrendStrength,
		// Trend direction
		features.TrendDirection,
		// Order flow
		math.Tanh(features.BuyPressure),
		math.Tanh(features.SellPressure),
		// Support/Resistance strength
		features.SRStrength,
	}
	return vector
}

// neuralNetworkPredict applies simple neural network
func (ml *DirectionML) neuralNetworkPredict(features []float64) []float64 {
	// Simple feed-forward network: input -> hidden -> output
	inputSize := len(features)
	hiddenSize := ml.neuronCount
	outputSize := 3 // UP, DOWN, SIDEWAYS

	// Initialize weights if not exists
	if ml.modelWeights == nil {
		totalWeights := inputSize*hiddenSize + hiddenSize + hiddenSize*outputSize + outputSize
		ml.modelWeights = make([]float64, totalWeights)
		// Xavier initialization
		for i := range ml.modelWeights {
			ml.modelWeights[i] = (rand.Float64() - 0.5) * math.Sqrt(6.0/float64(inputSize+outputSize))
		}
	}

	// Forward pass
	hidden := make([]float64, hiddenSize)
	weightIdx := 0

	// Input to hidden
	for h := 0; h < hiddenSize; h++ {
		sum := 0.0
		for i := 0; i < inputSize; i++ {
			sum += features[i] * ml.modelWeights[weightIdx]
			weightIdx++
		}
		// Add bias
		sum += ml.modelWeights[weightIdx]
		weightIdx++
		// ReLU activation
		hidden[h] = math.Max(0, sum)
	}

	// Hidden to output
	output := make([]float64, outputSize)
	for o := 0; o < outputSize; o++ {
		sum := 0.0
		for h := 0; h < hiddenSize; h++ {
			sum += hidden[h] * ml.modelWeights[weightIdx]
			weightIdx++
		}
		// Add bias
		sum += ml.modelWeights[weightIdx]
		weightIdx++
		output[o] = sum
	}

	return output
}

func (ml *DirectionML) initializeWeights() {
	// Initialize with small random weights
	features := []string{
		"price_position", "volume_ratio", "rsi14", "macd_hist",
		"bollinger_pos", "trend_strength", "trend_direction",
		"buy_pressure", "sell_pressure", "atr14",
	}

	directions := []string{"UP", "DOWN", "SIDEWAYS"}

	for _, direction := range directions {
		for _, feature := range features {
			key := "weights_" + direction + "_" + feature
			ml.weights[key] = (rand.Float64() - 0.5) * 0.1 // Small random weights
		}
		ml.biases[direction] = 0.0
	}
}

func (da *DirectionAnalyzer) savePrediction(prediction DirectionPrediction) {
	featuresJSON, _ := json.Marshal(prediction.Features)

	_, err := da.db.Exec(`
		INSERT INTO direction_predictions 
		(symbol, timestamp, direction, confidence, price_target, current_price, 
		 time_horizon, features, created_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
	`, prediction.Symbol, prediction.Timestamp, prediction.Direction,
		prediction.Confidence, prediction.PriceTarget, prediction.CurrentPrice,
		prediction.TimeHorizon, featuresJSON, prediction.CreatedAt)

	if err != nil {
		log.Printf("Failed to save direction prediction: %v", err)
	}
}

func (da *DirectionAnalyzer) publishPrediction(prediction DirectionPrediction) {
	// Build a tolerant payload compatible with DirectionSignal consumers
	payload := map[string]interface{}{
		"symbol":        prediction.Symbol,
		"timestamp":     prediction.Timestamp.Format(time.RFC3339),
		"direction":     prediction.Direction,
		"confidence":    prediction.Confidence,
		"price_target":  prediction.PriceTarget,
		"current_price": prediction.CurrentPrice,
		"time_horizon":  prediction.TimeHorizon,
		"features":      prediction.Features,
		// Provide model identifiers so downstream consumers (api-gateway, frontend)
		// can display which model produced the signal. Mark these as traditional.
		"model_type": "TRADITIONAL",
		"model_used": "DirectionAnalyzer",
	}

	data, err := json.Marshal(payload)
	if err != nil {
		log.Printf("Failed to marshal prediction payload: %v", err)
		return
	}

	err = da.kafkaProducer.WriteMessages(context.Background(),
		kafka.Message{
			Key:   []byte(prediction.Symbol),
			Value: data,
		})

	if err != nil {
		log.Printf("Failed to publish prediction: %v", err)
	} else {
		log.Printf("📡 Published prediction to Kafka: %s %s (%.2f%%)", prediction.Symbol, prediction.Direction, prediction.Confidence*100)
		// also publish per-model analysis to model_analyses topic and DB
		payload := map[string]interface{}{
			"symbol":     prediction.Symbol,
			"model_name": "DirectionAnalyzer",
			"prediction": prediction.Direction,
			"confidence": prediction.Confidence,
			"timestamp":  prediction.CreatedAt.Format(time.RFC3339),
			"features":   prediction.Features,
		}
		go PublishModelAnalysisDBAndKafka(context.Background(), da.db, da.kafkaBrokers, payload)
	}
}

// Helper functions for technical indicators
func softmax(scores []float64) []float64 {
	max := scores[0]
	for _, score := range scores {
		if score > max {
			max = score
		}
	}

	var sum float64
	result := make([]float64, len(scores))

	for i, score := range scores {
		result[i] = math.Exp(score - max)
		sum += result[i]
	}

	for i := range result {
		result[i] /= sum
	}

	return result
}
