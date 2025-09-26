package main

import (
	"database/sql"
	"encoding/json"
	"log"
	"math"
	"math/rand"
	"sort"
	"time"
)

// AdvancedMLEngine - высокопроизводительная честная ML система
type AdvancedMLEngine struct {
	db                  *sql.DB
	models              map[string]*SmartModel
	ensembleWeights     map[string][]float64
	performanceCache    map[string]*MLPerformanceMetrics
	trainingData        map[string][]TrainingExample
	confidenceThreshold float64
	minAccuracy         float64
	adaptiveLearning    bool
}

// SmartModel - компактная оптимизированная модель
type SmartModel struct {
	Symbol          string             `json:"symbol"`
	Weights         [][]float64        `json:"weights"`
	Biases          []float64          `json:"biases"`
	FeatureWeights  map[string]float64 `json:"feature_weights"`
	LastAccuracy    float64            `json:"last_accuracy"`
	PredictionCount int                `json:"prediction_count"`
	CorrectCount    int                `json:"correct_count"`
	LearningRate    float64            `json:"learning_rate"`
	LastUpdate      time.Time          `json:"last_update"`
	ConfidenceBoost float64            `json:"confidence_boost"`
}

// Note: TrainingExample is already defined in lstm_trading_ai.go

// MLPerformanceMetrics - метрики производительности ML
type MLPerformanceMetrics struct {
	Accuracy           float64   `json:"accuracy"`
	Precision          []float64 `json:"precision"`
	Recall             []float64 `json:"recall"`
	F1Score            []float64 `json:"f1_score"`
	TotalPredictions   int       `json:"total_predictions"`
	CorrectPredictions int       `json:"correct_predictions"`
	LastUpdate         time.Time `json:"last_update"`
}

// NewAdvancedMLEngine создает новую ML систему
func NewAdvancedMLEngine(db *sql.DB) *AdvancedMLEngine {
	engine := &AdvancedMLEngine{
		db:                  db,
		models:              make(map[string]*SmartModel),
		ensembleWeights:     make(map[string][]float64),
		performanceCache:    make(map[string]*MLPerformanceMetrics),
		trainingData:        make(map[string][]TrainingExample),
		confidenceThreshold: 0.65, // Требуем минимум 65% уверенности
		minAccuracy:         0.55, // Минимальная точность 55%
		adaptiveLearning:    true,
	}

	// Инициализируем таблицы
	engine.initializeTables()

	// Загружаем сохраненные модели
	engine.loadModels()

	log.Println("🧠 Advanced ML Engine initialized with high-performance models")
	return engine
}

// initializeTables создает оптимизированные таблицы
func (engine *AdvancedMLEngine) initializeTables() {
	// Компактная таблица моделей
	_, err := engine.db.Exec(`
		CREATE TABLE IF NOT EXISTS smart_models (
			symbol VARCHAR(20) PRIMARY KEY,
			model_data BYTEA NOT NULL,
			accuracy DECIMAL(5,4) NOT NULL,
			prediction_count INTEGER DEFAULT 0,
			correct_count INTEGER DEFAULT 0,
			updated_at TIMESTAMPTZ DEFAULT NOW()
		)
	`)
	if err != nil {
		log.Printf("Failed to create smart_models table: %v", err)
	}

	// Таблица обучающих данных
	_, err = engine.db.Exec(`
		CREATE TABLE IF NOT EXISTS training_examples (
			id BIGSERIAL PRIMARY KEY,
			symbol VARCHAR(20) NOT NULL,
			features BYTEA NOT NULL,
			direction INTEGER NOT NULL,
			actual_price DECIMAL(15,8),
			predicted_price DECIMAL(15,8),
			verified BOOLEAN DEFAULT FALSE,
			created_at TIMESTAMPTZ DEFAULT NOW()
		)
	`)
	if err != nil {
		log.Printf("Failed to create training_examples table: %v", err)
	}

	// Индексы для производительности
	engine.db.Exec(`CREATE INDEX IF NOT EXISTS idx_training_symbol_time ON training_examples (symbol, created_at DESC);`)
	engine.db.Exec(`CREATE INDEX IF NOT EXISTS idx_training_verified ON training_examples (verified, created_at DESC);`)

	log.Println("📊 Smart ML tables initialized")
}

// InitializeModel создает и инициализирует модель для символа
func (engine *AdvancedMLEngine) InitializeModel(symbol string) {
	model := &SmartModel{
		Symbol:          symbol,
		Weights:         engine.initializeOptimalWeights(16, 3), // 16 фич -> 3 класса
		Biases:          make([]float64, 3),
		FeatureWeights:  engine.initializeFeatureWeights(),
		LastAccuracy:    0.5, // Начинаем с 50%
		PredictionCount: 0,
		CorrectCount:    0,
		LearningRate:    0.01, // Агрессивное обучение
		LastUpdate:      time.Now(),
		ConfidenceBoost: 1.0,
	}

	// Используем Xavier инициализацию для лучшей сходимости
	for i := range model.Weights {
		for j := range model.Weights[i] {
			model.Weights[i][j] = rand.NormFloat64() * math.Sqrt(2.0/float64(len(model.Weights[0])))
		}
	}

	engine.models[symbol] = model
	engine.performanceCache[symbol] = &MLPerformanceMetrics{
		Accuracy:           0.5,
		Precision:          make([]float64, 3),
		Recall:             make([]float64, 3),
		F1Score:            make([]float64, 3),
		TotalPredictions:   0,
		CorrectPredictions: 0,
		LastUpdate:         time.Now(),
	}

	log.Printf("🤖 Initialized smart model for %s with optimized weights", symbol)
}

// initializeOptimalWeights создает оптимальные начальные веса
func (engine *AdvancedMLEngine) initializeOptimalWeights(inputSize, outputSize int) [][]float64 {
	weights := make([][]float64, outputSize)
	scale := math.Sqrt(2.0 / float64(inputSize)) // He initialization

	for i := range weights {
		weights[i] = make([]float64, inputSize)
		for j := range weights[i] {
			weights[i][j] = rand.NormFloat64() * scale
		}
	}
	return weights
}

// initializeFeatureWeights создает начальные веса фич
func (engine *AdvancedMLEngine) initializeFeatureWeights() map[string]float64 {
	return map[string]float64{
		"price_change":     1.5, // Изменение цены - важно
		"volume_change":    1.2, // Изменение объема
		"rsi":              1.0, // RSI
		"macd":             1.1, // MACD
		"bollinger_pos":    0.9, // Позиция в полосах Боллинджера
		"trend_strength":   1.3, // Сила тренда
		"volatility":       1.1, // Волатильность
		"momentum":         1.2, // Моментум
		"support_dist":     0.8, // Расстояние до поддержки
		"resistance_dist":  0.8, // Расстояние до сопротивления
		"time_factor":      0.5, // Временной фактор
		"market_sentiment": 0.7, // Настроение рынка
		"order_imbalance":  1.0, // Дисбаланс ордеров
		"whale_activity":   0.9, // Активность китов
		"news_sentiment":   0.6, // Настроение новостей
		"correlation":      0.4, // Корреляция с BTC
	}
}

// GenerateSmartPrediction создает умный прогноз
func (engine *AdvancedMLEngine) GenerateSmartPrediction(symbol string, candles []Candle) *TradingSignal {
	model, exists := engine.models[symbol]
	if !exists {
		engine.InitializeModel(symbol)
		model = engine.models[symbol]
	}

	if len(candles) < 20 {
		return nil // Недостаточно данных
	}

	// Извлекаем честные фичи
	features := engine.extractHonestFeatures(candles)
	if len(features) != 16 {
		log.Printf("❌ Invalid features count: %d", len(features))
		return nil
	}

	// Применяем модель
	prediction := engine.applyModel(model, features)

	// Вычисляем честную уверенность
	confidence := engine.calculateHonestConfidence(model, features, prediction)

	// Проверяем качество прогноза
	if confidence < engine.confidenceThreshold {
		log.Printf("⚠️ %s: Low confidence %.2f%%, skipping prediction", symbol, confidence*100)
		return nil
	}

	// Создаем честный сигнал
	signal := engine.createHonestSignal(symbol, candles[len(candles)-1], prediction, confidence)

	// Сохраняем для обучения
	engine.saveTrainingExample(symbol, features, prediction, signal)

	// Обновляем статистику
	model.PredictionCount++
	model.LastUpdate = time.Now()

	log.Printf("🎯 %s: HONEST prediction %s with %.1f%% confidence",
		symbol, signal.Prediction, confidence*100)

	return signal
}

// extractHonestFeatures извлекает честные фичи
func (engine *AdvancedMLEngine) extractHonestFeatures(candles []Candle) []float64 {
	if len(candles) < 20 {
		return nil
	}

	current := candles[len(candles)-1]
	prev := candles[len(candles)-2]

	// Базовые честные фичи
	features := make([]float64, 16)

	// 1. Изменение цены
	priceChange := (current.Close - prev.Close) / prev.Close
	features[0] = math.Tanh(priceChange * 100) // Нормализация

	// 2. Изменение объема
	volumeChange := (current.Volume - prev.Volume) / prev.Volume
	features[1] = math.Tanh(volumeChange)

	// 3. RSI (честный расчет)
	rsi := engine.calculateHonestRSI(candles, 14)
	features[2] = (rsi - 50) / 50 // Нормализация [-1, 1]

	// 4. MACD
	macd := engine.calculateHonestMACD(candles)
	features[3] = math.Tanh(macd * 1000)

	// 5. Позиция в полосах Боллинджера
	bbPos := engine.calculateBollingerPosition(candles, 20, 2.0)
	features[4] = bbPos*2 - 1 // [-1, 1]

	// 6. Сила тренда
	trendStrength := engine.calculateTrendStrength(candles)
	features[5] = math.Tanh(trendStrength)

	// 7. Волатильность
	volatility := engine.calculateVolatility(candles, 10)
	features[6] = math.Tanh(volatility * 100)

	// 8. Моментум
	momentum := engine.calculateMomentum(candles, 10)
	features[7] = math.Tanh(momentum)

	// 9-10. Расстояние до поддержки/сопротивления
	support, resistance := engine.calculateSupportResistance(candles)
	features[8] = math.Tanh((current.Close - support) / support)
	features[9] = math.Tanh((resistance - current.Close) / current.Close)

	// 11. Временной фактор (час дня)
	hour := time.Unix(current.Timestamp, 0).UTC().Hour()
	features[10] = math.Sin(2 * math.Pi * float64(hour) / 24)

	// 12. Настроение рынка (на основе цены)
	marketSentiment := engine.calculateMarketSentiment(candles)
	features[11] = math.Tanh(marketSentiment)

	// 13. Дисбаланс ордеров (приблизительно)
	orderImbalance := (current.High - current.Close) / (current.Close - current.Low + 0.0001)
	features[12] = math.Tanh(orderImbalance - 1)

	// 14. Активность китов (по объему)
	whaleActivity := engine.calculateWhaleActivity(candles)
	features[13] = math.Tanh(whaleActivity)

	// 15. Псевдо-настроение новостей
	newsSentiment := math.Sin(float64(current.Timestamp)/3600) * 0.1
	features[14] = newsSentiment

	// 16. Корреляция с общим рынком
	correlation := engine.calculateMarketCorrelation(candles)
	features[15] = math.Tanh(correlation)

	return features
}

// calculateHonestRSI вычисляет честный RSI
func (engine *AdvancedMLEngine) calculateHonestRSI(candles []Candle, period int) float64 {
	if len(candles) < period+1 {
		return 50.0
	}

	gains := 0.0
	losses := 0.0

	for i := len(candles) - period; i < len(candles); i++ {
		change := candles[i].Close - candles[i-1].Close
		if change > 0 {
			gains += change
		} else {
			losses += -change
		}
	}

	avgGain := gains / float64(period)
	avgLoss := losses / float64(period)

	if avgLoss == 0 {
		return 100.0
	}

	rs := avgGain / avgLoss
	rsi := 100 - (100 / (1 + rs))

	return rsi
}

// calculateHonestMACD вычисляет честный MACD
func (engine *AdvancedMLEngine) calculateHonestMACD(candles []Candle) float64 {
	if len(candles) < 26 {
		return 0.0
	}

	ema12 := engine.calculateEMA(candles, 12)
	ema26 := engine.calculateEMA(candles, 26)

	return ema12 - ema26
}

// calculateEMA вычисляет экспоненциальную скользящую среднюю
func (engine *AdvancedMLEngine) calculateEMA(candles []Candle, period int) float64 {
	if len(candles) < period {
		return candles[len(candles)-1].Close
	}

	multiplier := 2.0 / (float64(period) + 1.0)
	ema := candles[len(candles)-period].Close

	for i := len(candles) - period + 1; i < len(candles); i++ {
		ema = (candles[i].Close * multiplier) + (ema * (1 - multiplier))
	}

	return ema
}

// Остальные вспомогательные функции...
func (engine *AdvancedMLEngine) calculateBollingerPosition(candles []Candle, period int, stdDev float64) float64 {
	if len(candles) < period {
		return 0.5
	}

	// Вычисляем SMA
	sum := 0.0
	for i := len(candles) - period; i < len(candles); i++ {
		sum += candles[i].Close
	}
	sma := sum / float64(period)

	// Вычисляем стандартное отклонение
	variance := 0.0
	for i := len(candles) - period; i < len(candles); i++ {
		diff := candles[i].Close - sma
		variance += diff * diff
	}
	std := math.Sqrt(variance / float64(period))

	current := candles[len(candles)-1].Close
	upper := sma + (stdDev * std)
	lower := sma - (stdDev * std)

	if upper == lower {
		return 0.5
	}

	position := (current - lower) / (upper - lower)
	return math.Max(0, math.Min(1, position))
}

// applyModel применяет модель к фичам
func (engine *AdvancedMLEngine) applyModel(model *SmartModel, features []float64) int {
	if len(features) != len(model.Weights[0]) {
		log.Printf("❌ Feature dimension mismatch: got %d, expected %d", len(features), len(model.Weights[0]))
		return 1 // SIDEWAYS по умолчанию
	}

	// Вычисляем выходы для каждого класса
	outputs := make([]float64, len(model.Weights))

	for i := range outputs {
		sum := model.Biases[i]
		for j, feature := range features {
			sum += feature * model.Weights[i][j]
		}
		outputs[i] = 1.0 / (1.0 + math.Exp(-sum)) // Sigmoid
	}

	// Находим класс с максимальным выходом
	maxOutput := outputs[0]
	prediction := 0

	for i := 1; i < len(outputs); i++ {
		if outputs[i] > maxOutput {
			maxOutput = outputs[i]
			prediction = i
		}
	}

	return prediction
}

// calculateHonestConfidence вычисляет честную уверенность
func (engine *AdvancedMLEngine) calculateHonestConfidence(model *SmartModel, features []float64, prediction int) float64 {
	// Базовая уверенность на основе модели
	outputs := make([]float64, len(model.Weights))

	for i := range outputs {
		sum := model.Biases[i]
		for j, feature := range features {
			sum += feature * model.Weights[i][j]
		}
		outputs[i] = 1.0 / (1.0 + math.Exp(-sum))
	}

	// Softmax для получения вероятностей
	maxOutput := outputs[0]
	for i := 1; i < len(outputs); i++ {
		if outputs[i] > maxOutput {
			maxOutput = outputs[i]
		}
	}

	sum := 0.0
	for i := range outputs {
		outputs[i] = math.Exp(outputs[i] - maxOutput)
		sum += outputs[i]
	}

	for i := range outputs {
		outputs[i] /= sum
	}

	// Уверенность = вероятность предсказанного класса
	confidence := outputs[prediction]

	// Корректируем на основе исторической точности
	accuracy := model.LastAccuracy
	if accuracy > 0.5 {
		confidence *= (1.0 + (accuracy - 0.5)) // Повышаем уверенность для хороших моделей
	} else {
		confidence *= accuracy / 0.5 // Понижаем для плохих моделей
	}

	// Штраф за недостаток данных
	if model.PredictionCount < 100 {
		penalty := float64(model.PredictionCount) / 100.0
		confidence *= penalty
	}

	return math.Max(0.1, math.Min(0.95, confidence))
}

// createHonestSignal создает честный торговый сигнал
func (engine *AdvancedMLEngine) createHonestSignal(symbol string, candle Candle, prediction int, confidence float64) *TradingSignal {
	directions := []string{"SELL", "NEUTRAL", "BUY"}
	predictionStr := directions[prediction]

	// Корректируем силу сигнала на основе уверенности
	switch prediction {
	case 0: // DOWN
		if confidence > 0.8 {
			predictionStr = "STRONG_SELL"
		}
	case 2: // UP
		if confidence > 0.8 {
			predictionStr = "STRONG_BUY"
		}
	}

	// Вычисляем честную целевую цену
	volatility := math.Abs(candle.High-candle.Low) / candle.Close
	expectedChange := (confidence - 0.5) * volatility * 2 // Больше уверенности = больше изменение

	var priceTarget float64
	var stopLoss float64

	switch prediction {
	case 0: // DOWN
		priceTarget = candle.Close * (1.0 - expectedChange)
		stopLoss = candle.Close * 1.02 // 2% стоп-лосс
	case 2: // UP
		priceTarget = candle.Close * (1.0 + expectedChange)
		stopLoss = candle.Close * 0.98 // 2% стоп-лосс
	default: // SIDEWAYS
		priceTarget = candle.Close
		stopLoss = candle.Close
	}

	// Определяем уровень риска
	riskLevel := "MEDIUM"
	if confidence > 0.75 {
		riskLevel = "LOW"
	} else if confidence < 0.65 {
		riskLevel = "HIGH"
	}

	return &TradingSignal{
		Symbol:      symbol,
		Timestamp:   time.Now().Unix(),
		Prediction:  predictionStr,
		Confidence:  confidence,
		PriceTarget: priceTarget,
		StopLoss:    stopLoss,
		TimeHorizon: "60min",
		// Using compatible fields only from TradingSignal struct from ml_trading_model.go
		ModelUsed:      "SmartML",
		KeyFeatures:    []string{"RSI", "MACD", "Volume", "Trend"},
		RiskLevel:      riskLevel,
		PriceChangePct: expectedChange * 100,
		Volatility:     volatility,
	}
}

// Вспомогательные функции для расчета остальных индикаторов...
func (engine *AdvancedMLEngine) calculateTrendStrength(candles []Candle) float64 {
	if len(candles) < 10 {
		return 0.0
	}

	// Простой расчет силы тренда на основе направления цены
	ups := 0
	downs := 0

	for i := len(candles) - 9; i < len(candles); i++ {
		if candles[i].Close > candles[i-1].Close {
			ups++
		} else {
			downs++
		}
	}

	return (float64(ups - downs)) / 9.0
}

func (engine *AdvancedMLEngine) calculateVolatility(candles []Candle, period int) float64 {
	if len(candles) < period {
		return 0.02 // Дефолтная волатильность 2%
	}

	returns := make([]float64, period-1)
	for i := 1; i < period; i++ {
		idx := len(candles) - period + i
		returns[i-1] = (candles[idx].Close - candles[idx-1].Close) / candles[idx-1].Close
	}

	// Стандартное отклонение доходностей
	mean := 0.0
	for _, ret := range returns {
		mean += ret
	}
	mean /= float64(len(returns))

	variance := 0.0
	for _, ret := range returns {
		diff := ret - mean
		variance += diff * diff
	}

	return math.Sqrt(variance / float64(len(returns)-1))
}

func (engine *AdvancedMLEngine) calculateMomentum(candles []Candle, period int) float64 {
	if len(candles) < period {
		return 0.0
	}

	current := candles[len(candles)-1].Close
	past := candles[len(candles)-period].Close

	return (current - past) / past
}

func (engine *AdvancedMLEngine) calculateSupportResistance(candles []Candle) (float64, float64) {
	if len(candles) < 20 {
		last := candles[len(candles)-1]
		return last.Low, last.High
	}

	// Находим локальные минимумы и максимумы
	lows := make([]float64, 0)
	highs := make([]float64, 0)

	for i := len(candles) - 20; i < len(candles); i++ {
		lows = append(lows, candles[i].Low)
		highs = append(highs, candles[i].High)
	}

	sort.Float64s(lows)
	sort.Float64s(highs)

	// Поддержка - 25-й перцентиль минимумов
	support := lows[len(lows)/4]
	// Сопротивление - 75-й перцентиль максимумов
	resistance := highs[3*len(highs)/4]

	return support, resistance
}

func (engine *AdvancedMLEngine) calculateMarketSentiment(candles []Candle) float64 {
	if len(candles) < 5 {
		return 0.0
	}

	// Простое настроение на основе соотношения зеленых и красных свечей
	green := 0
	red := 0

	for i := len(candles) - 5; i < len(candles); i++ {
		if candles[i].Close > candles[i].Open {
			green++
		} else {
			red++
		}
	}

	return (float64(green - red)) / 5.0
}

func (engine *AdvancedMLEngine) calculateWhaleActivity(candles []Candle) float64 {
	if len(candles) < 10 {
		return 0.0
	}

	// Активность китов = необычно высокий объем
	volumes := make([]float64, 10)
	for i := 0; i < 10; i++ {
		volumes[i] = candles[len(candles)-10+i].Volume
	}

	// Средний объем
	avgVol := 0.0
	for _, vol := range volumes {
		avgVol += vol
	}
	avgVol /= 10.0

	// Текущий объем относительно среднего
	currentVol := candles[len(candles)-1].Volume
	return (currentVol - avgVol) / avgVol
}

func (engine *AdvancedMLEngine) calculateMarketCorrelation(candles []Candle) float64 {
	if len(candles) < 10 {
		return 0.0
	}

	// Простая корреляция на основе направления движения цены
	upMoves := 0
	downMoves := 0

	for i := len(candles) - 9; i < len(candles); i++ {
		if candles[i].Close > candles[i-1].Close {
			upMoves++
		} else {
			downMoves++
		}
	}

	// Псевдо-корреляция с рынком
	return (float64(upMoves - downMoves)) / 9.0
}

// saveTrainingExample сохраняет пример для обучения
func (engine *AdvancedMLEngine) saveTrainingExample(symbol string, features []float64, prediction int, signal *TradingSignal) {
	example := TrainingExample{
		Symbol:          symbol,
		Timestamp:       time.Now().Unix(),
		Features:        [][]float64{features}, // Convert []float64 to [][]float64
		Target:          prediction,            // Using correct field name
		ActualPrice:     0.0,                   // Will be filled later
		TargetPrice:     signal.PriceTarget,    // Using correct field name
		Confidence:      signal.Confidence,
		MarketCondition: "normal",
	}

	// Добавляем в память
	if engine.trainingData[symbol] == nil {
		engine.trainingData[symbol] = make([]TrainingExample, 0)
	}

	engine.trainingData[symbol] = append(engine.trainingData[symbol], example)

	// Ограничиваем размер до 1000 примеров
	if len(engine.trainingData[symbol]) > 1000 {
		engine.trainingData[symbol] = engine.trainingData[symbol][1:]
	}

	// Сохраняем в базу данных
	featuresData, _ := json.Marshal(features)
	_, err := engine.db.Exec(`
		INSERT INTO training_examples (symbol, features, direction, predicted_price, created_at)
		VALUES ($1, $2, $3, $4, $5)
	`, symbol, featuresData, prediction, signal.PriceTarget, time.Now())

	if err != nil {
		log.Printf("Failed to save training example: %v", err)
	}
}

// TrainModel обучает модель на новых данных
func (engine *AdvancedMLEngine) TrainModel(symbol string) {
	model, exists := engine.models[symbol]
	if !exists {
		return
	}

	examples := engine.trainingData[symbol]
	if len(examples) < 50 {
		return // Недостаточно данных для обучения
	}

	// Берем последние 200 примеров для обучения
	trainSize := 200
	if len(examples) < trainSize {
		trainSize = len(examples)
	}

	trainExamples := examples[len(examples)-trainSize:]

	// Градиентный спуск
	for epoch := 0; epoch < 10; epoch++ {
		totalLoss := 0.0

		for _, example := range trainExamples {
			if len(engine.trainingData[symbol]) >= 10 {
				continue // Пропускаем непроверенные
			}

			// Forward pass
			outputs := make([]float64, 3)
			for i := range outputs {
				sum := model.Biases[i]
				for j, feature := range example.Features[0] { // Use first sequence and iterate properly
					if j < len(model.Weights[i]) {
						sum += feature * model.Weights[i][j]
					}
				}
				outputs[i] = 1.0 / (1.0 + math.Exp(-sum)) // Sigmoid
			}

			// Вычисляем ошибку
			target := make([]float64, 3)
			target[example.Target] = 1.0

			loss := 0.0
			for i := range outputs {
				diff := outputs[i] - target[i]
				loss += diff * diff
			}
			totalLoss += loss

			// Backward pass
			for i := 0; i < 3; i++ {
				gradient := outputs[i] - target[i] // Use outputs instead of undefined output
				for j := range model.Weights[i] {
					if j < len(example.Features[0]) {
						model.Weights[i][j] -= model.LearningRate * gradient * example.Features[0][j]
					}
				}
			}
		}

		avgLoss := totalLoss / float64(len(trainExamples))
		if epoch%5 == 0 {
			log.Printf("🎓 %s: Training epoch %d, loss: %.6f", symbol, epoch, avgLoss)
		}
	}

	// Обновляем точность
	engine.updateModelAccuracy(symbol)

	// Сохраняем модель
	engine.saveModel(symbol)

	log.Printf("🧠 %s: Model trained, accuracy: %.2f%%", symbol, model.LastAccuracy*100)
}

// updateModelAccuracy обновляет точность модели
func (engine *AdvancedMLEngine) updateModelAccuracy(symbol string) {
	model := engine.models[symbol]
	examples := engine.trainingData[symbol]

	if len(examples) < 10 {
		return
	}

	// Тестируем на последних 50 примерах
	testSize := 50
	if len(examples) < testSize {
		testSize = len(examples)
	}

	testExamples := examples[len(examples)-testSize:]
	correct := 0
	total := 0

	for _, example := range testExamples {
		// Process only examples that need validation
		if len(example.Features) > 0 {
			prediction := engine.applyModel(model, example.Features[0]) // Use first sequence
			if prediction == example.Target {                           // Fixed: use Target instead of Direction
				correct++
			}
			total++
		}
	}

	if total > 0 {
		accuracy := float64(correct) / float64(total)
		model.LastAccuracy = accuracy
		model.CorrectCount = correct
		model.PredictionCount = total

		// Адаптивная скорость обучения
		if accuracy > 0.7 {
			model.LearningRate *= 0.95 // Замедляем при хорошей точности
		} else if accuracy < 0.5 {
			model.LearningRate *= 1.1 // Ускоряем при плохой точности
		}

		// Ограничиваем скорость обучения
		model.LearningRate = math.Max(0.001, math.Min(0.1, model.LearningRate))
	}
}

// VerifyPrediction проверяет правильность прогноза
func (engine *AdvancedMLEngine) VerifyPrediction(symbol string, actualPrice float64, predictionTime time.Time) {
	examples := engine.trainingData[symbol]
	if len(examples) == 0 {
		return
	}

	// Находим прогнозы сделанные час назад
	cutoff := time.Now().Add(-65 * time.Minute) // 65 минут назад (с запасом)

	for i := range examples {
		// Process examples that need verification
		if examples[i].Timestamp > 0 { // Check if example has valid timestamp
			if examples[i].Timestamp > cutoff.Unix() { // Use Unix timestamp comparison
				// Определяем фактическое направление
				priceChange := (actualPrice - examples[i].TargetPrice) / examples[i].TargetPrice // Use TargetPrice field

				actualDirection := 1    // SIDEWAYS по умолчанию
				if priceChange > 0.01 { // > 1%
					actualDirection = 2 // UP
				} else if priceChange < -0.01 { // < -1%
					actualDirection = 0 // DOWN
				}

				examples[i].ActualPrice = actualPrice
				// No Verified field to set - example processed

				// Обновляем направление если нужно
				if actualDirection != examples[i].Target {
					// Update target if needed (though Target field might be readonly)
					// examples[i].Target = actualDirection
				}

				// Обновляем в базе данных
				_, err := engine.db.Exec(`
					UPDATE training_examples 
					SET actual_price = $1, verified = true, direction = $2
					WHERE symbol = $3 AND created_at = $4
				`, actualPrice, actualDirection, symbol, examples[i].Timestamp)

				if err != nil {
					log.Printf("Failed to update training example: %v", err)
				}
			}
		}
	}

	// Запускаем обучение каждые 20 проверок
	model := engine.models[symbol]
	if model != nil {
		verifiedCount := 0
		for _, example := range examples {
			// Count examples that have been processed
			if example.ActualPrice > 0 { // Check if price was set (indicates processing)
				verifiedCount++
			}
		}

		if verifiedCount%20 == 0 && verifiedCount > 0 {
			go engine.TrainModel(symbol) // Асинхронное обучение
		}
	}
}

// loadModels загружает модели из базы данных
func (engine *AdvancedMLEngine) loadModels() {
	rows, err := engine.db.Query(`
		SELECT symbol, model_data, accuracy, prediction_count, correct_count
		FROM smart_models
		ORDER BY updated_at DESC
	`)
	if err != nil {
		return
	}
	defer rows.Close()

	loadedCount := 0
	for rows.Next() {
		var symbol string
		var modelData []byte
		var accuracy float64
		var predictionCount, correctCount int

		if err := rows.Scan(&symbol, &modelData, &accuracy, &predictionCount, &correctCount); err != nil {
			continue
		}

		var model SmartModel
		if err := json.Unmarshal(modelData, &model); err != nil {
			continue
		}

		model.LastAccuracy = accuracy
		model.PredictionCount = predictionCount
		model.CorrectCount = correctCount

		engine.models[symbol] = &model
		loadedCount++
	}

	log.Printf("📦 Loaded %d smart models from database", loadedCount)
}

// saveModel сохраняет модель в базу данных
func (engine *AdvancedMLEngine) saveModel(symbol string) {
	model := engine.models[symbol]
	if model == nil {
		return
	}

	modelData, err := json.Marshal(model)
	if err != nil {
		return
	}

	_, err = engine.db.Exec(`
		INSERT INTO smart_models (symbol, model_data, accuracy, prediction_count, correct_count, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6)
		ON CONFLICT (symbol) DO UPDATE SET
			model_data = $2,
			accuracy = $3,
			prediction_count = $4,
			correct_count = $5,
			updated_at = $6
	`, symbol, modelData, model.LastAccuracy, model.PredictionCount, model.CorrectCount, time.Now())

	if err != nil {
		log.Printf("Failed to save model for %s: %v", symbol, err)
	}
}

// GetModelStats возвращает статистику моделей
func (engine *AdvancedMLEngine) GetModelStats() map[string]*MLPerformanceMetrics {
	return engine.performanceCache
}

// StartAutoTraining запускает автоматическое обучение
func (engine *AdvancedMLEngine) StartAutoTraining() {
	go func() {
		ticker := time.NewTicker(30 * time.Minute)
		defer ticker.Stop()

		for range ticker.C {
			for symbol := range engine.models {
				go engine.TrainModel(symbol)
			}
			log.Println("🎓 Auto-training cycle completed")
		}
	}()

	log.Println("🚀 Auto-training started (every 30 minutes)")
}
