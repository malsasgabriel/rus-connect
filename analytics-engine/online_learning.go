package main

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

const (
	defaultBootstrapMinCandles      = 20
	defaultMinCandlesRequired       = 60
	defaultPredictionHorizonMinutes = 5  // Reduced from 15 to 5 for faster learning feedback
	defaultNeutralThreshold         = 0.001
	defaultTrainEpochs              = 30
	defaultRetrainInterval          = 5 * time.Minute
	defaultMaxTrainingExamples      = 4000
	maxPendingExamplesPerSymbol     = 1500

	defaultSidewaysCeilingPct = 0.72
	defaultNeutralFloor       = 0.00025
	defaultNeutralCeil        = 0.00350
	defaultNeutralAdjustStep  = 0.00010

	defaultAutopilotInterval         = 2 * time.Minute
	defaultAutopilotSilenceTimeout   = 15 * time.Minute
	defaultAutopilotActionCooldown   = 5 * time.Minute
	defaultAutopilotMinDirectional   = 0.18
	defaultAutopilotMinSignalSamples = 18
	defaultAutopilotResetMinAccuracy = 0.48
	defaultAutopilotResetPredictions = 150
)

type trainingConfig struct {
	bootstrapMinCandles      int
	minCandlesRequired       int
	predictionHorizonMinutes int
	neutralThreshold         float64
	neutralThresholdFloor    float64
	neutralThresholdCeil     float64
	neutralAdjustStep        float64
	sidewaysCeilingPct       float64
	trainEpochs              int
	retrainInterval          time.Duration
	maxTrainingExamples      int
}

type automationConfig struct {
	enabled             bool
	interval            time.Duration
	silenceTimeout      time.Duration
	actionCooldown      time.Duration
	minDirectionalRate  float64
	minSignalSamples    int
	resetMinAccuracy    float64
	resetMinPredictions int
}

func loadTrainingConfig() trainingConfig {
	cfg := trainingConfig{
		bootstrapMinCandles:      getEnvInt("ML_MIN_CANDLES_BOOTSTRAP", defaultBootstrapMinCandles),
		minCandlesRequired:       getEnvInt("ML_MIN_CANDLES", defaultMinCandlesRequired),
		predictionHorizonMinutes: getEnvInt("ML_PREDICTION_HORIZON_MIN", defaultPredictionHorizonMinutes),
		neutralThreshold:         getEnvFloat("ML_NEUTRAL_THRESHOLD", defaultNeutralThreshold),
		neutralThresholdFloor:    getEnvFloat("ML_NEUTRAL_THRESHOLD_FLOOR", defaultNeutralFloor),
		neutralThresholdCeil:     getEnvFloat("ML_NEUTRAL_THRESHOLD_CEIL", defaultNeutralCeil),
		neutralAdjustStep:        getEnvFloat("ML_NEUTRAL_THRESHOLD_STEP", defaultNeutralAdjustStep),
		sidewaysCeilingPct:       getEnvFloat("ML_SIDEWAYS_CEILING", defaultSidewaysCeilingPct),
		trainEpochs:              getEnvInt("ML_TRAIN_EPOCHS", defaultTrainEpochs),
		retrainInterval:          getEnvDuration("ML_RETRAIN_INTERVAL", defaultRetrainInterval),
		maxTrainingExamples:      getEnvInt("ML_MAX_TRAINING_EXAMPLES", defaultMaxTrainingExamples),
	}

	if cfg.bootstrapMinCandles < 3 {
		cfg.bootstrapMinCandles = 3
	}
	if cfg.minCandlesRequired < 3 {
		cfg.minCandlesRequired = 3
	}
	if cfg.bootstrapMinCandles > cfg.minCandlesRequired {
		cfg.bootstrapMinCandles = cfg.minCandlesRequired
	}
	if cfg.predictionHorizonMinutes < 1 {
		cfg.predictionHorizonMinutes = defaultPredictionHorizonMinutes
	}
	if cfg.neutralThreshold <= 0 || cfg.neutralThreshold >= 0.1 {
		cfg.neutralThreshold = defaultNeutralThreshold
	}
	if cfg.neutralThresholdFloor <= 0 || cfg.neutralThresholdFloor >= cfg.neutralThresholdCeil {
		cfg.neutralThresholdFloor = defaultNeutralFloor
	}
	if cfg.neutralThresholdCeil <= cfg.neutralThresholdFloor || cfg.neutralThresholdCeil >= 0.1 {
		cfg.neutralThresholdCeil = defaultNeutralCeil
	}
	if cfg.neutralAdjustStep <= 0 || cfg.neutralAdjustStep > 0.01 {
		cfg.neutralAdjustStep = defaultNeutralAdjustStep
	}
	if cfg.sidewaysCeilingPct < 0.4 || cfg.sidewaysCeilingPct > 0.95 {
		cfg.sidewaysCeilingPct = defaultSidewaysCeilingPct
	}
	if cfg.trainEpochs < 1 {
		cfg.trainEpochs = defaultTrainEpochs
	}
	if cfg.maxTrainingExamples < 200 {
		cfg.maxTrainingExamples = 200
	}

	return cfg
}

func loadAutomationConfig() automationConfig {
	enabled := getEnvInt("ML_AUTOPILOT_ENABLED", 1) == 1
	cfg := automationConfig{
		enabled:             enabled,
		interval:            getEnvDuration("ML_AUTOPILOT_INTERVAL", defaultAutopilotInterval),
		silenceTimeout:      getEnvDuration("ML_AUTOPILOT_SILENCE_TIMEOUT", defaultAutopilotSilenceTimeout),
		actionCooldown:      getEnvDuration("ML_AUTOPILOT_ACTION_COOLDOWN", defaultAutopilotActionCooldown),
		minDirectionalRate:  getEnvFloat("ML_AUTOPILOT_MIN_DIRECTIONAL_RATE", defaultAutopilotMinDirectional),
		minSignalSamples:    getEnvInt("ML_AUTOPILOT_MIN_SIGNAL_SAMPLES", defaultAutopilotMinSignalSamples),
		resetMinAccuracy:    getEnvFloat("ML_AUTOPILOT_RESET_MIN_ACCURACY", defaultAutopilotResetMinAccuracy),
		resetMinPredictions: getEnvInt("ML_AUTOPILOT_RESET_MIN_PREDICTIONS", defaultAutopilotResetPredictions),
	}
	if cfg.interval < 30*time.Second {
		cfg.interval = defaultAutopilotInterval
	}
	if cfg.silenceTimeout < 2*time.Minute {
		cfg.silenceTimeout = defaultAutopilotSilenceTimeout
	}
	if cfg.actionCooldown < 30*time.Second {
		cfg.actionCooldown = defaultAutopilotActionCooldown
	}
	if cfg.minDirectionalRate <= 0 || cfg.minDirectionalRate >= 1 {
		cfg.minDirectionalRate = defaultAutopilotMinDirectional
	}
	if cfg.minSignalSamples < 5 {
		cfg.minSignalSamples = defaultAutopilotMinSignalSamples
	}
	if cfg.resetMinAccuracy <= 0 || cfg.resetMinAccuracy >= 1 {
		cfg.resetMinAccuracy = defaultAutopilotResetMinAccuracy
	}
	if cfg.resetMinPredictions < 20 {
		cfg.resetMinPredictions = defaultAutopilotResetPredictions
	}
	return cfg
}

func (ae *AnalyticsEngine) initializeOnlineLearning() {
	cfg := loadTrainingConfig()
	autoCfg := loadAutomationConfig()
	ae.bootstrapMinCandles = cfg.bootstrapMinCandles
	ae.steadyMinCandles = cfg.minCandlesRequired
	ae.setMinCandlesRequired(cfg.bootstrapMinCandles)
	ae.featureEngine = NewSimpleFeatureEngine()
	ae.normalizerManager = NewFeatureNormalizerManager(16)
	ae.models = make(map[string]*SimpleNeuralNetwork)
	ae.pendingExamples = make(map[string][]PendingExample)
	ae.trainingData = make(map[string][]TrainingExample)
	ae.lastTrainedAt = make(map[string]time.Time)
	ae.predictionHorizonMinutes = cfg.predictionHorizonMinutes
	ae.neutralThreshold = cfg.neutralThreshold
	ae.neutralThresholdFloor = cfg.neutralThresholdFloor
	ae.neutralThresholdCeil = cfg.neutralThresholdCeil
	ae.neutralAdjustStep = cfg.neutralAdjustStep
	ae.sidewaysCeilingPct = cfg.sidewaysCeilingPct
	ae.symbolNeutralThresholds = make(map[string]float64)
	ae.trainEpochs = cfg.trainEpochs
	ae.retrainInterval = cfg.retrainInterval
	ae.maxTrainingExamples = cfg.maxTrainingExamples
	ae.autopilotEnabled = autoCfg.enabled
	ae.autopilotInterval = autoCfg.interval
	ae.autopilotSilenceTimeout = autoCfg.silenceTimeout
	ae.autopilotActionCooldown = autoCfg.actionCooldown
	ae.autopilotMinDirectional = autoCfg.minDirectionalRate
	ae.autopilotMinSignalSamples = autoCfg.minSignalSamples
	ae.autopilotResetMinAccuracy = autoCfg.resetMinAccuracy
	ae.autopilotResetMinPredictions = autoCfg.resetMinPredictions
	ae.autopilotLastAction = make(map[string]time.Time)
	log.Printf("ML thresholds: bootstrap=%d steady=%d neutral=%.6f", ae.bootstrapMinCandles, ae.steadyMinCandles, ae.neutralThreshold)
	log.Printf("ML autopilot: enabled=%v interval=%s silence_timeout=%s cooldown=%s min_directional=%.3f",
		ae.autopilotEnabled, ae.autopilotInterval, ae.autopilotSilenceTimeout, ae.autopilotActionCooldown, ae.autopilotMinDirectional)

	if ae.db != nil {
		if err := ae.normalizerManager.LoadAll(ae); err != nil {
			log.Printf("Failed to load feature normalizers: %v", err)
		}
	}
}

func (ae *AnalyticsEngine) runMLAutopilot(ctx context.Context) {
	if !ae.autopilotEnabled {
		return
	}
	log.Printf("ML autopilot loop started")
	ticker := time.NewTicker(ae.autopilotInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			log.Printf("ML autopilot loop stopped")
			return
		case <-ticker.C:
			for _, symbol := range ae.symbols {
				ae.evaluateSymbolAutopilot(strings.ToUpper(symbol))
			}
		}
	}
}

type signalWindowStats struct {
	Total    int
	Up       int
	Down     int
	Sideways int
}

func (ae *AnalyticsEngine) querySignalWindowStats(symbol string, minutes int) (signalWindowStats, error) {
	stats := signalWindowStats{}
	if ae.db == nil {
		return stats, sql.ErrConnDone
	}
	query := fmt.Sprintf(`
		SELECT direction, count()
		FROM direction_predictions
		WHERE symbol = ? AND created_at >= now() - INTERVAL %d MINUTE
		GROUP BY direction
	`, minutes)
	rows, err := ae.db.Query(query, symbol)
	if err != nil {
		return stats, err
	}
	defer rows.Close()
	for rows.Next() {
		var direction string
		var count int
		if scanErr := rows.Scan(&direction, &count); scanErr != nil {
			continue
		}
		stats.Total += count
		switch strings.ToUpper(direction) {
		case "UP":
			stats.Up += count
		case "DOWN":
			stats.Down += count
		default:
			stats.Sideways += count
		}
	}
	return stats, nil
}

func (ae *AnalyticsEngine) evaluateSymbolAutopilot(symbol string) {
	now := time.Now().UTC()
	model := ae.getOrCreateModel(symbol)

	ae.mlMu.Lock()
	readyAt := ae.historyReadyAt[symbol]
	emitted := ae.signalsEmittedBySymbol[symbol]
	lastTrained := ae.lastTrainedAt[symbol]
	trainLen := len(ae.trainingData[symbol])
	lastAction := ae.autopilotLastAction[symbol]
	ae.mlMu.Unlock()

	if !lastAction.IsZero() && now.Sub(lastAction) < ae.autopilotActionCooldown {
		return
	}

	action := ""
	reason := ""
	currentThreshold := ae.getNeutralThreshold(symbol)
	nextThreshold := currentThreshold
	directionalRate := 0.0
	sidewaysRate := 0.0

	stats, statsErr := ae.querySignalWindowStats(symbol, 60)
	if statsErr == nil && stats.Total > 0 {
		directionalRate = float64(stats.Up+stats.Down) / float64(stats.Total)
		sidewaysRate = float64(stats.Sideways) / float64(stats.Total)
	}

	if model.PredictionCount >= ae.autopilotResetMinPredictions && model.GetAccuracy() < ae.autopilotResetMinAccuracy {
		action = "reset_model_retrain"
		reason = "low_live_accuracy"
	} else if !readyAt.IsZero() && emitted == 0 && now.Sub(readyAt) >= ae.autopilotSilenceTimeout && trainLen >= MIN_TRAINING_EXAMPLES {
		action = "force_retrain"
		reason = "no_signals_after_history_ready"
	} else if statsErr == nil && stats.Total >= ae.autopilotMinSignalSamples && directionalRate < ae.autopilotMinDirectional {
		nextThreshold = math.Max(ae.neutralThresholdFloor, currentThreshold-ae.neutralAdjustStep)
		action = "tighten_threshold_retrain"
		reason = "directional_rate_too_low"
	} else if trainLen >= MIN_TRAINING_EXAMPLES && time.Since(lastTrained) >= ae.retrainInterval*3 {
		action = "scheduled_retrain"
		reason = "stale_model_age"
	}

	if action == "" {
		return
	}

	if action == "reset_model_retrain" {
		ae.mlMu.Lock()
		ae.models[symbol] = NewSimpleNeuralNetwork(symbol)
		ae.mlMu.Unlock()
		log.Printf("autopilot action: symbol=%s action=%s reason=%s", symbol, action, reason)
	} else if action == "tighten_threshold_retrain" && nextThreshold < currentThreshold-1e-12 {
		ae.mlMu.Lock()
		ae.symbolNeutralThresholds[symbol] = nextThreshold
		ae.mlMu.Unlock()
		log.Printf("autopilot threshold tighten: symbol=%s old=%.6f new=%.6f directional_rate=%.3f",
			symbol, currentThreshold, nextThreshold, directionalRate)
	}

	ae.maybeRetrain(symbol, true)

	ae.mlMu.Lock()
	ae.autopilotLastAction[symbol] = now
	ae.mlMu.Unlock()

	if err := ae.saveAutomationEvent(symbol, action, reason, directionalRate, sidewaysRate, model.GetAccuracy(), nextThreshold, trainLen); err != nil {
		log.Printf("Failed to save automation event for %s: %v", symbol, err)
	}
}

func (ae *AnalyticsEngine) processOnlineLearning(candle Candle) {
	if ae.featureEngine == nil {
		return
	}

	features := ae.featureEngine.AddCandle(candle)
	if len(features) == 0 {
		return
	}

	ae.resolvePendingExamples(candle.Symbol, candle.Timestamp, candle.Close)

	model := ae.getOrCreateModel(candle.Symbol)
	normalizer := ae.normalizerManager.Get(candle.Symbol)

	normalizedFeatures := features
	if normalizer.Fitted {
		normalizedFeatures = normalizer.Transform(features)
	}

	predClass, rawConfidence, probs := model.PredictWithDistribution(normalizedFeatures)
	predDirection := classToDirection(predClass)

	ae.enqueuePendingExample(PendingExample{
		Symbol:             candle.Symbol,
		Features:           append([]float64(nil), features...),
		Timestamp:          time.Unix(candle.Timestamp, 0).UTC(),
		PredictedDirection: predClass,
		EntryPrice:         candle.Close,
		TimeHorizon:        ae.predictionHorizonMinutes,
	})

	// Get trust stage and apply confidence dampening
	trustStage := ae.getModelTrustStage(model)
	confidence := ApplyTrustStageDampening(rawConfidence, trustStage, model.PredictionCount)

	if confidence < ae.confidenceThreshold {
		return
	}

	symbolThreshold := ae.getNeutralThreshold(candle.Symbol)
	signal := DirectionSignal{
		Symbol:          candle.Symbol,
		Timestamp:       candle.Timestamp,
		Direction:       predDirection,
		Confidence:      confidence,
		ClassProbs:      probs,
		CurrentPrice:    candle.Close,
		PriceTarget:     estimatePriceTarget(candle.Close, predClass, symbolThreshold),
		TimeHorizon:     ae.predictionHorizonMinutes,
		LabelHorizonMin: ae.predictionHorizonMinutes,
		ModelType:       "simple_nn_online",
		ModelUsed:       "simple_nn_online",
		Version:         "v2-real-training",
		RiskLevel:       inferRiskLevel(confidence),
		Factors:         []string{"price_action", "volume", "momentum"},
		TrustStage:      trustStage,
		ModelAgeSec:     int64(time.Since(model.LastUpdate).Seconds()),
		PredictionCount: model.PredictionCount,
	}
	ae.emitDirectionSignal(signal)
}

func (ae *AnalyticsEngine) enqueuePendingExample(example PendingExample) {
	ae.mlMu.Lock()
	defer ae.mlMu.Unlock()

	pending := ae.pendingExamples[example.Symbol]
	pending = append(pending, example)
	
	// FIX: Limit pending examples and clean up old ones
	maxPending := 500 // Reduced from 1500 to save memory
	if len(pending) > maxPending {
		pending = pending[len(pending)-maxPending:]
	}
	ae.pendingExamples[example.Symbol] = pending
}

func (ae *AnalyticsEngine) resolvePendingExamples(symbol string, currentTimestamp int64, currentPrice float64) {
	if currentPrice <= 0 {
		return
	}

	var matured []PendingExample
	var stillPending []PendingExample
	var expired []PendingExample // Track expired examples for cleanup

	ae.mlMu.Lock()
	pending := ae.pendingExamples[symbol]
	currentTime := time.Unix(currentTimestamp, 0)

	for _, p := range pending {
		deadline := p.Timestamp.Unix() + int64(p.TimeHorizon*60)
		expiryTime := p.Timestamp.Add(time.Duration(p.TimeHorizon*60+30) * time.Minute) // 30 min grace period

		if currentTimestamp >= deadline {
			matured = append(matured, p)
		} else if currentTime.After(expiryTime) {
			// Example is too old, discard it
			expired = append(expired, p)
		} else {
			stillPending = append(stillPending, p)
		}
	}

	// FIX: Properly clean up to allow GC to reclaim memory
	ae.pendingExamples[symbol] = stillPending
	ae.mlMu.Unlock()

	if len(expired) > 0 {
		log.Printf("🗑️ Cleaned up %d expired pending examples for %s", len(expired), symbol)
	}

	if len(matured) == 0 {
		return
	}

	log.Printf("📊 Resolving %d matured pending examples for %s", len(matured), symbol)

	model := ae.getOrCreateModel(symbol)
	neutralThreshold := ae.getNeutralThreshold(symbol)
	newExamples := make([]TrainingExample, 0, len(matured))
	correctCount := 0
	
	for _, p := range matured {
		if p.EntryPrice <= 0 {
			continue
		}
		deadline := p.Timestamp.Unix() + int64(p.TimeHorizon*60)
		outcomePrice := ae.resolveOutcomePrice(symbol, deadline, currentPrice)
		ret := (outcomePrice - p.EntryPrice) / p.EntryPrice
		target := classFromReturn(ret, neutralThreshold)
		
		isCorrect := (target == p.PredictedDirection)
		if isCorrect {
			correctCount++
		}
		
		model.UpdateAccuracy(isCorrect)
		newExamples = append(newExamples, TrainingExample{
			Features: p.Features,
			Target:   target,
			Ts:       deadline,
		})
		
		log.Printf("✅ Resolved: %s predicted=%d actual=%d correct=%v entry=%.6f exit=%.6f ret=%.4f%%",
			symbol, p.PredictedDirection, target, isCorrect, p.EntryPrice, outcomePrice, ret*100)
	}
	if len(newExamples) == 0 {
		return
	}

	ae.mlMu.Lock()
	// FIX: Properly trim training data to allow GC to reclaim memory
	buffer := ae.trainingData[symbol]
	buffer = append(buffer, newExamples...)
	if len(buffer) > ae.maxTrainingExamples {
		// Create new slice to allow GC to reclaim old memory
		newBuffer := make([]TrainingExample, ae.maxTrainingExamples)
		copy(newBuffer, buffer[len(buffer)-ae.maxTrainingExamples:])
		buffer = newBuffer
	}
	ae.trainingData[symbol] = buffer
	ae.mlMu.Unlock()
	
	log.Printf("📈 Model %s updated: %d new examples, accuracy=%.4f (%d/%d correct)",
		symbol, len(newExamples), model.GetAccuracy(), model.CorrectCount, model.PredictionCount)

	ae.adjustNeutralThreshold(symbol, buffer)
	ae.maybeRetrain(symbol, false)
}

func (ae *AnalyticsEngine) seedTrainingDataFromHistory(symbol string, candles []Candle) int {
	if len(candles) < ae.predictionHorizonMinutes+3 {
		return 0
	}
	tempFE := NewSimpleFeatureEngine()
	seeded := make([]TrainingExample, 0, len(candles))
	neutralThreshold := ae.getNeutralThreshold(symbol)

	for idx, c := range candles {
		features := tempFE.AddCandle(c)
		if len(features) == 0 {
			continue
		}
		futureIdx := idx + ae.predictionHorizonMinutes
		if futureIdx >= len(candles) {
			break
		}
		entry := c.Close
		future := candles[futureIdx].Close
		if entry <= 0 || future <= 0 {
			continue
		}
		ret := (future - entry) / entry
		target := classFromReturn(ret, neutralThreshold)
		seeded = append(seeded, TrainingExample{
			Features: append([]float64(nil), features...),
			Target:   target,
			Ts:       candles[futureIdx].Timestamp,
		})
	}
	if len(seeded) == 0 {
		return 0
	}
	ae.mlMu.Lock()
	buffer := append(ae.trainingData[symbol], seeded...)
	if len(buffer) > ae.maxTrainingExamples {
		buffer = buffer[len(buffer)-ae.maxTrainingExamples:]
	}
	ae.trainingData[symbol] = buffer
	ae.mlMu.Unlock()
	ae.adjustNeutralThreshold(symbol, buffer)
	return len(seeded)
}

func (ae *AnalyticsEngine) maybeRetrain(symbol string, force bool) {
	model := ae.getOrCreateModel(symbol)

	ae.mlMu.Lock()
	rawExamples := append([]TrainingExample(nil), ae.trainingData[symbol]...)
	lastTrained := ae.lastTrainedAt[symbol]
	ae.mlMu.Unlock()
	neutralThreshold := ae.getNeutralThreshold(symbol)

	if len(rawExamples) < MIN_TRAINING_EXAMPLES {
		return
	}

	shouldRetrain := force || model.ShouldRetrain() || time.Since(lastTrained) >= ae.retrainInterval
	if !shouldRetrain {
		return
	}

	classWeights := ae.computeClassWeights24h(rawExamples)
	normalizer := ae.normalizerManager.Get(symbol)
	rawFeatures := make([][]float64, 0, len(rawExamples))
	for _, ex := range rawExamples {
		rawFeatures = append(rawFeatures, ex.Features)
	}
	normalizer.Fit(rawFeatures)

	normalizedExamples := make([]TrainingExample, 0, len(rawExamples))
	for _, ex := range rawExamples {
		normalizedExamples = append(normalizedExamples, TrainingExample{
			Features: normalizer.Transform(ex.Features),
			Target:   ex.Target,
			Ts:       ex.Ts,
		})
	}

	report := model.Train(normalizedExamples, ae.trainEpochs, classWeights)

	if ae.db != nil {
		if err := model.SaveToDB(ae.db); err != nil {
			log.Printf("Failed to save model weights for %s: %v", symbol, err)
		}
		if err := normalizer.SaveToDB(ae, symbol); err != nil {
			log.Printf("Failed to save feature normalizer for %s: %v", symbol, err)
		}
		if err := ae.saveTrainingEvent(symbol, report, classWeights, neutralThreshold, ae.getModelTrustStage(model)); err != nil {
			log.Printf("Failed to save training event for %s: %v", symbol, err)
		}
	}

	ae.mlMu.Lock()
	ae.lastTrainedAt[symbol] = time.Now()
	ae.mlMu.Unlock()

	if ae.steadyMinCandles > 0 && ae.getMinCandlesRequired() < ae.steadyMinCandles {
		ae.setMinCandlesRequired(ae.steadyMinCandles)
		log.Printf("inference_started: switched to steady threshold after retrain (%d)", ae.steadyMinCandles)
	}
}

func (ae *AnalyticsEngine) forceRetrain(symbol string) (int, error) {
	if symbol == "" {
		for _, s := range ae.symbols {
			ae.maybeRetrain(strings.ToUpper(s), true)
		}
		return len(ae.symbols), nil
	}
	ae.maybeRetrain(strings.ToUpper(symbol), true)
	return 1, nil
}

func (ae *AnalyticsEngine) getOrCreateModel(symbol string) *SimpleNeuralNetwork {
	ae.mlMu.Lock()
	defer ae.mlMu.Unlock()

	if model, ok := ae.models[symbol]; ok {
		return model
	}

	model := NewSimpleNeuralNetwork(symbol)
	if ae.db != nil && getEnvInt("ML_LOAD_MODEL_WEIGHTS", 0) == 1 {
		if err := model.LoadFromDB(ae.db, symbol); err != nil {
			log.Printf("Failed to load model weights for %s: %v", symbol, err)
			model = NewSimpleNeuralNetwork(symbol)
		}
		if !model.IsShapeValid() {
			log.Printf("Invalid model weight shape for %s, using fresh initialization", symbol)
			model = NewSimpleNeuralNetwork(symbol)
		}
	}
	ae.models[symbol] = model
	return model
}

func (ae *AnalyticsEngine) resolveOutcomePrice(symbol string, maturityTS int64, fallback float64) float64 {
	if ae.featureEngine == nil {
		return fallback
	}
	history := ae.featureEngine.GetHistory(symbol)
	if len(history) == 0 {
		return fallback
	}

	bestPrice := history[len(history)-1].Close
	bestDelta := int64(math.MaxInt64)
	for _, c := range history {
		delta := c.Timestamp - maturityTS
		if delta < 0 {
			delta = -delta
		}
		if delta < bestDelta {
			bestDelta = delta
			bestPrice = c.Close
		}
	}
	if bestPrice <= 0 {
		return fallback
	}
	return bestPrice
}

func (ae *AnalyticsEngine) adjustNeutralThreshold(symbol string, examples []TrainingExample) {
	if len(examples) == 0 {
		return
	}
	start := 0
	if len(examples) > 240 {
		start = len(examples) - 240
	}
	window := examples[start:]
	sideways := 0
	for _, ex := range window {
		if ex.Target == 1 {
			sideways++
		}
	}
	sidewaysRate := float64(sideways) / float64(len(window))
	current := ae.getNeutralThreshold(symbol)
	next := current
	if sidewaysRate > ae.sidewaysCeilingPct {
		next = math.Max(ae.neutralThresholdFloor, current-ae.neutralAdjustStep)
	} else if sidewaysRate < ae.sidewaysCeilingPct*0.60 {
		next = math.Min(ae.neutralThresholdCeil, current+ae.neutralAdjustStep*0.50)
	}
	if math.Abs(next-current) > 1e-12 {
		ae.mlMu.Lock()
		ae.symbolNeutralThresholds[symbol] = next
		ae.mlMu.Unlock()
		log.Printf("adaptive neutral threshold: symbol=%s old=%.6f new=%.6f sideways_rate=%.3f",
			symbol, current, next, sidewaysRate)
	}
}

func (ae *AnalyticsEngine) computeClassWeights24h(examples []TrainingExample) [3]float64 {
	counts := [3]float64{0, 0, 0}
	cutoff := time.Now().Add(-24 * time.Hour).Unix()
	for _, ex := range examples {
		if ex.Target < 0 || ex.Target > 2 {
			continue
		}
		if ex.Ts == 0 || ex.Ts >= cutoff {
			counts[ex.Target]++
		}
	}
	total := counts[0] + counts[1] + counts[2]
	if total == 0 {
		return [3]float64{1, 1, 1}
	}

	weights := [3]float64{1, 1, 1}
	nonZero := 0.0
	for i := 0; i < 3; i++ {
		if counts[i] > 0 {
			weights[i] = total / (3.0 * counts[i])
			nonZero++
		}
	}
	if counts[1] > counts[0]+counts[2] {
		weights[0] *= 1.25
		weights[2] *= 1.25
		weights[1] *= 0.90
	}
	if nonZero == 0 {
		return [3]float64{1, 1, 1}
	}
	return weights
}

func (ae *AnalyticsEngine) getModelTrustStage(model *SimpleNeuralNetwork) string {
	if model == nil || !model.Trained {
		return "cold_start"
	}
	if model.PredictionCount < 100 {
		return "warming"
	}
	return "trained"
}

func (ae *AnalyticsEngine) getNeutralThreshold(symbol string) float64 {
	ae.mlMu.Lock()
	defer ae.mlMu.Unlock()
	if ae.symbolNeutralThresholds == nil {
		ae.symbolNeutralThresholds = make(map[string]float64)
	}
	if v, ok := ae.symbolNeutralThresholds[symbol]; ok {
		return v
	}
	ae.symbolNeutralThresholds[symbol] = ae.neutralThreshold
	return ae.neutralThreshold
}

func (ae *AnalyticsEngine) getMinCandlesRequired() int {
	return int(atomic.LoadInt32(&ae.minCandlesRequired))
}

func (ae *AnalyticsEngine) setMinCandlesRequired(v int) {
	atomic.StoreInt32(&ae.minCandlesRequired, int32(v))
}

func classFromReturn(ret, neutralThreshold float64) int {
	if ret > neutralThreshold {
		return 2
	}
	if ret < -neutralThreshold {
		return 0
	}
	return 1
}

func classToDirection(class int) string {
	switch class {
	case 0:
		return "DOWN"
	case 2:
		return "UP"
	default:
		return "SIDEWAYS"
	}
}

func inferRiskLevel(confidence float64) string {
	switch {
	case confidence >= 0.80:
		return "LOW"
	case confidence >= 0.65:
		return "MEDIUM"
	default:
		return "HIGH"
	}
}

func estimatePriceTarget(price float64, class int, neutralThreshold float64) float64 {
	if price <= 0 {
		return price
	}
	move := math.Max(0.0015, neutralThreshold*1.8)
	switch class {
	case 0:
		return price * (1 - move)
	case 2:
		return price * (1 + move)
	default:
		return price
	}
}

func getEnvInt(name string, fallback int) int {
	raw := os.Getenv(name)
	if raw == "" {
		return fallback
	}
	v, err := strconv.Atoi(raw)
	if err != nil {
		return fallback
	}
	return v
}

func getEnvFloat(name string, fallback float64) float64 {
	raw := os.Getenv(name)
	if raw == "" {
		return fallback
	}
	v, err := strconv.ParseFloat(raw, 64)
	if err != nil {
		return fallback
	}
	return v
}

func getEnvDuration(name string, fallback time.Duration) time.Duration {
	raw := os.Getenv(name)
	if raw == "" {
		return fallback
	}
	v, err := time.ParseDuration(raw)
	if err != nil {
		return fallback
	}
	return v
}
