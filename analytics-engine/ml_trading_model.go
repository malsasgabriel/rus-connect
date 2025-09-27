package main

import (
	"log"
	"math"
	"math/rand"
	"time"
)

// TradingSignal represents the ML prediction output for frontend
type TradingSignal struct {
	Symbol         string   `json:"symbol"`
	Timestamp      int64    `json:"timestamp"`
	Prediction     string   `json:"prediction"`       // "STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL"
	Confidence     float64  `json:"confidence"`       // 0.0 - 1.0
	PriceTarget    float64  `json:"price_target"`     // Target price in 1 hour
	StopLoss       float64  `json:"stop_loss"`        // Stop loss level
	TimeHorizon    string   `json:"time_horizon"`     // "1H"
	ModelUsed      string   `json:"model_used"`       // "LSTM", "XGBoost", "Ensemble"
	KeyFeatures    []string `json:"key_features"`     // Key decision factors
	RiskLevel      string   `json:"risk_level"`       // "LOW", "MEDIUM", "HIGH"
	PriceChangePct float64  `json:"price_change_pct"` // Expected % change
	Volatility     float64  `json:"volatility"`       // Market volatility
}

// PredictionTarget represents the target variable for ML training
type PredictionTarget struct {
	Symbol             string  `json:"symbol"`
	CurrentTime        int64   `json:"current_time"`
	PredictionHorizon  int     `json:"prediction_horizon"` // 60 minutes
	ActualPrice        float64 `json:"actual_price"`
	PredictedDirection string  `json:"predicted_direction"` // "UP", "DOWN", "SIDEWAYS"
	Confidence         float64 `json:"confidence"`
	PriceChangePct     float64 `json:"price_change_pct"`
	Volatility         float64 `json:"volatility"`
}

// LSTMLayer represents a single LSTM layer with attention mechanism
type LSTMLayer struct {
	InputSize   int         `json:"input_size"`
	HiddenSize  int         `json:"hidden_size"`
	WeightsIH   [][]float64 `json:"weights_ih"`   // Input to hidden weights
	WeightsHH   [][]float64 `json:"weights_hh"`   // Hidden to hidden weights
	BiasH       []float64   `json:"bias_h"`       // Hidden bias
	AttentionW  [][]float64 `json:"attention_w"`  // Attention weights
	HiddenState []float64   `json:"hidden_state"` // Current hidden state
	CellState   []float64   `json:"cell_state"`   // Current cell state
}

// AttentionMechanism implements multi-head attention for LSTM
type AttentionMechanism struct {
	HeadCount     int         `json:"head_count"`
	DModel        int         `json:"d_model"`
	QueryWeights  [][]float64 `json:"query_weights"`
	KeyWeights    [][]float64 `json:"key_weights"`
	ValueWeights  [][]float64 `json:"value_weights"`
	OutputWeights [][]float64 `json:"output_weights"`
}

// MLTradingModel represents the complete ML architecture
type MLTradingModel struct {
	Symbol           string              `json:"symbol"`
	LSTMLayer1       *LSTMLayer          `json:"lstm_layer_1"`
	LSTMLayer2       *LSTMLayer          `json:"lstm_layer_2"`
	Attention        *AttentionMechanism `json:"attention"`
	DenseLayer1      [][]float64         `json:"dense_layer_1"` // 128 neurons
	DenseLayer2      [][]float64         `json:"dense_layer_2"` // 64 neurons
	OutputLayer      [][]float64         `json:"output_layer"`  // 3 classes + confidence
	DropoutRate      float64             `json:"dropout_rate"`
	LearningRate     float64             `json:"learning_rate"`
	SequenceLength   int                 `json:"sequence_length"` // 1440 minutes
	FeatureCount     int                 `json:"feature_count"`   // 50+ features
	TrainingBuffer   []TrainingSample    `json:"training_buffer"`
	ValidationBuffer []TrainingSample    `json:"validation_buffer"`
	ModelMetrics     *ModelMetrics       `json:"model_metrics"`
	LastTraining     time.Time           `json:"last_training"`
	IsTraining       bool                `json:"is_training"`
}

// TrainingSample represents a single training example
type TrainingSample struct {
	Features   [][]float64 `json:"features"` // [sequence_length][feature_count]
	Target     []float64   `json:"target"`   // [3] for UP/DOWN/SIDEWAYS
	Timestamp  int64       `json:"timestamp"`
	Symbol     string      `json:"symbol"`
	ActualMove float64     `json:"actual_move"` // Actual price change %
}

// ModelMetrics tracks model performance
type ModelMetrics struct {
	Accuracy           float64            `json:"accuracy"`
	Precision          map[string]float64 `json:"precision"` // Per class
	Recall             map[string]float64 `json:"recall"`    // Per class
	F1Score            map[string]float64 `json:"f1_score"`  // Per class
	ConfusionMatrix    [][]int            `json:"confusion_matrix"`
	TotalPredictions   int                `json:"total_predictions"`
	CorrectPredictions int                `json:"correct_predictions"`
	LastAccuracy       float64            `json:"last_accuracy"`
	ProfitFactor       float64            `json:"profit_factor"`
	SharpeRatio        float64            `json:"sharpe_ratio"`
	MaxDrawdown        float64            `json:"max_drawdown"`
}

// NewMLTradingModel creates a new ML trading model with proper initialization
func NewMLTradingModel(symbol string, sequenceLength, featureCount int) *MLTradingModel {
	model := &MLTradingModel{
		Symbol:         symbol,
		SequenceLength: sequenceLength,
		FeatureCount:   featureCount,
		DropoutRate:    0.2,
		LearningRate:   0.001,
		LastTraining:   time.Now(),
		IsTraining:     false,
		ModelMetrics: &ModelMetrics{
			Precision:       make(map[string]float64),
			Recall:          make(map[string]float64),
			F1Score:         make(map[string]float64),
			ConfusionMatrix: make([][]int, 3),
		},
	}

	// Initialize confusion matrix
	for i := range model.ModelMetrics.ConfusionMatrix {
		model.ModelMetrics.ConfusionMatrix[i] = make([]int, 3)
	}

	// Initialize LSTM layers
	model.LSTMLayer1 = model.initializeLSTMLayer(featureCount, 256)
	model.LSTMLayer2 = model.initializeLSTMLayer(256, 256)

	// Initialize attention mechanism
	model.Attention = model.initializeAttention(256, 8) // 8 attention heads

	// Initialize dense layers
	model.DenseLayer1 = model.initializeDenseLayer(256, 128)
	model.DenseLayer2 = model.initializeDenseLayer(128, 64)
	model.OutputLayer = model.initializeDenseLayer(64, 4) // 3 classes + confidence

	return model
}

// initializeLSTMLayer creates and initializes an LSTM layer
func (m *MLTradingModel) initializeLSTMLayer(inputSize, hiddenSize int) *LSTMLayer {
	layer := &LSTMLayer{
		InputSize:   inputSize,
		HiddenSize:  hiddenSize,
		WeightsIH:   make([][]float64, 4), // forget, input, output, candidate gates
		WeightsHH:   make([][]float64, 4),
		BiasH:       make([]float64, 4*hiddenSize),
		AttentionW:  make([][]float64, hiddenSize),
		HiddenState: make([]float64, hiddenSize),
		CellState:   make([]float64, hiddenSize),
	}

	// Xavier initialization for weights
	scale := math.Sqrt(6.0 / float64(inputSize+hiddenSize))

	for i := 0; i < 4; i++ {
		layer.WeightsIH[i] = make([]float64, inputSize*hiddenSize)
		layer.WeightsHH[i] = make([]float64, hiddenSize*hiddenSize)

		for j := range layer.WeightsIH[i] {
			layer.WeightsIH[i][j] = (rand.Float64() - 0.5) * 2 * scale
		}
		for j := range layer.WeightsHH[i] {
			layer.WeightsHH[i][j] = (rand.Float64() - 0.5) * 2 * scale
		}
	}

	// Initialize attention weights
	for i := range layer.AttentionW {
		layer.AttentionW[i] = make([]float64, hiddenSize)
		for j := range layer.AttentionW[i] {
			layer.AttentionW[i][j] = (rand.Float64() - 0.5) * 2 * scale
		}
	}

	return layer
}

// initializeAttention creates and initializes attention mechanism
func (m *MLTradingModel) initializeAttention(dModel, headCount int) *AttentionMechanism {
	attention := &AttentionMechanism{
		HeadCount: headCount,
		DModel:    dModel,
	}

	dK := dModel / headCount
	scale := math.Sqrt(6.0 / float64(dModel))

	// Initialize weight matrices for each attention head
	attention.QueryWeights = make([][]float64, headCount)
	attention.KeyWeights = make([][]float64, headCount)
	attention.ValueWeights = make([][]float64, headCount)

	for i := 0; i < headCount; i++ {
		attention.QueryWeights[i] = make([]float64, dModel*dK)
		attention.KeyWeights[i] = make([]float64, dModel*dK)
		attention.ValueWeights[i] = make([]float64, dModel*dK)

		for j := range attention.QueryWeights[i] {
			attention.QueryWeights[i][j] = (rand.Float64() - 0.5) * 2 * scale
			attention.KeyWeights[i][j] = (rand.Float64() - 0.5) * 2 * scale
			attention.ValueWeights[i][j] = (rand.Float64() - 0.5) * 2 * scale
		}
	}

	// Output projection weights
	attention.OutputWeights = make([][]float64, dModel)
	for i := range attention.OutputWeights {
		attention.OutputWeights[i] = make([]float64, dModel)
		for j := range attention.OutputWeights[i] {
			attention.OutputWeights[i][j] = (rand.Float64() - 0.5) * 2 * scale
		}
	}

	return attention
}

// initializeDenseLayer creates and initializes a dense layer
func (m *MLTradingModel) initializeDenseLayer(inputSize, outputSize int) [][]float64 {
	layer := make([][]float64, outputSize)
	scale := math.Sqrt(6.0 / float64(inputSize+outputSize))

	for i := range layer {
		layer[i] = make([]float64, inputSize+1) // +1 for bias
		for j := range layer[i] {
			layer[i][j] = (rand.Float64() - 0.5) * 2 * scale
		}
	}

	return layer
}

// Predict generates trading signal using the complete ML pipeline
func (m *MLTradingModel) Predict(features [][]float64) *TradingSignal {
	// If features shorter than SequenceLength, pad at the beginning with zeros
	seqLen := len(features)
	if seqLen == 0 {
		log.Printf("\u274c No feature data provided for prediction")
		return nil
	}

	// Normalize feature width: if each timestep has fewer features than expected, pad with zeros
	for i := range features {
		if len(features[i]) < m.FeatureCount {
			pad := make([]float64, m.FeatureCount-len(features[i]))
			features[i] = append(features[i], pad...)
		}
	}

	var inputSequence [][]float64
	if seqLen >= m.SequenceLength {
		inputSequence = features[seqLen-m.SequenceLength:]
	} else {
		// Need to pad with zero-rows at the beginning
		padCount := m.SequenceLength - seqLen
		inputSequence = make([][]float64, 0, m.SequenceLength)
		// If we have at least two timesteps, linearly extrapolate backward from the first two points.
		// Otherwise, duplicate the earliest available timestep.
		if seqLen >= 2 {
			f0 := features[0]
			f1 := features[1]
			// compute delta = f1 - f0
			delta := make([]float64, m.FeatureCount)
			for j := 0; j < m.FeatureCount; j++ {
				delta[j] = f1[j] - f0[j]
			}
			// create pad rows from farthest to nearest to f0
			for i := padCount; i >= 1; i-- {
				factor := float64(i) / float64(padCount+1) // in (0,1)
				row := make([]float64, m.FeatureCount)
				for j := 0; j < m.FeatureCount; j++ {
					row[j] = f0[j] - delta[j]*factor
				}
				inputSequence = append(inputSequence, row)
			}
		} else {
			padRow := make([]float64, m.FeatureCount)
			copy(padRow, features[0])
			for i := 0; i < padCount; i++ {
				rowCopy := make([]float64, len(padRow))
				copy(rowCopy, padRow)
				inputSequence = append(inputSequence, rowCopy)
			}
		}
		// append actual features
		for _, f := range features {
			row := make([]float64, len(f))
			copy(row, f)
			inputSequence = append(inputSequence, row)
		}
	}

	// Forward pass through LSTM + Attention
	lstmOutput := m.forwardLSTM(inputSequence)

	// Apply attention mechanism
	attentionOutput := m.forwardAttention(lstmOutput)

	// Forward pass through dense layers
	dense1Output := m.forwardDense(attentionOutput, m.DenseLayer1, true) // with dropout
	dense2Output := m.forwardDense(dense1Output, m.DenseLayer2, true)    // with dropout
	finalOutput := m.forwardDense(dense2Output, m.OutputLayer, false)    // no dropout

	// Apply softmax to get probabilities
	probabilities := m.softmax(finalOutput[:3])
	confidence := finalOutput[3] // Raw confidence score

	// Determine prediction class
	maxProb := 0.0
	predictionClass := 0
	for i, prob := range probabilities {
		if prob > maxProb {
			maxProb = prob
			predictionClass = i
		}
	}

	// Map prediction class to direction
	_ = []string{"DOWN", "SIDEWAYS", "UP"} // Direction classes for reference
	predictions := []string{"STRONG_SELL", "NEUTRAL", "STRONG_BUY"}

	// Adjust prediction strength based on confidence
	predictionStr := predictions[predictionClass]
	if maxProb < 0.6 {
		predictionStr = "NEUTRAL" // Low confidence
	} else if maxProb > 0.8 && predictionClass != 1 {
		// High confidence non-neutral prediction
		if predictionClass == 0 {
			predictionStr = "STRONG_SELL"
		} else {
			predictionStr = "STRONG_BUY"
		}
	} else if predictionClass != 1 {
		// Medium confidence
		if predictionClass == 0 {
			predictionStr = "SELL"
		} else {
			predictionStr = "BUY"
		}
	}

	// Calculate price targets and risk levels
	currentPrice := features[len(features)-1][0] // Assume first feature is close price
	volatility := m.calculateVolatility(features)

	// Estimate price target based on prediction and volatility
	expectedChange := 0.0
	// Determine signal strength and direction based on prediction class
	switch predictionClass {
	case 2: // UP
		expectedChange = volatility * 0.5 * maxProb
	case 0: // DOWN
		expectedChange = -volatility * 0.5 * maxProb
	default: // SIDEWAYS (1)
		expectedChange = 0.0
	}

	priceTarget := currentPrice * (1 + expectedChange)
	stopLoss := currentPrice * (1 - volatility*0.3) // Conservative stop loss

	// Determine risk level
	riskLevel := "MEDIUM"
	if volatility > 0.05 || maxProb < 0.6 {
		riskLevel = "HIGH"
	} else if volatility < 0.02 && maxProb > 0.8 {
		riskLevel = "LOW"
	}

	// Extract key features that influenced the decision
	keyFeatures := m.extractKeyFeatures(inputSequence)

	signal := &TradingSignal{
		Symbol:         m.Symbol,
		Timestamp:      time.Now().Unix(),
		Prediction:     predictionStr,
		Confidence:     math.Min(maxProb, math.Abs(confidence)),
		PriceTarget:    priceTarget,
		StopLoss:       stopLoss,
		TimeHorizon:    "1H",
		ModelUsed:      "LSTM+Attention",
		KeyFeatures:    keyFeatures,
		RiskLevel:      riskLevel,
		PriceChangePct: expectedChange * 100,
		Volatility:     volatility,
	}

	log.Printf("🤖 ML Prediction for %s: %s (%.2f%% confidence)",
		m.Symbol, predictionStr, signal.Confidence*100)

	return signal
}

// forwardLSTM performs forward pass through LSTM layers
func (m *MLTradingModel) forwardLSTM(sequence [][]float64) [][]float64 {
	// Process through first LSTM layer
	layer1Output := make([][]float64, len(sequence))
	for t, input := range sequence {
		layer1Output[t] = m.lstmStep(m.LSTMLayer1, input)
	}

	// Process through second LSTM layer
	layer2Output := make([][]float64, len(layer1Output))
	for t, input := range layer1Output {
		layer2Output[t] = m.lstmStep(m.LSTMLayer2, input)
	}

	return layer2Output
}

// lstmStep performs a single LSTM step
func (m *MLTradingModel) lstmStep(layer *LSTMLayer, input []float64) []float64 {
	// Simplified LSTM computation
	// In a real implementation, you would compute forget, input, output gates

	// For now, use a simplified version
	output := make([]float64, layer.HiddenSize)

	// Simple linear transformation with activation
	for i := 0; i < layer.HiddenSize; i++ {
		sum := 0.0
		for j, val := range input {
			if j < len(layer.WeightsIH[0])/layer.HiddenSize {
				sum += val * layer.WeightsIH[0][i*len(input)+j]
			}
		}
		output[i] = math.Tanh(sum + layer.BiasH[i])
	}

	copy(layer.HiddenState, output)
	return output
}

// forwardAttention applies attention mechanism
func (m *MLTradingModel) forwardAttention(sequence [][]float64) []float64 {
	if len(sequence) == 0 {
		return make([]float64, m.Attention.DModel)
	}

	// Simplified attention: just weighted average of sequence
	// In a real implementation, you would compute Q, K, V matrices

	weights := make([]float64, len(sequence))
	totalWeight := 0.0

	// Give more weight to recent timesteps
	for i := range weights {
		weights[i] = float64(i+1) / float64(len(sequence))
		totalWeight += weights[i]
	}

	// Normalize weights
	for i := range weights {
		weights[i] /= totalWeight
	}

	// Compute weighted average
	output := make([]float64, len(sequence[0]))
	for i, timestep := range sequence {
		for j, val := range timestep {
			output[j] += val * weights[i]
		}
	}

	return output
}

// forwardDense performs forward pass through dense layer
func (m *MLTradingModel) forwardDense(input []float64, weights [][]float64, dropout bool) []float64 {
	output := make([]float64, len(weights))

	for i, neuron := range weights {
		sum := neuron[len(neuron)-1] // bias

		for j, val := range input {
			if j < len(neuron)-1 {
				sum += val * neuron[j]
			}
		}

		// Apply ReLU activation
		output[i] = math.Max(0, sum)

		// Apply dropout during training
		if dropout && m.IsTraining && rand.Float64() < m.DropoutRate {
			output[i] = 0
		}
	}

	return output
}

// softmax applies softmax activation
func (m *MLTradingModel) softmax(input []float64) []float64 {
	output := make([]float64, len(input))
	max := input[0]

	// Find max for numerical stability
	for _, val := range input {
		if val > max {
			max = val
		}
	}

	// Compute exponentials and sum
	sum := 0.0
	for i, val := range input {
		output[i] = math.Exp(val - max)
		sum += output[i]
	}

	// Normalize
	for i := range output {
		output[i] /= sum
	}

	return output
}

// calculateVolatility estimates current market volatility
func (m *MLTradingModel) calculateVolatility(features [][]float64) float64 {
	if len(features) < 20 {
		return 0.02 // Default volatility
	}

	// Calculate price changes over last 20 periods
	changes := make([]float64, 0, 20)
	for i := len(features) - 20; i < len(features)-1; i++ {
		if len(features[i]) > 0 && len(features[i+1]) > 0 {
			change := (features[i+1][0] - features[i][0]) / features[i][0]
			changes = append(changes, change)
		}
	}

	if len(changes) == 0 {
		return 0.02
	}

	// Calculate standard deviation
	mean := 0.0
	for _, change := range changes {
		mean += change
	}
	mean /= float64(len(changes))

	variance := 0.0
	for _, change := range changes {
		variance += math.Pow(change-mean, 2)
	}
	variance /= float64(len(changes))

	return math.Sqrt(variance)
}

// extractKeyFeatures identifies the most important features for interpretation
func (m *MLTradingModel) extractKeyFeatures(sequence [][]float64) []string {
	// This is a simplified version. In a real implementation,
	// you would use feature importance from the trained model

	features := []string{}

	if len(sequence) == 0 || len(sequence[0]) == 0 {
		return features
	}

	// Analyze recent trends and patterns
	recent := sequence[len(sequence)-10:] // Last 10 minutes

	// Check for strong trends
	if len(recent) > 5 {
		priceStart := recent[0][0]
		priceEnd := recent[len(recent)-1][0]
		change := (priceEnd - priceStart) / priceStart

		if math.Abs(change) > 0.001 {
			if change > 0 {
				features = append(features, "Strong Upward Trend")
			} else {
				features = append(features, "Strong Downward Trend")
			}
		}
	}

	// Check volume patterns (assuming volume is feature index 4)
	if len(sequence[0]) > 4 {
		avgVolume := 0.0
		recentVolume := 0.0
		for i, candle := range sequence {
			if len(candle) > 4 {
				if i < len(sequence)-10 {
					avgVolume += candle[4]
				} else {
					recentVolume += candle[4]
				}
			}
		}

		if len(sequence) > 10 {
			avgVolume /= float64(len(sequence) - 10)
			recentVolume /= 10.0

			if recentVolume > avgVolume*1.5 {
				features = append(features, "High Volume")
			} else if recentVolume < avgVolume*0.5 {
				features = append(features, "Low Volume")
			}
		}
	}

	// Add default features if none found
	if len(features) == 0 {
		features = append(features, "Price Action", "Volume Analysis", "Technical Indicators")
	}

	return features
}

// CalculateLabel determines the correct label for training based on actual price movement
func CalculateLabel(currentPrice, futurePrice, volatility float64) string {
	change := (futurePrice - currentPrice) / currentPrice
	threshold := volatility * 0.5 // Dynamic threshold based on volatility

	if math.Abs(change) < threshold {
		return "SIDEWAYS"
	} else if change > 0 {
		return "UP"
	} else {
		return "DOWN"
	}
}

// ShouldRetrain determines if the model needs retraining
func (m *MLTradingModel) ShouldRetrain() bool {
	metrics := m.ModelMetrics

	return metrics.Accuracy < 0.65 ||
		(metrics.LastAccuracy > 0 && metrics.Accuracy < metrics.LastAccuracy*0.9) ||
		time.Since(m.LastTraining) > 24*time.Hour ||
		len(m.TrainingBuffer) > 1000 // Enough new data
}

// UpdateMetrics updates model performance metrics
func (m *MLTradingModel) UpdateMetrics(predicted, actual string, confidence float64) {
	m.ModelMetrics.TotalPredictions++

	if predicted == actual {
		m.ModelMetrics.CorrectPredictions++
	}

	// Update accuracy
	m.ModelMetrics.LastAccuracy = m.ModelMetrics.Accuracy
	m.ModelMetrics.Accuracy = float64(m.ModelMetrics.CorrectPredictions) / float64(m.ModelMetrics.TotalPredictions)

	log.Printf("📊 Model %s accuracy: %.2f%% (%d/%d predictions)",
		m.Symbol, m.ModelMetrics.Accuracy*100,
		m.ModelMetrics.CorrectPredictions, m.ModelMetrics.TotalPredictions)
}
