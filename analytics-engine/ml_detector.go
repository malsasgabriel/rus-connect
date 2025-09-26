package main

import (
	"math"
	"sync"
)

// MLPumpDetector is a lightweight ML predictor used to approximate an LSTM-style model.
// It is intentionally simple to keep in-process inference fast (<50ms) for 5k+ streams.
type MLPumpDetector struct {
	featureEngine *FeatureEngine
	mu            sync.Mutex

	// tiny state for memory, simulating a single hidden layer RNN/LSTM cell
	hidden []float64
	// weights: 4 hidden units, 5 inputs (volume spike, price change 5m, price change 15m, RSI, volatility)
	W1 [][]float64
	b1 []float64
	W2 []float64
	b2 float64
}

func NewMLPumpDetector() *MLPumpDetector {
	// initialize 4 hidden units with small random-like values (deterministic seed not required for production)
	W1 := make([][]float64, 4)
	for i := range W1 {
		W1[i] = []float64{0.1, 0.1, 0.1, 0.1, 0.1}
	}
	return &MLPumpDetector{
		featureEngine: NewFeatureEngine(),
		hidden:        make([]float64, 4),
		W1:            W1,
		b1:            []float64{0.0, 0.0, 0.0, 0.0},
		W2:            []float64{0.2, 0.2, 0.2, 0.2},
		b2:            0.0,
	}
}

// sigmoid helper
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Predict performs a lightweight forward pass to approximate an ML model.
func (m *MLPumpDetector) Predict(features FeatureSet) float64 {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Build an input vector from available features (normalized rough ranges)
	input := []float64{
		minMaxNormalize(features.VolumeSpikeRatio, 0.0, 10.0),
		minMaxNormalize(features.PriceChange5m, -0.2, 0.2),
		minMaxNormalize(features.PriceChange15m, -0.5, 0.5),
		minMaxNormalize(features.RSI, 0.0, 100.0),
		minMaxNormalize(features.Volatility5m, 0.0, 0.1),
		// Enhanced features
		minMaxNormalize(features.VolumeMomentum, -1.0, 1.0),
		minMaxNormalize(features.PriceMomentum, -0.3, 0.3),
		minMaxNormalize(features.BollingerPosition, 0.0, 1.0),
		minMaxNormalize(features.TrendStrength, -0.1, 0.1),
	}

	// Adjust weights for enhanced features
	if len(m.W1[0]) < len(input) {
		// Expand weights for new features
		for i := range m.W1 {
			for len(m.W1[i]) < len(input) {
				m.W1[i] = append(m.W1[i], 0.1) // Initialize new weights
			}
		}
	}

	// hidden layer (4 nodes)
	newHidden := make([]float64, 4)
	for i := 0; i < 4; i++ {
		sum := m.b1[i]
		for j := 0; j < len(input) && j < len(m.W1[i]); j++ {
			sum += m.W1[i][j] * input[j]
		}
		newHidden[i] = math.Tanh(sum)
	}
	m.hidden = newHidden

	// output neuron
	out := m.b2
	for i := 0; i < 4; i++ {
		out += m.W2[i] * m.hidden[i]
	}
	return sigmoid(out)
}

// ProcessCandle is the main method for processing a new candle, generating features, and predicting.
func (m *MLPumpDetector) ProcessCandle(candle Candle) (float64, FeatureSet) {
	features := m.featureEngine.AddCandle(candle)
	probability := m.Predict(features)
	return probability, features
}

func minMaxNormalize(v, min, max float64) float64 {
	if max == min {
		return 0
	}
	if v < min {
		return 0
	}
	if v > max {
		return 1
	}
	return (v - min) / (max - min)
}
