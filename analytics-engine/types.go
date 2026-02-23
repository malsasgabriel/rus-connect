package main

import (
	"encoding/json"
	"time"
)

// Candle represents a single candlestick
type Candle struct {
	Symbol    string  `json:"symbol"`
	Timestamp int64   `json:"timestamp"`
	Open      float64 `json:"open"`
	High      float64 `json:"high"`
	Low       float64 `json:"low"`
	Close     float64 `json:"close"`
	Volume    float64 `json:"volume"`
}

// DirectionSignal represents a trading signal
type DirectionSignal struct {
	Symbol          string          `json:"symbol"`
	Timestamp       int64           `json:"timestamp"`
	Direction       string          `json:"direction"`
	Confidence      float64         `json:"confidence"`
	ClassProbs      ClassProbs      `json:"class_probs"`
	CurrentPrice    float64         `json:"current_price"`
	PriceTarget     float64         `json:"price_target"`
	TimeHorizon     int             `json:"time_horizon"`
	LabelHorizonMin int             `json:"label_horizon_min"`
	StopLoss        float64         `json:"stop_loss"`
	Volatility      float64         `json:"volatility"` // Market volatility (0-1 scale)
	Factors         []string        `json:"factors"`
	RiskLevel       string          `json:"risk_level"`
	TrustStage      string          `json:"trust_stage"`
	ModelAgeSec     int64           `json:"model_age_sec"`
	PredictionCount int             `json:"prediction_count"`
	ModelType       string          `json:"model_type"`
	ModelUsed       string          `json:"model_used"`
	Version         string          `json:"version"`
	Features        json.RawMessage `json:"features"`
	ActualDirection *string         `json:"actual_direction,omitempty"`
	CreatedAt       time.Time       `json:"created_at"`
	TimestampISO    string          `json:"timestamp_iso,omitempty"`
}

// SimpleNeuralNetwork represents a simple neural network for trading
type SimpleNeuralNetwork struct {
	Symbol                string      `json:"symbol"`
	InputToHiddenWeights  [][]float64 `json:"input_to_hidden_weights"`  // Input to Hidden weights
	HiddenToOutputWeights [][]float64 `json:"hidden_to_output_weights"` // Hidden to Output weights
	HiddenBiases          []float64   `json:"hidden_biases"`            // Hidden biases
	OutputBiases          []float64   `json:"output_biases"`            // Output biases
	LearningRate          float64     `json:"learning_rate"`
	LastAccuracy          float64     `json:"last_accuracy"`
	PredictionCount       int         `json:"prediction_count"`
	CorrectCount          int         `json:"correct_count"`
	LastUpdate            time.Time   `json:"last_update"`
	Trained               bool        `json:"trained"`
}

// TrainingExample represents a training example
type TrainingExample struct {
	Features []float64 `json:"features"`
	Target   int       `json:"target"` // 0=DOWN, 1=SIDEWAYS, 2=UP
	Ts       int64     `json:"ts"`
}

// ClassProbs stores class probabilities for DOWN/SIDEWAYS/UP.
type ClassProbs struct {
	Down     float64 `json:"down"`
	Sideways float64 `json:"sideways"`
	Up       float64 `json:"up"`
}

// PendingExample represents a prediction waiting for label generation
type PendingExample struct {
	Symbol             string    `json:"symbol"`
	Features           []float64 `json:"features"`
	Timestamp          time.Time `json:"timestamp"`
	PredictedDirection int       `json:"predicted_direction"` // 0=DOWN, 1=SIDEWAYS, 2=UP
	EntryPrice         float64   `json:"entry_price"`
	TimeHorizon        int       `json:"time_horizon"` // Minutes until we check actual outcome
}

// Feedback represents user feedback on a prediction
type Feedback struct {
	Symbol        string  `json:"symbol"`
	Timestamp     int64   `json:"timestamp"`
	PredictedProb float64 `json:"predicted_prob"`
	ActualPump    bool    `json:"actual_pump"`
	Confidence    float64 `json:"confidence"`
	Notes         string  `json:"notes"`
}

// FeatureNormalizer normalizes features using z-score normalization
type FeatureNormalizer struct {
	Means   []float64 `json:"means"`
	Stddevs []float64 `json:"stddevs"`
	Fitted  bool      `json:"fitted"`
}
