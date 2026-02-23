package main

import (
	"database/sql"
	"encoding/json"
	"log"
)

// SaveToDB saves the neural network weights to the database
func (nn *SimpleNeuralNetwork) SaveToDB(db *sql.DB) error {
	inputToHiddenWeightsJSON, err := json.Marshal(nn.InputToHiddenWeights)
	if err != nil {
		return err
	}

	hiddenToOutputWeightsJSON, err := json.Marshal(nn.HiddenToOutputWeights)
	if err != nil {
		return err
	}

	hiddenBiasesJSON, err := json.Marshal(nn.HiddenBiases)
	if err != nil {
		return err
	}

	outputBiasesJSON, err := json.Marshal(nn.OutputBiases)
	if err != nil {
		return err
	}

	query := `INSERT INTO model_weights
		(symbol, weights, biases, hidden_weights, hidden_biases, learning_rate, last_accuracy, prediction_count, correct_count, updated_at)
	VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, now())`

	_, err = db.Exec(query, nn.Symbol, inputToHiddenWeightsJSON, outputBiasesJSON,
		hiddenToOutputWeightsJSON, hiddenBiasesJSON,
		nn.LearningRate, nn.LastAccuracy, nn.PredictionCount, nn.CorrectCount)

	if err != nil {
		return err
	}

	log.Printf("✅ Saved model weights for %s to database", nn.Symbol)
	return nil
}

// LoadFromDB loads the neural network weights from the database
func (nn *SimpleNeuralNetwork) LoadFromDB(db *sql.DB, symbol string) error {
	var inputToHiddenWeightsJSON, hiddenToOutputWeightsJSON, hiddenBiasesJSON, outputBiasesJSON []byte
	var learningRate, lastAccuracy sql.NullFloat64
	var predictionCount, correctCount sql.NullInt64

	query := `SELECT weights, biases, hidden_weights, hidden_biases, learning_rate, last_accuracy, prediction_count, correct_count
		FROM model_weights
		WHERE symbol = ?
		ORDER BY updated_at DESC
		LIMIT 1`

	err := db.QueryRow(query, symbol).Scan(&inputToHiddenWeightsJSON, &outputBiasesJSON,
		&hiddenToOutputWeightsJSON, &hiddenBiasesJSON,
		&learningRate, &lastAccuracy, &predictionCount, &correctCount)

	if err != nil {
		if err == sql.ErrNoRows {
			log.Printf("No saved model weights found for %s, using random initialization", symbol)
			return nil // Not an error, just no saved model yet
		}
		return err
	}

	// Unmarshal input-to-hidden weights
	if err := json.Unmarshal(inputToHiddenWeightsJSON, &nn.InputToHiddenWeights); err != nil {
		return err
	}

	// Unmarshal hidden-to-output weights
	if err := json.Unmarshal(hiddenToOutputWeightsJSON, &nn.HiddenToOutputWeights); err != nil {
		return err
	}

	// Unmarshal hidden biases
	if err := json.Unmarshal(hiddenBiasesJSON, &nn.HiddenBiases); err != nil {
		return err
	}

	// Unmarshal output biases
	if err := json.Unmarshal(outputBiasesJSON, &nn.OutputBiases); err != nil {
		return err
	}

	// Set other fields
	if learningRate.Valid {
		nn.LearningRate = learningRate.Float64
	}
	if lastAccuracy.Valid {
		nn.LastAccuracy = lastAccuracy.Float64
	}
	if predictionCount.Valid {
		nn.PredictionCount = int(predictionCount.Int64)
	}
	if correctCount.Valid {
		nn.CorrectCount = int(correctCount.Int64)
	}

	nn.Symbol = symbol

	// Only mark as trained if we have valid accuracy data and it meets the minimum threshold
	// This prevents marking invalid or outdated models as trained
	const MIN_ACCURACY_THRESHOLD = 0.55
	if lastAccuracy.Valid && lastAccuracy.Float64 >= MIN_ACCURACY_THRESHOLD {
		nn.Trained = true
		log.Printf("✅ Loaded model weights for %s from database (accuracy: %.2f%%, marked as trained)",
			symbol, nn.LastAccuracy*100)
	} else {
		nn.Trained = false
		if lastAccuracy.Valid {
			log.Printf("⚠️ Loaded model weights for %s from database (accuracy: %.2f%% < %.2f%%, marked as untrained - will retrain)",
				symbol, nn.LastAccuracy*100, MIN_ACCURACY_THRESHOLD*100)
		} else {
			log.Printf("⚠️ Loaded model weights for %s from database (no accuracy data, marked as untrained - will retrain)",
				symbol)
		}
	}

	return nil
}
