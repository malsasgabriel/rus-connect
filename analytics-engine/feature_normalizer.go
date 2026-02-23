package main

import (
	"encoding/json"
	"log"
	"math"
	"sync"
)

// NewFeatureNormalizer creates a new feature normalizer
func NewFeatureNormalizer(numFeatures int) *FeatureNormalizer {
	return &FeatureNormalizer{
		Means:   make([]float64, numFeatures),
		Stddevs: make([]float64, numFeatures),
		Fitted:  false,
	}
}

// Fit calculates mean and standard deviation from training data
func (fn *FeatureNormalizer) Fit(data [][]float64) {
	if len(data) == 0 {
		log.Printf("Warning: FeatureNormalizer.Fit called with empty data")
		return
	}

	numFeatures := len(data[0])
	if len(fn.Means) != numFeatures {
		log.Printf("Warning: Feature count mismatch in Fit: expected %d, got %d", len(fn.Means), numFeatures)
		fn.Means = make([]float64, numFeatures)
		fn.Stddevs = make([]float64, numFeatures)
	}

	// Calculate means
	for i := 0; i < numFeatures; i++ {
		sum := 0.0
		count := 0
		for _, row := range data {
			if i < len(row) {
				sum += row[i]
				count++
			}
		}
		if count > 0 {
			fn.Means[i] = sum / float64(count)
		}
	}

	// Calculate standard deviations
	for i := 0; i < numFeatures; i++ {
		sumSq := 0.0
		count := 0
		for _, row := range data {
			if i < len(row) {
				diff := row[i] - fn.Means[i]
				sumSq += diff * diff
				count++
			}
		}
		if count > 0 {
			fn.Stddevs[i] = math.Sqrt(sumSq / float64(count))

			// Avoid division by zero - use 1.0 if stddev is too small
			if fn.Stddevs[i] < 1e-10 {
				fn.Stddevs[i] = 1.0
			}
		} else {
			fn.Stddevs[i] = 1.0
		}
	}

	fn.Fitted = true
	log.Printf("✅ FeatureNormalizer fitted with %d samples, %d features", len(data), numFeatures)
}

// Transform normalizes features using z-score normalization
func (fn *FeatureNormalizer) Transform(features []float64) []float64 {
	if !fn.Fitted {
		// If not fitted, return features as-is with a warning
		log.Printf("Warning: FeatureNormalizer not fitted, returning unnormalized features")
		return features
	}

	if len(features) != len(fn.Means) {
		log.Printf("Warning: Feature count mismatch in Transform: expected %d, got %d", len(fn.Means), len(features))
		return features
	}

	normalized := make([]float64, len(features))
	for i, f := range features {
		normalized[i] = (f - fn.Means[i]) / fn.Stddevs[i]
	}
	return normalized
}

// FitTransform fits the normalizer and transforms the data in one step
func (fn *FeatureNormalizer) FitTransform(data [][]float64) [][]float64 {
	fn.Fit(data)

	normalized := make([][]float64, len(data))
	for i, row := range data {
		normalized[i] = fn.Transform(row)
	}
	return normalized
}

// SaveToDB saves normalizer parameters to database
func (fn *FeatureNormalizer) SaveToDB(ae *AnalyticsEngine, symbol string) error {
	data, err := json.Marshal(fn)
	if err != nil {
		return err
	}

	query := `INSERT INTO feature_normalizers (symbol, params, updated_at) 
			  VALUES (?, ?, now())`

	_, err = ae.db.Exec(query, symbol, data)
	return err
}

// LoadFromDB loads normalizer parameters from database
func (fn *FeatureNormalizer) LoadFromDB(ae *AnalyticsEngine, symbol string) error {
	var data []byte
	query := `SELECT params FROM feature_normalizers WHERE symbol = ? ORDER BY updated_at DESC LIMIT 1`

	err := ae.db.QueryRow(query, symbol).Scan(&data)
	if err != nil {
		return err
	}

	if err := json.Unmarshal(data, fn); err != nil {
		return err
	}

	// Ensure Fitted flag is preserved (it should be in JSON, but verify)
	if len(fn.Means) > 0 && len(fn.Stddevs) > 0 {
		// If we have valid parameters, mark as fitted
		fn.Fitted = true
		log.Printf("✅ Loaded feature normalizer for %s from database (fitted: %v)", symbol, fn.Fitted)
	}

	return nil
}

// FeatureNormalizerManager manages normalizers for multiple symbols
type FeatureNormalizerManager struct {
	normalizers map[string]*FeatureNormalizer
	mu          sync.RWMutex
	numFeatures int
}

// NewFeatureNormalizerManager creates a new manager
func NewFeatureNormalizerManager(numFeatures int) *FeatureNormalizerManager {
	return &FeatureNormalizerManager{
		normalizers: make(map[string]*FeatureNormalizer),
		numFeatures: numFeatures,
	}
}

// Get returns normalizer for a symbol, creating if doesn't exist
func (fnm *FeatureNormalizerManager) Get(symbol string) *FeatureNormalizer {
	fnm.mu.RLock()
	normalizer, exists := fnm.normalizers[symbol]
	fnm.mu.RUnlock()

	if !exists {
		fnm.mu.Lock()
		// Double-check after acquiring write lock
		normalizer, exists = fnm.normalizers[symbol]
		if !exists {
			normalizer = NewFeatureNormalizer(fnm.numFeatures)
			fnm.normalizers[symbol] = normalizer
		}
		fnm.mu.Unlock()
	}

	return normalizer
}

// LoadAll loads all normalizers from database
func (fnm *FeatureNormalizerManager) LoadAll(ae *AnalyticsEngine) error {
	query := `SELECT symbol, params FROM feature_normalizers`
	rows, err := ae.db.Query(query)
	if err != nil {
		return err
	}
	defer rows.Close()

	loaded := 0
	for rows.Next() {
		var symbol string
		var data []byte

		if err := rows.Scan(&symbol, &data); err != nil {
			log.Printf("Error scanning normalizer row: %v", err)
			continue
		}

		normalizer := NewFeatureNormalizer(fnm.numFeatures)
		if err := json.Unmarshal(data, normalizer); err != nil {
			log.Printf("Error unmarshaling normalizer for %s: %v", symbol, err)
			continue
		}

		// Ensure Fitted flag is preserved when loading from DB
		if len(normalizer.Means) > 0 && len(normalizer.Stddevs) > 0 {
			normalizer.Fitted = true
		}

		fnm.mu.Lock()
		fnm.normalizers[symbol] = normalizer
		fnm.mu.Unlock()
		loaded++
	}

	log.Printf("✅ Loaded %d feature normalizers from database", loaded)
	return nil
}
