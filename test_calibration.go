package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"time"

	_ "github.com/lib/pq"
)

func main() {
	// PostgreSQL connection
	dsn := os.Getenv("PG_DSN")
	if dsn == "" {
		dsn = "host=localhost user=admin password=password dbname=predpump sslmode=disable"
	}

	db, err := sql.Open("postgres", dsn)
	if err != nil {
		log.Fatalf("Failed to connect to PostgreSQL: %v", err)
	}
	defer db.Close()

	if err := db.Ping(); err != nil {
		log.Fatalf("Failed to ping PostgreSQL: %v", err)
	}

	fmt.Println("✅ Database connection is valid")

	// Test the query that's used in performCalibration
	symbol := "BTCUSDT"
	fmt.Printf("🔧 Querying historical predictions for %s...\n", symbol)
	rows, err := db.Query(`
		SELECT confidence, 
		       CASE 
			       WHEN actual_direction = direction THEN 1 
				   ELSE 0 
			   END as is_correct
		FROM direction_predictions 
		WHERE symbol = $1 
		  AND confidence IS NOT NULL 
		  AND actual_direction IS NOT NULL
		ORDER BY timestamp DESC 
		LIMIT 1000
	`, symbol)

	if err != nil {
		log.Printf("❌ Failed to query historical predictions for %s: %v", symbol, err)
		return
	}
	defer rows.Close()

	// Initialize calibration bins
	calibrationBinsN := 10
	calibrationBins := make([]float64, calibrationBinsN)
	binCounts := make([]int, calibrationBinsN)
	binCorrect := make([]int, calibrationBinsN)

	// Process historical predictions
	fmt.Printf("🔧 Processing historical predictions for %s...\n", symbol)
	rowCount := 0
	for rows.Next() {
		var confidence float64
		var isCorrect int

		if err := rows.Scan(&confidence, &isCorrect); err != nil {
			log.Printf("❌ Failed to scan prediction row for %s: %v", symbol, err)
			continue
		}

		// Determine which bin this confidence falls into
		binIndex := int(confidence * float64(calibrationBinsN))
		if binIndex < 0 {
			binIndex = 0
		}
		if binIndex >= calibrationBinsN {
			binIndex = calibrationBinsN - 1
		}

		// Update bin statistics
		binCounts[binIndex]++
		if isCorrect == 1 {
			binCorrect[binIndex]++
		}

		rowCount++
	}

	fmt.Printf("🔧 Processed %d historical predictions for %s\n", rowCount, symbol)

	// Calculate bin accuracies
	for i := range calibrationBins {
		if binCounts[i] > 0 {
			calibrationBins[i] = float64(binCorrect[i]) / float64(binCounts[i])
		} else {
			// If no data in bin, use a default value based on adjacent bins or overall accuracy
			calibrationBins[i] = 0.5
		}
	}

	// Calculate emit threshold (minimum confidence required for signal emission)
	emitThreshold := 0.7

	// Save calibration data to database
	binsData, err := json.Marshal(calibrationBins)
	if err != nil {
		log.Printf("❌ Failed to marshal calibration bins for %s: %v", symbol, err)
		return
	}

	fmt.Printf("🔧 Saving calibration data for %s: threshold=%.2f, bins=%v\n", symbol, emitThreshold, calibrationBins)

	result, err := db.Exec(`
		INSERT INTO model_calibration (symbol, bins, emit_threshold, updated_at)
		VALUES ($1, $2, $3, $4)
		ON CONFLICT (symbol) DO UPDATE SET
			bins = $2,
			emit_threshold = $3,
			updated_at = $4
	`, symbol, binsData, emitThreshold, time.Now())

	if err != nil {
		log.Printf("❌ Failed to save calibration for %s: %v", symbol, err)
		return
	}

	// Log the result of the database operation
	rowsAffected, err := result.RowsAffected()
	if err != nil {
		log.Printf("⚠️ Could not get rows affected for %s: %v", symbol, err)
	} else {
		fmt.Printf("✅ Calibration saved for %s: threshold=%.2f, rows affected=%d\n", symbol, emitThreshold, rowsAffected)
	}

	fmt.Println("🔧 Model calibration process completed")
}
