package main

import (
	"database/sql"
	"fmt"
	"log"
	"os"
	"time"

	_ "github.com/ClickHouse/clickhouse-go/v2"
)

func main() {
	dsn := os.Getenv("CH_DSN")
	if dsn == "" {
		dsn = "clickhouse://app:app_password@localhost:9000/default?dial_timeout=5s&max_execution_time=60"
	}

	db, err := sql.Open("clickhouse", dsn)
	if err != nil {
		log.Fatalf("Failed to connect to ClickHouse: %v", err)
	}
	defer db.Close()

	// Test connection
	if err := db.Ping(); err != nil {
		log.Fatalf("Failed to ping ClickHouse: %v", err)
	}

	fmt.Println("=== ClickHouse Data Status ===\n")

	// 1. Check candle_1m table
	fmt.Println("1. candle_1m table (raw market data):")
	var candleCount int
	if err := db.QueryRow("SELECT COUNT(*) FROM candle_1m").Scan(&candleCount); err != nil {
		fmt.Printf("   Error: %v\n", err)
	} else {
		fmt.Printf("   Total candles: %d\n", candleCount)
	}

	var minCandleTS, maxCandleTS sql.NullInt64
	if err := db.QueryRow("SELECT MIN(timestamp), MAX(timestamp) FROM candle_1m").Scan(&minCandleTS, &maxCandleTS); err != nil {
		fmt.Printf("   Timestamp range error: %v\n", err)
	} else {
		if minCandleTS.Valid {
			fmt.Printf("   Min timestamp: %s\n", time.Unix(minCandleTS.Int64, 0).Format("2006-01-02 15:04:05"))
		}
		if maxCandleTS.Valid {
			fmt.Printf("   Max timestamp: %s\n", time.Unix(maxCandleTS.Int64, 0).Format("2006-01-02 15:04:05"))
		}
	}

	var distinctSymbols int
	if err := db.QueryRow("SELECT COUNT(DISTINCT symbol) FROM candle_1m").Scan(&distinctSymbols); err != nil {
		fmt.Printf("   Distinct symbols error: %v\n", err)
	} else {
		fmt.Printf("   Distinct symbols: %d\n", distinctSymbols)
	}

	// Show sample symbols
	rows, err := db.Query("SELECT DISTINCT symbol FROM candle_1m ORDER BY symbol LIMIT 10")
	if err == nil {
		defer rows.Close()
		fmt.Print("   Sample symbols: ")
		var symbols []string
		for rows.Next() {
			var s string
			if err := rows.Scan(&s); err == nil {
				symbols = append(symbols, s)
			}
		}
		fmt.Println(symbols)
	}

	fmt.Println()

	// 2. Check candle_cache table
	fmt.Println("2. candle_cache table (processed candles):")
	var cacheCount int
	if err := db.QueryRow("SELECT COUNT(*) FROM candle_cache").Scan(&cacheCount); err != nil {
		fmt.Printf("   Error: %v\n", err)
	} else {
		fmt.Printf("   Total candles: %d\n", cacheCount)
	}

	if err := db.QueryRow("SELECT MIN(timestamp), MAX(timestamp) FROM candle_cache").Scan(&minCandleTS, &maxCandleTS); err != nil {
		fmt.Printf("   Timestamp range error: %v\n", err)
	} else {
		if minCandleTS.Valid {
			fmt.Printf("   Min timestamp: %s\n", time.Unix(minCandleTS.Int64, 0).Format("2006-01-02 15:04:05"))
		}
		if maxCandleTS.Valid {
			fmt.Printf("   Max timestamp: %s\n", time.Unix(maxCandleTS.Int64, 0).Format("2006-01-02 15:04:05"))
		}
	}

	fmt.Println()

	// 3. Check direction_predictions table
	fmt.Println("3. direction_predictions table (ML predictions):")
	var predCount int
	if err := db.QueryRow("SELECT COUNT(*) FROM direction_predictions").Scan(&predCount); err != nil {
		fmt.Printf("   Error: %v\n", err)
	} else {
		fmt.Printf("   Total predictions: %d\n", predCount)
	}

	var minPredTS, maxPredTS sql.NullTime
	if err := db.QueryRow("SELECT MIN(timestamp), MAX(timestamp) FROM direction_predictions").Scan(&minPredTS, &maxPredTS); err != nil {
		fmt.Printf("   Timestamp range error: %v\n", err)
	} else {
		if minPredTS.Valid {
			fmt.Printf("   Min timestamp: %s\n", minPredTS.Time.Format("2006-01-02 15:04:05"))
		}
		if maxPredTS.Valid {
			fmt.Printf("   Max timestamp: %s\n", maxPredTS.Time.Format("2006-01-02 15:04:05"))
		}
	}

	if err := db.QueryRow("SELECT COUNT(DISTINCT symbol) FROM direction_predictions").Scan(&distinctSymbols); err != nil {
		fmt.Printf("   Distinct symbols error: %v\n", err)
	} else {
		fmt.Printf("   Distinct symbols: %d\n", distinctSymbols)
	}

	// Show prediction symbols
	rows, err = db.Query("SELECT DISTINCT symbol FROM direction_predictions ORDER BY symbol LIMIT 10")
	if err == nil {
		defer rows.Close()
		fmt.Print("   Prediction symbols: ")
		var symbols []string
		for rows.Next() {
			var s string
			if err := rows.Scan(&s); err == nil {
				symbols = append(symbols, s)
			}
		}
		if len(symbols) > 0 {
			fmt.Println(symbols)
		} else {
			fmt.Println("(none)")
		}
	}

	fmt.Println()

	// 4. Show prediction statistics by direction
	fmt.Println("4. Prediction distribution (last 24 hours):")
	distRows, err := db.Query(`
		SELECT direction, count() as cnt, avg(confidence) as avg_conf
		FROM direction_predictions
		WHERE created_at >= now() - INTERVAL 24 HOUR
		GROUP BY direction
		ORDER BY direction
	`)
	if err != nil {
		fmt.Printf("   Error: %v\n", err)
	} else {
		defer distRows.Close()
		found := false
		for distRows.Next() {
			found = true
			var direction string
			var count int64
			var avgConf float64
			if err := distRows.Scan(&direction, &count, &avgConf); err == nil {
				fmt.Printf("   %s: %d predictions (avg confidence: %.4f)\n", direction, count, avgConf)
			}
		}
		if !found {
			fmt.Println("   No predictions in last 24 hours")
		}
	}

	fmt.Println()

	// 5. Show latest predictions
	fmt.Println("5. Latest 10 predictions:")
	sampleRows, err := db.Query(`
		SELECT symbol, timestamp, direction, confidence, trust_stage, model_used, created_at
		FROM direction_predictions
		ORDER BY created_at DESC
		LIMIT 10
	`)
	if err != nil {
		fmt.Printf("   Error: %v\n", err)
	} else {
		defer sampleRows.Close()
		found := false
		for sampleRows.Next() {
			found = true
			var symbol, direction, trustStage, modelUsed string
			var ts, created time.Time
			var confidence float64
			if err := sampleRows.Scan(&symbol, &ts, &direction, &confidence, &trustStage, &modelUsed, &created); err == nil {
				fmt.Printf("   [%s] %s %s (conf: %.4f, stage: %s, model: %s)\n",
					ts.Format("15:04:05"), symbol, direction, confidence, trustStage, modelUsed)
			}
		}
		if !found {
			fmt.Println("   No predictions found")
		}
	}

	fmt.Println()

	// 6. Check model performance (if actual_direction is populated)
	fmt.Println("6. Model accuracy (where actual_direction is known):")
	var accuracyCount int
	if err := db.QueryRow(`
		SELECT COUNT(*)
		FROM direction_predictions
		WHERE actual_direction != '' AND actual_direction IS NOT NULL
	`).Scan(&accuracyCount); err != nil {
		fmt.Printf("   Error: %v\n", err)
	} else {
		fmt.Printf("   Predictions with labels: %d\n", accuracyCount)
		if accuracyCount > 0 {
			var accuracy sql.NullFloat64
			if err := db.QueryRow(`
				SELECT avg(if(actual_direction = direction, 1.0, 0.0)) * 100
				FROM direction_predictions
				WHERE actual_direction != '' AND actual_direction IS NOT NULL
			`).Scan(&accuracy); err != nil {
				fmt.Printf("   Accuracy calculation error: %v\n", err)
			} else if accuracy.Valid {
				fmt.Printf("   Accuracy: %.2f%%\n", accuracy.Float64)
			}
		}
	}

	fmt.Println()
	fmt.Println("=== Analysis Complete ===")
}
