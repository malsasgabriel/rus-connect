package main

import (
	"database/sql"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	_ "github.com/lib/pq"
)

func main() {
	start := flag.String("start", "2025-09-15", "start date YYYY-MM-DD")
	end := flag.String("end", "2025-09-21", "end date YYYY-MM-DD")
	bins := flag.Int("bins", 10, "number of calibration bins")
	threshold := flag.Float64("target", 0.55, "target accuracy for emit threshold")
	move := flag.Float64("move", 0.01, "price move threshold (fraction) to consider UP/DOWN, e.g. 0.005 = 0.5%")
	syms := flag.String("symbols", "", "comma-separated list of symbols to calibrate (default: all)")
	flag.Parse()

	dsn := os.Getenv("PG_DSN")
	if dsn == "" {
		dsn = "host=localhost user=admin password=password dbname=predpump sslmode=disable"
	}

	db, err := sql.Open("postgres", dsn)
	if err != nil {
		log.Fatalf("Failed to open DB: %v", err)
	}
	defer db.Close()

	sTime, _ := time.Parse("2006-01-02", *start)
	eTime, _ := time.Parse("2006-01-02", *end)

	rows, err := db.Query(`
		SELECT id, symbol, timestamp, direction, confidence, time_horizon
		FROM direction_predictions
		WHERE timestamp >= $1 AND timestamp <= $2
		ORDER BY timestamp ASC
	`, sTime, eTime.Add(24*time.Hour))
	if err != nil {
		log.Fatalf("Query failed: %v", err)
	}
	defer rows.Close()

	type Row struct {
		ID         int64
		Symbol     string
		TS         time.Time
		Direction  string
		Confidence float64
		Horizon    int
	}

	predsBySymbol := make(map[string][]Row)

	for rows.Next() {
		var r Row
		if err := rows.Scan(&r.ID, &r.Symbol, &r.TS, &r.Direction, &r.Confidence, &r.Horizon); err != nil {
			log.Printf("scan err: %v", err)
			continue
		}
		predsBySymbol[r.Symbol] = append(predsBySymbol[r.Symbol], r)
	}

	// prepare calibration table
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS model_calibration (
			symbol VARCHAR(20) PRIMARY KEY,
			bins JSONB,
			emit_threshold DOUBLE PRECISION,
			updated_at TIMESTAMPTZ DEFAULT NOW()
		)
	`)
	if err != nil {
		log.Printf("failed to create calibration table: %v", err)
	}

	// build symbol filter if provided
	symbolFilter := map[string]bool{}
	if *syms != "" {
		for _, s := range strings.Split(*syms, ",") {
			symbolFilter[strings.TrimSpace(s)] = true
		}
	}

	for sym, preds := range predsBySymbol {
		if len(symbolFilter) > 0 {
			if !symbolFilter[sym] {
				continue
			}
		}
		log.Printf("Processing %s: %d predictions", sym, len(preds))
		n := *bins
		counts := make([]int, n)
		corrects := make([]int, n)

		for _, p := range preds {
			// determine actual price at timestamp + horizon minutes
			targetTs := p.TS.Add(time.Duration(p.Horizon) * time.Minute)
			// look up nearest candle at or after targetTs
			var closePrice float64
			err := db.QueryRow(`
				SELECT close FROM candle_cache WHERE symbol=$1 AND timestamp >= $2 ORDER BY timestamp ASC LIMIT 1
			`, sym, targetTs.Unix()).Scan(&closePrice)
			if err != nil {
				continue
			}

			// find price at prediction time (closest candle at or before p.TS)
			var startPrice float64
			err = db.QueryRow(`
				SELECT close FROM candle_cache WHERE symbol=$1 AND timestamp <= $2 ORDER BY timestamp DESC LIMIT 1
			`, sym, p.TS.Unix()).Scan(&startPrice)
			if err != nil {
				continue
			}

			// compute movement
			mv := (closePrice - startPrice) / startPrice
			actual := "SIDEWAYS"
			if mv > *move {
				actual = "UP"
			} else if mv < -*move {
				actual = "DOWN"
			}

			// map prediction to UP/DOWN/SIDEWAYS
			predClass := "SIDEWAYS"
			switch p.Direction {
			case "STRONG_BUY", "BUY", "UP", "STRONG_UP":
				predClass = "UP"
			case "STRONG_SELL", "SELL", "DOWN", "STRONG_DOWN":
				predClass = "DOWN"
			default:
				predClass = "SIDEWAYS"
			}

			// map confidence to bin
			idx := int(p.Confidence * float64(n))
			if idx < 0 {
				idx = 0
			}
			if idx >= n {
				idx = n - 1
			}
			counts[idx]++
			if predClass == actual {
				corrects[idx]++
			}
		}

		binsArr := make([]float64, n)
		for i := 0; i < n; i++ {
			if counts[i] > 0 {
				binsArr[i] = float64(corrects[i]) / float64(counts[i])
			} else {
				binsArr[i] = 0
			}
		}

		// find minimal threshold where bin accuracy >= target
		thr := *threshold
		for i := 0; i < n; i++ {
			if binsArr[i] >= *threshold {
				thr = float64(i) / float64(n)
				break
			}
		}

		binsJson, _ := json.Marshal(binsArr)
		_, err = db.Exec(`
			INSERT INTO model_calibration (symbol, bins, emit_threshold, updated_at)
			VALUES ($1, $2, $3, NOW())
			ON CONFLICT (symbol) DO UPDATE SET bins = $2, emit_threshold = $3, updated_at = NOW()
		`, sym, binsJson, thr)
		if err != nil {
			log.Printf("failed to write calibration for %s: %v", sym, err)
			continue
		}

		log.Printf("Calibration for %s: emit_threshold=%.2f, bins=%v", sym, thr, binsArr)
	}

	fmt.Println("Calibration complete")
}
