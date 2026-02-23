package main

import (
	"database/sql"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"strings"
	"time"

	_ "github.com/lib/pq"
)

func main() {
	start := flag.String("start", "2025-09-15", "start date YYYY-MM-DD")
	end := flag.String("end", "2025-09-21", "end date YYYY-MM-DD")
	syms := flag.String("symbols", "", "comma-separated symbols to process (default: all)")
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

	// determine symbols to process
	symbols := []string{}
	if *syms == "" {
		// list symbols from direction_predictions in range
		rows, err := db.Query(`SELECT DISTINCT symbol FROM direction_predictions WHERE timestamp >= $1 AND timestamp <= $2 ORDER BY symbol`, sTime, eTime.Add(24*time.Hour))
		if err != nil {
			log.Fatalf("query symbols: %v", err)
		}
		defer rows.Close()
		for rows.Next() {
			var s string
			if err := rows.Scan(&s); err == nil {
				symbols = append(symbols, s)
			}
		}
	} else {
		for _, s := range strings.Split(*syms, ",") {
			symbols = append(symbols, strings.TrimSpace(s))
		}
	}

	if len(symbols) == 0 {
		log.Println("No symbols to process")
		return
	}

	// ensure model_platt table
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS model_platt (
			symbol VARCHAR(20) PRIMARY KEY,
			param_a DOUBLE PRECISION,
			param_b DOUBLE PRECISION,
			updated_at TIMESTAMPTZ DEFAULT NOW()
		)
	`)
	if err != nil {
		log.Printf("failed to create model_platt table: %v", err)
	}

	type Row struct {
		ID         int64
		Symbol     string
		TS         time.Time
		Direction  string
		Confidence float64
		Horizon    int
	}

	for _, sym := range symbols {
		// load predictions for symbol in window
		rows, err := db.Query(`
			SELECT id, symbol, timestamp, direction, confidence, time_horizon
			FROM direction_predictions
			WHERE symbol=$1 AND timestamp >= $2 AND timestamp <= $3
			ORDER BY timestamp ASC
		`, sym, sTime, eTime.Add(24*time.Hour))
		if err != nil {
			log.Printf("query failed for %s: %v", sym, err)
			continue
		}

		preds := make([]Row, 0)
		for rows.Next() {
			var r Row
			if err := rows.Scan(&r.ID, &r.Symbol, &r.TS, &r.Direction, &r.Confidence, &r.Horizon); err != nil {
				continue
			}
			preds = append(preds, r)
		}
		rows.Close()

		if len(preds) == 0 {
			log.Printf("No predictions for %s in window", sym)
			continue
		}

		// build training pairs (score, label)
		scores := make([]float64, 0, len(preds))
		labels := make([]float64, 0, len(preds))

		for _, p := range preds {
			// find target close price
			targetTs := p.TS.Add(time.Duration(p.Horizon) * time.Minute)
			var closePrice float64
			if err := db.QueryRow(`SELECT close FROM candle_cache WHERE symbol=$1 AND timestamp >= $2 ORDER BY timestamp ASC LIMIT 1`, p.Symbol, targetTs.Unix()).Scan(&closePrice); err != nil {
				continue
			}
			var startPrice float64
			if err := db.QueryRow(`SELECT close FROM candle_cache WHERE symbol=$1 AND timestamp <= $2 ORDER BY timestamp DESC LIMIT 1`, p.Symbol, p.TS.Unix()).Scan(&startPrice); err != nil {
				continue
			}
			move := (closePrice - startPrice) / startPrice
			actual := 0.0
			if move > 0.01 {
				actual = 1.0
			}
			scores = append(scores, p.Confidence)
			labels = append(labels, actual)
		}

		if len(scores) == 0 {
			log.Printf("No usable samples for %s", sym)
			continue
		}

		// Platt scaling per-symbol
		a := 0.0
		b := 0.0
		lr := 0.1
		for iter := 0; iter < 2000; iter++ {
			gradA := 0.0
			gradB := 0.0
			loss := 0.0
			for i, s := range scores {
				x := a*s + b
				p := 1.0 / (1.0 + math.Exp(-x))
				y := labels[i]
				loss += -(y*math.Log(p+1e-12) + (1.0-y)*math.Log(1.0-p+1e-12))
				gradA += (p - y) * s
				gradB += (p - y)
			}
			gradA /= float64(len(scores))
			gradB /= float64(len(scores))
			a -= lr * gradA
			b -= lr * gradB
			if iter%200 == 0 {
				lr *= 0.99
			}
			if loss/float64(len(scores)) < 0.1 {
				break
			}
		}

		// save per-symbol params
		_, err = db.Exec(`INSERT INTO model_platt (symbol, param_a, param_b, updated_at) VALUES ($1,$2,$3,NOW()) ON CONFLICT (symbol) DO UPDATE SET param_a=$2, param_b=$3, updated_at=NOW()`, sym, a, b)
		if err != nil {
			log.Printf("failed to save platt params for %s: %v", sym, err)
			continue
		}

		fmt.Printf("Platt for %s: a=%.6f b=%.6f samples=%d\n", sym, a, b, len(scores))
	}

	fmt.Println("Offline Platt calibration complete")
}
