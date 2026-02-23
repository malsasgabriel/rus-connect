package main

import (
	"database/sql"
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
	end := flag.String("end", "2025-09-29", "end date YYYY-MM-DD")
	syms := flag.String("symbols", "", "comma-separated symbols")
	move := flag.Float64("move", 0.005, "price move threshold fraction (default 0.005 = 0.5%)")
	batchSize := flag.Int("batch", 100, "batch size for updates")
	flag.Parse()

	dsn := os.Getenv("PG_DSN")
	if dsn == "" {
		dsn = "host=localhost user=admin password=password dbname=predpump sslmode=disable"
	}

	db, err := sql.Open("postgres", dsn)
	if err != nil {
		log.Fatalf("DB open: %v", err)
	}
	defer db.Close()

	sTime, _ := time.Parse("2006-01-02", *start)
	eTime, _ := time.Parse("2006-01-02", *end)

	symbols := []string{}
	if *syms == "" {
		// list all symbols from direction_predictions
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

	fmt.Printf("Updating actual directions for %v from %s to %s (move=%.4f)\n", symbols, sTime.Format("2006-01-02"), eTime.Format("2006-01-02"), *move)

	for _, sym := range symbols {
		fmt.Printf("Processing %s...\n", sym)

		// Get predictions that don't have actual_direction set yet
		rows, err := db.Query(`
            SELECT id, timestamp, direction, time_horizon
            FROM direction_predictions
            WHERE symbol=$1 AND timestamp >= $2 AND timestamp <= $3 AND actual_direction IS NULL
            ORDER BY timestamp ASC
        `, sym, sTime, eTime.Add(24*time.Hour))
		if err != nil {
			fmt.Printf("%s - query preds err: %v\n", sym, err)
			continue
		}

		type Prediction struct {
			ID      int64
			TS      time.Time
			Dir     string
			Horizon int
		}

		predictions := []Prediction{}
		for rows.Next() {
			var p Prediction
			if err := rows.Scan(&p.ID, &p.TS, &p.Dir, &p.Horizon); err == nil {
				predictions = append(predictions, p)
			}
		}
		rows.Close()

		fmt.Printf("%s - found %d predictions to update\n", sym, len(predictions))

		// Process in batches
		updated := 0
		for i := 0; i < len(predictions); i += *batchSize {
			end := i + *batchSize
			if end > len(predictions) {
				end = len(predictions)
			}

			batch := predictions[i:end]

			// Start transaction
			tx, err := db.Begin()
			if err != nil {
				fmt.Printf("%s - failed to begin transaction: %v\n", sym, err)
				continue
			}

			// Prepare statement
			stmt, err := tx.Prepare(`UPDATE direction_predictions SET actual_direction = $1 WHERE id = $2`)
			if err != nil {
				tx.Rollback()
				fmt.Printf("%s - failed to prepare statement: %v\n", sym, err)
				continue
			}

			for _, p := range batch {
				// Get close at target
				targetTs := p.TS.Add(time.Duration(p.Horizon) * time.Minute)
				var closePrice float64
				if err := db.QueryRow(`SELECT close FROM candle_cache WHERE symbol=$1 AND timestamp >= $2 ORDER BY timestamp ASC LIMIT 1`, sym, targetTs.Unix()).Scan(&closePrice); err != nil {
					continue
				}
				var startPrice float64
				if err := db.QueryRow(`SELECT close FROM candle_cache WHERE symbol=$1 AND timestamp <= $2 ORDER BY timestamp DESC LIMIT 1`, sym, p.TS.Unix()).Scan(&startPrice); err != nil {
					continue
				}

				mv := (closePrice - startPrice) / startPrice
				actual := "SIDEWAYS"
				if mv > *move {
					actual = "UP"
				} else if mv < -*move {
					actual = "DOWN"
				}

				// Update the prediction
				if _, err := stmt.Exec(actual, p.ID); err != nil {
					fmt.Printf("%s - failed to update prediction %d: %v\n", sym, p.ID, err)
					continue
				}

				updated++
			}

			// Close statement
			stmt.Close()

			// Commit transaction
			if err := tx.Commit(); err != nil {
				fmt.Printf("%s - failed to commit transaction: %v\n", sym, err)
				continue
			}

			fmt.Printf("%s - updated %d predictions in batch\n", sym, len(batch))
		}

		fmt.Printf("%s - completed: %d predictions updated\n", sym, updated)
	}

	fmt.Println("All symbols processed!")
}
