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
	syms := flag.String("symbols", "", "comma-separated symbols")
	move := flag.Float64("move", 0.005, "price move threshold fraction (default 0.005 = 0.5%)")
	usePlatt := flag.Bool("platt", true, "apply per-symbol Platt scaling parameters if available")
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
		// list all symbols from model_calibration
		rows, err := db.Query(`SELECT symbol FROM model_calibration ORDER BY symbol`)
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

	fmt.Printf("Computing emitted accuracy for %v from %s to %s (move=%.4f)\n", symbols, sTime.Format("2006-01-02"), eTime.Format("2006-01-02"), *move)

	for _, sym := range symbols {
		var thr float64
		err := db.QueryRow(`SELECT emit_threshold FROM model_calibration WHERE symbol=$1`, sym).Scan(&thr)
		if err != nil {
			fmt.Printf("%s - no calibration entry (err: %v)\n", sym, err)
			continue
		}

		// try to load platt params
		var pa, pb float64
		havePlatt := false
		if *usePlatt {
			err = db.QueryRow(`SELECT param_a, param_b FROM model_platt WHERE symbol=$1`, sym).Scan(&pa, &pb)
			if err == nil {
				havePlatt = true
			}
		}

		rows, err := db.Query(`
            SELECT timestamp, direction, confidence, time_horizon
            FROM direction_predictions
            WHERE symbol=$1 AND timestamp >= $2 AND timestamp <= $3
            ORDER BY timestamp ASC
        `, sym, sTime, eTime.Add(24*time.Hour))
		if err != nil {
			fmt.Printf("%s - query preds err: %v\n", sym, err)
			continue
		}

		type R struct {
			TS      time.Time
			Dir     string
			Conf    float64
			Horizon int
		}
		preds := []R{}
		for rows.Next() {
			var r R
			if err := rows.Scan(&r.TS, &r.Dir, &r.Conf, &r.Horizon); err == nil {
				preds = append(preds, r)
			}
		}
		rows.Close()

		emitted := 0
		correct := 0
		total := len(preds)

		for _, p := range preds {
			conf := p.Conf
			if havePlatt {
				x := pa*conf + pb
				conf = 1.0 / (1.0 + math.Exp(-x))
			}
			if conf < thr {
				continue
			}
			emitted++
			// get close at target
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
			predClass := "SIDEWAYS"
			switch p.Dir {
			case "STRONG_BUY", "BUY", "UP", "STRONG_UP":
				predClass = "UP"
			case "STRONG_SELL", "SELL", "DOWN", "STRONG_DOWN":
				predClass = "DOWN"
			default:
				predClass = "SIDEWAYS"
			}
			if predClass == actual {
				correct++
			}
		}

		acc := 0.0
		if emitted > 0 {
			acc = float64(correct) / float64(emitted)
		}
		fmt.Printf("%s - thr=%.3f emitted=%d total_preds=%d correct=%d accuracy=%.3f\n", sym, thr, emitted, total, correct, acc)
	}
}
