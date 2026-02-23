package main

import (
	"database/sql"
	"fmt"
	"log"
	"os"

	_ "github.com/lib/pq"
)

func main() {
	dsn := os.Getenv("PG_DSN")
	if dsn == "" {
		dsn = "host=localhost user=admin password=password dbname=predpump sslmode=disable"
	}

	db, err := sql.Open("postgres", dsn)
	if err != nil {
		log.Fatalf("open db: %v", err)
	}
	defer db.Close()

	var cnt int
	if err := db.QueryRow("select count(*) from direction_predictions").Scan(&cnt); err != nil {
		fmt.Printf("direction_predictions: error: %v\n", err)
	} else {
		fmt.Printf("direction_predictions: %d rows\n", cnt)
	}

	if err := db.QueryRow("select count(*) from model_calibration").Scan(&cnt); err != nil {
		fmt.Printf("model_calibration: error: %v\n", err)
	} else {
		fmt.Printf("model_calibration: %d rows\n", cnt)
	}

	if err := db.QueryRow("select count(*) from model_platt").Scan(&cnt); err != nil {
		fmt.Printf("model_platt: error: %v\n", err)
	} else {
		fmt.Printf("model_platt: %d rows\n", cnt)
	}

	// sample a few predictions
	rows, err := db.Query("select symbol, to_char(timestamp,'YYYY-MM-DD HH24:MI:SS'), direction, confidence from direction_predictions order by created_at desc limit 10")
	if err != nil {
		fmt.Printf("sample predictions: error: %v\n", err)
		return
	}
	defer rows.Close()

	fmt.Println("\nSample latest direction_predictions:")
	for rows.Next() {
		var sym, ts, dir string
		var conf float64
		if err := rows.Scan(&sym, &ts, &dir, &conf); err != nil {
			break
		}
		fmt.Printf("%s %s %s %.3f\n", ts, sym, dir, conf)
	}
}
