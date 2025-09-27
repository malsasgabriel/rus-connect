package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"time"

	_ "github.com/lib/pq"
)

func main() {
	dsn := os.Getenv("PG_DSN")
	if dsn == "" {
		dsn = "host=localhost user=admin password=password dbname=predpump sslmode=disable"
	}
	db, err := sql.Open("postgres", dsn)
	if err != nil {
		panic(err)
	}
	defer db.Close()

	rows, err := db.Query("SELECT symbol, bins, emit_threshold, updated_at FROM model_calibration ORDER BY symbol")
	if err != nil {
		panic(err)
	}
	defer rows.Close()

	for rows.Next() {
		var sym string
		var binsRaw []byte
		var thr float64
		var updated time.Time
		if err := rows.Scan(&sym, &binsRaw, &thr, &updated); err != nil {
			panic(err)
		}
		var bins []float64
		json.Unmarshal(binsRaw, &bins)
		fmt.Printf("%s - emit_threshold=%.3f updated=%s bins=%v\n", sym, thr, updated.Format(time.RFC3339), bins)
	}
}
