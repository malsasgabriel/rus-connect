package main

import (
	"database/sql"
	"fmt"
	"log"
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
		log.Fatalf("db open: %v", err)
	}
	defer db.Close()

	var cnt int
	if err := db.QueryRow(`SELECT COUNT(*) FROM direction_predictions`).Scan(&cnt); err != nil {
		log.Fatalf("count err: %v", err)
	}
	fmt.Printf("direction_predictions rows: %d\n", cnt)

	var minTS, maxTS sql.NullTime
	if err := db.QueryRow(`SELECT MIN(timestamp), MAX(timestamp) FROM direction_predictions`).Scan(&minTS, &maxTS); err != nil {
		log.Fatalf("minmax err: %v", err)
	}
	if minTS.Valid {
		fmt.Printf("min timestamp: %s\n", minTS.Time.Format(time.RFC3339))
	}
	if maxTS.Valid {
		fmt.Printf("max timestamp: %s\n", maxTS.Time.Format(time.RFC3339))
	}

	// list distinct symbols
	rows, err := db.Query(`SELECT DISTINCT symbol FROM direction_predictions ORDER BY symbol`)
	if err != nil {
		log.Fatalf("symbols query: %v", err)
	}
	defer rows.Close()
	fmt.Println("symbols:")
	for rows.Next() {
		var s string
		if err := rows.Scan(&s); err == nil {
			fmt.Println(" -", s)
		}
	}
}
