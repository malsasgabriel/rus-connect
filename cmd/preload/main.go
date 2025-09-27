package main

import (
	"database/sql"
	"log"
	"os"
	"time"

	_ "github.com/lib/pq"
)

func main() {
	// Minimal preload tool: call analytics-engine's HistoricalDataLoader via HTTP? Instead, reuse DB DSN and call analytics engine code directly is complex.
	// We'll spawn analytics-engine's historical loader by invoking the existing program in-process via `go run` would be heavy.
	log.Println("Preload helper: please run the analytics-engine or use the existing HistoricalDataLoader within analytics-engine to fetch data.")
	log.Println("For now, use: go run ./analytics-engine (it will preload historical candles and save to DB).")
	// Fall back: ensure DB is reachable
	dsn := os.Getenv("PG_DSN")
	if dsn == "" {
		dsn = "host=postgres user=admin password=password dbname=predpump sslmode=disable"
	}
	db, err := sql.Open("postgres", dsn)
	if err != nil {
		log.Fatalf("Failed to open DB: %v", err)
	}
	defer db.Close()
	if err := db.Ping(); err != nil {
		log.Fatalf("Failed to ping DB: %v", err)
	}
	log.Println("DB reachable. Exiting. Run analytics-engine to perform preload.")
	time.Sleep(1 * time.Second)
}
