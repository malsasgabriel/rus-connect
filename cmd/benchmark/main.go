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
		log.Fatalf("Failed to connect to database: %v", err)
	}
	defer db.Close()

	// Configure connection pool for benchmarking
	db.SetMaxOpenConns(25)
	db.SetMaxIdleConns(5)
	db.SetConnMaxLifetime(5 * time.Minute)

	if err := db.Ping(); err != nil {
		log.Fatalf("Failed to ping database: %v", err)
	}

	fmt.Println("Running database performance benchmarks...")

	// Benchmark 1: Simple query performance
	start := time.Now()
	rows, err := db.Query("SELECT COUNT(*) FROM candle_1m")
	if err != nil {
		log.Printf("Query failed: %v", err)
		return
	}
	var count int
	rows.Next()
	rows.Scan(&count)
	rows.Close()
	queryTime := time.Since(start)
	fmt.Printf("Simple COUNT query: %v (result: %d rows)\n", queryTime, count)

	// Benchmark 2: Complex query with JOIN
	start = time.Now()
	rows, err = db.Query(`
		SELECT s.symbol, COUNT(*) as signal_count 
		FROM signals s 
		WHERE s.created_at > NOW() - INTERVAL '1 day'
		GROUP BY s.symbol 
		ORDER BY signal_count DESC 
		LIMIT 10
	`)
	if err != nil {
		log.Printf("Complex query failed: %v", err)
		return
	}
	defer rows.Close()
	complexQueryTime := time.Since(start)
	fmt.Printf("Complex query with GROUP BY: %v\n", complexQueryTime)

	// Benchmark 3: Connection pool stress test
	fmt.Println("Running connection pool stress test...")
	start = time.Now()
	done := make(chan bool, 50)
	for i := 0; i < 50; i++ {
		go func() {
			rows, err := db.Query("SELECT 1")
			if err != nil {
				log.Printf("Stress test query failed: %v", err)
			} else {
				rows.Close()
			}
			done <- true
		}()
	}

	// Wait for all goroutines to complete
	for i := 0; i < 50; i++ {
		<-done
	}
	stressTestTime := time.Since(start)
	fmt.Printf("50 concurrent queries: %v\n", stressTestTime)

	fmt.Println("Benchmark completed!")
}
