package main

import (
	"database/sql"
	"fmt"
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
		panic(err)
	}
	defer db.Close()

	rows, err := db.Query("SELECT column_name, data_type FROM information_schema.columns WHERE table_name='candle_cache'")
	if err != nil {
		panic(err)
	}
	defer rows.Close()
	fmt.Println("Schema of candle_cache:")
	for rows.Next() {
		var name, dtype string
		rows.Scan(&name, &dtype)
		fmt.Printf("%s: %s\n", name, dtype)
	}

	fmt.Println("\nSample rows:")
	r, err := db.Query("SELECT symbol, timestamp, close FROM candle_cache ORDER BY timestamp DESC LIMIT 5")
	if err != nil {
		panic(err)
	}
	defer r.Close()
	for r.Next() {
		var sym string
		var ts interface{}
		var close float64
		r.Scan(&sym, &ts, &close)
		fmt.Printf("%s %v %.6f\n", sym, ts, close)
	}
}
