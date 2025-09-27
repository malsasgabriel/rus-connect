package main

import (
	"database/sql"
	"flag"
	"fmt"
	"os"

	_ "github.com/lib/pq"
)

func main() {
	thr := flag.Float64("threshold", 0.40, "new emit threshold to set")
	sym := flag.String("symbol", "", "symbol to update (empty = all)")
	flag.Parse()

	dsn := os.Getenv("PG_DSN")
	if dsn == "" {
		dsn = "host=localhost user=admin password=password dbname=predpump sslmode=disable"
	}

	db, err := sql.Open("postgres", dsn)
	if err != nil {
		panic(err)
	}
	defer db.Close()

	if *sym == "" {
		res, err := db.Exec("UPDATE model_calibration SET emit_threshold=$1, updated_at=NOW()", *thr)
		if err != nil {
			panic(err)
		}
		n, _ := res.RowsAffected()
		fmt.Printf("Updated %d rows to emit_threshold=%.3f\n", n, *thr)
		return
	}

	res, err := db.Exec("UPDATE model_calibration SET emit_threshold=$1, updated_at=NOW() WHERE symbol=$2", *thr, *sym)
	if err != nil {
		panic(err)
	}
	n, _ := res.RowsAffected()
	fmt.Printf("Updated %d rows for %s to emit_threshold=%.3f\n", n, *sym, *thr)
}
