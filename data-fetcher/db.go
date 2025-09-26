package main

import (
	"database/sql"
	"errors"
	"log"
	"os"
	"time"

	_ "github.com/lib/pq"
)

var db *sql.DB

func InitDB() {
	dsn := os.Getenv("PG_DSN")
	if dsn == "" {
		dsn = "host=postgres user=admin password=password dbname=predpump sslmode=disable"
	}
	var err error
	db, err = sql.Open("postgres", dsn)
	if err != nil {
		log.Printf("[DB] failed to open: %v", err)
		db = nil
		return
	}
	if err := db.Ping(); err != nil {
		log.Printf("[DB] ping failed: %v", err)
		db = nil
		return
	}
	_, err = db.Exec(`CREATE TABLE IF NOT EXISTS candle_1m (
                id BIGSERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                open DECIMAL(18,8) NOT NULL,
                high DECIMAL(18,8) NOT NULL,
                low DECIMAL(18,8) NOT NULL,
                close DECIMAL(18,8) NOT NULL,
                volume DECIMAL(18,8) NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW())`)
	if err != nil {
		log.Printf("[DB] candle table init error: %v", err)
		db = nil
		return
	}
}

func GetDB() *sql.DB {
	return db
}

func SaveCandle(c Candle) error {
	if db == nil {
		return errors.New("db not initialized")
	}
	_, err := db.Exec(
		`INSERT INTO candle_1m (symbol, timestamp, open, high, low, close, volume) VALUES ($1,$2,$3,$4,$5,$6,$7)`,
		c.Symbol,
		time.Unix(c.Timestamp, 0).UTC(),
		c.Open,
		c.High,
		c.Low,
		c.Close,
		c.Volume,
	)
	return err
}
