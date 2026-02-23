package main

import (
	"database/sql"
	"errors"
	"log"
	"os"
	"time"

	_ "github.com/ClickHouse/clickhouse-go/v2"
)

var db *sql.DB

func InitDB() {
	dsn := os.Getenv("CH_DSN")
	if dsn == "" {
		dsn = "clickhouse://app:app_password@clickhouse:9000/default?dial_timeout=5s&max_execution_time=60"
	}
	var err error
	db, err = sql.Open("clickhouse", dsn)
	if err != nil {
		log.Printf("[DB] failed to open: %v", err)
		db = nil
		return
	}

	// Configure connection pool for data-fetcher (write-heavy, low concurrency)
	db.SetMaxOpenConns(10)
	db.SetMaxIdleConns(3)
	db.SetConnMaxLifetime(5 * time.Minute)

	if err := db.Ping(); err != nil {
		log.Printf("[DB] ping failed: %v", err)
		db = nil
		return
	}
	_, err = db.Exec(`CREATE TABLE IF NOT EXISTS candle_1m (
                symbol String,
                timestamp Int64,
                open Float64,
                high Float64,
                low Float64,
                close Float64,
                volume Float64,
                created_at DateTime DEFAULT now()
            ) ENGINE = ReplacingMergeTree(created_at)
            PARTITION BY toYYYYMM(toDateTime(timestamp))
            ORDER BY (symbol, timestamp)`)
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
		`INSERT INTO candle_1m (symbol, timestamp, open, high, low, close, volume) 
		 VALUES (?,?,?,?,?,?,?)`,
		c.Symbol,
		c.Timestamp,
		c.Open,
		c.High,
		c.Low,
		c.Close,
		c.Volume,
	)
	return err
}
