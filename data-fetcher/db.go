package main

import (
	"context"
	"database/sql"
	"errors"
	"log"
	"os"
	"sync"
	"time"

	_ "github.com/ClickHouse/clickhouse-go/v2"
)

const (
	defaultCHDSN   = "clickhouse://app:app_password@clickhouse:9000/default?dial_timeout=5s&max_execution_time=60"
	dbWriteTimeout = 10 * time.Second
	dbInitMinDelay = 1 * time.Second
	dbInitMaxDelay = 30 * time.Second
)

var (
	// Guards db: it is written by the (re)connect loop and read by every
	// publisher goroutine. Previously this was an unsynchronised global.
	dbMu sync.RWMutex
	db   *sql.DB
)

const createCandleTableDDL = `CREATE TABLE IF NOT EXISTS candle_1m (
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
ORDER BY (symbol, timestamp)`

func chDSN() string {
	if dsn := os.Getenv("CH_DSN"); dsn != "" {
		return dsn
	}
	return defaultCHDSN
}

func setDB(handle *sql.DB) {
	dbMu.Lock()
	db = handle
	dbMu.Unlock()
}

// GetDB returns the current handle, or nil while ClickHouse is unreachable.
func GetDB() *sql.DB {
	dbMu.RLock()
	defer dbMu.RUnlock()
	return db
}

func connectDB() (*sql.DB, error) {
	handle, err := sql.Open("clickhouse", chDSN())
	if err != nil {
		return nil, err
	}

	// Write-heavy, low concurrency.
	handle.SetMaxOpenConns(10)
	handle.SetMaxIdleConns(3)
	handle.SetConnMaxLifetime(5 * time.Minute)

	ctx, cancel := context.WithTimeout(context.Background(), dbWriteTimeout)
	defer cancel()

	if err := handle.PingContext(ctx); err != nil {
		handle.Close()
		return nil, err
	}
	if _, err := handle.ExecContext(ctx, createCandleTableDDL); err != nil {
		handle.Close()
		return nil, err
	}
	return handle, nil
}

// InitDB tries once synchronously and then keeps retrying in the background
// with exponential backoff, so a slow ClickHouse no longer disables writes for
// the lifetime of the process.
func InitDB() {
	if handle, err := connectDB(); err == nil {
		setDB(handle)
		log.Printf("[DB] connected to ClickHouse")
		return
	} else {
		log.Printf("[DB] initial connect failed, retrying in background: %v", err)
	}

	go func() {
		delay := dbInitMinDelay
		for {
			time.Sleep(delay)
			if delay < dbInitMaxDelay {
				delay *= 2
				if delay > dbInitMaxDelay {
					delay = dbInitMaxDelay
				}
			}
			handle, err := connectDB()
			if err != nil {
				log.Printf("[DB] reconnect failed: %v", err)
				continue
			}
			setDB(handle)
			log.Printf("[DB] connected to ClickHouse after retry")
			return
		}
	}()
}

var errDBNotReady = errors.New("db not initialized")

const insertCandleSQL = `INSERT INTO candle_1m (symbol, timestamp, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?)`

// SaveCandle writes a single candle. Prefer SaveCandles for bulk writes:
// ClickHouse is far happier with one batch than with N single-row inserts.
func SaveCandle(c Candle) error {
	return SaveCandles([]Candle{c})
}

// SaveCandles inserts a slice of candles in one batch.
func SaveCandles(candles []Candle) error {
	if len(candles) == 0 {
		return nil
	}
	handle := GetDB()
	if handle == nil {
		return errDBNotReady
	}

	ctx, cancel := context.WithTimeout(context.Background(), dbWriteTimeout)
	defer cancel()

	tx, err := handle.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	stmt, err := tx.PrepareContext(ctx, insertCandleSQL)
	if err != nil {
		tx.Rollback()
		return err
	}
	defer stmt.Close()

	for _, c := range candles {
		if _, err := stmt.ExecContext(ctx, c.Symbol, c.Timestamp, c.Open, c.High, c.Low, c.Close, c.Volume); err != nil {
			tx.Rollback()
			return err
		}
	}
	return tx.Commit()
}
