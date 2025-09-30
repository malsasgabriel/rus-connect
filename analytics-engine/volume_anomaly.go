package main

import (
	"context"
	"database/sql"
	"fmt"
	"log"

	"github.com/segmentio/kafka-go"
)

// VolumeAnomalyDetector detects spikes in volume per symbol
type VolumeAnomalyDetector struct {
	db            *sql.DB
	windowSeconds int64
	multiplier    float64
	history       map[string][]float64
	kafkaWriter   *kafka.Writer
}

func NewVolumeAnomalyDetector(db *sql.DB, kafkaWriter *kafka.Writer) *VolumeAnomalyDetector {
	// create table
	_, err := db.Exec(`CREATE TABLE IF NOT EXISTS volume_anomalies (
        id BIGSERIAL PRIMARY KEY,
        symbol VARCHAR(20) NOT NULL,
        timestamp TIMESTAMPTZ NOT NULL,
        volume DECIMAL(20,8) NOT NULL,
        avg_volume DECIMAL(20,8) NOT NULL,
        multiplier_used DECIMAL(5,2) NOT NULL,
        created_at TIMESTAMPTZ DEFAULT NOW()
    )`)
	if err != nil {
		log.Printf("Failed to create volume_anomalies table: %v", err)
	}

	return &VolumeAnomalyDetector{
		db:            db,
		windowSeconds: 60 * 60 * 24, // 24 hours window (candles assumed 1m)
		multiplier:    3.0,
		history:       make(map[string][]float64),
		kafkaWriter:   kafkaWriter,
	}
}

// ProcessCandle checks a candle and writes an anomaly if detected
func (v *VolumeAnomalyDetector) ProcessCandle(c Candle) {
	h := v.history[c.Symbol]
	h = append(h, c.Volume)
	if len(h) > 1440 { // keep last 24h
		h = h[len(h)-1440:]
	}
	v.history[c.Symbol] = h

	// compute average
	var sum float64
	for _, vv := range h {
		sum += vv
	}
	avg := 0.0
	if len(h) > 0 {
		avg = sum / float64(len(h))
	}

	if avg > 0 && c.Volume >= avg*v.multiplier {
		// anomaly detected
		log.Printf("🔔 Volume anomaly detected for %s: vol=%.8f avg=%.8f", c.Symbol, c.Volume, avg)

		// insert into DB
		_, err := v.db.Exec(`INSERT INTO volume_anomalies (symbol, timestamp, volume, avg_volume, multiplier_used) VALUES ($1, to_timestamp($2), $3, $4, $5)`,
			c.Symbol, c.Timestamp, c.Volume, avg, v.multiplier)
		if err != nil {
			log.Printf("Failed to insert volume anomaly: %v", err)
		}

		// publish to Kafka topic 'volume_anomalies' if writer present
		if v.kafkaWriter != nil {
			type msg struct {
				Symbol    string  `json:"symbol"`
				Timestamp int64   `json:"timestamp"`
				Volume    float64 `json:"volume"`
				Avg       float64 `json:"avg"`
			}
			m := msg{Symbol: c.Symbol, Timestamp: c.Timestamp, Volume: c.Volume, Avg: avg}
			// best-effort publish
			v.kafkaWriter.WriteMessages(context.Background(), kafka.Message{Key: []byte(c.Symbol), Value: []byte(fmt.Sprintf("%s %d %.8f %.8f", m.Symbol, m.Timestamp, m.Volume, m.Avg))})
		}
	}
}
