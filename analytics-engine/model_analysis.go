package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"log"

	"github.com/segmentio/kafka-go"
)

// EnsureModelAnalysesTable creates table for per-model analyses
func EnsureModelAnalysesTable(db *sql.DB) {
	_, err := db.Exec(`
        CREATE TABLE IF NOT EXISTS model_analyses (
            id BIGSERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            model_name VARCHAR(50) NOT NULL,
            prediction VARCHAR(50),
            confidence DECIMAL(5,4),
            payload JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    `)
	if err != nil {
		log.Printf("Failed to create model_analyses table: %v", err)
	}
}

// PublishModelAnalysisDBAndKafka writes model analysis to DB and Kafka topic `model_analyses`.
func PublishModelAnalysisDBAndKafka(db *sql.DB, brokers []string, payload map[string]interface{}) error {
	// Marshal payload
	b, err := json.Marshal(payload)
	if err != nil {
		return err
	}

	// Insert into DB (if db provided)
	symbol := ""
	if v, ok := payload["symbol"].(string); ok {
		symbol = v
	}
	modelName := ""
	if v, ok := payload["model_name"].(string); ok {
		modelName = v
	}
	prediction := ""
	if v, ok := payload["prediction"].(string); ok {
		prediction = v
	}
	confidence := 0.0
	if v, ok := payload["confidence"].(float64); ok {
		confidence = v
	}

	if db != nil {
		_, err = db.Exec(`INSERT INTO model_analyses (symbol, model_name, prediction, confidence, payload) VALUES ($1,$2,$3,$4,$5)`, symbol, modelName, prediction, confidence, string(b))
		if err != nil {
			log.Printf("Failed to insert model analysis: %v", err)
			// continue to publish to kafka regardless
		}
	}

	// Publish to Kafka topic model_analyses
	if len(brokers) == 0 {
		// In Docker Compose the broker host is `kafka:9092` inside the network
		brokers = []string{"kafka:9092"}
	}
	writer := &kafka.Writer{
		Addr:     kafka.TCP(brokers...),
		Topic:    "model_analyses",
		Balancer: &kafka.LeastBytes{},
	}
	defer writer.Close()

	err = writer.WriteMessages(context.Background(), kafka.Message{
		Key:   []byte(symbol),
		Value: b,
	})
	if err != nil {
		log.Printf("Failed to publish model analysis to Kafka: %v", err)
		return err
	}

	return nil
}
