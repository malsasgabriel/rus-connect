package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/segmentio/kafka-go"
)

func main() {
	brokers := os.Getenv("KAFKA_BROKERS")
	if brokers == "" {
		brokers = "localhost:9092"
	}
	topic := flag.String("topic", "direction_signals", "Kafka topic to read")
	count := flag.Int("n", 10, "Number of messages to read")
	flag.Parse()

	r := kafka.NewReader(kafka.ReaderConfig{
		Brokers:   []string{brokers},
		Topic:     *topic,
		GroupID:   "kafka-dump-tool",
		MinBytes:  1,
		MaxBytes:  10e6,
		Partition: 0,
	})
	defer r.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	fmt.Printf("Reading up to %d messages from topic %s (brokers=%s)\n", *count, *topic, brokers)

	for i := 0; i < *count; i++ {
		m, err := r.FetchMessage(ctx)
		if err != nil {
			log.Printf("fetch error: %v", err)
			break
		}
		var pretty map[string]interface{}
		if err := json.Unmarshal(m.Value, &pretty); err != nil {
			// Dump raw if unmarshal fails
			fmt.Printf("#%d offset=%d key=%s raw=%s\n", i+1, m.Offset, string(m.Key), string(m.Value))
		} else {
			b, _ := json.MarshalIndent(pretty, "", "  ")
			fmt.Printf("#%d offset=%d key=%s\n%s\n", i+1, m.Offset, string(m.Key), string(b))
		}
		if err := r.CommitMessages(ctx, m); err != nil {
			log.Printf("commit error: %v", err)
		}
	}

	fmt.Println("Done.")
}
