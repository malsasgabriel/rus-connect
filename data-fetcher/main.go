package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/segmentio/kafka-go"
)

// MarketData represents market ticker data.
type MarketData struct {
	Symbol       string
	Volume24h    float64
	AvgVolume7d  float64 // Not directly from Bybit, but used in Analytics Engine
	BidVolume    float64
	AskVolume    float64
	LargestOrder float64
	Price        float64
}

// DataPoint is a generic structure for Kafka messages
type DataPoint struct {
	Symbol string  `json:"symbol"`
	Price  float64 `json:"price"`
	Volume float64 `json:"volume"`
	Time   int64   `json:"timestamp"`
}

// CCXTClient defines the interface for fetching market data.
type CCXTClient interface {
	FetchTicker(symbol string) (MarketData, error)
	FetchOrderBook(symbol string) (OrderBookSnapshot, error)
	FetchTrades(symbol string) (TradesSnapshot, error)
	FetchKline(symbol string, interval string) ([]Candle, error)
}

// KafkaProducerWrapper wraps kafka.Writer for easier use.
type KafkaProducerWrapper struct {
	writer *kafka.Writer
}

func NewKafkaProducerWrapper(broker, topic string) *KafkaProducerWrapper {
	writer := &kafka.Writer{
		Addr:     kafka.TCP(broker),
		Topic:    topic,
		Balancer: &kafka.LeastBytes{},
	}
	return &KafkaProducerWrapper{writer: writer}
}

func (p *KafkaProducerWrapper) Publish(ctx context.Context, key []byte, value []byte) error {
	return p.writer.WriteMessages(ctx,
		kafka.Message{
			Key:   key,
			Value: value,
		},
	)
}

func (p *KafkaProducerWrapper) Close() error {
	return p.writer.Close()
}

// Global rate limiter
var globalRateLimiter = NewRateLimiter(200, 100) // capacity 200 tokens, refill 100 per second

// Global circuit breaker
var globalBreaker = NewCircuitBreaker(5, 10*time.Second)

func main() {
	startHealthServer()
	log.Println("\U0001F680 Data Fetcher: PredPump Radar started")

	// Initialize DB with retry
	for i := 0; i < 20; i++ {
		InitDB()
		if GetDB() != nil {
			break
		}
		log.Printf("[DB] init failed, retrying in 3s (%d/20)", i+1)
		time.Sleep(3 * time.Second)
	}
	if GetDB() != nil {
		log.Println("\u2705 [DB] initialized successfully")
		// ClickHouse relies on ORDER BY in table engine for efficient reads.
	} else {
		log.Println("\u26A0\uFE0F [DB] not initialized; candles won't be persisted to ClickHouse")
	}

	kafkaBrokers := os.Getenv("KAFKA_BROKERS")
	if kafkaBrokers == "" {
		kafkaBrokers = "kafka:9092"
	}

	tickerProducer := NewKafkaProducerWrapper(kafkaBrokers, "ticker")
	orderbookProducer := NewKafkaProducerWrapper(kafkaBrokers, "orderbook")
	tradesProducer := NewKafkaProducerWrapper(kafkaBrokers, "trades")
	candleProducer := NewKafkaProducerWrapper(kafkaBrokers, "candle_1m")

	defer func() {
		log.Println("\U0001F512 Closing Kafka producers...")
		tickerProducer.Close()
		orderbookProducer.Close()
		tradesProducer.Close()
		candleProducer.Close()
	}()

	// Top cryptocurrencies by market cap for real analysis
	symbols := []string{"BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"}

	cc := NewBybitREST()

	var wg sync.WaitGroup
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Graceful shutdown: listen for termination signals
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	// Start fetch loops
	for _, s := range symbols {
		wg.Add(1)
		go fetchLoopForSymbol(ctx, s, cc, tickerProducer, orderbookProducer, tradesProducer, candleProducer, &wg)
	}

	// Wait for shutdown signal
	go func() {
		sig := <-sigChan
		log.Printf("\U0001F6D1 Received signal %v, initiating graceful shutdown...", sig)
		cancel() // Cancel context to stop all goroutines
	}()

	// Wait for all goroutines to finish
	wg.Wait()

	// Close database connection
	if db := GetDB(); db != nil {
		log.Println("\U0001F512 Closing database connection...")
		if err := db.Close(); err != nil {
			log.Printf("\u274C Error closing database: %v", err)
		} else {
			log.Println("\u2705 Database connection closed")
		}
	}

	log.Println("\u2705 Data Fetcher stopped gracefully")
}

func startHealthServer() {
	port := os.Getenv("DATA_FETCHER_PORT")
	if port == "" {
		port = "8083"
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]interface{}{"status": "ok", "time": time.Now().UTC()})
	})
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]interface{}{"status": "ok", "time": time.Now().UTC()})
	})
	mux.HandleFunc("/readyz", func(w http.ResponseWriter, _ *http.Request) {
		db := GetDB()
		if db == nil || db.Ping() != nil {
			w.WriteHeader(http.StatusServiceUnavailable)
			_ = json.NewEncoder(w).Encode(map[string]interface{}{"status": "not_ready", "db_ready": false})
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]interface{}{"status": "ready", "db_ready": true})
	})

	srv := &http.Server{
		Addr:              ":" + port,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       15 * time.Second,
		WriteTimeout:      15 * time.Second,
		IdleTimeout:       60 * time.Second,
	}

	go func() {
		log.Printf("Health server listening on %s", srv.Addr)
		if err := srv.ListenAndServe(); err != nil {
			log.Printf("Health server stopped: %v", err)
		}
	}()
}

func fetchLoopForSymbol(
	ctx context.Context,
	symbol string,
	cc CCXTClient,
	tickerProducer *KafkaProducerWrapper,
	orderbookProducer *KafkaProducerWrapper,
	tradesProducer *KafkaProducerWrapper,
	candleProducer *KafkaProducerWrapper,
	wg *sync.WaitGroup,
) {
	defer wg.Done()
	// Larger intervals keep the load on the exchange API low.
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	candleTicker := time.NewTicker(1 * time.Minute)
	defer candleTicker.Stop()

	for {
		select {
		case <-ctx.Done():
			log.Printf("Stopping fetch loop for %s", symbol)
			return
		case <-ticker.C:
			globalRateLimiter.Acquire()
			if globalBreaker.IsOpen() {
				continue
			}

			// Fetch ticker (priority)
			md, err := cc.FetchTicker(symbol)
			if err != nil {
				globalBreaker.Fail()
				log.Printf("Ticker fetch error for %s: %v", symbol, err)

				backoff := time.Duration(100+rand.Intn(200)) * time.Millisecond
				select {
				case <-ctx.Done():
					return
				case <-time.After(backoff):
				}

				// Reuse the retried response instead of discarding it: the previous
				// code spent an extra API call and then dropped the data.
				md, err = cc.FetchTicker(symbol)
				if err != nil {
					globalBreaker.Fail()
					log.Printf("Ticker retry failed for %s: %v", symbol, err)
					continue
				}
			}

			dp := DataPoint{Symbol: md.Symbol, Price: md.Price, Volume: md.Volume24h, Time: time.Now().Unix()}
			// Use retry for ticker data (2 retries - less critical than candles)
			if err := publishToKafkaWithRetry(ctx, tickerProducer, "ticker", md.Symbol, dp, 2); err != nil {
				log.Printf("Failed to publish ticker for %s after retries: %v", symbol, err)
			} else {
				log.Printf("Published ticker for %s: Price=%.2f, Volume=%.2f", md.Symbol, md.Price, md.Volume24h)
				globalBreaker.Success() // Mark success if ticker fetch and publish are OK
			}

			// Fetch orderbook (lower priority, less frequent)
			if rand.Intn(100) < 30 { // Fetch 30% of the time, simulating 1s interval for selected pairs
				if ob, err := cc.FetchOrderBook(symbol); err == nil {
					if err := publishToKafka(ctx, orderbookProducer, "orderbook", ob.Symbol, ob); err != nil {
						log.Printf("Failed to publish orderbook for %s: %v", symbol, err)
					} else {
						log.Printf("Published orderbook for %s with %d bids and %d asks", ob.Symbol, len(ob.Bids), len(ob.Asks))
						globalBreaker.Success()
					}
				} else {
					globalBreaker.Fail()
					log.Printf("Orderbook fetch error for %s: %v", symbol, err)
				}
			}

			// Fetch trades (medium priority, less frequent)
			if rand.Intn(100) < 50 { // Fetch 50% of the time, simulating 3s interval for selected pairs
				if tr, err := cc.FetchTrades(symbol); err == nil {
					if err := publishToKafka(ctx, tradesProducer, "trades", tr.Symbol, tr); err != nil {
						log.Printf("Failed to publish trades for %s: %v", symbol, err)
					} else {
						log.Printf("Published %d trades for %s", len(tr.Trades), tr.Symbol)
						globalBreaker.Success()
					}
				} else {
					globalBreaker.Fail()
					log.Printf("Trades fetch error for %s: %v", symbol, err)
				}
			}

		case <-candleTicker.C:
			// Fetch 1m candles and persist (once per minute)
			if candles, err := cc.FetchKline(symbol, "1"); err == nil && len(candles) > 0 {
				for _, c := range candles {
					if err := SaveCandle(c); err != nil {
						log.Printf("[DB] Failed to save candle for %s: %v", symbol, err)
					} else {
						log.Printf("[DB] Saved 1m candle for %s: Close=%.2f, Volume=%.2f", c.Symbol, c.Close, c.Volume)
					}
					// Publish candles to Kafka with retry (critical for ML)
					if err := publishToKafkaWithRetry(ctx, candleProducer, "candle_1m", c.Symbol, c, 3); err != nil {
						log.Printf("\u274C Failed to publish candle for %s after retries: %v", symbol, err)
					} else {
						log.Printf("\u2705 Published 1m candle for %s to Kafka", c.Symbol)
					}
				}
			} else if err != nil {
				log.Printf("Kline fetch error for %s: %v", symbol, err)
				globalBreaker.Fail()
			}
		}
	}
}

func publishToKafka(ctx context.Context, producer *KafkaProducerWrapper, topic, key string, data interface{}) error {
	value, err := json.Marshal(data)
	if err != nil {
		return fmt.Errorf("failed to marshal data for topic %s: %w", topic, err)
	}
	return producer.Publish(ctx, []byte(key), value)
}

// publishToKafkaWithRetry publishes to Kafka with exponential backoff retry.
func publishToKafkaWithRetry(
	ctx context.Context,
	producer *KafkaProducerWrapper,
	topic, key string,
	data interface{},
	maxRetries int,
) error {
	value, err := json.Marshal(data)
	if err != nil {
		return fmt.Errorf("failed to marshal data for topic %s: %w", topic, err)
	}

	if maxRetries < 1 {
		maxRetries = 1
	}

	// lastErr must not be shadowed inside the loop, otherwise the error returned
	// below always wraps the (nil) marshal error and the real cause is lost.
	var lastErr error
	for attempt := 0; attempt < maxRetries; attempt++ {
		lastErr = producer.Publish(ctx, []byte(key), value)
		if lastErr == nil {
			return nil
		}

		if attempt < maxRetries-1 {
			// Exponential backoff: 100ms, 200ms, 400ms, 800ms, ...
			backoffDuration := time.Duration(1<<uint(attempt)) * 100 * time.Millisecond

			log.Printf("Kafka publish to %s failed (attempt %d/%d): %v. Retrying in %v...",
				topic, attempt+1, maxRetries, lastErr, backoffDuration)

			select {
			case <-ctx.Done():
				return fmt.Errorf("context cancelled during retry: %w", ctx.Err())
			case <-time.After(backoffDuration):
			}
		}
	}

	return fmt.Errorf("failed to publish to %s after %d attempts: %w", topic, maxRetries, lastErr)
}
