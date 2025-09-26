package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"os"
	"sync"
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
	rand.Seed(time.Now().UnixNano())
	log.Println("Data Fetcher: PredPump Radar started")

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
		log.Println("[DB] initialized successfully")
		// Ensure indexes are created
		_, err := GetDB().Exec(`
            CREATE INDEX IF NOT EXISTS idx_candle_symbol_time ON candle_1m (symbol, timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_candle_timestamp ON candle_1m (timestamp);
            CREATE INDEX IF NOT EXISTS idx_candle_volume ON candle_1m (volume DESC);
        `)
		if err != nil {
			log.Fatalf("[DB] Failed to create indexes: %v", err)
		}
	} else {
		log.Println("[DB] not initialized; candles won't be persisted to PostgreSQL")
	}

	kafkaBrokers := os.Getenv("KAFKA_BROKERS")
	if kafkaBrokers == "" {
		kafkaBrokers = "kafka:9092"
	}

	tickerProducer := NewKafkaProducerWrapper(kafkaBrokers, "ticker")
	orderbookProducer := NewKafkaProducerWrapper(kafkaBrokers, "orderbook")
	tradesProducer := NewKafkaProducerWrapper(kafkaBrokers, "trades")
	candleProducer := NewKafkaProducerWrapper(kafkaBrokers, "candle_1m")

	defer tickerProducer.Close()
	defer orderbookProducer.Close()
	defer tradesProducer.Close()
	defer candleProducer.Close()

	var symbols []string
	// Top 10 cryptocurrencies by market cap for real analysis
	symbols = []string{"BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"}

	cc := NewBybitREST()

	var wg sync.WaitGroup
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	for _, s := range symbols {
		wg.Add(1)
		go fetchLoopForSymbol(ctx, s, cc, tickerProducer, orderbookProducer, tradesProducer, candleProducer, &wg)
	}

	wg.Wait()
	log.Println("Data Fetcher stopped.")
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
	ticker := time.NewTicker(3 * time.Second)
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
				time.Sleep(50 * time.Millisecond)
				continue
			}

			// Fetch ticker (priority)
			md, err := cc.FetchTicker(symbol)
			if err != nil {
				globalBreaker.Fail()
				log.Printf("Ticker fetch error for %s: %v", symbol, err)
				backoff := time.Duration(100+rand.Intn(200)) * time.Millisecond
				time.Sleep(backoff)
				if _, err := cc.FetchTicker(symbol); err != nil {
					globalBreaker.Fail()
					time.Sleep(200 * time.Millisecond)
					continue
				}
			} else {
				dp := DataPoint{Symbol: md.Symbol, Price: md.Price, Volume: md.Volume24h, Time: time.Now().Unix()}
				if err := publishToKafka(ctx, tickerProducer, "ticker", md.Symbol, dp); err != nil {
					log.Printf("Failed to publish ticker for %s: %v", symbol, err)
				} else {
					log.Printf("Published ticker for %s: Price=%.2f, Volume=%.2f", md.Symbol, md.Price, md.Volume24h)
					globalBreaker.Success() // Mark success if ticker fetch and publish are OK
				}
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
					// Publish candles to Kafka for Analytics Engine
					if err := publishToKafka(ctx, candleProducer, "candle_1m", c.Symbol, c); err != nil {
						log.Printf("Failed to publish candle for %s: %v", symbol, err)
					} else {
						log.Printf("Published 1m candle for %s to Kafka", c.Symbol)
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
