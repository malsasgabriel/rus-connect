package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strconv"
	"time"
)

// HistoricalDataLoader loads historical candle data from Bybit API
type HistoricalDataLoader struct {
	baseURL string
	client  *http.Client
}

// BybitKlineResponse represents Bybit API response for kline data
type BybitKlineResponse struct {
	RetCode int    `json:"retCode"`
	RetMsg  string `json:"retMsg"`
	Result  struct {
		Category string     `json:"category"`
		Symbol   string     `json:"symbol"`
		List     [][]string `json:"list"`
	} `json:"result"`
}

func NewHistoricalDataLoader() *HistoricalDataLoader {
	return &HistoricalDataLoader{
		baseURL: "https://api.bybit.com",
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// LoadHistoricalCandles loads the last 1440 candles (24 hours) for a symbol
func (hdl *HistoricalDataLoader) LoadHistoricalCandles(symbol string) ([]Candle, error) {
	// Calculate timestamp for 24 hours ago
	endTime := time.Now().Unix() * 1000          // milliseconds
	startTime := endTime - (24 * 60 * 60 * 1000) // 24 hours ago

	url := fmt.Sprintf("%s/v5/market/kline?category=spot&symbol=%s&interval=1&start=%d&end=%d&limit=1000",
		hdl.baseURL, symbol, startTime, endTime)

	log.Printf("📥 Loading historical data for %s from Bybit API...", symbol)

	resp, err := hdl.client.Get(url)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch historical data for %s: %v", symbol, err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %v", err)
	}

	var bybitResp BybitKlineResponse
	if err := json.Unmarshal(body, &bybitResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %v", err)
	}

	if bybitResp.RetCode != 0 {
		return nil, fmt.Errorf("bybit API error: %s", bybitResp.RetMsg)
	}

	// Convert Bybit format to our Candle format
	candles := make([]Candle, 0, len(bybitResp.Result.List))

	for _, kline := range bybitResp.Result.List {
		if len(kline) < 6 {
			continue
		}

		timestamp, err := strconv.ParseInt(kline[0], 10, 64)
		if err != nil {
			continue
		}

		open, err := strconv.ParseFloat(kline[1], 64)
		if err != nil {
			continue
		}

		high, err := strconv.ParseFloat(kline[2], 64)
		if err != nil {
			continue
		}

		low, err := strconv.ParseFloat(kline[3], 64)
		if err != nil {
			continue
		}

		close, err := strconv.ParseFloat(kline[4], 64)
		if err != nil {
			continue
		}

		volume, err := strconv.ParseFloat(kline[5], 64)
		if err != nil {
			continue
		}

		candle := Candle{
			Symbol:    symbol,
			Timestamp: timestamp / 1000, // Convert to seconds
			Open:      open,
			High:      high,
			Low:       low,
			Close:     close,
			Volume:    volume,
		}

		candles = append(candles, candle)
	}

	// Sort candles by timestamp (oldest first)
	// Bybit returns newest first, but we want oldest first
	for i := 0; i < len(candles)/2; i++ {
		j := len(candles) - 1 - i
		candles[i], candles[j] = candles[j], candles[i]
	}

	log.Printf("✅ Loaded %d historical candles for %s", len(candles), symbol)
	return candles, nil
}

// LoadHistoricalCandlesWithRetry loads historical data with retry logic
func (hdl *HistoricalDataLoader) LoadHistoricalCandlesWithRetry(symbol string, maxRetries int) ([]Candle, error) {
	var candles []Candle
	var err error

	for attempt := 1; attempt <= maxRetries; attempt++ {
		candles, err = hdl.LoadHistoricalCandles(symbol)
		if err == nil {
			return candles, nil
		}

		log.Printf("⚠️ Attempt %d/%d failed for %s: %v", attempt, maxRetries, symbol, err)

		if attempt < maxRetries {
			// Exponential backoff
			sleepTime := time.Duration(attempt*2) * time.Second
			log.Printf("🔄 Retrying in %v...", sleepTime)
			time.Sleep(sleepTime)
		}
	}

	return nil, fmt.Errorf("failed to load historical data for %s after %d attempts: %v", symbol, maxRetries, err)
}

// PreloadAllSymbols loads historical data for all symbols
func (hdl *HistoricalDataLoader) PreloadAllSymbols(symbols []string) map[string][]Candle {
	result := make(map[string][]Candle)

	for _, symbol := range symbols {
		candles, err := hdl.LoadHistoricalCandlesWithRetry(symbol, 3)
		if err != nil {
			log.Printf("❌ Failed to load historical data for %s: %v", symbol, err)
			continue
		}

		// Ensure we have at least some candles
		if len(candles) < 100 {
			log.Printf("⚠️ Only got %d candles for %s, may affect prediction quality", len(candles), symbol)
		}

		result[symbol] = candles
	}

	return result
}
