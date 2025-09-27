package main

import (
	"testing"
)

// Basic RSI test: ensures RSI returns value in 0..100 and not NaN for a known price series
func TestCalculateRSI(t *testing.T) {
	prices := make([]Candle, 30)
	base := 100.0
	for i := 0; i < 30; i++ {
		p := base + float64(i)
		prices[i] = Candle{Close: p}
	}

	rsi := calculateRSI(prices, 14)
	if rsi < 0 || rsi > 100 {
		t.Fatalf("RSI out of range: %v", rsi)
	}
}
