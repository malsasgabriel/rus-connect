package main

// Candle represents a single candlestick
type Candle struct {
	Symbol    string
	Timestamp int64
	Open      float64
	High      float64
	Low       float64
	Close     float64
	Volume    float64
}
