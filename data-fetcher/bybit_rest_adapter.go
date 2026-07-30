package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"
)

const (
	bybitDefaultBaseURL = "https://api.bybit.com"
	bybitMaxBodyBytes   = 4 << 20 // 4 MiB is far above any market response
)

// BybitREST is a small REST adapter for the Bybit v5 spot API.
type BybitREST struct {
	httpClient *http.Client
	baseURL    string
}

func NewBybitREST() *BybitREST {
	base := strings.TrimRight(os.Getenv("BYBIT_BASE_URL"), "/")
	if base == "" {
		base = bybitDefaultBaseURL
	}
	return &BybitREST{
		httpClient: &http.Client{Timeout: 15 * time.Second},
		baseURL:    base,
	}
}

// snippet keeps error messages short but diagnosable.
func snippet(b []byte) string {
	s := strings.TrimSpace(string(b))
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) > 200 {
		return s[:200] + "..."
	}
	return s
}

// getJSON performs the request and decodes the Bybit envelope.
//
// The previous code unmarshalled straight into a map and ignored both the HTTP
// status and retCode, so a 403 from an edge proxy surfaced as an opaque
// "invalid character 'e'" with no clue about the real cause.
func (c *BybitREST) getJSON(endpoint string) (map[string]interface{}, error) {
	resp, err := c.httpClient.Get(endpoint)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, bybitMaxBodyBytes))
	if err != nil {
		return nil, fmt.Errorf("read body: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("bybit http %d (%s): %s", resp.StatusCode, resp.Header.Get("Content-Type"), snippet(body))
	}

	var root map[string]interface{}
	if err := json.Unmarshal(body, &root); err != nil {
		return nil, fmt.Errorf("bybit returned non-JSON (%s): %s", resp.Header.Get("Content-Type"), snippet(body))
	}

	// v5 always carries retCode; anything non-zero is an API level failure.
	if code, ok := toFloat(root["retCode"]); ok && code != 0 {
		msg, _ := root["retMsg"].(string)
		return nil, fmt.Errorf("bybit retCode=%d retMsg=%q", int(code), msg)
	}

	return root, nil
}

func (c *BybitREST) endpoint(path string, params url.Values) string {
	return c.baseURL + path + "?" + params.Encode()
}

// OrderBookSnapshot represents aggregated bid/ask data
type OrderBookSnapshot struct {
	Symbol string
	Bids   [][2]float64
	Asks   [][2]float64
	Time   int64
}

// Trade represents a single trade entry
type Trade struct {
	Price float64
	Size  float64
	Side  string
	Time  int64
}

// TradesSnapshot contains trades for a symbol
type TradesSnapshot struct {
	Symbol string
	Trades []Trade
	Time   int64
}

// FetchTicker fetches ticker data for a symbol
func (c *BybitREST) FetchTicker(symbol string) (MarketData, error) {
	root, err := c.getJSON(c.endpoint("/v5/market/tickers", url.Values{
		"category": {"spot"},
		"symbol":   {symbol},
	}))
	if err != nil {
		return MarketData{Symbol: symbol}, err
	}

	md := MarketData{Symbol: symbol, AvgVolume7d: 5000.0} // AvgVolume7d is a placeholder
	if res, ok := root["result"].(map[string]interface{}); ok {
		if list, ok := res["list"].([]interface{}); ok && len(list) > 0 {
			if item, ok := list[0].(map[string]interface{}); ok {
				parseTickerMap(item, &md)
			}
		}
	}
	if md.Price == 0 {
		return MarketData{}, errors.New("no valid price data for " + symbol)
	}
	return md, nil
}

// FetchOrderBook fetches the order book snapshot for a symbol
func (c *BybitREST) FetchOrderBook(symbol string) (OrderBookSnapshot, error) {
	root, err := c.getJSON(c.endpoint("/v5/market/orderbook", url.Values{
		"category": {"spot"},
		"symbol":   {symbol},
		"limit":    {"50"},
	}))
	if err != nil {
		return OrderBookSnapshot{Symbol: symbol}, err
	}

	ob := OrderBookSnapshot{Symbol: symbol, Time: time.Now().Unix()}
	if res, ok := root["result"].(map[string]interface{}); ok {
		// v5 spot uses "b"/"a"; older shapes used "bids"/"asks".
		ob.Bids = parseLevels(res, "b", "bids")
		ob.Asks = parseLevels(res, "a", "asks")
	}
	return ob, nil
}

func parseLevels(res map[string]interface{}, keys ...string) [][2]float64 {
	for _, key := range keys {
		raw, ok := res[key].([]interface{})
		if !ok {
			continue
		}
		levels := make([][2]float64, 0, len(raw))
		for _, item := range raw {
			arr, ok := item.([]interface{})
			if !ok || len(arr) < 2 {
				continue
			}
			p, _ := toFloat(arr[0])
			s, _ := toFloat(arr[1])
			levels = append(levels, [2]float64{p, s})
		}
		if len(levels) > 0 {
			return levels
		}
	}
	return nil
}

// FetchTrades fetches recent trades for a symbol
func (c *BybitREST) FetchTrades(symbol string) (TradesSnapshot, error) {
	root, err := c.getJSON(c.endpoint("/v5/market/recent-trade", url.Values{
		"category": {"spot"},
		"symbol":   {symbol},
		"limit":    {"50"},
	}))
	if err != nil {
		return TradesSnapshot{Symbol: symbol}, err
	}

	ts := TradesSnapshot{Symbol: symbol, Time: time.Now().Unix()}
	if res, ok := root["result"].(map[string]interface{}); ok {
		if list, ok := res["list"].([]interface{}); ok {
			for _, it := range list {
				obj, ok := it.(map[string]interface{})
				if !ok {
					continue
				}
				t := Trade{}
				if p, ok := toFloat(obj["price"]); ok {
					t.Price = p
				}
				if s, ok := toFloat(obj["size"]); ok {
					t.Size = s
				} else if s, ok := toFloat(obj["qty"]); ok {
					t.Size = s
				}
				if side, ok := obj["side"].(string); ok {
					t.Side = side
				}
				if ms, ok := toFloat(obj["time"]); ok {
					t.Time = int64(ms) / 1000 // Bybit returns milliseconds
				}
				ts.Trades = append(ts.Trades, t)
			}
		}
	}
	if len(ts.Trades) == 0 {
		return TradesSnapshot{}, errors.New("no trades data available for " + symbol)
	}
	return ts, nil
}

// FetchKline fetches candlestick data for a symbol.
// Bybit intervals: 1,3,5,15,30,60,120,240,360,720,D,W,M
func (c *BybitREST) FetchKline(symbol string, interval string) ([]Candle, error) {
	root, err := c.getJSON(c.endpoint("/v5/market/kline", url.Values{
		"category": {"spot"},
		"symbol":   {symbol},
		"interval": {interval},
		"limit":    {"1"},
	}))
	if err != nil {
		return nil, err
	}

	var candles []Candle
	if res, ok := root["result"].(map[string]interface{}); ok {
		if list, ok := res["list"].([]interface{}); ok {
			for _, item := range list {
				arr, ok := item.([]interface{})
				if !ok || len(arr) < 6 {
					continue
				}
				// arr[0] used to be cast with arr[0].(string), which panics on a
				// numeric timestamp.
				ms, ok := toFloat(arr[0])
				if !ok {
					continue
				}
				o, _ := toFloat(arr[1])
				h, _ := toFloat(arr[2])
				l, _ := toFloat(arr[3])
				cl, _ := toFloat(arr[4])
				v, _ := toFloat(arr[5])
				candles = append(candles, Candle{
					Symbol:    symbol,
					Timestamp: int64(ms) / 1000, // milliseconds -> seconds
					Open:      o,
					High:      h,
					Low:       l,
					Close:     cl,
					Volume:    v,
				})
			}
		}
	}
	return candles, nil
}

func parseTickerMap(src map[string]interface{}, md *MarketData) {
	if s, ok := src["symbol"].(string); ok {
		md.Symbol = s
	}
	if v, ok := toFloat(src["lastPrice"]); ok {
		md.Price = v
	} else if v, ok := toFloat(src["price"]); ok {
		md.Price = v
	}
	if v, ok := toFloat(src["volume24h"]); ok {
		md.Volume24h = v
	}
	if v, ok := toFloat(src["bid1Price"]); ok {
		md.BidVolume = v
	}
	if v, ok := toFloat(src["ask1Price"]); ok {
		md.AskVolume = v
	}
}

// toFloat accepts both the numeric and the stringified numbers Bybit mixes.
func toFloat(v interface{}) (float64, bool) {
	switch t := v.(type) {
	case float64:
		return t, true
	case json.Number:
		if f, err := t.Float64(); err == nil {
			return f, true
		}
	case string:
		if f, err := strconv.ParseFloat(strings.TrimSpace(t), 64); err == nil {
			return f, true
		}
	}
	return 0, false
}
