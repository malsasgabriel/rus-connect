package main

import (
	"encoding/json"
	"errors"
	"io"
	"log"
	"net/http"
	"net/url"
	"strconv"
	"time"
)

// BybitREST — CCXT-like REST adapter for Bybit spot API (simplified)
type BybitREST struct {
	httpClient *http.Client
	baseURL    string
}

func NewBybitREST() *BybitREST {
	return &BybitREST{httpClient: &http.Client{Timeout: 15 * time.Second}, baseURL: "https://api.bybit.com"}
}

// MarketData is defined in data-fetcher/main.go

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
	endpoint := c.baseURL + "/v5/market/tickers?category=spot&symbol=" + url.QueryEscape(symbol)
	resp, err := c.httpClient.Get(endpoint)
	if err != nil {
		log.Printf("BybitREST ticker error: %v", err)
		return MarketData{Symbol: symbol}, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return MarketData{Symbol: symbol}, err
	}
	var root map[string]interface{}
	if err := json.Unmarshal(body, &root); err != nil {
		return MarketData{Symbol: symbol}, err
	}
	md := MarketData{Symbol: symbol, AvgVolume7d: 5000.0} // AvgVolume7d is placeholder
	if res, ok := root["result"].(map[string]interface{}); ok {
		if list, ok := res["list"].([]interface{}); ok && len(list) > 0 {
			if item, ok := list[0].(map[string]interface{}); ok {
				parseTickerMap(item, &md)
			}
		}
	}
	// Ensure we have valid data - no fallbacks, use real API data only
	if md.Price == 0 {
		return MarketData{}, errors.New("no valid price data for " + symbol)
	}
	return md, nil
}

// FetchOrderBook fetches the order book snapshot for a symbol
func (c *BybitREST) FetchOrderBook(symbol string) (OrderBookSnapshot, error) {
	endpoint := c.baseURL + "/v5/market/orderbook?category=spot&symbol=" + url.QueryEscape(symbol) + "&limit=50"
	resp, err := c.httpClient.Get(endpoint)
	if err != nil {
		log.Printf("BybitREST orderbook error: %v", err)
		return OrderBookSnapshot{Symbol: symbol}, err
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return OrderBookSnapshot{Symbol: symbol}, err
	}
	var root map[string]interface{}
	if err := json.Unmarshal(b, &root); err != nil {
		return OrderBookSnapshot{Symbol: symbol}, err
	}
	ob := OrderBookSnapshot{Symbol: symbol, Time: time.Now().Unix()}
	if res, ok := root["result"].(map[string]interface{}); ok {
		if bids, ok := res["bids"].([]interface{}); ok {
			for _, bitem := range bids {
				if arr, ok := bitem.([]interface{}); ok && len(arr) >= 2 {
					p, _ := toFloat(arr[0])
					s, _ := toFloat(arr[1])
					ob.Bids = append(ob.Bids, [2]float64{p, s})
				}
			}
		}
		if asks, ok := res["asks"].([]interface{}); ok {
			for _, aitem := range asks {
				if arr, ok := aitem.([]interface{}); ok && len(arr) >= 2 {
					p, _ := toFloat(arr[0])
					s, _ := toFloat(arr[1])
					ob.Asks = append(ob.Asks, [2]float64{p, s})
				}
			}
		}
	}
	return ob, nil
}

// FetchTrades fetches recent trades for a symbol
func (c *BybitREST) FetchTrades(symbol string) (TradesSnapshot, error) {
	endpoint := c.baseURL + "/v5/market/recent-trade?category=spot&symbol=" + url.QueryEscape(symbol) + "&limit=50"
	resp, err := c.httpClient.Get(endpoint)
	if err != nil {
		log.Printf("BybitREST trades error: %v", err)
		return TradesSnapshot{Symbol: symbol}, err
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return TradesSnapshot{Symbol: symbol}, err
	}
	var root map[string]interface{}
	if err := json.Unmarshal(b, &root); err != nil {
		return TradesSnapshot{Symbol: symbol}, err
	}
	ts := TradesSnapshot{Symbol: symbol, Time: time.Now().Unix()}
	if res, ok := root["result"].(map[string]interface{}); ok {
		if list, ok := res["list"].([]interface{}); ok {
			for _, it := range list {
				if obj, ok := it.(map[string]interface{}); ok {
					t := Trade{}
					if p, ok := obj["price"].(string); ok {
						if v, err := strconv.ParseFloat(p, 64); err == nil {
							t.Price = v
						}
					} else if p, ok := obj["price"].(float64); ok {
						t.Price = p
					}
					if s, ok := obj["qty"].(string); ok {
						if v, err := strconv.ParseFloat(s, 64); err == nil {
							t.Size = v
						}
					} else if s, ok := obj["qty"].(float64); ok {
						t.Size = s
					}
					if side, ok := obj["side"].(string); ok {
						t.Side = side
					}
					if timeVal, ok := obj["time"].(string); ok {
						if v, err := strconv.ParseInt(timeVal, 10, 64); err == nil {
							t.Time = v / 1000 // Bybit returns ms, convert to seconds
						}
					} else if timeVal, ok := obj["time"].(float64); ok {
						t.Time = int64(timeVal) / 1000
					}
					ts.Trades = append(ts.Trades, t)
				}
			}
		}
	}
	if len(ts.Trades) == 0 {
		return TradesSnapshot{}, errors.New("no trades data available for " + symbol)
	}
	return ts, nil
}

// FetchKline fetches candlestick data for a symbol
func (c *BybitREST) FetchKline(symbol string, interval string) ([]Candle, error) {
	// Bybit interval: 1,3,5,15,30,60,120,240,360,720,D,W,M
	// Currently supporting '1' for 1-minute candles
	endpoint := c.baseURL + "/v5/market/kline?category=spot&symbol=" + url.QueryEscape(symbol) + "&interval=" + url.QueryEscape(interval) + "&limit=1" // Fetch last candle
	resp, err := c.httpClient.Get(endpoint)
	if err != nil {
		log.Printf("BybitREST kline error: %v", err)
		return nil, err
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	var root map[string]interface{}
	if err := json.Unmarshal(b, &root); err != nil {
		return nil, err
	}
	var candles []Candle
	if res, ok := root["result"].(map[string]interface{}); ok {
		if list, ok := res["list"].([]interface{}); ok {
			for _, item := range list {
				if arr, ok := item.([]interface{}); ok && len(arr) >= 6 {
					ts, _ := strconv.ParseInt(arr[0].(string), 10, 64)
					o, _ := toFloat(arr[1])
					h, _ := toFloat(arr[2])
					l, _ := toFloat(arr[3])
					cl, _ := toFloat(arr[4])
					v, _ := toFloat(arr[5])
					candles = append(candles, Candle{
						Symbol:    symbol,
						Timestamp: ts / 1000, // Bybit returns milliseconds, convert to seconds
						Open:      o,
						High:      h,
						Low:       l,
						Close:     cl,
						Volume:    v,
					})
				}
			}
		}
	}
	return candles, nil
}

func parseTickerMap(src map[string]interface{}, md *MarketData) {
	if s, ok := src["symbol"].(string); ok {
		md.Symbol = s
	}
	if p, ok := src["lastPrice"].(string); ok { // Changed from "last_price" to "lastPrice"
		if v, err := strconv.ParseFloat(p, 64); err == nil {
			md.Price = v
		}
	} else if p, ok := src["price"].(string); ok {
		if v, err := strconv.ParseFloat(p, 64); err == nil {
			md.Price = v
		}
	} else if p, ok := src["price"].(float64); ok {
		md.Price = p
	}
	if v, ok := src["volume24h"].(float64); ok {
		md.Volume24h = v
	} else if v, ok := src["volume24h"].(string); ok {
		if fv, err := strconv.ParseFloat(v, 64); err == nil {
			md.Volume24h = fv
		}
	}
	if b, ok := src["bid1Price"].(string); ok { // Changed from "bid_price" to "bid1Price"
		if bv, err := strconv.ParseFloat(b, 64); err == nil {
			md.BidVolume = bv
		}
	}
	if a, ok := src["ask1Price"].(string); ok { // Changed from "ask_price" to "ask1Price"
		if av, err := strconv.ParseFloat(a, 64); err == nil {
			md.AskVolume = av
		}
	}
	// `largest_order` is not available directly in Bybit v5 ticker response. Set a placeholder or derive.
	// For now, it remains 0 as per MarketData init, or can be simulated if needed.
	// if lo, ok := src["largest_order"].(float64); ok {
	// 	md.LargestOrder = lo
	// } else if lo, ok := src["largest_order"].(string); ok {
	// 	if lv, err := strconv.ParseFloat(lo, 64); err == nil {
	// 		md.LargestOrder = lv
	// 	}
	// }
}

func toFloat(v interface{}) (float64, bool) {
	switch t := v.(type) {
	case float64:
		return t, true
	case string:
		if f, err := strconv.ParseFloat(t, 64); err == nil {
			return f, true
		}
	}
	return 0, false
}
