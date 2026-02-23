package main

import (
	"database/sql"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	_ "github.com/lib/pq"
)

type BybitKlineResponse struct {
	RetCode int    `json:"retCode"`
	RetMsg  string `json:"retMsg"`
	Result  struct {
		Category string     `json:"category"`
		Symbol   string     `json:"symbol"`
		List     [][]string `json:"list"`
	} `json:"result"`
}

type Candle struct {
	Symbol    string
	Timestamp int64
	Open      float64
	High      float64
	Low       float64
	Close     float64
	Volume    float64
}

func fetchKlines(symbol string, startMs, endMs int64) ([]Candle, error) {
	client := &http.Client{Timeout: 30 * time.Second}
	base := "https://api.bybit.com/v5/market/kline"

	limit := 2000
	all := make([]Candle, 0)
	curStartMs := startMs
	curEndMs := endMs
	page := 0
	maxPages := 1000 // safety cap

	for curStartMs <= curEndMs && page < maxPages {
		page++
		u, _ := url.Parse(base)
		q := u.Query()
		q.Set("category", "spot")
		q.Set("symbol", symbol)
		q.Set("interval", "1")
		q.Set("start", strconv.FormatInt(curStartMs, 10))
		q.Set("end", strconv.FormatInt(curEndMs, 10))
		q.Set("limit", strconv.Itoa(limit))
		u.RawQuery = q.Encode()

		log.Printf("[DEBUG] Page %d: GET %s", page, u.String())
		resp, err := client.Get(u.String())
		if err != nil {
			log.Printf("[ERROR] HTTP request failed: %v", err)
			return nil, fmt.Errorf("http get: %w", err)
		}
		log.Printf("[DEBUG] HTTP status: %s", resp.Status)
		body, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			log.Printf("[ERROR] Read body failed: %v", err)
			return nil, err
		}

		var byresp BybitKlineResponse
		if err := json.Unmarshal(body, &byresp); err != nil {
			log.Printf("[ERROR] JSON unmarshal failed: %v", err)
			log.Printf("[DEBUG] Body: %s", string(body))
			return nil, fmt.Errorf("unmarshal: %w", err)
		}
		log.Printf("[DEBUG] Bybit retCode=%d retMsg=%s", byresp.RetCode, byresp.RetMsg)
		if byresp.RetCode != 0 {
			log.Printf("[ERROR] Bybit API error: %s", byresp.RetMsg)
			return nil, fmt.Errorf("bybit error: %s", byresp.RetMsg)
		}

		log.Printf("[INFO] Page %d: received %d klines (startMs=%d endMs=%d)", page, len(byresp.Result.List), curStartMs, curEndMs)
		if len(byresp.Result.List) == 0 {
			log.Printf("[INFO] No more data for %s at page %d", symbol, page)
			break
		}

		var maxTs int64
		var minTs int64
		for i, row := range byresp.Result.List {
			if len(row) < 6 {
				log.Printf("[WARN] Skipping short kline row: %v", row)
				continue
			}
			tsMs, err := strconv.ParseInt(row[0], 10, 64)
			if err != nil {
				log.Printf("[WARN] Bad timestamp: %v", row[0])
				continue
			}
			open, _ := strconv.ParseFloat(row[1], 64)
			high, _ := strconv.ParseFloat(row[2], 64)
			low, _ := strconv.ParseFloat(row[3], 64)
			closev, _ := strconv.ParseFloat(row[4], 64)
			vol, _ := strconv.ParseFloat(row[5], 64)

			c := Candle{
				Symbol:    symbol,
				Timestamp: tsMs / 1000,
				Open:      open,
				High:      high,
				Low:       low,
				Close:     closev,
				Volume:    vol,
			}
			all = append(all, c)
			if i == 0 {
				minTs = tsMs
				maxTs = tsMs
			} else {
				if tsMs > maxTs {
					maxTs = tsMs
				}
				if tsMs < minTs {
					minTs = tsMs
				}
			}
		}
		if maxTs == 0 && minTs == 0 {
			log.Printf("[INFO] No valid timestamps returned on page %d, stopping", page)
			break
		}
		// Determine ordering by parsing first and last rows directly
		var firstRowTs, lastRowTs int64
		if len(byresp.Result.List) > 0 {
			if t0, err := strconv.ParseInt(byresp.Result.List[0][0], 10, 64); err == nil {
				firstRowTs = t0
			}
			if tN, err := strconv.ParseInt(byresp.Result.List[len(byresp.Result.List)-1][0], 10, 64); err == nil {
				lastRowTs = tN
			}
		}

		if firstRowTs == 0 || lastRowTs == 0 {
			log.Printf("[WARN] Couldn't determine row ordering (first=%d last=%d); stopping to be safe", firstRowTs, lastRowTs)
			break
		}

		if firstRowTs > lastRowTs {
			// descending (newest first) — move curEndMs to oldest returned - 1 ms
			newEndMs := minTs - 1
			if newEndMs < curStartMs {
				log.Printf("[DEBUG] Reached start boundary after adjusting end; stopping")
				break
			}
			curEndMs = newEndMs
			log.Printf("[DEBUG] Descending order detected. New window: startMs=%d endMs=%d", curStartMs, curEndMs)
		} else {
			// ascending — move curStartMs to maxTs + 1 ms
			newStartMs := maxTs + 1
			if newStartMs <= curStartMs {
				log.Printf("[WARN] No forward progress (curStartMs=%d newStartMs=%d) on page %d, stopping to avoid infinite loop", curStartMs, newStartMs, page)
				break
			}
			curStartMs = newStartMs
			log.Printf("[DEBUG] Ascending order detected. New window: startMs=%d endMs=%d", curStartMs, curEndMs)
		}
	}

	if len(all) > 1 && all[0].Timestamp > all[len(all)-1].Timestamp {
		for i := 0; i < len(all)/2; i++ {
			j := len(all) - 1 - i
			all[i], all[j] = all[j], all[i]
		}
	}

	log.Printf("[RESULT] Total candles fetched for %s: %d", symbol, len(all))
	return all, nil
}

func insertCandles(db *sql.DB, symbol string, candles []Candle) error {
	if len(candles) == 0 {
		return nil
	}
	// Insert in batches to avoid PostgreSQL parameter limit (65535)
	const colsPerRow = 7
	const maxParams = 65535
	maxRowsPerBatch := maxParams / colsPerRow
	if maxRowsPerBatch <= 0 {
		maxRowsPerBatch = 1000
	}
	// but keep a safer smaller batch size
	batchSize := 1000
	if batchSize > maxRowsPerBatch {
		batchSize = maxRowsPerBatch
	}

	for start := 0; start < len(candles); start += batchSize {
		end := start + batchSize
		if end > len(candles) {
			end = len(candles)
		}

		valueStrings := make([]string, 0, end-start)
		args := make([]interface{}, 0, (end-start)*colsPerRow)
		for i := start; i < end; i++ {
			c := candles[i]
			idx := (i - start) * colsPerRow
			// placeholders for this batch start from 1..n
			valueStrings = append(valueStrings, fmt.Sprintf("($%d,$%d,$%d,$%d,$%d,$%d,$%d)", idx+1, idx+2, idx+3, idx+4, idx+5, idx+6, idx+7))
			args = append(args, symbol, c.Timestamp, c.Open, c.High, c.Low, c.Close, c.Volume)
		}

		query := fmt.Sprintf(`
		INSERT INTO candle_cache (symbol, timestamp, open, high, low, close, volume)
		VALUES %s
		ON CONFLICT (symbol, timestamp) DO UPDATE SET
			open = EXCLUDED.open,
			high = EXCLUDED.high,
			low = EXCLUDED.low,
			close = EXCLUDED.close,
			volume = EXCLUDED.volume
	`, strings.Join(valueStrings, ","))

		if _, err := db.Exec(query, args...); err != nil {
			return err
		}
	}

	return nil
}

func main() {
	symbolsFlag := flag.String("symbols", "BTCUSDT", "comma-separated symbols")
	startFlag := flag.String("start", "2025-09-15", "start date YYYY-MM-DD")
	endFlag := flag.String("end", "2025-09-21", "end date YYYY-MM-DD")
	flag.Parse()

	symbols := strings.Split(*symbolsFlag, ",")
	startT, err := time.Parse("2006-01-02", *startFlag)
	if err != nil {
		log.Fatalf("invalid start date: %v", err)
	}
	// end inclusive -> set to end of day
	endT, err := time.Parse("2006-01-02", *endFlag)
	if err != nil {
		log.Fatalf("invalid end date: %v", err)
	}
	endT = endT.Add(24*time.Hour - time.Second)

	startMs := startT.Unix() * 1000
	endMs := endT.Unix() * 1000

	dsn := os.Getenv("PG_DSN")
	if dsn == "" {
		dsn = "host=postgres user=admin password=password dbname=predpump sslmode=disable"
	}
	db, err := sql.Open("postgres", dsn)
	if err != nil {
		log.Fatalf("open db: %v", err)
	}
	defer db.Close()

	// Parallel fetch per symbol to speed up multiple-symbol downloads
	type result struct {
		sym     string
		candles []Candle
		err     error
	}

	ch := make(chan result, len(symbols))

	for _, sym := range symbols {
		sym := strings.TrimSpace(sym)
		if sym == "" {
			continue
		}
		go func(s string) {
			log.Printf("Fetching %s from %s to %s", s, startT.Format(time.RFC3339), endT.Format(time.RFC3339))
			candles, err := fetchKlines(s, startMs, endMs)
			ch <- result{sym: s, candles: candles, err: err}
		}(sym)
	}

	// collect results
	for i := 0; i < len(symbols); i++ {
		r := <-ch
		if r.err != nil {
			log.Printf("Failed to fetch %s: %v", r.sym, r.err)
			continue
		}
		log.Printf("Fetched %d candles for %s", len(r.candles), r.sym)
		if err := insertCandles(db, r.sym, r.candles); err != nil {
			log.Printf("Failed to insert candles for %s: %v", r.sym, err)
			continue
		}
		log.Printf("Inserted %d candles for %s into DB", len(r.candles), r.sym)
	}
}
