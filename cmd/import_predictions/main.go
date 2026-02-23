package main

import (
	"bufio"
	"database/sql"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"
	"strings"
	"time"

	_ "github.com/lib/pq"
)

// Expected CSV columns: symbol,timestamp,direction,confidence,time_horizon,price_target,features
// timestamp in RFC3339 or Unix seconds

func parseTimestamp(s string) (time.Time, error) {
	if s == "" {
		return time.Time{}, fmt.Errorf("empty timestamp")
	}
	if ts, err := strconv.ParseInt(s, 10, 64); err == nil {
		return time.Unix(ts, 0).UTC(), nil
	}
	// try RFC3339
	t, err := time.Parse(time.RFC3339, s)
	if err == nil {
		return t.UTC(), nil
	}
	// try date only
	t, err = time.Parse("2006-01-02", s)
	if err == nil {
		return t.UTC(), nil
	}
	return time.Time{}, fmt.Errorf("unrecognized timestamp format: %s", s)
}

func main() {
	file := flag.String("file", "", "path to CSV or JSON file (use .csv or .json)")
	dry := flag.Bool("dry", true, "dry-run (do not write to DB)")
	batch := flag.Int("batch", 500, "insert batch size")
	dsn := flag.String("dsn", "", "Postgres DSN (optional, env PG_DSN used if empty)")
	flag.Parse()

	if *file == "" {
		log.Fatalf("provide -file path")
	}

	f, err := os.Open(*file)
	if err != nil {
		log.Fatalf("open file: %v", err)
	}
	defer f.Close()

	ext := strings.ToLower(strings.TrimPrefix(filepathExt(*file), "."))

	// prepare DB
	dsnVal := *dsn
	if dsnVal == "" {
		dsnVal = os.Getenv("PG_DSN")
		if dsnVal == "" {
			dsnVal = "host=localhost user=admin password=password dbname=predpump sslmode=disable"
		}
	}

	var db *sql.DB
	if !*dry {
		db, err = sql.Open("postgres", dsnVal)
		if err != nil {
			log.Fatalf("db open: %v", err)
		}
		defer db.Close()
	}

	type Rec struct {
		Symbol      string
		TS          time.Time
		Dir         string
		Conf        float64
		Horizon     int
		PriceTarget sql.NullFloat64
		Features    string
	}

	var records []Rec

	if ext == "csv" {
		r := csv.NewReader(bufio.NewReader(f))
		// read header
		header, err := r.Read()
		if err != nil {
			log.Fatalf("read header: %v", err)
		}
		colIdx := map[string]int{}
		for i, c := range header {
			colIdx[strings.ToLower(strings.TrimSpace(c))] = i
		}
		for {
			rec, err := r.Read()
			if err == io.EOF {
				break
			}
			if err != nil {
				log.Printf("csv read err: %v", err)
				continue
			}
			ts, err := parseTimestamp(rec[colIdx["timestamp"]])
			if err != nil {
				log.Printf("ts parse err: %v", err)
				continue
			}
			conf, _ := strconv.ParseFloat(rec[colIdx["confidence"]], 64)
			horizon := 60
			if i, ok := colIdx["time_horizon"]; ok && rec[i] != "" {
				if v, err := strconv.Atoi(rec[i]); err == nil {
					horizon = v
				}
			}
			var pt sql.NullFloat64
			if i, ok := colIdx["price_target"]; ok && rec[i] != "" {
				if v, err := strconv.ParseFloat(rec[i], 64); err == nil {
					pt.Float64 = v
					pt.Valid = true
				}
			}
			feat := ""
			if i, ok := colIdx["features"]; ok {
				feat = rec[i]
			}
			records = append(records, Rec{Symbol: rec[colIdx["symbol"]], TS: ts, Dir: rec[colIdx["direction"]], Conf: conf, Horizon: horizon, PriceTarget: pt, Features: feat})
		}
	} else if ext == "json" {
		dec := json.NewDecoder(f)
		for {
			var obj map[string]interface{}
			if err := dec.Decode(&obj); err == io.EOF {
				break
			} else if err != nil {
				log.Fatalf("json decode: %v", err)
			}
			s, _ := obj["symbol"].(string)
			tsRaw := obj["timestamp"]
			var ts time.Time
			switch v := tsRaw.(type) {
			case float64:
				ts = time.Unix(int64(v), 0).UTC()
			case string:
				t, err := parseTimestamp(v)
				if err != nil {
					continue
				}
				ts = t
			}
			dir, _ := obj["direction"].(string)
			conf := 0.0
			if cf, ok := obj["confidence"].(float64); ok {
				conf = cf
			}
			horizon := 60
			if h, ok := obj["time_horizon"].(float64); ok {
				horizon = int(h)
			}
			pt := sql.NullFloat64{}
			if v, ok := obj["price_target"].(float64); ok {
				pt.Float64 = v
				pt.Valid = true
			}
			feats := ""
			if fobj, ok := obj["features"]; ok {
				if b, err := json.Marshal(fobj); err == nil {
					feats = string(b)
				}
			}
			records = append(records, Rec{Symbol: s, TS: ts, Dir: dir, Conf: conf, Horizon: horizon, PriceTarget: pt, Features: feats})
		}
	} else {
		log.Fatalf("unsupported extension: %s", ext)
	}

	fmt.Printf("Parsed %d records (dry=%v)\n", len(records), *dry)

	if *dry {
		return
	}

	// insert in batches
	tx, err := db.Begin()
	if err != nil {
		log.Fatalf("begin tx: %v", err)
	}
	stmtStr := `INSERT INTO direction_predictions (symbol, timestamp, direction, confidence, price_target, current_price, time_horizon, features, created_at) VALUES `
	inserted := 0
	for i := 0; i < len(records); i += *batch {
		end := i + *batch
		if end > len(records) {
			end = len(records)
		}
		parts := []string{}
		args := []interface{}{}
		ai := 1
		for _, r := range records[i:end] {
			parts = append(parts, fmt.Sprintf("($%d,$%d,$%d,$%d,$%d,$%d,$%d,$%d,NOW())", ai, ai+1, ai+2, ai+3, ai+4, ai+5, ai+6, ai+7))
			args = append(args, r.Symbol, r.TS, r.Dir, r.Conf, r.PriceTarget, nil, r.Horizon, r.Features)
			ai += 8
		}
		q := stmtStr + strings.Join(parts, ",")
		if _, err := tx.Exec(q, args...); err != nil {
			tx.Rollback()
			log.Fatalf("insert batch err: %v", err)
		}
		inserted += (end - i)
	}
	if err := tx.Commit(); err != nil {
		log.Fatalf("tx commit: %v", err)
	}
	fmt.Printf("Inserted %d records\n", inserted)
}

// filepathExt is a tiny helper, avoids importing path/filepath for single use
func filepathExt(name string) string {
	i := strings.LastIndex(name, ".")
	if i == -1 {
		return ""
	}
	return name[i:]
}
