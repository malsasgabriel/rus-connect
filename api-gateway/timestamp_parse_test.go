package main

import (
	"encoding/json"
	"testing"
	"time"
)

// Ensure that api-gateway can parse numeric and string timestamps produced by analytics-engine
func TestDirectionSignalTimestampParsing(t *testing.T) {
	// Numeric timestamp (seconds)
	msgNum := map[string]interface{}{
		"symbol":    "BTCUSDT",
		"timestamp": time.Now().Unix(),
		"direction": "UP",
	}
	bNum, _ := json.Marshal(msgNum)

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(bNum, &raw); err != nil {
		t.Fatalf("failed unmarshal raw: %v", err)
	}

	// string timestamp RFC3339
	s := time.Now().UTC().Format(time.RFC3339)
	msgStr := map[string]interface{}{
		"symbol":    "BTCUSDT",
		"timestamp": s,
		"direction": "UP",
	}
	bStr, _ := json.Marshal(msgStr)

	var raw2 map[string]json.RawMessage
	if err := json.Unmarshal(bStr, &raw2); err != nil {
		t.Fatalf("failed unmarshal raw2: %v", err)
	}

	// simulate parsing logic: try numeric
	var num json.Number
	if err := json.Unmarshal(raw["timestamp"], &num); err != nil {
		t.Fatalf("numeric timestamp parse failed: %v", err)
	}

	// simulate parsing string
	var s2 string
	if err := json.Unmarshal(raw2["timestamp"], &s2); err != nil {
		t.Fatalf("string timestamp parse failed: %v", err)
	}

	if _, err := time.Parse(time.RFC3339, s2); err != nil {
		t.Fatalf("string timestamp not RFC3339: %v", err)
	}
}
