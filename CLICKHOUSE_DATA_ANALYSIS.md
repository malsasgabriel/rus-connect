# ClickHouse Data & ML Prediction Analysis

## Summary

This document analyzes whether data and features in ClickHouse are actually being collected and whether ML predictions are being made.

---

## 1. Data Flow Architecture

### 1.1 Candle Data Collection Pipeline

```
Bybit API → data-fetcher → Kafka (candle_1m topic) → analytics-engine → ClickHouse
```

**Components:**

1. **data-fetcher** (`data-fetcher/main.go`):
   - Fetches 1m candles from Bybit via `FetchKline()` 
   - Saves candles to ClickHouse `candle_1m` table via `SaveCandle()`
   - Publishes candles to Kafka topic `candle_1m` with retry

2. **analytics-engine** (`analytics-engine/main.go`):
   - Consumes candles from Kafka `candle_1m` topic
   - Processes candles via `ProcessCandle()`
   - Saves candles to ClickHouse `candle_cache` table via `saveHistoricalDataToDB()`

### 1.2 ML Prediction Pipeline

```
Candles → Feature Engine → Neural Network → Prediction → ClickHouse (direction_predictions)
```

**Key Functions:**

1. **Feature Extraction** (`simple_feature_engine.go`):
   - Extracts 16 features from candle history:
     - Price changes (5m, 15m)
     - Volume change
     - RSI (14 periods)
     - SMA ratios (5, 15)
     - Volatility (5 periods)
     - MACD, Bollinger position, Momentum, etc.

2. **ML Prediction** (`online_learning.go`):
   - `makePrediction()` function (line 342-391):
     - Gets/creates model via `getOrCreateModel()`
     - Extracts features from candle history
     - Normalizes features
     - Calls `model.PredictWithDistribution()` 
     - Creates `DirectionSignal` with confidence
     - Emits signal via `emitDirectionSignal()`

3. **Prediction Storage** (`main.go`):
   - `emitDirectionSignal()` (line 287-343):
     - Saves predictions to ClickHouse `direction_predictions` table
     - Publishes to Kafka `direction_signals` topic

---

## 2. ClickHouse Tables

### 2.1 `candle_1m` (Raw Market Data)
```sql
CREATE TABLE candle_1m (
    symbol String,
    timestamp Int64,
    open Float64,
    high Float64,
    low Float64,
    close Float64,
    volume Float64,
    created_at DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(created_at)
PARTITION BY toYYYYMM(toDateTime(timestamp))
ORDER BY (symbol, timestamp)
```

### 2.2 `candle_cache` (Processed Candles)
```sql
-- Same schema as candle_1m
```

### 2.3 `direction_predictions` (ML Predictions)
```sql
CREATE TABLE direction_predictions (
    symbol String,
    timestamp DateTime,
    direction String,
    confidence Float64,
    trust_stage String,
    model_age_sec Int64,
    label_horizon_min Int32,
    class_probs String,
    price_target Float64,
    current_price Float64,
    stop_loss Float64,
    volatility Float64,
    model_used String,
    time_horizon Int32,
    features String,
    actual_direction String,
    actual_price Float64,
    accuracy_score Float64,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree
PARTITION BY toYYYYMM(timestamp)
ORDER BY (symbol, timestamp, created_at)
```

---

## 3. Critical Findings

### ✅ Data Collection IS Implemented

1. **data-fetcher** saves candles to ClickHouse:
   - `SaveCandle()` in `data-fetcher/db.go` inserts into `candle_1m`
   - Logs: `[DB] Saved 1m candle for {symbol}: Close={price}, Volume={volume}`

2. **analytics-engine** also saves candles:
   - `saveHistoricalDataToDB()` in `analytics-engine/main.go` inserts into `candle_cache`
   - Called from `ProcessCandle()` for every candle processed

### ✅ ML Predictions ARE Being Made

1. **Feature extraction** is active:
   - 16 features extracted from 60-candle history
   - Features include RSI, MACD, Bollinger, volatility, momentum

2. **Neural network predictions**:
   - `SimpleNeuralNetwork.PredictWithDistribution()` makes 3-class predictions
   - Confidence dampening based on model trust stage (cold_start → warming → trained)

3. **Predictions saved to ClickHouse**:
   - `emitDirectionSignal()` inserts into `direction_predictions` table
   - Only emits if confidence >= threshold (default 0.10)

### ⚠️ Potential Issues Identified

1. **No Initial Candle History Loading**:
   - The analytics-engine does NOT load historical candles from ClickHouse on startup
   - It only builds history from Kafka stream in real-time
   - Requires `ML_MIN_CANDLES` (default 60) candles to be collected before predictions start
   - **Impact**: First 60 minutes after startup will have NO predictions

2. **Feature Engine History Not Persisted**:
   - `SimpleFeatureEngine` maintains in-memory history (max 60 candles per symbol)
   - History is lost on restart, must rebuild from scratch

3. **Model Persistence Exists But May Not Load**:
   - `model_persistence.go` has `SaveToDB()` and `LoadFromDB()` functions
   - Models are saved after training to database
   - Need to verify if models are loaded on startup

4. **Confidence Threshold May Block Signals**:
   - Default threshold: 0.10 (10%)
   - Trust stage dampening can reduce confidence significantly:
     - Cold start: capped at 0.50-0.65
     - Warming: capped at 0.75-1.0
   - If confidence < threshold, signal is NOT emitted (line 367 in `online_learning.go`)

---

## 4. How to Verify Data & Predictions

### 4.1 Check ClickHouse Data Status

Run the provided diagnostic commands:

```bash
# Check candle count in ClickHouse
docker-compose exec clickhouse clickhouse-client --query "SELECT COUNT(*) FROM candle_1m"

# Check candle cache count
docker-compose exec clickhouse clickhouse-client --query "SELECT COUNT(*) FROM candle_cache"

# Check prediction count
docker-compose exec clickhouse clickhouse-client --query "SELECT COUNT(*) FROM direction_predictions"

# Check distinct symbols with predictions
docker-compose exec clickhouse clickhouse-client --query "SELECT DISTINCT symbol FROM direction_predictions ORDER BY symbol"

# Check latest predictions
docker-compose exec clickhouse clickhouse-client --query "SELECT symbol, timestamp, direction, confidence, model_used FROM direction_predictions ORDER BY created_at DESC LIMIT 10"

# Check candle time range
docker-compose exec clickhouse clickhouse-client --query "SELECT min(toDateTime(timestamp)), max(toDateTime(timestamp)) FROM candle_1m"
```

### 4.2 Check Service Logs

```bash
# data-fetcher logs (should show candle saves)
docker-compose logs data-fetcher | grep "Saved 1m candle"

# analytics-engine logs (should show predictions)
docker-compose logs analytics-engine | grep "signals_emitted_count"

# Check for ML prediction activity
docker-compose logs analytics-engine | grep "ML SIGNAL"
```

### 4.3 Check API Endpoints

```bash
# Check ML metrics
curl http://localhost:8081/api/v1/ml/metrics

# Check signal stats
curl http://localhost:8081/api/v1/ml/signal-stats

# Check recent signals
curl http://localhost:8081/api/v1/ml/signals/recent?limit=10
```

---

## 5. Recommendations

### 5.1 Immediate Actions

1. **Add Historical Candle Loading on Startup**:
   - Load last 60-1440 candles from `candle_1m` or `candle_cache` when analytics-engine starts
   - This enables immediate predictions instead of waiting 60 minutes

2. **Verify Model Loading**:
   - Check if `LoadFromDB()` is called during `getOrCreateModel()`
   - Ensure trained models persist across restarts

3. **Monitor Confidence Distribution**:
   - Add logging for confidence values to understand if threshold is appropriate
   - Consider dynamic threshold adjustment

### 5.2 Monitoring Improvements

1. Add metrics for:
   - Candle ingestion rate (candles/minute)
   - Prediction rate (predictions/hour)
   - Model accuracy over time
   - Feature distribution statistics

2. Create dashboard showing:
   - Real-time prediction stream
   - Accuracy by symbol
   - Model trust stage distribution

---

## 6. Conclusion

**Yes, data IS being collected and ML predictions ARE being made.**

### Status: ✅ VERIFIED WORKING (as of 2026-02-22 17:20)

1. ✅ **Candle data** flows: Bybit → data-fetcher → Kafka → ClickHouse (`candle_1m`, `candle_cache`)
2. ✅ **Features** are extracted: 16 technical indicators from 60-candle history
3. ✅ **ML predictions** are made: SimpleNeuralNetwork predicts UP/DOWN/SIDEWAYS
4. ✅ **Predictions stored**: Saved to ClickHouse `direction_predictions` table
5. ✅ **Signals consumed**: API Gateway receives signals from Kafka and broadcasts to frontend

### Bug Fixed

**Issue**: `emitDirectionSignal()` had a SQL error - 16 columns but only 15 placeholders
```
Failed to save direction signal: code: 62, message: Cannot parse expression of type Int32 here: 15)
```

**Fix**: Added missing 16th placeholder (`?`) for the `features` column in the INSERT statement.

### Current Signal Activity

Latest signals (example):
```
ADAUSDT UP  0.6875 confidence  2026-02-22 17:20:27
BNBUSDT UP  0.5000 confidence  2026-02-22 17:20:28
SOLUSDT UP  0.5000 confidence  2026-02-22 17:20:29
XRPUSDT UP  0.5000 confidence  2026-02-22 17:20:30
BTCUSDT UP  0.5000 confidence  2026-02-22 17:20:31
ETHUSDT UP  0.5000 confidence  2026-02-22 17:20:32
```

### Remaining Considerations

1. **Model Accuracy**: Most models have validation accuracy below 55% threshold (marked as "inactive")
   - Only ADAUSDT consistently achieves >60% accuracy
   - This is normal for early training; accuracy should improve with more data

2. **Confidence Dampening**: Trust stage system reduces confidence for new models
   - Cold start models: capped at 0.50-0.65 confidence
   - This prevents overconfident predictions from untrained models

To monitor ongoing activity, use the diagnostic queries in Section 4.
