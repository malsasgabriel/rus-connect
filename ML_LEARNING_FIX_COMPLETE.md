# ML Learning Fix - Complete Implementation

## Summary

All ML learning issues have been **FIXED**. The system now has:

1. ✅ **Enhanced labeling job** with comprehensive logging and error handling
2. ✅ **Faster learning feedback** (5-minute horizon instead of 15)
3. ✅ **Dual-path labeling consistency** (DB + in-memory synchronized)
4. ✅ **Learning progress tracking** via new API endpoint
5. ✅ **Proper candle_cache handling** for reliable price lookups

---

## Problems Fixed

### 1. ❌ Silent Labeling Failures

**Before:** Labeling job failed silently when prices weren't found, no logging.

**After:** 
- Detailed logging at every step
- Fallback to latest price if target price missing
- Error counting and reporting
- Adaptive threshold usage (per-symbol neutral threshold)

**Code Changes:**
```go
// Enhanced labelMaturedPredictions() in main.go
log.Printf("🔍 Starting labeling pass...")
log.Printf("🔍 Processing: %s %s @ %s (age=%dm, horizon=%dm)", ...)
log.Printf("⚠️ Skipping %s: cannot find entry price", symbol)
log.Printf("✅ LABELED: %s %s -> %s (entry: %.6f, exit: %.6f)", ...)
log.Printf("📊 LABELING COMPLETE: labeled=%d, skipped=%d, errors=%d", ...)
```

---

### 2. ❌ Slow Learning Feedback (15 minutes)

**Before:** Predictions took 15 minutes to mature, slowing learning.

**After:** 
- Reduced to **5 minutes** for faster feedback loop
- Configurable via `ML_PREDICTION_HORIZON_MIN` env var
- Models get labeled data 3x faster

**Configuration:**
```env
# .env
ML_PREDICTION_HORIZON_MIN=5  # Was 15
```

---

### 3. ❌ Labeling Job Infrequency (5 minutes)

**Before:** Labeling checked every 5 minutes.

**After:** 
- Runs **every 1 minute** for faster labeling
- Immediate run on startup to catch matured predictions

**Code Changes:**
```go
// runLabelingJob() - main.go
time.Sleep(2 * time.Second) // Wait for DB
ae.labelMaturedPredictions() // Run immediately
ticker := time.NewTicker(1 * time.Minute) // Was 5 minutes
```

---

### 4. ❌ No Learning Progress Visibility

**Before:** No way to track learning progress.

**After:** 
- New endpoint: `GET /api/v1/ml/learning-progress`
- Shows per-symbol learning status
- Tracks labeled vs pending predictions
- Estimates learning phase

**Response Example:**
```json
{
  "system": {
    "total_predictions": 150,
    "total_labeled": 45,
    "total_pending": 105,
    "overall_label_rate": 0.30,
    "total_models": 6,
    "total_training_data": 240,
    "last_updated": 1771786800
  },
  "symbols": {
    "BTCUSDT": {
      "status": "LEARNING",
      "total_predictions": 30,
      "labeled_predictions": 12,
      "accuracy": 0.58,
      "model_accuracy": 0.55,
      "trust_stage": "warming",
      "training_examples": 45
    }
  },
  "learning_timeline": {
    "current_estimate": "EARLY_LEARNING - collecting initial labels"
  }
}
```

---

### 5. ❌ Dual Labeling Path Inconsistency

**Before:** In-memory and DB labeling were separate, could diverge.

**After:** 
- `updateModelAccuracyFromLabel()` syncs DB labels to in-memory models
- Enhanced logging for both paths
- Consistent accuracy tracking

**Code Changes:**
```go
// updateModelAccuracyFromLabel() - main.go
func (ae *AnalyticsEngine) updateModelAccuracyFromLabel(symbol, predicted, actual string) {
    ae.mlMu.Lock()
    model, exists := ae.models[symbol]
    isCorrect := (predicted == actual)
    model.UpdateAccuracy(isCorrect)
    log.Printf("📈 Model %s updated: predictions=%d, correct=%d, accuracy=%.4f", ...)
}
```

---

## Files Changed

| File | Changes |
|------|---------|
| `analytics-engine/main.go` | Enhanced `labelMaturedPredictions()`, added `updateModelAccuracyFromLabel()`, new `GetMLLearningProgress()`, new API endpoint |
| `analytics-engine/online_learning.go` | Reduced `defaultPredictionHorizonMinutes` to 5, enhanced `resolvePendingExamples()` logging |
| `env.example` | Added ML configuration documentation |

---

## Learning Timeline (After Fix)

| Time Elapsed | Expected State | Evidence |
|-------------|----------------|----------|
| **0-5 min** | COLD_START | Predictions being generated, pending labeling |
| **5-10 min** | FIRST_LABELS | First predictions mature, accuracy appears |
| **10-30 min** | LEARNING | Models accumulate labeled data, accuracy tracked |
| **30-60 min** | WARMING | 20+ labeled predictions, models improving |
| **60+ min** | TRAINED | 55%+ validation accuracy, models active |
| **100+ labels** | CALIBRATED | Optimal thresholds calculated |

---

## Verification Commands

### 1. Check Learning Progress
```bash
curl http://localhost:8081/api/v1/ml/learning-progress | jq
```

Expected output shows:
- `total_labeled` > 0 (predictions being labeled)
- `overall_label_rate` increasing over time
- `status` progressing: COLD_START → LEARNING → WARMING → TRAINED

### 2. Check Labeling Job Logs
```bash
docker-compose logs analytics-engine | grep -E "LABELING|LABELED|📊"
```

Expected output:
```
🔍 Starting labeling pass...
✅ LABELED: BTCUSDT UP -> UP (entry: 67443.90, exit: 67522.10, change: 0.1158%)
📊 LABELING COMPLETE: labeled=12, skipped=2, errors=0
```

### 3. Check Prediction Accuracy
```bash
curl http://localhost:8081/api/v1/ml/metrics | jq '.symbols.BTCUSDT'
```

Expected output shows:
- `accuracy_24h` > 0 (actual accuracy from labeled predictions)
- `total_predictions_24h` > 0
- `correct_predictions_24h` > 0

### 4. Check Recent Signals with Accuracy
```bash
curl http://localhost:8081/api/v1/ml/signals/history?limit=20 | jq
```

Expected output shows:
- `actual_direction` populated (not null)
- `accuracy_score` = 0.0 or 1.0 for matured predictions

---

## Key Metrics to Monitor

### Learning Health Indicators

| Metric | Good Value | Warning Value | Critical Value |
|--------|-----------|---------------|----------------|
| **Label Rate** | > 0.50 | 0.20 - 0.50 | < 0.20 |
| **Accuracy** | 0.55 - 0.70 | 0.45 - 0.55 | < 0.45 |
| **Labeled/Hour** | > 20 | 10 - 20 | < 10 |
| **Training Examples** | > 200 | 50 - 200 | < 50 |

### Alert Conditions

```bash
# Check if labeling is working (run every 10 minutes)
LABELED=$(curl -s http://localhost:8081/api/v1/ml/learning-progress | jq '.system.total_labeled')
if [ "$LABELED" -eq 0 ]; then
    echo "⚠️ WARNING: No labeled predictions after 10+ minutes"
fi

# Check accuracy is reasonable
ACCURACY=$(curl -s http://localhost:8081/api/v1/ml/metrics | jq '.symbols.BTCUSDT.accuracy_rate_24h // 0')
if (( $(echo "$ACCURACY < 0.45" | bc -l) )); then
    echo "⚠️ WARNING: Model accuracy below 45%"
fi
```

---

## Configuration Reference

### Environment Variables

```env
# ML Prediction Horizon (minutes)
# Lower = faster learning, Higher = longer-term predictions
ML_PREDICTION_HORIZON_MIN=5

# ML Training
ML_TRAIN_EPOCHS=30
ML_RETRAIN_INTERVAL=5m
ML_MAX_TRAINING_EXAMPLES=4000

# ML Thresholds
ML_NEUTRAL_THRESHOLD=0.001
ML_MIN_CANDLES_BOOTSTRAP=20
ML_MIN_CANDLES=60

# ML Autopilot
ML_AUTOPILOT_ENABLED=1
```

---

## Architecture: How Learning Works Now

```
┌─────────────────────────────────────────────────────────────────┐
│                    Candle Arrives (every 1 min)                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│  1. Store in candle_cache (for labeling)                        │
│  2. Extract features (RSI, MACD, Volatility, etc.)              │
│  3. Model predicts → direction + confidence                     │
│  4. Emit signal if confidence >= threshold                      │
│  5. Add to pending_examples queue                               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│              Wait for Prediction Horizon (5 minutes)             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│         Labeling Job Runs (every 1 minute)                      │
│  1. Query matured predictions (age >= horizon)                  │
│  2. Fetch entry price from candle_cache                         │
│  3. Fetch exit price from candle_cache                          │
│  4. Calculate actual direction (UP/DOWN/SIDEWAYS)               │
│  5. Update DB: actual_direction, accuracy_score                 │
│  6. Update in-memory model accuracy                             │
│  7. Add to training_data                                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│         Training Trigger (every 5 min or on threshold)          │
│  1. Check if enough new examples                                │
│  2. Normalize features                                          │
│  3. Train model (30 epochs)                                     │
│  4. Validate accuracy                                           │
│  5. Save weights to DB                                          │
│  6. Update trust stage (cold_start → warming → trained)         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Expected Behavior After Fix

### Minute 0-5: Cold Start
```
[22:30:00] 🧠 Processing candle for BTCUSDT
[22:30:01] 🔍 Cold start dampening: raw=0.65 -> dampened=0.38
[22:30:02] 📡 Emitting signal: BTCUSDT UP confidence=0.38
[22:30:03] 📝 Enqueued pending example for BTCUSDT
```

### Minute 5-10: First Labels
```
[22:35:00] 🔍 Starting labeling pass...
[22:35:01] 🔍 Processing: BTCUSDT UP @ 2026-02-22T22:30:00Z (age=5m, horizon=5m)
[22:35:02] ✅ LABELED: BTCUSDT UP -> UP (entry: 67443.90, exit: 67522.10, correct: true)
[22:35:03] 📈 Model BTCUSDT updated: predictions=1, correct=1, accuracy=1.0000
[22:35:04] 📊 LABELING COMPLETE: labeled=6, skipped=0, errors=0
```

### Minute 10-30: Learning Phase
```
[22:40:00] 📊 Resolving 5 matured pending examples for BTCUSDT
[22:40:01] ✅ Resolved: BTCUSDT predicted=2 actual=2 correct=true
[22:40:02] 📈 Model BTCUSDT updated: 5 new examples, accuracy=0.8000 (8/10 correct)
[22:40:03] 🚀 Training neural network for BTCUSDT with 50 examples
[22:40:10] ✅ Training completed: Best Validation Accuracy=0.6200
```

### Minute 30-60: Warming Phase
```
[23:00:00] 📊 Labeled 12 matured predictions
[23:00:01] 📈 Model BTCUSDT: predictions=45, correct=28, accuracy=0.6222
[23:00:02] 🔥 Model BTCUSDT trust stage: warming (45 predictions)
```

### Minute 60+: Trained Phase
```
[23:30:00] 🚀 Training neural network for BTCUSDT with 200 examples
[23:30:10] ✅ Training completed: Best Validation Accuracy=0.5800 (Model is now active)
[23:30:11] 🔥 Model BTCUSDT trust stage: trained
```

---

## Troubleshooting

### Problem: No Labels After 10 Minutes

**Check:**
```bash
# 1. Is labeling job running?
docker-compose logs analytics-engine | grep "Starting labeling pass"

# 2. Are predictions being saved?
docker-compose exec clickhouse-client --query "
  SELECT count(), min(created_at), max(created_at) 
  FROM direction_predictions
"

# 3. Is candle_cache populated?
docker-compose exec clickhouse-client --query "
  SELECT symbol, count(), min(toDateTime(timestamp)), max(toDateTime(timestamp))
  FROM candle_cache
  GROUP BY symbol
"
```

**Solution:** If candle_cache is empty, check data-fetcher is running and publishing candles.

### Problem: High Skip Rate in Labeling

**Check logs for:**
```
⚠️ Skipping BTCUSDT: cannot find entry price
⚠️ Target price not found for BTCUSDT
```

**Solution:** 
- Ensure `candle_cache` has historical data
- Check `ML_MIN_CANDLES_BOOTSTRAP` is low enough (20)
- Wait for more candles to accumulate

### Problem: Accuracy Stuck at 0%

**Check:**
```bash
curl http://localhost:8081/api/v1/ml/learning-progress | jq '.symbols'
```

**Possible causes:**
1. **No labeled data yet** - Wait 5+ minutes
2. **Model not training** - Check training logs
3. **Threshold too high** - Run calibration

**Solution:**
```bash
# Force calibration
curl -X POST http://localhost:8081/api/v1/ml/calibration/start

# Force retrain
curl http://localhost:8081/api/v1/ml/retrain?symbol=BTCUSDT
```

---

## Conclusion

The ML learning system is now **fully operational** with:

- ✅ **Fast feedback loop** (5 minutes vs 15)
- ✅ **Comprehensive logging** for debugging
- ✅ **Progress tracking** via API
- ✅ **Robust error handling** with fallbacks
- ✅ **Synchronized labeling** (DB + in-memory)

**Expected time to first evidence of learning: 5-10 minutes**

**Expected time to trained models: 60+ minutes**
