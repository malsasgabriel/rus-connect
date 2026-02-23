# ML Signal Confidence Fix

## Problem
All ML trading signals were showing **99.9-100% confidence** even for **cold_start** models, which is unrealistic and unreliable.

### Root Cause
The neural network was outputting raw softmax probabilities directly as confidence without considering:
1. **Model trust stage** (cold_start vs warming vs trained)
2. **Model maturity** (number of predictions made)
3. **Overconfidence** in untrained models

## Solution

### 1. Added Confidence Dampening Function
Created `ApplyTrustStageDampening()` in `simple_neural_network.go`:

```go
func ApplyTrustStageDampening(rawConfidence float64, trustStage string, predictionCount int) float64
```

**Dampening rules by trust stage:**

| Trust Stage | Prediction Count | Confidence Cap | Floor |
|-------------|------------------|----------------|-------|
| **cold_start** | 0-50 | 0.50 → 0.65 | 0.30 → 0.50 |
| **warming** | 100-500 | 0.75 → 1.0 | - |
| **trained** | 500+ | 0.95 (max) | - |

### 2. Updated Signal Generation
Modified `online_learning.go` to:
1. Get raw confidence from model prediction
2. Calculate trust stage based on model state
3. Apply dampening based on trust stage and prediction count
4. Use dampened confidence for signal emission threshold

### 3. Added Prediction Count Tracking
- Added `prediction_count` field to `DirectionSignal` struct
- Added database column to track model maturity
- Updated signal emission to record prediction count

## Results

### Before Fix
```
TIME     SYMBOL   SIGNAL  CONFIDENCE  TRUST_STAGE
18:37:00 BTCUSDT  SELL    100.0%      cold_start
18:37:00 ETHUSDT  BUY     100.0%      cold_start
18:37:00 XRPUSDT  SELL    99.9%       cold_start
```

### After Fix
```
TIME     SYMBOL   SIGNAL  CONFIDENCE  TRUST_STAGE  PRED_COUNT
16:52:36 ETHUSDT  UP      46.9%       cold_start   0
16:52:35 SOLUSDT  UP      50.0%       cold_start   0
16:52:34 BNBUSDT  UP      48.5%       cold_start   0
16:52:32 XRPUSDT  UP      50.0%       cold_start   0
16:52:31 BTCUSDT  UP      50.0%       cold_start   0
16:52:30 ADAUSDT  DOWN    68.8%       warming      0
```

## Files Changed

1. **analytics-engine/simple_neural_network.go**
   - Added `ApplyTrustStageDampening()` function
   - Implements confidence dampening logic by trust stage

2. **analytics-engine/online_learning.go**
   - Updated signal generation to apply dampening
   - Added prediction count to signal

3. **analytics-engine/types.go**
   - Added `PredictionCount` field to `DirectionSignal`

4. **analytics-engine/main.go**
   - Added `prediction_count` database column
   - Updated signal insertion to include prediction count

## Confidence Behavior

### Cold Start Models (0-50 predictions)
- **Max confidence**: 50% → 65% (scales with prediction count)
- **Min confidence**: 30% → 50% (scales with prediction count)
- **Rationale**: Untrained models should not be trusted with high confidence

### Warming Models (100-500 predictions)
- **Max confidence**: 75% → 100% (scales with maturity)
- **Rationale**: Models with some history but not fully trained

### Trained Models (500+ predictions)
- **Max confidence**: 95% (hard cap to prevent overconfidence)
- **Rationale**: Even well-trained models can be wrong

## Verification Commands

```bash
# Check recent signals with confidence
docker exec rus-connect-clickhouse-1 clickhouse-client --query \
  "SELECT symbol, direction, round(confidence, 3) as conf, trust_stage, prediction_count, created_at \
   FROM direction_predictions ORDER BY created_at DESC LIMIT 10"

# Check ML metrics API
curl http://localhost:8081/api/v1/ml/metrics | python -c \
  "import sys,json; d=json.load(sys.stdin); \
   [print(f'{k}: conf={v[\"avg_confidence\"]:.3f}') for k,v in d['symbols'].items()]"

# Check model trust stage distribution
docker exec rus-connect-clickhouse-1 clickhouse-client --query \
  "SELECT trust_stage, count(), avg(confidence) as avg_conf \
   FROM direction_predictions GROUP BY trust_stage ORDER BY trust_stage"
```

## Next Steps for Further Improvement

1. **Implement confidence calibration** based on actual prediction accuracy
2. **Add Brier score** tracking for probability calibration
3. **Implement isotonic regression** for confidence calibration
4. **Add ensemble methods** to reduce variance in predictions
5. **Track confidence drift** over time to detect model degradation
