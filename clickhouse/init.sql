CREATE TABLE IF NOT EXISTS candle_1m
(
    symbol String,
    timestamp Int64,
    open Float64,
    high Float64,
    low Float64,
    close Float64,
    volume Float64,
    created_at DateTime DEFAULT now()
)
ENGINE = ReplacingMergeTree(created_at)
PARTITION BY toYYYYMM(toDateTime(timestamp))
ORDER BY (symbol, timestamp);

CREATE TABLE IF NOT EXISTS candle_cache
(
    symbol String,
    timestamp Int64,
    open Float64,
    high Float64,
    low Float64,
    close Float64,
    volume Float64,
    created_at DateTime DEFAULT now()
)
ENGINE = ReplacingMergeTree(created_at)
PARTITION BY toYYYYMM(toDateTime(timestamp))
ORDER BY (symbol, timestamp);

CREATE TABLE IF NOT EXISTS direction_predictions
(
    symbol String,
    timestamp DateTime,
    direction String,
    confidence Float64,
    price_target Float64,
    current_price Float64,
    time_horizon Int32,
    features String,
    actual_direction String,
    actual_price Float64,
    accuracy_score Float64,
    created_at DateTime DEFAULT now()
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(timestamp)
ORDER BY (symbol, timestamp, created_at);
