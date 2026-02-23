
export interface MarketPair {
  symbol: string;
  price: number;
  volume: number;
  anomaly_score: number;
  last_update: number;
}

export interface PumpSignal {
  symbol: string;
  probability: number;
  timestamp: number;
  alert: string;
}

export interface TradingSignal {
  symbol: string;
  timestamp: number;
  prediction: string;        // "STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL"
  confidence: number;        // 0.0 - 1.0
  price_target: number;      // Target price in 1 hour
  stop_loss: number;         // Stop loss level
  time_horizon: string;      // "1H"
  model_used: string;        // "LSTM", "XGBoost", "Ensemble"
  key_features: string[];    // Key decision factors
  risk_level: string;        // "LOW", "MEDIUM", "HIGH"
  price_change_pct: number;  // Expected % change
  volatility: number;        // Market volatility
  trust_stage?: string;      // "cold_start", "warming", "trained"
  model_age_sec?: number;    // Seconds since model update
  class_probs?: {            // Per-class probabilities
    down: number;
    sideways: number;
    up: number;
  };
}

// Define other types (OrderBook, Trade) if needed for detailed views later.

export interface CandleData {
  time: string | number; // ISO string or Unix timestamp
  open: number;
  high: number;
  low: number;
  close: number;
}
