import React, { useEffect, useState } from 'react';
import './index.css';
import { MarketTable } from './components/MarketTable';
import { MarketPair, PumpSignal, TradingSignal } from './types';
import { ProgressBar } from './components/ProgressBar';
// ✅ React-toastify completely removed to prevent popup notifications
import { StatusDashboard } from './components/StatusDashboard';
import { LearningDashboard } from './components/LearningDashboard';
import { TradingSignalDashboard } from './components/TradingSignalDashboard';
import { MLDashboard } from './components/MLDashboard';
import { AdvancedTraderMind } from './components/AdvancedTraderMind';
import { ModelVotingPanel } from './components/ModelVotingPanel';
import { TraderMindFull } from './components/TraderMindFull';
import { SignalsHistory } from './components/SignalsHistory';
import { fetchWithFallback } from './utils/api';

const SIGNALS_STORAGE_KEY = 'tradingSignals';

// Risk calculation function that considers both confidence and volatility
function calculateRiskLevel(confidence: number, volatility: number): string {
  // Risk Score = (1 - confidence) * volatility * 100
  // Lower confidence + higher volatility = higher risk
  const riskScore = (1 - confidence) * volatility * 100;
  
  if (riskScore < 0.5) return 'LOW';
  if (riskScore < 1.5) return 'MEDIUM';
  return 'HIGH';
}

function toNumber(value: unknown, fallback = 0): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function toPrediction(direction: string): string {
  if (direction === 'UP') return 'BUY';
  if (direction === 'DOWN') return 'SELL';
  return 'NEUTRAL';
}

const App: React.FC = () => {
  const [marketData, setMarketData] = useState<MarketPair[]>([]);
  const [tradingSignals, setTradingSignals] = useState<TradingSignal[]>(() => {
    // Load signals from localStorage on initial render
    try {
      const saved = localStorage.getItem(SIGNALS_STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        // Filter out signals older than 24 hours
        const now = Date.now() / 1000;
        const twentyFourHoursAgo = now - (24 * 60 * 60);
        return parsed.filter((s: TradingSignal) => s.timestamp > twentyFourHoursAgo);
      }
    } catch (e) {
      console.error('Failed to load signals from localStorage:', e);
    }
    return [];
  });
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected'>('connecting');
  const [updatesCount, setUpdatesCount] = useState<number>(0);
  const [pumpSignalsCount, setPumpSignalsCount] = useState<number>(0);
  const [mlSignalsCount, setMlSignalsCount] = useState<number>(0);
  const [progress, setProgress] = useState<number>(0); // For overall system progress
  const [activeTab, setActiveTab] = useState<'market' | 'signals' | 'learning' | 'ml' | 'trader' | 'history'>('trader');
  // Infrastructure metrics state
  const [infrastructureMetrics, setInfrastructureMetrics] = useState<{
    kafka?: { messages_per_sec?: number; connection_status?: string };
    database?: { response_latency?: number; active_connections?: number; connection_status?: string };
  } | null>(null);
  const [anomalyDistribution, setAnomalyDistribution] = useState<{ high: number; medium: number; low: number }>({ high: 0, medium: 0, low: 0 });

  useEffect(() => {
    const loadRecentSignals = async () => {
      try {
        const payload = await fetchWithFallback<{ signals?: any[] }>('/api/v1/ml/signals/recent?limit=100&hours=24');
        const incoming = (payload?.signals || []).map((s) => {
          const confidence = toNumber(s.confidence, 0);
          const volatility = toNumber(s.volatility, 0.02);
          return {
            symbol: String(s.symbol || ''),
            prediction: toPrediction(String(s.direction || 'SIDEWAYS')),
            confidence,
            price_target: toNumber(s.price_target, 0),
            stop_loss: toNumber(s.stop_loss, 0),
            price_change_pct: s.price_target && s.current_price
              ? ((toNumber(s.price_target, 0) - toNumber(s.current_price, 0)) / Math.max(toNumber(s.current_price, 1), 1e-9)) * 100
              : 0,
            risk_level: calculateRiskLevel(confidence, volatility),
            model_used: String(s.model_used || 'SimpleNN'),
            time_horizon: `${toNumber(s.time_horizon, 60)}min`,
            timestamp: toNumber(s.timestamp, Date.now() / 1000),
            volatility,
            key_features: ['Technical Analysis', 'ML Prediction'],
            trust_stage: String(s.trust_stage || 'cold_start'),
            model_age_sec: toNumber(s.model_age_sec, 0),
            class_probs: s.class_probs || { down: 0, sideways: 1, up: 0 },
          } as TradingSignal;
        }).filter((s) => s.symbol);

        const bySymbol = new Map<string, TradingSignal>();
        for (const signal of incoming) {
          const existing = bySymbol.get(signal.symbol);
          if (!existing || signal.timestamp > existing.timestamp) {
            bySymbol.set(signal.symbol, signal);
          }
        }

        const deduped = Array.from(bySymbol.values())
          .sort((a, b) => b.timestamp - a.timestamp)
          .slice(0, 20);

        if (deduped.length > 0) {
          setTradingSignals(deduped);
        }
      } catch (e) {
        console.error('Failed to load recent signals from API:', e);
      }
    };

    loadRecentSignals();
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(SIGNALS_STORAGE_KEY, JSON.stringify(tradingSignals));
    } catch (e) {
      console.error('Failed to save signals to localStorage:', e);
    }
  }, [tradingSignals]);

  useEffect(() => {
    // Create WebSocket connection using the same logic as MLDashboard
    let wsUrl: string;
    if (window.location.host.includes('localhost:3000')) {
      // When running locally in development, use the full WebSocket URL
      wsUrl = `ws://${window.location.hostname}:8080/ws`;
    } else {
      // When running in Docker, use the relative path which will be proxied
      wsUrl = '/ws';
    }
    
    console.log(`Connecting to WebSocket at: ${wsUrl}`);
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log('WebSocket Connected');
      setConnectionStatus('connected');
      // ✅ Connection notification removed - status shown in header only
      setProgress(25); // Indicate initial connection
    };

    ws.onmessage = (event: MessageEvent) => {
      const message = JSON.parse(event.data);
      setUpdatesCount((prev: number) => prev + 1);

      if (message.type === 'initial_data') {
        setMarketData(message.data);
        setProgress(50); // Data loaded
      } else if (message.type === 'market_update') {
        setMarketData((prevData: MarketPair[]) => {
          const newDataMap = new Map<string, MarketPair>(message.data.map((item: MarketPair) => [item.symbol, item]));
          return prevData.map((pair: MarketPair) => newDataMap.get(pair.symbol) || pair);
        });
        setProgress(75); // Continuous updates
      } else if (message.type === 'pump_signal_update') {
        const signal: PumpSignal = message.data;
        console.log('Pump Signal:', signal);
        setPumpSignalsCount((prev: number) => prev + 1);
        // ✅ Popup removed - signals now shown in table only

        setMarketData((prevData: MarketPair[]) =>
          prevData.map((pair: MarketPair) =>
            pair.symbol === signal.symbol
              ? { ...pair, anomaly_score: signal.probability * 100, last_update: Date.now() / 1000 }
              : pair
          )
        );
        setProgress(100); // Anomaly detected
      } else if (message.type === 'trading_signal_update') {
        const signal: TradingSignal = message.data;
        console.log('Trading Signal:', signal);
        setMlSignalsCount((prev: number) => prev + 1);
        
        // ✅ Popup removed - signals now shown in table only
        
        // Update trading signals list
        setTradingSignals((prevSignals: TradingSignal[]) => {
          // Remove old signal for same symbol if exists
          const filteredSignals = prevSignals.filter(s => s.symbol !== signal.symbol);
          // Add new signal
          const newSignals = [signal, ...filteredSignals];
          // Keep only last 20 signals
          return newSignals.slice(0, 20);
        });
        setProgress(100); // ML prediction completed
      } else if (message.type === 'direction_signal') {
        // 🤖 Handle ML direction signals from analytics engine
        const dirSignal = message.data;
        console.log('🤖 Direction Signal:', dirSignal);
        setMlSignalsCount((prev: number) => prev + 1);
        
        // Convert direction signal to trading signal format
        const tradingSignal: TradingSignal = {
          symbol: dirSignal.symbol,
          prediction: dirSignal.direction === 'UP' ? 'BUY' : 
                     dirSignal.direction === 'DOWN' ? 'SELL' : 'NEUTRAL',
          confidence: dirSignal.confidence,
          price_target: dirSignal.price_target,
          stop_loss: dirSignal.stop_loss || 0,  // ✅ FIXED
          price_change_pct: dirSignal.price_target && dirSignal.current_price 
              ? ((dirSignal.price_target - dirSignal.current_price) / dirSignal.current_price) * 100 
              : 0,
          risk_level: calculateRiskLevel(dirSignal.confidence, dirSignal.volatility || 0.02),
          model_used: dirSignal.model_used || 'SimpleNN',
          time_horizon: `${dirSignal.time_horizon || 60}min`,
          timestamp: dirSignal.timestamp || Date.now() / 1000,
          volatility: dirSignal.volatility || 0.02,
          key_features: ['Technical Analysis', 'ML Prediction'],
          trust_stage: dirSignal.trust_stage || 'cold_start',
          model_age_sec: dirSignal.model_age_sec || 0,
          class_probs: dirSignal.class_probs || { down: 0, sideways: 1, up: 0 }
        };
        
        // ✅ Popup removed - signals now shown in table only
        
        // Update trading signals list
        setTradingSignals((prevSignals: TradingSignal[]) => {
          // Remove old signal for same symbol if exists
          const filteredSignals = prevSignals.filter(s => s.symbol !== dirSignal.symbol);
          // Add new signal
          const newSignals = [tradingSignal, ...filteredSignals];
          // Keep only last 20 signals
          return newSignals.slice(0, 20);
        });
        setProgress(100); // ML prediction completed
      }
    };

    ws.onclose = () => {
      console.log('WebSocket Disconnected');
      setConnectionStatus('disconnected');
      // ✅ Disconnection notification removed - status shown in header only
      setProgress(0);
    };

    ws.onerror = (error: Event) => {
      console.error('WebSocket Error:', error);
      setConnectionStatus('disconnected');
      // ✅ Error notification removed - status shown in header only
      setProgress(0);
    };

    return () => {
      ws.close();
    };
  }, []);

  // Load infrastructure metrics
  useEffect(() => {
    const loadInfrastructureMetrics = async () => {
      try {
        const data = await fetchWithFallback('/api/v1/infrastructure/metrics');
        setInfrastructureMetrics(data);
      } catch (e) {
        console.error('Failed to load infrastructure metrics:', e);
      }
    };

    loadInfrastructureMetrics();
    const interval = setInterval(loadInfrastructureMetrics, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  // Load anomaly distribution from ML metrics
  useEffect(() => {
    const loadAnomalyDistribution = async () => {
      try {
        // Load real ML metrics to get anomaly distribution
        const mlMetrics = await fetchWithFallback('/api/v1/ml/metrics');
        const signalStats = await fetchWithFallback('/api/v1/ml/signal-stats?symbol=BTCUSDT&hours=24');

        let high = 0, medium = 0, low = 0;

        // Use backend-provided distribution/rates (new contract).
        const btcMetrics = mlMetrics?.symbols?.BTCUSDT;
        if (btcMetrics?.class_distribution_24h) {
          const directionalRate = Number(btcMetrics.directional_rate_24h || 0);
          if (directionalRate < 0.25) high++;
          else if (directionalRate < 0.45) medium++;
          else low++;
        }

        // Add signal-based anomalies
        if (signalStats?.total_signals) {
          const totalSignals = signalStats.total_signals;
          const correctPredictions = signalStats.correct_predictions || 0;
          const accuracy = totalSignals > 0 ? correctPredictions / totalSignals : 0;

          if (accuracy < 0.6) high += Math.floor(totalSignals * 0.3);
          else if (accuracy < 0.8) medium += Math.floor(totalSignals * 0.2);
          else low += Math.floor(totalSignals * 0.1);
        }

        // Ensure at least some data is shown
        if (high === 0 && medium === 0 && low === 0) {
          low = 1; // At least show some low anomalies
        }

        setAnomalyDistribution({ high, medium, low });
      } catch (e) {
        console.error('Failed to load anomaly distribution:', e);
        // Fallback to basic calculation from trading signals
        const signals = tradingSignals;
        let high = 0, medium = 0, low = 0;

        signals.forEach(signal => {
          const risk = calculateRiskLevel(signal.confidence, signal.volatility || 0.02);
          if (risk === 'HIGH') high++;
          else if (risk === 'MEDIUM') medium++;
          else low++;
        });

        setAnomalyDistribution({ high: high || 1, medium: medium || 1, low: low || 1 });
      }
    };

    loadAnomalyDistribution();
    const interval = setInterval(loadAnomalyDistribution, 15000); // Refresh every 15 seconds
    return () => clearInterval(interval);
  }, [tradingSignals]);

  return (
    <div className="min-h-screen bg-gray-900 text-white p-4">
      {/* ✅ ToastContainer completely removed to eliminate all popup notifications */}
      <header className="flex justify-between items-center mb-6">
        <div className="flex items-baseline space-x-3">
          <h1 className="text-3xl font-bold">PredPump Radar</h1>
          <span className="text-sm text-gray-400">(build: {new Date().toLocaleString()})</span>
        </div>
        <div className="flex items-center space-x-4">
          <span className={`text-sm ${connectionStatus === 'connected' ? 'text-green-500' : 'text-red-500'}`}>
            WS Status: {connectionStatus}
          </span>
          <span className="text-sm text-gray-400">Updates: {updatesCount}</span>
          <span className="text-sm text-yellow-400">Pump Signals: {pumpSignalsCount}</span>
          <span className="text-sm text-blue-400">ML Signals: {mlSignalsCount}</span>
        </div>
      </header>

      <ProgressBar progress={progress} />

      {/* Tab Navigation */}
      <div className="flex space-x-4 mb-6 flex-wrap gap-2">
        <button
          onClick={() => setActiveTab('market')}
          className={`px-4 py-2 rounded font-medium ${
            activeTab === 'market'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
          }`}
        >
          📊 Market Monitor
        </button>
        <button
          onClick={() => setActiveTab('signals')}
          className={`px-4 py-2 rounded font-medium ${
            activeTab === 'signals'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
          }`}
        >
          🤖 ML Signals
        </button>
        <button
          onClick={() => setActiveTab('history')}
          className={`px-4 py-2 rounded font-medium ${
            activeTab === 'history'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
          }`}
        >
          📜 Signals History
        </button>
        <button
          onClick={() => setActiveTab('learning')}
          className={`px-4 py-2 rounded font-medium ${
            activeTab === 'learning'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
          }`}
        >
          🧠 Learning Dashboard
        </button>
        <button
          onClick={() => setActiveTab('ml')}
          className={`px-4 py-2 rounded font-medium ${
            activeTab === 'ml'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
          }`}
        >
          🤖 ML Metrics
        </button>
        <button
          onClick={() => setActiveTab('trader')}
          className={`px-4 py-2 rounded font-medium ${
            activeTab === 'trader'
              ? 'bg-green-600 text-white'
              : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
          }`}
        >
          🧠 Trader Mind
        </button>
      </div>

      {activeTab === 'market' && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <StatusDashboard
              wsStatus={connectionStatus}
              kafkaThroughput={
                infrastructureMetrics?.kafka?.messages_per_sec 
                  ? `${Math.round(infrastructureMetrics.kafka.messages_per_sec)} msg/s`
                  : "N/A"
              }
              dbStatus={
                infrastructureMetrics?.database?.connection_status 
                  ? infrastructureMetrics.database.connection_status === 'healthy' 
                    ? 'Connected' 
                    : 'Disconnected'
                  : "N/A"
              }
              anomalyDistribution={anomalyDistribution}
            />
            {/* TODO: Add detailed pair view and charts */}
          </div>

          <MarketTable data={marketData} />
          {/* TODO: Add detailed pair view and charts */}
        </>
      )}

      {activeTab === 'signals' && (
        <TradingSignalDashboard signals={tradingSignals} />
      )}

      {activeTab === 'history' && (
        <SignalsHistory initialLimit={100} initialHours={168} />
      )}

      {activeTab === 'learning' && (
        <LearningDashboard />
      )}

      {activeTab === 'trader' && (
        <>
          <AdvancedTraderMind />
          <div className="mt-6">
            <TraderMindFull />
          </div>
        </>
      )}
      {activeTab === 'ml' && (
        <MLDashboard />
      )}

      {/* Floating quick-open button for Trader Mind (visible even if nav is missing) */}
      <button
        onClick={() => setActiveTab('trader')}
        aria-label="Open Trader Mind"
        className="fixed right-4 bottom-4 z-50 px-4 py-2 bg-green-600 hover:bg-green-500 rounded-full shadow-lg text-white"
      >
        🧠 Trader Mind
      </button>
    </div>
  );
};

export default App;
