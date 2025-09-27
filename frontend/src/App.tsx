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

const API_GATEWAY_WS = import.meta.env.VITE_API_GATEWAY_WS || 'ws://localhost:8080/ws';

const App: React.FC = () => {
  const [marketData, setMarketData] = useState<MarketPair[]>([]);
  const [tradingSignals, setTradingSignals] = useState<TradingSignal[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected'>('connecting');
  const [updatesCount, setUpdatesCount] = useState<number>(0);
  const [pumpSignalsCount, setPumpSignalsCount] = useState<number>(0);
  const [mlSignalsCount, setMlSignalsCount] = useState<number>(0);
  const [progress, setProgress] = useState<number>(0); // For overall system progress
  const [activeTab, setActiveTab] = useState<'market' | 'learning' | 'signals' | 'ml' | 'trader'>('trader');
  // add trader mind tab state
  // ...existing code...

  useEffect(() => {
    const ws = new WebSocket(API_GATEWAY_WS);

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
          price_target: dirSignal.price_target || dirSignal.priceTarget,
          stop_loss: dirSignal.current_price || dirSignal.currentPrice || 0,
          price_change_pct: 0, // Calculate based on target vs current
          risk_level: dirSignal.confidence > 0.7 ? 'LOW' : 
                     dirSignal.confidence > 0.5 ? 'MEDIUM' : 'HIGH',
          model_used: dirSignal.model_used || 'unknown',
          time_horizon: `${dirSignal.time_horizon || 60}min`,
          timestamp: dirSignal.timestamp || Date.now() / 1000,
          volatility: 0.02, // Default volatility
          key_features: ['Technical Analysis', 'ML Prediction']
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
      <div className="flex space-x-4 mb-6">
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
              kafkaThroughput="N/A" // Mocked for now
              dbStatus="N/A" // Mocked for now
              anomalyDistribution={{ high: 0, medium: 0, low: 0 }} // Mocked for now
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