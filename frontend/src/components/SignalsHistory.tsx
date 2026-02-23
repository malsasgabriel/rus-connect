import React, { useEffect, useState } from 'react';
import { fetchWithFallback } from '../utils/api';

export interface HistoricalSignal {
  id?: number;
  symbol: string;
  direction: string;
  confidence: number;
  price_target: number;
  current_price: number;
  time_horizon: number;
  timestamp: string;
  stop_loss: number;
  volatility: number;
  trust_stage: string;
  model_age_sec: number;
  model_used: string;
  class_probs: { down: number; sideways: number; up: number };
  label_horizon_min: number;
  created_at: string;
  actual_direction?: string;
  actual_price?: number;
  accuracy_score?: number;
}

interface SignalsHistoryProps {
  initialLimit?: number;
  initialHours?: number;
}

export const SignalsHistory: React.FC<SignalsHistoryProps> = ({ 
  initialLimit = 100, 
  initialHours = 168 // 7 days
}) => {
  const [signals, setSignals] = useState<HistoricalSignal[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [limit, setLimit] = useState(initialLimit);
  const [hours, setHours] = useState(initialHours);
  const [symbolFilter, setSymbolFilter] = useState('');
  const [directionFilter, setDirectionFilter] = useState('ALL');
  const [totalSignals, setTotalSignals] = useState(0);

  const loadSignals = async () => {
    setLoading(true);
    setError(null);
    try {
      let url = `/api/v1/ml/signals/history?limit=${limit}&hours=${hours}`;
      if (symbolFilter) {
        url += `&symbol=${symbolFilter.toUpperCase()}`;
      }
      if (directionFilter !== 'ALL') {
        url += `&direction=${directionFilter}`;
      }
      
      const data = await fetchWithFallback<{ 
        signals: HistoricalSignal[]; 
        count: number;
        total?: number;
      }>(url);
      
      setSignals(data.signals || []);
      setTotalSignals(data.count || data.total || data.signals?.length || 0);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load signals');
      console.error('Failed to load signals history:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadSignals();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [limit, hours, symbolFilter, directionFilter]);

  const getSignalColor = (direction: string) => {
    switch (direction) {
      case 'UP':
        return 'text-green-400 bg-green-900/20 border border-green-500';
      case 'DOWN':
        return 'text-red-400 bg-red-900/20 border border-red-500';
      case 'SIDEWAYS':
        return 'text-gray-300 bg-gray-900/20 border border-gray-500';
      default:
        return 'text-gray-300 bg-gray-900/20 border border-gray-500';
    }
  };

  const getSignalIcon = (direction: string) => {
    switch (direction) {
      case 'UP':
        return '📈';
      case 'DOWN':
        return '📉';
      case 'SIDEWAYS':
        return '➡️';
      default:
        return '❓';
    }
  };

  const getTrustStageColor = (stage: string) => {
    switch (stage) {
      case 'trained':
        return 'bg-green-900 text-green-400';
      case 'warming':
        return 'bg-yellow-900 text-yellow-400';
      case 'cold_start':
      default:
        return 'bg-red-900 text-red-400';
    }
  };

  const getAccuracyColor = (accuracy: number) => {
    if (accuracy >= 0.7) return 'text-green-400';
    if (accuracy >= 0.5) return 'text-yellow-400';
    return 'text-red-400';
  };

  const uniqueSymbols = Array.from(new Set(signals.map(s => s.symbol))).sort();

  const filteredSignals = signals.filter(s => {
    if (directionFilter !== 'ALL' && s.direction !== directionFilter) return false;
    return true;
  });

  const stats = {
    total: filteredSignals.length,
    up: filteredSignals.filter(s => s.direction === 'UP').length,
    down: filteredSignals.filter(s => s.direction === 'DOWN').length,
    sideways: filteredSignals.filter(s => s.direction === 'SIDEWAYS').length,
    avgConfidence: filteredSignals.length > 0 
      ? (filteredSignals.reduce((sum, s) => sum + s.confidence, 0) / filteredSignals.length * 100).toFixed(1)
      : '0',
    withLabels: filteredSignals.filter(s => s.actual_direction).length,
    accuracy: (() => {
      const labeled = filteredSignals.filter(s => s.actual_direction && s.accuracy_score !== undefined);
      if (labeled.length === 0) return 'N/A';
      const avg = labeled.reduce((sum, s) => sum + (s.accuracy_score || 0), 0) / labeled.length;
      return (avg * 100).toFixed(1);
    })(),
  };

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <h2 className="text-xl font-bold mb-4 flex items-center">
        📜 Signals History
        <span className="ml-2 text-sm text-gray-400">
          ({stats.total} of {totalSignals} signals)
        </span>
      </h2>

      {/* Filters */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
        <div>
          <label className="block text-xs text-gray-400 mb-1">Symbol</label>
          <select
            value={symbolFilter}
            onChange={(e) => setSymbolFilter(e.target.value)}
            className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
          >
            <option value="">All Symbols</option>
            {uniqueSymbols.map(sym => (
              <option key={sym} value={sym}>{sym}</option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-xs text-gray-400 mb-1">Direction</label>
          <select
            value={directionFilter}
            onChange={(e) => setDirectionFilter(e.target.value)}
            className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
          >
            <option value="ALL">All Directions</option>
            <option value="UP">UP Only</option>
            <option value="DOWN">DOWN Only</option>
            <option value="SIDEWAYS">SIDEWAYS Only</option>
          </select>
        </div>

        <div>
          <label className="block text-xs text-gray-400 mb-1">Time Range</label>
          <select
            value={hours}
            onChange={(e) => setHours(Number(e.target.value))}
            className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
          >
            <option value={24}>Last 24 Hours</option>
            <option value={72}>Last 3 Days</option>
            <option value={168}>Last 7 Days</option>
            <option value={720}>Last 30 Days</option>
          </select>
        </div>

        <div>
          <label className="block text-xs text-gray-400 mb-1">Limit</label>
          <select
            value={limit}
            onChange={(e) => setLimit(Number(e.target.value))}
            className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
          >
            <option value={50}>50 signals</option>
            <option value={100}>100 signals</option>
            <option value={200}>200 signals</option>
            <option value={500}>500 signals</option>
          </select>
        </div>

        <div className="flex items-end">
          <button
            onClick={loadSignals}
            className="w-full bg-blue-600 hover:bg-blue-500 text-white font-medium py-2 px-4 rounded text-sm transition-colors"
          >
            🔄 Refresh
          </button>
        </div>
      </div>

      {/* Stats Summary */}
      <div className="grid grid-cols-2 md:grid-cols-7 gap-4 mb-6 p-4 bg-gray-900 rounded-lg">
        <div className="text-center">
          <div className="text-2xl font-bold text-white">{stats.total}</div>
          <div className="text-xs text-gray-400">Total</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-green-400">{stats.up}</div>
          <div className="text-xs text-gray-400">UP</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-red-400">{stats.down}</div>
          <div className="text-xs text-gray-400">DOWN</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-gray-400">{stats.sideways}</div>
          <div className="text-xs text-gray-400">SIDEWAYS</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-blue-400">{stats.avgConfidence}%</div>
          <div className="text-xs text-gray-400">Avg Confidence</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-purple-400">{stats.withLabels}</div>
          <div className="text-xs text-gray-400">Labeled</div>
        </div>
        <div className="text-center">
          <div className={`text-2xl font-bold ${getAccuracyColor(Number(stats.accuracy) / 100)}`}>
            {stats.accuracy}%
          </div>
          <div className="text-xs text-gray-400">Accuracy</div>
        </div>
      </div>

      {/* Loading State */}
      {loading && (
        <div className="text-center py-12 text-gray-400">
          <div className="text-4xl mb-2">⏳</div>
          <p>Loading signals history...</p>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="text-center py-12 text-red-400 bg-red-900/20 rounded-lg border border-red-500">
          <div className="text-4xl mb-2">❌</div>
          <p>{error}</p>
          <button
            onClick={loadSignals}
            className="mt-4 bg-red-600 hover:bg-red-500 text-white font-medium py-2 px-4 rounded text-sm transition-colors"
          >
            Retry
          </button>
        </div>
      )}

      {/* Signals Table */}
      {!loading && !error && filteredSignals.length === 0 && (
        <div className="text-center py-12 text-gray-400">
          <div className="text-4xl mb-2">🔍</div>
          <p>No signals found</p>
          <p className="text-sm">Try adjusting your filters</p>
        </div>
      )}

      {!loading && !error && filteredSignals.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead className="text-xs text-gray-400 uppercase bg-gray-700">
              <tr>
                <th className="px-3 py-3 whitespace-nowrap">Time</th>
                <th className="px-3 py-3 whitespace-nowrap">Symbol</th>
                <th className="px-3 py-3 whitespace-nowrap">Direction</th>
                <th className="px-3 py-3 whitespace-nowrap">Confidence</th>
                <th className="px-3 py-3 whitespace-nowrap">Current Price</th>
                <th className="px-3 py-3 whitespace-nowrap">Target Price</th>
                <th className="px-3 py-3 whitespace-nowrap">Stop Loss</th>
                <th className="px-3 py-3 whitespace-nowrap">Volatility</th>
                <th className="px-3 py-3 whitespace-nowrap">Model</th>
                <th className="px-3 py-3 whitespace-nowrap">Trust Stage</th>
                <th className="px-3 py-3 whitespace-nowrap">Horizon</th>
                <th className="px-3 py-3 whitespace-nowrap">Actual</th>
                <th className="px-3 py-3 whitespace-nowrap">Accuracy</th>
              </tr>
            </thead>
            <tbody>
              {filteredSignals.map((signal, index) => (
                <tr
                  key={`${signal.symbol}-${signal.timestamp}-${index}`}
                  className="border-b border-gray-700 hover:bg-gray-700/50 transition-colors"
                >
                  {/* Time */}
                  <td className="px-3 py-3 text-gray-300 whitespace-nowrap">
                    <div className="text-xs">
                      {new Date(signal.created_at).toLocaleDateString()}
                    </div>
                    <div className="text-xs text-gray-500">
                      {new Date(signal.created_at).toLocaleTimeString()}
                    </div>
                  </td>

                  {/* Symbol */}
                  <td className="px-3 py-3 font-bold text-white whitespace-nowrap">
                    {signal.symbol}
                  </td>

                  {/* Direction */}
                  <td className="px-3 py-3 whitespace-nowrap">
                    <div className="flex items-center">
                      <span className="mr-2">{getSignalIcon(signal.direction)}</span>
                      <span className={`font-semibold px-2 py-1 rounded text-xs ${getSignalColor(signal.direction)}`}>
                        {signal.direction}
                      </span>
                    </div>
                  </td>

                  {/* Confidence */}
                  <td className="px-3 py-3 whitespace-nowrap">
                    <div className="flex items-center">
                      <div className="w-16 bg-gray-600 rounded-full h-2 mr-2">
                        <div
                          className={`h-2 rounded-full ${
                            signal.confidence > 0.7 ? 'bg-green-500' :
                            signal.confidence > 0.5 ? 'bg-yellow-500' : 'bg-red-500'
                          }`}
                          style={{ width: `${signal.confidence * 100}%` }}
                        ></div>
                      </div>
                      <span className="text-white font-semibold text-xs">
                        {(signal.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  </td>

                  {/* Current Price */}
                  <td className="px-3 py-3 text-gray-300 font-mono text-xs whitespace-nowrap">
                    ${signal.current_price?.toFixed(6) || 'N/A'}
                  </td>

                  {/* Target Price */}
                  <td className="px-3 py-3 text-blue-400 font-mono text-xs whitespace-nowrap">
                    ${signal.price_target?.toFixed(6) || 'N/A'}
                  </td>

                  {/* Stop Loss */}
                  <td className="px-3 py-3 text-red-400 font-mono text-xs whitespace-nowrap">
                    ${signal.stop_loss?.toFixed(6) || 'N/A'}
                  </td>

                  {/* Volatility */}
                  <td className="px-3 py-3 text-orange-400 font-mono text-xs whitespace-nowrap">
                    {(signal.volatility * 100).toFixed(2)}%
                  </td>

                  {/* Model */}
                  <td className="px-3 py-3 text-purple-400 text-xs whitespace-nowrap">
                    {signal.model_used || 'SimpleNN'}
                  </td>

                  {/* Trust Stage */}
                  <td className="px-3 py-3 whitespace-nowrap">
                    <span className={`px-2 py-1 rounded text-xs font-semibold ${getTrustStageColor(signal.trust_stage)}`}>
                      {signal.trust_stage || 'cold_start'}
                    </span>
                  </td>

                  {/* Horizon */}
                  <td className="px-3 py-3 text-blue-400 text-xs whitespace-nowrap">
                    {signal.label_horizon_min || signal.time_horizon || 60}min
                  </td>

                  {/* Actual Direction */}
                  <td className="px-3 py-3 whitespace-nowrap">
                    {signal.actual_direction ? (
                      <span className={`font-semibold text-xs ${
                        signal.actual_direction === signal.direction 
                          ? 'text-green-400' 
                          : 'text-red-400'
                      }`}>
                        {getSignalIcon(signal.actual_direction)} {signal.actual_direction}
                      </span>
                    ) : (
                      <span className="text-gray-500 text-xs">Pending</span>
                    )}
                  </td>

                  {/* Accuracy */}
                  <td className="px-3 py-3 whitespace-nowrap">
                    {signal.accuracy_score !== undefined ? (
                      <span className={`font-semibold text-xs ${getAccuracyColor(signal.accuracy_score)}`}>
                        {(signal.accuracy_score * 100).toFixed(0)}%
                      </span>
                    ) : (
                      <span className="text-gray-500 text-xs">-</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Class Probabilities Distribution (for most recent signal per symbol) */}
      {filteredSignals.length > 0 && (
        <div className="mt-6 pt-4 border-t border-gray-700">
          <h3 className="text-sm font-semibold text-gray-400 mb-3">Latest Class Probabilities by Symbol</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            {Array.from(new Set(filteredSignals.map(s => s.symbol))).slice(0, 6).map(symbol => {
              const latestSignal = filteredSignals.find(s => s.symbol === symbol);
              if (!latestSignal?.class_probs) return null;
              return (
                <div key={symbol} className="bg-gray-900 rounded-lg p-3">
                  <div className="font-bold text-white text-sm mb-2">{symbol}</div>
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="text-green-400">UP:</span>
                      <span className="text-white">{(latestSignal.class_probs.up * 100).toFixed(0)}%</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-400">SIDE:</span>
                      <span className="text-white">{(latestSignal.class_probs.sideways * 100).toFixed(0)}%</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-red-400">DOWN:</span>
                      <span className="text-white">{(latestSignal.class_probs.down * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};
