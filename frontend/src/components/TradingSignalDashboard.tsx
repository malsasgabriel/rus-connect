import React from 'react';
import { TradingSignal } from '../types';

interface TradingSignalDashboardProps {
  signals: TradingSignal[];
}

export const TradingSignalDashboard: React.FC<TradingSignalDashboardProps> = ({ signals }) => {
  const getSignalColor = (prediction: string) => {
    switch (prediction) {
      case 'BUY':
        return 'text-green-400 bg-green-900/20 border border-green-500';
      case 'NEUTRAL':
        return 'text-gray-300 bg-gray-900/20 border border-gray-500';
      case 'SELL':
        return 'text-red-400 bg-red-900/20 border border-red-500';
      default:
        return 'text-gray-300 bg-gray-900/20 border border-gray-500';
    }
  };

  const getSignalIcon = (prediction: string) => {
    switch (prediction) {
      case 'BUY':
        return '📈';
      case 'NEUTRAL':
        return '➡️';
      case 'SELL':
        return '📉';
      default:
        return '❓';
    }
  };

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'LOW':
        return 'text-green-400';
      case 'MEDIUM':
        return 'text-yellow-400';
      case 'HIGH':
        return 'text-red-400';
      default:
        return 'text-gray-400';
    }
  };

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <h2 className="text-xl font-bold mb-4 flex items-center">
        🤖 AI Trading Signals
        <span className="ml-2 text-sm text-gray-400">({signals.length} active)</span>
      </h2>
      
      {signals.length === 0 ? (
        <div className="text-center py-8 text-gray-400">
          <div className="text-4xl mb-2">🔍</div>
          <p>No trading signals available</p>
          <p className="text-sm">Waiting for ML predictions...</p>
        </div>
      ) : (
        <>
          {/* Signals Table */}
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left">
              <thead className="text-xs text-gray-400 uppercase bg-gray-700">
                <tr>
                  <th className="px-4 py-3">Time</th>
                  <th className="px-4 py-3">Symbol</th>
                  <th className="px-4 py-3">Signal</th>
                  <th className="px-4 py-3">Confidence</th>
                  <th className="px-4 py-3">Target Price</th>
                  <th className="px-4 py-3">Stop Loss</th>
                  <th className="px-4 py-3">Risk</th>
                  <th className="px-4 py-3">Model</th>
                  <th className="px-4 py-3">Trust</th>
                  <th className="px-4 py-3">Horizon</th>
                </tr>
              </thead>
              <tbody>
                {signals.map((signal, index) => (
                  <tr
                    key={`${signal.symbol}-${signal.timestamp}-${index}`}
                    className="border-b border-gray-700 hover:bg-gray-700/50"
                  >
                    {/* Time */}
                    <td className="px-4 py-3 text-gray-300">
                      {new Date(signal.timestamp * 1000).toLocaleTimeString()}
                    </td>
                    
                    {/* Symbol */}
                    <td className="px-4 py-3 font-bold text-white">
                      {signal.symbol}
                    </td>
                    
                    {/* Signal */}
                    <td className="px-4 py-3">
                      <div className="flex items-center">
                        <span className="mr-2">{getSignalIcon(signal.prediction)}</span>
                        <span className={`font-semibold ${
                          signal.prediction === 'BUY' ? 'text-green-400' :
                          signal.prediction === 'SELL' ? 'text-red-400' : 'text-gray-400'
                        }`}>
                          {signal.prediction}
                        </span>
                      </div>
                    </td>
                    
                    {/* Confidence */}
                    <td className="px-4 py-3">
                      <div className="flex items-center">
                        <div className="w-12 bg-gray-600 rounded-full h-2 mr-2">
                          <div
                            className={`h-2 rounded-full ${
                              signal.confidence > 0.7 ? 'bg-green-500' :
                              signal.confidence > 0.5 ? 'bg-yellow-500' : 'bg-red-500'
                            }`}
                            style={{ width: `${signal.confidence * 100}%` }}
                          ></div>
                        </div>
                        <span className="text-white font-semibold">
                          {(signal.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </td>
                    
                    {/* Target Price */}
                    <td className="px-4 py-3 text-blue-400 font-semibold">
                      ${signal.price_target ? signal.price_target.toFixed(6) : 'N/A'}
                    </td>
                    
                    {/* Stop Loss */}
                    <td className="px-4 py-3 text-red-400 font-semibold">
                      ${signal.stop_loss ? signal.stop_loss.toFixed(6) : 'N/A'}
                    </td>
                    
                    {/* Risk */}
                    <td className="px-4 py-3">
                      <span className={`px-2 py-1 rounded text-xs font-semibold ${
                        signal.risk_level === 'LOW' ? 'bg-green-900 text-green-400' :
                        signal.risk_level === 'MEDIUM' ? 'bg-yellow-900 text-yellow-400' :
                        'bg-red-900 text-red-400'
                      }`}>
                        {signal.risk_level}
                      </span>
                    </td>
                    
                    {/* Model */}
                    <td className="px-4 py-3 text-purple-400">
                      {signal.model_used}
                    </td>

                    {/* Trust */}
                    <td className="px-4 py-3">
                      <span className={`px-2 py-1 rounded text-xs font-semibold ${
                        signal.trust_stage === 'trained' ? 'bg-green-900 text-green-400' :
                        signal.trust_stage === 'warming' ? 'bg-yellow-900 text-yellow-400' :
                        'bg-red-900 text-red-400'
                      }`}>
                        {signal.trust_stage || 'cold_start'}
                      </span>
                    </td>
                    
                    {/* Horizon */}
                    <td className="px-4 py-3 text-blue-400">
                      {signal.time_horizon}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {/* Summary Stats */}
      {signals.length > 0 && (
        <div className="mt-6 pt-4 border-t border-gray-700">
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-center">
            <div>
              <div className="text-2xl font-bold text-green-400">
                {signals.filter(s => s.prediction.includes('BUY')).length}
              </div>
              <div className="text-xs text-gray-400">Buy Signals</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-red-400">
                {signals.filter(s => s.prediction.includes('SELL')).length}
              </div>
              <div className="text-xs text-gray-400">Sell Signals</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-gray-400">
                {signals.filter(s => s.prediction === 'NEUTRAL').length}
              </div>
              <div className="text-xs text-gray-400">Neutral</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-blue-400">
                {(signals.reduce((sum, s) => sum + s.confidence, 0) / signals.length * 100).toFixed(1)}%
              </div>
              <div className="text-xs text-gray-400">Avg Confidence</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-purple-400">
                {signals.filter(s => s.risk_level === 'HIGH').length}
              </div>
              <div className="text-xs text-gray-400">High Risk</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
