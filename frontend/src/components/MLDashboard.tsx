import React, { useState, useEffect } from 'react';
import { fetchWithFallback } from '../utils/api';

interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  roc_auc: number;
  confidence: number;
  calibration_progress: number;
  last_updated: number;
}

interface SymbolMetrics {
  lstm: ModelMetrics;
  xgboost: ModelMetrics;
  transformer: ModelMetrics;
  meta_learner: ModelMetrics;
  ensemble: ModelMetrics;
}

interface SystemMetrics {
  overall_health: string;
  total_models: number;
  healthy_models: number;
  average_accuracy: number;
  average_confidence: number;
  last_updated: number;
}

interface TemporalAnalysis {
  hourly_performance: { [hour: string]: number };
  daily_performance: { [day: string]: number };
}

interface RiskMetrics {
  value_at_risk: number;
  expected_shortfall: number;
  stability_score: number;
  correlation_exposure: number;
}

interface CalibrationStatus {
  status: string;
  progress: number;
  eta: number;
  last_calibrated: number;
}

interface MLData {
  system: SystemMetrics;
  symbols: { [symbol: string]: SymbolMetrics };
  temporal_analysis: TemporalAnalysis;
  risk_metrics: RiskMetrics;
}

interface CalibrationData {
  models: { [symbol: string]: { [model: string]: CalibrationStatus } };
  system: {
    overall_status: string;
    completed: number;
    total: number;
    eta: number;
  };
}

export const MLDashboard: React.FC = () => {
  const [mlData, setMlData] = useState<MLData | null>(null);
  const [calibrationData, setCalibrationData] = useState<CalibrationData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const [mlData, calibrationData] = await Promise.all([
          fetchWithFallback('/api/v1/ml/metrics'),
          fetchWithFallback('/api/v1/ml/calibration')
        ]);

        setMlData(mlData);
        setCalibrationData(calibrationData);

        setError(null);
      } catch (err: any) {
        setError(err.message);
        console.error('Error fetching ML data:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 30000); // Update every 30 seconds

    return () => clearInterval(interval);
  }, []);

  const formatTimestamp = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleString();
  };

  const getHealthColor = (health: string) => {
    switch (health) {
      case 'EXCELLENT': return 'text-green-500';
      case 'GOOD': return 'text-blue-500';
      case 'WARNING': return 'text-yellow-500';
      case 'CRITICAL': return 'text-red-500';
      default: return 'text-gray-500';
    }
  };

  const getCalibrationStatusColor = (status: string) => {
    switch (status) {
      case 'COMPLETE': return 'text-green-500';
      case 'CALIBRATING': return 'text-blue-500';
      case 'PENDING': return 'text-yellow-500';
      case 'ERROR': return 'text-red-500';
      default: return 'text-gray-500';
    }
  };

  const startAutoCalibration = async () => {
    try {
      const response = await fetchWithFallback('/api/v1/ml/calibration/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        alert('Auto calibration started successfully!');
      } else {
        alert('Failed to start auto calibration');
      }
    } catch (error) {
      console.error('Error starting auto calibration:', error);
      alert('Error starting auto calibration');
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="text-xl text-gray-400">Loading ML Dashboard...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-900/20 border border-red-500 rounded-lg p-6">
        <h2 className="text-2xl font-bold text-red-400 mb-4">Error</h2>
        <p className="text-red-300">{error}</p>
        <button 
          onClick={() => window.location.reload()}
          className="mt-4 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
        >
          Retry
        </button>
      </div>
    );
  }

  if (!mlData || !calibrationData) {
    return (
      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-2xl font-bold text-white mb-4">ML Dashboard</h2>
        <p className="text-gray-400">No data available</p>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 rounded-lg p-6 space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-white">🤖 ML Performance Dashboard</h2>
        <button 
          onClick={startAutoCalibration}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 flex items-center"
        >
          <span>Start Auto Calibration</span>
        </button>
      </div>

      {/* System Overview */}
      <div className="bg-gray-700 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-white mb-4">System Overview</h3>
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          <div className="text-center">
            <div className={`text-2xl font-bold ${getHealthColor(mlData.system.overall_health)}`}>
              {mlData.system.overall_health}
            </div>
            <div className="text-sm text-gray-400">Health</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-400">
              {mlData.system.average_accuracy.toFixed(2)}
            </div>
            <div className="text-sm text-gray-400">Avg Accuracy</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-400">
              {mlData.system.average_confidence.toFixed(2)}
            </div>
            <div className="text-sm text-gray-400">Avg Confidence</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-yellow-400">
              {mlData.system.healthy_models}/{mlData.system.total_models}
            </div>
            <div className="text-sm text-gray-400">Healthy Models</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-400">
              {formatTimestamp(mlData.system.last_updated)}
            </div>
            <div className="text-sm text-gray-400">Last Updated</div>
          </div>
        </div>
      </div>

      {/* Model Comparison */}
      <div className="bg-gray-700 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-white mb-4">Model Performance Comparison</h3>
        {Object.entries(mlData.symbols).map(([symbol, metrics]) => (
          <div key={symbol} className="mb-6 last:mb-0">
            <h4 className="text-md font-semibold text-gray-300 mb-2">{symbol}</h4>
            <div className="overflow-x-auto">
              <table className="w-full text-sm text-left">
                <thead className="text-xs text-gray-400 uppercase bg-gray-600">
                  <tr>
                    <th className="px-4 py-2">Model</th>
                    <th className="px-4 py-2">Accuracy</th>
                    <th className="px-4 py-2">Precision</th>
                    <th className="px-4 py-2">Recall</th>
                    <th className="px-4 py-2">F1-Score</th>
                    <th className="px-4 py-2">ROC-AUC</th>
                    <th className="px-4 py-2">Confidence</th>
                    <th className="px-4 py-2">Calibration</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(metrics).map(([modelName, modelMetrics]) => (
                    <tr key={modelName} className="border-b border-gray-600">
                      <td className="px-4 py-2 font-medium text-white capitalize">{modelName.replace('_', ' ')}</td>
                      <td className="px-4 py-2 text-blue-400">{(modelMetrics.accuracy * 100).toFixed(1)}%</td>
                      <td className="px-4 py-2 text-green-400">{(modelMetrics.precision * 100).toFixed(1)}%</td>
                      <td className="px-4 py-2 text-yellow-400">{(modelMetrics.recall * 100).toFixed(1)}%</td>
                      <td className="px-4 py-2 text-purple-400">{(modelMetrics.f1_score * 100).toFixed(1)}%</td>
                      <td className="px-4 py-2 text-pink-400">{(modelMetrics.roc_auc * 100).toFixed(1)}%</td>
                      <td className="px-4 py-2 text-indigo-400">{(modelMetrics.confidence * 100).toFixed(1)}%</td>
                      <td className="px-4 py-2">
                        <div className="flex items-center">
                          <div className="w-16 bg-gray-600 rounded-full h-2 mr-2">
                            <div 
                              className="h-2 rounded-full bg-green-500" 
                              style={{ width: `${modelMetrics.calibration_progress * 100}%` }}
                            ></div>
                          </div>
                          <span className="text-xs text-gray-300">
                            {(modelMetrics.calibration_progress * 100).toFixed(0)}%
                          </span>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ))}
      </div>

      {/* Calibration Status */}
      <div className="bg-gray-700 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-white mb-4">Calibration Status</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h4 className="text-md font-semibold text-gray-300 mb-2">System Status</h4>
            <div className="flex items-center justify-between p-3 bg-gray-600 rounded">
              <span className={`font-semibold ${getCalibrationStatusColor(calibrationData.system.overall_status)}`}>
                {calibrationData.system.overall_status}
              </span>
              <span className="text-gray-400">
                {calibrationData.system.completed}/{calibrationData.system.total} completed
              </span>
            </div>
            {calibrationData.system.eta > 0 && (
              <div className="mt-2 text-sm text-gray-400">
                Estimated time remaining: {Math.floor(calibrationData.system.eta / 60)}m {calibrationData.system.eta % 60}s
              </div>
            )}
          </div>
          <div>
            <h4 className="text-md font-semibold text-gray-300 mb-2">Model Status</h4>
            <div className="space-y-2">
              {Object.entries(calibrationData.models).map(([symbol, models]) => (
                <div key={symbol} className="p-3 bg-gray-600 rounded">
                  <div className="font-medium text-white">{symbol}</div>
                  <div className="flex flex-wrap gap-2 mt-1">
                    {Object.entries(models).map(([modelName, status]) => (
                      <div key={modelName} className="flex items-center">
                        <span className="text-xs text-gray-400 mr-1">{modelName}:</span>
                        <span className={`text-xs font-semibold ${getCalibrationStatusColor(status.status)}`}>
                          {status.progress === 1 ? '✓' : `${(status.progress * 100).toFixed(0)}%`}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Temporal Analysis */}
      <div className="bg-gray-700 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-white mb-4">Temporal Performance Analysis</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="text-md font-semibold text-gray-300 mb-2">Hourly Performance</h4>
            <div className="h-64 overflow-y-auto">
              <div className="grid grid-cols-6 gap-2">
                {Object.entries(mlData.temporal_analysis.hourly_performance).map(([hour, performance]) => (
                  <div key={hour} className="flex flex-col items-center">
                    <div className="text-xs text-gray-400">{hour}</div>
                    <div 
                      className="w-6 bg-blue-500 rounded-t" 
                      style={{ height: `${performance * 100}%` }}
                    ></div>
                    <div className="text-xs text-gray-300 mt-1">{(performance * 100).toFixed(0)}%</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div>
            <h4 className="text-md font-semibold text-gray-300 mb-2">Daily Performance</h4>
            <div className="h-64 overflow-y-auto">
              <div className="space-y-2">
                {Object.entries(mlData.temporal_analysis.daily_performance).map(([day, performance]) => (
                  <div key={day} className="flex items-center">
                    <div className="w-20 text-xs text-gray-400">{day}</div>
                    <div className="flex-1 bg-gray-600 rounded-full h-4">
                      <div 
                        className="bg-green-500 h-4 rounded-full" 
                        style={{ width: `${performance * 100}%` }}
                      ></div>
                    </div>
                    <div className="w-12 text-xs text-gray-300 text-right ml-2">
                      {(performance * 100).toFixed(0)}%
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Risk Metrics */}
      <div className="bg-gray-700 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-white mb-4">Risk Metrics</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-red-400">
              {(mlData.risk_metrics.value_at_risk * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-gray-400">Value at Risk</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-orange-400">
              {(mlData.risk_metrics.expected_shortfall * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-gray-400">Expected Shortfall</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-400">
              {mlData.risk_metrics.stability_score}/100
            </div>
            <div className="text-sm text-gray-400">Stability Score</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-yellow-400">
              {mlData.risk_metrics.correlation_exposure.toFixed(2)}
            </div>
            <div className="text-sm text-gray-400">Correlation Exposure</div>
          </div>
        </div>
      </div>
    </div>
  );
};