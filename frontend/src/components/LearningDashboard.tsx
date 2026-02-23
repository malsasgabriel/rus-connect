import React, { useState, useEffect } from 'react';
import { fetchWithFallback } from '../utils/api';

interface LearningMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  sample_count: number;
  last_updated: number;
  model_version: number;
}

interface ModelStats {
  learning_rate: number;
  adaptation_rate: number;
  model_weights_count: number;
  feature_stats_count: number;
  last_retrain: number;
  pending_feedback: number;
  auto_feedback_enabled: boolean;
  pump_threshold: number;
  status: string;
  features: string[];
}

export const LearningDashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<LearningMetrics | null>(null);
  const [stats, setStats] = useState<ModelStats | null>(null);
  const [feedbackForm, setFeedbackForm] = useState({
    symbol: '',
    predicted_prob: 0,
    actual_pump: false,
    confidence: 1.0,
    notes: ''
  });
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);
        const [metricsData, statsData] = await Promise.all([
          fetchWithFallback('/api/v1/admin/performance'),
          fetchWithFallback('/api/v1/admin/model-stats')
        ]);
        setMetrics(metricsData);
        setStats(statsData);
      } catch (error: any) {
        console.error('Failed to fetch learning data:', error);
        setError(error.message || 'Failed to fetch learning data');
        setMetrics(null);
        setStats(null);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 30000); // Update every 30 seconds
    
    return () => clearInterval(interval);
  }, []);

  const submitFeedback = async (e: React.FormEvent) => {
    e.preventDefault();
    
    try {
      const response = await fetchWithFallback('/api/v1/admin/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...feedbackForm,
          timestamp: Math.floor(Date.now() / 1000)
        }),
      });
      
      if (response.ok) {
        alert('Feedback submitted successfully!');
        setFeedbackForm({
          symbol: '',
          predicted_prob: 0,
          actual_pump: false,
          confidence: 1.0,
          notes: ''
        });
      } else {
        alert('Failed to submit feedback');
      }
    } catch (error: any) {
      console.error('Error submitting feedback:', error);
      alert('Error submitting feedback: ' + (error.message || error.toString()));
    }
  };

  const formatTimestamp = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleString();
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'text-green-500';
      case 'learning': return 'text-blue-500';
      case 'error': return 'text-red-500';
      default: return 'text-gray-500';
    }
  };

  return (
    <div className="bg-gray-800 rounded-lg p-6 space-y-6">
      <h2 className="text-2xl font-bold text-white">🧠 Continuous Learning Dashboard</h2>
      
      {error && (
        <div className="bg-red-800 text-white p-3 rounded">
          Error: {error}
        </div>
      )}
      
      {loading ? (
        <div className="text-center py-4">Loading...</div>
      ) : (
        <>
          {/* Performance Metrics */}
          {metrics && (
            <div className="bg-gray-700 rounded-lg p-4">
              <h3 className="text-lg font-semibold text-white mb-4">Model Performance</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-400">{(metrics.accuracy * 100).toFixed(1)}%</div>
                  <div className="text-sm text-gray-400">Accuracy</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-400">{(metrics.precision * 100).toFixed(1)}%</div>
                  <div className="text-sm text-gray-400">Precision</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-yellow-400">{(metrics.recall * 100).toFixed(1)}%</div>
                  <div className="text-sm text-gray-400">Recall</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-purple-400">{(metrics.f1_score * 100).toFixed(1)}%</div>
                  <div className="text-sm text-gray-400">F1-Score</div>
                </div>
              </div>
              <div className="mt-4 text-center">
                <div className="text-lg text-white">Samples: {metrics.sample_count}</div>
                <div className="text-sm text-gray-400">Last Updated: {formatTimestamp(metrics.last_updated)}</div>
              </div>
            </div>
          )}

          {/* Model Statistics */}
          {stats && (
            <div className="bg-gray-700 rounded-lg p-4">
              <h3 className="text-lg font-semibold text-white mb-4">Learning System Status</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <div className={`text-lg font-semibold ${getStatusColor(stats.status)}`}>
                    Status: (stats.status || '').toUpperCase()
                  </div>
                  <div className="text-sm text-gray-400 mt-2">
                    Learning Rate: {stats.learning_rate}
                  </div>
                  <div className="text-sm text-gray-400">
                    Adaptation Rate: {stats.adaptation_rate}
                  </div>
                  <div className="text-sm text-gray-400">
                    Pump Threshold: {(stats.pump_threshold * 100).toFixed(0)}%
                  </div>
                </div>
                <div>
                  <div className="text-sm text-gray-400">
                    Model Weights: {stats.model_weights_count}
                  </div>
                  <div className="text-sm text-gray-400">
                    Feature Stats: {stats.feature_stats_count}
                  </div>
                  <div className="text-sm text-gray-400">
                    Pending Feedback: {stats.pending_feedback}
                  </div>
                  <div className="text-sm text-gray-400">
                    Last Retrain: {formatTimestamp(stats.last_retrain)}
                  </div>
                  <div className="text-sm text-gray-400">
                    Auto Feedback: {stats.auto_feedback_enabled ? '✅ Enabled' : '❌ Disabled'}
                  </div>
                </div>
              </div>
            </div>
          )}
        </>
      )}

      {/* Manual Feedback Form */}
      <div className="bg-gray-700 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-white mb-4">Submit Manual Feedback</h3>
        <form onSubmit={submitFeedback} className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Symbol
              </label>
              <input
                type="text"
                value={feedbackForm.symbol}
                onChange={(e) => setFeedbackForm({...feedbackForm, symbol: (e.target.value || '').toUpperCase()})}
                className="w-full px-3 py-2 bg-gray-600 text-white rounded border border-gray-500 focus:border-blue-500 focus:outline-none"
                placeholder="e.g., BTCUSDT"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Predicted Probability
              </label>
              <input
                type="number"
                step="0.01"
                min="0"
                max="1"
                value={feedbackForm.predicted_prob}
                onChange={(e) => setFeedbackForm({...feedbackForm, predicted_prob: parseFloat(e.target.value)})}
                className="w-full px-3 py-2 bg-gray-600 text-white rounded border border-gray-500 focus:border-blue-500 focus:outline-none"
                placeholder="0.75"
                required
              />
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Actual Pump Occurred?
              </label>
              <select
                value={feedbackForm.actual_pump.toString()}
                onChange={(e) => setFeedbackForm({...feedbackForm, actual_pump: e.target.value === 'true'})}
                className="w-full px-3 py-2 bg-gray-600 text-white rounded border border-gray-500 focus:border-blue-500 focus:outline-none"
              >
                <option value="false">No - False Alarm</option>
                <option value="true">Yes - Actual Pump</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Confidence (0-1)
              </label>
              <input
                type="number"
                step="0.1"
                min="0"
                max="1"
                value={feedbackForm.confidence}
                onChange={(e) => setFeedbackForm({...feedbackForm, confidence: parseFloat(e.target.value)})}
                className="w-full px-3 py-2 bg-gray-600 text-white rounded border border-gray-500 focus:border-blue-500 focus:outline-none"
              />
            </div>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Notes (optional)
            </label>
            <textarea
              value={feedbackForm.notes}
              onChange={(e) => setFeedbackForm({...feedbackForm, notes: e.target.value})}
              className="w-full px-3 py-2 bg-gray-600 text-white rounded border border-gray-500 focus:border-blue-500 focus:outline-none"
              rows={3}
              placeholder="Additional context about this prediction..."
            />
          </div>
          
          <button
            type="submit"
            className="w-full px-4 py-2 bg-blue-600 text-white font-medium rounded hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            Submit Feedback
          </button>
        </form>
      </div>

      {/* Feature List */}
      {stats && stats.features && (
        <div className="bg-gray-700 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-4">Active Features</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
            {stats.features.map((feature, index) => (
              <div key={index} className="px-3 py-1 bg-gray-600 text-sm text-gray-300 rounded">
                {feature.replace(/_/g, ' ').toUpperCase()}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
