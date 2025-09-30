import * as React from 'react';
import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
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

interface CalibrationData {
  models: { [symbol: string]: { [model: string]: CalibrationStatus } };
  system: {
    overall_status: string;
    completed: number;
    total: number;
    eta: number;
  };
}

interface InfrastructureMetrics {
  cpu: {
    usage_percent: number;
    core_count: number;
  };
  memory: {
    usage_percent: number;
    used_gb: number;
    total_gb: number;
  };
  kafka: {
    lag: number;
    messages_per_sec: number;
    consumption_rate: number;
    connection_status: string;
  };
  database: {
    queries_per_sec: number;
    response_latency: number;
    active_connections: number;
    connection_status: string;
  };
  last_updated: number;
}

interface MLData {
  system: SystemMetrics;
  symbols: { [symbol: string]: SymbolMetrics };
  temporal_analysis: TemporalAnalysis;
  risk_metrics: RiskMetrics;
}

// Add cache interface
interface DataCache {
  mlData: MLData | null;
  calibrationData: CalibrationData | null;
  infrastructureData: InfrastructureMetrics | null;
  timestamp: number;
}

// Cache expiration time (5 minutes)
const CACHE_EXPIRATION = 5 * 60 * 1000;

// Memoized components for performance optimization
const ControlPanel = React.memo(({ 
  timeRange, 
  setTimeRange, 
  autoRefresh, 
  setAutoRefresh, 
  refreshing, 
  fetchData,
  availableSymbols,
  selectedSymbols,
  handleSymbolToggle,
  handleSelectAllSymbols,
  handleDeselectAllSymbols
}: {
  timeRange: string;
  setTimeRange: (value: string) => void;
  autoRefresh: string;
  setAutoRefresh: (value: string) => void;
  refreshing: boolean;
  fetchData: () => void;
  availableSymbols: string[];
  selectedSymbols: string[];
  handleSymbolToggle: (symbol: string) => void;
  handleSelectAllSymbols: () => void;
  handleDeselectAllSymbols: () => void;
}) => {
  return (
    <div className="bg-gray-700 rounded-lg p-4">
      <h3 className="text-lg font-semibold text-white mb-4">Control Panel</h3>
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {/* Time Range Selector */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Time Range</label>
          <select 
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value)}
            className="w-full bg-gray-600 border border-gray-500 rounded px-3 py-2 text-white"
          >
            <option value="5m">5 Minutes</option>
            <option value="15m">15 Minutes</option>
            <option value="1h">1 Hour</option>
            <option value="6h">6 Hours</option>
            <option value="24h">24 Hours</option>
          </select>
        </div>
        
        {/* Auto-refresh Options */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Auto Refresh</label>
          <select 
            value={autoRefresh}
            onChange={(e) => setAutoRefresh(e.target.value)}
            className="w-full bg-gray-600 border border-gray-500 rounded px-3 py-2 text-white"
          >
            <option value="off">Off</option>
            <option value="5">5 Seconds</option>
            <option value="10">10 Seconds</option>
            <option value="30">30 Seconds</option>
            <option value="60">1 Minute</option>
          </select>
        </div>
        
        {/* Manual Refresh Button */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Refresh</label>
          <button
            onClick={fetchData}
            disabled={refreshing}
            className={`w-full px-4 py-2 rounded flex items-center justify-center ${
              refreshing 
                ? 'bg-gray-500 cursor-not-allowed' 
                : 'bg-blue-600 hover:bg-blue-700'
            } text-white`}
          >
            {refreshing ? (
              <>
                <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Refreshing...
              </>
            ) : (
              'Refresh Data'
            )}
          </button>
        </div>
        
        {/* Symbol Filters */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Symbols</label>
          <div className="flex space-x-2">
            <button
              onClick={handleSelectAllSymbols}
              className="px-2 py-1 bg-green-600 text-white rounded text-xs hover:bg-green-700"
            >
              Select All
            </button>
            <button
              onClick={handleDeselectAllSymbols}
              className="px-2 py-1 bg-red-600 text-white rounded text-xs hover:bg-red-700"
            >
              Deselect All
            </button>
          </div>
        </div>
      </div>
      
      {/* Symbol Filter Checkboxes */}
      <div className="mt-4">
        <label className="block text-sm font-medium text-gray-300 mb-2">Filter by Symbols:</label>
        <div className="flex flex-wrap gap-2">
          {availableSymbols.map(symbol => (
            <label key={symbol} className="inline-flex items-center">
              <input
                type="checkbox"
                checked={selectedSymbols.includes(symbol)}
                onChange={() => handleSymbolToggle(symbol)}
                className="rounded text-blue-600 focus:ring-blue-500"
              />
              <span className="ml-1 text-sm text-gray-300">{symbol}</span>
            </label>
          ))}
        </div>
      </div>
    </div>
  );
});

const SystemAlertsPanel = React.memo(() => {
  return (
    <div className="bg-gray-700 rounded-lg p-4">
      <h3 className="text-lg font-semibold text-white mb-4">System Alerts</h3>
      <div className="space-y-2">
        {/* Example alerts - in a real implementation, these would come from an API */}
        <div className="flex items-center p-3 bg-red-900/30 border border-red-500 rounded">
          <span className="text-red-500 text-xl mr-2">🚨</span>
          <div>
            <div className="font-medium text-red-400">Critical: Database Connection Lost</div>
            <div className="text-sm text-gray-300">Connection to PostgreSQL database failed. Retrying in 30 seconds.</div>
            <div className="text-xs text-gray-400 mt-1">2025-09-30 14:30:25</div>
          </div>
        </div>
        <div className="flex items-center p-3 bg-yellow-900/30 border border-yellow-500 rounded">
          <span className="text-yellow-500 text-xl mr-2">⚠️</span>
          <div>
            <div className="font-medium text-yellow-400">Warning: High Memory Usage</div>
            <div className="text-sm text-gray-300">Memory usage exceeded 85% threshold. Consider restarting services.</div>
            <div className="text-xs text-gray-400 mt-1">2025-09-30 14:25:10</div>
          </div>
        </div>
        <div className="flex items-center p-3 bg-green-900/30 border border-green-500 rounded">
          <span className="text-green-500 text-xl mr-2">✅</span>
          <div>
            <div className="font-medium text-green-400">Normal: System Calibrated</div>
            <div className="text-sm text-gray-300">All models successfully calibrated and running normally.</div>
            <div className="text-xs text-gray-400 mt-1">2025-09-30 14:20:45</div>
          </div>
        </div>
      </div>
    </div>
  );
});

export const MLDashboard: React.FC = () => {
  const [mlData, setMlData] = useState<MLData | null>(null);
  const [calibrationData, setCalibrationData] = useState<CalibrationData | null>(null);
  const [infrastructureData, setInfrastructureData] = useState<InfrastructureMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // WebSocket reference
  const wsRef = useRef<WebSocket | null>(null);
  
  // Cache reference
  const cacheRef = useRef<DataCache>({
    mlData: null,
    calibrationData: null,
    infrastructureData: null,
    timestamp: 0
  });
  
  // Control panel state
  const [timeRange, setTimeRange] = useState('1h');
  const [autoRefresh, setAutoRefresh] = useState('30s');
  const [selectedSymbols, setSelectedSymbols] = useState<string[]>([]);
  const [refreshing, setRefreshing] = useState(false);

  // Get available symbols from data
  const availableSymbols = useMemo(() => {
    return mlData ? Object.keys(mlData.symbols) : [];
  }, [mlData]);

  // Filter data based on selected symbols
  const filteredMlData = useMemo(() => {
    if (!mlData) return null;
    return {
      ...mlData,
      symbols: selectedSymbols.length > 0 
        ? Object.fromEntries(
            Object.entries(mlData.symbols).filter(([symbol]) => 
              selectedSymbols.includes(symbol)
            )
          )
        : mlData.symbols
    };
  }, [mlData, selectedSymbols]);

  const filteredCalibrationData = useMemo(() => {
    if (!calibrationData) return null;
    return {
      ...calibrationData,
      models: selectedSymbols.length > 0 
        ? Object.fromEntries(
            Object.entries(calibrationData.models).filter(([symbol]) => 
              selectedSymbols.includes(symbol)
            )
          )
        : calibrationData.models
    };
  }, [calibrationData, selectedSymbols]);

  // Check if cache is valid
  const isCacheValid = useCallback(() => {
    const now = Date.now();
    return (now - cacheRef.current.timestamp) < CACHE_EXPIRATION;
  }, []);

  // Get data from cache if valid
  const getDataFromCache = useCallback(() => {
    if (isCacheValid()) {
      return {
        mlData: cacheRef.current.mlData,
        calibrationData: cacheRef.current.calibrationData,
        infrastructureData: cacheRef.current.infrastructureData
      };
    }
    return null;
  }, [isCacheValid]);

  // Save data to cache
  const saveDataToCache = useCallback((mlData: MLData, calibrationData: CalibrationData, infrastructureData: InfrastructureMetrics) => {
    cacheRef.current = {
      mlData,
      calibrationData,
      infrastructureData,
      timestamp: Date.now()
    };
  }, []);

  // Aggregate data for charts
  const aggregateChartData = useCallback((data: MLData) => {
    // Aggregate accuracy data by symbol for bar charts
    const accuracyData: { symbol: string; accuracy: number }[] = [];
    
    // Aggregate confidence data by symbol for gauges
    const confidenceData: { symbol: string; confidence: number }[] = [];
    
    // Aggregate temporal performance data
    const hourlyPerformanceData: { hour: string; performance: number }[] = [];
    const dailyPerformanceData: { day: string; performance: number }[] = [];
    
    Object.entries(data.symbols).forEach(([symbol, metrics]) => {
      // Calculate average accuracy across all models for this symbol
      const avgAccuracy = Object.values(metrics).reduce((sum, model) => sum + model.accuracy, 0) / Object.keys(metrics).length;
      accuracyData.push({ symbol, accuracy: avgAccuracy });
      
      // Calculate average confidence across all models for this symbol
      const avgConfidence = Object.values(metrics).reduce((sum, model) => sum + model.confidence, 0) / Object.keys(metrics).length;
      confidenceData.push({ symbol, confidence: avgConfidence });
    });
    
    // Process temporal data if available
    if (data.temporal_analysis) {
      Object.entries(data.temporal_analysis.hourly_performance || {}).forEach(([hour, performance]) => {
        hourlyPerformanceData.push({ hour, performance });
      });
      
      Object.entries(data.temporal_analysis.daily_performance || {}).forEach(([day, performance]) => {
        dailyPerformanceData.push({ day, performance });
      });
    }
    
    return {
      accuracyData,
      confidenceData,
      hourlyPerformanceData,
      dailyPerformanceData
    };
  }, []);

  // Validate API response
  const validateApiResponse = useCallback((data: any, dataType: string): boolean => {
    if (!data) {
      console.error(`Invalid ${dataType} response: null or undefined`);
      return false;
    }
    
    // Add specific validation based on data type
    switch (dataType) {
      case 'mlData':
        return typeof data === 'object' && 
               data.system !== undefined && 
               data.symbols !== undefined;
      case 'calibrationData':
        return typeof data === 'object' && 
               data.models !== undefined && 
               data.system !== undefined;
      case 'infrastructureData':
        return typeof data === 'object' && 
               data.cpu !== undefined && 
               data.memory !== undefined && 
               data.kafka !== undefined && 
               data.database !== undefined;
      default:
        return true;
    }
  }, []);

  const fetchData = useCallback(async () => {
    try {
      setRefreshing(true);
      
      // Try to get data from cache first
      const cachedData = getDataFromCache();
      if (cachedData) {
        setMlData(cachedData.mlData);
        setCalibrationData(cachedData.calibrationData);
        setInfrastructureData(cachedData.infrastructureData);
        setLoading(false);
        setRefreshing(false);
        return;
      }
      
      // Fetch fresh data
      const [mlData, calibrationData, infrastructureData] = await Promise.all([
        fetchWithFallback('/api/v1/ml/metrics'),
        fetchWithFallback('/api/v1/ml/calibration'),
        fetchWithFallback('/api/v1/infrastructure/metrics')
      ]);
      
      // Validate responses
      if (!validateApiResponse(mlData, 'mlData')) {
        throw new Error('Invalid ML metrics data received');
      }
      
      if (!validateApiResponse(calibrationData, 'calibrationData')) {
        throw new Error('Invalid calibration data received');
      }
      
      if (!validateApiResponse(infrastructureData, 'infrastructureData')) {
        throw new Error('Invalid infrastructure data received');
      }
      
      // Save to state
      setMlData(mlData);
      setCalibrationData(calibrationData);
      setInfrastructureData(infrastructureData);
      
      // Save to cache
      saveDataToCache(mlData, calibrationData, infrastructureData);
      
      setError(null);
      
      // Initialize selected symbols with all available symbols if none selected
      if (selectedSymbols.length === 0 && mlData) {
        setSelectedSymbols(Object.keys(mlData.symbols));
      }
    } catch (err: any) {
      setError(err.message);
      console.error('Error fetching ML data:', err);
      
      // Try to use cached data as fallback
      const cachedData = getDataFromCache();
      if (cachedData) {
        setMlData(cachedData.mlData);
        setCalibrationData(cachedData.calibrationData);
        setInfrastructureData(cachedData.infrastructureData);
        setError(null);
      }
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [selectedSymbols.length, getDataFromCache, saveDataToCache, validateApiResponse]);

  // Initialize WebSocket connection
  const initWebSocket = useCallback(() => {
    // Close existing connection if any
    if (wsRef.current) {
      wsRef.current.close();
    }

    // Create new WebSocket connection
    // Use the proxy path which will be handled by Vite
    const wsUrl = '/ws';
    
    console.log(`Connecting to WebSocket at: ${wsUrl}`);
    console.log(`Window location: ${window.location.host}`);
    console.log(`Includes localhost:3000: ${window.location.host.includes('localhost:3000')}`);
    
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connected');
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        
        // Handle different message types
        switch (message.type) {
          case 'market_update':
            // Handle market data updates if needed
            break;
          case 'direction_signal':
            // Handle ML direction signals if needed
            break;
          case 'pump_signal_update':
            // Handle pump signal updates if needed
            break;
          default:
            // Handle other message types if needed
            break;
        }
      } catch (err) {
        console.error('Error parsing WebSocket message:', err);
      }
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      // Attempt to reconnect after 5 seconds
      setTimeout(() => {
        initWebSocket();
      }, 5000);
    };

    ws.onerror = (err) => {
      console.error('WebSocket error:', err);
    };
  }, []);

  // Debounce function for frequent updates
  const debounce = useCallback((func: (...args: any[]) => void, delay: number) => {
    let timeoutId: ReturnType<typeof setTimeout>;
    return (...args: any[]) => {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(() => func(...args), delay);
    };
  }, []);

  // Debounced fetch data function
  const debouncedFetchData = useMemo(() => {
    return debounce(fetchData, 300);
  }, [fetchData, debounce]);

  // Handle symbol toggle
  const handleSymbolToggle = useCallback((symbol: string) => {
    if (selectedSymbols.includes(symbol)) {
      setSelectedSymbols(selectedSymbols.filter(s => s !== symbol));
    } else {
      setSelectedSymbols([...selectedSymbols, symbol]);
    }
  }, [selectedSymbols]);

  const handleSelectAllSymbols = useCallback(() => {
    if (mlData) {
      setSelectedSymbols(Object.keys(mlData.symbols));
    }
  }, [mlData]);

  const handleDeselectAllSymbols = useCallback(() => {
    setSelectedSymbols([]);
  }, []);

  useEffect(() => {
    fetchData();
    initWebSocket();
    
    // Set up auto-refresh interval
    let intervalId: ReturnType<typeof setInterval> | null = null;
    if (autoRefresh !== 'off') {
      const intervalMs = parseInt(autoRefresh) * 1000;
      intervalId = setInterval(debouncedFetchData, intervalMs);
    }
    
    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
      // Close WebSocket connection on unmount
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [fetchData, initWebSocket, autoRefresh, debouncedFetchData]);

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

      // Since fetchWithFallback returns parsed JSON, not a Response object,
      // we need to handle the response differently
      if (response && response.status === 'success') {
        alert('Auto calibration started successfully!');
      } else if (response && response.status) {
        // Handle response with status field
        alert(`Auto calibration response: ${response.status} - ${response.message || 'No message'}`);
      } else {
        // Handle unexpected response format
        console.error('Unexpected response format:', response);
        alert(`Auto calibration response: ${JSON.stringify(response)}`);
      }
    } catch (error: any) {
      console.error('Error starting auto calibration:', error);
      alert(`Error starting auto calibration: ${error.message || error.toString() || 'Unknown error'}`);
    }
  };

  if (loading && !mlData) {
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

  if (!mlData || !calibrationData || !infrastructureData) {
    return (
      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-2xl font-bold text-white mb-4">ML Dashboard</h2>
        <p className="text-gray-400">No data available</p>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 rounded-lg p-6 space-y-6">
      {/* Control Panel */}
      <ControlPanel 
        timeRange={timeRange}
        setTimeRange={setTimeRange}
        autoRefresh={autoRefresh}
        setAutoRefresh={setAutoRefresh}
        refreshing={refreshing}
        fetchData={fetchData}
        availableSymbols={availableSymbols}
        selectedSymbols={selectedSymbols}
        handleSymbolToggle={handleSymbolToggle}
        handleSelectAllSymbols={handleSelectAllSymbols}
        handleDeselectAllSymbols={handleDeselectAllSymbols}
      />

      {/* System Alerts Panel */}
      <SystemAlertsPanel />

      {/* Dashboard Header */}
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-white">🤖 ML Performance Dashboard</h2>
        <button 
          onClick={startAutoCalibration}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 flex items-center"
        >
          <span>Start Auto Calibration</span>
        </button>
      </div>

      {/* System Status */}
      <div className="bg-gray-700 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-white mb-4">System Status</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Overall Health */}
          <div className="bg-gray-600 rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <h4 className="font-medium text-gray-300">Overall Health</h4>
              <div className={`px-2 py-1 rounded text-xs font-semibold ${getHealthColor(mlData.system.overall_health).replace('text-', 'bg-').replace('-500', '-900/30 border border-').replace('text', 'text')}`}>
                {mlData.system.overall_health}
              </div>
            </div>
            <div className="text-3xl font-bold text-center mb-2">
              <span className={getHealthColor(mlData.system.overall_health)}>
                {mlData.system.overall_health}
              </span>
            </div>
            <div className="text-center text-sm text-gray-400">
              System is running normally
            </div>
          </div>
          
          {/* Uptime and Components */}
          <div className="bg-gray-600 rounded-lg p-4">
            <h4 className="font-medium text-gray-300 mb-3">System Components</h4>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <div className="flex items-center">
                  <div className="w-3 h-3 rounded-full bg-green-500 mr-2"></div>
                  <span className="text-gray-300">PostgreSQL</span>
                </div>
                <span className="text-xs bg-green-900/30 text-green-400 px-2 py-1 rounded">Connected</span>
              </div>
              <div className="flex justify-between items-center">
                <div className="flex items-center">
                  <div className="w-3 h-3 rounded-full bg-green-500 mr-2"></div>
                  <span className="text-gray-300">Kafka</span>
                </div>
                <span className="text-xs bg-green-900/30 text-green-400 px-2 py-1 rounded">Connected</span>
              </div>
              <div className="flex justify-between items-center">
                <div className="flex items-center">
                  <div className="w-3 h-3 rounded-full bg-green-500 mr-2"></div>
                  <span className="text-gray-300">ML Engine</span>
                </div>
                <span className="text-xs bg-green-900/30 text-green-400 px-2 py-1 rounded">Running</span>
              </div>
              <div className="flex justify-between items-center">
                <div className="flex items-center">
                  <div className="w-3 h-3 rounded-full bg-green-500 mr-2"></div>
                  <span className="text-gray-300">API Gateway</span>
                </div>
                <span className="text-xs bg-green-900/30 text-green-400 px-2 py-1 rounded">Available</span>
              </div>
            </div>
          </div>
          
          {/* Metrics and Latency */}
          <div className="bg-gray-600 rounded-lg p-4">
            <h4 className="font-medium text-gray-300 mb-3">Performance Metrics</h4>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-400">Uptime</span>
                <span className="text-white font-mono">24d 16h 32m</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Avg Latency</span>
                <span className="text-white font-mono">42ms</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Models Active</span>
                <span className="text-white font-mono">{mlData.system.healthy_models}/{mlData.system.total_models}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Last Updated</span>
                <span className="text-white font-mono text-xs">{formatTimestamp(mlData.system.last_updated)}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Enhanced Model Performance Monitoring */}
      <div className="bg-gray-700 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-white mb-4">Enhanced Model Performance Monitoring</h3>
        
        {/* Comparative Performance Tables */}
        <div className="mb-8">
          <h4 className="text-md font-semibold text-gray-300 mb-4">Comparative Performance Tables</h4>
          {Object.entries(mlData.symbols).map(([symbol, metrics]) => (
            <div key={symbol} className="mb-6 last:mb-0">
              <h5 className="text-md font-semibold text-gray-300 mb-2">{symbol}</h5>
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
        
        {/* Model Drift Visualization */}
        <div className="mb-8">
          <h4 className="text-md font-semibold text-gray-300 mb-4">Model Drift Visualization</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-gray-600 rounded-lg p-4">
              <h5 className="font-medium text-gray-300 mb-3">Drift Over Time</h5>
              <div className="h-48 flex items-end space-x-2">
                {/* Mock drift data visualization */}
                {[80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25].map((value, index) => (
                  <div key={index} className="flex flex-col items-center flex-1">
                    <div 
                      className="w-full bg-blue-500 rounded-t" 
                      style={{ height: `${value}%` }}
                    ></div>
                    <div className="text-xs text-gray-400 mt-1">D{index+1}</div>
                  </div>
                ))}
              </div>
              <div className="text-xs text-gray-400 mt-2 text-center">
                Days ago
              </div>
            </div>
            <div className="bg-gray-600 rounded-lg p-4">
              <h5 className="font-medium text-gray-300 mb-3">Feature Drift</h5>
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-300">Price Features</span>
                    <span className="text-gray-400">Low Drift</span>
                  </div>
                  <div className="w-full bg-gray-500 rounded-full h-2">
                    <div className="bg-green-500 h-2 rounded-full" style={{ width: '20%' }}></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-300">Volume Features</span>
                    <span className="text-gray-400">Medium Drift</span>
                  </div>
                  <div className="w-full bg-gray-500 rounded-full h-2">
                    <div className="bg-yellow-500 h-2 rounded-full" style={{ width: '60%' }}></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-300">Technical Indicators</span>
                    <span className="text-gray-400">High Drift</span>
                  </div>
                  <div className="w-full bg-gray-500 rounded-full h-2">
                    <div className="bg-red-500 h-2 rounded-full" style={{ width: '85%' }}></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        {/* Learning/Validation Curves */}
        <div>
          <h4 className="text-md font-semibold text-gray-300 mb-4">Learning/Validation Curves</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-gray-600 rounded-lg p-4">
              <h5 className="font-medium text-gray-300 mb-3">Training Progress</h5>
              <div className="h-48 relative">
                {/* Mock learning curve */}
                <div className="absolute inset-0 flex items-end">
                  <div className="w-full h-40 border-l border-b border-gray-500">
                    {/* X-axis */}
                    <div className="absolute bottom-0 left-0 right-0 flex justify-between text-xs text-gray-400">
                      <span>0</span>
                      <span>50</span>
                      <span>100</span>
                      <span>150</span>
                      <span>200</span>
                    </div>
                    {/* Y-axis */}
                    <div className="absolute left-0 top-0 bottom-0 flex flex-col justify-between text-xs text-gray-400">
                      <span>1.0</span>
                      <span>0.8</span>
                      <span>0.6</span>
                      <span>0.4</span>
                      <span>0.2</span>
                      <span>0.0</span>
                    </div>
                    {/* Curve */}
                    <svg className="absolute inset-0 w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
                      <polyline 
                        fill="none" 
                        stroke="#3b82f6" 
                        strokeWidth="2" 
                        points="0,100 10,80 20,65 30,55 40,50 50,45 60,42 70,40 80,38 90,37 100,36"
                      />
                    </svg>
                  </div>
                </div>
                <div className="text-xs text-gray-400 mt-2 text-center">
                  Epochs
                </div>
              </div>
            </div>
            <div className="bg-gray-600 rounded-lg p-4">
              <h5 className="font-medium text-gray-300 mb-3">Validation Performance</h5>
              <div className="h-48 relative">
                <div className="absolute inset-0 flex items-end">
                  <div className="w-full h-40 border-l border-b border-gray-500">
                    {/* X-axis */}
                    <div className="absolute bottom-0 left-0 right-0 flex justify-between text-xs text-gray-400">
                      <span>0</span>
                      <span>50</span>
                      <span>100</span>
                      <span>150</span>
                      <span>200</span>
                    </div>
                    {/* Y-axis */}
                    <div className="absolute left-0 top-0 bottom-0 flex flex-col justify-between text-xs text-gray-400">
                      <span>1.0</span>
                      <span>0.8</span>
                      <span>0.6</span>
                      <span>0.4</span>
                      <span>0.2</span>
                      <span>0.0</span>
                    </div>
                    {/* Training curve */}
                    <svg className="absolute inset-0 w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
                      <polyline 
                        fill="none" 
                        stroke="#3b82f6" 
                        strokeWidth="2" 
                        points="0,100 10,80 20,65 30,55 40,50 50,45 60,42 70,40 80,38 90,37 100,36"
                      />
                    </svg>
                    {/* Validation curve */}
                    <svg className="absolute inset-0 w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
                      <polyline 
                        fill="none" 
                        stroke="#10b981" 
                        strokeWidth="2" 
                        points="0,100 10,85 20,70 30,60 40,55 50,50 60,48 70,46 80,45 90,44 100,43"
                      />
                    </svg>
                  </div>
                </div>
                <div className="flex items-center justify-center mt-2 space-x-4">
                  <div className="flex items-center">
                    <div className="w-3 h-3 bg-blue-500 mr-1"></div>
                    <span className="text-xs text-gray-400">Training</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-3 h-3 bg-green-500 mr-1"></div>
                    <span className="text-xs text-gray-400">Validation</span>
                  </div>
                </div>
                <div className="text-xs text-gray-400 mt-2 text-center">
                  Epochs
                </div>
              </div>
            </div>
          </div>
        </div>
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
            {calibrationData.system.completed === 0 && calibrationData.system.total > 0 && (
              <div className="mt-2 text-sm text-yellow-400">
                ⚠️ No calibration data yet. Please start automatic calibration.
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
        {calibrationData.system.completed === 0 && (
          <div className="mt-4 p-3 bg-blue-900/30 border border-blue-500 rounded text-blue-300 text-sm">
            <p>ℹ️ The system needs verified training examples to perform calibration. </p>
            <p>Calibration will automatically progress as the system generates and verifies predictions.</p>
            <p>Alternatively, you can start automatic calibration to begin the process.</p>
          </div>
        )}
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

      {/* Business Metrics */}
      <div className="bg-gray-700 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-white mb-4">Business Metrics</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Accuracy Bar Charts */}
          <div className="bg-gray-600 rounded-lg p-4">
            <h4 className="font-medium text-gray-300 mb-3">Accuracy by Symbol</h4>
            <div className="space-y-4">
              {Object.entries(mlData.symbols).map(([symbol, metrics]) => {
                // Calculate average accuracy across all models for this symbol
                const avgAccuracy = Object.values(metrics).reduce((sum, model) => sum + model.accuracy, 0) / Object.keys(metrics).length;
                return (
                  <div key={symbol}>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-300">{symbol}</span>
                      <span className="text-gray-400">{(avgAccuracy * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-gray-500 rounded-full h-2">
                      <div 
                        className="bg-blue-500 h-2 rounded-full" 
                        style={{ width: `${avgAccuracy * 100}%` }}
                      ></div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
          
          {/* Confidence Gauges */}
          <div className="bg-gray-600 rounded-lg p-4">
            <h4 className="font-medium text-gray-300 mb-3">Confidence Indicators</h4>
            <div className="space-y-4">
              {Object.entries(mlData.symbols).map(([symbol, metrics]) => {
                // Calculate average confidence across all models for this symbol
                const avgConfidence = Object.values(metrics).reduce((sum, model) => sum + model.confidence, 0) / Object.keys(metrics).length;
                return (
                  <div key={symbol}>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-300">{symbol}</span>
                      <span className="text-gray-400">{(avgConfidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-gray-500 rounded-full h-2">
                      <div 
                        className={`h-2 rounded-full ${avgConfidence > 0.8 ? 'bg-green-500' : avgConfidence > 0.6 ? 'bg-yellow-500' : 'bg-red-500'}`} 
                        style={{ width: `${avgConfidence * 100}%` }}
                      ></div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
          
          {/* 24-Hour Signal Statistics */}
          <div className="bg-gray-600 rounded-lg p-4">
            <h4 className="font-medium text-gray-300 mb-3">24-Hour Signal Statistics</h4>
            <div className="space-y-4">
              {/* Mock data for signal statistics */}
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-gray-300">Total Signals</span>
                  <span className="text-gray-400">1,247</span>
                </div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-gray-300">Correct Predictions</span>
                  <span className="text-green-400">892 (71.5%)</span>
                </div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-gray-300">Incorrect Predictions</span>
                  <span className="text-red-400">355 (28.5%)</span>
                </div>
                <div className="mt-3">
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-300">Direction Distribution</span>
                  </div>
                  <div className="flex space-x-1">
                    <div className="flex-1 text-center">
                      <div className="text-xs text-gray-400">UP</div>
                      <div className="text-lg font-bold text-green-400">42%</div>
                    </div>
                    <div className="flex-1 text-center">
                      <div className="text-xs text-gray-400">DOWN</div>
                      <div className="text-lg font-bold text-red-400">31%</div>
                    </div>
                    <div className="flex-1 text-center">
                      <div className="text-xs text-gray-400">SIDEWAYS</div>
                      <div className="text-lg font-bold text-yellow-400">27%</div>
                    </div>
                  </div>
                </div>
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

      {/* Infrastructure Monitoring */}
      <div className="bg-gray-700 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-white mb-4">Infrastructure Monitoring</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* CPU and Memory Usage */}
          <div className="bg-gray-600 rounded-lg p-4">
            <h4 className="font-medium text-gray-300 mb-3">System Resources</h4>
            <div className="space-y-4">
              {/* CPU Usage */}
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-300">CPU Usage</span>
                  <span className="text-gray-400">{infrastructureData.cpu.usage_percent.toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-500 rounded-full h-3">
                  <div 
                    className={`h-3 rounded-full ${
                      infrastructureData.cpu.usage_percent > 80 
                        ? 'bg-red-500' 
                        : infrastructureData.cpu.usage_percent > 60 
                          ? 'bg-yellow-500' 
                          : 'bg-green-500'
                    }`} 
                    style={{ width: `${infrastructureData.cpu.usage_percent}%` }}
                  ></div>
                </div>
                <div className="text-xs text-gray-400 mt-1">
                  {infrastructureData.cpu.core_count} cores
                </div>
              </div>
              
              {/* Memory Usage */}
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-300">Memory Usage</span>
                  <span className="text-gray-400">
                    {infrastructureData.memory.used_gb.toFixed(1)} GB / {infrastructureData.memory.total_gb.toFixed(1)} GB
                  </span>
                </div>
                <div className="w-full bg-gray-500 rounded-full h-3">
                  <div 
                    className={`h-3 rounded-full ${
                      infrastructureData.memory.usage_percent > 85 
                        ? 'bg-red-500' 
                        : infrastructureData.memory.usage_percent > 70 
                          ? 'bg-yellow-500' 
                          : 'bg-green-500'
                    }`} 
                    style={{ width: `${infrastructureData.memory.usage_percent}%` }}
                  ></div>
                </div>
              </div>
            </div>
          </div>
          
          {/* Kafka Metrics */}
          <div className="bg-gray-600 rounded-lg p-4">
            <h4 className="font-medium text-gray-300 mb-3">Kafka Metrics</h4>
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-gray-700 rounded p-3">
                <div className="text-sm text-gray-400">Consumer Lag</div>
                <div className="text-xl font-bold text-blue-400">{infrastructureData.kafka.lag.toLocaleString()}</div>
              </div>
              <div className="bg-gray-700 rounded p-3">
                <div className="text-sm text-gray-400">Messages/sec</div>
                <div className="text-xl font-bold text-green-400">{infrastructureData.kafka.messages_per_sec.toLocaleString()}</div>
              </div>
              <div className="bg-gray-700 rounded p-3">
                <div className="text-sm text-gray-400">Consumption Rate</div>
                <div className="text-xl font-bold text-purple-400">{infrastructureData.kafka.consumption_rate.toLocaleString()}</div>
              </div>
              <div className="bg-gray-700 rounded p-3">
                <div className="text-sm text-gray-400">Status</div>
                <div className={`font-semibold ${
                  infrastructureData.kafka.connection_status === 'healthy' 
                    ? 'text-green-400' 
                    : 'text-red-400'
                }`}>
                  {infrastructureData.kafka.connection_status}
                </div>
              </div>
            </div>
          </div>
          
          {/* Database Metrics */}
          <div className="bg-gray-600 rounded-lg p-4">
            <h4 className="font-medium text-gray-300 mb-3">Database Metrics</h4>
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-gray-700 rounded p-3">
                <div className="text-sm text-gray-400">Queries/sec</div>
                <div className="text-xl font-bold text-blue-400">{infrastructureData.database.queries_per_sec}</div>
              </div>
              <div className="bg-gray-700 rounded p-3">
                <div className="text-sm text-gray-400">Response Latency</div>
                <div className="text-xl font-bold text-yellow-400">{infrastructureData.database.response_latency.toFixed(1)}ms</div>
              </div>
              <div className="bg-gray-700 rounded p-3">
                <div className="text-sm text-gray-400">Active Connections</div>
                <div className="text-xl font-bold text-green-400">{infrastructureData.database.active_connections}</div>
              </div>
              <div className="bg-gray-700 rounded p-3">
                <div className="text-sm text-gray-400">Status</div>
                <div className={`font-semibold ${
                  infrastructureData.database.connection_status === 'healthy' 
                    ? 'text-green-400' 
                    : 'text-red-400'
                }`}>
                  {infrastructureData.database.connection_status}
                </div>
              </div>
            </div>
          </div>
          
          {/* Last Updated */}
          <div className="bg-gray-600 rounded-lg p-4 flex items-center justify-center">
            <div className="text-center">
              <div className="text-sm text-gray-400 mb-1">Last Updated</div>
              <div className="text-gray-300 font-mono">
                {formatTimestamp(infrastructureData.last_updated)}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};