import React from 'react';

interface StatusDashboardProps {
  wsStatus: 'connecting' | 'connected' | 'disconnected';
  kafkaThroughput: string; // e.g., "100 msg/s"
  dbStatus: string; // e.g., "Connected", "Disconnected"
  anomalyDistribution: {
    high: number;
    medium: number;
    low: number;
  };
}

export const StatusDashboard: React.FC<StatusDashboardProps> = ({
  wsStatus,
  kafkaThroughput,
  dbStatus,
  anomalyDistribution,
}) => {
  const getStatusColor = (status: string) => {
    if (status === 'connected' || status === 'Connected') return 'text-green-500';
    if (status === 'connecting') return 'text-yellow-500';
    return 'text-red-500';
  };

  return (
    <div className="bg-gray-800 p-4 rounded-lg shadow-md col-span-1">
      <h2 className="text-xl font-semibold mb-4 text-gray-200">System Status</h2>
      <div className="space-y-2">
        <div className="flex justify-between items-center">
          <span className="text-gray-400">WebSocket:</span>
          <span className={`font-medium ${getStatusColor(wsStatus)}`}>{wsStatus}</span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-gray-400">Kafka Throughput:</span>
          <span className="font-medium text-gray-200">{kafkaThroughput}</span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-gray-400">PostgreSQL:</span>
          <span className={`font-medium ${getStatusColor(dbStatus)}`}>{dbStatus}</span>
        </div>
        <div className="pt-2">
          <h3 className="text-md font-medium text-gray-300">Anomaly Distribution:</h3>
          <div className="flex justify-between text-sm text-gray-400 mt-1">
            <span>High: <span className="text-red-400">{anomalyDistribution.high}</span></span>
            <span>Medium: <span className="text-orange-400">{anomalyDistribution.medium}</span></span>
            <span>Low: <span className="text-green-400">{anomalyDistribution.low}</span></span>
          </div>
        </div>
      </div>
    </div>
  );
};
