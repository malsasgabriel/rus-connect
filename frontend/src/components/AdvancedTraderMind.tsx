import React, { useEffect, useState } from 'react';
import { fetchWithFallback } from '../utils/api';

interface TraderMindResp {
  symbol: string;
  samples: number;
  vote_counts: Record<string, number>;
  avg_confidence: number;
  final_decision: { action: string; confidence: number };
}

export const AdvancedTraderMind: React.FC = () => {
  const [symbol, setSymbol] = useState<string>('BTCUSDT');
  const [data, setData] = useState<TraderMindResp | null>(null);

  useEffect(() => {
    const fetcher = async () => {
      try {
        const json = await fetchWithFallback(`/api/v1/trader-mind/${symbol}`);
        setData(json);
      } catch (e) {
        console.error(e);
      }
    };
    fetcher();
    const id = setInterval(fetcher, 5000);
    return () => clearInterval(id);
  }, [symbol]);

  return (
    <div className="p-4 bg-gray-800 rounded">
      <h2 className="text-xl font-semibold mb-2">Trader Mind — {symbol}</h2>
      <div className="mb-4">
        <label className="mr-2">Symbol:</label>
        <input value={symbol} onChange={e => setSymbol(e.target.value.toUpperCase())} className="px-2 py-1 rounded bg-gray-700" />
      </div>
      {data ? (
        <div>
          <div className="mb-2">Samples: {data.samples}</div>
          <div className="mb-2">Avg confidence: {(data.avg_confidence*100).toFixed(1)}%</div>
          <div className="mb-4">Final: {data.final_decision.action} ({(data.final_decision.confidence*100).toFixed(1)}%)</div>
          <div className="grid grid-cols-3 gap-2">
            {Object.entries(data.vote_counts).map(([k,v]) => (
              <div key={k} className="p-2 bg-gray-700 rounded">
                <div className="text-sm text-gray-300">{k}</div>
                <div className="text-2xl font-bold">{v}</div>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div>Loading...</div>
      )}
    </div>
  );
};
