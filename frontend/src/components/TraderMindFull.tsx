import React, { useEffect, useState } from 'react';
import { ModelAnalysisCard } from './ModelAnalysisCard';
import { fetchWithFallback } from '../utils/api';

type Summary = {
  samples: number;
  vote_counts: Record<string, number>;
  avg_confidence: number;
  final_decision: { action: string; confidence: number };
};

export const TraderMindFull: React.FC = () => {
  const [symbol, setSymbol] = useState('BTCUSDT');
  const [summary, setSummary] = useState<Summary | null>(null);
  const [aggregations, setAggregations] = useState<Record<string, any>>({});
  const [recent, setRecent] = useState<any[]>([]);
  const [risk, setRisk] = useState<any>(null);

  const [error, setError] = useState<string | null>(null);

  const fetchAll = async () => {
    try {
      setError(null);
      const j = await fetchWithFallback(`/api/v1/trader-mind/full/${symbol}`);
      setSummary(j.summary ?? null);
      setAggregations(j.model_aggregations ?? {});
      setRecent(j.recent_model_events ?? []);
      setRisk(j.risk_estimate ?? null);
    } catch (e: any) {
      console.error('trader mind full fetch', e);
      setError(String(e));
    }
  };

  useEffect(() => {
    fetchAll();
    const t = setInterval(fetchAll, 4000);
    return () => clearInterval(t);
  }, [symbol]);

  return (
    <div className="p-4 bg-gray-800 rounded">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-2xl font-bold">Trader Mind Full — {symbol}</h2>
        <div>
          <label className="mr-2">Symbol</label>
          <input value={symbol} onChange={e=>setSymbol(e.target.value.toUpperCase())} className="px-2 py-1 rounded bg-gray-700" />
        </div>
      </div>

      {error && (
        <div className="mb-4 p-2 bg-red-800 text-white rounded">Backend unreachable: {error}</div>
      )}

      {summary ? (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <div className="bg-gray-700 p-3 rounded">
            <div className="text-sm text-gray-300">Samples</div>
            <div className="text-2xl font-bold">{summary.samples}</div>
            <div className="text-sm mt-2">Avg confidence: {(summary.avg_confidence*100).toFixed(1)}%</div>
            <div className="mt-2">Final: {summary.final_decision.action} ({(summary.final_decision.confidence*100).toFixed(1)}%)</div>
          </div>
          <div className="md:col-span-2">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {Object.entries(aggregations).map(([model, agg]: any) => (
                <ModelAnalysisCard key={model} model={model} count={agg.count||0} avg_confidence={agg.avg_confidence||0} last_prediction={agg.last_prediction||''} confidences={agg.confidences||[]} />
              ))}
            </div>
          </div>
        </div>
      ) : (
        <div className="mb-4">Loading summary...</div>
      )}

      <div className="mb-4 bg-gray-700 p-3 rounded">
        <h3 className="font-semibold mb-2">Recent Model Events</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="text-left text-gray-400">
                <th>Model</th>
                <th>Prediction</th>
                <th>Confidence</th>
                <th>When</th>
              </tr>
            </thead>
            <tbody>
              {recent.map((r,i)=> (
                <tr key={i} className="border-t border-gray-600">
                  <td className="py-2">{r.model_name}</td>
                  <td>{r.prediction}</td>
                  <td>{r.confidence!=null? (r.confidence*100).toFixed(1)+'%':'-'}</td>
                  <td>{r.created_at}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="bg-gray-700 p-3 rounded">
        <h3 className="font-semibold mb-2">Risk Estimate</h3>
        <div>Volatility (est): {risk? (risk.volatility*100).toFixed(2)+'%':'N/A'}</div>
      </div>
    </div>
  );
};
