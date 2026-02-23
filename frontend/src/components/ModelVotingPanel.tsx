import React, { useEffect, useState } from 'react';
import { fetchWithFallback } from '../utils/api';
import { ModelAnalysisCard } from './ModelAnalysisCard';

type ModelAgg = {
  count: number;
  avg_confidence: number;
  last_prediction: string;
  confidences: number[];
};

export const ModelVotingPanel: React.FC<{ symbol?: string }> = ({ symbol: initial = 'BTCUSDT' }) => {
  const [symbol, setSymbol] = useState(initial);
  const [data, setData] = useState<Record<string, ModelAgg>>({});
  const [loading, setLoading] = useState(false);
  const [recentEvents, setRecentEvents] = useState<any[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);

  const fetchData = async (sym?: string) => {
    const s = sym || symbol;
    setLoading(true);
    try {
      const j = await fetchWithFallback(`/api/v1/trader-mind/full/${s}`);
      const models = j.model_aggregations || {};
      setRecentEvents(j.recent_model_events || []);
      const norm: Record<string, ModelAgg> = {};
      for (const k of Object.keys(models)) {
        const m = models[k];
        norm[k] = {
          count: m.count || 0,
          avg_confidence: m.avg_confidence || 0,
          last_prediction: m.last_prediction || '',
          confidences: m.confidences || [],
        };
      }
      setData(norm);
    } catch (e) {
      console.error('fetch trader mind full', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const t = setInterval(() => fetchData(), 5000);
    return () => clearInterval(t);
  }, [symbol]);

  // histogram helper
  const buildHistogram = (arr: number[], buckets = 6) => {
    if (!arr || arr.length === 0) return new Array(buckets).fill(0);
    const mins = Math.min(...arr);
    const maxs = Math.max(...arr);
    const span = Math.max(1e-6, maxs - mins);
    const out = new Array(buckets).fill(0);
    for (const v of arr) {
      const idx = Math.min(buckets - 1, Math.floor(((v - mins) / span) * buckets));
      out[idx]++;
    }
    return out;
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          <h2 className="text-xl font-bold">Model Voting — {symbol}</h2>
          <input value={symbol} onChange={e=>setSymbol((e.target.value || '').toUpperCase())} className="px-2 py-1 rounded bg-gray-700" />
          <button onClick={()=>fetchData(symbol)} className="px-2 py-1 bg-blue-600 rounded">Refresh</button>
        </div>
        <div className="text-sm text-gray-400">{loading ? 'Loading...' : ''}</div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {Object.keys(data).length === 0 && <div className="text-gray-400">No data</div>}
        {Object.entries(data).map(([model, agg]) => (
          <div key={model} onClick={()=>setSelectedModel(model)}>
            <ModelAnalysisCard
              model={model}
              count={agg.count}
              avg_confidence={agg.avg_confidence}
              last_prediction={agg.last_prediction}
              confidences={agg.confidences}
            />
          </div>
        ))}
      </div>

      {/* Drilldown modal */}
      {selectedModel && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-gray-900 p-4 rounded w-11/12 md:w-3/4">
            <div className="flex justify-between items-center mb-3">
              <h3 className="text-lg font-bold">{selectedModel} — Details</h3>
              <button onClick={()=>setSelectedModel(null)} className="px-2 py-1 bg-gray-700 rounded">Close</button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-semibold mb-2">Confidence histogram</h4>
                {(() => {
                  const agg = data[selectedModel];
                  const hist = buildHistogram(agg?.confidences || [], 8);
                  const max = Math.max(...hist, 1);
                  return (
                    <svg width="100%" height={120} className="bg-gray-800 p-2 rounded">
                      {hist.map((v,i)=> {
                        const bw = 100 / hist.length;
                        const h = (v / max) * 80;
                        const x = i * bw + 2;
                        return <rect key={i} x={`${x}%`} y={100 - h} width={`${bw-4}%`} height={h} fill="#60a5fa" />
                      })}
                    </svg>
                  )
                })()}
              </div>

              <div>
                <h4 className="font-semibold mb-2">Recent events (filtered)</h4>
                <div className="max-h-48 overflow-auto bg-gray-800 p-2 rounded">
                  {(recentEvents.filter(r=>r.model_name===selectedModel)).slice(0,50).map((r,i)=> (
                    <div key={i} className="border-b border-gray-700 py-1 text-sm">
                      <div className="flex justify-between"><div className="font-medium">{r.prediction}</div><div className="text-gray-400">{r.confidence!=null? (r.confidence*100).toFixed(1)+'%':''}</div></div>
                      <div className="text-xs text-gray-400">{r.timestamp || r.created_at}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
