import React from 'react';

type Props = {
  model: string;
  count: number;
  avg_confidence: number;
  last_prediction: string;
  confidences: number[];
};

export const ModelAnalysisCard: React.FC<Props> = ({ model, count, avg_confidence, last_prediction, confidences }) => {
  // small sparkline using inline SVG
  const sparkPath = (() => {
    if (!confidences || confidences.length === 0) return '';
    const w = 120; const h = 30;
    const max = Math.max(...confidences);
    const min = Math.min(...confidences);
    const range = Math.max(1e-6, max - min);
    return confidences.map((v, i) => {
      const x = (i / (confidences.length - 1 || 1)) * w;
      const y = h - ((v - min) / range) * h;
      return `${i===0? 'M':'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
    }).join(' ');
  })();

  return (
    <div className="bg-gray-800 p-3 rounded shadow-sm">
      <div className="flex justify-between items-center">
        <h3 className="font-semibold">{model}</h3>
        <span className="text-sm text-gray-400">samples: {count}</span>
      </div>
      <div className="mt-2 text-sm text-gray-200">Avg conf: {(avg_confidence*100).toFixed(1)}%</div>
      <div className="mt-1 text-sm text-gray-300">Last: {last_prediction || '-'}</div>
      <div className="mt-2">
        <svg width="120" height="30" className="block">
          <path d={sparkPath} stroke="#60a5fa" strokeWidth={1.5} fill="none" />
        </svg>
      </div>
    </div>
  );
};
