import React, { useEffect, useState, useRef } from 'react';
import { ModelAnalysisCard } from './ModelAnalysisCard';
import { fetchWithFallback } from '../utils/api';
import { CandleData } from '../types';
import {
  createChart,
  ColorType,
  ChartOptions,
  IChartApi,
  CandlestickData,
} from 'lightweight-charts';

type Summary = {
  samples: number;
  vote_counts: Record<string, number>;
  avg_confidence: number;
  final_decision?: { action: string; confidence: number };
};

export const TraderMindFull: React.FC = () => {
  const [symbol, setSymbol] = useState('BTCUSDT');
  const [summary, setSummary] = useState<Summary | null>(null);
  const [aggregations, setAggregations] = useState<Record<string, any>>({});
  const [recent, setRecent] = useState<any[]>([]);
  const [risk, setRisk] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [candleData, setCandleData] = useState<CandleData[]>([]);
  const [chartLoading, setChartLoading] = useState<boolean>(false);

  const [error, setError] = useState<string | null>(null);

  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<any>(null); // Candlestick series reference

  const fetchAll = async () => {
    try {
      setError(null);
      setLoading(true);
      const j = await fetchWithFallback(`/api/v1/trader-mind/full/${symbol}`);
      setSummary(j.summary ?? null);
      setAggregations(j.model_aggregations ?? {});
      setRecent(j.recent_model_events ?? []);
      setRisk(j.risk_estimate ?? null);
    } catch (e: any) {
      console.error('trader mind full fetch', e);
      setError(e.message || String(e));
      // Reset data on error
      setSummary(null);
      setAggregations({});
      setRecent([]);
      setRisk(null);
    } finally {
      setLoading(false);
    }
  };

  // Fetch candle data for the chart
  useEffect(() => {
    const fetchCandleData = async () => {
      setChartLoading(true);
      try {
        // In a real implementation, this would come from your API
        // For now, we'll simulate candle data
        const simulatedCandles: CandleData[] = [];
        const now = Date.now();
        const oneMinute = 60 * 1000;

        // Generate last 60 minutes of simulated candle data
        for (let i = 60; i >= 0; i--) {
          const time = new Date(now - i * oneMinute).toISOString().slice(0, 16).replace('T', ' ') + ':00';
          const basePrice = 40000 + (Math.random() - 0.5) * 100; // Random base price around 40000
          const open = basePrice;
          const close = open + (Math.random() - 0.5) * 200; // Random close within 200 range
          const high = Math.max(open, close) + Math.random() * 100; // Random high
          const low = Math.min(open, close) - Math.random() * 100; // Random low

          simulatedCandles.push({
            time: time,
            open: parseFloat(open.toFixed(2)),
            high: parseFloat(high.toFixed(2)),
            low: parseFloat(low.toFixed(2)),
            close: parseFloat(close.toFixed(2)),
          });
        }

        setCandleData(simulatedCandles);
      } catch (e: any) {
        console.error('Error fetching candle data:', e);
        setError(e.message || 'Failed to fetch candle data');
      } finally {
        setChartLoading(false);
      }
    };

    fetchCandleData();
    // Update candle data every minute
    const id = setInterval(fetchCandleData, 60000);
    return () => clearInterval(id);
  }, [symbol]);

  // Initialize and update chart
  useEffect(() => {
    if (!chartContainerRef.current || candleData.length === 0) return;

    // Clean up previous chart if exists
    if (chartRef.current) {
      chartRef.current.remove();
    }

    // Chart options
    const chartOptions: any = {
      width: chartContainerRef.current.clientWidth,
      height: 400,
      layout: {
        background: { type: ColorType.Solid, color: 'rgba(31, 41, 55, 1)' }, // bg-gray-800
        textColor: 'rgba(156, 163, 175, 1)', // text-gray-400
        fontSize: 12,
        fontFamily: 'Trebuchet MS, sans-serif',
        attributionLogo: false,
      },
      grid: {
        vertLines: { color: 'rgba(55, 65, 81, 0.5)', style: 0, visible: true }, // border-gray-700
        horzLines: { color: 'rgba(55, 65, 81, 0.5)', style: 0, visible: true }, // border-gray-700
      },
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 5,
        barSpacing: 6,
        minBarSpacing: 1,
        fixLeftEdge: true,
        lockVisibleTimeRangeOnResize: true,
        rightBarStaysOnScroll: true,
        borderVisible: true,
        borderColor: 'rgba(55, 65, 81, 1)',
        visible: true,
        ticksVisible: true,
      },
      rightPriceScale: {
        visible: true,
        borderColor: 'rgba(55, 65, 81, 1)',
        autoScale: true,
        mode: 0,
        invertScale: false,
        alignLabels: true,
      },
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
        horzTouchDrag: true,
        vertTouchDrag: true,
      },
      handleScale: {
        axisPressedMouseMove: true,
        mouseWheel: true,
        pinch: true,
        axisDoubleClickReset: true,
      },
      crosshair: {
        mode: 0,
        vertLine: {
          color: 'rgba(255, 255, 255, 0.1)',
          width: 1,
          style: 0,
          visible: true,
          labelVisible: true,
          labelBackgroundColor: 'rgba(41, 98, 255, 0.75)',
        },
        horzLine: {
          color: 'rgba(255, 255, 255, 0.1)',
          width: 1,
          style: 0,
          visible: true,
          labelVisible: true,
          labelBackgroundColor: 'rgba(41, 98, 255, 0.75)',
        },
      },
      watermark: {
        visible: false,
        color: 'rgba(0, 0, 0, 0)',
        text: '',
        fontSize: 48,
        horzAlign: 'center',
        vertAlign: 'center',
        fontFamily: 'Trebuchet MS, sans-serif',
        fontStyle: '',
      },
    };

    // Create chart
    const chart = createChart(chartContainerRef.current, chartOptions);
    chartRef.current = chart;

    // Create candlestick series
    const candlestickSeries = chart.addCandlestickSeries({
      upColor: 'rgba(0, 150, 136, 1)',
      downColor: 'rgba(255, 85, 85, 1)',
      borderDownColor: 'rgba(255, 85, 85, 1)',
      borderUpColor: 'rgba(0, 150, 136, 1)',
      wickDownColor: 'rgba(255, 85, 85, 1)',
      wickUpColor: 'rgba(0, 150, 136, 1)',
    });
    seriesRef.current = candlestickSeries;

    // Format data for chart
    const formattedData = candleData.map((candle) => ({
      time: candle.time as string, // Chart expects string format like 'YYYY-MM-DD HH:mm'
      open: candle.open,
      high: candle.high,
      low: candle.low,
      close: candle.close,
    })) as CandlestickData[];

    // Add data to series
    candlestickSeries.setData(formattedData);

    // Handle window resize
    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };

    window.addEventListener('resize', handleResize);

    // Cleanup function
    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [candleData]);

  useEffect(() => {
    fetchAll();
    const t = setInterval(fetchAll, 4000);
    return () => clearInterval(t);
  }, [symbol]);

  return (
    <div className="p-4 bg-gray-800 rounded">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-2xl font-bold">Trader Mind Full — {symbol}</h2>
        <div className="flex items-center gap-4">
          <div>
            <label className="mr-2">Symbol</label>
            <input value={symbol} onChange={e=>setSymbol((e.target.value || '').toUpperCase())} className="px-2 py-1 rounded bg-gray-700" />
          </div>
          <div className="text-sm text-gray-400">Real-time 1m chart</div>
        </div>
      </div>

      {error && (
        <div className="mb-4 p-2 bg-red-800 text-white rounded">Backend unreachable: {error}</div>
      )}

      {/* Chart container */}
      <div className="mb-6">
        <div ref={chartContainerRef} className="w-full h-96" />
        {chartLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-800 bg-opacity-75 rounded">
            <div className="text-white">Loading chart data...</div>
          </div>
        )}
      </div>

      {loading ? (
        <div className="mb-4">Loading...</div>
      ) : summary ? (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <div className="bg-gray-700 p-3 rounded">
            <div className="text-sm text-gray-300">Samples</div>
            <div className="text-2xl font-bold">{summary.samples}</div>
            <div className="text-sm mt-2">Avg confidence: {(summary.avg_confidence*100).toFixed(1)}%</div>
            {summary.final_decision && summary.final_decision.action && (
              <div className="mt-2">Final: {summary.final_decision.action} ({((summary.final_decision.confidence || 0) * 100).toFixed(1)}%)</div>
            )}
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
        <div className="mb-4">No data available</div>
      )}

      <div className="mb-4 bg-gray-70 p-3 rounded">
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
              {recent.length > 0 ? (
                recent.map((r,i)=> (
                  <tr key={i} className="border-t border-gray-600">
                    <td className="py-2">{r.model_name}</td>
                    <td>{r.prediction}</td>
                    <td>{r.confidence!=null? (r.confidence*100).toFixed(1)+'%':'-'}</td>
                    <td>{r.created_at}</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={4} className="py-2 text-center text-gray-400">No recent events</td>
                </tr>
              )}
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
