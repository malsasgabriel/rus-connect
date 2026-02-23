import React, { useEffect, useState, useRef } from 'react';
import { fetchWithFallback } from '../utils/api';
import { CandleData } from '../types';
import {
  createChart,
  ColorType,
  ChartOptions,
  IChartApi,
  CandlestickData,
} from 'lightweight-charts';

interface TraderMindResp {
  symbol: string;
  summary?: {
    model?: string;
    last_prediction?: string;
    confidence?: number;
    accuracy?: number;
    samples?: number;
    vote_counts?: Record<string, number>;
    avg_confidence?: number;
    final_decision?: { action: string; confidence: number };
  };
  // Legacy fields for backward compatibility
  samples?: number;
  vote_counts?: Record<string, number>;
  avg_confidence?: number;
  final_decision?: { action: string; confidence: number };
}

export const AdvancedTraderMind: React.FC = () => {
  const [symbol, setSymbol] = useState<string>('BTCUSDT');
  const [data, setData] = useState<TraderMindResp | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [candleData, setCandleData] = useState<CandleData[]>([]);
  const [chartLoading, setChartLoading] = useState<boolean>(false);

  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<any>(null); // Candlestick series reference

  // Fetch trader mind data
  useEffect(() => {
    const fetcher = async () => {
      try {
        setLoading(true);
        const json = await fetchWithFallback(`/api/v1/trader-mind/${symbol}`);
        setData(json);
        setError(null);
      } catch (e: any) {
        console.error(e);
        setError(e.message || 'Failed to fetch data');
        setData(null);
      } finally {
        setLoading(false);
      }
    };
    fetcher();
    const id = setInterval(fetcher, 5000);
    return () => clearInterval(id);
  }, [symbol]);

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

  return (
    <div className="p-4 bg-gray-800 rounded">
      <h2 className="text-xl font-semibold mb-2">Trader Mind — {symbol}</h2>
      <div className="mb-4 flex flex-wrap items-center gap-4">
        <div>
          <label className="mr-2">Symbol:</label>
          <input
            value={symbol}
            onChange={e => setSymbol((e.target.value || '').toUpperCase())}
            className="px-2 py-1 rounded bg-gray-700"
          />
        </div>
        <div className="text-sm text-gray-400">Real-time 1m chart</div>
      </div>
      
      {error && (
        <div className="mb-4 p-2 bg-red-800 text-white rounded">
          Error: {error}
        </div>
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
        <div>Loading trader mind data...</div>
      ) : data ? (
        <div>
          {data.summary ? (
            <>
              {data.summary.samples !== undefined && (
                <div className="mb-2">Samples: {data.summary.samples}</div>
              )}
              {data.summary.avg_confidence !== undefined && (
                <div className="mb-2">Avg confidence: {(data.summary.avg_confidence*100).toFixed(1)}%</div>
              )}
              {data.summary.final_decision && data.summary.final_decision.action && (
                <div className="mb-4">Final: {data.summary.final_decision.action} ({((data.summary.final_decision.confidence || 0) * 100).toFixed(1)}%)</div>
              )}
              {data.summary.last_prediction && (
                <div className="mb-2">Last prediction: {data.summary.last_prediction}</div>
              )}
              {data.summary.confidence !== undefined && (
                <div className="mb-2">Confidence: {(data.summary.confidence*100).toFixed(1)}%</div>
              )}
              {data.summary.accuracy !== undefined && (
                <div className="mb-2">Accuracy: {(data.summary.accuracy*100).toFixed(1)}%</div>
              )}
              {data.summary.vote_counts && Object.keys(data.summary.vote_counts).length > 0 && (
                <div className="grid grid-cols-3 gap-2">
                  {Object.entries(data.summary.vote_counts).map(([k,v]) => (
                    <div key={k} className="p-2 bg-gray-700 rounded">
                      <div className="text-sm text-gray-300">{k}</div>
                      <div className="text-2xl font-bold">{v}</div>
                    </div>
                  ))}
                </div>
              )}
            </>
          ) : (
            <>
              {data.samples !== undefined && (
                <div className="mb-2">Samples: {data.samples}</div>
              )}
              {data.avg_confidence !== undefined && (
                <div className="mb-2">Avg confidence: {(data.avg_confidence*100).toFixed(1)}%</div>
              )}
              {data.final_decision && data.final_decision.action && (
                <div className="mb-4">Final: {data.final_decision.action} ({((data.final_decision.confidence || 0) * 100).toFixed(1)}%)</div>
              )}
              {data.vote_counts && Object.keys(data.vote_counts).length > 0 && (
                <div className="grid grid-cols-3 gap-2">
                  {Object.entries(data.vote_counts).map(([k,v]) => (
                    <div key={k} className="p-2 bg-gray-700 rounded">
                      <div className="text-sm text-gray-300">{k}</div>
                      <div className="text-2xl font-bold">{v}</div>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      ) : (
        <div>No data available</div>
      )}
    </div>
  );
};
