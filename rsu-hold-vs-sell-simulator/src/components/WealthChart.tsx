import React, { useState } from 'react';
import { SimulationOutput, StrategyKey } from '../types';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from 'recharts';
import { LineChart as ChartIcon, DollarSign, TrendingUp } from 'lucide-react';

interface WealthChartProps {
  simulation: SimulationOutput;
}

export const WealthChart: React.FC<WealthChartProps> = ({ simulation }) => {
  const [chartMode, setChartMode] = useState<'wealth' | 'pricePct'>('wealth');

  const [activeStrategies, setActiveStrategies] = useState<Record<StrategyKey, boolean>>({
    hold: true,
    sp500: true,
    nasdaq: false,
    sector: true,
    cash: false,
    hybrid5050: true,
    ltcg1yr: false,
  });

  const toggleStrategy = (key: StrategyKey) => {
    setActiveStrategies(prev => ({ ...prev, [key]: !prev[key] }));
  };

  const initialStockPrice = simulation.trajectory[0]?.stockPrice || 1;
  const initialSp500Price = simulation.trajectory[0]?.sp500Price || 1;

  const chartData = simulation.trajectory.map(t => {
    const stockPricePct = Math.round(((t.stockPrice - initialStockPrice) / initialStockPrice) * 100);
    const sp500PricePct = Math.round(((t.sp500Price - initialSp500Price) / initialSp500Price) * 100);

    return {
      quarter: t.quarterLabel,
      hold: t.holdValue,
      sp500: t.sp500Value,
      nasdaq: t.nasdaqValue,
      sector: t.sectorValue,
      cash: t.cashValue,
      hybrid5050: t.hybrid5050Value,
      stockPrice: t.stockPrice,
      sp500Price: t.sp500Price,
      stockPricePct,
      sp500PricePct,
    };
  });

  const formatCurrency = (val: number) =>
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(val);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-slate-950/95 border border-slate-700/80 rounded-xl p-3.5 shadow-2xl backdrop-blur-md text-xs font-sans min-w-[210px]">
          <div className="font-extrabold text-white border-b border-slate-800 pb-1.5 mb-2 flex items-center justify-between">
            <span>{label}</span>
            <span className="text-slate-400 font-normal">Quarter Snapshot</span>
          </div>
          <div className="space-y-1.5">
            {payload.map((p: any) => (
              <div key={p.dataKey} className="flex items-center justify-between gap-4">
                <span className="flex items-center gap-1.5 font-medium" style={{ color: p.color }}>
                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: p.color }}></span>
                  {p.name}:
                </span>
                <span className="font-extrabold text-white">
                  {chartMode === 'wealth'
                    ? formatCurrency(p.value)
                    : `${p.value > 0 ? '+' : ''}${p.value}% ($${p.payload.stockPrice})`}
                </span>
              </div>
            ))}
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-slate-900/80 backdrop-blur-md border border-slate-800 rounded-2xl p-5 sm:p-6 shadow-xl">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-5">
        <div className="flex items-center gap-2.5">
          <div className="p-2 bg-emerald-500/10 text-emerald-400 rounded-lg border border-emerald-500/20">
            <ChartIcon className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-lg font-bold text-white">
              {chartMode === 'wealth' ? 'Portfolio Wealth Trajectory' : 'Stock Price Growth % Since Start Year'}
            </h3>
            <p className="text-xs text-slate-400">
              {chartMode === 'wealth'
                ? 'Quarterly cumulative wealth ($) across strategy options'
                : `Price performance of ${simulation.config.symbol} vs S&P 500 benchmark since ${simulation.config.startYear}`}
            </p>
          </div>
        </div>

        {/* View Mode Toggle */}
        <div className="flex items-center gap-2">
          <div className="grid grid-cols-2 gap-1 bg-slate-950 p-1 rounded-xl border border-slate-800">
            <button
              onClick={() => setChartMode('wealth')}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-1 ${
                chartMode === 'wealth'
                  ? 'bg-emerald-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <DollarSign className="w-3.5 h-3.5" /> Portfolio ($)
            </button>
            <button
              onClick={() => setChartMode('pricePct')}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-1 ${
                chartMode === 'pricePct'
                  ? 'bg-emerald-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <TrendingUp className="w-3.5 h-3.5" /> Price Growth (%)
            </button>
          </div>
        </div>
      </div>

      {/* Strategy Selector Chips (Wealth Mode Only) */}
      {chartMode === 'wealth' && (
        <div className="flex flex-wrap items-center gap-1.5 mb-4 pb-3 border-b border-slate-800/60">
          {(Object.keys(simulation.strategies) as StrategyKey[]).map(key => {
            const strat = simulation.strategies[key];
            const isActive = activeStrategies[key];
            return (
              <button
                key={key}
                onClick={() => toggleStrategy(key)}
                className={`px-2.5 py-1 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all ${
                  isActive
                    ? 'bg-slate-800 text-white border border-slate-700 shadow-sm'
                    : 'bg-slate-950/60 text-slate-500 border border-transparent hover:text-slate-300'
                }`}
              >
                <span
                  className="w-2.5 h-2.5 rounded-full"
                  style={{ backgroundColor: isActive ? strat.color : '#475569' }}
                ></span>
                {key === 'hold'
                  ? `Hold (${simulation.config.symbol})`
                  : key === 'sp500'
                  ? 'S&P 500'
                  : key === 'nasdaq'
                  ? 'Nasdaq 100'
                  : key === 'sector'
                  ? `Sector (${simulation.config.sectorEtf})`
                  : key === 'cash'
                  ? 'Cash'
                  : key === 'hybrid5050'
                  ? 'Hybrid 50/50'
                  : '1-Yr LTCG'}
              </button>
            );
          })}
        </div>
      )}

      {/* Chart Canvas */}
      <div className="h-[360px] sm:h-[420px] w-full pt-2">
        <ResponsiveContainer width="100%" height="100%">
          {chartMode === 'wealth' ? (
            <AreaChart data={chartData} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
              <defs>
                <linearGradient id="gradientHold" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={simulation.strategies.hold.color} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={simulation.strategies.hold.color} stopOpacity={0.0} />
                </linearGradient>
                <linearGradient id="gradientSp500" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={simulation.strategies.sp500.color} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={simulation.strategies.sp500.color} stopOpacity={0.0} />
                </linearGradient>
              </defs>

              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
              <XAxis
                dataKey="quarter"
                stroke="#64748b"
                fontSize={11}
                tickLine={false}
                axisLine={{ stroke: '#334155' }}
              />
              <YAxis
                stroke="#64748b"
                fontSize={11}
                tickLine={false}
                axisLine={{ stroke: '#334155' }}
                tickFormatter={val => `$${val >= 1000000 ? `${(val / 1000000).toFixed(1)}M` : `${(val / 1000).toFixed(0)}k`}`}
              />
              <Tooltip content={<CustomTooltip />} />

              {activeStrategies.hold && (
                <Area
                  type="monotone"
                  dataKey="hold"
                  name={`Pure Hold (${simulation.config.symbol})`}
                  stroke={simulation.strategies.hold.color}
                  strokeWidth={2.5}
                  fill="url(#gradientHold)"
                />
              )}

              {activeStrategies.sp500 && (
                <Area
                  type="monotone"
                  dataKey="sp500"
                  name="S&P 500 (SPY)"
                  stroke={simulation.strategies.sp500.color}
                  strokeWidth={3}
                  fill="url(#gradientSp500)"
                />
              )}

              {activeStrategies.nasdaq && (
                <Area
                  type="monotone"
                  dataKey="nasdaq"
                  name="Nasdaq 100 (QQQ)"
                  stroke={simulation.strategies.nasdaq.color}
                  strokeWidth={2}
                  fill="none"
                />
              )}

              {activeStrategies.sector && (
                <Area
                  type="monotone"
                  dataKey="sector"
                  name={`Sector ETF (${simulation.config.sectorEtf})`}
                  stroke={simulation.strategies.sector.color}
                  strokeWidth={2}
                  fill="none"
                />
              )}

              {activeStrategies.cash && (
                <Area
                  type="monotone"
                  dataKey="cash"
                  name="Cash (HYSA)"
                  stroke={simulation.strategies.cash.color}
                  strokeWidth={2}
                  strokeDasharray="4 4"
                  fill="none"
                />
              )}

              {activeStrategies.hybrid5050 && (
                <Area
                  type="monotone"
                  dataKey="hybrid5050"
                  name="Hybrid 50/50 Strategy"
                  stroke={simulation.strategies.hybrid5050.color}
                  strokeWidth={2.5}
                  fill="none"
                />
              )}
            </AreaChart>
          ) : (
            <LineChart data={chartData} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
              <XAxis
                dataKey="quarter"
                stroke="#64748b"
                fontSize={11}
                tickLine={false}
                axisLine={{ stroke: '#334155' }}
              />
              <YAxis
                stroke="#64748b"
                fontSize={11}
                tickLine={false}
                axisLine={{ stroke: '#334155' }}
                tickFormatter={val => `${val > 0 ? '+' : ''}${val}%`}
              />
              <Tooltip content={<CustomTooltip />} />

              <Line
                type="monotone"
                dataKey="stockPricePct"
                name={`${simulation.config.symbol} Stock Price %`}
                stroke={simulation.strategies.hold.color}
                strokeWidth={3}
                dot={{ r: 3, fill: simulation.strategies.hold.color }}
              />

              <Line
                type="monotone"
                dataKey="sp500PricePct"
                name="S&P 500 Index %"
                stroke={simulation.strategies.sp500.color}
                strokeWidth={3}
                dot={{ r: 3, fill: simulation.strategies.sp500.color }}
              />
            </LineChart>
          )}
        </ResponsiveContainer>
      </div>
    </div>
  );
};
