import React from 'react';
import type { UserHousingInputs } from '../types';
import { generateNetWorthComparison } from '../utils/calculator';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from 'recharts';
import { TrendingUp, CheckCircle2 } from 'lucide-react';

interface RentVsBuyComparisonChartProps {
  inputs: UserHousingInputs;
}

export const RentVsBuyComparisonChart: React.FC<RentVsBuyComparisonChartProps> = ({ inputs }) => {
  const data = generateNetWorthComparison(inputs);

  const fmt = (val: number) =>
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(val);

  const year10Buy = data[data.length - 1]?.buyNetWorth || 0;
  const year10Rent = data[data.length - 1]?.rentAndInvestNetWorth || 0;
  const diff = year10Rent - year10Buy;

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl text-slate-100 space-y-4">
      
      {/* Title */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            <TrendingUp className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <span>10-Year Net Worth Trajectory: Buying vs Renting & Investing</span>
            </h3>
            <p className="text-xs text-slate-400">
              Compares 10-year wealth of Homeownership equity vs Renting + investing monthly cash savings into S&P 500 (7% return).
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 text-xs font-mono font-bold bg-slate-950 p-2.5 rounded-xl border border-slate-800">
          <div>
            <span className="text-[10px] text-emerald-400 font-sans block">10-Yr Buy Equity</span>
            <span>{fmt(year10Buy)}</span>
          </div>
          <span className="text-slate-700">|</span>
          <div>
            <span className="text-[10px] text-cyan-400 font-sans block">10-Yr Rent & Invest</span>
            <span>{fmt(year10Rent)}</span>
          </div>
        </div>
      </div>

      {/* Recharts Line Chart */}
      <div className="h-72 w-full pt-2">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 10, right: 30, left: 10, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              dataKey="year"
              stroke="#94a3b8"
              fontSize={11}
              tickFormatter={(v) => `Yr ${v}`}
            />
            <YAxis
              stroke="#94a3b8"
              fontSize={11}
              tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
            />
            <Tooltip
              contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '12px', fontSize: '12px' }}
              formatter={(val: any) => [fmt(Number(val) || 0), 'Net Worth']}
              labelFormatter={(label) => `Year ${label}`}
            />
            <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
            <Line
              type="monotone"
              dataKey="buyNetWorth"
              name="Homeownership Net Equity (4% Apprec.)"
              stroke="#10b981"
              strokeWidth={3}
              dot={{ r: 4, fill: '#10b981' }}
            />
            <Line
              type="monotone"
              dataKey="rentAndInvestNetWorth"
              name="Renting & Investing Savings (7% S&P500)"
              stroke="#06b6d4"
              strokeWidth={3}
              dot={{ r: 4, fill: '#06b6d4' }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Explanatory Insight */}
      <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 text-xs text-slate-300 flex items-center gap-2">
        <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
        <span>
          {diff > 0
            ? `Renting and investing the monthly cash savings yields +${fmt(diff)} more in liquid net worth over 10 years without property tax or maintenance risk!`
            : `Buying this home builds +${fmt(-diff)} more net wealth over 10 years through principal paydown and home appreciation!`}
        </span>
      </div>

    </div>
  );
};
