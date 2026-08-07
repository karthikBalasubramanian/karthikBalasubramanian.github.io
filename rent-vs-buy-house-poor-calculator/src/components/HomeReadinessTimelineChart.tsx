import React from 'react';
import type { UserHousingInputs, HousePoorAnalysis } from '../types';
import { ResponsiveContainer, ComposedChart, Bar, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend, ReferenceLine } from 'recharts';
import { Target, Sparkles, CheckCircle2 } from 'lucide-react';

interface HomeReadinessTimelineChartProps {
  inputs: UserHousingInputs;
  analysis: HousePoorAnalysis;
}

export const HomeReadinessTimelineChart: React.FC<HomeReadinessTimelineChartProps> = ({ inputs, analysis }) => {
  const data = analysis.readinessTimeline;
  const rainyDayTarget = analysis.rainyDayBufferTarget;
  const raisePct = inputs.annualSalaryRaisePercent ?? 3.0;

  const fmt = (val: number) =>
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(val);

  const readyPoint = data.find((p) => p.isReadyToBuy) || data[data.length - 1];

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl text-slate-100 space-y-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            <Target className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <span>Homeownership Readiness & Cashflow Roadmap</span>
              <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded-full bg-emerald-950 text-emerald-300 border border-emerald-800">
                {raisePct}% Annual Salary Raise
              </span>
            </h3>
            <p className="text-xs text-slate-400">
              Projects monthly cashflow surplus over time as salary grows & down payment increases while renting.
            </p>
          </div>
        </div>

        {readyPoint && (
          <div className="flex items-center gap-2 bg-emerald-950/80 border border-emerald-700/60 p-2.5 rounded-xl font-mono text-xs text-emerald-200">
            <Sparkles className="w-4 h-4 text-emerald-400 shrink-0" />
            <div>
              <span className="text-[10px] text-emerald-400 font-sans block font-bold">Target Readiness Year</span>
              <span className="font-extrabold text-sm text-white">Year {readyPoint.year} ({readyPoint.calendarYear})</span>
            </div>
          </div>
        )}
      </div>

      {/* Chart */}
      <div className="h-72 w-full pt-2">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 15, right: 30, left: 10, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              dataKey="calendarYear"
              stroke="#94a3b8"
              fontSize={11}
              tickFormatter={(v, idx) => `Yr ${idx} (${v})`}
            />
            <YAxis
              stroke="#94a3b8"
              fontSize={11}
              tickFormatter={(v) => `$${(v / 1000).toFixed(1)}k`}
            />
            <Tooltip
              contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '12px', fontSize: '12px' }}
              formatter={(val: any, name: any) => [fmt(Number(val) || 0), String(name || '')]}
              labelFormatter={(label) => `Calendar Year ${label}`}
            />
            <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
            
            <ReferenceLine
              y={rainyDayTarget}
              stroke="#eab308"
              strokeDasharray="4 4"
              label={{ value: `Rainy Day Target (+$${rainyDayTarget}/mo)`, fill: '#eab308', fontSize: 11, position: 'top' }}
            />

            <Bar
              dataKey="monthlyTakeHome"
              name="Monthly Net Take-Home (3% Raises)"
              fill="#38bdf8"
              opacity={0.35}
              barSize={24}
              radius={[4, 4, 0, 0]}
            />
            <Line
              type="monotone"
              dataKey="monthlyPiti"
              name="Monthly PITI Housing Cost"
              stroke="#f43f5e"
              strokeWidth={2.5}
              dot={{ r: 3, fill: '#f43f5e' }}
            />
            <Line
              type="monotone"
              dataKey="monthlyCashflowSurplus"
              name="Monthly Cashflow Surplus (Take Home - PITI - Lifestyle)"
              stroke="#10b981"
              strokeWidth={3.5}
              dot={{ r: 5, fill: '#10b981' }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Insight Banner */}
      <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 text-xs text-slate-300 flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
          <span>
            {analysis.isReadyToBuyToday
              ? `You are ready to buy TODAY with a +$${Math.round(analysis.leftoverCashBufferBuy).toLocaleString()}/mo cashflow surplus exceeding your $${rainyDayTarget}/mo target!`
              : `Renting today while growing salary (+3%/yr) allows you to comfortably buy in ${readyPoint.calendarYear} with a +$${Math.round(readyPoint.monthlyCashflowSurplus).toLocaleString()}/mo surplus!`}
          </span>
        </div>
        <div className="text-slate-400 font-mono text-[11px]">
          Down Payment Growth: <span className="text-emerald-300 font-bold">${Math.round(readyPoint.totalDownPaymentSaved).toLocaleString()}</span>
        </div>
      </div>
    </div>
  );
};
