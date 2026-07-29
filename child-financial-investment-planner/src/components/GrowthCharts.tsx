import React, { useState } from 'react';
import { YearProjectionRow, ParentInputs } from '../types';
import { SCENARIO_RATES } from '../data/accountData';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  BarChart,
  Bar,
} from 'recharts';
import { TrendingUp, DollarSign, Award, Layers, Sparkles, Filter } from 'lucide-react';

interface GrowthChartsProps {
  projections: YearProjectionRow[];
  inputs: ParentInputs;
}

export const GrowthCharts: React.FC<GrowthChartsProps> = ({ projections, inputs }) => {
  const [maxAgeHorizon, setMaxAgeHorizon] = useState<18 | 30 | 60>(60);
  const [chartMode, setChartMode] = useState<'scenarios' | 'growth_vs_principal' | 'account_breakdown'>('scenarios');

  const filteredProjections = projections.filter((p) => p.age <= maxAgeHorizon);

  const formatCurrency = (val: number) =>
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(val);

  const age18Row = projections.find((p) => p.age === 18) || projections[projections.length - 1];
  const age60Row = projections.find((p) => p.age === 60) || projections[projections.length - 1];

  // Prepare data for Recharts
  const chartData = filteredProjections.map((p) => ({
    age: `Age ${p.age}`,
    numericAge: p.age,
    year: p.year,
    TotalContributed: p.totalContributed,
    Conservative: p.conservativeBalance,
    Moderate: p.moderateBalance,
    Optimistic: p.optimisticBalance,
    ModerateGrowth: p.moderateGrowth,
    Plan529: p.accountBalances['529_plan']?.moderate || 0,
    TrumpAccount: p.accountBalances['trump_account']?.moderate || 0,
    CustodialRoth: p.accountBalances['custodial_roth_ira']?.moderate || 0,
    UTMA: p.accountBalances['utma_ugma']?.moderate || 0,
    TaxSavings: p.taxSavingsEstimate,
  }));

  return (
    <div className="space-y-8">
      {/* Top Controls & Milestone Cards */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 shadow-xs">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-100 dark:border-slate-800 pb-4">
          <div>
            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-indigo-50 text-indigo-700 dark:bg-indigo-950/60 dark:text-indigo-300 border border-indigo-100 dark:border-indigo-900/50 text-[11px] font-bold uppercase tracking-wider mb-2">
              <TrendingUp className="w-3.5 h-3.5 text-indigo-600 dark:text-indigo-400" /> Growth Trajectory Engine
            </div>
            <h2 className="text-xl sm:text-2xl font-bold text-slate-900 dark:text-white tracking-tight">
              Growth Potential: Conservative, Moderate &amp; Optimistic
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Visualize compound interest trajectories from Age {inputs.childCurrentAge} through Age {maxAgeHorizon}
            </p>
          </div>

          {/* Age Horizon Selectors */}
          <div className="flex items-center gap-1.5 bg-slate-100 dark:bg-slate-800 p-1 rounded-xl">
            <span className="text-[11px] text-slate-400 font-semibold px-2 uppercase tracking-wider">Timeline:</span>
            <button
              id="horizon-18"
              onClick={() => setMaxAgeHorizon(18)}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                maxAgeHorizon === 18
                  ? 'bg-indigo-600 text-white shadow-xs'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900'
              }`}
            >
              Age 0-18 (Childhood)
            </button>
            <button
              id="horizon-30"
              onClick={() => setMaxAgeHorizon(30)}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                maxAgeHorizon === 30
                  ? 'bg-indigo-600 text-white shadow-xs'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900'
              }`}
            >
              Age 0-30 (Young Adult)
            </button>
            <button
              id="horizon-60"
              onClick={() => setMaxAgeHorizon(60)}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                maxAgeHorizon === 60
                  ? 'bg-indigo-600 text-white shadow-xs'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900'
              }`}
            >
              Age 0-60 (Retirement)
            </button>
          </div>
        </div>

        {/* Milestone Summary Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-6">
          <div className="bg-slate-50 dark:bg-slate-800/40 p-4 rounded-xl border border-slate-200 dark:border-slate-800">
            <div className="text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
              Conservative (5.0% CAGR)
            </div>
            <div className="text-xl font-bold font-mono text-slate-900 dark:text-white mt-1">
              {formatCurrency(
                maxAgeHorizon === 18 ? age18Row.conservativeBalance : age60Row.conservativeBalance
              )}
            </div>
            <p className="text-[11px] text-slate-400 mt-0.5">Bond &amp; stable index returns</p>
          </div>

          <div className="bg-indigo-50/80 dark:bg-indigo-950/40 p-4 rounded-xl border border-indigo-100 dark:border-indigo-900/50">
            <div className="text-[11px] font-bold text-indigo-700 dark:text-indigo-400 uppercase tracking-wider">
              Moderate Baseline (7.5% CAGR)
            </div>
            <div className="text-xl font-extrabold font-mono text-indigo-900 dark:text-indigo-300 mt-1">
              {formatCurrency(
                maxAgeHorizon === 18 ? age18Row.moderateBalance : age60Row.moderateBalance
              )}
            </div>
            <p className="text-[11px] text-indigo-600 dark:text-indigo-400 mt-0.5">Balanced S&amp;P 500 index portfolio</p>
          </div>

          <div className="bg-slate-50 dark:bg-slate-800/40 p-4 rounded-xl border border-slate-200 dark:border-slate-800">
            <div className="text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
              Optimistic (10.0% CAGR)
            </div>
            <div className="text-xl font-bold font-mono text-slate-900 dark:text-white mt-1">
              {formatCurrency(
                maxAgeHorizon === 18 ? age18Row.optimisticBalance : age60Row.optimisticBalance
              )}
            </div>
            <p className="text-[11px] text-slate-400 mt-0.5">Historical U.S. stock average</p>
          </div>
        </div>
      </div>

      {/* Chart View Mode Controls */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <button
            id="chart-mode-scenarios"
            onClick={() => setChartMode('scenarios')}
            className={`px-3.5 py-2 rounded-xl text-xs font-bold transition-all flex items-center gap-1.5 ${
              chartMode === 'scenarios'
                ? 'bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900 shadow-xs'
                : 'bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-400 border border-slate-200 dark:border-slate-800'
            }`}
          >
            <TrendingUp className="w-3.5 h-3.5" /> Multi-Scenario Growth
          </button>
          <button
            id="chart-mode-growth"
            onClick={() => setChartMode('growth_vs_principal')}
            className={`px-3.5 py-2 rounded-xl text-xs font-bold transition-all flex items-center gap-1.5 ${
              chartMode === 'growth_vs_principal'
                ? 'bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900 shadow-xs'
                : 'bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-400 border border-slate-200 dark:border-slate-800'
            }`}
          >
            <DollarSign className="w-3.5 h-3.5" /> Contributions vs. Interest
          </button>
          <button
            id="chart-mode-accounts"
            onClick={() => setChartMode('account_breakdown')}
            className={`px-3.5 py-2 rounded-xl text-xs font-bold transition-all flex items-center gap-1.5 ${
              chartMode === 'account_breakdown'
                ? 'bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900 shadow-xs'
                : 'bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-400 border border-slate-200 dark:border-slate-800'
            }`}
          >
            <Layers className="w-3.5 h-3.5" /> Account Type Breakdown
          </button>
        </div>
      </div>

      {/* Recharts Container */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 shadow-xs">
        <div className="h-[420px] w-full">
          {chartMode === 'scenarios' && (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} margin={{ top: 10, right: 30, left: 20, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorCons" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.0} />
                  </linearGradient>
                  <linearGradient id="colorMod" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0.0} />
                  </linearGradient>
                  <linearGradient id="colorOpt" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0.0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                <XAxis dataKey="age" tick={{ fontSize: 11 }} />
                <YAxis
                  tickFormatter={(val) => `$${(val / 1000).toFixed(0)}k`}
                  tick={{ fontSize: 11 }}
                />
                <Tooltip
                  formatter={(value: any) => [formatCurrency(Number(value)), '']}
                  contentStyle={{ backgroundColor: '#0f172a', borderRadius: '12px', border: 'none', color: '#fff', fontSize: '12px' }}
                />
                <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                <Area
                  type="monotone"
                  dataKey="Optimistic"
                  stroke="#8b5cf6"
                  strokeWidth={2.5}
                  fillOpacity={1}
                  fill="url(#colorOpt)"
                  name="Optimistic (10.0%)"
                />
                <Area
                  type="monotone"
                  dataKey="Moderate"
                  stroke="#10b981"
                  strokeWidth={3}
                  fillOpacity={1}
                  fill="url(#colorMod)"
                  name="Moderate (7.5%)"
                />
                <Area
                  type="monotone"
                  dataKey="Conservative"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  fillOpacity={1}
                  fill="url(#colorCons)"
                  name="Conservative (5.0%)"
                />
                <Area
                  type="monotone"
                  dataKey="TotalContributed"
                  stroke="#64748b"
                  strokeWidth={2}
                  strokeDasharray="4 4"
                  fill="none"
                  name="Total Principal Invested"
                />
              </AreaChart>
            </ResponsiveContainer>
          )}

          {chartMode === 'growth_vs_principal' && (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} margin={{ top: 10, right: 30, left: 20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                <XAxis dataKey="age" tick={{ fontSize: 11 }} />
                <YAxis
                  tickFormatter={(val) => `$${(val / 1000).toFixed(0)}k`}
                  tick={{ fontSize: 11 }}
                />
                <Tooltip
                  formatter={(value: any) => [formatCurrency(Number(value)), '']}
                  contentStyle={{ backgroundColor: '#0f172a', borderRadius: '12px', border: 'none', color: '#fff', fontSize: '12px' }}
                />
                <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                <Area
                  type="monotone"
                  dataKey="TotalContributed"
                  stackId="1"
                  stroke="#3b82f6"
                  fill="#3b82f6"
                  name="Parent Contributions ($ Principal)"
                />
                <Area
                  type="monotone"
                  dataKey="ModerateGrowth"
                  stackId="1"
                  stroke="#10b981"
                  fill="#10b981"
                  name="Compounded Interest & Growth ($)"
                />
              </AreaChart>
            </ResponsiveContainer>
          )}

          {chartMode === 'account_breakdown' && (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 30, left: 20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                <XAxis dataKey="age" tick={{ fontSize: 11 }} />
                <YAxis
                  tickFormatter={(val) => `$${(val / 1000).toFixed(0)}k`}
                  tick={{ fontSize: 11 }}
                />
                <Tooltip
                  formatter={(value: any) => [formatCurrency(Number(value)), '']}
                  contentStyle={{ backgroundColor: '#0f172a', borderRadius: '12px', border: 'none', color: '#fff', fontSize: '12px' }}
                />
                <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                <Bar dataKey="Plan529" stackId="a" fill="#10b981" name="529 Plan" />
                <Bar dataKey="TrumpAccount" stackId="a" fill="#f59e0b" name="Trump Account" />
                <Bar dataKey="CustodialRoth" stackId="a" fill="#8b5cf6" name="Custodial Roth IRA" />
                <Bar dataKey="UTMA" stackId="a" fill="#3b82f6" name="UTMA/UGMA" />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>
    </div>
  );
};
