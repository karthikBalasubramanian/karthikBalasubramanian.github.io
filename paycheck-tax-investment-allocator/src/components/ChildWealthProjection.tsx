import React, { useMemo, useState } from 'react';
import { UserFinancialInputs, TaxBreakdownResult } from '../types';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';
import { Baby, GraduationCap, TrendingUp, Sparkles, Award, ShieldCheck, DollarSign } from 'lucide-react';

interface ChildWealthProjectionProps {
  inputs: UserFinancialInputs;
  taxResult: TaxBreakdownResult;
}

export const ChildWealthProjection: React.FC<ChildWealthProjectionProps> = ({
  inputs,
  taxResult,
}) => {
  const [returnRate, setReturnRate] = useState<number>(7); // 7% standard market return

  const biweeklyChildContribution =
    taxResult.plan529Biweekly +
    taxResult.custodialAccountBiweekly +
    taxResult.trumpAccountBiweekly +
    taxResult.custodialIraBiweekly;

  const annualChildContribution = biweeklyChildContribution * 26;

  // Compute 18-year projection timeline
  const projectionData = useMemo(() => {
    const data = [];
    let cumulativeContribution = 0;
    let balance529 = 0;
    let balanceCustodial = 0;
    let balanceTrump = 0;
    let balanceCustodialIra = 0;

    const r = returnRate / 100;

    for (let year = 0; year <= 18; year++) {
      if (year === 0) {
        data.push({
          year: `Age ${year}`,
          contributions: 0,
          totalValue: 0,
          val529: 0,
          valCustodial: 0,
          valTrump: 0,
          valIra: 0,
        });
      } else {
        const annual529 = taxResult.plan529Biweekly * 26;
        const annualCustodial = taxResult.custodialAccountBiweekly * 26;
        const annualTrump = taxResult.trumpAccountBiweekly * 26;
        const annualIra = taxResult.custodialIraBiweekly * 26;

        balance529 = (balance529 + annual529) * (1 + r);
        balanceCustodial = (balanceCustodial + annualCustodial) * (1 + r);
        balanceTrump = (balanceTrump + annualTrump) * (1 + r);
        balanceCustodialIra = (balanceCustodialIra + annualIra) * (1 + r);

        cumulativeContribution += annualChildContribution;
        const totalValue = balance529 + balanceCustodial + balanceTrump + balanceCustodialIra;

        data.push({
          year: `Age ${year}`,
          contributions: Math.round(cumulativeContribution),
          totalValue: Math.round(totalValue),
          val529: Math.round(balance529),
          valCustodial: Math.round(balanceCustodial),
          valTrump: Math.round(balanceTrump),
          valIra: Math.round(balanceCustodialIra),
        });
      }
    }
    return data;
  }, [
    taxResult.plan529Biweekly,
    taxResult.custodialAccountBiweekly,
    taxResult.trumpAccountBiweekly,
    taxResult.custodialIraBiweekly,
    returnRate,
    annualChildContribution,
  ]);

  const finalYearData = projectionData[18] || { totalValue: 0, contributions: 0 };
  const totalInterestEarned = Math.max(0, finalYearData.totalValue - finalYearData.contributions);

  const fmt = (val: number) =>
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(val);

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-2xl text-slate-100 space-y-5">
      
      {/* Top Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2">
          <div className="p-2 rounded-xl bg-teal-950 text-teal-400 border border-teal-800/50">
            <Baby className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-bold text-white">Child Wealth & Education Growth (18-Year Horizon)</h3>
            <p className="text-xs text-slate-400">
              Biweekly Child Allocation: <span className="text-teal-300 font-bold font-mono">{fmt(biweeklyChildContribution)}</span>/bw (${annualChildContribution.toLocaleString()}/yr)
            </p>
          </div>
        </div>

        {/* Return Rate Selector */}
        <div className="flex items-center gap-2 bg-slate-950 p-1.5 rounded-xl border border-slate-800 text-xs">
          <span className="text-slate-400">Est. Annual Return:</span>
          {[5, 7, 9].map((rate) => (
            <button
              key={rate}
              onClick={() => setReturnRate(rate)}
              className={`px-2.5 py-0.5 rounded-lg font-bold font-mono transition-all ${
                returnRate === rate
                  ? 'bg-teal-600 text-white shadow'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {rate}%
            </button>
          ))}
        </div>
      </div>

      {/* Highlights Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        
        <div className="bg-slate-950 border border-slate-800 rounded-xl p-3.5 space-y-1">
          <span className="text-xs font-semibold text-slate-400">Total Out-of-Pocket Contribution</span>
          <div className="text-xl font-bold font-mono text-slate-200">
            {fmt(finalYearData.contributions)}
          </div>
          <p className="text-[10px] text-slate-500">Principal saved over 18 years</p>
        </div>

        <div className="bg-slate-950 border border-teal-900/60 rounded-xl p-3.5 space-y-1">
          <span className="text-xs font-semibold text-teal-300 flex items-center gap-1">
            <Sparkles className="w-3.5 h-3.5 text-yellow-400" /> Compound Interest Earned
          </span>
          <div className="text-xl font-bold font-mono text-teal-300">
            +{fmt(totalInterestEarned)}
          </div>
          <p className="text-[10px] text-slate-400">Growth from compounding returns</p>
        </div>

        <div className="bg-gradient-to-br from-teal-950/80 to-slate-950 border border-teal-500/50 rounded-xl p-3.5 space-y-1">
          <span className="text-xs font-bold text-emerald-300 uppercase tracking-wider">
            Projected Child Portfolio at Age 18
          </span>
          <div className="text-2xl font-black font-mono text-emerald-400">
            {fmt(finalYearData.totalValue)}
          </div>
          <p className="text-[10px] text-emerald-300">Ready for College, House Down Payment, or Retirement!</p>
        </div>

      </div>

      {/* Growth Chart */}
      <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 space-y-2">
        <div className="flex justify-between items-center text-xs text-slate-400">
          <span>Child Wealth Growth Trajectory (Age 0 to 18)</span>
          <span className="font-mono text-slate-300">Assuming {returnRate}% compounding</span>
        </div>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={projectionData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="colorTotal" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#14b8a6" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#14b8a6" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="colorPrincipal" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="year" stroke="#64748b" tick={{ fontSize: 11 }} />
              <YAxis
                stroke="#64748b"
                tick={{ fontSize: 11 }}
                tickFormatter={(val) => `$${(val / 1000).toFixed(0)}k`}
              />
              <Tooltip
                contentStyle={{ backgroundColor: '#020617', borderColor: '#334155', borderRadius: '12px' }}
                itemStyle={{ fontSize: '12px' }}
                formatter={(val: any) => [fmt(val as number), '']}
              />
              <Area
                type="monotone"
                dataKey="totalValue"
                name="Projected Total Wealth"
                stroke="#14b8a6"
                strokeWidth={3}
                fillOpacity={1}
                fill="url(#colorTotal)"
              />
              <Area
                type="monotone"
                dataKey="contributions"
                name="Principal Contributions"
                stroke="#6366f1"
                strokeWidth={2}
                fillOpacity={1}
                fill="url(#colorPrincipal)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

    </div>
  );
};
