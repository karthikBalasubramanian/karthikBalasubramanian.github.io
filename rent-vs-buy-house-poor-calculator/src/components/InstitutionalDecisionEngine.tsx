import React from 'react';
import type { UserHousingInputs, HousePoorAnalysis } from '../types';
import { Landmark, Dices, PiggyBank, Scale, TrendingUp, Info } from 'lucide-react';

interface InstitutionalDecisionEngineProps {
  inputs: UserHousingInputs;
  analysis: HousePoorAnalysis;
}

export const InstitutionalDecisionEngine: React.FC<InstitutionalDecisionEngineProps> = ({ analysis }) => {
  const inst = analysis.institutional;

  const fmt = (val: number) =>
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(val);

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-2xl text-slate-100 space-y-6">
      
      {/* Engine Title Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 border-b border-slate-800 pb-4">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-gradient-to-br from-purple-500/20 to-indigo-500/20 text-purple-400 border border-purple-500/30">
            <Landmark className="w-6 h-6" />
          </div>
          <div>
            <h3 className="text-base sm:text-lg font-bold text-white flex items-center gap-2">
              <span>Institutional Quantitative Decision Engine</span>
              <span className="text-xs font-mono font-bold px-2.5 py-0.5 rounded-full bg-purple-950 text-purple-300 border border-purple-800">
                5 Wall-Street Layers
              </span>
            </h3>
            <p className="text-xs text-slate-400">
              Unrecoverable Costs, Dynamic Tax Shields, Crossover Break-Even, NPV Differential, & Monte Carlo Simulations.
            </p>
          </div>
        </div>

        {/* Monte Carlo Confidence Badge */}
        <div className="flex items-center gap-2.5 bg-gradient-to-r from-purple-950 to-indigo-950 border border-purple-700/60 p-3 rounded-xl font-mono text-xs text-purple-200 shrink-0">
          <Dices className="w-5 h-5 text-purple-400 shrink-0" />
          <div>
            <span className="text-[10px] text-purple-300 font-sans block font-bold">Monte Carlo Confidence (1k Sims)</span>
            <span className="font-extrabold text-sm text-emerald-400">{inst.monteCarloConfidenceScore}% Win Rate</span>
          </div>
        </div>
      </div>

      {/* 5 Institutional Layer Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        
        {/* LAYER 1: Unrecoverable Cost Equation */}
        <div className="bg-slate-950 border border-slate-800 rounded-xl p-5 space-y-3.5">
          <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
            <div className="flex items-center gap-2 text-rose-400 font-bold text-sm">
              <Scale className="w-4 h-4 text-rose-400" />
              <span>Layer 1: Unrecoverable Cost Equation</span>
            </div>
            <span className="text-[10px] font-mono bg-rose-950 text-rose-300 px-2 py-0.5 rounded border border-rose-800">
              The "5% Rule"
            </span>
          </div>

          <div className="grid grid-cols-2 gap-2.5 font-mono text-center text-xs">
            <div className="bg-slate-900 border border-slate-800 p-2.5 rounded-lg">
              <span className="text-[10px] text-slate-400 font-sans block">Monthly Unrecoverable Buy</span>
              <span className="font-extrabold text-rose-400 text-base">{fmt(inst.unrecoverableBuyMonthly)}</span>
              <span className="text-[9px] text-slate-500 font-sans block">Interest + Tax + Maint + CapOp</span>
            </div>

            <div className="bg-slate-900 border border-slate-800 p-2.5 rounded-lg">
              <span className="text-[10px] text-slate-400 font-sans block">Monthly Unrecoverable Rent</span>
              <span className="font-extrabold text-amber-300 text-base">{fmt(inst.unrecoverableRentMonthly)}</span>
              <span className="text-[9px] text-slate-500 font-sans block">100% Lost to Rent</span>
            </div>
          </div>

          {/* Simple Explanation Callout */}
          <div className="bg-slate-900/90 border border-indigo-900/60 p-3 rounded-lg space-y-1 text-xs">
            <div className="flex items-center gap-1.5 text-indigo-300 font-bold text-[11px]">
              <Info className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
              <span>Explanation:</span>
            </div>
            <p className="text-[11px] text-slate-300 leading-relaxed font-sans">
              When you pay your mortgage, think of it as two buckets: <strong className="text-rose-300">The Burned Bucket</strong> (interest, tax, repairs) and <strong className="text-emerald-300">Your Piggy Bank</strong> (principal paydown). Renting burns 100% of cash, whereas buying puts <strong className="text-emerald-300">{fmt(inst.principalEquityPaydownMonthly)}/mo</strong> straight back into your own piggy bank!
            </p>
          </div>
        </div>

        {/* LAYER 2: Dynamic Tax Shield Engine */}
        <div className="bg-slate-950 border border-slate-800 rounded-xl p-5 space-y-3.5">
          <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
            <div className="flex items-center gap-2 text-emerald-400 font-bold text-sm">
              <PiggyBank className="w-4 h-4 text-emerald-400" />
              <span>Layer 2: Dynamic Tax Shield Engine</span>
            </div>
            <span className="text-[10px] font-mono bg-emerald-950 text-emerald-300 px-2 py-0.5 rounded border border-emerald-800">
              SALT Cap $10k
            </span>
          </div>

          <div className="grid grid-cols-2 gap-2.5 font-mono text-center text-xs">
            <div className="bg-slate-900 border border-slate-800 p-2.5 rounded-lg">
              <span className="text-[10px] text-slate-400 font-sans block">Monthly IRS Tax Refund</span>
              <span className="font-extrabold text-emerald-400 text-base">+{fmt(inst.taxShieldMonthlyRefund)}/mo</span>
              <span className="text-[9px] text-slate-500 font-sans block">Income Tax Reduction</span>
            </div>

            <div className="bg-slate-900 border border-slate-800 p-2.5 rounded-lg">
              <span className="text-[10px] text-slate-400 font-sans block">Effective After-Tax PITI</span>
              <span className="font-extrabold text-cyan-300 text-base">{fmt(inst.afterTaxMonthlyPiti)}/mo</span>
              <span className="text-[9px] text-slate-500 font-sans block">Real Out-of-Pocket</span>
            </div>
          </div>

          {/* Simple Explanation Callout */}
          <div className="bg-slate-900/90 border border-indigo-900/60 p-3 rounded-lg space-y-1 text-xs">
            <div className="flex items-center gap-1.5 text-indigo-300 font-bold text-[11px]">
              <Info className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
              <span>Explanation:</span>
            </div>
            <p className="text-[11px] text-slate-300 leading-relaxed font-sans">
              Think of the government giving you a monthly coupon for buying a home. Because mortgage interest is tax-deductible, Uncle Sam reduces your income taxes by <strong className="text-emerald-300">+{fmt(inst.taxShieldMonthlyRefund)} every single month</strong>!
            </p>
          </div>
        </div>

        {/* LAYER 3: Crossover Break-Even Horizon (T*) */}
        <div className="bg-slate-950 border border-slate-800 rounded-xl p-5 space-y-3.5">
          <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
            <div className="flex items-center gap-2 text-cyan-400 font-bold text-sm">
              <TrendingUp className="w-4 h-4 text-cyan-400" />
              <span>Layer 3: Crossover Break-Even Horizon (T*)</span>
            </div>
            <span className="text-[10px] font-mono bg-cyan-950 text-cyan-300 px-2 py-0.5 rounded border border-cyan-800">
              Inflation Shield
            </span>
          </div>

          <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-xl text-center space-y-1 font-mono">
            <span className="text-[10px] text-slate-400 font-sans block uppercase font-bold">Break-Even Holding Horizon</span>
            <div className="text-2xl font-extrabold text-cyan-300">
              Year {inst.crossoverBreakEvenYear} <span className="text-xs text-slate-400 font-normal">({new Date().getFullYear() + inst.crossoverBreakEvenYear})</span>
            </div>
            <span className="text-[10px] text-slate-400 font-sans block">
              Point where cumulative buying costs become cheaper than rising rent
            </span>
          </div>

          {/* Simple Explanation Callout */}
          <div className="bg-slate-900/90 border border-indigo-900/60 p-3 rounded-lg space-y-1 text-xs">
            <div className="flex items-center gap-1.5 text-indigo-300 font-bold text-[11px]">
              <Info className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
              <span>Explanation:</span>
            </div>
            <p className="text-[11px] text-slate-300 leading-relaxed font-sans">
              Rent gets more expensive every year as landlords raise prices. Your mortgage principal & interest are locked in forever. In <strong className="text-cyan-300">Year {inst.crossoverBreakEvenYear}</strong>, the seesaw tips — after Year {inst.crossoverBreakEvenYear}, owning is cheaper every month for the rest of your life!
            </p>
          </div>
        </div>

        {/* LAYER 4 & 5: Terminal Net Worth NPV & Monte Carlo Risk Engine */}
        <div className="bg-slate-950 border border-slate-800 rounded-xl p-5 space-y-3.5">
          <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
            <div className="flex items-center gap-2 text-purple-400 font-bold text-sm">
              <Dices className="w-4 h-4 text-purple-400" />
              <span>Layer 4 & 5: NPV Terminal Wealth & Monte Carlo</span>
            </div>
            <span className="text-[10px] font-mono bg-purple-950 text-purple-300 px-2 py-0.5 rounded border border-purple-800">
              1,000 Stochastic Sims
            </span>
          </div>

          <div className="grid grid-cols-2 gap-2.5 font-mono text-center text-xs">
            <div className="bg-slate-900 border border-slate-800 p-2.5 rounded-lg">
              <span className="text-[10px] text-slate-400 font-sans block">10-Yr Buy Net Worth (Post-6% Fee)</span>
              <span className="font-extrabold text-emerald-400 text-base">{fmt(inst.terminalNetWorthBuy10Yr)}</span>
            </div>

            <div className="bg-slate-900 border border-slate-800 p-2.5 rounded-lg">
              <span className="text-[10px] text-slate-400 font-sans block">10-Yr Rent Net Worth (Post-Tax)</span>
              <span className="font-extrabold text-cyan-300 text-base">{fmt(inst.terminalNetWorthRent10Yr)}</span>
            </div>
          </div>

          {/* Simple Explanation Callout */}
          <div className="bg-slate-900/90 border border-indigo-900/60 p-3 rounded-lg space-y-1 text-xs">
            <div className="flex items-center gap-1.5 text-indigo-300 font-bold text-[11px]">
              <Info className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
              <span>Explanation:</span>
            </div>
            <p className="text-[11px] text-slate-300 leading-relaxed font-sans">
              Instead of assuming sunny financial weather, we simulated 1,000 financial weather forecasts (repair shocks, stock market swings). Result: You have an <strong className="text-emerald-300">{inst.monteCarloConfidenceScore}% probability of sun</strong> (outperforming renting) over 10 years!
            </p>
          </div>
        </div>

      </div>

    </div>
  );
};
