import React from 'react';
import type { UserHousingInputs, HousePoorAnalysis } from '../types';
import { Home, CheckCircle2, Calculator } from 'lucide-react';

interface BuyOptimizationEngineProps {
  inputs: UserHousingInputs;
  analysis: HousePoorAnalysis;
  onChange: (updated: Partial<UserHousingInputs>) => void;
}

export const BuyOptimizationEngine: React.FC<BuyOptimizationEngineProps> = ({
  inputs,
  analysis,
  onChange,
}) => {
  const spec = analysis.hedonicSpecMapping;
  const opt = analysis.expenseOptimization;
  const stress = analysis.stressTestMetrics;

  const fmt = (val: number) =>
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(val);

  const applyCategoryTrim = (category: string, trimAmount: number) => {
    const currentCatVal = (inputs.lifestyle as any)[category] || 0;
    const newCatVal = Math.max(0, currentCatVal - trimAmount);
    onChange({
      lifestyle: {
        ...inputs.lifestyle,
        [category]: newCatVal,
      },
    });
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-2xl text-slate-100 space-y-6">
      
      {/* Title Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 border-b border-slate-800 pb-4">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-gradient-to-br from-indigo-500/20 to-purple-500/20 text-indigo-400 border border-indigo-500/30">
            <Calculator className="w-6 h-6" />
          </div>
          <div>
            <h3 className="text-base sm:text-lg font-bold text-white flex items-center gap-2">
              <span>4-Stage Buy-Optimization & Sensitivity Matrix</span>
              <span className="text-xs font-mono font-bold px-2 py-0.5 rounded-full bg-indigo-950 text-indigo-300 border border-indigo-800">
                Multi-Variable Model
              </span>
            </h3>
            <p className="text-xs text-slate-400">
              Combines Reverse PITI Solving, Hedonic Spec Mapping, Expense Optimization, & Stress Testing.
            </p>
          </div>
        </div>
      </div>

      {/* 4-Stage Architecture Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        
        {/* STAGE 1 & 2: Reverse PITI & Hedonic Spec Mapping */}
        <div className="bg-slate-950 border border-slate-800 rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div className="flex items-center gap-2 text-emerald-400 font-bold text-sm">
              <span className="px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 text-xs border border-emerald-800 font-mono">Stage 1 & 2</span>
              <span>Hedonic Property Spec Mapping</span>
            </div>
            <span className="text-xs font-mono text-slate-400">{spec.zipCode} ({spec.cityName})</span>
          </div>

          <div className="grid grid-cols-2 gap-3 text-center font-mono">
            <div className="bg-slate-900 border border-slate-800 p-3 rounded-xl">
              <span className="text-[10px] text-slate-400 font-sans block">Max Affordable Home Price</span>
              <span className="text-xl font-extrabold text-emerald-400 block">{fmt(analysis.maxSafeHomePrice)}</span>
              <span className="text-[9px] text-slate-500 font-sans">Based on +${analysis.rainyDayBufferTarget}/mo buffer</span>
            </div>

            <div className="bg-slate-900 border border-slate-800 p-3 rounded-xl">
              <span className="text-[10px] text-slate-400 font-sans block">Affordable SqFt</span>
              <span className="text-xl font-extrabold text-cyan-400 block">{spec.affordableSqFt.toLocaleString()} sqft</span>
              <span className="text-[9px] text-slate-500 font-sans">@ ${spec.pricePerSqFt}/sqft in {spec.zipCode}</span>
            </div>
          </div>

          <div className="bg-emerald-950/40 border border-emerald-800/60 p-3.5 rounded-xl space-y-1.5 text-xs">
            <div className="flex items-center justify-between text-emerald-200 font-semibold">
              <span className="flex items-center gap-1.5">
                <Home className="w-4 h-4 text-emerald-400" />
                <span>Affordable Layout in ZIP {spec.zipCode}:</span>
              </span>
              <span className="font-mono text-white font-extrabold text-sm">
                {spec.estimatedBeds} Beds • {spec.estimatedBaths} Baths
              </span>
            </div>
            <p className="text-[11px] text-slate-300 leading-relaxed font-sans">
              With your current monthly surplus cash, you can comfortably buy a <strong className="text-emerald-300">{spec.estimatedBeds}-Bed / {spec.estimatedBaths}-Bath home (~{spec.affordableSqFt.toLocaleString()} sqft)</strong> in {spec.cityName} today while preserving a +${analysis.rainyDayBufferTarget}/mo rainy day cushion!
            </p>
          </div>
        </div>

        {/* STAGE 3: Lifestyle Expense Optimization Matrix */}
        <div className="bg-slate-950 border border-slate-800 rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div className="flex items-center gap-2 text-indigo-400 font-bold text-sm">
              <span className="px-2 py-0.5 rounded bg-indigo-950 text-indigo-300 text-xs border border-indigo-800 font-mono">Stage 3</span>
              <span>Lifestyle Expense Optimization</span>
            </div>
            {opt.monthlyPaymentGap > 0 ? (
              <span className="text-xs font-mono text-rose-400 font-bold">Shortfall: -{fmt(opt.monthlyPaymentGap)}/mo</span>
            ) : (
              <span className="text-xs font-mono text-emerald-400 font-bold">0 Shortfall</span>
            )}
          </div>

          {opt.monthlyPaymentGap > 0 ? (
            <div className="space-y-2 text-xs">
              <p className="text-[11px] text-slate-300">
                To unlock your target <strong className="text-white">{fmt(inputs.targetHomePrice)}</strong> dream home today, trim these lifestyle categories:
              </p>

              <div className="space-y-1.5 font-mono">
                {opt.categoryTrims.map(
                  (trim) =>
                    trim.recommendedTrim > 0 && (
                      <div key={trim.category} className="bg-slate-900 border border-slate-800 p-2.5 rounded-lg flex items-center justify-between flex-wrap gap-2 text-xs">
                        <div>
                          <span className="text-slate-200 font-sans font-semibold block">{trim.label}</span>
                          <span className="text-[10px] text-slate-400">Current: ${trim.currentAmount}/mo $\rightarrow$ Cut -${trim.recommendedTrim}</span>
                        </div>
                        <button
                          onClick={() => applyCategoryTrim(trim.category, trim.recommendedTrim)}
                          className="px-2.5 py-1 text-[10px] font-bold rounded bg-indigo-900/60 text-indigo-200 border border-indigo-700/60 hover:bg-indigo-800 transition-all font-sans"
                        >
                          Trim -${trim.recommendedTrim}/mo
                        </button>
                      </div>
                    )
                )}
              </div>

              <div className="bg-indigo-950/40 border border-indigo-800/60 p-2.5 rounded-lg text-[11px] text-indigo-200 flex items-center justify-between">
                <span>Total Buying Power Unlocked:</span>
                <span className="font-mono font-bold text-emerald-300">+{fmt(opt.totalTrimmed)}/mo cashflow</span>
              </div>
            </div>
          ) : (
            <div className="bg-emerald-950/30 border border-emerald-800/50 p-4 rounded-xl text-center space-y-1 text-xs">
              <CheckCircle2 className="w-6 h-6 text-emerald-400 mx-auto" />
              <span className="font-bold text-emerald-300 block">Zero Payment Shortfall!</span>
              <p className="text-slate-300 text-[11px]">
                Your current monthly income and lifestyle budget easily support your target home price of {fmt(inputs.targetHomePrice)}!
              </p>
            </div>
          )}
        </div>

        {/* STAGE 4: House-Poor Stress Tester & Reserve Buffer Gauges */}
        <div className="md:col-span-2 bg-slate-950 border border-slate-800 rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div className="flex items-center gap-2 text-amber-400 font-bold text-sm">
              <span className="px-2 py-0.5 rounded bg-amber-950 text-amber-300 text-xs border border-amber-800 font-mono">Stage 4</span>
              <span>House-Poor Risk Matrix & Stress Tester</span>
            </div>
            <span className={`text-xs font-mono font-bold px-2 py-0.5 rounded ${
              stress.riskLevel === 'house_poor'
                ? 'bg-rose-950 text-rose-400 border border-rose-800'
                : stress.riskLevel === 'moderate'
                ? 'bg-amber-950 text-amber-400 border border-amber-800'
                : 'bg-emerald-950 text-emerald-400 border border-emerald-800'
            }`}>
              {stress.riskLabel}
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs font-mono">
            
            {/* Housing Expense Ratio */}
            <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-xl space-y-1">
              <span className="text-[10px] text-slate-400 font-sans block uppercase font-bold">Housing Expense Ratio (HER)</span>
              <div className="text-xl font-extrabold text-white">
                {stress.housingExpenseRatio}% <span className="text-xs text-slate-400 font-normal">of gross</span>
              </div>
              <span className="text-[10px] text-slate-400 font-sans block">
                {stress.housingExpenseRatio <= 28 ? '✓ Benchmark (<28%)' : '⚠️ Above 28% benchmark'}
              </span>
            </div>

            {/* Daily Residue Cash Buffer */}
            <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-xl space-y-1">
              <span className="text-[10px] text-slate-400 font-sans block uppercase font-bold">Monthly Residue Cash Buffer</span>
              <div className={`text-xl font-extrabold ${analysis.leftoverCashBufferBuy >= analysis.rainyDayBufferTarget ? 'text-emerald-400' : 'text-rose-400'}`}>
                {fmt(analysis.leftoverCashBufferBuy)} <span className="text-xs text-slate-400 font-normal">/mo</span>
              </div>
              <span className="text-[10px] text-slate-400 font-sans block">
                Target: +${analysis.rainyDayBufferTarget}/mo
              </span>
            </div>

            {/* Reserve Buffer Duration */}
            <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-xl space-y-1">
              <span className="text-[10px] text-slate-400 font-sans block uppercase font-bold">Annual Cash Cushion Duration</span>
              <div className="text-xl font-extrabold text-cyan-400">
                {stress.reserveBufferMonths} <span className="text-xs text-slate-400 font-normal">months</span>
              </div>
              <span className="text-[10px] text-slate-400 font-sans block">
                Runway from leftover monthly cashflow
              </span>
            </div>

          </div>
        </div>

      </div>

      {/* Expandable Financial Engineering Methodology & Mathematical Formulas Section */}
      <div className="border-t border-slate-800 pt-4">
        <details className="group bg-slate-950 border border-slate-800 rounded-xl overflow-hidden transition-all">
          <summary className="flex items-center justify-between p-4 cursor-pointer select-none text-xs font-bold text-slate-300 hover:text-white hover:bg-slate-900/60 transition-all">
            <div className="flex items-center gap-2">
              <span className="p-1.5 rounded-lg bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 font-mono text-[10px]">f(x)</span>
              <span>Financial Engineering Methodology & Mathematical Formulas</span>
            </div>
            <span className="text-indigo-400 group-open:rotate-180 transition-transform font-mono">▼</span>
          </summary>

          <div className="p-5 border-t border-slate-800 text-xs text-slate-300 space-y-4 font-sans bg-slate-950/80 leading-relaxed">
            <p>
              This engine runs a four-stage <strong className="text-white">Multi-Variable Affordability & Sensitivity Model</strong> combining Reverse PITI Solving, Hedonic Spatial Mapping, Greedy Expense Optimization, and Risk Stress Testing.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 font-mono text-[11px]">
              
              {/* Stage 1 Formula */}
              <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-lg space-y-1.5">
                <span className="text-emerald-400 font-bold block font-sans text-xs">Stage 1: Reverse PITI Capacity Solver</span>
                <div className="bg-slate-950 p-2 rounded border border-slate-800 text-emerald-300 overflow-x-auto text-[10px]">
                  V_max = P_PITI / [ (1 - d) · (r(1+r)^n / ((1+r)^n - 1)) + (t_tax + t_ins + t_pmi)/12 ]
                </div>
                <p className="text-[10px] text-slate-400 font-sans">
                  Solves algebraically for max home price V_max given your available surplus cash capacity P_PITI.
                </p>
              </div>

              {/* Stage 2 Formula */}
              <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-lg space-y-1.5">
                <span className="text-cyan-400 font-bold block font-sans text-xs">Stage 2: Hedonic Physical Spec Mapping</span>
                <div className="bg-slate-950 p-2 rounded border border-slate-800 text-cyan-300 overflow-x-auto text-[10px]">
                  Affordable SqFt = V_max / Price_per_SqFt_ZIP
                </div>
                <p className="text-[10px] text-slate-400 font-sans">
                  Maps V_max against target ZIP code median price/sqft to determine exact bedroom & bathroom layout limits.
                </p>
              </div>

              {/* Stage 3 Formula */}
              <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-lg space-y-1.5">
                <span className="text-indigo-400 font-bold block font-sans text-xs">Stage 3: Greedy Expense Optimization</span>
                <div className="bg-slate-950 p-2 rounded border border-slate-800 text-indigo-300 overflow-x-auto text-[10px]">
                  ΔP = PITI(V_target) - P_PITI  ==&gt;  Σ ΔE_i = ΔP
                </div>
                <p className="text-[10px] text-slate-400 font-sans">
                  Calculates monthly payment shortfall ΔP and optimizes non-essential spend categories (dining out, streaming) to bridge the gap.
                </p>
              </div>

              {/* Stage 4 Formula */}
              <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-lg space-y-1.5">
                <span className="text-amber-400 font-bold block font-sans text-xs">Stage 4: House-Poor Stress Tester</span>
                <div className="bg-slate-950 p-2 rounded border border-slate-800 text-amber-300 overflow-x-auto text-[10px]">
                  HER = PITI / Gross_Income,  RB = (Surplus × 12) / Living_Cost
                </div>
                <p className="text-[10px] text-slate-400 font-sans">
                  Evaluates Housing Expense Ratio (HER) and Liquidity Reserve Buffer (RB) to stress test against unexpected financial shocks.
                </p>
              </div>

            </div>
          </div>
        </details>
      </div>

    </div>
  );
};
