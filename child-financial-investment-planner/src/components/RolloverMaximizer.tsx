import React, { useState } from 'react';
import { ParentInputs } from '../types';
import {
  ShieldCheck,
  Zap,
  ArrowRight,
  TrendingUp,
  DollarSign,
  Info,
  Sparkles,
  CheckCircle,
  HelpCircle,
  BookOpen,
  ExternalLink,
} from 'lucide-react';

interface RolloverMaximizerProps {
  inputs: ParentInputs;
}

export const RolloverMaximizer: React.FC<RolloverMaximizerProps> = ({ inputs }) => {
  const [rollover529Amount, setRollover529Amount] = useState<number>(35000);
  const [trumpAge18Balance, setTrumpAge18Balance] = useState<number>(175000);
  const [growthRate, setGrowthRate] = useState<number>(7.5);

  const formatCurrency = (val: number) =>
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(val);

  const yearsToCompounding60 = 60 - 18; // 42 years from age 18 to 60

  // 529 Roth Rollover Growth
  const projected529RothAt60 = rollover529Amount * Math.pow(1 + growthRate / 100, yearsToCompounding60);

  // Trump Account IRA Growth
  const projectedTrumpAt60 = trumpAge18Balance * Math.pow(1 + growthRate / 100, yearsToCompounding60);

  return (
    <div className="space-y-8">
      {/* Intro Header */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 sm:p-8 shadow-xs">
        <div className="max-w-3xl">
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-indigo-50 text-indigo-700 dark:bg-indigo-950/60 dark:text-indigo-300 border border-indigo-100 dark:border-indigo-900/50 text-[11px] font-bold uppercase tracking-wider mb-2">
            <Zap className="w-3.5 h-3.5 text-indigo-600 dark:text-indigo-400" /> SECURE 2.0 &amp; Lifetime IRA Rollover Guide
          </div>
          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900 dark:text-white">
            Maximizing 529 Plans &amp; Trump Accounts to Age 60
          </h2>
          <p className="text-slate-500 dark:text-slate-400 text-xs sm:text-sm mt-1.5 leading-relaxed">
            Unused 529 funds no longer get wasted or penalized. Learn how SECURE 2.0 allows up to <strong className="font-mono text-slate-800 dark:text-slate-200">$35,000</strong> to roll over tax-free into a Roth IRA, and how Trump Account funds compound from age 18 to 60 into a nest egg.
          </p>
        </div>
      </div>

      {/* Global Rate Selector */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-4 rounded-xl text-xs shadow-xs">
        <span className="font-bold text-slate-700 dark:text-slate-300 flex items-center gap-1.5 text-[11px] uppercase tracking-wider text-slate-400">
          <TrendingUp className="w-4 h-4 text-indigo-600" /> Compounding Interest Rate (Age 18 to 60):
        </span>
        <div className="flex items-center gap-2">
          <button
            id="rate-5"
            onClick={() => setGrowthRate(5.0)}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${growthRate === 5 ? 'bg-indigo-600 text-white' : 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300'}`}
          >
            5.0% (Conservative)
          </button>
          <button
            id="rate-7.5"
            onClick={() => setGrowthRate(7.5)}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${growthRate === 7.5 ? 'bg-indigo-600 text-white' : 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300'}`}
          >
            7.5% (Moderate)
          </button>
          <button
            id="rate-10"
            onClick={() => setGrowthRate(10.0)}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${growthRate === 10 ? 'bg-indigo-600 text-white' : 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300'}`}
          >
            10.0% (Optimistic)
          </button>
        </div>
      </div>

      {/* Section 1: 529 Maximization Strategies */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 sm:p-8 shadow-xs space-y-6">
        <div className="flex items-start justify-between gap-4 border-b border-slate-100 dark:border-slate-800 pb-4">
          <div>
            <div className="inline-flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-wider text-indigo-600 dark:text-indigo-400">
              <ShieldCheck className="w-4 h-4" /> Strategy 1: SECURE 2.0 529-to-Roth IRA Rollover
            </div>
            <h3 className="text-xl font-bold text-slate-900 dark:text-white mt-1">
              How to Maximize a 529 Plan Without Risking Overfunding Penalties
            </h3>
          </div>
          <span className="px-3 py-1 rounded-full bg-indigo-50 text-indigo-700 dark:bg-indigo-950/60 text-[11px] font-mono font-bold border border-indigo-100">
            $35,000 Lifetime Cap
          </span>
        </div>

        {/* Interactive 529 Rollover Calculator */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 bg-slate-50 dark:bg-slate-800/30 p-6 rounded-xl border border-slate-200 dark:border-slate-800">
          <div className="space-y-4">
            <label className="text-xs font-bold text-slate-700 dark:text-slate-300 block">
              Rollover Amount from 529 to Roth IRA (Max $35,000)
            </label>
            <input
              id="529-rollover-slider"
              type="range"
              min="5000"
              max="35000"
              step="1000"
              value={rollover529Amount}
              onChange={(e) => setRollover529Amount(Number(e.target.value))}
              className="w-full accent-indigo-600 cursor-pointer"
            />
            <div className="flex justify-between text-xs font-mono font-bold text-slate-900 dark:text-white">
              <span>Selected Rollover: ${rollover529Amount.toLocaleString()}</span>
              <span>Max: $35,000</span>
            </div>
            <p className="text-[11px] text-slate-500">
              Transferred directly into child's Roth IRA starting at age 18/22 (subject to annual IRA limit, e.g., $7k/yr for 5 years).
            </p>
          </div>

          <div className="bg-slate-900 text-white p-5 rounded-xl border border-slate-800 flex flex-col justify-between">
            <div>
              <div className="text-[10px] text-indigo-300 font-bold uppercase tracking-wider">
                Value at Age 60 (100% Tax-Free)
              </div>
              <div className="text-3xl font-extrabold font-mono text-indigo-400 mt-1">
                {formatCurrency(projected529RothAt60)}
              </div>
              <p className="text-xs text-slate-400 mt-1">
                ${rollover529Amount.toLocaleString()} rolled over at age 18 growing at {growthRate}% CAGR for 42 years!
              </p>
            </div>
            <div className="text-[11px] text-indigo-300 mt-4 border-t border-slate-800 pt-2 font-mono">
              ✓ Zero income tax upon withdrawal at age 60
            </div>
          </div>
        </div>

        {/* 529 Maximization Checklist */}
        <div className="space-y-3 pt-2">
          <h4 className="text-sm font-bold text-slate-900 dark:text-white">
            Top 5 Ways to Fully Maximize Your 529 Plan:
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
            <div className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-800 space-y-1">
              <div className="font-bold text-slate-900 dark:text-white flex items-center gap-1.5">
                <CheckCircle className="w-4 h-4 text-emerald-600" /> 1. Execute SECURE 2.0 Roth Rollover
              </div>
              <p className="text-slate-600 dark:text-slate-400">
                Roll up to $35,000 leftover balance into beneficiary's Roth IRA after 15 years.
              </p>
            </div>

            <div className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-800 space-y-1">
              <div className="font-bold text-slate-900 dark:text-white flex items-center gap-1.5">
                <CheckCircle className="w-4 h-4 text-emerald-600" /> 2. Utilize 5-Year Superfunding
              </div>
              <p className="text-slate-600 dark:text-slate-400">
                Gift up to $95,000 at once ($190k joint) without gift tax penalty, front-loading 5 years of compounding!
              </p>
            </div>

            <div className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-800 space-y-1">
              <div className="font-bold text-slate-900 dark:text-white flex items-center gap-1.5">
                <CheckCircle className="w-4 h-4 text-emerald-600" /> 3. Transfer Beneficiary to Siblings
              </div>
              <p className="text-slate-600 dark:text-slate-400">
                If child #1 gets a full scholarship, change the beneficiary to a sibling, cousin, or parent with zero tax.
              </p>
            </div>

            <div className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-800 space-y-1">
              <div className="font-bold text-slate-900 dark:text-white flex items-center gap-1.5">
                <CheckCircle className="w-4 h-4 text-emerald-600" /> 4. Pay Student Loans &amp; K-12 Tuition
              </div>
              <p className="text-slate-600 dark:text-slate-400">
                Use up to $10,000/yr for private K-12 school tuition and up to $10,000 lifetime for student loan payoffs.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Section 2: Trump Account to IRA Rollover */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 sm:p-8 shadow-xs space-y-6">
        <div className="flex items-start justify-between gap-4 border-b border-slate-100 dark:border-slate-800 pb-4">
          <div>
            <div className="inline-flex items-center gap-1.5 text-xs font-bold text-amber-600 dark:text-amber-400">
              <Zap className="w-4 h-4" /> Strategy 2: Trump Account Age 18 IRA Rollover Pathway
            </div>
            <h3 className="text-xl font-bold text-slate-900 dark:text-white mt-1">
              Building Multi-Million Dollar Wealth via Trump Account Rollover
            </h3>
          </div>
          <span className="px-3 py-1 rounded-full bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300 text-xs font-bold">
            $5,000 / Year Cap
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 bg-amber-950/20 p-6 rounded-xl border border-amber-200 dark:border-amber-900/60">
          <div className="space-y-4">
            <label className="text-xs font-bold text-slate-700 dark:text-slate-300 block">
              Projected Trump Account Balance at Age 18
            </label>
            <input
              id="trump-balance-slider"
              type="range"
              min="50000"
              max="300000"
              step="5000"
              value={trumpAge18Balance}
              onChange={(e) => setTrumpAge18Balance(Number(e.target.value))}
              className="w-full accent-amber-600 cursor-pointer"
            />
            <div className="flex justify-between text-xs font-bold text-slate-900 dark:text-white">
              <span>Age 18 Balance: ${trumpAge18Balance.toLocaleString()}</span>
              <span>($5,000/yr for 18 yrs)</span>
            </div>
            <p className="text-[11px] text-slate-500">
              At age 18, funds transition into an IRA. If left untouched until age 60 (without adding another dollar), compounding takes over!
            </p>
          </div>

          <div className="bg-slate-900 text-white p-5 rounded-xl border border-slate-800 flex flex-col justify-between">
            <div>
              <div className="text-xs text-amber-400 font-bold uppercase tracking-wider">
                Compounded IRA Value at Age 60
              </div>
              <div className="text-3xl font-extrabold font-mono text-amber-300 mt-1">
                {formatCurrency(projectedTrumpAt60)}
              </div>
              <p className="text-xs text-slate-300 mt-1">
                Growing from age 18 to 60 (42 years) at {growthRate}% CAGR!
              </p>
            </div>
            <div className="text-[11px] text-amber-400 mt-4 border-t border-slate-800 pt-2 font-mono">
              ⚡ Exponential multi-million dollar early retirement headstart
            </div>
          </div>
        </div>
      </div>

      {/* Official Government & Tax Citation Cards */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 shadow-xs space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-100 dark:border-slate-800 pb-3">
          <div>
            <h4 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <BookOpen className="w-4 h-4 text-indigo-600" /> Statutory &amp; IRS Authority References
            </h4>
            <p className="text-xs text-slate-500 mt-0.5">
              Verify SECURE 2.0 Section 126 tax rules and IRA compounding guidelines directly on official federal websites.
            </p>
          </div>
          <span className="text-[10px] font-mono text-indigo-700 dark:text-indigo-300 bg-indigo-50 dark:bg-indigo-950 px-2.5 py-1 rounded-md border border-indigo-100 dark:border-indigo-900">
            IRS Notice 2024-02 &amp; IRC § 529(c)(3)(E)
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <a
            href="https://www.irs.gov/newsroom/irs-issues-guidance-on-secure-20-act-provisions"
            target="_blank"
            rel="noopener noreferrer"
            className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700/80 hover:border-indigo-500 transition-all text-xs group space-y-1.5"
          >
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-bold text-indigo-600 dark:text-indigo-400 font-mono">IRS.gov</span>
              <ExternalLink className="w-3.5 h-3.5 text-slate-400 group-hover:text-indigo-600 dark:group-hover:text-indigo-400" />
            </div>
            <div className="font-bold text-slate-900 dark:text-white group-hover:text-indigo-600 dark:group-hover:text-indigo-300">
              IRS SECURE 2.0 Act Guidance
            </div>
            <p className="text-[11px] text-slate-500 dark:text-slate-400">
              Official IRS newsroom breakdown of Section 126 529-to-Roth IRA transfers and eligibility rules.
            </p>
          </a>

          <a
            href="https://www.irs.gov/publications/p970"
            target="_blank"
            rel="noopener noreferrer"
            className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700/80 hover:border-indigo-500 transition-all text-xs group space-y-1.5"
          >
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-bold text-indigo-600 dark:text-indigo-400 font-mono">IRS.gov</span>
              <ExternalLink className="w-3.5 h-3.5 text-slate-400 group-hover:text-indigo-600 dark:group-hover:text-indigo-400" />
            </div>
            <div className="font-bold text-slate-900 dark:text-white group-hover:text-indigo-600 dark:group-hover:text-indigo-300">
              IRS Publication 970 (529 Plans)
            </div>
            <p className="text-[11px] text-slate-500 dark:text-slate-400">
              Detailed IRS tax code manual for qualified education savings accounts and non-qualified penalties.
            </p>
          </a>

          <a
            href="https://www.irs.gov/publications/p590a"
            target="_blank"
            rel="noopener noreferrer"
            className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700/80 hover:border-indigo-500 transition-all text-xs group space-y-1.5"
          >
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-bold text-indigo-600 dark:text-indigo-400 font-mono">IRS.gov</span>
              <ExternalLink className="w-3.5 h-3.5 text-slate-400 group-hover:text-indigo-600 dark:group-hover:text-indigo-400" />
            </div>
            <div className="font-bold text-slate-900 dark:text-white group-hover:text-indigo-600 dark:group-hover:text-indigo-300">
              IRS Publication 590-A (IRAs)
            </div>
            <p className="text-[11px] text-slate-500 dark:text-slate-400">
              Federal contribution limits, rollover rules, and early withdrawal exemptions for Traditional &amp; Roth IRAs.
            </p>
          </a>
        </div>
      </div>
    </div>
  );
};
