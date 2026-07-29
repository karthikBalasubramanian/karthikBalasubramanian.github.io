import React from 'react';
import { ParentInputs, InvestmentGoal, AllocationResult, AccountId } from '../types';
import { ACCOUNT_DATA } from '../data/accountData';
import { calculateOptimalAllocation } from '../utils/financialCalculators';
import {
  Sliders,
  Sparkles,
  PieChart,
  CheckCircle2,
  AlertCircle,
  HelpCircle,
  DollarSign,
  TrendingUp,
  RotateCcw,
  ExternalLink,
} from 'lucide-react';

interface PortfolioOptimizerProps {
  inputs: ParentInputs;
  onUpdateInputs: (updated: Partial<ParentInputs>) => void;
  allocationResults: AllocationResult[];
}

export const PortfolioOptimizer: React.FC<PortfolioOptimizerProps> = ({
  inputs,
  onUpdateInputs,
  allocationResults,
}) => {
  const formatCurrency = (val: number) =>
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(val);

  const totalMonthlyEquivalent = inputs.monthlyContribution + inputs.yearlyLumpSum / 12;
  const totalAnnualEquivalent = totalMonthlyEquivalent * 12;

  const customSum = Object.values(inputs.customAllocations).reduce((a: number, b: number) => a + b, 0);

  const goalsList: { id: InvestmentGoal; label: string; desc: string; icon: string }[] = [
    {
      id: 'education_focused',
      label: '100% Education Focus',
      desc: 'Maximizes 529 plans & Coverdell ESAs for college, private K-12, & trade school.',
      icon: '🎓',
    },
    {
      id: 'balanced_growth',
      label: 'Balanced Education & Wealth',
      desc: 'Optimal mix of 529 for college + Trump Account for age 18 conversion.',
      icon: '⚖️',
    },
    {
      id: 'long_term_wealth',
      label: 'Max Long-Term Wealth (Age 60)',
      desc: 'Emphasizes Trump Accounts & Custodial IRAs to compound past age 18.',
      icon: '🚀',
    },
    {
      id: 'maximum_flexibility',
      label: 'Maximum Spending Flexibility',
      desc: 'Heavy allocation to UTMA/UGMA & Taxable Trusts for non-education expenses.',
      icon: '🔓',
    },
  ];

  const handleCustomSliderChange = (accId: AccountId, val: number) => {
    const updated = { ...inputs.customAllocations, [accId]: val };
    onUpdateInputs({ customAllocations: updated });
  };

  const handleResetToAuto = () => {
    const defaultAllocs: Record<AccountId, number> = {
      '529_plan': 45,
      trump_account: 30,
      custodial_roth_ira: 15,
      utma_ugma: 10,
      coverdell_esa: 0,
      taxable_brokerage: 0,
    };
    onUpdateInputs({ customAllocations: defaultAllocs });
  };

  return (
    <div className="space-y-8">
      {/* Intro Banner */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 sm:p-8 shadow-xs">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div>
            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-indigo-50 text-indigo-700 dark:bg-indigo-950/60 dark:text-indigo-300 border border-indigo-100 dark:border-indigo-900/50 text-[11px] font-bold uppercase tracking-wider mb-2">
              <Sparkles className="w-3.5 h-3.5 text-indigo-600 dark:text-indigo-400" /> Smart Account Allocation Engine
            </div>
            <h2 className="text-xl sm:text-2xl font-bold text-slate-900 dark:text-white tracking-tight">
              Optimize Distribution for Your Monthly Contribution
            </h2>
            <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400 mt-1">
              Adjust your monthly budget, child age, and primary goal below. Our financial rules engine calculates the ideal split across 529s, Trump Accounts, and IRAs.
            </p>
          </div>
          <div className="bg-indigo-50/80 dark:bg-indigo-950/40 p-4 rounded-xl border border-indigo-100 dark:border-indigo-900/50 shrink-0 text-center">
            <div className="text-[11px] text-indigo-700 dark:text-indigo-400 font-semibold uppercase tracking-wider">Total Annual Investment</div>
            <div className="text-2xl font-extrabold font-mono text-indigo-900 dark:text-indigo-300 mt-0.5">
              {formatCurrency(totalAnnualEquivalent)}
            </div>
            <div className="text-[11px] font-mono text-indigo-600 dark:text-indigo-400 mt-0.5">
              (${formatCurrency(totalMonthlyEquivalent)}/month)
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Controls Column (5 cols) */}
        <div className="lg:col-span-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 shadow-xs space-y-6">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 flex items-center gap-2 border-b border-slate-100 dark:border-slate-800 pb-3">
            <Sliders className="w-4 h-4 text-indigo-600" /> Investment Parameters
          </h3>

          {/* Monthly Budget Slider */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-xs">
              <label className="font-semibold text-slate-700 dark:text-slate-300">Monthly Contribution ($/month)</label>
              <span className="font-bold font-mono text-indigo-600 dark:text-indigo-400 text-sm">
                ${inputs.monthlyContribution}
              </span>
            </div>
            <input
              id="monthly-budget-slider"
              type="range"
              min="25"
              max="3000"
              step="25"
              value={inputs.monthlyContribution}
              onChange={(e) => onUpdateInputs({ monthlyContribution: Number(e.target.value) })}
              className="w-full accent-indigo-600 cursor-pointer"
            />
            <div className="relative w-full h-4 text-[10px] font-mono text-slate-400">
              <span className="absolute left-0">$25/mo</span>
              <span className="absolute -translate-x-1/2" style={{ left: '16%' }}>$500/mo</span>
              <span className="absolute -translate-x-1/2" style={{ left: '49.6%' }}>$1,500/mo</span>
              <span className="absolute right-0 text-right">$3,000/mo</span>
            </div>
          </div>

          {/* Child Age Slider */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-xs">
              <label className="font-semibold text-slate-700 dark:text-slate-300">Child Current Age</label>
              <span className="font-bold text-blue-600 dark:text-blue-400 text-sm">
                {inputs.childCurrentAge} {inputs.childCurrentAge === 0 ? '(Newborn)' : 'years old'}
              </span>
            </div>
            <input
              id="child-age-slider"
              type="range"
              min="0"
              max="17"
              step="1"
              value={inputs.childCurrentAge}
              onChange={(e) => onUpdateInputs({ childCurrentAge: Number(e.target.value) })}
              className="w-full accent-blue-600 cursor-pointer"
            />
            <div className="relative w-full h-4 text-[10px] text-slate-400">
              <span className="absolute left-0">Age 0</span>
              <span className="absolute -translate-x-1/2" style={{ left: '29.4%' }}>Age 5</span>
              <span className="absolute -translate-x-1/2" style={{ left: '58.8%' }}>Age 10</span>
              <span className="absolute right-0 text-right">Age 17</span>
            </div>
          </div>

          {/* Extra Annual Lump Sum */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-xs">
              <label className="font-semibold text-slate-700 dark:text-slate-300">Extra Annual Lump-Sum ($/year)</label>
              <span className="font-bold text-purple-600 dark:text-purple-400 text-sm">
                ${inputs.yearlyLumpSum}
              </span>
            </div>
            <input
              id="annual-lumpsum-slider"
              type="range"
              min="0"
              max="10000"
              step="250"
              value={inputs.yearlyLumpSum}
              onChange={(e) => onUpdateInputs({ yearlyLumpSum: Number(e.target.value) })}
              className="w-full accent-purple-600 cursor-pointer"
            />
            <p className="text-[11px] text-slate-500">e.g., Birthday gifts, tax refunds, grandparent contributions</p>
          </div>

          {/* Child Earned Income */}
          <div className="space-y-2 pt-2 border-t border-slate-100 dark:border-slate-800">
            <div className="flex justify-between items-center text-xs">
              <label className="font-semibold text-slate-700 dark:text-slate-300 flex items-center gap-1">
                Child Earned Income ($/year)
              </label>
              <span className="font-bold text-amber-600 dark:text-amber-400 text-xs">
                ${inputs.childEarnedIncome}/yr
              </span>
            </div>
            <input
              id="child-earned-income-input"
              type="number"
              min="0"
              max="7000"
              step="100"
              value={inputs.childEarnedIncome}
              onChange={(e) => onUpdateInputs({ childEarnedIncome: Number(e.target.value) })}
              className="w-full text-xs px-3 py-2 rounded-lg border border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-slate-900 dark:text-slate-100"
            />
            <p className="text-[11px] text-slate-500">
              Required to unlock <strong>Custodial Roth IRA</strong> eligibility (babysitting, modeling, family business wage).
            </p>
          </div>

          {/* Primary Strategy Goal */}
          <div className="space-y-3 pt-2 border-t border-slate-100 dark:border-slate-800">
            <label className="text-[11px] font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 block">
              Primary Financial Strategy Goal
            </label>
            <div className="space-y-2">
              {goalsList.map((g) => {
                const isSelected = inputs.primaryGoal === g.id;
                return (
                  <button
                    key={g.id}
                    id={`goal-button-${g.id}`}
                    onClick={() => onUpdateInputs({ primaryGoal: g.id })}
                    className={`w-full text-left p-3 rounded-xl border text-xs transition-all flex items-start gap-3 ${
                      isSelected
                        ? 'bg-indigo-50/80 dark:bg-indigo-950/50 border-indigo-500 text-slate-900 dark:text-white shadow-xs ring-1 ring-indigo-500/30'
                        : 'bg-slate-50 dark:bg-slate-800/40 border-slate-200 dark:border-slate-800 text-slate-700 dark:text-slate-300 hover:border-slate-300'
                    }`}
                  >
                    <span className="text-lg">{g.icon}</span>
                    <div>
                      <div className="font-bold text-slate-900 dark:text-white">{g.label}</div>
                      <div className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">{g.desc}</div>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right Output Column (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          {/* Smart Allocation Results Header */}
          <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 shadow-xs">
            <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800 pb-4">
              <div>
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 flex items-center gap-2">
                  <PieChart className="w-4 h-4 text-indigo-600" /> Recommended Account Allocation
                </h3>
                <p className="text-xs text-slate-500 mt-0.5">
                  Calculated based on your monthly budget of <strong className="font-mono text-slate-800 dark:text-slate-200">${inputs.monthlyContribution}/mo</strong>
                </p>
              </div>

              {customSum !== 100 && (
                <button
                  id="reset-auto-button"
                  onClick={handleResetToAuto}
                  className="inline-flex items-center gap-1 text-xs text-slate-500 hover:text-indigo-600 dark:hover:text-indigo-400 underline font-medium"
                >
                  <RotateCcw className="w-3 h-3" /> Reset to Auto
                </button>
              )}
            </div>

            {/* Allocation List */}
            <div className="space-y-4 mt-5">
              {allocationResults.map((res) => {
                const acc = ACCOUNT_DATA[res.accountId];
                return (
                  <div
                    key={res.accountId}
                    className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-800/30 space-y-3"
                  >
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <span className={`w-2.5 h-2.5 rounded-full ${acc.bgColor} border ${acc.borderColor}`} />
                        <span className="font-bold text-sm text-slate-900 dark:text-white">{res.accountName}</span>
                        <span className="text-[10px] px-2 py-0.5 rounded-full bg-slate-200/70 dark:bg-slate-700 text-slate-600 dark:text-slate-300 font-mono">
                          {acc.shortName}
                        </span>
                      </div>

                      <div className="flex items-center gap-3 text-xs font-bold">
                        <span className="text-slate-400 font-mono">{res.percentage}% split</span>
                        <span className="text-indigo-600 dark:text-indigo-400 text-sm font-extrabold font-mono">
                          ${res.monthlyAmount}/mo
                        </span>
                      </div>
                    </div>

                    {/* Progress Bar & Slider for Custom Override */}
                    <div className="space-y-1">
                      <div className="w-full bg-slate-200/80 dark:bg-slate-700 h-1.5 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-indigo-600 transition-all duration-300"
                          style={{ width: `${res.percentage}%` }}
                        />
                      </div>
                      <div className="flex items-center justify-between text-[11px] text-slate-400">
                        <span>Adjust manual %:</span>
                        <input
                          id={`custom-slider-${res.accountId}`}
                          type="range"
                          min="0"
                          max="100"
                          step="5"
                          value={inputs.customAllocations[res.accountId] ?? res.percentage}
                          onChange={(e) => handleCustomSliderChange(res.accountId, Number(e.target.value))}
                          className="w-32 accent-indigo-600 cursor-pointer"
                        />
                      </div>
                    </div>

                    {/* Recommendation Reason & Impact */}
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-[11px] pt-1.5 border-t border-slate-200/60 dark:border-slate-700/60">
                      <div className="text-slate-500 dark:text-slate-400 italic">
                        "{res.recommendationReason}"
                      </div>
                      <div className="text-right text-slate-600 dark:text-slate-300 font-medium">
                        Age 18 Balance: <strong className="text-indigo-600 dark:text-indigo-400 font-mono">{formatCurrency(res.projectedAge18Moderate)}</strong>
                        <span className="block text-[10px] text-slate-400 font-mono">
                          Age 60 IRA Growth: {formatCurrency(res.projectedAge60Moderate)}
                        </span>
                      </div>
                    </div>

                    {/* Official Backing Links */}
                    {acc.officialLinks && acc.officialLinks.length > 0 && (
                      <div className="pt-2 border-t border-slate-200/60 dark:border-slate-700/40 flex flex-wrap items-center gap-2 text-[10px]">
                        <span className="text-slate-400 font-bold uppercase tracking-wider text-[9px]">Official Sources:</span>
                        {acc.officialLinks.map((link, lIdx) => (
                          <a
                            key={lIdx}
                            href={link.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-indigo-600 dark:text-indigo-300 hover:border-indigo-500 font-medium transition-all"
                            title={link.description}
                          >
                            <span>{link.source}</span>
                            <ExternalLink className="w-2.5 h-2.5 opacity-70" />
                          </a>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Custom Split Total Warning */}
            {customSum !== 100 && (
              <div className="mt-4 p-3 rounded-xl bg-amber-50 dark:bg-amber-950/40 border border-amber-300 dark:border-amber-800 text-amber-800 dark:text-amber-300 text-xs flex items-center justify-between">
                <span className="flex items-center gap-1.5 font-semibold">
                  <AlertCircle className="w-4 h-4 shrink-0" /> Custom allocations sum to {customSum}% (should equal 100%)
                </span>
                <button
                  id="reset-allocations-button"
                  onClick={handleResetToAuto}
                  className="px-2.5 py-1 rounded-md bg-amber-600 text-white font-bold text-[11px]"
                >
                  Auto Balance
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
