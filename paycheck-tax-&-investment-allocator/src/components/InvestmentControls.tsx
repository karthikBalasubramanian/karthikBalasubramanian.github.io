import React from 'react';
import { UserFinancialInputs } from '../types';
import { TAX_LIMITS_2026 } from '../data/taxRates';
import { Shield, TrendingUp, PiggyBank, Baby, Building, Sparkles, Check, Info, Award, Zap } from 'lucide-react';

interface InvestmentControlsProps {
  inputs: UserFinancialInputs;
  onChange: (updated: Partial<UserFinancialInputs>) => void;
}

export const InvestmentControls: React.FC<InvestmentControlsProps> = ({ inputs, onChange }) => {
  const isBiweekly = inputs.payFrequency !== 'annual';
  const payPeriods = 26;

  // Max calculations
  const max401kAnnual = inputs.age >= 50
    ? TAX_LIMITS_2026.TRADITIONAL_401K_MAX + TAX_LIMITS_2026.TRADITIONAL_401K_CATCHUP
    : TAX_LIMITS_2026.TRADITIONAL_401K_MAX;
  const max401kBiweekly = Math.round((max401kAnnual / payPeriods) * 100) / 100;

  const maxHsaAnnual = inputs.hsaCoverage === 'family'
    ? TAX_LIMITS_2026.HSA_FAMILY_MAX + (inputs.age >= 55 ? TAX_LIMITS_2026.HSA_CATCHUP : 0)
    : TAX_LIMITS_2026.HSA_SINGLE_MAX + (inputs.age >= 55 ? TAX_LIMITS_2026.HSA_CATCHUP : 0);
  const maxHsaBiweekly = Math.round((maxHsaAnnual / payPeriods) * 100) / 100;

  const maxIraAnnual = inputs.age >= 50
    ? TAX_LIMITS_2026.IRA_MAX + TAX_LIMITS_2026.IRA_CATCHUP
    : TAX_LIMITS_2026.IRA_MAX;
  const maxIraBiweekly = Math.round((maxIraAnnual / payPeriods) * 100) / 100;

  // ESPP Benefit calculation with 25% paycheck cap and $21,250 payroll limit ($25,000 IRS annual FMV purchase limit with 15% discount)
  const biweeklyGross = inputs.grossSalary;
  const esppPct = Math.min(Math.max(0, inputs.esppPercent || 0), 25);
  const uncappedEsppContrib = (biweeklyGross * esppPct) / 100;
  const discountFrac = (inputs.esppDiscountPercent || 15) / 100;
  const maxEsppAnnualPayroll = 25000 * (1 - discountFrac); // $21,250 for 15% discount
  const maxEsppBiweekly = maxEsppAnnualPayroll / payPeriods;
  const esppContrib = Math.min(uncappedEsppContrib, maxEsppBiweekly);
  const isEsppAnnualCapped = uncappedEsppContrib > maxEsppBiweekly;
  const esppGain = esppContrib > 0 ? esppContrib * (discountFrac / (1 - discountFrac)) : 0;

  // Company Match Calculation
  const matchPercent = inputs.companyMatchPercent ?? 100;
  const matchCapPercent = inputs.companyMatchUpToPercent ?? 6;
  const employee401kBiweekly = (inputs.traditional401k || 0) + (inputs.roth401k || 0);
  const employee401kPercent = biweeklyGross > 0 ? (employee401kBiweekly / biweeklyGross) * 100 : 0;
  const matchedPercent = Math.min(employee401kPercent, matchCapPercent);
  const companyMatchBiweekly = biweeklyGross * (matchedPercent / 100) * (matchPercent / 100);
  const companyMatchAnnual = companyMatchBiweekly * 26;

  return (
    <div className="space-y-6">
      
      {/* 1. Pre-Tax Retirement & Health Accounts */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl text-slate-100 space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center gap-2">
            <div className="p-2 rounded-xl bg-purple-950 text-purple-400 border border-purple-800/50">
              <Shield className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-white">Pre-Tax Deductions (Lowers Taxable Income)</h3>
              <p className="text-xs text-slate-400">Deducted before income taxes are computed</p>
            </div>
          </div>
        </div>

        {/* Employer 401(k) Match Banner / Controls */}
        <div className="bg-gradient-to-r from-emerald-950/80 via-slate-900 to-indigo-950/80 border border-emerald-800/60 rounded-xl p-4 space-y-3">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800/80 pb-2.5">
            <div className="flex items-center gap-2">
              <div className="p-1.5 rounded-lg bg-emerald-900/50 text-emerald-400 border border-emerald-700/50">
                <Sparkles className="w-4 h-4" />
              </div>
              <div>
                <h4 className="text-xs font-bold text-emerald-300 uppercase tracking-wider">
                  Employer 401(k) Company Match ("Free Money")
                </h4>
                <p className="text-[11px] text-slate-300">
                  Company matches <span className="font-bold text-white">{matchPercent}%</span> of your contributions up to <span className="font-bold text-white">{matchCapPercent}%</span> of salary.
                </p>
              </div>
            </div>

            <div className="flex items-center gap-1.5 flex-wrap">
              <button
                onClick={() => onChange({ companyMatchPercent: 100, companyMatchUpToPercent: 6 })}
                className="text-[10px] font-bold px-2 py-1 rounded bg-emerald-900/60 text-emerald-200 border border-emerald-700 hover:bg-emerald-800"
              >
                100% for 6%
              </button>
              <button
                onClick={() => onChange({ companyMatchPercent: 50, companyMatchUpToPercent: 6 })}
                className="text-[10px] font-bold px-2 py-1 rounded bg-slate-800 text-slate-300 border border-slate-700 hover:bg-slate-700"
              >
                50% for 6%
              </button>
              <button
                onClick={() => onChange({ companyMatchPercent: 100, companyMatchUpToPercent: 4 })}
                className="text-[10px] font-bold px-2 py-1 rounded bg-slate-800 text-slate-300 border border-slate-700 hover:bg-slate-700"
              >
                100% for 4%
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div>
              <label className="block text-[10px] font-semibold text-slate-400 mb-1">Company Match %</label>
              <div className="relative">
                <input
                  type="number"
                  min={0}
                  max={200}
                  value={matchPercent}
                  onChange={(e) => onChange({ companyMatchPercent: Math.max(0, parseFloat(e.target.value) || 0) })}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg py-1.5 px-3 text-xs font-mono font-bold text-white focus:outline-none focus:border-emerald-500"
                />
                <span className="absolute right-3 top-1.5 text-slate-500 text-xs font-bold">%</span>
              </div>
            </div>

            <div>
              <label className="block text-[10px] font-semibold text-slate-400 mb-1">Max Match Salary Cap</label>
              <div className="relative">
                <input
                  type="number"
                  min={0}
                  max={25}
                  value={matchCapPercent}
                  onChange={(e) => onChange({ companyMatchUpToPercent: Math.max(0, parseFloat(e.target.value) || 0) })}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg py-1.5 px-3 text-xs font-mono font-bold text-white focus:outline-none focus:border-emerald-500"
                />
                <span className="absolute right-3 top-1.5 text-slate-500 text-xs font-bold">% of Salary</span>
              </div>
            </div>

            <div className="bg-emerald-950/90 border border-emerald-700/80 rounded-xl p-2.5 flex flex-col justify-center">
              <span className="text-[10px] uppercase font-bold text-emerald-400 tracking-wider">Free Employer Money</span>
              <div className="text-base font-bold text-white font-mono">
                +${companyMatchBiweekly.toFixed(2)} <span className="text-[10px] font-normal text-emerald-300">/ biweekly</span>
              </div>
              <span className="text-[10px] font-mono text-emerald-300">
                (${companyMatchAnnual.toLocaleString('en-US', { maximumFractionDigits: 0 })} / year)
              </span>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          
          {/* Traditional 401(k) */}
          <div className="bg-slate-950 border border-slate-800 rounded-xl p-3.5 space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-xs font-bold text-purple-300">401(k) Traditional</label>
              <button
                onClick={() => onChange({ traditional401k: max401kBiweekly, traditional401kIsPercent: false })}
                className="text-[10px] font-bold px-2 py-0.5 rounded bg-purple-900/50 text-purple-300 border border-purple-700/50 hover:bg-purple-800/60"
              >
                Max (${max401kBiweekly}/bw)
              </button>
            </div>

            <div className="relative">
              <span className="absolute left-3 top-2 text-slate-500 font-bold">$</span>
              <input
                type="number"
                value={inputs.traditional401k || ''}
                onChange={(e) =>
                  onChange({ traditional401k: Math.max(0, parseFloat(e.target.value) || 0) })
                }
                className="w-full bg-slate-900 border border-slate-700 rounded-lg py-1.5 pl-7 pr-3 text-xs font-mono font-bold text-white focus:outline-none focus:border-purple-500"
                placeholder="Biweekly $"
              />
            </div>
            <p className="text-[10px] text-slate-400">
              IRS Annual Limit: <span className="font-mono text-slate-200">${max401kAnnual.toLocaleString()}</span>
            </p>
          </div>

          {/* HSA Health Savings */}
          <div className="bg-slate-950 border border-slate-800 rounded-xl p-3.5 space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-xs font-bold text-teal-300">HSA Health Savings</label>
              <button
                onClick={() => onChange({ hsa: maxHsaBiweekly })}
                className="text-[10px] font-bold px-2 py-0.5 rounded bg-teal-900/50 text-teal-300 border border-teal-700/50 hover:bg-teal-800/60"
              >
                Max (${maxHsaBiweekly}/bw)
              </button>
            </div>

            <div className="flex gap-2">
              <div className="relative flex-1">
                <span className="absolute left-3 top-2 text-slate-500 font-bold">$</span>
                <input
                  type="number"
                  value={inputs.hsa || ''}
                  onChange={(e) => onChange({ hsa: Math.max(0, parseFloat(e.target.value) || 0) })}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg py-1.5 pl-7 pr-3 text-xs font-mono font-bold text-white focus:outline-none focus:border-teal-500"
                  placeholder="Biweekly $"
                />
              </div>
              <select
                value={inputs.hsaCoverage}
                onChange={(e) => onChange({ hsaCoverage: e.target.value as 'single' | 'family' })}
                className="bg-slate-900 border border-slate-700 rounded-lg text-xs py-1.5 px-2 text-slate-200 focus:outline-none"
              >
                <option value="single">Single</option>
                <option value="family">Family</option>
              </select>
            </div>
            <p className="text-[10px] text-slate-400">
              Saves Federal, State & FICA taxes! Max: <span className="font-mono text-slate-200">${maxHsaAnnual.toLocaleString()}</span>
            </p>
          </div>

          {/* Flexible Spending FSA */}
          <div className="bg-slate-950 border border-slate-800 rounded-xl p-3.5 space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-xs font-bold text-sky-300">FSA Account</label>
              <button
                onClick={() => onChange({ fsa: Math.round((3200 / 26) * 100) / 100 })}
                className="text-[10px] font-bold px-2 py-0.5 rounded bg-sky-900/50 text-sky-300 border border-sky-700/50 hover:bg-sky-800/60"
              >
                Max ($123/bw)
              </button>
            </div>

            <div className="relative">
              <span className="absolute left-3 top-2 text-slate-500 font-bold">$</span>
              <input
                type="number"
                value={inputs.fsa || ''}
                onChange={(e) => onChange({ fsa: Math.max(0, parseFloat(e.target.value) || 0) })}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg py-1.5 pl-7 pr-3 text-xs font-mono font-bold text-white focus:outline-none focus:border-sky-500"
                placeholder="Biweekly $"
              />
            </div>
            <p className="text-[10px] text-slate-400">Healthcare/Dependent care pre-tax flex account.</p>
          </div>

        </div>
      </div>

      {/* 2. Post-Tax Retirement, Child Accounts & ESPP */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl text-slate-100 space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center gap-2">
            <div className="p-2 rounded-xl bg-emerald-950 text-emerald-400 border border-emerald-800/50">
              <TrendingUp className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-white">
                Post-Tax Accounts & Child Wealth Allocations
              </h3>
              <p className="text-xs text-slate-400">Roth IRA, 529 College, Custodial/Trump accounts, & Company ESPP</p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          
          {/* Roth IRA / Post-tax IRA */}
          <div className="bg-slate-950 border border-slate-800 rounded-xl p-3.5 space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-xs font-bold text-emerald-300">Roth IRA (Post-Tax)</label>
              <button
                onClick={() => onChange({ rothIra: maxIraBiweekly })}
                className="text-[10px] font-bold px-2 py-0.5 rounded bg-emerald-900/50 text-emerald-300 border border-emerald-700/50 hover:bg-emerald-800/60"
              >
                Max (${maxIraBiweekly}/bw)
              </button>
            </div>

            <div className="relative">
              <span className="absolute left-3 top-2 text-slate-500 font-bold">$</span>
              <input
                type="number"
                value={inputs.rothIra || ''}
                onChange={(e) => onChange({ rothIra: Math.max(0, parseFloat(e.target.value) || 0) })}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg py-1.5 pl-7 pr-3 text-xs font-mono font-bold text-white focus:outline-none focus:border-emerald-500"
                placeholder="Biweekly $"
              />
            </div>
            <p className="text-[10px] text-slate-400">
              Tax-free growth & withdrawals! Max: <span className="font-mono text-slate-200">${maxIraAnnual.toLocaleString()}</span>/yr
            </p>
          </div>

          {/* 529 College Savings */}
          <div className="bg-slate-950 border border-slate-800 rounded-xl p-3.5 space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5">
                <Baby className="w-3.5 h-3.5 text-sky-400" />
                <label className="text-xs font-bold text-sky-300">529 College Savings</label>
              </div>
              <button
                onClick={() => onChange({ plan529: 250 })}
                className="text-[10px] font-bold px-2 py-0.5 rounded bg-sky-900/50 text-sky-300 border border-sky-700/50 hover:bg-sky-800/60"
              >
                Set $250/bw
              </button>
            </div>

            <div className="relative">
              <span className="absolute left-3 top-2 text-slate-500 font-bold">$</span>
              <input
                type="number"
                value={inputs.plan529 || ''}
                onChange={(e) => onChange({ plan529: Math.max(0, parseFloat(e.target.value) || 0) })}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg py-1.5 pl-7 pr-3 text-xs font-mono font-bold text-white focus:outline-none focus:border-sky-500"
                placeholder="Biweekly $"
              />
            </div>
            <p className="text-[10px] text-slate-400">
              Tax-free growth for education. Many states offer state tax deductions!
            </p>
          </div>

          {/* Custodial / Trump / Child Accounts */}
          <div className="bg-slate-950 border border-slate-800 rounded-xl p-3.5 space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5">
                <PiggyBank className="w-3.5 h-3.5 text-teal-400" />
                <label className="text-xs font-bold text-teal-300">Custodial & Trump Child Accounts</label>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-[10px] text-slate-400">Custodial UTMA</label>
                <div className="relative">
                  <span className="absolute left-2 top-1 text-slate-500 font-bold">$</span>
                  <input
                    type="number"
                    value={inputs.custodialAccount || ''}
                    onChange={(e) => onChange({ custodialAccount: Math.max(0, parseFloat(e.target.value) || 0) })}
                    className="w-full bg-slate-900 border border-slate-700 rounded py-1 pl-5 pr-2 text-xs font-mono font-bold text-white"
                  />
                </div>
              </div>
              <div>
                <label className="text-[10px] text-slate-400">Trump Account</label>
                <div className="relative">
                  <span className="absolute left-2 top-1 text-slate-500 font-bold">$</span>
                  <input
                    type="number"
                    value={inputs.trumpAccount || ''}
                    onChange={(e) => onChange({ trumpAccount: Math.max(0, parseFloat(e.target.value) || 0) })}
                    className="w-full bg-slate-900 border border-slate-700 rounded py-1 pl-5 pr-2 text-xs font-mono font-bold text-white"
                  />
                </div>
              </div>
            </div>
            <p className="text-[10px] text-slate-400">
              Child trust / UTMA wealth growth accounts for kids.
            </p>
          </div>

          {/* Company ESPP */}
          <div className="bg-slate-950 border border-slate-800 rounded-xl p-3.5 space-y-2 md:col-span-2 lg:col-span-3">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <Building className="w-4 h-4 text-indigo-400" />
                <div>
                  <label className="text-xs font-bold text-indigo-300">
                    Company Employee Stock Purchase Plan (ESPP)
                  </label>
                  <p className="text-[11px] text-slate-400">
                    Buy company stock at a 15% discount. Max contribution: <span className="font-bold text-indigo-300">25% per paycheck</span> & <span className="font-bold text-indigo-300">$21,250/yr payroll limit</span> ($25k FMV stock).
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-1.5 flex-wrap">
                <button
                  onClick={() => onChange({ esppPercent: 15 })}
                  className="text-[10px] font-bold px-2 py-1 rounded bg-slate-800 text-slate-300 border border-slate-700 hover:bg-slate-700"
                >
                  15% Standard
                </button>
                <button
                  onClick={() => onChange({ esppPercent: 25 })}
                  className="text-[10px] font-bold px-2.5 py-1 rounded-lg bg-indigo-900/60 text-indigo-300 border border-indigo-700/60 hover:bg-indigo-800"
                >
                  25% Paycheck Cap
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-1">
              <div className="space-y-1">
                <div className="flex justify-between text-xs text-slate-300">
                  <span>Contribution %</span>
                  <span className="font-bold text-indigo-400 font-mono">{inputs.esppPercent}% of Gross</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={25}
                  step={1}
                  value={Math.min(inputs.esppPercent, 25)}
                  onChange={(e) => onChange({ esppPercent: parseInt(e.target.value) || 0 })}
                  className="w-full accent-indigo-500 cursor-pointer"
                />
              </div>

              <div className="bg-slate-900 p-2.5 rounded-lg border border-slate-800 text-xs flex items-center justify-between">
                <div>
                  <span className="text-slate-400 block">Biweekly Contribution:</span>
                  {isEsppAnnualCapped && (
                    <span className="text-[10px] text-amber-400 font-bold">Capped at IRS $21,250/yr payroll limit ($25k FMV)</span>
                  )}
                </div>
                <span className="font-bold font-mono text-white">${esppContrib.toFixed(0)}/bw</span>
              </div>

              <div className="bg-emerald-950/60 border border-emerald-800/60 p-2.5 rounded-lg text-xs flex items-center justify-between">
                <span className="text-emerald-300 font-semibold flex items-center gap-1">
                  <Award className="w-3.5 h-3.5" /> 15% Discount Value:
                </span>
                <span className="font-bold font-mono text-emerald-400">+${esppGain.toFixed(0)}/bw</span>
              </div>
            </div>
          </div>

        </div>
      </div>

    </div>
  );
};
