import React, { useState } from 'react';
import { UserFinancialInputs, TaxBreakdownResult } from '../types';
import { Calendar, TrendingUp, CheckCircle2, AlertCircle, Zap, Shield, Sparkles, ArrowRight, DollarSign } from 'lucide-react';

interface PaycheckScheduleTableProps {
  inputs: UserFinancialInputs;
  taxResult: TaxBreakdownResult;
  onChange: (updated: Partial<UserFinancialInputs>) => void;
}

export const PaycheckScheduleTable: React.FC<PaycheckScheduleTableProps> = ({
  inputs,
  taxResult,
  onChange,
}) => {
  const [filterMode, setFilterMode] = useState<'all' | 'milestones'>('all');
  const schedule = taxResult.schedule;
  const periods = schedule.periods;

  const bonusPeriodNum = schedule.bonusPeriodNumber;
  const max401kPeriod = schedule.maxOutPayPeriod401k;

  const fmt = (val: number) =>
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(val);

  const displayedPeriods = filterMode === 'milestones'
    ? periods.filter(p => p.isBonusPeriod || p.is401kCapHit || p.isHsaCapHit || p.isEsppCapHit || p.isSocialSecurityCapHit || p.periodNumber === 1 || p.periodNumber === 26)
    : periods;

  const netPayRaise = schedule.latePhaseNetBiweekly - schedule.earlyPhaseNetBiweekly;

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-2xl text-slate-100 space-y-6">
      
      {/* Title & Timing Selector Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-purple-500/10 text-purple-400 border border-purple-500/20">
            <Calendar className="w-6 h-6" />
          </div>
          <div>
            <h3 className="text-base sm:text-lg font-bold text-white flex items-center gap-2">
              <span>26-Paycheck Chronological Timeline & Sequencing</span>
              <span className="text-xs font-mono font-bold px-2 py-0.5 rounded-full bg-purple-950 text-purple-300 border border-purple-800">
                2026 Tax Year
              </span>
            </h3>
            <p className="text-xs text-slate-400">
              Simulates how early bonus payouts cause 401(k), HSA, & ESPP caps to max out early, boosting late-year net cash!
            </p>
          </div>
        </div>

        {/* Bonus Paycheck Timing Selector */}
        {taxResult.grossAnnualBonus > 0 && (
          <div className="bg-slate-950 border border-amber-800/50 p-2.5 rounded-xl flex items-center gap-3">
            <Sparkles className="w-4 h-4 text-amber-400 shrink-0" />
            <div>
              <label className="block text-[10px] uppercase font-bold text-amber-300 tracking-wider">
                Bonus Paid On Paycheck #:
              </label>
              <select
                value={bonusPeriodNum}
                onChange={(e) => onChange({ bonusPayPeriodNumber: parseInt(e.target.value) || 4 })}
                className="bg-slate-900 border border-amber-700/60 rounded-lg text-xs font-mono font-bold text-amber-200 py-1 px-2 focus:outline-none"
              >
                {Array.from({ length: 26 }, (_, i) => i + 1).map((num) => (
                  <option key={num} value={num}>
                    Paycheck #{num} ({['Jan', 'Jan', 'Feb', 'Feb', 'Mar', 'Mar', 'Apr', 'Apr', 'May', 'May', 'Jun', 'Jun', 'Jul', 'Jul', 'Aug', 'Aug', 'Sep', 'Sep', 'Oct', 'Oct', 'Nov', 'Nov', 'Dec', 'Dec', 'Dec', 'Dec'][num - 1]})
                  </option>
                ))}
              </select>
            </div>
          </div>
        )}
      </div>

      {/* Phase Shift Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        
        {/* Phase 1: Early Paychecks */}
        <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 space-y-2">
          <div className="flex items-center justify-between text-xs text-slate-400 font-semibold">
            <span>Phase 1: Early-Year Paychecks</span>
            <span className="text-[10px] font-mono bg-slate-800 px-2 py-0.5 rounded text-slate-300">Paychecks #1 – #{max401kPeriod ? max401kPeriod - 1 : 18}</span>
          </div>
          <div className="text-xl font-bold font-mono text-white">
            {fmt(schedule.earlyPhaseNetBiweekly)} <span className="text-xs font-normal text-slate-400">/ biweekly</span>
          </div>
          <p className="text-[11px] text-slate-400">
            Standard pre-tax 401(k), HSA, & ESPP deductions are active from regular salary.
          </p>
        </div>

        {/* Bonus Paycheck Highlight */}
        {taxResult.grossAnnualBonus > 0 && (
          <div className="bg-gradient-to-br from-amber-950/70 via-slate-950 to-slate-950 border border-amber-700/60 rounded-xl p-4 space-y-2">
            <div className="flex items-center justify-between text-xs text-amber-300 font-semibold">
              <span className="flex items-center gap-1"><Sparkles className="w-3.5 h-3.5" /> Bonus Paycheck #{bonusPeriodNum}</span>
              <span className="text-[10px] font-mono bg-amber-900/60 px-2 py-0.5 rounded text-amber-200">Lump Sum Check</span>
            </div>
            <div className="text-xl font-bold font-mono text-amber-300">
              +{fmt(taxResult.bonusNetTakeHome)} <span className="text-xs font-normal text-amber-200/80">check in hand</span>
            </div>
            <p className="text-[11px] text-slate-300">
              Gross bonus (+{fmt(taxResult.grossAnnualBonus)}) minus 401(k), HSA, ESPP & supplemental taxes.
            </p>
          </div>
        )}

        {/* Phase 2: Late-Year Paychecks (Post Max-Out) */}
        <div className="bg-gradient-to-br from-emerald-950/80 via-slate-950 to-slate-950 border border-emerald-500/60 rounded-xl p-4 space-y-2">
          <div className="flex items-center justify-between text-xs text-emerald-300 font-semibold">
            <span className="flex items-center gap-1"><Zap className="w-3.5 h-3.5" /> Phase 2: Post Max-Out Paychecks</span>
            <span className="text-[10px] font-mono bg-emerald-950 px-2 py-0.5 rounded text-emerald-200 border border-emerald-700/60">Paychecks #{max401kPeriod || 19} – #26</span>
          </div>
          <div className="text-xl font-bold font-mono text-emerald-400 flex items-baseline gap-2">
            <span>{fmt(schedule.latePhaseNetBiweekly)}</span>
            {netPayRaise > 0 && (
              <span className="text-xs font-bold text-emerald-300 font-sans">
                (+{fmt(netPayRaise)}/bw raise!)
              </span>
            )}
          </div>
          <p className="text-[11px] text-slate-300">
            {max401kPeriod ? `401(k) maxed out on Paycheck #${max401kPeriod}!` : 'Deductions stop once annual IRS limits are reached.'} Net take-home pay increases automatically!
          </p>
        </div>
      </div>

      {/* Filter Buttons */}
      <div className="flex items-center justify-between flex-wrap gap-2 pt-2">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-slate-400">View Mode:</span>
          <button
            onClick={() => setFilterMode('all')}
            className={`text-xs font-bold px-3 py-1 rounded-lg transition-all ${
              filterMode === 'all'
                ? 'bg-purple-600 text-white shadow'
                : 'bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-white'
            }`}
          >
            All 26 Paychecks
          </button>
          <button
            onClick={() => setFilterMode('milestones')}
            className={`text-xs font-bold px-3 py-1 rounded-lg transition-all ${
              filterMode === 'milestones'
                ? 'bg-purple-600 text-white shadow'
                : 'bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-white'
            }`}
          >
            Key Milestones & Bonus Only
          </button>
        </div>

        <div className="flex items-center gap-2 text-[10px] text-slate-400">
          <span className="inline-block w-2.5 h-2.5 rounded-full bg-amber-500"></span> Bonus Paycheck
          <span className="inline-block w-2.5 h-2.5 rounded-full bg-purple-500 ml-2"></span> 401(k) Maxed
          <span className="inline-block w-2.5 h-2.5 rounded-full bg-teal-500 ml-2"></span> HSA Maxed
          <span className="inline-block w-2.5 h-2.5 rounded-full bg-indigo-500 ml-2"></span> ESPP Maxed
        </div>
      </div>

      {/* 26-Paycheck Full Chronological Table */}
      <div className="overflow-x-auto border border-slate-800 rounded-xl">
        <table className="w-full text-left text-xs font-mono border-collapse">
          <thead>
            <tr className="bg-slate-950 text-slate-400 border-b border-slate-800 font-sans text-[11px]">
              <th className="p-2.5">Paycheck</th>
              <th className="p-2.5 text-right">Gross Income</th>
              <th className="p-2.5 text-right">Employee 401(k)</th>
              <th className="p-2.5 text-right">Co. Match</th>
              <th className="p-2.5 text-right">HSA</th>
              <th className="p-2.5 text-right">Taxes Paid</th>
              <th className="p-2.5 text-right">ESPP</th>
              <th className="p-2.5 text-right font-bold text-green-400">Net Take-Home</th>
              <th className="p-2.5 text-right">YTD 401(k)</th>
              <th className="p-2.5 text-right">YTD HSA</th>
              <th className="p-2.5 text-right">YTD ESPP</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 bg-slate-900/40">
            {displayedPeriods.map((p) => {
              const isMaxedOutLater = p.is401kCapHit || p.isHsaCapHit;
              return (
                <tr
                  key={p.periodNumber}
                  className={`transition-colors hover:bg-slate-800/50 ${
                    p.isBonusPeriod
                      ? 'bg-amber-950/40 font-bold border-l-4 border-l-amber-500'
                      : p.periodNumber === max401kPeriod
                      ? 'bg-purple-950/40 border-l-4 border-l-purple-500'
                      : isMaxedOutLater
                      ? 'bg-emerald-950/20'
                      : ''
                  }`}
                >
                  <td className="p-2.5 font-sans font-semibold text-slate-200 flex items-center gap-1.5 whitespace-nowrap">
                    <span>{p.label}</span>
                    {p.isBonusPeriod && (
                      <span className="text-[9px] bg-amber-500 text-slate-950 font-bold px-1.5 py-0.5 rounded">
                        BONUS
                      </span>
                    )}
                    {p.periodNumber === max401kPeriod && (
                      <span className="text-[9px] bg-purple-600 text-white font-bold px-1.5 py-0.5 rounded">
                        401k MAX
                      </span>
                    )}
                  </td>

                  <td className="p-2.5 text-right font-bold text-slate-200">
                    ${Math.round(p.totalGross).toLocaleString()}
                  </td>

                  <td className="p-2.5 text-right text-purple-300">
                    {p.employee401k > 0 ? (
                      `-$${Math.round(p.employee401k).toLocaleString()}`
                    ) : (
                      <span className="text-slate-500 font-bold">$0 (Maxed)</span>
                    )}
                  </td>

                  <td className="p-2.5 text-right text-emerald-400">
                    +${Math.round(p.employerMatch).toLocaleString()}
                  </td>

                  <td className="p-2.5 text-right text-teal-300">
                    {p.hsa > 0 ? (
                      `-$${Math.round(p.hsa).toLocaleString()}`
                    ) : (
                      <span className="text-slate-500 font-bold">$0 (Maxed)</span>
                    )}
                  </td>

                  <td className="p-2.5 text-right text-rose-400">
                    -${Math.round(p.totalTaxes).toLocaleString()}
                  </td>

                  <td className="p-2.5 text-right text-indigo-300">
                    {p.esppContribution > 0 ? (
                      `-$${Math.round(p.esppContribution).toLocaleString()}`
                    ) : (
                      <span className="text-slate-500 font-bold">$0 (Maxed)</span>
                    )}
                  </td>

                  <td className="p-2.5 text-right font-bold text-green-400 text-sm bg-slate-950/40">
                    ${Math.round(p.netTakeHomePay).toLocaleString()}
                  </td>

                  <td className="p-2.5 text-right text-purple-300">
                    ${Math.round(p.ytd401kEmployee).toLocaleString()}
                    {p.is401kCapHit && <span className="text-[9px] text-purple-400 block font-bold">✓ Capped</span>}
                  </td>

                  <td className="p-2.5 text-right text-teal-300">
                    ${Math.round(p.ytdHsaTotal).toLocaleString()}
                    {p.isHsaCapHit && <span className="text-[9px] text-teal-400 block font-bold">✓ Capped</span>}
                  </td>

                  <td className="p-2.5 text-right text-indigo-300">
                    ${Math.round(p.ytdEsppPayroll).toLocaleString()}
                    {p.isEsppCapHit && <span className="text-[9px] text-indigo-400 block font-bold">✓ Capped</span>}
                  </td>
                </tr>
              );
            })}
          </tbody>
          <tfoot>
            {(() => {
              const totGross = periods.reduce((acc, p) => acc + p.totalGross, 0);
              const tot401k = periods.reduce((acc, p) => acc + p.employee401k, 0);
              const totMatch = periods.reduce((acc, p) => acc + p.employerMatch, 0);
              const totHsa = periods.reduce((acc, p) => acc + p.hsa, 0);
              const totTaxes = periods.reduce((acc, p) => acc + p.totalTaxes, 0);
              const totEspp = periods.reduce((acc, p) => acc + p.esppContribution, 0);
              const totNet = periods.reduce((acc, p) => acc + p.netTakeHomePay, 0);

              return (
                <tr className="bg-slate-950 border-t-2 border-purple-500/50 text-white font-extrabold text-xs">
                  <td className="p-3 font-sans text-purple-400 uppercase tracking-wider font-extrabold">
                    ANNUAL TOTAL (26 Paychecks)
                  </td>
                  <td className="p-3 text-right text-slate-100 font-bold">
                    ${Math.round(totGross).toLocaleString()}
                  </td>
                  <td className="p-3 text-right text-purple-300 font-bold">
                    -${Math.round(tot401k).toLocaleString()}
                  </td>
                  <td className="p-3 text-right text-emerald-400 font-bold">
                    +${Math.round(totMatch).toLocaleString()}
                  </td>
                  <td className="p-3 text-right text-teal-300 font-bold">
                    -${Math.round(totHsa).toLocaleString()}
                  </td>
                  <td className="p-3 text-right text-rose-400 font-bold">
                    -${Math.round(totTaxes).toLocaleString()}
                  </td>
                  <td className="p-3 text-right text-indigo-300 font-bold">
                    -${Math.round(totEspp).toLocaleString()}
                  </td>
                  <td className="p-3 text-right font-black text-green-400 text-sm bg-purple-950/60 border-l border-r border-purple-500/30">
                    ${Math.round(totNet).toLocaleString()}
                  </td>
                  <td className="p-3 text-right text-purple-300 font-bold">
                    ${Math.round(tot401k).toLocaleString()}
                  </td>
                  <td className="p-3 text-right text-teal-300 font-bold">
                    ${Math.round(totHsa).toLocaleString()}
                  </td>
                  <td className="p-3 text-right text-indigo-300 font-bold">
                    ${Math.round(totEspp).toLocaleString()}
                  </td>
                </tr>
              );
            })()}
          </tfoot>
        </table>
      </div>

    </div>
  );
};
