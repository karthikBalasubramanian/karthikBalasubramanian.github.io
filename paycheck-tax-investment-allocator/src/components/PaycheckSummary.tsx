import React from 'react';
import { UserFinancialInputs, TaxBreakdownResult } from '../types';
import { DollarSign, CheckCircle2, TrendingUp, Landmark, Shield, Wallet, ArrowDownRight, Sparkles } from 'lucide-react';

interface PaycheckSummaryProps {
  inputs: UserFinancialInputs;
  taxResult: TaxBreakdownResult;
}

export const PaycheckSummary: React.FC<PaycheckSummaryProps> = ({ inputs, taxResult }) => {
  const isBiweekly = inputs.payFrequency !== 'annual';
  const mul = isBiweekly ? 1 : 26;

  const fmt = (val: number) =>
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(
      val * mul
    );

  const netPay = taxResult.netTakeHomePayBiweekly * mul;
  const grossPay = taxResult.grossBiweekly * mul;
  const totalInvested =
    (taxResult.preTaxDeductionsBiweekly + taxResult.postTaxContributionsBiweekly) * mul;
  const totalInvestedPct = taxResult.percentages.preTax + taxResult.percentages.postTax;

  const annualNetTotal = Math.round(taxResult.schedule?.totalNetTakeHomeAnnual || taxResult.netTakeHomePayAnnual);
  const monthlyNetAverage = Math.round(annualNetTotal / 12);
  const displayBiweeklyNet = taxResult.schedule?.earlyPhaseNetBiweekly || taxResult.netTakeHomePayBiweekly;

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-2xl text-slate-100 space-y-5">
      
      {/* Top Banner */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2">
          <div className="p-2 rounded-xl bg-green-950 text-green-400 border border-green-800/50">
            <Wallet className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-bold text-white">Net Take-Home Paycheck</h3>
            <p className="text-xs text-slate-400">Net Cash in Hand = Gross Income − Pre-Tax Deductions − Taxes − ESPP</p>
          </div>
        </div>
        <span className="text-xs font-bold font-mono px-2.5 py-1 rounded-lg bg-green-950 text-green-300 border border-green-800/60">
          {isBiweekly ? 'Biweekly' : 'Annual'} View
        </span>
      </div>

      {/* Main Metric Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
        
        {/* Net Take-Home Pay */}
        <div className="bg-gradient-to-br from-green-950/80 via-slate-900 to-slate-950 border-2 border-green-500/50 rounded-2xl p-4 shadow-lg space-y-1 relative overflow-hidden">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-green-300 uppercase tracking-wider">
              Net Cash in Hand
            </span>
            <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-green-900/60 text-green-200 border border-green-700/50">
              Paycheck #1
            </span>
          </div>

          <div className="text-2xl sm:text-3xl font-black font-mono text-green-400">
            {fmt(displayBiweeklyNet)}
          </div>
          <div className="text-xs text-slate-300 flex items-center justify-between pt-1 border-t border-green-900/50">
            <span className="text-emerald-300 font-bold font-mono">${monthlyNetAverage.toLocaleString()}/mo avg</span>
            <span className="font-mono text-slate-400">${annualNetTotal.toLocaleString()}/yr</span>
          </div>

          {taxResult.postTaxContributionsBiweekly > 0 ? (
            <div className="mt-1.5 pt-1.5 border-t border-green-900/60 text-xs text-slate-300 flex items-center justify-between font-mono bg-emerald-950/50 -mx-4 -mb-4 p-2.5">
              <span className="text-[11px] font-sans text-slate-300">Remaining After Post-Tax:</span>
              <span className="text-emerald-300 font-bold text-sm">{fmt(taxResult.netTakeHomeAfterPostTaxBiweekly)}</span>
            </div>
          ) : (
            <div className="mt-1.5 pt-1 text-[10px] text-slate-400 font-sans italic">
              Gross − Pre-Tax 401(k)/HSA − Taxes
            </div>
          )}
        </div>

        {/* Total Wealth Invested */}
        <div className="bg-slate-950 border border-indigo-900/60 rounded-2xl p-4 space-y-1">
          <span className="text-xs font-semibold text-indigo-300 uppercase tracking-wider flex items-center gap-1">
            <TrendingUp className="w-3.5 h-3.5" /> Total Wealth Invested
          </span>
          <div className="text-2xl font-black font-mono text-indigo-400">
            {fmt(taxResult.preTaxDeductionsBiweekly + taxResult.postTaxContributionsBiweekly)}
          </div>
          <div className="text-xs text-slate-400 flex items-center justify-between pt-1 border-t border-slate-800">
            <span>{totalInvestedPct.toFixed(1)}% of Gross</span>
            <span className="font-mono text-slate-400">${((taxResult.preTaxDeductionsAnnual + taxResult.postTaxContributionsAnnual)).toLocaleString()}/yr</span>
          </div>
          {taxResult.companyMatchBiweekly > 0 && (
            <div className="mt-1 pt-1 border-t border-indigo-950 flex items-center justify-between text-[10px] text-emerald-400 font-bold">
              <span>+ Co. Match:</span>
              <span className="font-mono">+${(taxResult.companyMatchBiweekly * mul).toFixed(0)}/bw</span>
            </div>
          )}
        </div>

        {/* Total Taxes */}
        <div className="bg-slate-950 border border-rose-900/60 rounded-2xl p-4 space-y-1">
          <span className="text-xs font-semibold text-rose-300 uppercase tracking-wider flex items-center gap-1">
            <Landmark className="w-3.5 h-3.5" /> Total Taxes Paid
          </span>
          <div className="text-2xl font-black font-mono text-rose-400">
            {fmt(taxResult.totalTaxesBiweekly)}
          </div>
          <div className="text-xs text-slate-400 flex items-center justify-between pt-1 border-t border-slate-800">
            <span>{taxResult.percentages.taxes.toFixed(1)}% of Gross</span>
            <span className="font-mono text-slate-400">${(taxResult.totalTaxesAnnual).toLocaleString()}/yr</span>
          </div>
        </div>

        {/* Pre-Tax Savings Tax Benefit */}
        <div className="bg-slate-950 border border-purple-900/60 rounded-2xl p-4 space-y-1">
          <span className="text-xs font-semibold text-purple-300 uppercase tracking-wider flex items-center gap-1">
            <Shield className="w-3.5 h-3.5" /> Pre-Tax Deductions
          </span>
          <div className="text-2xl font-black font-mono text-purple-400">
            {fmt(taxResult.preTaxDeductionsBiweekly)}
          </div>
          <div className="text-xs text-slate-400 flex items-center justify-between pt-1 border-t border-slate-800">
            <span>{taxResult.percentages.preTax.toFixed(1)}% of Gross</span>
            <span className="font-mono text-slate-400">${(taxResult.preTaxDeductionsAnnual).toLocaleString()}/yr</span>
          </div>
        </div>

      </div>

      {/* Paycheck Allocation Progress Bar */}
      <div className="space-y-1.5 pt-1">
        <div className="flex justify-between text-xs text-slate-400">
          <span>Gross Salary Allocation ({fmt(taxResult.grossBiweekly)})</span>
          <span className="font-mono text-slate-300">
            Take-Home: {taxResult.percentages.takeHome.toFixed(1)}% | Taxes: {taxResult.percentages.taxes.toFixed(1)}% | Investments: {totalInvestedPct.toFixed(1)}%
          </span>
        </div>
        <div className="w-full h-4 bg-slate-950 rounded-full overflow-hidden flex p-0.5 border border-slate-800">
          <div
            style={{ width: `${taxResult.percentages.preTax}%` }}
            className="bg-purple-500 h-full transition-all duration-300"
            title={`Pre-Tax: ${taxResult.percentages.preTax.toFixed(1)}%`}
          />
          <div
            style={{ width: `${taxResult.percentages.taxes}%` }}
            className="bg-rose-500 h-full transition-all duration-300"
            title={`Taxes: ${taxResult.percentages.taxes.toFixed(1)}%`}
          />
          <div
            style={{ width: `${taxResult.percentages.postTax}%` }}
            className="bg-emerald-500 h-full transition-all duration-300"
            title={`Post-Tax: ${taxResult.percentages.postTax.toFixed(1)}%`}
          />
          <div
            style={{ width: `${taxResult.percentages.takeHome}%` }}
            className="bg-green-500 h-full rounded-r-full transition-all duration-300"
            title={`Take-Home: ${taxResult.percentages.takeHome.toFixed(1)}%`}
          />
        </div>
      </div>

      {/* Annual One-Time Bonus Breakdown Card */}
      {taxResult.grossAnnualBonus > 0 && (
        <div className="bg-gradient-to-r from-slate-950 via-amber-950/30 to-slate-950 border border-amber-600/40 rounded-xl p-4 space-y-3">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-amber-900/40 pb-2.5">
            <div className="flex items-center gap-2">
              <div className="p-1.5 rounded-lg bg-amber-900/60 text-amber-300 border border-amber-600/50">
                <Sparkles className="w-4.5 h-4.5" />
              </div>
              <div>
                <h4 className="text-xs font-bold text-amber-200 uppercase tracking-wider flex items-center gap-2">
                  <span>One-Time Annual Performance Bonus Breakdown</span>
                </h4>
                <p className="text-[11px] text-slate-300">
                  {inputs.annualBonusIsPercent ? `${inputs.annualBonusPercent}% of Gross Salary` : 'Fixed Annual Bonus'} • Taxed as IRS Supplemental Wages (22% Fed + State + FICA)
                </p>
              </div>
            </div>

            <div className="text-right">
              <span className="text-[10px] uppercase font-bold text-slate-400 block">Gross Bonus</span>
              <span className="text-base font-black font-mono text-amber-300">
                +${Math.round(taxResult.grossAnnualBonus).toLocaleString()} / year
              </span>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 text-xs">
            {/* Employee 401(k) Deduction from Bonus */}
            <div className="bg-slate-900/80 border border-slate-800 p-2.5 rounded-lg space-y-1">
              <span className="text-[10px] text-purple-300 block font-semibold">Bonus 401(k) Deferral</span>
              <div className="font-mono font-bold text-purple-400 text-sm">
                {taxResult.bonus401kContribution > 0 ? `-$${Math.round(taxResult.bonus401kContribution).toLocaleString()}` : '$0'}
              </div>
              <span className="text-[10px] text-slate-400 block">
                {taxResult.bonus401kContribution > 0 ? 'Pre-tax employee deduction' : 'Not applied or capped'}
              </span>
            </div>

            {/* Employee HSA Deduction from Bonus */}
            {taxResult.bonusHsaContribution > 0 && (
              <div className="bg-slate-900/80 border border-slate-800 p-2.5 rounded-lg space-y-1">
                <span className="text-[10px] text-teal-300 block font-semibold">Bonus HSA Contribution</span>
                <div className="font-mono font-bold text-teal-400 text-sm">
                  -${Math.round(taxResult.bonusHsaContribution).toLocaleString()}
                </div>
                <span className="text-[10px] text-slate-400 block">Pre-tax HSA deduction</span>
              </div>
            )}

            {/* Employee ESPP Deduction from Bonus */}
            {taxResult.bonusEsppContribution > 0 && (
              <div className="bg-slate-900/80 border border-slate-800 p-2.5 rounded-lg space-y-1">
                <span className="text-[10px] text-indigo-300 block font-semibold">Bonus ESPP Contribution</span>
                <div className="font-mono font-bold text-indigo-400 text-sm">
                  -${Math.round(taxResult.bonusEsppContribution).toLocaleString()}
                </div>
                <span className="text-[10px] text-slate-400 block">Post-tax stock purchase</span>
              </div>
            )}

            {/* Supplemental Taxes */}
            <div className="bg-slate-900/80 border border-slate-800 p-2.5 rounded-lg space-y-1">
              <span className="text-[10px] text-rose-300 block font-semibold">Supplemental Taxes</span>
              <div className="font-mono font-bold text-rose-400 text-sm">
                -${Math.round(taxResult.bonusTotalTaxes).toLocaleString()}
              </div>
              <span className="text-[10px] text-rose-300/80 block">
                (Fed 22% + State/FICA ~{Math.round((taxResult.bonusTotalTaxes / (taxResult.grossAnnualBonus || 1)) * 100)}%)
              </span>
            </div>

            {/* Net Bonus Check in Hand */}
            <div className="bg-emerald-950/70 border border-emerald-700/60 p-2.5 rounded-lg space-y-1">
              <span className="text-[10px] text-emerald-300 block font-bold uppercase tracking-wider">Net Bonus Take-Home</span>
              <div className="font-mono font-black text-green-300 text-base">
                +${Math.round(taxResult.bonusNetTakeHome).toLocaleString()}
              </div>
              <span className="text-[10px] text-emerald-200 block">
                Lump sum check in hand
              </span>
            </div>

            {/* Total Combined Annual Net */}
            <div className="bg-indigo-950/70 border border-indigo-700/60 p-2.5 rounded-lg space-y-1">
              <span className="text-[10px] text-indigo-300 block font-bold uppercase tracking-wider">Total Combined Annual Net</span>
              <div className="font-mono font-black text-indigo-200 text-base">
                ${Math.round(taxResult.totalCombinedNetAnnual).toLocaleString()} / yr
              </div>
              <span className="text-[10px] text-indigo-300 block">
                Base Net + Bonus Net
              </span>
            </div>
          </div>

          {taxResult.bonusCompanyMatch > 0 && (
            <p className="text-[10px] text-purple-300 font-sans italic pt-1">
              + Employer Bonus 401(k) Match (+${Math.round(taxResult.bonusCompanyMatch).toLocaleString()}) is deposited directly into your 401(k) retirement account.
            </p>
          )}
        </div>
      )}

      {/* Step-by-Step Waterfall List */}
      <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 space-y-2 text-xs font-mono">
        <div className="font-sans font-bold text-slate-200 border-b border-slate-800 pb-2 flex items-center justify-between">
          <span>Paycheck Waterfall Calculation</span>
          <span className="text-slate-400 text-[11px]">Exact arithmetic</span>
        </div>

        <div className="flex justify-between py-1 text-slate-200 font-bold">
          <span>1. Gross Salary</span>
          <span className="text-blue-400">{fmt(taxResult.grossBiweekly)}</span>
        </div>

        <div className="flex justify-between py-1 text-purple-300 pl-4 border-l-2 border-purple-800/50">
          <span>Less: Pre-Tax Deductions (401k + HSA + FSA)</span>
          <span>-{fmt(taxResult.preTaxDeductionsBiweekly)}</span>
        </div>

        <div className="flex justify-between py-1 text-slate-300 font-semibold border-t border-slate-900">
          <span>= Taxable Income Base</span>
          <span>{fmt(taxResult.taxableGrossBiweekly)}</span>
        </div>

        <div className="flex justify-between py-1 text-rose-300 pl-4 border-l-2 border-rose-800/50">
          <span>Less: Total Taxes (Federal + State + FICA + SDI)</span>
          <span>-{fmt(taxResult.totalTaxesBiweekly)}</span>
        </div>

        <div className="flex justify-between py-2 text-green-400 font-bold text-sm border-t border-slate-800 bg-slate-900/60 px-2 rounded-lg mt-1">
          <span className="font-sans">Final Net Take-Home Pay (Bank Account)</span>
          <span>{fmt(taxResult.netTakeHomePayBiweekly)}</span>
        </div>

        {taxResult.postTaxContributionsBiweekly > 0 && (
          <div className="flex justify-between py-1 text-emerald-300/80 pl-4 border-l-2 border-emerald-800/50 text-[11px]">
            <span>Less: Post-Tax Allocations (Roth/529/Child/ESPP)</span>
            <span>-{fmt(taxResult.postTaxContributionsBiweekly)}</span>
          </div>
        )}

        {taxResult.companyMatchBiweekly > 0 && (
          <p className="text-[10px] text-emerald-400/90 font-sans italic pt-2 border-t border-slate-900">
            * Note: Employer 401(k) Match (+{fmt(taxResult.companyMatchBiweekly)}) is deposited directly into your 401(k) retirement account and is excluded from liquid net take-home pay.
          </p>
        )}
      </div>

    </div>
  );
};
